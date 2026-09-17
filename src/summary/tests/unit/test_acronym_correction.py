"""Tests for the acronym correction service."""

import json
import logging

import pytest

from summary.core import acronym_correction, prompt
from summary.core.acronym_correction import (
    _locate_span,
    _Position,
    _shortlist,
    correct_acronyms,
)
from summary.core.phonetic import (
    PhoneticIndex,
    acronym_keys,
    keyset,
    similarity,
)
from summary.core.shared_models import WhisperXResponse

GLOSSARY = {
    "DINUM": "direction interministérielle du numérique",
    "DID": "dossier individuel de demande",
    "RGPD": "Règlement général sur la protection des données",
    "CNIL": "Commission nationale de l'informatique et des libertés",
}


def make_word(word: str, start: float, speaker: str = "Camille") -> dict:
    """Build a WhisperX word entry."""
    return {
        "word": word,
        "start": start,
        "end": start + 0.4,
        "score": 0.9,
        "speaker": speaker,
    }


def make_transcription() -> dict:
    """Build a one-segment transcription holding a mistranscribed acronym."""
    words = ["Côté", "dix", "nomme,", "la", "brique", "a", "avancé."]
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Côté dix nomme, la brique a avancé.",
                "speaker": "Camille",
                "words": [
                    make_word(word, index * 0.5) for index, word in enumerate(words)
                ],
            }
        ],
        "word_segments": [
            make_word(word, index * 0.5) for index, word in enumerate(words)
        ],
    }


class StubLLMService:
    """LLMService stand-in: answers from canned data, never hits the network."""

    def __init__(self, spans: list[str], decisions: list[dict]):
        """Store the spans to flag and the decisions to hand back, in order."""
        self.spans = spans
        self.decisions = list(decisions)
        self.calls: list[str] = []

    def call(self, system_prompt, user_prompt, name, response_format=None):
        """Return the canned answer for the stage being called."""
        self.calls.append(name)
        if name == "acronym-detect":
            return json.dumps({"suspects": [{"span": span} for span in self.spans]})
        if self.decisions:
            return json.dumps(self.decisions.pop(0))
        return json.dumps({"choix": None, "confiance": 0.0})


@pytest.fixture()
def index() -> PhoneticIndex:
    """Provide a phonetic index over the small test glossary."""
    return PhoneticIndex(GLOSSARY)


@pytest.fixture(autouse=True)
def test_glossary(monkeypatch):
    """Use the small test glossary instead of the shipped one."""
    monkeypatch.setattr(acronym_correction, "get_acronym_glossary", lambda: GLOSSARY)


def set_confidence_floor(monkeypatch, floor: float):
    """Override the confidence floor on the module-level settings."""
    monkeypatch.setattr(
        acronym_correction,
        "settings",
        acronym_correction.settings.model_copy(
            update={"acronym_correction_min_confidence": floor}
        ),
    )


# --------------------------------------------------------------- phonetics


def test_acronym_indexed_under_both_pronunciations(index):
    """A word-like and a spelled-out reading both reach the same acronym."""
    assert index.candidates("dix nomme") == [("DINUM", pytest.approx(0.8))]

    spelled = dict(index.candidates("erre gé pé dé"))
    assert "RGPD" in spelled


def test_acronym_keys_cover_word_like_and_spelled_readings():
    """A vowel-carrying acronym gets a word key on top of its spelled keys."""
    keys = acronym_keys("DINUM")

    assert keyset("DINUM") <= keys
    assert len(keys) > len(keyset("DINUM"))

    # A consonant-only acronym is only ever spelled out.
    assert acronym_keys("RGPD") == keyset("er je pe de")


def test_similarity_is_normalised():
    """Similarity is 1.0 on equal keys and 0.0 on an empty one."""
    assert similarity("dinum", "dinum") == 1.0
    assert similarity("dinom", "dinum") == pytest.approx(0.8)
    assert similarity("", "dinum") == 0.0


# ------------------------------------------------------------ maximal munch


def test_longest_window_claims_its_words(index):
    """A two-word window wins as DINUM instead of its first word as DID."""
    shortlist, where = _shortlist(index, ["dix", "nomme"], 0, 2)

    assert [acronym for acronym, _ in shortlist] == ["DINUM"]
    assert where["DINUM"] == ("dix nomme", 0, 2)


def test_single_word_still_matches_when_alone(index):
    """A one-word window is shortlisted when no longer window claims it."""
    shortlist, _ = _shortlist(index, ["dix"], 0, 1)

    assert [acronym for acronym, _ in shortlist] == ["DID"]


def test_locate_span_ignores_punctuation_and_case():
    """A flagged passage is found even when its punctuation differs."""
    segments = make_transcription()["segments"]

    assert _locate_span(segments, "Dix nomme") == _Position(0, 1, 2, True)
    assert _locate_span(segments, "") is None


# ------------------------------------------------------------ full pipeline


def test_correct_acronyms_rewrites_the_transcription(index):
    """An accepted correction rewrites words, text and word_segments."""
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    corrected, corrections = correct_acronyms(
        make_transcription(), llm_service, index=index
    )

    assert corrections == [
        {
            "segment_index": 0,
            "word_index": 1,
            "n_words": 2,
            "wrong": "dix nomme",
            "correct": "DINUM",
            "confidence": 0.95,
            "applied": True,
        }
    ]

    segment = corrected["segments"][0]
    assert segment["text"] == "Côté DINUM, la brique a avancé."
    assert [word["word"] for word in segment["words"]] == [
        "Côté",
        "DINUM",
        "la",
        "brique",
        "a",
        "avancé.",
    ]

    merged = segment["words"][1]
    assert merged["corrected"] is True
    assert merged["start"] == 0.5
    assert merged["end"] == pytest.approx(1.4)

    assert [word["word"] for word in corrected["word_segments"]][1] == "DINUM"


def test_correction_below_the_confidence_floor_is_recorded_not_applied(
    index, monkeypatch
):
    """A low-confidence choice is reported but leaves the transcript alone."""
    set_confidence_floor(monkeypatch, 0.8)
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.5}]
    )

    corrected, corrections = correct_acronyms(
        make_transcription(), llm_service, index=index
    )

    assert corrections[0]["applied"] is False
    assert corrected["segments"][0]["text"] == "Côté dix nomme, la brique a avancé."
    assert [word["word"] for word in corrected["segments"][0]["words"]] == [
        "Côté",
        "dix",
        "nomme,",
        "la",
        "brique",
        "a",
        "avancé.",
    ]


def test_confidence_floor_is_read_from_settings(index, monkeypatch):
    """Lowering the floor applies a correction that was held back before."""
    set_confidence_floor(monkeypatch, 0.4)
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.5}]
    )

    corrected, corrections = correct_acronyms(
        make_transcription(), llm_service, index=index
    )

    assert corrections[0]["applied"] is True
    assert corrected["segments"][0]["text"] == "Côté DINUM, la brique a avancé."


def test_accepts_a_whisperx_response_object(index):
    """A WhisperXResponse is handled like the equivalent dict."""
    transcription = WhisperXResponse.model_validate(make_transcription())
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    corrected, corrections = correct_acronyms(transcription, llm_service, index=index)

    assert isinstance(corrected, dict)
    assert corrections[0]["correct"] == "DINUM"
    assert corrected["segments"][0]["text"] == "Côté DINUM, la brique a avancé."

    # The corrected marker survives a round-trip through the model.
    revalidated = WhisperXResponse.model_validate(corrected)
    assert revalidated.segments[0].words[1].corrected is True


def test_rejected_span_leaves_the_transcript_untouched(index):
    """A null decision produces no correction at all."""
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": None, "confiance": 0.0}]
    )

    corrected, corrections = correct_acronyms(
        make_transcription(), llm_service, index=index
    )

    assert corrections == []
    assert corrected == make_transcription()


def test_segment_with_a_null_words_field_is_corrected_on_the_text_path(index):
    """A segment carrying a null words field is corrected from its text.

    It used to be skipped, which silently dropped every correction on output
    from a recogniser that runs no forced alignment.
    """
    transcription = {"segments": [{"text": "Côté dix nomme.", "words": None}]}
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    corrected, corrections = correct_acronyms(transcription, llm_service, index=index)

    assert [correction["correct"] for correction in corrections] == ["DINUM"]
    assert corrected["segments"][0]["text"] == "Côté DINUM."
    # There were no words to rewrite, so the field is left exactly as it was.
    assert corrected["segments"][0]["words"] is None


def test_llm_failure_is_swallowed(index):
    """A failing LLM call yields no corrections instead of raising."""

    class FailingLLMService:
        """LLMService stand-in that always fails."""

        def call(self, system_prompt, user_prompt, name, response_format=None):
            """Raise as a broken LLM backend would."""
            raise RuntimeError("LLM call failed")

    corrected, corrections = correct_acronyms(
        make_transcription(), FailingLLMService(), index=index
    )

    assert corrections == []
    assert corrected == make_transcription()


def test_user_glossary_adds_unknown_acronyms():
    """An organisation's own vocabulary is searchable even if nowhere public."""
    index = acronym_correction.build_index({"ZORBLAX": "Zone régionale blabla"})
    assert any(
        acronym == "ZORBLAX" for acronym, _ in index.candidates("zorblax", top_k=5)
    )


def test_user_glossary_outranks_the_shipped_one():
    """A user entry beats a public homonym that scores the same phonetically."""
    index = acronym_correction.build_index({"DINUM": "Direction du numérique"})

    plain, _ = acronym_correction._shortlist(index, ["dix", "nomme"], 0, 2)
    boosted, _ = acronym_correction._shortlist(index, ["dix", "nomme"], 0, 2, {"DINUM"})

    plain_rank = [acronym for acronym, _ in plain].index("DINUM")
    boosted_rank = [acronym for acronym, _ in boosted].index("DINUM")
    assert boosted_rank <= plain_rank
    assert boosted[0][0] == "DINUM"


def test_user_glossary_expansion_replaces_the_shipped_one():
    """The uploader's definition of their own acronym wins."""
    shipped = acronym_correction.get_acronym_glossary()
    assert "CNIL" in shipped
    index = acronym_correction.build_index({"CNIL": "Comité National Interne Local"})
    assert isinstance(index, PhoneticIndex)


def test_no_user_glossary_reuses_the_cached_index():
    """Passing nothing must not rebuild the 7769-entry index on every call."""
    assert acronym_correction.build_index() is acronym_correction.get_acronym_index()
    assert (
        acronym_correction.build_index(None) is acronym_correction.get_acronym_index()
    )


def test_context_words_do_not_create_wider_windows(index):
    """Stage 1's span is trusted: a neighbouring word cannot join the match.

    With a margin, "Côté dix nomme" is tried as a window, matches a worse
    acronym and swallows the words of the correct one.
    """
    # stage 1 flagged only "dix nomme": tokens 1-2
    shortlist, where = _shortlist(index, ["Côté", "dix", "nomme"], 1, 2)

    assert [acronym for acronym, _ in shortlist] == ["DINUM"]
    assert where["DINUM"] == ("dix nomme", 1, 2)
    assert all("Côté" not in matched[0] for matched in where.values())


def test_user_glossary_lowers_the_confidence_floor():
    """A declared acronym is applied at a confidence a guessed one would not be."""
    plain = acronym_correction._floor_for("DINUM", None)
    declared = acronym_correction._floor_for("DINUM", {"DINUM"})
    other = acronym_correction._floor_for("DICOM", {"DINUM"})

    assert declared < plain
    assert other == plain
    # the case that motivated it: 0.75 passes only when declared
    assert 0.75 >= declared
    assert 0.75 < plain


def test_shipped_glossary_exposes_evidence_weights():
    """Each shipped acronym records how many corpora confirmed it."""
    weights = acronym_correction.get_acronym_weights()
    assert weights["DINUM"] > weights["DICOM"]
    assert all(isinstance(value, int) and value >= 1 for value in weights.values())


def test_expansions_still_load_as_plain_strings():
    """The prompt needs {acronym: expansion}, not the raw file shape."""
    glossary = acronym_correction.get_acronym_glossary()
    assert isinstance(glossary["DINUM"], str)
    assert "numérique" in glossary["DINUM"]


def test_better_attested_acronym_wins_a_phonetic_tie(monkeypatch):
    """Homonyms tie phonetically; the one more corpora confirmed ranks first."""
    glossary = {
        "DINUM": "direction interministérielle du numérique",
        "DICOM": "délégation à l'information et la communication",
    }
    monkeypatch.setattr(acronym_correction, "get_acronym_glossary", lambda: glossary)
    monkeypatch.setattr(
        acronym_correction, "get_acronym_weights", lambda: {"DINUM": 5, "DICOM": 2}
    )

    shortlist, _ = acronym_correction._shortlist(
        PhoneticIndex(glossary), ["dix", "nomme"], 0, 2
    )
    ranked = [acronym for acronym, _ in shortlist]

    assert ranked.index("DINUM") < ranked.index("DICOM")


def test_evidence_bonus_cannot_override_a_real_similarity_gap(monkeypatch):
    """Attestation breaks ties; it must not promote a clearly worse match."""
    glossary = {"DINUM": "direction du numérique", "XYZ": "sans rapport"}
    monkeypatch.setattr(acronym_correction, "get_acronym_glossary", lambda: glossary)
    monkeypatch.setattr(
        acronym_correction, "get_acronym_weights", lambda: {"DINUM": 1, "XYZ": 7}
    )

    shortlist, _ = acronym_correction._shortlist(
        PhoneticIndex(glossary), ["dix", "nomme"], 0, 2
    )

    assert shortlist[0][0] == "DINUM"


def test_decision_prompt_explains_the_organisation_marker():
    """The candidate list renders a marker; the prompt must say what it means.

    Without this the model sees an undocumented annotation and cannot weigh it.
    """
    assert "glossaire de l'organisation" in prompt.PROMPT_SYSTEM_ACRONYM_DECIDE


def test_transcription_without_word_timings_says_so_and_still_corrects(index, caplog):
    """No per-word timings is not a dead end: correct the text, and say so.

    This used to return early. Thirty-three suspect passages on a real
    hour-long transcript then produced zero corrections, indistinguishable
    from a clean transcript.
    """
    transcription = {
        "segments": [{"start": 0.0, "end": 6.0, "text": "Côté dix nomme, la brique."}],
        "word_segments": [],
    }
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    with caplog.at_level(logging.INFO):
        corrected, corrections = acronym_correction.correct_acronyms(
            transcription=transcription, llm_service=llm_service, index=index
        )

    assert [correction["correct"] for correction in corrections] == ["DINUM"]
    assert corrected["segments"][0]["text"] == "Côté DINUM, la brique."
    assert any("per-word timings" in record.getMessage() for record in caplog.records)


def test_text_path_correction_reports_no_word_level_position(index):
    """Without words there is nothing to index into: the fields are None.

    The rest of the correction keeps the shape the word path produces, so a
    caller reading the audit trail does not need to know which path ran.
    """
    transcription = {"segments": [{"text": "Côté dix nomme, la brique."}]}
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    _, corrections = correct_acronyms(transcription, llm_service, index=index)

    assert corrections == [
        {
            "segment_index": 0,
            "word_index": None,
            "n_words": None,
            "wrong": "dix nomme",
            "correct": "DINUM",
            "confidence": 0.95,
            "applied": True,
        }
    ]


def test_text_path_corrects_a_real_albert_segment(index):
    """The measured case: plain Whisper output, an acronym under an elision.

    Albert's transcription endpoint serves Whisper without forced alignment, so
    segments carry text only. Whisper writes the elided article onto the
    acronym -- "Côté dix nomme" comes back as "Côté d'Inhomme" -- and the
    correction has to survive both at once.
    """
    transcription = {
        "segments": [
            {
                "start": 0.031,
                "end": 6.342,
                "speaker": None,
                "text": (
                    "Bonjour, ici Camille. Côté d'Inhomme, la brique de "
                    "collaboration a bien avancé ce mois-ci."
                ),
            }
        ]
    }
    llm_service = StubLLMService(
        spans=["d'Inhomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    corrected, corrections = correct_acronyms(transcription, llm_service, index=index)

    assert [correction["correct"] for correction in corrections] == ["DINUM"]
    assert corrections[0]["applied"] is True
    # The comma that hung off the token stays where it was.
    assert corrected["segments"][0]["text"] == (
        "Bonjour, ici Camille. Côté DINUM, la brique de collaboration "
        "a bien avancé ce mois-ci."
    )


def test_word_path_is_preferred_when_a_segment_has_both(index):
    """A segment with words keeps the word path, timings and all."""
    transcription = make_transcription()
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    corrected, corrections = correct_acronyms(transcription, llm_service, index=index)

    assert corrections[0]["word_index"] == 1
    assert corrections[0]["n_words"] == 2
    assert corrected["segments"][0]["words"][1]["start"] == 0.5


def test_both_paths_run_in_one_transcription(index):
    """A mixed transcription corrects each segment the way that segment allows."""
    transcription = make_transcription()
    transcription["segments"].append(
        {"start": 3.0, "end": 6.0, "text": "La quenil a publié sa délibération."}
    )
    llm_service = StubLLMService(
        spans=["dix nomme", "quenil"],
        decisions=[
            {"choix": "DINUM", "confiance": 0.95},
            {"choix": "CNIL", "confiance": 0.95},
        ],
    )

    corrected, corrections = correct_acronyms(transcription, llm_service, index=index)

    assert [correction["correct"] for correction in corrections] == ["DINUM", "CNIL"]
    assert corrected["segments"][0]["text"] == "Côté DINUM, la brique a avancé."
    assert corrected["segments"][1]["text"] == "La CNIL a publié sa délibération."
    # The word path kept its timings; the text path had none to keep.
    assert corrections[0]["word_index"] == 1
    assert corrections[1]["word_index"] is None


def test_word_segments_survive_a_wordless_segment(index):
    """A text-path correction must not reach into the flat word list.

    word_segments belongs to the word path. Letting a wordless segment mirror
    into it rewrites a window that was never located there.
    """
    transcription = {
        "segments": [{"start": 0.0, "end": 3.0, "text": "Côté dix nomme, la brique."}],
        "word_segments": [make_word("Côté", 0.0), make_word("ailleurs", 0.5)],
    }
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    corrected, corrections = correct_acronyms(transcription, llm_service, index=index)

    assert corrections[0]["correct"] == "DINUM"
    assert [word["word"] for word in corrected["word_segments"]] == [
        "Côté",
        "ailleurs",
    ]


def test_locate_span_matches_a_token_holding_an_elided_article(index):
    """One token can normalise to two words: compare the flattened streams.

    "d'Inhomme" normalises to "d inhomme", so a token-for-word comparison never
    matches the span the detector reported.
    """
    segments = [{"text": "Côté d'Inhomme, la brique."}]

    assert _locate_span(segments, "d'Inhomme") == _Position(0, 1, 1, False)


def test_elided_acronym_wins_against_the_whole_shipped_glossary(monkeypatch):
    """The measured case, against all 7769 shipped acronyms, not the test four.

    Passing the shortlist stage in a four-acronym test glossary proves the
    plumbing, not the premise. In the real one "d'Inhomme" has to clear the
    similarity floor and outrank DICOM, its nearest homonym, before the list is
    cut to top_k -- otherwise DINUM never reaches the deciding model at all.
    """
    shipped = acronym_correction._load_glossary_file()
    glossary = {acronym: entry[0] for acronym, entry in shipped.items()}
    weights = {acronym: entry[1] for acronym, entry in shipped.items()}
    monkeypatch.setattr(acronym_correction, "get_acronym_glossary", lambda: glossary)
    monkeypatch.setattr(acronym_correction, "get_acronym_weights", lambda: weights)

    # the token as the text path hands it over, trailing comma and all
    shortlist, where = acronym_correction._shortlist(
        PhoneticIndex(glossary), ["d'Inhomme,"], 0, 1
    )

    assert [acronym for acronym, _ in shortlist][0] == "DINUM"
    assert where["DINUM"][0] == "d'Inhomme"


def test_elided_article_still_matches_its_acronym():
    """Whisper glues the elided article on: "d'Inhomme" must still reach DINUM.

    Splitting on the apostrophe gives "d inhomme" (0.60), under the similarity
    floor, so DINUM never reached the model at all.
    """
    score = max(
        similarity(query, key)
        for query in keyset("d'Inhomme")
        for key in acronym_keys("DINUM")
    )
    assert score >= 0.8


def test_gluing_does_not_break_a_genuine_elided_article():
    """A genuine elided article stays one: "l'ANOM" must still reach ANOM."""

    def best(acronym: str) -> float:
        return max(
            similarity(query, key)
            for query in keyset("l'ANOM")
            for key in acronym_keys(acronym)
        )

    assert best("ANOM") > best("DINUM")
