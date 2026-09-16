"""Tests for the acronym correction service."""

import json

import pytest

from summary.core import acronym_correction
from summary.core.acronym_correction import (
    _locate_span,
    _shortlist,
    correct_acronyms,
)
from summary.core.phonetic import PhoneticIndex, acronym_keys, keyset, similarity
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
    words = [make_word("dix", 0.0), make_word("nomme", 0.5)]

    shortlist, where = _shortlist(index, words, 0, 2)

    assert [acronym for acronym, _ in shortlist] == ["DINUM"]
    assert where["DINUM"] == ("dix nomme", 0, 2)


def test_single_word_still_matches_when_alone(index):
    """A one-word window is shortlisted when no longer window claims it."""
    words = [make_word("dix", 0.0)]

    shortlist, _ = _shortlist(index, words, 0, 1)

    assert [acronym for acronym, _ in shortlist] == ["DID"]


def test_locate_span_ignores_punctuation_and_case():
    """A flagged passage is found even when its punctuation differs."""
    segments = make_transcription()["segments"]

    assert _locate_span(segments, "Dix nomme") == (0, 1, 2)
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


def test_segment_without_words_is_skipped(index):
    """A segment carrying a null words field does not break the chain."""
    transcription = {"segments": [{"text": "Côté dix nomme.", "words": None}]}
    llm_service = StubLLMService(
        spans=["dix nomme"], decisions=[{"choix": "DINUM", "confiance": 0.95}]
    )

    corrected, corrections = correct_acronyms(transcription, llm_service, index=index)

    assert corrections == []
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
    words = [
        {"word": "dix", "start": 0.0, "end": 0.3},
        {"word": "nomme", "start": 0.3, "end": 0.7},
    ]
    index = acronym_correction.build_index({"DINUM": "Direction du numérique"})

    plain, _ = acronym_correction._shortlist(index, words, 0, 2)
    boosted, _ = acronym_correction._shortlist(index, words, 0, 2, {"DINUM"})

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
