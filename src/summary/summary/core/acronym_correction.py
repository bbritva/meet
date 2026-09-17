"""Correct acronyms that speech recognition transcribed as ordinary words.

The chain has four stages:

1. an LLM reads the transcript and flags the passages that make no sense;
2. every flagged passage is located in the word stream, and each window around
   it is matched against a glossary of ~7700 French administrative acronyms by
   pronunciation (see `summary.core.phonetic`);
3. an LLM picks the acronym that fits the meaning of the sentence, or rejects
   the whole shortlist, and reports its confidence;
4. accepted corrections above the configured confidence floor are written into
   the transcription, and the rewritten tokens are marked `corrected`.

Precision matters more than recall here: a wrong correction changes the
meaning of a document colleagues read, while a missed one merely leaves the
transcript as it was. Corrections below the floor are recorded but not applied.

Measured end to end on synthetic mocks: about 50% recall and 85% precision.
"""

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from summary.core.config import get_settings
from summary.core.phonetic import MIN_WORD_LIKE_LENGTH, PhoneticIndex
from summary.core.prompt import (
    FORMAT_ACRONYM_DECIDE,
    FORMAT_ACRONYM_DETECT,
    PROMPT_SYSTEM_ACRONYM_DECIDE,
    PROMPT_SYSTEM_ACRONYM_DETECT,
    PROMPT_USER_ACRONYM_DECIDE,
)

settings = get_settings()

logger = logging.getLogger(__name__)

GLOSSARY_PATH = Path(__file__).parent / "data" / "acronyms.json"

# Segments sent to the detection LLM in one call.
SEGMENTS_PER_BATCH = 4

# Longest word window matched against the glossary.
MAX_WINDOW_WORDS = 4

# Words looked at on each side of a flagged passage: the LLM usually reports a
# span slightly wider or narrower than the actual error.
# Words of context added either side of the passage stage 1 flagged. Zero: the
# detector's span is trusted exactly. A margin lets stage 2 invent windows that
# were never flagged -- on "Côté dix nomme" the three-word window matches COTRIM
# (0.70), claims the words, and blocks the correct "dix nomme" -> DINUM (0.80).
# Measured on the mocks: margin 0 raises recall from 47% to 56% at equal precision.
WINDOW_MARGIN = 0
# An acronym supplied by the organisation outranks one mined from public corpora:
# the uploader knows their own vocabulary. Large enough to beat a phonetic near-tie
# (DINUM 0.80 vs DICOM 0.80), small enough not to force a clearly worse match.
USER_GLOSSARY_BONUS = 0.20
# Homonyms arrive phonetically tied: "dix nomme" matches DINUM and DICOM at 0.80
# each, so the winner was arbitrary. The shipped glossary records how many
# independent corpora confirmed each acronym -- DINUM 5, DICOM 2 -- which breaks
# the tie. Deliberately small: it must not override a real similarity gap.
EVIDENCE_BONUS_PER_SOURCE = 0.015
EVIDENCE_BONUS_CAP = 0.06

# Longest token sequence considered when locating a flagged passage.
MAX_LOCATE_WORDS = 8

# Common French words are never an acronym on their own, and they match a lot
# of short acronyms by sound.
STOPWORDS = frozenset(
    """le la les un une des du de et en dans pour par sur avec sans sous au aux
ce cet cette ces il elle ils elles on nous vous je tu me te se qui que quoi dont ou
est sont a ont ete etre avoir fait plus moins tres bien mal tout tous toute toutes
mais donc car ni or si comme quand alors ainsi aussi encore deja puis enfin
cela ceci celui celle leur leurs son sa ses mon ma mes ton ta tes notre votre
peut peuvent doit doivent va vont faire dire voir savoir pouvoir vouloir
oui non merci bonjour oh euh ah eh bon ben""".split()
)


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _load_glossary_file() -> dict[str, list]:
    """Read the shipped {acronym: [expansion, source_count]} file once."""
    with GLOSSARY_PATH.open(encoding="utf-8") as glossary_file:
        return json.load(glossary_file)


def get_acronym_glossary() -> dict[str, str]:
    """The {acronym: expansion} mapping shipped with the service."""
    return {
        acronym: entry[0] for acronym, entry in _load_glossary_file().items()
    }


def get_acronym_weights() -> dict[str, int]:
    """How many independent corpora confirmed each shipped acronym."""
    return {
        acronym: entry[1] for acronym, entry in _load_glossary_file().items()
    }


@lru_cache(maxsize=1)
def get_acronym_index() -> PhoneticIndex:
    """Build the phonetic index once per worker process."""
    return PhoneticIndex(get_acronym_glossary())


def build_index(user_glossary: Optional[dict[str, str]] = None) -> PhoneticIndex:
    """Index the shipped glossary, extended by an organisation's own entries.

    A user entry with the same acronym replaces the shipped expansion: the
    uploader's definition of their own vocabulary wins.
    """
    if not user_glossary:
        return get_acronym_index()
    merged = dict(get_acronym_glossary())
    merged.update(user_glossary)
    return PhoneticIndex(merged)


def _normalize(text: str) -> str:
    """Lower-case, strip accents and punctuation, for token comparison."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9 ]", " ", stripped).strip()


def _as_dict(transcription: Any) -> dict[str, Any]:
    """Return a plain dict for a WhisperXResponse or an already-plain dict."""
    if hasattr(transcription, "model_dump"):
        return transcription.model_dump()
    return json.loads(json.dumps(transcription))


def _parse_json_object(payload: Optional[str]) -> dict[str, Any]:
    """Parse the first JSON object found in an LLM answer."""
    if not payload:
        return {}
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _detect_suspect_spans(segments: list[dict[str, Any]], llm_service) -> list[str]:
    """Ask the LLM which passages of the transcript make no sense."""
    suspects: list[str] = []

    for start in range(0, len(segments), SEGMENTS_PER_BATCH):
        batch = segments[start : start + SEGMENTS_PER_BATCH]
        user_prompt = "\n".join(
            f"[{segment.get('speaker') or ''}] {segment.get('text', '').strip()}"
            for segment in batch
        )
        try:
            answer = llm_service.call(
                PROMPT_SYSTEM_ACRONYM_DETECT,
                user_prompt,
                name="acronym-detect",
                response_format=FORMAT_ACRONYM_DETECT,
            )
        except Exception as exc:
            logger.warning("Acronym detection failed on a batch: %s", exc)
            continue

        for suspect in _parse_json_object(answer).get("suspects") or []:
            span = (suspect or {}).get("span", "") if isinstance(suspect, dict) else ""
            if span.strip():
                suspects.append(span.strip())

    return suspects


def _locate_span(
    segments: list[dict[str, Any]], span: str
) -> Optional[tuple[int, int, int]]:
    """Find a flagged passage in the word stream.

    The LLM answers with text, not indices, and its punctuation and casing
    rarely match the tokens exactly, so the match is done on normalised token
    sequences, with the longest contiguous run as a fallback.

    Args:
        segments: the transcription segments.
        span: the passage reported by the detection stage.

    Returns:
        A (segment index, word index, word count) tuple, or None when the
        passage cannot be found.
    """
    targets = [token for token in _normalize(span).split() if token]
    if not targets:
        return None

    best: Optional[tuple[int, int, int]] = None
    best_length = 0

    for segment_index, segment in enumerate(segments):
        words = segment.get("words") or []
        normalized = [_normalize(word.get("word", "")) for word in words]

        for index in range(len(words)):
            for length in range(1, min(MAX_LOCATE_WORDS, len(words) - index) + 1):
                window = [
                    token for token in normalized[index : index + length] if token
                ]
                if window == targets:
                    return segment_index, index, length

        for index in range(len(words)):
            length = 0
            while (
                index + length < len(words)
                and length < len(targets)
                and normalized[index + length] == targets[length]
            ):
                length += 1
            if length > best_length:
                best = (segment_index, index, length)
                best_length = length

    return best


def _window_candidates(
    index: PhoneticIndex, words: list[dict[str, Any]], low: int, high: int
) -> list[tuple[int, int, str, list[tuple[str, float]]]]:
    """Score every word window of a region against the glossary."""
    scored = []
    for start in range(low, high):
        for length in range(1, min(MAX_WINDOW_WORDS, high - start) + 1):
            tokens = [
                word.get("word", "").strip(" ,.;:!?")
                for word in words[start : start + length]
            ]
            text = " ".join(token for token in tokens if token)
            if not text.strip():
                continue
            if length == 1 and _normalize(text) in STOPWORDS:
                continue
            candidates = [
                (acronym, score)
                for acronym, score in index.candidates(
                    text,
                    top_k=settings.acronym_correction_top_k,
                    min_similarity=settings.acronym_correction_min_similarity,
                )
                if len(re.sub(r"[^A-Za-z0-9]", "", acronym)) >= MIN_WORD_LIKE_LENGTH
            ]
            if candidates:
                scored.append((length, start, text, candidates))
    return scored


def _shortlist(
    index: PhoneticIndex,
    words: list[dict[str, Any]],
    word_index: int,
    n_words: int,
    user_acronyms: Optional[set[str]] = None,
) -> tuple[list[tuple[str, float]], dict[str, tuple[str, int, int]]]:
    """Build one ranked shortlist for the region around a flagged passage.

    Windows are claimed longest first: once a window owns its word positions,
    shorter overlapping windows are skipped. Without this maximal munch, "dix"
    matching DI at 1.00 outranks "dix nomme" matching DINUM at 0.80 and splits
    the acronym in two.

    Returns:
        The ranked (acronym, similarity) shortlist, and a mapping from acronym
        to the (text, word index, word count) window it matched.
    """
    low = max(0, word_index - WINDOW_MARGIN)
    high = min(len(words), word_index + n_words + WINDOW_MARGIN)

    windows = _window_candidates(index, words, low, high)
    windows.sort(key=lambda window: (-window[0], window[1]))
    weights = get_acronym_weights()

    pool: dict[str, tuple[float, str, int, int]] = {}
    claimed: set[int] = set()
    for length, start, text, candidates in windows:
        positions = set(range(start, start + length))
        if positions & claimed:
            continue
        claimed |= positions
        for acronym, raw_score in candidates:
            if user_acronyms and acronym in user_acronyms:
                score = min(1.0, raw_score + USER_GLOSSARY_BONUS)
            else:
                evidence = min(
                    EVIDENCE_BONUS_CAP,
                    EVIDENCE_BONUS_PER_SOURCE * (weights.get(acronym, 1) - 1),
                )
                score = min(1.0, raw_score + evidence)
            if acronym not in pool or score > pool[acronym][0]:
                pool[acronym] = (score, text, start, length)

    ranked = sorted(pool.items(), key=lambda item: -item[1][0])[
        : settings.acronym_correction_top_k
    ]
    shortlist = [(acronym, entry[0]) for acronym, entry in ranked]
    where = {acronym: (entry[1], entry[2], entry[3]) for acronym, entry in ranked}
    return shortlist, where


def _decide(
    sentence: str,
    span: str,
    shortlist: list[tuple[str, float]],
    llm_service,
    user_glossary: Optional[dict[str, str]] = None,
) -> tuple[Optional[str], float]:
    """Ask the LLM to pick an acronym from the shortlist, or reject it.

    Entries from the organisation's own glossary are marked, so the model can
    prefer them over homonyms mined from public corpora.
    """
    glossary = dict(get_acronym_glossary())
    if user_glossary:
        glossary.update(user_glossary)
    lines = []
    for acronym, _ in shortlist:
        line = f"- {acronym}"
        if glossary.get(acronym):
            line += f" ({glossary[acronym][:70]})"
        if user_glossary and acronym in user_glossary:
            line += "  [glossaire de l'organisation]"
        lines.append(line)
    candidates = "\n".join(lines)
    user_prompt = PROMPT_USER_ACRONYM_DECIDE.format(
        sentence=sentence.strip(), span=span, candidates=candidates
    )

    try:
        answer = llm_service.call(
            PROMPT_SYSTEM_ACRONYM_DECIDE,
            user_prompt,
            name="acronym-decide",
            response_format=FORMAT_ACRONYM_DECIDE,
        )
    except Exception as exc:
        logger.warning("Acronym decision failed for '%s': %s", span, exc)
        return None, 0.0

    decision = _parse_json_object(answer)
    choice = decision.get("choix")
    if not choice:
        return None, 0.0

    try:
        confidence = float(decision.get("confiance") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return choice, confidence


def _rewrite_text(text: str, tokens: list[str], acronym: str) -> str:
    """Replace a word sequence inside a segment text, keeping punctuation."""
    if not tokens:
        return text
    pattern = r"\W+".join(re.escape(token) for token in tokens if token)
    if not pattern:
        return text
    return re.sub(pattern, acronym, text, count=1)


def _apply_correction(segment: dict[str, Any], correction: dict[str, Any]) -> bool:
    """Rewrite one word window of a segment into its acronym."""
    words = list(segment.get("words") or [])
    word_index = correction["word_index"]
    n_words = correction["n_words"]
    if word_index + n_words > len(words):
        return False

    replaced = words[word_index : word_index + n_words]
    scores = [word.get("score") for word in replaced if word.get("score") is not None]
    merged = {
        "word": correction["correct"],
        "start": replaced[0].get("start"),
        "end": replaced[-1].get("end"),
        "score": min(scores) if scores else None,
        "speaker": replaced[0].get("speaker"),
        "corrected": True,
    }

    segment["words"] = words[:word_index] + [merged] + words[word_index + n_words :]
    segment["text"] = _rewrite_text(
        segment.get("text", ""),
        [word.get("word", "").strip(" ,.;:!?") for word in replaced],
        correction["correct"],
    )
    return True


def _apply_to_word_segments(
    word_segments: list[dict[str, Any]], correction: dict[str, Any], merged: list[str]
) -> None:
    """Mirror an applied correction into the flat word_segments list."""
    for index in range(len(word_segments) - len(merged) + 1):
        window = [
            _normalize(word.get("word", ""))
            for word in word_segments[index : index + len(merged)]
        ]
        if window != [_normalize(token) for token in merged]:
            continue
        replaced = word_segments[index : index + len(merged)]
        scores = [
            word.get("score") for word in replaced if word.get("score") is not None
        ]
        word_segments[index : index + len(merged)] = [
            {
                "word": correction["correct"],
                "start": replaced[0].get("start"),
                "end": replaced[-1].get("end"),
                "score": min(scores) if scores else None,
                "speaker": replaced[0].get("speaker"),
                "corrected": True,
            }
        ]
        return


def _floor_for(acronym: str, user_acronyms: Optional[set[str]]) -> float:
    """The confidence an acronym must reach before it is written back.

    An organisation that uploaded an acronym has already vouched for it, so the
    bar is lower: a 0.75 decision on a declared acronym is better evidence than
    0.75 on one guessed from a public corpus. The reported confidence stays the
    model's own, so the audit trail is not inflated.
    """
    floor = settings.acronym_correction_min_confidence
    if user_acronyms and acronym in user_acronyms:
        floor -= settings.acronym_correction_user_glossary_relief
    return floor


def correct_acronyms(
    transcription: Any,
    llm_service,
    index: Optional[PhoneticIndex] = None,
    user_glossary: Optional[dict[str, str]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Correct the acronyms a speech recognition model got wrong.

    Args:
        transcription: a WhisperXResponse or the equivalent plain dict.
        llm_service: the LLMService used for the detection and decision stages.
        index: phonetic index to match against; defaults to the glossary
            shipped with the service, extended by `user_glossary`.
        user_glossary: an organisation's own {acronym: expansion} entries. They
            extend the shipped glossary and outrank it on a phonetic near-tie,
            since the uploader knows their own vocabulary.

    Returns:
        The corrected transcription as a dict, and the list of corrections.
        Every correction records its position, the wrong and correct text, the
        confidence reported by the LLM, and whether it was applied: corrections
        below `acronym_correction_min_confidence` are reported but left out of
        the transcription.
    """
    corrected = _as_dict(transcription)
    segments = corrected.get("segments") or []
    if not segments:
        return corrected, []

    phonetic_index = index if index is not None else build_index(user_glossary)
    user_acronyms = set(user_glossary) if user_glossary else None

    located: list[tuple[str, tuple[int, int, int]]] = []
    seen: set[tuple[int, int, int]] = set()
    for span in _detect_suspect_spans(segments, llm_service):
        position = _locate_span(segments, span)
        if position is None:
            logger.debug("Suspect span not located in the transcript: %s", span)
        elif position not in seen:
            seen.add(position)
            located.append((span, position))

    corrections: list[dict[str, Any]] = []
    for span, (segment_index, word_index, n_words) in located:
        words = list(segments[segment_index].get("words") or [])
        shortlist, where = _shortlist(
            phonetic_index, words, word_index, n_words, user_acronyms
        )
        if not shortlist:
            continue

        choice, confidence = _decide(
            segments[segment_index].get("text", ""),
            span,
            shortlist,
            llm_service,
            user_glossary,
        )
        if not choice or choice not in where:
            continue

        wrong, start, length = where[choice]
        corrections.append(
            {
                "segment_index": segment_index,
                "word_index": start,
                "n_words": length,
                "wrong": wrong,
                "correct": choice,
                "confidence": confidence,
                "applied": confidence >= _floor_for(choice, user_acronyms),
            }
        )

    word_segments = list(corrected.get("word_segments") or [])
    # Apply from the end so earlier word indices stay valid.
    for correction in sorted(
        (item for item in corrections if item["applied"]),
        key=lambda item: (-item["segment_index"], -item["word_index"]),
    ):
        segment = corrected["segments"][correction["segment_index"]]
        original = [
            word.get("word", "")
            for word in (segment.get("words") or [])[
                correction["word_index"] : correction["word_index"]
                + correction["n_words"]
            ]
        ]
        if _apply_correction(segment, correction) and word_segments:
            _apply_to_word_segments(word_segments, correction, original)

    if word_segments:
        corrected["word_segments"] = word_segments

    logger.info(
        "Acronym correction: %d correction(s) found, %d applied",
        len(corrections),
        sum(1 for correction in corrections if correction["applied"]),
    )
    return corrected, corrections
