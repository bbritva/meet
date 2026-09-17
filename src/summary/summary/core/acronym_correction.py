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

Stages 2 and 4 run on one of two paths, chosen per segment. A segment carrying
`words[]` -- meet-whisperx runs a forced alignment pass, so its segments do --
is located and rewritten word by word, and the correction inherits the timings
of the words it replaced. A segment carrying only `text` -- plain Whisper, as
served by Albert's transcription endpoint -- is tokenised from that text and
rewritten at character offsets. The text path reports no word-level timings,
but it is the only one that works without forced alignment.

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
from typing import Any, NamedTuple, Optional

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

# Punctuation that hangs off a token and is not part of the word it carries.
# WhisperX attaches it to the word ("nomme,"), and so does splitting a segment
# text on whitespace, so both paths strip the same set.
EDGE_PUNCTUATION = " ,.;:!?"

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
    return {acronym: entry[0] for acronym, entry in _load_glossary_file().items()}


def get_acronym_weights() -> dict[str, int]:
    """How many independent corpora confirmed each shipped acronym."""
    return {acronym: entry[1] for acronym, entry in _load_glossary_file().items()}


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


class _Position(NamedTuple):
    """Where a flagged passage sits, in whichever shape its segment has.

    `token_index` and `n_tokens` index the segment's tokens -- its `words[]`
    entries when `has_words`, otherwise the whitespace-separated chunks of its
    text. `has_words` is what picks the path at every later stage.
    """

    segment_index: int
    token_index: int
    n_tokens: int
    has_words: bool


def _text_token_spans(text: str) -> list[tuple[int, int]]:
    """Return the character span of every whitespace-separated token of a text.

    The single tokeniser of the text path: `_segment_tokens` reads the tokens
    through it and `_apply_text_correction` rewrites through it, so the tokens
    a window was matched on and the offsets it is spliced at cannot drift.
    """
    return [match.span() for match in re.finditer(r"\S+", text)]


def _segment_tokens(segment: dict[str, Any]) -> tuple[list[str], bool]:
    """Return a segment's tokens, and whether they came from per-word timings.

    Output of a recogniser with forced alignment carries `words[]`, and the
    tokens are those words. Plain Whisper output carries only `text`, and the
    tokens are its whitespace-separated chunks -- the same shape, punctuation
    still attached, which is why the two paths can share everything between
    locating and deciding.
    """
    words = segment.get("words") or []
    if words:
        return [word.get("word", "") or "" for word in words], True
    text = segment.get("text", "") or ""
    return [text[start:end] for start, end in _text_token_spans(text)], False


def _locate_span(segments: list[dict[str, Any]], span: str) -> Optional[_Position]:
    """Find a flagged passage among a segment's tokens.

    The LLM answers with text, not indices, and its punctuation and casing
    rarely match the tokens exactly, so the match is done on normalised token
    sequences, with the longest contiguous run as a fallback.

    One token can normalise to several words -- `_normalize("d'Inhomme")` is
    "d inhomme" -- so the comparison is made on the flattened stream. Without
    that, a span Whisper glued an elided article onto never locates at all.

    Args:
        segments: the transcription segments.
        span: the passage reported by the detection stage.

    Returns:
        The position of the passage, or None when it cannot be found.
    """
    targets = [token for token in _normalize(span).split() if token]
    if not targets:
        return None

    best: Optional[_Position] = None
    best_matched = 0

    for segment_index, segment in enumerate(segments):
        tokens, has_words = _segment_tokens(segment)
        normalized = [_normalize(token).split() for token in tokens]

        for index in range(len(tokens)):
            for length in range(1, min(MAX_LOCATE_WORDS, len(tokens) - index) + 1):
                window = [
                    word
                    for token in normalized[index : index + length]
                    for word in token
                ]
                if window == targets:
                    return _Position(segment_index, index, length, has_words)

        for index in range(len(tokens)):
            length = 0
            matched = 0
            while index + length < len(tokens) and matched < len(targets):
                words = normalized[index + length]
                if targets[matched : matched + len(words)] != words:
                    break
                matched += len(words)
                length += 1
            if matched > best_matched:
                best = _Position(segment_index, index, length, has_words)
                best_matched = matched

    return best


def _window_candidates(
    index: PhoneticIndex, tokens: list[str], low: int, high: int
) -> list[tuple[int, int, str, list[tuple[str, float]]]]:
    """Score every token window of a region against the glossary."""
    scored = []
    for start in range(low, high):
        for length in range(1, min(MAX_WINDOW_WORDS, high - start) + 1):
            stripped = [
                token.strip(EDGE_PUNCTUATION)
                for token in tokens[start : start + length]
            ]
            text = " ".join(token for token in stripped if token)
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
    tokens: list[str],
    token_index: int,
    n_tokens: int,
    user_acronyms: Optional[set[str]] = None,
) -> tuple[list[tuple[str, float]], dict[str, tuple[str, int, int]]]:
    """Build one ranked shortlist for the region around a flagged passage.

    Takes plain token texts, so it is shared by both paths: the tokens are the
    segment's words when it has per-word timings, and the whitespace-separated
    chunks of its text when it does not.

    Windows are claimed longest first: once a window owns its token positions,
    shorter overlapping windows are skipped. Without this maximal munch, "dix"
    matching DI at 1.00 outranks "dix nomme" matching DINUM at 0.80 and splits
    the acronym in two.

    Returns:
        The ranked (acronym, similarity) shortlist, and a mapping from acronym
        to the (text, token index, token count) window it matched.
    """
    low = max(0, token_index - WINDOW_MARGIN)
    high = min(len(tokens), token_index + n_tokens + WINDOW_MARGIN)

    windows = _window_candidates(index, tokens, low, high)
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


def _apply_word_correction(
    segment: dict[str, Any], position: _Position, acronym: str
) -> bool:
    """Merge a window of timed words into one corrected word, and fix the text.

    The merged word spans the replaced ones, so the correction keeps the
    timings the alignment pass produced.
    """
    words = list(segment.get("words") or [])
    if position.n_tokens < 1 or position.token_index + position.n_tokens > len(words):
        return False

    replaced = words[position.token_index : position.token_index + position.n_tokens]
    scores = [word.get("score") for word in replaced if word.get("score") is not None]
    merged = {
        "word": acronym,
        "start": replaced[0].get("start"),
        "end": replaced[-1].get("end"),
        "score": min(scores) if scores else None,
        "speaker": replaced[0].get("speaker"),
        "corrected": True,
    }

    segment["words"] = (
        words[: position.token_index]
        + [merged]
        + words[position.token_index + position.n_tokens :]
    )
    segment["text"] = _rewrite_text(
        segment.get("text", ""),
        [word.get("word", "").strip(EDGE_PUNCTUATION) for word in replaced],
        acronym,
    )
    return True


def _apply_text_correction(
    segment: dict[str, Any], position: _Position, acronym: str
) -> bool:
    """Splice a token window of a segment text into its acronym.

    There are no words to merge on this path, so the whole edit is on the text.
    The window is addressed by character offset rather than by pattern, because
    the tokens were read from this very text: the offsets are exact, where a
    pattern would rewrite the first identical passage instead of this one. The
    punctuation hanging off the window is kept, so "d'Inhomme," becomes
    "DINUM," and not "DINUM".
    """
    text = segment.get("text", "") or ""
    spans = _text_token_spans(text)
    last = position.token_index + position.n_tokens - 1
    if position.n_tokens < 1 or last >= len(spans):
        return False

    start = spans[position.token_index][0]
    end = spans[last][1]
    while start < end and text[start] in EDGE_PUNCTUATION:
        start += 1
    while end > start and text[end - 1] in EDGE_PUNCTUATION:
        end -= 1

    segment["text"] = text[:start] + acronym + text[end:]
    return True


def _apply_correction(
    segment: dict[str, Any], position: _Position, acronym: str
) -> bool:
    """Rewrite one located window of a segment into its acronym."""
    if position.has_words:
        return _apply_word_correction(segment, position, acronym)
    return _apply_text_correction(segment, position, acronym)


def _apply_to_word_segments(
    word_segments: list[dict[str, Any]], acronym: str, merged: list[str]
) -> None:
    """Mirror an applied correction into the flat word_segments list.

    Only the word path calls this. A segment without per-word timings has no
    words anywhere, so there is nothing to mirror.
    """
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
                "word": acronym,
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
        Every correction records the wrong and correct text, the confidence
        reported by the LLM, and whether it was applied: corrections below
        `acronym_correction_min_confidence` are reported but left out of the
        transcription. `word_index` and `n_words` locate the correction in the
        segment's words, and are None for a correction found on the text path,
        where the segment carried no per-word timings to index into.
    """
    corrected = _as_dict(transcription)
    segments = corrected.get("segments") or []
    if not segments:
        return corrected, []

    # Whisper served without forced alignment (Albert's endpoint, for one)
    # returns segments with no per-word timings. The text path handles them, but
    # the corrections it finds carry no timings of their own, so say so rather
    # than let a caller wonder where they went.
    if not any(segment.get("words") for segment in segments):
        logger.info(
            "No per-word timings in this transcription (%d segments): it came "
            "from a speech recogniser without forced alignment. Acronyms are "
            "matched on the segment text instead, and the corrections carry no "
            "word-level timings.",
            len(segments),
        )

    phonetic_index = index if index is not None else build_index(user_glossary)
    user_acronyms = set(user_glossary) if user_glossary else None

    located: list[tuple[str, _Position]] = []
    seen: set[_Position] = set()
    for span in _detect_suspect_spans(segments, llm_service):
        position = _locate_span(segments, span)
        if position is None:
            logger.debug("Suspect span not located in the transcript: %s", span)
        elif position not in seen:
            seen.add(position)
            located.append((span, position))

    # Each correction is kept next to the position it was matched at: the
    # position drives the rewrite, while the correction is the audit trail
    # handed back to the caller.
    found: list[tuple[dict[str, Any], _Position]] = []
    for span, position in located:
        segment = segments[position.segment_index]
        tokens, _ = _segment_tokens(segment)
        shortlist, where = _shortlist(
            phonetic_index,
            tokens,
            position.token_index,
            position.n_tokens,
            user_acronyms,
        )
        if not shortlist:
            continue

        choice, confidence = _decide(
            segment.get("text", ""), span, shortlist, llm_service, user_glossary
        )
        if not choice or choice not in where:
            continue

        wrong, start, length = where[choice]
        found.append(
            (
                {
                    "segment_index": position.segment_index,
                    "word_index": start if position.has_words else None,
                    "n_words": length if position.has_words else None,
                    "wrong": wrong,
                    "correct": choice,
                    "confidence": confidence,
                    "applied": confidence >= _floor_for(choice, user_acronyms),
                },
                position._replace(token_index=start, n_tokens=length),
            )
        )

    corrections = [correction for correction, _ in found]
    word_segments = list(corrected.get("word_segments") or [])
    # Apply from the end so the token indices of earlier windows stay valid --
    # on the text path, so do their character offsets.
    for correction, position in sorted(
        (item for item in found if item[0]["applied"]),
        key=lambda item: (-item[1].segment_index, -item[1].token_index),
    ):
        segment = corrected["segments"][position.segment_index]
        original = [
            word.get("word", "")
            for word in (segment.get("words") or [])[
                position.token_index : position.token_index + position.n_tokens
            ]
        ]
        applied = _apply_correction(segment, position, correction["correct"])
        if applied and position.has_words and word_segments:
            _apply_to_word_segments(word_segments, correction["correct"], original)

    if word_segments:
        corrected["word_segments"] = word_segments

    logger.info(
        "Acronym correction: %d correction(s) found, %d applied",
        len(corrections),
        sum(1 for correction in corrections if correction["applied"]),
    )
    return corrected, corrections
