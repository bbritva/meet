"""Detect name cues in French transcript segments.

Three kinds, and keeping them apart is the whole point:

  ``self_id``   "Ici Camille", "Camille à l'appareil", "Bonjour, c'est
                Thomas", "Thomas au micro", "Gaël Prigent, bonjour".
                Labels the speaker of the CURRENT segment. Strongest.
  ``handoff``   "Julien, tu démarres ?", "Bruno, tu peux…", "Sophie, à toi".
                Suggests the NEXT different speaker. Weaker: the person
                addressed may say nothing.
  ``mention``   any other occurrence of a roster name. Tells you nothing
                about who is speaking and MUST NOT produce an attribution.

Two traps drive the design.

**Anchoring.** A name span is only accepted when it matches the attendee
roster (see `match.py`). Extracting "the capitalised word after c'est" finds
"C'est un", "C'est le", "C'est exactement" all day long. The single
exception is a strong self-identification whose name is nowhere on the
roster -- a late joiner -- and that path is deliberately narrow: strongest
templates only, capitalised but not an acronym, and silent whenever the span
is a near-miss on a roster name (a near-miss means "unsure", never "new
person").

**First-person substitution.** "Je remplace Léa", "au nom de Léa", "à la
place de Léa" all put a roster name in the mouth of someone who is *not*
that person. Attributing Léa there is the single most expensive error in the
corpus, so any self-identification whose name is preceded by substitution
phrasing is demoted to ``mention``.
"""

# The detector is a long linear scan over one closed list of French
# templates. Splitting it to satisfy a complexity budget would scatter
# rules that are only correct read in order, so it stays as measured.
# ruff: noqa: PLR0912, PLR0913, PLR0915, PLR0917

import re
import unicodedata
from dataclasses import dataclass, field

from summary.core.speaker_cues.match import (
    FUZZY_GREY_ZONE,
    FUZZY_MATCH_THRESHOLD,
    best_roster_similarity,
    match_span,
    normalise,
)

SELF_ID = "self_id"
HANDOFF = "handoff"
MENTION = "mention"

_PUNCT = '.,;:!?…«»"()[]–—'
#: Punctuation that ends a sentence, as opposed to merely separating inside
#: one. The difference matters: a template may reach across the comma of
#: "Gaël Prigent, bonjour", but never across the full stop of "...parlé à
#: Damien. Bonjour à tous." -- the name there belongs to the other sentence.
_HARD_PUNCT = ".!?…"
_APOSTROPHES = "’ʼ‘"

#: A fuzzy threshold no score can reach: mentions match exactly or not at all.
_EXACT_ONLY = 1.01

#: Words that cannot start a name. This is the "C'est un / le / la" guard.
_FUNCTION_WORDS = {
    "le",
    "la",
    "les",
    "l",
    "un",
    "une",
    "des",
    "du",
    "de",
    "d",
    "ce",
    "cet",
    "cette",
    "ces",
    "c",
    "mon",
    "ma",
    "mes",
    "ton",
    "ta",
    "tes",
    "son",
    "sa",
    "ses",
    "notre",
    "nos",
    "votre",
    "vos",
    "leur",
    "leurs",
    "je",
    "j",
    "tu",
    "il",
    "elle",
    "on",
    "nous",
    "vous",
    "ils",
    "elles",
    "me",
    "te",
    "se",
    "y",
    "en",
    "que",
    "qu",
    "qui",
    "quoi",
    "dont",
    "ou",
    "et",
    "mais",
    "donc",
    "or",
    "ni",
    "car",
    "ne",
    "n",
    "pas",
    "plus",
    "moins",
    "tres",
    "bien",
    "deja",
    "encore",
    "toujours",
    "jamais",
    "aussi",
    "alors",
    "apres",
    "avant",
    "pour",
    "par",
    "sur",
    "sous",
    "dans",
    "avec",
    "sans",
    "chez",
    "vers",
    "entre",
    "a",
    "au",
    "aux",
    "si",
    "tout",
    "toute",
    "tous",
    "toutes",
    "meme",
    "comme",
    "quand",
    "lors",
    "ainsi",
    "exactement",
    "simplement",
    "vraiment",
    "surtout",
    "peut",
    "etre",
    "cela",
    "ca",
    "rien",
    "trop",
    "assez",
    "autre",
    "autres",
    "beaucoup",
    "possible",
    "normal",
    "clair",
    "faux",
    "vrai",
}

#: Phrasings that put somebody else's name in the current speaker's mouth.
_SUBSTITUTION = (
    "je remplace",
    "je le remplace",
    "je la remplace",
    "en remplacement de",
    "a la place de",
    "au nom de",
    "de la part de",
    "pour le compte de",
    "je represente",
    "je parle pour",
    "je parle au nom",
    "je supplee",
    "je viens a la place",
    "je prends la suite de",
    "je reprends le dossier de",
)

_MAX_SPAN_TOKENS = 3
_SUBSTITUTION_WINDOW = 8


@dataclass
class Token:
    """One transcript word, punctuation stripped off but remembered."""

    raw: str  # original case, no surrounding punctuation
    norm: str  # lowercase, accents stripped, apostrophes unified
    index: int
    brk_before: bool = False
    brk_after: bool = False
    hard_before: bool = False  # the break is a sentence end, not a comma
    hard_after: bool = False


@dataclass
class Cue:
    """One detected name cue."""

    kind: str
    segment_index: int
    speaker: str
    span: str
    quote: str
    trigger: str
    name: str | None = None
    email: str = ""
    tier: str = "none"
    similarity: float = 0.0
    ambiguous: bool = False
    candidates: list[str] = field(default_factory=list)
    reason: str = ""


def _is_hard(punctuation: str) -> bool:
    return any(char in _HARD_PUNCT for char in punctuation)


def _unify(text: str) -> str:
    for apostrophe in _APOSTROPHES:
        text = text.replace(apostrophe, "'")
    return text


def tokenize(text: str) -> list[Token]:
    """Split a segment into tokens, treating punctuation as a hard boundary.

    Punctuation is attached to the word in some segments ("Thomas.") and a
    standalone token in others ("Nourdine , désolé") because French
    typography puts a space before `?`, `!`, `:` and `;`. Both shapes have
    to produce the same tokens, or the exact-match tier never fires.
    """
    tokens: list[Token] = []
    pending_break = True  # start of segment counts as a boundary
    pending_hard = True
    for piece in _unify(text).split():
        stripped = piece.strip(_PUNCT)
        if not stripped:
            pending_break = True
            pending_hard = pending_hard or _is_hard(piece)
            if tokens:
                tokens[-1].brk_after = True
                tokens[-1].hard_after = tokens[-1].hard_after or _is_hard(piece)
            continue
        lead = piece[: len(piece) - len(piece.lstrip(_PUNCT))]
        trail = piece[len(piece.rstrip(_PUNCT)) :]
        token = Token(
            raw=stripped,
            norm=re.sub(r"[^a-z']", "", _strip_accents_lower(stripped)),
            index=len(tokens),
            brk_before=pending_break or bool(lead),
            brk_after=bool(trail),
            hard_before=pending_hard or _is_hard(lead),
            hard_after=_is_hard(trail),
        )
        tokens.append(token)
        pending_break = bool(trail)
        pending_hard = _is_hard(trail)
    return tokens


def _strip_accents_lower(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _spans_after(tokens: list[Token], start: int) -> list[tuple[int, int]]:
    """Spans of 1..3 tokens starting at `start`, stopping at a break/function word."""
    spans: list[tuple[int, int]] = []
    end = start
    while end < len(tokens) and end - start < _MAX_SPAN_TOKENS:
        token = tokens[end]
        if token.norm in _FUNCTION_WORDS:
            break
        if end > start and tokens[end - 1].brk_after:
            break
        spans.append((start, end + 1))
        if token.brk_after:
            break
        end += 1
    return spans


def _spans_before(tokens: list[Token], stop: int) -> list[tuple[int, int]]:
    """Spans of 1..3 tokens ending just before `stop`.

    The *soft* break between the span and the trigger is part of the template
    ("Gaël Prigent, bonjour"), so it is ignored; a break *inside* the span
    still ends it. A sentence end is never skipped, or the mention in
    "J'en ai parlé à Damien. Bonjour à tous." would read as Damien
    introducing himself. Function words are not a stop condition here,
    because "ma tilde" -- Whisper's rendering of "Mathilde" -- starts with one.
    """
    spans: list[tuple[int, int]] = []
    if stop < 1 or tokens[stop - 1].hard_after:
        return spans
    start = stop - 1
    while start >= 0 and stop - start <= _MAX_SPAN_TOKENS:
        if start < stop - 1 and tokens[start].brk_after:
            break
        spans.append((start, stop))
        if tokens[start].brk_before:
            break
        start -= 1
    return spans


def _text_of(tokens: list[Token], span: tuple[int, int]) -> str:
    return " ".join(t.raw for t in tokens[span[0] : span[1]])


def _quote(tokens: list[Token], lo: int, hi: int) -> str:
    lo = max(0, lo)
    hi = min(len(tokens), hi)
    return " ".join(t.raw for t in tokens[lo:hi])


def _is_substituted(tokens: list[Token], span_start: int) -> bool:
    """True when the name at `span_start` is introduced as somebody else's.

    The window stops at the previous sentence end: "Je remplace Bruno. Ici
    Sylvie." is a substitution followed by a perfectly good self-id, and the
    first sentence must not poison the second.
    """
    window_start = max(0, span_start - _SUBSTITUTION_WINDOW)
    for position in range(span_start - 1, window_start - 1, -1):
        if tokens[position].hard_after:
            window_start = position + 1
            break
    before = " ".join(t.norm for t in tokens[window_start:span_start])
    return any(phrase in before for phrase in _SUBSTITUTION)


def _looks_like_a_name(token: Token) -> bool:
    """Capitalised, not an acronym, long enough to be a first name."""
    raw = token.raw
    if len(raw) < 3 or not raw[0].isupper():
        return False
    if raw.isupper():  # DINUM, ANSSI, RGPD, DSFR, PCA
        return False
    return all(c.isalpha() or c in "-'" for c in raw)


def _resolve_span(
    tokens: list[Token],
    spans: list[tuple[int, int]],
    attendees: list[dict[str, str]],
    fuzzy_threshold: float,
) -> tuple[tuple[int, int] | None, object]:
    """Pick the best-matching span among `spans`. Returns (span, SpanMatch)."""
    best_span = None
    best_match = None
    for span in spans:
        text = _text_of(tokens, span)
        match = match_span(text, attendees, fuzzy_threshold=fuzzy_threshold)
        if not match.candidates:
            continue
        score = match.candidates[0].score
        if (
            best_match is None
            or score > best_match.candidates[0].score
            or (
                score == best_match.candidates[0].score
                and (span[1] - span[0]) > (best_span[1] - best_span[0])
            )
        ):
            best_span, best_match = span, match
    return best_span, best_match


def _make_cue(
    kind: str,
    segment_index: int,
    speaker: str,
    tokens: list[Token],
    span: tuple[int, int],
    match: object,
    trigger: str,
    quote_lo: int,
    quote_hi: int,
) -> Cue:
    best = match.candidates[0]
    return Cue(
        kind=kind,
        segment_index=segment_index,
        speaker=speaker,
        span=_text_of(tokens, span),
        quote=_quote(tokens, quote_lo, quote_hi),
        trigger=trigger,
        name=best.name,
        email=best.email,
        tier=best.tier,
        similarity=best.score,
        ambiguous=match.is_ambiguous,
        candidates=[c.name for c in match.candidates],
    )


def detect_cues(
    segments: list[dict],
    attendees: list[dict[str, str]],
    fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
    allow_unknown_self_id: bool = True,
) -> list[Cue]:
    """Return every cue found in `segments`, in reading order."""
    cues: list[Cue] = []
    for index, segment in enumerate(segments):
        cues.extend(
            _detect_in_segment(
                index,
                segment.get("speaker") or "",
                tokenize(segment.get("text") or ""),
                attendees,
                fuzzy_threshold,
                allow_unknown_self_id,
            )
        )
    return cues


def _detect_in_segment(
    index: int,
    speaker: str,
    tokens: list[Token],
    attendees: list[dict[str, str]],
    fuzzy_threshold: float,
    allow_unknown_self_id: bool,
) -> list[Cue]:
    cues: list[Cue] = []
    consumed: set[int] = set()

    def take(span: tuple[int, int]) -> None:
        consumed.update(range(span[0], span[1]))

    for position in range(len(tokens)):
        trigger = _self_id_trigger(tokens, position)
        if trigger is None:
            continue
        direction, trigger_name, trigger_end = trigger
        if direction == "after":
            spans = _spans_after(tokens, trigger_end)
            quote_lo, quote_hi = position, trigger_end + _MAX_SPAN_TOKENS
        else:
            spans = _spans_before(tokens, position)
            quote_lo, quote_hi = position - _MAX_SPAN_TOKENS, trigger_end

        span, match = _resolve_span(tokens, spans, attendees, fuzzy_threshold)

        if span is not None:
            if _is_substituted(tokens, span[0]):
                cue = _make_cue(
                    MENTION,
                    index,
                    speaker,
                    tokens,
                    span,
                    match,
                    trigger_name,
                    quote_lo,
                    quote_hi,
                )
                cue.reason = "first-person substitution: the speaker is not this person"
                cues.append(cue)
                take(span)
                continue
            cues.append(
                _make_cue(
                    SELF_ID,
                    index,
                    speaker,
                    tokens,
                    span,
                    match,
                    trigger_name,
                    quote_lo,
                    quote_hi,
                )
            )
            take(span)
            continue

        unknown = _unknown_self_id(
            tokens, spans, attendees, trigger_name, allow_unknown_self_id
        )
        if unknown is not None:
            unknown_span, reason = unknown
            if reason:
                continue
            if _is_substituted(tokens, unknown_span[0]):
                continue
            cues.append(
                Cue(
                    kind=SELF_ID,
                    segment_index=index,
                    speaker=speaker,
                    span=_text_of(tokens, unknown_span),
                    quote=_quote(tokens, quote_lo, quote_hi),
                    trigger=trigger_name,
                    name=_text_of(tokens, unknown_span),
                    email="",
                    tier="out_of_roster",
                    similarity=1.0,
                )
            )
            take(unknown_span)

    for position in range(len(tokens)):
        trigger_name = _handoff_trigger(tokens, position)
        if trigger_name is None:
            continue
        spans = _spans_before(tokens, position)
        if not spans or not tokens[position - 1].brk_after:
            # The name must be adjacent to the pronoun and set off by
            # punctuation: "Julien, tu ...". Administrative French is full of
            # "je vous propose" / "à votre avis".
            continue
        span, match = _resolve_span(tokens, spans, attendees, fuzzy_threshold)
        if span is None or any(i in consumed for i in range(span[0], span[1])):
            continue
        if match.candidates[0].tier == "fuzzy":
            # A handoff is already indirect evidence; a mangled name on top
            # of it is not worth the precision it would cost.
            continue
        cues.append(
            _make_cue(
                HANDOFF,
                index,
                speaker,
                tokens,
                span,
                match,
                trigger_name,
                span[0],
                position + 3,
            )
        )
        take(span)

    for span in _remaining_name_spans(tokens, consumed, attendees):
        match = match_span(
            _text_of(tokens, span), attendees, fuzzy_threshold=_EXACT_ONLY
        )
        if not match.candidates:
            continue
        cue = _make_cue(
            MENTION,
            index,
            speaker,
            tokens,
            span,
            match,
            "third-person",
            max(0, span[0] - 3),
            span[1] + 3,
        )
        cue.reason = "third-person mention: proves the person exists, nothing else"
        cues.append(cue)

    return cues


def _self_id_trigger(tokens: list[Token], position: int) -> tuple[str, str, int] | None:
    """Return (direction, template name, end index) if a self-id template fires."""
    token = tokens[position]
    nxt = tokens[position + 1] if position + 1 < len(tokens) else None

    if token.norm == "ici" and token.brk_before:
        return "after", "ici <nom>", position + 1
    if token.norm in ("c'est", "cest") and token.brk_before:
        return "after", "c'est <nom>", position + 1
    if token.norm == "a" and nxt is not None and nxt.norm == "l'appareil":
        return "before", "<nom> à l'appareil", position + 2
    if token.norm == "au" and nxt is not None and nxt.norm == "micro":
        return "before", "<nom> au micro", position + 2
    if token.norm in ("bonjour", "bonsoir") and position > 0 and token.brk_before:
        return "before", "<nom>, bonjour", position + 1
    return None


def _handoff_trigger(tokens: list[Token], position: int) -> str | None:
    """Return the template name if a handoff template fires at `position`."""
    if position == 0:
        return None
    token = tokens[position]
    nxt = tokens[position + 1] if position + 1 < len(tokens) else None

    if token.norm in ("tu", "vous") and nxt is not None:
        return "<nom>, %s ..." % token.norm
    if token.norm == "a" and nxt is not None and nxt.norm in ("toi", "vous"):
        return "<nom>, à %s" % nxt.norm
    if token.norm in ("vas-y", "vasy", "allez-y"):
        return "<nom>, %s" % token.norm
    return None


def _unknown_self_id(
    tokens: list[Token],
    spans: list[tuple[int, int]],
    attendees: list[dict[str, str]],
    trigger_name: str,
    allowed: bool,
) -> tuple[tuple[int, int], str] | None:
    """A strong self-id whose name is nowhere on the roster (a late joiner).

    Narrow on purpose: this is the largest precision liability in the whole
    resolver, serving exactly one speaker in the corpus.
    """
    if not allowed or not spans:
        return None
    if trigger_name not in ("ici <nom>", "c'est <nom>", "<nom> à l'appareil"):
        return None
    span = spans[0]
    if span[1] - span[0] != 1:
        span = (span[0], span[0] + 1)
    token = tokens[span[0]]
    if not _looks_like_a_name(token):
        return None
    similarity = best_roster_similarity(token.raw, attendees)
    if similarity >= FUZZY_GREY_ZONE:
        # Near-miss on somebody real: stay silent rather than invent a person.
        return span, "grey zone (%.2f) against the roster" % similarity
    return span, ""


def _remaining_name_spans(
    tokens: list[Token], consumed: set[int], attendees: list[dict[str, str]]
) -> list[tuple[int, int]]:
    """1-2 token spans not already used by a stronger cue."""
    spans: list[tuple[int, int]] = []
    for start in range(len(tokens)):
        if start in consumed:
            continue
        for length in (2, 1):
            end = start + length
            if end > len(tokens):
                continue
            if any(i in consumed for i in range(start, end)):
                continue
            if length == 2 and tokens[start].brk_after:
                continue
            if normalise(_text_of(tokens, (start, end))):
                spans.append((start, end))
    return spans
