"""Match a detected name span against the attendee roster.

Three layers, cheapest first, and the first one that fires wins:

  1. **exact**       -- the span, as written, is a CN, a first name or a
                        last name of the roster.
  2. **normalised**  -- same after lowercasing, stripping accents and
                        dropping anything that is not a letter.
  3. **fuzzy**       -- for names Whisper mangled.

Why this fuzzy layer
--------------------
`difflib.SequenceMatcher` alone and a phonetic key alone each fail on part
of case 03, in opposite ways, so we take `max(ratio_on_letters,
ratio_on_phonetic_key)`:

  * `sois raya` -> `Soraya`: the letter ratio is 0.86, but the phonetic key
    *hurts* here, because the French rule `oi -> wa` turns `soisraya` into
    `swasraya` and moves it away from `soraya`.
  * `vingt sang` -> `Vincent`: the letter ratio is only 0.63, because the
    silent letters of `vingt` have no counterpart in `vincent`. The nasal
    rules (`in`/`an` before a consonant collapse to a single symbol, and the
    silent consonant that follows a nasal is dropped) are what rescue it.

So neither is redundant. The phonetic key is a small hand-written
French-adapted transform rather than Soundex or Metaphone: both of those are
tuned for English and, more importantly, neither models French nasal vowels,
which is precisely what `vingt sang` needs. It is ~20 lines of `re.sub`, it
needs no dependency, and every rule it applies is one a French speaker can
name.

Mangled names cross word boundaries (`carie msahli` is two tokens for one
two-word name, `ma tilde` is two tokens for one first name), so the caller
offers spans of 1-3 consecutive tokens and we score each of them.

Ambiguity is a first-class answer: `Jean` matches two attendees exactly, and
the honest output is "two candidates", not a coin toss.
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

#: A span must reach this to be considered a fuzzy hit at all.
#:
#: 0.75 is the weakest link in the resolver, and knowingly so. Running every
#: transcript against every *other* case's invite -- 30 runs where any output
#: is wrong by construction -- produces two hits, both here: "Salima" reaches
#: 0.80 against "Sahli", and "Nourdine" reaches 0.80 against "Nourry".
#: Raising the threshold to 0.85 removes both and still scores 17/17 on the
#: six cases, but it leaves only 0.007 of margin on "carie msahli" and
#: "sois raya" (both 0.857). 0.75 keeps 0.107 of margin on the names we
#: actually have to recover, which is the better trade on this evidence.
#: `score.py --sweep` prints both columns.
FUZZY_MATCH_THRESHOLD = 0.75

#: Below this, a span is not a roster name in any reading of the evidence.
#: Between this and FUZZY_MATCH_THRESHOLD is the grey zone: near-miss, so the
#: caller must emit nothing rather than invent an out-of-roster person.
FUZZY_GREY_ZONE = 0.58

#: Two attendees whose best scores are this close are not distinguishable.
AMBIGUITY_EPSILON = 0.05


@dataclass
class Candidate:
    """One attendee the span could designate."""

    name: str
    email: str
    score: float
    tier: str  # "exact" | "normalised" | "fuzzy"


@dataclass
class SpanMatch:
    """The result of matching one span against the whole roster."""

    span: str
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        """The strongest candidate, or None when the span matched nobody."""
        return self.candidates[0] if self.candidates else None

    @property
    def is_ambiguous(self) -> bool:
        """True when two different attendees are within epsilon of the best."""
        if len(self.candidates) < 2:
            return False
        return (
            self.candidates[0].score - self.candidates[1].score
        ) <= AMBIGUITY_EPSILON


def strip_accents(text: str) -> str:
    """Return `text` without diacritics (é -> e, ç -> c)."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def normalise(text: str) -> str:
    """Lowercase, strip accents, keep letters only (spaces dropped)."""
    return re.sub(r"[^a-z]", "", strip_accents(text).lower())


def phonetic_key(text: str) -> str:
    """A small French-adapted phonetic key.

    Models what actually breaks Whisper transcriptions of French names:
    silent `h`, the two readings of `c` and `g`, the vowel digraphs, the
    three nasal vowels, and silent final consonants. Nasals become the
    digits 1/2/3 so they cannot be confused with the letters around them.
    """
    key = normalise(text)
    if not key:
        return ""

    key = key.replace("ph", "f")
    key = key.replace("ch", "X")  # a single sound; placeholder keeps it atomic
    key = key.replace("qu", "k").replace("q", "k")
    key = key.replace("h", "")
    key = re.sub(r"c([eiy])", r"s\1", key)
    key = key.replace("ck", "k").replace("c", "k")
    key = re.sub(r"g([eiy])", r"j\1", key)
    key = key.replace("eau", "o").replace("au", "o")
    key = key.replace("ai", "e").replace("ei", "e")
    key = key.replace("ou", "u")
    key = key.replace("oi", "wa")
    key = key.replace("y", "i")

    # Nasal vowels: a vowel + n/m that is NOT followed by another vowel or
    # nasal consonant ("vin" is nasal, "vine" and "vinne" are not).
    key = re.sub(r"(ain|ein|in|im|un|um)(?![aeiouwnm])", "1", key)
    key = re.sub(r"(an|am|en|em)(?![aeiouwnm])", "2", key)
    key = re.sub(r"(on|om)(?![aeiouwnm])", "3", key)

    # g/t/d right after a nasal are silent in French: vingt, long, sang, grand.
    # Deliberately narrow -- it must not eat a pronounced consonant, so `s`
    # and `p` are excluded (the `s` of "vincent" is heard).
    key = re.sub(r"([123])[gtd]+", r"\1", key)

    key = re.sub(r"[tdspxzg]+$", "", key)  # silent final consonants
    key = re.sub(r"e$", "", key)  # silent final e
    key = key.replace("z", "s")
    key = re.sub(r"(.)\1+", r"\1", key)  # double letters are one sound
    return key


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def similarity(span: str, name: str) -> float:
    """Letter-level and phonetic similarity of two strings, whichever is higher."""
    return max(
        _ratio(normalise(span), normalise(name)),
        _ratio(phonetic_key(span), phonetic_key(name)),
    )


def _keys_of(attendee: dict[str, str]) -> list[str]:
    """The strings of an attendee a speaker might actually say."""
    full = attendee["name"].strip()
    parts = [p for p in re.split(r"[\s-]+", full) if p]
    keys = [full]
    if len(parts) > 1:
        keys.append(parts[0])  # first name alone
        keys.append(parts[-1])  # last name alone
    return keys


def match_span(
    span: str,
    attendees: list[dict[str, str]],
    fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
) -> SpanMatch:
    """Match one span of transcript words against the roster.

    Returns every attendee the span could designate, best first. An empty
    candidate list means "not a roster name".
    """
    result = SpanMatch(span=span)
    if not span.strip():
        return result

    span_norm = normalise(span)
    if not span_norm:
        return result

    scored: list[Candidate] = []
    for attendee in attendees:
        best_score = 0.0
        best_tier = "fuzzy"
        for key in _keys_of(attendee):
            if span.strip() == key:
                score, tier = 1.0, "exact"
            elif span_norm == normalise(key):
                score, tier = 1.0, "normalised"
            else:
                score, tier = similarity(span, key), "fuzzy"
            if score > best_score or (
                score == best_score and tier != "fuzzy" and best_tier == "fuzzy"
            ):
                best_score, best_tier = score, tier
        if best_tier != "fuzzy" or best_score >= fuzzy_threshold:
            scored.append(
                Candidate(
                    name=attendee["name"],
                    email=attendee.get("email", ""),
                    score=best_score,
                    tier=best_tier,
                )
            )

    scored.sort(key=lambda c: (-c.score, c.name))
    result.candidates = scored
    return result


def best_roster_similarity(span: str, attendees: list[dict[str, str]]) -> float:
    """Highest similarity of `span` to any roster name, threshold ignored.

    Used to detect the grey zone: a span that nearly matches somebody is a
    reason to stay silent, not a reason to declare a new person.
    """
    best = 0.0
    for attendee in attendees:
        for key in _keys_of(attendee):
            best = max(best, similarity(span, key))
    return best
