"""French phonetic matching, used to recover mistranscribed acronyms.

Speech recognition substitutes fluent French for acronyms it does not know:
"dix nomme" for DINUM, "quenil" for CNIL. Those are far apart in edit distance
but near-identical in sound, so matching has to happen on pronunciation.

Two pronunciations are indexed for every acronym:

- word-like, when the acronym is read as a word (DINUM -> /dinom/)
- spelled out, letter by letter (RGPD -> "erre ge pe de" -> /ERZePeDe/)

Nasal vowels are indexed both as nasals and as vowel + n, because the ASR
output rarely agrees with the glossary on that distinction.

Implemented with the standard library only: the summary service ships no
phonetic dependency.
"""

import re
import unicodedata

# Pronunciation of each letter of the alphabet, used to spell acronyms out.
_LETTER = {
    "a": "a",
    "b": "be",
    "c": "se",
    "d": "de",
    "e": "e",
    "f": "ef",
    "g": "je",
    "h": "aS",
    "i": "i",
    "j": "ji",
    "k": "ka",
    "l": "el",
    "m": "em",
    "n": "en",
    "o": "o",
    "p": "pe",
    "q": "ku",
    "r": "er",
    "s": "es",
    "t": "te",
    "u": "u",
    "v": "ve",
    "w": "dublEve",
    "x": "iks",
    "y": "igrek",
    "z": "zed",
}

# Ordered rewrite rules, French orthography -> coarse phonemes. Upper-case
# symbols stand for nasals and digraphs so they never collide with a plain
# letter of the alphabet.
_RULES = [
    (r"eaux?", "o"),
    (r"aux?", "o"),
    (r"eau", "o"),
    (r"ou", "u"),
    (r"oi", "wa"),
    (r"ai|ei", "e"),
    (r"eu|oeu|oe", "e"),
    (r"(ain|ein|aim|eim)(?![aeiouy])", "E"),
    (r"(in|im|yn|ym)(?![aeiouynm])", "E"),
    (r"(un|um)(?![aeiouynm])", "E"),
    (r"(an|am|en|em)(?![aeiouynm])", "A"),
    (r"(on|om)(?![aeiouynm])", "O"),
    (r"ch", "S"),
    (r"ph", "f"),
    (r"gn", "N"),
    (r"qu", "k"),
    (r"q", "k"),
    (r"c(?=[eiy])", "s"),
    (r"c", "k"),
    (r"ss", "s"),
    (r"g(?=[eiy])", "j"),
    (r"(?<=[aeiouy])s(?=[aeiouy])", "z"),
    (r"h", ""),
    (r"y", "i"),
    (r"x", "ks"),
]

_NASAL_SYMBOLS = ("A", "E", "O")

# Minimum number of letters for an acronym to be indexed as a word. Shorter
# ones sound like half the French dictionary and only add noise.
MIN_WORD_LIKE_LENGTH = 3

# Above this difference in key length, two keys cannot be close enough to be
# worth a Levenshtein computation.
_MAX_LENGTH_GAP = 3


def _strip_accents(text: str) -> str:
    """Lower-case text and remove its diacritics."""
    lowered = text.lower().replace("’", "'")
    return "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )


def phon(text: str, nasal: bool = True) -> str:
    """Return a coarse French pronunciation key for a piece of text.

    Args:
        text: the text to phonemise.
        nasal: when False, "an" / "in" / "on" are kept as vowel + n instead of
            being collapsed into a nasal symbol.

    Returns:
        The pronunciation key, or an empty string when the text holds no
        pronounceable letter.
    """
    cleaned = _strip_accents(text)
    cleaned = re.sub(r"[^a-z' ]", " ", cleaned).replace("'", " ")

    rules = (
        _RULES
        if nasal
        else [(pattern, rep) for pattern, rep in _RULES if rep not in _NASAL_SYMBOLS]
    )

    keys = []
    for word in cleaned.split():
        key = word
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        # Silent final consonants and mute "e".
        key = re.sub(r"(ks|e|es|ent|s|t|d|x|z|p|g)$", "", key)
        key = re.sub(r"er$|ez$", "e", key)
        key = re.sub(r"(.)\1+", r"\1", key)
        keys.append(key)

    return re.sub(r"(.)\1+", r"\1", "".join(keys))


def keyset(text: str) -> set[str]:
    """Return every plausible pronunciation key for a piece of text.

    French elision is the reason this is not a single key. Whisper writes the
    elided article onto the acronym -- "d'Inhomme" for DINUM -- and splitting on
    the apostrophe gives "d inhomme", which scores 0.60 against DINUM and falls
    under the similarity floor. Glued together it scores 0.80. Both readings are
    generated because neither is right in general: gluing "l'ANOM" into "lanom"
    would be just as wrong.
    """
    keys: set[str] = set()
    for variant in (text, text.replace("\u2019", "'").replace("'", "")):
        keys.add(phon(variant, True))
        keys.add(phon(variant, False))
    return {key for key in keys if key}


def acronym_keys(acronym: str) -> set[str]:
    """Return every plausible pronunciation key for an acronym.

    Covers the spelled-out reading ("RGPD" said "erre ge pe de") and, when the
    acronym is long enough and holds a vowel, the word-like reading ("DINUM"
    said "dinom").
    """
    letters = [char for char in _strip_accents(acronym) if char.isalpha()]
    keys = keyset(" ".join(_LETTER.get(letter, letter) for letter in letters))

    core = re.sub(r"[^a-zA-Z]", "", acronym)
    if len(core) >= MIN_WORD_LIKE_LENGTH and re.search(
        r"[aeiouy]", core, re.IGNORECASE
    ):
        keys |= keyset(core)

    return {key for key in keys if key}


def _levenshtein(left: str, right: str) -> int:
    """Return the edit distance between two strings."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    """Return a 0..1 similarity between two pronunciation keys."""
    if not left or not right:
        return 0.0
    return 1.0 - _levenshtein(left, right) / max(len(left), len(right))


def spans(words: list[str], max_words: int = 4):
    """Yield every 1..max_words window over a word list.

    Args:
        words: the words to window over.
        max_words: the longest window to produce.

    Yields:
        (start index, window length, joined text) tuples.
    """
    for start in range(len(words)):
        for length in range(1, max_words + 1):
            if start + length <= len(words):
                yield start, length, " ".join(words[start : start + length])


class PhoneticIndex:
    """Reverse index from pronunciation key to the acronyms that sound like it."""

    def __init__(self, glossary: dict[str, str]):
        """Build the index from an {acronym: expansion} glossary."""
        self.by_key: dict[str, set[str]] = {}
        for acronym in glossary:
            for key in acronym_keys(acronym):
                self.by_key.setdefault(key, set()).add(acronym)

    def candidates(
        self, text: str, top_k: int = 5, min_similarity: float = 0.62
    ) -> list[tuple[str, float]]:
        """Return the acronyms that sound closest to a piece of text.

        Args:
            text: the transcribed span to match.
            top_k: how many candidates to keep.
            min_similarity: similarity below which a candidate is dropped.

        Returns:
            (acronym, similarity) pairs, best first.
        """
        queries = keyset(text)
        if not queries:
            return []

        scored: dict[str, float] = {}
        for key, acronyms in self.by_key.items():
            if all(abs(len(key) - len(query)) > _MAX_LENGTH_GAP for query in queries):
                continue
            score = max(similarity(query, key) for query in queries)
            if score < min_similarity:
                continue
            for acronym in acronyms:
                if score > scored.get(acronym, 0.0):
                    scored[acronym] = score

        ranked = sorted(scored.items(), key=lambda item: (-item[1], len(item[0])))
        return ranked[:top_k]
