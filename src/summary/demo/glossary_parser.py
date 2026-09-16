"""Read a plain-text acronym glossary uploaded by a user.

Demo scaffolding. Nothing here is part of the acronym feature: it only turns
an uploaded file into the `{acronym: expansion}` mapping that
`correct_acronyms(user_glossary=...)` already accepts.

Tolerant on purpose. Somebody preparing a glossary on stage should not have to
remember which separator we chose, so `=`, `;` and `:` all work, in the same
file. Blank lines and `#` comments are skipped, whitespace is trimmed.

Strict on one point: text only. A `.docx` or a `.xlsx` is a zip container, and
silently reading bytes out of one would produce nonsense entries rather than an
error. The request was a plain-text glossary; anything else is refused with a
message instead of being half-read.

Case is preserved exactly as typed. `build_index` merges with a plain
`dict.update` and does not normalise, so `Typst` replaces the shipped entry
while `TYPST` would add a second, duplicate one. That is the shipped
behaviour; the parser reports it rather than hiding it.
"""

from __future__ import annotations

import os

#: A file extension we accept. The empty string covers `my-glossary` with no
#: extension at all, which is a normal way to keep a scratch list.
ALLOWED_SUFFIXES = frozenset({"", ".txt", ".csv", ".text", ".list"})

#: Tried in order on each line. The first one found splits the line, so
#: `Docs: Éditeur` splits on `:` and `A=B: C` keeps `B: C` as the expansion.
SEPARATORS = ("=", ";", ":")

#: Longest an acronym key may reasonably be. A longer left-hand side means the
#: line was not an entry (a sentence with a colon in it, most likely).
MAX_ACRONYM_LENGTH = 40


class GlossaryError(ValueError):
    """The upload cannot be read as a plain-text glossary at all."""


def check_filename(filename: str) -> None:
    """Refuse anything that is not plain text, by extension.

    Args:
        filename: the name as uploaded.

    Raises:
        GlossaryError: the extension is not one of `ALLOWED_SUFFIXES`.
    """
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise GlossaryError(
            "« %s » n'est pas un fichier texte. Formats acceptés : %s."
            % (
                filename,
                ", ".join(sorted(s or "(sans extension)" for s in ALLOWED_SUFFIXES)),
            )
        )


def decode(payload: bytes) -> str:
    """Decode an uploaded file as UTF-8, then Latin-1, or refuse it.

    Args:
        payload: the raw bytes uploaded.

    Returns:
        The decoded text.

    Raises:
        GlossaryError: the bytes are not text (a NUL byte, or neither codec).
    """
    if b"\x00" in payload:
        raise GlossaryError(
            "Le fichier contient des octets nuls : ce n'est pas du texte brut"
            " (un .docx ou un .xlsx est une archive zip, pas un fichier texte)."
        )
    for codec in ("utf-8", "latin-1"):
        try:
            return payload.decode(codec)
        except UnicodeDecodeError:
            continue
    raise GlossaryError("Le fichier n'est lisible ni en UTF-8 ni en Latin-1.")


def parse(text: str) -> tuple[dict[str, str], list[str]]:
    """Split a glossary file into entries and rejected lines.

    Args:
        text: the decoded file content.

    Returns:
        The `{acronym: expansion}` mapping, and one human-readable message per
        line that was not usable. The messages are shown in the page: a
        glossary that silently drops half its lines is worse than one that
        says which lines it dropped.
    """
    entries: dict[str, str] = {}
    skipped: list[str] = []

    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        position = min(
            (line.find(sep) for sep in SEPARATORS if line.find(sep) > 0),
            default=-1,
        )
        if position < 0:
            skipped.append(
                "ligne %d : aucun séparateur (=, ; ou :) — « %s »"
                % (number, line[:60])
            )
            continue

        acronym = line[:position].strip()
        expansion = line[position + 1 :].strip()

        if not acronym or not expansion:
            skipped.append(
                "ligne %d : acronyme ou développé vide — « %s »" % (number, line[:60])
            )
            continue
        if len(acronym) > MAX_ACRONYM_LENGTH:
            skipped.append(
                "ligne %d : partie gauche trop longue pour un acronyme (%d"
                " caractères) — « %s »" % (number, len(acronym), acronym[:60])
            )
            continue
        if acronym in entries:
            skipped.append(
                "ligne %d : « %s » déjà défini plus haut, la dernière"
                " définition gagne" % (number, acronym)
            )

        entries[acronym] = expansion

    return entries, skipped


def parse_upload(filename: str, payload: bytes) -> tuple[dict[str, str], list[str]]:
    """Check the name, decode the bytes, parse the lines.

    Args:
        filename: the name as uploaded.
        payload: the raw bytes uploaded.

    Returns:
        The `{acronym: expansion}` mapping and the rejected-line messages.

    Raises:
        GlossaryError: the file is not plain text.
    """
    check_filename(filename)
    return parse(decode(payload))
