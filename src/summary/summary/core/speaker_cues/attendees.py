"""Minimal iCalendar ATTENDEE reader.

The attendee list is the only roster available when Visio metadata is
missing. La Suite Calendars speaks CalDAV, so the input is a real RFC 5545
file and we only need one line shape out of it:

    ATTENDEE;CN=Prénom Nom;ROLE=...;PARTSTAT=...;RSVP=TRUE:mailto:x@y

Two details make a naive `for line in open(...)` wrong:

  * **folding is measured in octets, not characters** (RFC 5545 §3.1), so a
    line containing `Estève` folds earlier than a pure-ASCII one. We unfold
    on the raw bytes and decode afterwards -- that is also the only way a
    fold that splits a multi-byte character survives.
  * parameter values may be quoted and may contain escaped separators.

Stdlib only, on purpose: no `icalendar` dependency for four fields.
"""

import re
from typing import Any

_FOLD = re.compile(rb"\r\n[ \t]|\n[ \t]")


def unfold(raw: bytes) -> list[str]:
    """Unfold an iCalendar byte stream and return its logical lines."""
    joined = _FOLD.sub(b"", raw)
    text = joined.decode("utf-8")
    return [line for line in re.split(r"\r\n|\n", text) if line]


def _split_unescaped(value: str, sep: str) -> list[str]:
    """Split on `sep`, honouring backslash escapes and double quotes."""
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for char in value:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == sep and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def _parse_content_line(line: str) -> tuple[str, dict[str, str], str]:
    """Return (property name, parameters, value) for one unfolded line."""
    # The first unquoted, unescaped ':' separates name+params from the value.
    quoted = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == ":" and not quoted:
            head, value = line[:index], line[index + 1 :]
            break
    else:
        return line.upper(), {}, ""

    pieces = _split_unescaped(head, ";")
    name = pieces[0].upper()
    params: dict[str, str] = {}
    for piece in pieces[1:]:
        if "=" in piece:
            key, _, param_value = piece.partition("=")
            params[key.upper()] = param_value
    return name, params, value


def parse_attendees(path: str) -> list[dict[str, str]]:
    """Read an `.ics` file and return its attendees as `[{name, email}]`.

    Order follows the file. Attendees without a `CN` are skipped -- a roster
    entry with no name is useless to a name-cue resolver. Duplicate emails
    are collapsed, keeping the first occurrence.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    return parse_attendees_text(raw)


def parse_attendees_text(raw: bytes) -> list[dict[str, str]]:
    """Same as `parse_attendees`, on an in-memory byte stream."""
    attendees: list[dict[str, str]] = []
    seen: set[str] = set()

    for line in unfold(raw):
        name, params, value = _parse_content_line(line)
        if name != "ATTENDEE":
            continue
        common_name = params.get("CN", "").strip()
        if not common_name:
            continue
        email = value.strip()
        if email.lower().startswith("mailto:"):
            email = email[len("mailto:") :]
        email = email.strip()
        key = email.lower() or common_name.lower()
        if key in seen:
            continue
        seen.add(key)
        attendees.append({"name": common_name, "email": email})

    return attendees


def normalise_attendees(attendees: Any) -> list[dict[str, str]]:
    """Turn a task payload's attendee list into the roster the matcher expects.

    Accepts pydantic models (`Attendee`), plain dicts, or `None`, and returns
    `[{"name": ..., "email": ...}]`.

    Entries with no usable name are dropped: `match.py` derives its lookup
    keys by splitting the name, so a nameless roster entry would contribute
    only empty keys and could never match anything useful.

    Args:
        attendees: The payload's attendee list, or None.

    Returns:
        A roster list, empty when there is nothing usable.
    """
    if not attendees:
        return []

    roster: list[dict[str, str]] = []
    seen: set[str] = set()
    for attendee in attendees:
        if isinstance(attendee, dict):
            name = attendee.get("name") or ""
            email = attendee.get("email") or ""
        else:
            name = getattr(attendee, "name", "") or ""
            email = getattr(attendee, "email", "") or ""

        name = str(name).strip()
        email = str(email).strip()
        if not name:
            continue

        key = email.lower() or name.lower()
        if key in seen:
            continue
        seen.add(key)
        roster.append({"name": name, "email": email})

    return roster
