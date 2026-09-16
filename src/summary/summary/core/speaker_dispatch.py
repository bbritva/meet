"""Choose a speaker resolver based on what input is actually available.

    metadata.json present (Visio + the VAD agent)
        -> resolve_speaker_identities()            VAD overlap, the stronger base
    no usable metadata (Dictaphone, or the collector was off)
        -> resolve_speaker_identities_from_cues()  attendee list + spoken name cues

`celery_worker.py` calls this instead of testing `payload.metadata is not
None`, which used to skip speaker assignment entirely whenever metadata was
missing OR unreadable — leaving `SPEAKER_00` in the document with nothing to
explain it. Falling through to the cue resolver is strictly better: it can only
add names the VAD path was never going to produce.
"""

# The signature names every input a resolver can use, on purpose: the
# whole job of this module is choosing between them.
# ruff: noqa: PLR0913

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from summary.core.speaker_cues.resolve import resolve_speaker_identities_from_cues
from summary.core.user_assign import AssignmentResult, resolve_speaker_identities

logger = logging.getLogger(__name__)

VAD = "vad"
CUES = "cues"
NONE = "none"


@dataclass
class DispatchResult:
    """What came back, and which resolver produced it."""

    result: AssignmentResult
    source: str  # VAD | CUES | NONE
    reason: str  # why this path was taken, for logs and tests
    fell_back: bool = False  # the VAD path was tried and did not work


def _is_useful(result: AssignmentResult | None) -> bool:
    return bool(result and result.assignments)


def resolve_speakers(
    transcription: Any,
    *,
    metadata: dict | None = None,
    recording_start: datetime | None = None,
    recording_end: datetime | None = None,
    attendees: list[dict[str, str]] | None = None,
    fall_back_on_empty_vad: bool = True,
    cue_options: dict[str, Any] | None = None,
) -> DispatchResult:
    """Resolve `SPEAKER_XX` labels by whichever route the inputs allow.

    The VAD path wins whenever it is available AND produces something, because
    it measures who actually spoke rather than inferring it from words.
    """
    cue_options = dict(cue_options or {})

    have_vad = (
        bool(metadata) and recording_start is not None and recording_end is not None
    )
    have_cues = bool(attendees)

    if have_vad:
        try:
            vad = resolve_speaker_identities(
                metadata, transcription, recording_start, recording_end
            )
        except Exception as error:
            logger.warning(
                "VAD assignment failed (%s); considering the cue path", error
            )
            vad = None

        if _is_useful(vad):
            return DispatchResult(vad, VAD, "metadata present and usable")

        if not (fall_back_on_empty_vad and have_cues):
            return DispatchResult(
                vad or AssignmentResult(), VAD, "metadata present but produced nothing"
            )
        logger.info("VAD produced no assignment; falling back to name cues")

    if have_cues:
        cues = resolve_speaker_identities_from_cues(
            attendees, transcription, **cue_options
        )
        reason = "no usable metadata" if not have_vad else "VAD produced nothing"
        return DispatchResult(cues, CUES, reason, fell_back=have_vad)

    labels = sorted(
        {
            seg.get("speaker")
            for seg in (transcription.get("segments") or [])
            if seg.get("speaker")
        }
    )
    return DispatchResult(
        AssignmentResult(assignments=[], unassigned_speakers=labels),
        NONE,
        "neither metadata nor an attendee list",
    )
