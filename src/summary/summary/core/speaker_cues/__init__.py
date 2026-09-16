"""Fallback speaker attribution: attendee list + spoken name cues.

Used when the per-participant VAD metadata is missing, which is the common
case: `METADATA_COLLECTOR_ENABLED` and the per-recording `collect_metadata`
option both default to off, and Dictaphone never has any. Returns the same
`AssignmentResult` as `summary.core.user_assign.resolve_speaker_identities`,
so `apply_to()` works unchanged whichever resolver produced it.

The VAD path stays the stronger one and keeps priority: it measures who spoke
rather than inferring it from words. This package only covers the cases where
it has nothing to measure.
"""

from summary.core.speaker_cues.attendees import (
    normalise_attendees,
    parse_attendees,
    parse_attendees_text,
)
from summary.core.speaker_cues.cues import HANDOFF, MENTION, SELF_ID, Cue, detect_cues
from summary.core.speaker_cues.resolve import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ResolutionTrace,
    resolve_speaker_identities_from_cues,
)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "HANDOFF",
    "MENTION",
    "SELF_ID",
    "Cue",
    "ResolutionTrace",
    "detect_cues",
    "normalise_attendees",
    "parse_attendees",
    "parse_attendees_text",
    "resolve_speaker_identities_from_cues",
]
