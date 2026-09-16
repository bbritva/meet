"""Fallback speaker resolver: attendee list + name cues, no VAD metadata.

    metadata.json present (Visio + VAD)  -> resolve_speaker_identities()
    no metadata (Dictaphone, gates off)  -> resolve_speaker_identities_from_cues()

Two detectors feed it: `cues.py` (regex templates) and `llm_cues.py` (Albert).
They are interchangeable because they return the same `Cue` list; everything
below -- the scores, the threshold, the ambiguity rule, the uniqueness rule --
is shared, so a detector can only change WHAT IS SEEN, never what counts.

The return type is the same `AssignmentResult` the VAD resolver returns, so
`apply_to()` works unchanged whichever one produced it. `participant_id` is
the attendee's email; `score` is a 0..1 confidence.

How evidence turns into a score
-------------------------------
Each cue carries a base confidence that reflects how much it proves:

    self_id, exact or normalised roster hit      0.95 / 0.92
    self_id, name recovered phonetically         0.45 + 0.40 * similarity
    self_id, speaker not on the roster           0.80
    handoff ("Julien, tu démarres ?")            0.65
    mention                                      never any evidence at all

Several cues pointing at the same person combine with a noisy-OR: three
self-identifications are more certain than one, but no amount of evidence
reaches 1.0. Cues pointing at *different* people compete; if the two best
identities for one label are within `AMBIGUITY_EPSILON`, the label is
unassigned -- "I cannot tell" is a correct answer and a wrong name is not.

Three rules exist purely for precision:

  * a cue whose span matched two attendees equally well (the two Jeans)
    produces no evidence at all, for anybody;
  * a handoff only counts if the named person speaks within the next two
    segments;
  * after scoring, the same attendee claimed by two labels keeps the
    stronger claim only; the weaker label is unassigned. The loser is never
    reassigned to its runner-up, because that would turn one uncertainty
    into two confident guesses.
"""

# The knobs are deliberately explicit rather than a config object: this
# is the measured contract of the resolver, and every default here is a
# number the corpus justified.
# ruff: noqa: PLR0912, PLR0913, PLR0917

from dataclasses import dataclass, field
from typing import Any

from summary.core.speaker_cues.cues import (
    HANDOFF,
    MENTION,
    SELF_ID,
    Cue,
    detect_cues,
)
from summary.core.speaker_cues.match import (
    AMBIGUITY_EPSILON,
    FUZZY_MATCH_THRESHOLD,
)
from summary.core.user_assign import AssignmentResult, SpeakerAssignment

#: Below this combined confidence a speaker keeps its SPEAKER_XX label.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

#: How far a handoff may look ahead for the person it named.
HANDOFF_LOOKAHEAD_SEGMENTS = 2

_BASE_SCORE = {
    "exact": 0.95,
    "normalised": 0.92,
    "out_of_roster": 0.80,
}
_HANDOFF_SCORE = 0.65
_FUZZY_INTERCEPT = 0.45
_FUZZY_SLOPE = 0.40

#: An out-of-roster speaker has no email. `apply_to()` only reads
#: `speaker_label` and `participant_name`, so an empty id is harmless and is
#: the honest encoding of "named, but matched to nobody in the invite".
NO_PARTICIPANT_ID = ""


@dataclass
class Evidence:
    """One cue, once it has been attached to a speaker label."""

    speaker: str
    name: str
    participant_id: str
    score: float
    cue: Cue


@dataclass
class ResolutionTrace:
    """Everything the resolver considered. For `score.py` and for debugging."""

    cues: list[Cue] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)


def _cue_score(cue: Cue) -> float:
    if cue.kind == HANDOFF:
        return _HANDOFF_SCORE
    if cue.tier in _BASE_SCORE:
        return _BASE_SCORE[cue.tier]
    return min(0.9, _FUZZY_INTERCEPT + _FUZZY_SLOPE * cue.similarity)


def _noisy_or(scores: list[float]) -> float:
    combined = 0.0
    for score in scores:
        combined = combined + score - combined * score
    return min(combined, 0.99)


def _handoff_target(segments: list[dict], cue: Cue, lookahead: int) -> str | None:
    """The next different speaker after the segment that carried the handoff."""
    for index in range(cue.segment_index + 1, len(segments)):
        speaker = segments[index].get("speaker")
        if not speaker or speaker == cue.speaker:
            continue
        if index - cue.segment_index > lookahead:
            return None
        return speaker
    return None


def _collect_evidence(
    segments: list[dict], cues: list[Cue], trace: ResolutionTrace
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for cue in cues:
        if cue.kind == MENTION:
            continue
        if cue.ambiguous:
            trace.rejected.append(
                (
                    cue.speaker,
                    "%r matches %s equally well -- no evidence"
                    % (cue.span, " / ".join(cue.candidates)),
                )
            )
            continue
        if cue.kind == SELF_ID:
            target = cue.speaker
        elif cue.kind == HANDOFF:
            target = _handoff_target(segments, cue, HANDOFF_LOOKAHEAD_SEGMENTS)
            if target is None:
                trace.rejected.append(
                    (
                        cue.speaker,
                        "handoff to %s: nobody else speaks within %d segments"
                        % (cue.name, HANDOFF_LOOKAHEAD_SEGMENTS),
                    )
                )
                continue
        else:
            continue
        if not target or not cue.name:
            continue
        evidence.append(
            Evidence(
                speaker=target,
                name=cue.name,
                participant_id=cue.email or NO_PARTICIPANT_ID,
                score=_cue_score(cue),
                cue=cue,
            )
        )
    return evidence


def _speaker_labels(segments: list[dict]) -> list[str]:
    labels: list[str] = []
    for segment in segments:
        speaker = segment.get("speaker")
        if speaker and speaker not in labels:
            labels.append(speaker)
    return labels


#: The two detection stages. Everything after detection is identical.
DETECTORS = ("regex", "llm")


def resolve_speaker_identities_from_cues(
    attendees: list[dict[str, str]],
    transcription: Any,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
    allow_unknown_self_id: bool = True,
    detector: str = "regex",
    detector_options: dict[str, Any] | None = None,
    trace: ResolutionTrace | None = None,
) -> AssignmentResult:
    """Match WhisperX speaker labels to attendees using spoken name cues.

    Args:
        attendees: `[{name, email}]`, typically from `attendees.parse_attendees`.
        transcription: WhisperX dict (or object) with `segments`.
        confidence_threshold: below this a speaker keeps its SPEAKER_XX label.
        fuzzy_threshold: how close a mangled span must be to a roster name.
            Only the `regex` detector uses it.
        allow_unknown_self_id: accept a self-identification whose name is not
            on the roster (a late joiner). Turn it off to never emit a name
            that is not an invitee.
        detector: `"regex"` for the template detector in `cues.py`, `"llm"` for
            the Albert-backed one in `llm_cues.py`. ONLY the detection stage
            changes: the scoring, the threshold, the ambiguity rule and the
            one-attendee-one-label rule below are shared, on purpose.
        detector_options: forwarded to the LLM detector (`complete`, `model`,
            `cache`, `cache_path`, `batch_size`, `stats`). Ignored by `regex`.
        trace: optional, filled with cues and evidence for reporting.

    Returns:
        The same `AssignmentResult` shape as `resolve_speaker_identities`.
    """
    if detector not in DETECTORS:
        raise ValueError(
            "unknown detector %r, expected one of %s" % (detector, DETECTORS)
        )

    segments = (
        transcription.get("segments")
        if isinstance(transcription, dict)
        else getattr(transcription, "segments", None)
    ) or []

    trace = trace if trace is not None else ResolutionTrace()
    if detector == "llm":
        # Imported here so the regex path never pays for the network module.
        from summary.core.speaker_cues.llm_cues import detect_cues_llm  # noqa: PLC0415

        trace.cues = detect_cues_llm(
            segments,
            attendees,
            allow_unknown_self_id=allow_unknown_self_id,
            rejected=trace.rejected,
            **(detector_options or {}),
        )
    else:
        trace.cues = detect_cues(
            segments,
            attendees,
            fuzzy_threshold=fuzzy_threshold,
            allow_unknown_self_id=allow_unknown_self_id,
        )
    trace.evidence = _collect_evidence(segments, trace.cues, trace)

    by_speaker: dict[str, dict[str, list[Evidence]]] = {}
    for item in trace.evidence:
        identity = item.participant_id or ("name:" + item.name)
        by_speaker.setdefault(item.speaker, {}).setdefault(identity, []).append(item)

    # Each speaker is decided on its own evidence. No global assignment:
    # maximising a bipartite matching would hand one Jean to each label and
    # call the ambiguity solved.
    claims: dict[str, SpeakerAssignment] = {}
    for speaker, identities in by_speaker.items():
        ranked = sorted(
            (
                (
                    _noisy_or([e.score for e in items]),
                    items[0].name,
                    items[0].participant_id,
                )
                for items in identities.values()
            ),
            key=lambda row: (-row[0], row[1]),
        )
        score, name, participant_id = ranked[0]
        if len(ranked) > 1 and score - ranked[1][0] <= AMBIGUITY_EPSILON:
            trace.rejected.append(
                (speaker, "two identities tie: %s vs %s" % (name, ranked[1][1]))
            )
            continue
        if score < confidence_threshold:
            trace.rejected.append(
                (
                    speaker,
                    "confidence %.2f below threshold %.2f"
                    % (score, confidence_threshold),
                )
            )
            continue
        claims[speaker] = SpeakerAssignment(
            speaker_label=speaker,
            participant_id=participant_id,
            participant_name=name,
            score=round(score, 4),
        )

    # One attendee, one label: keep the stronger claim, drop the weaker.
    # Dropping, never reassigning -- the runner-up was not good enough on its
    # own merits a moment ago and nothing has changed.
    winners: dict[str, SpeakerAssignment] = {}
    for assignment in sorted(claims.values(), key=lambda a: -a.score):
        if not assignment.participant_id:
            continue
        held = winners.get(assignment.participant_id)
        if held is None:
            winners[assignment.participant_id] = assignment
            continue
        trace.rejected.append(
            (
                assignment.speaker_label,
                "%s is already claimed by %s with a stronger score (%.2f > %.2f)"
                % (
                    assignment.participant_name,
                    held.speaker_label,
                    held.score,
                    assignment.score,
                ),
            )
        )
        claims.pop(assignment.speaker_label, None)

    result = AssignmentResult()
    for label in _speaker_labels(segments):
        assignment = claims.get(label)
        if assignment is None:
            result.unassigned_speakers.append(label)
        else:
            result.assignments.append(assignment)
    return result
