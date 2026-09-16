"""The dispatch picks the right resolver, and degrades instead of giving up.

Every test asserts WHICH resolver ran, not how accurate it was -- accuracy is
covered by the cue and match tests. So the transcript here is a handful of
segments carrying clean self-identifications, built inline rather than read
from a corpus.

`_metadata_from` synthesises VAD events that agree with the transcript
timings, and names its participants "VAD ..." so the two paths produce
different names. That is what proves which one ran.
"""

import unittest
from datetime import datetime, timedelta, timezone

from summary.core.speaker_dispatch import CUES, NONE, VAD, resolve_speakers

ATTENDEES = [
    {"name": "Sylvie Aubert", "email": "sylvie.aubert@dinum.gouv.fr"},
    {"name": "Bruno Delcourt", "email": "bruno.delcourt@dinum.gouv.fr"},
]

RECORDING_START = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)


def _transcript():
    """A short meeting where both speakers name themselves."""
    rows = [
        ("SPEAKER_00", " Bonjour a tous, ici Sylvie.", 0.0, 3.0),
        ("SPEAKER_01", " Bonjour, c'est Bruno.", 3.5, 6.0),
        ("SPEAKER_00", " On commence par le budget.", 6.5, 9.0),
        ("SPEAKER_01", " Tres bien, je prends des notes.", 9.5, 12.0),
    ]
    return {
        "segments": [
            {"speaker": speaker, "text": text, "start": start, "end": end}
            for speaker, text, start, end in rows
        ]
    }


def _metadata_from(transcript):
    """Build VAD metadata that agrees with the transcript's timings."""
    events = []
    seen = {}
    for segment in transcript["segments"]:
        participant_id = "pid-" + segment["speaker"]
        seen[participant_id] = segment["speaker"]
        for kind, offset in (
            ("speech_start", segment["start"]),
            ("speech_end", segment["end"]),
        ):
            events.append(
                {
                    "participant_id": participant_id,
                    "type": kind,
                    "timestamp": (
                        RECORDING_START + timedelta(seconds=offset)
                    ).isoformat(),
                    "data": None,
                }
            )
    participants = [
        {"participantId": participant_id, "name": "VAD %s" % label}
        for participant_id, label in sorted(seen.items())
    ]
    end = RECORDING_START + timedelta(seconds=transcript["segments"][-1]["end"] + 1)
    return {"events": events, "participants": participants}, RECORDING_START, end


class Dispatch(unittest.TestCase):
    """Dispatch."""

    def test_metadata_present_uses_vad(self):
        """Metadata present uses vad."""
        transcript = _transcript()
        metadata, start, end = _metadata_from(transcript)
        got = resolve_speakers(
            transcript,
            metadata=metadata,
            recording_start=start,
            recording_end=end,
            attendees=ATTENDEES,
        )
        self.assertEqual(got.source, VAD)
        self.assertFalse(got.fell_back)
        # VAD names win over the cue names, proving which path ran
        self.assertTrue(
            all(a.participant_name.startswith("VAD ") for a in got.result.assignments)
        )

    def test_no_metadata_uses_cues(self):
        """No metadata uses cues."""
        got = resolve_speakers(_transcript(), attendees=ATTENDEES)
        self.assertEqual(got.source, CUES)
        self.assertFalse(got.fell_back)
        self.assertEqual(
            {a.speaker_label: a.participant_name for a in got.result.assignments},
            {"SPEAKER_00": "Sylvie Aubert", "SPEAKER_01": "Bruno Delcourt"},
        )

    def test_unusable_metadata_falls_back_to_cues(self):
        """Unusable metadata falls back to cues."""
        # metadata present but the events are junk -> VAD yields nothing
        got = resolve_speakers(
            _transcript(),
            metadata={"events": [], "participants": []},
            recording_start=RECORDING_START,
            recording_end=RECORDING_START + timedelta(hours=1),
            attendees=ATTENDEES,
        )
        self.assertEqual(got.source, CUES)
        self.assertTrue(got.fell_back, "should record that VAD was tried first")
        self.assertTrue(got.result.assignments)

    def test_broken_metadata_does_not_crash(self):
        """Broken metadata does not crash."""
        got = resolve_speakers(
            _transcript(),
            metadata={"events": [{"nope": 1}], "participants": []},
            recording_start=RECORDING_START,
            recording_end=RECORDING_START + timedelta(hours=1),
            attendees=ATTENDEES,
        )
        self.assertEqual(got.source, CUES)

    def test_neither_input_leaves_every_label_unassigned(self):
        """Neither input leaves every label unassigned."""
        transcript = _transcript()
        got = resolve_speakers(transcript)
        self.assertEqual(got.source, NONE)
        self.assertEqual(got.result.assignments, [])
        labels = sorted({s["speaker"] for s in transcript["segments"]})
        self.assertEqual(sorted(got.result.unassigned_speakers), labels)

    def test_fallback_can_be_disabled(self):
        """Fallback can be disabled."""
        got = resolve_speakers(
            _transcript(),
            metadata={"events": [], "participants": []},
            recording_start=RECORDING_START,
            recording_end=RECORDING_START + timedelta(hours=1),
            attendees=ATTENDEES,
            fall_back_on_empty_vad=False,
        )
        self.assertEqual(got.source, VAD)
        self.assertEqual(got.result.assignments, [])

    def test_a_partial_vad_result_is_not_topped_up_by_cues(self):
        """VAD wins as soon as it says anything; cues never top it up.

        The conservative reading: whenever the VAD path works at all, the
        output is exactly what it was before this fallback existed.
        """
        transcript = _transcript()
        metadata, start, end = _metadata_from(transcript)
        # drop one participant's events, so VAD can only resolve the other
        metadata["events"] = [
            e for e in metadata["events"] if e["participant_id"] != "pid-SPEAKER_01"
        ]
        got = resolve_speakers(
            transcript,
            metadata=metadata,
            recording_start=start,
            recording_end=end,
            attendees=ATTENDEES,
        )
        self.assertEqual(got.source, VAD)
        self.assertEqual(
            [a.participant_name for a in got.result.assignments], ["VAD SPEAKER_00"]
        )
        self.assertEqual(got.result.unassigned_speakers, ["SPEAKER_01"])
