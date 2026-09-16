"""Unit tests for the LLM cue detector -- the guards above all.

No network. Every test injects a stub `complete(system, user)` that returns a
canned reply, because the point is not what Albert answers, it is what this
module refuses to do with an answer. The guards are the reason a model may be
allowed near an attribution at all:

  * a name that is not on the roster is rejected, unless it is a self_id;
  * a `mention` never becomes an attribution, whatever name it carries;
  * a name matching two attendees equally well attributes nobody.
"""

import json
import unittest

from summary.core.speaker_cues.cues import MENTION, SELF_ID
from summary.core.speaker_cues.llm_cues import (
    SegmentCache,
    build_system_prompt,
    build_user_prompt,
    detect_cues_llm,
    parse_response,
)
from summary.core.speaker_cues.resolve import resolve_speaker_identities_from_cues

ROSTER = [
    {"name": "Bruno Delcourt", "email": "bruno.delcourt@dinum.gouv.fr"},
    {"name": "Léa Marchand", "email": "lea.marchand@dinum.gouv.fr"},
    {"name": "Sylvie Aubert", "email": "sylvie.aubert@dinum.gouv.fr"},
]

ROSTER_JEAN = [
    {"name": "Hélène Charpentier", "email": "helene.charpentier@dinum.gouv.fr"},
    {"name": "Jean Dupont", "email": "jean.dupont@dinum.gouv.fr"},
    {"name": "Jean Moreau", "email": "jean.moreau@dinum.gouv.fr"},
]


def segments(*rows):
    """`[(speaker, text)]` -> WhisperX-shaped segments."""
    return [{"speaker": speaker, "text": " " + text} for speaker, text in rows]


class StubLLM:
    """Returns one canned reply per call and remembers what it was asked."""

    def __init__(self, *replies):
        """Queue the canned replies, one per expected call."""
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user):
        """Call  ."""
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError("the detector asked more times than expected")
        return self.replies.pop(0)


def reply(*rows):
    """`[(index, type, name)]` -> the JSON array the model is supposed to emit."""
    return json.dumps(
        [
            {"index": index, "type": kind, "name": name, "raw": "extrait"}
            for index, kind, name in rows
        ],
        ensure_ascii=False,
    )


def detect(segs, roster, stub, **options):
    """Detect."""
    return detect_cues_llm(
        segs, roster, complete=stub, cache=SegmentCache(None), **options
    )


def resolve(segs, roster, stub, **options):
    """Resolve."""
    return resolve_speaker_identities_from_cues(
        roster,
        {"segments": segs},
        detector="llm",
        detector_options={"complete": stub, "cache": SegmentCache(None)},
        **options,
    )


# --------------------------------------------------------------------------
class TestRosterGuard(unittest.TestCase):
    """A name the invite never heard of only ever comes from a self_id."""

    def test_off_roster_name_on_a_handoff_is_rejected(self):
        """Off roster name on a handoff is rejected."""
        segs = segments(
            ("SPEAKER_00", "Amandine, tu nous dis un mot ?"),
            ("SPEAKER_01", "Volontiers, sur le budget."),
        )
        rejected = []
        cues = detect(
            segs,
            ROSTER,
            StubLLM(reply((0, "handoff", "Amandine"), (1, "none", None))),
            rejected=rejected,
        )
        self.assertEqual(cues, [])
        self.assertTrue(any("Amandine" in reason for _, reason in rejected))

    def test_off_roster_name_on_a_handoff_attributes_nobody(self):
        """Off roster name on a handoff attributes nobody."""
        segs = segments(
            ("SPEAKER_00", "Amandine, tu nous dis un mot ?"),
            ("SPEAKER_01", "Volontiers, sur le budget."),
        )
        result = resolve(
            segs, ROSTER, StubLLM(reply((0, "handoff", "Amandine"), (1, "none", None)))
        )
        self.assertEqual(result.assignments, [])
        self.assertEqual(result.unassigned_speakers, ["SPEAKER_00", "SPEAKER_01"])

    def test_off_roster_name_on_a_mention_is_rejected(self):
        """Off roster name on a mention is rejected."""
        segs = segments(("SPEAKER_00", "Amandine nous a envoyé sa note."))
        cues = detect(segs, ROSTER, StubLLM(reply((0, "mention", "Amandine"))))
        self.assertEqual(cues, [])

    def test_off_roster_name_on_a_self_id_is_kept(self):
        """Off roster name on a self id is kept."""
        segs = segments(("SPEAKER_00", "Bonjour, Nourdine, désolé du retard."))
        cues = detect(segs, ROSTER, StubLLM(reply((0, "self_id", "Nourdine"))))
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].kind, SELF_ID)
        self.assertEqual(cues[0].name, "Nourdine")
        self.assertEqual(cues[0].tier, "out_of_roster")
        self.assertEqual(cues[0].email, "")

    def test_off_roster_self_id_can_be_switched_off(self):
        """Off roster self id can be switched off."""
        segs = segments(("SPEAKER_00", "Bonjour, Nourdine, désolé du retard."))
        cues = detect(
            segs,
            ROSTER,
            StubLLM(reply((0, "self_id", "Nourdine"))),
            allow_unknown_self_id=False,
        )
        self.assertEqual(cues, [])

    def test_off_roster_self_id_must_look_like_a_name(self):
        """The model may not smuggle a clause through the one open door."""
        segs = segments(("SPEAKER_00", "Je travaille à la sous-direction."))
        for invented in (
            "la sous-direction technique du ministère",
            "le responsable",
            "DINUM",
            "x",
        ):
            cues = detect(segs, ROSTER, StubLLM(reply((0, "self_id", invented))))
            self.assertEqual(cues, [], invented)

    def test_a_roster_name_keeps_its_email_and_canonical_spelling(self):
        """A roster name keeps its email and canonical spelling."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."))
        cues = detect(segs, ROSTER, StubLLM(reply((0, "self_id", "sylvie aubert"))))
        self.assertEqual(cues[0].name, "Sylvie Aubert")
        self.assertEqual(cues[0].email, "sylvie.aubert@dinum.gouv.fr")
        self.assertEqual(cues[0].tier, "normalised")


class TestMentionNeverAttributes(unittest.TestCase):
    """The most expensive error in the corpus, and the model cannot make it."""

    def test_mention_of_a_roster_name_produces_a_mention_cue(self):
        """Mention of a roster name produces a mention cue."""
        segs = segments(("SPEAKER_00", "Je remplace Léa pendant son congé."))
        cues = detect(segs, ROSTER, StubLLM(reply((0, "mention", "Léa Marchand"))))
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].kind, MENTION)
        self.assertEqual(cues[0].name, "Léa Marchand")

    def test_mention_of_a_roster_name_attributes_nobody(self):
        """Mention of a roster name attributes nobody."""
        segs = segments(
            ("SPEAKER_00", "Je remplace Léa pendant son congé."),
            ("SPEAKER_01", "Très bien, on vous écoute."),
        )
        result = resolve(
            segs,
            ROSTER,
            StubLLM(reply((0, "mention", "Léa Marchand"), (1, "none", None))),
        )
        self.assertEqual(result.assignments, [])

    def test_a_meeting_of_mentions_only_attributes_nobody(self):
        """A meeting of mentions only attributes nobody."""
        segs = segments(
            ("SPEAKER_00", "Comme Bruno l'a signalé au comité."),
            ("SPEAKER_01", "Sylvie nous enverra la note."),
            ("SPEAKER_02", "Léa est en congé cette semaine."),
        )
        result = resolve(
            segs,
            ROSTER,
            StubLLM(
                reply(
                    (0, "mention", "Bruno Delcourt"),
                    (1, "mention", "Sylvie Aubert"),
                    (2, "mention", "Léa Marchand"),
                )
            ),
        )
        self.assertEqual(result.assignments, [])
        self.assertEqual(len(result.unassigned_speakers), 3)


class TestSelfContradiction(unittest.TestCase):
    """One voice cannot be Mounir and talk about Mounir."""

    def test_a_self_id_contradicted_by_the_same_voice_is_dropped(self):
        """A self id contradicted by the same voice is dropped."""
        segs = segments(
            ("SPEAKER_00", "Bruno nous a envoyé ses chiffres hier soir."),
            ("SPEAKER_01", "Une limitation par clé réglerait le cas."),
            ("SPEAKER_00", "Bruno, il faudra le relancer une dernière fois."),
        )
        rejected = []
        cues = detect(
            segs,
            ROSTER,
            StubLLM(
                reply(
                    (0, "mention", "Bruno Delcourt"),
                    (1, "none", None),
                    (2, "self_id", "Bruno Delcourt"),
                )
            ),
            rejected=rejected,
        )
        self.assertEqual([cue.kind for cue in cues], [MENTION])
        self.assertTrue(any("third person" in reason for _, reason in rejected))

    def test_a_contradicted_self_id_attributes_nobody(self):
        """A contradicted self id attributes nobody."""
        segs = segments(
            ("SPEAKER_00", "Bruno nous a envoyé ses chiffres hier soir."),
            ("SPEAKER_01", "Une limitation par clé réglerait le cas."),
            ("SPEAKER_00", "Bruno, il faudra le relancer une dernière fois."),
        )
        result = resolve(
            segs,
            ROSTER,
            StubLLM(
                reply(
                    (0, "mention", "Bruno Delcourt"),
                    (1, "none", None),
                    (2, "self_id", "Bruno Delcourt"),
                )
            ),
        )
        self.assertEqual(result.assignments, [])

    def test_another_voice_mentioning_the_name_changes_nothing(self):
        """The contradiction has to come from the SAME label."""
        segs = segments(
            ("SPEAKER_00", "Bruno nous a envoyé ses chiffres hier soir."),
            ("SPEAKER_01", "Bonjour, c'est Bruno."),
        )
        result = resolve(
            segs,
            ROSTER,
            StubLLM(
                reply(
                    (0, "mention", "Bruno Delcourt"), (1, "self_id", "Bruno Delcourt")
                )
            ),
        )
        self.assertEqual(
            {a.speaker_label: a.participant_name for a in result.assignments},
            {"SPEAKER_01": "Bruno Delcourt"},
        )

    def test_a_handoff_is_not_a_contradiction(self):
        """Naming somebody and then handing them the floor is consistent."""
        segs = segments(
            ("SPEAKER_00", "Bruno nous a envoyé ses chiffres."),
            ("SPEAKER_00", "Bruno, à toi."),
            ("SPEAKER_01", "Merci, sur le calendrier."),
        )
        result = resolve(
            segs,
            ROSTER,
            StubLLM(
                reply(
                    (0, "mention", "Bruno Delcourt"),
                    (1, "handoff", "Bruno Delcourt"),
                    (2, "none", None),
                )
            ),
        )
        self.assertEqual(
            {a.speaker_label: a.participant_name for a in result.assignments},
            {"SPEAKER_01": "Bruno Delcourt"},
        )


class TestAmbiguityAndScoring(unittest.TestCase):
    """Ambiguity and scoring."""

    def test_a_shared_first_name_attributes_nobody(self):
        """A shared first name attributes nobody."""
        segs = segments(
            ("SPEAKER_00", "Jean, tu peux faire un export ?"),
            ("SPEAKER_01", "Oui, je sors le tableau ce soir."),
        )
        result = resolve(
            segs, ROSTER_JEAN, StubLLM(reply((0, "handoff", "Jean"), (1, "none", None)))
        )
        self.assertEqual(result.assignments, [])

    def test_a_self_id_reaches_the_threshold(self):
        """A self id reaches the threshold."""
        segs = segments(
            ("SPEAKER_00", "Ici Sylvie."), ("SPEAKER_01", "Bonjour, c'est Bruno.")
        )
        result = resolve(
            segs,
            ROSTER,
            StubLLM(
                reply((0, "self_id", "Sylvie Aubert"), (1, "self_id", "Bruno Delcourt"))
            ),
        )
        self.assertEqual(
            {a.speaker_label: a.participant_name for a in result.assignments},
            {"SPEAKER_00": "Sylvie Aubert", "SPEAKER_01": "Bruno Delcourt"},
        )
        self.assertTrue(all(a.score >= 0.6 for a in result.assignments))

    def test_a_handoff_to_somebody_who_stays_silent_is_dropped(self):
        """A handoff to somebody who stays silent is dropped."""
        segs = segments(
            ("SPEAKER_00", "Bruno, à toi."),
            ("SPEAKER_00", "Bon, personne ne répond."),
            ("SPEAKER_00", "On passe au point suivant."),
            ("SPEAKER_01", "Sur le calendrier maintenant."),
        )
        result = resolve(
            segs,
            ROSTER,
            StubLLM(
                reply(
                    (0, "handoff", "Bruno Delcourt"),
                    (1, "none", None),
                    (2, "none", None),
                    (3, "none", None),
                )
            ),
        )
        self.assertEqual(result.assignments, [])


class TestDefensiveParsing(unittest.TestCase):
    """A malformed answer costs recall on one batch. Never a crash."""

    def test_garbage_yields_no_cues(self):
        """Garbage yields no cues."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."))
        self.assertEqual(detect(segs, ROSTER, StubLLM("je ne peux pas répondre")), [])

    def test_truncated_json_yields_no_cues(self):
        """Truncated json yields no cues."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."))
        self.assertEqual(detect(segs, ROSTER, StubLLM('[{"index": 0, "type": ')), [])

    def test_empty_answer_yields_no_cues(self):
        """Empty answer yields no cues."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."))
        self.assertEqual(detect(segs, ROSTER, StubLLM("")), [])

    def test_a_fenced_code_block_is_unwrapped(self):
        """A fenced code block is unwrapped."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."))
        answer = "```json\n%s\n```" % reply((0, "self_id", "Sylvie Aubert"))
        cues = detect(segs, ROSTER, StubLLM(answer))
        self.assertEqual(cues[0].name, "Sylvie Aubert")

    def test_prose_around_the_array_is_ignored(self):
        """Prose around the array is ignored."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."))
        answer = "Voici l'analyse :\n%s\nJ'espère que cela convient." % reply(
            (0, "self_id", "Sylvie Aubert")
        )
        self.assertEqual(detect(segs, ROSTER, StubLLM(answer))[0].name, "Sylvie Aubert")

    def test_an_index_outside_the_batch_is_dropped(self):
        """An index outside the batch is dropped."""
        self.assertEqual(
            parse_response(reply((7, "self_id", "Sylvie Aubert")), [0, 1]), {}
        )

    def test_a_type_outside_the_vocabulary_is_dropped(self):
        """A type outside the vocabulary is dropped."""
        self.assertEqual(parse_response(reply((0, "guess", "Sylvie Aubert")), [0]), {})

    def test_scraping_survives_a_broken_array(self):
        """Scraping survives a broken array."""
        broken = (
            '[{"index": 0, "type": "self_id", "name": "Sylvie Aubert", '
            '"raw": "Ici Sylvie"},, ]'
        )
        parsed = parse_response(broken, [0])
        self.assertEqual(parsed[0]["name"], "Sylvie Aubert")

    def test_a_segment_the_model_skipped_is_not_cached(self):
        """A gap must be re-asked next run, not frozen into the cache."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."), ("SPEAKER_01", "Bonjour."))
        cache = SegmentCache(None)
        detect_cues_llm(
            segs, ROSTER, complete=StubLLM(reply((0, "none", None))), cache=cache
        )
        self.assertEqual(len(cache.entries), 1)


class TestBatchingAndCache(unittest.TestCase):
    """Batching and cache."""

    def test_segments_are_batched_and_indexed(self):
        """Segments are batched and indexed."""
        segs = segments(*[("SPEAKER_00", "Phrase numéro %d." % i) for i in range(5)])
        stub = StubLLM(
            reply(*[(i, "none", None) for i in (0, 1)]),
            reply(*[(i, "none", None) for i in (2, 3)]),
            reply((4, "none", None)),
        )
        detect(segs, ROSTER, stub, batch_size=2)
        self.assertEqual(len(stub.calls), 3)
        self.assertIn("[0] (SPEAKER_00)", stub.calls[0][1])
        self.assertIn("[1] (SPEAKER_00)", stub.calls[0][1])
        self.assertIn("[4] (SPEAKER_00)", stub.calls[2][1])

    def test_the_roster_is_in_the_system_prompt(self):
        """The roster is in the system prompt."""
        prompt = build_system_prompt(ROSTER)
        for attendee in ROSTER:
            self.assertIn(attendee["name"], prompt)
        self.assertIn('"self_id|handoff|mention|none"', prompt)

    def test_the_user_prompt_carries_index_and_label(self):
        """The user prompt carries index and label."""
        self.assertEqual(
            build_user_prompt([(3, "SPEAKER_02", "Bonjour.")]),
            "Segments à analyser :\n\n[3] (SPEAKER_02) Bonjour.",
        )

    def test_a_cached_segment_is_never_asked_again(self):
        """A cached segment is never asked again."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."))
        cache = SegmentCache(None)
        stub = StubLLM(reply((0, "self_id", "Sylvie Aubert")))
        first = detect_cues_llm(segs, ROSTER, complete=stub, cache=cache)
        second = detect_cues_llm(segs, ROSTER, complete=stub, cache=cache)
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(first[0].name, second[0].name)

    def test_the_cache_key_follows_the_roster(self):
        """Same words, other invite: the answer may not be reused."""
        text = " Ici Sylvie."
        self.assertNotEqual(
            SegmentCache.key("m", [a["name"] for a in ROSTER], text),
            SegmentCache.key("m", [a["name"] for a in ROSTER_JEAN], text),
        )

    def test_the_cache_key_is_independent_of_batching(self):
        """The cache key is independent of batching."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."), ("SPEAKER_01", "Bonjour."))
        cache = SegmentCache(None)
        detect_cues_llm(
            segs,
            ROSTER,
            complete=StubLLM(reply((0, "self_id", "Sylvie Aubert"), (1, "none", None))),
            cache=cache,
            batch_size=8,
        )
        stub = StubLLM()  # any call now raises
        cues = detect_cues_llm(segs, ROSTER, complete=stub, cache=cache, batch_size=1)
        self.assertEqual(stub.calls, [])
        self.assertEqual(cues[0].name, "Sylvie Aubert")


class TestDetectorSwitch(unittest.TestCase):
    """Detector switch."""

    def test_an_unknown_detector_is_refused(self):
        """An unknown detector is refused."""
        with self.assertRaises(ValueError):
            resolve_speaker_identities_from_cues(
                ROSTER, {"segments": []}, detector="magic"
            )

    def test_the_regex_detector_is_still_the_default(self):
        """The regex detector is still the default."""
        segs = segments(("SPEAKER_00", "Ici Sylvie."))
        result = resolve_speaker_identities_from_cues(ROSTER, {"segments": segs})
        self.assertEqual(
            [a.participant_name for a in result.assignments], ["Sylvie Aubert"]
        )


class MaxCallsCap(unittest.TestCase):
    """The call ceiling bounds cost and degrades to regex, never silently drops."""

    def _many(self, n):
        rows = [
            ("SPEAKER_00", "On avance sur le dossier numéro %d." % i) for i in range(n)
        ]
        # a clean self-id in the very last segment, well past any cap
        rows.append(("SPEAKER_01", "Bonjour, ici Sylvie."))
        return segments(*rows)

    def test_cap_stops_calling(self):
        """Cap stops calling."""
        segs = self._many(35)  # 36 segments -> 3 batches of 12
        stub = StubLLM(reply(*[(i, "none", None) for i in range(12)]))
        stats = {}
        detect_cues_llm(
            segs,
            ROSTER,
            complete=stub,
            cache=SegmentCache(None),
            batch_size=12,
            max_calls=1,
            stats=stats,
        )
        self.assertEqual(stats["calls"], 1)
        self.assertTrue(stats["capped"])
        self.assertEqual(len(stub.calls), 1)  # stub would raise on a 2nd call

    def test_capped_segments_fall_back_to_regex(self):
        """Capped segments fall back to regex."""
        segs = self._many(35)
        stub = StubLLM(reply(*[(i, "none", None) for i in range(12)]))
        cues = detect_cues_llm(
            segs,
            ROSTER,
            complete=stub,
            cache=SegmentCache(None),
            batch_size=12,
            max_calls=1,
        )
        # the tail was never sent to the model, but regex still finds "ici Sylvie"
        found = [c for c in cues if c.name == "Sylvie Aubert" and c.kind == SELF_ID]
        self.assertTrue(found, "regex fallback did not cover the capped tail")
        self.assertEqual(
            found[0].segment_index,
            len(segs) - 1,
            "fallback cue carries the wrong segment index",
        )

    def test_not_capped_when_ceiling_is_high(self):
        """Not capped when ceiling is high."""
        segs = self._many(11)  # 12 segments -> 1 batch
        stub = StubLLM(reply(*[(i, "none", None) for i in range(12)]))
        stats = {}
        detect_cues_llm(
            segs,
            ROSTER,
            complete=stub,
            cache=SegmentCache(None),
            batch_size=12,
            max_calls=60,
            stats=stats,
        )
        self.assertFalse(stats["capped"])
        self.assertEqual(stats["calls"], 1)
