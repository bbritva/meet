"""Unit tests for the cue detector and for how cues become attributions."""

import unittest

from summary.core.speaker_cues.cues import (
    HANDOFF,
    MENTION,
    SELF_ID,
    detect_cues,
    tokenize,
)
from summary.core.speaker_cues.resolve import resolve_speaker_identities_from_cues

ROSTER = [
    {"name": "Bruno Delcourt", "email": "bruno.delcourt@dinum.gouv.fr"},
    {"name": "Léa Marchand", "email": "lea.marchand@dinum.gouv.fr"},
    {"name": "Marc Teyssier", "email": "marc.teyssier@culture.gouv.fr"},
    {"name": "Sylvie Aubert", "email": "sylvie.aubert@dinum.gouv.fr"},
]

ROSTER_JEAN = [
    {"name": "Hélène Charpentier", "email": "helene.charpentier@dinum.gouv.fr"},
    {"name": "Jean Dupont", "email": "jean.dupont@dinum.gouv.fr"},
    {"name": "Jean Moreau", "email": "jean.moreau@dinum.gouv.fr"},
]

ROSTER_MANGLED = [
    {"name": "Karim Sahli", "email": "karim.sahli@dinum.gouv.fr"},
    {"name": "Mathilde Fontaine", "email": "mathilde.fontaine@dinum.gouv.fr"},
    {"name": "Vincent Delaunay", "email": "vincent.delaunay@dinum.gouv.fr"},
]


def segment(text, speaker="SPEAKER_00"):
    """Segment."""
    return {"speaker": speaker, "text": text}


def cues_of(texts, roster=ROSTER, **kwargs):
    """Cues of."""
    segments = [segment(t) if isinstance(t, str) else segment(*t) for t in texts]
    return detect_cues(segments, roster, **kwargs)


def kinds(cues, kind):
    """Kinds."""
    return [c for c in cues if c.kind == kind]


class TestTokenizer(unittest.TestCase):
    """Punctuation is attached in some segments and detached in others."""

    def test_attached_punctuation_is_stripped(self):
        """Attached punctuation is stripped."""
        tokens = tokenize(" Bonjour, c'est Thomas.")
        self.assertEqual([t.raw for t in tokens], ["Bonjour", "c'est", "Thomas"])
        self.assertTrue(tokens[0].brk_after)
        self.assertTrue(tokens[-1].brk_after)

    def test_detached_punctuation_is_a_boundary_not_a_token(self):
        """Detached punctuation is a boundary not a token."""
        tokens = tokenize(" Bonjour, c'est Nourdine , désolé pour le retard.")
        self.assertEqual(
            [t.raw for t in tokens],
            ["Bonjour", "c'est", "Nourdine", "désolé", "pour", "le", "retard"],
        )
        self.assertTrue(tokens[2].brk_after)

    def test_both_shapes_yield_the_same_name_token(self):
        """Both shapes yield the same name token."""
        attached = tokenize("Bonjour, c'est Thomas.")[-1].raw
        detached = tokenize("Bonjour, c'est Thomas .")[-1].raw
        self.assertEqual(attached, detached)


class TestSelfIdTemplates(unittest.TestCase):
    """Self id templates."""

    def test_ici_prenom(self):
        """Ici prenom."""
        cues = kinds(cues_of([" Bonjour à tous. Ici Sylvie."]), SELF_ID)
        self.assertEqual([c.name for c in cues], ["Sylvie Aubert"])

    def test_cest_prenom(self):
        """Cest prenom."""
        cues = kinds(cues_of([" Bonjour, c'est Marc. J'ai repris la grille."]), SELF_ID)
        self.assertEqual([c.name for c in cues], ["Marc Teyssier"])

    def test_prenom_a_l_appareil(self):
        """Prenom a l appareil."""
        cues = kinds(
            cues_of([" Bruno à l'appareil. Sur le principe je suis favorable."]),
            SELF_ID,
        )
        self.assertEqual([c.name for c in cues], ["Bruno Delcourt"])

    def test_prenom_au_micro(self):
        """Prenom au micro."""
        cues = kinds(cues_of([" Marc au micro, je prends la suite."]), SELF_ID)
        self.assertEqual([c.name for c in cues], ["Marc Teyssier"])

    def test_nom_complet_bonjour(self):
        """Nom complet bonjour."""
        roster = [{"name": "Gaël Prigent", "email": "gael.prigent@interieur.gouv.fr"}]
        cues = kinds(
            cues_of([" Gaël Prigent, bonjour. Sur la partie hébergement."], roster),
            SELF_ID,
        )
        self.assertEqual([c.name for c in cues], ["Gaël Prigent"])

    def test_a_self_id_labels_its_own_speaker(self):
        """A self id labels its own speaker."""
        cues = kinds(cues_of([(" Ici Sylvie.", "SPEAKER_02")]), SELF_ID)
        self.assertEqual(cues[0].speaker, "SPEAKER_02")


class TestCestIsNotAName(unittest.TestCase):
    """The naive "capitalised word after c'est" rule fires on all of these."""

    NOT_NAMES = [
        " C'est un schéma qu'on a déjà rencontré sur le stockage objet.",
        " C'est la durée que l'ANSSI recommande pour ce type de journal.",
        " C'est le prix de l'arbitrage.",
        " C'est exactement le genre de point que le comité relèvera.",
        " C'est une dette qui grossit toute seule.",
        " Ce n'est pas un problème d'outil, c'est un problème de circuit.",
        " C'est entendu, on prévoira un créneau de soirée.",
    ]

    def test_no_cue_is_produced(self):
        """No cue is produced."""
        for text in self.NOT_NAMES:
            self.assertEqual(kinds(cues_of([text]), SELF_ID), [], text)

    def test_an_acronym_is_never_a_late_joiner(self):
        """An acronym is never a late joiner."""
        # Capitalised, follows a strong template, still not a person.
        self.assertEqual(kinds(cues_of([" Ici DINUM, service support."]), SELF_ID), [])


class TestSentenceBoundaries(unittest.TestCase):
    """A template may reach across a comma, never across a full stop."""

    def test_a_mention_before_bonjour_is_not_a_self_id(self):
        """A mention before bonjour is not a self id."""
        # "... à Damien." then a new sentence "Bonjour à tous." The <nom>,
        # bonjour template must not reach back into the previous sentence.
        roster = [{"name": "Damien Laroche", "email": "damien.laroche@dinum.gouv.fr"}]
        cues = cues_of([" J'en ai parlé à Damien. Bonjour à tous."], roster)
        self.assertEqual(kinds(cues, SELF_ID), [])
        self.assertEqual([c.name for c in kinds(cues, MENTION)], ["Damien Laroche"])

    def test_a_mention_before_vous_is_not_a_handoff(self):
        """A mention before vous is not a handoff."""
        roster = [{"name": "Damien Laroche", "email": "damien.laroche@dinum.gouv.fr"}]
        cues = cues_of([" Le point vient de Damien. Vous verrez le document."], roster)
        self.assertEqual(kinds(cues, HANDOFF), [])

    def test_a_comma_is_still_crossed(self):
        """A comma is still crossed."""
        roster = [{"name": "Gaël Prigent", "email": "gael.prigent@interieur.gouv.fr"}]
        cues = kinds(cues_of([" Gaël Prigent, bonjour."], roster), SELF_ID)
        self.assertEqual([c.name for c in cues], ["Gaël Prigent"])

    def test_the_tokenizer_tells_the_two_apart(self):
        """The tokenizer tells the two apart."""
        soft = tokenize("Prigent, bonjour")
        hard = tokenize("Prigent. Bonjour")
        self.assertTrue(soft[0].brk_after)
        self.assertFalse(soft[0].hard_after)
        self.assertTrue(hard[0].hard_after)


class TestFirstPersonSubstitution(unittest.TestCase):
    """The most expensive error in the corpus: case 05, "je remplace Léa"."""

    def test_je_remplace_lea_is_a_mention_not_a_self_id(self):
        """Je remplace lea is a mention not a self id."""
        cues = cues_of(
            [
                (
                    " Je remplace Léa sur le volet accessibilité pendant son congé.",
                    "SPEAKER_03",
                )
            ]
        )
        self.assertEqual(kinds(cues, SELF_ID), [])
        self.assertEqual([c.name for c in kinds(cues, MENTION)], ["Léa Marchand"])

    def test_lea_is_never_attributed_to_the_substitute(self):
        """Lea is never attributed to the substitute."""
        segments = [
            segment(" Bonjour, c'est Nourdine , désolé pour le retard.", "SPEAKER_03"),
            segment(" Je remplace Léa sur le volet accessibilité.", "SPEAKER_03"),
        ]
        result = resolve_speaker_identities_from_cues(ROSTER, {"segments": segments})
        names = [a.participant_name for a in result.assignments]
        self.assertNotIn("Léa Marchand", names)
        self.assertEqual(names, ["Nourdine"])

    def test_substitution_phrasing_demotes_even_a_strong_template(self):
        """Substitution phrasing demotes even a strong template."""
        # Same template that normally wins, poisoned by the phrasing before it.
        cues = cues_of([" Je parle au nom de Léa à l'appareil."])
        self.assertEqual(kinds(cues, SELF_ID), [])

    def test_substitution_does_not_bleed_into_the_next_sentence(self):
        """Substitution does not bleed into the next sentence."""
        # "Je remplace Bruno." poisons nothing after the full stop:
        # "Ici Sylvie." is a perfectly good self-identification.
        cues = cues_of([" Je remplace Bruno. Ici Sylvie."])
        self.assertEqual([c.name for c in kinds(cues, SELF_ID)], ["Sylvie Aubert"])
        self.assertEqual([c.name for c in kinds(cues, MENTION)], ["Bruno Delcourt"])

    def test_a_third_person_mention_alone_attributes_nobody(self):
        """A third person mention alone attributes nobody."""
        segments = [
            segment(
                " Léa est en congé cette semaine, elle m'a transmis ses éléments.",
                "SPEAKER_00",
            )
        ]
        result = resolve_speaker_identities_from_cues(ROSTER, {"segments": segments})
        self.assertEqual(result.assignments, [])
        self.assertEqual(result.unassigned_speakers, ["SPEAKER_00"])


class TestOutOfRosterSelfId(unittest.TestCase):
    """Out of roster self id."""

    def test_a_late_joiner_keeps_the_spoken_name_and_no_email(self):
        """A late joiner keeps the spoken name and no email."""
        cues = kinds(
            cues_of([" Bonjour, c'est Nourdine , désolé pour le retard."]), SELF_ID
        )
        self.assertEqual([c.name for c in cues], ["Nourdine"])
        self.assertEqual(cues[0].email, "")
        self.assertEqual(cues[0].tier, "out_of_roster")

    def test_the_path_can_be_switched_off(self):
        """The path can be switched off."""
        cues = cues_of(
            [" Bonjour, c'est Nourdine , désolé."], allow_unknown_self_id=False
        )
        self.assertEqual(kinds(cues, SELF_ID), [])

    def test_a_grey_zone_name_produces_nothing_at_all(self):
        """A grey zone name produces nothing at all."""
        # "Marceau" is close enough to Marc Teyssier to be a mangling and far
        # enough to be somebody else. Unsure means silent -- neither an
        # attribution to Marc nor the invention of a new person.
        cues = cues_of([" Bonjour, c'est Marceau, du service support."])
        self.assertEqual(kinds(cues, SELF_ID), [])

    def test_a_clear_mangling_is_still_read_as_the_invitee(self):
        """A clear mangling is still read as the invitee."""
        # Above the fuzzy threshold the roster wins: with only an invite to
        # go on, "Sylvia" is Sylvie Aubert misheard, not a new person.
        cues = kinds(cues_of([" Bonjour, c'est Sylvia, je commence."]), SELF_ID)
        self.assertEqual([(c.name, c.tier) for c in cues], [("Sylvie Aubert", "fuzzy")])


class TestHandoff(unittest.TestCase):
    """Handoff."""

    def test_handoff_labels_the_next_different_speaker(self):
        """Handoff labels the next different speaker."""
        segments = [
            segment(
                " Je vous propose de commencer. Julien, tu démarres ?", "SPEAKER_00"
            ),
            segment(" Volontiers. La migration est terminée.", "SPEAKER_01"),
        ]
        roster = [{"name": "Julien Mercier", "email": "julien.mercier@dinum.gouv.fr"}]
        result = resolve_speaker_identities_from_cues(roster, {"segments": segments})
        self.assertEqual(
            [(a.speaker_label, a.participant_name) for a in result.assignments],
            [("SPEAKER_01", "Julien Mercier")],
        )

    def test_a_pronoun_without_an_adjacent_name_is_not_a_handoff(self):
        """A pronoun without an adjacent name is not a handoff."""
        # Administrative French is full of "je vous propose" / "à votre avis".
        cues = cues_of([" Je vous propose de garder les questions pour la fin."])
        self.assertEqual(kinds(cues, HANDOFF), [])

    def test_the_name_must_be_set_off_by_punctuation(self):
        """The name must be set off by punctuation."""
        cues = cues_of([" Comme Bruno tu sais bien, le calendrier glisse."])
        self.assertEqual(kinds(cues, HANDOFF), [])

    def test_a_handoff_nobody_answers_is_dropped(self):
        """A handoff nobody answers is dropped."""
        segments = [
            segment(" Bruno, tu peux faire un export ?", "SPEAKER_00"),
            segment(" Je continue sur le point précédent.", "SPEAKER_00"),
            segment(" Toujours moi.", "SPEAKER_00"),
            segment(" Bon, on passe à la suite.", "SPEAKER_00"),
            segment(" Oui, d'accord.", "SPEAKER_01"),
        ]
        result = resolve_speaker_identities_from_cues(ROSTER, {"segments": segments})
        self.assertEqual(result.assignments, [])


class TestAmbiguousHandoffIsNotAGuess(unittest.TestCase):
    """Case 04: "Jean, tu peux ...?" with two Jeans on the invite."""

    def test_neither_jean_is_attributed(self):
        """Neither jean is attributed."""
        segments = [
            segment(
                " Ici Hélène , je pilote. Jean, tu peux faire un export ?", "SPEAKER_00"
            ),
            segment(" Oui, je sors le tableau ce soir.", "SPEAKER_01"),
            segment(
                " Jean, tu confirmes que la ligne peut recevoir ce montant ?",
                "SPEAKER_00",
            ),
            segment(" Sur le plan comptable, oui.", "SPEAKER_02"),
        ]
        result = resolve_speaker_identities_from_cues(
            ROSTER_JEAN, {"segments": segments}
        )
        self.assertEqual(
            [(a.speaker_label, a.participant_name) for a in result.assignments],
            [("SPEAKER_00", "Hélène Charpentier")],
        )
        self.assertEqual(result.unassigned_speakers, ["SPEAKER_01", "SPEAKER_02"])

    def test_the_cue_is_still_detected_as_ambiguous(self):
        """The cue is still detected as ambiguous."""
        cues = kinds(
            cues_of([" Jean, tu peux faire un export ?"], ROSTER_JEAN), HANDOFF
        )
        self.assertEqual(len(cues), 1)
        self.assertTrue(cues[0].ambiguous)
        self.assertEqual(cues[0].candidates, ["Jean Dupont", "Jean Moreau"])


class TestMangledSelfIdEndToEnd(unittest.TestCase):
    """Mangled self id end to end."""

    def test_names_split_across_tokens_are_recovered(self):
        """Names split across tokens are recovered."""
        segments = [
            segment(" Bonjour à tous. Ici carie msahli.", "SPEAKER_00"),
            segment(
                " ma tilde à l'appareil. Sur le chiffrement, tout va bien.",
                "SPEAKER_01",
            ),
            segment(" Ici vingt sang, à la sous-direction technique.", "SPEAKER_02"),
        ]
        result = resolve_speaker_identities_from_cues(
            ROSTER_MANGLED, {"segments": segments}
        )
        self.assertEqual(
            [(a.speaker_label, a.participant_name) for a in result.assignments],
            [
                ("SPEAKER_00", "Karim Sahli"),
                ("SPEAKER_01", "Mathilde Fontaine"),
                ("SPEAKER_02", "Vincent Delaunay"),
            ],
        )

    def test_a_mangled_name_stays_below_a_clean_one(self):
        """A mangled name stays below a clean one."""
        clean = resolve_speaker_identities_from_cues(
            ROSTER_MANGLED, {"segments": [segment(" Ici Karim.", "SPEAKER_00")]}
        ).assignments[0]
        mangled = resolve_speaker_identities_from_cues(
            ROSTER_MANGLED, {"segments": [segment(" Ici carie msahli.", "SPEAKER_00")]}
        ).assignments[0]
        self.assertGreater(clean.score, mangled.score)


class TestUniquenessAndThreshold(unittest.TestCase):
    """Uniqueness and threshold."""

    def test_one_attendee_cannot_hold_two_labels(self):
        """One attendee cannot hold two labels."""
        segments = [
            segment(" Ici Bruno, je commence.", "SPEAKER_00"),
            segment(" Bruno, tu peux faire un export ?", "SPEAKER_00"),
            segment(" Oui, tout de suite.", "SPEAKER_01"),
        ]
        result = resolve_speaker_identities_from_cues(ROSTER, {"segments": segments})
        self.assertEqual(
            [(a.speaker_label, a.participant_name) for a in result.assignments],
            [("SPEAKER_00", "Bruno Delcourt")],
        )
        # The weaker claim is dropped, never handed to its runner-up.
        self.assertEqual(result.unassigned_speakers, ["SPEAKER_01"])

    def test_the_threshold_is_honoured(self):
        """The threshold is honoured."""
        segments = [segment(" Ici Bruno, je commence.", "SPEAKER_00")]
        assigned = resolve_speaker_identities_from_cues(
            ROSTER, {"segments": segments}, confidence_threshold=0.9
        )
        self.assertEqual(len(assigned.assignments), 1)
        refused = resolve_speaker_identities_from_cues(
            ROSTER, {"segments": segments}, confidence_threshold=0.99
        )
        self.assertEqual(refused.assignments, [])
        self.assertEqual(refused.unassigned_speakers, ["SPEAKER_00"])

    def test_repeated_self_ids_raise_confidence(self):
        """Repeated self ids raise confidence."""
        once = resolve_speaker_identities_from_cues(
            ROSTER, {"segments": [segment(" Ici Bruno.", "SPEAKER_00")]}
        ).assignments[0]
        twice = resolve_speaker_identities_from_cues(
            ROSTER,
            {
                "segments": [
                    segment(" Ici Bruno.", "SPEAKER_00"),
                    segment(" Bruno à l'appareil, je reprends.", "SPEAKER_00"),
                ]
            },
        ).assignments[0]
        self.assertGreater(twice.score, once.score)


class TestResultShape(unittest.TestCase):
    """The whole point of the fallback: apply_to() must work unchanged."""

    def test_apply_to_replaces_the_labels(self):
        """Apply to replaces the labels."""
        diarization = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": " Ici Bruno.",
                    "words": [{"word": "Ici", "speaker": "SPEAKER_00"}],
                },
                {"speaker": "SPEAKER_01", "text": " Rien d'utile ici."},
            ],
            "word_segments": [{"word": "Ici", "speaker": "SPEAKER_00"}],
        }
        result = resolve_speaker_identities_from_cues(ROSTER, diarization)
        applied = result.apply_to(diarization)
        self.assertEqual(applied["segments"][0]["speaker"], "Bruno Delcourt")
        self.assertEqual(
            applied["segments"][0]["words"][0]["speaker"], "Bruno Delcourt"
        )
        self.assertEqual(applied["segments"][1]["speaker"], "SPEAKER_01")
        self.assertEqual(applied["word_segments"][0]["speaker"], "Bruno Delcourt")

    def test_every_label_is_either_assigned_or_unassigned_exactly_once(self):
        """Every label is either assigned or unassigned exactly once."""
        segments = [
            segment(" Ici Bruno.", "SPEAKER_00"),
            segment(" Rien à signaler.", "SPEAKER_01"),
            segment(" Ici Sylvie.", "SPEAKER_02"),
        ]
        result = resolve_speaker_identities_from_cues(ROSTER, {"segments": segments})
        seen = [
            a.speaker_label for a in result.assignments
        ] + result.unassigned_speakers
        self.assertEqual(sorted(seen), ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
