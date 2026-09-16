"""Unit tests for the roster matcher."""

import unittest
from difflib import SequenceMatcher

from summary.core.speaker_cues.match import (
    FUZZY_GREY_ZONE,
    best_roster_similarity,
    match_span,
    normalise,
    phonetic_key,
    similarity,
)

ROSTER_03 = [
    {"name": "Karim Sahli", "email": "karim.sahli@dinum.gouv.fr"},
    {"name": "Mathilde Fontaine", "email": "mathilde.fontaine@dinum.gouv.fr"},
    {"name": "Patrick Estève", "email": "patrick.esteve@dinum.gouv.fr"},
    {"name": "Soraya Benali", "email": "soraya.benali@ssi.gouv.fr"},
    {"name": "Vincent Delaunay", "email": "vincent.delaunay@dinum.gouv.fr"},
]

ROSTER_04 = [
    {"name": "Hélène Charpentier", "email": "helene.charpentier@dinum.gouv.fr"},
    {"name": "Jean Dupont", "email": "jean.dupont@dinum.gouv.fr"},
    {"name": "Jean Moreau", "email": "jean.moreau@dinum.gouv.fr"},
    {"name": "Rachid Benyoub", "email": "rachid.benyoub@dinum.gouv.fr"},
]

ROSTER_05 = [
    {"name": "Bruno Delcourt", "email": "bruno.delcourt@dinum.gouv.fr"},
    {"name": "Léa Marchand", "email": "lea.marchand@dinum.gouv.fr"},
    {"name": "Marc Teyssier", "email": "marc.teyssier@culture.gouv.fr"},
    {"name": "Sylvie Aubert", "email": "sylvie.aubert@dinum.gouv.fr"},
]


class TestNormalisation(unittest.TestCase):
    """Normalisation."""

    def test_accents_and_punctuation_are_dropped(self):
        """Accents and punctuation are dropped."""
        self.assertEqual(normalise("Bérangère"), "berangere")
        self.assertEqual(normalise("Gaël Prigent"), "gaelprigent")
        self.assertEqual(normalise("Estève"), "esteve")

    def test_silent_h_disappears_from_the_phonetic_key(self):
        """Silent h disappears from the phonetic key."""
        self.assertEqual(phonetic_key("Mathilde"), phonetic_key("matilde"))

    def test_nasal_vowels_collapse_with_their_silent_consonant(self):
        """Nasal vowels collapse with their silent consonant."""
        # "vingt" and "vin" sound alike; the g and t are not pronounced.
        self.assertEqual(phonetic_key("vingt sang"), phonetic_key("Vincent"))

    def test_a_pronounced_consonant_after_a_nasal_survives(self):
        """A pronounced consonant after a nasal survives."""
        # The s of "Vincent" is heard; the rule must not eat it.
        self.assertNotEqual(phonetic_key("Vincent"), phonetic_key("vint"))


class TestExactAndNormalisedTiers(unittest.TestCase):
    """Exact and normalised tiers."""

    def test_first_name_matches_exactly(self):
        """First name matches exactly."""
        match = match_span("Rachid", ROSTER_04)
        self.assertEqual(match.best.name, "Rachid Benyoub")
        self.assertEqual(match.best.tier, "exact")
        self.assertEqual(match.best.email, "rachid.benyoub@dinum.gouv.fr")

    def test_accents_are_not_required(self):
        """Accents are not required."""
        match = match_span("Helene", ROSTER_04)
        self.assertEqual(match.best.name, "Hélène Charpentier")
        self.assertEqual(match.best.tier, "normalised")

    def test_a_word_that_is_nobody_matches_nobody(self):
        """A word that is nobody matches nobody."""
        for word in ("un", "le", "exactement", "beaucoup", "bonjour"):
            self.assertEqual(match_span(word, ROSTER_04).candidates, [], word)


class TestAmbiguousFirstName(unittest.TestCase):
    """The trap of case 04: two invitees share a first name."""

    def test_jean_returns_both_jeans(self):
        """Jean returns both jeans."""
        match = match_span("Jean", ROSTER_04)
        self.assertEqual(
            [c.name for c in match.candidates], ["Jean Dupont", "Jean Moreau"]
        )

    def test_jean_is_flagged_ambiguous(self):
        """Jean is flagged ambiguous."""
        self.assertTrue(match_span("Jean", ROSTER_04).is_ambiguous)

    def test_a_full_name_disambiguates(self):
        """A full name disambiguates."""
        match = match_span("Jean Moreau", ROSTER_04)
        self.assertFalse(match.is_ambiguous)
        self.assertEqual(match.best.name, "Jean Moreau")

    def test_an_unshared_first_name_is_not_ambiguous(self):
        """An unshared first name is not ambiguous."""
        self.assertFalse(match_span("Hélène", ROSTER_04).is_ambiguous)


class TestMangledNamesCrossingTokenBoundaries(unittest.TestCase):
    """The trap of case 03: one name, several transcribed tokens."""

    CROSS_BOUNDARY = [
        ("carie msahli", "Karim Sahli"),
        ("ma tilde", "Mathilde Fontaine"),
        ("sois raya", "Soraya Benali"),
        ("vingt sang", "Vincent Delaunay"),
    ]

    def test_multi_token_spans_recover_the_invitee(self):
        """Multi token spans recover the invitee."""
        for span, expected in self.CROSS_BOUNDARY:
            match = match_span(span, ROSTER_03)
            self.assertIsNotNone(match.best, span)
            self.assertEqual(match.best.name, expected, span)
            self.assertEqual(match.best.tier, "fuzzy", span)
            self.assertFalse(match.is_ambiguous, span)

    def test_a_single_token_of_a_split_name_is_not_enough(self):
        """A single token of a split name is not enough."""
        # "vingt" alone is too short a hook; the resolver must look at the
        # 2-token span or stay silent.
        self.assertEqual(match_span("vingt", ROSTER_03).candidates, [])

    def test_the_phonetic_key_is_what_rescues_vingt_sang(self):
        """The phonetic key is what rescues vingt sang."""
        letters = SequenceMatcher(
            None, normalise("vingt sang"), normalise("Vincent")
        ).ratio()
        self.assertLess(letters, 0.75)
        self.assertGreaterEqual(similarity("vingt sang", "Vincent"), 0.75)

    def test_the_letter_ratio_is_what_rescues_sois_raya(self):
        """The letter ratio is what rescues sois raya."""
        # oi -> wa moves "soisraya" away from "soraya": the phonetic key
        # alone would miss this one, which is why we take the max of both.
        phonetic = SequenceMatcher(
            None, phonetic_key("sois raya"), phonetic_key("Soraya")
        ).ratio()
        self.assertLess(phonetic, 0.75)
        self.assertGreaterEqual(similarity("sois raya", "Soraya"), 0.75)


class TestOutOfRosterDistance(unittest.TestCase):
    """Out of roster distance."""

    def test_a_late_joiner_is_far_from_everybody(self):
        """A late joiner is far from everybody."""
        # Case 05: Nourdine is on nobody's invite. If this ever crept into
        # the grey zone the resolver would have an ordering bug.
        self.assertLess(best_roster_similarity("Nourdine", ROSTER_05), FUZZY_GREY_ZONE)

    def test_a_real_invitee_is_not_far(self):
        """A real invitee is not far."""
        self.assertGreaterEqual(best_roster_similarity("Sylvie", ROSTER_05), 0.99)
