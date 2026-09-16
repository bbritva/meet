"""Tests for the glue between the worker and the cue resolver.

The ported modules are covered by their own tests. What is NOT ported, and so
has no coverage behind it, is the adapter that lets the cue detector talk
through the application's `LLMService`, and the settings that drive it.

The LLM detector is not the default, so nothing else in the suite would catch
a wrong argument order here: it would fail only in production, with the LLM
detector switched on.
"""

import unittest
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from summary.core.celery_worker import _cue_options
from summary.core.config import AuthorizedTenant, Settings
from summary.core.speaker_cues.llm_cues import transport_from_llm_service


class RecordingLLMService:
    """Stands in for `LLMService`, remembering exactly how it was called."""

    def __init__(self, answer="[]"):
        """Set the canned answer the fake service returns."""
        self.answer = answer
        self.calls = []

    def call(self, system_prompt, user_prompt, name=None, response_format=None):
        """Record one call and return the canned answer."""
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "name": name,
            }
        )
        return self.answer


class TestTransportAdapter(unittest.TestCase):
    """The adapter from LLMService to the detector's `complete(system, user)`."""

    def test_system_and_user_prompts_keep_their_order(self):
        """System and user prompts keep their order."""
        service = RecordingLLMService()
        complete = transport_from_llm_service(service)

        complete("SYSTEM", "USER")

        self.assertEqual(len(service.calls), 1)
        self.assertEqual(service.calls[0]["system_prompt"], "SYSTEM")
        self.assertEqual(service.calls[0]["user_prompt"], "USER")

    def test_the_answer_is_passed_straight_through(self):
        """The answer is passed straight through."""
        complete = transport_from_llm_service(RecordingLLMService('[{"index": 0}]'))
        self.assertEqual(complete("s", "u"), '[{"index": 0}]')

    def test_calls_are_named_for_observability(self):
        """Calls are named for observability."""
        service = RecordingLLMService()
        transport_from_llm_service(service)("s", "u")
        self.assertEqual(service.calls[0]["name"], "speaker-cues")

    def test_a_none_answer_becomes_an_empty_string(self):
        """A None answer becomes an empty string.

        `LLMService.call` returns the message content, which the API may
        report as null. The parser expects a string.
        """
        self.assertEqual(
            transport_from_llm_service(RecordingLLMService(None))("s", "u"), ""
        )


def _settings(**overrides):
    """Build a Settings with the required fields filled in."""
    base = {
        "authorized_tenants": (
            AuthorizedTenant(
                webhook_url="https://example.com/webhook",
                id="test-tenant",
                api_key=SecretStr("test-api-token"),
                webhook_api_key=SecretStr("test-webhook-api-key"),
            ),
        ),
        "aws_storage_bucket_name": "bucket",
        "aws_s3_endpoint_url": "minio:9000",
        "aws_s3_access_key_id": "key",
        "aws_s3_secret_access_key": SecretStr("secret"),
        "whisperx_api_key": SecretStr("whisperx"),
        "llm_base_url": "https://example.com",
        "llm_api_key": SecretStr("llm"),
        "llm_model": "model",
    }
    base.update(overrides)
    return Settings(**base)


class TestCueOptions(unittest.TestCase):
    """The worker reads the settings, so the cue modules never have to."""

    def test_the_regex_default_builds_no_llm_client(self):
        """The regex default builds no LLM client."""
        options = _cue_options("task-1", "user-1")
        self.assertEqual(options["detector"], "regex")
        self.assertNotIn("detector_options", options)

    def test_the_thresholds_come_from_the_settings(self):
        """The thresholds come from the settings."""
        options = _cue_options("task-1", "user-1")
        self.assertEqual(options["confidence_threshold"], 0.6)
        self.assertEqual(options["fuzzy_threshold"], 0.75)
        self.assertTrue(options["allow_unknown_self_id"])

    @patch("summary.core.celery_worker.analytics")
    @patch("summary.core.celery_worker.LLMService")
    def test_the_llm_detector_gets_a_transport_and_a_ceiling(
        self, mock_llm_service, mock_analytics
    ):
        """The llm detector gets a transport and a ceiling."""
        mock_llm_service.return_value = RecordingLLMService()
        # Settings is frozen, so swap the whole object rather than one field.
        with patch(
            "summary.core.celery_worker.settings",
            _settings(resolve_speaker_cues_detector="llm"),
        ):
            options = _cue_options("task-1", "user-1")

        detector_options = options["detector_options"]
        self.assertTrue(callable(detector_options["complete"]))
        self.assertEqual(detector_options["max_calls"], 400)
        self.assertEqual(detector_options["batch_size"], 12)


class TestDetectorSetting(unittest.TestCase):
    """A misconfigured detector must fail at startup, not per task."""

    def test_the_two_known_detectors_are_accepted(self):
        """The two known detectors are accepted."""
        for detector in ("regex", "llm"):
            self.assertEqual(
                _settings(
                    resolve_speaker_cues_detector=detector
                ).resolve_speaker_cues_detector,
                detector,
            )

    def test_an_unknown_detector_is_refused(self):
        """An unknown detector is refused.

        Without this the worker would swallow the resolver's ValueError and
        log "skipping speaker assignment", so a typo would look like the
        feature quietly not working.
        """
        with pytest.raises(ValidationError, match="resolve_speaker_cues_detector"):
            _settings(resolve_speaker_cues_detector="Regex")
