"""Celery workers."""

# ruff: noqa: PLR0913

import json
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import requests
import sentry_sdk
from celery import Celery, signals
from celery.utils.log import get_task_logger
from openai.types.audio import Transcription
from requests import exceptions

from summary.core.acronym_correction import correct_acronyms
from summary.core.analytics import MetadataManager, get_analytics
from summary.core.config import get_settings
from summary.core.docs_service import create_document_in_lasuite_docs
from summary.core.file_service import (
    CorruptedAudioFile,
    FileService,
    FileServiceException,
    TranscribeError,
)
from summary.core.llm_service import LLMException, LLMObservability, LLMService
from summary.core.locales import get_locale
from summary.core.models import (
    PushToDocsBaseConfig,
    RecordingMetadata,
    SummarizeTaskJob,
    TranscribeTaskJob,
)
from summary.core.prompt import (
    FORMAT_NEXT_STEPS,
    FORMAT_PLAN,
    PROMPT_SYSTEM_CLEANING,
    PROMPT_SYSTEM_NEXT_STEP,
    PROMPT_SYSTEM_PART,
    PROMPT_SYSTEM_PLAN,
    PROMPT_SYSTEM_TLDR,
    PROMPT_USER_PART,
)
from summary.core.shared_models import (
    SummarizeWebhookFailurePayload,
    SummarizeWebhookSuccessPayload,
    TranscribeWebhookFailurePayload,
    TranscribeWebhookSuccessPayload,
    WhisperXResponse,
    webhook_payload_adapter,
)
from summary.core.speaker_cues.attendees import normalise_attendees
from summary.core.speaker_cues.llm_cues import transport_from_llm_service
from summary.core.speaker_dispatch import resolve_speakers
from summary.core.transcript_formatter import TranscriptFormatter
from summary.core.webhook_service import (
    call_webhook_v2,
)

settings = get_settings()
analytics = get_analytics()

metadata_manager = MetadataManager()

logger = get_task_logger(__name__)

celery = Celery(
    __name__,
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    broker_connection_retry_on_startup=True,
    # To store the tasks args too in results and make the
    # V2 API work
    result_extended=True,
)

celery.config_from_object("summary.core.celery_config")

if settings.sentry_dsn and settings.sentry_is_enabled:

    @signals.celeryd_init.connect
    def init_sentry(**_kwargs):
        """Initialize sentry."""
        sentry_sdk.init(dsn=settings.sentry_dsn, enable_tracing=True)


file_service = FileService()


def transcribe_audio(
    *,
    task_id: str,
    language: str,
    cloud_storage_url: str,
    raises: bool = False,
):
    """Transcribe an audio file using WhisperX.

    Downloads the audio from a cloud storage URL, sends it to
    WhisperX for transcription, and tracks metadata throughout the process.

    Returns the transcription object, or None if the file could not be retrieved.
    """
    logger.info("Initiating WhisperX client")

    # Transcription
    try:
        with file_service.prepare_audio_file(
            cloud_storage_url=cloud_storage_url,
        ) as (audio_file, metadata):
            metadata_manager.track(task_id, {"audio_length": metadata["duration"]})

            # Compute language parameter
            if language is None:
                language = settings.whisperx_default_language
                logger.info(
                    "No language specified, using default from settings: %s",
                    (language or "auto-detect"),
                )
            else:
                logger.info(
                    "Querying transcription in '%s' language",
                    language,
                )

            # Call remote service for transcription
            transcription_start_time = time.time()

            api_key = settings.whisperx_api_key.get_secret_value()
            base_url = settings.whisperx_base_url

            # We use a manual call to the transcripion endpoint, and we do not
            # directly use the OpenAI lib for this.
            # This is because, depending on the requested response format,
            # the OpenAI lib will cast the response to a different dataclass,
            # which can result in stripping out keys & data that we are interested in.
            # This is in particular true for word_segments and words.
            # WhisperX response is slightly different from OpenAI STT endpoints
            # response.
            # At the same time "diarized_json" should be the value
            # provided to STT endpoints in our context.
            url = urljoin(base_url.rstrip("/") + "/", "audio/transcriptions")
            res = requests.post(
                url,
                data={
                    "model": settings.whisperx_asr_model,
                    "language": language,
                    "timestamp_granularities": ["word", "segment"],
                    "response_format": "diarized_json",
                },
                # `requests` omits the Content-Type header when given a
                # bare file object. Some OpenAI-compatible ASR endpoints
                # reject that multipart body (the Albert API returns 500),
                # so send an explicit filename and content type.
                files={
                    "file": (
                        os.path.basename(audio_file.name),
                        audio_file,
                        "application/octet-stream",
                    )
                },
                headers={"Authorization": f"Bearer {api_key}"},
                # Mimic OpenAI's timeout settings
                timeout=(60, 10 * 60),
            )
            if res.status_code == 400:
                logger.info(
                    "WhisperX transcription failed, "
                    "likely due to a corrupted audio file: %s",
                    res.text,
                )
                raise CorruptedAudioFile("WhisperX coudln't decode the audio file.")

            try:
                res.raise_for_status()
            except requests.exceptions.HTTPError:
                logger.exception("WhisperX transcription failed")
                # We reraise the error so that it can be retried by celery
                raise

            transcription_json: dict[str, Any] = res.json()
            # We remove the "usage" key from the transcription_json dictionary
            # as it may cause issues with parsing inside the Transcription model
            # Some API don't share the exact same structure for the "usage" key
            transcription_json.pop("usage", None)

            # We force the use of the Transcription model here
            # to avoid changing too much code for now.
            # Note that it should be WhisperXResponse instead.
            transcription = Transcription.model_validate(
                # We add a dummy "text" to make the model validate,
                # Some API responses lack the "text" key.
                {"text": "", **transcription_json},
                extra="allow",
                strict=False,
            )

            # Logging
            transcription_duration = round(time.time() - transcription_start_time, 2)
            metadata_manager.track(
                task_id,
                {"transcription_time": transcription_duration},
            )
            logger.info(
                "Transcription received in %.2f seconds.", transcription_duration
            )
            logger.debug("Transcription: \n %s", transcription)

    except FileServiceException as e:
        # For v2 pipeline we want failures not silent errors like this
        if raises:
            raise e
        redacted_cloud_storage_url = (
            cloud_storage_url.split("?", 1)[0] if cloud_storage_url else None
        )
        logger.exception(
            ("Unexpected error while preparing file %s "),
            redacted_cloud_storage_url,
        )
        return None

    metadata_manager.track_transcription_metadata(task_id, transcription)
    return transcription


def _read_recording_metadata(
    recording_metadata: RecordingMetadata | None, task_id
) -> dict | None:
    """Fetch the VAD metadata blob, or None when it cannot be used.

    An unreadable blob is no longer fatal: it is one of the two cases the
    cue-based fallback exists for, so we log and let the caller try that route
    instead of skipping speaker assignment entirely.

    Args:
        recording_metadata: The payload's metadata block, or None.
        task_id: current task id, for logging purposes

    Returns:
        The parsed metadata dict, or None.
    """
    if recording_metadata is None:
        return None

    logger.debug(
        "recording_start_dt: %s ; recording_end_dt: %s",
        recording_metadata.started_at,
        recording_metadata.ended_at,
    )
    try:
        return file_service.read_cloud_storage_json(
            recording_metadata.cloud_storage_url
        )
    except FileServiceException as exc:
        logger.error(
            "Error reading metadata for task %s; falling back to name cues. Error: %s",
            task_id,
            exc,
        )
        return None


def _cue_options(task_id, user_sub: str) -> dict:
    """Build the cue resolver's keyword arguments from the settings.

    Settings are read here and passed down, so the cue modules themselves stay
    free of any configuration dependency -- which is also what lets their unit
    tests run without an environment.

    Args:
        task_id: current task id, used as the observability session id
        user_sub: owner of the recording, for observability

    Returns:
        Keyword arguments for `resolve_speaker_identities_from_cues`.
    """
    options = {
        "confidence_threshold": settings.resolve_speaker_cues_confidence_threshold,
        "fuzzy_threshold": settings.resolve_speaker_cues_fuzzy_threshold,
        "allow_unknown_self_id": settings.resolve_speaker_cues_allow_unknown_self_id,
        "detector": settings.resolve_speaker_cues_detector,
    }

    if settings.resolve_speaker_cues_detector != "llm":
        return options

    # Only the LLM detector needs a transport, and it is not the default: a
    # regex run must never build an LLM client it will not use.
    user_has_tracing_consent = analytics.is_feature_enabled(
        "summary-tracing-consent", distinct_id=user_sub
    )
    llm_service = LLMService(
        llm_observability=LLMObservability(
            user_has_tracing_consent=user_has_tracing_consent,
            session_id=task_id,
            user_id=user_sub,
        )
    )
    options["detector_options"] = {
        "complete": transport_from_llm_service(llm_service),
        "model": settings.llm_model,
        "batch_size": settings.resolve_speaker_cues_llm_batch_size,
        "max_calls": settings.resolve_speaker_cues_max_llm_calls,
    }
    return options


def resolve_speaker_identities_and_apply_to(
    *,
    transcription: WhisperXResponse,
    recording_metadata: RecordingMetadata | None,
    task_id,
    attendees=None,
    user_sub: str = "",
) -> WhisperXResponse:
    """Assign users to detected speakers and rewrite the transcriptions.

    Dispatches between the two resolvers: VAD overlap when the recording
    metadata is present and usable, spoken name cues otherwise. The VAD path
    keeps priority because it measures who spoke rather than inferring it.

    Args:
        transcription: output of meet-whisperx after transcription and diarization
        recording_metadata: Metadata of the recording, or None
        task_id: current task id, for logging purposes
        attendees: the meeting's attendee list, or None
        user_sub: owner of the recording, for observability

    Returns:
        The transcription with speaker labels replaced, or unchanged when no
        resolver could say anything.
    """
    logger.debug("Running speaker resolution")
    try:
        metadata = (
            _read_recording_metadata(recording_metadata, task_id)
            if settings.is_resolve_speaker_identities_enabled
            else None
        )
        roster = (
            normalise_attendees(attendees)
            if settings.is_resolve_speaker_cues_enabled
            else []
        )

        dispatched = resolve_speakers(
            transcription.model_dump(),
            metadata=metadata,
            recording_start=(
                recording_metadata.started_at if recording_metadata else None
            ),
            recording_end=recording_metadata.ended_at if recording_metadata else None,
            attendees=roster,
            cue_options=_cue_options(task_id, user_sub),
        )
        logger.info(
            "Speaker resolution for task %s: source=%s (%s), %d assigned,"
            " %d unassigned",
            task_id,
            dispatched.source,
            dispatched.reason,
            len(dispatched.result.assignments),
            len(dispatched.result.unassigned_speakers),
        )

        new_transcription = dispatched.result.apply_to(transcription.model_dump())
        return WhisperXResponse.model_validate(new_transcription)

    except Exception as exc:
        logger.exception(
            "Speaker resolution failed for task %s; skipping"
            " speaker assignment. Error: %s",
            task_id,
            exc,
        )
        return transcription


def _correct_acronyms_in(
    *, transcription: WhisperXResponse, user_sub: str, task_id: str
) -> WhisperXResponse:
    """Correct mistranscribed acronyms and rewrite the transcription.

    Args:
        transcription: output of meet-whisperx, possibly with speakers resolved
        user_sub: owner of the recording, for observability
        task_id: current task id, used as the observability session id
    """
    user_has_tracing_consent = analytics.is_feature_enabled(
        "summary-tracing-consent", distinct_id=user_sub
    )
    llm_service = LLMService(
        llm_observability=LLMObservability(
            user_has_tracing_consent=user_has_tracing_consent,
            session_id=task_id,
            user_id=user_sub,
        )
    )

    corrected, corrections = correct_acronyms(
        transcription=transcription,
        llm_service=llm_service,
    )
    logger.info(
        "Acronym correction for task %s: %d correction(s), %d applied",
        task_id,
        len(corrections),
        sum(1 for correction in corrections if correction["applied"]),
    )
    return WhisperXResponse.model_validate(corrected)


def format_transcript(
    transcription,
    context_language: str | None,
    language: str,
    download_link: str | None,
    form_link: str | None,
) -> str:
    """Format a transcription into readable content with a title.

    Resolves the locale from context_language / language, then uses
    TranscriptFormatter to produce markdown content and a title.

    Returns a (content, title) tuple.
    """
    locale = get_locale(context_language, language)
    formatter = TranscriptFormatter(locale)

    return formatter.format(
        transcription,
        download_link=download_link,
        form_link=form_link,
    )


def format_actions(llm_output: dict) -> str:
    """Format the actions from the LLM output into a markdown list.

    format:
    - [ ] Action title Assignée à : assignee1, assignee2, Échéance : due_date
    """
    lines = []
    for action in llm_output.get("actions", []):
        title = action.get("title", "").strip()
        assignees = ", ".join(action.get("assignees", [])) or "-"
        due_date = action.get("due_date") or "-"
        line = f"- [ ] {title} Assignée à : {assignees}, Échéance : {due_date}"
        lines.append(line)
    if lines:
        return "### Prochaines étapes\n\n" + "\n".join(lines)
    return ""


def summarize_transcription_internals(
    *, distinct_id: str, transcript: str, session_id: str
) -> str:
    """Generate a summary from the provided transcription text.

    1. Uses an LLM to generate a TL;DR summary of the transcription.
    2. Breaks the transcription into parts and summarizes each part.
    3. Cleans up the combined summary
    4. Generates next steps.
    """
    logger.info(
        "Starting summarization task | Owner: %s",
        distinct_id,
    )

    user_has_tracing_consent = analytics.is_feature_enabled(
        "summary-tracing-consent", distinct_id=distinct_id
    )

    # NOTE: We must instantiate a new LLMObservability client for each task invocation
    # because the masking function needs to be user-specific. The masking function is
    # baked into the Langfuse client at initialization time, so we can't reuse
    # a singleton client. This is a performance trade-off we accept to ensure per-user
    # privacy controls in observability traces.
    llm_observability = LLMObservability(
        user_has_tracing_consent=user_has_tracing_consent,
        session_id=session_id,
        user_id=distinct_id,
    )
    llm_service = LLMService(llm_observability=llm_observability)

    tldr = llm_service.call(PROMPT_SYSTEM_TLDR, transcript, name="tldr")

    logger.info("TLDR generated")

    parts = llm_service.call(
        PROMPT_SYSTEM_PLAN, transcript, name="parts", response_format=FORMAT_PLAN
    )
    logger.info("Plan generated")

    res = json.loads(parts)
    parts = res.get("titles", [])
    logger.info("Parts to summarize: %s", parts)
    parts_summarized = []
    for part in parts:
        prompt_user_part = PROMPT_USER_PART.format(part=part, transcript=transcript)
        logger.info("Summarizing part: %s", part)
        parts_summarized.append(
            llm_service.call(PROMPT_SYSTEM_PART, prompt_user_part, name="part")
        )

    logger.info("Parts summarized")

    raw_summary = "\n\n".join(parts_summarized)

    next_steps = llm_service.call(
        PROMPT_SYSTEM_NEXT_STEP,
        transcript,
        name="next-steps",
        response_format=FORMAT_NEXT_STEPS,
    )

    next_steps = format_actions(json.loads(next_steps))

    logger.info("Next steps generated")

    cleaned_summary = llm_service.call(
        PROMPT_SYSTEM_CLEANING, raw_summary, name="cleaning"
    )
    logger.info("Summary cleaned")

    summary = tldr + "\n\n" + cleaned_summary + "\n\n" + next_steps

    llm_observability.flush()
    logger.debug("LLM observability flushed")

    return summary


##################################################################################
# Tasks v2
##################################################################################


def _should_push_to_docs(
    payload: TranscribeTaskJob | SummarizeTaskJob,
) -> bool:
    """Determines if the transcription should be pushed to docs.

    Based on the payload and settings.
    """
    if not payload.push_to_docs_config:
        reason = "Push to docs is not requested in the payload"
    elif not settings.is_lasuite_docs_integration_enabled:
        reason = "Docs integration is disabled"
    elif not settings.get_authorized_tenant(
        tenant_id=payload.tenant_id
    ).allowed_push_to_docs:
        reason = "Tenant is not allowed to push to docs"
    else:
        return True

    logger.info("Push to docs is not requested: %s", reason)
    return False


def _should_auto_create_summary(payload: TranscribeTaskJob) -> bool:
    """Determines if the transcription should have an auto-created summary.

    Based on the payload and settings.
    """
    if (
        payload.push_to_docs_config is None
        or not payload.push_to_docs_config.auto_create_summary
    ):
        reason = "Auto create summary is not requested in the payload"
    elif not settings.is_summary_enabled:
        reason = "Summary feature is disabled"
    else:
        return True

    logger.info("Auto create summary is not requested: %s", reason)
    return False


@celery.task(
    max_retries=3,
    queue=settings.call_webhook_queue_v2,
    autoretry_for=[exceptions.RequestException],
)
def call_webhook_v2_task(
    payload: dict,
    tenant_id: str,
):
    """Calls a webhook asynchrously (retry handled by celery)."""
    call_webhook_v2(
        payload=webhook_payload_adapter.validate_python(payload), tenant_id=tenant_id
    )


@celery.task(
    bind=True,
    autoretry_for=[
        exceptions.RequestException,
    ],
    max_retries=settings.celery_max_retries,
    queue=settings.transcribe_queue_v2,
)
def process_audio_transcribe_v2_task(
    self,
    payload: dict,
):
    """Process an audio file by transcribing it.

    This Celery task orchestrates:
    1. Audio transcription via WhisperX
    2. Store transcript result on S3
    3. Webhook submission

    Args:
        self: Celery task instance (passed on with bind=True)
        payload: Serialized dictionary of TranscribeSummarizeTaskCreationV2
    """
    payload = TranscribeTaskJob.model_validate(payload)
    logger.info(
        "Transcribing for object received | Owner: %s",
        payload.user_sub,
    )

    job_id = self.request.id

    try:
        transcription_res = WhisperXResponse(
            **transcribe_audio(  # type: ignore
                task_id=job_id,
                cloud_storage_url=payload.cloud_storage_url,
                language=payload.language,
                raises=True,
            ).model_dump()
        )
    except TranscribeError as e:
        failure_payload = TranscribeWebhookFailurePayload(
            job_id=job_id,
            error_code=e.error_code,
        )
        call_webhook_v2_task.apply_async(
            args=[failure_payload.model_dump(), payload.tenant_id]
        )
        return failure_payload.model_dump()

    # Assign speakers and rewrite transcription/diarization output
    can_resolve_from_vad = (
        settings.is_resolve_speaker_identities_enabled and payload.metadata is not None
    )
    can_resolve_from_cues = settings.is_resolve_speaker_cues_enabled and bool(
        payload.attendees
    )
    if can_resolve_from_vad or can_resolve_from_cues:
        try:
            transcription_res = resolve_speaker_identities_and_apply_to(
                transcription=transcription_res,
                recording_metadata=payload.metadata,
                task_id=job_id,
                attendees=payload.attendees,
                user_sub=payload.user_sub,
            )
        except Exception as e:
            logger.error(f"Failed to resolve speaker identities, skipping: {e}")

    # Correct the acronyms the speech recognition model transcribed as words
    if settings.is_acronym_correction_enabled:
        try:
            transcription_res = _correct_acronyms_in(
                transcription=transcription_res,
                user_sub=payload.user_sub,
                task_id=job_id,
            )
        except Exception as e:
            logger.error(f"Failed to correct acronyms, skipping: {e}")

    should_push_to_docs = _should_push_to_docs(payload)
    # We do it synchronously for now
    if should_push_to_docs:
        if payload.push_to_docs_config is None:
            raise ValueError("Push to docs config is missing")

        # Format output
        content = format_transcript(
            transcription_res.model_dump(),
            payload.context_language,
            payload.language,
            payload.push_to_docs_config.download_link,
            payload.push_to_docs_config.form_link,
        )

        create_document_in_lasuite_docs(
            content=content,
            title=payload.push_to_docs_config.title,
            email=payload.push_to_docs_config.user_email,
            sub=payload.user_sub,
        )

        if _should_auto_create_summary(payload):
            locale = get_locale(payload.context_language, payload.language)

            summarize_v2_task.apply_async(
                args=[
                    SummarizeTaskJob(
                        received_at=datetime.now(timezone.utc),
                        tenant_id=payload.tenant_id,
                        user_sub=payload.user_sub,
                        user_email=payload.user_email,
                        push_to_docs_config=PushToDocsBaseConfig(
                            user_email=payload.push_to_docs_config.user_email,
                            title=locale.summary_title_template.format(
                                title=payload.push_to_docs_config.title
                            ),
                        ),
                        content=content,
                    ).model_dump()
                ],
            )

    file_service.store_transcript(
        transcript=transcription_res,
        job_id=job_id,
    )

    success_payload = TranscribeWebhookSuccessPayload(
        job_id=job_id,
        transcription_data_url=file_service.get_transcript_signed_url(job_id),
    )
    call_webhook_v2_task.apply_async(
        args=[success_payload.model_dump(), payload.tenant_id]
    )
    metadata_manager.capture(job_id, settings.posthog_transcript_success)

    return success_payload.model_dump()


@signals.task_prerun.connect(sender=process_audio_transcribe_v2_task)
def task_started_transcript(task_id=None, task=None, args=None, **kwargs):
    """Signal handler called before task execution begins."""
    if args:
        metadata_manager.create(task_id, TranscribeTaskJob.model_validate(args[0]))


@signals.task_retry.connect(sender=process_audio_transcribe_v2_task)
def task_retry_handler_transcript(request=None, reason=None, einfo=None, **kwargs):
    """Signal handler called when task execution retries."""
    metadata_manager.retry(request.id)


@signals.task_failure.connect(sender=process_audio_transcribe_v2_task)
def handle_transcribe_v2_failed(  # noqa: PLR0917
    sender,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **kw,
):
    """Handle the failure of transcribe_v2_task.

    Tracks the failure event in analytics and sends a failure webhook to the client.
    """
    logger.error(
        "Transcribe task %s failed, no more retries left, sending failure webhook.",
        task_id,
    )
    metadata_manager.capture(
        task_id,
        settings.posthog_transcript_failure,
        {"exception_type": type(exception).__name__},
    )
    call_webhook_v2_task.apply_async(
        args=[
            TranscribeWebhookFailurePayload(
                job_id=task_id,
                error_code="unknown_error",
            ).model_dump(),
            args[0]["tenant_id"],
        ]
    )


@celery.task(
    bind=True,
    autoretry_for=[LLMException, Exception],
    max_retries=settings.celery_max_retries,
    queue=settings.summarize_queue_v2,
)
def summarize_v2_task(
    self,
    payload: dict,
):
    """Generate a summary from the provided content.

    This Celery task performs the following operations:
    1. Run summary internals
    2. Sends the final summary via webhook.
    """
    payload = SummarizeTaskJob.model_validate(payload)
    summary = summarize_transcription_internals(
        distinct_id=payload.user_sub,
        transcript=payload.content,
        session_id=self.request.id,
    )
    job_id = self.request.id
    file_service.store_summary(summary=summary, job_id=job_id)

    if _should_push_to_docs(payload):
        if payload.push_to_docs_config is None:
            raise ValueError("Push to docs config is missing")

        create_document_in_lasuite_docs(
            content=summary,
            title=payload.push_to_docs_config.title,
            email=payload.push_to_docs_config.user_email,
            sub=payload.user_sub,
        )

    success_payload = SummarizeWebhookSuccessPayload(
        job_id=job_id,
        summary_data_url=file_service.get_summary_signed_url(job_id),
    )
    call_webhook_v2_task.apply_async(
        args=[success_payload.model_dump(), payload.tenant_id]
    )
    metadata_manager.capture(job_id, settings.posthog_summary_success)

    return success_payload.model_dump()


@signals.task_prerun.connect(sender=summarize_v2_task)
def task_started_summary(task_id=None, task=None, args=None, **kwargs):
    """Signal handler called before task execution begins."""
    if args:
        metadata_manager.create(task_id, SummarizeTaskJob.model_validate(args[0]))


@signals.task_retry.connect(sender=summarize_v2_task)
def task_retry_handler_summary(request=None, reason=None, einfo=None, **kwargs):
    """Signal handler called when task execution retries."""
    metadata_manager.retry(request.id)


@signals.task_failure.connect(sender=summarize_v2_task)
def handle_summarize_v2_failed(  # noqa: PLR0917
    sender,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **kw,
):
    """Handle the failure of summarize_v2_task.

    Tracks the failure event in analytics and sends a failure webhook to the client.
    """
    logger.warn(
        "Summary task %s failed, no more retries left, sending failure webhook.",
        task_id,
    )
    metadata_manager.capture(
        task_id,
        settings.posthog_summary_failure,
        {"exception_type": type(exception).__name__},
    )
    call_webhook_v2_task.apply_async(
        args=[
            SummarizeWebhookFailurePayload(
                job_id=task_id,
                error_code="unknown_error",
            ).model_dump(),
            args[0]["tenant_id"],
        ]
    )
