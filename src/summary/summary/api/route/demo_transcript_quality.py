"""Demo-only route: run a dropped transcript through the correction stages.

Why this route exists
---------------------
The two transcript-quality features (speaker resolution from name cues, and
acronym correction) run inside `process_audio_transcribe_v2_task`, which starts
from `cloud_storage_url` -- an audio file. There is no way to show what the two
stages do to a transcript without transcribing audio first, which is slow, and
which is not the thing being demonstrated.

So this route takes the transcript directly and skips transcription. That is
the whole bypass, and it lives here, in a route that is clearly demo-scoped.
Nothing in the audio ingestion path is changed by it.

What it does NOT do
-------------------
It does not reimplement the pipeline. It calls `demo/run_demo.run_pipeline`,
which calls the three functions `celery_worker` calls, in `celery_worker`'s
order:

    resolve_speaker_identities_and_apply_to()   speakers
    _correct_acronyms_in()  ->  correct_acronyms()
    format_transcript()                         markdown

and it runs that twice: once with both feature flags closed (the "before"
transcript, which is today's Dictaphone behaviour) and once with both open.

Honesty
-------
The numbers this route returns are measured on the run that just happened, not
asserted. `stats` is the delta of `run_demo.STATS` across the run, so a caller
can see whether the model was actually called or the disk cache answered.
`warnings` collects the WARNING lines the stages emit, because
`acronym_correction._decide` catches its own exceptions and returns
`(None, 0.0)`: with the model unreachable the stage fails **open**, produces
fewer corrections, and otherwise looks exactly like a clean run.
`flagged_spans` is what stage 1 flagged, which is the only signal separating
"the glossary could not have helped here" (never flagged) from "the model
looked and declined".
"""

import json
import logging
import os
import sys
import threading
import uuid

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router_demo = APIRouter()

#: `src/summary/demo` has no `__init__.py`, so it is not importable as a
#: package. The scaffolding lives there rather than in `summary/` on purpose:
#: it is demo code and it does not ship inside the service's own package.
DEMO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "demo",
)

#: `run_demo`'s slots (`TRACE_SLOT`, `CORRECTIONS_SLOT`, `STATS`) are module
#: globals, so two overlapping runs would scramble each other's evidence.
LOCK = threading.Lock()

#: Reports from `/run`, so `/publish` pushes exactly the markdown the page was
#: showing rather than running the pipeline a second time. Bounded: this is a
#: demo, not storage.
RUNS: dict[str, dict] = {}
MAX_RUNS = 8

#: `run_demo.install_scaffolding` rebinds names on the `celery_worker` module.
#: Installing it twice would wrap the wrappers, so it happens once.
_SCAFFOLDING = {"installed": False}

#: What `acronym_correction._detect_suspect_spans` flagged on the last run.
SPANS_SLOT: dict = {"spans": []}


class DemoInputError(ValueError):
    """An uploaded file cannot be used. The message is shown in the page."""


def _demo_modules():
    """Import the demo scaffolding, adding `demo/` to the path first.

    Imported inside the handler rather than at module import time: a broken
    demo import must not stop the service's real routes from registering.
    """
    if DEMO_DIR not in sys.path:
        sys.path.insert(0, DEMO_DIR)
    import glossary_parser  # noqa: PLC0415
    import run_demo  # noqa: PLC0415

    return run_demo, glossary_parser


def _install_once(run_demo, celery_worker, acronym_correction, model):
    """Install the caching/observing scaffolding and the stage-1 capture."""
    if _SCAFFOLDING["installed"]:
        return
    run_demo.install_scaffolding(celery_worker, model)

    real_detect = acronym_correction._detect_suspect_spans

    def detect_and_capture(segments, llm_service):
        spans = real_detect(segments, llm_service)
        SPANS_SLOT["spans"] = list(spans)
        return spans

    acronym_correction._detect_suspect_spans = detect_and_capture
    _SCAFFOLDING["installed"] = True


class WarningCapture(logging.Handler):
    """Keeps the WARNING lines a degraded run leaves behind."""

    def __init__(self):
        """Start with an empty buffer, at WARNING."""
        super().__init__(level=logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record):
        """Store the formatted message, never the credentials in it."""
        try:
            self.lines.append(record.getMessage())
        except Exception:  # noqa: S110  a broken log line must not stop the run
            pass


def _read_transcript(payload: bytes, filename: str):
    """Parse an uploaded WhisperX JSON into the model the pipeline expects.

    Raises:
        DemoInputError: the bytes are not JSON, or not a WhisperX response.
    """
    from pydantic import ValidationError  # noqa: PLC0415

    from summary.core.shared_models import WhisperXResponse  # noqa: PLC0415

    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DemoInputError(
            "« %s » n'est pas un JSON lisible : %s" % (filename, exc)
        ) from exc

    try:
        WhisperXResponse.model_validate(raw)
    except ValidationError as exc:
        raise DemoInputError(
            "« %s » n'a pas la forme d'une réponse WhisperX (segments avec"
            " words). Détail : %s" % (filename, str(exc)[:300])
        ) from exc
    return raw


def _read_calendar(payload: bytes, filename: str) -> list[dict]:
    """Parse an uploaded `.ics` into `[{name, email}]`.

    Raises:
        DemoInputError: the bytes are not a readable iCalendar stream.
    """
    from summary.core.speaker_cues.attendees import (  # noqa: PLC0415
        parse_attendees_text,
    )

    try:
        return parse_attendees_text(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DemoInputError(
            "« %s » n'est pas un fichier iCalendar lisible : %s" % (filename, exc)
        ) from exc


def _speaker_rows(trace, transcription) -> list[dict]:
    """The best cue behind each name that survived into the transcript.

    Mirrors `run_demo.name_evidence`, but returns fields rather than a
    formatted French sentence, so the page can lay them out itself.
    """
    if trace is None:
        return []

    speakers = {
        segment.get("speaker")
        for segment in transcription.model_dump().get("segments") or []
    }

    best: dict[str, object] = {}
    for item in trace.evidence:
        held = best.get(item.name)
        if held is None or item.score > held.score:
            best[item.name] = item

    rows = []
    for name in sorted(best):
        if name not in speakers:
            continue
        item = best[name]
        rows.append(
            {
                "name": name,
                "label": item.speaker,
                "score": round(item.score, 2),
                "kind": item.cue.kind,
                "quote": item.cue.quote or item.cue.span,
                "segment_index": item.cue.segment_index,
            }
        )
    return rows


def _correction_rows(corrections, glossary, transcription) -> list[dict]:
    """One row per correction, with the sentence it sits in."""
    sentences = {
        index: (segment.get("text") or "").strip()
        for index, segment in enumerate(
            transcription.model_dump().get("segments") or []
        )
    }
    return [
        {
            "wrong": correction["wrong"],
            "correct": correction["correct"],
            "confidence": round(correction["confidence"], 2),
            "applied": correction["applied"],
            "from_user_glossary": correction["correct"] in glossary,
            "segment_index": correction["segment_index"],
            "word_index": correction["word_index"],
            "sentence": sentences.get(correction["segment_index"], ""),
        }
        for correction in corrections
    ]


def _explain(exc: Exception) -> str:
    """A message for the page. Never the API key, never a stack trace."""
    name = type(exc).__name__
    text = str(exc)
    if "CacheMiss" in name:
        return "Réponse absente du cache et passage hors ligne : %s" % text
    return (
        "L'appel au modèle a échoué (%s). Albert est peut-être injoignable."
        " Détail : %s" % (name, text[:300])
    )


@router_demo.post("/demo/transcript-quality/run")
async def run_transcript_quality_demo(
    transcript: UploadFile = File(...),  # noqa: B008
    glossary: UploadFile | None = File(None),  # noqa: B008
    calendar: UploadFile | None = File(None),  # noqa: B008
):
    """Run one dropped transcript through the pipeline, twice.

    Returns the "before" markdown (both flags closed), the "after" markdown
    (both open, with the dropped glossary and the attendees from the `.ics`),
    and the evidence behind every decision either stage made.
    """
    try:
        run_demo, glossary_parser = _demo_modules()
    except Exception as exc:
        logger.exception("Demo scaffolding failed to import")
        return JSONResponse(
            {"error": "Le harnais de démo n'a pas pu être chargé : %s" % exc},
            status_code=500,
        )

    from summary.core import acronym_correction, celery_worker  # noqa: PLC0415
    from summary.core.config import get_settings  # noqa: PLC0415
    from summary.core.shared_models import WhisperXResponse  # noqa: PLC0415

    settings = get_settings()

    # ---- the three uploads -------------------------------------------------
    try:
        raw = _read_transcript(await transcript.read(), transcript.filename or "?")
    except DemoInputError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    entries: dict[str, str] = {}
    skipped: list[str] = []
    if glossary is not None and glossary.filename:
        try:
            entries, skipped = glossary_parser.parse_upload(
                glossary.filename, await glossary.read()
            )
        except glossary_parser.GlossaryError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    attendees: list[dict] = []
    if calendar is not None and calendar.filename:
        try:
            attendees = _read_calendar(await calendar.read(), calendar.filename)
        except DemoInputError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    # ---- the two runs ------------------------------------------------------
    warnings = WarningCapture()
    logger_names = (
        "summary.core.acronym_correction",
        "summary.core.llm_service",
        "summary.core.speaker_dispatch",
        "summary.core.celery_worker",
    )
    for name in logger_names:
        logging.getLogger(name).addHandler(warnings)

    try:
        with LOCK:
            _install_once(
                run_demo, celery_worker, acronym_correction, settings.llm_model
            )
            hits_before = run_demo.STATS["hits"]
            misses_before = run_demo.STATS["misses"]
            SPANS_SLOT["spans"] = []

            before = run_demo.run_pipeline(
                celery_worker,
                WhisperXResponse.model_validate(raw),
                attendees,
                run_demo.BEFORE,
                "demo-upload-before",
                "fr",
            )
            after = run_demo.run_pipeline(
                celery_worker,
                WhisperXResponse.model_validate(raw),
                attendees,
                run_demo.AFTER,
                "demo-upload-after",
                "fr",
                user_glossary=entries or None,
            )
            flagged = list(SPANS_SLOT["spans"])
            stats = {
                "hits": run_demo.STATS["hits"] - hits_before,
                "misses": run_demo.STATS["misses"] - misses_before,
            }
    except Exception as exc:
        logger.exception("Demo pipeline run failed")
        return JSONResponse({"error": _explain(exc)}, status_code=502)
    finally:
        for name in logger_names:
            logging.getLogger(name).removeHandler(warnings)

    segments = raw.get("segments") or []
    report = {
        "run_id": uuid.uuid4().hex,
        "transcript": {
            "filename": transcript.filename,
            "segments": len(segments),
            "words": sum(len(segment.get("words") or []) for segment in segments),
            "language": raw.get("language"),
        },
        "glossary": {
            "filename": glossary.filename if glossary else None,
            "entries": [{"acronym": k, "expansion": v} for k, v in entries.items()],
            "count": len(entries),
            "skipped": skipped,
        },
        "attendees": {
            "filename": calendar.filename if calendar else None,
            "list": attendees,
            "count": len(attendees),
        },
        "before": {"markdown": before.markdown},
        "after": {
            "markdown": after.markdown,
            # The corrected transcript itself, not only its rendered markdown.
            # A caller that wants to *keep* the result -- Dictaphone's import
            # route stores it as a real recording's transcript -- needs the
            # structured WhisperX response (segments, words, speakers), because
            # markdown cannot be turned back into one. Additive: the `/demo`
            # page ignores this field and renders exactly what it rendered
            # before.
            "transcript": after.transcription.model_dump(),
        },
        "corrections": _correction_rows(
            after.corrections, entries, after.transcription
        ),
        "speakers": _speaker_rows(after.trace, after.transcription),
        "flagged_spans": flagged,
        "stats": stats,
        "warnings": warnings.lines,
        "min_confidence": settings.acronym_correction_min_confidence,
        "model": settings.llm_model,
    }

    RUNS[report["run_id"]] = report
    for stale in list(RUNS)[:-MAX_RUNS]:
        RUNS.pop(stale, None)
    return report


class PublishRequest(BaseModel):
    """Which markdown of which stored run to push to Docs."""

    run_id: str
    which: str = "after"
    title: str = "Démo — transcript corrigé"


@router_demo.post("/demo/transcript-quality/publish")
def publish_transcript_quality_demo(request: PublishRequest):
    """Create one document in Docs from a run this route already produced.

    Behind its own call on purpose: publishing on every run would litter Docs
    with a document per test. The markdown comes from `RUNS`, not from the
    caller, so only something this route actually produced can be published.
    """
    report = RUNS.get(request.run_id)
    if report is None:
        return JSONResponse(
            {"error": "Aucun résultat à publier : relancez la correction."},
            status_code=400,
        )
    if request.which not in ("before", "after"):
        return JSONResponse(
            {"error": "« which » doit valoir « before » ou « after »."},
            status_code=400,
        )

    run_demo, _ = _demo_modules()
    from summary.core import celery_worker  # noqa: PLC0415

    capture = run_demo.DocIdCapture()
    docs_logger = logging.getLogger("summary.core.docs_service")
    # The id is only ever logged, at INFO, and `create_document_in_lasuite_docs`
    # returns None. Under uvicorn the effective level is WARNING, so the record
    # is dropped before any handler sees it: the document is created and the id
    # is lost. Measured, not guessed -- Docs answered 201 while this route
    # reported "no id". Lowered for the duration of the call, then restored.
    previous_level = docs_logger.level
    docs_logger.setLevel(logging.INFO)
    docs_logger.addHandler(capture)
    try:
        celery_worker.create_document_in_lasuite_docs(
            content=report[request.which]["markdown"],
            title=request.title,
            email=run_demo.DOCS_EMAIL,
            sub=run_demo.DOCS_SUB,
        )
    except Exception as exc:
        logger.exception("Demo publish to Docs failed")
        return JSONResponse(
            {"error": "La publication dans Docs a échoué : %s" % str(exc)[:300]},
            status_code=502,
        )
    finally:
        docs_logger.removeHandler(capture)
        docs_logger.setLevel(previous_level)

    if not capture.ids:
        return JSONResponse(
            {"error": "Docs n'a pas renvoyé d'identifiant de document."},
            status_code=502,
        )
    document_id = capture.ids[-1]
    return {
        "id": document_id,
        "url": run_demo.DOCS_BROWSER_BASE % document_id,
    }
