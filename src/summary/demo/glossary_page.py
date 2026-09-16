r"""Drop a glossary, see the transcript corrected with it.

A one-page demo of `correct_acronyms(user_glossary=...)`, which landed in
`6429eb44` and has been inert ever since: `celery_worker._correct_acronyms_in`
never passes a mapping, so `build_index`, the `USER_GLOSSARY_BONUS` and the
`[glossaire de l'organisation]` prompt marker have never fired outside unit
tests. This page is the first thing that exercises them.

What it does NOT do
-------------------
It does not reimplement the pipeline and it does not copy `run_demo.py`. It
imports it and calls `run_demo.run_pipeline`, which calls the three functions
`celery_worker` calls, in `celery_worker`'s order:

    resolve_speaker_identities_and_apply_to()   speakers
    _correct_acronyms_in()  ->  correct_acronyms()
    format_transcript()                         markdown

Two seams are added on top of `run_demo.install_scaffolding`, both of which
observe rather than change:

  * `celery_worker.correct_acronyms` is wrapped once more, to inject
    `user_glossary=`. `_correct_acronyms_in` resolves that name as a module
    global at call time, so the injection lands at the exact call boundary the
    worker would use if it ever passed a glossary. Nothing inside
    `acronym_correction.py` is touched.
  * `acronym_correction._detect_suspect_spans` is wrapped to record what stage
    1 flagged. The README used to say "la démo n'instrumente pas l'étape 1";
    it is instrumented here, because it is the only signal that separates "the
    glossary could not possibly help" (the passage was never flagged, so no
    shortlist exists) from "the model refused" (it was flagged and decided
    against). Without it, a glossary that changes nothing is unexplainable.

Honesty
-------
The transcript, the `.ics` and the four attendees are mocks written for this
demo. No real meeting has ever gone through this pipeline. The ground truth is
`errors.json`, written before the measurement.

Usage, on the host (not in a container, nothing is rebuilt):

    cd src/summary
    PYTHONHASHSEED=0 uv run --no-project --python 3.13 \\
        --with 'fastapi[standard]' --with python-multipart \\
        --with pydantic-settings --with celery --with redis --with minio \\
        --with openai --with posthog --with requests \\
        --with 'sentry-sdk[fastapi,celery]' --with langfuse \\
        python demo/glossary_page.py

then open http://localhost:8799.

`--no-project` is the load-bearing flag, and the dependencies are listed by
hand for the same reason. Without it `uv` tries to build the project in
`src/summary` first, and that build fails on a conflict that predates this
page: setuptools' flat-layout discovery sees both `demo/` and `summary/` as
top-level packages and refuses to guess ("Multiple top-level packages
discovered in a flat-layout"). `--with-editable .` hits the same wall. No
`PYTHONPATH` is needed: the two `sys.path.insert` calls below already put
`demo/` and `src/summary` on the path.

`PYTHONHASHSEED=0` is not decorative: `PhoneticIndex.by_key` holds acronyms in
a `set` and `candidates()` breaks ties by iteration order, so without a pinned
seed the shortlist -- and therefore the `acronym-decide` prompt -- changes
between processes and the disk cache misses.
"""

# A demo server is one linear story like `run_demo.py` is, and the long lines
# are French sentences shown in the page.
# ruff: noqa: T201, E501

from __future__ import annotations

import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))

PORT = 8799
HTML_PATH = os.path.join(HERE, "glossary_page.html")
SAMPLE_PATH = os.path.join(HERE, "sample-glossary.txt")

TITLE_GLOSSARY = "Réunion architecture — transcript corrigé avec glossaire"


def _load_env_file(path: str) -> None:
    """Fill missing environment variables from a compose env file.

    The settings object is built at import time and needs the whole service
    configuration, which lives in `env.d/development/summary` -- the same file
    `celery-summary-transcribe` is started with. Values already present in the
    environment win, so a caller can override any of them.

    `host.docker.internal` is rewritten to `localhost`: the file is written for
    a container, and this process runs on the host.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            value = value.replace("host.docker.internal", "localhost")
            os.environ.setdefault(key, value)


# Before any `summary` import: `config.py` builds its settings at import time.
_load_env_file(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
        "env.d",
        "development",
        "summary",
    )
)
os.environ.setdefault("IS_ACRONYM_CORRECTION_ENABLED", "true")
os.environ.setdefault("IS_RESOLVE_SPEAKER_CUES_ENABLED", "true")
os.environ.setdefault("RESOLVE_SPEAKER_CUES_DETECTOR", "llm")

import logging  # noqa: E402

import run_demo  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from glossary_parser import GlossaryError, parse_upload  # noqa: E402
from pydantic import BaseModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

#: The glossary the injected `user_glossary=` argument reads, and the spans
#: stage 1 flagged. Module globals like `run_demo`'s own slots, and guarded by
#: the same lock: two overlapping requests would scramble them.
GLOSSARY_SLOT: dict = {"glossary": None}
SPANS_SLOT: dict = {"spans": []}
LOCK = threading.Lock()

#: The last run of each kind, kept so "Publier dans Docs" can push exactly the
#: markdown the page showed rather than running the pipeline again.
LAST: dict = {"with": None, "without": None}


def install_seams(celery_worker, acronym_correction) -> None:
    """Add the glossary injection and the stage-1 capture, once.

    Both wrap what is already installed and delegate to it. Neither changes a
    decision: one supplies an argument `correct_acronyms` already accepts, the
    other copies out a list on its way past.
    """
    captured_correct = celery_worker.correct_acronyms

    def correct_with_glossary(**kwargs):
        kwargs.setdefault("user_glossary", GLOSSARY_SLOT["glossary"])
        return captured_correct(**kwargs)

    celery_worker.correct_acronyms = correct_with_glossary

    real_detect = acronym_correction._detect_suspect_spans

    def detect_and_capture(segments, llm_service):
        spans = real_detect(segments, llm_service)
        SPANS_SLOT["spans"] = list(spans)
        return spans

    acronym_correction._detect_suspect_spans = detect_and_capture


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


class Pipeline:
    """Holds the imported modules and the mock input, loaded once."""

    def __init__(self):
        """Import `summary`, install the scaffolding, read the mocks."""
        from summary.core import acronym_correction, celery_worker  # noqa: PLC0415
        from summary.core.config import get_settings  # noqa: PLC0415
        from summary.core.shared_models import WhisperXResponse  # noqa: PLC0415
        from summary.core.speaker_cues.attendees import (  # noqa: PLC0415
            parse_attendees,
        )

        self.celery_worker = celery_worker
        self.whisperx_response = WhisperXResponse
        self.settings = get_settings()
        self.min_confidence = self.settings.acronym_correction_min_confidence

        run_demo.install_scaffolding(celery_worker, self.settings.llm_model)
        install_seams(celery_worker, acronym_correction)

        flow = os.path.join(run_demo.INPUT_DIR, "flow-1-technique")
        self.raw = run_demo.load(os.path.join(flow, "03-transcript-whisperx.json"))
        self.errors = run_demo.load(os.path.join(flow, "errors.json"))["errors"]
        self.attendees = parse_attendees(os.path.join(flow, "02-attendees.ics"))
        self.before_markdown: str | None = None

    def _run(self, flags, job_id):
        return run_demo.run_pipeline(
            self.celery_worker,
            self.whisperx_response.model_validate(self.raw),
            self.attendees,
            flags,
            job_id,
            "fr",
        )

    def before(self) -> str:
        """The AVANT transcript: both feature flags closed, no model call."""
        if self.before_markdown is None:
            with LOCK:
                GLOSSARY_SLOT["glossary"] = None
                self.before_markdown = self._run(run_demo.BEFORE, "demo-before").markdown
        return self.before_markdown

    def run(self, glossary: dict[str, str] | None) -> dict:
        """Run the full pipeline once and score it against `errors.json`.

        The warning capture is not decoration. `_decide` catches its own
        exceptions and returns `(None, 0.0)`: with Albert unreachable, the
        stage fails **open**, the run comes back with fewer corrections and
        nothing propagates. Measured, not assumed -- pointing `LLM_BASE_URL` at
        a dead port yields HTTP 200 and seven corrections instead of eight.
        A degraded passage that looks exactly like a clean one is the kind of
        silence this demo is supposed to refuse, so the warnings the stage logs
        are collected and shown in the page.
        """
        warnings = WarningCapture()
        names = (
            "summary.core.acronym_correction",
            "summary.core.llm_service",
            "summary.core.speaker_dispatch",
        )
        for name in names:
            logging.getLogger(name).addHandler(warnings)
        try:
            with LOCK:
                GLOSSARY_SLOT["glossary"] = glossary or None
                SPANS_SLOT["spans"] = []
                result = self._run(run_demo.AFTER, "demo-glossary")
                flagged = list(SPANS_SLOT["spans"])
                GLOSSARY_SLOT["glossary"] = None
        finally:
            for name in names:
                logging.getLogger(name).removeHandler(warnings)
        return self._report(result, glossary or {}, flagged, warnings.lines)

    def _report(
        self,
        result,
        glossary: dict[str, str],
        flagged: list[str],
        warnings: list[str],
    ) -> dict:
        """Turn one run into the numbers and rows the page renders."""
        scored = run_demo.score(result.corrections, self.errors)

        verdicts: dict[str, dict] = {}
        for error, correction in scored["caught"]:
            verdicts[error["id"]] = {"verdict": "corrigée", "correction": correction}
        for error, correction in scored["below_floor"]:
            verdicts[error["id"]] = {
                "verdict": "sous le seuil",
                "correction": correction,
            }
        for error in scored["missed"]:
            verdicts.setdefault(error["id"], {"verdict": "manquée", "correction": None})

        rows = []
        for error in scored["in_scope"]:
            found = verdicts.get(error["id"], {"verdict": "manquée", "correction": None})
            correction = found["correction"]
            rows.append(
                {
                    "id": error["id"],
                    "wrong": error["wrong"],
                    "correct": error["correct"],
                    "verdict": found["verdict"],
                    "confidence": correction["confidence"] if correction else None,
                    "from_user": bool(correction)
                    and correction["correct"] in glossary,
                    "in_shipped_glossary": error["correct"].upper()
                    in self._shipped_keys(),
                }
            )

        sentences = {
            index: (segment.get("text") or "").strip()
            for index, segment in enumerate(
                result.transcription.model_dump().get("segments") or []
            )
        }

        corrections = [
            {
                "wrong": correction["wrong"],
                "correct": correction["correct"],
                "confidence": correction["confidence"],
                "applied": correction["applied"],
                "from_user": correction["correct"] in glossary,
                "segment_index": correction["segment_index"],
                "sentence": sentences.get(correction["segment_index"], ""),
            }
            for correction in result.corrections
        ]

        false_positives = [
            {
                "wrong": correction["wrong"],
                "correct": correction["correct"],
                "confidence": correction["confidence"],
                "applied": correction["applied"],
                "segment_index": correction["segment_index"],
                "from_user": correction["correct"] in glossary,
            }
            for correction in scored["false_positives"]
        ]

        return {
            "counts": {
                "in_scope": len(scored["in_scope"]),
                "caught": len(scored["caught"]),
                "below_floor": len(scored["below_floor"]),
                "missed": len(scored["missed"]),
                "false_positives": len(scored["false_positives"]),
            },
            "rows": rows,
            "corrections": corrections,
            "false_positives": false_positives,
            "flagged_spans": flagged,
            "warnings": warnings,
            "markdown": result.markdown,
            "min_confidence": self.min_confidence,
        }

    def _shipped_keys(self) -> set[str]:
        from summary.core.acronym_correction import (  # noqa: PLC0415
            get_acronym_glossary,
        )

        return {key.upper() for key in get_acronym_glossary()}

    def publish(self, markdown: str, title: str) -> str | None:
        """Create one document in Docs and return its id.

        `create_document_in_lasuite_docs` returns None, so the id is read back
        from the log line it emits -- the same trick `run_demo.py` uses, and
        the reason `docs_service.py` stays untouched.
        """
        capture = run_demo.DocIdCapture()
        logger = logging.getLogger("summary.core.docs_service")
        logger.addHandler(capture)
        try:
            self.celery_worker.create_document_in_lasuite_docs(
                content=markdown,
                title=title,
                email=run_demo.DOCS_EMAIL,
                sub=run_demo.DOCS_SUB,
            )
        finally:
            logger.removeHandler(capture)
        return capture.ids[-1] if capture.ids else None


PIPELINE: Pipeline | None = None


def pipeline() -> Pipeline:
    """Build the pipeline holder on first use."""
    global PIPELINE  # noqa: PLW0603
    if PIPELINE is None:
        PIPELINE = Pipeline()
    return PIPELINE


app = FastAPI(title="Démo glossaire d'acronymes")


@app.get("/")
def index():
    """Serve the page."""
    return FileResponse(HTML_PATH, media_type="text/html; charset=utf-8")


@app.get("/api/sample")
def sample():
    """Hand back the bundled sample glossary, as text."""
    with open(SAMPLE_PATH, encoding="utf-8") as handle:
        return {"filename": os.path.basename(SAMPLE_PATH), "text": handle.read()}


@app.get("/api/before")
def before():
    """The AVANT transcript, so the page can show it next to the others."""
    try:
        return {"markdown": pipeline().before()}
    except Exception as exc:
        return JSONResponse({"error": _explain(exc)}, status_code=500)


@app.post("/api/run")
async def run(file: UploadFile = File(None)):  # noqa: B008
    """Run the pipeline, with the uploaded glossary or without one.

    No file means the control run: exactly the same code path with
    `user_glossary=None`, which is what the service does today.
    """
    glossary: dict[str, str] = {}
    skipped: list[str] = []
    if file is not None and file.filename:
        try:
            glossary, skipped = parse_upload(file.filename, await file.read())
        except GlossaryError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not glossary:
            return JSONResponse(
                {
                    "error": "Aucune entrée lisible dans ce fichier.",
                    "skipped": skipped,
                },
                status_code=400,
            )

    try:
        report = pipeline().run(glossary)
    except Exception as exc:
        return JSONResponse({"error": _explain(exc)}, status_code=502)

    report["glossary"] = [{"acronym": k, "expansion": v} for k, v in glossary.items()]
    report["skipped"] = skipped
    report["with_glossary"] = bool(glossary)
    LAST["with" if glossary else "without"] = report
    return report


class PublishRequest(BaseModel):
    """Which of the two stored runs to push to Docs."""

    which: str = "with"


@app.post("/api/publish")
def publish(request: PublishRequest):
    """Create a third document in Docs from the run the page is showing.

    Deliberately behind its own button: publishing on every upload would
    litter Docs with a document per test run.
    """
    report = LAST.get(request.which)
    if report is None:
        return JSONResponse(
            {"error": "Rien à publier : lancez d'abord une correction."},
            status_code=400,
        )
    suffix = "" if request.which == "with" else " (sans glossaire)"
    try:
        document_id = pipeline().publish(report["markdown"], TITLE_GLOSSARY + suffix)
    except Exception as exc:
        return JSONResponse({"error": _explain(exc)}, status_code=502)
    if not document_id:
        return JSONResponse(
            {"error": "Docs n'a pas renvoyé d'identifiant de document."},
            status_code=502,
        )
    return {"id": document_id, "url": run_demo.DOCS_BROWSER_BASE % document_id}


def _explain(exc: Exception) -> str:
    """A message for the page. Never the API key, never a stack trace."""
    name = type(exc).__name__
    text = str(exc)
    if "CacheMiss" in name:
        return (
            "Réponse absente du cache et passage hors ligne : %s" % text
        )
    return (
        "L'appel au modèle a échoué (%s). Albert est peut-être injoignable ;"
        " le cache disque couvre le passage sans glossaire, pas les nouvelles"
        " décisions. Détail : %s" % (name, text[:300])
    )


if __name__ == "__main__":
    print("Démo glossaire : http://localhost:%d" % PORT)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
