r"""Before / after demo for the two transcript-quality features.

Runs one mocked WhisperX transcript through the pipeline twice and pushes both
results to La Suite Docs as two documents.

    BEFORE   is_resolve_speaker_cues_enabled = False
             is_acronym_correction_enabled   = False
    AFTER    both True, resolve_speaker_cues_detector = "llm"

BEFORE is not hand-written. It is what `celery_worker` does when both gates are
closed: neither stage is called and `format_transcript` runs on the raw
WhisperX JSON. That is today's behaviour for Dictaphone, which never has VAD
metadata -- so `SPEAKER_00` in the "before" document is genuine, not staged.

What this script does NOT do
---------------------------
It does not reimplement either feature. It calls the three functions
`celery_worker` calls, in the order `celery_worker` calls them:

    resolve_speaker_identities_and_apply_to()   speakers
    _correct_acronyms_in()                      acronyms
    format_transcript()                         markdown

and then the existing `create_document_in_lasuite_docs()`. The only
monkeypatching is demo scaffolding that observes without changing behaviour:

  * `LLMService` -> `CachingLLMService`, so the demo replays offline;
  * `LLMObservability` -> a lazy holder, so a cached run builds no client;
  * `_cue_options` -> the real one plus a `trace`, so the evidence behind
    every name can be printed. `resolve_speaker_identities_from_cues` already
    accepts that `trace` argument; nothing about the resolution changes.

Usage (inside `celery-summary-transcribe`, where `/app` is `src/summary`):

    docker exec -e IS_ACRONYM_CORRECTION_ENABLED=true \\
                -e IS_RESOLVE_SPEAKER_CUES_ENABLED=true \\
                -e RESOLVE_SPEAKER_CUES_DETECTOR=llm \\
                -e PYTHONHASHSEED=0 \\
                celery-summary-transcribe python /app/demo/run_demo.py

    --offline   fail on any cache miss instead of calling Albert
    --no-docs   run the pipeline and the report, push nothing
"""

# A demo script is one long linear story on purpose; splitting it into
# helpers would hide the order of operations the demo is about.
# The long lines are markdown table rows inside the README template:
# wrapping them would break the table.
# ruff: noqa: PLR0912, PLR0913, PLR0915, PLR0917, T201, E501

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
# `python /app/demo/run_demo.py` puts /app/demo on the path, not /app, so the
# `summary` package next to it has to be added explicitly.
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))

from llm_cache import CachingLLMService, LazyObservability  # noqa: E402

DEMO_ROOT = HERE
INPUT_DIR = os.path.join(DEMO_ROOT, "input")
OUT_DIR = os.path.join(DEMO_ROOT, "out")
CACHE_PATH = os.path.join(DEMO_ROOT, "cache", "llm-cache.json")

DOCS_BROWSER_BASE = "http://localhost:8700/docs/%s/"

#: The Docs account the two documents are created for. Local Keycloak realm
#: `impress`, which is also the account used to open them in the browser.
DOCS_EMAIL = "impress@impress.world"
DOCS_SUB = "impress@impress.world"

TITLE_BEFORE = "Réunion architecture — transcript brut"
TITLE_AFTER = "Réunion architecture — transcript corrigé"

#: Ground-truth error types the acronym feature is not meant to catch.
OUT_OF_SCOPE_TYPES = ("number", "proper_noun")


@dataclass
class Flags:
    """The two settings the demo switches, named as in `config.py`."""

    is_resolve_speaker_cues_enabled: bool
    is_acronym_correction_enabled: bool


BEFORE = Flags(
    is_resolve_speaker_cues_enabled=False, is_acronym_correction_enabled=False
)
AFTER = Flags(is_resolve_speaker_cues_enabled=True, is_acronym_correction_enabled=True)


@dataclass
class RunResult:
    """Everything one pipeline run produced."""

    markdown: str
    transcription: object
    trace: object = None
    dispatch_log: list = field(default_factory=list)
    corrections: list = field(default_factory=list)


# --------------------------------------------------------------------------
# scaffolding: cache, trace, log capture
# --------------------------------------------------------------------------
STATS: dict = {"hits": 0, "misses": 0}
TRACE_SLOT: dict = {"trace": None}
CORRECTIONS_SLOT: dict = {"corrections": []}
#: Flipped for the offline replay pass. Read at call time, so the scaffolding
#: is installed exactly once and never wraps itself.
OFFLINE: dict = {"value": False}


def install_scaffolding(celery_worker, model: str):
    """Wrap the LLM service, the observability holder, the cue options.

    Each wrapper delegates to the real thing. None of them changes what the
    features decide -- they only make the run cacheable and observable.
    """

    def make_service(llm_observability):
        return CachingLLMService(
            cache_path=CACHE_PATH,
            observability_kwargs=llm_observability.kwargs,
            model=model,
            offline=OFFLINE["value"],
            stats=STATS,
        )

    celery_worker.LLMObservability = LazyObservability
    celery_worker.LLMService = make_service

    real_cue_options = celery_worker._cue_options

    def cue_options_with_trace(task_id, user_sub):
        options = real_cue_options(task_id, user_sub)
        options["trace"] = TRACE_SLOT["trace"]
        return options

    celery_worker._cue_options = cue_options_with_trace

    # `_correct_acronyms_in` keeps the corrections list to itself (it only
    # logs a count), so capture it where it is produced. `correct_acronyms`
    # runs untouched; this only copies its second return value out.
    real_correct = celery_worker.correct_acronyms

    def correct_and_capture(**kwargs):
        corrected, corrections = real_correct(**kwargs)
        CORRECTIONS_SLOT["corrections"] = corrections
        return corrected, corrections

    celery_worker.correct_acronyms = correct_and_capture


class LogCapture(logging.Handler):
    """Keeps the INFO lines that prove each stage actually ran."""

    def __init__(self):
        """Start with an empty buffer, at INFO."""
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        """Store the formatted message."""
        try:
            self.lines.append(record.getMessage())
        except Exception:  # noqa: S110  a broken log line must not stop the demo
            pass


class DocIdCapture(logging.Handler):
    """Reads the document id out of the docs_service success log.

    `create_document_in_lasuite_docs` returns None, so the id it logs is the
    only way to build a URL without touching `docs_service.py`.
    """

    def __init__(self):
        """Start with no ids collected, at INFO."""
        super().__init__(level=logging.INFO)
        self.ids: list[str] = []

    def emit(self, record):
        """Pick up `%s` argument 0 of the delivery-success line."""
        if str(record.msg).startswith("Delivery success") and record.args:
            self.ids.append(str(record.args[0]))


# --------------------------------------------------------------------------
# the pipeline, in `celery_worker` order
# --------------------------------------------------------------------------
def run_pipeline(celery_worker, whisperx_response, attendees, flags, job_id, language):
    """Mirror `process_audio_transcribe_v2_task`'s stages for one flag set.

    The gates are copied from `celery_worker.py` (the block that starts at
    "Assign speakers and rewrite transcription/diarization output"), with the
    two settings replaced by `flags`. `metadata` is None throughout: this run
    is the Dictaphone shape, where VAD metadata never exists.
    """
    from summary.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()

    transcription = whisperx_response
    trace = None
    corrections: list = []

    # Speakers. `metadata is not None` is False here, so the VAD branch of the
    # real condition can never open -- exactly the Dictaphone situation.
    can_resolve_from_vad = settings.is_resolve_speaker_identities_enabled and False
    can_resolve_from_cues = flags.is_resolve_speaker_cues_enabled and bool(attendees)
    if can_resolve_from_vad or can_resolve_from_cues:
        from summary.core.speaker_cues.resolve import ResolutionTrace  # noqa: PLC0415

        trace = ResolutionTrace()
        TRACE_SLOT["trace"] = trace
        transcription = celery_worker.resolve_speaker_identities_and_apply_to(
            transcription=transcription,
            recording_metadata=None,
            task_id=job_id,
            attendees=attendees,
            user_sub=DOCS_SUB,
        )

    # Acronyms.
    if flags.is_acronym_correction_enabled:
        CORRECTIONS_SLOT["corrections"] = []
        transcription = celery_worker._correct_acronyms_in(
            transcription=transcription,
            user_sub=DOCS_SUB,
            task_id=job_id,
        )
        corrections = CORRECTIONS_SLOT["corrections"]

    # Markdown.
    markdown = celery_worker.format_transcript(
        transcription.model_dump(),
        None,
        language,
        None,
        None,
    )
    return RunResult(
        markdown=markdown,
        transcription=transcription,
        trace=trace,
        corrections=corrections,
    )


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------
def _tidy_quote(quote: str, span: str) -> str:
    """Trim a cue quote down to something a slide can show.

    The LLM detector quotes the line it was given, which carries the batching
    prefix `[12] (SPEAKER_02)`. Cosmetic only -- nothing downstream reads it.
    """
    text = re.sub(r"^\s*\[\d+\]\s*\([^)]*\)\s*", "", quote or "").strip()
    text = text or (span or "").strip()
    cut = re.split(r"(?<=[.?!])\s", text, maxsplit=1)[0]
    if len(cut) > 70:
        cut = cut[:69].rstrip() + "…"
    return cut


def name_evidence(trace, transcription) -> list[str]:
    """One line per assigned name: where the resolver got it from."""
    if trace is None:
        return []

    speakers = {}
    for segment in transcription.model_dump().get("segments") or []:
        speakers.setdefault(segment.get("speaker"), True)

    best: dict[str, object] = {}
    for item in trace.evidence:
        held = best.get(item.name)
        if held is None or item.score > held.score:
            best[item.name] = item

    lines = []
    for name in sorted(best):
        if name not in speakers:
            continue
        item = best[name]
        lines.append(
            "%s ← %s, « %s », segment %d (confiance %.2f, label %s)"
            % (
                name,
                item.cue.kind,
                _tidy_quote(item.cue.quote, item.cue.span),
                item.cue.segment_index,
                item.score,
                item.speaker,
            )
        )
    return lines


def acronym_evidence(corrections) -> tuple[list[str], list[str]]:
    """Applied corrections and the ones the confidence floor blocked."""
    applied, blocked = [], []
    for correction in corrections:
        line = '%s ← "%s", confidence %.2f, %s' % (
            correction["correct"],
            correction["wrong"],
            correction["confidence"],
            "applied" if correction["applied"] else "NOT applied (below floor)",
        )
        line += " [segment %d, word %d]" % (
            correction["segment_index"],
            correction["word_index"],
        )
        (applied if correction["applied"] else blocked).append(line)
    return applied, blocked


# --------------------------------------------------------------------------
# score against ground truth
# --------------------------------------------------------------------------
def speaker_score(before_transcription, after_transcription, errors):
    """Check the recovered names against the `speaker` field of `errors.json`.

    `errors.json` records who really said each faulty segment. It is the only
    per-segment name ground truth that exists for this mock, and it was
    written for the acronym work -- not for this check, which is what makes it
    usable as an independent yardstick.
    """
    before_segments = before_transcription.model_dump().get("segments") or []
    after_segments = after_transcription.model_dump().get("segments") or []

    rows = []
    for error in errors:
        index = error["segment_index"]
        if index >= len(after_segments):
            continue
        label = before_segments[index].get("speaker")
        got = after_segments[index].get("speaker")
        expected = error["speaker"]
        rows.append((index, label, expected, got, got == expected))

    seen: dict[str, tuple] = {}
    for _index, label, expected, got, ok in rows:
        seen.setdefault(label, (label, expected, got, ok))
    return sorted(seen.values())


def _overlaps(correction, error) -> bool:
    if correction["segment_index"] != error["segment_index"]:
        return False
    start = correction["word_index"]
    end = start + correction["n_words"] - 1
    return any(start <= index <= end for index in error["word_indices"])


def score(corrections, errors):
    """Four buckets, matched on segment index plus word-range overlap.

    A correction with the right answer but below the confidence floor is NOT
    a catch -- it is its own bucket, because the transcript still shows the
    wrong word.
    """
    in_scope = [e for e in errors if e["type"] not in OUT_OF_SCOPE_TYPES]
    out_of_scope = [e for e in errors if e["type"] in OUT_OF_SCOPE_TYPES]

    caught, below_floor, missed, false_positives = [], [], [], []
    used = set()

    for error in in_scope:
        match = None
        for index, correction in enumerate(corrections):
            if index in used or not _overlaps(correction, error):
                continue
            match = (index, correction)
            break
        if match is None:
            missed.append(error)
            continue
        index, correction = match
        used.add(index)
        right = (
            correction["correct"].strip().lower() == error["correct"].strip().lower()
        )
        if right and correction["applied"]:
            caught.append((error, correction))
        elif right:
            below_floor.append((error, correction))
        else:
            missed.append(error)
            false_positives.append(correction)

    for index, correction in enumerate(corrections):
        if index in used:
            continue
        hit_out_of_scope = any(_overlaps(correction, e) for e in out_of_scope)
        if correction["applied"] or hit_out_of_scope:
            false_positives.append(correction)

    return {
        "in_scope": in_scope,
        "out_of_scope": out_of_scope,
        "caught": caught,
        "below_floor": below_floor,
        "missed": missed,
        "false_positives": false_positives,
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
HONESTY = """## Honnêteté — à lire avant tout le reste

- **La transcription et la liste des participants sont fabriquées.** Le fichier
  WhisperX, l'invitation `.ics` et les quatre noms sont des mocks écrits pour
  cette démonstration. Aucune vraie réunion n'est passée dans ce pipeline.
- Les erreurs mesurées sont **nos propres erreurs**, écrites dans `errors.json`
  avant la mesure. Ce sont des chiffres synthétiques, pas des résultats de
  terrain.
- Le rappel de 0,74 du résolveur de locuteurs est une **estimation ponctuelle** :
  `temperature=0` n'est pas reproductible sur Albert, les répétitions donnent le
  même score mais sur des items différents.
- Un nombre de tests unitaires n'est pas une mesure de justesse, et n'est pas
  présenté comme telle ici.
- Le modèle est celui que la pile a déjà configuré : `LLM_MODEL=openweight-large`
  chez Albert (`openai/gpt-oss-120b`). La démo n'en choisit pas un autre pour
  l'occasion.
- Les deux documents publiés viennent d'un passage **entièrement en cache**,
  sans un seul appel réseau. Les réponses en cache, elles, ont bien été
  produites par Albert lors d'un passage précédent sur cette même entrée.
"""


def misses_prose(result, min_confidence: float) -> str:
    """Explain the misses from what was measured, not from a fixed story.

    This paragraph used to be hard-coded: DINUM missed because the widened
    window matched COTRIM, and four free-software names absent from the
    glossary. Both statements became false once `WINDOW_MARGIN` dropped to 0,
    and a paragraph that contradicts the table above it is worse than no
    paragraph. Everything that can move between runs is now read back from the
    score buckets and from the glossary itself.
    """
    from summary.core.acronym_correction import (  # noqa: PLC0415
        get_acronym_glossary,
    )

    glossary = get_acronym_glossary()
    known = {key.upper() for key in glossary}

    def label(error) -> str:
        return "**`%s` → %s (%s)**" % (error["wrong"], error["correct"], error["id"])

    lines = []

    for error, correction in result["below_floor"]:
        text = (
            "- %s est **trouvé** à %.2f et remonté dans la liste d'audit, mais"
            " sous le seuil de %.2f : le document garde le mot de Whisper."
            % (label(error), correction["confidence"], min_confidence)
        )
        if error["id"] == "e1":
            text += (
                " C'est un changement de comportement : au passage précédent il"
                " était **manqué**, parce que la fenêtre élargie « Côté dix"
                " nomme » matchait COTRIM (0.70), réclamait les mots et bloquait"
                " « dix nomme » → DINUM. `WINDOW_MARGIN = 0` (commit 28729b37)"
                " supprime cette fenêtre que personne n'avait signalée, et le bon"
                " candidat atteint l'arbitrage — sans convaincre le modèle pour"
                " autant. Corrigé, il ne l'est toujours pas."
            )
        lines.append(text + "\n")

    absent = [e for e in result["missed"] if e["correct"].upper() not in known]
    present = [e for e in result["missed"] if e["correct"].upper() in known]

    if absent:
        lines.append(
            "- %s : absents du glossaire administratif de %d entrées. Rien à"
            " décider si le candidat n'existe pas.\n"
            % (", ".join(label(error) for error in absent), len(glossary))
        )
    if present:
        lines.append(
            "- %s : ces entrées **sont** pourtant dans le glossaire de %d"
            " entrées. Le blocage ne vient donc pas d'un candidat manquant :"
            " soit l'étape 1 n'a pas signalé le passage, soit le modèle a"
            " refusé à l'étape 2. La démo n'instrumente pas l'étape 1 et ne"
            " tranche pas entre les deux.\n"
            % (", ".join(label(error) for error in present), len(glossary))
        )

    count = len(result["false_positives"])
    if count:
        lines.append(
            "- **%d faux positif(s)** sur ce transcript : un mot a été remplacé"
            " par une réponse qui n'est pas celle de la vérité terrain.\n" % count
        )
    else:
        lines.append(
            "- **Zéro faux positif** sur ce transcript. C'est le sens du"
            " compromis : le seuil de %.2f est réglé pour la précision, pas"
            " pour le rappel.\n" % min_confidence
        )

    return "".join(lines)


def build_report(context) -> str:
    """The evidence tables, the score and the refusal case, as markdown."""
    out = []
    add = out.append

    add("## 1. Les deux documents\n")
    for label, title, doc_id in context["documents"]:
        if doc_id:
            add("- **%s** — %s\n  %s" % (label, title, DOCS_BROWSER_BASE % doc_id))
        else:
            add("- **%s** — %s (non publié)" % (label, title))
    add("")
    add(
        "Captures d'écran des deux documents ouverts dans Docs :"
        " `/Users/macair/dinum/screenshots/demo-docs-transcript-brut-v2.png`"
        " et `…-corrige-v2.png`.\n"
    )
    add(
        "Markdown correspondant, tel qu'il a été poussé :"
        " `demo/out/flow-1-before.md` et `demo/out/flow-1-after.md`.\n"
    )

    add("## 2. Preuve par nom (résolveur d'indices, détecteur `llm`)\n")
    add("Format : `Nom ← type d'indice, « citation », segment`\n")
    add("```")
    for line in context["names"] or ["(aucun nom attribué)"]:
        add(line)
    add("```")
    add("")
    unassigned = context["unassigned"]
    add(
        "Labels laissés en `SPEAKER_XX` dans le document corrigé : **%s**\n"
        % (", ".join(unassigned) if unassigned else "aucun")
    )

    rows = context["speakers"]
    correct = sum(1 for _, _, _, ok in rows if ok)
    add(
        "Contrôle contre la vérité terrain — le champ `speaker` de"
        " `errors.json` dit qui a réellement parlé sur chaque segment fautif."
        " **%d / %d** labels retrouvés correctement. Le contrôle ne couvre que"
        " %d des 4 labels : `errors.json` est une vérité terrain d'acronymes,"
        " pas de locuteurs, et un participant qui n'a commis aucune erreur"
        " d'acronyme n'y figure pas. Les 4 noms du document corrigé sont"
        " néanmoins les 4 bons.\n" % (correct, len(rows), len(rows))
    )
    add("| label WhisperX | attendu | obtenu | |")
    add("|---|---|---|---|")
    for label, expected, got, ok in rows:
        add("| %s | %s | %s | %s |" % (label, expected, got, "✅" if ok else "❌"))
    add("")

    add("## 3. Preuve par acronyme\n")
    add('Format : `ACRONYME ← "ce que Whisper a écrit", confiance, appliqué ou non`\n')
    add("### Appliqués (confiance ≥ %.2f)\n" % context["min_confidence"])
    add("```")
    for line in context["applied"] or ["(aucun)"]:
        add(line)
    add("```")
    add("")
    add("### Sous le seuil — signalés mais **NON appliqués**\n")
    add(
        "C'est l'histoire de la précision : la correction est trouvée, elle est"
        " remontée dans la liste d'audit, et le document n'est pas modifié.\n"
    )
    add("```")
    for line in context["blocked"] or ["(aucun)"]:
        add(line)
    add("```")
    add("")

    result = context["score"]
    add("## 4. Score contre `errors.json`\n")
    add(
        "Vérité terrain : **%d erreurs** au total, dont **%d erreurs d'acronyme**"
        " (le périmètre de la fonctionnalité) et **%d hors périmètre**.\n"
        % (
            len(result["in_scope"]) + len(result["out_of_scope"]),
            len(result["in_scope"]),
            len(result["out_of_scope"]),
        )
    )
    add("| Résultat | Nombre |")
    add("|---|---|")
    add(
        "| corrigées (appliquées, bonne réponse) | **%d / %d** |"
        % (len(result["caught"]), len(result["in_scope"]))
    )
    add(
        "| bonne réponse mais sous le seuil (non appliquée) | %d |"
        % len(result["below_floor"])
    )
    add("| manquées | %d |" % len(result["missed"]))
    add("| faux positifs | %d |" % len(result["false_positives"]))
    add("")

    add("### Détail, erreur par erreur\n")
    add("| id | Whisper a écrit | attendu | résultat |")
    add("|---|---|---|---|")
    verdicts = {}
    for error, correction in result["caught"]:
        verdicts[error["id"]] = "corrigée (%.2f)" % correction["confidence"]
    for error, correction in result["below_floor"]:
        verdicts[error["id"]] = (
            "trouvée mais sous le seuil (%.2f) — non appliquée"
            % correction["confidence"]
        )
    for error in result["missed"]:
        verdicts.setdefault(error["id"], "manquée")
    for error in result["in_scope"]:
        add(
            "| %s | %s | %s | %s |"
            % (
                error["id"],
                error["wrong"],
                error["correct"],
                verdicts.get(error["id"], "manquée"),
            )
        )
    add("")

    over = [
        (error, correction)
        for error, correction in result["caught"]
        if correction["wrong"].strip().lower() != error["wrong"].strip().lower()
    ]
    if over:
        add("### Effet de bord observé : la fenêtre mange des mots voisins\n")
        add(
            "La règle de « plus longue fenêtre d'abord » a remplacé ici plus"
            " de mots que nécessaire. Le mot juste arrive, mais la phrase perd"
            " un mot autour. À regarder avant toute mise en production — ce"
            " n'est pas une invention de contenu, c'est une phrase abîmée.\n"
        )
        add("| attendu | fenêtre réellement remplacée | phrase obtenue |")
        add("|---|---|---|")
        for error, correction in over:
            add(
                "| `%s` → %s | `%s` → %s | %s |"
                % (
                    error["wrong"],
                    error["correct"],
                    correction["wrong"],
                    correction["correct"],
                    context["sentences"].get(correction["segment_index"], ""),
                )
            )
        add("")

    add("### Ce qui a été manqué, et pourquoi\n")
    add(misses_prose(result, context["min_confidence"]))

    add("### Hors périmètre — la correction d'acronymes ne les vise pas\n")
    add("| id | Whisper a écrit | attendu | type |")
    add("|---|---|---|---|")
    for error in result["out_of_scope"]:
        add(
            "| %s | %s | %s | %s |"
            % (error["id"], error["wrong"], error["correct"], error["type"])
        )
    add("")
    add(
        "Ce sont deux nombres (`treize`→`trente`, `dix-huit`→`dix-sept`) et un nom"
        " propre (`Marie-Anne`→`Marianne`). Le glossaire ne contient que des"
        " acronymes administratifs : rien dans cette fonctionnalité ne peut les"
        " rattraper, et il ne faut pas le lui reprocher.\n"
    )

    add("## 5. Le refus — cas `06-no-cues`\n")
    add(
        "La question qui vient toujours : « comment vous évitez de mettre des mots"
        " dans la bouche de quelqu'un ? » Réponse : sur une réunion sans aucun"
        " indice de nom, le résolveur n'attribue rien.\n"
    )
    refusal = context["refusal"]
    add("- participants invités : %s" % ", ".join(refusal["attendees"]))
    add("- labels dans la transcription : %s" % ", ".join(refusal["labels"]))
    add("- noms attribués : **%s**" % (", ".join(refusal["assigned"]) or "aucun"))
    add("- labels laissés en `SPEAKER_XX` : **%s**" % ", ".join(refusal["unassigned"]))
    add("")
    add("Ce que le modèle a vu, et ce qui a été refusé :\n")
    add("```")
    for line in refusal["cues"] or ["(aucun indice détecté)"]:
        add(line)
    for line in refusal["rejected"]:
        add(line)
    add("```")
    add("")
    add(
        "Le piège est le segment 2, « comme Damien l'avait signalé » : une mention"
        " à la troisième personne. `resolve.py` ne transforme **jamais** une"
        " mention en attribution. La vérité terrain dit que SPEAKER_00 est"
        " réellement Fabienne Roussel — et que ne pas l'attribuer est la bonne"
        " réponse.\n"
    )

    add("## 6. Reproductibilité hors ligne\n")
    add(
        "Toutes les réponses d'Albert sont en cache sur disque"
        " (`demo/cache/llm-cache.json`, clé = hash du modèle et des deux"
        " prompts). Le second passage est rejoué avec `--offline`, qui"
        " transforme tout défaut de cache en erreur :\n"
    )
    add("```")
    add(
        "passage 1 : %d réponses lues en cache, %d appels à Albert"
        % (context["stats1"]["hits"], context["stats1"]["misses"])
    )
    add(
        "passage 2 (--offline, tout défaut de cache = erreur) : %d en cache, %d appels"
        % (context["stats2"]["hits"], context["stats2"]["misses"])
    )
    add("markdown identique entre les deux passages : %s" % context["replay_identical"])
    add("```")
    add("")

    add("## 7. Les lignes de journal qui prouvent que les étapes ont tourné\n")
    add("```")
    for line in context["logs"]:
        add(line)
    add("```")
    return "\n".join(out)


def variance_section() -> str:
    """Summarise the extra samples in `out/report-<tag>.md`, if any.

    Read back from the files rather than retyped, so the README can never
    claim a number no run produced. Brief §9: the point estimate is a point
    estimate, and the only honest way to show that is to show the spread.
    """
    import glob  # noqa: PLC0415

    paths = sorted(glob.glob(os.path.join(OUT_DIR, "report-*.md")))
    if not paths:
        return ""

    out = ["\n## 8. Variance entre deux passages identiques\n"]
    out.append(
        "Même entrée, même code, même seuil — seule la génération d'Albert"
        " change. `temperature=0` n'est pas reproductible chez eux : ce"
        " tableau est la preuve, pas une excuse écrite après coup.\n"
    )
    out.append("| passage | corrigées | sous le seuil | manquées | faux positifs |")
    out.append("|---|---|---|---|---|")

    rows = [("passage de référence (celui publié)", os.path.join(OUT_DIR, "report.md"))]
    rows += [(os.path.basename(p), p) for p in paths]

    blocks = []
    for label, path in rows:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        numbers = re.findall(
            r"^\| (?:corrigées|bonne réponse|manquées|faux positifs)"
            r"[^|]*\| \*{0,2}([0-9]+)",
            text,
            re.M,
        )
        if len(numbers) == 4:
            out.append("| %s | %s | %s | %s | %s |" % (label, *numbers))
        block = re.search(r"### Sous le seuil.*?```\n(.*?)```", text, re.S)
        if block:
            blocks.append((label, block.group(1).strip()))

    out.append("")
    out.append("Corrections sous le seuil observées, passage par passage :\n")
    out.append("```")
    for label, block in blocks:
        out.append("%s:" % label)
        for line in block.splitlines():
            out.append("  " + line)
    out.append("```")
    out.append("")
    out.append(
        "C'est exactement ce que le seuil est censé faire : quand le modèle"
        " n'est pas sûr, la correction est **remontée dans la liste d'audit**"
        " et le document reste tel quel. Une erreur laissée en place se"
        " corrige à la relecture ; un mot inventé, non.\n"
    )
    return "\n".join(out)


#: Appended to README.md after the report. It lives here, and not typed by
#: hand into README.md, because `main()` rewrites that file wholesale: a
#: section written straight into the markdown would disappear on the next run.
GLOSSARY_PAGE = """
## 9. La page « glossaire de l'organisation »

`correct_acronyms(..., user_glossary=...)` existe depuis `6429eb44` et n'a
jamais servi : `celery_worker._correct_acronyms_in` ne passe aucun mapping, donc
`build_index`, le `USER_GLOSSARY_BONUS` (0,20) et le marqueur
`[glossaire de l'organisation]` du prompt ne se déclenchent pas en production.
`demo/glossary_page.py` est la première chose qui les exerce.

On dépose un glossaire en texte brut, la page rejoue le même pipeline que
`run_demo.py` — elle l'importe, elle ne le recopie pas — avec et sans
glossaire, et affiche l'avant/après, la liste des corrections et le score.

```sh
cd src/summary
PYTHONHASHSEED=0 uv run --python 3.13 python demo/glossary_page.py
```

Puis http://localhost:8799. Rien n'est reconstruit, aucun conteneur n'est
touché : la page tourne sur l'hôte et lit `env.d/development/summary`, le même
fichier que `celery-summary-transcribe`. Un bouton séparé « Publier dans Docs »
crée un troisième document ; il n'y a pas de publication automatique, sinon
chaque essai laisserait un document derrière lui.

| | |
|---|---|
| `glossary_page.py` | le serveur : deux passages, le score, la publication |
| `glossary_page.html` | la page : dépôt par glisser-déposer **et** par clic |
| `glossary_parser.py` | l'analyseur tolérant (`=`, `;`, `:`), texte brut uniquement |
| `sample-glossary.txt` | le glossaire d'exemple, **figé avant la mesure** |

### Ce que la mesure donne

Même entrée, même seuil (0,80), 12 erreurs d'acronyme dans le périmètre.

| passage | corrigées | sous le seuil | manquées | faux positifs |
|---|---|---|---|---|
| sans glossaire (témoin, rejoué depuis le cache) | **7 / 12** | 1 | 4 | 0 |
| avec `sample-glossary.txt` (15 entrées) | **7 / 12** | 0 | 5 | 1 |

Le glossaire **n'améliore pas le score sur ce transcript**. Le seul faux positif
est `dix nomme` → DICOM à 0,70 : sous le seuil, donc **non appliqué** au
document — mais c'est bien un faux positif de plus qu'au passage témoin, et il
est compté comme tel.

Il faut lire ces deux lignes avec la variance en tête : le passage `run2`, même
code et **sans** glossaire, donne 6/12 avec 2 sous le seuil (§8). Un écart d'une
unité ne distingue donc pas un effet du glossaire d'un tirage d'Albert.

### DINUM franchit-il le seuil ? Non.

L'arithmétique attendue — 0,75 + 0,20 = 0,95 — additionne deux nombres qui n'ont
rien à voir. Le bonus s'applique à la **similarité phonétique**, dans
`_shortlist` ; `_decide` reçoit la liste et **jette les scores**
(`for acronym, _ in shortlist`). Le seuil, lui, compare la `confiance` rendue
par le modèle. Le bonus ne peut donc pas déplacer le nombre que le seuil
regarde. Ce qu'il fait vraiment : DINUM passe de 0,80 à 1,00 dans la liste et
gagne le marqueur `[glossaire de l'organisation]` dans le prompt.

Mesuré en rejouant la seule décision `e1` dix fois, cache contourné :

| condition | acronyme choisi | confiance | au-dessus de 0,80 |
|---|---|---|---|
| sans glossaire | DINUM 5/5 | 0,65 – 0,85 | **1 / 5** |
| avec glossaire | DINUM 4/5, DICOM 1/5 | 0,60 – 0,85 | **2 / 5** |

Autrement dit : DINUM était déjà choisi sans glossaire, et le glossaire ne le
fait pas franchir le seuil de façon fiable. À n = 5, l'écart ne se distingue pas
du bruit.

### Les quatre manquées : le glossaire ne peut structurellement rien y faire

- `type est` → Typst, `bloc note` → BlockNote, `Christ` → Grist : **l'étape 1
  ne signale jamais ces passages** (9 passages signalés sur 12 erreurs, la page
  les affiche). Sans signalement, aucune liste de candidats n'est construite :
  il n'y a rien sur quoi un glossaire pourrait peser. Pourtant la cible est bien
  en tête de liste quand on la construit à la main (Typst 0,80 → 1,00 avec le
  glossaire, BlockNote déjà à 1,00).
- `dos pecs` → Docspec : le passage **est** signalé, mais Docspec n'entre jamais
  dans la liste. Sa similarité phonétique est sous
  `acronym_correction_min_similarity` (0,62), et le bonus est ajouté **après**
  ce filtre. Un glossaire ne rattrape donc pas un mot phonétiquement éloigné.

### Une croyance à corriger

L'idée que ces cinq noms manqueraient parce qu'un corpus administratif ignore le
vocabulaire d'une équipe est **fausse ici** : les 15 termes du glossaire
d'exemple, DINUM, Typst, Docspec, BlockNote et Grist compris, sont **déjà** dans
le glossaire livré de 7769 entrées. Sur ce transcript, la démonstration ne peut
pas raconter « votre vocabulaire est absent ». Ce qu'elle montre est plus utile :
où la fonctionnalité bloque réellement — à l'étape 1, et au filtre de
similarité.

### Si Albert est injoignable

`_decide` rattrape ses propres exceptions et rend `(None, 0.0)` : l'étape échoue
**en silence**, le passage revient avec moins de corrections et rien ne remonte.
Vérifié en pointant `LLM_BASE_URL` sur un port mort — HTTP 200, sept corrections
au lieu de huit, aucune erreur. La page collecte donc les avertissements du
journal et affiche un bandeau rouge « passage dégradé » : un passage abîmé qui
ressemble à un passage propre est exactement le genre de silence que cette démo
doit refuser.

### Honnêteté

Le transcript, l'invitation `.ics` et les quatre participants sont des **mocks**
écrits pour cette démonstration ; aucune vraie réunion n'est passée dans ce
pipeline. La vérité terrain est `errors.json`, écrite avant la mesure.
`sample-glossary.txt` a été figé avant la mesure et n'a pas été retouché après
coup : ajuster les développés en voyant le score reviendrait à régler la démo
pour qu'elle flatte le résultat.

Capture : `/Users/macair/dinum/screenshots/demo-glossary-page.png`.
"""


README_HEADER = """# Démo — qualité des transcriptions : avant / après

Deux fonctionnalités, un seul transcript, deux documents La Suite Docs.

| | AVANT | APRÈS |
|---|---|---|
| `is_resolve_speaker_cues_enabled` | `False` | `True` |
| `is_acronym_correction_enabled` | `False` | `True` |
| `resolve_speaker_cues_detector` | — | `llm` |

L'« avant » n'est pas écrit à la main. C'est ce que fait `celery_worker.py`
quand les deux drapeaux sont fermés : aucune des deux étapes n'est appelée et
`format_transcript` tourne sur le JSON WhisperX brut. C'est le comportement
d'aujourd'hui pour Dictaphone, qui n'a jamais de métadonnées VAD — d'où les
`SPEAKER_00` du document brut.

Le script appelle les fonctions de `celery_worker.py`, dans son ordre :
`resolve_speaker_identities_and_apply_to` → `_correct_acronyms_in` →
`format_transcript`, puis `create_document_in_lasuite_docs`. Il ne réimplémente
rien et ne modifie ni les deux fonctionnalités, ni `docs_service.py`.

La démo vit dans `src/summary/demo/` et non à la racine du dépôt, parce que
seul `./src/summary` est monté dans les conteneurs (`/app`) : c'est la seule
place d'où elle peut tourner dans le vrai environnement du service.

| | |
|---|---|
| `run_demo.py` | le script : deux passages, les preuves, le score, la publication |
| `llm_cache.py` | cache disque devant `LLMService`, pour rejouer hors ligne |
| `input/` | copies en lecture seule des mocks (`mocks/flow-1-technique`, `speaker-attribution/cases/06-no-cues`) |
| `cache/` | les réponses d'Albert, **versionnées** : sans elles la démo dépend du wifi de la salle |
| `out/` | les markdown poussés dans Docs et les rapports |

## Comment le rejouer

```sh
docker exec -e IS_ACRONYM_CORRECTION_ENABLED=true \\
            -e IS_RESOLVE_SPEAKER_CUES_ENABLED=true \\
            -e RESOLVE_SPEAKER_CUES_DETECTOR=llm \\
            -e PYTHONHASHSEED=0 \\
            celery-summary-transcribe python /app/demo/run_demo.py
```

`--offline` interdit tout appel réseau (le cache doit suffire), `--no-docs`
saute la publication.

`PYTHONHASHSEED=0` n'est pas décoratif : `phonetic.PhoneticIndex` range les
acronymes dans un `set`, et `candidates()` départage les ex æquo par ordre
d'itération. Sans graine fixée, la liste de candidats — donc le prompt
`acronym-decide` — change d'un processus à l'autre et le cache disque rate.
La graine ne change aucune décision, elle enlève seulement l'aléa qui
empêcherait de rejouer la démo hors ligne.

"""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def load(path):
    """Read a JSON file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    """Run both passes, score them, publish, write the README."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="fail on a cache miss")
    parser.add_argument("--no-docs", action="store_true", help="do not publish")
    parser.add_argument(
        "--doc-ids",
        default="",
        help="two existing Docs ids, comma separated: reuse them in the report"
        " instead of creating a second pair. Use it when only the report text"
        " changed -- the two documents hold the transcripts, not the report.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="second sample of the same input: use its own cache, write"
        " out/report-<tag>.md, leave README.md and Docs alone. Albert is not"
        " reproducible at temperature 0, so this measures the spread instead"
        " of pretending there is none.",
    )
    args = parser.parse_args()
    if args.tag:
        args.no_docs = True
        global CACHE_PATH  # noqa: PLW0603
        CACHE_PATH = os.path.join(DEMO_ROOT, "cache", "llm-cache-%s.json" % args.tag)

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    from summary.core import celery_worker  # noqa: PLC0415
    from summary.core.config import get_settings  # noqa: PLC0415
    from summary.core.shared_models import WhisperXResponse  # noqa: PLC0415
    from summary.core.speaker_cues.attendees import parse_attendees  # noqa: PLC0415

    settings = get_settings()

    # The AFTER run must be the real settings path, not a simulated one.
    problems = []
    if not settings.is_resolve_speaker_cues_enabled:
        problems.append("IS_RESOLVE_SPEAKER_CUES_ENABLED must be true")
    if not settings.is_acronym_correction_enabled:
        problems.append("IS_ACRONYM_CORRECTION_ENABLED must be true")
    if settings.resolve_speaker_cues_detector != "llm":
        problems.append("RESOLVE_SPEAKER_CUES_DETECTOR must be llm")
    if problems:
        print("Refusing to run: " + "; ".join(problems), file=sys.stderr)
        return 2

    # `phonetic.PhoneticIndex.by_key` maps a key to a **set** of acronyms, and
    # `candidates()` breaks score+length ties by iteration order. Python
    # randomises string hashing per process, so without a pinned seed the
    # shortlist -- and therefore the `acronym-decide` prompt -- changes between
    # runs and the disk cache misses. Pinning the seed changes no decision; it
    # only removes the randomness that would defeat the cache on stage.
    if os.environ.get("PYTHONHASHSEED") != "0":
        print(
            "WARNING: PYTHONHASHSEED is not 0. The acronym shortlist is built"
            " from a set, so the cache will miss on replay. Re-run with"
            " -e PYTHONHASHSEED=0.",
            file=sys.stderr,
        )

    OFFLINE["value"] = args.offline
    install_scaffolding(celery_worker, settings.llm_model)

    logs = LogCapture()
    for name in (
        "summary.core.celery_worker",
        "summary.core.acronym_correction",
        "summary.core.speaker_dispatch",
    ):
        logging.getLogger(name).addHandler(logs)

    os.makedirs(OUT_DIR, exist_ok=True)

    flow = os.path.join(INPUT_DIR, "flow-1-technique")
    raw = load(os.path.join(flow, "03-transcript-whisperx.json"))
    errors = load(os.path.join(flow, "errors.json"))["errors"]
    attendees = parse_attendees(os.path.join(flow, "02-attendees.ics"))

    # Bare, outside any try/except: the wrappers fail open, so a schema
    # mismatch here would otherwise look like a model that found nothing.
    WhisperXResponse.model_validate(raw)

    print("=" * 78)
    print(
        "Entrée : mocks/flow-1-technique — %d segments, %d participants invités"
        % (len(raw["segments"]), len(attendees))
    )
    print("Modèle : %s (Albert)" % settings.llm_model)
    print("PAS de métadonnées VAD : c'est la situation Dictaphone.")
    print("=" * 78)

    # ---------------- BEFORE ----------------
    before = run_pipeline(
        celery_worker,
        WhisperXResponse.model_validate(raw),
        attendees,
        BEFORE,
        "demo-before",
        "fr",
    )
    with open(
        os.path.join(OUT_DIR, "flow-1-before.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write(before.markdown)

    # ---------------- AFTER -----------------
    stats_before_after = dict(STATS)
    after = run_pipeline(
        celery_worker,
        WhisperXResponse.model_validate(raw),
        attendees,
        AFTER,
        "demo-after",
        "fr",
    )
    stats1 = {
        "hits": STATS["hits"] - stats_before_after["hits"],
        "misses": STATS["misses"] - stats_before_after["misses"],
    }
    with open(
        os.path.join(OUT_DIR, "flow-1-after.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write(after.markdown)

    if after.markdown == before.markdown:
        print(
            "STOP: the AFTER transcript is identical to BEFORE. Both stages"
            " fail open, so this is a silent no-op, not a negative result.",
            file=sys.stderr,
        )
        for line in logs.lines:
            print("  log | " + line, file=sys.stderr)
        return 3

    # ------------- offline replay -----------
    offline_stats_start = dict(STATS)
    OFFLINE["value"] = True
    replay = run_pipeline(
        celery_worker,
        WhisperXResponse.model_validate(raw),
        attendees,
        AFTER,
        "demo-after",
        "fr",
    )
    stats2 = {
        "hits": STATS["hits"] - offline_stats_start["hits"],
        "misses": STATS["misses"] - offline_stats_start["misses"],
    }
    OFFLINE["value"] = args.offline

    # ------------- refusal case -------------
    case = os.path.join(INPUT_DIR, "06-no-cues")
    case_raw = load(os.path.join(case, "03-transcript-whisperx.json"))
    case_attendees = parse_attendees(os.path.join(case, "02-attendees.ics"))
    WhisperXResponse.model_validate(case_raw)
    refusal_run = run_pipeline(
        celery_worker,
        WhisperXResponse.model_validate(case_raw),
        case_attendees,
        Flags(
            is_resolve_speaker_cues_enabled=True, is_acronym_correction_enabled=False
        ),
        "demo-refusal",
        "fr",
    )
    refusal_segments = refusal_run.transcription.model_dump().get("segments") or []
    refusal_labels = []
    for segment in case_raw["segments"]:
        if segment.get("speaker") and segment["speaker"] not in refusal_labels:
            refusal_labels.append(segment["speaker"])
    refusal_after_labels = []
    for segment in refusal_segments:
        if segment.get("speaker") and segment["speaker"] not in refusal_after_labels:
            refusal_after_labels.append(segment["speaker"])
    refusal_assigned = [
        label for label in refusal_after_labels if not label.startswith("SPEAKER_")
    ]
    refusal_cues = [
        "cue %-8s segment %-3d speaker %-11s « %s » -> %s"
        % (
            cue.kind,
            cue.segment_index,
            cue.speaker,
            (cue.span or cue.quote).strip()[:60],
            cue.name or "(aucun nom)",
        )
        for cue in (refusal_run.trace.cues if refusal_run.trace else [])
    ]
    refusal_rejected = [
        "rejeté %-11s %s" % (label, reason)
        for label, reason in (refusal_run.trace.rejected if refusal_run.trace else [])
    ]
    with open(
        os.path.join(OUT_DIR, "06-no-cues-after.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write(refusal_run.markdown)

    # ---------------- evidence ---------------
    names = name_evidence(after.trace, after.transcription)
    applied, blocked = acronym_evidence(after.corrections)
    result = score(after.corrections, errors)

    after_labels = []
    for segment in after.transcription.model_dump().get("segments") or []:
        speaker = segment.get("speaker")
        if speaker and speaker not in after_labels:
            after_labels.append(speaker)
    unassigned = [label for label in after_labels if label.startswith("SPEAKER_")]

    # ---------------- publish ----------------
    documents = [
        ("AVANT", TITLE_BEFORE, None),
        ("APRÈS", TITLE_AFTER, None),
    ]
    if args.doc_ids:
        reused = [part.strip() for part in args.doc_ids.split(",") if part.strip()]
        documents = [
            ("AVANT", TITLE_BEFORE, reused[0]),
            ("APRÈS", TITLE_AFTER, reused[1]),
        ]
    elif not args.no_docs:
        doc_ids = DocIdCapture()
        logging.getLogger("summary.core.docs_service").addHandler(doc_ids)
        celery_worker.create_document_in_lasuite_docs(
            content=before.markdown, title=TITLE_BEFORE, email=DOCS_EMAIL, sub=DOCS_SUB
        )
        celery_worker.create_document_in_lasuite_docs(
            content=after.markdown, title=TITLE_AFTER, email=DOCS_EMAIL, sub=DOCS_SUB
        )
        if len(doc_ids.ids) == 2:
            documents = [
                ("AVANT", TITLE_BEFORE, doc_ids.ids[0]),
                ("APRÈS", TITLE_AFTER, doc_ids.ids[1]),
            ]

    context = {
        "documents": documents,
        "names": names,
        "unassigned": unassigned,
        "speakers": speaker_score(before.transcription, after.transcription, errors),
        "sentences": {
            index: (segment.get("text") or "").strip()
            for index, segment in enumerate(
                after.transcription.model_dump().get("segments") or []
            )
        },
        "applied": applied,
        "blocked": blocked,
        "min_confidence": settings.acronym_correction_min_confidence,
        "score": result,
        "refusal": {
            "attendees": [a["name"] for a in case_attendees],
            "labels": refusal_labels,
            "assigned": refusal_assigned,
            "unassigned": [
                label for label in refusal_after_labels if label.startswith("SPEAKER_")
            ],
            "cues": refusal_cues,
            "rejected": refusal_rejected,
        },
        "stats1": stats1,
        "stats2": stats2,
        "replay_identical": "oui" if replay.markdown == after.markdown else "NON",
        "logs": [
            line
            for line in logs.lines
            if "resolution for task" in line.lower()
            or line.startswith("Acronym correction")
        ],
    }

    report = build_report(context)
    print()
    print(report)

    if args.tag:
        path = os.path.join(OUT_DIR, "report-%s.md" % args.tag)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")
        print()
        print("Écrit : %s (échantillon de variance, README.md non touché)" % path)
        return 0

    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    with open(os.path.join(DEMO_ROOT, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(
            README_HEADER
            + "\n"
            + HONESTY
            + "\n"
            + report
            + "\n"
            + variance_section()
            + GLOSSARY_PAGE
        )

    print()
    print(
        "Écrit : demo/README.md, demo/out/report.md, demo/out/flow-1-{before,after}.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
