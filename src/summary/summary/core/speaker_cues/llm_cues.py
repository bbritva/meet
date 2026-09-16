"""Detect name cues with a language model instead of regex templates.

Same job as `cues.py`, same output type (`Cue`), same downstream: `resolve.py`
turns cues into an `AssignmentResult` and nothing about the threshold, the
noisy-OR or the uniqueness rule changes. Only the DETECTION stage is swapped.

Why
---
The template detector scores 17/17 on cases 01-06 and 0/19 on cases 07-12.
That is not a bug in the templates, it is what a closed list of six
formulations does out of sample: « Je me présente : X », « Alors moi c'est X »
and « je passe la parole à X » are not on the list, so nothing is detected.
A model reads the sentence instead of matching it.

The gateway
-----------
This module owns no transport and no credentials. It calls an injected
`complete(system, user) -> str`; the worker builds one from the application's
`LLMService` via `transport_from_llm_service()`, so the endpoint, the model and
the API key are the ones already configured in `config.py` (`LLM_BASE_URL`,
`LLM_MODEL`, `LLM_API_KEY` -- Albert, in the deployed configuration) and
observability keeps working for free.

That injection is also what keeps the tests offline: they pass a stub and this
module never imports a socket.

What the model is allowed to change, and what it is not
-------------------------------------------------------
The model widens *recall*, never *what counts as an attribution*. Four guards,
all applied here, after the answer comes back:

  1. a `name` that is not on the roster is REJECTED, unless the type is
     `self_id` -- an uninvited person who introduces themselves is the one
     documented case (case 05, case 11) where a name comes from the transcript
     alone;
  2. `mention` becomes a `MENTION` cue, which `resolve.py` never turns into
     evidence. A model that decides « je remplace Léa » identifies the speaker
     is therefore harmless: the kind it returned already disqualifies it;
  3. roster matching is EXACT or NORMALISED only -- the model is asked to copy
     a roster name verbatim, so a near-miss means it invented something;
  4. a name that matches two attendees equally well (the two Jeans) is marked
     ambiguous and `resolve.py` drops it;
  5. a `self_id` for P is dropped when the SAME anonymous label also mentions
     P in the third person somewhere else in the meeting. The model answers
     one segment at a time and cannot see that it just contradicted itself.

Measured accuracy
-----------------
Three corpora, all synthetic, ordered from most to least flattering:

  * **self-identification heavy** (cases 01-12): precision 1.00, recall 0.74;
  * **handoff-dense** (13-18): precision 0.80, recall 0.44. Handoff-only
    recall 0.47. Built one-failure-mode-per-case, so this is a worst case;
  * **long and realistic** (19-21, 25-28 min, ~1 cue per speaker becomes 3-5):
    **precision 0.92, recall 0.79**, handoff-only recall 0.80. This is the
    closest thing we have to a real meeting and the number to quote.

Recall on handoffs is roughly half that on self-identifications, because a
self-id is an observation ("Ici Camille" — the speaker IS Camille) while a
handoff is a prediction about who speaks next, and meetings interrupt.

The limit is DETECTION, not arbitration. The model sees 73 of 108 ground-truth
cues (regex sees 31). A global bipartite assignment and a closed-set
elimination rule were both implemented and measured: the first changed one
label in 63 runs, the second never fired in 42. Neither is shipped.

What it misses, measured on the held-out cases 07-12
----------------------------------------------------
Five labels out of nineteen, in three families:

  * **the telephone register.** « Vous avez Myriam Toussaint au bout du fil »
    and « Vous êtes bien avec Madeleine Fabre » both come back as `mention`.
    The model reads the name as somebody else's, which is what those words
    would mean outside a meeting opening;
  * **a name mangled into an ordinary word.** « carton, je t'écoute » for
    Xavier Carthon comes back as `none`. « cent drine » for Sandrine and
    « quatre mer » for Quatremer are recovered, so this is not hopeless, just
    unreliable;
  * **the right name under the wrong kind.** « Alors, seau laine, pour le
    ministère de la Santé » is decoded to Solène Pruvost — correctly, from two
    common nouns — but typed `handoff` instead of `self_id`, and `resolve.py`
    then drops it because the person named is the one already speaking. This
    one is worth a future rule: a handoff whose target turns out to be the
    current speaker is a self-identification wearing the wrong label.

And one refusal that cost a true positive: « Je passe la parole à Sébastien »
came back as a `handoff` with `name: null`. Rule 4 of the prompt (say nothing
rather than risk a wrong name) firing where it did not need to.

`temperature=0` does not make this reproducible: two cold runs both scored
14/19 on cases 07-12, on DIFFERENT labels. Every figure above is one draw;
treat them as +/- one label per case. The committed cache freezes that draw.

Caching
-------
One cache entry per SEGMENT, keyed by a hash of (prompt version, model,
roster, segment text). Batching is an API-call optimisation, not part of the
key, so re-running with a different batch size still hits the cache. With the
cache committed, `score.py` runs offline and reproduces the published numbers
exactly.
"""

# Wide signatures are the injection points the tests rely on (transport,
# cache, batch size, ceiling), and the branchiness is the defensive
# parsing and the guards. Both are the point of the module.
# ruff: noqa: PLR0911, PLR0912, PLR0913, PLR0917

import hashlib
import json
import logging
import os
import re
import time

from summary.core.llm_service import LLMException
from summary.core.speaker_cues.cues import HANDOFF, MENTION, SELF_ID, Cue
from summary.core.speaker_cues.match import match_span

logger = logging.getLogger(__name__)

#: Bump when the prompt changes, or cached answers from the old prompt would
#: be silently reused and the numbers would mean nothing.
PROMPT_VERSION = "3"

#: The cache is in-memory by default. The prototype wrote it to disk so the
#: published scores could be reproduced offline; inside the worker that would
#: mean writing into the installed package directory, so a caller who wants
#: persistence has to ask for it by passing an explicit path.
DEFAULT_CACHE_PATH = None

#: Segments per API call. One call per segment would be ~40 calls per meeting;
#: one call for the whole meeting makes the model lose the index mapping.
DEFAULT_BATCH_SIZE = 12
# Hard ceiling on model calls for ONE transcript. Calls scale as
# segments/batch_size, so a pathological transcript (e.g. a segmenter bug
# emitting one segment per word) would otherwise scale without bound. Past
# the cap we stop asking and leave the remaining segments unclassified; the
# caller falls back to the regex detector for those.
#
# Sizing: measured density on the case corpus is ~8.9 segments/min, i.e.
# ~45 calls/hour at batch_size=12. The ceiling must never bite a real
# meeting, only pathology:
#     1 h  ~=  45 calls        4 h  ~= 178 calls
#     2 h  ~=  89 calls        8 h  ~= 356 calls
# 400 leaves headroom past a full day of talking (~4800 segments) while
# still bounding a word-level segmentation bug (~750 calls/hour).
DEFAULT_MAX_CALLS = 400

DEFAULT_TIMEOUT = 180
MAX_OUTPUT_TOKENS = 4096

#: A fuzzy threshold no score can reach: the model is told to copy a roster
#: name verbatim, so only exact and normalised hits are roster membership.
_EXACT_ONLY = 1.01

TYPES = {"self_id", "handoff", "mention", "none"}
_TYPE_TO_KIND = {"self_id": SELF_ID, "handoff": HANDOFF, "mention": MENTION}

#: An out-of-roster self-identification may only produce a short, name-shaped
#: string. Without this the model could hand back a whole clause.
_MAX_UNKNOWN_TOKENS = 3


class LLMUnavailable(RuntimeError):
    """No cached answer and no way to ask for one."""


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Tu analyses la transcription automatique d'une réunion administrative \
française. Elle a été produite par un système de reconnaissance vocale : les \
noms propres y sont souvent mal transcrits.

Voici la liste des personnes invitées à cette réunion (le « répertoire ») :
%(roster)s

On te donne des segments numérotés de la transcription, chacun précédé de \
l'étiquette anonyme de son locuteur (SPEAKER_00, SPEAKER_01, …). Ces \
étiquettes ne sont PAS des noms : la reconnaissance vocale sait seulement que \
deux segments portant la même étiquette ont été prononcés par la même voix.

Pour CHAQUE segment, dis s'il contient un indice sur l'identité d'un locuteur, \
et lequel :

- "self_id" : la personne qui parle donne SON PROPRE nom.
- "handoff" : la personne qui parle nomme QUELQU'UN D'AUTRE, soit pour lui \
donner la parole, soit pour l'interpeller directement.
- "mention" : un nom de personne est prononcé, mais il ne désigne ni celui qui \
parle, ni celui à qui on donne la parole.
- "none" : ce segment ne contient aucun nom de personne.

Règles, dans cet ordre :

1. Les noms sont MAL TRANSCRITS. « carie msahli » est Karim Sahli, « ma tilde » \
est Mathilde, « vingt sang » est Vincent, « sois raya » est Soraya. Un nombre \
ou un nom commun placé là où seul un nom de personne peut se trouver — \
« Vingt sang, tu peux nous dire… » — reste un nom de personne : c'est la place \
du mot dans la phrase qui compte, pas le mot lui-même.
2. Le champ "name" doit contenir EXACTEMENT un nom du répertoire ci-dessus, \
recopié tel qu'il y est écrit, jamais la forme entendue dans la transcription.
3. N'INVENTE JAMAIS un nom absent du répertoire. Unique exception : un \
"self_id" d'une personne qui n'est pas invitée (elle arrive en cours de \
réunion, elle se présente) ; recopie alors le prénom prononcé, tel quel.
4. Dans le doute, réponds "name": null. Ne rien dire coûte beaucoup moins cher \
qu'un nom faux : un nom faux met des paroles dans la bouche de quelqu'un qui \
ne les a pas prononcées.
5. Si le prénom prononcé correspond à plusieurs personnes du répertoire (deux \
« Jean »), réponds "name": null.
6. « Je remplace Léa », « au nom de Léa », « à la place de Léa » : celui qui \
parle n'est PAS Léa. C'est une "mention", jamais un "self_id".
7. Les étiquettes servent à trancher entre "self_id" et "handoff" : dans un \
"handoff", c'est une AUTRE étiquette qui prend la parole juste après. Si la \
même étiquette continue de parler, un nom prononcé au début de son tour est \
son propre nom.
8. Un nom propre accolé à une salutation au début d'un tour de parole est une \
présentation, pas une interpellation : « Gaël Prigent, bonjour. » est un \
"self_id" de Gaël Prigent.

Réponds UNIQUEMENT par un tableau JSON, un objet par segment fourni, dans \
l'ordre, sans aucun texte avant ni après :

[{"index": <numéro du segment>, "type": "self_id|handoff|mention|none", \
"name": "<nom du répertoire, ou null>", "raw": "<extrait exact du segment>"}]
"""


def build_system_prompt(attendees: list[dict[str, str]]) -> str:
    """The system message: role, roster, and the strict JSON contract."""
    roster = "\n".join("- %s" % a["name"] for a in attendees)
    return SYSTEM_PROMPT % {"roster": roster}


def build_user_prompt(batch: list[tuple[int, str, str]]) -> str:
    """The numbered segments of one batch, each with its anonymous label.

    The label is in the transcript already (`03-transcript-whisperx.json`), so
    handing it over invents nothing. It is what separates « Gaël Prigent,
    bonjour » spoken by Gaël from the same words spoken to Gaël: in a handoff
    the next segment carries a different label.
    """
    lines = ["Segments à analyser :", ""]
    for index, speaker, text in batch:
        lines.append(
            "[%d] (%s) %s" % (index, speaker or "SPEAKER_??", (text or "").strip())
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
class SegmentCache:
    """One entry per segment, keyed by prompt version + model + roster + text.

    Batching is deliberately NOT part of the key: it is an API-call
    optimisation, and a re-run with another batch size must still be free.
    """

    def __init__(self, path: str | None = DEFAULT_CACHE_PATH):
        """Load the cache at `path`, or start an empty in-memory one."""
        self.path = path
        self.entries: dict[str, dict] = {}
        self.dirty = False
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                self.entries = json.load(handle)

    @staticmethod
    def key(model: str, roster: list[str], text: str) -> str:
        """Return the cache key for one segment under one roster."""
        blob = json.dumps(
            [PROMPT_VERSION, model, roster, text],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        """Return the cached answer for `key`, or None, counting the hit."""
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def set(self, key: str, value: dict) -> None:
        """Store one answer, marking the cache dirty."""
        self.entries[key] = value
        self.dirty = True

    def save(self) -> None:
        """Write the cache back to disk, if it has a path and has changed."""
        if not (self.path and self.dirty):
            return
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(
                self.entries, handle, ensure_ascii=False, indent=1, sort_keys=True
            )
            handle.write("\n")
        self.dirty = False


# --------------------------------------------------------------------------
# parsing -- defensive on purpose
# --------------------------------------------------------------------------
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_response(text: str, indices: list[int]) -> dict[int, dict]:
    """Pull `{index, type, name, raw}` objects out of whatever came back.

    Three levels of tolerance, because a malformed answer must cost recall on
    one batch, never a crash:

      1. the whole reply parses as a JSON array;
      2. the first `[...]` span in the reply parses;
      3. individual `{...}` objects are scraped out one by one.

    Anything whose `index` is not in `indices`, or whose `type` is outside the
    closed vocabulary, is dropped.
    """
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    candidates = [text]
    array = _ARRAY_RE.search(text)
    if array:
        candidates.append(array.group(0))

    rows: list = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            rows = parsed
            break
    if not rows:
        for match in _OBJECT_RE.finditer(text):
            try:
                rows.append(json.loads(match.group(0)))
            except (ValueError, TypeError):
                continue

    allowed = set(indices)
    out: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        if isinstance(index, str) and index.strip().lstrip("-").isdigit():
            index = int(index.strip())
        if not isinstance(index, int) or index not in allowed:
            continue
        kind = row.get("type")
        if kind not in TYPES:
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            name = None
        raw = row.get("raw")
        out[index] = {
            "type": kind,
            "name": name.strip() if name else None,
            "raw": raw.strip() if isinstance(raw, str) else "",
        }
    return out


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def classify_segments(
    segments: list[dict],
    attendees: list[dict[str, str]],
    complete=None,
    model: str = "",
    cache: SegmentCache | None = None,
    cache_path: str | None = DEFAULT_CACHE_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_calls: int = DEFAULT_MAX_CALLS,
    stats: dict | None = None,
) -> dict[int, dict]:
    """Return `{segment index: {type, name, raw}}`, cached answers included.

    `complete(system, user)` is the transport, and it is required as soon as a
    segment is missing from the cache: this module never builds one itself, so
    the credentials and the endpoint stay in `LLMService`. Tests inject a stub,
    which is why nothing here can reach the network.

    Args:
        segments: WhisperX segments.
        attendees: The roster, `[{name, email}]`.
        complete: `complete(system, user) -> str` transport.
        model: Cache-key discriminator only; the model actually used is the
            one `LLMService` is configured with.
        cache: A `SegmentCache`, or None to build one from `cache_path`.
        cache_path: Path of a JSON cache, or None for memory only.
        batch_size: Segments per model call.
        max_calls: Hard ceiling on model calls for one transcript.
        stats: Optional dict, filled with call counters.

    Returns:
        `{segment index: {type, name, raw}}`.
    """
    cache = cache if cache is not None else SegmentCache(cache_path)
    roster = [a["name"] for a in attendees]
    system = build_system_prompt(attendees)

    results: dict[int, dict] = {}
    todo: list[tuple[int, str, str]] = []
    keys: dict[int, str] = {}
    for index, segment in enumerate(segments):
        text = (segment.get("text") or "").strip()
        key = SegmentCache.key(model, roster, text)
        keys[index] = key
        hit = cache.get(key)
        if hit is not None:
            results[index] = hit
        else:
            todo.append((index, segment.get("speaker") or "", text))

    calls = 0
    elapsed = 0.0
    capped = False
    if todo:
        if complete is None:
            raise LLMUnavailable(
                "the LLM cue detector has no transport: pass complete=..., "
                "built from the application LLMService with "
                "transport_from_llm_service()"
            )

        for start in range(0, len(todo), batch_size):
            if max_calls is not None and calls >= max_calls:
                # Ceiling reached. Stop asking; the segments still in `todo`
                # simply get no LLM answer and the caller degrades to regex.
                capped = True
                break
            batch = todo[start : start + batch_size]
            indices = [index for index, _, _ in batch]
            began = time.time()
            try:
                reply = complete(system, build_user_prompt(batch))
            except LLMException as error:
                raise LLMUnavailable("the LLM is unreachable: %s" % error) from error
            elapsed += time.time() - began
            calls += 1
            answers = parse_response(reply, indices)
            for index, _, _ in batch:
                answer = answers.get(index)
                if answer is None:
                    # Nothing usable came back for this segment. Do NOT cache
                    # the gap: the next run gets to ask again.
                    results[index] = {"type": "none", "name": None, "raw": ""}
                    continue
                cache.set(keys[index], answer)
                results[index] = answer
        cache.save()

    if stats is not None:
        stats.update(
            {
                "calls": calls,
                "segments": len(segments),
                "asked": len(todo),
                "cache_hits": cache.hits,
                "seconds": round(elapsed, 2),
                "capped": capped,
                "max_calls": max_calls,
            }
        )
    return results


# --------------------------------------------------------------------------
# answer -> Cue, with the guards
# --------------------------------------------------------------------------
def _looks_like_a_name(text: str) -> bool:
    """Short, capitalised, alphabetic, not an acronym."""
    tokens = text.split()
    if not tokens or len(tokens) > _MAX_UNKNOWN_TOKENS:
        return False
    for token in tokens:
        cleaned = token.strip(".,;:!?«»\"'()")
        if len(cleaned) < 2 or not cleaned[0].isupper() or cleaned.isupper():
            return False
        if not all(c.isalpha() or c in "-'" for c in cleaned):
            return False
    return True


def cue_from_answer(
    index: int,
    segment: dict,
    answer: dict,
    attendees: list[dict[str, str]],
    allow_unknown_self_id: bool = True,
) -> tuple[Cue | None, str]:
    """Turn one model answer into a `Cue`, or into a reason for dropping it.

    This is where every guard lives. `resolve.py` is not changed and must not
    be: whatever the model returns, what reaches the scorer is a `Cue` of a
    kind the existing rules already know how to distrust.
    """
    kind = _TYPE_TO_KIND.get(answer.get("type") or "")
    if kind is None:
        return None, ""

    speaker = segment.get("speaker") or ""
    text = (segment.get("text") or "").strip()
    raw = (answer.get("raw") or "").strip() or text
    name = answer.get("name")
    if not name:
        return None, "%s with no name" % answer.get("type")

    # Roster membership: exact or normalised only. The model was told to copy
    # a roster name verbatim, so anything softer means it made one up.
    match = match_span(name, attendees, fuzzy_threshold=_EXACT_ONLY)
    if match.candidates:
        best = match.candidates[0]
        return (
            Cue(
                kind=kind,
                segment_index=index,
                speaker=speaker,
                span=name,
                quote=raw,
                trigger="llm:%s" % answer["type"],
                name=best.name,
                email=best.email,
                tier=best.tier,
                similarity=best.score,
                ambiguous=match.is_ambiguous,
                candidates=[c.name for c in match.candidates],
                reason=(
                    "third-person mention: proves the person exists, nothing else"
                    if kind == MENTION
                    else ""
                ),
            ),
            "",
        )

    # GUARD: off the roster. Only a self-identification may introduce a name
    # the invite never heard of -- and only a name-shaped one.
    if kind is not SELF_ID:
        return None, "%r is not on the roster and the cue is a %s -- rejected" % (
            name,
            answer["type"],
        )
    if not allow_unknown_self_id:
        return None, "%r is not on the roster and out-of-roster self-id is off" % name
    if not _looks_like_a_name(name):
        return None, "%r is not on the roster and does not look like a name" % name
    return (
        Cue(
            kind=SELF_ID,
            segment_index=index,
            speaker=speaker,
            span=name,
            quote=raw,
            trigger="llm:self_id",
            name=name,
            email="",
            tier="out_of_roster",
            similarity=1.0,
        ),
        "",
    )


def drop_self_contradictions(
    cues: list[Cue], rejected: list[tuple[str, str]] | None = None
) -> list[Cue]:
    """GUARD: one voice cannot be Mounir and talk about Mounir.

    The model is answering segment by segment, so it can happily label
    « Mounir nous a envoyé ses chiffres » a mention and, twenty segments
    later, « Mounir, il faudra le relancer » a self-identification -- both by
    the same anonymous label. Read together the two answers contradict each
    other: nobody refers to themselves in the third person.

    So a `self_id` for P is dropped whenever the SAME label also produced a
    third-person mention of P. This costs recall in the rare case where the
    mention was the wrong answer, and that is the intended trade: this is the
    guard that keeps a meeting with no cue at all producing no name at all.

    Found on `12-mentions-seules`, where the model read a dislocated
    « Mounir, il faudra le relancer une dernière fois » as a vocative.
    """
    mentioned: dict[str, set[str]] = {}
    for cue in cues:
        if cue.kind == MENTION and cue.name:
            mentioned.setdefault(cue.speaker, set()).add(cue.name)

    kept: list[Cue] = []
    for cue in cues:
        if cue.kind is SELF_ID and cue.name in mentioned.get(cue.speaker, ()):
            if rejected is not None:
                rejected.append(
                    (
                        cue.speaker,
                        "segment %d: self_id %r dropped -- the same voice mentions "
                        "%s in the third person elsewhere"
                        % (cue.segment_index, cue.span, cue.name),
                    )
                )
            continue
        kept.append(cue)
    return kept


def transport_from_llm_service(llm_service, name: str = "speaker-cues"):
    """Adapt an application `LLMService` to the `complete(system, user)` shape.

    Keeps the endpoint, the model and the API key in `LLMService` (and keeps
    Langfuse observability working), so this module needs no settings and no
    credentials of its own.

    Args:
        llm_service: A configured `summary.core.llm_service.LLMService`.
        name: Observability name for the calls.

    Returns:
        A `complete(system, user) -> str` callable.
    """

    def complete(system: str, user: str) -> str:
        return llm_service.call(system, user, name=name) or ""

    return complete


def detect_cues_llm(
    segments: list[dict],
    attendees: list[dict[str, str]],
    allow_unknown_self_id: bool = True,
    rejected: list[tuple[str, str]] | None = None,
    **options,
) -> list[Cue]:
    """Same contract as `cues.detect_cues`, one model call per batch.

    `options` are forwarded to `classify_segments` (`complete`, `model`,
    `cache`, `cache_path`, `batch_size`, `max_calls`, `stats`).
    """
    answers = classify_segments(segments, attendees, **options)

    # Segments the model never saw (call ceiling reached) fall back to the
    # regex detector, so a cap degrades recall instead of silently dropping
    # the tail of a long transcript.
    missing = [i for i in range(len(segments)) if i not in answers]
    fallback_cues: list[Cue] = []
    if missing:
        from summary.core.speaker_cues.cues import (  # noqa: PLC0415
            detect_cues as _regex_detect,
        )

        window = [segments[i] for i in missing]
        remap = dict(enumerate(missing))
        for cue in _regex_detect(
            window, attendees, allow_unknown_self_id=allow_unknown_self_id
        ):
            cue.segment_index = remap[cue.segment_index]
            fallback_cues.append(cue)

    cues: list[Cue] = list(fallback_cues)
    for index, segment in enumerate(segments):
        answer = answers.get(index)
        if not answer:
            continue
        cue, reason = cue_from_answer(
            index,
            segment,
            answer,
            attendees,
            allow_unknown_self_id=allow_unknown_self_id,
        )
        if cue is not None:
            cues.append(cue)
        elif reason and rejected is not None:
            rejected.append(
                (segment.get("speaker") or "", "segment %d: %s" % (index, reason))
            )
    return drop_self_contradictions(cues, rejected)
