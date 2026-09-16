"""A disk cache in front of `LLMService`, so the demo replays without network.

Demo scaffolding. Nothing here is part of either feature: it wraps the
application `LLMService` without changing it, and both features keep calling
`llm_service.call(...)` exactly as they do in `celery_worker.py`.

Why a cache at the `LLMService` level rather than per feature
------------------------------------------------------------
The speaker-cue detector already owns a `SegmentCache`, the acronym corrector
owns nothing. Caching one layer lower -- the single `call(system, user, name)`
both of them go through -- gives one mechanism, one file, and covers every
stage: `speaker-cues`, `acronym-detect` and `acronym-decide`.

The key is a hash of (model, name, system prompt, user prompt). Nothing else:
not the environment, not the API key, not the date. Two runs that ask the
model the same question therefore hit, and a run that asks a *different*
question misses and is honest about it.

`offline=True` turns a miss into an error instead of a network call. That is
what makes "it replays offline" a checked claim rather than a hope: the demo
script runs the whole pipeline a second time with `--offline` and asserts the
call count is zero.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Optional


class CacheMiss(RuntimeError):
    """An offline run needed an answer that is not in the cache."""


def _key(model: str, name: str, system: str, user: str) -> str:
    digest = hashlib.sha256()
    for part in (model, name, system, user):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class LazyObservability:
    """Stands in for `LLMObservability` until a real call is actually needed.

    `celery_worker` builds the observability object before the service, so a
    fully cached run would otherwise still construct a Langfuse/OpenAI client
    for calls it never makes. Holding the keyword arguments and building the
    real object on the first cache miss keeps an offline run free of clients.
    """

    def __init__(self, **kwargs: Any):
        """Record the arguments `celery_worker` passed."""
        self.kwargs = kwargs


class CachingLLMService:
    """Same `.call()` shape as `LLMService`, answers from disk when it can."""

    def __init__(
        self,
        cache_path: str,
        observability_kwargs: dict,
        model: str,
        offline: bool = False,
        stats: Optional[dict] = None,
    ):
        """Open (or start) the cache at `cache_path`."""
        self._path = cache_path
        self._observability_kwargs = observability_kwargs
        self._model = model
        self._offline = offline
        self._stats = stats if stats is not None else {}
        self._stats.setdefault("hits", 0)
        self._stats.setdefault("misses", 0)
        self._real = None
        self._entries: dict[str, dict] = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                self._entries = json.load(handle)

    def _service(self):
        if self._real is None:
            # Imported late so a fully cached run never touches the module.
            from summary.core.llm_service import (  # noqa: PLC0415
                LLMObservability,
                LLMService,
            )

            self._real = LLMService(
                llm_observability=LLMObservability(**self._observability_kwargs)
            )
        return self._real

    def _save(self) -> None:
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as handle:
            json.dump(
                self._entries, handle, ensure_ascii=False, indent=1, sort_keys=True
            )

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        name: str,
        response_format: Optional[Mapping[str, Any]] = None,
    ):
        """Return the cached reply, or ask the real service and cache it."""
        key = _key(self._model, name, system_prompt, user_prompt)
        hit = self._entries.get(key)
        if hit is not None:
            self._stats["hits"] += 1
            return hit["reply"]

        if self._offline:
            raise CacheMiss(
                "offline run: no cached answer for a %r call (key %s)"
                % (name, key[:12])
            )

        reply = self._service().call(
            system_prompt, user_prompt, name, response_format=response_format
        )
        self._stats["misses"] += 1
        self._entries[key] = {"name": name, "model": self._model, "reply": reply}
        # Written after every miss: a crash halfway through a long run keeps
        # the answers already paid for.
        self._save()
        return reply
