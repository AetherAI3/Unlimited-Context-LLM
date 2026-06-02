"""Thin, OPTIONAL client of the closed AetherCloud API — the *only* seam to hosted.

Moat boundary (see BUILD_PLAN §9): this file **calls** the closed API; it never **contains**
any hosted logic. No closed schema, no admission gate, no learner, no router, no usage/policy
layer, no signing internals — nothing here lets a fork reconstitute the hosted service. It is a
dumb request/response over HTTPS, and it is **None by default** on a :class:`~aether_context.session.Session`
(the engine runs fully local with no client at all).

Two operations, both plain HTTP with the standard library (no extra dependency, matching the
Ollama path):

  * :meth:`retrieve` — ask the hosted pool for the nearest slices to a query (the hosted
    counterpart of the local pool's ``search``). Returns local :class:`Slice` objects so the
    rest of the engine never knows whether a slice came from disk or from hosted.
  * :meth:`submit` — hand the session's local harvest candidates to the hosted intake on
    close (a one-way push; what hosted *does* with them is closed and not our concern here).

Fail-soft: with no ``base_url`` the client is inert — every method is a clean no-op that never
raises into a run. A network error is wrapped as a typed
:class:`~aether_context.errors.BackendUnavailable` only when the caller explicitly opted in by
configuring a ``base_url``; the session's close path still swallows it so a hosted outage never
crashes a local run.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable, Sequence

import numpy as np

from aether_context._log import get_logger
from aether_context.context_pool import Slice
from aether_context.errors import BackendUnavailable

logger = get_logger(__name__)

#: Default HTTP timeout (seconds) for hosted calls — generous but bounded.
DEFAULT_TIMEOUT: int = 30
#: Retrieval endpoint path appended to ``base_url``.
_RETRIEVE_PATH = "/v1/retrieve"
#: Harvest-submit endpoint path appended to ``base_url``.
_SUBMIT_PATH = "/v1/harvest"


class AtlasClient:
    """A no-op-by-default HTTPS client of the closed AetherCloud API.

    Args:
        base_url: The hosted API base URL. ``None`` (the default) makes every method an inert
            no-op — this is the fully-local mode and the moat-safe default.
        token: Optional bearer token sent as ``Authorization: Bearer <token>``.
        timeout: Per-request timeout in seconds.

    The client holds **no** atlas state; it only serializes a request and parses a response.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url: str | None = base_url.rstrip("/") if base_url else None
        self._token: str | None = token
        self._timeout: int = int(timeout)

    @property
    def configured(self) -> bool:
        """Whether a ``base_url`` was provided (else the client is an inert no-op)."""
        return self.base_url is not None

    # -- retrieve (hosted counterpart of pool.search) -------------------------
    def retrieve(self, query: Sequence[float] | np.ndarray, k: int) -> list[Slice]:
        """Return up to ``k`` hosted slices nearest to ``query`` (``[]`` if unconfigured).

        ``query`` is the 256-dim retrieval embedding (a list or numpy vector). With no
        ``base_url`` this is a clean no-op returning ``[]`` — the engine then relies solely on
        the local pool. A configured-but-failing call raises a typed
        :class:`~aether_context.errors.BackendUnavailable`; callers on the hot path should
        treat it as fail-soft (the session does).
        """
        if not self.configured:
            return []
        if k <= 0:
            return []
        vec = np.asarray(query, dtype=np.float32).reshape(-1).tolist()
        body = {"query": vec, "k": int(k)}
        resp = self._post(_RETRIEVE_PATH, body)
        return self._parse_slices(resp)

    # -- submit (one-way harvest push on close) -------------------------------
    def submit(self, candidates: Iterable[Any]) -> int:
        """Push harvest ``candidates`` to the hosted intake; return the count accepted.

        Each candidate is expected to expose ``.text``, ``.vector`` and ``.tags`` (the
        session's :class:`~aether_context.session.HarvestCandidate`). With no ``base_url``
        this is a no-op returning ``0``. The session calls this inside a fail-soft guard on
        close, so a hosted error never crashes the local run.
        """
        items = list(candidates)
        if not self.configured or not items:
            return 0
        payload = {"candidates": [self._candidate_to_json(c) for c in items]}
        self._post(_SUBMIT_PATH, payload)
        return len(items)

    # -- HTTP plumbing (stdlib only) ------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        """POST ``body`` as JSON to ``base_url + path``; return the parsed JSON response."""
        assert self.base_url is not None  # guarded by callers
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise BackendUnavailable(
                f"AetherCloud API returned HTTP {exc.code} from {url}",
                hint="Check the base_url, token, and that the hosted service is reachable; "
                "or omit atlas_client to run fully local.",
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise BackendUnavailable(
                f"Could not reach the AetherCloud API at {url}: {exc}",
                hint="Check connectivity / base_url; or omit atlas_client to run fully local.",
            ) from exc
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise BackendUnavailable(
                f"AetherCloud API returned a non-JSON response from {url}: {exc}",
                hint="The endpoint may be wrong; verify base_url points at the API root.",
            ) from exc
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _candidate_to_json(candidate: Any) -> dict:
        """Serialize a harvest candidate to a plain JSON dict (text + vector + tags)."""
        vector = getattr(candidate, "vector", None)
        vec_list = (
            np.asarray(vector, dtype=np.float32).reshape(-1).tolist()
            if vector is not None
            else []
        )
        return {
            "text": str(getattr(candidate, "text", "")),
            "vector": vec_list,
            "tags": dict(getattr(candidate, "tags", {}) or {}),
        }

    @staticmethod
    def _parse_slices(resp: dict) -> list[Slice]:
        """Parse the hosted retrieve response into local :class:`Slice` objects.

        Tolerant of a missing/empty ``slices`` list and of partial records — a malformed
        record is skipped (logged), never fatal, so a hosted format drift degrades softly.
        """
        out: list[Slice] = []
        raw_slices = resp.get("slices")
        if not isinstance(raw_slices, list):
            return out
        for rec in raw_slices:
            if not isinstance(rec, dict):
                continue
            vec = rec.get("vector")
            if vec is None:
                continue
            try:
                vector = np.asarray(vec, dtype=np.float32).reshape(-1)
                out.append(
                    Slice(
                        id=str(rec.get("id", "")),
                        session=str(rec.get("session", "")),
                        vector=vector,
                        text=str(rec.get("text", "")),
                        tokens=int(rec.get("tokens", 0)),
                        meta=dict(rec.get("meta", {}) or {}),
                        score=float(rec.get("score", 0.0)),
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.debug("skipping malformed hosted slice record: %s", exc)
                continue
        return out


__all__ = ["AtlasClient", "DEFAULT_TIMEOUT"]
