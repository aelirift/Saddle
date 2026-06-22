"""Embedding client — saddle's torch-free handle on the local embed server.

saddle's process (running inside rayxiv4's venv) has ``httpx`` but no ``torch``.
The embedding model lives in the host's system ``python3``. This module is the
bridge: it lazily *auto-starts* :mod:`saddle.embed_server` under that system
interpreter, waits for the model to load, and then embeds text over loopback —
so callers just ask for vectors and never think about the process boundary.

Design choices that matter:
  * **Auto-start, reuse-if-present.** The first ``embed`` call probes the
    configured port; if a server already answers (this process started one
    earlier, or another saddle process / a manual launch did) it is reused, so
    the heavy model is loaded at most once per host, not once per process.
  * **Detached child.** The server is spawned with ``start_new_session=True``
    and its logs tee to ``~/.saddle/embed_server.log`` — it survives the
    caller and a second saddle process simply rebinds-fails and reuses it.
  * **No hard-coding.** Host / port / model / interpreter all come from env
    (``SADDLE_EMBED_*``) with sane defaults, so a tenant can repoint the model
    without touching code.

The vector dimension is *discovered* from the running model (``/health``),
never assumed — the DKB sizes its vec table from :pyattr:`Embedder.dim`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

_log = logging.getLogger("saddle.embed")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8631
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_START_TIMEOUT = 90.0  # first model load is the slow part


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    return v.strip() if v and v.strip() else default


def _embed_python() -> str:
    """Interpreter that has torch + sentence_transformers (NOT saddle's venv)."""
    env = os.environ.get("SADDLE_EMBED_PYTHON")
    if env and env.strip():
        return env.strip()
    system = Path("/usr/bin/python3")
    if system.exists():
        return str(system)
    return "python3"


def _log_path() -> Path:
    return Path(os.path.expanduser("~/.saddle")) / "embed_server.log"


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into unit vectors. Lets tests swap a fake."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HttpEmbedder:
    """Embedder backed by the local :mod:`saddle.embed_server` subprocess."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        model: str | None = None,
        *,
        start_timeout: float | None = None,
    ) -> None:
        self._host = host or _env("SADDLE_EMBED_HOST", _DEFAULT_HOST)
        self._port = int(port or _env("SADDLE_EMBED_PORT", str(_DEFAULT_PORT)))
        self._model = model or _env("SADDLE_EMBED_MODEL", _DEFAULT_MODEL)
        self._start_timeout = float(
            start_timeout
            if start_timeout is not None
            else _env("SADDLE_EMBED_START_TIMEOUT", str(_DEFAULT_START_TIMEOUT))
        )
        self._base = f"http://{self._host}:{self._port}"
        self._client = httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._dim = 0
        self._ready = False
        self._lock = threading.Lock()

    # --- readiness / lifecycle ------------------------------------------
    def _probe(self) -> dict | None:
        """Return the /health dict, or None if nothing is listening yet."""
        try:
            r = self._client.get(f"{self._base}/health", timeout=2.0)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            return None
        if r.status_code == 200:
            return r.json()
        return None

    def _spawn(self) -> None:
        server = Path(__file__).resolve().parent / "embed_server.py"
        lp = _log_path()
        lp.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(lp, "a", encoding="utf-8")
        cmd = [
            _embed_python(), str(server),
            "--host", self._host,
            "--port", str(self._port),
            "--model", self._model,
            "--preload",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ),
        )
        _log.info(
            "spawned embed server pid=%s on %s (model=%s, log=%s)",
            self._proc.pid, self._base, self._model, lp,
        )

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            if self._probe() is None:
                self._spawn()
            deadline = time.time() + self._start_timeout
            while time.time() < deadline:
                health = self._probe()
                if health and health.get("status") == "ok":
                    self._dim = int(health.get("dim") or 0)
                    self._ready = True
                    _log.info("embed server ready (dim=%d)", self._dim)
                    return
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"embed server exited rc={self._proc.returncode}; "
                        f"see {_log_path()}"
                    )
                time.sleep(0.5)
            raise TimeoutError(
                f"embed server not ready after {self._start_timeout:.0f}s "
                f"(see {_log_path()})"
            )

    # --- public API ------------------------------------------------------
    @property
    def dim(self) -> int:
        self._ensure_ready()
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        self._ensure_ready()
        r = self._client.post(f"{self._base}/embed", json={"texts": items})
        r.raise_for_status()
        data = r.json()
        if data.get("dim"):
            self._dim = int(data["dim"])
        return data["vectors"]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# --- process-global singleton -------------------------------------------
_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Embedder:
    """Return the shared embedder, constructing the default on first use."""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = HttpEmbedder()
    return _EMBEDDER


def set_embedder(embedder: Embedder | None) -> None:
    """Install an embedder (e.g. a fake in tests). None clears it."""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        _EMBEDDER = embedder


def reset_embedder() -> None:
    """Drop the cached embedder — next :func:`get_embedder` rebuilds it."""
    set_embedder(None)
