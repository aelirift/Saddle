"""Standalone embedding server — runs under SYSTEM python3, not saddle's venv.

saddle runs inside rayxiv4's venv, which has neither ``torch`` nor
``sentence_transformers``. Those live in the host's system ``python3``. Rather
than drag a multi-hundred-MB ML stack into saddle's runtime, the embedder is a
*separate process*: this tiny HTTP server is launched (by ``saddle.embed``)
under ``/usr/bin/python3``, loads the model once, and answers ``/embed`` over
loopback. saddle stays torch-free and talks to it with plain ``httpx``.

Hard constraints (do not break these):
  * stdlib + ``sentence_transformers`` ONLY. No ``httpx``, no ``saddle`` imports
    — this file is imported by an interpreter that has none of them.
  * Launched BY FILE PATH (``python3 .../embed_server.py``), never as a saddle
    module, so it must be runnable with ``saddle`` absent from ``sys.path``.

Endpoints:
  GET  /health  -> {"status": "ok", "model": "...", "dim": N}
  POST /embed   -> body {"texts": [...]}  ->  {"model","dim","vectors":[[...]]}

The model loads lazily on first need; ``/health`` reports ``loading`` until the
weights are resident, which is how the client knows when the server is ready.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Force fully-offline, deterministic single-threaded loads BEFORE the ML stack
# is imported — the model is already cached on disk; never hit the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_log = logging.getLogger("saddle.embed_server")

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# --- Model singleton: loaded once, guarded so concurrent requests don't race --
_model = None
_model_name = ""
_dim = 0
_load_lock = threading.Lock()


def _load(model_name: str):
    """Load (once) and return the SentenceTransformer; stamp ``_dim``."""
    global _model, _model_name, _dim
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:  # someone else won the race
            return _model
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        _dim = int(model.get_sentence_embedding_dimension())
        _model_name = model_name
        _model = model
        _log.info("loaded %s (dim=%d)", model_name, _dim)
        return _model


def _encode(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` -> unit-normalized float vectors (cosine == dot)."""
    model = _load(_model_name or _DEFAULT_MODEL)
    vecs = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [[float(x) for x in row] for row in vecs]


class _Handler(BaseHTTPRequestHandler):
    # Quiet the default per-request stderr spam; route through logging instead.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        _log.debug(fmt, *args)

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — stdlib dispatch name
        if self.path.rstrip("/") in ("/health", ""):
            ready = _model is not None
            self._send(
                200,
                {
                    "status": "ok" if ready else "loading",
                    "model": _model_name or _DEFAULT_MODEL,
                    "dim": _dim,
                },
            )
            return
        self._send(404, {"error": f"no such path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 — stdlib dispatch name
        if self.path.rstrip("/") != "/embed":
            self._send(404, {"error": f"no such path: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw or b"{}")
            texts = req.get("texts")
            if not isinstance(texts, list) or not all(
                isinstance(t, str) for t in texts
            ):
                self._send(400, {"error": "body must be {'texts': [str, ...]}"})
                return
            vectors = _encode(texts) if texts else []
            self._send(
                200,
                {"model": _model_name or _DEFAULT_MODEL, "dim": _dim, "vectors": vectors},
            )
        except Exception as exc:  # noqa: BLE001 — return the error, never crash the loop
            _log.exception("embed failed")
            self._send(500, {"error": str(exc)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="saddle embedding server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8631)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--preload", action="store_true", help="load model before serving")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global _model_name
    _model_name = args.model
    if args.preload:
        _load(args.model)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    _log.info("serving on http://%s:%d (model=%s)", args.host, args.port, args.model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover — manual stop
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
