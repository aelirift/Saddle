"""MiniMax image generation client.

Reuses the same API key as the text caller (`providers.minimax.api_key`) but
calls a different endpoint. MiniMax's image-gen API is at
`https://api.minimaxi.chat/v1/image_generation`.

Usage:
    caller = MiniMaxImageCaller()
    png_bytes = await caller.generate(
        prompt="anime-style fighter in idle stance",
        aspect_ratio="3:4",
    )
    Path("out.png").write_bytes(png_bytes)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

import httpx

_log = logging.getLogger("saddle.llm.image_gen")

_IMAGE_API_URL = "https://api.minimaxi.chat/v1/image_generation"
_DEFAULT_MODEL = "image-01"


def _config_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("RAYXI_LLM_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repo_root / "config" / "llm_config.json",
            Path.cwd() / "config" / "llm_config.json",
            Path.home() / ".config" / "rayxi" / "llm_config.json",
            # Removed: hardcoded cross-project path leaked local filesystem.
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        deduped.append(candidate)
        seen.add(key)
    return deduped


def _resolve_config_path() -> Path | None:
    for candidate in _config_candidates():
        if candidate.exists():
            return candidate
    return None


def _load_api_key() -> str:
    env_key = os.environ.get("MINIMAX_API_KEY")
    if env_key:
        return env_key
    config_path = _resolve_config_path()
    if config_path is None:
        raise FileNotFoundError(
            "No MiniMax image config found. Set MINIMAX_API_KEY or RAYXI_LLM_CONFIG to enable image generation."
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return cfg["providers"]["minimax"]["api_key"]


class MiniMaxImageCaller:
    """Thin async client over MiniMax /v1/image_generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout_seconds: float = 180.0,
    ) -> None:
        self._api_key = api_key or _load_api_key()
        self._model = model
        self._timeout = timeout_seconds

    async def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "1:1",
        n: int = 1,
        response_format: str = "base64",
        label: str = "",
    ) -> bytes:
        """Generate an image and return raw PNG bytes.

        Args:
            prompt: the image description (include "transparent background" when needed)
            aspect_ratio: "1:1" | "3:4" | "4:3" | "9:16" | "16:9"
            n: number of images to generate (returns the first)
            response_format: "base64" or "url"
            label: optional log tag
        """
        body = {
            "model": self._model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "response_format": response_format,
            "n": n,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        tag = label or prompt[:40]
        _log.info("MiniMax image: generating '%s' (%s)", tag, aspect_ratio)

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(_IMAGE_API_URL, headers=headers, json=body)
                if resp.status_code != 200:
                    err = resp.text[:400]
                    raise RuntimeError(f"MiniMax image {resp.status_code}: {err}")
                data = resp.json()
                # _extract_png may do a blocking HTTP fetch for url-format
                # responses; run in thread pool to avoid blocking the loop.
                return await asyncio.to_thread(self._extract_png, data, response_format)
            except Exception as exc:
                last_err = exc
                _log.warning("MiniMax image attempt %d failed: %s", attempt + 1, str(exc)[:160])
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"MiniMax image failed after 3 attempts: {last_err}")

    async def generate_many(
        self,
        specs: list[tuple[str, str, str]],
        *,
        aspect_ratio: str = "1:1",
        concurrency: int = 4,
    ) -> dict[str, bytes]:
        """Generate multiple images in parallel.

        Each individual image call is routed through the LLMPool's
        "image_gen" intent — the pool's per-(provider, intent) router
        + ProcessPool memory pacing apply. The semaphore here only
        bounds in-flight per-batch concurrency so a 50-image batch
        doesn't fan out 50 concurrent HTTP calls.

        Args:
            specs: list of (label, prompt, aspect_ratio_override or "") tuples
            aspect_ratio: default ratio if spec has empty override
            concurrency: max parallel requests
        """
        from saddle.llm.llm_pool import get_llm_pool

        pool = get_llm_pool()
        sem = asyncio.Semaphore(concurrency)
        results: dict[str, bytes] = {}

        async def _do(label: str, prompt: str, ratio_override: str) -> None:
            async with sem:
                try:
                    png = await pool.call(
                        intent="image_gen",
                        prompt=prompt,
                        aspect_ratio=ratio_override or aspect_ratio,
                        label=label,
                    )
                    results[label] = png
                    _log.info("MiniMax image: %s → %d bytes", label, len(png))
                except Exception as exc:
                    _log.error("MiniMax image: %s FAILED: %s", label, exc)

        await asyncio.gather(*[_do(l, p, r) for l, p, r in specs])
        return results

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _extract_png(self, data: dict, response_format: str) -> bytes:
        """Pull PNG bytes out of the JSON response, regardless of shape variant."""
        # MiniMax returns either `data.image_base64` (list) or `data` (list of urls)
        if response_format == "base64":
            # Try common shapes
            if "data" in data:
                d = data["data"]
                if isinstance(d, dict) and "image_base64" in d:
                    lst = d["image_base64"]
                    if isinstance(lst, list) and lst:
                        return base64.b64decode(lst[0])
                    if isinstance(lst, str):
                        return base64.b64decode(lst)
                if isinstance(d, list) and d:
                    first = d[0]
                    if isinstance(first, dict):
                        b64 = first.get("b64_json") or first.get("image_base64")
                        if b64:
                            return base64.b64decode(b64)
            # Some variants put it top-level
            if "image_base64" in data:
                b64 = data["image_base64"]
                if isinstance(b64, list) and b64:
                    return base64.b64decode(b64[0])
                return base64.b64decode(b64)
        elif response_format == "url":
            if "data" in data and isinstance(data["data"], dict) and "image_urls" in data["data"]:
                urls = data["data"]["image_urls"]
                if isinstance(urls, list) and urls:
                    # Sync httpx.get — caller is async, so this should be
                    # wrapped in asyncio.to_thread by the caller. Kept sync
                    # here because _extract_png is a sync method used in
                    # both sync/async contexts. 60s timeout bounds the block.
                    import httpx as _h
                    return _h.get(urls[0], timeout=60).content
        raise RuntimeError(f"MiniMax image: cannot extract PNG from response: {json.dumps(data)[:400]}")
