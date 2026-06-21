"""Project-specific, multi-tenant LLM policy for saddle.

The connection pool was lifted wholesale from a sibling project, but saddle
owns its OWN provider priority, per-provider caps, and pool sizing —
decoupled from that project's hardcoded values. Secrets are NOT duplicated:
API keys are read from a shared key file (the sibling's
``config/llm_config.json``) via the policy's ``keys_from`` pointer, while
everything that is POLICY (priority order, caps, memory ceilings) lives in
saddle's own ``config/llm_policy.json``.

Multi-tenant layering
---------------------
One saddle process serves many tenants and projects. Policy resolves in
three layers, each deep-merged onto the one below (nested dicts merge,
scalars/lists replace):

    base      config/llm_policy.json                              (shared default)
    tenant    config/tenants/<tenant>/policy.json                 (optional)
    project   config/tenants/<tenant>/projects/<project>/policy.json (optional)

So a tenant can override just one provider's cap, or a project can pin its
own ``priority`` / ``keys_from``, without restating the whole policy. The
overlay tree lives under ``SADDLE_CONFIG_DIR`` (default ``<repo>/config``).

Resolution order for the file locations:
  base policy  ← ``SADDLE_LLM_POLICY`` env, else ``<repo>/config/llm_policy.json``,
                 else ``<cwd>/config/llm_policy.json``
  shared keys  ← ``SADDLE_LLM_CONFIG`` env, else policy ``keys_from``,
                 else ``RAYXI_LLM_CONFIG`` env

Because ``keys_from`` is part of the layered policy, a tenant overlay that
points at its own key file gives that tenant key isolation for free.

``merged_config(ctx)`` overlays POLICY fields onto the SECRET key entries
and returns ``{"providers": {...}}`` — the exact shape the pool's caller
factories already consume, so no factory plumbing changes. Every accessor
takes an optional :class:`~saddle.context.Context`; omitting it resolves
the ambient (env + cwd) context.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from saddle.context import Context
from saddle.context import default as _default_ctx

_log = logging.getLogger("saddle.llm.policy")

# Saddle defaults — used when the policy file is absent or silent.
# claude_agent leads (the Agent SDK is saddle's primary), minimax is the
# funded fallback. Override entirely via config/llm_policy.json.
_DEFAULT_PRIORITY: list[str] = ["claude_agent", "minimax"]
_DEFAULT_POOL: dict[str, float] = {
    "max_procs": 24,
    "mem_ceiling_pct": 80.0,
    "mem_resume_pct": 70.0,
}


def _repo_root() -> Path:
    # src/saddle/llm/policy.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("could not read %s: %s", path, exc)
        return {}


def _config_dir() -> Path:
    """Root of the config tree (base policy + tenant overlays)."""
    env = os.environ.get("SADDLE_CONFIG_DIR")
    if env and env.strip():
        return Path(env).expanduser()
    return _repo_root() / "config"


def _policy_path() -> Path | None:
    candidates: list[Path] = []
    env = os.environ.get("SADDLE_LLM_POLICY")
    if env and env.strip():
        candidates.append(Path(env).expanduser())
    candidates.append(_config_dir() / "llm_policy.json")
    candidates.append(_repo_root() / "config" / "llm_policy.json")
    candidates.append(Path.cwd() / "config" / "llm_policy.json")
    for c in candidates:
        if c.exists():
            return c
    return None


@lru_cache(maxsize=1)
def _base_policy() -> dict:
    """Read + cache the BASE (tenant-agnostic) policy file."""
    path = _policy_path()
    if path is None:
        _log.warning("no llm_policy.json found — using saddle defaults")
        return {}
    return _read_json(path)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive merge: nested dicts merge, everything else replaces."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=128)
def _resolve_cached(tenant: str, project: str) -> dict:
    """Layered policy: base ← tenant overlay ← project overlay.

    Cached by the (already-slugified) tenant/project strings. Overlay
    files are optional; a missing file is simply skipped.
    """
    policy = dict(_base_policy())
    tenants = _config_dir() / "tenants"
    t_path = tenants / tenant / "policy.json"
    if t_path.exists():
        policy = _deep_merge(policy, _read_json(t_path))
    p_path = tenants / tenant / "projects" / project / "policy.json"
    if p_path.exists():
        policy = _deep_merge(policy, _read_json(p_path))
    return policy


def resolve_policy(ctx: Context | None = None) -> dict:
    """Return the fully-resolved policy for ``ctx`` (ambient ctx if None)."""
    c = ctx or _default_ctx()
    return _resolve_cached(c.tenant, c.project)


def load_policy(ctx: Context | None = None) -> dict:
    """Back-compat alias for :func:`resolve_policy`."""
    return resolve_policy(ctx)


def reset_policy_cache() -> None:
    """Drop cached policy layers — for tests / config hot-reload."""
    _base_policy.cache_clear()
    _resolve_cached.cache_clear()


def _shared_keys_path(policy: dict) -> Path | None:
    candidates: list[Path] = []
    env = os.environ.get("SADDLE_LLM_CONFIG")
    if env and env.strip():
        candidates.append(Path(env).expanduser())
    keys_from = policy.get("keys_from")
    if keys_from:
        candidates.append(Path(str(keys_from)).expanduser())
    rayxi = os.environ.get("RAYXI_LLM_CONFIG")
    if rayxi and rayxi.strip():
        candidates.append(Path(rayxi).expanduser())
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_shared_keys(policy: dict) -> dict:
    """Read the shared SECRET key file (api keys live here, not in saddle)."""
    path = _shared_keys_path(policy)
    if path is None:
        _log.warning("no shared key file resolved — keyless providers only")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("could not read shared keys %s: %s", path, exc)
        return {}


def active_priority(ctx: Context | None = None) -> list[str]:
    """Provider rotation, leftmost wins. Policy-driven, saddle-owned."""
    pr = resolve_policy(ctx).get("priority")
    if isinstance(pr, list) and pr:
        return [str(x) for x in pr]
    return list(_DEFAULT_PRIORITY)


def pool_settings(ctx: Context | None = None) -> dict:
    """Process-pool sizing + memory ceilings. Policy overlays the defaults."""
    settings = dict(_DEFAULT_POOL)
    overlay = resolve_policy(ctx).get("pool")
    if isinstance(overlay, dict):
        settings.update(overlay)
    return settings


def merged_config(ctx: Context | None = None) -> dict:
    """Return ``{"providers": {...}}`` — shared SECRET keys overlaid with
    saddle POLICY fields (caps, model) for ``ctx``. Same shape the old
    ``_load_config`` returned, so the caller factories are unchanged.

    ``claude_agent`` is keyless (subscription auth), so it always appears in
    the merged set even when the shared key file has no entry for it.
    """
    policy = resolve_policy(ctx)
    shared_providers = dict(_load_shared_keys(policy).get("providers", {}))
    policy_providers = policy.get("providers", {})
    if not isinstance(policy_providers, dict):
        policy_providers = {}

    names = (
        set(shared_providers)
        | set(policy_providers)
        | set(active_priority(ctx))
    )
    merged: dict[str, dict] = {}
    for name in names:
        entry = dict(shared_providers.get(name, {}))      # secrets first
        overlay = policy_providers.get(name, {})          # policy overlay
        if isinstance(overlay, dict):
            entry.update(overlay)
        merged[name] = entry
    return {"providers": merged}
