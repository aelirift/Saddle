"""Tenant + project context — saddle's multi-tenant addressing handle.

Saddle is a shared harness: one process serves many *tenants* (isolated
owners — a person, org, or workspace) and many *projects* per tenant.
Everything that touches per-owner state — LLM routing policy, the
intake/todo store, fairness quotas — is keyed by a :class:`Context`.

A Context is resolved ONCE at the edge (a CLI flag, a future service
request) and threaded through the call. Resolution order per field:

    tenant   explicit arg -> $SADDLE_TENANT -> OS user -> "default"
    project  explicit arg -> $SADDLE_PROJECT -> git repo name -> cwd name

Both fields are slugified to ``[a-z0-9_-]`` because they become
filesystem path segments (``config/tenants/<tenant>/...``) and database
filter keys — an unsanitised value would be a path-traversal / injection
vector. The dataclass is frozen + slugified in ``__post_init__`` so every
Context in the system is already safe however it was constructed, and is
hashable so it can key an ``lru_cache``.
"""

from __future__ import annotations

import getpass
import os
import re
from dataclasses import dataclass
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_DEFAULT_TENANT = "default"
_DEFAULT_PROJECT = "default"


def _slug(value: str, *, fallback: str) -> str:
    """Lowercase, collapse unsafe runs to '-', trim. Fallback if empty."""
    s = _SLUG_RE.sub("-", (value or "").strip().lower()).strip("-")
    return s or fallback


def _git_root(start: Path) -> Path | None:
    """Nearest enclosing git repo, bounded at ``$HOME``.

    The walk stops before testing ``$HOME`` itself so a home-level repo
    (dotfiles, a personal monorepo root) is never mistaken for "the
    project" when the actual working dir isn't its own repo. Callers
    fall back to the cwd basename in that case.
    """
    cur = start.resolve()
    try:
        home = Path.home().resolve()
    except Exception:  # noqa: BLE001 — no home (rare); no ceiling
        home = None
    for parent in (cur, *cur.parents):
        if home is not None and parent == home:
            break
        if (parent / ".git").exists():
            return parent
    return None


@dataclass(frozen=True)
class Context:
    """Immutable ``(tenant, project)`` addressing handle.

    Frozen + slugified -> hashable + always safe as a path/DB key.
    """

    tenant: str
    project: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tenant", _slug(self.tenant, fallback=_DEFAULT_TENANT)
        )
        object.__setattr__(
            self, "project", _slug(self.project, fallback=_DEFAULT_PROJECT)
        )

    @property
    def key(self) -> str:
        """Stable ``tenant/project`` string for labels + log lines."""
        return f"{self.tenant}/{self.project}"


def _default_tenant() -> str:
    env = os.environ.get("SADDLE_TENANT")
    if env and env.strip():
        return env
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 — no user context (rare); fall through
        user = ""
    return user or _DEFAULT_TENANT


def _default_project(cwd: Path | None = None) -> str:
    env = os.environ.get("SADDLE_PROJECT")
    if env and env.strip():
        return env
    base = (cwd or Path.cwd()).resolve()
    root = _git_root(base)
    return (root or base).name


def resolve(
    tenant: str | None = None,
    project: str | None = None,
    *,
    cwd: str | Path | None = None,
) -> Context:
    """Resolve a Context from explicit args, env, and the working dir.

    Empty/whitespace args are treated as "not given" and fall through to
    the env/host defaults.
    """
    cwd_path = Path(cwd) if cwd is not None else None
    t = tenant if (tenant and tenant.strip()) else _default_tenant()
    p = project if (project and project.strip()) else _default_project(cwd_path)
    return Context(tenant=t, project=p)


def default() -> Context:
    """The ambient Context (env + cwd) used where no ctx is passed in."""
    return resolve()
