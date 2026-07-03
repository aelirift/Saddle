"""Single source of truth for *what project is in focus*.

saddle observes ONE project at a time: the ``(tenant, project)`` it is bound
to, whose code lives under :func:`saddle.context.code_root`. More than one
layer has to agree on that boundary —

  * **Layer 0 — doctrine guard** fences file actions to paths *inside* the
    focus root (the ``stay-in-project-focus`` rule).
  * **Layer 1 — intake** flags asks that target work *outside* the focus
    project, so a "go improve <other project>" prompt is surfaced rather than
    silently itemized as in-bounds work.

If each layer computed "the focus" its own way they could disagree — the guard
fencing one root while intake itemizes against another. This module is the ONE
place the focus identity is defined so the layers cannot drift apart. Any
future layer that needs to reason about scope imports it from here too.

Two flavours of "in focus":

* **Path focus** — *is this filesystem target inside the focus repo?* — is
  deterministic and exact, and lives where it is used (the doctrine fence's own
  containment check). This module owns the *root* that check fences against.
* **Intent focus** — *does this natural-language ask belong to the focus
  project or to a different one?* — is a judgment the intake LLM makes. This
  module hands it the focus IDENTITY (project name + root) to judge against, via
  :func:`focus_descriptor`. It deliberately does NOT enumerate sibling project
  names: "in focus" is judged against THIS project, and anything about another
  codebase is out — without baking today's repo list into the code.
"""

from __future__ import annotations

from pathlib import Path

from saddle.context import Context, code_root
from saddle.context import default as _default_ctx


def focus_root(cwd: str | Path | None = None) -> Path:
    """The focus project's code root — the SAME path Layer 3 parses and the
    doctrine fence guards. Resolved by :func:`saddle.context.code_root`
    (``$SADDLE_CODE_ROOT`` -> enclosing git repo -> cwd)."""
    return code_root(cwd)


def focus_project(ctx: Context | None = None) -> str:
    """The focus project's name — the ``project`` half of the ``(tenant,
    project)`` isolation key. Falls back to the ambient context."""
    return (ctx or _default_ctx()).project


def focus_descriptor(ctx: Context | None = None) -> str:
    """A one-line description of the focus to inject into an LLM prompt: the
    project name and its root path. Carries the focus IDENTITY without listing
    sibling projects, so the model judges "belongs to this project vs. another"
    against a concrete target rather than a hardcoded vocabulary."""
    ctx = ctx or _default_ctx()
    return f'"{ctx.project}", the project whose code is rooted at {focus_root()}'


# === The turn's ACTIVE SCOPE SET (mediator design §4) ========================
#
# One focus is the DEFAULT, not the LIMIT: a prompt that spans two projects
# ("audit rayxiv4 AND check saddle") makes both active for the turn. Intake
# records the set after per-item scope assignment; the doctrine fence and the
# turn-end stages read it back, so all layers again agree on one boundary —
# now a set instead of a single root. The marker is per-session and
# atomically replaced (same discipline as the design-gate markers).

import json as _json
import os as _os
import tempfile as _tempfile


def _scopes_path(session: str) -> Path:
    from saddle.store import default_db_path

    safe = "".join(
        c if (c.isalnum() or c in "-_") else "_" for c in session
    ) or "default"
    return default_db_path().parent / "scopes" / f"{safe}.json"


def record_active_scopes(
    session: str,
    tenant: str,
    projects: list[str],
    asks: dict[str, list[str]] | None = None,
) -> None:
    """Persist the turn's active project slugs for ``session`` (optionally with
    the asks routed to each sibling — the material the per-project drift check
    weighs). Best-effort by contract: an IO failure is logged by the caller's
    stderr, never raised — at worst the fence falls back to single-focus
    behavior (stricter, never looser)."""
    p = _scopes_path(session)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = _tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    with _os.fdopen(fd, "w", encoding="utf-8") as fh:
        _json.dump({
            "tenant": tenant,
            "projects": sorted(set(projects)),
            "asks": {k: list(v) for k, v in (asks or {}).items()},
        }, fh)
    _os.replace(tmp, p)


def _read_scopes_doc(session: str, tenant: str) -> dict:
    try:
        doc = _json.loads(_scopes_path(session).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict) or doc.get("tenant") != tenant:
        return {}  # tenant mismatch = absent — the hard fence between owners
    return doc


def active_scopes(session: str, tenant: str) -> list[str]:
    """The project slugs active for ``session`` (empty when never recorded or
    recorded for a different tenant)."""
    doc = _read_scopes_doc(session, tenant)
    return [str(s) for s in doc.get("projects", []) if str(s).strip()]


def active_scope_asks(session: str, tenant: str) -> dict[str, list[str]]:
    """Per-sibling asks recorded with the turn's scope set: slug -> the asks
    the intake routed to that project. Empty when none were recorded."""
    doc = _read_scopes_doc(session, tenant)
    asks = doc.get("asks")
    if not isinstance(asks, dict):
        return {}
    return {
        str(k): [str(a) for a in v]
        for k, v in asks.items()
        if isinstance(v, list) and v
    }


def active_roots(session: str, ctx: Context | None = None) -> list[str]:
    """Every code root inside the turn's scope set: the focus root FIRST (the
    ambient project is always in scope), then the registry root of every other
    active project. The doctrine fence's "inside" is containment in ANY of
    these."""
    from saddle import registry

    ctx = ctx or _default_ctx()
    roots = [str(focus_root())]
    if session:
        known = registry.known_projects(ctx.tenant)
        for slug in active_scopes(session, ctx.tenant):
            root = known.get(slug)
            if root and root not in roots:
                roots.append(root)
    return roots
