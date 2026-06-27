"""Cross-project authorization grants for the doctrine scope-fence.

The doctrine ``stay-in-project-focus`` rule blocks any edit / write / delete
whose target lives outside the focus project, unless the action carries
``cross_project_task=true`` evidence. That evidence is reachable from the
``saddle guard`` CLI, but NOT from a tool call:
:func:`saddle.doctrine.actions_from_tool` builds the Action with no evidence,
so an agent operating through Edit / Write / Bash has no channel to present a
legitimate cross-project authorization. The fence's own message says the work
is allowed when the task is "explicitly cross-project" -- but nothing lets the
tool path *say so*. This module is that missing channel.

A *grant* is an explicit, persistent, auditable record that the user authorized
work spanning a named set of project roots. The doctrine hook consults grants
ONLY when the scope-fence would otherwise block: if the focus root and the
blocked target both fall under granted roots for the active tenant, the
cross-project move is authorized and allowed -- with a NOTICE on stderr, never
silently. A target under no granted root still blocks, so the fence keeps
protecting against wandering into an unrelated repo. Other rules
(``no-unwired-delete``, ``disposition-coherent``) are not scope concerns and
are never overridden here.

Grants live in ``<saddle-data-dir>/cross_project.json`` (the dir that holds the
saddle DB), so they share the install's isolation. Each grant carries its
authorized roots, the tenant it applies to, a reason, and a timestamp -- a
reviewer sees exactly what was authorized, for whom, and why. Grants are
additive and revocable; they never loosen any rule other than the focus fence.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _grants_path() -> Path:
    """Where grants are parked -- beside the saddle DB so they share the
    install's data dir and isolation."""
    from saddle.store import default_db_path
    return default_db_path().parent / "cross_project.json"


@dataclass(frozen=True)
class Grant:
    """One explicit authorization spanning a set of project roots."""

    roots: tuple[str, ...]
    tenant: str = ""          # "" / "*" -> applies to every tenant
    reason: str = ""
    source: str = "cli"
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "roots": list(self.roots),
            "tenant": self.tenant,
            "reason": self.reason,
            "source": self.source,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Grant":
        return cls(
            roots=tuple(str(r) for r in d.get("roots", ())),
            tenant=str(d.get("tenant", "")),
            reason=str(d.get("reason", "")),
            source=str(d.get("source", "cli")),
            ts=float(d.get("ts", 0.0)),
        )


def _resolve(p: str) -> str:
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:  # noqa: BLE001
        return str(p)


def load_grants() -> list[Grant]:
    try:
        data = json.loads(_grants_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[Grant] = []
    for d in data.get("grants", []):
        try:
            out.append(Grant.from_dict(d))
        except Exception:  # noqa: BLE001 -- one bad record must not nuke the rest
            continue
    return out


def save_grants(grants: Iterable[Grant]) -> None:
    p = _grants_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"grants": [g.to_dict() for g in grants]}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def grant(roots: Iterable[str], *, tenant: str = "", reason: str = "",
          source: str = "cli") -> Grant:
    g = Grant(
        roots=tuple(_resolve(r) for r in roots),
        tenant=tenant, reason=reason, source=source, ts=time.time(),
    )
    save_grants([*load_grants(), g])
    return g


def revoke_all() -> int:
    n = len(load_grants())
    save_grants([])
    return n


def _tenant_matches(g: Grant, tenant: str) -> bool:
    return g.tenant in ("", "*") or g.tenant == tenant


def authorized_roots(tenant: str) -> list[str]:
    roots: list[str] = []
    for g in load_grants():
        if _tenant_matches(g, tenant):
            roots.extend(g.roots)
    return roots


def _under(root: str, target: str) -> bool:
    try:
        r = Path(root).expanduser().resolve()
        t = Path(target).expanduser()
        if not t.is_absolute():
            t = r / t
        t = t.resolve()
        return t == r or r in t.parents
    except Exception:  # noqa: BLE001
        return False


def is_authorized(target: str, focus: str, *, tenant: str) -> bool:
    """A cross-project target is authorized iff BOTH the focus root and the
    target fall under granted roots for ``tenant`` -- i.e. the move is between
    explicitly-authorized projects, not a wander into an ungranted one."""
    roots = authorized_roots(tenant)
    if not roots:
        return False
    focus_ok = any(_under(r, focus) for r in roots)
    target_ok = any(_under(r, target) for r in roots)
    return focus_ok and target_ok


def authorize_tool(tool_name: str, tool_input, *, focus: str,
                   tenant: str | None = None) -> str | None:
    """If every out-of-focus target of this tool call is cross-project
    authorized, return a human notice string; else ``None``. The doctrine hook
    calls this only after the scope-fence has blocked, so ``None`` means the
    block stands."""
    from saddle.doctrine import actions_from_tool
    if tenant is None:
        try:
            from saddle.context import resolve
            tenant = resolve(None, None).tenant
        except Exception:  # noqa: BLE001
            tenant = ""
    actions = actions_from_tool(tool_name, tool_input, project_root=focus)
    targets = [a.target for a in actions]
    if not targets:
        return None
    for t in targets:
        if _under(focus, t):
            continue  # in-focus targets were never the fence's concern
        if not is_authorized(t, focus, tenant=tenant):
            return None
    return f"cross-project authorized by grant (tenant={tenant!r}): {', '.join(targets)}"
