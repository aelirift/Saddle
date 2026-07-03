"""Project registry — the projects saddle has seen, learned from evidence.

Mediator design §4 (docs/design/mediator_front_end.md): saddle attaches to the
AGENT, not to one project, so it must know WHICH project each piece of work
belongs to — switching when the user switches, and holding several at once
when one prompt spans two projects. That routing needs an authority for "what
projects exist and where their code lives": this registry.

Entries are LEARNED, not configured: the first time a (tenant, root) shows up
in evidence — the hook's working directory, a file path an edit touches — the
project is recorded with ``confirmed=False``. Confirmation is a front-plane
courtesy (the user acknowledges the new project once) so a typo'd path never
silently becomes a phantom project; an unconfirmed entry still routes.

Storage: ``$SADDLE_HOME/projects.json`` — small, atomic-replace on write (the
same discipline as the design-gate markers), readable by every hook without a
database round-trip.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from saddle.context import _slug  # the one slugifier — ids must match Context's


def _registry_path() -> Path:
    from saddle.store import default_db_path

    return default_db_path().parent / "projects.json"


def _load() -> dict:
    try:
        doc = json.loads(_registry_path().read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(doc: dict) -> None:
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
        os.replace(tmp, p)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def register_root(tenant: str, root: str | Path, *, ts: float) -> str:
    """Record that ``root`` is a project of ``tenant``; return its slug.

    Idempotent: an already-known root only refreshes ``last_seen``. The slug is
    the root's directory name run through the SAME slugifier Context uses, so a
    registry slug always equals the ``ctx.project`` an agent working in that
    root resolves to."""
    r = Path(root).expanduser().resolve()
    slug = _slug(r.name, fallback="default")
    doc = _load()
    projects = doc.setdefault(tenant, {})
    entry = projects.get(slug)
    if entry is None:
        projects[slug] = {
            "root": str(r), "first_seen": ts, "last_seen": ts, "confirmed": False,
        }
    else:
        entry["last_seen"] = ts
        # A moved project (same name, new root) follows the newest evidence —
        # the old root no longer exists as this project's home.
        if entry.get("root") != str(r) and not Path(entry.get("root", "")).exists():
            entry["root"] = str(r)
    _save(doc)
    return slug


def known_projects(tenant: str) -> dict[str, str]:
    """slug -> root for every project this tenant has been seen working in."""
    return {
        slug: str(e.get("root", ""))
        for slug, e in _load().get(tenant, {}).items()
        if e.get("root")
    }


def confirm(tenant: str, slug: str) -> bool:
    """Mark a learned project as user-acknowledged. True if it existed."""
    doc = _load()
    entry = doc.get(tenant, {}).get(slug)
    if entry is None:
        return False
    entry["confirmed"] = True
    _save(doc)
    return True


def unconfirmed(tenant: str) -> list[str]:
    """Slugs recorded from evidence but never user-acknowledged — the front
    plane surfaces each ONCE so a phantom path can be corrected early."""
    return [
        slug for slug, e in _load().get(tenant, {}).items()
        if not e.get("confirmed")
    ]


def project_for_path(tenant: str, path: str | Path) -> str | None:
    """Which known project contains ``path``? Longest-root match wins (a
    nested checkout resolves to the nested project, not its parent). ``None``
    when no known root contains it."""
    try:
        t = Path(path).expanduser().resolve()
    except OSError:
        return None
    best: tuple[int, str] | None = None
    for slug, root in known_projects(tenant).items():
        try:
            r = Path(root).resolve()
        except OSError:
            continue
        if t == r or r in t.parents:
            score = len(r.parts)
            if best is None or score > best[0]:
                best = (score, slug)
    return best[1] if best else None


def descriptor_block(tenant: str, *, exclude: str = "") -> str:
    """One line per known project (name + root) for an LLM prompt that must
    route work to the right project — concrete identities, never a guessed
    vocabulary. ``exclude`` drops the ambient project (already described by
    the focus descriptor). Empty string when nothing else is known."""
    rows = [
        f'- "{slug}" — its code is rooted at {root}'
        for slug, root in sorted(known_projects(tenant).items())
        if slug != exclude
    ]
    return "\n".join(rows)
