"""Loader for the global DKB seed corpus.

The seed is saddle's pre-researched, offline body of design wisdom — universal
best practices, anti-patterns, and principles that apply to every tenant. It
ships as data (``data/dkb_seed.json``), never hard-coded in logic, so it can be
extended offline by adding entries with new stable ids.

Loading is **idempotent**: each entry carries a stable id, and an entry already
present is skipped, so ``saddle kb seed`` can run any number of times (install,
upgrade, hot-reload) without duplicating the corpus. Seeded entries are global
scope and ``source=seed``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from saddle.dkb import DKB, get_dkb
from saddle.models import ACTIVE, SEED, Knowledge

_log = logging.getLogger("saddle.seed")


def seed_path() -> Path:
    """Path to the packaged seed corpus."""
    return Path(__file__).resolve().parent / "data" / "dkb_seed.json"


def load_seed_entries(path: Path | None = None) -> list[Knowledge]:
    """Parse the seed JSON into global :class:`Knowledge` rows (not yet stored)."""
    p = path or seed_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    entries: list[Knowledge] = []
    for e in data.get("entries", []):
        entries.append(
            Knowledge(
                kind=str(e["kind"]),
                title=str(e["title"]),
                body=str(e["body"]),
                tags=[str(t) for t in e.get("tags", [])],
                source=SEED,
                id=str(e["id"]),
            )
        )
    return entries


def seed_dkb(dkb: DKB | None = None, *, path: Path | None = None) -> dict:
    """Reconcile the seed corpus into the DKB: add what's new, retire what's gone.

    The corpus file is the *desired active seed set*; this brings the DKB into
    line with it. Every current entry is added if absent (and skipped if already
    present — keyed on its stable id, so re-running adds nothing). Any
    **superseded** seed entry — one still active in the DKB but no longer named
    by the corpus, e.g. an id renamed across versions (``kno_seed_x`` →
    ``knowledge_seed_x``) — is retired so it leaves the search indexes instead of
    lingering as a duplicate of its replacement. Retired, never deleted (the
    no-delete rule): the canonical row stays, auditable forever.

    Returns ``{"total", "added", "skipped", "retired"}``.
    """
    d = dkb or get_dkb()
    entries = load_seed_entries(path)
    current_ids = {k.id for k in entries}
    added = 0
    for k in entries:
        # Cheap existence pre-check first: it skips the expensive embed for the
        # common already-present case (every run after the first). The check and
        # the insert are not one atomic step, though, so two seeders racing on a
        # fresh db can both read "absent" and both insert — the loser hits a
        # PRIMARY KEY violation. Catch it: a concurrent seeder having inserted
        # this stable id IS the idempotent outcome this loader promises.
        if d.get_knowledge(k.id) is not None:
            continue
        try:
            d.add_knowledge(k)
            added += 1
        except sqlite3.IntegrityError:
            _log.debug("DKB seed: %s already inserted by a concurrent seeder", k.id)
    retired = _retire_superseded(d, current_ids)
    result = {
        "total": len(entries),
        "added": added,
        "skipped": len(entries) - added,
        "retired": retired,
    }
    _log.info(
        "DKB seed: %d entries (%d added, %d already present, %d superseded retired)",
        result["total"], result["added"], result["skipped"], result["retired"],
    )
    return result


def _retire_superseded(d: DKB, current_ids: set[str]) -> int:
    """Retire every active *seed* entry whose id the corpus no longer names.

    This is what makes an id rename safe: once the corpus drops the old id, the
    stale row is retired rather than left behind as a second copy of its
    replacement. Scope is deliberately narrow — only ``source=seed`` rows, so
    tenant/project knowledge harvested from audits is never touched — and the
    retire is conservative: an entry the corpus still names is never retired
    here, so a hand-retired seed stays retired (we don't resurrect it).

    Guarded against the degenerate empty corpus: saddle always ships a non-empty
    seed, so an empty ``current_ids`` can only mean the corpus failed to load,
    and retiring on that bad read would wipe the whole active corpus. Retire
    nothing in that case — refusing to act on a degenerate desired-state, not
    papering over it.
    """
    if not current_ids:
        return 0
    retired = 0
    for k in d.list_knowledge(sources=[SEED], status=ACTIVE, limit=100_000):
        if k.id not in current_ids and d.retire_knowledge(k.id):
            retired += 1
    return retired
