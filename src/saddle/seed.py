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
from pathlib import Path

from saddle.dkb import DKB, get_dkb
from saddle.models import SEED, Knowledge

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
    """Upsert the seed corpus into the DKB, skipping entries already present.

    Returns ``{"total", "added", "skipped"}``.
    """
    d = dkb or get_dkb()
    entries = load_seed_entries(path)
    added = 0
    for k in entries:
        if d.get_knowledge(k.id) is None:
            d.add_knowledge(k)
            added += 1
    result = {"total": len(entries), "added": added, "skipped": len(entries) - added}
    _log.info(
        "DKB seed: %d entries (%d added, %d already present)",
        result["total"], result["added"], result["skipped"],
    )
    return result
