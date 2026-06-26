"""DKB seed corpus + its loader.

The seed is saddle's offline body of design wisdom, shipped as data
(``data/dkb_seed.json``) and loaded idempotently, keyed by each entry's stable
id. These tests pin two things the corpus depends on — that every entry is
well-formed and self-describing, and that re-seeding never duplicates — without
an embed server or vector backend, since the loader only parses JSON and dedups
on id.
"""
from __future__ import annotations

import sqlite3

from saddle.models import ACTIVE, KNOWLEDGE_KINDS, PRINCIPLE, RETIRED, SEED, Knowledge
from saddle.seed import load_seed_entries, seed_dkb


def test_seed_corpus_is_wellformed_and_self_describing():
    """Every packaged entry parses, carries a self-describing ``knowledge_``
    id (no leftover abbreviation), a known kind, and the seed source — and the
    ids are unique, which is what makes dedup-by-id sound."""
    entries = load_seed_entries()
    assert entries, "seed corpus must not be empty"
    ids = [k.id for k in entries]
    assert len(set(ids)) == len(ids), "seed ids must be unique (the dedup key)"
    for k in entries:
        assert k.id.startswith("knowledge_seed_"), k.id
        assert k.kind in KNOWLEDGE_KINDS, (k.id, k.kind)
        assert k.source == SEED
        assert k.title and k.body, k.id


class _FakeDKB:
    """Minimal stand-in exposing only what ``seed_dkb`` touches — id-keyed
    get/add plus the reconcile pair (list/retire) — so the loader's idempotency
    AND its retire-superseded path are tested without the real vector backend or
    embed server. ``add_knowledge`` raises on a duplicate id, mirroring the
    SQLite PRIMARY KEY the real DKB relies on; ``retire_knowledge`` flips status
    to RETIRED and keeps the row, mirroring the no-delete rule."""

    def __init__(self) -> None:
        self.rows: dict[str, Knowledge] = {}

    def get_knowledge(self, kid):
        return self.rows.get(kid)

    def add_knowledge(self, k):
        if k.id in self.rows:
            raise sqlite3.IntegrityError(f"duplicate id {k.id}")
        self.rows[k.id] = k
        return k

    def list_knowledge(self, ctx=None, *, kinds=None, sources=None,
                        status=ACTIVE, limit=200):
        out = []
        for k in self.rows.values():
            if sources is not None and k.source not in sources:
                continue
            if status is not None and k.status != status:
                continue
            out.append(k)
        return out[:limit]

    def retire_knowledge(self, kid):
        k = self.rows.get(kid)
        if k is None or k.status != ACTIVE:
            return False
        k.status = RETIRED
        return True


def test_seed_dkb_is_idempotent():
    """First run inserts the whole corpus; a second run adds nothing because
    every stable id is already present — so ``saddle kb seed`` can run on every
    install/upgrade without duplicating the corpus."""
    d = _FakeDKB()
    first = seed_dkb(d)
    assert first["total"] > 0
    assert first["added"] == first["total"] and first["skipped"] == 0
    assert first["retired"] == 0           # nothing superseded on a fresh db

    second = seed_dkb(d)
    assert second["added"] == 0
    assert second["skipped"] == second["total"] == first["total"]
    assert second["retired"] == 0
    assert len(d.rows) == first["total"]   # no duplicate rows minted


def test_seed_dkb_retires_superseded_ids():
    """A seed entry under an old id the corpus no longer names is RETIRED (never
    deleted) on the next seed run — so an id rename (``kno_seed_x`` ->
    ``knowledge_seed_x``) can't strand the old entry as a duplicate of its
    replacement. The canonical row stays; it just leaves the active set."""
    d = _FakeDKB()
    # Simulate a db seeded under an old id scheme no longer in the corpus.
    legacy = Knowledge(kind=PRINCIPLE, title="old", body="superseded entry",
                       source=SEED, id="kno_seed_legacy")
    d.add_knowledge(legacy)

    res = seed_dkb(d)
    assert res["retired"] == 1
    assert d.rows["kno_seed_legacy"].status == RETIRED   # retired, not deleted
    assert "kno_seed_legacy" in d.rows                    # canonical row kept

    # The active seed set now equals exactly the current corpus.
    active = {k.id for k in d.list_knowledge(sources=[SEED], status=ACTIVE)}
    assert "kno_seed_legacy" not in active
    assert active == {k.id for k in load_seed_entries()}

    # Idempotent: a settled corpus retires nothing more on re-run.
    again = seed_dkb(d)
    assert again["retired"] == 0


def test_retire_superseded_never_fires_on_empty_corpus(tmp_path):
    """An empty corpus can only mean a load failure — never a legitimate "retire
    everything". The reconcile must refuse to wipe the active seed set in that
    case, leaving every existing entry active."""
    d = _FakeDKB()
    seed_dkb(d)                                          # populate from the real corpus
    before = {k.id for k in d.list_knowledge(sources=[SEED], status=ACTIVE)}
    assert before

    empty = tmp_path / "empty_seed.json"
    empty.write_text('{"version": 1, "entries": []}', encoding="utf-8")
    res = seed_dkb(d, path=empty)
    assert res["total"] == 0 and res["retired"] == 0     # guarded: nothing wiped
    after = {k.id for k in d.list_knowledge(sources=[SEED], status=ACTIVE)}
    assert after == before                               # active set untouched
