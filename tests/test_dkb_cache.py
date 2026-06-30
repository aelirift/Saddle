"""DKB cache tier — the bounded, evictable memory the design wisdom store grew.

These pin the invariants that make memory safe to grow forever while staying
bounded:

* a ``reference`` is ALWAYS non-durable (the cache invariant), no matter what the
  caller passes;
* the cache tier stays under its per-scope cap — enforced at write time AND by an
  explicit cleanup — evicting the COLDEST first (LRU, LFU tie-break);
* expired cache rows are excluded from retrieval and purged by cleanup;
* the DURABLE tier (facts, lessons, …) is NEVER evicted;
* retrieval bumps access stats (the eviction signal);
* an old (pre-cache) db migrates forward additively.

A deterministic in-process embedder keeps it offline — no embed server, no real
vectors — so the cache logic is exercised, not the model. sqlite-vec is real
(the index must accept the writes/deletes), but its ranking is not asserted here;
keyword (FTS) matches carry the retrieval assertions.
"""
from __future__ import annotations

import math
import sqlite3
import time

import pytest

from saddle.context import Context
from saddle.dkb import DKB
from saddle.models import FACT, MANUAL, REFERENCE, Knowledge

CTX = Context(tenant="acme", project="game")


class _FakeEmbedder:
    """Deterministic, offline embedder: a fixed-dim unit vector hashed from the
    text. Distinct texts get distinct vectors (enough for sqlite-vec to store and
    search); semantic quality is irrelevant — FTS carries the keyword assertions."""

    dim = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for i, ch in enumerate((t or "").encode()):
                v[i % self.dim] += ch
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _dkb() -> DKB:
    return DKB(":memory:", embedder=_FakeEmbedder())


def _ref(title: str, body: str = "x", **kw) -> Knowledge:
    return Knowledge(kind=REFERENCE, title=title, body=body,
                     scope_tenant="acme", scope_project="game", source=MANUAL, **kw)


def _fact(title: str, body: str = "x", **kw) -> Knowledge:
    return Knowledge(kind=FACT, title=title, body=body,
                     scope_tenant="acme", scope_project="game", source=MANUAL, **kw)


# -- the cache invariant -----------------------------------------------------

def test_reference_is_forced_non_durable_even_if_caller_says_durable():
    d = _dkb()
    k = d.add_knowledge(_ref("cache row", durable=True))   # caller lies
    assert k.durable is False and k.is_cache is True
    # and it persisted that way
    assert d.get_knowledge(k.id).durable is False


def test_fact_defaults_durable():
    d = _dkb()
    k = d.add_knowledge(_fact("identity"))
    assert k.durable is True and k.is_cache is False


# -- write-time cap enforcement (self-bounding) ------------------------------

def test_write_time_cap_evicts_coldest_and_spares_durable(monkeypatch):
    monkeypatch.setenv("SADDLE_KB_CACHE_MAX", "2")
    d = _dkb()
    d.add_knowledge(_fact("durable fact"))                 # never counts/evicts
    a = d.add_knowledge(_ref("A"))
    b = d.add_knowledge(_ref("B"))
    c = d.add_knowledge(_ref("C"))                          # pushes cache to 3 > 2
    cache = [k for k in d.list_knowledge(CTX) if k.is_cache]
    durable = [k for k in d.list_knowledge(CTX) if k.durable]
    assert len(cache) == 2                                  # bounded to the cap
    assert len(durable) == 1                                # durable untouched
    # All three were equally cold (never retrieved); the OLDEST (A) is evicted.
    ids = {k.id for k in cache}
    assert a.id not in ids and b.id in ids and c.id in ids


def test_eviction_prefers_the_coldest(monkeypatch):
    # Eviction keeps the WARMEST: a retrieved entry survives, a never-retrieved
    # one is dropped. Warmth is set via the same signal retrieval uses
    # (hits/last_used); the cap is applied by an explicit cleanup so the test is
    # deterministic regardless of the vector index's ranking.
    monkeypatch.setenv("SADDLE_KB_CACHE_MAX", "0")          # no write-time eviction
    d = _dkb()
    a = d.add_knowledge(_ref("A"))
    b = d.add_knowledge(_ref("B"))
    c = d.add_knowledge(_ref("C"))
    e = d.add_knowledge(_ref("D"))
    # Warm C and D (as a retrieval would); A and B stay cold.
    d._touch_locked_safe([c.id, e.id], time.time())
    d.cleanup(max_cache_per_scope=2)                        # keep 2 warmest
    ids = {k.id for k in d.list_knowledge(CTX) if k.is_cache}
    assert ids == {c.id, e.id}                              # cold A/B evicted
    assert a.id not in ids and b.id not in ids


# -- expiry ------------------------------------------------------------------

def test_born_expired_reference_is_purged_on_write():
    # A reference written already past its TTL is born dead — the write-time
    # sweep drops it immediately rather than storing a row that can never be read.
    d = _dkb()
    dead = d.add_knowledge(_ref("dead", expires_at=time.time() - 100))
    assert d.get_knowledge(dead.id) is None


def test_expired_cache_excluded_from_search_then_purged_by_cleanup(monkeypatch):
    monkeypatch.setenv("SADDLE_KB_CACHE_MAX", "0")          # disable cap; isolate expiry
    d = _dkb()
    live = d.add_knowledge(_ref("live", body="searchterm live"))
    dead = d.add_knowledge(_ref("dead", body="searchterm dead"))
    # Simulate the TTL elapsing AFTER the row was resident (can't time-travel, so
    # push expires_at into the past directly — bypassing the write-time sweep).
    d._conn.execute("UPDATE knowledge SET expires_at=? WHERE id=?",
                    (time.time() - 100, dead.id))
    d._conn.commit()
    hits = d.search_knowledge(CTX, "searchterm", k=5)
    got = {kn.id for kn, _ in hits}
    assert live.id in got and dead.id not in got           # expired not returned
    assert d.get_knowledge(dead.id) is not None             # still on disk pre-gc
    res = d.cleanup()
    assert res["expired"] == 1
    assert d.get_knowledge(dead.id) is None                 # purged
    assert d.get_knowledge(live.id) is not None


# -- cleanup never touches the durable tier ----------------------------------

def test_cleanup_never_evicts_durable(monkeypatch):
    monkeypatch.setenv("SADDLE_KB_CACHE_MAX", "1")
    d = _dkb()
    facts = [d.add_knowledge(_fact(f"fact {i}", body=f"f{i}")) for i in range(5)]
    [d.add_knowledge(_ref(f"ref {i}")) for i in range(5)]
    d.cleanup()
    surviving = {k.id for k in d.list_knowledge(CTX)}
    for f in facts:
        assert f.id in surviving                            # every durable fact kept
    cache = [k for k in d.list_knowledge(CTX) if k.is_cache]
    assert len(cache) == 1                                  # cache bounded to cap


def test_cleanup_bounds_every_scope(monkeypatch):
    monkeypatch.setenv("SADDLE_KB_CACHE_MAX", "0")          # add freely; bound at cleanup
    d = _dkb()
    for i in range(3):
        d.add_knowledge(Knowledge(kind=REFERENCE, title=f"g{i}", body="g",
                                  scope_tenant="acme", scope_project="game", source=MANUAL))
        d.add_knowledge(Knowledge(kind=REFERENCE, title=f"s{i}", body="s",
                                  scope_tenant="acme", scope_project="shop", source=MANUAL))
    res = d.cleanup(max_cache_per_scope=1)
    assert res["scopes"] == 2                               # both scopes bounded
    assert res["evicted"] == 4                              # 2 dropped from each scope
    game = [k for k in d.list_knowledge(Context("acme", "game")) if k.is_cache]
    shop = [k for k in d.list_knowledge(Context("acme", "shop")) if k.is_cache]
    assert len(game) == 1 and len(shop) == 1


# -- access tracking ---------------------------------------------------------

def test_search_bumps_hits_and_last_used():
    d = _dkb()
    k = d.add_knowledge(_fact("topic", body="distinctive-keyword body"))
    before = d.get_knowledge(k.id)
    assert before.hits == 0 and before.last_used == 0
    d.search_knowledge(CTX, "distinctive-keyword", k=5)
    after = d.get_knowledge(k.id)
    assert after.hits == 1 and after.last_used > 0


# -- provenance round-trips --------------------------------------------------

def test_provenance_and_ttl_round_trip():
    d = _dkb()
    prov = {"source": "src/foo.py", "fingerprint": "abc123"}
    future = time.time() + 10_000
    k = d.add_knowledge(_ref("snap", provenance=prov, expires_at=future))
    got = d.get_knowledge(k.id)
    assert got.provenance == prov
    assert got.expires_at == future


# -- migration of a pre-cache db ---------------------------------------------

def test_old_schema_db_migrates_forward(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE knowledge (id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
        "title TEXT NOT NULL, body TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', "
        "scope_tenant TEXT NOT NULL DEFAULT '', scope_project TEXT NOT NULL DEFAULT '', "
        "source TEXT NOT NULL DEFAULT 'seed', status TEXT NOT NULL DEFAULT 'active', "
        "ts REAL NOT NULL);"
    )
    conn.execute("INSERT INTO knowledge(id,kind,title,body,ts) "
                 "VALUES('knowledge_old','lesson','old','body',1.0)")
    conn.commit()
    conn.close()

    d = DKB(path, embedder=_FakeEmbedder())                 # migrates on open
    old = d.get_knowledge("knowledge_old")
    assert old is not None
    assert old.durable is True and old.hits == 0 and old.expires_at == 0
    # writing the new shape works against the migrated table
    d.add_knowledge(_fact("new"))
    assert len(d.list_knowledge()) >= 2
    d.close()
    # reopening is idempotent (no duplicate-column error)
    DKB(path, embedder=_FakeEmbedder()).close()
