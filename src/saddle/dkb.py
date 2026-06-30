"""Design Knowledge Base — saddle's persistent, semantically-searchable memory.

The DKB holds two families of knowledge in ONE store (one schema, one hybrid
index, one scope ladder), because they are retrieved the same way and the split
is a property of an entry, not a reason for a second subsystem:

* **Design wisdom** — curated best practices, anti-patterns, principles, and
  lessons (the seed corpus + lessons auto-harvested from real bugs). Layer 2
  retrieves these before designing, so every design stands on everything learned
  so far instead of being re-derived cold.
* **Memory** — plain facts saddle would otherwise re-derive every session by
  re-reading files or re-searching the web: its own identity, a project's stable
  invariants (``fact``), and a *cache* of materialized file/web lookups
  (``reference``). This is the "stop figuring the same thing out every time"
  tier — retrieved on demand (see :mod:`saddle.recall`), NEVER prepended
  wholesale, so context cannot balloon.

Two tiers, by an entry's ``durable`` bit:

* **Durable** (all design wisdom + curated ``fact`` s) — hard-won and
  irreproducible, so **retired, never deleted**: a retired entry leaves the two
  search indexes but stays in the canonical ``knowledge`` table, auditable
  forever. Never auto-evicted.
* **Cache** (``reference`` s — forced non-durable on write) — reconstructible
  from their source, so they are **bounded and evictable**: :meth:`DKB.cleanup`
  (and the write-time cap in :meth:`DKB.add_knowledge`) DELETE the coldest /
  expired cache rows so storage never grows without bound. This is the lesson
  "any data store needs a bounded cleanup process" applied to the cache itself,
  and it does NOT violate the no-delete rule — that rule protects irreproducible
  knowledge; a cache you can rebuild by re-reading its source is exempt.

Retrieval is **hybrid**: an FTS5 keyword index and a sqlite-vec (vec0) semantic
index are queried in parallel and fused with Reciprocal Rank Fusion, so both an
exact term match (a rare identifier the embedder would blur) and a paraphrase
surface the right entry — pure semantic search does NOT subsume keyword search,
which is exactly why both run. The scope ladder and ``active`` filter run
*inside* both index queries; retrieval bumps ``hits``/``last_used`` (the eviction
signal) and excludes expired cache rows.

Both artifacts live in the shared ``saddle.db`` (its own connection so the vector
extension stays off the intake path): :class:`~saddle.models.Knowledge` (one DKB
entry, scope-laddered like policy — a query for ``(t, p)`` sees global ∪ that
tenant ∪ that project, never another tenant's private rows) and
:class:`~saddle.models.Design` (one design Layer 2 produced, kept for audit/reuse).

Embedding dimension is discovered from the running model and frozen into
``dkb_meta`` on first use; a later model with a different dimension is rejected
loudly rather than silently corrupting the index.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Iterable

from saddle import ids
from saddle.context import Context
from saddle.embed import Embedder, get_embedder
from saddle.models import (
    ACTIVE,
    CACHE_KINDS,
    DESIGN_STATUSES,
    KNOWLEDGE_KINDS,
    KNOWLEDGE_SOURCES,
    KNOWLEDGE_STATUSES,
    REFERENCE,
    RETIRED,
    Design,
    Knowledge,
)
from saddle.store import default_db_path

_log = logging.getLogger("saddle.dkb")

# Reciprocal Rank Fusion constant — standard 60; dampens the head so a single
# index can't dominate the fused order.
_RRF_C = 60
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

# Default cap on CACHE (non-durable) entries per scope, before the coldest are
# evicted. A bound, not a hard-coded constant: it is the env-overridable default
# (``$SADDLE_KB_CACHE_MAX``), because "how much cache is worth keeping" is
# deployment policy, while the durable tier is never capped.
_DEFAULT_CACHE_MAX = 500


def _cache_max() -> int:
    """Per-scope cache cap from ``$SADDLE_KB_CACHE_MAX`` (default
    :data:`_DEFAULT_CACHE_MAX`). A value <= 0 disables the cap (unbounded cache —
    explicit opt-out, never the silent default)."""
    raw = os.environ.get("SADDLE_KB_CACHE_MAX", "")
    if not raw.strip():
        return _DEFAULT_CACHE_MAX
    try:
        return int(raw)
    except ValueError:
        _log.warning("SADDLE_KB_CACHE_MAX=%r is not an int; using %d", raw, _DEFAULT_CACHE_MAX)
        return _DEFAULT_CACHE_MAX

_DKB_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '[]',
    scope_tenant  TEXT NOT NULL DEFAULT '',
    scope_project TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT 'seed',
    status        TEXT NOT NULL DEFAULT 'active',
    ts            REAL NOT NULL,
    durable       INTEGER NOT NULL DEFAULT 1,
    hits          INTEGER NOT NULL DEFAULT 0,
    last_used     REAL NOT NULL DEFAULT 0,
    expires_at    REAL NOT NULL DEFAULT 0,
    provenance    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_knowledge_scope
    ON knowledge(scope_tenant, scope_project, status);

CREATE TABLE IF NOT EXISTS design (
    id         TEXT PRIMARY KEY,
    tenant     TEXT NOT NULL,
    project    TEXT NOT NULL,
    intake_id  TEXT NOT NULL DEFAULT '',
    ask        TEXT NOT NULL,
    summary    TEXT NOT NULL DEFAULT '',
    problem    TEXT NOT NULL DEFAULT '',
    approach   TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    satisfies  TEXT NOT NULL DEFAULT '[]',
    avoids     TEXT NOT NULL DEFAULT '[]',
    heeds      TEXT NOT NULL DEFAULT '[]',
    meta       TEXT NOT NULL DEFAULT '{}',
    status     TEXT NOT NULL DEFAULT 'draft',
    ts         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_design_tp ON design(tenant, project, ts);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    kid UNINDEXED, kind UNINDEXED,
    scope_tenant UNINDEXED, scope_project UNINDEXED,
    title, body, tags
);

CREATE TABLE IF NOT EXISTS dkb_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _vec_blob(xs: list[float]) -> bytes:
    return struct.pack("%df" % len(xs), *xs)


def _fts_query(text: str) -> str:
    """Turn arbitrary prompt text into a safe FTS5 OR-of-terms MATCH string.

    Every token is double-quoted so FTS5 operators in the user's text can never
    be interpreted; OR (not the implicit AND) maximizes recall for retrieval.
    """
    terms = [t for t in _WORD_RE.findall(text.lower()) if len(t) >= 2]
    return " OR ".join(f'"{t}"' for t in terms)


def _jl(raw: str) -> list[str]:
    try:
        v = json.loads(raw or "[]")
        return [str(x) for x in v] if isinstance(v, list) else []
    except (TypeError, ValueError):
        return []


def _provenance(raw: str) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        return {}


def _row_to_knowledge(r: sqlite3.Row) -> Knowledge:
    return Knowledge(
        kind=r["kind"], title=r["title"], body=r["body"], tags=_jl(r["tags"]),
        scope_tenant=r["scope_tenant"], scope_project=r["scope_project"],
        source=r["source"], status=r["status"], id=r["id"], ts=r["ts"],
        durable=bool(r["durable"]), hits=int(r["hits"]),
        last_used=float(r["last_used"]), expires_at=float(r["expires_at"]),
        provenance=_provenance(r["provenance"]),
    )


def _row_to_design(r: sqlite3.Row) -> Design:
    return Design(
        ask=r["ask"], summary=r["summary"], problem=r["problem"],
        approach=r["approach"], body=r["body"],
        satisfies=_jl(r["satisfies"]), avoids=_jl(r["avoids"]), heeds=_jl(r["heeds"]),
        meta=json.loads(r["meta"] or "{}"), status=r["status"], id=r["id"],
        tenant=r["tenant"], project=r["project"], intake_id=r["intake_id"], ts=r["ts"],
    )


class DKB:
    """SQLite + sqlite-vec backed Design Knowledge Base.

    Owns its own connection (vector extension loaded) to the shared db file.
    The embedder is resolved lazily, so opening a DKB never spawns the embed
    server until something actually needs a vector.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        raw = default_db_path() if path is None else path
        self._path = str(raw)
        if self._path != ":memory:":
            p = Path(self._path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(p)
        self._lock = threading.Lock()
        self._embedder = embedder
        self._dim = 0
        self._conn = self._connect(self._path)
        self._conn.executescript(_DKB_SCHEMA)
        self._migrate()
        self._conn.commit()

    @staticmethod
    def _connect(path: str) -> sqlite3.Connection:
        import sqlite_vec

        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # Cache columns added after the original schema shipped. ``CREATE TABLE IF
    # NOT EXISTS`` never alters an existing table, so a db seeded before this
    # change keeps the old shape; bring it forward additively (ALTER ADD COLUMN
    # is cheap and the NOT NULL DEFAULTs backfill every existing row as a durable,
    # never-used entry — exactly what a pre-cache row IS).
    _CACHE_COLUMNS: tuple[tuple[str, str], ...] = (
        ("durable", "INTEGER NOT NULL DEFAULT 1"),
        ("hits", "INTEGER NOT NULL DEFAULT 0"),
        ("last_used", "REAL NOT NULL DEFAULT 0"),
        ("expires_at", "REAL NOT NULL DEFAULT 0"),
        ("provenance", "TEXT NOT NULL DEFAULT '{}'"),
    )

    def _migrate(self) -> None:
        """Additively bring an older ``knowledge`` table up to the cache schema.

        The cache-eviction index lives here (not in the base schema script) so it
        is built only AFTER the columns it spans exist — opening an old db ALTERs
        the columns in, then the index; opening a fresh one finds the columns from
        the CREATE TABLE and just adds the index. Both end identical."""
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(knowledge)")}
        for name, decl in self._CACHE_COLUMNS:
            if name not in have:
                self._conn.execute(f"ALTER TABLE knowledge ADD COLUMN {name} {decl}")
        # Eviction ladder: cleanup walks active CACHE (durable=0) rows per scope by
        # warmth, so index the exact (durable, status, scope, warmth) shape it scans.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_knowledge_cache ON knowledge"
            "(durable, status, scope_tenant, scope_project, last_used, hits)"
        )

    def _has_vec(self) -> bool:
        """Whether the sqlite-vec table exists yet — true once any entry has been
        embedded. Lets cleanup/purge touch the vector index WITHOUT spawning the
        embed server (which :meth:`_ensure_vec` would, to read the model dim)."""
        r = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_vec'"
        ).fetchone()
        return r is not None

    def _emb(self) -> Embedder:
        return self._embedder if self._embedder is not None else get_embedder()

    def _ensure_vec(self) -> int:
        """Create the vec table at the embedder's dim (once); guard mismatch."""
        if self._dim:
            return self._dim
        row = self._conn.execute(
            "SELECT value FROM dkb_meta WHERE key='embed_dim'"
        ).fetchone()
        dim = self._emb().dim
        if row is not None:
            stored = int(row["value"])
            if stored != dim:
                raise RuntimeError(
                    f"DKB embed dim mismatch: index built at {stored}, model now "
                    f"{dim}. Reindex required (model changed)."
                )
            self._dim = stored
            return stored
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vec USING vec0("
            f"kid TEXT PRIMARY KEY, kind TEXT, scope_tenant TEXT, "
            f"scope_project TEXT, embedding float[{dim}])"
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO dkb_meta(key,value) VALUES('embed_dim',?)",
            (str(dim),),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO dkb_meta(key,value) VALUES('embed_model',?)",
            (getattr(self._emb(), "_model", ""),),
        )
        self._conn.commit()
        self._dim = dim
        return dim

    # -- knowledge writes -------------------------------------------------
    def add_knowledge(self, k: Knowledge) -> Knowledge:
        """Insert a DKB entry into the canonical table + both search indexes.

        Enforces the cache invariant on write: a :data:`~saddle.models.REFERENCE`
        is reconstructible, so it is ALWAYS stored non-durable (``durable=0``)
        regardless of what the caller passed — that is what makes it eligible for
        eviction and keeps "is this row a cache row" a single fact (the
        ``durable`` column), never a kind-list that can drift. After a cache row
        lands, the scope's cap is enforced immediately, so the store is
        self-bounding without waiting for an explicit ``cleanup`` / cron pass.
        """
        if k.kind not in KNOWLEDGE_KINDS:
            raise ValueError(f"unknown knowledge kind {k.kind!r}")
        if k.source not in KNOWLEDGE_SOURCES:
            raise ValueError(f"unknown knowledge source {k.source!r}")
        if k.status not in KNOWLEDGE_STATUSES:
            raise ValueError(f"unknown knowledge status {k.status!r}")
        if k.scope_project and not k.scope_tenant:
            raise ValueError("scope_project requires scope_tenant (project implies tenant)")
        # A reference IS the cache tier; force it non-durable so the invariant
        # "cache kind <=> evictable" holds no matter how the caller built it.
        if k.kind in CACHE_KINDS:
            k.durable = False
        k.id = k.id or ids.record_id(ids.KIND_KNOWLEDGE)
        k.ts = k.ts or time.time()
        tags_json = json.dumps(list(k.tags))
        prov_json = json.dumps(dict(k.provenance or {}))
        emb = self._emb().embed([f"{k.title}\n\n{k.body}"])[0]
        with self._lock:
            self._ensure_vec()
            try:
                self._conn.execute(
                    "INSERT INTO knowledge(id,kind,title,body,tags,scope_tenant,"
                    "scope_project,source,status,ts,durable,hits,last_used,"
                    "expires_at,provenance) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (k.id, k.kind, k.title, k.body, tags_json, k.scope_tenant,
                     k.scope_project, k.source, k.status, k.ts,
                     1 if k.durable else 0, int(k.hits), float(k.last_used),
                     float(k.expires_at), prov_json),
                )
                if k.status == ACTIVE:
                    self._index(k, tags_json, emb)
                self._conn.commit()
            except Exception:
                # Never strand an open write transaction on a failed insert
                # (e.g. a duplicate id racing a concurrent seeder, or an index
                # write that fails). Roll back so the connection is clean for
                # the next caller, then re-raise — a genuine duplicate id is a
                # programming error everywhere except the idempotent seed path,
                # which catches IntegrityError itself.
                self._conn.rollback()
                raise
            # Self-bounding: a freshly added cache row may push its scope over the
            # cap, so evict the coldest right here (also clears expired rows in the
            # scope). Durable adds never trigger eviction — that tier is uncapped.
            if k.status == ACTIVE and not k.durable:
                self._evict_scope_locked(k.scope_tenant, k.scope_project, time.time())
                self._conn.commit()
        return k

    def _index(self, k: Knowledge, tags_json: str, emb: list[float]) -> None:
        """Add a row to the FTS + vec indexes (active entries only)."""
        self._conn.execute(
            "INSERT INTO knowledge_fts(kid,kind,scope_tenant,scope_project,"
            "title,body,tags) VALUES(?,?,?,?,?,?,?)",
            (k.id, k.kind, k.scope_tenant, k.scope_project, k.title, k.body, tags_json),
        )
        self._conn.execute(
            "INSERT INTO knowledge_vec(kid,kind,scope_tenant,scope_project,"
            "embedding) VALUES(?,?,?,?,?)",
            (k.id, k.kind, k.scope_tenant, k.scope_project, _vec_blob(emb)),
        )

    def retire_knowledge(self, kid: str) -> bool:
        """Retire (never delete): flip status + drop from both search indexes."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE knowledge SET status=? WHERE id=? AND status=?",
                (RETIRED, kid, ACTIVE),
            )
            if cur.rowcount:
                self._conn.execute("DELETE FROM knowledge_fts WHERE kid=?", (kid,))
                self._conn.execute("DELETE FROM knowledge_vec WHERE kid=?", (kid,))
            self._conn.commit()
            return cur.rowcount > 0

    # -- cache cleanup: keep the reconstructible tier bounded -------------
    # The ONLY place the DKB deletes rows. It deletes CACHE rows only
    # (``durable=0``) — a row you can rebuild by re-reading its source — so the
    # no-delete rule that protects irreproducible knowledge is never crossed.
    # Every delete is logged (no silent truncation: a bounded cache that drops
    # what it dropped, loudly, is the contract).

    def _purge_locked(self, kids: list[str]) -> int:
        """Hard-delete cache rows by id from the canonical table + both indexes.

        Lock must be held. The ``durable=0`` guard on the canonical delete is the
        safety net: even if a durable id were passed by mistake, it survives (and
        contributes 0 to the count), so this can never erase curated knowledge.
        Index rows for a (wrongly) durable id are not touched because such an id
        never reaches here — callers select strictly ``durable=0``."""
        if not kids:
            return 0
        deleted = 0
        has_vec = self._has_vec()
        for kid in kids:
            cur = self._conn.execute(
                "DELETE FROM knowledge WHERE id=? AND durable=0", (kid,)
            )
            if not cur.rowcount:
                continue  # durable or already gone — leave the indexes alone
            deleted += 1
            self._conn.execute("DELETE FROM knowledge_fts WHERE kid=?", (kid,))
            if has_vec:
                self._conn.execute("DELETE FROM knowledge_vec WHERE kid=?", (kid,))
        return deleted

    def _expired_cache_kids_locked(
        self, st: str | None, sp: str | None, now: float
    ) -> list[str]:
        """Active cache rows past their TTL (``0 < expires_at < now``), optionally
        within one scope. Lock must be held."""
        sql = (
            "SELECT id FROM knowledge WHERE durable=0 AND status=? "
            "AND expires_at>0 AND expires_at<?"
        )
        args: list = [ACTIVE, now]
        if st is not None:
            sql += " AND scope_tenant=? AND scope_project=?"
            args.extend([st, sp or ""])
        return [r["id"] for r in self._conn.execute(sql, args).fetchall()]

    def _overflow_cache_kids_locked(self, st: str, sp: str, cap: int) -> list[str]:
        """The COLDEST active cache rows in one scope beyond ``cap`` — the ones to
        evict. Coldness = least-recently-used, LFU (fewest ``hits``) then oldest
        as tie-breaks. Lock must be held; ``cap<=0`` means uncapped (evict none)."""
        if cap <= 0:
            return []
        n = self._conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge WHERE durable=0 AND status=? "
            "AND scope_tenant=? AND scope_project=?",
            (ACTIVE, st, sp),
        ).fetchone()["c"]
        if n <= cap:
            return []
        # Keep the ``cap`` warmest; the remainder (coldest first) are evicted.
        rows = self._conn.execute(
            "SELECT id FROM knowledge WHERE durable=0 AND status=? "
            "AND scope_tenant=? AND scope_project=? "
            "ORDER BY last_used ASC, hits ASC, ts ASC LIMIT ?",
            (ACTIVE, st, sp, n - cap),
        ).fetchall()
        return [r["id"] for r in rows]

    def _evict_scope_locked(self, st: str, sp: str, now: float, cap: int | None = None) -> int:
        """Drop expired + over-cap cache rows for ONE scope. Lock must be held.
        Returns how many rows were purged (caller commits)."""
        cap = _cache_max() if cap is None else cap
        kids = self._expired_cache_kids_locked(st, sp, now)
        kids += [k for k in self._overflow_cache_kids_locked(st, sp, cap) if k not in kids]
        n = self._purge_locked(kids)
        if n:
            _log.info(
                "DKB cache: evicted %d entr%s from scope [%s] (cap=%d)",
                n, "y" if n == 1 else "ies", f"{st}/{sp}" if st else "global", cap,
            )
        return n

    def cleanup(self, *, max_cache_per_scope: int | None = None) -> dict:
        """Bound the cache tier across EVERY scope: purge expired cache rows, then
        evict the coldest of any scope over its cap. The durable tier is never
        touched. This is the full cross-scope sweep behind ``saddle kb gc``; the
        write-time path in :meth:`add_knowledge` keeps a single scope bounded
        between sweeps. Returns ``{"expired", "evicted", "scopes"}``."""
        cap = _cache_max() if max_cache_per_scope is None else max_cache_per_scope
        now = time.time()
        with self._lock:
            expired = self._purge_locked(self._expired_cache_kids_locked(None, None, now))
            scopes = self._conn.execute(
                "SELECT DISTINCT scope_tenant, scope_project FROM knowledge "
                "WHERE durable=0 AND status=?", (ACTIVE,),
            ).fetchall()
            evicted = 0
            touched = 0
            for s in scopes:
                kids = self._overflow_cache_kids_locked(
                    s["scope_tenant"], s["scope_project"], cap
                )
                got = self._purge_locked(kids)
                if got:
                    touched += 1
                evicted += got
            self._conn.commit()
        result = {"expired": expired, "evicted": evicted, "scopes": touched}
        _log.info(
            "DKB cleanup: %d expired, %d evicted across %d scope(s) (cap=%d)",
            expired, evicted, touched, cap,
        )
        return result

    # -- knowledge reads --------------------------------------------------
    def get_knowledge(self, kid: str) -> Knowledge | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM knowledge WHERE id=?", (kid,)
            ).fetchone()
        return _row_to_knowledge(r) if r else None

    def list_knowledge(
        self,
        ctx: Context | None = None,
        *,
        kinds: Iterable[str] | None = None,
        sources: Iterable[str] | None = None,
        status: str | None = ACTIVE,
        limit: int = 200,
    ) -> list[Knowledge]:
        """List entries, optionally scope-laddered to ``ctx`` and filtered."""
        sql = ["SELECT * FROM knowledge WHERE 1=1"]
        args: list = []
        if ctx is not None:
            sql.append("AND scope_tenant IN ('', ?) AND scope_project IN ('', ?)")
            args.extend([ctx.tenant, ctx.project])
        if status is not None:
            sql.append("AND status=?")
            args.append(status)
        for col, vals in (("kind", kinds), ("source", sources)):
            vs = list(vals) if vals is not None else None
            if vs:
                sql.append(f"AND {col} IN ({','.join('?' * len(vs))})")
                args.extend(vs)
        sql.append("ORDER BY ts DESC LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_knowledge(r) for r in rows]

    def _scope_args(self, ctx: Context | None) -> tuple[str, list]:
        if ctx is None:
            return ("scope_tenant='' AND scope_project=''", [])
        return ("scope_tenant IN ('', ?) AND scope_project IN ('', ?)",
                [ctx.tenant, ctx.project])

    def _kind_clause(self, kinds: Iterable[str] | None) -> tuple[str, list]:
        vs = list(kinds) if kinds is not None else None
        if not vs:
            return ("", [])
        return (f" AND kind IN ({','.join('?' * len(vs))})", vs)

    def _fts_hits(self, ctx, query, kinds, n) -> list[str]:
        match = _fts_query(query)
        if not match:
            return []
        scope_sql, scope_args = self._scope_args(ctx)
        kind_sql, kind_args = self._kind_clause(kinds)
        sql = (
            "SELECT kid FROM knowledge_fts WHERE knowledge_fts MATCH ? "
            f"AND {scope_sql}{kind_sql} ORDER BY rank LIMIT {int(n)}"
        )
        with self._lock:
            rows = self._conn.execute(sql, [match, *scope_args, *kind_args]).fetchall()
        return [r["kid"] for r in rows]

    def _vec_hits(self, ctx, query, kinds, n) -> list[str]:
        try:
            qv = _vec_blob(self._emb().embed([query])[0])
        except Exception as exc:  # noqa: BLE001 — degrade to FTS-only if embed down
            _log.warning("vector retrieval unavailable (%s); FTS-only", exc)
            return []
        scope_sql, scope_args = self._scope_args(ctx)
        kind_sql, kind_args = self._kind_clause(kinds)
        with self._lock:
            self._ensure_vec()
            sql = (
                "SELECT kid FROM knowledge_vec WHERE embedding MATCH ? "
                f"AND k={int(n)} AND {scope_sql}{kind_sql} ORDER BY distance"
            )
            rows = self._conn.execute(sql, [qv, *scope_args, *kind_args]).fetchall()
        return [r["kid"] for r in rows]

    def search_knowledge(
        self,
        ctx: Context | None,
        query: str,
        *,
        k: int = 8,
        kinds: Iterable[str] | None = None,
    ) -> list[tuple[Knowledge, float]]:
        """Hybrid scope-laddered retrieval: FTS5 ∪ vec0, fused with RRF.

        Returns up to ``k`` (Knowledge, fused_score) pairs, best first. If the
        embed server is down, degrades to keyword-only rather than failing.
        """
        q = (query or "").strip()
        if not q:
            return []
        kinds = list(kinds) if kinds is not None else None
        cand = max(int(k) * 4, 32)
        fts = self._fts_hits(ctx, q, kinds, cand)
        vec = self._vec_hits(ctx, q, kinds, cand)
        scores: dict[str, float] = {}
        for ranked in (fts, vec):
            for rank, kid in enumerate(ranked):
                scores[kid] = scores.get(kid, 0.0) + 1.0 / (_RRF_C + rank + 1)
        # Walk ALL candidates by fused score (not a pre-truncated top-k) so a
        # skipped expired/retired row doesn't shrink the result below k — a stale
        # cache row must not crowd out a live hit.
        ranked_all = sorted(scores.items(), key=lambda kv: -kv[1])
        now = time.time()
        out: list[tuple[Knowledge, float]] = []
        for kid, score in ranked_all:
            if len(out) >= int(k):
                break
            kn = self.get_knowledge(kid)
            if kn is None or kn.status != ACTIVE:
                continue
            if kn.expires_at and kn.expires_at < now:
                continue  # expired cache row — excluded; cleanup will purge it
            out.append((kn, score))
        # Retrieval is the eviction signal: bump access stats on what we return so
        # the cache's coldest (the never-recalled rows) are what cleanup drops.
        self._touch_locked_safe([kn.id for kn, _ in out], now)
        return out

    def _touch_locked_safe(self, kids: list[str], now: float) -> None:
        """Record a retrieval: ``hits += 1`` and ``last_used = now`` for each id.
        Acquires the lock itself (callers are outside it). Best-effort — a touch
        failure must never sink a successful read, so it is logged, not raised."""
        if not kids:
            return
        ph = ",".join("?" * len(kids))
        try:
            with self._lock:
                self._conn.execute(
                    f"UPDATE knowledge SET hits=hits+1, last_used=? WHERE id IN ({ph})",
                    [now, *kids],
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            _log.warning("DKB touch failed for %d id(s): %s", len(kids), exc)

    # -- design artifacts -------------------------------------------------
    def add_design(self, ctx: Context, design: Design) -> Design:
        if design.status not in DESIGN_STATUSES:
            raise ValueError(f"unknown design status {design.status!r}")
        design.id = design.id or ids.record_id(ids.KIND_DESIGN)
        design.tenant = ctx.tenant
        design.project = ctx.project
        design.ts = design.ts or time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO design(id,tenant,project,intake_id,ask,summary,"
                "problem,approach,body,satisfies,avoids,heeds,meta,status,ts)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (design.id, ctx.tenant, ctx.project, design.intake_id, design.ask,
                 design.summary, design.problem, design.approach, design.body,
                 json.dumps(list(design.satisfies)), json.dumps(list(design.avoids)),
                 json.dumps(list(design.heeds)), json.dumps(design.meta),
                 design.status, design.ts),
            )
            self._conn.commit()
        return design

    def get_design(self, ctx: Context, did: str) -> Design | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM design WHERE id=? AND tenant=? AND project=?",
                (did, ctx.tenant, ctx.project),
            ).fetchone()
        return _row_to_design(r) if r else None

    def list_designs(
        self, ctx: Context, *, status: str | None = None, limit: int = 50
    ) -> list[Design]:
        sql = ["SELECT * FROM design WHERE tenant=? AND project=?"]
        args: list = [ctx.tenant, ctx.project]
        if status is not None:
            sql.append("AND status=?")
            args.append(status)
        sql.append("ORDER BY ts DESC LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_design(r) for r in rows]

    def set_design_status(self, ctx: Context, did: str, status: str) -> bool:
        if status not in DESIGN_STATUSES:
            raise ValueError(f"unknown design status {status!r}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE design SET status=? WHERE id=? AND tenant=? AND project=?",
                (status, did, ctx.tenant, ctx.project),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def update_design_meta(self, ctx: Context, did: str, patch: dict) -> bool:
        """Merge ``patch`` into a design's ``meta`` JSON in place; return False if
        no such design exists. Lets a later stage record a post-design outcome —
        e.g. the convergence trail (:mod:`saddle.converge`) — onto the design it
        belongs to, without rewriting the row or disturbing ``status`` (an
        implementation outcome is not a re-verdict on the design's quality)."""
        with self._lock:
            r = self._conn.execute(
                "SELECT meta FROM design WHERE id=? AND tenant=? AND project=?",
                (did, ctx.tenant, ctx.project),
            ).fetchone()
            if r is None:
                return False
            try:
                meta = json.loads(r["meta"]) if r["meta"] else {}
            except (TypeError, ValueError):
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            meta.update(patch)
            self._conn.execute(
                "UPDATE design SET meta=? WHERE id=? AND tenant=? AND project=?",
                (json.dumps(meta), did, ctx.tenant, ctx.project),
            )
            self._conn.commit()
        return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- process-wide singleton ---------------------------------------------
_dkb: DKB | None = None


def get_dkb() -> DKB:
    global _dkb
    if _dkb is None:
        _dkb = DKB()
    return _dkb


def set_dkb(dkb: DKB | None) -> None:
    global _dkb
    _dkb = dkb


def reset_dkb() -> None:
    global _dkb
    if _dkb is not None:
        try:
            _dkb.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
    _dkb = None
