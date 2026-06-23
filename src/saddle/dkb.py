"""Design Knowledge Base — the persistent memory Layer 2 reasons over.

The DKB is saddle's accumulated design wisdom: pre-researched best practices and
anti-patterns (the seed corpus), plus lessons auto-harvested from real bugs and
weak designs caught while the harness runs. Layer 2 retrieves from it before
designing, so every design stands on everything learned so far instead of being
re-derived cold each time.

Two artifacts live here, both in the shared ``saddle.db`` (its own connection so
the vector extension stays off the intake path):

* :class:`~saddle.models.Knowledge` — one DKB entry. Scope-laddered exactly like
  policy (global / tenant / project): a query for ``(t, p)`` sees global ∪ that
  tenant ∪ that project, never another tenant's private lessons.
* :class:`~saddle.models.Design` — one design Layer 2 produced, kept for audit
  and reuse.

Retrieval is **hybrid**: an FTS5 keyword index and a sqlite-vec (vec0) semantic
index are queried in parallel and fused with Reciprocal Rank Fusion, so both an
exact term match and a paraphrase surface the right entry. The scope ladder and
``active`` filter run *inside* both index queries. Entries are **retired, never
deleted** (the no-delete rule): a retired entry leaves the two search indexes
but stays in the canonical ``knowledge`` table, auditable forever.

Embedding dimension is discovered from the running model and frozen into
``dkb_meta`` on first use; a later model with a different dimension is rejected
loudly rather than silently corrupting the index.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Iterable

from saddle.context import Context
from saddle.embed import Embedder, get_embedder
from saddle.models import (
    ACTIVE,
    DESIGN_STATUSES,
    KNOWLEDGE_KINDS,
    KNOWLEDGE_SOURCES,
    KNOWLEDGE_STATUSES,
    RETIRED,
    Design,
    Knowledge,
)
from saddle.store import _new_id, default_db_path

_log = logging.getLogger("saddle.dkb")

# Reciprocal Rank Fusion constant — standard 60; dampens the head so a single
# index can't dominate the fused order.
_RRF_C = 60
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

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
    ts            REAL NOT NULL
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


def _row_to_knowledge(r: sqlite3.Row) -> Knowledge:
    return Knowledge(
        kind=r["kind"], title=r["title"], body=r["body"], tags=_jl(r["tags"]),
        scope_tenant=r["scope_tenant"], scope_project=r["scope_project"],
        source=r["source"], status=r["status"], id=r["id"], ts=r["ts"],
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
        """Insert a DKB entry into the canonical table + both search indexes."""
        if k.kind not in KNOWLEDGE_KINDS:
            raise ValueError(f"unknown knowledge kind {k.kind!r}")
        if k.source not in KNOWLEDGE_SOURCES:
            raise ValueError(f"unknown knowledge source {k.source!r}")
        if k.status not in KNOWLEDGE_STATUSES:
            raise ValueError(f"unknown knowledge status {k.status!r}")
        if k.scope_project and not k.scope_tenant:
            raise ValueError("scope_project requires scope_tenant (project implies tenant)")
        k.id = k.id or _new_id("kno")
        k.ts = k.ts or time.time()
        tags_json = json.dumps(list(k.tags))
        emb = self._emb().embed([f"{k.title}\n\n{k.body}"])[0]
        with self._lock:
            self._ensure_vec()
            self._conn.execute(
                "INSERT INTO knowledge(id,kind,title,body,tags,scope_tenant,"
                "scope_project,source,status,ts) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (k.id, k.kind, k.title, k.body, tags_json, k.scope_tenant,
                 k.scope_project, k.source, k.status, k.ts),
            )
            if k.status == ACTIVE:
                self._index(k, tags_json, emb)
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
        top = sorted(scores.items(), key=lambda kv: -kv[1])[: int(k)]
        out: list[tuple[Knowledge, float]] = []
        for kid, score in top:
            kn = self.get_knowledge(kid)
            if kn is not None and kn.status == ACTIVE:
                out.append((kn, score))
        return out

    # -- design artifacts -------------------------------------------------
    def add_design(self, ctx: Context, design: Design) -> Design:
        if design.status not in DESIGN_STATUSES:
            raise ValueError(f"unknown design status {design.status!r}")
        design.id = design.id or _new_id("dsg")
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
