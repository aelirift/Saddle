"""Durable, (tenant, project)-scoped store for the topic mind-map (Slice #3).

The topic mind-map is the live-brain's third tracking grain — the DAG of
discussion threads that CONTAINS the item ledger and the settled designs (they
hang off topic nodes) and whose one active node is the commitment. This module
is only the durable STRUCTURE + its invariants; the per-turn LLM classifier that
proposes DAG changes (open / advance / split / merge / close-with-evidence /
reopen, each user-confirmed) is a later slice that mirrors
:mod:`saddle.item_ledger` / :mod:`saddle.livegoal`.

Design source: ``docs/design/topic_mind_map.md``. The store BAKES the invariants
that must never depend on a caller remembering them:

* **No hard delete.** There is no delete method. A closed or dropped node is
  SOFT — kept, queryable ("have we tried X? why did it fail?"), reopenable. This
  is the retirement gate made structural: nothing durable vanishes silently.
* **Work nodes: no evidence, no close.** A ``work`` node does not close on build;
  it advances ``none -> built_untested -> tested_unclosed -> closed``. Closing one
  requires it to have reached ``tested_unclosed`` AND an evidence string — the
  two middle states are reminder states saddle nags on until then.
* **Root / topic can't close over open children.** A ``root`` / ``topic`` node
  refuses to close while any child is still OPEN (the "all children closed"
  half of its closure rule; the "user confirms" half is engine-level).
* **Reopen two ways** — set the reopen flag on a closed node, or spawn a NEW node
  with the same ``topic_key`` (query the prior thread by key).

Shares the one ``saddle.db`` (its own ``topic_node`` / ``topic_edge`` tables, its
own connection) with the intake store and the fork ledger. WAL mode + a
process-wide lock around a single shared connection, exactly like
:class:`~saddle.store.SqliteStore`. Every row is stamped ``(tenant, project)`` and
every query filtered on them — the same isolation boundary the other stores hold;
edges carry the pair too, so an edge can never bridge two tenants' nodes.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from saddle import ids
from saddle.context import Context
from saddle.models import (
    CLOSED,
    DROPPED,
    EDGE_BACK,
    OPEN,
    TN_ROOT,
    TN_TOPIC,
    TN_WORK,
    TOPIC_EDGE_KINDS,
    TOPIC_NODE_TYPES,
    TOPIC_STATUSES,
    TOPIC_TESTING_ORDER,
    TOPIC_TESTING_STATES,
    TS_BUILT_UNTESTED,
    TS_CLOSED,
    TS_NONE,
    TS_TESTED_UNCLOSED,
    TopicEdge,
    TopicNode,
)
from saddle.store import default_db_path

# A closed / dropped node keeps a COMPACT why-summary — bounded so the closed set
# never bloats the store (a few hundred tokens/node, per the design's bounding).
_CONTEXT_CAP = 2000

_TOPIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_node (
    id               TEXT PRIMARY KEY,
    tenant           TEXT NOT NULL,
    project          TEXT NOT NULL,
    title            TEXT NOT NULL,
    type             TEXT NOT NULL DEFAULT 'topic',
    topic_key        TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'open',
    testing_state    TEXT NOT NULL DEFAULT 'none',
    recorded_context TEXT NOT NULL DEFAULT '',
    reopen_flag      INTEGER NOT NULL DEFAULT 0,
    ts               REAL NOT NULL,
    updated_ts       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_topic_node_tp
    ON topic_node(tenant, project, status, updated_ts);
CREATE INDEX IF NOT EXISTS ix_topic_node_key
    ON topic_node(tenant, project, topic_key);

CREATE TABLE IF NOT EXISTS topic_edge (
    tenant     TEXT NOT NULL,
    project    TEXT NOT NULL,
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'contains',
    condition  TEXT NOT NULL DEFAULT '',
    ts         REAL NOT NULL,
    PRIMARY KEY (tenant, project, parent_id, child_id, kind)
);
CREATE INDEX IF NOT EXISTS ix_topic_edge_parent
    ON topic_edge(tenant, project, parent_id);
CREATE INDEX IF NOT EXISTS ix_topic_edge_child
    ON topic_edge(tenant, project, child_id);
"""


def _clip_context(text: str) -> str:
    s = (text or "").strip()
    return s if len(s) <= _CONTEXT_CAP else s[: _CONTEXT_CAP - 1] + "…"


def _fold_context(existing: str, addition: str) -> str:
    """Fold a new why-note into a node's recorded context, bounded. The newest
    note leads (it's what the next reader needs first); older context trails until
    the cap trims it."""
    addition = (addition or "").strip()
    if not addition:
        return _clip_context(existing)
    existing = (existing or "").strip()
    return _clip_context(f"{addition}\n{existing}" if existing else addition)


def _row_to_node(row: sqlite3.Row) -> TopicNode:
    return TopicNode(
        title=row["title"],
        type=row["type"],
        topic_key=row["topic_key"],
        status=row["status"],
        testing_state=row["testing_state"],
        recorded_context=row["recorded_context"],
        reopen_flag=bool(row["reopen_flag"]),
        id=row["id"],
        tenant=row["tenant"],
        project=row["project"],
        ts=row["ts"],
        updated_ts=row["updated_ts"],
    )


def _row_to_edge(row: sqlite3.Row) -> TopicEdge:
    return TopicEdge(
        parent_id=row["parent_id"],
        child_id=row["child_id"],
        kind=row["kind"],
        condition=row["condition"],
        tenant=row["tenant"],
        project=row["project"],
        ts=row["ts"],
    )


@runtime_checkable
class TopicStore(Protocol):
    """Persistence surface for the topic mind-map. Every method is scoped to
    ``ctx`` — the (tenant, project) filter is applied to every read and write."""

    def add_node(self, ctx: Context, node: TopicNode) -> TopicNode: ...
    def get_node(self, ctx: Context, node_id: str) -> TopicNode | None: ...
    def list_nodes(
        self,
        ctx: Context,
        *,
        status: str | None = None,
        type: str | None = None,
        topic_key: str | None = None,
        limit: int = 200,
    ) -> list[TopicNode]: ...
    def add_edge(self, ctx: Context, edge: TopicEdge) -> TopicEdge: ...
    def edges_from(self, ctx: Context, node_id: str) -> list[TopicEdge]: ...
    def edges_to(self, ctx: Context, node_id: str) -> list[TopicEdge]: ...
    def children(
        self, ctx: Context, node_id: str, *, kind: str | None = None
    ) -> list[TopicNode]: ...
    def open_children(self, ctx: Context, node_id: str) -> list[TopicNode]: ...
    def advance_testing(
        self, ctx: Context, node_id: str, state: str
    ) -> TopicNode: ...
    def close_node(
        self, ctx: Context, node_id: str, *, evidence: str = ""
    ) -> TopicNode: ...
    def drop_node(
        self, ctx: Context, node_id: str, *, summary: str = ""
    ) -> TopicNode: ...
    def reopen_node(
        self, ctx: Context, node_id: str, *, reason: str = ""
    ) -> TopicNode: ...
    def set_reopen_flag(
        self, ctx: Context, node_id: str, flag: bool = True
    ) -> TopicNode: ...
    def reminders(self, ctx: Context, *, limit: int = 50) -> list[TopicNode]: ...
    def close(self) -> None: ...


class SqliteTopicStore:
    """SQLite implementation of :class:`TopicStore`. Thread-safe via one lock
    around a single shared connection (``check_same_thread=False``)."""

    def __init__(self, path: str | Path | None = None) -> None:
        raw = default_db_path() if path is None else path
        self._path = str(raw)
        if self._path != ":memory:":
            p = Path(self._path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(p)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_TOPIC_SCHEMA)
        self._conn.commit()

    # -- writes -----------------------------------------------------------

    def add_node(self, ctx: Context, node: TopicNode) -> TopicNode:
        """Persist a new node, stamping id/tenant/project/ts. Validates the typed
        enums up front so an invalid kind/status/testing_state can never reach a
        row (they are unrepresentable, per the design)."""
        if node.type not in TOPIC_NODE_TYPES:
            raise ValueError(f"unknown topic node type {node.type!r}")
        if node.status not in TOPIC_STATUSES:
            raise ValueError(f"unknown topic status {node.status!r}")
        if node.testing_state not in TOPIC_TESTING_STATES:
            raise ValueError(f"unknown testing state {node.testing_state!r}")
        now = time.time()
        node.id = node.id or ids.record_id(ids.KIND_TOPIC)
        node.tenant = ctx.tenant
        node.project = ctx.project
        node.ts = node.ts or now
        node.updated_ts = node.updated_ts or node.ts
        node.recorded_context = _clip_context(node.recorded_context)
        with self._lock:
            self._conn.execute(
                "INSERT INTO topic_node(id,tenant,project,title,type,topic_key,"
                "status,testing_state,recorded_context,reopen_flag,ts,updated_ts)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (node.id, ctx.tenant, ctx.project, node.title, node.type,
                 node.topic_key, node.status, node.testing_state,
                 node.recorded_context, int(node.reopen_flag), node.ts,
                 node.updated_ts),
            )
            self._conn.commit()
        return node

    def add_edge(self, ctx: Context, edge: TopicEdge) -> TopicEdge:
        """Persist an edge between two EXISTING nodes in this (tenant, project).
        Both endpoints are checked to exist here — so an edge can never bridge to
        another tenant's node or dangle. A ``back_edge`` MUST carry a condition
        (an unconditional flow-back is meaningless — the cycle only ever fires on
        its named condition). Idempotent on ``(parent, child, kind)``."""
        if edge.kind not in TOPIC_EDGE_KINDS:
            raise ValueError(f"unknown edge kind {edge.kind!r}")
        if edge.kind == EDGE_BACK and not (edge.condition or "").strip():
            raise ValueError("a back_edge requires a condition")
        if edge.parent_id == edge.child_id:
            raise ValueError("a topic edge cannot point a node at itself")
        if self.get_node(ctx, edge.parent_id) is None:
            raise ValueError(f"edge parent {edge.parent_id!r} not found in scope")
        if self.get_node(ctx, edge.child_id) is None:
            raise ValueError(f"edge child {edge.child_id!r} not found in scope")
        edge.tenant = ctx.tenant
        edge.project = ctx.project
        edge.ts = edge.ts or time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO topic_edge"
                "(tenant,project,parent_id,child_id,kind,condition,ts)"
                " VALUES(?,?,?,?,?,?,?)",
                (ctx.tenant, ctx.project, edge.parent_id, edge.child_id,
                 edge.kind, edge.condition, edge.ts),
            )
            self._conn.commit()
        return edge

    def advance_testing(self, ctx: Context, node_id: str, state: str) -> TopicNode:
        """Advance a ``work`` node's testing state (``none -> built_untested ->
        tested_unclosed``). Only a work node has a testing state; the move must go
        FORWARD along :data:`TOPIC_TESTING_ORDER` (a build never un-happens). The
        terminal ``closed`` state is reached only through :meth:`close_node`, which
        gates on evidence — so advancing here can never sneak past that gate."""
        node = self._require_node(ctx, node_id)
        if node.type != TN_WORK:
            raise ValueError(f"testing state applies to a work node, not {node.type!r}")
        if state not in TOPIC_TESTING_STATES:
            raise ValueError(f"unknown testing state {state!r}")
        if state == TS_CLOSED:
            raise ValueError("close a work node via close_node (evidence-gated)")
        cur, new = TOPIC_TESTING_ORDER.index(node.testing_state), TOPIC_TESTING_ORDER.index(state)
        if new < cur:
            raise ValueError(
                f"testing state only advances: {node.testing_state!r} -> {state!r} is backwards"
            )
        return self._update(ctx, node_id, testing_state=state)

    def close_node(self, ctx: Context, node_id: str, *, evidence: str = "") -> TopicNode:
        """Close a node — SOFT (the row survives, reopenable). Per-type gates:

        * ``work``  — requires an ``evidence`` string AND that the node reached
          ``tested_unclosed`` (no evidence / not tested => refuse). The evidence
          folds into the recorded context and the testing state moves to ``closed``.
        * ``root`` / ``topic`` — refuse while any child is still OPEN (the
          "all children closed" half of the rule).
        The other types close directly. ``user confirms`` is the engine's job; the
        store enforces only what must be structurally true."""
        node = self._require_node(ctx, node_id)
        fields: dict = {"status": CLOSED}
        if node.type == TN_WORK:
            if not (evidence or "").strip():
                raise ValueError("closing a work node requires evidence")
            if node.testing_state not in (TS_TESTED_UNCLOSED, TS_CLOSED):
                raise ValueError(
                    "a work node must reach tested_unclosed before it can close "
                    f"(is {node.testing_state!r})"
                )
            fields["testing_state"] = TS_CLOSED
        elif node.type in (TN_ROOT, TN_TOPIC):
            if self.open_children(ctx, node_id):
                raise ValueError(f"{node.type} node has open children — cannot close yet")
        if (evidence or "").strip():
            fields["recorded_context"] = _fold_context(node.recorded_context, evidence)
        return self._update(ctx, node_id, **fields)

    def drop_node(self, ctx: Context, node_id: str, *, summary: str = "") -> TopicNode:
        """Drop a node — SOFT. Keeps the compact why-summary so the thread is
        never re-explored cold ("tried X; failed on Y; don't revisit unless Z")."""
        node = self._require_node(ctx, node_id)
        fields: dict = {"status": DROPPED}
        if (summary or "").strip():
            fields["recorded_context"] = _fold_context(node.recorded_context, summary)
        return self._update(ctx, node_id, **fields)

    def reopen_node(self, ctx: Context, node_id: str, *, reason: str = "") -> TopicNode:
        """Bring a closed / dropped node back to OPEN and clear the reopen flag.
        A reopened work node returns to ``tested_unclosed`` — a reminder state — so
        it re-nags for a fresh sign-off rather than silently reading as done."""
        node = self._require_node(ctx, node_id)
        fields: dict = {"status": OPEN, "reopen_flag": 0}
        if node.type == TN_WORK and node.testing_state == TS_CLOSED:
            fields["testing_state"] = TS_TESTED_UNCLOSED
        if (reason or "").strip():
            fields["recorded_context"] = _fold_context(node.recorded_context, reason)
        return self._update(ctx, node_id, **fields)

    def set_reopen_flag(self, ctx: Context, node_id: str, flag: bool = True) -> TopicNode:
        """Mark (or clear) a node as a reopen CANDIDATE without changing its
        status — the flag drives saddle's suggestion to reopen (e.g. a back-edge
        condition fired); the actual reopen stays user-confirmed."""
        self._require_node(ctx, node_id)
        return self._update(ctx, node_id, reopen_flag=int(bool(flag)))

    def _update(self, ctx: Context, node_id: str, **fields) -> TopicNode:
        """Patch a node's mutable columns + bump updated_ts, scoped to ctx. The
        single writer for every status / testing / context / flag change, so the
        (tenant, project) fence and the updated_ts bump are never forgotten."""
        cols = ", ".join(f"{k}=?" for k in fields)
        args = list(fields.values()) + [time.time(), node_id, ctx.tenant, ctx.project]
        with self._lock:
            self._conn.execute(
                f"UPDATE topic_node SET {cols}, updated_ts=?"
                " WHERE id=? AND tenant=? AND project=?",
                args,
            )
            self._conn.commit()
        got = self.get_node(ctx, node_id)
        assert got is not None  # _require_node already proved it exists in scope
        return got

    # -- reads ------------------------------------------------------------

    def _require_node(self, ctx: Context, node_id: str) -> TopicNode:
        node = self.get_node(ctx, node_id)
        if node is None:
            raise ValueError(f"topic node {node_id!r} not found in scope")
        return node

    def get_node(self, ctx: Context, node_id: str) -> TopicNode | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM topic_node WHERE id=? AND tenant=? AND project=?",
                (node_id, ctx.tenant, ctx.project),
            ).fetchone()
        return _row_to_node(row) if row is not None else None

    def list_nodes(
        self,
        ctx: Context,
        *,
        status: str | None = None,
        type: str | None = None,
        topic_key: str | None = None,
        limit: int = 200,
    ) -> list[TopicNode]:
        sql = ["SELECT * FROM topic_node WHERE tenant=? AND project=?"]
        args: list = [ctx.tenant, ctx.project]
        if status is not None:
            sql.append("AND status=?")
            args.append(status)
        if type is not None:
            sql.append("AND type=?")
            args.append(type)
        if topic_key is not None:
            sql.append("AND topic_key=?")
            args.append(topic_key)
        sql.append("ORDER BY updated_ts DESC LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_node(r) for r in rows]

    def edges_from(self, ctx: Context, node_id: str) -> list[TopicEdge]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM topic_edge WHERE tenant=? AND project=? AND parent_id=?"
                " ORDER BY ts",
                (ctx.tenant, ctx.project, node_id),
            ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def edges_to(self, ctx: Context, node_id: str) -> list[TopicEdge]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM topic_edge WHERE tenant=? AND project=? AND child_id=?"
                " ORDER BY ts",
                (ctx.tenant, ctx.project, node_id),
            ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def children(
        self, ctx: Context, node_id: str, *, kind: str | None = None
    ) -> list[TopicNode]:
        """The child nodes reached by edges out of ``node_id`` (``contains`` by
        default; pass ``kind`` to follow back-edges). Newest-updated first."""
        edges = self.edges_from(ctx, node_id)
        want = kind if kind is not None else "contains"
        child_ids = [e.child_id for e in edges if e.kind == want]
        nodes = [self.get_node(ctx, cid) for cid in child_ids]
        got = [n for n in nodes if n is not None]
        got.sort(key=lambda n: n.updated_ts, reverse=True)
        return got

    def open_children(self, ctx: Context, node_id: str) -> list[TopicNode]:
        return [n for n in self.children(ctx, node_id) if n.status == OPEN]

    def reminders(self, ctx: Context, *, limit: int = 50) -> list[TopicNode]:
        """Open work nodes parked in a build-but-not-signed-off state — the ones
        saddle actively nags on (loud, never silent). Newest-updated first."""
        nodes = self.list_nodes(ctx, status=OPEN, type=TN_WORK, limit=limit)
        return [n for n in nodes if n.is_reminder()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- process-wide singleton ---------------------------------------------
_topic_store: TopicStore | None = None


def get_topic_store() -> TopicStore:
    """Return the process-global topic store (lazily opened SqliteTopicStore)."""
    global _topic_store
    if _topic_store is None:
        _topic_store = SqliteTopicStore()
    return _topic_store


def set_topic_store(store: TopicStore) -> None:
    """Inject a store (tests, alternate backends)."""
    global _topic_store
    _topic_store = store


def reset_topic_store() -> None:
    """Close + drop the singleton — for tests."""
    global _topic_store
    if _topic_store is not None:
        try:
            _topic_store.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
    _topic_store = None
