"""Durable, (tenant, project)-scoped ledger of agent ACTION points.

The fork/binding ledger (:mod:`saddle.dialog_store`) records the decisions; this
records what the agent then *did* about them — one row per action point, each
with a saddle-assigned, self-describing id (``p47.act5`` = action 5 of exchange
47), so a change is auditable forever:

    "where's the map feature?"
        -> removed in action p47.act5, session X, exchange 47,
           reason: "not wired — no caller for f_detail_map()"

The id itself says which exchange the agent was serving when it acted, and is
letter-tagged (``p``/``act``) so it can never be mistaken for an IP / line range /
version (see :mod:`saddle.ids`).

If the user disputes the REASON ("it wasn't wired, so the fix was to hook it up,
not delete it"), :meth:`dispute_action` marks the action ``contested`` with that
counter-reason; it can later be ``reversed``. Nothing is hard-deleted — the trail
is the point.

Isolation is identical to every other saddle store: every row is stamped with
``(tenant, project)`` and every query filtered on them. Shares the one
``saddle.db`` (its own ``action`` table, its own connection) and the shared
``counter`` table — by name — with :mod:`saddle.dialog_store`, which keeps the
action ids unique within a (tenant, project). The ``act:<pn>`` counters are
touched only here; ``prompt`` / ``node:<pn>`` only by the fork store, so the two
connections never contend on the same counter row.
"""

from __future__ import annotations

import copy
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from saddle import ids
from saddle.context import Context
from saddle.models import (
    ACTION_KINDS,
    ACTION_STATUSES,
    ACT_CONTESTED,
    ACT_OTHER,
    ACT_RECORDED,
    Action,
)
from saddle.store import default_db_path

_ACTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS action (
    id          TEXT PRIMARY KEY,
    aid         TEXT NOT NULL,
    tenant      TEXT NOT NULL,
    project     TEXT NOT NULL,
    area        TEXT NOT NULL DEFAULT '',
    session     TEXT NOT NULL DEFAULT '',
    ts          REAL NOT NULL,
    pn          INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL DEFAULT 'other',
    summary     TEXT NOT NULL DEFAULT '',
    file        TEXT NOT NULL DEFAULT '',
    line_start  INTEGER NOT NULL DEFAULT 0,
    line_end    INTEGER NOT NULL DEFAULT 0,
    symbol      TEXT NOT NULL DEFAULT '',
    reason      TEXT NOT NULL DEFAULT '',
    choice_id   TEXT NOT NULL DEFAULT '',
    fork_id     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'recorded',
    dispute     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_action_tp ON action(tenant, project, ts);
CREATE INDEX IF NOT EXISTS ix_action_symbol ON action(tenant, project, symbol);
CREATE INDEX IF NOT EXISTS ix_action_aid ON action(tenant, project, aid);

-- Shared, by name, with saddle.dialog_store (idempotent create). The per-exchange
-- ``act:<pn>`` counters assign the ``p<pn>.act<seq>`` ids, unique within a
-- (tenant, project).
CREATE TABLE IF NOT EXISTS counter (
    tenant   TEXT NOT NULL,
    project  TEXT NOT NULL,
    name     TEXT NOT NULL,
    value    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant, project, name)
);
"""

# Columns added after the action table first shipped. Applied as an additive
# migration (ALTER TABLE ADD COLUMN) so an existing saddle.db gains them without
# losing data — never a silent recreate.
_ACTION_ADDED_COLUMNS = (("area", "TEXT NOT NULL DEFAULT ''"),)


def _row_to_action(r: sqlite3.Row) -> Action:
    return Action(
        summary=r["summary"],
        kind=r["kind"],
        file=r["file"],
        line_start=r["line_start"],
        line_end=r["line_end"],
        symbol=r["symbol"],
        reason=r["reason"],
        choice_id=r["choice_id"],
        fork_id=r["fork_id"],
        status=r["status"],
        dispute=r["dispute"],
        aid=r["aid"],
        id=r["id"],
        tenant=r["tenant"],
        project=r["project"],
        area=r["area"],
        session=r["session"],
        pn=r["pn"],
        ts=r["ts"],
    )


def _normalize(action: Action, ctx: Context, now: float) -> None:
    """Stamp scope/ts and clamp kind/status onto an action in place (id/aid are
    set by the store, which owns the number)."""
    action.tenant = ctx.tenant
    action.project = ctx.project
    action.area = ctx.area
    action.ts = action.ts or now
    if action.kind not in ACTION_KINDS:
        action.kind = ACT_OTHER
    if action.status not in ACTION_STATUSES:
        action.status = ACT_RECORDED


@runtime_checkable
class ActionStore(Protocol):
    """Persistence surface for the action-provenance ledger. Every method is
    scoped to ``ctx`` — one project's actions never surface in another's."""

    def add_action(self, ctx: Context, action: Action) -> Action: ...
    def get_action(self, ctx: Context, aid: str) -> Action | None: ...
    def find_actions(
        self,
        ctx: Context,
        *,
        symbol: str | None = None,
        file: str | None = None,
        status: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> list[Action]: ...
    def dispute_action(self, ctx: Context, aid: str, reason: str) -> bool: ...
    def set_action_status(self, ctx: Context, aid: str, status: str) -> bool: ...
    def close(self) -> None: ...


class InMemoryActionStore:
    """Dict-backed :class:`ActionStore` for tests, with identical ``ctx``
    filtering and per-exchange ``p<pn>.act<seq>`` counters."""

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}
        self._counters: dict[tuple[str, str, int], int] = {}

    def _next_aid(self, ctx: Context, pn: int) -> str:
        key = (ctx.tenant, ctx.project, int(pn))
        self._counters[key] = self._counters.get(key, 0) + 1
        return ids.action(pn, self._counters[key])

    def add_action(self, ctx: Context, action: Action) -> Action:
        _normalize(action, ctx, time.time())
        action.id = action.id or ids.record_id(ids.KIND_ACTION)
        action.aid = action.aid or self._next_aid(ctx, action.pn)
        self._actions[action.id] = copy.deepcopy(action)
        return action

    def _lookup(self, ctx: Context, aid: str) -> Action | None:
        for a in self._actions.values():
            if a.tenant != ctx.tenant or a.project != ctx.project:
                continue
            if a.aid == aid or a.id == aid:
                return a
        return None

    def get_action(self, ctx: Context, aid: str) -> Action | None:
        a = self._lookup(ctx, aid)
        return copy.deepcopy(a) if a is not None else None

    def find_actions(
        self,
        ctx: Context,
        *,
        symbol: str | None = None,
        file: str | None = None,
        status: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> list[Action]:
        q = (query or "").lower()
        out = []
        for a in self._actions.values():
            if a.tenant != ctx.tenant or a.project != ctx.project:
                continue
            if symbol is not None and a.symbol != symbol:
                continue
            if file is not None and a.file != file:
                continue
            if status is not None and a.status != status:
                continue
            if q and q not in (a.summary + " " + a.symbol + " " + a.file).lower():
                continue
            out.append(a)
        out.sort(key=lambda a: a.ts, reverse=True)
        return [copy.deepcopy(a) for a in out[: max(0, int(limit))]]

    def dispute_action(self, ctx: Context, aid: str, reason: str) -> bool:
        a = self._lookup(ctx, aid)
        if a is None:
            return False
        a.status = ACT_CONTESTED
        a.dispute = reason
        return True

    def set_action_status(self, ctx: Context, aid: str, status: str) -> bool:
        if status not in ACTION_STATUSES:
            raise ValueError(f"unknown action status {status!r}")
        a = self._lookup(ctx, aid)
        if a is None:
            return False
        a.status = status
        return True

    def close(self) -> None:  # nothing to release
        pass


class SqliteActionStore:
    """SQLite implementation of :class:`ActionStore`, sharing the one
    ``saddle.db`` and the ``counter`` table with the rest of saddle."""

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
        self._conn.executescript(_ACTION_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additively bring an older saddle.db up to the current columns. New
        installs already have them from the schema; this only fills gaps on a DB
        that predates a column — no data is dropped or recreated."""
        have = {
            r["name"] for r in self._conn.execute("PRAGMA table_info(action)")
        }
        for col, decl in _ACTION_ADDED_COLUMNS:
            if col not in have:
                self._conn.execute(f"ALTER TABLE action ADD COLUMN {col} {decl}")

    def _next_aid(self, ctx: Context, pn: int) -> str:
        """Atomic ``p<pn>.act<seq>`` from the shared counter table. Each exchange
        has its own ``act:<pn>`` counter, touched only by this store, so its own
        lock is enough to serialize it (no cross-store contention)."""
        name = f"act:{int(pn)}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO counter(tenant,project,name,value) VALUES(?,?,?,0)"
                " ON CONFLICT(tenant,project,name) DO NOTHING",
                (ctx.tenant, ctx.project, name),
            )
            self._conn.execute(
                "UPDATE counter SET value=value+1"
                " WHERE tenant=? AND project=? AND name=?",
                (ctx.tenant, ctx.project, name),
            )
            row = self._conn.execute(
                "SELECT value FROM counter WHERE tenant=? AND project=? AND name=?",
                (ctx.tenant, ctx.project, name),
            ).fetchone()
            self._conn.commit()
        return ids.action(pn, int(row["value"]))

    def add_action(self, ctx: Context, action: Action) -> Action:
        _normalize(action, ctx, time.time())
        action.id = action.id or ids.record_id(ids.KIND_ACTION)
        action.aid = action.aid or self._next_aid(ctx, action.pn)
        with self._lock:
            self._conn.execute(
                "INSERT INTO action(id,aid,tenant,project,area,session,ts,pn,kind,"
                "summary,file,line_start,line_end,symbol,reason,choice_id,"
                "fork_id,status,dispute)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    action.id, action.aid, ctx.tenant, ctx.project, ctx.area,
                    action.session, action.ts, action.pn, action.kind,
                    action.summary, action.file, action.line_start, action.line_end,
                    action.symbol, action.reason, action.choice_id, action.fork_id,
                    action.status, action.dispute,
                ),
            )
            self._conn.commit()
        return action

    def get_action(self, ctx: Context, aid: str) -> Action | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM action WHERE tenant=? AND project=? AND (aid=? OR id=?)",
                (ctx.tenant, ctx.project, aid, aid),
            ).fetchone()
        return _row_to_action(row) if row is not None else None

    def find_actions(
        self,
        ctx: Context,
        *,
        symbol: str | None = None,
        file: str | None = None,
        status: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> list[Action]:
        sql = ["SELECT * FROM action WHERE tenant=? AND project=?"]
        args: list = [ctx.tenant, ctx.project]
        if symbol is not None:
            sql.append("AND symbol=?")
            args.append(symbol)
        if file is not None:
            sql.append("AND file=?")
            args.append(file)
        if status is not None:
            sql.append("AND status=?")
            args.append(status)
        if query:
            sql.append("AND (summary LIKE ? OR symbol LIKE ? OR file LIKE ?)")
            like = f"%{query}%"
            args.extend([like, like, like])
        sql.append("ORDER BY ts DESC LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_action(r) for r in rows]

    def dispute_action(self, ctx: Context, aid: str, reason: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE action SET status=?, dispute=?"
                " WHERE tenant=? AND project=? AND (aid=? OR id=?)",
                (ACT_CONTESTED, reason, ctx.tenant, ctx.project, aid, aid),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def set_action_status(self, ctx: Context, aid: str, status: str) -> bool:
        if status not in ACTION_STATUSES:
            raise ValueError(f"unknown action status {status!r}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE action SET status=?"
                " WHERE tenant=? AND project=? AND (aid=? OR id=?)",
                (status, ctx.tenant, ctx.project, aid, aid),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- process-wide singleton ---------------------------------------------
_action_store: ActionStore | None = None


def get_action_store() -> ActionStore:
    """Return the process-global action store (lazily opened SqliteActionStore)."""
    global _action_store
    if _action_store is None:
        _action_store = SqliteActionStore()
    return _action_store


def set_action_store(store: ActionStore) -> None:
    """Inject an action store (tests, alternate backends)."""
    global _action_store
    _action_store = store


def reset_action_store() -> None:
    """Close + drop the singleton — for tests."""
    global _action_store
    if _action_store is not None:
        try:
            _action_store.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
    _action_store = None
