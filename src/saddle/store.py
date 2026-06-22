"""SQLite-backed, multi-tenant store for saddle intake records + items.

One database, every row stamped with ``(tenant, project)``, every query
filtered on them — that pair IS the isolation boundary, so an id from one
tenant can never resolve another tenant's row. SQLite in WAL mode handles
many tenants/projects writing concurrently across processes; a process-wide
lock serializes the single shared connection in-process. The :class:`Store`
protocol lets a heavier backend (Postgres) drop in later without touching
callers.

The "todo list" is not a separate table — it's the cross-intake view of
open actionable items (:meth:`SqliteStore.todos`), so a (tenant, project)
accumulates a running backlog across prompts for free.

Location: ``$SADDLE_HOME/saddle.db`` (default ``~/.saddle/saddle.db``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from saddle.context import Context
from saddle.models import (
    ITEM_STATUSES,
    OPEN,
    TODO_KINDS,
    Intake,
    Item,
)

_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS intake (
    id          TEXT PRIMARY KEY,
    tenant      TEXT NOT NULL,
    project     TEXT NOT NULL,
    ts          REAL NOT NULL,
    raw_prompt  TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    meta        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_intake_tp ON intake(tenant, project, ts);

CREATE TABLE IF NOT EXISTS item (
    id          TEXT PRIMARY KEY,
    intake_id   TEXT NOT NULL,
    tenant      TEXT NOT NULL,
    project     TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    ask         TEXT NOT NULL,
    source_text TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'open',
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_item_tp ON item(tenant, project, status);
CREATE INDEX IF NOT EXISTS ix_item_intake ON item(intake_id);
"""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _default_db_path() -> Path:
    home = os.environ.get("SADDLE_HOME")
    base = (
        Path(home).expanduser()
        if home and home.strip()
        else Path.home() / ".saddle"
    )
    return base / "saddle.db"


def default_db_path() -> Path:
    """Public accessor for the saddle db location (shared with the DKB)."""
    return _default_db_path()


def _row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        kind=row["kind"],
        ask=row["ask"],
        source_text=row["source_text"],
        detail=row["detail"],
        status=row["status"],
        id=row["id"],
        intake_id=row["intake_id"],
        seq=row["seq"],
        ts=row["ts"],
    )


def _row_to_intake(row: sqlite3.Row, items: list[Item]) -> Intake:
    return Intake(
        raw_prompt=row["raw_prompt"],
        summary=row["summary"],
        items=items,
        meta=json.loads(row["meta"] or "{}"),
        id=row["id"],
        tenant=row["tenant"],
        project=row["project"],
        ts=row["ts"],
    )


@runtime_checkable
class Store(Protocol):
    """Persistence surface. Every method is scoped to ``ctx`` — the
    (tenant, project) filter is applied to every read and write."""

    def save_intake(self, ctx: Context, intake: Intake) -> Intake: ...
    def get_intake(self, ctx: Context, intake_id: str) -> Intake | None: ...
    def list_intakes(self, ctx: Context, *, limit: int = 50) -> list[Intake]: ...
    def list_items(
        self,
        ctx: Context,
        *,
        intake_id: str | None = None,
        kinds: Iterable[str] | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> list[Item]: ...
    def todos(self, ctx: Context, *, status: str = OPEN) -> list[Item]: ...
    def set_item_status(
        self, ctx: Context, item_id: str, status: str
    ) -> bool: ...
    def close(self) -> None: ...


class SqliteStore:
    """SQLite implementation of :class:`Store`. Thread-safe via one lock
    around a single shared connection (``check_same_thread=False``)."""

    def __init__(self, path: str | Path | None = None) -> None:
        raw = _default_db_path() if path is None else path
        self._path = str(raw)
        if self._path != ":memory:":
            p = Path(self._path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(p)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        self._conn.commit()

    # -- writes -----------------------------------------------------------

    def save_intake(self, ctx: Context, intake: Intake) -> Intake:
        """Persist an intake + its items in one transaction. Stamps ids,
        ts, seq, tenant, project onto the passed object and returns it."""
        now = time.time()
        intake.id = intake.id or _new_id("int")
        intake.tenant = ctx.tenant
        intake.project = ctx.project
        intake.ts = intake.ts or now
        rows = []
        for i, item in enumerate(intake.items):
            item.id = item.id or _new_id("itm")
            item.intake_id = intake.id
            item.tenant = ctx.tenant
            item.project = ctx.project
            item.seq = i
            item.ts = item.ts or now
            if item.status not in ITEM_STATUSES:
                item.status = OPEN
            rows.append(
                (item.id, item.intake_id, ctx.tenant, ctx.project, item.seq,
                 item.kind, item.ask, item.source_text, item.detail,
                 item.status, item.ts)
            )
        with self._lock:
            self._conn.execute(
                "INSERT INTO intake(id,tenant,project,ts,raw_prompt,summary,meta)"
                " VALUES(?,?,?,?,?,?,?)",
                (intake.id, ctx.tenant, ctx.project, intake.ts,
                 intake.raw_prompt, intake.summary, json.dumps(intake.meta)),
            )
            if rows:
                self._conn.executemany(
                    "INSERT INTO item(id,intake_id,tenant,project,seq,kind,ask,"
                    "source_text,detail,status,ts)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
            self._conn.commit()
        return intake

    def set_item_status(
        self, ctx: Context, item_id: str, status: str
    ) -> bool:
        if status not in ITEM_STATUSES:
            raise ValueError(f"unknown item status {status!r}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE item SET status=? WHERE id=? AND tenant=? AND project=?",
                (status, item_id, ctx.tenant, ctx.project),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # -- reads ------------------------------------------------------------

    def get_intake(self, ctx: Context, intake_id: str) -> Intake | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM intake WHERE id=? AND tenant=? AND project=?",
                (intake_id, ctx.tenant, ctx.project),
            ).fetchone()
        if row is None:
            return None
        items = self.list_items(ctx, intake_id=intake_id)
        return _row_to_intake(row, items)

    def list_intakes(self, ctx: Context, *, limit: int = 50) -> list[Intake]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM intake WHERE tenant=? AND project=?"
                " ORDER BY ts DESC LIMIT ?",
                (ctx.tenant, ctx.project, int(limit)),
            ).fetchall()
        return [_row_to_intake(r, []) for r in rows]

    def list_items(
        self,
        ctx: Context,
        *,
        intake_id: str | None = None,
        kinds: Iterable[str] | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> list[Item]:
        sql = ["SELECT * FROM item WHERE tenant=? AND project=?"]
        args: list = [ctx.tenant, ctx.project]
        if intake_id is not None:
            sql.append("AND intake_id=?")
            args.append(intake_id)
        ks = list(kinds) if kinds is not None else None
        if ks:
            sql.append(f"AND kind IN ({','.join('?' * len(ks))})")
            args.extend(ks)
        if status is not None:
            sql.append("AND status=?")
            args.append(status)
        sql.append("ORDER BY intake_id, seq LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_item(r) for r in rows]

    def todos(self, ctx: Context, *, status: str = OPEN) -> list[Item]:
        """Open actionable items (task/directive) — the running todo list."""
        return self.list_items(ctx, kinds=TODO_KINDS, status=status)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- process-wide singleton ---------------------------------------------
_store: Store | None = None


def get_store() -> Store:
    """Return the process-global store (lazily opened SqliteStore)."""
    global _store
    if _store is None:
        _store = SqliteStore()
    return _store


def set_store(store: Store) -> None:
    """Inject a store (tests, alternate backends)."""
    global _store
    _store = store


def reset_store() -> None:
    """Close + drop the singleton — for tests."""
    global _store
    if _store is not None:
        try:
            _store.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
    _store = None
