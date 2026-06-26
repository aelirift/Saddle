"""Durable, (tenant, project)-scoped ledger for conversational intent.

This is the part of saddle that OUTLIVES the agent's context window. When the
agent offers the user a set of labeled options, that :class:`~saddle.models.Fork`
is persisted here; when the user picks one, that :class:`~saddle.models.Binding`
is persisted here too. Because the ledger is saddle's — not the agent's — it
survives however hard the agent thrashes or compacts: saddle still knows that in
THIS project, ``a)`` means *that* option of *that* fork, hours later.

Isolation is the whole point. Every row is stamped with ``(tenant, project)``
and every query is filtered on them — the same boundary :mod:`saddle.store`
enforces — so a pick in project X can never resolve a fork offered in project Y.
``session`` is a finer, optional filter *within* a project (which agent
conversation), never a substitute for the project fence.

Shares the one ``saddle.db`` (its own ``fork`` / ``binding`` tables, its own
connection) with the intake store and the DKB. WAL mode + a process-wide lock
around a single shared connection, exactly like :class:`~saddle.store.SqliteStore`.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from saddle import ids
from saddle.context import Context
from saddle.models import (
    BIND_METHODS,
    FORK_OPEN,
    FORK_STATUSES,
    Binding,
    Fork,
    ForkOption,
)
from saddle.store import default_db_path

_DIALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS fork (
    id          TEXT PRIMARY KEY,
    tenant      TEXT NOT NULL,
    project     TEXT NOT NULL,
    area        TEXT NOT NULL DEFAULT '',
    session     TEXT NOT NULL DEFAULT '',
    ts          REAL NOT NULL,
    prompt      TEXT NOT NULL DEFAULT '',
    source_text TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'open',
    options     TEXT NOT NULL DEFAULT '[]',
    pn          INTEGER NOT NULL DEFAULT 0,
    seq         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_fork_tp ON fork(tenant, project, status, ts);

CREATE TABLE IF NOT EXISTS binding (
    id         TEXT PRIMARY KEY,
    fork_id    TEXT NOT NULL,
    tenant     TEXT NOT NULL,
    project    TEXT NOT NULL,
    area       TEXT NOT NULL DEFAULT '',
    session    TEXT NOT NULL DEFAULT '',
    ts         REAL NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    choice_id  TEXT NOT NULL DEFAULT '',
    user_text  TEXT NOT NULL DEFAULT '',
    method     TEXT NOT NULL DEFAULT 'label',
    confidence REAL NOT NULL DEFAULT 1.0,
    resolved   INTEGER NOT NULL DEFAULT 1,
    reason     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_binding_tp ON binding(tenant, project, ts);
CREATE INDEX IF NOT EXISTS ix_binding_fork ON binding(fork_id);

-- The numbering authority for the whole conversational ledger. One row per
-- (tenant, project, name); ``next_counter`` increments and returns atomically.
-- Names: ``prompt`` (the exchange #), ``node:<pn>`` (forks within an exchange),
-- ``act:<pn>`` (actions within an exchange, used by the action store). Sharing
-- one table keeps every id unique within a (tenant, project).
CREATE TABLE IF NOT EXISTS counter (
    tenant   TEXT NOT NULL,
    project  TEXT NOT NULL,
    name     TEXT NOT NULL,
    value    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant, project, name)
);
"""

# Columns added after the fork/binding tables first shipped. Applied as an
# additive migration (ALTER TABLE ADD COLUMN) so an existing saddle.db gains the
# numbering columns without losing data — never a silent recreate.
_FORK_ADDED_COLUMNS = (("pn", "INTEGER NOT NULL DEFAULT 0"),
                       ("seq", "INTEGER NOT NULL DEFAULT 0"),
                       ("area", "TEXT NOT NULL DEFAULT ''"))
_BINDING_ADDED_COLUMNS = (("choice_id", "TEXT NOT NULL DEFAULT ''"),
                          ("area", "TEXT NOT NULL DEFAULT ''"))


def _options_to_json(options: list[ForkOption]) -> str:
    return json.dumps(
        [
            {"label": o.label, "text": o.text, "recommended": bool(o.recommended)}
            for o in options
        ]
    )


def _options_from_json(raw: str) -> list[ForkOption]:
    try:
        data = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    out: list[ForkOption] = []
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict):
                out.append(
                    ForkOption(
                        label=str(d.get("label", "")),
                        text=str(d.get("text", "")),
                        recommended=bool(d.get("recommended", False)),
                    )
                )
    return out


def _row_to_fork(r: sqlite3.Row) -> Fork:
    return Fork(
        options=_options_from_json(r["options"]),
        prompt=r["prompt"],
        source_text=r["source_text"],
        status=r["status"],
        id=r["id"],
        tenant=r["tenant"],
        project=r["project"],
        area=r["area"],
        session=r["session"],
        ts=r["ts"],
        pn=r["pn"],
        seq=r["seq"],
    )


def _row_to_binding(r: sqlite3.Row) -> Binding:
    return Binding(
        fork_id=r["fork_id"],
        label=r["label"],
        choice_id=r["choice_id"],
        user_text=r["user_text"],
        method=r["method"],
        confidence=r["confidence"],
        resolved=bool(r["resolved"]),
        reason=r["reason"],
        id=r["id"],
        tenant=r["tenant"],
        project=r["project"],
        area=r["area"],
        session=r["session"],
        ts=r["ts"],
    )


@runtime_checkable
class ForkStore(Protocol):
    """Persistence surface for the conversational-intent ledger. Every method
    is scoped to ``ctx`` — the (tenant, project) filter is applied to every read
    and write, so one project's forks and bindings never leak into another's."""

    def next_counter(self, ctx: Context, name: str) -> int: ...
    def current_counter(self, ctx: Context, name: str) -> int: ...
    def add_fork(self, ctx: Context, fork: Fork) -> Fork: ...
    def get_fork(self, ctx: Context, fork_id: str) -> Fork | None: ...
    def get_fork_by_node(self, ctx: Context, node_id: str) -> Fork | None: ...
    def open_forks(
        self, ctx: Context, *, session: str | None = None, limit: int = 20
    ) -> list[Fork]: ...
    def set_fork_status(self, ctx: Context, fork_id: str, status: str) -> bool: ...
    def add_binding(self, ctx: Context, binding: Binding) -> Binding: ...
    def latest_binding(
        self, ctx: Context, *, session: str | None = None, resolved_only: bool = True
    ) -> Binding | None: ...
    def close(self) -> None: ...


class InMemoryForkStore:
    """Dict-backed :class:`ForkStore` for tests. Stores deep copies and filters
    on ``ctx`` exactly like the SQLite backend, so the isolation guarantee is
    identical in tests and in production."""

    def __init__(self) -> None:
        self._forks: dict[str, Fork] = {}
        self._bindings: dict[str, Binding] = {}
        self._counters: dict[tuple[str, str, str], int] = {}

    def next_counter(self, ctx: Context, name: str) -> int:
        key = (ctx.tenant, ctx.project, name)
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    def current_counter(self, ctx: Context, name: str) -> int:
        return self._counters.get((ctx.tenant, ctx.project, name), 0)

    def add_fork(self, ctx: Context, fork: Fork) -> Fork:
        now = time.time()
        fork.id = fork.id or ids.record_id(ids.KIND_FORK)
        fork.tenant = ctx.tenant
        fork.project = ctx.project
        fork.area = ctx.area
        fork.ts = fork.ts or now
        if fork.status not in FORK_STATUSES:
            fork.status = FORK_OPEN
        # The fork answers the latest user prompt (the current exchange); number
        # it within that exchange so its node_id is "p<pn>.f<seq>".
        if not fork.pn:
            fork.pn = self.current_counter(ctx, "prompt")
        if not fork.seq:
            fork.seq = self.next_counter(ctx, f"node:{fork.pn}")
        self._forks[fork.id] = copy.deepcopy(fork)
        return fork

    def get_fork(self, ctx: Context, fork_id: str) -> Fork | None:
        f = self._forks.get(fork_id)
        if f is None or f.tenant != ctx.tenant or f.project != ctx.project:
            return None
        return copy.deepcopy(f)

    def get_fork_by_node(self, ctx: Context, node_id: str) -> Fork | None:
        for f in self._forks.values():
            if (
                f.tenant == ctx.tenant
                and f.project == ctx.project
                and f.node_id == node_id
            ):
                return copy.deepcopy(f)
        return None

    def open_forks(
        self, ctx: Context, *, session: str | None = None, limit: int = 20
    ) -> list[Fork]:
        out = [
            f
            for f in self._forks.values()
            if f.tenant == ctx.tenant
            and f.project == ctx.project
            and f.status == FORK_OPEN
            and (session is None or f.session == session)
        ]
        out.sort(key=lambda f: f.ts, reverse=True)
        return [copy.deepcopy(f) for f in out[: max(0, int(limit))]]

    def set_fork_status(self, ctx: Context, fork_id: str, status: str) -> bool:
        if status not in FORK_STATUSES:
            raise ValueError(f"unknown fork status {status!r}")
        f = self._forks.get(fork_id)
        if f is None or f.tenant != ctx.tenant or f.project != ctx.project:
            return False
        f.status = status
        return True

    def add_binding(self, ctx: Context, binding: Binding) -> Binding:
        now = time.time()
        binding.id = binding.id or ids.record_id(ids.KIND_BINDING)
        binding.tenant = ctx.tenant
        binding.project = ctx.project
        binding.area = ctx.area
        binding.ts = binding.ts or now
        if binding.method not in BIND_METHODS:
            raise ValueError(f"unknown bind method {binding.method!r}")
        self._bindings[binding.id] = copy.deepcopy(binding)
        return binding

    def latest_binding(
        self, ctx: Context, *, session: str | None = None, resolved_only: bool = True
    ) -> Binding | None:
        out = [
            b
            for b in self._bindings.values()
            if b.tenant == ctx.tenant
            and b.project == ctx.project
            and (session is None or b.session == session)
            and (not resolved_only or b.resolved)
        ]
        if not out:
            return None
        out.sort(key=lambda b: b.ts, reverse=True)
        return copy.deepcopy(out[0])

    def close(self) -> None:  # nothing to release
        pass


class SqliteForkStore:
    """SQLite implementation of :class:`ForkStore`. Thread-safe via one lock
    around a single shared connection (``check_same_thread=False``), sharing the
    one ``saddle.db`` with the intake store and the DKB."""

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
        self._conn.executescript(_DIALOG_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additively bring an older saddle.db up to the current columns. New
        installs already have them from the schema; this only fills gaps on a DB
        that predates the numbering columns — no data is dropped or recreated."""
        for table, added in (("fork", _FORK_ADDED_COLUMNS),
                             ("binding", _BINDING_ADDED_COLUMNS)):
            have = {
                r["name"]
                for r in self._conn.execute(f"PRAGMA table_info({table})")
            }
            for col, decl in added:
                if col not in have:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {decl}"
                    )

    # -- numbering --------------------------------------------------------

    def next_counter(self, ctx: Context, name: str) -> int:
        """Atomically increment and return the (tenant, project, name) counter —
        the source of every prompt / node / action number. The lock makes the
        read-modify-write a single critical section even across processes' rows."""
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
        return int(row["value"])

    def current_counter(self, ctx: Context, name: str) -> int:
        """Read the counter without advancing it (0 if it has never been used)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM counter WHERE tenant=? AND project=? AND name=?",
                (ctx.tenant, ctx.project, name),
            ).fetchone()
        return int(row["value"]) if row is not None else 0

    # -- writes -----------------------------------------------------------

    def add_fork(self, ctx: Context, fork: Fork) -> Fork:
        now = time.time()
        fork.id = fork.id or ids.record_id(ids.KIND_FORK)
        fork.tenant = ctx.tenant
        fork.project = ctx.project
        fork.area = ctx.area
        fork.ts = fork.ts or now
        if fork.status not in FORK_STATUSES:
            fork.status = FORK_OPEN
        # Number the fork within the latest exchange -> node_id "p<pn>.f<seq>".
        if not fork.pn:
            fork.pn = self.current_counter(ctx, "prompt")
        if not fork.seq:
            fork.seq = self.next_counter(ctx, f"node:{fork.pn}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO fork(id,tenant,project,area,session,ts,prompt,"
                "source_text,status,options,pn,seq) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fork.id, ctx.tenant, ctx.project, ctx.area, fork.session,
                    fork.ts, fork.prompt, fork.source_text, fork.status,
                    _options_to_json(fork.options), fork.pn, fork.seq,
                ),
            )
            self._conn.commit()
        return fork

    def set_fork_status(self, ctx: Context, fork_id: str, status: str) -> bool:
        if status not in FORK_STATUSES:
            raise ValueError(f"unknown fork status {status!r}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE fork SET status=? WHERE id=? AND tenant=? AND project=?",
                (status, fork_id, ctx.tenant, ctx.project),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def add_binding(self, ctx: Context, binding: Binding) -> Binding:
        now = time.time()
        binding.id = binding.id or ids.record_id(ids.KIND_BINDING)
        binding.tenant = ctx.tenant
        binding.project = ctx.project
        binding.area = ctx.area
        binding.ts = binding.ts or now
        if binding.method not in BIND_METHODS:
            raise ValueError(f"unknown bind method {binding.method!r}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO binding(id,fork_id,tenant,project,area,session,ts,"
                "label,choice_id,user_text,method,confidence,resolved,reason)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    binding.id, binding.fork_id, ctx.tenant, ctx.project, ctx.area,
                    binding.session, binding.ts, binding.label, binding.choice_id,
                    binding.user_text, binding.method, float(binding.confidence),
                    1 if binding.resolved else 0, binding.reason,
                ),
            )
            self._conn.commit()
        return binding

    # -- reads ------------------------------------------------------------

    def get_fork(self, ctx: Context, fork_id: str) -> Fork | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM fork WHERE id=? AND tenant=? AND project=?",
                (fork_id, ctx.tenant, ctx.project),
            ).fetchone()
        return _row_to_fork(row) if row is not None else None

    def get_fork_by_node(self, ctx: Context, node_id: str) -> Fork | None:
        """Look up a fork by its assigned ``node_id`` ("p<pn>.f<seq>") — the way
        saddle verifies that a cited id names a fork it actually minted. A
        malformed or unminted node_id resolves to ``None`` (so a prose number that
        merely looks like an id is NOT treated as a citation)."""
        parsed = ids.parse_fork_node(node_id)
        if parsed is None:
            return None
        pn, seq = parsed
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM fork WHERE tenant=? AND project=? AND pn=? AND seq=?"
                " ORDER BY ts DESC LIMIT 1",
                (ctx.tenant, ctx.project, pn, seq),
            ).fetchone()
        return _row_to_fork(row) if row is not None else None

    def open_forks(
        self, ctx: Context, *, session: str | None = None, limit: int = 20
    ) -> list[Fork]:
        sql = [
            "SELECT * FROM fork WHERE tenant=? AND project=? AND status=?"
        ]
        args: list = [ctx.tenant, ctx.project, FORK_OPEN]
        if session is not None:
            sql.append("AND session=?")
            args.append(session)
        sql.append("ORDER BY ts DESC LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_fork(r) for r in rows]

    def latest_binding(
        self, ctx: Context, *, session: str | None = None, resolved_only: bool = True
    ) -> Binding | None:
        sql = ["SELECT * FROM binding WHERE tenant=? AND project=?"]
        args: list = [ctx.tenant, ctx.project]
        if session is not None:
            sql.append("AND session=?")
            args.append(session)
        if resolved_only:
            sql.append("AND resolved=1")
        sql.append("ORDER BY ts DESC LIMIT 1")
        with self._lock:
            row = self._conn.execute(" ".join(sql), args).fetchone()
        return _row_to_binding(row) if row is not None else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- process-wide singleton ---------------------------------------------
_fork_store: ForkStore | None = None


def get_fork_store() -> ForkStore:
    """Return the process-global fork store (lazily opened SqliteForkStore)."""
    global _fork_store
    if _fork_store is None:
        _fork_store = SqliteForkStore()
    return _fork_store


def set_fork_store(store: ForkStore) -> None:
    """Inject a fork store (tests, alternate backends)."""
    global _fork_store
    _fork_store = store


def reset_fork_store() -> None:
    """Close + drop the singleton — for tests."""
    global _fork_store
    if _fork_store is not None:
        try:
            _fork_store.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
    _fork_store = None
