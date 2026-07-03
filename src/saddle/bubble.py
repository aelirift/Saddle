"""Client-agnostic outbox for saddle's outbound voice — the bubble-up channel.

saddle SPEAKS on every turn: it itemizes the ask, catches drift, reviews a
design, harvests a lesson. Reaching the *agent* is solved — a hook prints
``additionalContext`` on stdout and the model reads it. Reaching the *human* is
not: a hook's stderr is shown only in an interactive TTY, and under an SDK /
service host (``CLAUDE_CODE_ENTRYPOINT=sdk-py``) it is swallowed. An AFK user, or
any non-terminal client, never sees what saddle said. That is the gap this
module closes.

A :class:`~saddle.models.BubbleEvent` is one such message, made DURABLE and
CLIENT-AGNOSTIC by persisting it two ways at once:

* a canonical ``bubble`` table in the shared ``saddle.db`` — any process or
  client can query it (the source of truth, ``(tenant, project)``-scoped exactly
  like every other saddle store), and
* a per-session JSONL mirror under ``<data-dir>/bubbles/<session>.jsonl`` — one
  event per line, append-only, so a client with no SQLite at all can ``tail -f``
  saddle's voice in real time.

Neither channel assumes a particular agent or host: the launcher panel, a plain
``tail``, the ``saddle bubble`` CLI, and the MCP ``bubble_recent`` tool all read
the *same* events. The mirror is a convenience derived from the table — if a
mirror write fails the canonical row still lands, so saddle never goes silent
because a directory was unwritable.

The render contract lives here too (:func:`render_bubbles`): the one place that
turns events into the human-readable block every client shows, so the panel and
the CLI never drift in how they present saddle's voice.

Isolation is identical to every other saddle store: every row is stamped with
``(tenant, project)`` and every query filtered on them; ``session`` is a finer,
optional filter *within* a project, never a substitute for the project fence.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from saddle import ids
from saddle.context import Context
from saddle.models import (
    BUBBLE_LEVELS,
    BUBBLE_NOTICE,
    BubbleEvent,
)
from saddle.store import default_db_path

_BUBBLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS bubble (
    id        TEXT PRIMARY KEY,
    tenant    TEXT NOT NULL,
    project   TEXT NOT NULL,
    session   TEXT NOT NULL DEFAULT '',
    ts        REAL NOT NULL,
    level     TEXT NOT NULL DEFAULT 'notice',
    stage     TEXT NOT NULL DEFAULT '',
    title     TEXT NOT NULL DEFAULT '',
    text      TEXT NOT NULL,
    meta      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_bubble_tp ON bubble(tenant, project, ts);
CREATE INDEX IF NOT EXISTS ix_bubble_session
    ON bubble(tenant, project, session, ts);
"""


# -- serialization (table row <-> event <-> JSONL line) ----------------------

def _meta_json(meta) -> str:
    try:
        return json.dumps(meta or {})
    except (TypeError, ValueError):
        return "{}"


def _meta_load(raw: str) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        return {}


def _row_to_event(r: sqlite3.Row) -> BubbleEvent:
    return BubbleEvent(
        text=r["text"],
        level=r["level"],
        stage=r["stage"],
        title=r["title"],
        session=r["session"],
        meta=_meta_load(r["meta"]),
        id=r["id"],
        tenant=r["tenant"],
        project=r["project"],
        ts=r["ts"],
    )


def event_to_dict(e: BubbleEvent) -> dict:
    """Flat, JSON-safe dict — the JSONL mirror line AND the wire form a client
    receives. Field names match the table columns so any reader decodes once."""
    return {
        "id": e.id,
        "tenant": e.tenant,
        "project": e.project,
        "session": e.session,
        "ts": e.ts,
        "level": e.level,
        "stage": e.stage,
        "title": e.title,
        "text": e.text,
        "meta": e.meta or {},
    }


def event_from_dict(d: dict) -> BubbleEvent:
    return BubbleEvent(
        text=str(d.get("text", "")),
        level=str(d.get("level", BUBBLE_NOTICE)),
        stage=str(d.get("stage", "")),
        title=str(d.get("title", "")),
        session=str(d.get("session", "")),
        meta=d.get("meta") if isinstance(d.get("meta"), dict) else {},
        id=str(d.get("id", "")),
        tenant=str(d.get("tenant", "")),
        project=str(d.get("project", "")),
        ts=float(d.get("ts", 0.0) or 0.0),
    )


def _normalize(e: BubbleEvent, ctx: Context, now: float) -> None:
    """Stamp scope/id/ts and clamp the level onto an event in place. A bad level
    is clamped to NOTICE rather than dropped — saddle's voice must always land;
    a mis-tagged level is never worth losing the message."""
    e.id = e.id or ids.record_id(ids.KIND_BUBBLE)
    e.tenant = ctx.tenant
    e.project = ctx.project
    e.ts = e.ts or now
    if e.level not in BUBBLE_LEVELS:
        e.level = BUBBLE_NOTICE


# -- per-session JSONL mirror (the no-SQLite tail channel) -------------------

def _bubbles_dir() -> Path:
    """Where the per-session mirrors live — alongside the saddle DB so they
    share the install's data dir and isolation."""
    return default_db_path().parent / "bubbles"


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name) or "default"


def mirror_path(tenant: str, project: str, session: str) -> Path:
    """The append-only JSONL a client can ``tail`` for one session's bubbles:
    ``<data-dir>/bubbles/<tenant>/<project>/<session>.jsonl``. Scoped by
    ``(tenant, project)`` so a content-bearing mirror honours the SAME isolation
    fence as the table — two tenants that happen to reuse a session id can never
    write into one another's tail file."""
    return _bubbles_dir() / _safe(tenant) / _safe(project) / f"{_safe(session)}.jsonl"


def _append_mirror(e: BubbleEvent) -> None:
    """Append one event as a JSONL line. Best-effort: a mirror failure must not
    fail the canonical DB write (the table is the source of truth)."""
    p = mirror_path(e.tenant, e.project, e.session)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event_to_dict(e), ensure_ascii=False) + "\n")
    except OSError as exc:  # noqa: BLE001 — the row already persisted; mirror is a convenience
        print(f"bubble: mirror write failed ({exc!r})", file=sys.stderr)


# -- store -------------------------------------------------------------------

@runtime_checkable
class BubbleStore(Protocol):
    """Persistence surface for saddle's outbound voice. Every method is scoped
    to ``ctx`` — one project's bubbles never surface in another's."""

    def emit(self, ctx: Context, event: BubbleEvent) -> BubbleEvent: ...
    def recent(
        self,
        ctx: Context,
        *,
        session: str | None = None,
        since_ts: float | None = None,
        level: str | None = None,
        limit: int = 50,
        any_project: bool = False,
    ) -> list[BubbleEvent]: ...
    def close(self) -> None: ...


class InMemoryBubbleStore:
    """Dict-backed :class:`BubbleStore` for tests + the staged runner's capture
    assertions. Same ``ctx`` filtering as the SQLite backend; no JSONL mirror
    (the mirror is a property of the durable store, exercised by its own test)."""

    def __init__(self) -> None:
        self._events: list[BubbleEvent] = []

    def emit(self, ctx: Context, event: BubbleEvent) -> BubbleEvent:
        _normalize(event, ctx, time.time())
        self._events.append(copy.deepcopy(event))
        return event

    def recent(
        self,
        ctx: Context,
        *,
        session: str | None = None,
        since_ts: float | None = None,
        level: str | None = None,
        limit: int = 50,
        any_project: bool = False,
    ) -> list[BubbleEvent]:
        out = [
            e
            for e in self._events
            if e.tenant == ctx.tenant
            and (any_project or e.project == ctx.project)
            and (session is None or e.session == session)
            and (since_ts is None or e.ts > since_ts)
            and (level is None or e.level == level)
        ]
        out.sort(key=lambda e: (e.ts, e.id), reverse=True)
        return [copy.deepcopy(e) for e in out[: max(0, int(limit))]]

    def close(self) -> None:  # nothing to release
        pass


class SqliteBubbleStore:
    """SQLite implementation of :class:`BubbleStore`, sharing the one
    ``saddle.db`` (its own ``bubble`` table, its own connection). Each emit also
    appends to the per-session JSONL mirror so a no-SQLite client can tail it."""

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
        self._conn.executescript(_BUBBLE_SCHEMA)
        self._conn.commit()

    def emit(self, ctx: Context, event: BubbleEvent) -> BubbleEvent:
        _normalize(event, ctx, time.time())
        with self._lock:
            self._conn.execute(
                "INSERT INTO bubble(id,tenant,project,session,ts,level,stage,"
                "title,text,meta) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    event.id, ctx.tenant, ctx.project, event.session, event.ts,
                    event.level, event.stage, event.title, event.text,
                    _meta_json(event.meta),
                ),
            )
            self._conn.commit()
        _append_mirror(event)
        return event

    def recent(
        self,
        ctx: Context,
        *,
        session: str | None = None,
        since_ts: float | None = None,
        level: str | None = None,
        limit: int = 50,
        any_project: bool = False,
    ) -> list[BubbleEvent]:
        # any_project widens to the whole tenant (session/level/since still
        # bind) — the turn-end reader must see a mixed-scope turn's bubbles
        # across every project ledger they landed in (mediator design §4).
        if any_project:
            sql = ["SELECT * FROM bubble WHERE tenant=?"]
            args: list = [ctx.tenant]
        else:
            sql = ["SELECT * FROM bubble WHERE tenant=? AND project=?"]
            args = [ctx.tenant, ctx.project]
        if session is not None:
            sql.append("AND session=?")
            args.append(session)
        if since_ts is not None:
            sql.append("AND ts > ?")
            args.append(float(since_ts))
        if level is not None:
            sql.append("AND level=?")
            args.append(level)
        sql.append("ORDER BY ts DESC, id DESC LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_event(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- process-wide singleton ---------------------------------------------
_bubble_store: BubbleStore | None = None


def get_bubble_store() -> BubbleStore:
    """Return the process-global bubble store (lazily opened SqliteBubbleStore)."""
    global _bubble_store
    if _bubble_store is None:
        _bubble_store = SqliteBubbleStore()
    return _bubble_store


def set_bubble_store(store: BubbleStore) -> None:
    """Inject a bubble store (tests, alternate backends)."""
    global _bubble_store
    _bubble_store = store


def reset_bubble_store() -> None:
    """Close + drop the singleton — for tests."""
    global _bubble_store
    if _bubble_store is not None:
        try:
            _bubble_store.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
    _bubble_store = None


# --- convenience: emit + read through the singleton ----------------------

def emit_bubble(
    ctx: Context,
    text: str,
    *,
    level: str = BUBBLE_NOTICE,
    stage: str = "",
    title: str = "",
    session: str = "",
    meta: dict | None = None,
) -> BubbleEvent:
    """Build a :class:`BubbleEvent` and persist it (table + mirror) in one call —
    the form the hooks use so saddle's voice reaches a human on every turn."""
    event = BubbleEvent(
        text=text, level=level, stage=stage, title=title,
        session=session, meta=dict(meta or {}),
    )
    return get_bubble_store().emit(ctx, event)


def recent_bubbles(
    ctx: Context,
    *,
    session: str | None = None,
    since_ts: float | None = None,
    level: str | None = None,
    limit: int = 50,
    any_project: bool = False,
) -> list[BubbleEvent]:
    """Most-recent-first bubbles for ``ctx`` (optionally one session / since a
    timestamp / one level). The read path the CLI + MCP tool share.
    ``any_project`` widens to every project of the tenant — the turn-end
    stages read a mixed-scope turn's bubbles across ledgers."""
    return get_bubble_store().recent(
        ctx, session=session, since_ts=since_ts, level=level, limit=limit,
        any_project=any_project,
    )


# --- the render contract: events -> the human block every client shows ----

_LEVEL_MARK = {"info": "·", "notice": "•", "alert": "⚠"}


def render_bubbles(events: Iterable[BubbleEvent], *, newest_last: bool = True) -> str:
    """Turn events into the one human-readable block every client renders, so the
    panel, the CLI, and the MCP tool never drift in presentation. ``events`` is
    accepted newest-first (as :func:`recent_bubbles` returns); set
    ``newest_last`` to print chronologically (latest at the bottom, panel-style)."""
    evs = list(events)
    if newest_last:
        evs = list(reversed(evs))
    lines: list[str] = []
    for e in evs:
        mark = _LEVEL_MARK.get(e.level, "•")
        when = time.strftime("%H:%M:%S", time.localtime(e.ts)) if e.ts else "--:--:--"
        tag = f" {e.stage}" if e.stage else ""
        head = f"{mark} [{when}{tag}]"
        if e.title:
            head += f" {e.title}"
        lines.append(head)
        for ln in (e.text or "").splitlines() or [""]:
            lines.append(f"    {ln}")
    return "\n".join(lines)
