"""bubble — saddle's client-agnostic outbound voice (the bubble-up channel).

saddle SPEAKS every turn (itemize, drift, design review, lesson). Reaching the
*agent* is solved (a hook prints ``additionalContext``); reaching the *human* is
not — a hook's stderr is swallowed under an SDK host. The bubble outbox closes
that gap by persisting each message two ways: the canonical ``bubble`` table in
``saddle.db`` and a per-session JSONL mirror a no-SQLite client can ``tail``.
These tests pin the contract that makes that durable + safe:

* the store round-trips an event and stamps id / scope / ts (both backends),
* a bad level is clamped (saddle's voice must always land), an explicit id/ts
  is preserved, and a zero ts is stamped to ``now``,
* the JSONL mirror is written, appends per event, decodes back, AND is isolated
  by ``(tenant, project)`` — two tenants reusing a session id never share a file
  (the leak this module's mirror_path signature exists to prevent),
* ``recent`` is isolated by tenant + project and honours the session / level /
  since_ts / limit filters, newest-first,
* the wire form round-trips, the render contract is stable (level marks, stage
  tag, multiline indent, chronological ordering), and
* the live PreToolUse doctrine hook deposits a bubble on a BLOCK and an
  out-of-focus EDIT WARN (both alert) and on a cross-project ALLOW (notice) —
  proving the emit side is wired, not just the store.

``SADDLE_HOME`` is redirected to tmp so the db + mirrors never touch the real
install; the shared ``conftest`` resets the bubble-store singleton around every
test so the convenience/hook paths open a fresh db per test.
"""
from __future__ import annotations

import io
import json
import time

import pytest

from saddle import crossproject, ids
from saddle.bubble import (
    InMemoryBubbleStore,
    SqliteBubbleStore,
    emit_bubble,
    event_from_dict,
    event_to_dict,
    get_bubble_store,
    mirror_path,
    recent_bubbles,
    render_bubbles,
    set_bubble_store,
)
from saddle.context import Context, resolve
from saddle.models import (
    BUBBLE_ALERT,
    BUBBLE_INFO,
    BUBBLE_NOTICE,
    BubbleEvent,
)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Park the saddle data dir (db + JSONL mirrors) under tmp_path."""
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    yield


def _ctx(tenant: str = "aeli", project: str = "rayxiv4") -> Context:
    return Context(tenant=tenant, project=project)


# --- store round-trip + normalization (both backends) -----------------------

def test_sqlite_emit_stamps_and_roundtrips():
    store = SqliteBubbleStore()
    ctx = _ctx()
    ev = store.emit(ctx, BubbleEvent(
        text="hello", level=BUBBLE_ALERT, stage="guard", title="t",
        session="S1", meta={"k": "v"}))
    assert ev.id.startswith("bubble_")
    assert ev.tenant == "aeli" and ev.project == "rayxiv4" and ev.ts > 0
    got = store.recent(ctx)
    assert len(got) == 1
    g = got[0]
    assert g.text == "hello" and g.level == "alert" and g.stage == "guard"
    assert g.title == "t" and g.session == "S1" and g.meta == {"k": "v"}
    assert g.id == ev.id and g.ts == ev.ts
    store.close()


def test_inmemory_emit_roundtrips():
    store = InMemoryBubbleStore()
    ctx = _ctx()
    store.emit(ctx, BubbleEvent(text="a", session="S1"))
    got = store.recent(ctx)
    assert len(got) == 1 and got[0].text == "a" and got[0].level == "notice"


def test_emit_clamps_unknown_level():
    store = SqliteBubbleStore()
    ev = store.emit(_ctx(), BubbleEvent(text="x", level="screaming"))
    assert ev.level == BUBBLE_NOTICE  # a mis-tagged level is never worth losing
    store.close()


def test_emit_preserves_explicit_id_and_ts():
    store = InMemoryBubbleStore()
    ev = store.emit(_ctx(), BubbleEvent(text="x", id="bubble_fixed", ts=123.0))
    assert ev.id == "bubble_fixed" and ev.ts == 123.0


def test_emit_stamps_zero_ts_to_now():
    store = InMemoryBubbleStore()
    before = time.time()
    ev = store.emit(_ctx(), BubbleEvent(text="x"))  # ts defaults to 0.0
    assert ev.ts >= before  # falsy ts is replaced so the message always lands


# --- the JSONL mirror (the no-SQLite tail channel) --------------------------

def test_sqlite_emit_writes_jsonl_mirror():
    store = SqliteBubbleStore()
    ctx = _ctx()
    ev = store.emit(ctx, BubbleEvent(text="mirror me", session="S1", meta={"n": 1}))
    p = mirror_path(ctx.tenant, ctx.project, "S1")
    assert p.exists()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d == event_to_dict(ev)
    assert event_from_dict(d).text == "mirror me"  # a tail client decodes it
    store.close()


def test_mirror_appends_per_event():
    store = SqliteBubbleStore()
    ctx = _ctx()
    store.emit(ctx, BubbleEvent(text="one", session="S1"))
    store.emit(ctx, BubbleEvent(text="two", session="S1"))
    p = mirror_path(ctx.tenant, ctx.project, "S1")
    assert len(p.read_text(encoding="utf-8").splitlines()) == 2
    store.close()


def test_mirror_isolated_by_tenant_project():
    """Two tenants reusing one session id must NOT share a tail file — the
    content-bearing mirror honours the same (tenant, project) fence as the
    table."""
    store = SqliteBubbleStore()
    a, b = _ctx("aeli", "rayxiv4"), _ctx("bob", "rayxiv4")
    store.emit(a, BubbleEvent(text="aeli secret", session="S1"))
    store.emit(b, BubbleEvent(text="bob secret", session="S1"))
    pa = mirror_path("aeli", "rayxiv4", "S1")
    pb = mirror_path("bob", "rayxiv4", "S1")
    assert pa != pb
    assert "aeli secret" in pa.read_text() and "bob secret" not in pa.read_text()
    assert "bob secret" in pb.read_text() and "aeli secret" not in pb.read_text()
    store.close()


# --- (tenant, project) isolation on the table -------------------------------

def test_recent_isolated_by_project():
    store = SqliteBubbleStore()
    a, b = _ctx("aeli", "projA"), _ctx("aeli", "projB")
    store.emit(a, BubbleEvent(text="A only"))
    store.emit(b, BubbleEvent(text="B only"))
    assert [e.text for e in store.recent(a)] == ["A only"]
    assert [e.text for e in store.recent(b)] == ["B only"]
    store.close()


def test_recent_isolated_by_tenant():
    store = SqliteBubbleStore()
    a, b = _ctx("aeli", "p"), _ctx("bob", "p")
    store.emit(a, BubbleEvent(text="aeli"))
    store.emit(b, BubbleEvent(text="bob"))
    assert [e.text for e in store.recent(a)] == ["aeli"]
    assert [e.text for e in store.recent(b)] == ["bob"]
    store.close()


# --- recent() filters + ordering --------------------------------------------

def test_recent_newest_first():
    store = SqliteBubbleStore()
    ctx = _ctx()
    store.emit(ctx, BubbleEvent(text="first", ts=100.0))
    store.emit(ctx, BubbleEvent(text="second", ts=200.0))
    store.emit(ctx, BubbleEvent(text="third", ts=300.0))
    assert [e.text for e in store.recent(ctx)] == ["third", "second", "first"]
    store.close()


def test_recent_session_filter():
    store = SqliteBubbleStore()
    ctx = _ctx()
    store.emit(ctx, BubbleEvent(text="s1", session="S1"))
    store.emit(ctx, BubbleEvent(text="s2", session="S2"))
    assert [e.text for e in store.recent(ctx, session="S1")] == ["s1"]
    store.close()


def test_recent_level_filter():
    store = SqliteBubbleStore()
    ctx = _ctx()
    store.emit(ctx, BubbleEvent(text="i", level=BUBBLE_INFO))
    store.emit(ctx, BubbleEvent(text="a", level=BUBBLE_ALERT))
    assert [e.text for e in store.recent(ctx, level="alert")] == ["a"]
    store.close()


def test_recent_since_ts_filter():
    store = SqliteBubbleStore()
    ctx = _ctx()
    store.emit(ctx, BubbleEvent(text="old", ts=100.0))
    store.emit(ctx, BubbleEvent(text="new", ts=200.0))
    assert [e.text for e in store.recent(ctx, since_ts=150.0)] == ["new"]
    store.close()


def test_recent_limit():
    store = SqliteBubbleStore()
    ctx = _ctx()
    for i in range(5):
        store.emit(ctx, BubbleEvent(text=str(i), ts=float(i + 1)))
    assert [e.text for e in store.recent(ctx, limit=2)] == ["4", "3"]
    store.close()


# --- the wire form (JSONL line / client payload) ----------------------------

def test_event_wire_roundtrips():
    ev = BubbleEvent(
        text="t", level="alert", stage="guard", title="ti", session="S1",
        meta={"a": 1}, id="bubble_x", tenant="aeli", project="p", ts=42.0)
    assert event_from_dict(event_to_dict(ev)) == ev


# --- the singletons + convenience helpers -----------------------------------

def test_emit_and_recent_bubbles_via_singleton():
    ctx = _ctx()
    emit_bubble(ctx, "via convenience", level=BUBBLE_ALERT, stage="guard", session="S1")
    got = recent_bubbles(ctx, session="S1")
    assert len(got) == 1 and got[0].text == "via convenience" and got[0].level == "alert"


def test_set_bubble_store_injects():
    mem = InMemoryBubbleStore()
    set_bubble_store(mem)
    ctx = _ctx()
    emit_bubble(ctx, "injected")
    assert get_bubble_store() is mem
    assert [e.text for e in recent_bubbles(ctx)] == ["injected"]


# --- the render contract (the one block every client shows) -----------------

def test_render_bubbles_contract():
    evs = [
        BubbleEvent(text="line1\nline2", level="alert", stage="guard", ts=0.0, id="b2"),
        BubbleEvent(text="info msg", level="info", stage="", ts=0.0, id="b1"),
    ]
    out = render_bubbles(evs)  # default newest_last: input is newest-first
    assert "⚠" in out and "·" in out          # alert + info level marks
    assert "guard" in out                       # stage tag
    assert "    line1" in out and "    line2" in out  # multiline indent
    assert "--:--:--" in out                    # ts=0 -> placeholder time
    # newest_last reverses: the newest (evs[0], alert) prints at the BOTTOM
    assert out.index("info msg") < out.index("line1")


def test_render_bubbles_newest_first_keeps_order():
    evs = [
        BubbleEvent(text="alpha", ts=0.0, id="b2"),
        BubbleEvent(text="omega", ts=0.0, id="b1"),
    ]
    out = render_bubbles(evs, newest_last=False)
    assert out.index("alpha") < out.index("omega")


def test_render_bubbles_empty_is_empty_string():
    assert render_bubbles([]) == ""


# --- the id kind ------------------------------------------------------------

def test_bubble_id_kind_registered():
    assert ids.KIND_BUBBLE in ids.ID_KINDS
    rid = ids.record_id(ids.KIND_BUBBLE)
    head, _, tail = rid.partition("_")
    assert head == "bubble" and len(tail) == 12


# --- the live emit side: the doctrine hook deposits bubbles ------------------

def _run_doctrine_hook(payload, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    from saddle import doctrine_hook
    return doctrine_hook.main()


def test_doctrine_hook_block_emits_alert_bubble(tmp_path, monkeypatch):
    """A scope-fence BLOCK must reach the human as an ALERT guard bubble — the
    whole reason the outbox exists (stderr alone is swallowed under an SDK host).
    A cross-project DELETE still hard-blocks (only edits were downgraded to warn),
    so it is the case that proves the block -> alert-bubble emit side."""
    monkeypatch.setenv("SADDLE_TENANT", "aeli")
    monkeypatch.setenv("SADDLE_PROJECT", "rayxiv4")
    focus = tmp_path / "focus"
    (focus / "src").mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "src").mkdir(parents=True)
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(focus))
    rc = _run_doctrine_hook(
        {"tool_name": "Bash",
         "tool_input": {"command": f"rm {outside / 'src/x.py'}"},
         "session_id": "HS1"},
        monkeypatch)
    assert rc == 0
    got = recent_bubbles(resolve(), session="HS1")
    assert any(
        e.level == "alert" and e.stage == "guard" and "BLOCKED" in e.text
        for e in got
    )


def test_doctrine_hook_out_of_focus_edit_emits_alert_bubble(tmp_path, monkeypatch):
    """A cross-project EDIT no longer blocks (Option A) but must never be silent:
    with NO covering grant it is allowed-with-warning and deposits an ALERT guard
    bubble naming the out-of-focus move — the durable surface an AFK human sees."""
    monkeypatch.setenv("SADDLE_TENANT", "aeli")
    monkeypatch.setenv("SADDLE_PROJECT", "rayxiv4")
    focus = tmp_path / "focus"
    (focus / "src").mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "src").mkdir(parents=True)
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(focus))
    rc = _run_doctrine_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": str(outside / "src/x.py")},
         "session_id": "HS3"},
        monkeypatch)
    assert rc == 0
    got = recent_bubbles(resolve(), session="HS3")
    assert any(
        e.level == "alert" and e.stage == "guard" and "OUT-OF-FOCUS" in e.text
        for e in got
    )


def test_doctrine_hook_cross_project_allow_emits_notice_bubble(tmp_path, monkeypatch):
    """A grant-authorized cross-project edit is a NOTICE bubble — an AFK human
    SEES the allow that stderr alone would hide."""
    monkeypatch.setenv("SADDLE_TENANT", "aeli")
    monkeypatch.setenv("SADDLE_PROJECT", "rayxiv4")
    focus = tmp_path / "focus"
    (focus / "src").mkdir(parents=True)
    other = tmp_path / "other"
    (other / "src").mkdir(parents=True)
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(focus))
    crossproject.grant([str(focus), str(other)], tenant="*")
    rc = _run_doctrine_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": str(other / "src/x.py")},
         "session_id": "HS2"},
        monkeypatch)
    assert rc == 0
    got = recent_bubbles(resolve(), session="HS2")
    assert any(
        e.level == "notice" and e.stage == "guard" and "cross-project ALLOW" in e.text
        for e in got
    )
