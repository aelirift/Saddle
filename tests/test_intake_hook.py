"""intake_hook — the UserPromptSubmit bubble-up hook.

Verifies saddle's per-prompt *observation* channel (the counterpart to the
PreToolUse enforcement hook): that a submitted prompt is replayed for drift,
itemized, and that the commitment + scope surface ride back as
``additionalContext`` — and that EVERY failure path fails OPEN (exit 0, never a
block). The LLM itemize is stubbed so the suite stays deterministic and offline;
the drift/commitment/scope paths are exercised for real against the in-memory
ledger.
"""
from __future__ import annotations

import io
import json

import pytest

from saddle.context import Context
from saddle.dialog import IntentTracker, set_tracker
from saddle.dialog_store import InMemoryForkStore
from saddle.models import Intake, Item


# --- transcript builders (same on-disk shape the adapter parses) -----------

def _asst(text="", *, session="s1", uuid="", sidechain=False):
    content = [{"type": "text", "text": text}] if text else []
    return {
        "type": "assistant", "sessionId": session, "uuid": uuid,
        "isSidechain": sidechain, "message": {"role": "assistant", "content": content},
    }


def _user(text, *, session="s1", uuid=""):
    return {
        "type": "user", "sessionId": session, "uuid": uuid, "isSidechain": False,
        "message": {"role": "user", "content": text},
    }


_FORK = "Here's the call on caching:\na) in-memory LRU\nb) redis\nc) sqlite cache"


def _write_transcript(tmp_path, objs):
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in objs) + "\n")
    return p


# --- harness ---------------------------------------------------------------

def _run_hook(payload, monkeypatch, capsys, tmp_path, *, tracker=None, env=None):
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(tmp_path))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    set_tracker(tracker or IntentTracker(store=InMemoryForkStore()))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    from saddle import intake_hook
    rc = intake_hook.main()
    return rc, capsys.readouterr()


def _context(out) -> str:
    """The additionalContext saddle injected (stdout JSON), or '' if it stayed silent."""
    if not out.out.strip():
        return ""
    return json.loads(out.out)["hookSpecificOutput"]["additionalContext"]


# --- fail-open / no-op edges -----------------------------------------------

def test_empty_stdin_is_silent(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    from saddle import intake_hook
    assert intake_hook.main() == 0
    assert capsys.readouterr().out.strip() == ""


def test_unparseable_payload_fails_open(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    from saddle import intake_hook
    assert intake_hook.main() == 0
    out = capsys.readouterr()
    assert out.out.strip() == ""
    assert "unparseable" in out.err


def test_blank_prompt_is_silent(monkeypatch, capsys, tmp_path):
    rc, out = _run_hook({"prompt": "   ", "session_id": "s1"}, monkeypatch, capsys, tmp_path,
                        env={"SADDLE_HOOK_ITEMIZE": "0"})
    assert rc == 0 and out.out.strip() == ""


def test_itemize_failure_degrades_not_blocks(monkeypatch, capsys, tmp_path):
    async def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr("saddle.intake.decompose", boom)
    rc, out = _run_hook(
        {"prompt": "audit the whole pipeline end to end please", "session_id": "s1"},
        monkeypatch, capsys, tmp_path,
    )
    assert rc == 0                                  # never blocks
    assert "itemization unavailable" in _context(out)


# --- the slow path: itemized know/do list + scope warning ------------------

def _fake_intake(prompt):
    return Intake(
        raw_prompt=prompt,
        summary="Fix the camera and (out of scope) build a separate launcher.",
        items=[
            Item(kind="task", ask="Fix the camera-follow bug in the launcher"),
            Item(kind="task", ask="Build the matrix terminal launcher in the other repo"),
        ],
        meta={
            "scope_warning": "1 of 2 item(s) target work OUTSIDE the focus project (game).",
            "out_of_focus": [{"n": 2, "ask": "Build the matrix terminal launcher in the other repo",
                              "reason": "a different repository/product"}],
            "todo_items": 2, "audit_complete": True,
        },
    )


def test_substantive_prompt_itemizes_and_surfaces_scope(monkeypatch, capsys, tmp_path):
    async def fake(prompt, ctx=None, **k):
        return _fake_intake(prompt)
    monkeypatch.setattr("saddle.intake.decompose", fake)
    rc, out = _run_hook(
        {"prompt": "fix the camera follow bug and build the matrix launcher", "session_id": "s1"},
        monkeypatch, capsys, tmp_path,
    )
    ctx = _context(out)
    assert rc == 0
    assert "Fix the camera-follow bug" in ctx                 # the know/do list rendered
    assert "SCOPE" in ctx and "out of focus" in ctx           # the cross-project drift surfaced
    assert "saddle [acme/game]" in ctx                        # spoken under the isolation key


def test_trivial_prompt_skips_the_llm(monkeypatch, capsys, tmp_path):
    async def boom(*a, **k):                                   # must NOT be reached
        raise AssertionError("decompose called for a trivial prompt")
    monkeypatch.setattr("saddle.intake.decompose", boom)
    rc, out = _run_hook({"prompt": "proceed", "session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    assert "itemization unavailable" not in _context(out)      # it was skipped, not failed


# --- the fast path: drift caught from the transcript -----------------------

def test_replay_surfaces_a_then_b_drift(monkeypatch, capsys, tmp_path):
    tp = _write_transcript(tmp_path, [
        _user("let's decide the cache", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),                          # offers a/b/c
        _user("a", uuid="u3"),                                 # binds a
        _asst(text="Going with option b — it's faster.", uuid="u4"),  # DRIFT
    ])
    rc, out = _run_hook(
        {"prompt": "what's the status?", "session_id": "s1", "transcript_path": str(tp)},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    ctx = _context(out)
    assert rc == 0
    assert "DRIFT" in ctx and "you committed" in ctx


def test_current_prompt_pick_binds_commitment(monkeypatch, capsys, tmp_path):
    # the agent offered a fork (in the transcript); the user's CURRENT prompt is
    # the pick — the commitment must surface THIS turn, not a turn late.
    tp = _write_transcript(tmp_path, [
        _user("decide the cache", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),
    ])
    rc, out = _run_hook(
        {"prompt": "a", "session_id": "s1", "transcript_path": str(tp)},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    ctx = _context(out)
    assert rc == 0
    assert "standing commitment" in ctx and "p1.f1.a" in ctx


def test_cursor_makes_replay_idempotent(monkeypatch, capsys, tmp_path):
    # one durable ledger across two hook fires (a fresh process each time IRL);
    # the persisted cursor means the same drift is NOT re-surfaced next prompt.
    tracker = IntentTracker(store=InMemoryForkStore())
    tp = _write_transcript(tmp_path, [
        _asst(text=_FORK, uuid="u1"),
        _user("a", uuid="u2"),
        _asst(text="Going with option b.", uuid="u3"),         # DRIFT
    ])
    payload = {"prompt": "status?", "session_id": "s1", "transcript_path": str(tp)}
    _, out1 = _run_hook(payload, monkeypatch, capsys, tmp_path, tracker=tracker,
                        env={"SADDLE_HOOK_ITEMIZE": "0"})
    assert "DRIFT" in _context(out1)
    _, out2 = _run_hook(payload, monkeypatch, capsys, tmp_path, tracker=tracker,
                        env={"SADDLE_HOOK_ITEMIZE": "0"})
    assert "DRIFT" not in _context(out2)                       # cursor advanced past it
