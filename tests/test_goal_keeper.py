"""The goal-keeper — an active, unmet goal refuses the stop and drives the
agent back to work (user directive 2026-07-03, superseding the Stop hook's
original observe-only stance). Blocks only on the completion audit's verdict
(goal_active, not complete, not awaiting the user), caps consecutive forced
continuations, and alerts the human at the cap instead of spinning.
"""

from __future__ import annotations

import io
import json

import pytest

from saddle.completion import CompletionVerdict
from saddle.context import Context

_CTX = Context(tenant="acme", project="game")


def _run_stop(monkeypatch, capsys, tmp_path, verdict, *, session="s1", env=None,
              user_text="finish everything", user_uuid="u1"):
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(tmp_path))
    monkeypatch.setenv("SADDLE_HOOK_CODE", "0")
    monkeypatch.setenv("SADDLE_HOOK_LESSON", "0")
    monkeypatch.setenv("SADDLE_HOOK_VOICE", "0")
    monkeypatch.setenv("SADDLE_HOOK_DESIGN", "0")
    monkeypatch.setenv("SADDLE_GOAL_KEEPER", "1")  # on by default; test_off_switch overrides
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)

    async def _canned(goal, reply, ctx=None, **kw):
        return verdict

    monkeypatch.setattr("saddle.completion.audit_completion", _canned)
    tp = tmp_path / "t.jsonl"
    lines = [
        {"type": "user", "uuid": user_uuid,
         "message": {"content": [{"type": "text", "text": user_text}]}},
        {"type": "assistant", "uuid": "a1",
         "message": {"content": [{"type": "text", "text": "status update."}]}},
    ]
    tp.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"session_id": session, "transcript_path": str(tp)}
    )))
    from saddle import stop_hook

    rc = stop_hook.main()
    return rc, capsys.readouterr()


def _decision(out) -> dict:
    for ln in out.out.splitlines():
        try:
            doc = json.loads(ln)
        except ValueError:
            continue
        if doc.get("decision"):
            return doc
    return {}


_UNMET = CompletionVerdict(goal_active=True, complete=False,
                           missing=["the play-test of every feature"])


def test_unmet_active_goal_blocks_the_stop(monkeypatch, capsys, tmp_path):
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET)
    assert rc == 0
    doc = _decision(out)
    assert doc.get("decision") == "block"
    assert "keep working" in doc["reason"]
    assert "play-test of every feature" in doc["reason"]


def test_awaiting_user_never_blocks(monkeypatch, capsys, tmp_path):
    v = CompletionVerdict(goal_active=True, complete=False,
                          awaiting_user=True, missing=["a decision"])
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, v)
    assert _decision(out) == {}


def test_no_active_goal_never_blocks(monkeypatch, capsys, tmp_path):
    v = CompletionVerdict(goal_active=False, complete=False, missing=["x"])
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, v)
    assert _decision(out) == {}


def test_complete_goal_never_blocks(monkeypatch, capsys, tmp_path):
    v = CompletionVerdict(goal_active=True, complete=True)
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, v)
    assert _decision(out) == {}


def test_off_switch(monkeypatch, capsys, tmp_path):
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET,
                        env={"SADDLE_GOAL_KEEPER": "0"})
    assert _decision(out) == {}


def test_cap_alerts_then_stays_quiet_no_loop(monkeypatch, capsys, tmp_path):
    from saddle.bubble import recent_bubbles

    # 3 consecutive blocks, then the 4th turn alerts instead of blocking.
    for i in range(3):
        rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET)
        assert _decision(out).get("decision") == "block", f"turn {i} should block"
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET)
    assert _decision(out) == {}
    alerts = [b for b in recent_bubbles(_CTX, level="alert")
              if b.meta.get("keeper_capped")]
    assert len(alerts) == 1 and "stuck" in alerts[0].text
    # STICKY: on the SAME user turn (same anchor uuid) it must NOT re-arm — it
    # stays quiet instead of the old 3-on / 1-off perpetual loop.
    for _ in range(3):
        rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET)
        assert _decision(out) == {}, "same user turn must not block again"


def test_new_user_message_re_arms_the_keeper(monkeypatch, capsys, tmp_path):
    # Cap out the first user turn.
    for _ in range(4):
        _run_stop(monkeypatch, capsys, tmp_path, _UNMET)
    # A NEW user instruction (new anchor uuid) puts the human back in the loop —
    # the keeper re-arms and drives the fresh directive.
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET,
                        user_text="ok keep going and finish it", user_uuid="u2")
    assert _decision(out).get("decision") == "block"


def test_user_stop_directive_stands_down(monkeypatch, capsys, tmp_path):
    # The goal is still unmet, but the user just said stop — the keeper must
    # honor it and NOT override the command.
    for text in ("stop the build", "stop", "hold on", "pause"):
        rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET,
                            user_text=text, user_uuid="stop-" + text[:4])
        assert _decision(out) == {}, f"{text!r} should stand the keeper down"


def test_mixed_stop_then_go_is_not_a_halt(monkeypatch, capsys, tmp_path):
    # "stop X ... keep going with Y" is a GO — the keeper still drives it.
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET,
                        user_text="stop the goal keeper from overriding me, "
                                  "then keep going and rebuild the class",
                        user_uuid="mixed1")
    assert _decision(out).get("decision") == "block"


def test_progress_resets_the_cap_counter(monkeypatch, capsys, tmp_path):
    rc, out = _run_stop(monkeypatch, capsys, tmp_path, _UNMET)
    assert _decision(out).get("decision") == "block"
    done = CompletionVerdict(goal_active=True, complete=True)
    _run_stop(monkeypatch, capsys, tmp_path, done)
    from saddle.stop_hook import _keeper_count

    assert _keeper_count("s1") == 0
