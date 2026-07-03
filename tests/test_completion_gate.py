"""The completion gate — a "finished" claim is judged against the goal AS THE
USER MEANT IT (broad clauses + still-open ledger asks), never against the
sub-list the reply enumerates. Founding incident: a goal auto-cleared on a
confident summary while its "test everything"/"AAA quality" clauses were open.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from saddle.completion import CompletionVerdict, audit_completion
from saddle.context import Context

_CTX = Context(tenant="acme", project="game")


def _canned(doc: dict):
    calls = {"n": 0}

    async def call(system, prompt, *, json_mode=False, label=""):
        calls["n"] += 1
        assert label == "completion/gate"
        # The high-level-interpretation instruction must ride every call.
        assert "Never narrow the goal" in system
        return json.dumps(doc)

    return call, calls


def test_overclaim_is_flagged_with_missing_items():
    call, _ = _canned({
        "claims_done": True, "complete": False,
        "missing": ["every feature play-tested", "the quality bar reached"],
    })
    v = asyncio.run(audit_completion(
        "fix the 8 issues and play test every feature and fix all",
        "All eight issues are fixed and verified end-to-end.",
        _CTX, caller=call,
    ))
    assert v.overclaim
    assert "every feature play-tested" in v.missing


def test_honest_status_report_is_not_a_claim():
    call, _ = _canned({"claims_done": False, "complete": False, "missing": []})
    v = asyncio.run(audit_completion(
        "fix everything", "Fixed 3 of 8; the map and camera remain.", _CTX,
        caller=call,
    ))
    assert not v.overclaim


def test_justified_completion_stays_silent():
    call, _ = _canned({"claims_done": True, "complete": True, "missing": []})
    v = asyncio.run(audit_completion(
        "rename the flag", "Renamed, tests green, verified in the app.", _CTX,
        caller=call,
    ))
    assert not v.overclaim


def test_empty_reply_skips_the_llm():
    async def boom(*a, **k):
        raise AssertionError("no call expected")

    v = asyncio.run(audit_completion("goal", "   ", _CTX, caller=boom))
    assert v == CompletionVerdict()


def test_stop_hook_bubbles_overclaim_for_the_drain(monkeypatch, capsys, tmp_path):
    """End to end: an overclaiming turn produces a STAGE_COMPLETION alert whose
    meta carries origin=turn-end, so the next turn's drain delivers it to the
    agent."""
    from saddle.bubble import recent_bubbles

    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(tmp_path))
    monkeypatch.setenv("SADDLE_HOOK_CODE", "0")
    monkeypatch.setenv("SADDLE_HOOK_LESSON", "0")
    monkeypatch.setenv("SADDLE_HOOK_VOICE", "0")
    monkeypatch.setenv("SADDLE_HOOK_DESIGN", "0")

    async def _overclaim(goal, reply, ctx=None, **kw):
        return CompletionVerdict(
            claims_done=True, complete=False,
            missing=["the play-test of every feature"],
        )

    monkeypatch.setattr("saddle.completion.audit_completion", _overclaim)

    tp = tmp_path / "t.jsonl"
    lines = [
        {"type": "user", "uuid": "u1",
         "message": {"content": [{"type": "text", "text": "fix ALL of it"}]}},
        {"type": "assistant", "uuid": "a1",
         "message": {"content": [{"type": "text",
                                  "text": "All done and verified end-to-end."}]}},
    ]
    tp.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"session_id": "s1", "transcript_path": str(tp)}
    )))
    from saddle import stop_hook

    assert stop_hook.main() == 0
    alerts = [b for b in recent_bubbles(_CTX, level="alert")
              if b.stage == "completion"]
    assert len(alerts) == 1
    assert "finished" in alerts[0].text and "still open" in alerts[0].text
    assert alerts[0].meta.get("origin") == "turn-end"
    assert "play-test of every feature" in alerts[0].text
