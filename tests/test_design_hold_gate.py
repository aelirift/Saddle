"""The design-hold gate in the doctrine hook (Part 1): the auto-settle is
suppressed under an active HOLD, the deterministic hold-deny decision, and a
smoke run of the PreToolUse hook end to end (DENY under hold, ALLOW otherwise).
"""

from __future__ import annotations

import io
import json

import pytest

from saddle.context import Context
from saddle.hold import (
    HOLD_AUTONOMOUS,
    HOLD_HELD_STATE,
    HoldPosture,
    write_posture,
)

_CTX = Context(tenant="acme", project="game")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(tmp_path))
    return tmp_path


# -- _design_outcome: HOLD suppresses the auto-settle; else it settles --------

@pytest.fixture
def _stub_audit_and_settle(monkeypatch):
    from saddle.design import AuditVerdict
    from saddle.models import Design

    async def _clean_audit(goal, approach, ctx=None, **kw):
        return AuditVerdict(ok=True, issues=[])

    settled: list[str] = []

    async def _settle(goal, approach, ctx=None, *, approved_by="converged", **kw):
        settled.append(approach)
        return Design(ask=goal, id="design_s", status="final")

    monkeypatch.setattr("saddle.design.audit_proposal", _clean_audit)
    monkeypatch.setattr("saddle.design.settle_approach", _settle)
    return settled


def test_clean_audit_is_held_not_settled_under_hold(_stub_audit_and_settle):
    from saddle import doctrine_hook

    write_posture("s1", HoldPosture(state=HOLD_HELD_STATE, held={"goal": "g"}))
    out = doctrine_hook._design_outcome(_CTX, "g", "some concrete approach", "s1")
    assert out is not None
    assert out.meta.get("held") is True
    assert out.meta.get("settled") is not True
    assert _stub_audit_and_settle == []          # NOT settled under a hold


def test_clean_audit_settles_under_default(_stub_audit_and_settle):
    from saddle import doctrine_hook

    # No posture written -> DEFAULT
    out = doctrine_hook._design_outcome(_CTX, "g", "some concrete approach", "s2")
    assert out is not None
    assert out.meta.get("settled") is True
    assert _stub_audit_and_settle == ["some concrete approach"]


def test_clean_audit_settles_under_autonomous(_stub_audit_and_settle):
    from saddle import doctrine_hook

    write_posture("s3", HoldPosture(state=HOLD_AUTONOMOUS))
    out = doctrine_hook._design_outcome(_CTX, "g", "some concrete approach", "s3")
    assert out.meta.get("settled") is True
    assert _stub_audit_and_settle == ["some concrete approach"]


# -- _hold_denies_edit: the deterministic decision ---------------------------

class _Turn:
    approach = ""


def test_hold_denies_unapproved_code_edit():
    from saddle.doctrine_hook import _hold_denies_edit

    write_posture("s1", HoldPosture(state=HOLD_HELD_STATE, held={"goal": "g"}))
    assert _hold_denies_edit("Edit", "s1", _Turn()) is True        # fail-closed
    assert _hold_denies_edit("Write", "s1", _Turn()) is True
    assert _hold_denies_edit("Bash", "s1", _Turn()) is False       # not a code edit


def test_hold_opens_after_approval():
    from saddle.doctrine_hook import _hold_denies_edit

    write_posture("s1", HoldPosture(state=HOLD_HELD_STATE, held={"goal": "g"},
                                    approved_design_id="design_x"))
    assert _hold_denies_edit("Edit", "s1", _Turn()) is False       # approved


def test_default_and_autonomous_never_deny():
    from saddle.doctrine_hook import _hold_denies_edit

    assert _hold_denies_edit("Edit", "never-written", _Turn()) is False  # DEFAULT
    write_posture("s2", HoldPosture(state=HOLD_AUTONOMOUS))
    assert _hold_denies_edit("Edit", "s2", _Turn()) is False


# -- the PreToolUse hook, end to end (smoke) ---------------------------------

def _transcript(tmp_path):
    tp = tmp_path / "t.jsonl"
    lines = [
        {"type": "user", "uuid": "u1",
         "message": {"content": [{"type": "text", "text": "build it, but show me the plan first"}]}},
        {"type": "assistant", "uuid": "a1",
         "message": {"content": [{"type": "text", "text": "Here is my plan: do X then Y."}]}},
    ]
    tp.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return tp


def test_hook_denies_a_code_edit_under_hold(tmp_path, monkeypatch, capsys):
    from saddle import doctrine_hook

    write_posture("s1", HoldPosture(state=HOLD_HELD_STATE, held={"goal": "build it"}))
    tp = _transcript(tmp_path)
    payload = {
        "tool_name": "Edit",
        "tool_input": {"file_path": str(tmp_path / "x.py"), "old_string": "a",
                       "new_string": "b"},
        "session_id": "s1",
        "transcript_path": str(tp),
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert doctrine_hook.main() == 0
    out = capsys.readouterr().out
    doc = json.loads(out.strip().splitlines()[-1])
    assert doc["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "approve" in doc["hookSpecificOutput"]["permissionDecisionReason"].lower()


def test_hook_allows_a_code_edit_with_no_hold(tmp_path, monkeypatch):
    """DEFAULT posture + no transcript -> the design stage no-ops (no LLM) and the
    hook allows without wedging (no deny decision emitted)."""
    from saddle import doctrine_hook

    payload = {
        "tool_name": "Edit",
        "tool_input": {"file_path": str(tmp_path / "x.py"), "old_string": "a",
                       "new_string": "b"},
        "session_id": "s-default",
        "transcript_path": "",   # no transcript -> design stage returns early
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert doctrine_hook.main() == 0


def test_intake_hook_runs_without_wedging(tmp_path, monkeypatch):
    """The UserPromptSubmit hook returns 0 with the LLM stages (including the new
    review stage) disabled — a smoke that the wiring imports + runs clean."""
    from saddle import intake_hook

    for knob in ("SADDLE_HOOK_ITEMIZE", "SADDLE_HOOK_INTENT", "SADDLE_HOOK_REVIEW"):
        monkeypatch.setenv(knob, "0")
    payload = {"prompt": "please add a metrics endpoint to the server",
               "session_id": "s1", "transcript_path": ""}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert intake_hook.main() == 0
