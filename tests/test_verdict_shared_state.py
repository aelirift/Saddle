"""Gap 5 — the completion gate's verdict is shared state for Stage 3.

Pins: the Stop hook persists its newest verdict per session; Stage 3's
design audit loads it and enriches the audited GOAL with the gate's
still-missing items as a binding in-goal declaration, so a forced
continuation is never re-judged as drift by a stage reading older state.
"""

from __future__ import annotations

import asyncio

from saddle.completion import CompletionVerdict, latest_verdict, persist_verdict
from saddle.context import Context

_CTX = Context(tenant="acme", project="game")


def _home(monkeypatch, tmp_path):
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")


def test_persist_then_load_roundtrip(monkeypatch, tmp_path) -> None:
    _home(monkeypatch, tmp_path)
    v = CompletionVerdict(goal_active=True, complete=False,
                          missing=["drive the vendor buy", "trade panel"])
    persist_verdict("s1", v)
    got = latest_verdict("s1")
    assert got is not None
    assert got.goal_active and not got.complete
    assert got.missing == ["drive the vendor buy", "trade panel"]
    assert latest_verdict("other-session") is None


def test_stage3_goal_carries_the_missing_items(monkeypatch, tmp_path) -> None:
    _home(monkeypatch, tmp_path)
    persist_verdict("s1", CompletionVerdict(
        goal_active=True, complete=False, missing=["close the mail gap"]))

    seen: dict = {}

    async def _fake_audit(goal, approach, ctx=None, **kw):
        seen["goal"] = goal

        class _V:
            has_issues = False

        return _V()

    async def _fake_settle(goal, approach, ctx=None, **kw):
        class _D:
            id = "d1"
            summary = "s"

        return _D()

    import saddle.design as design

    monkeypatch.setattr(design, "audit_proposal", _fake_audit)
    monkeypatch.setattr(design, "settle_approach", _fake_settle)
    from saddle.doctrine_hook import _design_outcome

    _design_outcome(_CTX, "finish the sweep", "close the mail gap now",
                    session="s1")
    assert "close the mail gap" in seen["goal"]
    assert "in-goal by definition" in seen["goal"]


def test_stage3_goal_untouched_without_active_gap(monkeypatch, tmp_path) -> None:
    _home(monkeypatch, tmp_path)
    persist_verdict("s1", CompletionVerdict(goal_active=True, complete=True))

    seen: dict = {}

    async def _fake_audit(goal, approach, ctx=None, **kw):
        seen["goal"] = goal

        class _V:
            has_issues = True
            issues = ["x"]

        return _V()

    import saddle.design as design

    monkeypatch.setattr(design, "audit_proposal", _fake_audit)
    from saddle.doctrine_hook import _design_outcome

    _design_outcome(_CTX, "finish the sweep", "some approach", session="s1")
    assert seen["goal"] == "finish the sweep"
