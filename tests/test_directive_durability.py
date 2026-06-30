"""Directive durability gate — closes design_issues Gap 4.

Not every ``directive``-kind ask is a STANDING rule; many are task-local
instructions ("respond with only JSON") that, persisted as policy, get enforced
on every future task out of context. These pin that the gate splits durable from
task-local, fails SAFE toward not-promoting, and that the orchestrator persists
only the standing ones.
"""
from __future__ import annotations

import asyncio

import pytest

from saddle.context import Context
from saddle.intake import classify_directive_durability
from saddle.llm import policy
from saddle.models import DIRECTIVE, TASK, Intake, Item
from saddle.orchestrator import _promote_directives

CTX = Context(tenant="acme", project="game")


class FakeCaller:
    """Returns a canned JSON verdict; records calls. ``boom`` raises (to prove the
    gate fails safe). Mirrors the intake test callers."""

    def __init__(self, response: str = "{}", *, boom: bool = False):
        self.response = response
        self.boom = boom
        self.calls: list[dict] = []

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        if self.boom:
            raise RuntimeError("provider down")
        self.calls.append({"system": system, "prompt": prompt, "label": label})
        return self.response


def _dir(text: str) -> Item:
    return Item(kind=DIRECTIVE, ask=text)


def _classify(items, caller):
    return asyncio.run(classify_directive_durability(caller, items, CTX))


def test_splits_standing_from_task_local():
    items = [_dir("always fail loud"), _dir("respond with only JSON")]
    caller = FakeCaller(
        '{"verdicts": [{"index": 1, "durability": "standing"}, '
        '{"index": 2, "durability": "task"}]}'
    )
    res = _classify(items, caller)
    assert [i.ask for i in res["standing"]] == ["always fail loud"]
    assert [i.ask for i in res["task"]] == ["respond with only JSON"]
    assert res["checked"] is True


def test_unclassified_item_defaults_to_task():
    items = [_dir("always fail loud"), _dir("make THIS design game-agnostic")]
    # model only classifies item 1 -> item 2 must default to task (not promoted).
    caller = FakeCaller('{"verdicts": [{"index": 1, "durability": "standing"}]}')
    res = _classify(items, caller)
    assert [i.ask for i in res["standing"]] == ["always fail loud"]
    assert [i.ask for i in res["task"]] == ["make THIS design game-agnostic"]
    assert res["checked"] is False                     # coverage incomplete -> surfaced


def test_fails_safe_to_task_when_classify_errors():
    items = [_dir("a"), _dir("b")]
    res = _classify(items, FakeCaller(boom=True))
    assert res["standing"] == []                       # promote NOTHING on failure
    assert [i.ask for i in res["task"]] == ["a", "b"]
    assert res["checked"] is False


def test_no_directive_items_short_circuits():
    res = _classify([], FakeCaller(boom=True))          # boom never reached
    assert res == {"standing": [], "task": [], "checked": True}


# -- orchestrator promotes only the standing ones -----------------------------

@pytest.fixture
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SADDLE_CONFIG_DIR", str(tmp_path))
    base = tmp_path / "llm_policy.json"
    base.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SADDLE_LLM_POLICY", str(base))
    policy.reset_policy_cache()
    yield
    policy.reset_policy_cache()


def test_orchestrator_promotes_only_standing(_isolate_config):
    intake = Intake(raw_prompt="x", items=[
        _dir("never delete code without classifying it"),   # standing
        _dir("respond with only the JSON object"),          # task-local
        Item(kind=TASK, ask="build the thing"),             # not a directive
    ])
    caller = FakeCaller(
        '{"verdicts": [{"index": 1, "durability": "standing"}, '
        '{"index": 2, "durability": "task"}]}'
    )
    promoted = asyncio.run(_promote_directives(CTX, intake, "project", caller))
    assert promoted == ["never delete code without classifying it"]
    effective = policy.directives(CTX)
    assert "never delete code without classifying it" in effective
    assert "respond with only the JSON object" not in effective   # Gap 4: not persisted
