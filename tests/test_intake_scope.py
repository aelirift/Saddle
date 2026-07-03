"""Intake focus gate — only ACTIONS (tasks) can cross the project boundary.

Discussing, asking about, or referencing another project is not drift; only a
task that would modify another project's code is. These tests pin that seam:
a task-free prompt never even calls the model, only tasks are weighed, and an
out-of-focus task is surfaced at its ORIGINAL position in the user's full item
list (not the filtered task list). The LLM is faked so the classifier's wiring
— filter, dispatch, index-map, warn — is exercised deterministically.
"""
from __future__ import annotations

import asyncio
import json

from saddle.context import Context
from saddle.intake import _classify_scope
from saddle.models import Item

CTX = Context(tenant="acme", project="game")


class FakeCaller:
    """Records every call; returns a canned JSON verdict. ``forbidden=True``
    turns any call into a failure — the assertion that no model is consulted
    when there is nothing to weigh."""

    def __init__(self, response: str = "{}", *, forbidden: bool = False):
        self.response = response
        self.forbidden = forbidden
        self.calls: list[dict] = []

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        if self.forbidden:
            raise AssertionError("focus LLM was called with no task to weigh")
        self.calls.append({"system": system, "prompt": prompt, "label": label})
        return self.response


def _scope(items, caller):
    return asyncio.run(_classify_scope(caller, items, CTX))


def test_task_free_prompt_skips_the_focus_call():
    # Questions, context, and decisions about another project are discussion,
    # not drift — there is nothing to act on, so the model is never consulted.
    items = [
        Item(kind="question", ask="what does the other project's parser do?"),
        Item(kind="context", ask="the other project uses redis"),
        Item(kind="decision", ask="pick a backend for the other project"),
    ]
    caller = FakeCaller(forbidden=True)
    res = _scope(items, caller)
    assert caller.calls == []                       # no model call at all
    assert res["out_of_focus"] == [] and res["scope_warning"] == ""
    assert res["scope_checked"] is True             # nothing to check = checked


def test_directive_about_other_project_is_not_an_action():
    # A standing rule that mentions another project is still discussion: it does
    # not, by itself, modify that project's code, so it is not weighed.
    items = [Item(kind="directive", ask="prefer tabs in the other repo")]
    caller = FakeCaller(forbidden=True)
    res = _scope(items, caller)
    assert caller.calls == []
    assert res["scope_warning"] == "" and res["scope_checked"] is True


def test_only_tasks_are_sent_and_indices_map_back():
    items = [
        Item(kind="task", ask="fix the camera-follow bug in game"),     # idx 0, in focus
        Item(kind="question", ask="how does the other repo cache?"),    # idx 1, discussion
        Item(kind="task", ask="refactor the other repo's parser.py"),   # idx 2, out of focus
    ]
    # The model only ever sees the two TASKS, numbered 1 and 2 in its listing.
    verdict = json.dumps({"verdicts": [
        {"index": 1, "scope": "in_focus", "reason": "this project"},
        {"index": 2, "scope": "out_of_focus", "reason": "another repo's code"},
    ]})
    caller = FakeCaller(verdict)
    res = _scope(items, caller)

    # the discussion question never reached the model
    assert len(caller.calls) == 1
    sent = caller.calls[0]["prompt"]
    assert "parser.py" in sent and "how does the other repo cache" not in sent

    # the out-of-focus task is reported at its ORIGINAL position (item 3 of 3)
    assert res["out_of_focus"] == [
        {"n": 3, "ask": "refactor the other repo's parser.py",
         "reason": "another repo's code"}
    ]
    assert res["scope_checked"] is True
    assert "1 of 2 action(s) target work outside" in res["scope_warning"]
    assert "outside game" in res["scope_warning"]


def test_in_focus_only_tasks_produce_no_warning():
    items = [Item(kind="task", ask="add momentum to the warrior kit in game")]
    verdict = json.dumps(
        {"verdicts": [{"index": 1, "scope": "in_focus", "reason": "this project"}]})
    res = _scope(items, FakeCaller(verdict))
    assert res["out_of_focus"] == [] and res["scope_warning"] == ""
    assert res["scope_checked"] is True


def test_unclassified_task_surfaces_review_warning():
    # The model returns no verdict for the lone task -> scope_checked False,
    # surfaced as a manual-review warning rather than silently waved through.
    items = [Item(kind="task", ask="edit the other repo's config")]
    res = _scope(items, FakeCaller(json.dumps({"verdicts": []})))
    assert res["out_of_focus"] == []
    assert res["scope_checked"] is False
    assert "did not classify every action" in res["scope_warning"]
