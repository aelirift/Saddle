"""Multi-project scope routing (mediator design §4).

Covers: the learned project registry, the per-session active scope set, the
doctrine fence widened to the scope set (+ the scratch-space exemption), the
intake router stamping items with sibling projects, and the turn-end harvest
filing lessons into each project's own ledger.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from saddle import registry
from saddle.context import Context
from saddle.focus import active_roots, active_scope_asks, active_scopes, record_active_scopes


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    return tmp_path


# --- registry ----------------------------------------------------------------

def test_register_learns_and_is_idempotent(tmp_path):
    root = tmp_path / "projects" / "rayxiv4"
    root.mkdir(parents=True)
    slug = registry.register_root("aeli", root, ts=100.0)
    assert slug == "rayxiv4"
    assert registry.known_projects("aeli") == {"rayxiv4": str(root.resolve())}
    # second sighting refreshes, never duplicates
    assert registry.register_root("aeli", root, ts=200.0) == "rayxiv4"
    assert list(registry.known_projects("aeli")) == ["rayxiv4"]


def test_registry_slug_matches_context_project(tmp_path):
    root = tmp_path / "My Project"
    root.mkdir()
    slug = registry.register_root("aeli", root, ts=1.0)
    assert slug == Context(tenant="aeli", project=root.name).project


def test_project_for_path_longest_root_wins(tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "nested"
    inner.mkdir(parents=True)
    registry.register_root("aeli", outer, ts=1.0)
    registry.register_root("aeli", inner, ts=1.0)
    assert registry.project_for_path("aeli", inner / "x.py") == "nested"
    assert registry.project_for_path("aeli", outer / "y.py") == "outer"
    assert registry.project_for_path("aeli", tmp_path / "elsewhere.py") is None


def test_unconfirmed_then_confirm(tmp_path):
    root = tmp_path / "newproj"
    root.mkdir()
    registry.register_root("aeli", root, ts=1.0)
    assert registry.unconfirmed("aeli") == ["newproj"]
    assert registry.confirm("aeli", "newproj") is True
    assert registry.unconfirmed("aeli") == []


def test_tenants_are_isolated(tmp_path):
    root = tmp_path / "p1"
    root.mkdir()
    registry.register_root("aeli", root, ts=1.0)
    assert registry.known_projects("other") == {}
    assert registry.project_for_path("other", root / "f.py") is None


# --- the active scope set ------------------------------------------------------

def test_scope_set_roundtrip(tmp_path):
    record_active_scopes("s1", "aeli", ["rayxiv4", "saddle"],
                         asks={"saddle": ["audit the drift catches"]})
    assert active_scopes("s1", "aeli") == ["rayxiv4", "saddle"]
    assert active_scope_asks("s1", "aeli") == {"saddle": ["audit the drift catches"]}
    # a different tenant reads nothing — the hard fence between owners
    assert active_scopes("s1", "other") == []


def test_active_roots_resolve_via_registry(tmp_path, monkeypatch):
    focus = tmp_path / "rayxiv4"
    sib = tmp_path / "saddle"
    focus.mkdir(); sib.mkdir()
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(focus))
    monkeypatch.setenv("SADDLE_TENANT", "aeli")
    monkeypatch.setenv("SADDLE_PROJECT", "rayxiv4")
    registry.register_root("aeli", sib, ts=1.0)
    record_active_scopes("s2", "aeli", ["rayxiv4", "saddle"])
    roots = active_roots("s2")
    assert str(focus) in roots[0]
    assert str(sib.resolve()) in roots


# --- the doctrine fence over the scope set -------------------------------------

def test_fence_allows_second_active_project(tmp_path, monkeypatch):
    import tempfile

    from saddle.doctrine import gate_tool_call

    # pytest tmp_path lives under the real temp tree; point the scratch
    # exemption elsewhere so this test exercises the fence, not the exemption.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "faketmp"))
    focus = tmp_path / "rayxiv4"; focus.mkdir()
    sib = tmp_path / "saddle"; sib.mkdir()
    inside = gate_tool_call(
        "Edit", {"file_path": str(sib / "src" / "x.py")},
        project_root=str(focus), extra_roots=(str(sib),),
    )
    assert inside.allowed and inside.rule_id is None  # clean, not a warn
    outside = gate_tool_call(
        "Edit", {"file_path": str(tmp_path / "third" / "y.py")},
        project_root=str(focus), extra_roots=(str(sib),),
    )
    assert outside.allowed and outside.severity == "warn"


def test_fence_exempts_scratch_space(tmp_path):
    import tempfile

    from saddle.doctrine import gate_tool_call

    focus = tmp_path / "proj"; focus.mkdir()
    v = gate_tool_call(
        "Write", {"file_path": f"{tempfile.gettempdir()}/claude-x/scratch/probe.mjs"},
        project_root=str(focus),
    )
    assert v.allowed and v.severity != "warn"


def test_fence_still_blocks_cross_project_delete(tmp_path):
    from saddle.doctrine import gate_tool_call

    focus = tmp_path / "proj"; focus.mkdir()
    sib = tmp_path / "sib"; sib.mkdir()
    v = gate_tool_call(
        "Bash", {"command": f"rm {tmp_path / 'third' / 'f.py'}"},
        project_root=str(focus), extra_roots=(str(sib),),
    )
    assert not v.allowed


# --- intake routing -------------------------------------------------------------

def _scope_caller(routes: dict[int, str]):
    """A caller that answers ONLY the scope call; itemize/audit are not used."""

    async def call(system, prompt, *, json_mode=False, label=""):
        assert label == "intake/scope"
        verdicts = [
            {"index": i, "scope": s, "reason": "test"} for i, s in routes.items()
        ]
        return json.dumps({"verdicts": verdicts})

    return call


def test_classify_scope_stamps_sibling_and_warns_outside(tmp_path, monkeypatch):
    from saddle.intake import _classify_scope
    from saddle.models import Item

    sib = tmp_path / "saddle"; sib.mkdir()
    registry.register_root("aeli", sib, ts=1.0)
    ctx = Context(tenant="aeli", project="rayxiv4")
    items = [
        Item(kind="task", ask="fix the minimap"),
        Item(kind="task", ask="audit saddle's drift catches"),
        Item(kind="task", ask="rewrite my router firmware"),
        Item(kind="context", ask="the demo felt cluttered"),
    ]
    info = asyncio.run(_classify_scope(
        _scope_caller({1: "focus", 2: "saddle", 3: "outside"}), items, ctx,
    ))
    assert items[1].project == "saddle"          # routed sibling — stamped
    assert items[0].project == ""                # ambient — untouched
    assert info["item_projects"] == {2: "saddle"}
    assert [o["n"] for o in info["out_of_focus"]] == [3]
    assert "unrecognized" in info["scope_warning"] or "outside" in info["scope_warning"]
    assert info["scope_checked"] is True


def test_items_persist_their_routed_project(tmp_path, monkeypatch):
    from saddle.models import Intake, Item
    from saddle.store import SqliteStore

    store = SqliteStore(tmp_path / "s.db")
    ctx = Context(tenant="aeli", project="rayxiv4")
    intake = Intake(raw_prompt="x", items=[
        Item(kind="task", ask="ambient work"),
        Item(kind="task", ask="sibling work", project="saddle"),
    ])
    store.save_intake(ctx, intake)
    items = store.list_items(ctx)
    by_ask = {i.ask: i.project for i in items}
    # ambient item carries the ambient project; the routed one keeps its route
    assert by_ask["ambient work"] == "rayxiv4"
    saddle_items = store.list_items(Context(tenant="aeli", project="saddle"))
    assert [i.ask for i in saddle_items] == ["sibling work"]
    store.close()


# --- per-project lesson harvest ---------------------------------------------------

def test_turn_issues_group_by_project(tmp_path, monkeypatch):
    from saddle.bubble import emit_bubble
    from saddle.stop_hook import _turn_issues

    a = Context(tenant="aeli", project="rayxiv4")
    b = Context(tenant="aeli", project="saddle")
    emit_bubble(a, "design issues", stage="design", session="s1",
                meta={"issues": ["band-aid in the zone loader"]})
    emit_bubble(b, "design issues", stage="design", session="s1",
                meta={"issues": ["gate swallowed a timeout"]})
    grouped = _turn_issues(a, "s1", 0.0)
    assert grouped == {
        "rayxiv4": ["band-aid in the zone loader"],
        "saddle": ["gate swallowed a timeout"],
    }
