"""crossproject — explicit, persistent cross-project authorization grants the
doctrine scope-fence consults.

The scope-fence (see test_doctrine_guard / test_doctrine_hook) blocks any edit
outside the focus project. ``actions_from_tool`` attaches no
``cross_project_task`` evidence, so a *grant* is the only sanctioned way to
authorize a cross-project move through the tool path. These tests pin:

* the grant store round-trips and resolves roots,
* tenant scoping and path containment,
* ``authorize_tool``'s per-target rule (every out-of-focus target must be
  granted, else the whole call stays blocked),
* and that the PreToolUse hook honours a grant for the scope-fence -- and ONLY
  the scope-fence (a grant never rescues a code-delete or any non-scope rule).

``SADDLE_HOME`` is redirected to a tmp dir so grants never touch the real
install; ``SADDLE_CODE_ROOT`` pins the focus for the hook path.
"""
from __future__ import annotations

import io
import json

import pytest

from saddle import crossproject


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Park the saddle data dir (and thus the grants file) under tmp_path."""
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    yield


def _projects(tmp_path):
    a = tmp_path / "projA"
    b = tmp_path / "projB"
    c = tmp_path / "projC"
    for p in (a, b, c):
        (p / "src").mkdir(parents=True)
    return a, b, c


# --- the grant store --------------------------------------------------------

def test_grant_roundtrips_and_persists(tmp_path):
    a, b, _ = _projects(tmp_path)
    g = crossproject.grant([str(a), str(b)], tenant="aeli", reason="why", source="cli")
    assert g.tenant == "aeli" and g.reason == "why" and g.ts > 0
    loaded = crossproject.load_grants()
    assert len(loaded) == 1
    assert set(loaded[0].roots) == {str(a.resolve()), str(b.resolve())}
    assert (tmp_path / "cross_project.json").exists()


def test_grant_resolves_roots(tmp_path, monkeypatch):
    a, _, _ = _projects(tmp_path)
    monkeypatch.chdir(tmp_path)
    crossproject.grant(["projA"])  # relative -- must be resolved to absolute
    assert crossproject.load_grants()[0].roots == (str(a.resolve()),)


def test_grants_are_additive(tmp_path):
    a, b, c = _projects(tmp_path)
    crossproject.grant([str(a)])
    crossproject.grant([str(b), str(c)])
    assert len(crossproject.load_grants()) == 2


def test_revoke_all_clears(tmp_path):
    a, b, _ = _projects(tmp_path)
    crossproject.grant([str(a), str(b)])
    assert crossproject.revoke_all() == 1
    assert crossproject.load_grants() == []


def test_load_grants_missing_file_is_empty(tmp_path):
    assert crossproject.load_grants() == []


def test_load_grants_tolerates_garbage(tmp_path):
    (tmp_path / "cross_project.json").write_text("{not json", encoding="utf-8")
    assert crossproject.load_grants() == []


# --- tenant scoping ---------------------------------------------------------

def test_authorized_roots_tenant_filter(tmp_path):
    a, b, c = _projects(tmp_path)
    crossproject.grant([str(a)], tenant="aeli")
    crossproject.grant([str(b)], tenant="bob")
    crossproject.grant([str(c)], tenant="")        # empty -> every tenant
    aeli = crossproject.authorized_roots("aeli")
    assert str(a.resolve()) in aeli and str(c.resolve()) in aeli
    assert str(b.resolve()) not in aeli            # bob's grant is invisible to aeli


def test_authorized_roots_wildcard_tenant(tmp_path):
    a, _, _ = _projects(tmp_path)
    crossproject.grant([str(a)], tenant="*")
    assert str(a.resolve()) in crossproject.authorized_roots("anyone")


# --- containment + is_authorized -------------------------------------------

def test_under_containment(tmp_path):
    a, b, _ = _projects(tmp_path)
    assert crossproject._under(str(a), str(a / "src" / "x.py"))
    assert crossproject._under(str(a), str(a))          # the root itself
    assert not crossproject._under(str(a), str(b / "x.py"))


def test_is_authorized_requires_focus_and_target(tmp_path):
    a, b, c = _projects(tmp_path)
    crossproject.grant([str(a), str(b)], tenant="aeli")
    # focus in A, target in B -> both granted -> authorized
    assert crossproject.is_authorized(str(b / "src/x.py"), str(a), tenant="aeli")
    # focus in A, target in C -> C ungranted -> not authorized
    assert not crossproject.is_authorized(str(c / "src/x.py"), str(a), tenant="aeli")
    # wrong tenant sees no grants
    assert not crossproject.is_authorized(str(b / "src/x.py"), str(a), tenant="bob")


def test_is_authorized_false_without_grants(tmp_path):
    a, b, _ = _projects(tmp_path)
    assert not crossproject.is_authorized(str(b / "x.py"), str(a), tenant="aeli")


# --- authorize_tool ---------------------------------------------------------

def test_authorize_tool_grants_cross_project_edit(tmp_path):
    a, b, _ = _projects(tmp_path)
    crossproject.grant([str(a), str(b)], tenant="*")
    note = crossproject.authorize_tool(
        "Edit", {"file_path": str(b / "src/x.py")}, focus=str(a))
    assert note is not None and "cross-project authorized" in note


def test_authorize_tool_blocks_ungranted_target(tmp_path):
    a, b, c = _projects(tmp_path)
    crossproject.grant([str(a), str(b)], tenant="*")
    note = crossproject.authorize_tool(
        "Edit", {"file_path": str(c / "src/x.py")}, focus=str(a))
    assert note is None


def test_authorize_tool_no_targets_returns_none(tmp_path):
    a, b, _ = _projects(tmp_path)
    crossproject.grant([str(a), str(b)], tenant="*")
    # Read never mutates -> actions_from_tool yields nothing -> None
    assert crossproject.authorize_tool(
        "Read", {"file_path": str(b / "x.py")}, focus=str(a)) is None


def test_authorize_tool_partial_grant_blocks_whole_call(tmp_path):
    a, b, c = _projects(tmp_path)
    crossproject.grant([str(a), str(b)], tenant="*")
    # one rm target granted (B), one ungranted (C) -> the whole call stays blocked
    cmd = f"rm {b / 'src/x.py'} {c / 'src/y.py'}"
    assert crossproject.authorize_tool("Bash", {"command": cmd}, focus=str(a)) is None


def test_authorize_tool_in_focus_targets_need_no_grant(tmp_path):
    a, _, _ = _projects(tmp_path)
    # no grant at all; an in-focus edit has no out-of-focus target to authorize
    note = crossproject.authorize_tool(
        "Edit", {"file_path": str(a / "src/x.py")}, focus=str(a))
    assert note is not None  # nothing violated -> a (vacuous) authorization note


# --- the PreToolUse hook honours grants for the fence, and only the fence ----

def _run_hook(payload, focus, monkeypatch, capsys):
    import tempfile

    # These tests emulate real sibling PROJECTS under pytest's tmp_path, which
    # lives in the real temp tree — point the fence's scratch exemption at an
    # empty dir so the siblings read as projects, not scratch space.
    monkeypatch.setattr(tempfile, "tempdir", str(focus) + "-faketmp")
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(focus))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    from saddle import doctrine_hook
    rc = doctrine_hook.main()
    return rc, capsys.readouterr()


def test_hook_allows_granted_cross_project_edit(tmp_path, monkeypatch, capsys):
    a, b, _ = _projects(tmp_path)
    crossproject.grant([str(a), str(b)], tenant="*")
    rc, out = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": str(b / "src/x.py")}},
        focus=a, monkeypatch=monkeypatch, capsys=capsys)
    assert rc == 0
    assert out.out.strip() == ""               # no deny emitted -> allowed
    assert "cross-project ALLOW" in out.err     # and it says so, loudly


def test_hook_warns_ungranted_sibling_edit_even_with_a_grant(tmp_path, monkeypatch, capsys):
    a, b, c = _projects(tmp_path)
    crossproject.grant([str(a), str(b)], tenant="*")   # grants A<->B, not C
    rc, out = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": str(c / "src/x.py")}},
        focus=a, monkeypatch=monkeypatch, capsys=capsys)
    assert rc == 0
    assert out.out.strip() == ""                  # an EDIT is never denied now
    # C is under no grant -> a genuine wander -> the loud ALERT surface, not a notice
    assert "WARN out-of-focus" in out.err


def test_hook_no_grant_warns_sibling_edit(tmp_path, monkeypatch, capsys):
    a, b, _ = _projects(tmp_path)
    rc, out = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": str(b / "src/x.py")}},
        focus=a, monkeypatch=monkeypatch, capsys=capsys)
    assert rc == 0
    assert out.out.strip() == ""                  # an EDIT is never denied now
    assert "WARN out-of-focus" in out.err         # no grant -> loud alert, never silent


def test_hook_grant_does_not_override_code_delete(tmp_path, monkeypatch, capsys):
    """A grant overrides ONLY the scope-fence rules (SCOPE_FENCE_RULE_IDS). An
    in-focus code delete trips ``no-unwired-delete`` -- a non-scope code-safety
    invariant -- so even with a grant covering the focus, that block must stand."""
    a, b, _ = _projects(tmp_path)
    crossproject.grant([str(a), str(b)], tenant="*")
    rc, out = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm src/saddle/x.py"}},
        focus=a, monkeypatch=monkeypatch, capsys=capsys)
    assert rc == 0
    doc = json.loads(out.out)["hookSpecificOutput"]
    assert doc["permissionDecision"] == "deny"
    assert "no-unwired-delete" in doc["permissionDecisionReason"]
