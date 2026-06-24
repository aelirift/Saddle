"""doctrine — the tool-call passthrough (actions_from_tool / gate_tool_call) and
the PreToolUse hook entry point.

The gate proper is covered by ``test_doctrine_guard``; here we verify the *front
edge*: that a tool invocation is translated into the right Action(s), that
``gate_tool_call`` enforces them, and that ``saddle.doctrine_hook`` speaks the
PreToolUse deny/allow protocol. ``project_root`` / ``SADDLE_CODE_ROOT`` are
explicit so the scope-fence reasons about a synthetic root with no real FS.
"""
from __future__ import annotations

import io
import json

from saddle import doctrine as d
from saddle.doctrine import SEED_RULES, actions_from_tool, gate_tool_call

ROOT = "/work/focus"


def _gate(tool_name, tool_input):
    return gate_tool_call(tool_name, tool_input, project_root=ROOT, rules=SEED_RULES)


# --- actions_from_tool -----------------------------------------------------

def test_write_maps_to_write_path_action():
    acts = actions_from_tool("Write", {"file_path": "/work/focus/a.py"}, project_root=ROOT)
    assert [(a.nverb, a.target, a.target_kind) for a in acts] == [
        ("write", "/work/focus/a.py", "path")
    ]


def test_edit_family_maps_to_edit():
    for name, key in (
        ("Edit", "file_path"), ("MultiEdit", "file_path"),
        ("NotebookEdit", "notebook_path"), ("Update", "file_path"),
    ):
        acts = actions_from_tool(name, {key: "x.py"}, project_root=ROOT)
        assert len(acts) == 1 and acts[0].nverb == "edit", name


def test_write_without_path_is_noop():
    assert actions_from_tool("Write", {}, project_root=ROOT) == []


def test_read_and_grep_are_noops():
    assert actions_from_tool("Read", {"file_path": "/x.py"}, project_root=ROOT) == []
    assert actions_from_tool("Grep", {"pattern": "foo"}, project_root=ROOT) == []


def test_bash_rm_multiple_operands():
    acts = actions_from_tool("Bash", {"command": "rm a.py b.py"}, project_root=ROOT)
    assert [a.target for a in acts] == ["a.py", "b.py"]
    assert all(a.nverb == "delete" and a.target_kind == "path" for a in acts)


def test_bash_git_rm():
    acts = actions_from_tool("Bash", {"command": "git rm src/x.py"}, project_root=ROOT)
    assert [a.target for a in acts] == ["src/x.py"]


def test_bash_segments_split_on_operators():
    acts = actions_from_tool(
        "Bash", {"command": "rm -rf build && rm c.py"}, project_root=ROOT)
    assert [a.target for a in acts] == ["build", "c.py"]


def test_bash_non_delete_is_noop():
    assert actions_from_tool("Bash", {"command": "echo hi && ls"}, project_root=ROOT) == []


def test_bash_unparseable_is_noop():
    # an unbalanced quote makes shlex raise -> we assert on nothing we can't read
    assert actions_from_tool("Bash", {"command": "rm 'oops"}, project_root=ROOT) == []


# --- gate_tool_call --------------------------------------------------------

def test_gate_blocks_edit_outside_focus():
    v = _gate("Edit", {"file_path": "/work/sibling/x.py"})
    assert not v.allowed and v.rule_id == "stay-in-project-focus"


def test_gate_allows_edit_inside_focus():
    v = _gate("Edit", {"file_path": "/work/focus/src/x.py"})
    assert v.allowed


def test_gate_blocks_bash_rm_code():
    v = _gate("Bash", {"command": "rm src/saddle/llm/llm_pool.py"})
    assert not v.allowed and v.rule_id == "no-unwired-delete"


def test_gate_allows_bash_rm_non_code():
    v = _gate("Bash", {"command": "rm build/artifact.bin && rm notes.txt"})
    assert v.allowed


def test_gate_blocks_rm_of_sibling_code_on_scope():
    v = _gate("Bash", {"command": "rm /work/sibling/foo.py"})
    assert not v.allowed and v.rule_id == "stay-in-project-focus"


def test_gate_read_is_allowed():
    v = _gate("Read", {"file_path": "/work/sibling/x.py"})
    assert v.allowed and "does not mutate" in v.reason


# --- the PreToolUse hook ---------------------------------------------------

def _run_hook(payload, monkeypatch, capsys):
    monkeypatch.setenv("SADDLE_CODE_ROOT", ROOT)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    from saddle import doctrine_hook
    rc = doctrine_hook.main()
    return rc, capsys.readouterr()


def test_hook_denies_out_of_focus_edit(monkeypatch, capsys):
    rc, out = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": "/work/sibling/x.py"}},
        monkeypatch, capsys,
    )
    assert rc == 0
    doc = json.loads(out.out)
    dec = doc["hookSpecificOutput"]
    assert dec["permissionDecision"] == "deny"
    assert "stay-in-project-focus" in dec["permissionDecisionReason"]


def test_hook_denies_code_delete(monkeypatch, capsys):
    rc, out = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm src/saddle/x.py"}},
        monkeypatch, capsys,
    )
    assert rc == 0
    assert json.loads(out.out)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_allows_in_focus_edit(monkeypatch, capsys):
    rc, out = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": "/work/focus/src/x.py"}},
        monkeypatch, capsys,
    )
    assert rc == 0
    assert out.out.strip() == ""  # allow -> no decision emitted


def test_hook_allows_read(monkeypatch, capsys):
    rc, out = _run_hook(
        {"tool_name": "Read", "tool_input": {"file_path": "/work/sibling/x.py"}},
        monkeypatch, capsys,
    )
    assert rc == 0 and out.out.strip() == ""


def test_hook_empty_stdin_allows(monkeypatch, capsys):
    monkeypatch.setenv("SADDLE_CODE_ROOT", ROOT)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    from saddle import doctrine_hook
    assert doctrine_hook.main() == 0
    assert capsys.readouterr().out.strip() == ""


def test_hook_unparseable_fails_open(monkeypatch, capsys):
    monkeypatch.setenv("SADDLE_CODE_ROOT", ROOT)
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    from saddle import doctrine_hook
    assert doctrine_hook.main() == 0
    out = capsys.readouterr()
    assert out.out.strip() == ""  # no deny emitted
    assert "unparseable" in out.err
