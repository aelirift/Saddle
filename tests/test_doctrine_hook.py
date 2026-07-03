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


# --- actions_from_tool: shell WRITE channels (the mutate-the-other-way half) -

def _writes(command):
    acts = actions_from_tool("Bash", {"command": command}, project_root=ROOT)
    return [a.target for a in acts if a.nverb == "write"]


def test_bash_redirect_write():
    # `>` creates/overwrites a file just like Write — it must surface as a write.
    assert _writes("cat tmpl > out.py") == ["out.py"]


def test_bash_redirect_append_write():
    assert _writes("echo line >> notes.py") == ["notes.py"]


def test_bash_redirect_attached_no_space():
    # shlex keeps `>out.py` as one token; the lexical pass still catches it.
    assert _writes("cat tmpl >out.py") == ["out.py"]


def test_bash_redirect_quoted_target():
    assert _writes("cat t > 'my file.py'") == ["my file.py"]


def test_bash_tee_writes_every_operand():
    assert _writes("echo x | tee a.py b.py") == ["a.py", "b.py"]


def test_bash_cp_writes_destination_only():
    # source is a READ; only the destination is a write.
    assert _writes("cp a.py dst/b.py") == ["dst/b.py"]


def test_bash_mv_writes_destination_only():
    assert _writes("mv a.py dst/b.py") == ["dst/b.py"]


def test_bash_dd_writes_of_target():
    assert _writes("dd if=a.img of=out.py bs=1M") == ["out.py"]


def test_bash_sed_in_place_writes_file_not_script():
    # the `s/a/b/g` script carries no suffix; only the .py operand is a file.
    assert _writes("sed -i 's/a/b/g' x.py") == ["x.py"]


def test_bash_sed_without_in_place_is_noop():
    # plain sed streams to stdout — no file write.
    assert _writes("sed 's/a/b/g' x.py") == []


def test_bash_redirect_to_devnull_is_ignored():
    # /dev/null is not another project — must not surface as a write.
    assert _writes("pytest -q > /dev/null") == []


def test_bash_redirect_to_tmp_is_ignored():
    assert _writes("pytest -q > /tmp/out.log 2>&1") == []


def test_bash_fd_dup_names_no_file():
    # `2>&1` redirects a descriptor, not a file — nothing to assert on.
    assert _writes("make 2>&1") == []


def test_bash_quoted_redirect_glyph_is_inert():
    # a `>` inside a string is text, not a redirection.
    assert _writes('echo "a > b"') == []


def test_bash_write_and_delete_coexist():
    acts = actions_from_tool(
        "Bash", {"command": "rm old.py && cp new.py dst/new.py"}, project_root=ROOT)
    assert [(a.nverb, a.target) for a in acts] == [
        ("delete", "old.py"), ("write", "dst/new.py")]


# --- gate_tool_call --------------------------------------------------------

def test_gate_warns_edit_outside_focus():
    # cross-project EDIT is allowed-with-warning now, not blocked.
    v = _gate("Edit", {"file_path": "/work/sibling/x.py"})
    assert v.allowed and v.severity == "warn"
    assert v.rule_id == "stay-in-project-focus"


def test_gate_allows_edit_inside_focus():
    v = _gate("Edit", {"file_path": "/work/focus/src/x.py"})
    assert v.allowed


def test_gate_blocks_bash_rm_code():
    v = _gate("Bash", {"command": "rm src/saddle/llm/llm_pool.py"})
    assert not v.allowed and v.rule_id == "no-unwired-delete"


def test_gate_allows_bash_rm_non_code():
    v = _gate("Bash", {"command": "rm build/artifact.bin && rm notes.txt"})
    assert v.allowed


def test_gate_blocks_rm_of_sibling_on_cross_project_delete():
    # a DELETE outside focus still hard-blocks (deletes are not downgraded).
    v = _gate("Bash", {"command": "rm /work/sibling/foo.py"})
    assert not v.allowed and v.rule_id == "no-cross-project-delete"


def test_gate_read_is_allowed():
    v = _gate("Read", {"file_path": "/work/sibling/x.py"})
    assert v.allowed and "does not mutate" in v.reason


# --- gate_tool_call: shell writes face the scope-fence like Edit/Write -------
# A shell WRITE into a sibling is a cross-project EDIT/WRITE, so (like the Edit
# and Write tools) it now WARNS rather than blocks — surfaced, never silent.

def test_gate_warns_redirect_write_to_sibling():
    v = _gate("Bash", {"command": "cat tmpl > /work/sibling/x.py"})
    assert v.allowed and v.severity == "warn" and v.rule_id == "stay-in-project-focus"


def test_gate_warns_cp_into_sibling():
    v = _gate("Bash", {"command": "cp a.py /work/sibling/b.py"})
    assert v.allowed and v.severity == "warn" and v.rule_id == "stay-in-project-focus"


def test_gate_warns_dd_into_sibling():
    v = _gate("Bash", {"command": "dd if=a.img of=/work/sibling/b.py"})
    assert v.allowed and v.severity == "warn" and v.rule_id == "stay-in-project-focus"


def test_gate_warns_sed_in_place_on_sibling():
    v = _gate("Bash", {"command": "sed -i 's/a/b/g' /work/sibling/x.py"})
    assert v.allowed and v.severity == "warn" and v.rule_id == "stay-in-project-focus"


def test_gate_allows_redirect_write_inside_focus():
    v = _gate("Bash", {"command": "cat tmpl > /work/focus/out.py"})
    assert v.allowed


def test_gate_allows_redirect_to_devnull():
    v = _gate("Bash", {"command": "pytest -q > /dev/null 2>&1"})
    assert v.allowed


def test_gate_allows_redirect_to_tmp():
    v = _gate("Bash", {"command": "make build > /tmp/build.log"})
    assert v.allowed


# --- the PreToolUse hook ---------------------------------------------------

def _run_hook(payload, monkeypatch, capsys):
    monkeypatch.setenv("SADDLE_CODE_ROOT", ROOT)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    from saddle import doctrine_hook
    rc = doctrine_hook.main()
    return rc, capsys.readouterr()


def test_hook_warns_on_out_of_focus_edit(monkeypatch, capsys):
    # Option A: an out-of-focus EDIT is allowed-with-warning, not denied — no
    # permission decision is emitted, and the warn is surfaced loudly on stderr.
    rc, out = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": "/work/sibling/x.py"}},
        monkeypatch, capsys,
    )
    assert rc == 0
    assert out.out.strip() == ""                 # no deny decision emitted
    assert "WARN out-of-focus" in out.err        # but never silent


def test_hook_denies_code_delete(monkeypatch, capsys):
    rc, out = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm src/saddle/x.py"}},
        monkeypatch, capsys,
    )
    assert rc == 0
    assert json.loads(out.out)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_deny_heralds_block_on_screen(monkeypatch, capsys):
    """ask #3 — a BLOCK is saddle's loudest act; the human must SEE it, not just
    the agent. A cross-project DELETE still hard-blocks (only edits downgraded to
    warn), and the deny JSON carries a top-level ``systemMessage`` (the one channel
    Claude Code renders on screen) naming saddle + the blocked tool + the rule."""
    rc, out = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm /work/sibling/foo.py"}},
        monkeypatch, capsys,
    )
    assert rc == 0
    doc = json.loads(out.out)
    assert doc["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "saddle BLOCKED Bash" in doc["systemMessage"]
    assert "no-cross-project-delete" in doc["systemMessage"]


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


# --- Stage 3: the pre-code design review (observation) on the same hook ------
#
# After the deterministic guard ALLOWS the first code edit of a turn, the hook
# reads the agent's pre-edit reasoning from the transcript and audits it. These
# fake out the LLM (monkeypatch audit_proposal) and assert the wiring: a flaw is a
# loud ALERT bubble + agent additionalContext (never a permissionDecision), a
# clean approach is silent, no recorded approach is its own finding, a failed
# audit fails loud, and the gate fires exactly once per turn.
from saddle.context import Context

_DCTX = Context(tenant="acme", project="game")


def _t_user(text, uuid):
    return {"type": "user", "sessionId": "s1", "uuid": uuid, "isSidechain": False,
            "message": {"role": "user", "content": text}}


def _t_asst(text, uuid):
    return {"type": "assistant", "sessionId": "s1", "uuid": uuid, "isSidechain": False,
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def _transcript(tmp_path, objs):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in objs) + "\n")
    return str(p)


def _patch_audit(monkeypatch, fn):
    monkeypatch.setattr("saddle.design.audit_proposal", fn)


def _canned_audit(verdict):
    async def _audit(goal, approach, ctx=None, **kw):
        return verdict
    return _audit


def _run_design_hook(payload, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SADDLE_CODE_ROOT", ROOT)
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    from saddle import doctrine_hook
    rc = doctrine_hook.main()
    return rc, capsys.readouterr()


def _edit_payload(tp, *, tool="Edit", path="/work/focus/x.py", session="s1"):
    return {"tool_name": tool, "tool_input": {"file_path": path},
            "session_id": session, "transcript_path": tp}


def test_design_gate_flags_a_bandaid_approach(monkeypatch, capsys, tmp_path):
    from saddle.bubble import recent_bubbles
    from saddle.design import AuditVerdict

    _patch_audit(monkeypatch, _canned_audit(
        AuditVerdict(ok=False, issues=["band-aid: swallow-and-log the error"])))
    tp = _transcript(tmp_path, [
        _t_user("fix the timeout", "u1"),
        _t_asst("I'll wrap the failing call in try/except and log it.", "u2"),
    ])
    rc, out = _run_design_hook(_edit_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0
    # the agent sees it via additionalContext — and NO permissionDecision (the
    # review observes, it never auto-approves or blocks the edit).
    dec = json.loads(out.out)["hookSpecificOutput"]
    assert dec["hookEventName"] == "PreToolUse"
    assert "permissionDecision" not in dec
    assert "band-aid: swallow-and-log the error" in dec["additionalContext"]
    # and the human sees the durable per-stage ALERT bubble.
    alerts = recent_bubbles(_DCTX, level="alert")
    assert any(b.stage == "design" and "band-aid" in b.text for b in alerts)


def test_design_bandaid_heralds_on_screen(monkeypatch, capsys, tmp_path):
    """ask #3 — the Stage-3 finding reaches the HUMAN via ``systemMessage`` too, not
    only the agent's ``additionalContext``. A caught band-aid is visible on screen
    while staying OBSERVATION (still no permissionDecision — it never blocks)."""
    from saddle.design import AuditVerdict

    _patch_audit(monkeypatch, _canned_audit(
        AuditVerdict(ok=False, issues=["band-aid: swallow-and-log the error"])))
    tp = _transcript(tmp_path, [
        _t_user("fix the timeout", "u1"),
        _t_asst("I'll wrap the failing call in try/except and log it.", "u2"),
    ])
    rc, out = _run_design_hook(_edit_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0
    doc = json.loads(out.out)
    assert "band-aid: swallow-and-log the error" in doc["systemMessage"]
    assert "permissionDecision" not in doc["hookSpecificOutput"]


def _patch_settle(monkeypatch, *, design_id="design_test1", summary="the plan"):
    from saddle.models import DESIGN_FINAL, Design

    settled = {"n": 0, "approved_by": None}

    async def _settle(goal, approach, ctx=None, *, approved_by="converged", **kw):
        settled["n"] += 1
        settled["approved_by"] = approved_by
        return Design(ask=goal, summary=summary, approach=approach,
                      body=approach, status=DESIGN_FINAL, id=design_id)

    monkeypatch.setattr("saddle.design.settle_approach", _settle)
    return settled


def test_design_gate_clean_approach_settles_the_design(monkeypatch, capsys, tmp_path):
    """Mediator loop step 4: an approach that audits clean at the first code edit
    IS the agreement — it is recorded as a settled design (Stage 4's substrate)
    and said once, briefly."""
    from saddle.bubble import recent_bubbles
    from saddle.design import AuditVerdict

    _patch_audit(monkeypatch, _canned_audit(AuditVerdict(ok=True, issues=[])))
    settled = _patch_settle(monkeypatch)
    tp = _transcript(tmp_path, [
        _t_user("bound the cache", "u1"),
        _t_asst("Root cause: unbounded map. I'll add an LRU with eviction.", "u2"),
    ])
    rc, out = _run_design_hook(_edit_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0
    assert settled["n"] == 1
    assert "Plan agreed and recorded" in out.out    # agent + human both hear it once
    notices = [b for b in recent_bubbles(_DCTX, level="notice") if b.stage == "design"]
    assert len(notices) == 1 and notices[0].meta.get("settled") is True


def test_design_gate_strict_mode_denies_until_clean(monkeypatch, capsys, tmp_path):
    """SADDLE_GATE_MODE=deny: unresolved plan issues HOLD the edit (deny doc on
    stdout), and a revised approach on the next attempt re-reviews, settles, and
    opens the gate — the discussion holds the floor, then lets go."""
    from saddle.design import AuditVerdict

    calls = {"n": 0}

    async def _flip_audit(goal, approach, ctx=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return AuditVerdict(ok=False, issues=["band-aid: swallow-and-log"])
        return AuditVerdict(ok=True, issues=[])

    _patch_audit(monkeypatch, _flip_audit)
    settled = _patch_settle(monkeypatch)
    monkeypatch.setenv("SADDLE_GATE_MODE", "deny")
    tp = _transcript(tmp_path, [
        _t_user("fix the timeout", "u1"),
        _t_asst("I'll wrap it in try/except and log.", "u2"),
    ])
    rc, out = _run_design_hook(_edit_payload(tp), monkeypatch, capsys, tmp_path)
    assert rc == 0
    doc = json.loads(out.out.strip().splitlines()[0])
    assert doc["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "holding this code change" in doc["hookSpecificOutput"]["permissionDecisionReason"]
    assert settled["n"] == 0

    # second attempt: the (revised) approach audits clean -> settle -> allow
    rc2, out2 = _run_design_hook(_edit_payload(tp), monkeypatch, capsys, tmp_path)
    assert rc2 == 0
    assert calls["n"] == 2                          # re-reviewed, not blind-denied
    assert settled["n"] == 1
    assert "permissionDecision" not in out2.out     # no deny -> the edit proceeds


def test_design_gate_no_recorded_design_is_an_alert(monkeypatch, capsys, tmp_path):
    from saddle.bubble import recent_bubbles

    called = {"audit": False}

    async def _forbidden(*a, **k):
        called["audit"] = True
        return None
    _patch_audit(monkeypatch, _forbidden)
    # user prompt then STRAIGHT to the edit — no spoken approach.
    tp = _transcript(tmp_path, [_t_user("just do it", "u1")])
    rc, out = _run_design_hook(
        _edit_payload(tp, tool="Write"), monkeypatch, capsys, tmp_path)

    assert rc == 0
    assert called["audit"] is False                # no approach -> no LLM call
    dec = json.loads(out.out)["hookSpecificOutput"]
    assert "without first laying out its plan" in dec["additionalContext"]
    alerts = recent_bubbles(_DCTX, level="alert")
    assert any(b.stage == "design" and "without first laying out its plan" in b.text for b in alerts)


def test_design_gate_fires_once_per_turn(monkeypatch, capsys, tmp_path):
    from saddle.design import AuditVerdict

    calls = {"n": 0}

    async def _counting(goal, approach, ctx=None, **kw):
        calls["n"] += 1
        return AuditVerdict(ok=False, issues=["issue"])
    _patch_audit(monkeypatch, _counting)
    tp = _transcript(tmp_path, [
        _t_user("do the thing", "u1"),
        _t_asst("here's my approach", "u2"),
    ])
    payload = _edit_payload(tp)
    _run_design_hook(payload, monkeypatch, capsys, tmp_path)   # first edit -> audits
    _run_design_hook(payload, monkeypatch, capsys, tmp_path)   # same turn -> skipped

    assert calls["n"] == 1                          # anchored on the user-prompt uuid


def test_design_gate_audit_failure_fails_loud(monkeypatch, capsys, tmp_path):
    from saddle.bubble import recent_bubbles

    async def _boom(goal, approach, ctx=None, **kw):
        raise RuntimeError("provider down")
    _patch_audit(monkeypatch, _boom)
    tp = _transcript(tmp_path, [
        _t_user("do the thing", "u1"),
        _t_asst("my approach here", "u2"),
    ])
    rc, out = _run_design_hook(_edit_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0                                  # observation never blocks
    # a classified ALERT bubble naming what saddle did NOT verify this turn.
    alerts = recent_bubbles(_DCTX, level="alert")
    assert any(b.stage == "design" and "did not run" in b.text for b in alerts)
    assert "did not run" in out.out                 # the agent learns it too


def test_design_gate_skips_non_edit_tools(monkeypatch, capsys, tmp_path):
    from saddle.design import AuditVerdict

    calls = {"n": 0}

    async def _counting(*a, **k):
        calls["n"] += 1
        return AuditVerdict()
    _patch_audit(monkeypatch, _counting)
    tp = _transcript(tmp_path, [_t_user("run tests", "u1"), _t_asst("ok", "u2")])
    rc, out = _run_design_hook(
        {"tool_name": "Bash", "tool_input": {"command": "pytest -q"},
         "session_id": "s1", "transcript_path": tp},
        monkeypatch, capsys, tmp_path)

    assert rc == 0 and out.out.strip() == ""
    assert calls["n"] == 0                          # a non-edit tool never audits


def test_design_gate_not_reached_when_guard_denies(monkeypatch, capsys, tmp_path):
    from saddle.design import AuditVerdict
    from saddle.doctrine import Verdict

    calls = {"n": 0}

    async def _counting(*a, **k):
        calls["n"] += 1
        return AuditVerdict()
    _patch_audit(monkeypatch, _counting)
    # No seed rule blocks an Edit any more (cross-project edits only WARN), so
    # force a deny to prove the invariant under test: a BLOCKED tool call never
    # reaches Stage 3 — the design review rides only on the guard's ALLOW.
    monkeypatch.setattr(
        "saddle.doctrine.gate_tool_call",
        lambda *a, **k: Verdict(False, "needs-review", "forced block for test"),
    )
    tp = _transcript(tmp_path, [_t_user("do it", "u1"), _t_asst("approach", "u2")])
    rc, out = _run_design_hook(_edit_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0
    dec = json.loads(out.out)["hookSpecificOutput"]
    assert dec["permissionDecision"] == "deny"      # forced guard block
    assert calls["n"] == 0                           # so Stage 3 was never reached


def test_design_gate_runs_on_cross_project_authorized_edit(monkeypatch, capsys, tmp_path):
    """A cross-project EDIT a USER grant covers is ALLOWED-with-a-NOTICE (Option A:
    edits warn, they don't block) — and it is still the FIRST code edit of a turn, so
    Stage 3 (the anti-band-aid design review) must run on it, not be skipped.
    Regression guard: the cross-project allow path must reach the design stage and
    never ``return 0`` before it, or the audit is silently skipped for all authorized
    cross-project work (the common case during an AFK multi-repo session)."""
    from saddle.bubble import recent_bubbles
    from saddle.design import AuditVerdict

    _patch_audit(monkeypatch, _canned_audit(
        AuditVerdict(ok=False, issues=["band-aid: swallow-and-log the error"])))
    tp = _transcript(tmp_path, [
        _t_user("fix the timeout", "u1"),
        _t_asst("I'll wrap the failing call in try/except and log it.", "u2"),
    ])
    # authorize focus<->sibling so the scope-fence block is overridden by the grant.
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    from saddle import crossproject
    crossproject.grant(["/work/focus", "/work/sibling"], tenant="acme", reason="test")

    rc, out = _run_design_hook(
        _edit_payload(tp, path="/work/sibling/x.py"), monkeypatch, capsys, tmp_path)

    assert rc == 0
    dec = json.loads(out.out)["hookSpecificOutput"]
    # the edit was ALLOWED via the grant -> NO deny decision is emitted ...
    assert "permissionDecision" not in dec
    # ... and Stage 3 STILL ran: the band-aid rides agent context + a design ALERT.
    assert "band-aid: swallow-and-log the error" in dec["additionalContext"]
    alerts = recent_bubbles(_DCTX, level="alert")
    assert any(b.stage == "design" and "band-aid" in b.text for b in alerts)


def test_design_gate_disabled_by_env(monkeypatch, capsys, tmp_path):
    from saddle.design import AuditVerdict

    calls = {"n": 0}

    async def _counting(*a, **k):
        calls["n"] += 1
        return AuditVerdict()
    _patch_audit(monkeypatch, _counting)
    monkeypatch.setenv("SADDLE_HOOK_DESIGN", "0")
    tp = _transcript(tmp_path, [_t_user("do it", "u1"), _t_asst("approach", "u2")])
    rc, out = _run_design_hook(_edit_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0 and out.out.strip() == ""
    assert calls["n"] == 0
