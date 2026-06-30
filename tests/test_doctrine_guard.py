"""doctrine — the deterministic pre-action gate.

These are the lessons from saddle's own drift, frozen as red/green checks. They
exercise the gate directly (``evaluate(action, SEED_RULES)`` with an explicit
``project_root``) so nothing here touches the real filesystem or policy store —
the scope-fence reasons purely about path containment, which ``pathlib`` resolves
lexically for non-existent synthetic roots.
"""
from __future__ import annotations

import pytest

from saddle.doctrine import (
    SEED_RULES,
    Action,
    CheckRule,
    Verdict,
    evaluate,
)

ROOT = "/work/focus"


def _v(action: Action) -> Verdict:
    return evaluate(action, SEED_RULES)


# --- scope fence (stay-in-project-focus) -----------------------------------

def test_warns_edit_outside_focus():
    # Option A: a cross-project EDIT is ALLOWED but WARNED (not blocked) — it
    # surfaces loudly, it does not hard-gate a file edit.
    v = _v(Action("edit", "/work/sibling/src/x.py", "path", project_root=ROOT))
    assert v.allowed
    assert v.severity == "warn"
    assert v.rule_id == "stay-in-project-focus"
    assert "cross_project_task" in v.required_evidence


def test_allows_edit_inside_focus():
    v = _v(Action("edit", "/work/focus/src/saddle/x.py", "path", project_root=ROOT))
    assert v.allowed
    assert v.rule_id is None


def test_allows_relative_path_inside_focus():
    # a bare relative path is taken relative to the focus root, not some cwd
    v = _v(Action("write", "src/saddle/new.py", "path", project_root=ROOT))
    assert v.allowed


def test_allows_cross_project_with_explicit_evidence():
    v = _v(Action(
        "edit", "/work/sibling/src/x.py", "path",
        project_root=ROOT, evidence={"cross_project_task": "true"},
    ))
    assert v.allowed


def test_scope_fence_warns_every_cross_project_edit_verb():
    # every edit/write/create spelling outside focus WARNS (allowed, surfaced).
    for verb in ("edit", "write", "create", "patch", "overwrite", "modify"):
        v = _v(Action(verb, "/work/sibling/x.py", "path", project_root=ROOT))
        assert v.allowed, verb
        assert v.severity == "warn", verb
        assert v.rule_id == "stay-in-project-focus", verb


def test_scope_fence_blocks_every_cross_project_delete_verb():
    # every delete spelling outside focus stays a HARD BLOCK — file removal in a
    # sibling repo is never downgraded to a warning.
    for verb in ("delete", "rm", "unlink", "purge", "remove"):
        v = _v(Action(verb, "/work/sibling/x.py", "path", project_root=ROOT))
        assert not v.allowed, verb
        assert v.rule_id == "no-cross-project-delete", verb
        assert "cross_project_task" in v.required_evidence, verb


# --- no-unwired-delete + disposition-coherent ------------------------------

def test_blocks_delete_code_without_disposition():
    v = _v(Action("delete", "saddle/llm/llm_pool.py", "code", project_root=ROOT))
    assert not v.allowed
    assert v.rule_id == "no-unwired-delete"
    assert "disposition" in v.required_evidence


def test_delete_source_file_path_needs_disposition():
    # the live case: a .py PATH inside the focus passes the scope fence, but the
    # delete rule still demands a disposition — "unused" is not presumptively dead
    v = _v(Action("delete", "/work/focus/src/saddle/llm/llm_pool.py", "path",
                  project_root=ROOT))
    assert not v.allowed
    assert v.rule_id == "no-unwired-delete"


def test_blocks_delete_with_unknown_disposition():
    v = _v(Action("delete", "x.py", "code", project_root=ROOT,
                  evidence={"disposition": "deadcode"}))
    assert not v.allowed
    assert v.rule_id == "disposition-coherent"
    assert "must be one of" in v.reason


def test_blocks_delete_with_incoherent_disposition():
    # disposition present (clears no-unwired-delete) but missing its companion
    v = _v(Action("delete", "x.py", "code", project_root=ROOT,
                  evidence={"disposition": "superseded"}))
    assert not v.allowed
    assert v.rule_id == "disposition-coherent"
    assert "replaced_by" in v.reason


def test_allows_delete_with_coherent_superseded():
    v = _v(Action("delete", "x.py", "code", project_root=ROOT, evidence={
        "disposition": "superseded", "replaced_by": "FallbackCaller chain",
    }))
    assert v.allowed


def test_allows_delete_with_scaffold_disposition():
    v = _v(Action("delete", "x.py", "code", project_root=ROOT, evidence={
        "disposition": "scaffold", "wire_target": "orchestrator._promote",
    }))
    assert v.allowed


def test_allows_delete_domain_excluded():
    v = _v(Action("delete", "x.py", "code", project_root=ROOT, evidence={
        "disposition": "domain_excluded", "reason": "image gen — no saddle need",
    }))
    assert v.allowed


def test_cross_project_delete_blocks_on_scope_first():
    # first-block-wins: the cross-project DELETE fence is listed before the
    # in-focus disposition rule, so an out-of-focus code delete blocks on scope.
    v = _v(Action("delete", "/work/sibling/x.py", "code", project_root=ROOT))
    assert not v.allowed
    assert v.rule_id == "no-cross-project-delete"


# --- verb normalisation + non-mutating verbs -------------------------------

@pytest.mark.parametrize("verb", ["rm", "unlink", "remove", "purge", "drop", "del"])
def test_delete_synonyms_normalise(verb):
    v = _v(Action(verb, "x.py", "code", project_root=ROOT))
    assert not v.allowed
    assert v.rule_id == "no-unwired-delete"


def test_unknown_verb_allows():
    v = _v(Action("inspect", "/work/sibling/x.py", "path", project_root=ROOT))
    assert v.allowed
    assert v.rule_id is None


def test_read_like_verb_not_fenced():
    # only mutating verbs are fenced; a non-mutating verb passes even out-of-tree
    v = _v(Action("open", "/elsewhere/secret.py", "path", project_root=ROOT))
    assert v.allowed


# --- data-authored rules + pluggable predicates (the hybrid) ---------------

def test_data_rule_from_dict_require_evidence():
    rule = CheckRule.from_dict({
        "id": "needs-ticket",
        "kind": "require_evidence",
        "verbs": ["edit"],
        "requires_evidence": ["ticket"],
        "message": "edits need a ticket",
    })
    blocked = evaluate(Action("edit", "x.py", "code", project_root=ROOT), [rule])
    assert not blocked.allowed and blocked.rule_id == "needs-ticket"
    ok = evaluate(
        Action("edit", "x.py", "code", project_root=ROOT, evidence={"ticket": "JIRA-1"}),
        [rule],
    )
    assert ok.allowed


def test_data_rule_scope_fence_from_dict():
    rule = CheckRule.from_dict({
        "id": "fence",
        "kind": "scope_fence",
        "verbs": ["edit"],
        "target_kind": "path",
        "override_evidence": "approved",
    })
    out = evaluate(Action("edit", "/elsewhere/x.py", "path", project_root=ROOT), [rule])
    assert not out.allowed and out.rule_id == "fence"


def test_predicate_registry_pluggable():
    def _always_block(action, rule):
        return False, "nope"

    rule = CheckRule(
        id="custom", kind="predicate", verbs=frozenset({"edit"}),
        predicate="always_block",
    )
    v = evaluate(
        Action("edit", "x.py", "code", project_root=ROOT), [rule],
        predicates={"always_block": _always_block},
    )
    assert not v.allowed and v.rule_id == "custom" and v.reason == "nope"


def test_unknown_predicate_is_skipped_not_crash():
    rule = CheckRule(
        id="bad", kind="predicate", verbs=frozenset({"edit"}), predicate="missing",
    )
    v = evaluate(Action("edit", "x.py", "code", project_root=ROOT), [rule])
    assert v.allowed  # unknown predicate -> skipped, nothing blocks


def test_warn_severity_allows_with_warning():
    rule = CheckRule(
        id="soft", kind="require_evidence", verbs=frozenset({"edit"}),
        requires_evidence=("note",), severity="warn", message="prefer a note",
    )
    v = evaluate(Action("edit", "x.py", "code", project_root=ROOT), [rule])
    assert v.allowed
    assert v.severity == "warn"
    assert v.rule_id == "soft"


def test_block_beats_warn_regardless_of_order():
    warn = CheckRule(id="w", kind="require_evidence", verbs=frozenset({"delete"}),
                     requires_evidence=("note",), severity="warn")
    block = CheckRule(id="b", kind="require_evidence", verbs=frozenset({"delete"}),
                      requires_evidence=("disp",), severity="block")
    v = evaluate(Action("delete", "x.py", "code", project_root=ROOT), [warn, block])
    assert not v.allowed and v.rule_id == "b"


# --- verdict rendering -----------------------------------------------------

def test_verdict_render_block_and_allow():
    blocked = _v(Action("delete", "x.py", "code", project_root=ROOT))
    assert blocked.render().startswith("BLOCK [no-unwired-delete]")
    assert "needs evidence: disposition" in blocked.render()
    allowed = _v(Action("edit", "src/x.py", "path", project_root=ROOT))
    assert allowed.render().startswith("ALLOW")
