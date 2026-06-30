"""design.py audit stage — the Layer-2 gate that tests DESIGN, not just code.

The codemap (Layer 3) gates mechanical dataflow. This is the OTHER half of the
instruction: the design pipeline must also gate INTENT and BEST PRACTICE. The
audit stage is that gate — it is handed the binding DIRECTIVES and the DKB's
ANTI-PATTERNS and must flag a design that violates them, hand-waves, or band-aids
the symptom. A clean design passes untouched; every flaw it catches is harvested
back into the DKB as a durable lesson, so the gate makes the system smarter by
use (saddle's structural answer to RayXI's "gate that only checked declarations").

No real LLM and no embedder: a fixture caller dispatches canned replies by label
(audit verdicts popped in sequence to script convergence / non-convergence) and a
stub DKB serves one anti-pattern and records what gets harvested.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from saddle.context import Context
from saddle.design import (
    AuditVerdict,
    _AuditParseError,
    _parse_audit_reply,
    audit_proposal,
    design_for,
)
from saddle.models import (
    ANTI_PATTERN,
    AUDIT,
    DESIGN_FINAL,
    DESIGN_FLAGGED,
    LESSON,
    Knowledge,
)

# The anti-pattern the DKB serves into the audit — its title must reach the audit
# prompt, proving the gate tests against best-practice knowledge, not just code.
_ANTI = Knowledge(
    kind=ANTI_PATTERN,
    title="Magic-number balancing",
    body="Scaling raw numbers to hit a target instead of fixing the kit's design.",
    tags=["balance"],
)
# A binding directive that must likewise reach the audit prompt.
_DIRECTIVE = "never hard-code game values"


def _audit_reply(verdict: dict) -> str:
    """Render a ``{"ok", "issues"}`` verdict as the LINE contract the audit stage
    now expects (verdict line + one issue per line). The tests still express cases
    as dicts for readability; this is the wire format the real LLM emits since the
    audit moved OFF JSON (one stray quote in one issue used to discard them all)."""
    issues = [str(i) for i in (verdict.get("issues") or [])]
    if verdict.get("ok") and not issues:
        return "OK"
    return "ISSUES\n" + "\n".join(issues)


class FixtureCaller:
    """Records prompts by label; pops audit verdicts in order to script the loop."""

    def __init__(self, audit_verdicts: list[dict]) -> None:
        self.prompts: dict[str, str] = {}
        self._audits = list(audit_verdicts)

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False,
                       label: str = "") -> str:
        self.prompts[label] = prompt
        if label == "design/diagnose":
            return json.dumps({
                "problem": "the kit has no rotation",
                "approach": "design a real resource loop, not a number bump",
                "risks": ["a multiplier hides the missing loop"],
            })
        if label in ("design/body", "design/revise"):
            return "## Design\nA concrete resource loop with costs and cooldowns."
        if label == "design/audit":
            verdict = self._audits.pop(0) if self._audits else {"ok": True, "issues": []}
            return _audit_reply(verdict)
        if label == "design/index":
            return json.dumps({"summary": "Resource-loop kit",
                               "satisfies": [], "avoids": [], "heeds": []})
        if label == "design/harvest":
            return json.dumps({"entries": [{
                "kind": "lesson",
                "title": "Bump is not balance",
                "body": "A flat multiplier never fixes a kit that lacks a rotation.",
                "tags": ["balance", "design"],
            }]})
        return json.dumps({"entries": []})


class StubDKB:
    """Serves one anti-pattern (only when asked for that kind) and records harvests."""

    def __init__(self) -> None:
        self.added: list[Knowledge] = []

    def search_knowledge(self, ctx, query, *, k=8, kinds=None):
        if kinds and ANTI_PATTERN in kinds:
            return [(_ANTI, 1.0)]
        return []

    def list_knowledge(self, ctx=None, *, limit=200, **kw):
        return list(self.added)

    def add_knowledge(self, kn: Knowledge) -> Knowledge:
        self.added.append(kn)
        return kn

    def add_design(self, ctx, design):
        return design


def _design(caller, dkb, *, harvest, max_audits):
    return asyncio.run(design_for(
        goal="make the mage hit harder",
        ctx=Context(tenant="acme", project="game"),
        caller=caller,
        dkb=dkb,
        directives=[_DIRECTIVE],
        persist=False,
        harvest=harvest,
        surface=False,          # isolate the audit gate from the code-surface stage
        max_audits=max_audits,
    ))


def test_audit_flags_directive_and_anti_pattern_violations():
    """The gate tests DESIGN INTENT: a design the audit never blesses ends FLAGGED,
    its issues recorded — and the binding directive AND the DKB anti-pattern were
    both put in front of the audit, proving it judges against intent + best
    practice, not merely mechanical facts."""
    caller = FixtureCaller([
        {"ok": False, "issues": ["hard-codes the damage constant", "no rotation"]},
        {"ok": False, "issues": ["still a symptom patch"]},
    ])
    design = _design(caller, StubDKB(), harvest=False, max_audits=2)

    # never converged -> flagged, all issues across both passes recorded.
    assert design.status == DESIGN_FLAGGED
    assert design.meta["audit_clean"] is False
    assert design.meta["audits_run"] == 2
    assert design.meta["issues"] == [
        "hard-codes the damage constant", "no rotation", "still a symptom patch",
    ]

    # the audit was actually handed the intent + best-practice context it gates on.
    audit_prompt = caller.prompts["design/audit"]
    assert _DIRECTIVE in audit_prompt            # binding directive reached the gate
    assert "Magic-number balancing" in audit_prompt   # DKB anti-pattern reached it

    # surface stage was off -> the code gate never ran; this is the design gate.
    assert "design/surface" not in caller.prompts


def test_clean_design_passes_and_harvests_nothing():
    """A sound design is blessed on the first audit: FINAL, no issues, and even with
    harvesting ON nothing is filed — there were no caught flaws to learn from."""
    caller = FixtureCaller([{"ok": True, "issues": []}])
    dkb = StubDKB()
    design = _design(caller, dkb, harvest=True, max_audits=2)

    assert design.status == DESIGN_FINAL
    assert design.meta["audit_clean"] is True
    assert design.meta["audits_run"] == 1        # converged immediately, no revise
    assert design.meta["issues"] == []
    assert design.meta["harvested"] == 0
    assert dkb.added == []                        # clean design teaches nothing
    assert "design/harvest" not in caller.prompts


def test_caught_flaws_are_harvested_into_the_dkb():
    """The self-improvement loop: a flaw caught on the first audit and fixed on the
    revise becomes a durable DKB lesson (source=audit), scoped to the project — so
    the gate that caught it makes the next design smarter."""
    caller = FixtureCaller([
        {"ok": False, "issues": ["band-aid: patches the symptom"]},
        {"ok": True, "issues": []},
    ])
    dkb = StubDKB()
    design = _design(caller, dkb, harvest=True, max_audits=2)

    # converged after one revise; the caught flaw is still recorded for harvest.
    assert design.status == DESIGN_FINAL
    assert design.meta["audit_clean"] is True
    assert design.meta["audits_run"] == 2
    assert design.meta["issues"] == ["band-aid: patches the symptom"]
    assert design.meta["harvested"] == 1

    # the flaw was filed back as a project-scoped, audit-sourced lesson.
    assert len(dkb.added) == 1
    lesson = dkb.added[0]
    assert lesson.source == AUDIT
    assert lesson.kind == LESSON
    assert lesson.scope_tenant == "acme"
    assert lesson.scope_project == "game"

    # the harvest stage was driven by the actual caught flaw, not a re-derivation.
    assert "band-aid: patches the symptom" in caller.prompts["design/harvest"]


# === audit_proposal — Stage 3's gate over an ALREADY-written approach =========
#
# design_for GENERATES a design and audits its own output; audit_proposal audits
# an approach the supervised agent already wrote in its transcript, through the
# SAME _SYS_AUDIT stage. These pin that standalone seam against a faked caller +
# DKB so the wiring (retrieve anti-patterns -> audit -> verdict) stays offline.

_CTX = Context(tenant="acme", project="game")


class _AuditCaller:
    """Records the audit prompt; returns a canned verdict. ``boom=True`` raises, to
    prove a failed classify PROPAGATES (fail-loud)."""

    def __init__(self, verdict: dict | None = None, *, boom: bool = False) -> None:
        self.verdict = verdict if verdict is not None else {"ok": True, "issues": []}
        self.boom = boom
        self.prompt = ""

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False,
                       label: str = "") -> str:
        if self.boom:
            raise RuntimeError("provider down")
        self.prompt = prompt
        return _audit_reply(self.verdict)


class _AntiDKB:
    """Serves canned (Knowledge, score) anti-patterns, only when asked for that kind."""

    def __init__(self, anti=None) -> None:
        self._anti = list(anti or [])

    def search_knowledge(self, ctx, query, *, k=8, kinds=None):
        if kinds and ANTI_PATTERN in kinds:
            return list(self._anti)
        return []


def _audit(approach, *, caller, dkb, directives=None,
           goal="make the mage hit harder") -> AuditVerdict:
    return asyncio.run(audit_proposal(
        goal, approach, _CTX, caller=caller, dkb=dkb,
        directives=directives if directives is not None else [_DIRECTIVE],
    ))


def test_audit_proposal_clean_approach_is_ok():
    """A sound approach audits clean: ok, no issues — and the anti-pattern the DKB
    served was actually weighed (considered counts it)."""
    caller = _AuditCaller({"ok": True, "issues": []})
    v = _audit("Design a real resource loop with costs and cooldowns.",
               caller=caller, dkb=_AntiDKB([(_ANTI, 1.0)]))
    assert isinstance(v, AuditVerdict)
    assert v.ok is True and v.has_issues is False
    assert v.considered == 1


def test_audit_proposal_flags_a_bandaid():
    caller = _AuditCaller({"ok": False, "issues": ["band-aid: swallow-and-log the error"]})
    v = _audit("Wrap the failing call in try/except and log it.",
               caller=caller, dkb=_AntiDKB([(_ANTI, 1.0)]))
    assert v.ok is False and v.has_issues is True
    assert v.issues == ["band-aid: swallow-and-log the error"]


def test_audit_proposal_feeds_directive_and_anti_pattern_to_the_audit():
    """The gate judges against INTENT + best practice: the binding directive AND
    the DKB anti-pattern both reach the audit prompt (same as design_for's audit)."""
    caller = _AuditCaller({"ok": True, "issues": []})
    _audit("some concrete approach", caller=caller, dkb=_AntiDKB([(_ANTI, 1.0)]))
    assert _DIRECTIVE in caller.prompt
    assert "Magic-number balancing" in caller.prompt


def test_audit_proposal_includes_the_global_seed_anti_pattern():
    """Stage 3's deliberate complement to Stage 2: it measures against UNIVERSAL
    wisdom, so a GLOBAL-scoped anti-pattern (scope_tenant='') is NOT dropped —
    unlike intent.history_drift, which weighs only this project's settled state."""
    seed = Knowledge(kind=ANTI_PATTERN, title="Silent fallback",
                     body="Degrading to a quiet default hides the real failure.",
                     scope_tenant="", scope_project="")          # global
    caller = _AuditCaller({"ok": True, "issues": []})
    v = _audit("some approach", caller=caller, dkb=_AntiDKB([(seed, 0.9)]))
    assert v.considered == 1
    assert "Silent fallback" in caller.prompt


def test_audit_proposal_empty_approach_raises():
    """The hook handles the 'no recorded design' case before calling; the engine
    guards an empty approach so it can never silently audit nothing."""
    with pytest.raises(ValueError, match="empty approach"):
        _audit("   ", caller=_AuditCaller(), dkb=_AntiDKB())


def test_audit_proposal_failed_classify_propagates():
    """fail-loud: a raising caller propagates OUT for run_stage to classify and
    bubble an ALERT — never swallowed into a false 'looked fine'."""
    with pytest.raises(RuntimeError, match="provider down"):
        _audit("some approach", caller=_AuditCaller(boom=True),
               dkb=_AntiDKB([(_ANTI, 1.0)]))


# === _parse_audit_reply — the LINE contract's loud-accounting invariants =======
#
# Fork A: the audit moved OFF a JSON {"ok","issues":[...]} reply because long
# prose hand-escaped into a JSON array is the harness's most fragile parse — one
# stray quote throws the WHOLE payload and discards every issue with it, a
# fabricated 'sound' (the cardinal sin). These pin every false-negative seam of
# the replacement: a malformed verdict RAISES (loud), and an ugly issue can never
# take the rest of the review down.


def test_parse_audit_reply_ok_alone_is_sound():
    assert _parse_audit_reply("OK") == (True, [])
    assert _parse_audit_reply("  OK  \n\n") == (True, [])


def test_parse_audit_reply_accepts_markdown_and_punctuated_verdicts():
    """The verdict reduces to its bare alpha token, so a model that bolds or
    punctuates the verdict line still parses — content matching is strict, glyphs
    are not."""
    assert _parse_audit_reply("**OK**") == (True, [])
    assert _parse_audit_reply("ISSUES:\n- a band-aid") == (False, ["a band-aid"])


def test_parse_audit_reply_collects_one_issue_per_line_stripping_bullets():
    ok, issues = _parse_audit_reply(
        "ISSUES\n- hard-codes the constant\n* no rotation\n1. uncovered ask"
    )
    assert ok is False
    assert issues == ["hard-codes the constant", "no rotation", "uncovered ask"]


def test_parse_audit_reply_issues_verdict_with_no_lines_preserves_prior_semantics():
    """ISSUES with nothing named -> (False, []), the same 'flagged not-ok but named
    nothing actionable' state the old {"ok": false, "issues": []} produced."""
    assert _parse_audit_reply("ISSUES") == (False, [])


def test_parse_audit_reply_strips_think_block_before_the_verdict():
    reply = "<think>let me weigh the directives</think>\nOK"
    assert _parse_audit_reply(reply) == (True, [])


def test_parse_audit_reply_an_ugly_issue_no_longer_takes_the_rest_down():
    """The whole point of Fork A: an issue containing the characters that USED to
    break json.loads (an unescaped quote, a brace, a comma) now survives as ONE
    issue, and the issues AROUND it are not lost with it."""
    reply = (
        'ISSUES\n'
        'the cooldown field "cooldown_s" is never read by the sweep\n'
        'the {damage} resolver is bypassed, so modifiers do not reach it\n'
        'no coverage of the save/load round-trip'
    )
    ok, issues = _parse_audit_reply(reply)
    assert ok is False
    assert len(issues) == 3
    assert issues[0] == 'the cooldown field "cooldown_s" is never read by the sweep'


def test_parse_audit_reply_empty_reply_raises_never_a_silent_pass():
    for blank in ("", "   ", "\n\n", "<think>only reasoning, no verdict</think>"):
        with pytest.raises(_AuditParseError):
            _parse_audit_reply(blank)


def test_parse_audit_reply_missing_verdict_raises_not_guessed_into_a_pass():
    """A reply that jumps straight to issues with NO verdict line is the cardinal
    seam: we must NOT read it as OK (drop the issues) NOR silently treat it as
    issues — we RAISE so the stage fails loud and the model re-emits."""
    with pytest.raises(_AuditParseError):
        _parse_audit_reply("- the cooldown isn't wired\n- no resource display")


def test_parse_audit_reply_preamble_before_verdict_raises():
    with pytest.raises(_AuditParseError):
        _parse_audit_reply("Here is my audit:\nISSUES\n- a band-aid")


def test_parse_audit_reply_ok_with_trailing_content_raises_never_trusts_ok():
    """An OK verdict followed by content is a contradiction; we refuse to trust OK
    over lines the model also chose to write (those lines might be real issues) —
    raise rather than risk a false negative."""
    with pytest.raises(_AuditParseError):
        _parse_audit_reply("OK\n- actually the cooldown isn't wired")


def test_parse_audit_reply_issue_crammed_onto_the_verdict_line_raises():
    """Demanding the whole first line reduce to exactly OK/ISSUES means an issue
    jammed onto the verdict line fails the match and raises — it can never be
    silently swallowed as a bare OK."""
    with pytest.raises(_AuditParseError):
        _parse_audit_reply("OK but one issue: the resource never regenerates")
