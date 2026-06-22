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

from saddle.context import Context
from saddle.design import design_for
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
            return json.dumps(verdict)
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
