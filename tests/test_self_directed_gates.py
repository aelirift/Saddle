"""Both supervisory gates must recognize SELF-DIRECTED work.

saddle develops and maintains itself; a request to change its own code, gates,
prompts, or configuration is the project's own legitimate work, not drift. Two
gates kept misfiring on exactly that:

* the intent gate (Stage 2, :data:`saddle.intent._SYS_HISTORY`) flagged building
  a new saddle feature as ``scope_creep`` and a user-directed change to a prior
  decision as ``reopens_decision``;
* the design gate (Stage 3, :data:`saddle.design._SYS_AUDIT`) blanket-applied
  binding directives written for a DIFFERENT task (producing a target project's
  artifacts) to unrelated self-directed work.

Both gates are LLM-judged, so the tune lives in their instruction — exactly like
the precedent in :data:`saddle.intake._SYSTEM_SCOPE`, which already treats
"configuring this assistant" as in-focus. These tests pin that the recognition
clause is PRESENT (a contract pin: a silent revert of the tune fails here) and
that the precedent it mirrors still says the same thing.
"""
from __future__ import annotations

from saddle.design import _SYS_AUDIT
from saddle.intake import _SYSTEM_SCOPE
from saddle.intent import _SYS_HISTORY


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def test_intent_gate_recognizes_self_directed_work():
    p = _norm(_SYS_HISTORY)
    # self-directed work on the assistant itself is explicitly not a divergence
    assert "self-directed work is not a divergence" in p
    assert "configure the assistant" in p or "configuring the assistant" in p
    # the settled designs are a subset, not the boundary -> new work is not creep
    assert "subset of the project's work" in p
    assert "not scope_creep" in p
    # the user re-deciding is legitimate; only a SILENT/unaware contradiction flags
    assert "re-deciding" in p
    assert "unawares" in p


def test_design_gate_judges_directive_applicability():
    p = _norm(_SYS_AUDIT)
    assert "applicability first" in p
    # self-directed work is not bound by directives meant for the built projects
    assert "self-directed work" in p
    assert "not bound by" in p
    # but genuinely-applicable directives are still enforced hard (no weakening)
    assert "enforce the directives that do apply" in p


def test_self_directed_recognition_mirrors_the_scope_gate_precedent():
    # The intake scope gate already treats configuring this assistant as in-focus;
    # the two new clauses extend that same recognition to the other two gates, so
    # the precedent must still hold (else they no longer mirror a live contract).
    s = _norm(_SYSTEM_SCOPE)
    assert "configuring this assistant counts as focus" in s
