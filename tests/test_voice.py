"""The voice chokepoint — saddle's user-facing language contract.

Locks the plain templates (structure + the words that carry the meaning), the
slug→plain mappings' fallback behavior, and that the speaking surfaces
(supervisor failure alerts, intent divergence renders) actually route through
the chokepoint instead of re-rolling their own jargon.
"""

from __future__ import annotations

from saddle import voice
from saddle.intent import IntentDivergence, IntentReport


def test_stage_plain_known_and_fallback():
    assert "request breakdown" in voice.stage_plain("intake")
    assert voice.stage_plain("brand_new_stage") == "brand_new_stage"


def test_failure_plain_known_and_fallback():
    assert voice.failure_plain("timeout") == "it ran out of time"
    assert voice.failure_plain("weird_new_cat") == "weird_new_cat"


def test_stage_failed_reads_plainly():
    text = voice.stage_failed(
        "intake", "timeout", "the request breakdown", "raise the ceiling.",
        "TimeoutError: ",
    )
    # Leads with what did not happen, names the plain cause, names what went
    # unchecked — and never leaks the bare category slug as the explanation.
    assert text.startswith("⚠ Saddle's request breakdown")
    assert "ran out of time" in text
    assert "went unchecked" in text
    assert "could not run (timeout)" not in text


def test_out_of_focus_names_project_and_stays_calm():
    text = voice.out_of_focus("Edit", "rayxiv4", "path: /elsewhere/x.py")
    assert "OUTSIDE rayxiv4" in text
    assert "was allowed" in text
    assert "OUT-OF-FOCUS" not in text  # the old shouting headline is retired


def test_design_issue_templates_lead_with_the_point():
    pre = voice.design_issues_pre_edit("  • issue one")
    end = voice.design_issues_turn_end("  • issue one")
    assert pre.startswith("Before code gets written")
    assert end.startswith("This turn proposed a plan but wrote no code")
    assert "issue one" in pre and "issue one" in end


def test_intent_divergence_renders_plain_headline_ref_last():
    d = IntentDivergence(
        kind="reopens_decision",
        what="asks to bring back the rejected cache design",
        why="the project closed this in the cache decision",
        ref="cache-decision-4",
    )
    out = d.render()
    lines = out.splitlines()
    # Plain headline first, no all-caps shouting, locator dead last.
    assert lines[0].startswith("• Goes against an earlier decision:")
    assert "RE-OPENS" not in out
    assert lines[-1].strip() == "(recorded as: cache-decision-4)"


def test_intent_report_sections_use_plain_head():
    r = IntentReport(
        divergences=[IntentDivergence(kind="scope_creep", what="x")], checked=True
    )
    (section,) = r.sections()
    assert section.startswith("This request conflicts with something the project already decided")
    assert "pulls against what the project has already settled" not in section


def test_kind_plain_fallback_never_shouts():
    assert voice.kind_plain("some_new_kind") == "some new kind"


def test_voice_contract_present_in_speaking_prompts():
    from saddle import design, intent

    assert "no engineering background" in design._SYS_AUDIT
    assert "plain everyday words" in design._SYS_HARVEST
    assert "plain everyday words" in intent._SYS_HISTORY
