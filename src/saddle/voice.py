"""saddle's voice — the ONE place its user-facing language is defined.

Founding directive (user, 2026-07-03, mediator design §3): everything saddle
says to a person is in common language. No unexplained technical terms, no
acronym soup, no idioms or regional expressions. A reader with no engineering
background follows it on first read.

Why a module and not a style note: a rule pasted into prompts and templates
decays — the same lesson that made the doctrine gate a hook. So the language
policy lives here as CODE every speaking surface imports:

* :data:`VOICE_CONTRACT` — the style contract appended to every LLM prompt
  whose output a person will read (stage findings, harvest lessons, discuss).
  Prevention at the source: the model writes plainly instead of being
  rewritten afterwards.
* The plain templates below — the recurring deterministic notices (a stage
  that could not run, an out-of-focus edit, a settled-state conflict), written
  once in plain words so no hook re-rolls its own jargon variant.
* :func:`audit_plainness` — the bounded LLM check that judges a rendered
  block against the contract, used at turn-end (where latency is cheap) and in
  tests. A finding is a loud alert, never a silent rewrite — saddle does not
  edit its own words behind the reader's back; it learns to write them right.

The STRUCTURAL join (header + section concatenation) stays in
:func:`saddle.supervisor.render_sections` — structure and language are
separate concerns; this module owns the words.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from saddle.context import Context

# ── The style contract for LLM-generated user-facing prose ─────────────────

VOICE_CONTRACT = (
    "\nHOW TO WRITE YOUR ANSWER (binding style contract):\n"
    "- Write for a reader with no engineering background.\n"
    "- Use everyday words. When a technical term is unavoidable, give its "
    "plain meaning right where it first appears, in the same sentence.\n"
    "- Name things fully: what it is and what it does — never a bare "
    "codename, abbreviation, or file path standing in for an explanation.\n"
    "- No idioms, slang, cultural or regional expressions, and no "
    "metaphors that assume the reader shares a culture.\n"
    "- Lead with the point — what happened or what is needed — then the "
    "detail.\n"
    "- Keep it short. No repetition, no filler, no hedging.\n"
)


# ── Plain names for saddle's own moving parts ───────────────────────────────

# Stage slug -> what that check IS, in plain words. Used anywhere a stage
# name would otherwise leak as jargon ("intake", "intent") into a person's
# reading.
STAGE_PLAIN: dict[str, str] = {
    "intake": "request breakdown (splitting your message into the things it asks for)",
    "intent": "conflict check (comparing the request against earlier decisions)",
    "design": "plan review (checking the approach before code is written)",
    "review": "approval posture (whether you asked to approve the plan before code)",
    "code": "code-matches-design check",
    "lesson": "lesson capture (recording what this turn taught)",
    "guard": "action gate (the check that runs before file changes)",
    "dialog": "conversation tracking",
}


def stage_plain(stage: str) -> str:
    """The plain-words name for a stage slug; the slug itself if unknown."""
    return STAGE_PLAIN.get(stage, stage)


# Failure category slug -> plain cause. Mirrors
# saddle.llm.retry_category.RETRY_CATEGORIES; a category with no entry falls
# back to the slug so a new category is never silently mistranslated.
_FAILURE_PLAIN: dict[str, str] = {
    "external_rate_limit": "the language-model service asked us to slow down",
    "provider_outage": "the language-model service was unreachable",
    "timeout": "it ran out of time",
    "empty_response": "the language model returned nothing",
    "parse_error": "the language model's answer came back malformed",
    "validation_failed": "the language model's answer failed saddle's checks",
    "output_too_large": "the answer was too large to handle",
    "input_too_large": "the request was too large for the language model",
    "provider_blocked": "the language-model service refused the request",
    "other": "an unexpected error",
}


def failure_plain(category: str) -> str:
    """The plain-words cause for a failure category slug."""
    return _FAILURE_PLAIN.get(category, category)


# ── Plain templates for the recurring deterministic notices ─────────────────


def stage_failed(stage: str, category: str, subject: str, remedy: str,
                 error: str) -> str:
    """A supervisory check that could not run — loud, plain, and specific about
    what therefore went UNCHECKED this turn (the anti-fail-open)."""
    return (
        f"⚠ Saddle's {stage_plain(stage)} did not run this turn — "
        f"{failure_plain(category)}. That means {subject} went unchecked. "
        f"{remedy} (technical detail for the developer: {error})"
    )


def out_of_focus(tool_name: str, project: str, detail: str) -> str:
    """An allowed-but-out-of-focus file change: the agent touched a file
    outside the project it is working on. Allowed, never silent."""
    return (
        f"Note: the agent used {tool_name} on a file OUTSIDE {project}, the "
        f"project it is working on. The change was allowed — please check it "
        f"was intended.\n{detail}"
    )


def design_issues_pre_edit(body: str) -> str:
    """Stage-3 finding, pre-edit variant: the plan has problems and code has
    not been written yet — the agent can still change course."""
    return (
        "Before code gets written: saddle reviewed the plan for this turn "
        f"and found problems worth fixing first:\n{body}"
    )


def design_issues_turn_end(body: str) -> str:
    """Stage-3 finding, turn-end variant: the turn proposed a plan in prose
    and wrote no code; the review runs so the NEXT turn starts corrected."""
    return (
        "This turn proposed a plan but wrote no code yet. Saddle reviewed "
        f"the proposal and found problems to fix before building it:\n{body}"
    )


def no_recorded_design() -> str:
    """Stage-3 finding: the agent edited code without laying out a plan."""
    return (
        "⚠ The agent started changing code without first laying out its "
        "plan. Saddle cannot review a plan that was never stated — the "
        "approach should be discussed before code is written."
    )


def design_gate_deny(body: str) -> str:
    """Strict-mode deny: the plan has unresolved problems, so the code edit is
    held until the plan is fixed — the discussion holds the floor."""
    return (
        "Saddle is holding this code change: the plan for this turn has "
        "problems that need resolving before code is written (strict design "
        f"gate is on):\n{body}\n"
        "To proceed: state your revised approach in your reply (saddle "
        "re-reviews it on your next edit attempt), or work it out with saddle "
        "directly via its design_propose tool. Once the plan reviews clean it "
        "is recorded as agreed and the gate opens."
    )


def design_settled(summary: str) -> str:
    """A clean plan was recorded as an agreed design — said once, briefly."""
    return (
        "Plan agreed and recorded. Saddle will check future code against it "
        f"and will not re-open this discussion unless the code drifts: {summary}"
    )


def design_hold_redirect(goal: str) -> str:
    """Design-Hold deny: the USER asked to see and approve the plan before any
    code is written, so this code change is held. The message REDIRECTS the
    agent — present the plan and wait for the user, do not keep retrying."""
    topic = f" for: {goal.strip()}" if (goal or "").strip() else ""
    return (
        "Saddle is holding this code change because you asked to review the "
        f"plan before any code is written{topic}. Present your plan to the "
        "user now and wait for their approval. Retrying the edit will not "
        "help — the change stays held until the user approves, then the gate "
        "opens on its own."
    )


def design_held_awaiting_approval(goal: str) -> str:
    """Design-Hold notice: the plan reviewed clean, but the user asked to be
    the one who approves it, so saddle records it as held (not auto-agreed) and
    waits for the user's word."""
    topic = f" for: {goal.strip()}" if (goal or "").strip() else ""
    return (
        "The plan looks sound, but you asked to approve it yourself before "
        f"code is written{topic}. Saddle is holding it for your approval "
        "instead of agreeing to it automatically. Tell the user the plan is "
        "ready, and it will proceed once they approve."
    )


def autonomy_engaged(condition: str = "") -> str:
    """Design-Hold notice: the user handed the assistant the wheel — proceed
    without stopping to get each plan approved. Names how to take it back."""
    extra = f" ({condition.strip()})" if (condition or "").strip() else ""
    return (
        "You are now in autonomous mode: the user asked you to proceed with "
        f"your own recommendations without stopping for plan approval{extra}. "
        "Keep working through the goal on your own judgment. This stays on "
        "until the user asks to review a plan or tells you to stop — either "
        "one takes the wheel back."
    )


def autonomy_reminder() -> str:
    """The per-turn on-screen reminder while autonomous mode is active, so the
    user always sees that the assistant is proceeding on its own judgment and
    knows how to end it."""
    return (
        "Autonomous mode is on: the assistant is proceeding on its own "
        "judgment without stopping for plan approval. To take the wheel back, "
        "ask to review a plan or tell it to stop."
    )


def approval_recorded(design_id: str = "") -> str:
    """Design-Hold notice: the user approved the plan, so saddle recorded it as
    an agreed design and opened the gate for code."""
    tail = f" (recorded as design {design_id})" if (design_id or "").strip() else ""
    return (
        "Approval recorded: the user approved the plan, so saddle has agreed "
        f"it and opened the gate{tail}. You can write the code now; saddle "
        "will check it against this plan from here on."
    )


def hold_released() -> str:
    """Design-Hold notice: an earlier hold or autonomous mode was cleared, so
    saddle is back to its normal behavior for this session."""
    return (
        "Saddle is back to its normal behavior for this session — the earlier "
        "request to hold or to proceed on your own has been cleared."
    )


def goal_keeper_reason(missing: list) -> str:
    """The stop-block reason: the goal is not finished, so keep working.
    Read by the AGENT (it resumes with this text) and shown to the human."""
    body = "\n".join(f"  • {m}" for m in missing) or "  • (the goal's remaining work)"
    return (
        "Saddle's goal-keeper: the goal you are driving is not finished, and "
        "you are not blocked on the user — so keep working instead of "
        f"stopping. Still open:\n{body}\n"
        "Pick the next open item and continue. If you are genuinely blocked "
        "on a decision only the user can make, ask that question directly."
    )


def turn_end_findings_head() -> str:
    """Header for the drain: findings saddle's end-of-turn checks produced
    AFTER the agent's last reply — the agent has not seen them yet."""
    return (
        "After your last reply, saddle's end-of-turn checks found the "
        "following. You have not seen these yet — address what still applies "
        "before continuing:"
    )


def settled_conflict_head() -> str:
    """Stage-2 header: the request conflicts with earlier decisions."""
    return (
        "This request conflicts with something the project already decided. "
        "If you mean to change that decision, say so and saddle will record "
        "the change; otherwise the earlier decision stands:"
    )


# Stage-2 divergence-kind headlines, plain. Keys mirror saddle.intent kinds.
KIND_PLAIN: dict[str, str] = {
    "contradicts_design": "Conflicts with an earlier design",
    "reopens_decision": "Goes against an earlier decision",
    "scope_creep": "Outside this project's current focus",
}


def kind_plain(kind: str) -> str:
    """Plain headline for a divergence kind; the slug itself if unknown."""
    return KIND_PLAIN.get(kind, kind.replace("_", " "))


# ── The bounded plainness check (enforcement, not aspiration) ───────────────

_SYS_PLAINNESS = (
    "You judge whether a short status message is readable by a person with "
    "no engineering background. Flag ONLY these problems:\n"
    "- a technical term, abbreviation, or codename used with no plain-words "
    "explanation at first use;\n"
    "- an idiom, slang, or cultural/regional expression;\n"
    "- the point buried: the reader cannot tell what happened or what is "
    "needed from the first sentence;\n"
    "- meaningless filler that says nothing.\n"
    "Do NOT flag: file paths or identifiers given alongside a plain "
    "explanation, ordinary words, brevity, or the name 'saddle' — that is "
    "the speaking assistant's own name and the reader knows it.\n"
    'Answer as JSON: {"issues": ["<one line per problem, quoting the '
    'offending words>", ...]} — an empty list when the text reads plainly.'
)


async def audit_plainness(text: str, ctx: "Context | None" = None) -> list[str]:
    """Judge ``text`` against the voice contract; return one line per problem
    (empty = reads plainly). Bounded by the ambient stage budget like every
    other supervisory LLM call; raises on caller failure so :func:`run_stage`
    classifies it loudly — a plainness check that cannot run must never pass
    silently."""
    body = (text or "").strip()
    if not body:
        return []
    from saddle.context import default as default_ctx
    from saddle.llm.callers import build_callers
    from saddle.llm.json_tools import extract_json_text

    import json

    caller = build_callers(ctx or default_ctx())["default"]
    raw = await caller(_SYS_PLAINNESS, body, json_mode=True, label="voice/plainness")
    doc = json.loads(extract_json_text(raw))
    issues = doc.get("issues") if isinstance(doc, dict) else None
    if not isinstance(issues, list):
        raise ValueError("plainness check returned no 'issues' list")
    return [str(i).strip() for i in issues if str(i).strip()]
