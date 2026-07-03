"""The staged supervisory runner — saddle's five live drift checkpoints, wired.

WHY THIS EXISTS
---------------
Supervision is not a single gate. Drift — divergence from the user's intent, the
project's established design, or good engineering practice — enters a turn at
FIVE distinct points, and a lone "does the proposed design match the vision?"
check (fired once, just before code) is blind to four of them:

  1. intake  — the prompt is mis-/under-understood before any work starts.
  2. intent  — this prompt pulls against the project's own design / history.
  3. design  — the design is a band-aid / misreads the root cause.
  4. code    — the code drifts from the design mid-implementation.
  5. lesson  — the lesson from this turn is never captured, so it recurs.

So supervision is STAGED across the whole turn (one check per entry point), and
this module is the one place that owns those stages — named, individually
bubbling, and each independently FAIL-CLASSIFIED. The full contract lives in
``docs/design/supervisory_pipeline.md``; the distilled principle is seeded as
``knowledge_seed_supervision_is_staged``.

THE DISCIPLINE THIS FRAMEWORK ENFORCES
--------------------------------------
Every stage runs through :func:`run_stage`, which guarantees the property the old
intake fail-open violated: **a stage that cannot run says so LOUDLY.** A stage
either

  * ran and found something to surface  -> a bubble at the finding's level;
  * ran and found nothing               -> silence (nothing drifted) — but only
                                           because it actually RAN to completion;
  * could not run (timeout, provider
    outage, a bug)                      -> a classified **ALERT** naming what
                                           saddle did NOT verify this turn.

The third case is the whole point. A swallowed exception that degrades to a
quiet "(unavailable)" string is exactly the silent fail-open
``knowledge_seed_fail_loud`` / ``knowledge_seed_silent_fallback`` forbid:
supervision didn't run and nobody decided to skip it. Here, failure is
classified (reusing the LLM retry taxonomy — wall-clock vs provider outage vs an
oversized-input contract gap) and bubbled as an ALERT, so the human and the
agent both learn the check was incomplete.

TWO CHANNELS, DELIBERATELY DIFFERENT
------------------------------------
* The **bubble** channel is PER-STAGE: each stage emits its own durable
  :class:`~saddle.models.BubbleEvent`, so a client can filter by stage, a stage
  failure is its own ALERT (not buried inside a NOTICE batch), and the render
  level is per-stage (intake ran fine = notice; intent found drift = alert).
  ``run_stage`` owns this channel — it is identical across every caller.
* The **agent-context** channel is COMBINED: a hook joins each stage's sections
  into one ``additionalContext`` blob the model reads once. :func:`agent_context`
  builds it. The stdout PROTOCOL around it (UserPromptSubmit's
  ``additionalContext`` vs PreToolUse's ``permissionDecision``) differs per hook,
  so it stays the hook's job — this module owns only the shared policy.

This module is presentation + orchestration; the VERDICTS stay owned by the
engines (``intake`` / ``dialog`` / ``design`` / DKB). A bubble carries a finding;
it is not itself the judgement (mirrors the :class:`BubbleEvent` contract note).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from saddle.bubble import emit_bubble
from saddle.context import Context
from saddle.llm.retry_category import categorize_retry, describe_category
from saddle.models import (
    BUBBLE_ALERT,
    BUBBLE_NOTICE,
    BubbleEvent,
)

_log = logging.getLogger("saddle.supervisor")


# -- what a stage RETURNS when it ran ----------------------------------------

@dataclass
class StageOutcome:
    """What a supervisory stage produced when it RAN to completion.

    ``sections`` are the human-readable text blocks the stage wants surfaced
    (empty = ran, nothing to say — a *silent success*, distinct from could-not-
    run). ``level`` is how loudly to render them: a routine stage result is a
    :data:`~saddle.models.BUBBLE_NOTICE`; a caught drift / band-aid is a
    :data:`~saddle.models.BUBBLE_ALERT`. ``title`` is an optional compact
    headline; ``meta`` is any structured payload to ride on the bubble (e.g. the
    raw findings, the intake id) for a richer client.
    """

    sections: list[str] = field(default_factory=list)
    level: str = BUBBLE_NOTICE
    title: str = ""
    meta: dict = field(default_factory=dict)


# -- what run_stage RETURNS to the caller ------------------------------------

@dataclass
class StageResult:
    """The outcome of running one stage through :func:`run_stage`.

    ``ok`` means the stage RAN to completion — NOT that it found nothing. A stage
    that ran and surfaced a drift is ``ok=True`` with a non-empty ``sections`` and
    an ``alert`` ``level``; only a stage that could not run is ``ok=False``, and
    then ``failure`` carries its classified category (e.g. ``timeout``,
    ``provider_outage``, ``input_too_large``) and ``remedy`` a one-line hint.
    ``bubble`` is the event emitted (``None`` when the stage was a silent
    success). ``sections`` is what the agent-context channel joins.
    """

    stage: str
    ok: bool = True
    sections: list[str] = field(default_factory=list)
    level: str = BUBBLE_NOTICE
    failure: str = ""
    remedy: str = ""
    bubble: BubbleEvent | None = None

    @property
    def spoke(self) -> bool:
        """Did this stage emit a bubble (find/fail), or stay silent (clean run)?"""
        return self.bubble is not None


# -- shared presentation -----------------------------------------------------

def render_sections(ctx: Context, sections: list[str]) -> str:
    """The one block-render every supervisory channel shares: a ``saddle [key]``
    header over the stage's sections, so the bubble outbox, the hook's stderr,
    and the agent-context blob never drift in how saddle's voice reads. Empty /
    whitespace-only sections are dropped so a render is never a header alone."""
    body = [s for s in sections if s and s.strip()]
    return f"━━ saddle [{ctx.key}] ━━\n" + "\n\n".join(body) if body else ""


def agent_context(ctx: Context, results: list[StageResult]) -> str:
    """Join every stage's sections into ONE ``additionalContext`` blob for the
    model — the combined agent channel (the bubbles already went out per-stage).
    Empty when no stage had anything to say, so a hook can skip the stdout emit."""
    sections = [s for r in results for s in r.sections]
    return render_sections(ctx, sections)


def system_message(
    ctx: Context,
    results: list["StageResult"],
    *,
    max_chars: int = 1800,
) -> str:
    """The USER-SCREEN channel — the digest a hook hands Claude Code's
    ``systemMessage`` stdout field, the ONE output path the harness renders to the
    watching HUMAN.

    Distinct from the other two channels and necessary because neither reaches a
    person: :func:`agent_context` feeds the MODEL's ``additionalContext`` (saddle's
    voice lands in the agent's context, never on screen), and a hook's stderr is
    shown only in an interactive TTY — under an SDK / service host
    (``CLAUDE_CODE_ENTRYPOINT=sdk-py``) it is swallowed. The durable bubble outbox
    persists every word but has no live display surface a human is watching. So
    without ``systemMessage`` saddle is invisible to the user it supervises — the
    exact "I don't see any outputs from saddle on screen" gap.

    The digest is deliberately NARROWER than the agent channel: only a stage that
    SPOKE (found drift, surfaced an itemization, or could-not-run) contributes —
    the routine readouts the model still wants (e.g. the standing commitment) are
    NOT screen heralds, so a clean turn where every stage was a silent success
    returns ``""`` and the caller skips the field. Capped so a long design alert
    can't flood the surface — the FULL text always remains in the outbox, this is
    only the on-screen herald."""
    spoke = [r for r in results if r.spoke]
    sections = [s for r in spoke for s in r.sections]
    body = render_sections(ctx, sections)
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n… (full detail in saddle outbox)"
    return body


# -- failure classification --------------------------------------------------

def classify_failure(exc: BaseException) -> tuple[str, str]:
    """Classify a stage failure into ``(category, remedy)`` using the SAME
    taxonomy the LLM-retry layer uses (:func:`categorize_retry`), so a wall-clock
    deadline, a provider outage, and an oversized-input contract gap are told
    apart by every surface that reports them. Both supervise liveness errors
    (:class:`~saddle.supervise.DeadlineExceeded` / ``Stalled``) subclass
    ``TimeoutError`` and so classify as ``timeout`` — the recurring intake hang."""
    category = categorize_retry(exception=exc)
    return category, describe_category(category)


# -- the core: run one stage with fail-classification + bubbling -------------

def run_stage(
    ctx: Context,
    stage: str,
    fn: Callable[[], StageOutcome | None],
    *,
    session: str = "",
    what: str = "",
) -> StageResult:
    """Run one supervisory stage with the discipline every stage shares.

    ``fn`` does the stage's work and returns a :class:`StageOutcome` (or ``None``
    for "ran, nothing to say"), or raises. ``what`` names what this stage verifies
    (e.g. "the prompt decomposition") for the failure message.

    Three outcomes, never a fourth:

    * **ran, produced sections** -> emit one bubble at the outcome's level;
      ``ok=True``.
    * **ran, produced nothing**  -> stay silent; ``ok=True``, no bubble (nothing
      drifted — but the stage genuinely ran).
    * **raised**                 -> classify the failure and emit an **ALERT**
      naming what saddle could not verify this turn; ``ok=False``. This is the
      generalization of the intake fail-loud fix: a stage NEVER silently no-ops
      and implies it passed.

    ``Exception`` is caught (covering the timeout / provider / contract errors a
    stage can hit); ``BaseException`` control-flow (``KeyboardInterrupt``,
    ``SystemExit``, ``asyncio.CancelledError``) is left to propagate — a
    cancelled turn is not a stage finding.
    """
    try:
        outcome = fn()
    except Exception as exc:  # noqa: BLE001 — a stage failure is SURFACED, never swallowed
        return _failed(ctx, stage, exc, session=session, what=what)

    sections = [s for s in (outcome.sections if outcome else []) if s and s.strip()]
    if not sections:
        return StageResult(stage=stage, ok=True)  # silent success — it ran, found nothing

    level = outcome.level if outcome else BUBBLE_NOTICE
    bubble = emit_bubble(
        ctx,
        render_sections(ctx, sections),
        level=level,
        stage=stage,
        title=(outcome.title if outcome else ""),
        session=session,
        meta=(outcome.meta if outcome else {}),
    )
    return StageResult(
        stage=stage, ok=True, sections=sections, level=level, bubble=bubble
    )


def _failed(
    ctx: Context, stage: str, exc: BaseException, *, session: str, what: str
) -> StageResult:
    """Turn a stage exception into a LOUD, classified ALERT — the anti-fail-open.

    Names the failure category, what saddle therefore did NOT verify this turn,
    and the remediation hint, so the gap is impossible to mistake for a pass."""
    category, remedy = classify_failure(exc)
    subject = what or f"stage {stage}"
    from saddle.voice import stage_failed

    text = stage_failed(
        stage, category, subject, remedy, f"{type(exc).__name__}: {exc}"
    )
    _log.warning("supervisory stage %s failed (%s): %r", stage, category, exc)
    bubble = emit_bubble(
        ctx,
        render_sections(ctx, [text]),
        level=BUBBLE_ALERT,
        stage=stage,
        title=f"{stage} unavailable",
        session=session,
        meta={"failure": category, "what": subject, "error": repr(exc)},
    )
    return StageResult(
        stage=stage,
        ok=False,
        sections=[text],
        level=BUBBLE_ALERT,
        failure=category,
        remedy=remedy,
        bubble=bubble,
    )


# -- async -> sync bridge for the stage bodies -------------------------------

def run_bounded(coro: Awaitable, *, seconds: float, what: str):
    """Drive an async supervisory step to completion under a DEADLINE, from the
    sync stage protocol. The deadline raises a typed
    :class:`~saddle.supervise.DeadlineExceeded` that :func:`run_stage` classifies
    and bubbles — so a wedged ``decompose`` / ``design_for`` becomes a loud ALERT
    instead of an unbounded hang or a swallowed timeout. ``seconds <= 0`` waits
    unbounded (an explicit opt-out, never the silent default that caused the
    150s converge hang)."""
    from saddle import supervise

    return asyncio.run(supervise.bounded(coro, seconds=seconds, what=what))
