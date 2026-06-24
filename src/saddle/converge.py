"""Layer 4 — the convergence controller: drive the coder to implement a design
until its completeness surface is satisfied.

WHY THIS EXISTS
---------------
Layers 1-3 produce a design and a way to CHECK it, but nothing DRIVES code to the
design. A human — or a fresh Claude — handed the prose still hits the exact
failure this whole harness exists to prevent: the design is sophisticated, but
once the model starts coding it FLATTENS. It implements a generic version, wires
half the surface, and announces it is done. ``design.py`` proved a model CAN
design above the surface; this layer proves the same model can be MADE to code to
that design without collapsing — by never handing over the whole task at once and
never trusting the coder's own claim of doneness.

THE LOOP (one design at a time)
-------------------------------
  load     — the design's persisted ``SurfaceManifest`` (``design.meta['surface']``)
             is the locked completeness floor: every code site the change must keep
             consistent.
  gate     — :func:`saddle.design.intent_drift` re-parses the target tree FRESH and
             returns the gaps (unsatisfied touchpoints). Empty == the code already
             satisfies the design.
  brief    — the coder (a tool-capable :class:`~saddle.llm.claude_agent.ChatSession`,
             Claude-Code-like) gets the design body, the full impact set
             (``manifest.format`` — "hand the implementer the full impact set"), and
             the CURRENT gaps. It edits real code.
  re-gate  — after EVERY coder turn the gate re-runs against fresh code. The coder's
             textual "done" is never accepted; the gate decides.
  re-prompt— remaining gaps go back as effect-grounded, finding-by-finding feedback
             (never "gate failed, try again"), with the progress delta so the coder
             sees what it just closed.
  terminate— three self-terminating exits (the watchdog rule): SUCCESS (gaps empty),
             STALL (the same gap set recurs — no progress / oscillation), EXHAUSTED
             (hard round cap). On any non-success the controller HALTS and surfaces
             the unsatisfied sites loudly — no silent degradation, no swap to a
             non-coding model.

WHAT IS DEFERRED (documented scope, not a band-aid)
---------------------------------------------------
The dogfood "Convergence Controller" design specifies edit-grained mutation
interception (gate after every Edit *inside* a turn via PostToolUse hooks), orphan
promotion (the floor grows upward as the coder writes new tracked sites), a
TotalCover meta-gate, and borrowed-invariants across multiple co-designed units.
Those need a richer coder Protocol (``set_mutation_interceptor``) and a
``CodeDerivedSurface`` from the gate, which today returns only gap ``Finding``\\ s.
This layer realizes the load-bearing structure — deny the whole picture, gate
every step against FRESH code, effect-grounded re-prompts, never trust a
self-claim, halt loudly — at TURN granularity, and leaves edit-granularity +
multi-unit ownership as the next layer up.

GAME-AGNOSTIC / STANDALONE
--------------------------
The controller imports no game module and no Agent SDK. The coder and the gate are
INJECTED: the default coder is a thin ``ChatSession`` adapter (the only SDK-aware
piece, lazily imported); the default gate is ``intent_drift`` bound to a code root.
Swapping either is writing a new adapter, not touching the loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from saddle.codemap import Finding, SurfaceManifest, refs
from saddle.design import intent_drift

if TYPE_CHECKING:  # pragma: no cover
    from saddle.context import Context
    from saddle.dkb import DKB
    from saddle.models import Design

_log = logging.getLogger("saddle.converge")

# --- terminal outcomes ----------------------------------------------------
CONVERGED = "converged"            # the gate is clean — code satisfies the design
ALREADY = "already_satisfied"      # the gate was clean before any coding ran
STALLED = "stalled"                # the same gap set recurred — no progress
EXHAUSTED = "exhausted"            # hit the hard round cap with gaps remaining
NO_SURFACE = "no_surface"          # design declared no surface — gate can't verify
CODER_FAILED = "coder_failed"      # coder turn crashed past its retries — halt
CODER_UNAVAILABLE = "coder_unavailable"  # coder could not START (claude CLI unreachable) — halt fast, never hang

_OK_OUTCOMES: frozenset[str] = frozenset({CONVERGED, ALREADY})

# saddle's coder persona. The discipline lives in the system prompt, but the
# load-bearing enforcement is the gate (Section "never trust a self-claim").
_CODER_SYSTEM = (
    "You are saddle's implementation coder: a disciplined senior engineer who "
    "implements a DESIGN into a real codebase EXACTLY, without flattening it to a "
    "generic version. You work against a LOCKED completeness surface — a set of "
    "code sites that must stay consistent. A separate gate re-checks the real code "
    "after every one of your turns; your own claim that the work is done is never "
    "trusted, so do not announce completion — just make the code satisfy the "
    "surface. Make real edits (real reads/writes/calls at the named sites), never "
    "stubs, comments, or placeholders. No band-aids, no hard-coding; implement the "
    "design's structure faithfully."
)


class Coder(Protocol):
    """The injected coder. An async context manager (open/close a live session)
    that takes one turn at a time. Implemented for production by
    :class:`ChatSessionCoder`; a fake satisfies the same shape in tests."""

    async def __aenter__(self) -> "Coder": ...
    async def __aexit__(self, *exc) -> None: ...
    async def turn(self, prompt: str) -> str: ...


class CoderTurnError(RuntimeError):
    """A coder turn crashed and did not recover within its bounded retries.
    Carries the last underlying error so halt-and-surface stays specific."""


@dataclass
class Round:
    """One coder-turn + re-gate cycle, recorded for the audit trail."""

    n: int
    gaps_before: int
    gaps_after: int
    closed: int
    findings_after: list[Finding] = field(default_factory=list)
    note: str = ""


@dataclass
class ConvergeResult:
    """The outcome of driving one design to (or short of) convergence."""

    design_id: str
    status: str
    rounds: list[Round] = field(default_factory=list)
    final_gaps: list[Finding] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status in _OK_OUTCOMES


class ChatSessionCoder:
    """Default coder: a tool-capable :class:`ChatSession` scoped to ``code_root``.

    The ONLY SDK-aware piece in this layer; it lazily imports ``claude_agent`` so
    ``import saddle.converge`` works on a host without the Agent SDK installed.
    """

    def __init__(
        self,
        *,
        cwd: str,
        system_prompt: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> None:
        self._cwd = str(cwd)
        self._system_prompt = system_prompt or _CODER_SYSTEM
        self._model = model
        self._effort = effort
        # Live stream sink: each text chunk the coder emits is handed to this as
        # it arrives (the CLI points it at stderr), so a turn is watchable in
        # real time instead of surfacing only after it completes.
        self._on_chunk = on_chunk
        self._session = None

    async def __aenter__(self) -> "ChatSessionCoder":
        from saddle.llm.claude_agent import ChatSession

        self._session = ChatSession(
            cwd=self._cwd,
            system_prompt=self._system_prompt,
            model=self._model,
            effort=self._effort,
        )
        await self._session.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def turn(self, prompt: str) -> str:
        assert self._session is not None
        parts: list[str] = []
        async for chunk in self._session.ask(prompt):
            if self._on_chunk is not None:
                self._on_chunk(chunk)
            parts.append(chunk)
        if self._on_chunk is not None and parts:
            self._on_chunk("\n")  # terminate the streamed block for the next line
        return "".join(parts)


def _sig(findings: list[Finding]) -> frozenset[str]:
    """Order-independent signature of a gap set — ``Finding`` carries a dict
    ``detail`` so it is not hashable; its ``__str__`` is the stable key."""
    return frozenset(str(f) for f in findings)


def _format_findings(findings: list[Finding]) -> str:
    return "\n".join(f"  {f}" for f in findings) if findings else "  (none)"


def _surface_text(design: "Design", code_root: str) -> str:
    """The full impact set of the design's declared surface, for the first brief.
    Best-effort: a parse failure just drops this context — the gaps still drive."""
    manifest = SurfaceManifest.from_dict(design.meta.get("surface"))
    if manifest.is_empty():
        return ""
    try:
        mods = refs.parse_project(code_root)
        return manifest.format(mods, root=code_root)
    except Exception:  # noqa: BLE001 — context, not contract
        _log.warning("converge: could not format surface for %s", design.id, exc_info=True)
        return ""


def _first_brief(
    design: "Design", surface_text: str, gaps: list[Finding], directives: list[str]
) -> str:
    parts = [
        "Implement this design into the codebase. Drive the real code until the "
        "completeness surface below is fully satisfied.",
        f"DESIGN: {design.summary or design.ask}",
        design.body,
    ]
    if directives:
        rules = "\n".join(f"- {d}" for d in directives)
        parts.append(f"BINDING RULES (must hold):\n{rules}")
    if surface_text:
        parts.append(
            "COMPLETENESS SURFACE — every code site this design must keep "
            "consistent (the full impact set; you need not rediscover what to "
            f"touch):\n{surface_text}"
        )
    parts.append(
        "CURRENT GAPS — the gate ran against the real code and these touchpoints "
        "are UNSATISFIED. Close every one with real code (a real read/write/call at "
        f"the named site):\n{_format_findings(gaps)}"
    )
    parts.append(
        "Make the edits now. After your turn the gate re-runs against the real "
        "code; any remaining gaps come straight back to you. Do not declare the "
        "work done — the gate decides."
    )
    return "\n\n".join(parts)


def _reprompt_brief(gaps: list[Finding], closed: list[Finding]) -> str:
    parts = ["The gate re-ran against the real code after your last turn."]
    if closed:
        parts.append(
            "CLOSED since last turn (good — keep these satisfied):\n"
            f"{_format_findings(closed)}"
        )
    parts.append(
        "STILL UNSATISFIED — close each of these with real code at the named site, "
        f"and do not regress what you just fixed:\n{_format_findings(gaps)}"
    )
    parts.append("Make the edits now. The gate will re-verify; do not declare done.")
    return "\n\n".join(parts)


async def _coder_turn(coder: Coder, prompt: str, *, retries: int) -> str:
    """Take one coder turn, retrying a crashed turn a bounded number of times.

    The coder is the lead Claude with no tool-capable fallback (minimax cannot
    drive Edit/Write), so a transient CLI/transport crash gets a few retries — and
    then HALTS (raises :class:`CoderTurnError`) rather than degrading. Only the
    ``turn`` call is guarded, so a bug in the surrounding loop still surfaces."""
    last: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            return await coder.turn(prompt)
        except Exception as exc:  # noqa: BLE001 — transient coder crash, bounded-retried
            last = exc
            _log.warning(
                "converge: coder turn failed (attempt %d/%d): %s",
                attempt + 1, retries + 1, str(exc)[:160],
            )
    raise CoderTurnError(str(last)) from last


def _persist(result: ConvergeResult, design: "Design", ctx: "Context | None",
             dkb: "DKB | None") -> None:
    """Record the convergence trail on the design (``design.meta['convergence']``)
    so a halted run leaves a durable, inspectable artifact — the design's own
    status is left untouched (an implementation outcome is not a design verdict)."""
    if ctx is None or not design.id:
        return
    from saddle.dkb import get_dkb

    trail = {
        "outcome": result.status,
        "rounds": [
            {"n": r.n, "gaps_before": r.gaps_before, "gaps_after": r.gaps_after,
             "closed": r.closed}
            for r in result.rounds
        ],
        "final_gaps": [str(f) for f in result.final_gaps],
        "error": result.error,
    }
    try:
        (dkb or get_dkb()).update_design_meta(ctx, design.id, {"convergence": trail})
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort
        _log.warning("converge: could not persist trail for %s: %s", design.id, exc)


async def converge_design(
    design: "Design",
    *,
    code_root: str | Path,
    coder: Coder | None = None,
    gate: Callable[[], list[Finding]] | None = None,
    directives: list[str] | None = None,
    ctx: "Context | None" = None,
    dkb: "DKB | None" = None,
    max_rounds: int = 8,
    stall_repeat: int = 3,
    turn_retries: int = 2,
    persist: bool = True,
    on_chunk: Callable[[str], None] | None = None,
) -> ConvergeResult:
    """Drive ``design`` into the code at ``code_root`` until its completeness
    surface is satisfied, gating after every coder turn.

    Injection points keep the loop game-agnostic and testable: ``coder`` defaults
    to a :class:`ChatSessionCoder` scoped to ``code_root``; ``gate`` defaults to
    :func:`saddle.design.intent_drift` bound to ``code_root`` (re-parsing fresh
    code every call). Both can be replaced without touching the loop. ``on_chunk``
    (when no explicit ``coder`` is given) streams the default coder's live output
    chunk-by-chunk — the CLI points it at stderr so a run is watchable.

    Returns a :class:`ConvergeResult` carrying the per-round trail and the final
    gap set. Never raises on a coder crash — it halts with ``status=CODER_FAILED``
    and the gaps surfaced — so the caller (CLI) maps the outcome to an exit code.
    """
    code_root = str(Path(code_root).expanduser())
    if gate is None:
        def gate() -> list[Finding]:  # bound to this design + root
            return intent_drift(design, root=code_root)

    manifest = SurfaceManifest.from_dict(design.meta.get("surface"))
    if manifest.is_empty():
        # No declared surface ⇒ the gate cannot verify anything. Report it rather
        # than spin a loop whose exit condition can never be observed.
        result = ConvergeResult(design.id, NO_SURFACE)
        if persist:
            _persist(result, design, ctx, dkb)
        _log.info("converge: design %s declared no surface — nothing to gate", design.id)
        return result

    gaps = await asyncio.to_thread(gate)
    if not gaps:
        result = ConvergeResult(design.id, ALREADY)
        if persist:
            _persist(result, design, ctx, dkb)
        return result

    # Only now — about to brief the coder — resolve the binding rules and the full
    # impact set; the trivial exits above never pay for either.
    if directives is None:
        directives = _resolve_directives(ctx)
    surface_text = await asyncio.to_thread(_surface_text, design, code_root)
    rounds: list[Round] = []
    history: list[frozenset[str]] = []
    closed: list[Finding] = []
    if coder is None:
        coder = ChatSessionCoder(cwd=code_root, on_chunk=on_chunk)

    try:
        await coder.__aenter__()
    except Exception as exc:  # noqa: BLE001 — coder couldn't START; halt fast
        # The coder could not even open a session (e.g. the `claude` CLI is
        # unreachable, so connect hit its deadline). Halt with a DISTINCT outcome
        # instead of hanging or masquerading as a mid-coding crash — the gate has
        # nothing to verify because no code was ever written.
        result = ConvergeResult(
            design.id, CODER_UNAVAILABLE, rounds=rounds, final_gaps=gaps,
            error=str(exc),
        )
        if persist:
            _persist(result, design, ctx, dkb)
        _log.warning("converge: design %s halted — coder unavailable: %s",
                     design.id, str(exc)[:160])
        return result

    try:
        for n in range(1, max(1, max_rounds) + 1):
            prompt = (
                _first_brief(design, surface_text, gaps, directives)
                if n == 1
                else _reprompt_brief(gaps, closed)
            )
            _log.info("converge: design %s round %d/%d — coder working on %d gap(s)",
                      design.id, n, max(1, max_rounds), len(gaps))
            try:
                await _coder_turn(coder, prompt, retries=turn_retries)
            except CoderTurnError as exc:
                result = ConvergeResult(
                    design.id, CODER_FAILED, rounds=rounds, final_gaps=gaps,
                    error=str(exc),
                )
                if persist:
                    _persist(result, design, ctx, dkb)
                _log.warning("converge: design %s halted — coder failed: %s",
                             design.id, str(exc)[:160])
                return result

            # Re-gate against FRESH code — the coder's self-claim is never accepted.
            after = await asyncio.to_thread(gate)
            after_sig = _sig(after)
            closed = [f for f in gaps if str(f) not in after_sig]
            rounds.append(Round(
                n=n, gaps_before=len(gaps), gaps_after=len(after),
                # the TRUE set-difference count — a count delta (before-after)
                # reads 0 when a round closes one gap and opens another, and
                # would disagree with the `closed` list, the log, and the reprompt
                closed=len(closed), findings_after=after,
            ))
            _log.info("converge: design %s round %d — %d gap(s) -> %d (closed %d)",
                      design.id, n, len(gaps), len(after), len(closed))

            if not after:
                result = ConvergeResult(design.id, CONVERGED, rounds=rounds)
                if persist:
                    _persist(result, design, ctx, dkb)
                return result

            # Stall: the same unsatisfied set recurring (stuck OR oscillating).
            history.append(after_sig)
            if history.count(after_sig) >= max(2, stall_repeat):
                result = ConvergeResult(design.id, STALLED, rounds=rounds, final_gaps=after)
                if persist:
                    _persist(result, design, ctx, dkb)
                _log.warning("converge: design %s halted — stalled on %d gap(s)",
                             design.id, len(after))
                return result
            gaps = after
    finally:
        try:
            await coder.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001 — session close is best-effort
            pass

    # Hard cap — a liveness net; stall detection usually fires first.
    result = ConvergeResult(design.id, EXHAUSTED, rounds=rounds, final_gaps=gaps)
    if persist:
        _persist(result, design, ctx, dkb)
    _log.warning("converge: design %s halted — exhausted %d round(s), %d gap(s) left",
                 design.id, max_rounds, len(gaps))
    return result


def _resolve_directives(ctx: "Context | None") -> list[str]:
    if ctx is None:
        return []
    try:
        from saddle.llm import policy
        return policy.directives(ctx)
    except Exception:  # noqa: BLE001 — directives are guidance, not a hard dep
        return []


def format_result(result: ConvergeResult, code_root: str) -> str:
    """Human-readable rendering of a convergence outcome for the CLI."""
    lines = [
        f"converge {result.design_id}: {result.status.upper()} against {code_root}"
    ]
    for r in result.rounds:
        lines.append(
            f"  round {r.n}: {r.gaps_before} gap(s) -> {r.gaps_after} (closed {r.closed})"
        )
    if result.error:
        lines.append(f"  error: {result.error}")
    if result.final_gaps:
        lines.append(f"\n{len(result.final_gaps)} unsatisfied touchpoint(s) at halt:")
        lines.extend(f"  {f}" for f in result.final_gaps)
    elif result.ok:
        lines.append("  surface COMPLETE — code satisfies the design")
    return "\n".join(lines)
