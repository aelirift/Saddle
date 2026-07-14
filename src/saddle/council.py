"""The design council — saddle chairs a two-critic panel for high-blast-radius work.

WHY THIS EXISTS
---------------
The pre-code design gate audits an approach with ONE model against the DKB
(:func:`saddle.design.audit_proposal`). That is right for routine work, but a
change with a wide blast radius — one that touches many code sites, alters a
settled design, or stands up a new subsystem — deserves more than a single
reviewer's read. A lone judge misses what a second, differently-mandated lens
would catch, and a single model's blind spots become the harness's blind spots.

So for NON-TRIVIAL work saddle CONVENES a council: it chairs, and pulls in
exactly TWO critic language models of DISTINCT model families (never two clones
of the same model — that would just double one blind spot). Each critic has a
separate, non-overlapping mandate and MUST name a concrete flaw or an explicit
PASS — no hedged "it's fine but". saddle then SYNTHESISES the two critiques into
one reconciled design; a genuinely unresolved conflict is surfaced to the USER as
a fork, never settled by fiat. Consensus still cannot launder a band-aid: the
existing anti-pattern audit floor (:func:`saddle.design.audit_proposal`) runs on
the synthesis before anything is settled.

TRIAGE — WHEN THE COUNCIL CONVENES
----------------------------------
Cost and latency are bounded by triaging on BLAST RADIUS. The design's
completeness surface (:class:`saddle.codemap.SurfaceManifest`) is computed first;
an empty or tiny surface SKIPS the council and keeps the single
``audit_proposal``. The council convenes only for a multi-site surface (or an
explicit user request via the ``council`` tool / ``force``).

BOUNDS (never wedge)
--------------------
* 1 critique round (two critics IN PARALLEL) + 1 synthesis + at most 1
  reconciliation — a hard ceiling on LLM calls.
* Per-critic wall-clock deadline; drop-and-quorum — the council proceeds on the
  critics that returned (need at least ``min_quorum``, default 1 of 2).
* If the panel cannot form (fewer than two distinct families available) or
  quorum fails, saddle FALLS BACK to the single ``audit_proposal`` and BUBBLES
  loudly — it never wedges on the council path.
* In autonomous mode a routine design skips the council (the user handed over
  the wheel); an explicit ``council`` call still convenes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from saddle.context import Context, default as _default_ctx
from saddle.llm.json_tools import call_json, strip_think
from saddle.llm.pool import tenant_gate
from saddle.voice import VOICE_CONTRACT

if TYPE_CHECKING:  # pragma: no cover
    from saddle.codemap import SurfaceManifest
    from saddle.dkb import DKB
    from saddle.llm.protocol import LLMCaller

_log = logging.getLogger("saddle.council")


# -- model families (for distinct-family dedup) ------------------------------

# A caller NAME maps to a model FAMILY: the council dedups by family so it never
# seats two variants of the same model (two DeepSeek tiers, say) as "two critics"
# — that would double one blind spot instead of covering two. Unknown names fall
# back to the name itself (a distinct family by default, never wrongly merged).
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("anthropic", "anthropic"),
    ("deepseek", "deepseek"),
    ("minimax", "minimax"),
    ("kimi", "moonshot"),
    ("glm", "zhipu"),
)


def model_family(caller_name: str) -> str:
    """The model family a caller name belongs to — the dedup key for council
    membership. ``deepseek_pro`` and ``deepseek_flash`` share ``deepseek``;
    ``claude_agent`` is ``anthropic``. An unrecognised name is its own family."""
    n = (caller_name or "").strip().lower()
    for prefix, family in _FAMILY_PREFIXES:
        if n.startswith(prefix):
            return family
    return n or "unknown"


# The synthetic caller entries build_callers adds — the fallback CHAIN and the
# parallel RACE — are compositions of real providers, not providers themselves,
# so they are never seated as a distinct critic ("default" is not a family).
_SYNTHETIC_CALLERS: frozenset[str] = frozenset({"default", "race"})


# -- membership selection (pure, testable) -----------------------------------

def select_members(
    available: set[str], cfg: dict, priority: list[str]
) -> tuple[str, list[str]]:
    """Choose the chair + up to two DISTINCT-family critics from the available
    callers.

    The chair (synthesis + the longevity lens) is the STRONGEST model:
    ``cfg['synthesis']`` if available, else the priority lead. Critics are drawn
    in preference order — ``cfg['members']`` first, then the active priority,
    then the rest — and DEDUPED BY FAMILY, so the two seats are always distinct
    families (or fewer, when the host has fewer families). Pure and deterministic
    for a given input, so the dedup is unit-testable without any LLM."""
    # Real providers only — the synthetic "default"/"race" chains are never a
    # critic seat (and never a distinct "family").
    avail = set(available) - _SYNTHETIC_CALLERS
    synth = str(cfg.get("synthesis") or "")
    chair = synth if synth in avail else next(
        (p for p in priority if p in avail), "")
    if not chair and avail:
        chair = sorted(avail)[0]

    ordered: list[str] = []
    for name in list(cfg.get("members") or []) + list(priority) + sorted(avail):
        if name in avail and name not in ordered:
            ordered.append(name)

    critics: list[str] = []
    seen_families: set[str] = set()
    for name in ordered:
        fam = model_family(name)
        if fam in seen_families:
            continue
        seen_families.add(fam)
        critics.append(name)
        if len(critics) == 2:
            break
    return chair, critics


# -- the two lenses + the synthesis chair ------------------------------------

_CRITIQUE_TAIL = (
    "\nYou MUST land a concrete verdict — never hedge with 'it is fine but'. "
    "Either name a SPECIFIC flaw (what is wrong, exactly where, and what it "
    "costs) or return an explicit PASS with the one-line reason it holds up. "
    "Respond with ONLY JSON: "
    '{"verdict": "pass|flaw", "concern": "the concrete flaw, or the PASS '
    'reason", "severity": "blocking|advisory", "recommendation": "the change '
    'you would make, or \\"\\" on a pass"}'
)

_LENS_LONGEVITY = (
    "You are the LONGEVITY & ARCHITECTURE critic on a two-critic design council. "
    "Your mandate — and ONLY yours — is whether this approach holds up over "
    "TIME. Judge: is the abstraction at the right level; will it extend to the "
    "obvious next cases without a rewrite; does it add coupling or a dependency "
    "that will be expensive to unwind; does it fit the system's existing grain "
    "or fight it; will a maintainer six months out understand and safely change "
    "it. Do NOT re-litigate whether it fixes the immediate bug — that is the "
    "other critic's mandate. Stay in your lane and go deep on durability."
    + _CRITIQUE_TAIL
)

_LENS_ROOTCAUSE = (
    "You are the ROOT-CAUSE & COMPLETENESS critic on a two-critic design "
    "council. Your mandate — and ONLY yours — is whether this approach fixes the "
    "REAL underlying problem and covers the WHOLE goal. Judge: does it reach the "
    "root cause or only a symptom; is it a band-aid (a tolerant default, a "
    "hardcoded backstop, a swallow-and-log) standing in for a structural fix; "
    "does it leave any part of the goal — any ask it folds, any site it must "
    "touch — unaddressed. Do NOT critique long-term architecture — that is the "
    "other critic's mandate. Stay in your lane and go deep on correctness and "
    "coverage." + _CRITIQUE_TAIL
)

_FORK_MARKER = "UNRESOLVED FORK:"

_SYS_SYNTHESIS = (
    "You are saddle, chairing a design council. You are given a GOAL, the "
    "agent's DRAFT approach, and TWO independent critiques (a longevity/"
    "architecture lens and a root-cause/completeness lens). Produce the single "
    "best design that reconciles both critiques: adopt what each got right, "
    "resolve where they overlap, and fix every flaw either one named. Write the "
    "reconciled design as PROSE / Markdown — concrete mechanisms, structure, and "
    "how each part satisfies the goal. Do NOT wrap it in JSON or a code fence.\n"
    "If — and ONLY if — the two critiques point in GENUINELY CONFLICTING "
    "directions that only the USER can adjudicate (a real tradeoff, not one you "
    "can resolve on the merits), do NOT pick one by fiat. Instead make the VERY "
    f"FIRST line exactly: '{_FORK_MARKER} <the choice, in one plain sentence>', "
    "then present both options and their tradeoffs. Otherwise emit no such line "
    "and just write the reconciled design. No preamble, no sign-off."
    + VOICE_CONTRACT
)

_SYS_RECONCILE = (
    "You are saddle, chairing a design council, on a RECONCILIATION pass. Your "
    "first synthesis flagged the two critiques as conflicting. Try HARDER to "
    "reconcile them on the merits now — most apparent conflicts dissolve into a "
    "design that honors both concerns. Produce that reconciled design as PROSE. "
    f"Emit the '{_FORK_MARKER} ...' first line ONLY if the conflict is a genuine "
    "tradeoff that the USER alone must decide; otherwise resolve it and write the "
    "single design. No preamble, no sign-off." + VOICE_CONTRACT
)


# -- results -----------------------------------------------------------------

@dataclass
class CritiqueResult:
    """One critic's verdict. ``ok`` is whether the critic RAN (returned within
    its deadline) — distinct from its ``verdict`` (pass / flaw). ``lens`` names
    its mandate; ``concern`` / ``recommendation`` carry the specific finding.
    A dropped critic is ``ok=False`` with ``error`` set."""

    member: str
    lens: str
    ok: bool = False
    verdict: str = ""
    concern: str = ""
    severity: str = ""
    recommendation: str = ""
    error: str = ""

    @property
    def blocking(self) -> bool:
        return self.ok and self.verdict == "flaw" and self.severity == "blocking"


@dataclass
class CouncilResult:
    """The council's outcome for one design.

    ``convened`` is whether the council actually ran (False = triaged out as
    trivial, or the panel could not form); ``fell_back`` is whether the caller
    must run the single ``audit_proposal`` instead (a tiny surface, an unformable
    panel, or a failed quorum) — always paired with a loud reason in ``dissent``.
    ``body`` is saddle's synthesised design (prose); ``index`` its small
    structured index; ``critiques`` the per-critic verdicts; ``members`` the
    caller names seated; ``forks`` the user-adjudicable conflicts the chair could
    not resolve; ``dissent`` a one-line note when the council could not deliver a
    settled design; ``audit_issues`` what the anti-pattern floor caught on the
    synthesis; ``design_id`` the settled design's id (empty unless settled)."""

    body: str = ""
    index: dict = field(default_factory=dict)
    dissent: str = ""
    forks: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)
    critiques: list[CritiqueResult] = field(default_factory=list)
    convened: bool = False
    fell_back: bool = False
    audit_issues: list[str] = field(default_factory=list)
    design_id: str = ""

    @property
    def settled(self) -> bool:
        return bool(self.design_id)


# -- triage ------------------------------------------------------------------

def surface_site_count(manifest: "SurfaceManifest") -> int:
    """Total declared surface sites across every kind — the blast-radius measure
    the triage floor compares against ``surface_min_sites``."""
    return sum(len(v) for v in manifest.to_dict().values())


# -- critique execution (parallel, per-critic deadline, drop-and-quorum) -----

def _critique_prompt(goal: str, draft: str) -> str:
    return (
        f"GOAL:\n{goal}\n\n"
        f"THE AGENT'S DRAFT APPROACH:\n{draft}\n\n"
        "Critique the draft through your lens only."
    )


async def _one_critique(
    caller: "LLMCaller", member: str, lens_system: str, lens_label: str,
    goal: str, draft: str, deadline: float,
) -> CritiqueResult:
    """Run one critic under its wall-clock deadline. A timeout / failure is
    CAUGHT here (not propagated) so drop-and-quorum can proceed on the survivors
    — the council must never wedge on one slow critic."""
    try:
        coro = call_json(
            caller, lens_system, _critique_prompt(goal, draft),
            label=f"council/{lens_label}",
        )
        payload = await (asyncio.wait_for(coro, timeout=deadline)
                         if deadline and deadline > 0 else coro)
    except Exception as exc:  # noqa: BLE001 — a dropped critic is quorum's job, not a wedge
        _log.warning("council critic %s (%s) dropped: %r", member, lens_label, exc)
        return CritiqueResult(member=member, lens=lens_label, ok=False, error=repr(exc))
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in ("pass", "flaw"):
        verdict = "flaw" if str(payload.get("concern") or "").strip() else "pass"
    return CritiqueResult(
        member=member, lens=lens_label, ok=True, verdict=verdict,
        concern=str(payload.get("concern") or "").strip(),
        severity=str(payload.get("severity") or "advisory").strip().lower(),
        recommendation=str(payload.get("recommendation") or "").strip(),
    )


async def _run_critiques(
    callers: dict, assignments: list[tuple[str, str, str]],
    goal: str, draft: str, ctx: Context, deadline: float,
) -> list[CritiqueResult]:
    """Fire both critics CONCURRENTLY under one tenant fairness slot, each on its
    own deadline. Returns every critic's result (dropped ones included) — quorum
    is judged by the caller."""
    async with tenant_gate(ctx):
        return list(await asyncio.gather(*[
            _one_critique(callers[name], name, system, label, goal, draft, deadline)
            for (name, system, label) in assignments
        ]))


# -- synthesis ---------------------------------------------------------------

def _synth_prompt(goal: str, draft: str, critiques: list[CritiqueResult]) -> str:
    blocks = []
    for c in critiques:
        tag = "blocking" if c.blocking else (c.severity or "advisory")
        blocks.append(
            f"CRITIQUE — {c.lens} ({c.member}), verdict {c.verdict} [{tag}]:\n"
            f"  concern: {c.concern or '(none)'}\n"
            f"  recommendation: {c.recommendation or '(none)'}"
        )
    body = "\n\n".join(blocks) or "(no critiques returned)"
    return (
        f"GOAL:\n{goal}\n\n"
        f"THE AGENT'S DRAFT APPROACH:\n{draft}\n\n"
        f"THE CRITIQUES:\n{body}\n\n"
        "Reconcile the critiques into the single best design."
    )


def _extract_forks(body: str) -> tuple[str, list[str]]:
    """Split any leading ``UNRESOLVED FORK: ...`` lines off the synthesis body.
    Returns ``(clean_body, forks)`` — the fork lines the chair could not resolve,
    for the user to adjudicate."""
    forks: list[str] = []
    kept: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if s.upper().startswith(_FORK_MARKER):
            q = s[len(_FORK_MARKER):].strip()
            if q:
                forks.append(q)
        else:
            kept.append(line)
    return "\n".join(kept).strip(), forks


async def _synthesize(
    caller: "LLMCaller", goal: str, draft: str,
    critiques: list[CritiqueResult], ctx: Context, *, reconcile: bool,
) -> tuple[str, list[str]]:
    system = _SYS_RECONCILE if reconcile else _SYS_SYNTHESIS
    label = "council/reconcile" if reconcile else "council/synthesis"
    async with tenant_gate(ctx):
        raw = await caller(system, _synth_prompt(goal, draft, critiques),
                           json_mode=False, label=label)
    return _extract_forks(strip_think(raw or "").strip())


# -- the entry point ---------------------------------------------------------

async def convene(
    goal: str,
    draft: str,
    ctx: Context | None = None,
    *,
    dkb: "DKB | None" = None,
    directives: list[str] | None = None,
    surface: "SurfaceManifest | None" = None,
    mods: list | None = None,
    callers: dict | None = None,
    force: bool = False,
    code_root: str | None = None,
) -> CouncilResult:
    """Convene the design council for ``goal`` / ``draft`` (the agent's approach).

    Triage first: an empty or tiny completeness surface (below
    ``surface_min_sites``) SKIPS the council — ``convened=False`` and the caller
    keeps the single ``audit_proposal``. ``force=True`` (an explicit council
    request) convenes regardless.

    On a non-trivial surface: seat the chair + two distinct-family critics, run
    both lenses in parallel under a per-critic deadline (drop-and-quorum), have
    saddle synthesise a reconciled design (≤1 reconciliation on an unresolved
    conflict), run the anti-pattern audit floor on the synthesis, and settle it
    (``approved_by="council"``) when it clears. An unresolved conflict is
    surfaced as a user fork (never settled); a failed panel/quorum falls back to
    the single audit and bubbles loudly (``fell_back=True``). Inject ``surface``
    / ``mods`` / ``callers`` / ``dkb`` to run offline (tests)."""
    goal = (goal or "").strip()
    draft = (draft or "").strip()
    if not goal or not draft:
        raise ValueError("council needs both a goal and a draft approach")
    ctx = ctx or _default_ctx()

    from saddle.llm.policy import active_priority, council_settings

    cfg = council_settings(ctx)
    priority = active_priority(ctx)

    # 1. Triage by blast radius — cheaply, before any LLM panel is built.
    manifest = surface
    if manifest is None:
        from saddle import design as _design

        rp = _design._resolve_code_root(code_root)
        if mods is None:
            mods = await _design._parse_code(rp)
        if not mods:
            # No code to weigh a blast radius against -> trivial; keep the single
            # audit_proposal. No callers built, no LLM consulted.
            return CouncilResult(convened=False,
                                 dissent="no code to size the change against")
        if callers is None:
            from saddle.llm.callers import build_callers

            callers = build_callers(ctx)
        diag = {"problem": "", "approach": draft}
        manifest, _fanout = await _design._surface(
            callers.get("default"), goal, diag, mods, rp
        )
    min_sites = int(cfg.get("surface_min_sites", 2) or 2)
    if not force and surface_site_count(manifest) < min_sites:
        return CouncilResult(convened=False,
                             dissent="surface too small to convene a council")

    # 2. Seat the panel: chair (strongest) + two DISTINCT-family critics.
    if callers is None:
        from saddle.llm.callers import build_callers

        callers = build_callers(ctx)
    chair, critics = select_members(set(callers), cfg, priority)
    if len(critics) < 2 or not chair:
        # Cannot form a two-family panel here -> fall back loudly, never wedge.
        return CouncilResult(
            convened=False, fell_back=True, members=critics,
            dissent="could not seat two distinct-family critics; "
                    "falling back to the single design audit",
        )
    # The longevity lens goes to the STRONGEST critic (the chair, when it is a
    # critic); the root-cause lens to the other.
    if chair in critics:
        longevity, rootcause = chair, next(c for c in critics if c != chair)
    else:
        longevity, rootcause = critics[0], critics[1]
    seated = [longevity, rootcause]

    # 3. One critique round — both lenses in parallel, per-critic deadline.
    deadline = float(cfg.get("critic_deadline_s", 90.0) or 0.0)
    assignments = [
        (longevity, _LENS_LONGEVITY, "longevity"),
        (rootcause, _LENS_ROOTCAUSE, "root-cause"),
    ]
    critiques = await _run_critiques(callers, assignments, goal, draft, ctx, deadline)
    quorum = int(cfg.get("min_quorum", 1) or 1)
    got = [c for c in critiques if c.ok]
    if len(got) < quorum:
        return CouncilResult(
            convened=True, fell_back=True, members=seated, critiques=critiques,
            dissent=f"only {len(got)} of 2 critics returned (need {quorum}); "
                    "falling back to the single design audit",
        )

    # 4. Synthesis (chair, prose) + ≤1 reconciliation on an unresolved conflict.
    body, forks = await _synthesize(callers[chair], goal, draft, got, ctx,
                                    reconcile=False)
    if forks:
        # The chair flagged a conflict -> exactly ONE reconciliation attempt.
        body, forks = await _synthesize(callers[chair], goal, draft, got, ctx,
                                        reconcile=True)

    result = CouncilResult(
        body=body, forks=forks, members=seated, critiques=critiques, convened=True,
    )
    if not body:
        result.fell_back = True
        result.dissent = "the chair produced no synthesis; falling back"
        return result

    # 5. The audit floor still runs — consensus cannot launder a band-aid.
    from saddle.design import audit_proposal

    verdict = await audit_proposal(goal, body, ctx, dkb=dkb, directives=directives)
    if verdict.has_issues:
        result.audit_issues = list(verdict.issues)
        result.dissent = "the anti-pattern audit floor flagged the synthesis"
        return result
    if forks:
        # A genuine, user-adjudicable conflict remains after reconciliation —
        # surface it; do NOT settle by fiat.
        result.dissent = "a conflict remains for the user to decide"
        return result

    # 6. Settle the reconciled design (approved_by="council").
    from saddle.design import settle_approach

    design = await settle_approach(
        goal, body, ctx, approved_by="council", dkb=dkb, code_root=code_root,
    )
    result.design_id = design.id
    result.index = {
        "summary": design.summary,
        "satisfies": list(design.satisfies),
        "avoids": list(design.avoids),
        "heeds": list(design.heeds),
    }
    return result


# -- rendering (for the MCP tool + design_propose) ---------------------------

def render_result(result: CouncilResult, ctx_key: str = "") -> str:
    """A plain-language readout of a council outcome for the MCP channel."""
    who = f" for {ctx_key}" if ctx_key else ""
    if result.settled:
        seats = ", ".join(result.members)
        head = (
            f"SETTLED by council{who} — recorded as agreed design "
            f"{result.design_id}. The council (saddle chairing {seats}) "
            "reconciled the critiques and the anti-pattern floor passed. The "
            "gate is open; saddle will check future code against this design."
        )
        return f"{head}\n\nTHE RECONCILED DESIGN:\n{result.body}"
    lines: list[str] = []
    if result.audit_issues:
        lines.append(
            "NOT SETTLED — the council reconciled the critiques, but the "
            "anti-pattern audit still found problems to resolve first:"
        )
        lines += [f"- {i}" for i in result.audit_issues]
    if result.forks:
        lines.append(
            "NEEDS YOUR DECISION — the council could not reconcile a genuine "
            "tradeoff. Please choose:"
        )
        lines += [f"- {f}" for f in result.forks]
    if not lines:
        lines.append(f"council did not settle a design: {result.dissent}")
    if result.body:
        lines.append("\nTHE COUNCIL'S SYNTHESIS SO FAR:\n" + result.body)
    return "\n".join(lines)
