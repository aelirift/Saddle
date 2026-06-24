"""Layer 2 — best-practice design over the Design Knowledge Base.

Layer 1 turns a prompt into discrete asks. Layer 2 turns a (folded) goal into a
proper design — and crucially, it thinks ABOVE the surface request. Left to its
own devices an LLM produces shallow, symptom-level designs: it retries a call
that hit a token limit, it multiplies every damage number to hit a target,
it spams one ability with no rotation. None of those examine whether the FRAME
is even right. The harness supplies that missing altitude as an explicit,
staged pipeline rather than hoping a single prompt does it:

  1. retrieve   — hybrid scope-laddered pull from the DKB on the raw goal.
  2. diagnose   — root cause vs symptom, reframe, alternatives, second-order
                  effects. The "is this even the right kit?" stage.
  3. surface    — ground the design in the target project's REAL code: declare the
                  completeness surface (which values/identities/boundaries the
                  change touches, named from the actual symbol menu) and derive
                  the existing fan-out of those things, so the design must address
                  every site instead of stopping at the symptom — and the manifest
                  persists as the bridge to the Layer-3 gate. No code ⇒ skipped.
  4. retrieve   — again, now sharpened by the diagnosis (better query ⇒ better
                  knowledge), unioned with the first pull.
  5. design     — synthesize the actual design as PROSE, covering the whole goal,
                  honoring the binding directives, drawing on best practices,
                  steering clear of anti-patterns, applying lessons.
  6. audit      — enforce it: directives violated, anti-patterns present,
                  hand-waving, band-aids, uncovered asks. Bounded revise loop.
  7. index      — extract the settled design's small structured fields (summary,
                  satisfies / avoids / heeds) for storage and display.
  8. harvest    — any flaw the audit caught becomes a durable lesson written back
                  into the DKB (source=audit), so the system gets smarter by use.

The design body is generated and revised as free text, NOT as a field inside a
JSON object: a multi-kilobyte design hand-escaped into a JSON string is the
harness's most fragile contract (one missed escape breaks the whole parse), so
the heavy artifact rides as prose and only the small index is JSON. Every LLM
step still returns ONE artifact and runs inside the tenant's fairness gate; sync
DKB / embedding work is offloaded so the event loop stays free.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from saddle.codemap import Finding, SurfaceManifest, refs
from saddle.context import Context, default as _default_ctx
from saddle.dkb import DKB, get_dkb
from saddle.llm import policy
from saddle.llm.json_tools import call_json
from saddle.llm.pool import tenant_gate
from saddle.models import (
    ANTI_PATTERN,
    AUDIT,
    BEST_PRACTICE,
    DESIGN_FINAL,
    DESIGN_FLAGGED,
    LESSON,
    PRINCIPLE,
    Design,
    Knowledge,
)

if TYPE_CHECKING:  # pragma: no cover
    from saddle.llm.protocol import LLMCaller

_log = logging.getLogger("saddle.design")

# Human-readable section headers for the knowledge block, in priority order.
_KIND_HEADERS: list[tuple[str, str]] = [
    (BEST_PRACTICE, "BEST PRACTICES (follow)"),
    (ANTI_PATTERN, "ANTI-PATTERNS (avoid)"),
    (LESSON, "LESSONS (apply)"),
    (PRINCIPLE, "PRINCIPLES (uphold)"),
]

_SYS_DIAGNOSE = (
    "You are the diagnosis stage of a design harness — you think ABOVE the "
    "surface request. You are given a GOAL and relevant DESIGN KNOWLEDGE. Do "
    "NOT design the solution yet; diagnose it:\n"
    "- problem: the real underlying problem / root cause behind the goal. "
    "Separate symptom from cause. If the goal is itself a symptom-level framing "
    "(e.g. 'increase the damage multipliers', 'retry the failing call'), name "
    "the ACTUAL problem (e.g. 'the kit has no rotation', 'the input contract is "
    "oversized') instead of taking the surface ask at face value.\n"
    "- approach: the structural direction you would take, including any REFRAME "
    "of the problem and the main alternatives you considered and why you "
    "rejected them. Prefer structural fixes over patches.\n"
    "- risks: second-order effects and failure modes a naive design would miss.\n"
    "Use the knowledge: avoid the listed anti-patterns, uphold the principles.\n"
    "Respond with ONLY JSON: "
    '{"problem": "...", "approach": "...", "risks": ["..."]}'
)

_SYS_SURFACE = (
    "You are the surface-mapping stage of a design harness. Before the design is "
    "written, declare its COMPLETENESS SURFACE: the specific code-level things "
    "this change touches that MUST stay consistent everywhere they appear, so the "
    "change cannot land half-wired (the classic failure: a new modifier that the "
    "counter, the description, the cooldown sweep, and the engine all ignore).\n"
    "You are given the GOAL, the DIAGNOSIS, and a SYMBOL MENU drawn from the "
    "ACTUAL target codebase — real field names, function names, call sites, and "
    "string-set declarations, ranked by how often they appear. GROUND EVERY NAME "
    "you emit in that menu; never invent one. If the code says `cooldown_s`, write "
    "`cooldown_s`, not `cooldown`.\n"
    "Declare eight kinds (ANY may be empty — only include a thing the GOAL "
    "actually touches):\n"
    "- values: a number whose effective value is base (+) modifiers and must "
    "reach every consumer through a resolver. Fields: name; field (the menu "
    "field); accessor (the resolver function(s) from the menu that return the "
    "effective value); producers (functions that legitimately build the base — a "
    "def builder, a deserializer).\n"
    "- identities: a closed set of string members (an enum-by-string). Fields: "
    "name; canonical (the member list — reuse a menu collection if one matches); "
    "source_symbol (the collection name that declares it); carriers (the field/"
    "var names that HOLD a value of this identity). If a carrier name is generic "
    "and reused across unrelated namespaces (e.g. `kind`, `type`, `id`), QUALIFY "
    "it as `object.field` (e.g. `effect.kind`) so a literal from a different "
    "namespace isn't mis-flagged — a bare `kind` matches everywhere.\n"
    "- boundaries: a value written authoritatively on the server that must be "
    "mirrored to the client. Fields: name; key (the field); replication_func (the "
    "function that packs it into the snapshot).\n"
    "- references: a name that must be REGISTERED in non-code substrates too, not "
    "only defined in code — a config file, the docs, a DB schema (the cooldown "
    "that lived in the engine but never in the skill DESCRIPTION). Fields: name; "
    "key (the token that must appear); substrates (globs relative to the project "
    "root where it must appear, e.g. `config/*.json`, `docs/**/*.md`).\n"
    "- persistence: a value that must survive a save/load round-trip. Fields: "
    "name; key (the field); save_func (the function that writes the save); "
    "load_func (the function that reads it back). It must be referenced in BOTH "
    "or it silently resets / loads garbage.\n"
    "- lifecycle: a DECLARED knob that must be READ to mean anything — an exported "
    "setting, a constant, a signal, a script-scope field. Fields: name; symbol "
    "(the declared identifier from the menu). A declaration nothing reads is a DEAD "
    "knob: it looks adjustable but changes nothing. Declare this for any setting "
    "the GOAL adds or relies on, so the gate proves some code actually consumes "
    "it.\n"
    "- authority: a function that MUTATES server-authoritative state and must "
    "check an authority guard before it writes, or a client can invoke it and "
    "desync / cheat. Fields: name; guard (the authority-check function(s) from the "
    "menu, e.g. `is_server`, `is_multiplayer_authority`); mutators (the function "
    "names that perform the authoritative write). The WRITE side of the trust "
    "boundary — declare it for any state a client must not set directly.\n"
    "- bindings: an INPUT keymap that must be unambiguous and fully reachable — "
    "every physical input does exactly ONE intended thing and every declared "
    "action is invocable, or a key opens a panel when the player meant to act (the "
    "number-row-fires-a-menu bug). Fields: name; keymap (the serialized keymap "
    "file relative to root, e.g. `project.godot`); families (a map of intent-family "
    "-> the action-name prefixes in that family, e.g. `{\"ability\": [\"ability_\"], "
    "\"panel\": [\"open_\"]}`); compatible (family pairs allowed to share one input, "
    "e.g. `[[\"ability\", \"card\"]]`; a family paired with itself permits same-"
    "family co-binds); programmatic (actions fired from code that need no key, so "
    "they aren't flagged dead). A trigger firing two incompatible families, or an "
    "action with no trigger, is the gap.\n"
    "Respond with ONLY JSON: "
    '{"values": [{"name": "...", "field": "...", "accessor": ["..."], '
    '"producers": ["..."]}], "identities": [{"name": "...", "canonical": ["..."], '
    '"source_symbol": "...", "carriers": ["..."]}], "boundaries": [{"name": "...", '
    '"key": "...", "replication_func": "..."}], "references": [{"name": "...", '
    '"key": "...", "substrates": ["..."]}], "persistence": [{"name": "...", '
    '"key": "...", "save_func": "...", "load_func": "..."}], "lifecycle": '
    '[{"name": "...", "symbol": "..."}], "authority": [{"name": "...", '
    '"guard": ["..."], "mutators": ["..."]}], "bindings": [{"name": "...", '
    '"keymap": "...", "families": {"fam": ["prefix_"]}, "compatible": '
    '[["famA", "famB"]], "programmatic": ["..."]}]}'
)

_SYS_BODY = (
    "You are the design stage of a design harness. Produce a best-practice "
    "design for the GOAL, guided by the DIAGNOSIS and the DESIGN KNOWLEDGE. "
    "Hard requirements: no hand-waving (every decision concrete, backed by a "
    "real mechanism); no band-aids (root-cause structure, never a symptom "
    "patch); think long-term; prefer automation and abstraction; NO "
    "hard-coding. Honor every binding DIRECTIVE. Cover the ENTIRE goal — if it "
    "folds several asks, address all of them in one coherent design.\n"
    "Write the design itself: concrete mechanisms, structure, data and control "
    "flow, and how each part satisfies the goal — detailed and actionable, not "
    "a sketch. Respond with the design as prose / Markdown. Do NOT wrap it in "
    "JSON and do NOT enclose the whole thing in a code fence. No preamble and "
    "no sign-off — emit only the design."
)

_SYS_INDEX = (
    "You are the indexing stage of a design harness. You are given a GOAL and a "
    "finished DESIGN. Read the design and extract a compact, faithful index of "
    "what it already says — do not invent or add anything. Short phrases only, "
    "no prose paragraphs:\n"
    "- summary: one or two sentences naming the design.\n"
    "- satisfies: the binding directives / best practices the design honors.\n"
    "- avoids: the anti-patterns it steers clear of.\n"
    "- heeds: the lessons it applies.\n"
    "Respond with ONLY JSON: "
    '{"summary": "...", "satisfies": ["..."], "avoids": ["..."], "heeds": ["..."]}'
)

_SYS_AUDIT = (
    "You are the audit stage of a design harness. You are given a GOAL, its "
    "current DESIGN, the binding DIRECTIVES, and the ANTI-PATTERNS to avoid. "
    "Audit hard and specifically. Flag any of:\n"
    "- a binding directive violated or ignored;\n"
    "- an anti-pattern present in the design;\n"
    "- hand-waving — a vague claim with no concrete mechanism behind it;\n"
    "- a band-aid — a symptom patch rather than a root-cause structure;\n"
    "- any part of the GOAL left uncovered.\n"
    "Each issue must name what is wrong and where. Do not invent issues; if the "
    "design is sound, return none.\n"
    "Respond with ONLY JSON: "
    '{"ok": true/false, "issues": ["..."]}. ok=true only when issues is empty.'
)

_SYS_HARVEST = (
    "You are the lesson-harvest stage of a design harness. A design just went "
    "through audit and the FLAWS below were caught and corrected. Extract any "
    "GENERALIZABLE design lesson or anti-pattern worth remembering for FUTURE "
    "designs — reusable wisdom, not a restatement of this one case. Emit an "
    "entry ONLY if it is genuinely general and not obvious; if nothing clears "
    "that bar, return an empty list.\n"
    "For each: kind ('lesson' or 'anti_pattern'), title (a short phrase), body "
    "(1-2 sentences of durable insight), tags (a few keywords).\n"
    "Respond with ONLY JSON: "
    '{"entries": [{"kind": "...", "title": "...", "body": "...", "tags": ["..."]}]}'
)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _format_knowledge(items: list[Knowledge]) -> str:
    """Render retrieved knowledge as a sectioned, kind-grouped block."""
    if not items:
        return "(no relevant knowledge yet)"
    by_kind: dict[str, list[Knowledge]] = {}
    for k in items:
        by_kind.setdefault(k.kind, []).append(k)
    out: list[str] = []
    for kind, header in _KIND_HEADERS:
        group = by_kind.get(kind)
        if not group:
            continue
        out.append(f"{header}:")
        for k in group:
            out.append(f"- {k.title}: {k.body}")
        out.append("")
    return "\n".join(out).strip() or "(no relevant knowledge yet)"


def _format_directives(directives: list[str]) -> str:
    if not directives:
        return "(none)"
    return "\n".join(f"{i + 1}. {d}" for i, d in enumerate(directives))


def _merge_knowledge(
    *lists: list[tuple[Knowledge, float]],
) -> list[Knowledge]:
    """Union retrieved knowledge across passes, dedup by id, keep first order."""
    seen: set[str] = set()
    out: list[Knowledge] = []
    for lst in lists:
        for k, _ in lst:
            if k.id not in seen:
                seen.add(k.id)
                out.append(k)
    return out


async def _search(dkb: DKB, *args, **kwargs):
    """Run the (sync) hybrid retrieval off the event loop."""
    return await asyncio.to_thread(dkb.search_knowledge, *args, **kwargs)


def _index_fields(payload: dict) -> dict:
    return {
        "summary": str(payload.get("summary", "")).strip(),
        "satisfies": [str(x).strip() for x in payload.get("satisfies", []) if str(x).strip()],
        "avoids": [str(x).strip() for x in payload.get("avoids", []) if str(x).strip()],
        "heeds": [str(x).strip() for x in payload.get("heeds", []) if str(x).strip()],
    }


def _diagnose_prompt(goal: str, kb: str) -> str:
    return f"GOAL:\n{goal}\n\nDESIGN KNOWLEDGE:\n{kb}"


def _surface_prompt(goal: str, diag: dict, menu: dict) -> str:
    return (
        f"GOAL:\n{goal}\n\n"
        "DIAGNOSIS:\n"
        f"- problem (root cause): {diag.get('problem', '')}\n"
        f"- approach: {diag.get('approach', '')}\n\n"
        "SYMBOL MENU (real names from the target codebase, ranked by use — ground "
        f"every name you emit in this menu):\n{json.dumps(menu, indent=2)}\n\n"
        "Declare the completeness surface this design touches."
    )


def _resolve_code_root(code_root: str | Path | None) -> Path | None:
    """The tree to ground the surface stage in: an explicit arg, else
    ``$SADDLE_CODE_ROOT``, else nothing. Deliberately NOT cwd-derived here — the
    harness serves many projects and never runs inside the target's working dir;
    the cwd fallback (:func:`saddle.context.code_root`) belongs at the human CLI
    edge, which passes the resolved root in. No root ⇒ the surface stage no-ops
    and the design pipeline runs unchanged (a brand-new project designs fine)."""
    if code_root is not None:
        return Path(code_root).expanduser()
    env = os.environ.get("SADDLE_CODE_ROOT")
    if env and env.strip():
        return Path(env).expanduser().resolve()
    return None


async def _parse_code(root: Path | None) -> list:
    """Parse the target project's code off the event loop. Best-effort: a missing
    or unreadable tree yields an empty list (then the surface stage no-ops) — the
    target project's filesystem is not saddle's contract to enforce."""
    if root is None:
        return []
    try:
        return await asyncio.to_thread(refs.parse_project, root)
    except Exception:  # noqa: BLE001 — code grounding is best-effort
        _log.warning("surface: could not parse code at %s; skipping", root, exc_info=True)
        return []


async def _surface(
    caller: "LLMCaller", goal: str, diag: dict, mods: list, root: Path | None = None
) -> tuple[SurfaceManifest, str]:
    """Declare the design's completeness surface (grounded in the project's real
    symbols) and return it with its current code fan-out. No code ⇒ empty manifest
    and no LLM call. ``root`` lets the fan-out also scan the file substrates
    (references) so the body sees what's missing from config/docs/schema, not only
    code. The LLM contract here is saddle's own, so a wholesale-malformed reply
    surfaces loudly (via call_json); an individual spec ROW the LLM left
    incomplete is dropped with a loud warning (SurfaceManifest.from_dict) so one
    missing field never aborts the design — never silently swallowed."""
    if not mods:
        return SurfaceManifest(), ""
    menu = await asyncio.to_thread(lambda: refs.symbols(mods).top())
    payload = await call_json(
        caller, _SYS_SURFACE, _surface_prompt(goal, diag, menu), label="design/surface"
    )
    manifest = SurfaceManifest.from_dict(payload)
    if manifest.is_empty():
        return manifest, ""
    fanout = await asyncio.to_thread(manifest.format, mods, root)
    return manifest, fanout


def _body_prompt(
    goal: str,
    diag: dict,
    kb: str,
    directives: list[str],
    prior_body: str | None,
    issues: list[str] | None,
    surface: str | None = None,
) -> str:
    risks = "; ".join(str(r) for r in diag.get("risks", []) if str(r).strip())
    parts = [
        f"GOAL:\n{goal}",
        "DIAGNOSIS:\n"
        f"- problem (root cause): {diag.get('problem', '')}\n"
        f"- approach: {diag.get('approach', '')}\n"
        f"- risks: {risks or '(none named)'}",
        f"BINDING DIRECTIVES:\n{_format_directives(directives)}",
        f"DESIGN KNOWLEDGE:\n{kb}",
    ]
    if surface:
        parts.append(
            "COMPLETENESS SURFACE — every existing code site that touches a value, "
            "identity, or boundary this design changes, derived from the actual "
            "codebase. Your design MUST state how EACH site stays correct after the "
            "change, and name any NEW site that must be added. Leave none silently "
            f"un-addressed:\n\n{surface}"
        )
    if prior_body and issues:
        bullet = "\n".join(f"- {i}" for i in issues)
        parts.append(
            "REVISE the design below to resolve these AUDIT ISSUES while still "
            "covering the whole goal. Return the FULL revised design, not a "
            f"diff.\n\nAUDIT ISSUES:\n{bullet}\n\nCURRENT DESIGN:\n{prior_body}"
        )
    return "\n\n".join(parts)


def _index_prompt(goal: str, body: str) -> str:
    return f"GOAL:\n{goal}\n\nDESIGN:\n{body}\n\nExtract the structured index."


def _audit_prompt(goal: str, body: str, directives: list[str], anti: str) -> str:
    return (
        f"GOAL:\n{goal}\n\n"
        f"DESIGN:\n{body}\n\n"
        f"BINDING DIRECTIVES:\n{_format_directives(directives)}\n\n"
        f"ANTI-PATTERNS TO AVOID:\n{anti}\n\n"
        "Return the issues found (empty if the design is sound)."
    )


def _harvest_prompt(goal: str, issues: list[str]) -> str:
    bullet = "\n".join(f"- {i}" for i in issues)
    return (
        f"GOAL:\n{goal}\n\nFLAWS CAUGHT AND CORRECTED IN AUDIT:\n{bullet}\n\n"
        "Extract any generalizable lessons or anti-patterns worth remembering."
    )


async def _harvest(
    caller: "LLMCaller", ctx: Context, dkb: DKB, goal: str, issues: list[str]
) -> int:
    """File generalizable lessons from caught flaws into the DKB (source=audit)."""
    payload = await call_json(
        caller, _SYS_HARVEST, _harvest_prompt(goal, issues), label="design/harvest"
    )
    raw = payload.get("entries")
    if not isinstance(raw, list) or not raw:
        return 0
    existing_titles = {
        _norm(k.title)
        for k in await asyncio.to_thread(dkb.list_knowledge, ctx, limit=1000)
    }
    added = 0
    for e in raw:
        if not isinstance(e, dict):
            continue
        title = str(e.get("title", "")).strip()
        body = str(e.get("body", "")).strip()
        if not title or not body or _norm(title) in existing_titles:
            continue
        kind = str(e.get("kind", "")).strip().lower()
        if kind not in (LESSON, ANTI_PATTERN):
            kind = LESSON
        kn = Knowledge(
            kind=kind, title=title, body=body,
            tags=[str(t).strip() for t in e.get("tags", []) if str(t).strip()],
            scope_tenant=ctx.tenant, scope_project=ctx.project, source=AUDIT,
        )
        await asyncio.to_thread(dkb.add_knowledge, kn)
        existing_titles.add(_norm(title))
        added += 1
    if added:
        _log.info("harvested %d lesson(s) into the DKB for %s", added, ctx.key)
    return added


async def design_for(
    goal: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
    dkb: DKB | None = None,
    directives: list[str] | None = None,
    persist: bool = True,
    harvest: bool = True,
    surface: bool = True,
    code_root: str | Path | None = None,
    max_audits: int = 2,
    retrieve_k: int = 8,
    intake_id: str = "",
) -> Design:
    """Produce a best-practice :class:`Design` for ``goal`` under ``ctx``.

    Runs the full diagnose → surface → retrieve → design → audit → harvest
    pipeline inside the tenant fairness gate. Persists the design (and any
    harvested lessons) unless told otherwise. Pass ``caller`` to bypass provider
    resolution (tests).

    The SURFACE stage grounds the design in the target project's real code: it
    declares the completeness surface (which values/identities/boundaries the
    change touches) and feeds the existing fan-out of those things into the body
    prompt, so the design must address every site rather than stop at the symptom.
    It runs only when a code root is resolvable (``code_root`` arg or
    ``$SADDLE_CODE_ROOT``); otherwise it no-ops and the pipeline is unchanged.
    """
    goal = (goal or "").strip()
    if not goal:
        raise ValueError("cannot design for an empty goal")
    ctx = ctx or _default_ctx()
    dkb = dkb or get_dkb()
    if directives is None:
        directives = policy.directives(ctx)
    if caller is None:
        from saddle.llm.callers import build_callers
        caller = build_callers(ctx)["default"]

    audits_run = 0
    clean = False
    all_issues: list[str] = []
    code_root_path = _resolve_code_root(code_root) if surface else None
    async with tenant_gate(ctx):
        # Parse the target code off-thread, overlapped with the LLM stages below —
        # it depends on nothing they produce, so its latency hides behind theirs.
        code_task = asyncio.create_task(_parse_code(code_root_path))

        # 1 + 2: broad retrieve, then diagnose above the surface ask.
        pass1 = await _search(dkb, ctx, goal, k=retrieve_k)
        diag = await call_json(
            caller, _SYS_DIAGNOSE,
            _diagnose_prompt(goal, _format_knowledge([k for k, _ in pass1])),
            label="design/diagnose",
        )
        problem = str(diag.get("problem", "")).strip()
        approach = str(diag.get("approach", "")).strip()
        risks = [str(r).strip() for r in diag.get("risks", []) if str(r).strip()]

        # 3: surface — declare the completeness surface, grounded in real symbols,
        # and derive the existing fan-out the body must address (empty if no code).
        mods = await code_task
        manifest, fanout = await _surface(caller, goal, diag, mods, code_root_path)

        # 4: sharpen retrieval with the diagnosis; pull anti-patterns for audit.
        sharp_q = f"{goal}\n{problem}\n{approach}"
        pass2 = await _search(dkb, ctx, sharp_q, k=retrieve_k)
        knowledge = _merge_knowledge(pass1, pass2)
        kb_block = _format_knowledge(knowledge)
        anti_hits = await _search(dkb, ctx, sharp_q, k=6, kinds=[ANTI_PATTERN])
        anti_block = _format_knowledge([k for k, _ in anti_hits])

        # 5: synthesize the design as PROSE (text, not a JSON field). A long
        # design hand-escaped into a JSON string is the harness's most fragile
        # contract; returning it as text removes that whole failure mode.
        body = await caller(
            _SYS_BODY,
            _body_prompt(goal, diag, kb_block, directives, None, None, fanout),
            json_mode=False, label="design/body",
        )

        # 6: audit + bounded revise loop — operates on the prose body.
        for _ in range(max(0, max_audits)):
            verdict = await call_json(
                caller, _SYS_AUDIT,
                _audit_prompt(goal, body, directives, anti_block),
                label="design/audit",
            )
            audits_run += 1
            issues = [str(i).strip() for i in verdict.get("issues", []) if str(i).strip()]
            if bool(verdict.get("ok")) and not issues:
                clean = True
                break
            if not issues:
                break  # flagged not-ok but named nothing actionable — stop
            all_issues.extend(issues)
            body = await caller(
                _SYS_BODY,
                _body_prompt(goal, diag, kb_block, directives, body, issues, fanout),
                json_mode=False, label="design/revise",
            )

        # 7: index the settled design into small, robust structured fields.
        index = _index_fields(
            await call_json(
                caller, _SYS_INDEX, _index_prompt(goal, body), label="design/index"
            )
        )

        # 8: harvest the caught flaws into the DKB.
        harvested = 0
        if harvest and all_issues:
            harvested = await _harvest(caller, ctx, dkb, goal, all_issues)

    design = Design(
        ask=goal,
        summary=index["summary"],
        problem=problem,
        approach=approach,
        body=body,
        satisfies=index["satisfies"],
        avoids=index["avoids"],
        heeds=index["heeds"],
        status=DESIGN_FINAL if clean else DESIGN_FLAGGED,
        intake_id=intake_id,
        meta={
            "risks": risks,
            "retrieved": [k.id for k in knowledge],
            "audits_run": audits_run,
            "audit_clean": clean,
            "issues": all_issues,
            "harvested": harvested,
            # The completeness surface the design committed to — the durable
            # bridge to the Layer-3 gate. Plain JSON; SurfaceManifest.from_dict
            # rebuilds the typed specs to run against real code after implementation.
            "surface": manifest.to_dict(),
        },
    )
    if persist:
        await asyncio.to_thread(dkb.add_design, ctx, design)
        _log.info(
            "design %s [%s] for %s: %d audits (clean=%s), %d harvested",
            design.id, design.status, ctx.key, audits_run, clean, harvested,
        )
    return design


def intent_drift(
    design: Design,
    *,
    root: str | Path | None = None,
    mods: list | None = None,
) -> list[Finding]:
    """Re-run a design's DECLARED surface against the code as it stands now and
    return where the implementation has DRIFTED from the intent the design
    committed to.

    This is the design's OWN gate, handed forward. Layer 2 enumerated the
    completeness surface at design time (the :class:`SurfaceManifest` persisted in
    ``design.meta['surface']``); this runs those exact specs — nobody re-types
    them — against a FRESH parse of the target tree. The surface is intent; the
    code is truth; the findings are the difference. Because the parse happens
    here, every edit made up to this call is seen: there is no cached map that can
    bless code it never read.

    ``root`` resolves like the surface stage (explicit arg, else
    ``$SADDLE_CODE_ROOT``); pass ``mods`` to gate an already-parsed tree instead.
    With no declared surface, or no code to compare against, there is nothing that
    can drift — returns empty. A non-empty result is always a real divergence.
    """
    manifest = SurfaceManifest.from_dict(design.meta.get("surface"))
    if manifest.is_empty():
        return []
    rp = _resolve_code_root(root)
    if mods is None:
        if rp is None:
            return []  # no code to compare the intent against — not a drift
        mods = refs.parse_project(rp)
    return manifest.gate(mods, root=rp)


def format_design(design: Design) -> str:
    """Human-readable rendering of a design for the CLI."""
    lines: list[str] = []
    if design.id:
        lines.append(f"design {design.id}  [{design.status}]  {design.tenant}/{design.project}")
    lines.append(f"ask: {design.ask}")
    if design.summary:
        lines.append(f"summary: {design.summary}")
    if design.problem:
        lines.append(f"\nroot cause:\n  {design.problem}")
    if design.approach:
        lines.append(f"\napproach:\n  {design.approach}")
    if design.body:
        lines.append(f"\ndesign:\n{design.body}")
    for label, vals in (("satisfies", design.satisfies), ("avoids", design.avoids),
                        ("heeds", design.heeds)):
        if vals:
            lines.append(f"\n{label}:")
            lines.extend(f"  - {v}" for v in vals)
    risks = design.meta.get("risks") or []
    if risks:
        lines.append("\nrisks (second-order):")
        lines.extend(f"  - {r}" for r in risks)
    if design.status == DESIGN_FLAGGED:
        lines.append("\n(note: audit did not converge clean — review the issues)")
    return "\n".join(lines)
