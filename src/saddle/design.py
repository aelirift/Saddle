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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from saddle.codemap import Finding, SurfaceManifest, refs
from saddle.context import Context, default as _default_ctx
from saddle.dkb import DKB, get_dkb
from saddle.llm import policy
from saddle.llm.json_tools import call_json, strip_think
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
    "- congruences: a server->client replication MIRROR whose state mutators must "
    "each pass an authority gate — the WHOLE congruence bug class, derived per-"
    "function across files instead of named mutator-by-mutator. A mirror service is "
    "any module defining `mirror_apply` (the client-side snapshot applier); the "
    "fields it writes are the replicated state; any public function that writes one "
    "of them is a mutator and must call a `guard` before it writes OR be unreachable "
    "from non-server code, else a HUD mutates the local mirror and the next snapshot "
    "reverts it. Fields: name; mirror_apply (the applier fn, e.g. "
    "`apply_replication_snapshot`); guard (the authority-check function(s)); exempt "
    "(functions the service legitimately leaves ungated). The mutator SET and its "
    "call sites are read from the AST — you only name the engine tokens. Declare it "
    "for any service that ships a client-side copy of server-owned state.\n"
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
    '"guard": ["..."], "mutators": ["..."]}], "congruences": [{"name": "...", '
    '"mirror_apply": "...", "guard": ["..."], "exempt": ["..."]}], '
    '"bindings": [{"name": "...", '
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
    "APPLICABILITY FIRST. A binding directive binds only the kind of task it was "
    "written for. Before flagging a directive as violated, check it actually "
    "GOVERNS this work: a directive about producing a specific project artifact (a "
    "required output format, a target-project constraint) does NOT bind unrelated "
    "work, and SELF-DIRECTED work on the assistant/harness ITSELF — changing its "
    "own code, gates, prompts, configuration, or behavior — is NOT bound by "
    "directives meant for the projects it builds or operates on. Do not flag an "
    "inapplicable directive, and do not raise a coverage gap against a goal a "
    "directive was never about. Enforce the directives that DO apply, hard.\n"
    "Each issue must name what is wrong and where. Do not invent issues; if the "
    "design is sound, flag nothing.\n\n"
    "Reply as PLAIN TEXT, NOT JSON — the issues are long prose, and prose escaped "
    "into a JSON string is fragile (one stray quote would discard every issue at "
    "once). Use this EXACT line format:\n"
    "- The FIRST line is the verdict and contains ONLY one word: OK if the design "
    "is sound, or ISSUES if you found any. Put nothing else on that line.\n"
    "- If the verdict is ISSUES, write each issue on its OWN line below — one "
    "issue per line, naming what is wrong and where. Every non-blank line after "
    "the verdict is read as exactly one issue.\n"
    "Emit nothing else: no preamble, no JSON, no code fence, no closing remark."
)


class _AuditParseError(ValueError):
    """The design-audit reply did not present a recognizable ``OK`` / ``ISSUES``
    verdict. Raised LOUDLY (never swallowed into a fabricated 'sound') so a
    malformed verdict fails the stage — :func:`saddle.supervisor.run_stage` turns
    it into a classified ALERT naming what saddle could not verify this turn. The
    cardinal sin this guards is the opposite: silently reading a no-verdict reply
    as OK and dropping the real issues with it.

    It surfaces through the retry taxonomy as ``other`` (a soft, surfaced
    disposition — the loud ALERT already carries this message verbatim), NOT as a
    swallow; coupling :mod:`saddle.llm.retry_category` to this design-layer type to
    relabel it would be an inversion for no behavioural gain."""


# A leading list bullet / number on an issue line — stripped COSMETICALLY when
# collecting issues. Removing the glyph never drops the issue's CONTENT; it only
# tidies the surfaced text.
_AUDIT_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _audit_verdict_token(line: str) -> str:
    """The verdict line reduced to its bare alpha token: every non-letter dropped,
    upper-cased. ``**OK.**`` -> ``OK``; ``issues:`` -> ``ISSUES``; ``OK looks
    good`` -> ``OKLOOKSGOOD`` (NOT a verdict). Demanding the WHOLE line reduce to
    exactly ``OK`` / ``ISSUES`` is the cardinal-sin guard: an issue crammed onto
    the verdict line can never be silently read as a bare OK — it fails the match
    and raises."""
    return re.sub(r"[^A-Za-z]", "", line or "").upper()


def _parse_audit_reply(text: str) -> tuple[bool, list[str]]:
    """Parse the design-audit reply's LINE contract into ``(ok, issues)``.

    NOT JSON — and that is the FIX, not an accident. The audit's issues are long
    free prose, and prose hand-escaped into a JSON array is the harness's most
    fragile parse (the same reason the design BODY rides as text — module
    docstring): one unescaped quote in ONE issue throws the WHOLE ``json.loads``
    and discards every OTHER issue with it — a fabricated 'sound', the cardinal
    sin. A line contract has no per-issue escaping to break, so a single ugly issue
    can never take the rest of the review down.

    Loud-accounting invariants — each closes the false-negative seam:
      • empty / whitespace-only reply                 -> raise (no verdict to
        trust; never guessed into a pass).
      • first non-blank line is the VERDICT and must reduce to EXACTLY ``OK`` or
        ``ISSUES`` (:func:`_audit_verdict_token`); any other first line -> raise. A
        missing / misplaced verdict, or an issue crammed onto the verdict line, is
        never read as OK.
      • ``OK`` is honored ONLY when no non-blank line follows it; ``OK`` with
        trailing content is a contradiction -> raise (never trust OK over content
        the model also chose to write).
      • ``ISSUES`` -> EVERY following non-blank line is exactly one issue, verbatim
        (a leading bullet is stripped cosmetically). No per-line marker is required,
        so there is no marker a model can forget that would silently drop a line,
        and no per-line parse step that could fail mid-stream.
      • ``ISSUES`` with no following content -> ``(False, [])``, preserving the
        prior contract's 'flagged not-ok but named nothing actionable' semantics.

    A think-block (reasoning models) is stripped first so it never counts as the
    verdict / a preamble. Raises :class:`_AuditParseError` on any unrecognized
    verdict — fail-loud for the supervisory runner to classify and bubble."""
    clean = strip_think(text or "").strip()
    lines = clean.splitlines()
    head_idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if head_idx is None:
        raise _AuditParseError("empty audit reply — no verdict to trust")
    token = _audit_verdict_token(lines[head_idx])
    rest = [ln.strip() for ln in lines[head_idx + 1:] if ln.strip()]
    if token == "OK":
        if rest:
            raise _AuditParseError(
                "audit reply opened OK but also wrote content — refusing to trust "
                f"OK over {len(rest)} trailing line(s): {rest[0]!r}"
            )
        return True, []
    if token != "ISSUES":
        raise _AuditParseError(
            "audit reply did not open with an OK / ISSUES verdict: "
            f"{lines[head_idx]!r}"
        )
    issues = [_AUDIT_BULLET_RE.sub("", ln).strip() for ln in rest]
    return False, [i for i in issues if i]

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
        "Return the verdict line (OK / ISSUES) and, if ISSUES, one issue per line."
    )


async def _audit_call(
    caller: "LLMCaller",
    goal: str,
    body: str,
    directives: list[str],
    anti: str,
    *,
    label: str,
) -> tuple[bool, list[str]]:
    """Run the audit stage and parse its LINE-contract reply into ``(ok, issues)``.

    The single audit entry point BOTH call sites share — :func:`design_for`'s
    self-audit of its OWN generated body, and :func:`audit_proposal`'s gate over
    the agent's prose — so the bar AND the parse are identical no matter who
    authored the design. Deliberately ``json_mode=False``: the reply is the plain
    verdict-line contract (:func:`_parse_audit_reply`), not JSON, because long
    prose issues escaped into a JSON array are the harness's most fragile parse —
    one stray quote would discard every issue at once. A malformed verdict raises
    (:class:`_AuditParseError`) and PROPAGATES for ``run_stage`` to classify and
    bubble; it is never swallowed into a false pass."""
    text = await caller(
        _SYS_AUDIT,
        _audit_prompt(goal, body, directives, anti),
        json_mode=False,
        label=label,
    )
    return _parse_audit_reply(text)


def _harvest_prompt(goal: str, issues: list[str]) -> str:
    bullet = "\n".join(f"- {i}" for i in issues)
    return (
        f"GOAL:\n{goal}\n\nFLAWS CAUGHT AND CORRECTED IN AUDIT:\n{bullet}\n\n"
        "Extract any generalizable lessons or anti-patterns worth remembering."
    )


async def _harvest(
    caller: "LLMCaller", ctx: Context, dkb: DKB, goal: str, issues: list[str]
) -> list[Knowledge]:
    """File generalizable lessons from caught flaws into the DKB (source=audit).
    Returns the lessons actually filed — deduped against existing titles — so a
    caller can name them (the turn-end harvest bubbles the titles it learned)."""
    payload = await call_json(
        caller, _SYS_HARVEST, _harvest_prompt(goal, issues), label="design/harvest"
    )
    raw = payload.get("entries")
    if not isinstance(raw, list) or not raw:
        return []
    existing_titles = {
        _norm(k.title)
        for k in await asyncio.to_thread(dkb.list_knowledge, ctx, limit=1000)
    }
    added: list[Knowledge] = []
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
        added.append(kn)
    if added:
        _log.info("harvested %d lesson(s) into the DKB for %s", len(added), ctx.key)
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

        # 6: audit + bounded revise loop — operates on the prose body. The audit
        # rides a LINE contract (verdict line + one issue per line), NOT JSON: the
        # issues are long prose and JSON-escaped prose is the harness's most fragile
        # parse — one stray quote would discard every issue at once (the same reason
        # the body above is text). _parse_audit_reply already strips/filters.
        for _ in range(max(0, max_audits)):
            ok_audit, issues = await _audit_call(
                caller, goal, body, directives, anti_block, label="design/audit",
            )
            audits_run += 1
            if ok_audit and not issues:
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
            harvested = len(await _harvest(caller, ctx, dkb, goal, all_issues))

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


@dataclass
class AuditVerdict:
    """The result of auditing an ALREADY-WRITTEN approach — Stage 3's engine, as
    opposed to :func:`design_for` which GENERATES a design and audits its own
    output. ``ok`` is True only when the approach is sound; ``issues`` names each
    flaw the audit caught — a band-aid, a misread root cause, hand-waving, an
    uncovered ask, a violated directive. ``considered`` is how many DKB
    anti-patterns were weighed against it (0 = nothing to measure against)."""

    ok: bool = True
    issues: list[str] = field(default_factory=list)
    considered: int = 0

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


async def audit_proposal(
    goal: str,
    approach: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
    dkb: DKB | None = None,
    directives: list[str] | None = None,
    retrieve_k: int = 6,
) -> AuditVerdict:
    """Audit an approach the agent has ALREADY proposed against the goal it
    serves — the pre-code design gate's engine (Stage 3).

    Distinct from :func:`design_for`, which runs the whole diagnose → … → harvest
    pipeline to PRODUCE a design: the supervised agent designs in its own
    transcript and never calls saddle, so Stage 3 lifts that prose out and audits
    IT — through the EXACT ``_SYS_AUDIT`` stage the design pipeline applies to its
    own body. The bar is therefore identical whether saddle authored the design or
    merely watched the agent author it: did the approach reach the root cause (not
    a symptom), honor the binding directives, avoid a band-aid (tolerant default /
    hardcoded backstop / swallow-and-log) instead of a structural fix, and cover
    the whole goal?

    The anti-patterns are pulled from the DKB with the GLOBAL seed corpus
    INCLUDED — measuring an approach against universal engineering wisdom is
    exactly Stage 3's job, the deliberate complement to Stage 2 (``intent``),
    which weighs only THIS project's own settled state. A failed classify
    PROPAGATES (fail-loud) for the supervisory runner to classify and bubble — it
    is never swallowed into a false 'looked fine'.
    """
    goal = (goal or "").strip()
    approach = (approach or "").strip()
    if not approach:
        raise ValueError("cannot audit an empty approach")
    ctx = ctx or _default_ctx()
    dkb = dkb or get_dkb()
    if directives is None:
        directives = policy.directives(ctx)
    if caller is None:
        from saddle.llm.callers import build_callers
        caller = build_callers(ctx)["default"]

    async with tenant_gate(ctx):
        anti_hits = await _search(
            dkb, ctx, f"{goal}\n{approach}", k=retrieve_k, kinds=[ANTI_PATTERN]
        )
        anti_block = _format_knowledge([k for k, _ in anti_hits])
        ok_audit, issues = await _audit_call(
            caller, goal, approach, directives, anti_block, label="design/gate-audit",
        )
    return AuditVerdict(
        ok=ok_audit and not issues,
        issues=issues,
        considered=len(anti_hits),
    )


@dataclass
class HarvestResult:
    """What a turn-end lesson harvest filed (Stage 5). ``titles`` names the durable
    lessons written to the DKB this turn; ``considered`` is how many caught flaws
    were fed in. ``harvested == 0`` with ``considered > 0`` means nothing cleared
    the 'genuinely general, not obvious' bar, or every lesson was already on file —
    a real outcome, not a failure (the gate is deliberately conservative)."""

    titles: list[str] = field(default_factory=list)
    considered: int = 0

    @property
    def harvested(self) -> int:
        return len(self.titles)


async def harvest_turn(
    goal: str,
    issues: list[str],
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
    dkb: DKB | None = None,
) -> HarvestResult:
    """Stage 5's engine: distil the flaws CAUGHT this turn into durable, reusable
    DKB lessons (``source=audit``), so the NEXT turn's design/intent retrieval
    stands on them — saddle's cumulative, not 'caught once and done', guarantee.

    Reuses :func:`_harvest` — the EXACT lesson-extraction the design pipeline runs
    over its OWN caught flaws — so a flaw caught by the live supervisor and one
    caught by ``design_for`` teach the DKB the same way. Deduped against existing
    titles, scoped to this tenant/project. No caught flaws ⇒ no LLM call ⇒ empty
    result (a clean turn teaches nothing). A failed classify PROPAGATES (fail-loud)
    for the supervisory runner to surface — never swallowed into a false 'learned'.
    """
    issues = [str(i).strip() for i in (issues or []) if str(i).strip()]
    if not issues:
        return HarvestResult()
    ctx = ctx or _default_ctx()
    dkb = dkb or get_dkb()
    if caller is None:
        from saddle.llm.callers import build_callers
        caller = build_callers(ctx)["default"]
    async with tenant_gate(ctx):
        added = await _harvest(caller, ctx, dkb, goal, issues)
    return HarvestResult(titles=[k.title for k in added], considered=len(issues))


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


@dataclass
class ConformanceDrift:
    """One settled design whose committed completeness surface the code, as it
    stands now, no longer satisfies. ``findings`` are the exact unsatisfied
    touchpoints :func:`intent_drift` raised; ``has_error`` is True when any of
    them is error-grade (a hard conformance break, not a soft gap)."""

    design_id: str
    summary: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(f.severity == "error" for f in self.findings)


@dataclass
class ConformanceResult:
    """Stage 4's verdict: which settled designs the current code has DRIFTED from.
    ``designs_checked`` is how many surfaced designs were gated (0 ⇒ nothing to
    verify — no settled design declared a surface, or there is no code); a
    non-empty ``drifts`` is always a real divergence (``intent_drift``'s
    guarantee)."""

    drifts: list[ConformanceDrift] = field(default_factory=list)
    designs_checked: int = 0

    @property
    def has_drift(self) -> bool:
        return bool(self.drifts)

    @property
    def has_error(self) -> bool:
        return any(d.has_error for d in self.drifts)


def conformance_scan(
    ctx: Context,
    *,
    dkb: DKB | None = None,
    root: str | Path | None = None,
    limit: int = 8,
    status: str = DESIGN_FINAL,
) -> ConformanceResult:
    """Re-verify the project's SETTLED designs against the code as it stands now —
    Stage 4's engine (turn-end code-vs-design conformance).

    Each design that reached the SURFACE stage carries a persisted
    :class:`SurfaceManifest` (``design.meta['surface']``) — the completeness floor
    the design committed to. This lists the project's most-recent ``limit``
    designs in ``status`` (default :data:`DESIGN_FINAL` — only a *settled* design
    is a contract the code must honor; a still-flagged design never converged, so
    gating code against it is premature), keeps the ones that declared a surface,
    parses the target tree ONCE, and re-runs each design's own gate
    (:func:`intent_drift`) against that fresh parse. The surface is intent, the
    code is truth; a design that turns up in ``drifts`` is one the agent's edits
    left unsatisfied — the "committed to a structural fix, then quietly wrote the
    swallow anyway" miss, caught retrospectively.

    No resolvable code root, or no settled design with a surface ⇒ an empty result
    (nothing can drift) — the documented silent case. Parsing once and threading
    ``mods`` into every gate keeps a turn-end scan O(one parse), not O(designs).
    The scan is read-only: it never mutates a design or its status (an
    implementation outcome is not a re-verdict on the design's quality)."""
    dkb = dkb or get_dkb()
    rp = _resolve_code_root(root)
    if rp is None:
        return ConformanceResult()  # no code to compare the intent against
    designs = dkb.list_designs(ctx, status=status, limit=limit)
    surfaced = [
        d for d in designs
        if not SurfaceManifest.from_dict(d.meta.get("surface")).is_empty()
    ]
    if not surfaced:
        return ConformanceResult()  # nothing declared a surface ⇒ nothing to gate
    mods = refs.parse_project(rp)
    result = ConformanceResult(designs_checked=len(surfaced))
    for d in surfaced:
        findings = intent_drift(d, root=rp, mods=mods)
        if findings:
            result.drifts.append(
                ConformanceDrift(
                    design_id=d.id, summary=d.summary or d.ask, findings=findings
                )
            )
    return result


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
