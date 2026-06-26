"""Deterministic congruence pre-pass — the codemap's 9th check, auto-derived.

WHY THIS EXISTS
---------------
RayXI shipped the client/server congruence bug FOUR times (mount_collection,
stat_allocation, auction_house, profession_skill_curve): a service that ships a
client mirror (an ``apply_*`` snapshot applier) whose state MUTATORS are ungated
and are CALLED DIRECTLY by a HUD — so the HUD mutates the local mirror, the next
server snapshot reverts it (the change flickers then vanishes), and the server
never learns the intent.

The audit layer already probes ``client_server_congruence`` with an LLM. But an
LLM probe is a JUDGEMENT — it can miss a case, and it cannot PROVE the class is
gone. The codemap's :func:`saddle.codemap.impact.impact_congruence` proves it
deterministically from the AST. The only thing it needs is a spec naming the
project's ENGINE tokens: the mirror-applier function name and the authority-guard
names. This module DERIVES that spec from the project's own real shape — no
hand-authored spec, no game vocabulary — so the deterministic check runs on every
audit and the bug class cannot silently regrow.

WHAT IS DERIVED (all from the parsed code, never declared)
----------------------------------------------------------
  * mirror-applier name — a function whose name matches the replicated-state
    applier convention (``apply_*`` / ``recv_*`` / ``*_synced``) AND is defined in
    AT LEAST TWO modules, i.e. a real shared replication CONVENTION rather than a
    one-off. Each such name seeds one :class:`CongruenceSpec`; the impact then
    derives, per service, the replicated fields that applier writes and every
    public mutator of them. A nominated name that turns out to have no mirror
    shape simply yields no gaps (the impact's write-correlation is the safety net).
  * guard names — the engine authority tokens (``is_server`` / ``_is_server`` /
    ``is_multiplayer_authority`` / ``is_multiplayer_master``) that are ACTUALLY
    CALLED somewhere in the project. These are Godot multiplayer ENGINE vocabulary
    (the same set :data:`saddle.codemap.gdref._AUTHORITY_GUARD_RE` matches), not
    game vocabulary, so naming them bakes in no genre.

If no shared applier convention is found, or no guard is ever called, nothing is
derived and the pre-pass is silently empty — the safe failure mode that lets it
run harmlessly on a project with no replication mirror at all.
"""
from __future__ import annotations

import re

from saddle.audit.finding import AuditFinding, CONGRUENCE, ERROR
from saddle.codemap import refs
from saddle.codemap.impact import impact_congruence
from saddle.codemap.specs import CongruenceSpec

# The audit concern target this deterministic pass corroborates (so its findings
# reconcile with the LLM probe's under one target id).
CONGRUENCE_CONCERN = "concern:client_server_congruence"

# Replicated-state applier naming — the client-side snapshot receiver convention.
# Mirrors saddle.codemap.gdref._CLIENT_FUNC_RE (the per-function domain classifier),
# kept here as the candidate filter for "which function names name a mirror applier".
_APPLIER_RE = re.compile(r"^_?recv_|^apply_|_synced$")

# Engine authority guards (Godot multiplayer). The same canonical set the codemap's
# _AUTHORITY_GUARD_RE matches — engine vocabulary, not game vocabulary. `_is_server`
# is the near-universal thin wrapper a project writes around the built-in, treated as
# a primitive here so the seed catches a project that only ever calls the wrapper.
_ENGINE_GUARDS: tuple[str, ...] = (
    "is_server", "_is_server", "is_multiplayer_authority", "is_multiplayer_master",
)

# A PROJECT authority-predicate WRAPPER is named by the engine/networking authority
# convention (server / authority / host / owner / master, optionally underscore-
# prefixed) AND its body actually calls an engine primitive. BOTH halves are required:
# the name alone would misread a game function coincidentally called `is_owner` as a
# guard, and the call alone would sweep in every gated MUTATOR (a mutator that calls
# `_is_server()` is gated, not itself a guard) — which, added to the guard set, would
# then wrongly mark its own callers "guarded". Name ∧ behaviour isolates real wrappers
# like RayXI's `_is_authority()` (named right, returns `ms.call("is_server")`).
_WRAPPER_NAME_RE = re.compile(
    r"^_?is_(?:server|authority|host|owner|master"
    r"|multiplayer_authority|multiplayer_master)$"
)

# An applier name must be defined in at least this many modules to count as a real
# replication CONVENTION (rather than an incidental apply_damage-style one-off).
_MIN_APPLIER_MODULES = 2

_FOUR_PART_FIX = (
    "Gate the mutator with the authority guard (`if not <guard>(): return`) AND "
    "route the caller through a server intent/RPC, then repaint from the mirror "
    "applier — the proven four-part fix: (1) guard the mutator, (2) IntentRouter "
    "client sender send_X, (3) server @rpc receiver _recv_X, (4) emit a repaint "
    "signal from the snapshot applier."
)


def _derive_guards(mods: list) -> tuple[str, ...]:
    """Every authority guard the project actually uses — the engine primitives it
    calls PLUS its own predicate wrappers around them.

    Two tiers, both grounded in real calls so nothing is invented:
      * engine primitives — the canonical Godot built-ins (:data:`_ENGINE_GUARDS`)
        that appear at a call site anywhere in the project;
      * project wrappers — functions named by the authority convention
        (:data:`_WRAPPER_NAME_RE`) whose body calls one of those primitives, e.g.
        RayXI's ``_is_authority()`` returning ``ms.call("is_server")``. Without this
        tier a mutator gated by the project's own wrapper reads as ungated — the
        exact false positive ``claim_choice`` produced.

    Order is stable (engine first, then wrappers sorted) and de-duplicated."""
    engine = tuple(g for g in _ENGINE_GUARDS if any(refs.calls_to(m, g) for m in mods))
    if not engine:
        return ()
    wrappers: set[str] = set()
    for m in mods:
        callers = {r.func for g in engine for r in refs.calls_to(m, g) if r.func}
        wrappers |= {fn for fn in callers if _WRAPPER_NAME_RE.match(fn)}
    # A wrapper already in the engine seed (e.g. `_is_server`) must not double-list.
    return (*engine, *(w for w in sorted(wrappers) if w not in engine))


def derive_congruence_specs(mods: list) -> list[CongruenceSpec]:
    """The project's congruence specs, derived from the parsed modules' real shape.

    One spec per shared mirror-applier convention; each carries the guard names the
    project actually uses (engine primitives + its own predicate wrappers, see
    :func:`_derive_guards`). Empty when no replication mirror is present — so the
    deterministic check fires with NO hand-authored spec, yet never invents one for
    a project that has no such service."""
    applier_modcount: dict[str, int] = {}
    for m in mods:
        defined = {fn for fn in refs.function_names(m) if _APPLIER_RE.search(fn)}
        for fn in defined:
            applier_modcount[fn] = applier_modcount.get(fn, 0) + 1
    appliers = sorted(
        fn for fn, c in applier_modcount.items() if c >= _MIN_APPLIER_MODULES
    )
    if not appliers:
        return []

    guards = _derive_guards(mods)
    if not guards:
        return []

    return [
        CongruenceSpec(name=f"congruence:{ap}", mirror_apply=ap, guard=guards)
        for ap in appliers
    ]


def _to_audit_finding(f) -> AuditFinding:
    """Project one codemap congruence gap into a grounded, high-confidence audit
    finding under the cross-cutting congruence concern. The title is shape-aware so
    the two halves of the bug class read true: an UNGATED mutator's local change is
    reverted by the next snapshot; a GATED-but-UNROUTED mutator's guard bails on a
    remote client and the intent is silently lost."""
    func = f.detail.get("func", "?")
    ext = list(f.detail.get("external_calls", []))
    evidence = [f.location, *ext]
    if f.detail.get("gap_kind") == "unrouted":
        title = (f"gated-but-unrouted replicated-state mutator {func!r} is called "
                 f"directly from client/UI code — on a remote client the authority "
                 f"guard bails, the click no-ops, and the server never learns the intent")
    else:
        title = (f"ungated replicated-state mutator {func!r} is called from "
                 f"non-server code — the local change is reverted by the next snapshot")
    return AuditFinding(
        target=CONGRUENCE_CONCERN,
        severity=ERROR,
        kind=CONGRUENCE,
        title=title,
        detail=f.message,
        evidence=evidence,
        suggestion=_FOUR_PART_FIX,
        confidence="high",
    )


def deterministic_congruence_findings(mods: list) -> list[AuditFinding]:
    """Every congruence gap the codemap proves on ``mods``, as audit findings.

    Auto-derives the specs, runs the codemap impact for each, and projects its gaps
    into grounded ``congruence`` findings. Deterministic and citation-backed — the
    complement to (not a replacement for) the LLM concern probe, which still catches
    the shapes off this axis (a server write never replicated at all)."""
    out: list[AuditFinding] = []
    for spec in derive_congruence_specs(mods):
        for gap in impact_congruence(mods, spec).gaps():
            out.append(_to_audit_finding(gap))
    return out
