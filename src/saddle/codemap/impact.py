"""The impact set — the COMPLETE code fan-out of one value / identity / boundary
/ lifecycle / authority.

These are the AST-derived kinds. value/identity/boundary fan out a value through
the code's dataflow; lifecycle asks the orthogonal question — does a declared
symbol have ANY read at all (a dead knob looks adjustable but changes nothing);
authority asks the trust-boundary WRITE question — does a mutator of server state
guard against a client invoking it. The two substrate kinds — cross-substrate
references and save/load persistence — fan out the same way in substrate.py
(their ``gaps()`` is likewise their ``check_*``); this module is the code-derived
portion of the map.

WHY THIS EXISTS
---------------
checks.py answers "is anything INCOMPLETE?" and returns only the gaps. The
user's real question is bigger: "if I add or remove this feature, what ELSE must
change?" Answering THAT needs the whole fan-out — every site that reads the
value, every site that writes it, every place the effective value is resolved,
across every domain. The gaps are just one slice of that set (the consumer reads
that nothing resolves).

This is RayXI's value_impact_map done right and, decisively, WIRED. RayXI built
the equivalent fan-out and then imported it from nobody, so it never ran at a
gate — the single defect that let the talent-cooldown bug ship. Here the impact
set is the ONE source the gate projects from: ``ValueImpact.gaps()`` IS
``check_value``. A Finding is, by construction, an uncovered entry in the impact
set, so the map you read for "what else must change" and the gate that blocks the
commit can never drift apart. One derivation, two readers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import refs
from .finding import Finding
from .pyref import Ref
from .specs import (
    AuthoritySpec,
    BoundarySpec,
    CongruenceSpec,
    IdentitySpec,
    LifecycleSpec,
    ValueSpec,
)

# Verification harnesses — build-oracle probes and unit/integration tests — call
# mutators DIRECTLY to exercise them, run with authority, and are NOT the client UI
# surface. They are excluded from the congruence caller scan by filename convention
# (probe/test/spec naming is universal across projects and languages), so a test that
# drives a mutator is never misread as a HUD that mutates replicated state.
#
# Matched on the BASENAME (not the whole path) for the file conventions, so a source
# file living under an incidental ancestor dir merely NAMED like a test — e.g. pytest's
# own ``/tmp/.../test_foo0/`` fixture root — is NOT mistaken for a harness. Only a real
# ``test/`` or ``tests/`` directory segment counts as a harness directory.
_HARNESS_FILE_RE = re.compile(r"(?:^test_|_test\.|_spec\.|_probe\.)")


def _is_harness_path(path: str) -> bool:
    segs = path.replace("\\", "/").lower().split("/")
    if _HARNESS_FILE_RE.search(segs[-1]):
        return True
    return any(seg in ("test", "tests") for seg in segs[:-1])


@dataclass
class ValueImpact:
    """Every site that touches a varying value. ``resolver_sites`` are where the
    effective value is computed; ``all_writes`` where the base is set;
    ``all_reads`` every base read, split by the properties into producer (the
    builder's own legit read), covered (resolved in scope) and uncovered (the
    gap a modifier never reaches)."""
    spec: ValueSpec
    resolver_sites: list[Ref] = field(default_factory=list)
    all_reads: list[Ref] = field(default_factory=list)
    all_writes: list[Ref] = field(default_factory=list)
    # Callee names that receive the resolved value as a call argument — their raw
    # reads are covered one hop out (see pyref.resolved_arg_callees).
    arg_covered_funcs: set[str] = field(default_factory=set)

    @property
    def _covering(self) -> set[tuple[str, str | None]]:
        # A (path, enclosing-func) is "covered" iff it contains a resolver call —
        # exactly the set check_value built, re-derived from the resolver sites.
        return {(r.path, r.func) for r in self.resolver_sites}

    def _is_covered(self, r: Ref, cov: set[tuple[str, str | None]]) -> bool:
        # Resolved in scope (a resolver call shares the read's function), OR the
        # read's function received the resolved value as an argument one hop out.
        return (r.path, r.func) in cov or r.func in self.arg_covered_funcs

    @property
    def producer_reads(self) -> list[Ref]:
        ex = self.spec.exempt_funcs
        return [r for r in self.all_reads if r.func in ex]

    @property
    def covered_reads(self) -> list[Ref]:
        ex, cov = self.spec.exempt_funcs, self._covering
        return [r for r in self.all_reads
                if r.func not in ex and self._is_covered(r, cov)]

    @property
    def uncovered_reads(self) -> list[Ref]:
        ex, cov = self.spec.exempt_funcs, self._covering
        return [r for r in self.all_reads
                if r.func not in ex and not self._is_covered(r, cov)]

    @property
    def domains(self) -> list[str]:
        return sorted({r.domain for r in
                       self.all_reads + self.all_writes + self.resolver_sites})

    def gaps(self) -> list[Finding]:
        resolvers = self.spec.resolvers
        return [
            Finding(
                check="value_propagation",
                severity="error",
                node_kind="value",
                thing=self.spec.name,
                message=(
                    f"reads base {self.spec.field!r} without resolving via "
                    f"{' / '.join(resolvers)} in scope — a change to "
                    f"{self.spec.name} (talent/buff/modifier) won't reach this site"
                ),
                location=r.location,
                detail={"func": r.func, "domain": r.domain, "kind": r.kind},
            )
            for r in self.uncovered_reads
        ]


@dataclass
class IdentityImpact:
    """Every site that touches an identity namespace: each canonical-set
    declaration, and every literal used AS the identity, split into members
    (in the canonical set — fine) and drift (not in it — the typo/divergence)."""
    spec: IdentitySpec
    decls: list[tuple[set, Ref]] = field(default_factory=list)
    member_refs: list[tuple[str, Ref]] = field(default_factory=list)
    drift_refs: list[tuple[str, Ref]] = field(default_factory=list)

    @property
    def domains(self) -> list[str]:
        return sorted({r.domain for _, r in
                       self.member_refs + self.drift_refs} |
                      {r.domain for _, r in self.decls})

    def gaps(self) -> list[Finding]:
        out: list[Finding] = []
        if len(self.decls) > 1:
            locs = [r.location for _, r in self.decls]
            _, dup = self.decls[1]
            out.append(Finding(
                check="identity_membership",
                severity="error",
                node_kind="identity",
                thing=self.spec.name,
                message=(
                    f"canonical set {self.spec.source_symbol!r} declared in "
                    f"{len(self.decls)} places {locs} — single-source it so the "
                    f"members can't drift apart"
                ),
                location=dup.location,
                detail={"locations": locs},
            ))
        for lit, r in self.drift_refs:
            out.append(Finding(
                check="identity_membership",
                severity="error",
                node_kind="identity",
                thing=self.spec.name,
                message=(
                    f"uses {lit!r} as a {self.spec.name} but it is not in the "
                    f"canonical set {sorted(self.spec.canonical)} — drift / typo "
                    f"(an enum 'lock' is a declaration; this is the enforcement)"
                ),
                location=r.location,
                detail={"literal": lit, "func": r.func, "domain": r.domain},
            ))
        return out


@dataclass
class BoundaryImpact:
    """Every site that touches a value crossing the server→client boundary:
    all writes (``server_writes`` are the authoritative ones), all reads
    (``packed_reads`` ship it in the snapshot, ``client_reads`` land it on
    screen)."""
    spec: BoundarySpec
    all_writes: list[Ref] = field(default_factory=list)
    all_reads: list[Ref] = field(default_factory=list)

    @property
    def server_writes(self) -> list[Ref]:
        return [w for w in self.all_writes if w.domain == "server"]

    @property
    def packed_reads(self) -> list[Ref]:
        return [r for r in self.all_reads if r.func == self.spec.replication_func]

    @property
    def packed(self) -> bool:
        return bool(self.packed_reads)

    @property
    def client_reads(self) -> list[Ref]:
        return [r for r in self.all_reads if r.domain == "client"]

    def gaps(self) -> list[Finding]:
        sw = self.server_writes
        if not sw:
            return []  # nothing authoritative to mirror
        out: list[Finding] = []
        if not self.packed:
            w = sw[0]
            out.append(Finding(
                check="boundary_mirror",
                severity="error",
                node_kind="boundary",
                thing=self.spec.name,
                message=(
                    f"server writes {self.spec.key!r} but it is not read into "
                    f"{self.spec.replication_func!r} — the change never crosses "
                    f"to the client (server-authoritative, client-blind)"
                ),
                location=w.location,
                detail={"server_writes": [x.location for x in sw]},
            ))
        if not self.client_reads:
            w = sw[0]
            out.append(Finding(
                check="boundary_mirror",
                severity="error",
                node_kind="boundary",
                thing=self.spec.name,
                message=(
                    f"server writes {self.spec.key!r} but no client-domain code "
                    f"reads it — the mirrored value never appears on screen"
                ),
                location=w.location,
                detail={"hint": "client must read the replicated value or it is invisible"},
            ))
        return out


@dataclass
class LifecycleImpact:
    """Every site that touches a declared symbol: ``decls`` is each declaration
    (the ``@export``/const/signal/script-scope field), ``uses`` is every read of
    it anywhere in the project. The completeness question is the simplest of all —
    a symbol with declarations and ZERO uses is a DEAD knob: it looks adjustable
    but consumes into nothing. This is orthogonal to value propagation: that asks
    whether a read sees the modifier; this asks whether the declaration has any
    read at all."""
    spec: LifecycleSpec
    decls: list[Ref] = field(default_factory=list)
    uses: list[Ref] = field(default_factory=list)

    @property
    def domains(self) -> list[str]:
        return sorted({r.domain for r in self.decls + self.uses})

    def gaps(self) -> list[Finding]:
        # Only a DECLARED symbol can be dead — no declaration means the spec names
        # nothing in this code (a naming/spec error, not a liveness gap), so stay
        # silent rather than cry. A declaration with any read is alive.
        if not self.decls or self.uses:
            return []
        d = self.decls[0]
        return [Finding(
            check="lifecycle_liveness",
            severity="error",
            node_kind="lifecycle",
            thing=self.spec.name,
            message=(
                f"{self.spec.symbol!r} is declared but never read anywhere — a "
                f"DEAD knob: it looks adjustable (a designer can set it) but "
                f"changes nothing, because no code consumes it"
            ),
            location=d.location,
            detail={"symbol": self.spec.symbol,
                    "declarations": [x.location for x in self.decls]},
        )]


@dataclass
class AuthorityImpact:
    """Every site that bears on a set of authoritative mutators: ``mutator_defs``
    is each mutator's definition; ``guard_calls`` is every authority-guard call
    that sits INSIDE one of those mutators. A mutator defined with no guard call in
    its body is unguarded — a client can invoke it and desync / cheat. This is the
    write-side trust boundary, the mirror of BoundaryImpact's read side."""
    spec: AuthoritySpec
    mutator_defs: list[Ref] = field(default_factory=list)
    guard_calls: list[Ref] = field(default_factory=list)

    @property
    def guarded_funcs(self) -> set[str]:
        return {r.func for r in self.guard_calls if r.func is not None}

    @property
    def unguarded(self) -> list[Ref]:
        g = self.guarded_funcs
        return [r for r in self.mutator_defs if r.name not in g]

    @property
    def domains(self) -> list[str]:
        return sorted({r.domain for r in self.mutator_defs + self.guard_calls})

    def gaps(self) -> list[Finding]:
        out: list[Finding] = []
        for r in self.unguarded:
            out.append(Finding(
                check="authority_guard",
                severity="error",
                node_kind="authority",
                thing=self.spec.name,
                message=(
                    f"{r.name!r} mutates authoritative state but calls no authority "
                    f"guard ({' / '.join(self.spec.guards)}) in its body — a client "
                    f"can invoke it and desync / cheat (server trusts the caller)"
                ),
                location=r.location,
                detail={"func": r.name, "domain": r.domain},
            ))
        return out


@dataclass
class MutatorSite:
    """One public state-mutator of a mirror service, with everything the congruence
    gate needs to judge it: where it is defined, the replicated-state writes it
    performs, whether it calls an authority guard, every call site OUTSIDE its own
    module, and which of those callers is the ``@rpc`` server route.

    A replicated-state mutator is safe on a real client ONLY when it is BOTH
    authority-gated (so the direct call no-ops off the authority) AND routed (so a
    remote client's intent reaches the server through an ``@rpc`` receiver). The two
    ways that breaks are the two halves of the bug class:

      * UNGATED + raw client call — the client mutates its LOCAL mirror and the next
        server snapshot reverts it (the change flickers then vanishes). [shape #1]
      * GATED but UNROUTED + raw client call — on a remote client the guard bails and
        the intent is silently lost; the click does nothing, the server never learns.
        [shape #2 — the weekly_vault claim shape RayXI shipped]

    ``raw_client_callers`` are the CLIENT-domain (UI) callers that are NOT the ``@rpc``
    route (the genuine HUD invocations); ``rpc_routed`` is whether any ``@rpc`` receiver
    calls this mutator (a real server path exists).

    A caller counts as a genuine divergence only when it is CLIENT-domain — a HUD that
    runs on the client (a Godot Control/CanvasLayer script, see gdref's UI-extends
    promotion). A SHARED-domain service that calls the mutator (e.g. a server-side
    loot/talent service) runs on the authority, so its gated call is correct, not a
    bug; counting it would be the false positive the rayxi sweep exposed. The high-
    precision client signal is what makes this an ERROR-severity deterministic check
    rather than a guess; the fuzzier shared-caller cases stay the LLM probe's job."""
    module: str
    name: str
    def_ref: Ref
    write_refs: list[Ref] = field(default_factory=list)
    guarded: bool = False
    external_calls: list[Ref] = field(default_factory=list)
    # (path, lineno) of external call sites whose enclosing function is an @rpc
    # receiver — the server route, told apart from raw UI calls.
    rpc_caller_lines: set = field(default_factory=set)

    @property
    def rpc_routed(self) -> bool:
        return bool(self.rpc_caller_lines)

    @property
    def risky_callers(self) -> list[Ref]:
        # Every non-server external caller (includes the @rpc route) — kept for the
        # impact's domain ledger and the human-readable format.
        return [r for r in self.external_calls if r.domain != "server"]

    @property
    def raw_client_callers(self) -> list[Ref]:
        # CLIENT-domain callers that are NOT the @rpc route — the genuine HUD calls
        # that put the mutator on a real client's invocation path. Shared-domain
        # (server-side service) callers are excluded: their gated call runs on the
        # authority and is correct.
        return [r for r in self.external_calls
                if r.domain == "client"
                and (r.path, r.lineno) not in self.rpc_caller_lines]

    @property
    def is_gap(self) -> bool:
        # Reachable from a raw client/UI call AND not in the one safe shape (gated AND
        # routed). An ungated mutator with only server-domain or @rpc-route callers is
        # server-internal, not a congruence divergence, so it stays silent.
        if not self.raw_client_callers:
            return False
        return (not self.guarded) or (not self.rpc_routed)

    @property
    def gap_kind(self) -> str:
        """Which half of the break this is — drives the finding's explanation. An
        ungated mutator is reported as ``ungated`` even if it is also unrouted: gating
        is the more fundamental missing guard and the message covers both fixes."""
        return "ungated" if not self.guarded else "unrouted"


@dataclass
class CongruenceImpact:
    """Every site that bears on the server→client mirror congruence of a project:
    each mirror service (a module defining ``mirror_apply``) and, per service, every
    public mutator of its replicated state classified as guarded / ungated and with
    its external call sites. The gap slice is the ungated-AND-client-reachable
    mutators — the exact two-way-broken shape the audit's ``client_server_congruence``
    concern names, here proven deterministically from the AST instead of left to an
    LLM to rediscover across files."""
    spec: CongruenceSpec
    services: list[Ref] = field(default_factory=list)
    mutators: list[MutatorSite] = field(default_factory=list)

    @property
    def gap_mutators(self) -> list[MutatorSite]:
        return [m for m in self.mutators if m.is_gap]

    @property
    def domains(self) -> list[str]:
        return sorted({r.domain for r in self.services}
                      | {m.def_ref.domain for m in self.mutators})

    def gaps(self) -> list[Finding]:
        out: list[Finding] = []
        for m in self.gap_mutators:
            callers = sorted({r.location for r in m.raw_client_callers})
            if m.gap_kind == "ungated":
                why = (
                    f"calls no authority guard ({' / '.join(self.spec.guards)}) in its "
                    f"body, so on a real client the caller mutates the LOCAL mirror and "
                    f"the next server snapshot reverts it (the change flickers then "
                    f"vanishes)"
                )
            else:
                why = (
                    "is authority-gated but has NO server route (no @rpc receiver calls "
                    "it), so on a real client the guard bails and the player's intent is "
                    "silently lost — the click does nothing and the server never learns"
                )
            out.append(Finding(
                check="client_server_congruence",
                severity="error",
                node_kind="congruence",
                thing=self.spec.name,
                message=(
                    f"{m.name!r} mutates replicated state on a mirror service "
                    f"(the module defines {self.spec.mirror_apply!r}) and is CALLED from "
                    f"{len(m.raw_client_callers)} raw client/UI site(s) "
                    f"({', '.join(callers)}) — it {why}. Gate the mutator with the guard "
                    f"AND route the caller through a server intent/RPC, repainting from "
                    f"the snapshot applier (the proven four-part fix)"
                ),
                location=m.def_ref.location,
                detail={"func": m.name, "domain": m.def_ref.domain,
                        "gap_kind": m.gap_kind, "routed": m.rpc_routed,
                        "writes": [w.location for w in m.write_refs],
                        "external_calls": [r.location for r in m.raw_client_callers]},
            ))
        return out


def impact_value(mods: list, spec: ValueSpec) -> ValueImpact:
    imp = ValueImpact(spec=spec)
    resolver_set = set(spec.resolvers)
    for m in mods:
        for res in spec.resolvers:
            imp.resolver_sites.extend(refs.calls_to(m, res))
        imp.all_reads.extend(refs.field_reads(m, spec.field))
        imp.all_writes.extend(refs.field_writes(m, spec.field))
        imp.arg_covered_funcs |= refs.resolved_arg_callees(m, resolver_set)
    return imp


def impact_identity(mods: list, spec: IdentitySpec) -> IdentityImpact:
    imp = IdentityImpact(spec=spec)
    for m in mods:
        imp.decls.extend(refs.collection_decls(m, spec.source_symbol))
    for m in mods:
        for lit, r in refs.identity_refs(m, spec.carriers):
            (imp.member_refs if lit in spec.canonical else imp.drift_refs).append((lit, r))
    return imp


def impact_boundary(mods: list, spec: BoundarySpec) -> BoundaryImpact:
    imp = BoundaryImpact(spec=spec)
    for m in mods:
        imp.all_writes.extend(refs.field_writes(m, spec.key))
        imp.all_reads.extend(refs.field_reads(m, spec.key))
    return imp


def impact_lifecycle(mods: list, spec: LifecycleSpec) -> LifecycleImpact:
    imp = LifecycleImpact(spec=spec)
    for m in mods:
        imp.decls.extend(refs.name_decls(m, spec.symbol))
        imp.uses.extend(refs.name_uses(m, spec.symbol))
    return imp


def impact_authority(mods: list, spec: AuthoritySpec) -> AuthorityImpact:
    imp = AuthorityImpact(spec=spec)
    mutators = set(spec.mutators)
    for m in mods:
        for name in spec.mutators:
            imp.mutator_defs.extend(refs.function_defs(m, name))
        for g in spec.guards:
            # A guard call counts only when it sits inside one of the mutators —
            # the guard must protect THIS write, not merely exist somewhere.
            for r in refs.calls_to(m, g):
                if r.func in mutators:
                    imp.guard_calls.append(r)
    return imp


def _module_callees(m) -> dict[str, set[str]]:
    """Module-local call graph: each function name → the set of same-module functions
    it calls. Built from ``calls_to`` (whose Refs carry the enclosing function in
    ``.func``), so it needs no new per-language primitive and works for every adapter.
    Cross-module and library calls fall away because only names this module defines are
    nodes — exactly the scope a transitive same-module write-following needs."""
    graph: dict[str, set[str]] = {}
    for callee in refs.function_names(m):
        for r in refs.calls_to(m, callee):
            if r.func:
                graph.setdefault(r.func, set()).add(callee)
    return graph


def _effective_writes(m, fn: str, callees: dict[str, set[str]],
                      follow_public: bool = False) -> list:
    """The replicated-state writes ``fn`` is RESPONSIBLE for: its own direct writes
    plus those of the same-module helpers it calls, transitively. A public mutator can
    land a replicated write only through a helper — chat_backend's ``send_message``
    writes the replicated ``_history`` solely via ``_record_history`` — and a direct-
    writes-only scan reads it as a non-writer, silently missing a real gated-but-
    unrouted client gap.

    PRIVATE helpers are always followed (an implementation detail of the caller).
    PUBLIC callees are followed only when ``follow_public`` is set — used for a GATED
    mutator, whose authority guard is the author's declaration "invoking me mutates
    authoritative state," so a write it reaches through a public delegate
    (``guild_system.kick`` → public ``leave`` → ``_guilds``) is still its
    responsibility for the per-action congruence question. For an UNGATED function
    (and for the snapshot applier, from which the replicated field set is derived)
    public callees are NOT followed: descending into an independent public mutator
    would double-attribute one logical write and could inflate the replicated set with
    a field the applier never lands. Cycle-safe via ``seen``; the entry ``fn`` always
    contributes its own writes regardless of name."""
    seen: set[str] = set()
    out: list = []
    stack = [fn]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        out.extend(refs.func_writes(m, cur))
        for nxt in callees.get(cur, ()):
            if nxt in seen:
                continue
            # Private helpers are always part of the caller's write responsibility; a
            # public callee is followed only from a gated mutator (follow_public).
            if nxt.startswith("_") or follow_public:
                stack.append(nxt)
    return out


def impact_congruence(mods: list, spec: CongruenceSpec) -> CongruenceImpact:
    """Derive the whole mirror-congruence fan-out from the parsed project — no
    hand-listed mutators. For every module defining ``mirror_apply`` (a mirror
    service), the replicated state is the set of base fields that ``mirror_apply``
    itself writes; each PUBLIC function writing one of those fields is a mutator,
    classified guarded/ungated by whether it calls a ``guard``, and annotated with
    its call sites in OTHER modules. Cross-file by construction — the per-function
    HUD-call ↔ ungated-mutator correlation an isolated per-file scan cannot see."""
    imp = CongruenceImpact(spec=spec)
    guards = set(spec.guards)
    exempt = set(spec.exempt)

    services = []
    for m in mods:
        defs = refs.function_defs(m, spec.mirror_apply)
        if defs:
            services.append(m)
            imp.services.extend(defs)
    if not services:
        return imp

    # The set of @rpc entry-point names each module defines — the server routes a
    # remote client's intent travels through. Computed once; a mutator called by one
    # of these is ROUTED (the safe shape), so the call is told apart from a raw HUD
    # call. Typically the IntentRouter autoload owns these, but an in-service @rpc
    # endpoint counts too.
    rpc_by_mod = {m.path: refs.rpc_receivers(m) for m in mods}

    for m in services:
        # Module-local call graph, built once per service, so a mutator's writes can
        # be followed through the private helpers it delegates to (the send_message →
        # _record_history → _history edge that a direct-only scan misses).
        callees = _module_callees(m)
        # Replicated state = the base fields mirror_apply writes when it lands the
        # snapshot on the client (counting its private appliers too). A mutator of
        # THOSE fields races the next snapshot; a write to any other field (a build-
        # time @export catalog) is not a mirror divergence and is correctly ignored.
        rep_fields = {w.name for w in _effective_writes(m, spec.mirror_apply, callees)}
        if not rep_fields:
            continue
        guard_funcs = {r.func for g in guards for r in refs.calls_to(m, g) if r.func}
        for fn in refs.function_names(m):
            if fn == spec.mirror_apply or fn in exempt or fn.startswith("_"):
                continue
            guarded = fn in guard_funcs
            direct = [w for w in refs.func_writes(m, fn) if w.name in rep_fields]
            # A gated mutator owns the writes it reaches through PUBLIC delegates too
            # (kick -> leave -> _guilds); an ungated function does not (its public
            # callee is an independent mutator), so follow_public tracks the guard.
            writes = [w for w in _effective_writes(m, fn, callees, follow_public=guarded)
                      if w.name in rep_fields]
            if not writes:
                continue  # touches no replicated state at all (a getter / unrelated write)
            # A function that touches replicated state ONLY through a private helper
            # (no direct write of its own) counts as a mutator only when it is
            # authority-GATED — the author's own `if not _is_server(): return` is the
            # declaration "this mutates authoritative state." An UNGATED function whose
            # sole replicated write is indirect is overwhelmingly a getter whose helper
            # lazily restores from save (collection_log.pages_state -> _ensure_restored)
            # or a cross-authority sync (store.apply_entitlement_grants ->
            # _record_ownership), NOT a per-action mirror mutation. Direct writes stay
            # the high-precision signal for the ungated (shape-1) case; the fuzzier
            # indirect-ungated cases are left to the LLM probe, not raised as a hard gap.
            if not direct and not guarded:
                continue
            defs = refs.function_defs(m, fn)
            if not defs:
                continue
            # The mutator's own parameter arity — the disambiguator that keeps a call
            # to a SAME-NAMED method on a different service (guild_system.invite/3 vs
            # group_framework.invite/2) from being charged to this one. A call whose
            # argument count cannot fit `[min_required, max_total]` is a different
            # function that merely shares the name; only when the arity is known on
            # BOTH sides and provably incompatible is the attribution dropped (an
            # indeterminate count never deletes a caller — the adapter's standing bias).
            arity = refs.func_arity(m, fn)
            # When name AND arity ALSO collide (chat_backend.kick/3 vs guild_system.kick/3),
            # the receiver decides: the service name this module declares for itself vs the
            # service a caller resolved its receiver from. An empty set (module declares no
            # locator identity) disables this filter — conservative, keeps every caller.
            my_services = refs.registered_service_names(m)
            ext: list[Ref] = []
            rpc_lines: set = set()
            for other in mods:
                # Verification harnesses (build-oracle probes / tests) drive mutators
                # directly with authority — they are not the client UI, so they never
                # make a mutator "client-reachable".
                if _is_harness_path(other.path):
                    continue
                rpc_here = rpc_by_mod.get(other.path, set())
                arg_counts = refs.call_arg_counts(other, fn) if arity is not None else {}
                recv_services = refs.call_receiver_services(other, fn) if my_services else {}
                for r in refs.calls_to(other, fn):
                    if arity is not None:
                        nargs = arg_counts.get(r.lineno)
                        if nargs is not None and not (arity[0] <= nargs <= arity[1]):
                            continue  # same name, incompatible arity — a different func
                    recv_svc = recv_services.get(r.lineno)
                    if recv_svc is not None and recv_svc not in my_services:
                        continue  # receiver resolved from a DIFFERENT named service
                    # An @rpc receiver calling the mutator is the server route — mark
                    # it (so it is not counted a raw client call) and note the mutator
                    # is routed, whether the receiver is in another module or this one.
                    if r.func in rpc_here:
                        rpc_lines.add((r.path, r.lineno))
                    if other.path == m.path:
                        continue  # a service calling its own mutator is server-internal
                    ext.append(r)
            imp.mutators.append(MutatorSite(
                module=m.path, name=fn, def_ref=defs[0], write_refs=writes,
                guarded=fn in guard_funcs, external_calls=ext,
                rpc_caller_lines=rpc_lines,
            ))
    return imp


def _loc_line(r: Ref) -> str:
    where = f"{r.func}()" if r.func else "<module>"
    return f"      {r.location}  [{r.domain}] {where}  {r.kind}"


def format_value_impact(imp: ValueImpact) -> str:
    s = imp.spec
    out = [f"VALUE {s.name!r}  (field {s.field!r}, resolves via {' / '.join(s.resolvers)})",
           f"  domains: {', '.join(imp.domains) or '(none)'}",
           f"  resolver call sites: {len(imp.resolver_sites)}"]
    for r in imp.resolver_sites:
        out.append(_loc_line(r))
    out.append(f"  base writes: {len(imp.all_writes)}")
    for r in imp.all_writes:
        out.append(_loc_line(r))
    out.append(f"  COVERED reads (modifier reaches): {len(imp.covered_reads)}")
    for r in imp.covered_reads:
        out.append(_loc_line(r))
    if imp.producer_reads:
        out.append(f"  producer reads (build base — exempt): {len(imp.producer_reads)}")
        for r in imp.producer_reads:
            out.append(_loc_line(r))
    out.append(f"  UNCOVERED reads (modifier MISSES — fix these): {len(imp.uncovered_reads)}")
    for r in imp.uncovered_reads:
        out.append(_loc_line(r))
    return "\n".join(out)


def format_identity_impact(imp: IdentityImpact) -> str:
    s = imp.spec
    out = [f"IDENTITY {s.name!r}  (canonical {s.source_symbol!r} = {sorted(s.canonical)})",
           f"  declarations: {len(imp.decls)}"]
    for _, r in imp.decls:
        out.append(_loc_line(r))
    out.append(f"  member uses (in canonical): {len(imp.member_refs)}")
    for lit, r in imp.member_refs:
        out.append(f"      {r.location}  [{r.domain}] {lit!r}")
    out.append(f"  DRIFT uses (NOT in canonical — fix these): {len(imp.drift_refs)}")
    for lit, r in imp.drift_refs:
        out.append(f"      {r.location}  [{r.domain}] {lit!r}")
    return "\n".join(out)


def format_lifecycle_impact(imp: LifecycleImpact) -> str:
    s = imp.spec
    out = [f"LIFECYCLE {s.name!r}  (symbol {s.symbol!r})",
           f"  declarations: {len(imp.decls)}"]
    for r in imp.decls:
        out.append(_loc_line(r))
    out.append(f"  reads: {len(imp.uses)}")
    for r in imp.uses:
        out.append(_loc_line(r))
    if imp.decls and not imp.uses:
        out.append("  DEAD: declared but read by nobody — fix or remove the knob")
    return "\n".join(out)


def format_authority_impact(imp: AuthorityImpact) -> str:
    s = imp.spec
    guarded = imp.guarded_funcs
    out = [f"AUTHORITY {s.name!r}  (guards: {' / '.join(s.guards)})",
           f"  mutators: {len(imp.mutator_defs)}"]
    for r in imp.mutator_defs:
        mark = "GUARDED" if r.name in guarded else "UNGUARDED — fix this"
        out.append(f"      [{mark}] {r.location}  [{r.domain}] {r.name}()")
    out.append(f"  guard calls inside mutators: {len(imp.guard_calls)}")
    for r in imp.guard_calls:
        out.append(_loc_line(r))
    return "\n".join(out)


def format_congruence_impact(imp: CongruenceImpact) -> str:
    s = imp.spec
    out = [f"CONGRUENCE {s.name!r}  (mirror services define {s.mirror_apply!r}; "
           f"guards: {' / '.join(s.guards)})",
           f"  mirror services: {len(imp.services)}"]
    for r in imp.services:
        out.append(f"      {r.location}  [{r.domain}]")
    out.append(f"  replicated-state mutators: {len(imp.mutators)}")
    for m in imp.mutators:
        if m.is_gap:
            n = len(m.raw_client_callers)
            if m.gap_kind == "ungated":
                mark = f"UNGATED + client-reachable — fix this ({n} raw caller(s))"
            else:
                mark = f"gated but UNROUTED + client-reachable — fix this ({n} raw caller(s))"
        elif m.guarded:
            route = "routed" if m.rpc_routed else "no client caller"
            mark = f"gated ({route})"
        else:
            mark = "ungated (server-internal — no client caller)"
        out.append(f"      [{mark}] {m.def_ref.location}  [{m.def_ref.domain}] {m.name}()")
        for r in m.raw_client_callers:
            out.append(f"          called by {r.location}  [{r.domain}] {r.func or '<module>'}()")
    return "\n".join(out)


def format_boundary_impact(imp: BoundaryImpact) -> str:
    s = imp.spec
    out = [f"BOUNDARY {s.name!r}  (key {s.key!r}, replicates via {s.replication_func!r})",
           f"  server writes: {len(imp.server_writes)}"]
    for r in imp.server_writes:
        out.append(_loc_line(r))
    out.append(f"  packed into snapshot: {'YES' if imp.packed else 'NO'} ({len(imp.packed_reads)} read(s))")
    for r in imp.packed_reads:
        out.append(_loc_line(r))
    out.append(f"  client reads: {len(imp.client_reads)}")
    for r in imp.client_reads:
        out.append(_loc_line(r))
    return "\n".join(out)
