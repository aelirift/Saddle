"""The impact set — the COMPLETE code fan-out of one value / identity / boundary.

These are the three DATAFLOW kinds. The two substrate kinds — cross-substrate
references and save/load persistence — fan out the same way in substrate.py
(their ``gaps()`` is likewise their ``check_*``); this module is not the whole
map, just its dataflow third.

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

from dataclasses import dataclass, field

from . import refs
from .finding import Finding
from .pyref import Ref
from .specs import BoundarySpec, IdentitySpec, ValueSpec


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
