"""The surface manifest — a design's declared touch-surface, and the bridge
between Layer 2 (design) and the Layer 3 gate.

WHY THIS EXISTS
---------------
The impact set answers "what does THIS value touch?" for one spec. A real design
touches several things at once (a talent that edits cooldown ALSO touches the
points counter, the skill description, the cast animation's duration…). The
manifest is the whole set the design commits to: every value that now varies,
every identity that must stay canonical, every value that crosses the
server→client boundary.

It is the bridge that makes "the high-level design must be comprehensive" real
AND verifiable:

  * Layer 2 EMITS a manifest when it designs a feature — forcing the design to
    enumerate its full surface instead of stopping at the symptom.
  * the manifest PERSISTS on the Design (rides in Design.meta, plain JSON), so
    later — when someone implements the design — the gate runs the design's OWN
    specs. Nobody re-types the touchpoints; the design already handed them over
    (the "hand over the impact set" lesson, end to end).
  * the gate is STILL code-derived: the manifest only says WHICH things matter;
    impact.py derives WHERE they are touched from the actual AST. Declaration
    names the thing; code proves the coverage. That split is the whole point —
    RayXI declared the relationships AND checked the declaration; here we declare
    only intent and verify against real dataflow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .finding import Finding
from .impact import (
    AuthorityImpact,
    BoundaryImpact,
    CongruenceImpact,
    IdentityImpact,
    LifecycleImpact,
    ValueImpact,
    format_authority_impact,
    format_boundary_impact,
    format_congruence_impact,
    format_identity_impact,
    format_lifecycle_impact,
    format_value_impact,
    impact_authority,
    impact_boundary,
    impact_congruence,
    impact_identity,
    impact_lifecycle,
    impact_value,
)
from .specs import (
    AuthoritySpec,
    BindingSpec,
    BoundarySpec,
    CongruenceSpec,
    IdentitySpec,
    LifecycleSpec,
    PersistenceSpec,
    ReferenceSpec,
    ValueSpec,
)
from .substrate import (
    format_binding_impact,
    format_persistence_impact,
    format_reference_impact,
    impact_binding,
    impact_persistence,
    impact_reference,
)

_log = logging.getLogger("saddle.codemap.manifest")


def _rows(items, conv, kind: str) -> list:
    """Convert each raw spec row, dropping (LOUDLY) any the LLM left malformed.

    A surface row that omits a required key is exactly the LLM-omission failure
    mode saddle exists to absorb: skip THAT row and keep the rest of the manifest
    (and the design body) rather than letting one missing field abort the whole
    design. The warning keeps every drop visible — resilient, never silent."""
    out: list = []
    for x in items or []:
        try:
            out.append(conv(x))
        except (KeyError, TypeError, AttributeError) as e:
            _log.warning("surface: dropping malformed %s spec %r (%s)", kind, x, e)
    return out


def _value_to_dict(s: ValueSpec) -> dict:
    return {"name": s.name, "field": s.field,
            "accessor": list(s.resolvers), "producers": list(s.producers)}


def _value_from_dict(d: dict) -> ValueSpec:
    acc = d.get("accessor")
    accessor = tuple(acc) if isinstance(acc, (list, tuple)) else acc
    return ValueSpec(name=d["name"], field=d["field"], accessor=accessor,
                     producers=tuple(d.get("producers", ())))


def _identity_to_dict(s: IdentitySpec) -> dict:
    return {"name": s.name, "canonical": sorted(s.canonical),
            "source_symbol": s.source_symbol, "carriers": sorted(s.carriers)}


def _identity_from_dict(d: dict) -> IdentitySpec:
    return IdentitySpec(name=d["name"], canonical=set(d.get("canonical", [])),
                        source_symbol=d["source_symbol"],
                        carriers=set(d.get("carriers", [])))


def _boundary_to_dict(s: BoundarySpec) -> dict:
    return {"name": s.name, "key": s.key, "replication_func": s.replication_func}


def _boundary_from_dict(d: dict) -> BoundarySpec:
    return BoundarySpec(name=d["name"], key=d["key"],
                        replication_func=d["replication_func"])


def _reference_to_dict(s: ReferenceSpec) -> dict:
    return {"name": s.name, "key": s.key, "substrates": list(s.substrates)}


def _reference_from_dict(d: dict) -> ReferenceSpec:
    return ReferenceSpec(name=d["name"], key=d["key"],
                         substrates=tuple(d.get("substrates", ())))


def _persistence_to_dict(s: PersistenceSpec) -> dict:
    return {"name": s.name, "key": s.key,
            "save_func": s.save_func, "load_func": s.load_func}


def _persistence_from_dict(d: dict) -> PersistenceSpec:
    return PersistenceSpec(name=d["name"], key=d["key"],
                           save_func=d["save_func"], load_func=d["load_func"])


def _lifecycle_to_dict(s: LifecycleSpec) -> dict:
    return {"name": s.name, "symbol": s.symbol}


def _lifecycle_from_dict(d: dict) -> LifecycleSpec:
    return LifecycleSpec(name=d["name"], symbol=d["symbol"])


def _authority_to_dict(s: AuthoritySpec) -> dict:
    return {"name": s.name, "guard": list(s.guards), "mutators": list(s.mutators)}


def _authority_from_dict(d: dict) -> AuthoritySpec:
    g = d.get("guard")
    guard = tuple(g) if isinstance(g, (list, tuple)) else g
    return AuthoritySpec(name=d["name"], guard=guard,
                         mutators=tuple(d.get("mutators", ())))


def _congruence_to_dict(s: CongruenceSpec) -> dict:
    return {"name": s.name, "mirror_apply": s.mirror_apply,
            "guard": list(s.guards), "exempt": list(s.exempt)}


def _congruence_from_dict(d: dict) -> CongruenceSpec:
    g = d.get("guard")
    guard = tuple(g) if isinstance(g, (list, tuple)) else g
    return CongruenceSpec(name=d["name"], mirror_apply=d["mirror_apply"],
                          guard=guard, exempt=tuple(d.get("exempt", ())))


def _binding_to_dict(s: BindingSpec) -> dict:
    return {"name": s.name, "keymap": s.keymap,
            "families": {k: list(v) for k, v in s.families.items()},
            "compatible": [list(p) for p in s.compatible],
            "programmatic": list(s.programmatic)}


def _binding_from_dict(d: dict) -> BindingSpec:
    return BindingSpec(
        name=d["name"], keymap=d["keymap"],
        families={k: tuple(v) for k, v in d.get("families", {}).items()},
        compatible=tuple(tuple(p) for p in d.get("compatible", ())),
        programmatic=tuple(d.get("programmatic", ())))


@dataclass
class SurfaceManifest:
    """The full set of things a design commits to touch. Serialises to plain
    JSON (so it can ride in ``Design.meta``) and round-trips back to the typed
    specs the gate runs."""
    values: list[ValueSpec] = field(default_factory=list)
    identities: list[IdentitySpec] = field(default_factory=list)
    boundaries: list[BoundarySpec] = field(default_factory=list)
    references: list[ReferenceSpec] = field(default_factory=list)
    persistence: list[PersistenceSpec] = field(default_factory=list)
    lifecycle: list[LifecycleSpec] = field(default_factory=list)
    authority: list[AuthoritySpec] = field(default_factory=list)
    congruences: list[CongruenceSpec] = field(default_factory=list)
    bindings: list[BindingSpec] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.values or self.identities or self.boundaries
                    or self.references or self.persistence or self.lifecycle
                    or self.authority or self.congruences or self.bindings)

    def to_dict(self) -> dict:
        return {
            "values": [_value_to_dict(s) for s in self.values],
            "identities": [_identity_to_dict(s) for s in self.identities],
            "boundaries": [_boundary_to_dict(s) for s in self.boundaries],
            "references": [_reference_to_dict(s) for s in self.references],
            "persistence": [_persistence_to_dict(s) for s in self.persistence],
            "lifecycle": [_lifecycle_to_dict(s) for s in self.lifecycle],
            "authority": [_authority_to_dict(s) for s in self.authority],
            "congruences": [_congruence_to_dict(s) for s in self.congruences],
            "bindings": [_binding_to_dict(s) for s in self.bindings],
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "SurfaceManifest":
        d = d or {}
        return cls(
            values=_rows(d.get("values"), _value_from_dict, "value"),
            identities=_rows(d.get("identities"), _identity_from_dict, "identity"),
            boundaries=_rows(d.get("boundaries"), _boundary_from_dict, "boundary"),
            references=_rows(d.get("references"), _reference_from_dict, "reference"),
            persistence=_rows(d.get("persistence"), _persistence_from_dict, "persistence"),
            lifecycle=_rows(d.get("lifecycle"), _lifecycle_from_dict, "lifecycle"),
            authority=_rows(d.get("authority"), _authority_from_dict, "authority"),
            congruences=_rows(d.get("congruences"), _congruence_from_dict, "congruence"),
            bindings=_rows(d.get("bindings"), _binding_from_dict, "binding"),
        )

    # --- the gate the design's own specs power ----------------------------
    def impacts(self, mods: list, root=None) -> dict[str, list]:
        """The complete fan-out of every declared surface element. ``references``
        scan files under ``root`` (skipped when no root is given — there is
        nothing to scan); everything else is derived from the parsed code."""
        out: dict[str, list] = {
            "values": [impact_value(mods, s) for s in self.values],
            "identities": [impact_identity(mods, s) for s in self.identities],
            "boundaries": [impact_boundary(mods, s) for s in self.boundaries],
            "persistence": [impact_persistence(mods, s) for s in self.persistence],
            "lifecycle": [impact_lifecycle(mods, s) for s in self.lifecycle],
            "authority": [impact_authority(mods, s) for s in self.authority],
            "congruences": [impact_congruence(mods, s) for s in self.congruences],
            "references": [],
            "bindings": [],
        }
        if root is not None:
            out["references"] = [impact_reference(root, s) for s in self.references]
            out["bindings"] = [impact_binding(root, s) for s in self.bindings]
        return out

    def gate(self, mods: list, root=None) -> list[Finding]:
        """Every unsatisfied touchpoint across the whole manifest — the gaps
        slice of the impacts above. Empty == complete. Pass ``root`` to also gate
        the file-substrate references (config/docs/schema presence)."""
        out: list[Finding] = []
        imps = self.impacts(mods, root=root)
        for group in imps.values():
            for imp in group:
                out.extend(imp.gaps())
        return out

    def format(self, mods: list, root=None) -> str:
        """Human/LLM-readable: the complete surface fan-out, section by section.
        This is the artifact handed to the design step so its prose addresses
        every existing site — and to a reviewer so 'what else must change' is
        answered, not guessed."""
        imps = self.impacts(mods, root=root)
        blocks: list[str] = []
        for imp in imps["values"]:
            blocks.append(format_value_impact(imp))
        for imp in imps["identities"]:
            blocks.append(format_identity_impact(imp))
        for imp in imps["boundaries"]:
            blocks.append(format_boundary_impact(imp))
        for imp in imps["references"]:
            blocks.append(format_reference_impact(imp))
        for imp in imps["persistence"]:
            blocks.append(format_persistence_impact(imp))
        for imp in imps["lifecycle"]:
            blocks.append(format_lifecycle_impact(imp))
        for imp in imps["authority"]:
            blocks.append(format_authority_impact(imp))
        for imp in imps["congruences"]:
            blocks.append(format_congruence_impact(imp))
        for imp in imps["bindings"]:
            blocks.append(format_binding_impact(imp))
        return "\n\n".join(blocks) if blocks else "(empty manifest)"
