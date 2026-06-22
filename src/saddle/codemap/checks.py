"""The completeness checks the gate runs against the code-derived map.

There are five, and this list is the source of truth for how many — kept honest
by ``tests/test_codemap_coverage.py``, which fails the moment a check, spec,
impact, exporter, or manifest field for a kind goes missing (so this docstring
can't quietly claim a stale count the way RayXI's maps drifted from the code).

Each check answers a question RayXI's declared maps could not, because each is
grounded in the AST or the real file substrate (actual reads/writes/calls/
registrations), not a declaration that the code can silently diverge from:

  value_propagation  — a VALUE that varies (a talent/buff/modifier can change it)
                       must be read through ONE accessor. Every base-field read
                       NOT covered by an accessor call in scope is a site a
                       modifier won't reach. This is the talent-cooldown bug:
                       the menu, the skill description, the hotbar sweep, and the
                       combat engine each read the base number, so spending a
                       point changed nothing they showed or enforced.

  identity_membership — an IDENTITY (an enum member / type name / key) must live
                       in ONE canonical set and never be spelled out of band.
                       Catches BOTH a duplicated canonical set (the drift enabler)
                       AND a literal used as the identity that isn't a member
                       (the typo/drift itself) — including in an `if x == "freeze"`
                       form, which RayXI's match-only scan never saw. This is why
                       "lock all enums" kept missing: the lock was a declaration,
                       not an enforced membership over actual code.

  boundary_mirror    — a value written on the SERVER (authoritative) must be
                       packed into the replication snapshot AND read by client
                       code, or the change is real on the server and invisible
                       on screen. This is the server/client split RayXI had no
                       axis for.

  reference_presence — a name defined in code must also be REGISTERED in the
                       non-code substrates that declare it (a config file, the
                       docs, a DB schema): the cooldown that lived in the engine
                       but never in the skill DESCRIPTION. Off the AST, so it
                       lives in substrate.py — but projects exactly like the rest.

  persistence_symmetry — a value that must survive a session has to be referenced
                       in BOTH the save and the load function, or it silently
                       resets / restores garbage. Code-derived (reads ∪ writes of
                       the key per function) but a save-file question, so it too
                       lives in substrate.py.

Each check is a THIN PROJECTION of its impact set: ``check_value`` is exactly
``impact_value(...).gaps()`` and likewise for every other kind. The dataflow
fan-out lives in impact.py and the two substrate fan-outs in substrate.py; a
check returns only its unsatisfied slice. Routing the gate and the map through
ONE derivation is the structural fix for RayXI's fatal split — a dataflow map
nobody imported alongside gates that only checked declarations. Here the gate
cannot disagree with the map, because it IS the map.
"""
from __future__ import annotations

from .finding import Finding
from .impact import impact_boundary, impact_identity, impact_value
from .specs import (
    BoundarySpec,
    IdentitySpec,
    PersistenceSpec,
    ReferenceSpec,
    ValueSpec,
)
from .substrate import impact_persistence, impact_reference

__all__ = [
    "ValueSpec",
    "IdentitySpec",
    "BoundarySpec",
    "ReferenceSpec",
    "PersistenceSpec",
    "check_value",
    "check_identity",
    "check_boundary",
    "check_reference",
    "check_persistence",
]


def check_value(mods: list, spec: ValueSpec) -> list[Finding]:
    return impact_value(mods, spec).gaps()


def check_identity(mods: list, spec: IdentitySpec) -> list[Finding]:
    return impact_identity(mods, spec).gaps()


def check_boundary(mods: list, spec: BoundarySpec) -> list[Finding]:
    return impact_boundary(mods, spec).gaps()


def check_reference(root, spec: ReferenceSpec) -> list[Finding]:
    return impact_reference(root, spec).gaps()


def check_persistence(mods: list, spec: PersistenceSpec) -> list[Finding]:
    return impact_persistence(mods, spec).gaps()
