"""saddle Layer 3 — the per-(tenant,project) code-derived completeness map.

WHY THIS EXISTS
---------------
Layer 1 itemizes a prompt; Layer 2 designs against the Design KB. Neither knows
the target project's CODE, so neither can answer "if I add/remove this feature,
what else must change?" That gap shipped the talent-points bug: a new spendable
cooldown modifier whose effect reached nothing — not the points counter, not the
skill description, not the hotbar sweep, not the combat engine — and the
server/client split hid half of it.

RayXI already had interaction maps (provides/reads_from), a value-impact map, a
requirements-traceability matrix, and an enum-drift check. The bug shipped
anyway because every one of those verifies a DECLARATION or an ARTIFACT'S
PRESENCE, not the code's actual dataflow — and the one map that did model
dataflow (value_impact_map) was imported by nobody, so it never ran at a gate.

This layer is the corrected version: a map ALWAYS derived from the AST, an
IMPACT SET that lists every site a thing touches (impact.py), checks that verify
EFFECT (a change reaches every consumer) by PROJECTING that impact set rather
than recomputing it, and a gate that is actually wired to run.
"""
from __future__ import annotations

from . import gdref, pyref, refs
from .checks import (
    check_boundary,
    check_identity,
    check_persistence,
    check_reference,
    check_value,
)
from .finding import Finding
from .gate import run_checks, run_paths
from .impact import (
    BoundaryImpact,
    IdentityImpact,
    ValueImpact,
    format_boundary_impact,
    format_identity_impact,
    format_value_impact,
    impact_boundary,
    impact_identity,
    impact_value,
)
from .manifest import SurfaceManifest
from .specs import (
    BoundarySpec,
    IdentitySpec,
    PersistenceSpec,
    ReferenceSpec,
    ValueSpec,
)
from .substrate import (
    PersistenceImpact,
    ReferenceImpact,
    format_persistence_impact,
    format_reference_impact,
    impact_persistence,
    impact_reference,
)

__all__ = [
    "pyref",
    "gdref",
    "refs",
    "Finding",
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
    "ValueImpact",
    "IdentityImpact",
    "BoundaryImpact",
    "ReferenceImpact",
    "PersistenceImpact",
    "impact_value",
    "impact_identity",
    "impact_boundary",
    "impact_reference",
    "impact_persistence",
    "format_value_impact",
    "format_identity_impact",
    "format_boundary_impact",
    "format_reference_impact",
    "format_persistence_impact",
    "SurfaceManifest",
    "run_checks",
    "run_paths",
]
