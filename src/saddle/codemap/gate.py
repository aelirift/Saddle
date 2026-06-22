"""The change-completeness gate: run the registered specs over a parsed code
map and return every unsatisfied touchpoint.

This is the WIRED entry point — the thing RayXI's value_impact_map never had.
RayXI built an equivalent map and then imported it from nobody, so it never ran
at a gate. Here, `run_paths` is what a commit hook / CI step / the design step's
post-emit verifier calls; an empty result is the only clean state.
"""
from __future__ import annotations

from collections.abc import Iterable

from . import refs
from .checks import (
    BoundarySpec,
    IdentitySpec,
    PersistenceSpec,
    ReferenceSpec,
    ValueSpec,
    check_boundary,
    check_identity,
    check_persistence,
    check_reference,
    check_value,
)
from .finding import Finding


def run_checks(
    mods: list,
    *,
    values: Iterable[ValueSpec] = (),
    identities: Iterable[IdentitySpec] = (),
    boundaries: Iterable[BoundarySpec] = (),
    persistence: Iterable[PersistenceSpec] = (),
    references: Iterable[ReferenceSpec] = (),
    root=None,
) -> list[Finding]:
    findings: list[Finding] = []
    for v in values:
        findings.extend(check_value(mods, v))
    for i in identities:
        findings.extend(check_identity(mods, i))
    for b in boundaries:
        findings.extend(check_boundary(mods, b))
    for p in persistence:
        findings.extend(check_persistence(mods, p))
    if root is not None:
        for r in references:
            findings.extend(check_reference(root, r))
    return findings


def run_paths(paths, **specs) -> list[Finding]:
    """Convenience: parse a mixed list of files (Python + GDScript, routed by
    extension), then run the checks. ``references`` need a ``root=`` to scan."""
    return run_checks(refs.parse_paths(paths), **specs)
