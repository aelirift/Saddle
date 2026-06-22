"""A single completeness finding emitted by a codemap check.

The codemap is the per-(tenant,project) Layer-3 artifact that records, for a
canonical *thing* (a VALUE like `cooldown` or an IDENTITY like a status-type
name), every code site that touches it — derived from the AST, never declared.
A Finding is what the gate raises when a site is missing, drifted, or unmirrored.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    check: str        # value_propagation | identity_membership | boundary_mirror
                      #   | reference_presence | persistence_symmetry
    severity: str     # "error" | "warn"
    node_kind: str    # value | identity | boundary | reference | persistence
    thing: str        # the canonical thing, e.g. "cooldown" / "status_type" / "mana"
    message: str
    location: str     # "path:line"
    detail: dict | None = None

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check} :: {self.thing}: {self.message}  ({self.location})"
