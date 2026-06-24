"""saddle audit surface — grounded probes that EMIT findings, never write code.

The complement to Layer 4's converge loop. Converge proves a model can be MADE to
implement a design without flattening it; audit proves saddle can ENUMERATE
everything that can be checked in a target project and verify each one — closing
the "I never come back / I never know what's left" gap with a coverage ledger.

  plan.build_plan(root)        -> AuditPlan        (everything that CAN be audited)
  run.run_audit(plan, ...)     -> AuditReport      (findings + coverage, never hangs)

Each target is grounded (ground.py) in real files + the symbol menu + registry
dataflow, probed once (probe.py), and individually bounded (supervise) so no probe
can wedge the run.
"""
from __future__ import annotations

from saddle.audit.finding import (
    AuditFinding,
    ERROR,
    INFO,
    WARN,
    sort_findings,
)
from saddle.audit.ground import Grounding, ground_target
from saddle.audit.plan import AuditPlan, AuditTarget, build_plan
from saddle.audit.probe import audit_target
from saddle.audit.run import AuditReport, Coverage, format_report, run_audit

__all__ = [
    "AuditFinding",
    "ERROR",
    "WARN",
    "INFO",
    "sort_findings",
    "Grounding",
    "ground_target",
    "AuditPlan",
    "AuditTarget",
    "build_plan",
    "audit_target",
    "AuditReport",
    "Coverage",
    "run_audit",
    "format_report",
]
