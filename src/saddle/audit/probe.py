"""The audit PROBE — one grounded LLM pass that emits findings, never code.

This is the inversion of Layer 4's converge loop: converge DRIVES a coder to
WRITE code until a gate is clean; the probe READS grounded evidence and EMITS
what is missing/wrong/half-wired as structured findings. No edits, no tools, no
coder — just judgement over real evidence.

GROUNDED-OR-NOTHING
-------------------
The probe is handed real files, the symbol menu, registry dataflow, and seed
matches (see ground.py). Its one hard rule is that every finding must CITE that
evidence — a ``path:line`` or ``registry:key`` a reader can open. A finding with
no citation is the exact failure an audit exists to avoid (a confident-sounding
hallucination), so the contract demands evidence and the report flags any finding
that arrives without it. One call returns ONE artifact: the findings for ONE
target (the per-call size contract holds — it is one target's verdict, not a
bundle of unrelated outputs).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from saddle.audit.finding import AuditFinding
from saddle.llm.json_tools import call_json

if TYPE_CHECKING:  # pragma: no cover
    from saddle.audit.ground import Grounding
    from saddle.audit.plan import AuditTarget
    from saddle.llm.protocol import LLMCaller

_log = logging.getLogger("saddle.audit.probe")

_SYS_PROBE = (
    "You are saddle's audit probe — a ruthless senior engineer reviewing a real "
    "codebase for what is MISSING, WRONG, or HALF-WIRED. You are given an AUDIT "
    "QUESTION and GROUNDING: real file contents, the project's symbol menu, and "
    "(when relevant) a registry's key->code-site dataflow and concern seed "
    "matches. You do not write code; you report gaps.\n"
    "HARD RULES:\n"
    "- Ground EVERY finding. Each must cite specific evidence from the grounding — "
    "a `path:line`, a `registry:key`, or a symbol name that appears above. A "
    "finding you cannot cite is a guess; do not emit it.\n"
    "- Report only REAL gaps: a declared thing with no implementation, an "
    "implemented thing nothing consumes, a server/client mismatch, code that "
    "contradicts the registry/doc, a panel that renders but cannot be used. Do NOT "
    "invent issues, restyle working code, or flag mere taste. If the grounding "
    "shows the thing is fine, say nothing.\n"
    "- Be specific and actionable: name WHAT is wrong, WHERE (the evidence), and a "
    "concrete fix direction.\n"
    "- Grade honestly: severity error|warn|info, and confidence high|medium|low. "
    "Low confidence is fine and useful — just mark it.\n"
    "For each finding emit: severity (error|warn|info), kind (missing_impl|"
    "dead_decl|congruence|contract_drift|orphan|ux|inconsistency, or a sharper one "
    "you name), title (one line), detail (the full explanation), evidence (a list "
    "of path:line / registry:key citations), suggestion (the fix direction), "
    "confidence (high|medium|low).\n"
    "Respond with ONLY JSON: "
    '{"findings": [{"severity": "...", "kind": "...", "title": "...", "detail": '
    '"...", "evidence": ["path:line"], "suggestion": "...", "confidence": "..."}]}. '
    "Return an empty list if the grounding reveals no real gap."
)


def probe_prompt(target: "AuditTarget", grounding_text: str) -> str:
    return (
        f"AUDIT TARGET: {target.title}  (kind: {target.kind})\n\n"
        f"AUDIT QUESTION:\n{target.question}\n\n"
        f"GROUNDING (real evidence — cite from here):\n{grounding_text}\n\n"
        "Report every real, grounded gap as findings. Empty list if there are none."
    )


async def audit_target(
    caller: "LLMCaller",
    target: "AuditTarget",
    grounding: "Grounding",
    *,
    label: str = "",
) -> list[AuditFinding]:
    """Run one grounded probe over ``target`` and return its findings.

    Malformed rows are dropped loudly (one bad finding never voids the batch); the
    LLM contract is saddle's own, so a wholesale-malformed reply surfaces via
    ``call_json``. Findings are stamped with ``target.id`` so the report reconciles
    them against coverage.
    """
    if grounding.is_empty():
        _log.warning("audit: target %s has no grounding — skipping probe", target.id)
        return []
    payload = await call_json(
        caller, _SYS_PROBE, probe_prompt(target, grounding.format()),
        label=label or f"audit/{target.kind}",
    )
    raw = payload.get("findings")
    if not isinstance(raw, list):
        _log.warning("audit: target %s — reply had no findings list", target.id)
        return []
    findings: list[AuditFinding] = []
    dropped = 0
    for row in raw:
        f = AuditFinding.from_dict(row, target=target.id)
        if f is None:
            dropped += 1
            continue
        findings.append(f)
    if dropped:
        _log.warning("audit: target %s — dropped %d malformed finding row(s)", target.id, dropped)
    return findings
