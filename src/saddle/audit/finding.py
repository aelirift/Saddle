"""The audit surface's output types — open-ended findings, not gate findings.

WHY A SEPARATE FINDING TYPE
---------------------------
``saddle.codemap.Finding`` is what the Layer-3 GATE raises: a DETERMINISTIC,
pre-declared completeness violation (a value not propagated, an identity drifted,
a boundary not mirrored). It answers a closed question — "does the code satisfy
THIS declared surface?" — with no LLM in the loop.

An AUDIT asks an OPEN question — "what is missing / wrong / half-wired in this
subsystem?" — answered by an LLM grounded in the real code, registries, and docs.
Its output is judgement, so each finding carries the things that make a judgement
trustworthy and actionable: graded ``severity``, a gap ``kind``, the ``evidence``
that grounds it (a finding with no citation is suspect by construction), a
concrete ``suggestion``, and the model's own ``confidence`` so a low-confidence
guess never reads as a hard defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- severity: how much this finding should block ------------------------
ERROR = "error"    # a real defect: missing impl, congruence break, dead contract
WARN = "warn"      # a likely problem worth a look — not proven broken
INFO = "info"      # an observation / suggestion, not a defect
SEVERITIES: frozenset[str] = frozenset({ERROR, WARN, INFO})
_SEV_RANK = {ERROR: 0, WARN: 1, INFO: 2}

# --- kind: the SHAPE of the gap (a small, documented, open vocabulary) ----
# These are guidance for the probe + grouping for the report, NOT a closed enum:
# an audit that invents a sharper kind for a real gap is doing its job.
MISSING_IMPL = "missing_impl"        # declared (registry/doc/signature) but no code implements it
DEAD_DECL = "dead_decl"              # implemented/declared but nothing consumes it (dead knob)
CONGRUENCE = "congruence"            # server and client disagree (state written one side, never mirrored)
CONTRACT_DRIFT = "contract_drift"    # code disagrees with the registry/doc that declares it
ORPHAN = "orphan"                    # exists but is unreferenced — scaffolding / facade
UX = "ux"                            # user-facing defect (transparent panel, unclickable button, dead key)
INCONSISTENCY = "inconsistency"      # two sites that must agree, don't
KINDS: frozenset[str] = frozenset({
    MISSING_IMPL, DEAD_DECL, CONGRUENCE, CONTRACT_DRIFT, ORPHAN, UX, INCONSISTENCY,
})

CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})


@dataclass(frozen=True)
class AuditFinding:
    """One thing an audit probe found missing, wrong, or half-wired.

    ``target`` ties it to the :class:`~saddle.audit.plan.AuditTarget` whose probe
    raised it (so coverage and findings reconcile). ``evidence`` is the load-
    bearing field — ``path:line`` or ``registry:key`` citations grounding the
    claim in something a reader can open; an empty list is a smell the report
    surfaces, not a silent pass.
    """

    target: str            # the audit target id this finding belongs to
    severity: str          # error | warn | info
    kind: str              # missing_impl | dead_decl | congruence | ... (open vocab)
    title: str             # one-line label
    detail: str            # what is missing / wrong, in full
    evidence: list[str] = field(default_factory=list)   # "path:line" / "registry:key" citations
    suggestion: str = ""   # concrete direction for the fix
    confidence: str = "medium"

    def __str__(self) -> str:
        where = f"  ({'; '.join(self.evidence)})" if self.evidence else "  (UNGROUNDED)"
        return f"[{self.severity}/{self.kind}] {self.target} :: {self.title}{where}"

    @property
    def grounded(self) -> bool:
        """A finding a reader can verify — it cites at least one real site."""
        return bool(self.evidence)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "severity": self.severity,
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "evidence": list(self.evidence),
            "suggestion": self.suggestion,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict, *, target: str = "") -> "AuditFinding | None":
        """Rebuild a finding from one LLM/JSON row, normalizing loose fields.

        Returns ``None`` for an unsalvageable row (no title AND no detail) so one
        malformed entry is dropped loudly by the caller, never aborting the batch.
        """
        if not isinstance(d, dict):
            return None
        title = str(d.get("title", "")).strip()
        detail = str(d.get("detail", "")).strip()
        if not title and not detail:
            return None
        sev = str(d.get("severity", "")).strip().lower()
        if sev not in SEVERITIES:
            sev = ERROR if sev in ("critical", "high", "blocker") else WARN
        kind = str(d.get("kind", "")).strip().lower() or INCONSISTENCY
        conf = str(d.get("confidence", "")).strip().lower()
        if conf not in CONFIDENCES:
            conf = "medium"
        ev_raw = d.get("evidence", [])
        if isinstance(ev_raw, str):
            ev_raw = [ev_raw]
        evidence = [str(e).strip() for e in (ev_raw or []) if str(e).strip()]
        return cls(
            target=str(d.get("target", "") or target).strip() or target,
            severity=sev, kind=kind,
            title=title or detail[:80],
            detail=detail or title,
            evidence=evidence,
            suggestion=str(d.get("suggestion", "")).strip(),
            confidence=conf,
        )


def sort_findings(findings: list[AuditFinding]) -> list[AuditFinding]:
    """Most-actionable first: severity, then grounded before ungrounded, then target."""
    return sorted(
        findings,
        key=lambda f: (_SEV_RANK.get(f.severity, 9), not f.grounded, f.target, f.title),
    )
