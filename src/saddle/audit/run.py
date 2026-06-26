"""The audit DRIVER — work the whole plan, never hang, prove coverage.

Ties the surface together: parse the target code ONCE, then for every planned
target ground it (deterministic) and run one probe (LLM), each probe individually
bounded by :mod:`saddle.supervise` so a wedged or slow probe fails that ONE target
and the run keeps going — the same liveness discipline that killed the converge
hang, applied per-probe. The outcome is an :class:`AuditReport` carrying every
finding AND the coverage ledger: which targets ran, which were skipped or failed,
and which are still pending — so "did we audit everything that can be audited?" is
a fact, not a hope.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from saddle import supervise
from saddle.audit.congruence import deterministic_congruence_findings
from saddle.audit.finding import AuditFinding, ERROR, WARN, sort_findings
from saddle.audit.ground import (
    ground_target, _iter_code_text, source_files, parse_sources, _TEXT_SCAN_EXTS,
    data_field_counts,
)
from saddle.audit.plan import AuditPlan
from saddle.audit.probe import audit_target

if TYPE_CHECKING:  # pragma: no cover
    from saddle.audit.plan import AuditTarget
    from saddle.context import Context
    from saddle.llm.protocol import LLMCaller

_log = logging.getLogger("saddle.audit.run")

# A generous safety net only — with the parse SCOPED to the project's real source
# dirs (not its generated build output) a whole-project parse is a second or two;
# this deadline exists so a pathological tree can never wedge the audit before a
# single probe runs — the liveness rule applied to the setup phase, not just probes.
_PARSE_DEADLINE_S = 180.0


@dataclass
class Coverage:
    """The completeness ledger — every planned target's disposition. A run is only
    'done' when no target is still pending (each ran, was deliberately skipped, or
    failed loud)."""

    planned: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)   # id -> why
    failed: dict[str, str] = field(default_factory=dict)    # id -> error

    @property
    def accounted(self) -> set[str]:
        return set(self.ran) | set(self.skipped) | set(self.failed)

    @property
    def pending(self) -> list[str]:
        acc = self.accounted
        return [t for t in self.planned if t not in acc]

    @property
    def complete(self) -> bool:
        """Every planned target has a verdict (ran/skipped/failed) — nothing left."""
        return not self.pending

    def to_dict(self) -> dict:
        return {
            "planned": list(self.planned), "ran": list(self.ran),
            "skipped": dict(self.skipped), "failed": dict(self.failed),
            "pending": self.pending, "complete": self.complete,
        }


@dataclass
class AuditReport:
    """The outcome of working an audit plan: all findings + the coverage ledger."""

    root: str
    findings: list[AuditFinding] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    started: float = 0.0
    finished: float = 0.0

    def errors(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == ERROR]

    def ungrounded(self) -> list[AuditFinding]:
        return [f for f in self.findings if not f.grounded]

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "started": self.started, "finished": self.finished,
            "counts": {
                "findings": len(self.findings),
                "errors": len(self.errors()),
                "warns": len([f for f in self.findings if f.severity == WARN]),
                "ungrounded": len(self.ungrounded()),
            },
            "coverage": self.coverage.to_dict(),
            "findings": [f.to_dict() for f in sort_findings(self.findings)],
        }


def _select(plan: AuditPlan, kinds: list[str] | None, only: list[str] | None) -> list["AuditTarget"]:
    out = plan.targets
    if kinds:
        kset = set(kinds)
        out = [t for t in out if t.kind in kset]
    if only:
        oset = set(only)
        out = [t for t in out if t.id in oset]
    return out


async def run_audit(
    plan: AuditPlan,
    *,
    caller: "LLMCaller | None" = None,
    ctx: "Context | None" = None,
    kinds: list[str] | None = None,
    only: list[str] | None = None,
    concurrency: int = 3,
    per_target_deadline_s: float = 300.0,
    persist: bool = True,
    out_dir: str | Path | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> AuditReport:
    """Work ``plan`` to completion and return the :class:`AuditReport`.

    ``caller`` defaults to the project's provider chain (so probes run through
    minimax/claude/etc.). Probes run with bounded ``concurrency``; each is wrapped
    in :func:`saddle.supervise.bounded` at ``per_target_deadline_s`` so no single
    probe can wedge the run — it is recorded as a failed target and the rest
    proceed. Findings + coverage persist to ``out_dir`` unless told otherwise.
    """
    root = Path(plan.root).expanduser().resolve()
    if ctx is None:
        from saddle.context import default as _default_ctx
        ctx = _default_ctx()
    if caller is None:
        from saddle.llm.callers import build_callers
        caller = build_callers(ctx)["default"]

    targets = _select(plan, kinds, only)
    report = AuditReport(root=str(root), started=time.time())
    report.coverage.planned = [t.id for t in targets]

    def emit(kind: str, **data) -> None:
        if on_event is not None:
            try:
                on_event({"event": kind, **data})
            except Exception:  # noqa: BLE001 — telemetry must never break the run
                pass

    emit("plan", targets=len(targets), root=str(root))
    if not targets:
        report.finished = time.time()
        return report

    # Parse + read the code ONCE; every target's grounding reuses it. SCOPED to the
    # project's real source dirs (plan.code_dirs) — never the whole repo, which on a
    # generator pipeline is mostly build OUTPUT and would be slow + memory-heavy +
    # would pollute the symbol menu. Bounded so even a pathological tree can't wedge
    # the audit before a probe runs.
    code_dirs = plan.code_dirs or None
    emit("parse", phase="start", code_dirs=plan.code_dirs)
    try:
        mods = await supervise.bounded(
            asyncio.to_thread(_safe_parse, root, code_dirs),
            seconds=_PARSE_DEADLINE_S, what="audit source parse",
        )
        # Only pay for the raw text scan if some target actually needs it (registry/seed).
        need_text = any(t.kind == "registry" or t.seeds for t in targets)
        if need_text:
            code_files = await supervise.bounded(
                asyncio.to_thread(source_files, root, code_dirs, exts=_TEXT_SCAN_EXTS),
                seconds=_PARSE_DEADLINE_S, what="audit source enumerate",
            )
            code_text = await supervise.bounded(
                asyncio.to_thread(_iter_code_text, code_files),
                seconds=_PARSE_DEADLINE_S, what="audit source read",
            )
        else:
            code_files, code_text = [], []
        # Data-field menu — built ONCE from the plan's data files so doc/package
        # probes can check a doc-claimed contract field against the DATA it lives in,
        # not only the code symbol menu (the false-`absent` class). Bounded like the
        # parse; a failure here degrades to "no menu", never wedges the run.
        try:
            data_counts = await supervise.bounded(
                asyncio.to_thread(data_field_counts, root, list(plan.data_files)),
                seconds=_PARSE_DEADLINE_S, what="audit data-field index",
            )
        except supervise.DeadlineExceeded:
            data_counts = {}
    except supervise.DeadlineExceeded as exc:
        # Setup itself wedged — fail the whole plan loud (every target pending->failed)
        # rather than hang. This is the liveness contract honored even before probes.
        for t in targets:
            report.coverage.failed[t.id] = f"setup wedged: {exc}"
        report.finished = time.time()
        emit("parse", phase="failed", reason=str(exc)[:120])
        if persist:
            _persist_report(report, plan, out_dir, root)
        return report
    emit("parse", phase="done", modules=len(mods), files=len(code_files))

    # Deterministic congruence pre-pass — the codemap's 9th check, auto-derived from
    # the parsed code (no hand-authored spec). Emits first-class, citation-backed
    # findings so the bug class RayXI shipped four times CANNOT silently regrow; the
    # LLM concern probe below stays as the open-ended complement (it catches shapes
    # off this axis — a server write never replicated at all). Bounded + best-effort:
    # a failure here degrades to "no deterministic findings", never wedges the run.
    try:
        cong = await supervise.bounded(
            asyncio.to_thread(deterministic_congruence_findings, mods),
            seconds=_PARSE_DEADLINE_S, what="audit congruence pre-pass",
        )
        report.findings.extend(cong)
        emit("congruence", phase="done", findings=len(cong))
    except Exception as exc:  # noqa: BLE001 — the pre-pass must never break the run
        _log.warning("audit: deterministic congruence pre-pass failed: %s", exc)
        emit("congruence", phase="failed", reason=str(exc)[:120])

    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0
    total = len(targets)
    lock = asyncio.Lock()

    async def _one(target: "AuditTarget") -> None:
        nonlocal done
        async with sem:
            try:
                grounding = await asyncio.to_thread(
                    ground_target, target, root,
                    mods=mods, code_files=code_files, code_text=code_text,
                    code_dirs=code_dirs, data_counts=data_counts,
                )
                if grounding.is_empty():
                    async with lock:
                        report.coverage.skipped[target.id] = "no grounding available"
                    emit("target", id=target.id, status="skipped", reason="no grounding")
                    return
                findings = await supervise.bounded(
                    audit_target(caller, target, grounding, label=f"audit/{target.kind}"),
                    seconds=per_target_deadline_s,
                    what=f"audit probe {target.id}",
                )
                async with lock:
                    report.findings.extend(findings)
                    report.coverage.ran.append(target.id)
                emit("target", id=target.id, status="ran", findings=len(findings))
            except supervise.DeadlineExceeded as exc:
                async with lock:
                    report.coverage.failed[target.id] = f"probe deadline: {exc}"
                emit("target", id=target.id, status="failed", reason="deadline")
            except Exception as exc:  # noqa: BLE001 — one probe failing never aborts the audit
                async with lock:
                    report.coverage.failed[target.id] = str(exc)[:200]
                emit("target", id=target.id, status="failed", reason=str(exc)[:120])
                _log.warning("audit: target %s failed: %s", target.id, str(exc)[:160])
            finally:
                async with lock:
                    done += 1
                    n = done
                emit("progress", done=n, total=total)

    await asyncio.gather(*(_one(t) for t in targets))
    report.finished = time.time()

    if persist:
        _persist_report(report, plan, out_dir, root)
    _log.info(
        "audit: %d/%d targets ran, %d findings (%d error), %d failed, %d pending",
        len(report.coverage.ran), total, len(report.findings),
        len(report.errors()), len(report.coverage.failed), len(report.coverage.pending),
    )
    return report


def _safe_parse(root: Path, code_dirs: list[str] | None) -> list:
    try:
        return parse_sources(root, code_dirs)
    except Exception:  # noqa: BLE001 — code grounding is best-effort
        _log.warning("audit: could not parse code at %s", root, exc_info=True)
        return []


def _persist_report(report: AuditReport, plan: AuditPlan, out_dir, root: Path) -> None:
    base = Path(out_dir).expanduser() if out_dir else (
        root / ".saddle" / "audit" / time.strftime("%Y%m%d_%H%M%S", time.localtime(report.started))
    )
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        (base / "findings.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        (base / "report.md").write_text(format_report(report), encoding="utf-8")
        _log.info("audit: report written to %s", base)
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort
        _log.warning("audit: could not persist report: %s", exc)


def format_report(report: AuditReport) -> str:
    cov = report.coverage
    lines = [
        f"# saddle audit — {report.root}",
        "",
        f"- targets planned: {len(cov.planned)}",
        f"- ran: {len(cov.ran)}   skipped: {len(cov.skipped)}   failed: {len(cov.failed)}   pending: {len(cov.pending)}",
        f"- coverage complete: {cov.complete}",
        f"- findings: {len(report.findings)}  ({len(report.errors())} error, "
        f"{len([f for f in report.findings if f.severity == WARN])} warn, "
        f"{len(report.ungrounded())} ungrounded)",
        "",
    ]
    if cov.failed:
        lines.append("## failed targets (probe could not complete)")
        lines.extend(f"- {tid}: {why}" for tid, why in cov.failed.items())
        lines.append("")
    if cov.pending:
        lines.append("## pending targets (never reached — coverage INCOMPLETE)")
        lines.extend(f"- {tid}" for tid in cov.pending)
        lines.append("")
    lines.append("## findings (most actionable first)")
    if not report.findings:
        lines.append("(none)")
    for f in sort_findings(report.findings):
        lines.append(f"\n### {f}")
        if f.detail and f.detail != f.title:
            lines.append(f"{f.detail}")
        if f.suggestion:
            lines.append(f"_fix:_ {f.suggestion}")
        lines.append(f"_confidence:_ {f.confidence}")
    return "\n".join(lines)
