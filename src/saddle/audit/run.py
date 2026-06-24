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
from saddle.audit.finding import AuditFinding, ERROR, WARN, sort_findings
from saddle.audit.ground import ground_target, _iter_code_text
from saddle.audit.plan import AuditPlan
from saddle.audit.probe import audit_target
from saddle.codemap import refs

if TYPE_CHECKING:  # pragma: no cover
    from saddle.audit.plan import AuditTarget
    from saddle.context import Context
    from saddle.llm.protocol import LLMCaller

_log = logging.getLogger("saddle.audit.run")


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

    # Parse + read the code ONCE; every target's grounding reuses it.
    emit("parse", phase="start")
    mods = await asyncio.to_thread(_safe_parse, root)
    code_files = await asyncio.to_thread(refs.project_files, root)
    # Only pay for the raw text scan if some target actually needs it (registry/seed).
    need_text = any(t.kind == "registry" or t.seeds for t in targets)
    code_text = await asyncio.to_thread(_iter_code_text, code_files) if need_text else []
    emit("parse", phase="done", modules=len(mods), files=len(code_files))

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


def _safe_parse(root: Path) -> list:
    try:
        return refs.parse_project(root)
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
