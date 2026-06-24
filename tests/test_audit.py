"""Tests for saddle's audit surface: the finding model, the coverage plan, the
deterministic grounding (incl. string-keyed registry dataflow), the LLM probe
(with a fake caller), and the driver's coverage ledger + per-probe liveness."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from saddle.audit.finding import AuditFinding, sort_findings
from saddle.audit.ground import (
    ground_target, registry_keys, scan_key_sites, scan_seeds, _iter_code_text,
)
from saddle.audit.plan import (
    AuditTarget, build_plan, REGISTRY, DOC, PACKAGE, CONCERN,
)
from saddle.audit.probe import audit_target
from saddle.audit.run import run_audit, Coverage


# --- finding model -------------------------------------------------------

def test_finding_from_dict_normalizes_and_flags_grounding():
    f = AuditFinding.from_dict({
        "severity": "blocker", "kind": "missing_impl", "title": "x not wired",
        "detail": "the key is declared but no code reads it",
        "evidence": ["config/a.json:registry", "src/m.py:10"], "confidence": "high",
    })
    assert f is not None
    assert f.severity == "error"          # "blocker" -> error
    assert f.grounded is True
    assert f.confidence == "high"

    bare = AuditFinding.from_dict({"title": "guess", "detail": "no cite"})
    assert bare is not None and bare.grounded is False  # ungrounded but kept + flagged

    assert AuditFinding.from_dict({"evidence": ["x"]}) is None  # no title AND no detail -> dropped
    assert AuditFinding.from_dict("not a dict") is None


def test_sort_findings_orders_by_severity_then_grounded():
    a = AuditFinding("t", "warn", "ux", "w", "d", evidence=["x:1"])
    b = AuditFinding("t", "error", "missing_impl", "e", "d", evidence=["y:2"])
    c = AuditFinding("t", "error", "orphan", "e2", "d")  # ungrounded error
    out = sort_findings([a, b, c])
    assert out[0] is b          # grounded error first
    assert out[1] is c          # ungrounded error next
    assert out[2] is a          # warn last


# --- plan / coverage enumeration -----------------------------------------

def _make_project(tmp: Path) -> Path:
    (tmp / "README.md").write_text("# Title\nthe core does X\n")
    (tmp / "docs").mkdir()
    (tmp / "docs" / "guide.md").write_text("the guide\n")
    (tmp / "config").mkdir()
    (tmp / "config" / "items.json").write_text(json.dumps({
        "sword": {"dmg": 5}, "shield": {"block": 3},
    }))
    (tmp / "config" / "scalar.json").write_text(json.dumps(42))  # not a registry
    pkg = tmp / "src" / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text(
        "def use(item):\n"
        "    if item == 'sword':\n"        # references the 'sword' key
        "        return 1\n"
        "    return 0\n"
    )
    return tmp


def test_build_plan_enumerates_docs_registries_packages_concerns(tmp_path):
    root = _make_project(tmp_path)
    plan = build_plan(root)
    ids = plan.ids()
    assert "doc:README.md" in ids
    assert "doc:docs/guide.md" in ids
    assert "registry:config/items.json" in ids
    assert "registry:config/scalar.json" not in ids       # bare scalar is not a contract
    assert "package:src/mypkg" in ids
    assert {t.id for t in plan.by_kind(CONCERN)}           # concerns present
    # round-trips through dict
    assert plan.to_dict()["root"] == str(root.resolve())
    from saddle.audit.plan import AuditPlan
    assert AuditPlan.from_dict(plan.to_dict()).ids() == ids


# --- grounding: string-keyed registry dataflow ---------------------------

def test_registry_keys_extracts_object_keys():
    keys = registry_keys({"sword": {"dmg": 5}, "shield": {"block": 3}})
    assert "sword" in keys and "shield" in keys


def test_scan_key_sites_finds_consumers_and_dead_contracts(tmp_path):
    root = _make_project(tmp_path)
    code_files = [str(root / "src" / "mypkg" / "core.py")]
    code_text = _iter_code_text(code_files)
    hits = scan_key_sites(root, ["sword", "shield"], code_text)
    assert hits["sword"], "the consumed key must cite a code site"
    assert hits["sword"][0].startswith("src/mypkg/core.py:")
    assert hits["shield"] == [], "an unreferenced key is a dead contract (empty)"


def test_scan_seeds_cites_matching_lines(tmp_path):
    root = _make_project(tmp_path)
    code_text = _iter_code_text([str(root / "src" / "mypkg" / "core.py")])
    hits = scan_seeds(root, [r"def use"], code_text)
    assert hits[r"def use"] and "core.py:1" in hits[r"def use"][0]


def test_ground_target_registry_assembles_files_and_dataflow(tmp_path):
    root = _make_project(tmp_path)
    target = AuditTarget(
        id="registry:config/items.json", kind=REGISTRY, title="items",
        question="audit", paths=["config/items.json"],
    )
    g = ground_target(target, root)
    assert "config/items.json" in g.files          # the registry text is in the bundle
    assert g.key_dataflow.get("sword")             # consumed key has a site
    assert g.key_dataflow.get("shield") == []      # dead contract surfaced
    assert g.symbol_menu                            # symbol menu present
    assert not g.is_empty()


# --- probe (fake caller) -------------------------------------------------

class _FakeCaller:
    """An LLMCaller stand-in that returns a canned JSON string."""

    def __init__(self, reply: str, *, delay: float = 0.0, boom: bool = False):
        self._reply = reply
        self._delay = delay
        self._boom = boom
        self.calls = 0

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._boom:
            raise RuntimeError("provider exploded")
        return self._reply


def _grounding_for(tmp_path):
    root = _make_project(tmp_path)
    target = AuditTarget(
        id="registry:config/items.json", kind=REGISTRY, title="items",
        question="audit the dataflow", paths=["config/items.json"],
    )
    return root, target, ground_target(target, root)


def test_audit_target_parses_and_tags_findings(tmp_path):
    _, target, g = _grounding_for(tmp_path)
    reply = json.dumps({"findings": [
        {"severity": "error", "kind": "dead_decl", "title": "shield unused",
         "detail": "the shield key is declared but no code reads it",
         "evidence": ["config/items.json:shield"], "confidence": "high"},
        {"title": "", "detail": ""},          # malformed -> dropped
    ]})
    caller = _FakeCaller(reply)
    findings = asyncio.run(audit_target(caller, target, g))
    assert len(findings) == 1                  # malformed row dropped
    assert findings[0].target == "registry:config/items.json"
    assert findings[0].kind == "dead_decl"


# --- driver: coverage ledger + per-probe liveness ------------------------

def test_run_audit_builds_coverage_and_collects_findings(tmp_path):
    root = _make_project(tmp_path)
    plan = build_plan(root)
    reply = json.dumps({"findings": [
        {"severity": "warn", "kind": "orphan", "title": "maybe",
         "detail": "something", "evidence": ["README.md:1"]},
    ]})
    caller = _FakeCaller(reply)
    report = asyncio.run(run_audit(
        plan, caller=caller, persist=False, concurrency=4,
    ))
    assert report.coverage.complete, f"pending: {report.coverage.pending}"
    assert set(report.coverage.ran) == plan.ids()      # every target ran
    assert len(report.findings) == len(plan.targets)   # one finding per target
    assert caller.calls == len(plan.targets)


def test_run_audit_bounds_a_hanging_probe_and_keeps_going(tmp_path):
    """The headline liveness guarantee: a probe that never returns fails THAT
    target via the deadline and the run still completes — no hang."""
    root = _make_project(tmp_path)
    plan = build_plan(root)
    caller = _FakeCaller("{}", delay=30.0)             # would hang far past the deadline
    report = asyncio.run(run_audit(
        plan, caller=caller, persist=False, concurrency=4,
        per_target_deadline_s=0.3,
    ))
    # Every target accounted for as FAILED (deadline) — none left pending, no hang.
    assert report.coverage.complete
    assert len(report.coverage.failed) == len(plan.targets)
    assert all("deadline" in why for why in report.coverage.failed.values())


def test_run_audit_isolates_a_crashing_probe(tmp_path):
    root = _make_project(tmp_path)
    plan = build_plan(root)
    caller = _FakeCaller("{}", boom=True)
    report = asyncio.run(run_audit(plan, caller=caller, persist=False))
    assert report.coverage.complete
    assert len(report.coverage.failed) == len(plan.targets)


def test_run_audit_persists_report(tmp_path):
    root = _make_project(tmp_path)
    plan = build_plan(root)
    caller = _FakeCaller(json.dumps({"findings": []}))
    out = tmp_path / "out"
    asyncio.run(run_audit(plan, caller=caller, persist=True, out_dir=out))
    assert (out / "findings.json").is_file()
    assert (out / "plan.json").is_file()
    assert (out / "report.md").is_file()
    data = json.loads((out / "findings.json").read_text())
    assert data["coverage"]["complete"] is True


def test_coverage_pending_tracks_unaccounted():
    cov = Coverage(planned=["a", "b", "c"], ran=["a"])
    cov.skipped["b"] = "no grounding"
    assert cov.pending == ["c"]
    assert cov.complete is False
