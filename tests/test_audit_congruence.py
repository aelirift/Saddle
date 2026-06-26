"""The audit-layer deterministic congruence pre-pass: auto-derive the spec from a
project's real shape (no hand-authored spec) and emit grounded findings, wired into
run_audit so the bug class RayXI shipped four times cannot silently regrow.

Covers: (1) the deriver nominates a mirror-applier defined across >=2 modules and
the engine guards the project actually calls; (2) it stays silent when there is no
shared replication convention; (3) deterministic_congruence_findings flags the
ungated, HUD-called mutator as a grounded congruence finding; (4) run_audit folds
those findings into the report ALONGSIDE the LLM probe findings."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from saddle.audit.congruence import (
    CONGRUENCE_CONCERN,
    derive_congruence_specs,
    deterministic_congruence_findings,
)
from saddle.audit.ground import parse_sources
from saddle.audit.plan import build_plan
from saddle.audit.run import run_audit

# Two mirror services sharing the apply_replication_snapshot convention (so the
# applier is a real CONVENTION, defined in >=2 modules, not a one-off) + a HUD that
# calls one service's UNGATED mutator directly. The HUD `extends CanvasLayer` (a Godot
# UI base), so it is a CLIENT module -- the signal that separates a genuine UI caller
# from a server-side service-to-service call. The other service is fully gated.
SVC_A = '''
func apply_replication_snapshot(snap):
    _state = snap["state"]
    changed.emit()

func _is_server() -> bool:
    return multiplayer.is_server()

func allocate_point(stat):
    _state[stat] += 1

func get_point(stat):
    return _state.get(stat, 0)
'''

SVC_B = '''
func apply_replication_snapshot(snap):
    _data = snap["data"]

func _is_server() -> bool:
    return multiplayer.is_server()

func set_value(k, v):
    if not _is_server():
        return
    _data[k] = v
'''

HUD = '''
extends CanvasLayer

func _on_alloc_pressed(stat):
    _svc_a.allocate_point(stat)
'''


def _make_gd_project(tmp: Path) -> Path:
    pkg = tmp / "src" / "game"
    pkg.mkdir(parents=True)
    (pkg / "stat_runtime.gd").write_text(SVC_A)
    (pkg / "bank_runtime.gd").write_text(SVC_B)
    (pkg / "stat_hud.gd").write_text(HUD)
    # a doc + registry so build_plan also has non-congruence targets to run
    (tmp / "README.md").write_text("# Game\nthe stat system replicates points\n")
    (tmp / "config").mkdir()
    (tmp / "config" / "stats.json").write_text(json.dumps({"str": {"base": 1}}))
    return tmp


def test_deriver_finds_shared_applier_and_called_guards(tmp_path):
    root = _make_gd_project(tmp_path)
    mods = parse_sources(root, ["src"])
    specs = derive_congruence_specs(mods)
    # one spec for the shared applier convention
    assert [s.mirror_apply for s in specs] == ["apply_replication_snapshot"]
    # the guard the project actually calls is carried (engine token, not game vocab)
    assert "_is_server" in specs[0].guards


def test_deriver_silent_without_shared_convention(tmp_path):
    # A single mirror service (applier defined in only ONE module) is not a
    # convention — the deriver nominates nothing rather than guessing.
    pkg = tmp_path / "src" / "game"
    pkg.mkdir(parents=True)
    (pkg / "only.gd").write_text(SVC_A)
    (pkg / "hud.gd").write_text(HUD)
    mods = parse_sources(tmp_path, ["src"])
    assert derive_congruence_specs(mods) == []


def test_deterministic_findings_flag_the_ungated_hud_called_mutator(tmp_path):
    root = _make_gd_project(tmp_path)
    mods = parse_sources(root, ["src"])
    findings = deterministic_congruence_findings(mods)
    # exactly the ungated, HUD-called mutator — set_value (gated) and get_point
    # (getter) are clean and absent
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "congruence"
    assert f.severity == "error"
    assert f.target == CONGRUENCE_CONCERN
    assert f.grounded                     # cites the def site + the HUD call site
    assert "allocate_point" in f.title
    assert any("stat_hud.gd" in e for e in f.evidence)
    assert f.confidence == "high"


class _EmptyCaller:
    """LLM stand-in that finds nothing — isolates the deterministic contribution."""
    def __init__(self):
        self.calls = 0

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        self.calls += 1
        return json.dumps({"findings": []})


def test_run_audit_includes_deterministic_congruence_finding(tmp_path):
    root = _make_gd_project(tmp_path)
    plan = build_plan(root)
    caller = _EmptyCaller()
    report = asyncio.run(run_audit(plan, caller=caller, persist=False, concurrency=4))
    # the run still completes its coverage ledger ...
    assert report.coverage.complete
    # ... and the deterministic congruence finding rode in even though EVERY LLM
    # probe returned nothing — the class is caught with no hand-authored spec.
    cong = [f for f in report.findings if f.kind == "congruence"]
    assert len(cong) == 1
    assert "allocate_point" in cong[0].title
    assert cong[0].target == CONGRUENCE_CONCERN
