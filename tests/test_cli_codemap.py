"""`saddle codemap <design_id>` — the WIRED gate, from the command line.

This is the line RayXI never had: a persisted design's DECLARED surface re-run
against the code as it stands, exiting nonzero on any gap so it drops into a
commit hook / CI step. The gate is stubbed at the DKB only (the design + its
surface are real, the code is real); no embedder, no network.
"""
from __future__ import annotations

from saddle.cli import main
from saddle.codemap import SurfaceManifest, ValueSpec
from saddle.models import Design

GAPS_PY = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"]

def cast(inst):
    cd = resolve_cd(inst)
    return cd if inst["cooldown_s"] > 0 else 0

def sweep(inst):
    return inst["cooldown_s"] - 1     # UNCOVERED
'''

COVERED_PY = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"]

def cast(inst):
    cd = resolve_cd(inst)
    return cd if inst["cooldown_s"] > 0 else 0
'''


def _design_with_surface() -> Design:
    surface = SurfaceManifest(
        values=[ValueSpec(name="cooldown", field="cooldown_s",
                          accessor="resolve_cd", producers=("build_def",))],
    ).to_dict()
    return Design(ask="reduce cooldown via a talent point",
                  meta={"surface": surface}, id="dsg_test")


class _StubDKB:
    def __init__(self, design: Design) -> None:
        self._design = design

    def get_design(self, ctx, did):
        return self._design if did == self._design.id else None


def test_cli_gate_exits_nonzero_on_gaps(tmp_path, monkeypatch, capsys):
    (tmp_path / "abil.py").write_text(GAPS_PY)
    monkeypatch.setattr("saddle.dkb.get_dkb", lambda: _StubDKB(_design_with_surface()))
    code = main(["codemap", "dsg_test", "--root", str(tmp_path), "--json"])
    out = capsys.readouterr().out
    assert code == 1                              # nonzero so a hook/CI blocks
    assert "value_propagation" in out
    assert "sweep" in out                         # the uncovered site named


def test_cli_gate_exits_zero_when_complete(tmp_path, monkeypatch, capsys):
    (tmp_path / "abil.py").write_text(COVERED_PY)
    monkeypatch.setattr("saddle.dkb.get_dkb", lambda: _StubDKB(_design_with_surface()))
    code = main(["codemap", "dsg_test", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "COMPLETE" in out


def test_cli_gate_unknown_design_exits_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("saddle.dkb.get_dkb", lambda: _StubDKB(_design_with_surface()))
    code = main(["codemap", "nope", "--root", str(tmp_path)])
    assert code == 2
    assert "no design" in capsys.readouterr().err
