"""`saddle converge <design_id>` — Layer 4 from the command line.

The convergence loop needs a live coder (the Agent SDK), so these tests drive
only the command paths that terminate BEFORE a coder is ever opened: a clean tree
(ALREADY, exit 0), a design with no declared surface (NO_SURFACE, exit 1), and an
unknown design id (exit 2). The DKB is stubbed; the surface and the code are real.
"""
from __future__ import annotations

from saddle.cli import main
from saddle.codemap import SurfaceManifest, ValueSpec
from saddle.models import Design

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


def _design_no_surface() -> Design:
    return Design(ask="something with no declared surface", meta={}, id="dsg_test")


class _StubDKB:
    def __init__(self, design: Design) -> None:
        self._design = design
        self.meta_updates: list[tuple] = []

    def get_design(self, ctx, did):
        return self._design if did == self._design.id else None

    def update_design_meta(self, ctx, did, patch):
        self.meta_updates.append((did, patch))
        return True


def test_cli_converge_already_satisfied_exits_zero(tmp_path, monkeypatch, capsys):
    (tmp_path / "abil.py").write_text(COVERED_PY)        # surface already complete
    stub = _StubDKB(_design_with_surface())
    monkeypatch.setattr("saddle.dkb.get_dkb", lambda: stub)
    code = main(["converge", "dsg_test", "--root", str(tmp_path), "--no-persist"])
    out = capsys.readouterr().out
    assert code == 0
    assert "ALREADY_SATISFIED" in out and "COMPLETE" in out


def test_cli_converge_no_surface_exits_one(tmp_path, monkeypatch, capsys):
    stub = _StubDKB(_design_no_surface())
    monkeypatch.setattr("saddle.dkb.get_dkb", lambda: stub)
    code = main(["converge", "dsg_test", "--root", str(tmp_path), "--no-persist"])
    out = capsys.readouterr().out
    assert code == 1
    assert "NO_SURFACE" in out


def test_cli_converge_unknown_design_exits_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("saddle.dkb.get_dkb", lambda: _StubDKB(_design_with_surface()))
    code = main(["converge", "nope", "--root", str(tmp_path)])
    assert code == 2
    assert "no design" in capsys.readouterr().err
