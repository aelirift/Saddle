"""conformance_scan — Stage 4's engine: re-verify the project's SETTLED designs
against the code as it stands now.

A design enumerates its completeness surface once (the SurfaceManifest in
``Design.meta``); ``conformance_scan`` pulls the project's recent settled designs,
keeps the ones that declared a surface, parses the tree ONCE, and re-runs each
design's own gate (``intent_drift``) against that fresh parse. A design whose
surface the current code no longer satisfies is the turn-end "code drifted from
the design" miss. These pin the engine against a stub DKB + a real tmp code tree,
no LLM and no real database.
"""
from __future__ import annotations

from saddle.codemap import SurfaceManifest, ValueSpec
from saddle.context import Context
from saddle.design import conformance_scan
from saddle.models import DESIGN_FINAL, DESIGN_FLAGGED, Design

_CTX = Context(tenant="acme", project="game")

# Routed cleanly through the resolver -> the design's surface is satisfied.
_CLEAN = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def sweep(inst):
    cd = resolve_cd(inst)        # routed through the resolver -> covered
    return cd - 1
'''

# A raw base read that bypasses the resolver -> the modifier misses -> DRIFT.
_DRIFTED = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def sweep(inst):
    return inst["cooldown_s"] - 1   # DRIFT: raw base read, modifier misses
'''


def _surface() -> dict:
    return SurfaceManifest(
        values=[ValueSpec("cooldown", "cooldown_s", "resolve_cd", ("build_def",))]
    ).to_dict()


def _design(*, did: str, status: str = DESIGN_FINAL, surface: bool = True,
            summary: str = "route cooldown reads through the resolver") -> Design:
    return Design(
        id=did, ask="route every cooldown read through the resolver",
        summary=summary, status=status,
        meta={"surface": _surface()} if surface else {},
    )


class _StubDKB:
    """Serves designs like the real DKB's ``list_designs`` — newest-first, filtered
    by ``status`` and capped by ``limit`` — so the status/limit contract the engine
    relies on is exercised, not bypassed."""

    def __init__(self, designs: list[Design]) -> None:
        self._designs = list(designs)

    def list_designs(self, ctx, *, status=None, limit=50):
        rows = [d for d in self._designs if status is None or d.status == status]
        return rows[: int(limit)]


def _write(tmp_path, body: str):
    (tmp_path / "abil.py").write_text(body)


def test_conformance_catches_drift_from_settled_design(tmp_path):
    _write(tmp_path, _DRIFTED)
    dkb = _StubDKB([_design(did="d1")])
    res = conformance_scan(_CTX, dkb=dkb, root=tmp_path)

    assert res.has_drift
    assert res.designs_checked == 1
    assert len(res.drifts) == 1
    d = res.drifts[0]
    assert d.design_id == "d1"
    assert d.summary == "route cooldown reads through the resolver"
    assert {f.detail["func"] for f in d.findings} == {"sweep"}
    assert all(f.check == "value_propagation" for f in d.findings)
    # value_propagation is error-grade -> the scan and the drift both flag error.
    assert d.has_error and res.has_error


def test_conformance_clean_code_does_not_drift(tmp_path):
    _write(tmp_path, _CLEAN)
    dkb = _StubDKB([_design(did="d1")])
    res = conformance_scan(_CTX, dkb=dkb, root=tmp_path)

    assert res.designs_checked == 1      # it WAS gated...
    assert not res.has_drift             # ...and the code satisfies it
    assert res.drifts == []
    assert not res.has_error


def test_conformance_only_gates_settled_designs(tmp_path):
    """A still-FLAGGED design never converged in audit, so gating code against it
    is premature — Stage 4 weighs only DESIGN_FINAL. Even with drifted code the
    flagged design is not gated (the status filter reaches the DKB)."""
    _write(tmp_path, _DRIFTED)
    dkb = _StubDKB([_design(did="dflag", status=DESIGN_FLAGGED)])
    res = conformance_scan(_CTX, dkb=dkb, root=tmp_path)

    assert res.designs_checked == 0      # the flagged design was filtered out
    assert not res.has_drift


def test_conformance_skips_designs_without_a_surface(tmp_path):
    """A settled design that never declared a surface has nothing to gate — it is
    not counted as checked and cannot drift (the documented silent case)."""
    _write(tmp_path, _DRIFTED)
    dkb = _StubDKB([_design(did="dnosurf", surface=False)])
    res = conformance_scan(_CTX, dkb=dkb, root=tmp_path)

    assert res.designs_checked == 0
    assert not res.has_drift


def test_conformance_no_code_root_is_silent(tmp_path, monkeypatch):
    """No resolvable code root -> nothing to compare the intent against -> an empty
    result, never a false 'drift'."""
    monkeypatch.delenv("SADDLE_CODE_ROOT", raising=False)
    dkb = _StubDKB([_design(did="d1")])
    res = conformance_scan(_CTX, dkb=dkb, root=None)

    assert res.designs_checked == 0
    assert not res.has_drift


def test_conformance_gates_every_settled_design_in_one_scan(tmp_path):
    """Two settled designs, one satisfied and one drifted, gated against the SAME
    fresh parse — only the drifted one surfaces, and both were counted as checked
    (the parse-once / gate-each contract)."""
    _write(tmp_path, _DRIFTED)
    dkb = _StubDKB([
        _design(did="dclean", summary="an unrelated, satisfied design"),
        _design(did="ddrift", summary="the cooldown resolver design"),
    ])
    # 'dclean' declares the SAME cooldown surface, so against _DRIFTED BOTH drift;
    # to isolate, give the clean one a surface the code already satisfies.
    dkb._designs[0].meta["surface"] = SurfaceManifest(
        values=[ValueSpec("cdr", "cd_reduction", "resolve_cd", ("build_def",))]
    ).to_dict()
    res = conformance_scan(_CTX, dkb=dkb, root=tmp_path)

    assert res.designs_checked == 2
    ids = {d.design_id for d in res.drifts}
    assert ids == {"ddrift"}             # only the cooldown design drifted
