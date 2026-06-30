"""Interprocedural resolved-value pass-down — the cross-module fixpoint.

The one-hop ``resolved_arg_callees`` only clears the FIRST callee a resolver-bound
local is handed to. Real "resolve once at entry, thread the resolved value down"
code (RayXI's model B) is many hops deep and crosses module boundaries, so the
one-hop scan flags every frame past the first — a field of false positives that
buries the genuine raw-base reads. ``codemap/passdown.py`` lifts coverage to the
whole project with a two-polarity (resolved / base) monotonic fixpoint.

These tests pin the THREE properties that make that cut sound, in BOTH languages:
  * a deep / cross-module resolved value clears every frame it reaches (the FP cut);
  * a genuinely raw read with no resolved path STAYS flagged (the cut never blinds);
  * the base-veto keeps a MIXED applier — reached by a resolved value on one path
    and a raw ``base_sources`` value on another — flagged, so the existential cut
    can't cost a false negative (saddle's cardinal sin).

Plus the source-exclusion invariant: the value's own resolvers / base_sources are
never reported as covered CONSUMERS even when a resolved value flows back into
them, because they are where the dataflow starts, not a sink.
"""
from __future__ import annotations

from saddle.codemap import gdref, passdown, pyref
from saddle.codemap.checks import check_value
from saddle.codemap.impact import impact_value
from saddle.codemap.specs import ValueSpec

# resolve_cd = resolver; cast resolves once, then threads cd through THREE more
# frames. The one-hop scan clears hop1 only; the fixpoint must clear hop2 + hop3
# too. sweep reads the base raw with no resolution anywhere — the only real gap.
PY_DEEP = '''
def resolve_cd(inst):
    return inst["cooldown_s"] * 0.9

def cast(inst):
    cd = resolve_cd(inst)
    hop1(inst, cd)

def hop1(inst, cd):
    hop2(inst, cd)

def hop2(inst, cd):
    hop3(inst, cd)

def hop3(inst, cd):
    note(inst["cooldown_s"])

def sweep(inst):
    if inst["cooldown_s"] <= 0:
        expire(inst)
'''

GD_DEEP = '''
func resolve_cd(inst):
    return inst["cooldown_s"] * 0.9

func cast(inst):
    var cd = resolve_cd(inst)
    hop1(inst, cd)

func hop1(inst, cd):
    hop2(inst, cd)

func hop2(inst, cd):
    hop3(inst, cd)

func hop3(inst, cd):
    note(inst["cooldown_s"])

func sweep(inst):
    if inst["cooldown_s"] <= 0:
        expire(inst)
'''

DEEP_SPEC = ValueSpec(name="cooldown", field="cooldown_s", accessor="resolve_cd")


def test_py_deep_passdown_clears_every_hop_but_keeps_raw_read():
    mods = pyref.parse_modules([("abil.py", PY_DEEP)])
    imp = impact_value(mods, DEEP_SPEC)
    # hop2 + hop3 are what the one-hop scan MISSES — the interprocedural lift clears them
    assert {"hop1", "hop2", "hop3"} <= imp.arg_covered_funcs
    assert "sweep" not in imp.arg_covered_funcs
    assert {r.func for r in imp.uncovered_reads} == {"sweep"}
    assert {f.detail["func"] for f in check_value(mods, DEEP_SPEC)} == {"sweep"}


def test_gd_deep_passdown_clears_every_hop_but_keeps_raw_read():
    mods = gdref.parse_modules([("abil.gd", GD_DEEP)])
    imp = impact_value(mods, DEEP_SPEC)
    assert {"hop1", "hop2", "hop3"} <= imp.arg_covered_funcs
    assert "sweep" not in imp.arg_covered_funcs
    assert {r.func for r in imp.uncovered_reads} == {"sweep"}
    assert {f.detail["func"] for f in check_value(mods, DEEP_SPEC)} == {"sweep"}


# Cross-module: the resolver and the deep applier live in DIFFERENT files. The
# fixpoint stitches sites across modules, so deep_apply (in b) is covered by a
# resolved value produced in a.
PY_MOD_A = '''
def resolve_cd(inst):
    return inst["cooldown_s"] * 0.9

def cast(inst):
    cd = resolve_cd(inst)
    deep_apply(inst, cd)
'''
PY_MOD_B = '''
def deep_apply(inst, cd):
    note(inst["cooldown_s"])
'''
GD_MOD_A = '''
func resolve_cd(inst):
    return inst["cooldown_s"] * 0.9

func cast(inst):
    var cd = resolve_cd(inst)
    deep_apply(inst, cd)
'''
GD_MOD_B = '''
func deep_apply(inst, cd):
    note(inst["cooldown_s"])
'''


def test_py_cross_module_passdown_clears():
    mods = pyref.parse_modules([("a.py", PY_MOD_A), ("b.py", PY_MOD_B)])
    imp = impact_value(mods, DEEP_SPEC)
    assert "deep_apply" in imp.arg_covered_funcs        # cleared across the file boundary
    assert not check_value(mods, DEEP_SPEC)             # no false-positive gap


def test_gd_cross_module_passdown_clears():
    mods = gdref.parse_modules([("a.gd", GD_MOD_A), ("b.gd", GD_MOD_B)])
    imp = impact_value(mods, DEEP_SPEC)
    assert "deep_apply" in imp.arg_covered_funcs
    assert not check_value(mods, DEEP_SPEC)


# Base-veto soundness keystone. With base_sources declared, _mixed is reached by a
# RESOLVED value (via via_resolved) AND a raw base value (via via_base) — it must
# stay flagged. _pure is reached ONLY by a resolved value — it must clear. This is
# the one property that keeps the EXISTENTIAL cut from costing a false negative.
PY_VETO = '''
def resolve_cd(inst):
    return inst["cooldown_s"] * 0.9

def get_base(aid):
    return REG[aid]

def via_resolved(inst):
    cd = resolve_cd(inst)
    _mixed(inst, cd)

def via_base(aid):
    base = get_base(aid)
    _mixed(base, base)

def _mixed(inst, cd):
    note(inst["cooldown_s"])

def pure_path(inst):
    cd = resolve_cd(inst)
    _pure(inst, cd)

def _pure(inst, cd):
    note(inst["cooldown_s"])
'''

GD_VETO = '''
func resolve_cd(inst):
    return inst["cooldown_s"] * 0.9

func get_base(aid):
    return REG[aid]

func via_resolved(inst):
    var cd = resolve_cd(inst)
    _mixed(inst, cd)

func via_base(aid):
    var base = get_base(aid)
    _mixed(base, base)

func _mixed(inst, cd):
    note(inst["cooldown_s"])

func pure_path(inst):
    var cd = resolve_cd(inst)
    _pure(inst, cd)

func _pure(inst, cd):
    note(inst["cooldown_s"])
'''

VETO_SPEC = ValueSpec(name="cooldown", field="cooldown_s",
                      accessor="resolve_cd", base_sources=("get_base",))


def test_py_base_veto_keeps_mixed_applier_flagged():
    mods = pyref.parse_modules([("abil.py", PY_VETO)])
    imp = impact_value(mods, VETO_SPEC)
    # _pure clears (resolved-only); _mixed does NOT (a raw base also reaches it)
    assert "_pure" in imp.arg_covered_funcs
    assert "_mixed" not in imp.arg_covered_funcs
    assert "_mixed" in {f.detail["func"] for f in check_value(mods, VETO_SPEC)}


def test_gd_base_veto_keeps_mixed_applier_flagged():
    mods = gdref.parse_modules([("abil.gd", GD_VETO)])
    imp = impact_value(mods, VETO_SPEC)
    assert "_pure" in imp.arg_covered_funcs
    assert "_mixed" not in imp.arg_covered_funcs
    assert "_mixed" in {f.detail["func"] for f in check_value(mods, VETO_SPEC)}


def test_py_no_base_sources_is_pure_existential_resolved():
    # Same fixture, but the spec names NO base_sources: with no base atom to find,
    # the veto can't fire, so _mixed clears too. This documents that declaring
    # base_sources is what BUYS the veto — its absence is pure existential-resolved.
    spec = ValueSpec(name="cooldown", field="cooldown_s", accessor="resolve_cd")
    mods = pyref.parse_modules([("abil.py", PY_VETO)])
    imp = impact_value(mods, spec)
    assert {"_pure", "_mixed"} <= imp.arg_covered_funcs


# Source-exclusion: a two-resolver chain feeds the FIRST resolver's output into the
# SECOND. The fixpoint sees a resolved atom reach the second resolver's param, but a
# resolver is the SOURCE of the value, never a covered consumer — only _spawn is.
PY_SOURCE = '''
def get_base(aid):
    return REG[aid]

def apply_mods(aid, d):
    return d

def resolve_cd(aid, d):
    return d

def cast(aid):
    d = get_base(aid)
    d = apply_mods(aid, d)
    d = resolve_cd(aid, d)
    _spawn(d)

def _spawn(d):
    x = d["cooldown_s"]
'''

SOURCE_SPEC = ValueSpec(name="cooldown", field="cooldown_s",
                        accessor=("apply_mods", "resolve_cd"))


def test_py_resolvers_never_reported_as_covered_consumers():
    mods = pyref.parse_modules([("abil.py", PY_SOURCE)])
    imp = impact_value(mods, SOURCE_SPEC)
    # only the genuine downstream sink is covered — neither resolver leaks in
    assert imp.arg_covered_funcs == {"_spawn"}


def test_name_ambiguity_contributes_no_passdown_coverage():
    # deep_apply is defined in TWO modules: the cross-module fixpoint cannot attribute
    # its parameters, so it MUST contribute neither coverage nor veto — a hard, sound
    # bail. (The one-hop resolved_arg_callees scan clears by NAME and the whole value
    # axis projects coverage by func name, so impact_value's arg_covered_funcs is
    # name-granular by construction — a pre-existing, feature-independent limitation
    # recorded separately in the audit, not the pass-down fixpoint's job to repair.)
    a = '''
def resolve_cd(inst):
    return inst["cooldown_s"] * 0.9
def cast(inst):
    cd = resolve_cd(inst)
    deep_apply(inst, cd)
def deep_apply(inst, cd):
    note(inst["cooldown_s"])
'''
    b = '''
def deep_apply(inst, cd):
    other(inst["cooldown_s"])
'''
    mods = pyref.parse_modules([("a.py", a), ("b.py", b)])
    # the NEW interprocedural fixpoint attributes nothing to the ambiguous callee
    assert "deep_apply" not in passdown.resolved_passdown_funcs(mods, {"resolve_cd"}, set())
    assert "deep_apply" not in passdown.base_reachable_funcs(mods, {"resolve_cd"}, set())


def test_resolved_passdown_funcs_empty_resolvers_is_empty():
    # A degenerate spec with no resolvers can clear nothing — never raises.
    mods = pyref.parse_modules([("abil.py", PY_DEEP)])
    assert passdown.resolved_passdown_funcs(mods, set(), set()) == set()


def test_coverage_memo_is_live_and_identity_guarded():
    # The _coverage memo was silently DEAD (it stored the bare result tuple, so the
    # load's `hit[0] is mods` identity guard could never match `covered`-the-set). This
    # pins that a repeat call on the SAME mods object is served from cache (same tuple),
    # while a DISTINCT mods object is a miss — never a false hit on a recycled id.
    rs, bs = frozenset({"resolve_cd"}), frozenset()
    mods = pyref.parse_modules([("abil.py", PY_DEEP)])
    first = passdown._coverage(mods, rs, bs)
    assert passdown._coverage(mods, rs, bs) is first      # memo hit — recompute avoided
    other = pyref.parse_modules([("abil.py", PY_DEEP)])
    assert passdown._coverage(other, rs, bs) is not first  # distinct object -> recompute


# ── Resolver-WRAPPERS (return polarity) ──────────────────────────────────────
# A wrapper RETURNS a resolved value and never a base the scan can see. A CALL to
# one is the structural twin of a direct resolver call (`def = wrap(id); use
# def.field` mirrors `def = resolve_def(id); use def.field`), so the consumer's raw
# reads clear via the SAME flow-insensitive `_covering` mechanism. These pin: a real
# wrapper clears its consumers; a BASE-returning wrapper does NOT (fail-closed, the
# cardinal-sin guard); polarity propagates through a chain; name-ambiguity bails; and
# the one known fail-OPEN residual (a cast-laundered base return) is surfaced loudly.
PY_WRAP = '''
def resolve_def(aid):
    return REG[aid]

def hotbar_def(aid):
    d = resolve_def(aid)
    return d

def cons_a(aid):
    d = hotbar_def(aid)
    return d["cooldown_s"]

def cons_b(aid):
    return hotbar_def(aid)["cooldown_s"]
'''
GD_WRAP = '''
func resolve_def(aid):
    return REG[aid]

func hotbar_def(aid):
    var d = resolve_def(aid)
    return d

func cons_a(aid):
    var d = hotbar_def(aid)
    return d["cooldown_s"]

func cons_b(aid):
    return hotbar_def(aid)["cooldown_s"]
'''
WRAP_SPEC = ValueSpec(name="cooldown", field="cooldown_s", accessor="resolve_def")


def test_py_resolver_wrapper_clears_consumers():
    mods = pyref.parse_modules([("hud.py", PY_WRAP)])
    assert "hotbar_def" in passdown.resolver_wrapper_funcs(mods, {"resolve_def"}, set())
    imp = impact_value(mods, WRAP_SPEC)
    flagged = {r.func for r in imp.uncovered_reads}
    assert "cons_a" not in flagged and "cons_b" not in flagged   # both consumers clear
    assert not check_value(mods, WRAP_SPEC)                       # no false-positive gap


def test_gd_resolver_wrapper_clears_consumers():
    mods = gdref.parse_modules([("hud.gd", GD_WRAP)])
    assert "hotbar_def" in passdown.resolver_wrapper_funcs(mods, {"resolve_def"}, set())
    imp = impact_value(mods, WRAP_SPEC)
    flagged = {r.func for r in imp.uncovered_reads}
    assert "cons_a" not in flagged and "cons_b" not in flagged
    assert not check_value(mods, WRAP_SPEC)


# Base-returning wrapper: good_wrap returns a resolved value (R); bad_wrap returns a
# RAW base directly (B). The B-return veto must keep bad_wrap OUT of the wrapper set,
# so cons_bad's un-resolved read STAYS caught. This is the keystone: the wrapper
# mechanism must never clear a consumer that reads a value the wrapper left un-resolved.
PY_WRAP_BASE = '''
def resolve_def(aid):
    return REG[aid]

def get_base(aid):
    return RAW[aid]

def good_wrap(aid):
    return resolve_def(aid)

def bad_wrap(aid):
    return get_base(aid)

def cons_good(aid):
    return good_wrap(aid)["cooldown_s"]

def cons_bad(aid):
    return bad_wrap(aid)["cooldown_s"]
'''
GD_WRAP_BASE = '''
func resolve_def(aid):
    return REG[aid]

func get_base(aid):
    return RAW[aid]

func good_wrap(aid):
    return resolve_def(aid)

func bad_wrap(aid):
    return get_base(aid)

func cons_good(aid):
    return good_wrap(aid)["cooldown_s"]

func cons_bad(aid):
    return bad_wrap(aid)["cooldown_s"]
'''
WRAP_BASE_SPEC = ValueSpec(name="cooldown", field="cooldown_s",
                           accessor="resolve_def", base_sources=("get_base",))


def test_py_base_returning_wrapper_stays_fail_closed():
    mods = pyref.parse_modules([("hud.py", PY_WRAP_BASE)])
    wraps = passdown.resolver_wrapper_funcs(mods, {"resolve_def"}, {"get_base"})
    assert "good_wrap" in wraps
    assert "bad_wrap" not in wraps                 # base-returning -> NOT blessed
    flagged = {f.detail["func"] for f in check_value(mods, WRAP_BASE_SPEC)}
    assert "cons_bad" in flagged                   # genuine un-resolved read STILL caught
    assert "cons_good" not in flagged              # the real wrapper consumer clears


def test_gd_base_returning_wrapper_stays_fail_closed():
    mods = gdref.parse_modules([("hud.gd", GD_WRAP_BASE)])
    wraps = passdown.resolver_wrapper_funcs(mods, {"resolve_def"}, {"get_base"})
    assert "good_wrap" in wraps
    assert "bad_wrap" not in wraps
    flagged = {f.detail["func"] for f in check_value(mods, WRAP_BASE_SPEC)}
    assert "cons_bad" in flagged
    assert "cons_good" not in flagged


# Multi-hop: outer returns inner(...) (a ("call", inner) atom); inner is a wrapper, so
# its RETURN polarity propagates up to outer. cons binds from outer and must clear —
# this exercises the ("call", fn) extraction in BOTH adapters and the fixpoint stitch.
PY_WRAP_CHAIN = '''
def resolve_def(aid):
    return REG[aid]

def inner(aid):
    return resolve_def(aid)

def outer(aid):
    return inner(aid)

def cons(aid):
    d = outer(aid)
    return d["cooldown_s"]
'''
GD_WRAP_CHAIN = '''
func resolve_def(aid):
    return REG[aid]

func inner(aid):
    return resolve_def(aid)

func outer(aid):
    return inner(aid)

func cons(aid):
    var d = outer(aid)
    return d["cooldown_s"]
'''


def test_py_chained_wrapper_propagates_polarity():
    mods = pyref.parse_modules([("hud.py", PY_WRAP_CHAIN)])
    assert {"inner", "outer"} <= passdown.resolver_wrapper_funcs(mods, {"resolve_def"}, set())
    assert "cons" not in {r.func for r in impact_value(mods, WRAP_SPEC).uncovered_reads}


def test_gd_chained_wrapper_propagates_polarity():
    mods = gdref.parse_modules([("hud.gd", GD_WRAP_CHAIN)])
    assert {"inner", "outer"} <= passdown.resolver_wrapper_funcs(mods, {"resolve_def"}, set())
    assert "cons" not in {r.func for r in impact_value(mods, WRAP_SPEC).uncovered_reads}


def test_name_ambiguous_wrapper_not_blessed():
    # hotbar_def is defined in TWO modules: a call to the name could hit the non-wrapper
    # definition in b, so _callee_polarity bails and the wrapper filter excludes it
    # (defined in >1 module). cons's read stays flagged — fail-closed on ambiguity.
    a = '''
def resolve_def(aid):
    return REG[aid]
def hotbar_def(aid):
    return resolve_def(aid)
def cons(aid):
    return hotbar_def(aid)["cooldown_s"]
'''
    b = '''
def hotbar_def(aid):
    return RAW[aid]
'''
    mods = pyref.parse_modules([("a.py", a), ("b.py", b)])
    assert "hotbar_def" not in passdown.resolver_wrapper_funcs(mods, {"resolve_def"}, set())
    assert "cons" in {r.func for r in impact_value(mods, WRAP_SPEC).uncovered_reads}


# KNOWN fail-OPEN residual — pinned so the limitation is visible and any future change
# to it is a conscious one. hotbar_def returns a resolved value on the has_runtime path
# AND a base laundered through an `as Dictionary` cast on the other. The structural
# B-veto cannot see a base hidden behind a cast/reassignment, so hotbar_def is
# (knowingly) blessed despite the base path. This is the return-axis twin of the
# laundered-base PARAM residual; recorded as a known false-negative class in the audit.
# Contrast: a DIRECT `return get_base(aid)` IS vetoed (test_*_base_returning_*) — only
# the cast launder slips, and only because a cast is opaque to the regex/AST scan.
GD_WRAP_LAUNDER = '''
func resolve_def(aid):
    return REG[aid]

func get_base(aid):
    return RAW[aid]

func has_runtime():
    return true

func hotbar_def(aid):
    var raw = get_base(aid)
    var base_def = raw as Dictionary
    if has_runtime():
        return resolve_def(aid)
    return base_def

func cons(aid):
    var d = hotbar_def(aid)
    return d["cooldown_s"]
'''


def test_gd_laundered_base_return_is_known_fail_open_residual():
    mods = gdref.parse_modules([("hud.gd", GD_WRAP_LAUNDER)])
    wraps = passdown.resolver_wrapper_funcs(mods, {"resolve_def"}, {"get_base"})
    assert "hotbar_def" in wraps          # fail-OPEN: cast-laundered base escapes the veto
    # The same launder via plain reassignment is the Python form of this residual; a
    # NON-laundered direct base return is caught (see base_returning_wrapper tests).


def test_laundered_base_wrapper_is_enumerated_not_silently_blessed():
    # The fail-OPEN residual is SURFACED, not blessed-and-forgotten: a mixed wrapper (R on
    # one path, cast-laundered base on the other) is ENUMERATED by name so a value-drift
    # audit can eyeball it. This is the doctrine — keep the strict structural rule AND name
    # the gap, rather than paper it with a runtime guess. The surface is the residual
    # EXACTLY: a clean all-resolved wrapper has zero fail-open risk and never appears.
    mods = gdref.parse_modules([("hud.gd", GD_WRAP_LAUNDER)])
    fo = passdown.laundered_base_wrapper_funcs(mods, {"resolve_def"}, {"get_base"})
    assert "hotbar_def" in fo                          # mixed wrapper surfaced by name
    spec = ValueSpec(name="cooldown", field="cooldown_s",
                     accessor="resolve_def", base_sources=("get_base",))
    assert impact_value(mods, spec).fail_open_wrappers == {"hotbar_def"}   # carried on report
    # A clean wrapper (every return provably resolved) is NOT surfaced — no noise on every
    # wrapper, only the ones where the bless genuinely rests on an unverifiable return.
    clean = gdref.parse_modules([("hud.gd", GD_WRAP)])
    assert passdown.laundered_base_wrapper_funcs(clean, {"resolve_def"}, set()) == set()
    assert impact_value(clean, WRAP_SPEC).fail_open_wrappers == set()
