"""Interprocedural resolved-value pass-down coverage — the cross-module fixpoint.

The one-hop ``resolved_arg_callees`` (in pyref/gdref) recognises a resolver-bound
LOCAL handed straight to a callee. Real "resolve once, pass the resolved value
down" code is DEEPER than one hop and crosses module boundaries: a project that
resolves at the activation entry (``def = resolve_def(id, def)``) then threads the
resolved ``def`` through a dozen call frames — commit, side-effects, summon,
echo — has consumers many hops from the resolver. The one-hop scan clears the
first callee and flags the rest, so the resolved-pass-down architecture reads as a
field of false positives that BURY the genuine raw-base reads (a HUD tooltip that
reads ``get_def_by_id(id).cooldown`` un-resolved).

This module lifts the coverage to the whole project. A function PARAMETER is
"resolved-on-entry" when the resolved value provably reaches it; a function with
such a parameter has its raw reads covered (same whole-function model the one-hop
scan uses). The reachability is a monotonic fixpoint over the cross-module call
graph, computed in TWO polarities so the false-positive cut stays sound:

  * R-reach (resolved): a parameter position is R-reachable if AT LEAST ONE call
    site passes a resolved atom there — an inline resolver call, a resolver-bound
    local of the caller, or an already-R-reachable parameter of the caller. This is
    EXISTENTIAL on purpose: real code launders the resolved value through
    containers / signal closures / ``duplicate()`` on some paths (provenance a
    static scan cannot follow), so requiring EVERY path to prove resolved would
    clear nothing. One provable resolved path is enough evidence the design routes
    the resolved value here.

  * B-reach (base): the same existential reachability for the BASE value — a
    ``base_sources`` accessor call, a base-bound local, or a B-reachable parameter.

A parameter is covered (resolved-on-entry) iff it is R-reachable AND NOT
B-reachable. The base-veto is what keeps EXISTENTIAL sound: an applier reached by a
resolved value on one path and a raw base value on another is NOT cleared, so the
genuine "someone passed the un-resolved value in" gap is never hidden. When a
project declares no ``base_sources``, there is no base atom to find, so coverage is
pure existential-resolved (still sound against the synthetic raw-read drift, which
has no resolved path at all and so is never cleared).

RESOLVER-WRAPPERS (return polarity). The one-hop scan and the param fixpoint above
both recognise resolution only when a resolver is CALLED — inline, or via a
resolver-bound local. Real code also resolves behind a HELPER: a project function
that fetches the base and returns the resolved value (``func _hotbar_def(id): ...;
return apply_mods(id, base)``), so every consumer does ``var d = _hotbar_def(id)``
and reads ``d.field`` with no resolver call in its OWN scope. Without recognising
the helper, every such consumer reads as a false positive — the same field-of-FPs
the param fixpoint exists to kill, one level up.

So the fixpoint also computes, in the SAME two polarities, each function's RETURN
polarity (existential, like reachability): a function is R-returning if AT LEAST ONE
return expression is resolved (an inline resolver call, a resolver-bound local, a
resolved-on-entry param, or a call to an R-returning wrapper), and B-returning if any
return is a base value by the mirror rule. A function that is R-returning AND NOT
B-returning is a resolver-WRAPPER; ``impact_value`` treats a CALL to it exactly like a
declared resolver call (resolution happens in the caller's scope). The rule is PURELY
STRUCTURAL — it reasons only about what each return expression syntactically is, never
about any runtime invariant of the project under analysis.

The B-returning veto is what keeps wrapper-blessing sound: a helper that returns a
base value on ANY path the scan can SEE (a direct ``return get_def_by_id(id)`` or a
``return <base-bound local>``) is B-returning, so it is NOT blessed and its callers
stay flagged — fail-closed on every base-return the analyzer can resolve. The only
base-return it cannot see is one laundered through a non-base reassignment before the
return (``base = raw as T; return base``) — the SAME bounded laundered-base residual
named above, now on the return axis: a wrapper with a resolved path and a
laundered-base sibling path is blessed (documented, not a new FN class). The mirror
laundering on the RESOLVED side (a resolved value reassigned before return) fails the
SAFE way — the wrapper is not recognised, its callers stay flagged (a false positive,
never a missed gap). Multi-hop wrappers (a wrapper returning another wrapper's result,
directly or via a local) ARE covered by the fixpoint; only the launder-through-
reassignment shapes are residual.

Name ambiguity is a hard bail: a callee defined in more than one module cannot have
its parameters attributed nor its return polarity trusted (a call to the name could
hit a non-wrapper definition), so its sites contribute nothing (no coverage, no veto,
no wrapper-blessing) — fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

# An argument at a call site (or a function's return expression), pre-classified by
# the per-language adapter as far as it can without cross-function context:
#   "R"            inline resolver call (``helper(x, resolve(x))`` / ``return resolve(x)``)
#   "B"            inline base_source call (``helper(get_def_by_id(id))``)
#   ("n", name)    a bare identifier — resolved against the caller's locals/params here
#   ("call", fn)   an inline call to a PROJECT function (not a resolver/base_source) —
#                  resolved against fn's RETURN polarity (a resolver-WRAPPER passes its
#                  resolved-ness through). Only emitted for unambiguous DIRECT calls
#                  (a bare ``fn(...)``), never a method/attribute call, so a method
#                  name colliding with an unrelated top-level wrapper can't leak.
#   None           anything else (a literal, a subscript, a compound expression)
Atom = Union[str, tuple, None]


@dataclass
class ModuleFacts:
    """Everything the cross-module fixpoint needs from ONE module, extracted by the
    language adapter so the graph algorithm itself stays language-agnostic.

    ``params`` maps each function defined here to its ORDERED parameter names (only
    functions whose signature parses). ``resolved_locals`` / ``base_locals`` map an
    enclosing function (``None`` = module scope) to the local names it binds from a
    resolver / base_source call. ``call_locals`` maps an enclosing function to
    ``{local_name: callee}`` for ``x = callee(...)`` where ``callee`` is a PROJECT
    function (not a resolver/base_source) — so ``x`` carries that callee's RETURN
    polarity (a local bound from a resolver-wrapper resolves). ``returns`` maps each
    function to the list of its return-expression ``Atom`` s (one per ``return``),
    classified the same way a call argument is. ``sites`` is every outgoing call as
    ``(caller_func, callee_name, [Atom, ...])`` with one atom per positional arg."""

    params: dict = field(default_factory=dict)
    resolved_locals: dict = field(default_factory=dict)
    base_locals: dict = field(default_factory=dict)
    call_locals: dict = field(default_factory=dict)
    returns: dict = field(default_factory=dict)
    sites: list = field(default_factory=list)


# Process-lifetime memo: impact_value calls this once per ValueSpec, but every spec
# of one design shares the same resolvers + base_sources and the same parsed module
# list, so the fixpoint is identical across them. Keyed by the identity of the mods
# list plus the token sets — and the entry HOLDS the mods list (the value is
# ``(mods, result)``) so that exact object stays alive while cached and CPython can
# never recycle its id() onto a different list, which would otherwise return a stale
# result for a same-id-but-different-content scan (a soundness hazard — a false clear).
# Bounded so a long-lived process (the live Stop hook) can't grow it without limit; a
# miss just recomputes the ~2s whole-project fixpoint.
_CACHE: dict = {}
_CACHE_CAP = 64


def resolved_passdown_funcs(mods, resolvers, base_sources=()) -> set:
    """Names of functions whose raw reads are covered because a resolved value
    provably reaches one of their parameters (R-reachable and not B-reachable).

    ``mods`` is the whole parsed project; ``resolvers`` / ``base_sources`` are the
    ValueSpec token sets. Returns a set of function NAMES — the same whole-function
    granularity ``ValueImpact.arg_covered_funcs`` already consumes."""
    return _coverage(mods, resolvers, base_sources)[0]


def base_reachable_funcs(mods, resolvers, base_sources=()) -> set:
    """Names of functions a RAW base value provably reaches (B-reachable on any
    parameter). This is the base-VETO set ``impact_value`` subtracts from the FULL
    covered set, not just the pass-down portion: the one-hop ``resolved_arg_callees``
    scan is base-blind, so unioning it back in could re-clear a mixed applier the
    pass-down fixpoint correctly flagged. Subtracting the base-reachable names makes
    the veto authoritative over BOTH scans. A removed name flags MORE reads, never
    fewer, so the subtraction can never cost a false negative — only safe noise."""
    return _coverage(mods, resolvers, base_sources)[1]


def resolver_wrapper_funcs(mods, resolvers, base_sources=()) -> set:
    """Names of resolver-WRAPPER functions: a project function that provably RETURNS a
    resolved value (R-returning) and NEVER a base value the analyzer can see
    (not B-returning). ``impact_value`` treats a CALL to one of these exactly like a
    declared resolver call — a caller that does ``var d = wrap(id)`` resolves in its own
    scope. Purely structural (what each return expression syntactically is); name
    ambiguity is excluded (a call could hit a non-wrapper definition). See the module
    docstring for the laundered-base residual and the multi-hop coverage."""
    return _coverage(mods, resolvers, base_sources)[2]


def laundered_base_wrapper_funcs(mods, resolvers, base_sources=()) -> set:
    """The fail-OPEN residual surface: resolver-WRAPPER names that are blessed (so their
    callers are NOT flagged) yet have at least one RETURN the scan cannot confirm is
    resolved — a cast-laundered base (``base = raw as T; return base``), an empty-literal
    fallback, or a resolved value the structural scan missed. These are the MIXED wrappers
    where the bless rests on faith for one path; the analyzer cannot structurally tell a
    legitimate null-guard fallback from a buggy unresolved base, so it surfaces the wrapper
    by NAME instead of guessing. ``impact_value`` carries this to
    ``ValueImpact.fail_open_wrappers`` so a value-drift audit can eyeball each one. A
    wrapper whose returns are ALL resolved has zero fail-open risk and never appears."""
    return _coverage(mods, resolvers, base_sources)[3]


def _coverage(mods, resolvers, base_sources) -> tuple:
    """``(covered, base_reachable, wrappers, fail_open)`` for one (mods, resolvers,
    base_sources), memoised. ``covered`` is the resolved-pass-down clear set (R-reachable,
    not B-reachable, minus the value's own source functions); ``base_reachable`` is the
    veto set; ``wrappers`` is the resolver-WRAPPER names (R-returning, not B-returning)
    whose CALL sites ``impact_value`` treats like a declared resolver call; ``fail_open``
    is the subset of ``wrappers`` blessed on an unverifiable (non-R, non-B) return — the
    enumerated laundered-base residual surface (see ``_run_fixpoint``)."""
    from . import refs

    resolvers = frozenset(resolvers)
    if not resolvers:
        return (set(), set(), set(), set())
    base_sources = frozenset(base_sources)
    ck = (id(mods), resolvers, base_sources)
    hit = _CACHE.get(ck)
    if hit is not None and hit[0] is mods:   # identity-verified — never a recycled id
        return hit[1]
    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()

    facts = [refs.passdown_facts(m, resolvers, base_sources) for m in mods]
    covered, base_reach, wrappers, faith = _run_fixpoint(facts)
    # The value's own SOURCE functions are never downstream consumers: a resolver
    # threaded its own resolved output back into itself (``def = resolve_def(id, def)``
    # on the line after ``apply_ability_modifiers``) makes the fixpoint see a resolved
    # atom reach the resolver's param, but the resolver is where the dataflow STARTS,
    # not a sink whose pass-down we are clearing — and its raw base read is already
    # exempt (``ValueSpec.exempt_funcs``). Likewise a base_source RETURNS the raw base;
    # it consumes nothing. Dropping both from the covered set only ever ADDS findings
    # (a removed name flags more reads, never fewer), so it can't cost a false negative.
    covered = covered - resolvers - base_sources
    # A declared resolver / base_source is never a "discovered" wrapper: the resolver
    # already covers in scope, and a base_source is the value's origin, not a relay.
    wrappers = wrappers - resolvers - base_sources
    # faith ⊆ wrappers (names), so the same declared-symbol cleanup applies.
    faith = faith - resolvers - base_sources
    out = (covered, base_reach, wrappers, faith)
    # Store WITH the mods object so the load's `hit[0] is mods` identity guard can fire:
    # the (id(mods), …) key alone is unsound (a GC'd list's id can be recycled), so the
    # guard re-confirms object identity before trusting a hit. Storing the bare `out`
    # (a 3-tuple) made `hit[0]` the `covered` SET — never `is mods` — so the memo was
    # silently DEAD and passdown_facts recomputed on every impact_* call. Found in the
    # 237-module rayxiv4 audit; correctness-safe (a dead memo never returns a wrong
    # answer) but a real cost, fixed here at the source rather than left as a residual.
    _CACHE[ck] = (mods, out)
    return out


def _run_fixpoint(facts) -> tuple:
    # Aggregate the per-module facts, keyed by module index so same-named functions
    # in different modules stay distinct.
    defs = {}                 # callee name -> module indices defining it
    params = {}               # (modidx, func) -> ordered param names
    resolved_locals = {}      # (modidx, func) -> resolver-bound locals
    base_locals = {}          # (modidx, func) -> base-bound locals
    call_locals = {}          # (modidx, func) -> {local: project-callee it is bound from}
    returns = {}              # (modidx, func) -> [Atom, ...] one per return statement
    sites_by_callee = {}      # callee -> [(modidx, caller, atoms)]

    for i, mf in enumerate(facts):
        for name, pnames in mf.params.items():
            defs.setdefault(name, set()).add(i)
            params[(i, name)] = pnames
        for caller, names in mf.resolved_locals.items():
            resolved_locals[(i, caller)] = set(names)
        for caller, names in mf.base_locals.items():
            base_locals[(i, caller)] = set(names)
        for caller, mapping in mf.call_locals.items():
            call_locals[(i, caller)] = dict(mapping)
        for caller, atoms in mf.returns.items():
            returns[(i, caller)] = list(atoms)
        for caller, callee, atoms in mf.sites:
            sites_by_callee.setdefault(callee, []).append((i, caller, atoms))

    r_reach = {}              # (modidx, func) -> R-reachable param positions
    b_reach = {}              # (modidx, func) -> B-reachable param positions
    r_return = set()          # (modidx, func) that EXISTENTIALLY return a resolved value
    b_return = set()          # (modidx, func) that EXISTENTIALLY return a base value

    def _callee_polarity(callee, ret_set) -> bool:
        # A callee whose RETURN provably carries the polarity — only when it is defined
        # in exactly ONE module. Name ambiguity is a hard bail: a call to the name could
        # hit a non-wrapper definition elsewhere, so it carries nothing (fail-closed).
        midx = defs.get(callee)
        if not midx or len(midx) != 1:
            return False
        return (next(iter(midx)), callee) in ret_set

    def _atom_is(atom, modidx, caller, want_local, want_reach, tag, ret_set):
        if atom == tag:                                       # inline resolver / base call
            return True
        if isinstance(atom, tuple) and len(atom) == 2:
            if atom[0] == "n":
                nm = atom[1]
                if nm in want_local.get((modidx, caller), ()):   # resolver/base-bound local
                    return True
                ps = params.get((modidx, caller))
                if ps:                                            # a reachable param of the caller
                    reachable = want_reach.get((modidx, caller), ())
                    if nm in [ps[p] for p in reachable if p < len(ps)]:
                        return True
                # a local bound from a polarity-returning wrapper carries that polarity
                wcallee = call_locals.get((modidx, caller), {}).get(nm)
                if wcallee is not None and _callee_polarity(wcallee, ret_set):
                    return True
            elif atom[0] == "call":                           # inline call to a wrapper
                if _callee_polarity(atom[1], ret_set):
                    return True
        return False

    changed = True
    while changed:
        changed = False
        # (1) Parameter reachability — the two-polarity existential as before, but
        # _atom_is now also consults call_locals + the return-polarity sets, so an arg
        # that is a wrapper-bound local or an inline wrapper call carries its polarity.
        for callee, modidxs in defs.items():
            if len(modidxs) != 1:                 # name ambiguity -> cannot attribute params
                continue
            mi = next(iter(modidxs))
            pnames = params.get((mi, callee))
            if not pnames:
                continue
            csites = sites_by_callee.get(callee)
            if not csites:
                continue
            key = (mi, callee)
            for pos in range(len(pnames)):
                atoms_here = [(atoms[pos] if pos < len(atoms) else None, smi, sc)
                              for (smi, sc, atoms) in csites]
                if pos not in r_reach.get(key, ()) and any(
                    _atom_is(a, smi, sc, resolved_locals, r_reach, "R", r_return)
                    for (a, smi, sc) in atoms_here
                ):
                    r_reach.setdefault(key, set()).add(pos)
                    changed = True
                if pos not in b_reach.get(key, ()) and any(
                    _atom_is(a, smi, sc, base_locals, b_reach, "B", b_return)
                    for (a, smi, sc) in atoms_here
                ):
                    b_reach.setdefault(key, set()).add(pos)
                    changed = True
        # (2) Return polarity — existential, the mirror of (1) over a function's OWN
        # return expressions. A function is R-returning if ANY return is resolved, and
        # B-returning if ANY return is a base value the analyzer can see. Mutually
        # recursive with (1): a `return <reachable-param>` depends on (1); a
        # `return wrapper(...)` depends on a polarity computed here. Monotone — both
        # sets only grow — so iterating to a fixpoint is order-independent and settles.
        for key, atoms in returns.items():
            mi, fn = key
            if key not in r_return and any(
                _atom_is(a, mi, fn, resolved_locals, r_reach, "R", r_return) for a in atoms
            ):
                r_return.add(key)
                changed = True
            if key not in b_return and any(
                _atom_is(a, mi, fn, base_locals, b_reach, "B", b_return) for a in atoms
            ):
                b_return.add(key)
                changed = True

    # Covered: a function has at least one parameter that is R-reachable and NOT
    # B-reachable (the base-veto). The veto is applied here, once, on the settled
    # fixpoints — never mid-iteration, so growing b_reach can't leave a stale
    # coverage decision behind. base_reach is every function a base value reaches on
    # ANY parameter — the veto set impact_value applies over the FULL covered set
    # (including the base-blind one-hop scan), so a mixed applier can't slip through.
    covered = set()
    for (mi, name), rposs in r_reach.items():
        bposs = b_reach.get((mi, name), set())
        if any(p not in bposs for p in rposs):
            covered.add(name)
    base_reach = {name for (mi, name) in b_reach}
    # Resolver-wrappers: R-returning AND NOT B-returning. The B-return veto is what
    # keeps this fail-closed — a helper that returns a base value on any path the scan
    # can see is excluded, so its callers stay flagged. Name ambiguity is excluded the
    # same way (a call to the name could hit a non-wrapper definition); an excluded
    # wrapper only leaves callers flagged (a false positive), never hides a gap.
    wrapper_keys = {(mi, name) for (mi, name) in r_return
                    if (mi, name) not in b_return and len(defs.get(name, ())) == 1}
    wrappers = {name for (mi, name) in wrapper_keys}
    # Fail-OPEN surface: a blessed wrapper that ALSO has a return the analyzer cannot
    # confirm is resolved — neither a visible resolver/R-atom nor a base/B-atom (a base
    # return would have vetoed the bless outright). That is exactly a MIXED wrapper (R on
    # one path, can't-confirm on another): a cast-laundered base `base = raw as T; return
    # base`, an empty-literal fallback, or a resolved value the scan missed. The bless
    # rests on FAITH for that return, so we ENUMERATE the wrapper instead of leaving the
    # residual as silent prose. `impact_value` carries this to `ValueImpact.fail_open_
    # wrappers` so a value-drift audit NAMES every wrapper blessed on an unverifiable
    # return. A wrapper whose returns are ALL R has zero fail-open risk and is absent.
    faith = set()
    for (mi, name) in wrapper_keys:
        for a in returns.get((mi, name), []):
            if not _atom_is(a, mi, name, resolved_locals, r_reach, "R", r_return):
                faith.add(name)
                break
    return covered, base_reach, wrappers, faith
