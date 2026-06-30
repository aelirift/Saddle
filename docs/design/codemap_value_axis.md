# The codemap value axis — value-drift impact analysis

> **Status: design contract.** This is the authoritative design the value-axis
> analyzer (`saddle.codemap.impact.impact_value`, its passdown fixpoint, and the
> two language frontends `pyref` / `gdref`) implements against. It is
> creator-facing (the harness's own machinery), not tenant-facing wisdom.

## What this is

The codemap has three axes. The **symbol/AST** axis answers *what calls what*; the
**interaction** axis answers *what wires to what*. The **value axis** — this doc —
answers a sharper question:

> When a value has a *resolved* form (base data run through a resolver that layers
> on modifiers/overrides/effects), does every consumer read the **resolved** value,
> or does some consumer read the **raw base** and silently miss the resolution?

That bug — a consumer binding the raw base where it should bind the resolved
value — is **value drift**. It is invisible to the symbol axis (the raw read is a
perfectly valid call) and invisible to the interaction axis (no wire is missing).
Only a value-aware pass catches it. The motivating real case: a HUD whose hover
range telegraph, out-of-range dim, resource gate, and charge pip each read an
ability's raw registry base instead of the talent/morph/class-resolved effective
def — so the player sees numbers for an ability they are not actually casting.

### The model

A `ValueSpec` declares one value's resolution shape:

- `name` — human label (e.g. `"ability_def"`).
- `field` — the field whose resolved-vs-raw split matters.
- `resolvers` — the functions that *produce the resolved form* (e.g.
  `resolve_def`, `apply_ability_modifiers`).
- `base_sources` — the functions that *produce the raw base* (e.g. `get_def_by_id`).
- `producers` / `accessor` — where the value originates / how it is read.

`impact_value(mods, spec) -> ValueImpact` walks every module and reports each site
that reads the raw base **without being covered** by a resolver. A covered site is
fine; an uncovered base read is flagged as drift.

## The two coverage mechanisms (do not conflate them)

Coverage — "is this base read actually safe because a resolver governs it?" — is
decided by **two independent mechanisms** in `impact.py`. They have different
reach and different veto rules, and the distinction is the crux of the whole axis:

| | Mechanism 1 — resolver-call-in-scope | Mechanism 2 — `arg_covered_funcs` |
|---|---|---|
| **What it means** | a resolver is *called in the same function* as the base read | the base value *flows as an argument into* a resolver, interprocedurally |
| **How computed** | `_covering = {(r.path, r.func) for r in resolver_sites}` | passdown fixpoint over the cross-module call graph |
| **Flow sensitivity** | flow-**insensitive** (anywhere in the function counts) | flow-following (argument reachability) |
| **Base veto** | **NOT** vetoed | **base-VETOED**: `arg_covered_funcs -= passdown.base_reachable_funcs` |

`_is_covered(r, cov)` is the OR of the two:
`(r.path, r.func) in _covering` **or** `r.func in arg_covered_funcs`.

The veto asymmetry is deliberate. Mechanism 1 says "a resolver runs right here, so
the author is clearly resolution-aware in this scope" — a local, syntactic fact
that needs no veto. Mechanism 2 reasons across function boundaries, where a value
can reach *both* a resolver and a raw base sink on different paths; there it must
fail **closed** on any base-reaching path, hence the veto.

## The passdown fixpoint — interprocedural two-polarity reachability

`passdown.py` runs an **existential, two-polarity** fixpoint over the call graph.
For every function it asks two questions independently:

- **R-reach:** can a *resolved* value reach this point? (some path runs a resolver)
- **B-reach:** can a *raw base* value reach this point? (some path runs a base source)

Both can be true at once — that is the whole reason the veto exists. The fixpoint
operates on **atoms** extracted per call-argument and per return-expression. An
atom is one of:

| Atom | Meaning |
|---|---|
| `"R"` | an inline resolver call (`resolve_def(...)`) |
| `"B"` | an inline base-source call (`get_def_by_id(...)`) |
| `("n", name)` | a bare identifier — resolve via the function's `call_locals` binding |
| `("call", fn)` | a bare DIRECT call to a project function — carries *that function's* return polarity |
| `None` | none of the above (a literal, a compound expression, an unknown) |

The atom vocabulary is **purely structural**: it reasons only about what each
expression *syntactically is*, never about any runtime invariant of the tenant
code. This is what makes the analysis sound to port across languages — `pyref`
(Python, AST-based) and `gdref` (GDScript, line/regex-based) emit the *same* atom
vocabulary, and the fixpoint is language-blind.

## The three false-positive classes closed

Three distinct shapes were producing **false drift flags** — a consumer marked as
reading raw base when it was in fact safe. Each was root-caused to a missing piece
of coverage reasoning, not patched at the report site.

### 1. String-literal base names (gdref)

A GDScript `.call("get_def_by_id", id)` names the base source as a **string
literal**, not a bare identifier. The early extractor only recognized bare-call
base sources, so a dispatch-style base read was invisible — and worse, a *resolver*
dispatched the same way was invisible too, so its coverage was lost and consumers
downstream were falsely flagged. Fixed by teaching `gdref` to recognize the
`.call("name", ...)` dispatch form for both resolvers and base sources.

### 2. Deep passdown (both frontends)

A base value passed *through one or more intermediate functions* before reaching a
resolver was not credited as covered — the original coverage was single-hop. The
passdown fixpoint closes this: `("call", fn)` and `("n", name)` atoms let polarity
propagate across as many hops as the call graph has, to a fixpoint.

### 3. Resolver-WRAPPER (this session's extension)

The sharpest class. A project helper that *is itself a resolver in effect* —

```gdscript
func _ability_hotbar_def(ability_id):
    var raw_def = _registry.call("get_def_by_id", ability_id)
    var base_def := raw_def as Dictionary
    if _has_runtime():
        return _runtime.call("apply_ability_modifiers", ability_id, base_def)  # R
    return base_def                                                            # laundered base
```

— is a **resolver-wrapper**: it R-returns on some path and is defined in exactly
one module. Four hud_controller consumers bind `var def := _ability_hotbar_def(id)`
and were *all* falsely flagged, because the analyzer saw them binding a value from
a plain helper, not from a declared resolver.

The fix treats **a call to a wrapper exactly like a declared resolver call** —
mechanism 1, flow-insensitive, not vetoed — because that is structurally what it
is. `resolver_wrapper_funcs(mods, resolvers, base_sources)` returns the wrapper
set from the fixpoint; `impact_value` then extends `resolver_sites` with every
call to a wrapper:

```python
wrappers = passdown.resolver_wrapper_funcs(mods, resolver_set, spec.base_sources)
if wrappers:
    for m in mods:
        for w in wrappers:
            imp.resolver_sites.extend(refs.calls_to(m, w))
```

It introduces **no new FN class** and changes nothing about base reads — it only
recognizes one more shape of "a resolver runs here."

## Return polarity is purely structural — and the fail-OPEN residual

A function is a wrapper only if it is **R-returning and NOT B-returning**, defined
in one module:

```python
wrapper_keys = {(mi, name) for (mi, name) in r_return
                if (mi, name) not in b_return and len(defs.get(name, ())) == 1}
```

The **B-veto fails CLOSED on every base return the scan can syntactically see.**
If a function returns `get_def_by_id(id)` on any path, it is B-returning, never
blessed, and its consumers stay flagged — this is the **cardinal-sin guard**, and
it has a dedicated test (`bad_wrap` returning `get_base(aid)` stays unblessed,
`cons_bad` stays flagged). The wrapper mechanism categorically cannot clear a real
base read.

There is exactly **one fail-OPEN residual**: a **laundered base**.

```gdscript
var base_def := raw_def as Dictionary   # a cast hides that this is the raw base
return base_def                          # the scan sees a bare local, not a base call
```

The cast launders the raw base through a local whose *binding expression* is no
longer a base-source call, so the structural scan cannot see the base origin. A
wrapper that R-returns on one path and returns a laundered base on another is
therefore blessed despite having a raw-base return path.

### The residual is ENUMERATED, not just documented

Documenting a fail-open hole as prose would let a real value-drift defect ship
while the analysis reads healthy. So the residual is **surfaced by a detector**,
not narrated. `laundered_base_wrapper_funcs(mods, resolvers, base_sources)` returns
every **blessed wrapper that also has a return the fixpoint could not confirm is
resolved** — i.e. a MIXED wrapper (R on one path, can't-confirm on another). The
computation is exact:

```python
faith = set()
for (mi, name) in wrapper_keys:            # the blessed wrappers
    for a in returns.get((mi, name), []):
        if not _atom_is(a, mi, name, resolved_locals, r_reach, "R", r_return):
            faith.add(name)                # at least one return rests on faith
            break
```

A wrapper whose returns are **all** provably-R has zero fail-open risk and never
appears — the surface is the residual *exactly*, not noise on every wrapper.
`impact_value` carries it on `ValueImpact.fail_open_wrappers`, so a value-drift
audit **names** every wrapper whose bless rests on an unverifiable return.

This is why the residual is **accepted, not hidden**:

- It requires a **mixed** wrapper — a pure laundered-base function never R-returns,
  so is never blessed.
- The analyzer **cannot structurally distinguish** a legitimate null-guard
  fallback (the common, correct pattern — return the unmodified base when there
  are no modifiers) from a buggy unresolved-base return. Forcing a flag here would
  re-introduce false positives on every correct null-guard wrapper.
- So the strict/correct structural rule stays, AND the detector enumerates the
  exact wrappers a human must eyeball.

This is the value-axis instance of the project doctrine: keep the strict behaviour
at the site, do not paper a structural limit with a runtime guess, **surface** the
gap as an auditable list.

### The detector on real rayxiv4 code

Run against the fixed `hud_controller.gd` template, for all three resolved fields:

```
[range_m       ] consumers flagged: -none-   |  fail-open wrappers: ['_ability_hotbar_def']
[resource_cost ] consumers flagged: -none-   |  fail-open wrappers: ['_ability_hotbar_def']
[charges       ] consumers flagged: -none-   |  fail-open wrappers: ['_ability_hotbar_def']
```

The four consumers are clear (the wrapper fix works), and the detector names the
**one** real mixed wrapper — `_ability_hotbar_def` — blessed on its `return
base_def` fallback. Auditor verdict on that one named wrapper: it returns the
talent/morph-resolved def when `talent_runtime` is present (the R path) and the
unmodified base only when there are no modifiers to apply (non-class game, or
before the autoload registers) — a **legitimate null-guard where base == effective**,
not an unresolved-base bug. The surface named it; the audit confirmed it benign.
That is the loop working as designed.

## Known residuals (the honest ledger)

| Residual | Axis | Status | Why it stands |
|---|---|---|---|
| **Laundered-base return** | both | **enumerated** via `fail_open_wrappers` | a `base = raw as T; return base` cast hides the base origin from a structural scan; flagging it as drift would false-positive every legitimate null-guard wrapper, so it is surfaced as an audit list instead |
| **Name-granular coverage** | mechanism 2 | bounded | `arg_covered_funcs` is keyed by function *name*, not `(path, name)` — a name collision across modules over-credits coverage. Mechanism 1 is `(path, func)`-precise; the wrapper bless requires single-module definition, which bounds this |
| **Line-scoped GD extraction** | gdref | conservative | GDScript has no AST; `gdref` reasons per logical line via regex, so a multi-line call argument or unusual formatting can miss an atom. A missed atom yields `None`, which never *grants* coverage |

## Found and fixed along the way

- **Dead coverage memo.** `_coverage` wrote a bare result-tuple to `_CACHE[ck]`
  while the load guarded on `hit[0] is mods` — comparing the *covered set* to the
  mods object, which is never identity-true, so the memo never hit and the fixpoint
  recomputed on every call. Correctness was never at risk (a dead cache is just
  slow), but it was a real bug found during the 237-module rayxiv4 audit. Fixed to
  store `(mods, out)` so the identity guard fires; pinned by
  `test_coverage_memo_is_live_and_identity_guarded`, which asserts a repeat call
  returns the *same object* and a fresh parse does not.
- **Earlier passdown atom bugs** (prior session): the Assign handler bound
  `call_locals` only on the `ast.Name` callee branch; the Return handler appended
  the return atom per function. Both are now covered by the wrapper/chain tests.

## Test coverage

`tests/test_codemap_passdown.py` pins the axis end-to-end, per frontend (PY_/GD_):

- **Positive wrapper** (`*_WRAP`): a hotbar_def wrapping `resolve_def` blesses its
  consumers (`cons_a` / `cons_b` clear).
- **Cardinal-sin guard** (`*_WRAP_BASE`): `bad_wrap` returning a raw base stays
  unblessed; `cons_bad` stays flagged. The mechanism cannot clear a real base read.
- **Multi-hop chain** (`*_WRAP_CHAIN`): `outer` returns `inner(...)`; both blessed.
- **Name ambiguity** (`test_name_ambiguous_wrapper_not_blessed`): a wrapper name
  defined in two modules is not blessed (single-module bless rule).
- **Laundered-base residual** (`GD_WRAP_LAUNDER` /
  `test_gd_laundered_base_return_is_known_fail_open_residual`): pins the fail-OPEN
  — a cast-laundered base return escapes the veto and the wrapper IS blessed.
- **Residual is enumerated**
  (`test_laundered_base_wrapper_is_enumerated_not_silently_blessed`): the mixed
  wrapper is named by `laundered_base_wrapper_funcs` and carried on
  `ValueImpact.fail_open_wrappers`; a clean all-R wrapper is absent (no noise).
- **Memo liveness** (`test_coverage_memo_is_live_and_identity_guarded`).
