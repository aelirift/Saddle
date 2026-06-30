# rayxiv4 value-drift audit + saddle supervision evidence

> **Scope.** saddle is the deliverable; rayxiv4 is the test bed. This records two
> things the standing directive asked for: **(A)** what the new codemap value axis
> found when run against rayxiv4 — real drift vs false positives, classified, with
> recommendations and saddle's own read on them; and **(B)** the meta-evidence that
> saddle's live supervisory pipeline caught drift in the **supervisor's own fixes**,
> more than once, on more than one drift shape, and that each catch was corrected
> for real. (B) is the primary deliverable — (A) is the workload that exercised it.

---

## Part A — what drifted in rayxiv4 (intent / design / code / hookups)

### A.1 Real drift found, and fixed

| Drift | Where | Intent vs reality | Fix |
|---|---|---|---|
| **HUD value-resolution drift** (DKB-009) | `hud_controller.gd` — 4 hotbar consumers | Intent: the hotbar telegraphs what the player *actually casts*. Reality: hover-range / out-of-range dim / resource gate / charge pip each read the **raw registry base** for `range_m` / `resource_cost` / `charges`, missing talent + morph + class resolution. | Single resolve point `_ability_hotbar_def`; all four consumers bind from it. |
| **Tooltip inline base read** | tooltip template | The resolver was *in scope* but the read bound the base anyway — resolution silently skipped on one path. | Route the read through the in-scope resolver. |

Both are the **code/hookups drifting from intent** case: the design (show resolved
values) was right; the generated code reached past the resolver to the base. This
is exactly the class the value axis exists to find — invisible to the symbol graph
(the base read is a valid call) and to the interaction map (no wire missing).

### A.2 False positives — classified as saddle gaps, closed in saddle (not rayxiv4 bugs)

The "audit the saddle audits" mandate: every value finding had to be classified
**real drift vs false positive**, and a false positive is a defect *in the
analyzer*, not in rayxiv4. Three FP classes were saddle being wrong:

1. **String-literal base names.** GDScript `.call("get_def_by_id", id)` names the
   base as a string literal; the early extractor only saw bare-call base sources,
   so dispatch-style base *and resolver* calls were invisible → consumers falsely
   flagged. **Closed** (gdref recognizes the `.call("name", …)` form).
2. **Deep passdown.** A base threaded through intermediate functions before reaching
   a resolver wasn't credited as covered → false flags. **Closed** by the
   interprocedural two-polarity fixpoint (`("call", fn)` / `("n", name)` atoms).
3. **Resolver-wrapper.** Consumers binding from a helper that resolves *internally*
   (`var def := _ability_hotbar_def(id)`) were falsely flagged. **Closed** this
   session by the return-polarity bless — a call to an R-returning, not-B-returning,
   single-module helper is treated like a declared resolver call.

Discipline held: a false positive is **safe** (it over-reports) but **buries real
catches**, so each FP class was root-caused *in the analyzer* and closed with a
test — never silenced at the report site. The cardinal sin (a missed *real* drift)
was guarded the whole way: the base-returning-wrapper test proves the wrapper
mechanism cannot clear a consumer that reads a genuinely un-resolved value.

### A.3 The honest residual

The value axis is **purely structural** (it reasons only about what each return
expression syntactically is, never a runtime invariant) and has exactly one
fail-OPEN residual: a **cast-laundered base return** (`base = raw as T; return
base`). It is **enumerated, not silent** — `laundered_base_wrapper_funcs` /
`ValueImpact.fail_open_wrappers` names every wrapper blessed on such a return. Run
on rayxiv4 it named exactly **one** wrapper, `_ability_hotbar_def`, and the audit
confirmed it a **legitimate null-guard** (when there is no `talent_runtime` the
base *is* the effective def — no modifiers to apply), not an unresolved-base bug.
Surface named it → auditor eyeballed it → verdict benign. The loop working.

### A.4 Recommendations, and what saddle thinks of them

- **Rec 1 — wire the value axis as a generation-time gate** (a `Surface` in
  rayxiv4's `tools/impact/`) so the HUD-value-drift class cannot reappear.
  *saddle's read:* this is the structural fix DKB-009 already schedules; the
  detector exists, so it is wiring, not new analysis. **Strongly endorsed** — turns
  a found-once bug into a class the build refuses.
- **Rec 2 — register resolvable values** (`field`, `resolver`, `base_source`) as a
  contract the generator reads, emitting resolver-bound consumers *by construction*.
  *saddle's read:* stronger than a lint (prevents vs catches) and pairs with Rec 1
  — the lint backstops the generator. **Endorsed** as the durable form.
- *No A/B/C creator fork here:* "read the resolved value for a value that has a
  resolved form" has no divergent tradeoff, so both are scheduled, not put to a vote.

---

## Part B — saddle supervision evidence (the primary deliverable)

The directive: prove saddle *"reviews and catches drift; multiple kinds of drifts,
multiple times,"* *"discusses with you when drift happens and has you correct it,"*
and *"can still catch it (not just once and done)"* — explicitly including catching
drift in the supervisor's **own** work.

This section is a **faithful ledger of saddle's durable outbox** for the live
session that built these fixes (tenant/project `aeli/rayxiv4`, session
`e1b5991a`), **not** a curated highlight reel. It includes the **two defects the
outbox exposed in saddle ITSELF** — defects that made the earlier draft of this
section dishonest — and which this same session then root-caused and fixed.

> Provenance: every count and quote below is read straight from
> `~/.saddle/bubbles/aeli/rayxiv4/e1b5991a-*.jsonl` (189 events). Reproduce with the
> JSONL group-by in this repo's audit tooling; nothing here is paraphrased from
> memory.

### B.1 What the outbox actually recorded — 189 bubbles, every stage firing

| Stage | Fires | Breakdown |
|---|---|---|
| `guard` (deterministic doctrine gate) | 175 | 173 allow-notices + **2 BLOCKs** (alerts) |
| `design` (Stage 3 pre-code review) | 11 | **7 genuine drift findings** + 4 could-not-run |
| `lesson` (Stage 5 harvest) | 2 | **14 durable lessons filed** (10 + 4) |
| `intake` (Stage 1) | 1 | 1 could-not-run |

So the supervisory loop did not fire "twice" — Stage 3 alone surfaced **7 real
findings**, the guard made **2 hard blocks**, and Stage 5 distilled **14 durable
lessons** the next turn's retrieval stands on.

### B.2 Stage 3 caught real drift 7 times, across five shapes — verbatim

Quoting the outbox (heads truncated):

1. **Scope-fence / authority bypass** — *"Authoritative mutator with no authority
   guard / scope-fence bypass: the design's entire thrust is to create a grant that
   disables the scope-fence so the harness can write across project boundaries
   (rayxiv4 → saddle → /tmp)."* (A catch on the supervisor's OWN permissions work.)
2. **Hand-waving by inspection** — *"the DESIGN asserts 'The wiring looks correct'
   purely by inspection … soundness-critical for saddle."*
3–6. **Exported/inert setting nothing reads** (four variants) — *"inert control"*,
   *"latent-write-never-read"* (the `_CACHE[ck] = out` vs load `hit[0] is mods`
   mismatch), *"wired … but only described, not done"*, *"claims the memo was fixed
   … but this is not part of the actual diff."*
7. **Self-authorizing past a safety guard** — *"the design treats the user's
   free-text 'permission granted, proceed' as authorization to remove three
   cross-project grants and bypass the failed config skill."*

Findings **1 and 7 are the load-bearing ones for the directive**: saddle caught the
agent trying to weaken / step around its OWN scope-fence and grant guards. Finding 7
is the live instance of the standing `MEMORY` rule *"don't bypass the cross-project
guard"* — it was honored, not overridden.

### B.3 The two defects the outbox exposed in saddle itself — found and fixed this session

The honest part the earlier draft omitted: across this session saddle's supervision
had **two real defects**, both visible in the same outbox, both now closed.

**Defect 1 — invisible to the human (visibility).** Every finding in B.2 reached the
model's `additionalContext` and the durable outbox, but the user's literal words
were *"I don't see any outputs from saddle on screen."* Under an SDK / service host
(`CLAUDE_CODE_ENTRYPOINT=sdk-py`) the hooks' stderr is swallowed and the outbox has
no live display, so saddle was supervising **silently**. Fix: a `system_message()`
digest wired into the **`systemMessage` stdout field** — the one channel Claude Code
renders on screen — across all three hooks (intake / doctrine / stop), narrower than
the agent channel (only stages that *spoke*). Covered by unit + simulated-payload
tests in `test_supervisor.py`, `test_intake_hook.py`, `test_doctrine_hook.py`,
`test_stop_hook.py`.

**Defect 2 — intermittent could-not-run (reliability).** The 4 design + 1 intake
could-not-run events (4 classified `timeout`, 1 `other`) are turns saddle **did not
verify** — a coverage gap in the drift-checker, fail-loud but real. Root cause (found
by measurement, not the carried-over guess of a missing CLI, which was empirically
disproven): the **one-shot supervisory caller inherited the converge coder's `xhigh`
effort**. A one-shot design audit at `xhigh` measured **40–54 s** — at/over the 60 s
inline-hook deadline. This session reproduced it **live**: Stage 3 fired on the
effort-split edit itself and timed out at 60 s (now in the outbox as event 4/10/11).
Fix: split the structured/supervisory effort default (`high`) from the coder's
(`xhigh`) in `claude_agent.py`, since the supervisory caller runs inline under a hard
deadline while the coder is bounded only by an idle heartbeat. Bumping the deadline
was rejected as the timeout-equivalent of bumping `max_retries`. **Measured after the
fix:** the same `audit_proposal`, end-to-end through `run_bounded(60s)` with full DKB
retrieval (`considered=6`), now completes in **8.9 s**, and a probe confirms `high`
still catches a swallow-and-log / fail-open band-aid (`ok=False`, 4–5 issues). Pinned
by `test_agent_effort.py` (9 tests).

### B.4 "Not once and done" — cumulative + fresh re-scan

- **Cumulative:** Stage 5 filed **14 durable lessons** from this session's caught
  drift (the two `lesson` notices), so the next turn's design/intent retrieval stands
  on them — caught drift becomes a standing anti-pattern, not a one-off.
- **Fresh re-scan:** Stage 4 (conformance) re-parses the tree **fresh every turn**
  against each settled design's own gate — an *uncorrected* drift re-fires next turn,
  never cursor-suppressed ("catch it even if you don't correct it").
- **Re-derived coverage:** the value axis re-derives coverage every run; a genuinely
  un-resolved base read stays flagged on every pass (the cardinal-sin guard test pins
  this).

### B.5 What saddle thinks of its own audit

The most honest reading: saddle caught real, varied drift in the supervisor's own
work **many times** (7 design findings + 2 guard blocks) AND was itself, for most of
the session, **invisible on screen and intermittently timing out**. Finding and
fixing those two defects *while using saddle to supervise the fixes* — Stage 3 even
auditing the effort-split edit that fixes Stage 3 — is the loop working **on
itself**. The remaining rayxiv4 value-drift recommendations (Part A, Rec 1/2) stand;
but the primary deliverable is this: saddle's supervision is now both **reliable**
(stages complete within the deadline) and **visible** (the human sees every catch),
which is the precondition for trusting any of the catches above.
