# The supervisory pipeline — five live stages (gap-2)

> **Status: design contract.** This is the authoritative design the staged
> runner and the two hooks implement against. It is creator-facing (the harness's
> own architecture), not tenant-facing wisdom — the distilled *principle* behind
> it is seeded into the DKB separately (`knowledge_seed_supervision_is_staged`).

## What this is

saddle wraps an AI coding agent. One **turn** = a user prompt → the agent reads,
designs, edits, and runs commands → the turn ends. saddle's job is to watch that
turn and catch **drift** — divergence from (a) the user's actual intent, (b) the
project's established design / history, and (c) good engineering practice —
*while the turn is happening*, and to carry every finding to a human through the
client-agnostic outbox (`saddle.bubble`) so an AFK or non-TTY user still sees it.

## Why "five stages" and not one gate

The earlier framing of this gap treated supervision as a single
**proposal-vs-vision** check: *does the agent's proposed design match the
vision?* — fired **once**, just before code. That is real and necessary, but it
is only **Stage 3** of five. Drift enters a turn at five distinct points, and a
single pre-code gate is blind to the other four:

| # | Drift-entry point | If only Stage 3 existed, this slips through |
|---|---|---|
| 1 | The prompt is mis-/under-understood before any work starts | the agent solves the wrong problem; supervision itself silently didn't run when itemize failed |
| 2 | This prompt pulls against the project's own design / prior decisions / focus | scope-creep and "you picked a) then asked for b)" contradictions |
| 3 | The agent's design is a band-aid / misreads the root cause | the classic proposal-vs-vision miss |
| 4 | The **code** drifts from the design mid-implementation | "design said structural fix, the diff added a try/except swallow" |
| 5 | The lesson from this turn is never captured | the same class of drift recurs next turn — "caught once and done" |

Supervision must therefore be **staged across the whole turn**, each stage
watching one entry point, each emitting a `BubbleEvent`. The stage labels are
already the contract on `BubbleEvent.stage`: `intake` / `intent` / `design` /
`code` / `lesson` (plus `guard` = the deterministic doctrine pre-action gate, and
`dialog` = the back-and-forth correction channel the stages use when they need
the human).

## Key finding: the stages are mostly wiring over existing engines

This pipeline is **not** five new subsystems. Each stage reuses a rich primitive
that already exists; gap-2 is the *orchestration* that names the stages, fires
them at the right moment, makes each one bubble, and closes the specific holes.

| Stage | Existing engine (reuse) | The gap to close (the actual work) |
|---|---|---|
| 1 intake | `intake.decompose` (itemize → coverage-audit → focus-scope) | ✅ **DONE** — `intake_hook._itemize_outcome` now runs through `supervisor.run_stage` under a `run_bounded` deadline: a timeout/error is classified + bubbled as a LOUD ALERT (root-cause named) instead of the old silent fail-open swallow |
| 2 intent | `intake._classify_scope` (cross-project axis) + `dialog`/transcript replay (pick-drift axis) | ✅ **DONE** — `intent.history_drift` adds the **project/design/history** axis: it retrieves the project's settled `Design`s + closed decisions from the DKB and an LLM flags a prompt that contradicts a design, re-opens a decision, or creeps past focus. Wired as a second `STAGE_INTENT` `run_stage` in `intake_hook` (bounded, fail-loud); a clean compare is silent, a brand-new project skips the LLM call |
| 3 design | `design.design_for` — its `_SYS_AUDIT` already flags band-aids, hand-waving, anti-patterns, uncovered asks | ✅ **DONE** — `design.audit_proposal` reuses the exact `_SYS_AUDIT` over the **already-written** approach (lifted from the transcript by `transcript.latest_turn`), not a freshly-generated one. Wired into `doctrine_hook` as a `STAGE_DESIGN` `run_stage`, fired **once per turn** (per-session anchor marker) on the first code-mutating edit that the guard allows; an empty approach is the "no recorded design" ALERT, a band-aid is an ALERT, a clean approach is silent; infra fails open, the audit itself fails loud. A turn-end pass (in `stop_hook`) closes the **prose-proposal blind spot** — an approach proposed in prose with no edit — sharing the same per-turn marker so exactly one reviews a turn; and the audit **reply is a line contract** (`OK`/`ISSUES` + one issue per line), not fragile prose-in-JSON |
| 4 code | `design.intent_drift(design)` — re-runs the design's persisted `SurfaceManifest.gate()` against a **fresh** parse of the tree | ✅ **DONE** — `design.conformance_scan` gates the project's **settled** (`DESIGN_FINAL`) designs that declared a surface, latest-N, parsing the tree **once** and re-running each design's own `intent_drift` gate against that shared parse. Wired into the new `stop_hook` as a `STAGE_CODE` `run_stage` at **turn-end**; an error-grade `Finding` is an ALERT, a warn-grade gap a NOTICE, a scan blow-up a classified ALERT (fail-loud); no settled+surfaced design ⇒ silent. Re-catches an uncorrected drift every turn (fresh parse, never a cursor) |
| 5 lesson | `design._harvest` — files generalizable lessons into the DKB (`source=audit`) | ✅ **DONE** — `design.harvest_turn` reuses `_harvest` over the drift CAUGHT this turn (Stages 3 + 4, read from the durable outbox since a per-session watermark) and files deduped, project-scoped lessons. Wired into `stop_hook` as a `STAGE_LESSON` `run_stage` after Stage 4; a NOTICE names what was learned, a clean turn is silent (no LLM call), a harvest that throws is a classified ALERT (fail-loud), and the watermark advances so the same drift is not re-harvested next turn |

## The five stages in detail

### Stage 1 — Intake (the prompt is broken down properly)
- **When:** `UserPromptSubmit`, before the agent acts.
- **What:** `decompose` itemizes the prompt exhaustively (two-pass: itemize +
  coverage-audit), then runs the focus-scope gate. The actionable items become
  the persisted todo list.
- **The gap (the loud one) — ✅ CLOSED.** When the itemize LLM call failed — the
  recurring `TimeoutError` — `_itemize_section` *used to* return
  `"(saddle: itemization unavailable this turn — …)"` and let the turn proceed.
  That was a **silent fail-open**: supervision didn't run and nobody decided to
  skip it. Per the no-band-aid principle (`knowledge_seed_fail_loud`,
  `knowledge_seed_silent_fallback`) that swallow-and-degrade is exactly what we
  forbid.
- **The fix (implemented).** `_itemize_outcome` now drives `decompose` under a
  deadline via `supervisor.run_bounded` and lets a failure PROPAGATE to
  `supervisor.run_stage`, which (a) **classifies** it (oversize-input contract gap
  vs provider outage vs wall-clock — the `supervise` typed errors
  `DeadlineExceeded`/`Stalled` both subclass `TimeoutError` and now classify as
  `timeout`, fixed at the source in `retry_category.categorize_retry`), (b) emits
  a per-stage **ALERT** bubble naming the class and what saddle could not verify
  this turn, and (c) the hook joins that statement into the agent's combined
  `additionalContext` so the *agent* also knows supervision was incomplete. The
  hook still never *blocks* (exit 0 — observation, not enforcement). A *bounded*
  retry is reserved for an external rate-limit alone
  (`knowledge_seed_blind_retry`); every other failure surfaces its root cause
  instead of being retried or swallowed.

### Stage 2 — Intent drift (this prompt vs project / design / history)
- **When:** after intake, before design.
- **What:** does **this** prompt's intent pull against what the project already
  is? Three axes:
  - **cross-project** (built): `_classify_scope` flags a TASK that would modify a
    *different* project than the focus.
  - **pick-drift** (built): transcript replay + `dialog.IntentTracker` surface
    "you committed a), the agent acted on b)".
  - **project/design/history** (✅ implemented — `intent.history_drift`): retrieve
    the project's settled `Design`s + closed decisions from the DKB and ask an LLM
    whether this ask contradicts a settled design, re-opens a closed decision, or
    creeps past the project's stated focus. Only **this tenant's** settled state is
    weighed — divergence from the global seed corpus is Stage 3's job, kept
    distinct so Stage 2 means "this project's own history". A brand-new project
    with nothing settled returns silent **without an LLM call**; a failed check is
    a classified ALERT (fail-loud, like Stage 1), never a swallowed pass.
- **Bubbles:** a NOTICE for a soft pull (scope-creep); an ALERT for a HARD pull —
  a contradicted design or a re-opened decision (`IntentReport.level`) — plus the
  pick-drift axis's own ALERT and the intake stage's scope warning. The human
  confirms or corrects via the `dialog` channel — annotate, never auto-refuse
  (refusal is the guard's job).

### Stage 3 — Design review (the anti-band-aid gate) — ✅ DONE
- **When:** Stage 3 fires at **two** moments, sharing **one** per-turn anchor
  marker so **exactly one** of them reviews a given turn:
  - **Pre-edit (primary):** the agent has a design but **before** it writes code
    (PreToolUse on the first code-mutating tool call of the turn —
    `Edit`/`Write`/`MultiEdit`/`NotebookEdit`). It runs **only after the
    deterministic guard allows** the call; a denied call never reaches it. The
    "first of the turn" is enforced by a per-session anchor marker
    (`design_gate/<session>.json` keyed on the user prompt's uuid), so the gate
    fires once per turn and stays silent on every subsequent edit of the same turn
    — regardless of clean/drift/failure outcome.
  - **Turn-end (the prose-proposal blind-spot closer):** the pre-edit gate is
    EDIT-gated, so a turn that proposes an approach in **prose** and writes **no
    file** (a "here's my plan / which option?" turn, a pure discussion turn)
    triggers no edit and is never reviewed. The `Stop` hook always fires at
    turn-end, so it closes the spot: if the pre-edit gate did **not** already mark
    this turn AND the agent actually spoke an approach, it audits that approach
    through the **same** `audit_proposal` engine, then marks the turn. Because both
    read/write the one anchor marker, the pre-edit gate (if any edit fired) and the
    turn-end closer never both review a turn. A turn with no spoken approach is a
    pure tool-run / no-op, not a blind spot: it is marked and stays silent
    (`SADDLE_HOOK_DESIGN=0` disables the closer; the audit itself still fails loud).
- **What:** (a) assert a design **exists** — `transcript.latest_turn` lifts the
  approach the agent *spoke* this turn (assistant prose since the last user
  prompt; `thinking` is never read, matching the deterministic-drift
  discrimination). An empty approach means the agent jumped straight to an edit
  with no recorded design — itself the finding ("discuss approach before
  coding"). (b) Review it via `design.audit_proposal`, which reuses
  `design_for`'s **exact** `_SYS_AUDIT` stage over the already-written approach
  (not a freshly generated design): did the agent reach the **root cause** (not a
  symptom), does it drift from intent / the binding directives, is it a
  **band-aid** (tolerant default / hardcoded backstop / swallow-and-log) instead
  of a structural fix? The audit is handed the binding directives **and** the
  DKB's anti-patterns — and unlike Stage 2 it **includes the global seed corpus**
  (universal best practice), the deliberate complement to Stage 2's
  this-project-only history axis.
- **Bubbles:** the verdict, as a per-stage `STAGE_DESIGN` bubble and joined into
  the agent's `additionalContext` (observation, never a `permissionDecision` —
  Stage 3 does not block; enforcement stays the guard's). A band-aid, a misread
  problem, or no recorded design → ALERT + the `dialog` correction loop with the
  human; a clean approach is silent. **Infra fails open** (a missing transcript,
  a disabled gate via `SADDLE_HOOK_DESIGN=0`, a non-edit tool → the turn proceeds
  unjudged rather than wedging on saddle's own plumbing), but **the audit itself
  fails loud** (a raising/timed-out classify is a classified ALERT via
  `run_stage`, never a swallowed "looked fine"). This is the original
  proposal-vs-vision check, now correctly scoped as one stage of five.
- **Reply contract (the loud-accounting fix) — ✅ CLOSED.** The audit's issues are
  long free prose, and `_SYS_AUDIT` *used to* return them hand-escaped into a JSON
  array (`{"ok":…, "issues":[…]}`). That was the harness's most fragile parse — the
  same reason the design BODY rides as text: **one** unescaped quote in **one**
  issue throws the whole `json.loads` and discards **every** issue with it, which a
  bare `except` would read as a "sound" design — a fabricated pass, the cardinal
  false-negative. The fix replaces it with a **line contract** (`_parse_audit_reply`):
  the first non-blank line is the verdict (reduces to exactly `OK` or `ISSUES`, else
  it **raises** — a missing verdict, a preamble, or an issue crammed onto the verdict
  line is never read as `OK`); under `ISSUES`, every following line is one issue
  verbatim. No per-issue JSON escaping to break, no per-line marker to forget — a
  single ugly issue can never take the rest of the review down, and an unrecognizable
  verdict fails **loud** through `run_stage` instead of silently passing. (Its
  companion: `retry_category` now labels a residual `json.JSONDecodeError` from the
  JSON `call_json` sites — diagnose/surface/index/harvest — as `parse_error`, by
  type, so those parses surface a precise category too.)

### Stage 4 — Code conformance (the code matches the design) — ✅ DONE
- **When:** turn-end, via the new `Stop` hook (`saddle.stop_hook`) — a
  retrospective replay over the code as it stands when the agent finishes, fired
  **every** turn. There is no edit-cursor: the tree is re-parsed **fresh** each
  time, which is exactly what makes an *uncorrected* drift re-surface next
  turn-end instead of being blessed by a stale map.
- **What:** `design.conformance_scan` retrieves the project's **settled**
  (`DESIGN_FINAL`) designs that declared a completeness surface — latest-N
  (`SADDLE_HOOK_CODE_LIMIT`, default 8) — and re-runs each one's own
  `intent_drift(design)` gate (its persisted `SurfaceManifest` from
  `design.meta['surface']`) against the current tree. The surface is intent, the
  code is truth, the `Finding`s are the difference. Only **settled** designs are
  gated: a `DESIGN_FLAGGED` design never converged, so holding code to it would be
  premature. The tree is parsed **once** and the shared parse is handed to every
  design's gate (O(one parse), not O(designs)). The scan is strictly **read-only**
  — it never mutates a design or its status. Catches the agent that committed to a
  structural fix in Stage 3 and then quietly wrote the swallow anyway.
- **Bubbles:** a per-stage `STAGE_CODE` bubble naming each drifting design and its
  unsatisfied touchpoints (the offending **function**, lifted from the finding's
  `detail['func']`, not just a line number). The level tracks the severity the
  codemap **already classified**: an **error**-grade `Finding` (a hard
  propagation/liveness break the design promised against) is an **ALERT**, a
  **warn**-grade gap a **NOTICE** — no fuzzy re-judgement. A `Stop` hook has **no**
  agent-context injection channel (unlike `UserPromptSubmit`/`PreToolUse`), so the
  durable bubble outbox is the read path, with a human-readable copy on **stderr**
  for an interactive TTY. The hook is **observe-only — exit 0 always, no stop
  decision**: a *blocking* `Stop` would force the agent to keep going, which is
  enforcement, not observation. Failure regimes mirror the other hooks: a
  HOOK-level infra error (unparseable payload, ctx resolution, no code root) fails
  **open** (exit 0 + a stderr log — saddle cannot speak for a turn it never
  resolved), but a STAGE that cannot run (a parse blow-up) fails **loud** through
  `run_stage` — a classified ALERT naming what saddle did **not** verify, never a
  swallowed pass. No settled+surfaced design, or no code root ⇒ nothing can drift
  ⇒ silent (a non-empty result is always a real divergence).
- **Scope note (deliberate).** In the *pure supervised* flow, Stage 3 reviews the
  agent's prose via `audit_proposal` and mints **no** `SurfaceManifest`, so Stage
  4 finds nothing to gate and stays silent — correct, not a gap. Stage 4 does its
  real work when designs were **formally produced** (the `design_for` /
  orchestrator path persists a surface), which is precisely the rayxiv4-audit use
  case: mint surfaced designs, then let Stage 4 verify the live code against them
  every turn-end.

### Stage 5 — Lesson harvest (lesson learnt) — ✅ DONE
- **When:** turn-end, after Stage 4, in the same `stop_hook` (a `STAGE_LESSON`
  `run_stage`).
- **What:** `design.harvest_turn` distils what THIS turn taught into durable,
  **tenant/project-scoped** DKB lessons (`source=audit`), deduped against existing
  titles, by reusing the EXACT `_harvest` the design pipeline runs over its own
  caught flaws — so a flaw the live supervisor catches teaches the DKB the same
  way `design_for`'s do. The material it learns from is **this turn's caught
  drift, read back from the durable outbox**: Stage 3's band-aids
  (`meta['issues']` on the design bubble) and Stage 4's conformance breaks
  (`meta['drifts'][*]['findings']` on the code bubble). Intake infra-failures and
  intent pulls are **deliberately excluded** — they are not generalizable design
  lessons. The turn boundary is a **per-session watermark** (the last turn-end's
  timestamp): only drift bubbled since it is harvested, so a lesson is filed once,
  not re-filed every turn an uncorrected drift is re-caught (the title-dedup is a
  second backstop). A clean turn (nothing caught) makes **no LLM call** and stays
  silent.
- **Bubbles:** a NOTICE naming the lessons written (with the count + titles on the
  bubble `meta` for a richer client), plus a stderr copy. A harvest that throws is
  **not** swallowed — it PROPAGATES to `run_stage` as a classified `STAGE_LESSON`
  ALERT (fail-loud), never a false "learned nothing". This is what makes
  supervision **cumulative** rather than "caught once and done": next turn's Stage
  2/3 retrieval stands on what this turn filed.

## Orchestration

- **One staged runner** (`supervise`-adjacent module) owns the five stages as
  named, individually-bubbling steps, shared by both hooks so the policy lives in
  one place, not re-rolled per caller. Each stage is independently
  **fail-classified** (Stage 1's discipline generalized): a stage that cannot run
  says so loudly via a bubble; it never silently no-ops and implies success.
- **Hook mapping:**
  - `UserPromptSubmit` (`intake_hook`) → **Stage 1 + Stage 2**.
  - `PreToolUse` (`doctrine_hook`) → the **guard** (deterministic, already live) +
    **Stage 3** on the first code-mutating call.
  - **Turn-end** (`Stop`/retrospective replay) → **Stage 3 (the prose-proposal
    blind-spot closer, only if the pre-edit gate did not already review this turn)
    + Stage 4 + Stage 5**.
- **"Not once and done":** the stages fire **every turn** and are idempotent. If
  the human corrects a Stage-3 band-aid and the agent still ships drift, Stage 4
  catches it at turn-end; if a lesson is harvested, the *next* turn's Stage 2/3
  retrieval re-applies it. Re-catch across stages and across turns is the design,
  not a special case.
- **Bubble levels:** `info` = background; `notice` = "saddle wants this read" (a
  stage result, a commitment); `alert` = "saddle needs a correction/decision —
  surface loudly." A turn whose Stage 1 failed, or whose Stage 3 caught a
  band-aid, is an `alert` turn.

## Boundaries (what this is NOT)

- The runner is **presentation + orchestration**; the *verdicts* stay owned by
  the engines (`intake`, `dialog`, `design`/DKB). A bubble carries a finding; it
  is not itself the judgment (mirrors the `BubbleEvent` contract note in
  `models.py`).
- Enforcement (DENY) stays the **doctrine guard's** job; the five stages
  **observe and surface** (and open a `dialog`), they do not block the turn —
  except the deterministic guard that already does.
- Opaque, undecidable inputs (an interpreter fed `python -c`, a heredoc) are
  **not** asserted on by the deterministic guard; surfacing them is precisely
  what the supervisory stages are for.

