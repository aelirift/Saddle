# The topic mind-map — the unifying layer of the project rep

**Status:** design (2026-07-20). Slice #3 of the live-brain. Not yet built.
**Not a change to** `dsg_072c39f16c55` (audit-orchestration probe DAG) **or**
`dsg_55c51b7bcb70` (Convergence Controller) — those govern *code-surface*
convergence; this is a *conversation/topic* tracker in the cross-project brain
layer, which the owner fenced OFF from the convergence controller. (Saddle's
conflict hook flagged a collision on the word "DAG" + a wrong-layer application —
a false positive, recorded here so the record is straight.)

## What it is
The third tracking grain, and the parent structure that unifies the other two:

- **commitment** (#76) — the *one* active node.
- **item ledger** (#77) — discrete asks; they hang off topic nodes.
- **settled designs** — hang off topic nodes.
- **topic mind-map** (this) — the DAG of discussion threads that *contains* all of
  the above. Items / designs / intent are views over it.

It is a field of the project rep (`topics`), surfaced in the rep's "OPEN THREADS"
render slot.

## Structure — a DAG, in practice a tree, with conditional back-edges
- Parent→child **contains** edges form the map; almost always a tree
  (rayxiv4 → stages → substeps).
- **Conditional back-edges** allowed: a node may *flow back* to a prior node when a
  named condition holds — the "circular logic on special conditions" (e.g. a
  closed topic reopens *iff* a design/intent it depended on changes). The
  condition is recorded on the edge; the cycle is never automatic — it fires only
  when its condition is met and (like every structural change) is **proposed and
  user-confirmed**.

## Node types (typed enum — invalid kinds unrepresentable)
Each type carries its own **closure semantics**:

| type | closes when |
|---|---|
| `root` | top-level ask; closes when all children closed + user confirms, or dropped |
| `topic` | work-area; closes when children closed + user confirms, or dropped |
| `decision` | user rules on it → **collapses into parent's recorded context** (no lingering node) |
| `work` | built **and** tested **and** user signs off **with evidence** (see states) |
| `question` | user answers it |
| `finding` | informational; attaches to a topic, never "open" |

## `work` testing-state machine (the "build-and-move-on but stay open" rule)
A `work` node does **not** close on build. It tracks a testing state and is
**reminded, never silently accepted** (this is loud, not silent degradation):

```
none  →  built_untested  →  tested_unclosed  →  closed
                                  ↑ evidence: created tests / playwright /
                                    regression results attached
```
- `built_untested` and `tested_unclosed` are **reminder states**, not steady states
  — saddle nags with status + evidence-gap until the user explicitly closes.
- **Closure requires evidence** (a test/playwright/regression result) + the user's
  word. No evidence, no close.

## Status + reopen
- `status ∈ {open, closed, dropped}`.
- A closed/dropped node keeps a **compact summary**: *what we tried, why it
  closed / failed / is-not-applicable, and enough context to NOT re-explore it
  cold* ("we tried genetic-algo for the non-thorp distribution; failed on X;
  don't revisit unless Y changes").
- **Reopen** two ways: set a `reopen` flag on the node, OR spawn a **new node with
  the same `topic_key`** — chosen when a design/intent change makes a dropped topic
  applicable again. The back-edge condition drives the *suggestion* to reopen.

## Persisted store (new — the first genuinely new durable structure in the brain)
A SQLite store alongside the fork / item stores, project-scoped:

- `topic_node(id, tenant, project, title, type, topic_key, status,
  testing_state, recorded_context, reopen_flag, ts, updated_ts)`
- `topic_edge(parent_id, child_id, kind ∈ {contains, back_edge}, condition, ts)`

**Bounding:** `recorded_context` is capped (a few hundred tokens/node); closed
nodes' summaries are compacted; the **open set** is what reminders surface; the
**closed set** is queryable ("have we tried X? why did it fail?") but never nags.

## Engine (build later — mirrors livegoal/ledger)
A per-turn classifier over the conversation proposes DAG changes — open / advance /
split / merge / rename / re-parent / close-with-evidence / reopen — each
**proposed and user-confirmed** (nothing implicit). Structural remaps carry
children + recorded context to the new shape (the "stage 1 splits into predesign +
design" case).

## Reminder cadence — trigger-based (settled)
Surface open loose-ends **after a big build completes, while testing, and on
`resume`**. Not a fixed interval.

## Fit-test refinements folded in (validated on this session's real DAG)
1. `work` testing-state machine (build-and-move-on but stay open + evidence-gated close).
2. Node types with per-type closure semantics.
3. `decision` nodes collapse into parent context; closed topics keep a compact
   why-summary; reopen via flag or same-`topic_key` node.
4. DAG-capable but tree-common (this session was a pure tree); back-edges are the
   deliberate exception.
