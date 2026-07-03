# Saddle as the conversation front end (Option A) — mediator design

**Decision (user, 2026-07-03):** saddle fronts the conversation. The user talks
to saddle; saddle conducts the technical design discussion with whichever agent
is doing the work; the agent's raw action stream never reaches the user's chat.
This supersedes the sidecar-only posture (hooks whispering at an agent whose
output the user still reads raw).

Two follow-up requirements land with the commit:

1. **Plain-language voice** — everything saddle says to the user is in common
   language: no unexplained technical terms, no acronym soup, and no regional
   or subcultural idioms. A reader with no engineering background follows it.
2. **Multi-project awareness** — saddle is hooked to the *agent*, not to a
   project. It must know which project each piece of work belongs to, switch
   automatically when the user switches, and handle one prompt that spans two
   projects at once — filing lessons, drift checks, and designs to the right
   project ledger, never mixing them.

---

## 1. The two conversation planes

| Plane | Parties | Language | Content |
|---|---|---|---|
| **Front** | user ↔ saddle | plain, common words | what's being built, the choices that need the user, proof no request was dropped, short progress digests |
| **Back** | saddle ↔ agent | technical | design rounds, best-practice pressure-testing, drift checks, settlement, execution feedback |

The agent's tool-call firehose (greps, reads, edits, probes) belongs to
neither plane. It streams to the **inspector** — a log surface the user opens
only on demand. Same bubble-up principle as the creator panel: decisions on
the panel, plumbing in the inspector.

## 2. The mediator loop (one user turn)

1. **Intake** — decompose the prompt into typed items (existing Stage 1),
   now with a **scope tag per item** (§4).
2. **Settled check** — for each actionable item: is it covered by a settled
   design in its project's ledger, details and all?
   - Covered → no discussion. Straight to execution, watched by the
     conformance gate (Stage 4).
   - Not covered / contradicts settled → a **design round** is warranted.
3. **Design rounds (back plane)** — saddle and the agent iterate:
   agent proposes an approach; saddle audits it against the project's DKB
   (lessons, anti-patterns, settled decisions, binding directives); critique
   goes back; repeat until the proposal audits clean. Code edits stay gated
   (deny-until-settled) for the item's scope while the round is open.
4. **Settlement** — a clean proposal is persisted as a settled design with its
   completeness surface. Approval source is recorded:
   `approved_by: user` (the user picked / confirmed on the front plane) or
   `approved_by: converged` (automated lane — saddle and agent reached
   agreement per the audit). Once settled, the topic goes quiet: Stage 2 sees
   coverage, Stage 4 enforces details-match every turn, and detected drift
   **re-opens the round** instead of shouting into a log.
5. **Execution watch** — the agent works; guard + conformance run as today;
   findings feed the round or the digest, not the user's chat.
6. **Turn digest (front plane)** — saddle closes the turn with a short plain
   render: what was decided, what changed, what's still open, what needs the
   user. Items ledger shows requested-vs-done so nothing is silently dropped.

The user is interrupted mid-turn only for a genuine fork or approval —
rendered as one clean block: what's broken/needed in plain words, the options,
the tradeoff, saddle's recommendation.

## 3. Plain-language contract (the voice)

- **One chokepoint.** Every user-facing string — intake summaries, digests,
  decision blocks, alerts — renders through a single voice module. No stage
  prints to the user directly. (Bubbles already share one render contract;
  this extends it with the language pass.)
- **The rules the chokepoint enforces:**
  - common words; a technical term may appear only immediately after its
    plain-word introduction ("the gate — the check that blocks code edits
    until the design is agreed — …");
  - every noun defined at first use (standing user directive);
  - no idioms, slang, or regionalisms; no metaphors that assume a culture;
  - lead with the point (what happened / what's needed), detail after;
  - short. A digest fits on a screen without scrolling.
- **Enforcement, not aspiration:** LLM-generated renders carry the style
  contract in their system prompt *and* pass a bounded plain-language check
  (same fail-loud discipline as other stages — an unreadable render is a
  finding, not a shrug). Static templates are rewritten to comply once.
- Saddle's **current bubble prose fails this bar** (e.g. "RE-OPENS A CLOSED
  DECISION [Single play-test event standing in for code review evidence]").
  Migrating existing renders through the chokepoint is part of this work,
  not a later nicety.

## 4. Project scope: from "one focus" to scope routing

Today (`focus.py`): saddle observes ONE project, resolved once per process
from `$SADDLE_PROJECT` → git root → cwd. That model is retired.

**New model — saddle attaches to the agent; scope is resolved per item.**

- **Project registry.** Saddle maintains a small registry of projects it has
  seen: slug → root path(s) + descriptor. Learned automatically the first
  time a root shows up (confirmed with the user on the front plane the first
  time, so a typo'd path never becomes a phantom project).
- **Per-event resolution.** Every hook payload / front-plane turn resolves
  scope from evidence, strongest first: explicit statement in the prompt →
  file paths being touched → the agent's cwd/git root → the session's last
  dominant scope. Switching projects therefore needs no reconfiguration —
  the evidence moves, saddle follows.
- **Per-item scoping.** Intake tags each decomposed item with its project.
  A prompt like "audit rayxiv4's pipeline and check whether saddle catches
  the drift" yields rayxiv4-scoped items and saddle-scoped items in the same
  turn. The turn holds a **scope set**, not a single focus.
- **Per-scope stages.** Every downstream stage runs against the ledger of the
  item it serves: intent drift for a rayxiv4 item consults rayxiv4's settled
  designs; a lesson harvested from saddle-work files with
  `scope_project=saddle`. Storage already keys everything by
  `(tenant, project)` — the change is routing, not schema.
- **Guard fence.** The path fence judges writes against the union of the
  turn's active scope roots. A write outside every active scope warns (or
  blocks, for deletes) exactly as today — but a two-project turn no longer
  false-alarms on the second project.
- **Digests group by project.** A mixed turn's digest reads as two short
  sections, one per project, so lessons/decisions stay visibly separate.
- `focus.py` remains the single source of truth for scope identity — its
  contract widens from "the focus" to "the active scope set". Both consumers
  (guard, intake) keep importing from the one place so they cannot disagree.

## 5. Surfaces

- **Front chat** — where the user types. Target: the launcher chat panel
  (already mediator-shaped) and/or `saddle chat` REPL. Both render the same
  voice-module output.
- **Digest feed** — the per-turn plain summaries + open-asks board; durable
  (bubble outbox is the feed), rendered by panel, CLI, and MCP identically.
- **Inspector** — the agent's raw stream (tool calls, technical alerts,
  stage internals), on demand only.

## 6. What exists → what changes

| Piece | Exists | Change |
|---|---|---|
| Intake decompose (Stage 1) | ✅ | + per-item scope tag |
| Intent drift vs settled (Stage 2) | ✅ | route per item scope |
| Design audit (Stage 3) | ✅ one-shot | becomes iterative round via MCP `design_propose` / critique loop |
| Deny-until-settled gate | guard exists | + "unsettled design" rule, per scope |
| Settlement (`design_for`, surfaces, Stage 4 scan) | ✅ dormant | live sessions settle designs; drift re-opens rounds |
| Lesson harvest (Stage 5) | ✅ | file to item's scope; feed *briefings*, not just post-hoc audits |
| MCP server (`discuss`, `recall`, …) | ✅ unreachable | wire into agent hosts; add round-trip design tools |
| Turn-end findings | die in outbox | drain into next turn's context + digest |
| Voice/render | per-stage prose | single chokepoint + plain-language contract |
| Focus | one static project | registry + per-item scope set |
| Front surface | REPL exists; panel is mediator-shaped | route user ↔ saddle ↔ agent |

## 7. Delivery order (each slice usable on its own)

1. **Voice chokepoint + plain-language contract** — rewrite existing renders
   through it. Immediate readability win in today's sidecar mode.
2. **Scope routing** — registry, per-item tags, per-scope stage routing,
   fence update. Fixes lesson cross-contamination and out-of-focus noise.
3. **Round-trip design loop** — MCP wiring + deny-until-settled + settlement
   from live sessions. Stage 4 comes alive; discussions become real.
4. **Turn digest** — the distillation + open-asks board on the existing
   panel/CLI surfaces (Option B as a slice of A).
5. **Front chat** — the user's conversation moves to saddle; agent behind it.

## 8. Open questions (defaults noted, none blocking slices 1–2)

- **Front surface primacy** — launcher panel vs `saddle chat` REPL as the
  first real front plane? *Default: launcher panel* (already mediator-shaped,
  already renders bubbles).
- **Gate strictness during migration** — deny-until-settled on from day one,
  or warn-first behind a mode flag per host? *Default: mode flag, warn-first
  for one adjustment period, then deny.*
- **Registry confirmation UX** — confirm a newly-seen project with the user
  once, or auto-admit and surface in the digest? *Default: confirm once.*
