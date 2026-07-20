# Saddle — Intent vs. Implementation Audit

**Date:** 2026-07-20 · **Method:** full read of the core modules (`dialog.py`,
`intent.py`, `orchestrator.py`, `intake.py`, `store.py`, `models.py`,
`doctrine.py`, `doctrine_hook.py`, `intake_hook.py`, `stop_hook.py`,
`design.py`, `council.py`, `mcp_server.py`, `recall.py`, `supervisor.py`,
`completion.py`, `codemap/checks.py`) against the owner's stated intent.

---

## The intent (the benchmark)

Saddle exists to absorb the *high-level thinking* so the owner supplies only
**intent + design path**. An LLM is deep in knowledge and can code/design, but
left alone it: (a) ships quick band-aid fixes instead of long-term
enterprise-grade design; (b) drifts every which way; (c) drops parts of a
multi-item prompt (6 asks → 3 done); (d) does not root-cause — it looks at one
surface layer instead of asking *"what caused THAT? then what caused that?
would this solve ALL cases? would it prevent recurrence forever?"*. Saddle is
meant to **proactively listen to + analyze the running conversation**, decide
what to do and how, apply the project's lessons / best-practices / anti-patterns
by reasoning *with* them, and only THEN talk to the LLM — preventing every drift
class (intent, context, code, design, selection, naming, verification).

## Verdict

Saddle's **architecture aspires to exactly this** — layered intake ("exhaustive,
nothing dropped"), an intent-drift stage, a doctrine gate that compiles lessons
to deterministic rules, a fork/pick drift tracker, and — critically — an
`orchestrate` decompose→design brain and a multi-lens `council`. But the
implementation has two structural failures that hollow out the intent:

1. **The brain is off the always-on path.** The rich pieces (`orchestrate`,
   `council`) run **only** when invoked voluntarily (CLI / MCP tools). The
   always-on hooks — which fire whether or not the agent cooperates — call the
   *lighter* primitives. So the strongest machinery is gated behind exactly the
   cooperation a drifting model withholds.
2. **Key mechanisms are inert or stubbed.** The item-completion ledger has no
   writer (`set_item_status` is never called). The "what is the user working on
   *now*" semantic matcher was explicitly deferred to an unbuilt "later layer";
   the commitment is a frozen fork-pick row that never re-evaluates against the
   live conversation.

Net: saddle today is a **disciplined array of reactive per-event hooks sharing a
durable ledger**, not the live-analysis brain the intent describes.

## Scorecard

| # | Dimension (from intent) | Verdict | One-line why |
|---|---|---|---|
| 1 | Live conversation model (proactive listening) | **PARTIAL** | Commitment = last explicit fork-pick; freezes through natural pivots; semantic "current goal" matcher deferred + unbuilt (`dialog.py:364-371`). |
| 2 | Exhaustive itemization (no dropped asks) | **PARTIAL** | Real itemize+audit convergence loop, but on non-convergence it ships partial with a cosmetic note; hook path runs only 1 audit pass (`intake.py:468-483`, `intake_hook.py:364`). |
| 3 | Item follow-through (all N tracked to done) | **MISSING** | `set_item_status` defined, **never called**; items accumulate OPEN forever; completion judged holistically, not per-item (`store.py:202-213`, `completion.py`). |
| 4 | Root-cause interrogation | **PARTIAL** | Single-pass diagnose prompt; the only loop resolves *fixes*, not *causes*; no 5-whys, no "prevent recurrence forever?" question (`design.py:83-99, 690-705`). |
| 5 | Drift taxonomy (7 classes) | **PARTIAL** | intent/code/selection MEET; design/naming/verification partial; **context drift has no detector** (see table below). |
| 6 | Quick-fix vs enterprise design | **PARTIAL** | Anti-band-aid push is real but **LLM-prose only**, no red/green enforce; strongest form (council) is voluntary (`design.py:188-224`, `doctrine.py:34-36`). |
| 7 | Lessons reasoned vs dumped | **PARTIAL** | Design path reasons *with* lessons; the general per-turn path just retrieves + injects text for the coder to maybe read (`recall.py:119-144`). |
| 8 | Supervision loop shape | **PARTIAL** | Three independent hooks + a durable ledger; the unifying `orchestrate`/`council` brain is quarantined on CLI/MCP (`orchestrator.py:266` called only from `cli.py`). |

### Drift-taxonomy detail (Dimension 5)

| Drift class | Detector | Verdict |
|---|---|---|
| intent | `history_drift` vs settled DKB (`intent.py:275`) | **MEETS** |
| context | *none* — only carry-forward re-injection | **MISSING** |
| code | conformance scan vs completeness surface (`design.py:948`, `codemap/checks.py`) | **MEETS** |
| design | `audit_proposal` band-aid/misread/uncovered (single pass) | **MEETS** (LLM-judged) |
| selection | `IntentTracker.check_action` equality, never-silent (`dialog.py:477`) | **MEETS** (strongest) |
| naming | design-time symbol-menu grounding only | **PARTIAL** |
| verification | completion gate over-claim catch | **PARTIAL** |

## The 5 deepest gaps (root-caused)

1. **No live goal model — the commitment is a frozen fork-pick ledger.** Root
   cause: a deterministic equality-checkable fork/pick ledger was chosen (to
   survive the context window) and the semantic "what's being worked on now"
   matcher was explicitly deferred to an unbuilt later layer; fork retirement is
   "never automatic" (`models.py:244`), so nothing re-evaluates the commitment
   against the live conversation. *Proven live 3×: it froze on `p5458`, on
   `p39`, and misread a deliberate resume-handoff as "declaring closure."*

2. **Item follow-through is schema-only.** Root cause: the `Item.status` model +
   `set_item_status` exist but have **no writer** anywhere; supervision judges
   whole-goal completion with one LLM verdict instead of tracing each decomposed
   ask to done/blocked. The "6 asks → all 6 tracked" guarantee has no runtime.

3. **Root-cause is a one-shot prompt attribute, not a driven dialectic.** Root
   cause: causal depth is encoded as a checklist the LLM is told to satisfy in a
   single diagnose call; the only iterative loop deepens *fixes*, not *causes*;
   recurrence-prevention was offloaded to DKB memory (remembered for *next*
   time) rather than *asked of this design*.

4. **The "analyze → decide → apply lessons → then talk" brain is disconnected
   from the always-on path.** Root cause: the coverage-preserving `orchestrate`
   fold and the two-critic `council` were built for the CLI/MCP surface where
   invocation is voluntary; the hooks call the lighter single-pass primitives.
   The best machinery is gated behind the cooperation a drifting model withholds
   — this one root cause *also* weakens gaps 2, 3, 6, and 7.

5. **Design-quality + drift enforcement is mostly LLM-graded prose, with holes.**
   Root cause: doctrine concedes "an LLM grading an LLM drifts" yet compiles only
   4 deterministic rules (delete + scope), leaving design taste, lessons-
   reasoning, and 4 of 7 drift classes on the very LLM-judgment tier it was built
   to distrust.

## Fix roadmap (highest leverage first)

- **R1 — Put the brain on the always-on path (closes/【weakens】 gaps 4, 2, 3, 6, 7).**
  Make the hooks call the *strong* primitives, not the light ones: `intake_hook`
  runs the full `orchestrate` decompose+coverage-fold; the design hook convenes
  the multi-lens `council` (or at least the root-cause lens) instead of a single
  `audit_proposal`. Do not depend on the agent to *choose* to call MCP tools.

- **R2 — Build the live goal model (gap 1).** Add the deferred semantic layer: a
  per-turn LLM step that reads the rolling conversation and either confirms or
  **supersedes** the frozen commitment when the topic has demonstrably moved,
  writing an automatic fork-retirement. The deterministic pick-ledger stays as
  the strong signal; the LLM layer keeps it *live*.

- **R3 — Make the item ledger real (gap 2).** Wire `set_item_status`: after each
  turn, an LLM/deterministic pass marks each open item addressed/blocked with
  evidence, so `todos` drains and "all N asks handled" becomes verifiable, not
  aspirational.

- **R4 — Turn root-cause into a driven loop (gap 3).** Replace the one-shot
  diagnose with an iterative interrogation: *why → why → would this cover ALL
  cases → would it prevent recurrence forever*, bounded, before a design is
  accepted.

- **R5 — Close the drift-taxonomy holes (gap 5).** Add a context-drift detector
  (post-compaction "does the agent still hold the goal?"), a cross-codebase
  naming/convention-drift check, and a real verification-drift prober ("was the
  claimed check actually run?").

**Sequencing note:** R1 is the root-of-roots — activating the existing brain on
the live path is more leverage than any new mechanism, because the good design
already exists; it just isn't wired to fire when it must.
