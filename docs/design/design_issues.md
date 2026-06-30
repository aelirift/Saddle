# Design issues — saddle's own DKB

> **Status: running gap log (creator-facing).** This is saddle's analogue of the
> rayxiv4 DKB principle: *never paper over a structural gap with a point-in-time
> runtime fallback (tolerant default, hardcoded backstop, blocklist, swallow-and-log).*
> Keep the strict/correct behaviour at the site and **file the gap here** so the
> contract / parser / detector gets fixed at the source. Each entry: the gap, the
> live evidence, the root cause, the proper fix, and what (if anything) shipped as a
> downstream mitigation.

---

## Gap 1 — fork detection is lexical, not semantic (OPEN — the root cause)

**The gap.** Stage 2 (intent) catches "pick-drift" by replaying the dialog: the
agent OFFERS a fork (labelled options), the user PICKS one (a binding), and a later
action is compared against that binding by qualified choice-id. The *catch* is
exact and correct. But fork **creation** and **binding** are decided by surface
cues, not meaning:

- a fork is minted when an agent line looks like an offer — a prompt that ends in
  `?` followed by `a)`/`b)`/`c)` (or `1.`/`2.`) option lines;
- a binding is minted when a later short user message looks like a pick — a bare
  `a`, `2`, or a proceed-word (`go`/`proceed`) mapped to the recommended option.

Neither step understands whether the agent is *actually offering the user a
decision* (vs. quoting, illustrating a format, or enumerating non-options), nor
whether the user's text is *actually selecting an option* (vs. an unrelated
instruction that merely happens to be short or contain a letter).

**Live evidence (this harness's own ledger, 2026-06-28).** Three false-positive
`resolved` forks were sitting in `~/.saddle/saddle.db`, all from a single
supervised session:

| Fork | What it really was | What the parser made of it |
|---|---|---|
| `p189.f1` | the agent's own **illustrative** `a) in-memory LRU / b) redis / c) sqlite` example, written to demonstrate a *render format* | a real decision "offered" to the user; the user's `go` (meaning "proceed with the implementation") was bound as **picking option a)** |
| `p64.f1` | an agent message that happened to enumerate two safety facts | the user "picked 2" — where the "pick" text was the compaction-resume **system boilerplate** `"This session is being continued…"` |
| `p64.f2` | same shape, a different agent message | same false `"2"` pick from the same boilerplate |

**Why it matters (this is not merely cosmetic).** A false-positive fork over-reports,
which is "safe" in isolation — but `latest_binding` returns the **most-recent**
resolved binding, so a junk fork minted *after* a real decision **buries** the real
one. A buried real commitment is a missed real drift by displacement — the cardinal
sin. Over-reporting here is one short step from under-reporting.

**Proper fix.** Fork detection must be **semantic**. Reuse the same LLM seam that
Stage 2 already uses for `intent.history_drift` to judge two questions before a
fork/binding is minted:

1. *In this agent message, is the agent genuinely OFFERING the user a decision
   among enumerated options* — or quoting / illustrating / listing facts?
2. *Does this user message SELECT one of the currently-open fork's options* — or is
   it an unrelated instruction that merely looks pick-shaped?

Keep the deterministic equality check for the *catch* (that half is correct and
must stay exact). Gate only **creation** and **binding** on the classifier.

**Explicitly forbidden (DKB discipline).** Do **not** paper this over with a
point-in-time blocklist — e.g. "ignore lines beginning `This session is being
continued`", or "skip forks whose source is a system message". That is the exact
runtime backstop the DKB principle prohibits; it would silence today's three
samples and let tomorrow's differently-worded boilerplate straight through. Fix the
detector, not the symptom.

---

## Gap 2 — retired forks kept surfacing their pick forever (SHIPPED: X)

**The gap (now closed).** `latest_binding` selected the most-recent `resolved`
binding with **no check that its fork was still live**, so a junk resolved fork
re-surfaced its pick as a "standing commitment" on **every** turn — even after the
decision was off the table.

**Fix shipped.** `ForkStore.latest_binding` gained `live_fork_only` (default
`False`, preserving every other caller); `dialog.active_binding` passes
`live_fork_only=True`. A binding whose fork was explicitly **retired**
(`FORK_SUPERSEDED`) is no longer returned as the live commitment. A binding whose
fork row is merely **missing** is kept — saddle never deletes forks, so absence is
an anomaly, not a retirement. Implemented at the store seam (in-memory + the SQLite
`LEFT JOIN … AND f.status != superseded`), covered by
`test_superseded_fork_drops_its_binding_from_the_live_commitment` and its SQLite
twin.

**Live ledger swept.** The three Gap-1 false forks (`p189.f1`, `p64.f1`, `p64.f2`)
were superseded via the `set_fork_status` API; `active_binding` now correctly
returns `None` (there is no real standing commitment — `go` meant "proceed", not a
fork pick).

> Relationship to Gap 1: X stops a junk fork from surfacing **once it is retired**;
> it does **not** stop junk from being **minted**. Gap 1 is the upstream root cause
> and still needs the proper fix.

---

## Gap 3 — the commitment readout nagged instead of carrying context (SHIPPED: Y)

**The gap (now closed).** `intake_hook._render_binding` led with
`standing commitment: p1.f1.a — 'go' (via recommended, confidence 1.00)` — an
opaque locator plus a reminder that *a decision happened*, which nags without
helping the agent act.

**Fix shipped.** The readout now restates **what the chosen option was**, as a
directive: the proposal it answered, the committed option's substance foregrounded
(`the user chose a) → in-memory LRU   ← this is what to do now`), the unchosen
options demoted to a drift-guard aside (`not chosen (so you don't drift to them):
…`), and the qualified choice-id carried **last** as a citation hint
(`[ref p1.f1.a — cite it when you act]`). Method/confidence noise dropped. Intake
tests updated (`carry-forward` / `this is what to do now` replace `standing
commitment` / `← committed`).

> Relationship: Y makes the readout **useful**; X keeps it from showing **stale**
> junk; Gap 1 (semantic detection) keeps junk from being **minted**. Y and X are
> downstream mitigations of a problem Gap 1 owns at the source.

---

## Gap 4 — task instructions get promoted as standing directives (SHIPPED: durability gate at promotion)

**The gap.** A *standing directive* is meant to be a rule the user wants honored on
**every future** design ("no band-aids", "fail loud"). But directives are captured
with no gate distinguishing a *durable rule* from a *this-task instruction*, so
prompt-local commands — output-format requirements, target-project constraints,
one-off framing — get persisted into a project's `policy.json` as if they were
standing. The design gate (Stage 3, `design.audit_proposal`) then hands the whole
accumulated set to itself as **BINDING DIRECTIVES** and judges *every* later
approach against them, including unrelated work.

**Live evidence (`config/tenants/aeli/projects/saddle/policy.json`, 2026-06-28).**
The `aeli/saddle` project policy held **8** directives, *all* captured while saddle
was producing rayxiv4's audit-DAG / convergence-controller **designs** — none a
standing rule for developing saddle the tool:

| Directive (abridged) | What it really was |
|---|---|
| "Respond with only the JSON object that satisfies the schema…" | the output contract of **one** schema'd LLM call |
| "…remain game-agnostic… no coupling to rayxiv4" (×3) | a constraint on the **rayxiv4** orchestration design |
| "…slice the work so the design-to-code collapse is prevented" | framing for **that** orchestration task |
| "Limit the audit output to reporting drifts and bugs only…" | the spec of **one** audit-output task |

When the user later directed work **on saddle itself** (tuning these very gates),
Stage 3 flagged the approach for violating directive #10 ("respond with only the
JSON object"), #4/#6 ("explain rayxiv4 independence"), #5 ("slicing strategy"),
etc. — a textbook false positive: task instructions applied as standing law to an
unrelated task.

**Why it matters.** Each false directive is a false "design issue" every turn,
training the user to ignore the channel (the same failure mode Gap 1 guards). And
because directives only **accumulate** (global ∪ tenant ∪ project, no inverse until
now), the pollution is monotonic — it only ever gets worse.

**Proper fix (the source) — SHIPPED 2026-06-28.** Promotion now gates durability.
`intake.classify_directive_durability` (mirroring `intake._classify_scope`) judges
each candidate directive *standing rule* vs *task-local instruction* via the same
LLM seam Stage 2/3 use, and `orchestrator._promote_directives` persists ONLY the
standing ones — task-local instructions still guide the current run but are never
written to policy, and the skip is logged (never silent). The classifier fails
SAFE toward not-promoting (an unclassified item, or a failed call, defaults to
`task`), because over-promotion is the harm: a real standing rule left un-persisted
is recoverable (`saddle directives --add`), a one-off persisted forever is not.
Pinned by `tests/test_directive_durability.py` (split / unclassified-defaults-to-task
/ fail-safe / orchestrator-promotes-only-standing).

**Downstream mitigations also shipped (2026-06-28) — defense in depth, kept:**
1. **Applicability clause** in `design._SYS_AUDIT` (and the self-directed clause in
   `intent._SYS_HISTORY`): both gates now judge whether a directive/design actually
   governs the work in front of them, and treat self-directed work on the assistant
   itself as in-bounds — mirroring `intake._SYSTEM_SCOPE`'s "configuring this
   assistant counts as in_focus". A backstop: it can't reliably override a directive
   handed to a gate as "binding", which is why the data fix below was also needed.
2. **`policy.demote_directive`** (+ `saddle directives --remove`): the missing
   inverse of `promote_directive`, so a mis-captured rule can be curated OUT through
   the same locked, cache-busting path instead of a hand-edit. The 8 polluting
   `aeli/saddle` directives were cleared with it; the design gate then audited a
   self-directed approach clean (`considered=6 ok=True`).

> Gap 4 is now closed at the source (promotion gates durability), so new task
> instructions no longer repollute policy. The two mitigations remain as defense in
> depth: the applicability clause stops a mis-applied directive at the gate even if
> one slips through, and `demote_directive` curates out anything already there.
