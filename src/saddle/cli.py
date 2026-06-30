"""saddle command-line entry point.

A small subcommand dispatcher over saddle's surfaces:

    saddle                          # interactive agentic chat (default)
    saddle chat                     # same, explicit
    saddle intake "<prompt>"        # Layer 1: decompose + record a prompt
    saddle design "<prompt>"        # Layer 1 + Layer 2: decompose → fold → design
    saddle todos                    # show the open todo backlog
    saddle directives               # list this project's standing rules
    saddle directives --add "<r>"   # promote a standing rule (--scope global/tenant/project)
    saddle directives --remove "<r>"# curate out a standing rule (--scope ...)
    saddle lesson "<text>"          # record a DKB design-wisdom entry by hand
    saddle remember "<text>"        # remember a project FACT (durable memory)
    saddle remember "<text>" --kind reference --source <path|url> --ttl 86400
                                    # cache a file/web lookup (bounded, evictable)
    saddle kb                       # list visible DKB entries
    saddle kb --search "<query>"    # hybrid-search the DKB (semantic ∪ keyword)
    saddle kb --seed                # load the global seed corpus (idempotent)
    saddle kb --gc                  # evict expired/over-cap cache entries
    saddle codemap <design_id>      # Layer 3: gate a design's surface against code
    saddle codemap --symbols        # dump the project's grounding symbol menu
    saddle converge <design_id>     # Layer 4: drive the coder to satisfy the design's surface
    saddle audit                    # audit a project: grounded probes -> findings + coverage
    saddle audit --list             # enumerate everything that CAN be audited (no LLM)
    saddle guard --verb <v> --target <p>   # Layer 0: pre-action doctrine gate (exit 2 = BLOCK)
    saddle cross-project list              # list cross-project authorization grants
    saddle cross-project grant <root>...   # authorize work spanning these project roots
    saddle cross-project revoke            # revoke all cross-project grants
    saddle bubble                          # read saddle's outbound voice (the bubble outbox)
    saddle bubble --level alert            # only the messages that need a decision
    saddle bubble --session <id> --json    # one session's bubbles, machine-readable

Addressing: every non-chat command takes ``--tenant`` / ``--project`` to
address a specific tenant+project; omitted, they resolve from the environment
(SADDLE_TENANT / SADDLE_PROJECT) and the current working directory.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys

from saddle.context import Context, resolve

_KB_KINDS = ["best_practice", "anti_pattern", "lesson", "principle", "fact", "reference"]
# Kinds the `remember` command writes — the MEMORY family (a durable fact or a
# reconstructible cache entry), as opposed to `lesson`'s design-wisdom kinds.
_MEMORY_KINDS = ["fact", "reference"]
_SCOPES = ["global", "tenant", "project"]
_BUBBLE_LEVELS = ["info", "notice", "alert"]


# -- progress narration ------------------------------------------------------
# saddle's layers (orchestrator, design, converge, the LLM pool) already emit
# their progress via ``logging`` at INFO. Nothing surfaced it: a bare CLI run
# installs no handler, so Python's last-resort handler drops INFO and every long
# command looked like a frozen black box. Narrate by default — INFO to stderr,
# results stay on stdout (so ``--json`` piping is unaffected) — with ``-q`` to
# silence and ``-v`` for debug.

class _LogFmt(logging.Formatter):
    """Clean narration: bare message for INFO, a ``[warning]``/``[error]`` tag
    above it, with any traceback appended."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            msg = f"[{record.levelname.lower()}] {msg}"
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        return msg


def _setup_logging(verbosity: int) -> None:
    """Route saddle's own loggers to stderr. ``verbosity``: <0 quiet (WARNING+),
    0 narrate (INFO), >0 debug. Scoped to the ``saddle`` logger so third-party
    INFO (httpx, the SDK) stays out of the narration."""
    if verbosity < 0:
        level = logging.WARNING
    elif verbosity == 0:
        level = logging.INFO
    else:
        level = logging.DEBUG
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LogFmt())
    log = logging.getLogger("saddle")
    log.handlers[:] = [handler]
    log.setLevel(level)
    log.propagate = False


def _verbosity(args: argparse.Namespace) -> int:
    """-1 quiet / 0 narrate / +n debug, from the global ``-q`` / ``-v`` flags.
    Both use ``SUPPRESS`` defaults so they land on the namespace from either
    side of the subcommand without clobbering each other."""
    if getattr(args, "quiet", False):
        return -1
    return getattr(args, "verbose", 0) or 0


def _emit_stderr(text: str) -> None:
    """Write a streamed coder chunk straight through to stderr, live."""
    sys.stderr.write(text)
    sys.stderr.flush()


def _run_chat() -> int:
    from saddle.chat import main as chat_main
    return chat_main()


def _read_text(arg: str | None) -> str:
    if arg is None or arg == "-":
        return sys.stdin.read().strip()
    return arg.strip()


def _add_ctx_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tenant", default=None)
    p.add_argument("--project", default=None)


def _scope_pair(ctx: Context, scope: str) -> tuple[str, str]:
    """(scope_tenant, scope_project) for a DKB entry placed at ``scope``."""
    if scope == "global":
        return ("", "")
    if scope == "tenant":
        return (ctx.tenant, "")
    return (ctx.tenant, ctx.project)


# -- Layer 1 -----------------------------------------------------------------

def _run_intake(args: argparse.Namespace) -> int:
    from saddle.intake import decompose, format_intake

    prompt = _read_text(args.prompt)
    if not prompt:
        print("intake: empty prompt (pass text or pipe via stdin)", file=sys.stderr)
        return 2
    ctx = resolve(args.tenant, args.project)
    intake = asyncio.run(decompose(prompt, ctx, max_audits=args.audits))
    if args.json:
        print(json.dumps(dataclasses.asdict(intake), indent=2))
    else:
        print(format_intake(intake))
    return 0


# -- Layer 1 + Layer 2 (the orchestrator) ------------------------------------

def _run_design(args: argparse.Namespace) -> int:
    from saddle.orchestrator import orchestrate, format_orchestration

    prompt = _read_text(args.prompt)
    if not prompt:
        print("design: empty prompt (pass text or pipe via stdin)", file=sys.stderr)
        return 2
    ctx = resolve(args.tenant, args.project)
    orc = asyncio.run(
        orchestrate(
            prompt, ctx,
            run_designs=not args.no_designs,
            max_audits=args.audits,
            retrieve_k=args.retrieve_k,
        )
    )
    if args.json:
        print(json.dumps(dataclasses.asdict(orc), indent=2))
    else:
        print(format_orchestration(orc))
    return 0


# -- todo backlog ------------------------------------------------------------

def _run_todos(args: argparse.Namespace) -> int:
    from saddle.store import get_store

    ctx = resolve(args.tenant, args.project)
    items = get_store().todos(ctx)
    if not items:
        print(f"no open todos for {ctx.key}")
        return 0
    print(f"{len(items)} open todo(s) for {ctx.key}:")
    for it in items:
        print(f"  [{it.kind:<9}] {it.ask}  ({it.id})")
    return 0


# -- standing directives -----------------------------------------------------

def _run_directives(args: argparse.Namespace) -> int:
    from saddle.llm import policy

    ctx = resolve(args.tenant, args.project)
    if args.add is not None:
        text = _read_text(args.add)
        if not text:
            print("directives --add: empty text", file=sys.stderr)
            return 2
        added = policy.promote_directive(ctx, text, scope=args.scope)
        verb = "added" if added else "already present"
        print(f"{verb} [{args.scope}]: {text}")
        return 0
    if args.remove is not None:
        text = _read_text(args.remove)
        if not text:
            print("directives --remove: empty text", file=sys.stderr)
            return 2
        removed = policy.demote_directive(ctx, text, scope=args.scope)
        verb = "removed" if removed else "not found"
        print(f"{verb} [{args.scope}]: {text}")
        return 0
    rules = policy.directives(ctx)
    if not rules:
        print(f"no standing directives for {ctx.key}")
        return 0
    print(f"{len(rules)} standing directive(s) for {ctx.key}:")
    for d in rules:
        print(f"  - {d}")
    return 0


# -- DKB: record a lesson by hand --------------------------------------------

def _run_lesson(args: argparse.Namespace) -> int:
    from saddle.dkb import get_dkb
    from saddle.models import MANUAL, Knowledge

    ctx = resolve(args.tenant, args.project)
    body = _read_text(args.text)
    if not body:
        print("lesson: empty text (pass text or pipe via stdin)", file=sys.stderr)
        return 2
    title = (args.title or body).strip()
    if len(title) > 80:
        title = title[:79].rstrip() + "…"
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    scope_tenant, scope_project = _scope_pair(ctx, args.scope)
    kn = Knowledge(
        kind=args.kind, title=title, body=body, tags=tags,
        scope_tenant=scope_tenant, scope_project=scope_project, source=MANUAL,
    )
    get_dkb().add_knowledge(kn)
    print(f"recorded {kn.kind} [{kn.scope}] {kn.id}: {title}")
    return 0


# -- DKB: remember a project fact / cache a lookup ---------------------------

def _run_remember(args: argparse.Namespace) -> int:
    """Write a MEMORY entry: a durable ``fact`` (the default — identity, a stable
    invariant) or a ``reference`` cache of a file/web lookup. A reference is
    forced non-durable by the DKB, so ``--ttl`` (expiry) and ``--source``
    (provenance for re-derivation) only make sense there, but are accepted on
    either kind. This is the write face of the same memory ``recall`` reads."""
    import time

    from saddle.dkb import get_dkb
    from saddle.models import MANUAL, Knowledge

    ctx = resolve(args.tenant, args.project)
    body = _read_text(args.text)
    if not body:
        print("remember: empty text (pass text or pipe via stdin)", file=sys.stderr)
        return 2
    title = (args.title or body).strip()
    if len(title) > 80:
        title = title[:79].rstrip() + "…"
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    scope_tenant, scope_project = _scope_pair(ctx, args.scope)
    now = time.time()
    expires_at = now + args.ttl if args.ttl and args.ttl > 0 else 0.0
    provenance: dict = {}
    if args.source:
        provenance = {"source": args.source, "fetched_at": now}
    kn = Knowledge(
        kind=args.kind, title=title, body=body, tags=tags,
        scope_tenant=scope_tenant, scope_project=scope_project, source=MANUAL,
        expires_at=expires_at, provenance=provenance,
    )
    get_dkb().add_knowledge(kn)
    tier = "cache" if kn.is_cache else "durable"
    print(f"remembered {kn.kind} [{kn.scope}] ({tier}) {kn.id}: {title}")
    return 0


# -- DKB: list / search / seed / gc ------------------------------------------

def _run_kb(args: argparse.Namespace) -> int:
    from saddle.dkb import get_dkb

    ctx = resolve(args.tenant, args.project)
    if args.seed:
        from saddle.seed import seed_dkb
        r = seed_dkb()
        print(
            f"seed: {r['total']} entries — {r['added']} added, "
            f"{r['skipped']} already present"
        )
        return 0
    if args.gc:
        cap = args.max_cache if args.max_cache and args.max_cache > 0 else None
        r = get_dkb().cleanup(max_cache_per_scope=cap)
        print(
            f"kb gc: {r['expired']} expired + {r['evicted']} over-cap evicted "
            f"across {r['scopes']} scope(s)"
        )
        return 0
    if args.search is not None:
        query = _read_text(args.search)
        if not query:
            print("kb --search: empty query", file=sys.stderr)
            return 2
        hits = get_dkb().search_knowledge(ctx, query, k=args.k)
        if not hits:
            print(f"no matches for {query!r} in {ctx.key}")
            return 0
        print(f"{len(hits)} hit(s) for {query!r} in {ctx.key}:")
        for kn, score in hits:
            print(f"  [{kn.kind:<13}] {score:.4f}  {kn.title}  <{kn.scope}>")
        return 0
    kinds = [args.kind] if args.kind else None
    items = get_dkb().list_knowledge(ctx, kinds=kinds, limit=args.limit)
    if not items:
        print(f"DKB empty for {ctx.key} (try: saddle kb --seed)")
        return 0
    print(f"{len(items)} DKB entr{'y' if len(items) == 1 else 'ies'} for {ctx.key}:")
    for kn in items:
        print(f"  [{kn.kind:<13}] {kn.title}  <{kn.scope}> ({kn.source})")
    return 0


# -- Layer 3: code-completeness gate -----------------------------------------

def _run_codemap(args: argparse.Namespace) -> int:
    from saddle.codemap import SurfaceManifest, refs
    from saddle.context import code_root

    root = args.root or str(code_root())

    # Inspect mode: the symbol menu a design's surface must be grounded in.
    if args.symbols:
        mods = refs.parse_project(root)
        menu = refs.symbols(mods).top()
        if args.json:
            print(json.dumps(menu, indent=2))
            return 0
        print(f"symbol menu for {root} ({len(mods)} module(s)):")
        for bucket in ("fields", "funcs", "calls"):
            print(f"\n{bucket}:")
            for name, cnt in menu[bucket].items():
                print(f"  {cnt:>5}  {name}")
        if menu["collections"]:
            print("\ncollections:")
            for name, members in menu["collections"].items():
                print(f"  {name} = {members}")
        return 0

    # Gate mode: re-run a persisted design's DECLARED surface against the code as
    # it stands now. Nonzero exit on any gap, so this drops straight into a
    # commit hook / CI step (the WIRED gate RayXI's value_impact_map never had).
    if not args.design:
        print("codemap: pass a design id to gate, or --symbols to inspect the menu",
              file=sys.stderr)
        return 2
    from saddle.dkb import get_dkb

    ctx = resolve(args.tenant, args.project)
    design = get_dkb().get_design(ctx, args.design)
    if design is None:
        print(f"codemap: no design {args.design!r} in {ctx.key}", file=sys.stderr)
        return 2
    manifest = SurfaceManifest.from_dict(design.meta.get("surface"))
    if manifest.is_empty():
        print(f"codemap: design {args.design} declared no surface — nothing to gate")
        return 0
    mods = refs.parse_project(root)
    findings = manifest.gate(mods, root=root)
    if args.json:
        print(json.dumps([dataclasses.asdict(f) for f in findings], indent=2))
    elif not findings:
        print(f"codemap: design {args.design} surface COMPLETE against {root} "
              f"({len(mods)} module(s)) — no gaps")
    else:
        print(f"codemap: {len(findings)} gap(s) for design {args.design} against {root}:")
        for f in findings:
            print(f"  {f}")
    return 1 if findings else 0


# -- Layer 4: drive the coder to satisfy a design's surface ------------------

def _run_converge(args: argparse.Namespace) -> int:
    from saddle.context import code_root
    from saddle.converge import converge_design, format_result
    from saddle.dkb import get_dkb

    ctx = resolve(args.tenant, args.project)
    design = get_dkb().get_design(ctx, args.design)
    if design is None:
        print(f"converge: no design {args.design!r} in {ctx.key}", file=sys.stderr)
        return 2
    root = args.root or str(code_root())
    # Stream the coder's work to stderr live (unless silenced) so a converge run
    # isn't a black box; the result still lands on stdout for piping.
    on_chunk = None if _verbosity(args) < 0 else _emit_stderr
    result = asyncio.run(
        converge_design(
            design, code_root=root, ctx=ctx,
            max_rounds=args.max_rounds, turn_retries=args.turn_retries,
            persist=not args.no_persist, on_chunk=on_chunk,
        )
    )
    if args.json:
        print(json.dumps({
            "design_id": result.design_id,
            "status": result.status,
            "rounds": [
                {"n": r.n, "gaps_before": r.gaps_before,
                 "gaps_after": r.gaps_after, "closed": r.closed}
                for r in result.rounds
            ],
            "final_gaps": [str(f) for f in result.final_gaps],
            "error": result.error,
        }, indent=2))
    else:
        print(format_result(result, root))
    return 0 if result.ok else 1


# -- audit: grounded probes -> findings + coverage ---------------------------

def _run_audit(args: argparse.Namespace) -> int:
    """Enumerate everything that can be audited under a project root and probe each
    target, grounded in real code/registries/docs. ``--list`` just prints the
    coverage plan (no LLM). A run prints the findings + the coverage ledger; the
    full report (findings.json / report.md) is written under the root unless
    ``--no-persist``. Exit 1 when any error-severity finding lands OR coverage is
    incomplete (a target failed/pending) — so it gates in CI."""
    from saddle.audit import build_plan, format_report, run_audit
    from saddle.context import code_root

    ctx = resolve(args.tenant, args.project)
    root = args.root or str(code_root())
    kinds = [k.strip() for k in (args.kinds or "").split(",") if k.strip()] or None
    plan = build_plan(root)

    # Attach any concern seeds (project-specific patterns the caller supplies — no
    # engine vocabulary is baked into saddle) to every concern target.
    seeds = [s for s in (args.seed or []) if s.strip()]
    if seeds:
        plan.targets = [
            dataclasses.replace(t, seeds=list(t.seeds) + seeds) if t.kind == "concern" else t
            for t in plan.targets
        ]

    if args.list:
        sel = [t for t in plan.targets if (not kinds or t.kind in kinds)]
        print(f"audit plan for {root}: {len(sel)} target(s) "
              f"({len(plan.targets)} total before --kinds filter)")
        for t in sel:
            print(f"  [{t.kind:<9}] {t.id}")
        return 0

    only = list(args.only) if args.only else None
    if args.limit and args.limit > 0:
        pool = [t for t in plan.targets if (not kinds or t.kind in kinds)
                and (not only or t.id in set(only))]
        only = [t.id for t in pool[: args.limit]]

    narrate = _verbosity(args) >= 0

    def on_event(ev: dict) -> None:
        if not narrate:
            return
        kind = ev.get("event")
        if kind == "plan":
            sys.stderr.write(f"[audit] {ev.get('targets', 0)} target(s) to probe under {ev.get('root','')}\n")
        elif kind == "parse" and ev.get("phase") == "start":
            dirs = ", ".join(ev.get("code_dirs") or []) or "(whole root)"
            sys.stderr.write(f"[audit] parsing source dirs: {dirs} …\n")
        elif kind == "parse" and ev.get("phase") == "done":
            sys.stderr.write(f"[audit] parsed {ev.get('modules', 0)} module(s) / {ev.get('files', 0)} file(s) — probing\n")
        elif kind == "parse" and ev.get("phase") == "failed":
            sys.stderr.write(f"[audit] SETUP WEDGED: {ev.get('reason','')}\n")
        elif kind == "progress":
            sys.stderr.write(f"\r[audit] {ev['done']}/{ev['total']} targets probed")
            sys.stderr.flush()
        elif kind == "target" and ev.get("status") == "ran":
            sys.stderr.write(f"\r[audit] {ev['id']} -> {ev['findings']} finding(s)\n")
        elif kind == "target" and ev.get("status") == "failed":
            sys.stderr.write(f"\r[audit] {ev['id']} FAILED: {ev.get('reason','')}\n")
        elif kind == "target" and ev.get("status") == "skipped":
            sys.stderr.write(f"\r[audit] {ev['id']} skipped: {ev.get('reason','')}\n")

    report = asyncio.run(run_audit(
        plan, ctx=ctx, kinds=kinds, only=only,
        concurrency=args.concurrency, per_target_deadline_s=args.deadline,
        persist=not args.no_persist,
        out_dir=args.out, on_event=on_event,
    ))
    if narrate:
        sys.stderr.write("\n")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    cov = report.coverage
    return 0 if (not report.errors() and cov.complete) else 1


# -- Layer 0: the pre-action doctrine gate -----------------------------------

def _parse_evidence(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``--evidence k=v`` repeats into a dict. A bare ``--evidence k``
    (no ``=``) is shorthand for ``k=true`` — convenient for boolean evidence
    like ``cross_project_task``."""
    out: dict[str, str] = {}
    for item in pairs or []:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[item.strip()] = "true"
    return out


def _run_guard(args: argparse.Namespace) -> int:
    """saddle's first passthrough: evaluate a proposed action against doctrine
    BEFORE it happens. Exit 0 = ALLOW, 2 = BLOCK — so a worker (or a coder
    wrapper, or a PreToolUse hook) can gate a step with
    ``saddle guard ... && <do the thing>`` and the gate runs with no LLM in the
    enforcement loop."""
    from saddle.context import code_root
    from saddle.doctrine import Action, guard

    ctx = resolve(args.tenant, args.project)
    root = args.root or str(code_root())
    action = Action(
        verb=args.verb,
        target=args.target,
        target_kind=args.kind,
        evidence=_parse_evidence(args.evidence),
        project_root=root,
    )
    verdict = guard(action, ctx)
    if args.json:
        print(json.dumps({
            "allowed": verdict.allowed,
            "rule_id": verdict.rule_id,
            "reason": verdict.reason,
            "required_evidence": list(verdict.required_evidence),
            "severity": verdict.severity,
        }, indent=2))
    else:
        print(verdict.render())
    return 0 if verdict.allowed else 2


# -- cross-project authorization grants --------------------------------------

def _run_crossproject(args: argparse.Namespace) -> int:
    """Manage the explicit, persistent grants the doctrine scope-fence consults.

    The fence blocks edits outside the focus project unless the work is
    explicitly cross-project; a *grant* is how a human (or this CLI) records that
    authorization durably. ``list`` shows what's authorized, ``grant`` adds a set
    of mutually-authorized roots, ``revoke`` clears them all."""
    import time

    from saddle import crossproject

    if args.action == "list":
        grants = crossproject.load_grants()
        if not grants:
            print("no cross-project grants (the scope-fence is fully closed)")
            return 0
        print(f"{len(grants)} cross-project grant(s):")
        for g in grants:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(g.ts)) if g.ts else "?"
            tnt = g.tenant or "(every tenant)"
            print(f"  [{tnt}] {when} via {g.source}")
            for r in g.roots:
                print(f"      root: {r}")
            if g.reason:
                print(f"      reason: {g.reason}")
        return 0

    if args.action == "grant":
        roots = [r for r in (args.roots or []) if r.strip()]
        if not roots:
            print("cross-project grant: pass one or more project roots to authorize",
                  file=sys.stderr)
            return 2
        tenant = args.tenant if args.tenant is not None else resolve(args.tenant, args.project).tenant
        g = crossproject.grant(roots, tenant=tenant, reason=args.reason or "", source="cli")
        print(f"granted [{g.tenant or 'every tenant'}] across {len(g.roots)} root(s):")
        for r in g.roots:
            print(f"  {r}")
        return 0

    if args.action == "revoke":
        n = crossproject.revoke_all()
        print(f"revoked {n} cross-project grant(s)")
        return 0

    return 2  # unreachable — argparse constrains the choices


# -- bubble outbox (saddle's client-agnostic outbound voice) -----------------

def _parse_since(raw: str | None) -> float | None:
    """Interpret ``--since`` as a relative age (``90s`` / ``30m`` / ``2h`` /
    ``1d``) or a raw epoch-seconds value, returning the absolute ``since_ts``
    floor recent-bubble queries compare against. Unparseable -> ``None`` (no
    floor) so a typo widens the window rather than hiding saddle's voice."""
    if not raw:
        return None
    import time

    s = raw.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s and s[-1] in units:
        try:
            return time.time() - float(s[:-1]) * units[s[-1]]
        except ValueError:
            return None
    try:
        return float(s)  # raw epoch seconds
    except ValueError:
        return None


def _run_bubble(args: argparse.Namespace) -> int:
    """Read saddle's outbound voice — the bubbles the hooks (and, soon, the
    staged supervisory runner) emit so an AFK / non-TTY human SEES what saddle
    said when stderr is swallowed under an SDK host. This is the CLI face of the
    same client-agnostic outbox the launcher panel and the MCP ``bubble_recent``
    tool read; all three go through one render contract (:func:`render_bubbles`)
    so they never drift in presentation. For a LIVE feed, ``tail -f`` the JSONL
    mirror; this is the point-in-time read."""
    from saddle.bubble import event_to_dict, recent_bubbles, render_bubbles

    ctx = resolve(args.tenant, args.project)
    events = recent_bubbles(
        ctx,
        session=args.session or None,
        since_ts=_parse_since(args.since),
        level=args.level,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([event_to_dict(e) for e in events], indent=2))
        return 0
    if not events:
        where = f" for session {args.session}" if args.session else ""
        print(f"no bubbles{where} in [{ctx.key}]")
        return 0
    print(render_bubbles(events))
    return 0


# -- parser ------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    # Global progress flags, shared by the top-level parser AND every subparser
    # (via ``parents``) so they're accepted on either side of the subcommand.
    # SUPPRESS defaults mean an unset flag never overwrites a value parsed from
    # the other side.
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("-v", "--verbose", action="count", default=argparse.SUPPRESS,
                   help="narrate more (repeat for debug)")
    g.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                   help="silence progress narration (warnings + errors only)")

    parser = argparse.ArgumentParser(
        prog="saddle", description="saddle LLM harness", parents=[g])
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("chat", help="interactive agentic chat (default)", parents=[g])

    p_in = sub.add_parser("intake", help="decompose + record a prompt (Layer 1)",
                          parents=[g])
    p_in.add_argument("prompt", nargs="?", help="prompt text; omit or '-' to read stdin")
    _add_ctx_flags(p_in)
    p_in.add_argument("--audits", type=int, default=2,
                      help="max coverage-audit passes (default 2)")
    p_in.add_argument("--json", action="store_true", help="dump the raw intake as JSON")

    p_dg = sub.add_parser("design", help="decompose → fold → design (Layer 1 + 2)",
                          parents=[g])
    p_dg.add_argument("prompt", nargs="?", help="prompt text; omit or '-' to read stdin")
    _add_ctx_flags(p_dg)
    p_dg.add_argument("--audits", type=int, default=2,
                      help="max design audit passes per goal (default 2)")
    p_dg.add_argument("--retrieve-k", type=int, default=8, dest="retrieve_k",
                      help="DKB hits to retrieve per design (default 8)")
    p_dg.add_argument("--no-designs", action="store_true", dest="no_designs",
                      help="stop after folding; don't run the design pipeline")
    p_dg.add_argument("--json", action="store_true", help="dump the orchestration as JSON")

    p_td = sub.add_parser("todos", help="show the open todo backlog", parents=[g])
    _add_ctx_flags(p_td)

    p_dir = sub.add_parser("directives", help="list / promote standing rules",
                           parents=[g])
    _add_ctx_flags(p_dir)
    p_dir.add_argument("--add", nargs="?", const="-", default=None,
                       metavar="TEXT", help="promote a directive (text or '-'/omit for stdin)")
    p_dir.add_argument("--remove", nargs="?", const="-", default=None,
                       metavar="TEXT", help="curate out a directive (text or '-'/omit for stdin)")
    p_dir.add_argument("--scope", choices=_SCOPES, default="project",
                       help="scope to promote / remove at (default project)")

    p_les = sub.add_parser("lesson", help="record a DKB entry by hand", parents=[g])
    p_les.add_argument("text", nargs="?", help="lesson text; omit or '-' to read stdin")
    _add_ctx_flags(p_les)
    p_les.add_argument("--kind", choices=_KB_KINDS, default="lesson",
                       help="DKB kind (default lesson)")
    p_les.add_argument("--title", default=None, help="title (defaults to the text)")
    p_les.add_argument("--tags", default="", help="comma-separated tags")
    p_les.add_argument("--scope", choices=_SCOPES, default="project",
                       help="visibility scope (default project)")

    p_rem = sub.add_parser("remember", help="remember a project fact / cache a lookup",
                           parents=[g])
    p_rem.add_argument("text", nargs="?", help="fact text; omit or '-' to read stdin")
    _add_ctx_flags(p_rem)
    p_rem.add_argument("--kind", choices=_MEMORY_KINDS, default="fact",
                       help="fact (durable) or reference (evictable cache); default fact")
    p_rem.add_argument("--title", default=None, help="title (defaults to the text)")
    p_rem.add_argument("--tags", default="", help="comma-separated tags")
    p_rem.add_argument("--scope", choices=_SCOPES, default="project",
                       help="visibility scope (default project)")
    p_rem.add_argument("--source", default="", metavar="PATH|URL",
                       help="provenance for a cached lookup (recorded for re-derivation)")
    p_rem.add_argument("--ttl", type=float, default=0.0, metavar="SECONDS",
                       help="expire a cache entry after this many seconds (0 = never)")

    p_kb = sub.add_parser("kb", help="Design Knowledge Base: list / search / seed / gc",
                          parents=[g])
    _add_ctx_flags(p_kb)
    p_kb.add_argument("--search", nargs="?", const="-", default=None,
                      metavar="QUERY", help="hybrid-search the DKB (text or '-'/omit for stdin)")
    p_kb.add_argument("--seed", action="store_true", help="load the global seed corpus")
    p_kb.add_argument("--gc", action="store_true",
                      help="evict expired + over-cap cache entries (bounded-storage cleanup)")
    p_kb.add_argument("--max-cache", type=int, default=0, dest="max_cache",
                      help="override the per-scope cache cap for this gc run (0 = use default)")
    p_kb.add_argument("--kind", choices=_KB_KINDS, default=None, help="filter list by kind")
    p_kb.add_argument("-k", type=int, default=8, dest="k", help="search hits to return")
    p_kb.add_argument("--limit", type=int, default=50, help="max entries to list")

    p_cm = sub.add_parser(
        "codemap",
        help="Layer 3: gate a design's completeness surface against code",
        parents=[g])
    p_cm.add_argument("design", nargs="?", help="design id to gate (omit when using --symbols)")
    _add_ctx_flags(p_cm)
    p_cm.add_argument("--symbols", action="store_true",
                      help="dump the project's grounding symbol menu instead of gating")
    p_cm.add_argument("--root", default=None,
                      help="code root to parse (default: $SADDLE_CODE_ROOT, else git root / cwd)")
    p_cm.add_argument("--json", action="store_true", help="machine-readable output")

    p_cv = sub.add_parser(
        "converge",
        help="Layer 4: drive the coder to implement a design until its surface is satisfied",
        parents=[g])
    p_cv.add_argument("design", help="design id to implement and converge")
    _add_ctx_flags(p_cv)
    p_cv.add_argument("--root", default=None,
                      help="code root the coder edits (default: $SADDLE_CODE_ROOT, else git root / cwd)")
    p_cv.add_argument("--max-rounds", type=int, default=8, dest="max_rounds",
                      help="hard cap on coder-turn + re-gate cycles (default 8)")
    p_cv.add_argument("--turn-retries", type=int, default=2, dest="turn_retries",
                      help="bounded retries on a crashed coder turn before halting (default 2)")
    p_cv.add_argument("--no-persist", action="store_true", dest="no_persist",
                      help="don't record the convergence trail onto the design")
    p_cv.add_argument("--json", action="store_true", help="machine-readable output")

    p_au = sub.add_parser(
        "audit",
        help="audit a project: grounded probes emit findings + a coverage ledger",
        parents=[g])
    _add_ctx_flags(p_au)
    p_au.add_argument("--root", default=None,
                      help="project root to audit (default: $SADDLE_CODE_ROOT, else git root / cwd)")
    p_au.add_argument("--list", action="store_true",
                      help="enumerate the audit plan (what CAN be audited) and exit — no LLM")
    p_au.add_argument("--kinds", default=None,
                      help="comma-separated target kinds to include (registry,doc,package,concern)")
    p_au.add_argument("--only", action="append", metavar="TARGET_ID",
                      help="restrict to specific target id(s) (repeatable)")
    p_au.add_argument("--limit", type=int, default=0,
                      help="cap the number of targets probed (0 = no cap)")
    p_au.add_argument("--seed", action="append", metavar="REGEX",
                      help="ground every concern probe with this code-pattern seed (repeatable)")
    p_au.add_argument("--concurrency", type=int, default=3,
                      help="probes to run in parallel (default 3)")
    p_au.add_argument("--deadline", type=float, default=300.0,
                      help="per-probe deadline in seconds before that target fails (default 300)")
    p_au.add_argument("--out", default=None,
                      help="directory for the written report (default: <root>/.saddle/audit/<ts>)")
    p_au.add_argument("--no-persist", action="store_true", dest="no_persist",
                      help="don't write the report to disk")
    p_au.add_argument("--json", action="store_true", help="machine-readable output")

    p_gd = sub.add_parser(
        "guard",
        help="Layer 0: gate a proposed action against doctrine before taking it",
        parents=[g])
    _add_ctx_flags(p_gd)
    p_gd.add_argument("--verb", required=True,
                      help="action verb (edit/write/create/delete/rm/...)")
    p_gd.add_argument("--target", required=True,
                      help="the path or code symbol the action would touch")
    p_gd.add_argument("--kind", choices=["auto", "path", "code"], default="auto",
                      help="how to read --target (default: auto-infer from shape)")
    p_gd.add_argument("--evidence", action="append", metavar="K=V",
                      help="evidence k=v (repeatable); bare k means k=true")
    p_gd.add_argument("--root", default=None,
                      help="focus-project root (default: $SADDLE_CODE_ROOT, else git root / cwd)")
    p_gd.add_argument("--json", action="store_true", help="machine-readable verdict")

    p_xp = sub.add_parser(
        "cross-project",
        help="manage cross-project authorization grants for the scope-fence",
        parents=[g])
    _add_ctx_flags(p_xp)
    p_xp.add_argument("action", choices=["list", "grant", "revoke"],
                      help="list grants, grant across roots, or revoke all")
    p_xp.add_argument("roots", nargs="*",
                      help="project roots to authorize together (for 'grant')")
    p_xp.add_argument("--reason", default="",
                      help="why this cross-project work is authorized (recorded on the grant)")

    p_bub = sub.add_parser(
        "bubble",
        help="read saddle's outbound voice (the client-agnostic bubble outbox)",
        parents=[g])
    _add_ctx_flags(p_bub)
    p_bub.add_argument("--session", default=None,
                       help="filter to one agent session id")
    p_bub.add_argument("--level", choices=_BUBBLE_LEVELS, default=None,
                       help="filter to one level (alert = needs a correction/decision)")
    p_bub.add_argument("--since", default=None, metavar="AGE",
                       help="only bubbles newer than AGE (90s/30m/2h/1d, or epoch seconds)")
    p_bub.add_argument("--limit", type=int, default=50,
                       help="max bubbles to show (default 50)")
    p_bub.add_argument("--json", action="store_true",
                       help="machine-readable list of bubble events")

    return parser


_DISPATCH = {
    "intake": _run_intake,
    "design": _run_design,
    "todos": _run_todos,
    "directives": _run_directives,
    "lesson": _run_lesson,
    "remember": _run_remember,
    "kb": _run_kb,
    "codemap": _run_codemap,
    "converge": _run_converge,
    "audit": _run_audit,
    "guard": _run_guard,
    "cross-project": _run_crossproject,
    "bubble": _run_bubble,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(_verbosity(args))
    if args.cmd is None or args.cmd == "chat":
        return _run_chat()
    handler = _DISPATCH.get(args.cmd)
    if handler is None:
        return 1  # unreachable — argparse rejects unknown subcommands
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
