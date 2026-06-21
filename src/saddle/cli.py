"""saddle command-line entry point.

A small subcommand dispatcher over saddle's surfaces:

    saddle                      # interactive agentic chat (default)
    saddle chat                 # same, explicit
    saddle intake "<prompt>"    # Layer 1: decompose + record a prompt
    saddle todos                # show the open todo backlog

``intake`` / ``todos`` take ``--tenant`` / ``--project`` to address a
specific tenant+project; omitted, they resolve from the environment
(SADDLE_TENANT / SADDLE_PROJECT) and the current working directory.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys

from saddle.context import resolve


def _run_chat() -> int:
    from saddle.chat import main as chat_main
    return chat_main()


def _read_prompt(arg: str | None) -> str:
    if arg is None or arg == "-":
        return sys.stdin.read().strip()
    return arg.strip()


def _run_intake(args: argparse.Namespace) -> int:
    from saddle.intake import decompose, format_intake

    prompt = _read_prompt(args.prompt)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saddle", description="saddle LLM harness")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("chat", help="interactive agentic chat (default)")

    p_in = sub.add_parser("intake", help="decompose + record a prompt (Layer 1)")
    p_in.add_argument("prompt", nargs="?", help="prompt text; omit or '-' to read stdin")
    p_in.add_argument("--tenant", default=None)
    p_in.add_argument("--project", default=None)
    p_in.add_argument("--audits", type=int, default=2,
                      help="max coverage-audit passes (default 2)")
    p_in.add_argument("--json", action="store_true", help="dump the raw intake as JSON")

    p_td = sub.add_parser("todos", help="show the open todo backlog")
    p_td.add_argument("--tenant", default=None)
    p_td.add_argument("--project", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd is None or args.cmd == "chat":
        return _run_chat()
    if args.cmd == "intake":
        return _run_intake(args)
    if args.cmd == "todos":
        return _run_todos(args)
    return 1  # unreachable — argparse rejects unknown subcommands


if __name__ == "__main__":
    raise SystemExit(main())
