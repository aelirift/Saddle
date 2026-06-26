"""Single source of truth for *what project is in focus*.

saddle observes ONE project at a time: the ``(tenant, project)`` it is bound
to, whose code lives under :func:`saddle.context.code_root`. More than one
layer has to agree on that boundary —

  * **Layer 0 — doctrine guard** fences file actions to paths *inside* the
    focus root (the ``stay-in-project-focus`` rule).
  * **Layer 1 — intake** flags asks that target work *outside* the focus
    project, so a "go improve <other project>" prompt is surfaced rather than
    silently itemized as in-bounds work.

If each layer computed "the focus" its own way they could disagree — the guard
fencing one root while intake itemizes against another. This module is the ONE
place the focus identity is defined so the layers cannot drift apart. Any
future layer that needs to reason about scope imports it from here too.

Two flavours of "in focus":

* **Path focus** — *is this filesystem target inside the focus repo?* — is
  deterministic and exact, and lives where it is used (the doctrine fence's own
  containment check). This module owns the *root* that check fences against.
* **Intent focus** — *does this natural-language ask belong to the focus
  project or to a different one?* — is a judgment the intake LLM makes. This
  module hands it the focus IDENTITY (project name + root) to judge against, via
  :func:`focus_descriptor`. It deliberately does NOT enumerate sibling project
  names: "in focus" is judged against THIS project, and anything about another
  codebase is out — without baking today's repo list into the code.
"""

from __future__ import annotations

from pathlib import Path

from saddle.context import Context, code_root
from saddle.context import default as _default_ctx


def focus_root(cwd: str | Path | None = None) -> Path:
    """The focus project's code root — the SAME path Layer 3 parses and the
    doctrine fence guards. Resolved by :func:`saddle.context.code_root`
    (``$SADDLE_CODE_ROOT`` -> enclosing git repo -> cwd)."""
    return code_root(cwd)


def focus_project(ctx: Context | None = None) -> str:
    """The focus project's name — the ``project`` half of the ``(tenant,
    project)`` isolation key. Falls back to the ambient context."""
    return (ctx or _default_ctx()).project


def focus_descriptor(ctx: Context | None = None) -> str:
    """A one-line description of the focus to inject into an LLM prompt: the
    project name and its root path. Carries the focus IDENTITY without listing
    sibling projects, so the model judges "belongs to this project vs. another"
    against a concrete target rather than a hardcoded vocabulary."""
    ctx = ctx or _default_ctx()
    return f'"{ctx.project}", the project whose code is rooted at {focus_root()}'
