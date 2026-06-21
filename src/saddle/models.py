"""Domain models for saddle's intake layer + store.

An :class:`Intake` is one decomposition of one user prompt into a list of
:class:`Item` s. Each Item is one discrete ask, classified by ``kind`` and
restated clearly in ``ask``. The persistent "todo list" is not a separate
type — it's the subset of items whose kind is in :data:`TODO_KINDS` that
are still open (see ``store.todos``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- Item kinds: the taxonomy Layer 1 classifies every ask into ----------
QUESTION = "question"      # wants an answer
TASK = "task"              # an action to perform -> goes on the todo list
DIRECTIVE = "directive"    # a standing rule / preference / constraint
CONTEXT = "context"        # background, no action needed
DECISION = "decision"      # a fork left to the assistant to choose

ITEM_KINDS: frozenset[str] = frozenset(
    {QUESTION, TASK, DIRECTIVE, CONTEXT, DECISION}
)
# Kinds that land on the persistent, actionable todo list.
TODO_KINDS: frozenset[str] = frozenset({TASK, DIRECTIVE})

# --- Item statuses -------------------------------------------------------
OPEN = "open"              # not yet handled
ANSWERED = "answered"      # a question that's been answered
DONE = "done"             # a task that's been completed
NOTED = "noted"            # context/directive acknowledged, no further action
ITEM_STATUSES: frozenset[str] = frozenset({OPEN, ANSWERED, DONE, NOTED})


@dataclass
class Item:
    """One discrete ask pulled out of a prompt.

    ``kind``/``ask``/``source_text``/``detail`` are produced by Layer 1;
    ``id``/``intake_id``/``seq``/``ts`` are stamped by the store on save.
    """

    kind: str
    ask: str
    source_text: str = ""
    detail: str = ""
    status: str = OPEN
    id: str = ""
    intake_id: str = ""
    seq: int = 0
    ts: float = 0.0

    def is_todo(self) -> bool:
        return self.kind in TODO_KINDS


@dataclass
class Intake:
    """One decomposition of one prompt, with its items + coverage meta."""

    raw_prompt: str
    summary: str = ""
    items: list[Item] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    tenant: str = ""
    project: str = ""
    ts: float = 0.0
