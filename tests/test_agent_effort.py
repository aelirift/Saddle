"""Effort split — the one-shot SUPERVISORY caller must not inherit the CODER's
deepest effort.

saddle has two LLM workload classes with different latency SLAs:

  * the CODER (:class:`~saddle.llm.claude_agent.ChatSession`) runs a multi-turn
    converge loop, bounded by an idle heartbeat — it wants the deepest pass
    (``xhigh``);
  * the STRUCTURED/SUPERVISORY one-shot caller
    (:class:`~saddle.llm.claude_agent.ClaudeAgentCaller`, ``max_turns=1``) backs
    every Stage 1-5 drift check, which runs INLINE inside a Claude Code hook
    under a ~60s deadline (the hook BLOCKS the turn until it returns).

When both shared ``xhigh`` the one-shot design audit measured 40-54s — at/over
the deadline — so the drift check intermittently TIMED OUT and bubbled "could
not verify" instead of actually catching drift. The fix gives the one-shot
caller its own faster default (``high``, measured ~10-12s with the SAME
findings). These tests pin that the two classes resolve DIFFERENT efforts and
the precedence of the overrides, with no real LLM and no real CLI.
"""
from __future__ import annotations

from saddle.llm.claude_agent import (
    ChatSession,
    ClaudeAgentCaller,
    _DEFAULT_CODER_EFFORT,
    _DEFAULT_STRUCTURED_EFFORT,
)

# The efforts that COMFORTABLY fit a ~60s inline hook deadline for a one-shot
# audit. xhigh/max are the slow band that caused the live timeout — a future
# "just use one effort everywhere" change must trip this set, loudly.
_FAST_BAND = {"low", "medium", "high"}


def _clear_effort_env(monkeypatch):
    for name in ("SADDLE_AGENT_EFFORT", "SADDLE_STRUCTURED_EFFORT", "SADDLE_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)


# -- the defining contract: the two classes differ -------------------------

def test_structured_and_coder_efforts_differ(monkeypatch):
    """The whole point of the split: the supervisory one-shot caller and the
    coder do NOT resolve the same effort by default. If they ever collapse back
    to one value, the inline-hook timeout returns — fail loud here first."""
    _clear_effort_env(monkeypatch)
    assert _DEFAULT_STRUCTURED_EFFORT != _DEFAULT_CODER_EFFORT
    assert ClaudeAgentCaller({})._effort != ChatSession()._effort


def test_structured_default_fits_hook_deadline(monkeypatch):
    """The supervisory default must live in the fast band — xhigh (the coder's
    setting) is exactly what blew the 60s deadline."""
    _clear_effort_env(monkeypatch)
    assert _DEFAULT_STRUCTURED_EFFORT in _FAST_BAND
    assert ClaudeAgentCaller({})._effort in _FAST_BAND


def test_coder_keeps_deepest_effort(monkeypatch):
    """The coder is bounded by an idle heartbeat, not the hook deadline, so it
    keeps the deepest pass — the split must not have lowered IT."""
    _clear_effort_env(monkeypatch)
    assert _DEFAULT_CODER_EFFORT == "xhigh"
    assert ChatSession()._effort == "xhigh"


# -- precedence: structured caller -----------------------------------------

def test_structured_cfg_effort_wins(monkeypatch):
    """An explicit per-provider ``effort`` beats every env + the default."""
    monkeypatch.setenv("SADDLE_STRUCTURED_EFFORT", "low")
    monkeypatch.setenv("SADDLE_AGENT_EFFORT", "medium")
    assert ClaudeAgentCaller({"effort": "xhigh"})._effort == "xhigh"


def test_structured_env_beats_global(monkeypatch):
    """``SADDLE_STRUCTURED_EFFORT`` tunes the inline drift-check effort and takes
    precedence over the global ``SADDLE_AGENT_EFFORT`` for the one-shot caller."""
    _clear_effort_env(monkeypatch)
    monkeypatch.setenv("SADDLE_STRUCTURED_EFFORT", "medium")
    monkeypatch.setenv("SADDLE_AGENT_EFFORT", "xhigh")
    assert ClaudeAgentCaller({})._effort == "medium"


def test_global_env_overrides_structured_default(monkeypatch):
    """With no structured-specific knob, the global override still wins — an
    operator pinning everything to one effort is honored (their explicit call)."""
    _clear_effort_env(monkeypatch)
    monkeypatch.setenv("SADDLE_AGENT_EFFORT", "max")
    assert ClaudeAgentCaller({})._effort == "max"


# -- precedence: coder ------------------------------------------------------

def test_coder_ctor_arg_wins(monkeypatch):
    """An explicit ``effort=`` argument beats env + default for the coder."""
    monkeypatch.setenv("SADDLE_AGENT_EFFORT", "low")
    assert ChatSession(effort="medium")._effort == "medium"


def test_coder_ignores_structured_env(monkeypatch):
    """The structured-only knob must NOT bleed into the coder — it tunes the
    supervisory caller alone."""
    _clear_effort_env(monkeypatch)
    monkeypatch.setenv("SADDLE_STRUCTURED_EFFORT", "low")
    assert ChatSession()._effort == "xhigh"


def test_coder_global_env_overrides(monkeypatch):
    """The global override still reaches the coder (both classes honor it)."""
    _clear_effort_env(monkeypatch)
    monkeypatch.setenv("SADDLE_AGENT_EFFORT", "high")
    assert ChatSession()._effort == "high"
