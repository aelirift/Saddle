"""Standing directives must be curatable OUT, not only promoted IN.

Promotion had no inverse, so a rule captured by mistake — e.g. a task-specific
instruction auto-promoted as if it were standing — could only be undone by hand-
editing policy. :func:`saddle.llm.policy.demote_directive` is the symmetric path.
These pin the round-trip, normalized matching, and scope isolation (a project
demote never strips a tenant/global rule).
"""
from __future__ import annotations

import pytest

from saddle.context import Context
from saddle.llm import policy


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    # Point BOTH the config dir AND the global base policy at a tmp tree so
    # promotes/demotes never touch the real policy files, and no ambient global
    # directive leaks into the effective set (the base policy is resolved
    # separately from the config dir). Reset the cache around each test.
    monkeypatch.setenv("SADDLE_CONFIG_DIR", str(tmp_path))
    base = tmp_path / "llm_policy.json"
    base.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SADDLE_LLM_POLICY", str(base))
    policy.reset_policy_cache()
    yield
    policy.reset_policy_cache()


CTX = Context(tenant="acme", project="game")


def test_promote_then_demote_round_trip():
    assert policy.promote_directive(CTX, "always fail loud", scope="project") is True
    assert "always fail loud" in policy.directives(CTX)
    assert policy.demote_directive(CTX, "always fail loud", scope="project") is True
    assert "always fail loud" not in policy.directives(CTX)


def test_demote_matches_normalized_text():
    policy.promote_directive(CTX, "No Band-Aids", scope="project")
    # different spacing + case still removes it
    assert policy.demote_directive(CTX, "  no   band-aids ", scope="project") is True
    assert policy.directives(CTX) == []


def test_demote_absent_directive_returns_false():
    assert policy.demote_directive(CTX, "never set", scope="project") is False


def test_project_demote_does_not_strip_tenant_rule():
    policy.promote_directive(CTX, "tenant-wide rule", scope="tenant")
    policy.promote_directive(CTX, "project rule", scope="project")
    # Removing at project scope leaves the tenant rule intact (no reaching up).
    assert policy.demote_directive(CTX, "tenant-wide rule", scope="project") is False
    assert "tenant-wide rule" in policy.directives(CTX)
    assert policy.demote_directive(CTX, "project rule", scope="project") is True
    assert "tenant-wide rule" in policy.directives(CTX)


def test_demote_empty_text_raises():
    with pytest.raises(ValueError):
        policy.demote_directive(CTX, "  ", scope="project")
