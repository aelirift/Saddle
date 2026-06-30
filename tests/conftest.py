"""Shared test isolation.

The bubble outbox (:mod:`saddle.bubble`) is a process-wide singleton that lazily
opens a ``SqliteBubbleStore`` at ``$SADDLE_HOME/saddle.db``. Because the intake
and doctrine hooks now emit bubbles IN-PROCESS during tests, that singleton would
otherwise cache the FIRST test's tmp db and leak it into every later test (which
each redirect ``SADDLE_HOME`` to a fresh tmp dir). Reset it around every test so
each starts against its own db — the same discipline ``reset_store`` /
``reset_dkb`` already give per file, applied globally for the one singleton the
hooks touch implicitly.
"""
from __future__ import annotations

import pytest

from saddle.bubble import reset_bubble_store


@pytest.fixture(autouse=True)
def _reset_bubble_store():
    reset_bubble_store()
    yield
    reset_bubble_store()
