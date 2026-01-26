"""Tests for git helper utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("git")  # provided by GitPython

from pleb.git_tools import require_clean_repo


def test_require_clean_repo_does_not_raise_on_dirty() -> None:
    # Avoid depending on an actual git repository in unit tests.
    repo = SimpleNamespace(is_dirty=lambda untracked_files=True: True)
    require_clean_repo(repo)  # should not raise
