"""Lightweight Git helpers used by the pipeline."""

from __future__ import annotations

try:
    from git import Repo  # type: ignore
except Exception:  # pragma: no cover
    Repo = object  # type: ignore
from .logging_utils import get_logger

logger = get_logger("pleb.git")

def checkout(repo: Repo, branch: str) -> None:
    """Check out a git branch.

    Args:
        repo: GitPython repository object.
        branch: Branch name to check out.
    """
    repo.git.checkout(branch)

def require_clean_repo(repo: Repo) -> None:
    """Warn if the git repository has uncommitted changes.

    Args:
        repo: GitPython repository object.
    """
    if repo.is_dirty(untracked_files=True):
        logger.warning("Repo has uncommitted changes. Results may be non-reproducible.")
