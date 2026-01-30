"""Lightweight Git helpers used by the pipeline.

These helpers wrap common GitPython operations with minimal logging.
"""

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

    Examples:
        Check out a branch::

            checkout(repo, "main")
    """
    repo.git.checkout(branch)


def require_clean_repo(repo: Repo) -> None:
    """Warn if the git repository has uncommitted changes.

    Args:
        repo: GitPython repository object.

    Notes:
        This function only logs a warning; it does not raise.
    """
    if repo.is_dirty(untracked_files=True):
        logger.warning("Repo has uncommitted changes. Results may be non-reproducible.")
