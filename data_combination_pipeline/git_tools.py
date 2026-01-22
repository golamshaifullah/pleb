from __future__ import annotations

try:
    from git import Repo  # type: ignore
except Exception:  # pragma: no cover
    Repo = object  # type: ignore
from .logging_utils import get_logger

logger = get_logger("data_combination_pipeline.git")

def checkout(repo: Repo, branch: str) -> None:
    repo.git.checkout(branch)

def require_clean_repo(repo: Repo) -> None:
    if repo.is_dirty(untracked_files=True):
        logger.warning("Repo has uncommitted changes. Results may be non-reproducible.")
