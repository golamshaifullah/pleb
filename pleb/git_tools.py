"""Lightweight Git helpers used by the pipeline.

These helpers wrap common GitPython operations with minimal logging.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
from typing import Iterator

try:
    from git import Repo  # type: ignore
except Exception:  # pragma: no cover
    Repo = object  # type: ignore
from .logging_utils import get_logger

logger = get_logger("pleb.git")


def checkout(repo: Repo, branch: str) -> None:
    """Check out a git branch.

    Parameters
    ----------
    repo : git.Repo
        GitPython repository object.
    branch : str
        Branch name to check out.

    Examples
    --------
    Check out a branch::

        checkout(repo, "main")
    """
    repo.git.checkout(branch)


def require_clean_repo(repo: Repo) -> None:
    """Warn if the git repository has uncommitted changes.

    Parameters
    ----------
    repo : git.Repo
        GitPython repository object.

    Notes
    -----
        This function only logs a warning; it does not raise.
    """
    if repo.is_dirty(untracked_files=True):
        logger.warning("Repo has uncommitted changes. Results may be non-reproducible.")


def current_branch_name(repo: Repo) -> str:
    """Return the current branch name, or ``""`` for detached HEAD."""
    try:
        return repo.active_branch.name
    except (TypeError, AttributeError, ValueError):
        return ""


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def branch_checked_out_in_worktree(repo_root: Path, branch: str) -> Path | None:
    """Return the worktree path where a branch is checked out, if any."""
    res = _git(Path(repo_root), "worktree", "list", "--porcelain")
    if res.returncode != 0:
        return None
    current_path: Path | None = None
    target = f"refs/heads/{branch}"
    for raw in res.stdout.splitlines():
        line = raw.strip()
        if not line:
            current_path = None
            continue
        if line.startswith("worktree "):
            current_path = Path(line.split(" ", 1)[1])
            continue
        if line == f"branch {target}":
            return current_path
    return None


@contextmanager
def temporary_detached_worktree(repo_root: Path, ref: str) -> Iterator[Path]:
    """Yield a detached temporary worktree rooted at ``ref``.

    The temporary worktree is removed on normal exit. If cleanup fails, the
    path is left in place and a warning is logged.
    """
    repo_root = Path(repo_root).resolve()
    worktree_path = Path(tempfile.mkdtemp(prefix="pleb_worktree_"))
    add = _git(repo_root, "worktree", "add", "--detach", str(worktree_path), ref)
    if add.returncode != 0:
        raise RuntimeError(
            f"git worktree add failed for ref {ref!r}: "
            f"{(add.stderr or add.stdout).strip()}"
        )
    try:
        yield worktree_path
    finally:
        rm = _git(repo_root, "worktree", "remove", str(worktree_path))
        if rm.returncode != 0:
            rm_force = _git(
                repo_root, "worktree", "remove", "--force", str(worktree_path)
            )
            if rm_force.returncode == 0:
                _git(repo_root, "worktree", "prune")
            else:
                logger.warning(
                    "Failed to remove temporary worktree %s: %s | force remove failed: %s",
                    worktree_path,
                    (rm.stderr or rm.stdout).strip(),
                    (rm_force.stderr or rm_force.stdout).strip(),
                )
