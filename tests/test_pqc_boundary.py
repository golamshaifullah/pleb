from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_pleb_does_not_import_pqc_detector_modules() -> None:
    offenders: list[str] = []
    for path in [*_python_files(REPO / "pleb"), *_python_files(REPO / "tests")]:
        if path.name == "test_pqc_boundary.py":
            continue
        for module in _imported_modules(path):
            if module == "pqc.detect" or module.startswith("pqc.detect."):
                offenders.append(f"{path.relative_to(REPO)} imports {module}")

    assert not offenders, "Detector logic belongs in pqc:\n" + "\n".join(offenders)
