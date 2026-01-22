from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pandas as pd

def _find_header_line(lines: List[str], startswith: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if line.strip().startswith(startswith):
            return i
    return None

def read_plklog(file: Path) -> pd.DataFrame:
    if not file.exists():
        raise FileNotFoundError(str(file))

    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header_i = _find_header_line(lines, "Param")
    if header_i is None:
        header_i = next((i for i, l in enumerate(lines) if "Param" in l and "Postfit" in l), None)
    if header_i is None:
        raise ValueError(f"Could not find plk table header in {file}")

    rows = []
    for line in lines[header_i + 1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("---") or s.lower().startswith("finishing"):
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        rows.append(parts[:6])

    df = pd.DataFrame(rows, columns=["Param", "Prefit", "Postfit", "Uncertainty", "Difference", "Fit"])
    return df

def read_covmat(file: Path) -> pd.DataFrame:
    if not file.exists():
        raise FileNotFoundError(str(file))

    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header_i = _find_header_line(lines, "Param")
    if header_i is None:
        header_i = next((i for i, l in enumerate(lines) if l.strip().startswith("Param")), None)
    if header_i is None:
        raise ValueError(f"Could not find covmat header in {file}")

    data = "".join(lines[header_i:])
    from io import StringIO
    df = pd.read_csv(StringIO(data), sep=r"\s+", engine="python")
    if "Param" not in df.columns:
        raise ValueError(f"Could not parse covmat into a table in {file}")
    df = df.set_index("Param")
    if "Finishing" in df.index:
        df = df.drop(index="Finishing")
    return df.apply(pd.to_numeric, errors="coerce")

def read_general2(file: Path) -> pd.DataFrame:
    if not file.exists():
        raise FileNotFoundError(str(file))

    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header_i = _find_header_line(lines, "sat")
    if header_i is None:
        header_i = next((i for i, l in enumerate(lines) if ("sat" in l and "post" in l and "err" in l)), None)
    if header_i is None:
        raise ValueError(f"Could not find general2 header in {file}")

    data_lines = []
    for line in lines[header_i:]:
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("finishing"):
            break
        data_lines.append(s)

    from io import StringIO
    data = "\n".join(data_lines) + "\n"
    df = pd.read_csv(StringIO(data), sep=r"\s+", engine="python")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

def read_tim_file(timfile: Path) -> pd.DataFrame:
    rows = []
    with open(timfile, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("C", "#")):
                continue
            if line.startswith(("FORMAT", "MODE", "TIME", "EFAC", "EQUAD", "JUMP", "INCLUDE", "SKIP", "TRACK")):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append(parts)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in range(min(4, df.shape[1])):
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df
