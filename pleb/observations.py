"""Source identities and lossless measurement parsing for release processing.

Identities live in sidecars, never in Tempo2's bounded flag array. Moving a
checkout does not change an identity; changing its source file does.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cached_property
import hashlib
from pathlib import Path

from .tim_utils import parse_tim_flags_from_line


@dataclass(frozen=True)
class Observation:
    line: int
    body: str
    prefix: str
    state: str
    filename: str
    frequency: Decimal
    mjd: Decimal
    uncertainty: Decimal
    site: str
    time: Decimal
    efac: Decimal
    equad: Decimal

    @cached_property
    def flags(self):
        return parse_tim_flags_from_line(self.body, start_col=5)

    @property
    def epoch(self):
        return self.mjd + (
            self.time + Decimal(self.flags.get("-addsat", "0"))
        ) / Decimal(86400)

    @property
    def measurement(self):
        return (
            Path(self.filename).name,
            self.frequency,
            self.mjd,
            self.uncertainty,
        )


def source_id(relative_path: str, file_sha256: str, line: int) -> str:
    """Identify a physical input row, including distinct duplicate copies."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or line < 1:
        raise ValueError("Source identity requires a relative path and positive line")
    value = f"{path.as_posix()}\0{file_sha256}\0{line}"
    return hashlib.sha256(value.encode()).hexdigest()


def read_observations(path: Path, *, allow_includes=False) -> list[Observation]:
    """Read physical FORMAT 1 measurements, including inactive observations.

    TIME, EFAC and EQUAD contexts are recorded so merging backend files cannot
    leak one file's directives into another. INCLUDEs belong in wrappers.
    """
    result = []
    time, efac, equad = Decimal(0), Decimal(1), Decimal(0)
    skip = ended = False
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        commented = parts[0].startswith(("C", "#")) or not parts[0].lower().strip("c")
        if parts[0].startswith("C") and parts[0] != "C":
            rest = parts[0].lstrip("C")
            parts = ["C", *([rest] if rest else []), *parts[1:]]
        if parts[0].startswith("#") and len(parts[0]) > 1:
            parts = ["#", parts[0][1:], *parts[1:]]
        if not commented:
            command = parts[0].upper()
            if command == "END":
                ended = True
            elif command in {"SKIP", "NOSKIP"}:
                skip = command == "SKIP"
            elif not (skip or ended):
                if command == "INCLUDE" and not allow_includes:
                    raise ValueError(
                        f"Expected a leaf TIM, found INCLUDE: {path}:{number}"
                    )
                if command in {
                    "PHASE",
                    "FREQ",
                    "SCALE",
                    "ECORR",
                    "T2EFAC",
                    "T2EQUAD",
                    "JUMP",
                    "TRACK",
                    "EMAX",
                    "EMIN",
                    "FMIN",
                    "FMAX",
                }:
                    raise ValueError(
                        f"Unsupported timing directive {command}: {path}:{number}"
                    )
                if command == "TIME":
                    time += Decimal(parts[1])
                elif command == "EFAC":
                    efac = Decimal(parts[1])
                elif command == "EQUAD":
                    equad = Decimal(parts[1])
        # Reviewed comment prefixes can contain several tokens. Locate the
        # actual measurement rather than relying on a fixed prefix length.
        for index in range(min(12, max(0, len(parts) - 4))):
            try:
                frequency, mjd, uncertainty = map(Decimal, parts[index + 1 : index + 4])
                if not all(v.is_finite() for v in (frequency, mjd, uncertainty)):
                    continue
                if not (frequency >= 0 and 30000 < mjd < 100000 and uncertainty > 0):
                    continue
            except (InvalidOperation, ValueError):
                continue
            if index and not commented:
                raise ValueError(f"Malformed active measurement: {path}:{number}")
            state = (
                "commented"
                if commented
                else "after_END" if ended else "SKIP" if skip else "active"
            )
            result.append(
                Observation(
                    number,
                    " ".join(parts[index:]),
                    " ".join(parts[:index]),
                    state,
                    parts[index],
                    frequency,
                    mjd,
                    uncertainty,
                    parts[index + 4],
                    time,
                    efac,
                    equad,
                )
            )
            break
    return result


def write_observations(path: Path, entries, *, weighted=True):
    """Write (identity, observation, keep, reason) entries and return row locations."""
    lines = ["FORMAT 1", f"MODE {1 if weighted else 0}"]
    time, efac, equad = Decimal(0), Decimal(1), Decimal(0)
    locations = []
    for identity, row, keep, reason in entries:
        if row.time != time:
            lines.append(f"TIME {row.time - time}")
            time = row.time
        for command, previous, current in (
            ("EFAC", efac, row.efac),
            ("EQUAD", equad, row.equad),
        ):
            if previous != current:
                lines.append(f"{command} {current}")
        efac, equad = row.efac, row.equad
        prefix = "" if keep else "C "
        lines.append(prefix + row.body)
        locations.append(
            dict(source_id=identity, line=len(lines), active=bool(keep), reason=reason)
        )
    if time:
        lines.append(f"TIME {-time}")
    if efac != 1:
        lines.append("EFAC 1")
    if equad:
        lines.append("EQUAD 0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return locations
