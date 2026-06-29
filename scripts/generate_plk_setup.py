#!/usr/bin/env python3
"""
Generate a tempo2 plk setup file with flag colour rules for .tim files.

Example:

    python3 make_plk_flag_colours.py \
        --root ./tim-files \
        --output plk_flag_colours.dat \
        --summary

Use the generated file with, for example:

    tempo2 -gr plk -setup plk_flag_colours.dat -colour -sys   -f pulsar.par pulsar.tim
    tempo2 -gr plk -setup plk_flag_colours.dat -colour -group -f pulsar.par pulsar.tim
    tempo2 -gr plk -setup plk_flag_colours.dat -colour -pta   -f pulsar.par pulsar.tim

The output file intentionally uses only simple plk setup syntax:

    background black
    line white
    flag -sys SOME_VALUE 2 16

No comments are written to the .dat file by default.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


DEFAULT_FLAGS = ["-sys", "-group", "-pta"]

# PGPLOT colour indices. 0 is background and 1 is foreground, so avoid both.
# "black" is intended for a black/dark background. It uses bright saturated
# colours and avoids the default dark blue and dark grey.
BLACK_BACKGROUND_COLOURS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]

# More conservative set for files that may be used on either black or white.
# This avoids yellow, light grey, dark grey, black, white, and default dark blue.
DUAL_BACKGROUND_COLOURS = [2, 3, 6, 8, 10, 11, 12, 13]

# Filled/open point symbols that are usually distinguishable in PGPLOT.
DEFAULT_SYMBOLS = [16, 17, 18, 21, 22, 23, 24, 25, 26]


def parse_int_list(text: str, option_name: str) -> List[int]:
    """Parse comma-separated or whitespace-separated integers."""
    raw = text.replace(",", " ").split()
    values: List[int] = []
    for item in raw:
        try:
            values.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{option_name} must contain only integers; got {item!r}"
            ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{option_name} must not be empty")
    return values


def iter_tim_files(root: Path) -> Iterable[Path]:
    """Yield .tim files recursively, case-insensitively."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".tim":
            yield path


def is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    # TEMPO-style comment line.
    if stripped.startswith("C ") or stripped.startswith("c "):
        return True
    return False


def split_tim_line(line: str) -> List[str] | None:
    """Tokenise one .tim line. Return None if the line is malformed."""
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        return None


def collect_flag_values(
    root: Path,
    flags: Sequence[str],
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """
    Scan .tim files and collect values for requested flags.

    Returns:
        values_by_flag, stats
    """
    wanted = set(flags)
    values_by_flag: Dict[str, Set[str]] = {flag: set() for flag in flags}

    stats = {
        "files_scanned": 0,
        "lines_read": 0,
        "blank_or_comment_lines": 0,
        "malformed_lines": 0,
        "missing_value_occurrences": 0,
        "unsupported_space_values": 0,
    }

    for tim_path in iter_tim_files(root):
        stats["files_scanned"] += 1
        try:
            handle = tim_path.open("r", encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"warning: could not read {tim_path}: {exc}", file=sys.stderr)
            continue

        with handle:
            for line in handle:
                stats["lines_read"] += 1

                if is_comment_or_blank(line):
                    stats["blank_or_comment_lines"] += 1
                    continue

                tokens = split_tim_line(line)
                if tokens is None:
                    stats["malformed_lines"] += 1
                    continue

                for i, token in enumerate(tokens):
                    if token not in wanted:
                        continue
                    if i + 1 >= len(tokens):
                        stats["missing_value_occurrences"] += 1
                        continue

                    value = tokens[i + 1]
                    # The plk setup parser is token-based. If a quoted .tim
                    # value contains literal whitespace, there is no safe way to
                    # represent it as a single setup-file token.
                    if any(ch.isspace() for ch in value):
                        stats["unsupported_space_values"] += 1
                        continue

                    values_by_flag[token].add(value)

    return values_by_flag, stats


def sort_values(values: Iterable[str], case_sensitive: bool) -> List[str]:
    if case_sensitive:
        return sorted(values)
    return sorted(values, key=lambda s: (s.casefold(), s))


def assign_pair(index: int, colours: Sequence[int], symbols: Sequence[int]) -> Tuple[int, int]:
    """
    Assign colour first, then symbol.

    For the first len(colours) values, the symbol is constant and colours vary.
    Then the symbol changes and colours cycle again.
    """
    colour = colours[index % len(colours)]
    symbol = symbols[(index // len(colours)) % len(symbols)]
    return colour, symbol


def build_output_lines(
    values_by_flag: Dict[str, Set[str]],
    flags: Sequence[str],
    colours: Sequence[int],
    symbols: Sequence[int],
    case_sensitive: bool,
    include_base_theme: bool,
    background: str,
    foreground: str,
) -> Tuple[List[str], Dict[str, bool]]:
    lines: List[str] = []
    reused_by_flag: Dict[str, bool] = {}

    if include_base_theme:
        lines.append(f"background {background}")
        lines.append(f"line {foreground}")

    max_unique_pairs = len(colours) * len(symbols)

    for flag in flags:
        values = sort_values(values_by_flag.get(flag, set()), case_sensitive)
        reused_by_flag[flag] = len(values) > max_unique_pairs

        for i, value in enumerate(values):
            colour, symbol = assign_pair(i, colours, symbols)
            lines.append(f"flag {flag} {value} {colour} {symbol}")

    return lines, reused_by_flag


def write_output(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Final newline is intentional: convenient for shell tools and concatenation.
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def print_summary(
    root: Path,
    output: Path,
    values_by_flag: Dict[str, Set[str]],
    flags: Sequence[str],
    stats: Dict[str, int],
    colours: Sequence[int],
    symbols: Sequence[int],
    reused_by_flag: Dict[str, bool],
) -> None:
    print(f"Root: {root}")
    print(f"Output: {output}")
    print(f".tim files scanned: {stats['files_scanned']}")
    print(f"Total lines read: {stats['lines_read']}")
    print(f"Blank/comment lines skipped: {stats['blank_or_comment_lines']}")
    print(f"Malformed lines skipped: {stats['malformed_lines']}")
    print(f"Flag occurrences missing values: {stats['missing_value_occurrences']}")
    print(f"Whitespace-containing values skipped: {stats['unsupported_space_values']}")
    print(f"Colours: {' '.join(map(str, colours))}")
    print(f"Symbols: {' '.join(map(str, symbols))}")
    print(f"Unique colour/symbol pairs per flag: {len(colours) * len(symbols)}")

    for flag in flags:
        n = len(values_by_flag.get(flag, set()))
        reused = "yes" if reused_by_flag.get(flag, False) else "no"
        print(f"{flag}: {n} unique values; pair reuse required: {reused}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan TEMPO/TEMPO2 .tim files and generate tempo2 plk "
            "setup colour rules for flag values."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory to scan recursively for .tim files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plk_flag_colours.dat"),
        help="Output .dat file. Default: plk_flag_colours.dat",
    )
    parser.add_argument(
        "--flags",
        nargs="+",
        default=DEFAULT_FLAGS,
        help="Flag names to extract. Default: -sys -group -pta",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print scan and assignment summary.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Sort values case-sensitively. Original spelling is always preserved.",
    )
    parser.add_argument(
        "--palette",
        choices=["black", "dual"],
        default="black",
        help=(
            "Built-in colour palette. 'black' is best for a black background; "
            "'dual' is more conservative for both black and white backgrounds. "
            "Default: black"
        ),
    )
    parser.add_argument(
        "--colours",
        type=lambda s: parse_int_list(s, "--colours"),
        default=None,
        help=(
            "Override colour cycle as comma- or space-separated integers, e.g. "
            "'2,3,5,6,7,8,10,11,12,13,15'."
        ),
    )
    parser.add_argument(
        "--symbols",
        type=lambda s: parse_int_list(s, "--symbols"),
        default=DEFAULT_SYMBOLS,
        help=(
            "Symbol cycle as comma- or space-separated integers. "
            "Default: 16,17,18,21,22,23,24,25,26"
        ),
    )
    parser.add_argument(
        "--background",
        default="black",
        help="Background colour name to write. Default: black",
    )
    parser.add_argument(
        "--foreground",
        default="white",
        help="Foreground/line colour name to write. Default: white",
    )
    parser.add_argument(
        "--flag-lines-only",
        action="store_true",
        help=(
            "Write only 'flag ...' lines. By default the file starts with "
            "'background <colour>' and 'line <colour>'."
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    output = args.output.expanduser()

    if not root.exists():
        parser.error(f"--root does not exist: {root}")
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")

    flags = list(dict.fromkeys(args.flags))  # de-duplicate while preserving order
    if not flags:
        parser.error("--flags must contain at least one flag name")
    for flag in flags:
        if not flag.startswith("-"):
            parser.error(f"flag names should include the leading '-': {flag!r}")

    if args.colours is not None:
        colours = args.colours
    elif args.palette == "black":
        colours = BLACK_BACKGROUND_COLOURS
    else:
        colours = DUAL_BACKGROUND_COLOURS

    symbols = args.symbols

    blocked_colours = {0, 1}
    bad = [c for c in colours if c in blocked_colours]
    if bad:
        parser.error(f"do not use PGPLOT colour indices 0 or 1 in --colours; got {bad}")

    values_by_flag, stats = collect_flag_values(root, flags)

    lines, reused_by_flag = build_output_lines(
        values_by_flag=values_by_flag,
        flags=flags,
        colours=colours,
        symbols=symbols,
        case_sensitive=args.case_sensitive,
        include_base_theme=not args.flag_lines_only,
        background=args.background,
        foreground=args.foreground,
    )

    write_output(output, lines)

    if args.summary:
        print_summary(
            root=root,
            output=output,
            values_by_flag=values_by_flag,
            flags=flags,
            stats=stats,
            colours=colours,
            symbols=symbols,
            reused_by_flag=reused_by_flag,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())