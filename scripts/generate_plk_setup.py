#!/usr/bin/env python3
"""
Generate tempo2 plk setup files with safe flag colour rules for .tim files.

Designed for a black-background plk session by default.

Typical usage:

  python3 generate_plk_setup.py \
      --root ./tim-files \
      --output-prefix plk_flag_colours \
      --summary

This creates split files by default:

  plk_flag_colours_sys.dat
  plk_flag_colours_group.dat
  plk_flag_colours_pta.dat

Use one setup file at a time, matching the flag used with -colour:

  tempo2 -gr plk -setup plk_flag_colours_sys.dat   -colour -sys   -f pulsar.par pulsar.tim
  tempo2 -gr plk -setup plk_flag_colours_group.dat -colour -group -f pulsar.par pulsar.tim
  tempo2 -gr plk -setup plk_flag_colours_pta.dat   -colour -pta   -f pulsar.par pulsar.tim

The generated .dat files use only these setup lines:

  background black
  line white
  flag <flag-name> <flag-value> <pgplot-colour-index> <pgplot-symbol-index>

No comments are written to the .dat files.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

DEFAULT_FLAGS = ["-sys", "-group", "-pta"]

# PGPLOT predefined colour indices, deliberately avoiding low-contrast choices.
#
# For a black background we avoid:
#   0  background / black
#   1  foreground / usually white
#   4  default dark blue, weak on black
#   14 dark grey
#   15 light grey, visible but explicitly a grey derivative
#
# The retained values are chromatic colours and chromatic derivatives:
#   red, green, cyan, magenta, yellow, orange,
#   yellow-green, green-cyan, blue-cyan, blue-magenta, red-magenta.
BLACK_CHROMATIC_COLOURS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# For white backgrounds, omit colours that are commonly too light on white.
# Still no grey derivatives.
WHITE_CHROMATIC_COLOURS = [2, 3, 4, 6, 8, 12, 13]

# Conservative palette for users who genuinely need one file usable on both
# black and white. This is smaller but safer.
DUAL_CHROMATIC_COLOURS = [2, 3, 6, 8, 12, 13]

# Default: one marker only. This removes marker-code ambiguity. Colours will
# cycle for large value sets, which is acceptable and safer.
SINGLE_SAFE_SYMBOL = [16]

# Optional: standard PGPLOT markers for users who want more distinct pairs.
# Enable with --multi-symbols only after the single-marker output runs cleanly.
MULTI_SAFE_SYMBOLS = [16, 17, 18, 21, 22, 23]

# Current plk uses fixed arrays of size 100 for setup-file flag rules.
# Keep a margin by default; raise only if you know what you are doing.
DEFAULT_MAX_FLAG_RULES_PER_FILE = 95

# Strict low-contrast filters by palette mode.
DISALLOWED_BY_PALETTE = {
    "black": {0, 1, 4, 14, 15},
    "white": {0, 1, 5, 7, 9, 10, 14, 15},
    "dual": {0, 1, 4, 5, 7, 9, 10, 11, 14, 15},
}


def parse_int_list(text: str, option_name: str) -> List[int]:
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


def validate_colours(
    colours: Sequence[int], palette: str, allow_low_contrast: bool
) -> None:
    outside = [c for c in colours if c < 2 or c > 15]
    if outside:
        raise ValueError(
            "PGPLOT point colour indices must be predefined non-background "
            f"indices in 2..15. Bad values: {outside}"
        )

    if not allow_low_contrast:
        disallowed = DISALLOWED_BY_PALETTE[palette]
        bad = [c for c in colours if c in disallowed]
        if bad:
            raise ValueError(
                f"palette {palette!r} disallows low-contrast/background-like "
                f"colour indices {sorted(disallowed)}. Bad values: {bad}. "
                "Use --allow-low-contrast only if you intentionally want this."
            )


def validate_symbols(symbols: Sequence[int]) -> None:
    outside = [s for s in symbols if s < 0 or s > 31]
    if outside:
        raise ValueError(
            "PGPLOT marker symbols should be standard symbols in 0..31. "
            f"Bad values: {outside}"
        )


def iter_tim_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".tim":
            yield path


def is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith("C ") or stripped.startswith("c "):
        return True
    return False


def split_tim_line(line: str) -> List[str] | None:
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        return None


def collect_flag_values(
    root: Path, flags: Sequence[str]
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    wanted = set(flags)
    values_by_flag: Dict[str, Set[str]] = {flag: set() for flag in flags}

    stats = {
        "files_scanned": 0,
        "lines_read": 0,
        "blank_or_comment_lines": 0,
        "malformed_lines": 0,
        "missing_value_occurrences": 0,
        "too_long_values": 0,
        "whitespace_values": 0,
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

                    # plk reads setup values using fixed 100-char buffers.
                    if len(token) >= 100 or len(value) >= 100:
                        stats["too_long_values"] += 1
                        continue
                    if any(ch.isspace() for ch in value):
                        stats["whitespace_values"] += 1
                        continue

                    values_by_flag[token].add(value)

    return values_by_flag, stats


def sort_values(values: Iterable[str], case_sensitive: bool) -> List[str]:
    if case_sensitive:
        return sorted(values)
    return sorted(values, key=lambda s: (s.casefold(), s))


def assign_pair(
    i: int, colours: Sequence[int], symbols: Sequence[int]
) -> Tuple[int, int]:
    # Colour cycles fastest; symbol changes after a full colour cycle.
    colour = colours[i % len(colours)]
    symbol = symbols[(i // len(colours)) % len(symbols)]
    return colour, symbol


def flag_to_suffix(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_") or "flag"


def build_flag_lines(
    flag: str, values: Sequence[str], colours: Sequence[int], symbols: Sequence[int]
) -> List[str]:
    lines: List[str] = []
    for i, value in enumerate(values):
        colour, symbol = assign_pair(i, colours, symbols)
        lines.append(f"flag {flag} {value} {colour} {symbol}")
    return lines


def write_dat(
    path: Path,
    lines: Sequence[str],
    include_theme: bool,
    background: str,
    foreground: str,
) -> None:
    out_lines: List[str] = []
    if include_theme:
        out_lines.append(f"background {background}")
        out_lines.append(f"line {foreground}")
    out_lines.extend(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def write_split_outputs(
    output_prefix: Path,
    values_by_flag: Dict[str, Set[str]],
    flags: Sequence[str],
    colours: Sequence[int],
    symbols: Sequence[int],
    case_sensitive: bool,
    include_theme: bool,
    background: str,
    foreground: str,
    max_rules_per_file: int,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}

    for flag in flags:
        values = sort_values(values_by_flag.get(flag, set()), case_sensitive)
        if len(values) > max_rules_per_file:
            raise RuntimeError(
                f"{flag} has {len(values)} unique values. This exceeds the configured "
                f"safe per-file limit of {max_rules_per_file}. Current plk uses a fixed "
                "100-rule array for setup-file flag rules. Split the data set, reduce "
                "the values, or raise --max-rules-per-file up to 100 at your own risk."
            )

        suffix = flag_to_suffix(flag)
        path = output_prefix.with_name(f"{output_prefix.name}_{suffix}.dat")
        lines = build_flag_lines(flag, values, colours, symbols)
        write_dat(path, lines, include_theme, background, foreground)
        outputs[flag] = path

    return outputs


def write_combined_output(
    output: Path,
    values_by_flag: Dict[str, Set[str]],
    flags: Sequence[str],
    colours: Sequence[int],
    symbols: Sequence[int],
    case_sensitive: bool,
    include_theme: bool,
    background: str,
    foreground: str,
    max_rules_per_file: int,
) -> Dict[str, Path]:
    all_lines: List[str] = []

    for flag in flags:
        values = sort_values(values_by_flag.get(flag, set()), case_sensitive)
        all_lines.extend(build_flag_lines(flag, values, colours, symbols))

    if len(all_lines) > max_rules_per_file:
        raise RuntimeError(
            f"combined output would contain {len(all_lines)} flag rules, exceeding "
            f"the safe limit of {max_rules_per_file}. Use split output instead."
        )

    write_dat(output, all_lines, include_theme, background, foreground)
    return {"combined": output}


def print_summary(
    root: Path,
    values_by_flag: Dict[str, Set[str]],
    flags: Sequence[str],
    stats: Dict[str, int],
    colours: Sequence[int],
    symbols: Sequence[int],
    outputs: Dict[str, Path],
    max_rules_per_file: int,
    palette: str,
) -> None:
    print(f"Root: {root}")
    print(f"Palette mode: {palette}")
    print(f".tim files scanned: {stats['files_scanned']}")
    print(f"Total lines read: {stats['lines_read']}")
    print(f"Blank/comment lines skipped: {stats['blank_or_comment_lines']}")
    print(f"Malformed lines skipped: {stats['malformed_lines']}")
    print(f"Flag occurrences missing values: {stats['missing_value_occurrences']}")
    print(f"Overlong flag/value tokens skipped: {stats['too_long_values']}")
    print(f"Whitespace-containing values skipped: {stats['whitespace_values']}")
    print(f"Colours used: {' '.join(map(str, colours))}")
    print(f"Symbols used: {' '.join(map(str, symbols))}")
    print(f"Unique colour/symbol pairs per flag: {len(colours) * len(symbols)}")
    print(f"Safe maximum flag rules per setup file: {max_rules_per_file}")

    for flag in flags:
        n = len(values_by_flag.get(flag, set()))
        pair_reuse = n > len(colours) * len(symbols)
        print(
            f"{flag}: {n} unique values; colour/symbol pair reuse: {'yes' if pair_reuse else 'no'}"
        )

    for key, path in outputs.items():
        print(f"Output {key}: {path}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate safe tempo2 plk flag-colour setup files from .tim files."
    )
    p.add_argument(
        "--root", required=True, type=Path, help="Root directory to scan recursively."
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("plk_flag_colours"),
        help="Prefix for split output files. Default: plk_flag_colours",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("plk_flag_colours.dat"),
        help="Output path used only with --combined.",
    )
    p.add_argument(
        "--flags",
        nargs="+",
        default=DEFAULT_FLAGS,
        help="Flags to scan. Default: -sys -group -pta",
    )
    p.add_argument("--summary", action="store_true", help="Print a summary.")
    p.add_argument(
        "--case-sensitive", action="store_true", help="Sort values case-sensitively."
    )
    p.add_argument(
        "--palette",
        choices=["black", "white", "dual"],
        default="black",
        help=(
            "Contrast palette. 'black' avoids black/white/grey/dark-blue indices; "
            "'white' avoids pale colours; 'dual' is smaller but usable on both. "
            "Default: black"
        ),
    )
    p.add_argument(
        "--colours",
        type=lambda s: parse_int_list(s, "--colours"),
        default=None,
        help="Override colour cycle, e.g. '2,3,5,6,7,8,9,10,11,12,13'.",
    )
    p.add_argument(
        "--allow-low-contrast",
        action="store_true",
        help="Allow colours normally rejected for the selected palette.",
    )
    p.add_argument(
        "--multi-symbols",
        action="store_true",
        help="Use several marker symbols. Default uses only symbol 16.",
    )
    p.add_argument(
        "--symbols",
        type=lambda s: parse_int_list(s, "--symbols"),
        default=None,
        help="Override symbols. Must be standard PGPLOT symbols in 0..31.",
    )
    p.add_argument(
        "--background",
        default="black",
        help="Background colour name written to .dat. Default: black",
    )
    p.add_argument(
        "--foreground",
        default="white",
        help="Foreground/line colour name written to .dat. Default: white",
    )
    p.add_argument(
        "--flag-lines-only",
        action="store_true",
        help="Write only flag lines, omitting background/line entries.",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Write one combined .dat file. Not recommended when total rules may exceed 100.",
    )
    p.add_argument(
        "--max-rules-per-file",
        type=int,
        default=DEFAULT_MAX_FLAG_RULES_PER_FILE,
        help="Safe flag-rule limit per .dat file. Default: 95; absolute plk array size is 100.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    if not root.exists():
        parser.error(f"--root does not exist: {root}")
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")

    flags = list(dict.fromkeys(args.flags))
    for flag in flags:
        if not flag.startswith("-"):
            parser.error(f"flag names must include the leading '-': {flag!r}")
        if len(flag) >= 100:
            parser.error(f"flag name is too long for plk's 100-char buffers: {flag!r}")

    if args.max_rules_per_file < 1 or args.max_rules_per_file > 100:
        parser.error("--max-rules-per-file must be between 1 and 100")

    if args.colours is not None:
        colours = args.colours
    elif args.palette == "black":
        colours = BLACK_CHROMATIC_COLOURS
    elif args.palette == "white":
        colours = WHITE_CHROMATIC_COLOURS
    else:
        colours = DUAL_CHROMATIC_COLOURS

    if args.symbols is not None:
        symbols = args.symbols
    elif args.multi_symbols:
        symbols = MULTI_SAFE_SYMBOLS
    else:
        symbols = SINGLE_SAFE_SYMBOL

    try:
        validate_colours(colours, args.palette, args.allow_low_contrast)
        validate_symbols(symbols)
    except ValueError as exc:
        parser.error(str(exc))

    values_by_flag, stats = collect_flag_values(root, flags)
    include_theme = not args.flag_lines_only

    try:
        if args.combined:
            outputs = write_combined_output(
                output=args.output.expanduser(),
                values_by_flag=values_by_flag,
                flags=flags,
                colours=colours,
                symbols=symbols,
                case_sensitive=args.case_sensitive,
                include_theme=include_theme,
                background=args.background,
                foreground=args.foreground,
                max_rules_per_file=args.max_rules_per_file,
            )
        else:
            outputs = write_split_outputs(
                output_prefix=args.output_prefix.expanduser(),
                values_by_flag=values_by_flag,
                flags=flags,
                colours=colours,
                symbols=symbols,
                case_sensitive=args.case_sensitive,
                include_theme=include_theme,
                background=args.background,
                foreground=args.foreground,
                max_rules_per_file=args.max_rules_per_file,
            )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.summary:
        print_summary(
            root=root,
            values_by_flag=values_by_flag,
            flags=flags,
            stats=stats,
            colours=colours,
            symbols=symbols,
            outputs=outputs,
            max_rules_per_file=args.max_rules_per_file,
            palette=args.palette,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
