#!/usr/bin/env python3
"""
tempo2_par_model_check.py

## GMS: This is a standalone script that must be invoked fdirectly on par files that need to be tested.

Standalone validator for Tempo2 .par files. It checks that binary-model
parameters in a par file are compatible with the selected Tempo2 BINARY model.

It is intentionally conservative: Tempo2 accepts many global parameters through
a common parser; this script primarily polices binary-model parameters and only
flags unknown non-binary tokens when --strict-known is requested.

Usage:
  python tempo2_par_model_check.py pulsar.par
  python tempo2_par_model_check.py --mode compatibility pulsar.par
  python tempo2_par_model_check.py --tempo2-src /path/to/tempo2 pulsar.par
  python tempo2_par_model_check.py --self-test

Exit codes:
  0: no errors
  1: validation errors
  2: command-line or file error
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ----------------------------- Normalisation -----------------------------

ALIASES = {
    # common Tempo/Tempo2 spellings
    "E": "ECC",
    "ECC": "ECC",
    "XDOT": "A1DOT",
    "X": "A1",        # seen in older files / literature; A1 is Tempo2's internal label
    "T0ASC": "TASC",
    "TASC": "TASC",
    "KIN": "KIN",
    "KOM": "KOM",
}

# A non-exhaustive list of global parameters. These are allowed regardless of
# BINARY model. Regex families cover most normal timing/noise/instrument terms.
GLOBAL_EXACT = {
    "PSR", "PSRJ", "PSRB", "RA", "RAJ", "DEC", "DECJ", "ELONG", "ELAT",
    "PMRA", "PMDEC", "PMELONG", "PMELAT", "PX", "POSEPOCH", "PEPOCH",
    "DMEPOCH", "DM", "DMDOT", "CM", "START", "FINISH", "EPHEM", "UNITS",
    "CLK", "CLKCORR", "CORRECT_TROPOSPHERE", "PLANET_SHAPIRO", "DILATEFREQ",
    "DILATEFREQREF", "TIMEEPH", "T2CMETHOD", "MODE", "TRACK", "TZRMJD",
    "TZRSITE", "TZRFRQ", "NE_SW", "SWM", "EPHVER", "NITS", "NTOA", "CHI2R",
    "TRES", "JUMP", "DMMODEL", "DMOFF", "RM", "PMRV", "DIST", "H0",
    "QIFUNC", "IFUNC", "SIFUNC", "GWM_AMP", "GWM_EPOCH", "GWM_PHI", "GWM_THETA",
}
GLOBAL_PATTERNS = [
    re.compile(r"^F\d+$"),                  # spin frequency derivatives F0, F1, ...
    re.compile(r"^P\d+$"),                  # period form P0, P1, ...
    re.compile(r"^DMX(_R\d+)?_?\d*$"),
    re.compile(r"^FD\d+$"),
    re.compile(r"^FDJUMP\d*$"),
    re.compile(r"^JUMP\d*$"),
    re.compile(r"^WAVE\d+$"),
    re.compile(r"^WAVEEPOCH$"),
    re.compile(r"^GL[A-Z0-9_]*$"),          # glitch families
    re.compile(r"^TN[A-Z0-9_]*$"),          # TempoNest/noise terms
    re.compile(r"^TNE[A-Z0-9_]*$"),
    re.compile(r"^EFAC\d*$"),
    re.compile(r"^EQUAD\d*$"),
    re.compile(r"^ECORR\d*$"),
    re.compile(r"^DMEFAC\d*$"),
    re.compile(r"^DMEQUAD\d*$"),
    re.compile(r"^T2E[A-Z0-9_]*$"),
    re.compile(r"^T2EFAC\d*$"),
    re.compile(r"^T2EQUAD\d*$"),
]

# Canonical binary-ish names. Only these are model-restricted by default.
BINARY_EXACT = {
    "A0", "B0", "A1", "A1DOT", "A1_2", "A1_3", "A1_4", "A1_5",
    "PB", "PBDOT", "XPBDOT", "ECC", "EDOT", "T0", "TASC", "OM", "OMDOT",
    "GAMMA", "SINI", "M2", "MTOT", "DR", "DTHETA", "H3", "H4", "STIG",
    "SHAPMAX", "KIN", "KOM", "K96", "KOMDOT", "KINDOT", "EPS1", "EPS2",
    "EPS1DOT", "EPS2DOT", "ORBWAVEC0", "ORBWAVES0", "ORBWAVE_OM",
    "ORBWAVE_EPOCH", "ORBWAVE", "ORBIFUNC", "NHARM", "AFAC", "NDDOP",
    "BINARY", "OM2DOT", "BETA", "XOMDOT", "MASSFN", "FB", "BPJ",
}
BINARY_PATTERNS = [
    re.compile(r"^FB\d+$"),
    re.compile(r"^BPJ\d+$"),
    re.compile(r"^A1_\d+$"),
    re.compile(r"^ECC_\d+$"),
    re.compile(r"^T0_\d+$"),
    re.compile(r"^PB_\d+$"),
    re.compile(r"^OM_\d+$"),
]


def canonical_param(raw: str) -> str:
    """Map a par-file keyword to a coarse canonical Tempo2-style name."""
    k = raw.strip().upper()
    k = ALIASES.get(k, k)
    if re.fullmatch(r"FB\d+", k):
        return "FB"
    if re.fullmatch(r"BPJ\d+", k):
        return "BPJ"
    # Keep DMX/FD/F/JUMP families distinct enough to classify as global.
    return k


def is_global_param(canon: str, raw: Optional[str] = None) -> bool:
    k = canon.upper()
    r = (raw or canon).upper()
    return k in GLOBAL_EXACT or any(p.match(k) or p.match(r) for p in GLOBAL_PATTERNS)


def is_binary_param(canon: str, raw: Optional[str] = None) -> bool:
    k = canon.upper()
    r = (raw or canon).upper()
    return k in BINARY_EXACT or any(p.match(k) or p.match(r) for p in BINARY_PATTERNS)


# ----------------------------- Rule metadata -----------------------------

@dataclass(frozen=True)
class ModelRule:
    allowed: Set[str]
    required_all: Tuple[str, ...] = ()
    required_any: Tuple[Tuple[str, ...], ...] = ()
    require_pairs: Tuple[Tuple[str, ...], ...] = ()
    derived: Set[str] = field(default_factory=set)
    compatibility: Set[str] = field(default_factory=set)
    notes: Dict[str, str] = field(default_factory=dict)


BT = {"PB", "A1", "ECC", "T0", "OM", "PBDOT", "A1DOT", "OMDOT", "EDOT", "GAMMA"}
DD = {
    "PB", "A1", "ECC", "T0", "OM", "PBDOT", "A1DOT", "OMDOT", "EDOT",
    "GAMMA", "SINI", "M2", "DR", "DTHETA", "XPBDOT",
}
ELL1 = {
    "PB", "FB", "A1", "A1DOT", "TASC", "EPS1", "EPS2", "EPS1DOT", "EPS2DOT",
    "PBDOT", "SINI", "M2", "ORBIFUNC",
}
ELL1_FORBIDDEN_NOTES = {
    "ECC": "ELL1 uses EPS1/EPS2 instead of ECC/E.",
    "OM": "ELL1 uses EPS1/EPS2 instead of OM.",
    "T0": "ELL1 uses TASC instead of T0.",
}

DEFAULT_RULES: Dict[str, ModelRule] = {
    "BT": ModelRule(
        allowed=BT,
        required_all=("PB", "A1", "ECC", "T0", "OM"),
        notes={"EDOT": "BT has derivative/update code for EDOT, but the delay path initializes edot to zero in current source; treat fitted EDOT with suspicion."},
    ),
    "BTJ": ModelRule(
        allowed=BT | {"BPJ"},
        required_all=("PB", "A1", "ECC", "T0", "OM"),
    ),
    "BTX": ModelRule(
        allowed=(BT | {"FB"}) - {"PB"},
        required_all=("A1", "ECC", "T0", "OM"),
        required_any=(("FB", "PB"),),
    ),
    "DD": ModelRule(
        allowed=DD,
        required_all=("PB", "A1", "ECC", "T0", "OM"),
    ),
    "DDS": ModelRule(
        allowed=(DD | {"SHAPMAX"}) - {"DR", "DTHETA", "XPBDOT"},
        required_all=("PB", "A1", "ECC", "T0", "OM", "SHAPMAX"),
        compatibility={"SINI"},
        notes={"SINI": "DDS source comments out SINI input and uses SHAPMAX; SINI is update/derivative-era compatibility, not the intended shape parameter."},
    ),
    "DDH": ModelRule(
        allowed=DD | {"H3", "STIG"},
        required_all=("PB", "A1", "ECC", "T0", "OM", "H3", "STIG"),
        require_pairs=(("H3", "STIG"),),
        compatibility={"SINI", "M2"},
        notes={"SINI": "DDH derives sin(i) from STIG.", "M2": "DDH derives the Shapiro mass term from H3/STIG."},
    ),
    "DDGR": ModelRule(
        # Current source has derivative/update branches for SINI and PBDOT, but it also computes
        # GR-derived values via mass2dd; keep them allowed but warn in strict mode.
        allowed={"PB", "A1", "ECC", "T0", "OM", "M2", "MTOT", "SINI", "PBDOT", "XPBDOT", "A1DOT"},
        required_all=("PB", "A1", "ECC", "T0", "OM", "M2", "MTOT"),
        derived={"GAMMA", "OMDOT", "DR", "DTHETA"},
        compatibility={"SINI", "PBDOT"},
        notes={
            "SINI": "DDGR computes SINI from masses and Keplerian parameters when setting derived parameters; current source also has SINI derivative/update paths.",
            "PBDOT": "DDGR computes PBDOT through mass2dd when setting derived parameters; current source also has PBDOT derivative/update paths.",
            "OMDOT": "DDGR computes OMDOT from the GR mass model.",
            "GAMMA": "DDGR computes GAMMA from the GR mass model.",
        },
    ),
    "DDK": ModelRule(
        allowed=(DD | {"KIN", "KOM", "K96", "KOMDOT", "KINDOT"}) - {"XPBDOT", "DR", "DTHETA"},
        required_all=("PB", "A1", "ECC", "T0", "OM", "KIN", "KOM"),
        compatibility={"SINI", "A1DOT"},
    ),
    "ELL1": ModelRule(
        allowed=ELL1,
        required_all=("A1", "TASC", "EPS1", "EPS2"),
        required_any=(("PB", "FB"),),
        notes=ELL1_FORBIDDEN_NOTES,
    ),
    "ELL1H": ModelRule(
        allowed=ELL1 | {"H3", "H4", "STIG", "NHARM", "OMDOT"},
        required_all=("A1", "TASC", "EPS1", "EPS2", "H3"),
        required_any=(("PB", "FB"),),
        notes={
            **ELL1_FORBIDDEN_NOTES,
            "STIG": "In ELL1H, H4/NHARM mode takes precedence and STIG is ignored if H4 is also supplied.",
            "NHARM": "ELL1H uses NHARM only with H4; NHARM without H4 is suspicious.",
        },
    ),
    "ELL1K": ModelRule(
        allowed=ELL1 | {"OMDOT", "LNEDOT", "EPS1DOT", "EPS2DOT"},
        required_all=("A1", "TASC", "EPS1", "EPS2"),
        required_any=(("PB", "FB"),),
        notes=ELL1_FORBIDDEN_NOTES,
    ),
    # Broad/superset models. These are deliberately permissive: use --tempo2-src
    # with the current tree if you need exact source-derived sets for these.
    "T2": ModelRule(
        allowed=BT | DD | ELL1 | {"H3", "H4", "STIG", "SHAPMAX", "KIN", "KOM", "MTOT", "FB", "NHARM", "A0", "B0"},
        required_any=(("PB", "FB"),),
    ),
    "T2-PTA": ModelRule(
        allowed=BT | DD | ELL1 | {"H3", "H4", "STIG", "SHAPMAX", "KIN", "KOM", "MTOT", "FB", "NHARM", "A0", "B0"},
        required_any=(("PB", "FB"),),
    ),
    "MSS": ModelRule(
        allowed=BT | DD | {"MTOT"},
        required_all=("PB", "A1", "ECC", "T0", "OM"),
    ),
}

MODEL_FILE_BY_NAME = {
    "BT": "BTmodel.C",
    "BTJ": "BTJmodel.C",
    "BTX": "BTXmodel.C",
    "DD": "DDmodel.C",
    "DDS": "DDSmodel.C",
    "DDH": "DDHmodel.C",
    "DDGR": "DDGRmodel.C",
    "DDK": "DDKmodel.C",
    "ELL1": "ELL1model.C",
    "ELL1H": "ELL1Hmodel.C",
    "ELL1K": "ELL1kmodel.C",
    "T2": "T2model.C",
    "T2-PTA": "T2-PTAmodel.C",
    "MSS": "MSSmodel.C",
}

PARAM_UNDERSCORE_TO_PAR = {
    "pb": "PB", "a1": "A1", "ecc": "ECC", "t0": "T0", "om": "OM",
    "pbdot": "PBDOT", "xpbdot": "XPBDOT", "a1dot": "A1DOT", "omdot": "OMDOT",
    "edot": "EDOT", "gamma": "GAMMA", "sini": "SINI", "m2": "M2", "mtot": "MTOT",
    "dr": "DR", "dtheta": "DTHETA", "eps1": "EPS1", "eps2": "EPS2", "eps1dot": "EPS1DOT",
    "eps2dot": "EPS2DOT", "tasc": "TASC", "fb": "FB", "h3": "H3", "h4": "H4",
    "stig": "STIG", "shapmax": "SHAPMAX", "kin": "KIN", "kom": "KOM", "nharm": "NHARM",
    "bpj": "BPJ", "a0": "A0", "b0": "B0", "orbifunc": "ORBIFUNC",
}


def source_derived_rules(src: pathlib.Path, base: Dict[str, ModelRule]) -> Dict[str, ModelRule]:
    """Optionally augment embedded rules with param_* references from a Tempo2 source tree."""
    out = dict(base)
    for model, rel in MODEL_FILE_BY_NAME.items():
        f = src / rel
        if not f.exists():
            continue
        text = f.read_text(errors="replace")
        found = set()
        for name in re.findall(r"\bparam_([A-Za-z0-9_]+)\b", text):
            par = PARAM_UNDERSCORE_TO_PAR.get(name.lower())
            if par and is_binary_param(par):
                found.add(par)
        if found:
            existing = out.get(model, ModelRule(allowed=set()))
            # Union: prevents false negatives from hand rules while letting current source
            # add newly referenced binary parameters.
            out[model] = ModelRule(
                allowed=set(existing.allowed) | found,
                required_all=existing.required_all,
                required_any=existing.required_any,
                require_pairs=existing.require_pairs,
                derived=set(existing.derived),
                compatibility=set(existing.compatibility),
                notes=dict(existing.notes),
            )
    return out


# ----------------------------- Par parsing -----------------------------

@dataclass(frozen=True)
class ParamRec:
    raw: str
    canon: str
    value: Optional[str]
    line_no: int
    text: str


def strip_inline_comment(line: str) -> str:
    # Tempo2 par files are whitespace-delimited; do not attempt full shell-style parsing.
    return line.split("#", 1)[0].strip()


def parse_par(path: pathlib.Path) -> Tuple[Optional[str], List[ParamRec]]:
    records: List[ParamRec] = []
    binary_model: Optional[str] = None
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        text = strip_inline_comment(line)
        if not text:
            continue
        if text.upper().startswith("C ") or text.upper() == "C":
            continue
        parts = text.split()
        if not parts:
            continue
        raw = parts[0]
        value = parts[1] if len(parts) > 1 else None
        canon = canonical_param(raw)
        rec = ParamRec(raw=raw, canon=canon, value=value, line_no=n, text=line.rstrip())
        records.append(rec)
        if canon == "BINARY" and value:
            binary_model = canonical_model(value)
    return binary_model, records


def canonical_model(value: str) -> str:
    m = value.strip().upper()
    if m == "ELL1K":
        return "ELL1K"
    if m in {"T2PTA", "T2_PTA"}:
        return "T2-PTA"
    return m


# ----------------------------- Validation -----------------------------

@dataclass(frozen=True)
class Diagnostic:
    severity: str  # ERROR, WARNING
    line_no: Optional[int]
    message: str


def has_param(records: Sequence[ParamRec], canon: str) -> bool:
    return any(r.canon == canon for r in records)


def validate(
    path: pathlib.Path,
    rules: Dict[str, ModelRule],
    *,
    mode: str = "strict",
    require_core: bool = True,
    strict_known: bool = False,
) -> List[Diagnostic]:
    binary, records = parse_par(path)
    diags: List[Diagnostic] = []

    present = {r.canon for r in records}
    binary_records = [r for r in records if r.canon != "BINARY" and is_binary_param(r.canon, r.raw)]

    if binary is None:
        for r in binary_records:
            diags.append(Diagnostic("ERROR", r.line_no, f"binary parameter {r.raw} is present but no BINARY model is set"))
        return diags

    if binary not in rules:
        diags.append(Diagnostic("ERROR", None, f"unknown or unsupported BINARY model {binary!r}"))
        return diags

    rule = rules[binary]

    if require_core:
        for p in rule.required_all:
            if p not in present:
                diags.append(Diagnostic("ERROR", None, f"BINARY {binary} requires {p}, but it is missing"))
        for group in rule.required_any:
            if not any(g in present for g in group):
                diags.append(Diagnostic("ERROR", None, f"BINARY {binary} requires one of {', '.join(group)}, but none is present"))

    for pair in rule.require_pairs:
        got = [p for p in pair if p in present]
        if got and len(got) != len(pair):
            missing = [p for p in pair if p not in present]
            diags.append(Diagnostic("ERROR", None, f"BINARY {binary} requires {', '.join(pair)} as a set; missing {', '.join(missing)}"))

    # Model-specific relationship checks not expressible as pure allow-list rules.
    if binary == "ELL1H":
        if "NHARM" in present and "H4" not in present:
            diags.append(Diagnostic("WARNING", None, "ELL1H NHARM is meaningful only with H4; Tempo2 logs an error and reverts if NHARM is supplied without H4"))
        if "H4" in present and "STIG" in present:
            diags.append(Diagnostic("WARNING", None, "ELL1H source gives H4/NHARM mode precedence and ignores STIG when H4 and STIG are both supplied"))

    for r in records:
        if r.canon == "BINARY":
            continue
        if is_global_param(r.canon, r.raw):
            continue
        if not is_binary_param(r.canon, r.raw):
            if strict_known:
                diags.append(Diagnostic("WARNING", r.line_no, f"unclassified non-binary parameter {r.raw}; not checked against BINARY {binary}"))
            continue

        if r.canon in rule.allowed:
            if r.canon in rule.compatibility:
                sev = "WARNING" if mode == "compatibility" else "WARNING"
                msg = rule.notes.get(r.canon, f"{r.raw} is accepted for BINARY {binary}, but is compatibility/suspicious in this model")
                diags.append(Diagnostic(sev, r.line_no, msg))
            elif r.canon in rule.notes and mode == "strict":
                diags.append(Diagnostic("WARNING", r.line_no, rule.notes[r.canon]))
            continue

        if r.canon in rule.derived:
            sev = "WARNING" if mode == "compatibility" else "ERROR"
            msg = rule.notes.get(r.canon, f"{r.raw} is derived internally by BINARY {binary}, not an independent model parameter")
            diags.append(Diagnostic(sev, r.line_no, msg))
            continue

        extra = rule.notes.get(r.canon, "")
        suffix = f" {extra}" if extra else ""
        diags.append(Diagnostic("ERROR", r.line_no, f"{r.raw} is not valid for BINARY {binary}.{suffix}"))

    return diags


def print_diags(path: pathlib.Path, diags: Sequence[Diagnostic]) -> None:
    if not diags:
        print(f"OK: {path}")
        return
    for d in diags:
        loc = f"{path}:{d.line_no}" if d.line_no is not None else str(path)
        print(f"{loc}: {d.severity}: {d.message}")


# ----------------------------- Self tests -----------------------------

SELF_TEST_CASES = [
    ("ell1_good", """
PSRJ J0000+0000
F0 100
DM 10
BINARY ELL1
A1 1
PB 1
TASC 58000
EPS1 1e-6
EPS2 2e-6
""", 0),
    ("ell1_bad_ecc", """
PSRJ J0000+0000
F0 100
BINARY ELL1
A1 1
PB 1
TASC 58000
EPS1 1e-6
EPS2 2e-6
ECC 0.1
""", 1),
    ("bt_good", """
PSRJ J0000+0000
F0 100
BINARY BT
A1 1
PB 1
T0 58000
ECC 0.1
OM 20
""", 0),
    ("bt_bad_eps", """
PSRJ J0000+0000
F0 100
BINARY BT
A1 1
PB 1
T0 58000
ECC 0.1
OM 20
EPS1 1e-6
""", 1),
    ("ddh_missing_stig", """
PSRJ J0000+0000
F0 100
BINARY DDH
A1 1
PB 1
T0 58000
ECC 0.1
OM 20
H3 1e-7
""", 1),
    ("no_binary_with_pb", """
PSRJ J0000+0000
F0 100
PB 1
A1 1
""", 1),
    ("ddgr_warn_sini", """
PSRJ J0000+0000
F0 100
BINARY DDGR
A1 1
PB 1
T0 58000
ECC 0.1
OM 20
M2 1.2
MTOT 2.4
SINI 0.9
""", 0),
    ("ddgr_reject_gamma", """
PSRJ J0000+0000
F0 100
BINARY DDGR
A1 1
PB 1
T0 58000
ECC 0.1
OM 20
M2 1.2
MTOT 2.4
GAMMA 0.001
""", 1),
]


def run_self_tests() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        for name, content, expect_error in SELF_TEST_CASES:
            f = root / f"{name}.par"
            f.write_text(content.strip() + "\n")
            diags = validate(f, DEFAULT_RULES, mode="strict")
            got_error = any(d.severity == "ERROR" for d in diags)
            ok = int(got_error) == expect_error
            print(f"{name}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                print_diags(f, diags)
                failures += 1
    return 1 if failures else 0


# ----------------------------- CLI -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate Tempo2 .par binary parameters against selected BINARY model")
    p.add_argument("par_files", nargs="*", type=pathlib.Path)
    p.add_argument("--mode", choices=("strict", "compatibility"), default="strict",
                   help="strict errors on derived/incompatible params; compatibility downgrades derived params to warnings")
    p.add_argument("--no-require-core", action="store_true",
                   help="do not require the usual/core parameters for the selected binary model")
    p.add_argument("--strict-known", action="store_true",
                   help="warn on unclassified non-binary parameters")
    p.add_argument("--tempo2-src", type=pathlib.Path,
                   help="optional local Tempo2 source tree; source param_* references are unioned into embedded model rules")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        return run_self_tests()

    rules = DEFAULT_RULES
    if args.tempo2_src:
        if not args.tempo2_src.exists():
            print(f"ERROR: --tempo2-src path does not exist: {args.tempo2_src}", file=sys.stderr)
            return 2
        rules = source_derived_rules(args.tempo2_src, DEFAULT_RULES)

    if args.list_models:
        for m in sorted(rules):
            print(f"{m}: {', '.join(sorted(rules[m].allowed))}")
        return 0

    if not args.par_files:
        print("ERROR: no par file supplied", file=sys.stderr)
        return 2

    any_error = False
    for f in args.par_files:
        if not f.exists():
            print(f"ERROR: file does not exist: {f}", file=sys.stderr)
            any_error = True
            continue
        diags = validate(
            f,
            rules,
            mode=args.mode,
            require_core=not args.no_require_core,
            strict_known=args.strict_known,
        )
        print_diags(f, diags)
        if any(d.severity == "ERROR" for d in diags):
            any_error = True
    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
