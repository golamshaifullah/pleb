"""Native raw-to-release stages used by the PLEB workflow runner.

Measurements always come from raw inputs. A review catalog contains decisions,
not replacement TOAs, fitted models or a processed checkpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import fields, replace
from fnmatch import fnmatchcase
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .observations import read_observations, source_id, write_observations

STAGES = (
    "ingest",
    "clocks",
    "normalise",
    "initial_models",
    "qc",
    "review",
    "models",
    "timing",
    "verify",
)
SECTIONS = {
    "release_version": None,
    "run": {"mode"},
    "source": {"raw", "inventory_sha256"},
    "ingest": None,
    "processing": None,
    "qc": {"mode", "settings", "apply"},
    "review": {"mode", "catalog", "sha256"},
    "timing": {
        "fit",
        "validate",
        "native",
        "iterations",
        "nobs",
        "ephem",
        "clock",
        "weighted",
        "parameters",
        "search",
    },
    "variants": None,
    "output": {"directory", "repository", "branch", "base", "dataset_path"},
    "verify": {"reference", "sha256", "reject_exact_duplicates"},
    "container": {"image", "sha256"},
    "clocks": {"include_container", "utc_stations", "start_mjd", "finish_mjd"},
}


class ReleaseError(ValueError):
    """An input, decision or stage cannot be reproduced safely."""


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else _stream_digest(stream)
        )


def _stream_digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _json_read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def resolve_path(config_path, value):
    path = Path(value).expanduser()
    return (
        (config_path.parent / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )


def load_release_config(path, overrides=None):
    """Validate the single release config, including misspelled keys."""
    from .config_io import _set_dotted_key, _parse_value_as_toml_literal
    from .dataset_fix import FixDatasetConfig

    path = Path(path).expanduser().resolve()
    cfg = tomllib.loads(path.read_text())
    for item in overrides or []:
        if "=" not in item:
            raise ReleaseError("Overrides require KEY=VALUE")
        key, value = item.split("=", 1)
        _set_dotted_key(cfg, key, _parse_value_as_toml_literal(value))
    if cfg.get("release_version") != 1 or cfg.get("run", {}).get("mode") != "release":
        raise ReleaseError("Expected release_version=1 and [run] mode='release'")
    for section, value in cfg.items():
        if section not in SECTIONS:
            raise ReleaseError(f"Unknown release section: {section}")
        allowed = SECTIONS[section]
        if section != "release_version" and not isinstance(value, dict):
            raise ReleaseError(f"Release section must be a table: {section}")
        if allowed is not None and set(value) - allowed:
            raise ReleaseError(
                f"Unknown {section} settings: {sorted(set(value) - allowed)}"
            )
    processing = cfg.get("processing", {})
    allowed = {f.name for f in fields(FixDatasetConfig)} | {
        "merge_backend_suffixes",
        "site_rules",
        "time_rules",
    }
    if set(processing) - allowed:
        raise ReleaseError(
            f"Unknown processing settings: {sorted(set(processing) - allowed)}"
        )
    for key in (
        "dedupe_action",
        "wsrt_p2_action",
        "qc_action",
        "qc_solar_action",
        "qc_orbital_phase_action",
    ):
        if processing.get(key, "comment") != "comment":
            raise ReleaseError(f"Release processing requires {key}='comment'")
    if (
        processing.get("overlap_exact_catalog_path")
        and "overlap_exact_map" in processing
    ):
        raise ReleaseError(
            "Declare overlap_exact_map or overlap_exact_catalog_path, not both"
        )
    if processing.get("remove_overlaps_exact", True) and not (
        "overlap_exact_map" in processing
        or processing.get("overlap_exact_catalog_path")
    ):
        raise ReleaseError(
            "Declare overlap_exact_map/catalog or disable remove_overlaps_exact"
        )
    if processing.get("remove_patterns"):
        raise ReleaseError(
            "Release processing retains measurements; remove_patterns must be empty"
        )
    if processing.get("apply", True) is not True or processing.get("dry_run", False):
        raise ReleaseError(
            "Release stages apply changes only to their isolated output dataset"
        )
    if (
        processing.get("backup", False)
        or processing.get("generate_alltim_variants")
        or processing.get("jump_reference_variants")
    ):
        raise ReleaseError(
            "Release stages use their run manifest and inline variants, without extra backup/model copies"
        )
    ingest = cfg.get("ingest", {})
    allowed_ingest = {
        "sources",
        "par_roots",
        "template_roots",
        "backends",
        "ignore_backends",
        "exclude_tim_paths",
        "pulsar_aliases",
        "lockfile_path",
        "lockfile_strict",
        "priority_order",
        "priority_roots",
        "pulsars",
    }
    if set(ingest) - allowed_ingest:
        raise ReleaseError(
            f"Unknown ingest settings: {sorted(set(ingest) - allowed_ingest)}"
        )
    for name, backend in ingest.get("backends", {}).items():
        if set(backend) - {"root", "tim_glob", "ignore", "ignore_suffixes"}:
            raise ReleaseError(f"Unknown ingest backend settings: {name}")
        if ".." in Path(backend.get("tim_glob", "*.tim")).parts:
            raise ReleaseError(
                "Backend globs cannot traverse outside their source root"
            )
    if cfg.get("qc", {}).get("mode", "replay") not in {"replay", "detect", "disabled"}:
        raise ReleaseError("QC mode must be replay, detect or disabled")
    qc_apply = cfg.get("qc", {}).get("apply", {})
    allowed_apply = {
        f.name for f in fields(FixDatasetConfig) if f.name.startswith("qc_")
    }
    if set(qc_apply) - allowed_apply:
        raise ReleaseError(
            f"Unknown QC application settings: {sorted(set(qc_apply) - allowed_apply)}"
        )
    for key in ("qc_action", "qc_solar_action", "qc_orbital_phase_action"):
        if qc_apply.get(key, "comment") != "comment":
            raise ReleaseError(f"Release QC requires {key}='comment'")
    review = cfg.get("review", {})
    if review.get("mode", "catalog") not in {"catalog", "interactive", "none"}:
        raise ReleaseError("Review mode must be catalog, interactive or none")
    if review.get("mode", "catalog") == "catalog" and not review.get("catalog"):
        raise ReleaseError("Catalog review requires a decision catalog")
    if int(cfg.get("timing", {}).get("iterations", 1)) < 1:
        raise ReleaseError("Timing iterations must be positive")
    timing = cfg.get("timing", {})
    if (
        timing.get("fit")
        or timing.get("validate")
        or cfg.get("qc", {}).get("mode") == "detect"
        or cfg.get("clocks", {}).get("include_container")
    ) and not timing.get("native", False):
        if not cfg.get("container", {}).get("image") or not cfg.get(
            "container", {}
        ).get("sha256"):
            raise ReleaseError("Container stages require an image and pinned SHA256")
    if timing.get("search") and (
        not timing.get("fit") or not timing.get("weighted", True)
    ):
        raise ReleaseError("Timing model search requires an explicit weighted fit")
    for parameter in timing.get("parameters", []):
        if (
            set(parameter) - {"pulsar", "variant", "name", "value"}
            or not parameter.get("name")
            or any(
                c in str(parameter.get("name", "")) + str(parameter.get("value", ""))
                for c in "\r\n"
            )
        ):
            raise ReleaseError(
                "Timing parameters require a name/value and optional pulsar/variant selectors"
            )
    for search in timing.get("search", []):
        if set(search) - {
            "pulsar",
            "variant",
            "candidates",
            "min_delta_bic",
            "common_noise_scale",
        } or not search.get("candidates"):
            raise ReleaseError(
                "Timing search requires candidate specifications and valid comparison settings"
            )
    for psr in cfg.get("ingest", {}).get("pulsars", []):
        for variant in cfg.get("variants", {"combined": ["*"]}):
            matches = [
                s
                for s in timing.get("search", [])
                if fnmatchcase(psr, s.get("pulsar", "*"))
                and s.get("variant", "combined") == variant
            ]
            if len(matches) > 1:
                raise ReleaseError(
                    "Combine timing candidates into one search rule per pulsar/variant"
                )
    if "combined" not in cfg.get("variants", {"combined": ["*"]}):
        raise ReleaseError("A combined variant is required")
    return path, cfg


def locked_json(config_path, section):
    name = section.get("catalog", section.get("reference"))
    if not name:
        return None
    path = resolve_path(config_path, name)
    if section.get("sha256") and file_digest(path) != section["sha256"]:
        raise ReleaseError(f"Input checksum differs: {path.name}")
    return _json_read(path)


def _ingest_mapping(config_path, cfg):
    from .ingest import _load_mapping

    # Materialised mappings are run outputs. Users edit only release.toml.
    raw = resolve_path(config_path, cfg["source"]["raw"])
    mapping = dict(cfg["ingest"])
    for key in ("sources", "par_roots", "template_roots"):
        mapping[key] = [str((raw / p).resolve()) for p in mapping.get(key, [])]
    mapping["priority_roots"] = {
        k: str((raw / p).resolve())
        for k, p in mapping.get("priority_roots", {}).items()
    }
    mapping["backends"] = {
        k: dict(v, root=str((raw / v["root"]).resolve()))
        for k, v in mapping["backends"].items()
    }
    if mapping.get("lockfile_path"):
        raise ReleaseError(
            "Release ingestion uses pinned source identities, not an external path lockfile"
        )
    # Validate through the existing ingestion loader before creating output.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pleb-ingest-plan-") as directory:
        path = Path(directory) / "ingest.json"
        json_write(path, mapping)
        loaded = _load_mapping(path)
    roots = (
        [b.root for b in loaded.backends]
        + list(loaded.sources)
        + list(loaded.par_roots)
        + list(loaded.template_roots)
    )
    if any(not p.is_relative_to(raw) for p in roots):
        raise ReleaseError("Ingestion sources must be inside source.raw")
    return raw, mapping, loaded


def input_inventory(config_path, cfg):
    """Resolve deterministic source choices before any processing or review."""
    from .ingest import (
        _find_timfiles,
        _find_parfiles,
        _find_template_files,
        _resolve_backend_keys,
    )

    raw, mapping, loaded = _ingest_mapping(config_path, cfg)
    chosen = _find_timfiles(
        loaded.backends,
        loaded.pulsar_aliases,
        loaded.ignore_backends,
        loaded.exclude_tim_paths,
        loaded.priority_order,
        loaded.priority_roots,
    )
    pars = _find_parfiles(loaded.par_roots, loaded.pulsar_aliases)
    pulsars = sorted(loaded.pulsars or chosen)
    records, sources, seen = [], {}, set()
    suffixes = cfg.get("processing", {}).get("merge_backend_suffixes", [])
    for psr in pulsars:
        if psr not in chosen or psr not in pars:
            raise ReleaseError(f"Missing raw PAR/TIM inputs for {psr}")
        seed = pars[psr].resolve()
        sources[seed.relative_to(raw).as_posix()] = file_digest(seed)
        for backend, path, _ in _resolve_backend_keys(psr, chosen[psr]):
            canonical = backend
            for suffix in suffixes:
                if canonical.endswith(suffix):
                    canonical = canonical[: -len(suffix)]
            relative = path.resolve().relative_to(raw).as_posix()
            checksum = sources.setdefault(relative, file_digest(path))
            for row in read_observations(path):
                identity = source_id(relative, checksum, row.line)
                key = (psr, canonical, identity)
                if key in seen:
                    continue  # One input referenced by two mapping entries.
                seen.add(key)
                records.append(
                    dict(
                        source_id=identity,
                        source=relative,
                        source_line=row.line,
                        source_sha256=checksum,
                        pulsar=psr,
                        backend=canonical,
                        ingest_file=f"{psr}/tims/{backend}.tim",
                    )
                )
    if len({r["source_id"] for r in records}) != len(records):
        raise ReleaseError(
            "The same source observation maps to multiple output backends"
        )
    for path in raw.rglob("*.clk"):
        sources[path.relative_to(raw).as_posix()] = file_digest(path)
    templates = _find_template_files(loaded.template_roots, loaded.pulsar_aliases)
    for psr in pulsars:
        for path in templates.get(psr, []):
            sources[path.resolve().relative_to(raw).as_posix()] = file_digest(path)
    inventory = dict(
        format_version=1,
        pulsars=pulsars,
        files=sources,
        observations=len(records),
        identities_sha256=digest(sorted(r["source_id"] for r in records)),
    )
    if (
        cfg["source"].get("inventory_sha256")
        and digest(inventory) != cfg["source"]["inventory_sha256"]
    ):
        raise ReleaseError("Raw input inventory differs from the pinned release")
    return raw, mapping, records, inventory


def _bindings_write(root, records):
    with gzip.open(root / "runtime/observations.jsonl.gz", "wt") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _bindings_read(root):
    with gzip.open(root / "runtime/observations.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream]


def bound_observations(dataset, bindings):
    grouped = defaultdict(list)
    for record in bindings:
        grouped[record["file"]].append(record)
    for name, records in sorted(grouped.items()):
        rows = read_observations(dataset / name)
        if len(rows) != len(records):
            raise ReleaseError(f"A stage added/deleted measurements: {name}")
        for record, row in zip(records, rows):
            if record.get("measurement_sha256") and record[
                "measurement_sha256"
            ] != observation_digest(row):
                raise ReleaseError(
                    f"A stage changed or reordered source measurements: {name}:{row.line}"
                )
            yield record, row


def observation_digest(row):
    """Physics invariants, excluding selection state and descriptive flags."""
    return digest(
        [
            Path(row.filename).name,
            str(row.frequency.normalize()),
            str(row.mjd.normalize()),
            str(row.uncertainty.normalize()),
            row.site,
            str(row.time.normalize()),
            str(row.efac.normalize()),
            str(row.equad.normalize()),
            {
                k: row.flags.get(k)
                for k in ("-addsat", "-padd", "-to", "-dmo", "-dm", "-efac", "-equad")
            },
        ]
    )


def _stage_ingest(root, cfg, mapping, records):
    from .ingest import ingest_dataset

    dataset = root / "dataset"
    path = root / "runtime/ingest.json"
    json_write(path, mapping)
    report = ingest_dataset(path, dataset, verify=True)
    if report["missing_parfiles"] or report["missing_timfiles"]:
        raise ReleaseError("Ingestion has missing input files")
    if (dataset / "ingest_reports").exists():
        shutil.move(str(dataset / "ingest_reports"), root / "reports/ingest")
    grouped, parsed = defaultdict(list), {}
    for record in records:
        name = record["ingest_file"]
        if name not in parsed:
            parsed[name] = {row.line: row for row in read_observations(dataset / name)}
        try:
            row = parsed[name][record["source_line"]]
        except KeyError as error:
            raise ReleaseError(
                f"Ingestion changed source line locations: {name}"
            ) from error
        for rule in cfg.get("processing", {}).get("site_rules", []):
            if (
                fnmatchcase(record["pulsar"], rule.get("pulsar", "*"))
                and fnmatchcase(record["backend"], rule.get("backend", "*"))
                and row.site == rule["source"]
            ):
                parts = row.body.split()
                parts[4] = rule["target"]
                row = replace(row, site=rule["target"], body=" ".join(parts))
        for rule in cfg.get("processing", {}).get("time_rules", []):
            from decimal import Decimal

            if (
                fnmatchcase(record["pulsar"], rule.get("pulsar", "*"))
                and fnmatchcase(record["backend"], rule.get("backend", "*"))
                and Decimal(str(rule.get("start_mjd", 0)))
                <= row.mjd
                <= Decimal(str(rule.get("finish_mjd", 100000)))
            ):
                row = replace(row, time=row.time + Decimal(str(rule["seconds"])))
        target = f"{record['pulsar']}/tims/{record['backend']}.tim"
        grouped[target].append((record, row))
    bindings = []
    for name, entries in sorted(grouped.items()):
        locations = write_observations(
            dataset / name,
            [
                (r["source_id"], row, row.state == "active", row.state)
                for r, row in entries
            ],
            weighted=cfg.get("timing", {}).get("weighted", True),
        )
        bindings.extend(
            dict(
                r,
                file=name,
                line=location["line"],
                measurement_sha256=observation_digest(row),
            )
            for (r, row), location in zip(entries, locations)
        )
    # Remove redundant ingest views, not physical source measurements.
    for name in parsed.keys() - grouped.keys():
        (dataset / name).unlink()
    _bindings_write(root, bindings)
    json_write(root / "reports/ingest.json", report)


def _processing_config(config_path, cfg, **overrides):
    from .dataset_fix import FixDatasetConfig

    values = dict(
        apply=True,
        backup=False,
        dry_run=False,
        remove_patterns=[],
        min_toas_per_backend_tim=0,
        dedupe_action="comment",
        qc_action="comment",
    )
    values.update(
        {
            k: v
            for k, v in cfg.get("processing", {}).items()
            if k not in {"merge_backend_suffixes", "site_rules", "time_rules"}
        }
    )
    # Native releases never read undeclared tables from a source checkout.
    values.setdefault("backend_bandwidths", {})
    if not values.get("overlap_exact_catalog_path"):
        values.setdefault("overlap_exact_map", {})
    for key, value in list(values.items()):
        if key.endswith("_path") and value:
            values[key] = str(resolve_path(config_path, value))
    values.update(overrides)
    return FixDatasetConfig(**values)


def _errors(value):
    if isinstance(value, dict):
        if value.get("error"):
            yield str(value["error"])
        for child in value.values():
            yield from _errors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _errors(child)


def _stage_normalise(root, config_path, cfg, pulsars):
    from .dataset_fix import fix_pulsar_dataset

    reports = []
    for number, psr in enumerate(pulsars, 1):
        reports.append(
            fix_pulsar_dataset(
                root / "dataset" / psr, _processing_config(config_path, cfg)
            )
        )
        if number % 10 == 0 or number == len(pulsars):
            print(f"normalise: {number}/{len(pulsars)} pulsars", flush=True)
    errors = list(_errors(reports))
    json_write(root / "reports/processing.json", reports)
    if errors:
        raise ReleaseError("Dataset processing failed: " + "; ".join(errors[:5]))
    list(bound_observations(root / "dataset", _bindings_read(root)))


def _stage_clocks(root, config_path, cfg):
    """Overlay supplied corrections onto pinned Tempo2 clocks, then explicit UTC."""
    import os
    from .tempo2 import container_runtime

    clockdir = root / "dataset/clockfiles"
    clocks = cfg.get("clocks", {})
    if clocks.get("include_container"):
        runtime = root / "runtime/clock_base"
        if cfg.get("timing", {}).get("native"):
            stock = Path(os.environ["TEMPO2"]) / "clock"
            shutil.copytree(stock, runtime)
        else:
            image = resolve_path(config_path, cfg["container"]["image"])
            program = "import os,shutil; from pathlib import Path; shutil.copytree(Path(os.environ.get('TEMPO2','/opt/tempo2/share/tempo2'))/'clock', '/pleb-runtime/clock_base')"
            command = [
                container_runtime(),
                "exec",
                "--bind",
                f"{root / 'runtime'}:/pleb-runtime",
                str(image),
                "python",
                "-c",
                program,
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            (root / "runtime/clock_setup.log").write_text(
                completed.stdout + "\n" + completed.stderr
            )
            if completed.returncode:
                raise ReleaseError(
                    "Container clock setup failed: " + completed.stderr.strip()
                )
    for station in clocks.get("utc_stations", []):
        if not station.isalnum():
            raise ReleaseError("UTC station names must be alphanumeric")
        clockdir.mkdir(exist_ok=True)
        (clockdir / f"{station}2utc_explicit.clk").write_text(
            f"# UTC({station}) UTC\n"
            "# Input timestamps are already UTC: explicit zero station correction.\n"
            f"{clocks.get('start_mjd', 40000)} 0.0\n{clocks.get('finish_mjd', 70000)} 0.0\n"
        )
    effective = root / "runtime/clock_base"
    if effective.exists():
        for path in clockdir.glob("*.clk"):
            shutil.copyfile(path, effective / path.name)
    else:
        effective = clockdir
    json_write(
        root / "reports/clocks.json",
        dict(
            effective={p.name: file_digest(p) for p in sorted(effective.glob("*.clk"))},
            supplied={p.name: file_digest(p) for p in sorted(clockdir.glob("*.clk"))},
            container_sha256=cfg.get("container", {}).get("sha256"),
        ),
    )


def _stage_qc(root, config_path, cfg, pulsars):
    if cfg.get("qc", {}).get("mode", "replay") != "detect":
        return
    import os
    import pandas as pd
    from .outlier_qc import PTAQCConfig
    from .dataset_fix import fix_pulsar_dataset
    from .tempo2 import container_runtime

    settings = dict(cfg["qc"].get("settings", {}))
    allowed = {f.name for f in fields(PTAQCConfig)}
    if set(settings) - allowed:
        raise ReleaseError(f"Unknown QC settings: {sorted(set(settings) - allowed)}")
    for key, value in list(settings.items()):
        if key.endswith("_path") and value:
            settings[key] = str(resolve_path(config_path, value))
    reports = []
    for psr in pulsars:
        par = root / "dataset" / psr / f"{psr}_combined.par"
        path = root / "qc" / f"{psr}_qc.csv"
        payload = root / "runtime" / f"qc_{psr}.json"
        json_write(
            payload,
            dict(root=str(root), par=str(par), output=str(path), settings=settings),
        )
        worker = Path(__file__).with_name("release_qc_worker.py")
        command = [sys.executable, str(worker), str(payload)]
        if not cfg.get("timing", {}).get("native"):
            image = resolve_path(config_path, cfg["container"]["image"])
            command = [
                container_runtime(),
                "exec",
                "--bind",
                f"{root}:{root}",
                "--bind",
                f"{worker}:{worker}:ro",
            ]
            for key, value in settings.items():
                if key.endswith("_path") and value:
                    folder = Path(value).parent
                    command += ["--bind", f"{folder}:{folder}:ro"]
            command += [str(image), "python", str(worker), str(payload)]
        environment = dict(os.environ)
        # The image's global /tmp cache/log directory may belong to another
        # user. All Python worker writes belong to this isolated, bound run.
        for name, directory in (
            ("PLEB_LOG_DIR", "logs"),
            ("MPLCONFIGDIR", "matplotlib"),
        ):
            target = root / "runtime" / directory
            target.mkdir(exist_ok=True)
            environment[name] = str(target)
            environment["APPTAINERENV_" + name] = str(target)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = "1"
            environment["APPTAINERENV_" + name] = "1"
        completed = subprocess.run(
            command, capture_output=True, text=True, env=environment
        )
        path.parent.mkdir(exist_ok=True)
        path.with_suffix(".worker.log").write_text(
            completed.stdout + "\n" + completed.stderr
        )
        if completed.returncode or not path.exists():
            raise ReleaseError(
                f"PQC failed for {psr}; see {path.with_suffix('.worker.log')}"
            )
        frame = pd.read_csv(path)
        frame = attach_qc_identities(
            frame,
            root / "dataset",
            _bindings_read(root),
            psr,
            settings.get("merge_tol_seconds", 2.0),
        )
        frame.to_csv(path, index=False)
        apply = dict(cfg["qc"].get("apply", {}))
        apply.update(
            qc_apply_only=True,
            qc_results_dir=root / "qc",
            qc_require_csv=True,
            qc_remove_outliers=True,
            qc_action="comment",
        )
        reports.append(
            fix_pulsar_dataset(
                par.parent, _processing_config(config_path, cfg, **apply)
            )
        )
    json_write(root / "reports/qc.json", reports)
    if list(_errors(reports)):
        raise ReleaseError("QC application failed; see reports/qc.json")


def attach_qc_identities(frame, dataset, bindings, pulsar, tolerance_seconds=2):
    """Attach source IDs only through a unique, one-to-one metadata match."""
    from decimal import Decimal

    candidates = defaultdict(list)
    for record, row in bound_observations(dataset, bindings):
        if record["pulsar"] == pulsar and row.state == "active":
            candidates[Path(record["file"]).name].append((record, row))
    used, identities = set(), []
    for entry in frame.to_dict("records"):
        name = next(
            (
                str(entry[k])
                for k in ("_timfile", "timfile", "_timfile_base")
                if k in entry and str(entry[k]) not in {"", "nan"}
            ),
            "",
        )
        epoch = Decimal(str(entry["mjd"]))
        frequency = Decimal(str(entry["freq"]))
        found = [
            (r, row)
            for r, row in candidates[Path(name).name]
            if r["source_id"] not in used
            and abs(row.epoch - epoch) * 86400 <= Decimal(str(tolerance_seconds))
            and abs(row.frequency - frequency) <= Decimal("0.000001")
        ]
        if len(found) != 1:
            raise ReleaseError(
                f"QC row has {len(found)} source matches: {pulsar}/{name}/{epoch}"
            )
        identity = found[0][0]["source_id"]
        used.add(identity)
        identities.append(identity)
    result = frame.copy()
    result["source_id"] = identities
    return result


def prepare_release_review(root, bindings):
    """Prepare native review inputs without exporting Feathers or refitting."""
    import pandas as pd
    from .qc_review import empty_overrides, write_overrides

    review = root / "review"
    review.mkdir(exist_ok=True)
    frames = defaultdict(list)
    for record, row in bound_observations(root / "dataset", bindings):
        frames[record["pulsar"]].append(
            dict(
                source_id=record["source_id"],
                pulsar=record["pulsar"],
                variant="combined",
                timfile=record["file"],
                mjd=str(row.epoch),
                freq=str(row.frequency),
                toa_unc_us=str(row.uncertainty),
                backend=row.flags.get("-sys", record["backend"]),
                bad_point=row.state != "active",
                filename=row.filename,
            )
        )
    for psr, rows in frames.items():
        data = pd.DataFrame(rows)
        qc_path = root / "qc" / f"{psr}_qc.csv"
        if qc_path.exists():
            detected = pd.read_csv(qc_path)
            if "source_id" not in detected:
                raise ReleaseError("QC output is missing portable source IDs")
            extra = [c for c in detected if c not in data or c == "source_id"]
            data = data.merge(
                detected[extra], on="source_id", how="left", validate="one_to_one"
            )
        data.to_csv(review / f"{psr}_qc.csv", index=False)
    overrides = review / "manual_qc_overrides.csv"
    if not overrides.exists():
        write_overrides(empty_overrides(), overrides)


def _interactive_catalog(root, bindings):
    from .qc_review import load_overrides, _latest_effective_overrides

    frame = _latest_effective_overrides(
        load_overrides(root / "review/manual_qc_overrides.csv")
    )
    choices = []
    for row in frame.to_dict("records"):
        if row["manual_action"] == "clear_manual":
            continue
        choices.append(
            dict(
                source_id=row.get("source_id") or row["review_id"],
                action="comment" if row["manual_action"] == "mark_bad" else "keep",
                reason=row.get("manual_reason", "expert review"),
                origin="manual_review",
                reviewer=row.get("reviewer", ""),
            )
        )
    return dict(
        format_version=1,
        identities_sha256=digest(sorted(r["source_id"] for r in bindings)),
        decisions=choices,
    )


def apply_review_catalog(dataset, bindings, catalog, *, weighted=True):
    """Apply explicit source decisions, rejecting missing/duplicate identities."""
    if catalog.get("format_version") != 1:
        raise ReleaseError("Unsupported review catalog version")
    decisions = {}
    for entry in catalog.get("decisions", []):
        identity = entry["source_id"]
        if identity in decisions or entry["action"] not in {"keep", "comment"}:
            raise ReleaseError("Repeated source identity or unsupported review action")
        decisions[identity] = entry
    actual = {r["source_id"] for r in bindings}
    if set(decisions) - actual:
        raise ReleaseError(
            f"Unmatched review decisions: {len(set(decisions) - actual)}"
        )
    if (
        catalog.get("identities_sha256")
        and digest(sorted(actual)) != catalog["identities_sha256"]
    ):
        raise ReleaseError("Review catalog belongs to different source observations")
    grouped = defaultdict(list)
    reactivated_files = set()
    for record, row in bound_observations(dataset, bindings):
        decision = decisions.get(record["source_id"])
        keep = decision["action"] == "keep" if decision else row.state == "active"
        if keep and row.state != "active":
            reactivated_files.add(record["file"])
        reason = decision.get("reason", "review") if decision else row.state
        grouped[record["file"]].append((record["source_id"], row, keep, reason))
    locations = {}
    for name, entries in grouped.items():
        locations[name] = write_observations(dataset / name, entries, weighted=weighted)
    return dict(
        decisions=len(decisions),
        actions=dict(Counter(d["action"] for d in decisions.values())),
        reactivated_files=sorted(reactivated_files),
    )


def selection_summary(dataset, bindings):
    by_pulsar = defaultdict(list)
    physical = Counter()
    duplicates = []
    measurements = []
    seen = {}
    for record, row in bound_observations(dataset, bindings):
        physical[row.state] += 1
        if row.state != "active":
            continue
        by_pulsar[record["pulsar"]].append(record["source_id"])
        measurements.append([record["source_id"], observation_digest(row)])
        key = (record["pulsar"], *row.measurement[:2], row.epoch, row.site)
        if key in seen:
            duplicates.append([seen[key], record["source_id"]])
        seen[key] = record["source_id"]
    return dict(
        format_version=1,
        active=sum(map(len, by_pulsar.values())),
        pulsars={
            p: dict(active=len(ids), selection_sha256=digest(sorted(ids)))
            for p, ids in sorted(by_pulsar.items())
        },
        selection_sha256=digest(sorted(i for ids in by_pulsar.values() for i in ids)),
        measurements_sha256=digest(sorted(measurements)),
        physical_states=dict(physical),
        exact_duplicate_active=duplicates,
    )


def timing_policy_digest(cfg):
    timing = cfg.get("timing", {})
    return digest(
        dict(
            clocks=cfg.get("clocks", {}),
            container_sha256=cfg.get("container", {}).get("sha256"),
            timing={k: timing.get(k) for k in ("clock", "ephem", "weighted")},
        )
    )


def _stage_models(root, config_path, cfg, pulsars, *, initial=False):
    from .dataset_fix import infer_and_apply_system_flags, update_parfile_jumps

    dataset = root / "dataset"
    settings = cfg.get("timing", {})
    variants = cfg.get("variants", {"combined": ["*"]})
    review_path = root / "reports/review.json"
    refresh = (
        set(_json_read(review_path).get("reactivated_files", []))
        if review_path.exists()
        else set()
    )
    for psr in pulsars:
        parent = dataset / psr
        tims = sorted((parent / "tims").glob("*.tim"))
        # Rows explicitly kept from provider comments need labels too.
        if not initial and cfg.get("processing", {}).get("infer_system_flags"):
            for path in tims:
                if path.relative_to(dataset).as_posix() in refresh:
                    infer_and_apply_system_flags(
                        path, _processing_config(config_path, cfg)
                    )
        active = {
            path: [r for r in read_observations(path) if r.state == "active"]
            for path in tims
        }
        for variant, patterns in variants.items():
            if not variant.replace("_", "").isalnum() or not isinstance(patterns, list):
                raise ReleaseError(
                    "Variants require simple names and backend glob lists"
                )
            selected = [
                p
                for p in tims
                if active[p]
                and any(fnmatchcase(p.stem, pattern) for pattern in patterns)
            ]
            wrapper = parent / f"{psr}_{variant}_all.tim"
            wrapper.write_text(
                "FORMAT 1\n"
                + "\n".join("INCLUDE tims/" + p.name for p in selected)
                + "\n"
            )
            par = parent / f"{psr}_{variant}.par"
            shutil.copyfile(parent / f"{psr}.par", par)
            systems = sorted(
                {
                    r.flags["-sys"]
                    for p in selected
                    for r in active[p]
                    if r.flags.get("-sys")
                }
            )
            update_parfile_jumps(
                par,
                "-sys",
                systems,
                prune_stale_jumps=True,
                ensure_ephem=settings.get("ephem"),
                ensure_clk=settings.get("clock"),
                apply=True,
                backup=False,
            )
            lines = [
                line
                for line in par.read_text().splitlines()
                if not line.split() or line.split()[0] not in {"MODE", "NITS"}
            ]
            lines += [
                f"MODE {1 if settings.get('weighted', True) else 0}",
                f"NITS {int(settings.get('iterations', 1))}",
            ]
            for entry in settings.get("parameters", []):
                if (
                    fnmatchcase(psr, entry.get("pulsar", "*"))
                    and entry.get("variant", "combined") == variant
                ):
                    name, value = entry["name"], entry["value"]
                    lines = [line for line in lines if not line.startswith(name + " ")]
                    lines.append(f"{name} {value}")
            par.write_text("\n".join(lines) + "\n")
    if not initial:
        for name in ("system_flag_table.json", "system_flag_table.toml"):
            path = dataset / name
            if path.exists():
                shutil.move(path, root / "reports" / name)


def _stage_timing(root, config_path, cfg, pulsars):
    settings = cfg.get("timing", {})
    if not (settings.get("fit") or settings.get("validate")):
        return
    from .tempo2 import build_singularity_prefix

    dataset = root / "dataset"
    native = settings.get("native", False)
    image = (
        resolve_path(config_path, cfg["container"]["image"])
        if not native
        else Path(".")
    )
    reports = []
    for psr in pulsars:
        for variant in cfg.get("variants", {"combined": ["*"]}):
            stem = f"{psr}/{psr}_{variant}"
            par, tim = dataset / (stem + ".par"), dataset / (stem + "_all.tim")
            if not any(
                line.startswith("INCLUDE ") for line in tim.read_text().splitlines()
            ):
                continue
            folder = root / "timing" / psr / variant

            def execute(source, destination):
                import os

                destination.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination / "input.par")
                clockdir = root / "runtime/clock_base"
                if not clockdir.exists():
                    clockdir = dataset / "clockfiles"
                binds = [(destination, "/fit")]
                if clockdir.exists():
                    binds.append((clockdir, "/release-clocks"))
                prefix = build_singularity_prefix(
                    root, Path("dataset"), image, native=native, extra_binds=binds
                )
                if not native:
                    prefix = prefix[:-1] + ["--pwd", "/fit", prefix[-1]]
                args = [
                    "tempo2",
                    "-f",
                    str(destination / "input.par") if native else "/fit/input.par",
                    str(tim) if native else "/data/" + stem + "_all.tim",
                    "-nobs",
                    str(settings.get("nobs", 120000)),
                    "-showchisq",
                ]
                if clockdir.exists():
                    args += [
                        "-clkdir",
                        str(clockdir) + "/" if native else "/release-clocks/",
                    ]
                args += ["-newpar"] if settings.get("fit") else ["-nofit"]
                environment = dict(os.environ)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                ):
                    environment[name] = "1"
                    environment["APPTAINERENV_" + name] = "1"
                command = prefix + args
                completed = subprocess.run(
                    command,
                    cwd=destination,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                text = completed.stdout + "\n" + completed.stderr
                (destination / "tempo2.log").write_text(text)
                json_write(destination / "command.json", command)
                if completed.returncode or "ERROR:" in text:
                    raise ReleaseError(
                        f"Tempo2 failed for {psr}/{variant}; see timing log"
                    )
                fitted = destination / "new.par"
                if settings.get("fit") and not fitted.is_file():
                    raise ReleaseError("Tempo2 did not write its fitted PAR")
                return fitted if settings.get("fit") else source, text

            fitted, text = execute(par, folder)
            if settings.get("fit"):
                from .timing_selection import fit_statistics, select_timing_model

                for rule in settings.get("search", []):
                    if not fnmatchcase(
                        psr, rule.get("pulsar", "*")
                    ) or variant != rule.get("variant", "combined"):
                        continue
                    fitted, selection = select_timing_model(
                        fitted,
                        fit_statistics(text),
                        rule["candidates"],
                        folder / "search",
                        execute,
                        min_delta_bic=float(rule.get("min_delta_bic", 10)),
                        common_noise_scale=rule.get("common_noise_scale", False),
                    )
                    json_write(folder / "model_selection.json", selection)
                shutil.copyfile(fitted, par)
            reports.append(
                dict(
                    pulsar=psr,
                    variant=variant,
                    fit=bool(settings.get("fit")),
                    iterations=settings.get("iterations", 1),
                )
            )
    json_write(root / "reports/timing.json", reports)


def _dataset_manifest(dataset):
    return {
        str(p.relative_to(dataset)): file_digest(p)
        for p in sorted(dataset.rglob("*"))
        if p.is_file()
    }


def publish_release(dataset, config_path, settings):
    """Commit only the finished dataset through a private Git index.

    The checkout and unrelated index entries are never touched. Existing output
    branches must already contain the exact requested result.
    """
    import os
    import tempfile

    repository = resolve_path(config_path, settings["repository"])
    branch = settings["branch"]
    prefix = Path(settings["dataset_path"])
    if prefix.is_absolute() or ".." in prefix.parts or str(prefix) in {".", ""}:
        raise ReleaseError(
            "Output dataset_path must be a nonempty repository-relative directory"
        )

    def git(*args, data=None, env=None):
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            input=data,
            env=env,
            capture_output=True,
        )
        if result.returncode:
            raise ReleaseError(result.stderr.decode().strip())
        return result.stdout

    git("check-ref-format", "--branch", branch)
    if (
        f"branch refs/heads/{branch}"
        in git("worktree", "list", "--porcelain").decode().splitlines()
    ):
        raise ReleaseError("Output branch is checked out; choose a new release branch")
    parent = git("rev-parse", settings.get("base", "HEAD")).decode().strip()
    with tempfile.TemporaryDirectory(prefix="pleb-release-index-") as temporary:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        git("read-tree", parent, env=environment)
        names = git("ls-files", "-z", "--", prefix.as_posix(), env=environment).split(
            b"\0"
        )
        if any(names):
            git(
                "update-index",
                "-z",
                "--index-info",
                data=b"".join(
                    b"0 " + b"0" * len(parent) + b"\t" + p + b"\0" for p in names if p
                ),
                env=environment,
            )
        entries = []
        for path in sorted(dataset.rglob("*")):
            if not path.is_file():
                continue
            name = (prefix / path.relative_to(dataset)).as_posix()
            oid = git("hash-object", "-w", "--stdin", data=path.read_bytes()).strip()
            entries.append(b"100644 " + oid + b"\t" + name.encode() + b"\0")
        git(
            "update-index",
            "-z",
            "--index-info",
            data=b"".join(entries),
            env=environment,
        )
        tree = git("write-tree", env=environment).decode().strip()
        existing = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "rev-parse",
                "--verify",
                f"refs/heads/{branch}",
            ],
            capture_output=True,
            text=True,
        )
        if existing.returncode == 0:
            commit = existing.stdout.strip()
            if git("rev-parse", commit + "^{tree}").decode().strip() != tree:
                raise ReleaseError("Output branch already contains a different release")
        else:
            commit = (
                git(
                    "commit-tree",
                    tree,
                    "-p",
                    parent,
                    data=b"Build reviewed release from raw inputs with PLEB\n",
                )
                .decode()
                .strip()
            )
            git("update-ref", f"refs/heads/{branch}", commit, "0" * len(parent))
    return dict(branch=branch, commit=commit, tree=tree)


def run_release(
    config_path, *, overrides=None, resume=False, plan=False, stop_after=None
):
    """Run native stages. Resume only verified state with unchanged inputs/code."""
    config_path, cfg = load_release_config(config_path, overrides)
    raw, mapping, records, inventory = input_inventory(config_path, cfg)
    catalog = (
        locked_json(config_path, cfg.get("review", {}))
        if cfg.get("review", {}).get("mode", "catalog") == "catalog"
        else None
    )
    reference = locked_json(config_path, cfg.get("verify", {}))
    root = resolve_path(config_path, cfg["output"]["directory"])
    if root == raw or root.is_relative_to(raw) or raw.is_relative_to(root):
        raise ReleaseError("Release output and raw input must be separate directories")
    package = Path(__file__).parent
    declared = {
        value: file_digest(resolve_path(config_path, value))
        for key, value in cfg.get("processing", {}).items()
        if key.endswith("_path") and value
    }
    signature = dict(
        config=digest(cfg),
        inventory=digest(inventory),
        review=digest(catalog),
        reference=digest(reference),
        declared_inputs=declared,
        code={
            p.relative_to(package).as_posix(): file_digest(p)
            for p in sorted(package.rglob("*.py"))
        },
    )
    if cfg.get("container", {}).get("sha256"):
        if (
            file_digest(resolve_path(config_path, cfg["container"]["image"]))
            != cfg["container"]["sha256"]
        ):
            raise ReleaseError("Container checksum differs")
    if catalog and catalog.get("identities_sha256") != inventory["identities_sha256"]:
        raise ReleaseError("Decision catalog does not match the raw input inventory")
    if plan:
        return dict(
            output=str(root),
            stages=list(STAGES),
            input=inventory,
            qc_mode=cfg.get("qc", {}).get("mode", "replay"),
            timing=cfg.get("timing", {}),
        )
    state_file = root / "runtime/state.json"
    if root.exists():
        if not resume or not state_file.exists():
            raise ReleaseError(
                "Output exists; use --resume with a verified run or choose a new directory"
            )
        state = _json_read(state_file)
        if state.get("status") == "failed":
            raise ReleaseError("A stage failed; choose a fresh output directory")
        if state["signature"] != signature:
            raise ReleaseError(
                "Config, code or input changed; choose a new output directory"
            )
        if state.get("dataset_manifest") != _dataset_manifest(root / "dataset"):
            raise ReleaseError("Run dataset changed outside PLEB; refusing resume")
    else:
        (root / "runtime").mkdir(parents=True)
        (root / "reports").mkdir()
        state = dict(signature=signature, completed=[], status="running")
        json_write(root / "runtime/resolved_config.json", cfg)
        json_write(root / "runtime/input_inventory.json", inventory)
        json_write(
            root / "runtime/software.json",
            {"python": sys.version, "pleb": importlib.metadata.version("pleb")},
        )
        json_write(state_file, state)
    for stage in STAGES:
        if stage in state["completed"]:
            continue
        try:
            if stage == "ingest":
                _stage_ingest(root, cfg, mapping, records)
            elif stage == "clocks":
                _stage_clocks(root, config_path, cfg)
            elif stage == "normalise":
                _stage_normalise(root, config_path, cfg, inventory["pulsars"])
            elif stage == "qc":
                _stage_qc(root, config_path, cfg, inventory["pulsars"])
            elif stage == "review":
                if catalog:
                    report = apply_review_catalog(
                        root / "dataset",
                        _bindings_read(root),
                        catalog,
                        weighted=cfg.get("timing", {}).get("weighted", True),
                    )
                    json_write(root / "reports/review.json", report)
                elif cfg.get("review", {}).get("mode") == "interactive":
                    bindings = _bindings_read(root)
                    if not state.get("awaiting_review"):
                        prepare_release_review(root, bindings)
                        state.update(
                            status="awaiting_review",
                            awaiting_review=True,
                            dataset_manifest=_dataset_manifest(root / "dataset"),
                        )
                        json_write(state_file, state)
                        return dict(
                            output=str(root),
                            status="awaiting_review",
                            command=f"pleb review --run-dir {root / 'review'}",
                            resume=f"pleb run --config {config_path} --resume",
                        )
                    catalog_now = _interactive_catalog(root, bindings)
                    report = apply_review_catalog(
                        root / "dataset",
                        bindings,
                        catalog_now,
                        weighted=cfg.get("timing", {}).get("weighted", True),
                    )
                    json_write(root / "reports/review_decisions.json", catalog_now)
                    json_write(root / "reports/review.json", report)
            elif stage in {"models", "initial_models"}:
                _stage_models(
                    root,
                    config_path,
                    cfg,
                    inventory["pulsars"],
                    initial=stage == "initial_models",
                )
            elif stage == "timing":
                _stage_timing(root, config_path, cfg, inventory["pulsars"])
            elif stage == "verify":
                summary = selection_summary(root / "dataset", _bindings_read(root))
                summary["timing_policy_sha256"] = timing_policy_digest(cfg)
                if (
                    cfg.get("verify", {}).get("reject_exact_duplicates", True)
                    and summary["exact_duplicate_active"]
                ):
                    raise ReleaseError("Release contains duplicate active measurements")
                keys = ["active", "selection_sha256", "pulsars"]
                keys += [
                    k
                    for k in ("measurements_sha256", "timing_policy_sha256")
                    if reference and k in reference
                ]
                if reference and any(summary[k] != reference[k] for k in keys):
                    raise ReleaseError(
                        "Release selection differs from the approved reference"
                    )
                json_write(root / "reports/selection.json", summary)
                json_write(
                    root / "reports/dataset_manifest.json",
                    _dataset_manifest(root / "dataset"),
                )
                if cfg.get("output", {}).get("branch"):
                    json_write(
                        root / "reports/published_release.json",
                        publish_release(root / "dataset", config_path, cfg["output"]),
                    )
            state["completed"].append(stage)
            state["dataset_manifest"] = _dataset_manifest(root / "dataset")
            state["status"] = "complete" if stage == "verify" else "running"
            json_write(state_file, state)
            print(f"{stage}: passed", flush=True)
            if stage == stop_after:
                return dict(output=str(root), status="prepared", input=inventory)
        except Exception as error:
            state.update(status="failed", failed_stage=stage, error=str(error))
            # A failed mutating stage must be restarted in a fresh workspace.
            json_write(state_file, state)
            raise
    return dict(
        output=str(root),
        status="complete",
        selection=_json_read(root / "reports/selection.json"),
    )
