"""Run the pinned PQC implementation with the release clock overlay.

Executed by the native release runner in its configured scientific runtime.
It does not patch installed packages or substitute a timing model.
"""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys


def main(payload_path):
    payload = json.loads(Path(payload_path).read_text())
    root = Path(payload["root"])
    stock = Path(os.environ.get("TEMPO2", "/opt/tempo2/share/tempo2"))
    runtime = root / "runtime/pqc_tempo2"
    if not runtime.exists():
        runtime.mkdir(parents=True)
        for path in stock.iterdir():
            if path.name == "clock":
                shutil.copytree(path, runtime / "clock")
            else:
                (runtime / path.name).symlink_to(
                    path, target_is_directory=path.is_dir()
                )
        for path in (root / "dataset/clockfiles").glob("*"):
            if path.is_file():
                shutil.copyfile(path, runtime / "clock" / path.name)
    os.environ["TEMPO2"] = str(runtime)
    from pleb.outlier_qc import PTAQCConfig, run_pqc_for_parfile
    import pleb.outlier_qc

    cfg = PTAQCConfig(**payload["settings"])
    output = Path(payload["output"])
    run_pqc_for_parfile(Path(payload["par"]), output, cfg)
    metadata = {
        "settings": asdict(cfg),
        "implementation_sha256": hashlib.sha256(
            Path(pleb.outlier_qc.__file__).read_bytes()
        ).hexdigest(),
        "packages": {},
        "clockfiles": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (runtime / "clock").glob("*")
            if p.is_file()
        },
    }
    for name in ("pleb", "pqc", "numpy", "scipy", "libstempo"):
        try:
            metadata["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][name] = "unknown"
    output.with_suffix(".runtime.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main(sys.argv[1])
