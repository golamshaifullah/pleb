from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


PulsarSelection = Union[str, List[str]]  # "ALL" or explicit list


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the data combination pipeline."""

    # Root of the data repository (contains pulsar folders like Jxxxx+xxxx/)
    home_dir: Path

    # Singularity/Apptainer image containing tempo2
    singularity_image: Path

    # Where to write the report output
    results_dir: Path = Path(".")

    # Branches you want to compare/diagnose
    branches: List[str] = field(default_factory=lambda: ["master", "EPTA+InPTA"])

    # Reference branch for change reports
    reference_branch: str = "master"

    # Pulsars to process: "ALL" or list
    pulsars: PulsarSelection = "ALL"

    # Output directory name: None -> timestamped
    outdir_name: Optional[str] = None

    # tempo2 settings
    epoch: str = "55000"

    # If True, re-run tempo2 even if outputs already exist
    force_rerun: bool = False

    # Pipeline toggles
    run_tempo2: bool = True
    make_toa_coverage_plots: bool = True
    make_change_reports: bool = True
    make_covariance_heatmaps: bool = True
    make_residual_plots: bool = True
    make_outlier_reports: bool = True

    # Plotting controls
    dpi: int = 120
    max_covmat_params: Optional[int] = None

    def resolved(self) -> "PipelineConfig":
        """Return a copy with paths expanded/resolved."""
        c = PipelineConfig(**{**asdict(self), "home_dir": Path(self.home_dir), "singularity_image": Path(self.singularity_image), "results_dir": Path(self.results_dir)})
        c.home_dir = c.home_dir.expanduser().resolve()
        c.singularity_image = c.singularity_image.expanduser().resolve()
        c.results_dir = c.results_dir.expanduser().resolve()
        return c

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # serialize Paths
        for k in ("home_dir", "singularity_image", "results_dir"):
            d[k] = str(d[k])
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineConfig":
        def p(x: Any) -> Path:
            return Path(x) if x is not None else Path(".")
        return PipelineConfig(
            home_dir=p(d["home_dir"]),
            singularity_image=p(d["singularity_image"]),
            results_dir=p(d.get("results_dir", ".")),
            branches=list(d.get("branches", ["master", "EPTA+InPTA"])),
            reference_branch=str(d.get("reference_branch", "master")),
            pulsars=d.get("pulsars", "ALL"),
            outdir_name=(None if d.get("outdir_name") in (None, "") else d.get("outdir_name")),
            epoch=str(d.get("epoch", "55000")),
            force_rerun=bool(d.get("force_rerun", False)),
            run_tempo2=bool(d.get("run_tempo2", True)),
            make_toa_coverage_plots=bool(d.get("make_toa_coverage_plots", True)),
            make_change_reports=bool(d.get("make_change_reports", True)),
            make_covariance_heatmaps=bool(d.get("make_covariance_heatmaps", True)),
            make_residual_plots=bool(d.get("make_residual_plots", True)),
            make_outlier_reports=bool(d.get("make_outlier_reports", True)),
            dpi=int(d.get("dpi", 120)),
            max_covmat_params=(None if d.get("max_covmat_params") in (None, "") else d.get("max_covmat_params")),
        )

    @staticmethod
    def load(path: Path) -> "PipelineConfig":
        """Load config from JSON or TOML."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return PipelineConfig.from_dict(data)

        if path.suffix.lower() in (".toml", ".tml"):
            if tomllib is None:
                raise RuntimeError("TOML config requested but tomllib is unavailable in this Python.")
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            # Accept either top-level keys or [pipeline] table
            if "pipeline" in data and isinstance(data["pipeline"], dict):
                data = data["pipeline"]
            return PipelineConfig.from_dict(data)

        raise ValueError(f"Unsupported config file type: {path.suffix}. Use .json or .toml")

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
