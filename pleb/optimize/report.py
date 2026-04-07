"""Human-readable optimization summaries."""

from __future__ import annotations

from pathlib import Path

from .models import OptimizationResult


def write_markdown_report(result: OptimizationResult) -> Path:
    """Write a compact Markdown report for an optimization study."""
    out_path = Path(result.out_dir) / "report.md"
    lines = [
        f"# Optimization Report: {result.config.study_name}",
        "",
        f"- Trials: {len(result.trials)}",
        f"- Sampler: `{result.config.sampler}`",
        f"- Execution mode: `{result.config.execution_mode}`",
    ]
    best = result.best_trial
    if best is None:
        lines.extend(["", "No successful trial was produced."])
    else:
        lines.extend(
            [
                "",
                "## Best Trial",
                "",
                f"- Trial ID: `{best.trial_id}`",
                f"- Score: `{best.score}`",
                f"- Run dir: `{best.run_dir}`",
                "",
                "## Parameters",
                "",
            ]
        )
        for key, value in sorted(best.params.items()):
            lines.append(f"- `{key}` = `{value}`")
        lines.extend(["", "## Metrics", ""])
        for key, value in sorted(best.metrics.items()):
            lines.append(f"- `{key}` = `{value}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
