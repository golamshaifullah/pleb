from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_review_synthesis.py"
    spec = importlib.util.spec_from_file_location("make_review_synthesis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_guess_variant_uses_qc_filename_suffix() -> None:
    mod = _load_module()

    assert mod._guess_variant(Path("J1909-3744.combined_qc.csv")) == "combined"
    assert mod._guess_variant(Path("J1909-3744.legacy_qc.csv")) == "legacy"
    assert mod._guess_variant(Path("J1909-3744.new_qc.csv")) == "new"


def test_choose_postfit_residual_column_prefers_raw_tempo2_err() -> None:
    mod = _load_module()

    df = pd.DataFrame(
        {
            "tempo2_post_us": [1.0, 2.0, 3.0],
            "tempo2_post": [1.0e-6, 2.0e-6, 3.0e-6],
            "tempo2_err": [0.5, 0.5, 0.5],
            "tempo2_err_us": [500000.0, 500000.0, 500000.0],
        }
    )

    choice = mod.choose_postfit_residual_column(df)

    assert choice.column == "tempo2_post_us"
    assert choice.error_column == "tempo2_err"


def test_scatter_by_decision_draws_errorbars_when_plot_errors_exist() -> None:
    mod = _load_module()

    frame = pd.DataFrame(
        {
            "mjd": [50000.0, 50001.0],
            "plot_residual_us": [1.2, -0.7],
            "plot_err_us": [0.3, 0.4],
            "reviewed_decision": ["KEEP", "BAD_TOA"],
            "manual_action": ["", ""],
        }
    )
    frame = mod.build_plot_subset(frame, max_keep_points=10)

    fig, ax = mod.plt.subplots()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_errorbar(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    ax.errorbar = fake_errorbar  # type: ignore[method-assign]
    mod._scatter_by_decision(
        ax,
        frame,
        "mjd",
        "plot_residual_us",
        xlabel="MJD",
        ylabel="residual",
        yerr="plot_err_us",
    )
    mod.plt.close(fig)

    assert len(calls) == 2
    assert "yerr" in calls[0][1]
    assert "yerr" in calls[1][1]


def test_choose_residual_display_limit_clips_extreme_outliers() -> None:
    mod = _load_module()

    frame = pd.DataFrame(
        {
            "plot_residual_us": [0.2, -0.3, 0.5, 0.4, 0.1, 1200.0, -1300.0],
            "plot_err_us": [0.15, 0.18, 0.2, 0.17, 0.19, 0.25, 0.25],
            "_decision": ["KEEP", "KEEP", "KEEP", "KEEP", "KEEP", "BAD_TOA", "BAD_TOA"],
        }
    )

    limit = mod._choose_residual_display_limit(frame)

    assert limit is not None
    assert limit < 100.0
    assert limit > 1.0


def test_make_variant_overview_plot_uses_axis_legends_not_figure_legend(tmp_path: Path) -> None:
    mod = _load_module()

    frame = pd.DataFrame(
        {
            "mjd": [50000.0, 50001.0, 50002.0],
            "freq": [1400.0, 1500.0, 1600.0],
            "plot_residual_us": [1.2, -0.7, 0.4],
            "plot_err_us": [0.3, 0.4, 0.2],
            "backend_group": ["A", "B", "A"],
            "_decision": ["KEEP", "BAD_TOA", "EVENT"],
            "_manual": [False, False, False],
        }
    )

    figure_legend_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    axis_legend_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real_figure_legend = mod.plt.Figure.legend
    real_axes_legend = mod.plt.Axes.legend

    def fake_figure_legend(self: object, *args: object, **kwargs: object) -> object:
        figure_legend_calls.append((args, kwargs))
        return real_figure_legend(self, *args, **kwargs)

    def fake_axes_legend(self: object, *args: object, **kwargs: object) -> object:
        axis_legend_calls.append((args, kwargs))
        return real_axes_legend(self, *args, **kwargs)

    mod.plt.Figure.legend = fake_figure_legend  # type: ignore[method-assign]
    mod.plt.Axes.legend = fake_axes_legend  # type: ignore[method-assign]
    try:
        mod.make_variant_overview_plot(
            frame,
            title="demo",
            residual_label="tempo2 post-fit residual [us]",
            out_path=tmp_path / "overview.png",
        )
    finally:
        mod.plt.Figure.legend = real_figure_legend  # type: ignore[method-assign]
        mod.plt.Axes.legend = real_axes_legend  # type: ignore[method-assign]

    assert figure_legend_calls == []
    assert len(axis_legend_calls) >= 2


def test_discover_artifacts_includes_model_check_outputs(tmp_path: Path) -> None:
    mod = _load_module()
    repo_root = tmp_path
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    run_dir = tmp_path / "results" / "j1909_step6_apply_delete" / "j1909_step6_apply_delete"
    (run_dir / "binary_analysis").mkdir(parents=True)
    (run_dir / "param_scan").mkdir(parents=True)
    (run_dir / "change_report").mkdir(parents=True)
    (run_dir / "binary_analysis" / "binary_analysis.tsv").write_text(
        "pulsar\tbranch\tBINARY\nJ1909-3744\tj1909_step6_apply_delete\tELL1\n",
        encoding="utf-8",
    )
    (run_dir / "param_scan" / "param_scan_j1909_step6_apply_delete.tsv").write_text(
        "pulsar\tbranch\tcandidate\tlrt_p_value\tlrt_delta_chisq\nJ1909-3744\tj1909_step6_apply_delete\tDM2\t0.01\t12.0\n",
        encoding="utf-8",
    )
    (run_dir / "change_report" / "MODEL_COMPARISON_main.tsv").write_text(
        "pulsar\tbranch\treference\tlrt_p_value\nJ1909-3744\tj1909_step6_apply_delete\tmain\t0.03\n",
        encoding="utf-8",
    )
    (run_dir / "change_report" / "NEW_PARAM_SIGNIFICANCE_main.tsv").write_text(
        "pulsar\tbranch\treference\tn_new_params\tn_new_sig_z>=3\nJ1909-3744\tj1909_step6_apply_delete\tmain\t2\t1\n",
        encoding="utf-8",
    )
    rt = mod.StageRuntime(
        spec=mod.STAGES[5],
        branch="j1909_step6_apply_delete",
        branch_exists=True,
        commit="abc123",
        run_dir=run_dir,
        run_dir_exists=True,
        notes="test",
    )

    rows = mod.discover_artifacts([rt], dataset_root, repo_root)
    paths = {row["path"] for row in rows}

    assert "results/j1909_step6_apply_delete/j1909_step6_apply_delete/binary_analysis/binary_analysis.tsv" in paths
    assert "results/j1909_step6_apply_delete/j1909_step6_apply_delete/param_scan/param_scan_j1909_step6_apply_delete.tsv" in paths
    assert "results/j1909_step6_apply_delete/j1909_step6_apply_delete/change_report/MODEL_COMPARISON_main.tsv" in paths
    assert "results/j1909_step6_apply_delete/j1909_step6_apply_delete/change_report/NEW_PARAM_SIGNIFICANCE_main.tsv" in paths


def test_build_stage_runtimes_and_discover_artifacts_support_branch_only_run_dirs(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    repo_root = tmp_path
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    (repo_root / "README.md").write_text("root\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo_root, check=True)

    subprocess.run(["git", "checkout", "-b", "j1909_step5_apply_comments"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    run_dir = repo_root / "results" / "j1909_step5_apply_comments" / "j1909_step5_apply_comments"
    (run_dir / "fix_dataset" / "j1909_step5_apply_comments").mkdir(parents=True)
    (run_dir / "fix_dataset" / "j1909_step5_apply_comments" / "fix_dataset_summary.tsv").write_text(
        "branch\tpulsar\nj1909_step5_apply_comments\tJ1909-3744\n",
        encoding="utf-8",
    )
    (run_dir / "run_report.pdf").write_bytes(b"%PDF-1.4\n")
    subprocess.run(["git", "add", "results"], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "step5 artifacts"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    subprocess.run(["git", "checkout", "main"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (repo_root / "results").exists():
        shutil.rmtree(repo_root / "results")

    args = mod.argparse.Namespace(
        psr="J1909-3744",
        slug="j1909",
        final_branch="j1909_step6_apply_delete",
        stage_branch=[],
        stage_run=[],
    )
    runtimes = mod.build_stage_runtimes(args, repo_root, repo_root / "results")
    step5 = next(rt for rt in runtimes if rt.spec.key == "step5_apply_comments")

    assert step5.run_dir_exists is True
    assert step5.notes == "branch_only"

    rows = mod.discover_artifacts(runtimes, repo_root / "dataset", repo_root)
    paths = {row["path"] for row in rows if row["stage"] == "step5_apply_comments"}
    assert "results/j1909_step5_apply_comments/j1909_step5_apply_comments/run_report.pdf" in paths
    assert (
        "results/j1909_step5_apply_comments/j1909_step5_apply_comments/fix_dataset/"
        "j1909_step5_apply_comments/fix_dataset_summary.tsv"
    ) in paths


def test_build_model_check_package_summarizes_unwired_outputs(tmp_path: Path) -> None:
    mod = _load_module()
    repo_root = tmp_path
    run_dir = tmp_path / "results" / "j1909_step6_apply_delete" / "j1909_step6_apply_delete"
    (run_dir / "binary_analysis").mkdir(parents=True)
    (run_dir / "param_scan").mkdir(parents=True)
    (run_dir / "change_report").mkdir(parents=True)
    (run_dir / "binary_analysis" / "binary_analysis.tsv").write_text(
        "pulsar\tbranch\tBINARY\tPB\nJ1909-3744\tj1909_step6_apply_delete\tELL1\t1.53\n",
        encoding="utf-8",
    )
    (run_dir / "param_scan" / "param_scan_j1909_step6_apply_delete.tsv").write_text(
        (
            "pulsar\tbranch\tcandidate\tlrt_p_value\tlrt_delta_chisq\tmax_param_z\n"
            "J1909-3744\tj1909_step6_apply_delete\tDM2\t0.04\t10.0\t2.2\n"
            "J1909-3744\tj1909_step6_apply_delete\tFB1\t0.01\t18.0\t4.5\n"
        ),
        encoding="utf-8",
    )
    (run_dir / "change_report" / "MODEL_COMPARISON_main.tsv").write_text(
        "pulsar\tbranch\treference\tdelta_redchisq\tlrt_delta_chisq\tlrt_p_value\nJ1909-3744\tj1909_step6_apply_delete\tmain\t-0.2\t9.0\t0.02\n",
        encoding="utf-8",
    )
    (run_dir / "change_report" / "NEW_PARAM_SIGNIFICANCE_main.tsv").write_text(
        "pulsar\tbranch\treference\tn_new_params\tn_new_with_numeric_sigma\tn_new_sig_z>=3\tmax_new_param_z\tmax_new_param\nJ1909-3744\tj1909_step6_apply_delete\tmain\t2\t2\t1\t5.1\tFB1\n",
        encoding="utf-8",
    )
    rt = mod.StageRuntime(
        spec=mod.STAGES[5],
        branch="j1909_step6_apply_delete",
        branch_exists=True,
        commit="abc123",
        run_dir=run_dir,
        run_dir_exists=True,
        notes="test",
    )
    artifact_rows = mod.discover_artifacts([rt], tmp_path / "dataset", repo_root)
    model_dir = tmp_path / "review" / "10_model_checks"

    package = mod.build_model_check_package(
        runtimes=[rt],
        artifact_rows=artifact_rows,
        repo_root=repo_root,
        model_dir=model_dir,
    )

    assert len(package["artifact_index_rows"]) == 4
    assert len(package["binary_rows"]) == 1
    assert package["param_scan_rows"][0]["candidate"] == "FB1"
    assert package["change_model_rows"][0]["lrt_p_value"] == "0.02"
    assert package["new_param_rows"][0]["max_new_param"] == "FB1"
    assert (model_dir / "model_checks.md").exists()


def test_qc_csvs_for_pulsar_has_fallback_glob(tmp_path: Path) -> None:
    mod = _load_module()
    run_dir = tmp_path / "results" / "j1909_step4_detect_selected"
    qc_dir = run_dir / "qc" / "j1909_step4_detect_selected"
    qc_dir.mkdir(parents=True)
    csv_path = qc_dir / "J1909-3744.combined_qc.csv"
    csv_path.write_text("col\n1\n", encoding="utf-8")

    found = mod._qc_csvs_for_pulsar(run_dir, "J1909-3744")

    assert csv_path in found


def test_parse_markdown_pdf_blocks_extracts_tables() -> None:
    mod = _load_module()

    text = """# Index

Intro paragraph.

| stage | branch | status |
| --- | --- | --- |
| step1 | j1909_step1_fix | yes |
| step2 | j1909_step2_detect_variants | no |
"""

    blocks = mod._parse_markdown_pdf_blocks(text)

    assert [block["kind"] for block in blocks] == ["text", "table"]
    assert blocks[1]["fields"] == ("stage", "branch", "status")
    assert blocks[1]["rows"][0]["branch"] == "j1909_step1_fix"
    assert blocks[1]["rows"][1]["status"] == "no"


def test_build_pdf_with_matplotlib_writes_pdf(tmp_path: Path) -> None:
    mod = _load_module()
    out_root = tmp_path / "review_package"
    dirs = mod.mkdirs(out_root)

    (out_root / "index.md").write_text(
        (
            "# Index\n\nA compact review package.\n\n"
            "| stage | branch | branch_exists | run_dir_exists | commit |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| step1_fix | j1909_step1_fix | yes | yes | abc123 |\n"
            "| step2_detect_variants | j1909_step2_detect_variants | yes | yes | def456 |\n"
        ),
        encoding="utf-8",
    )
    (dirs["decision"] / "final_data_quality_decision_sheet.md").write_text(
        "# Final decision\n\n- Decision: ACCEPT_CANDIDATE_FOR_HUMAN_SIGNOFF\n",
        encoding="utf-8",
    )
    (dirs["postfit"] / "postfit_review.md").write_text(
        "# Post-fit review\n\nResidual plots follow.\n",
        encoding="utf-8",
    )
    (dirs["model_checks"] / "model_checks.md").write_text(
        "# Model checks\n\nNo binary concerns found.\n",
        encoding="utf-8",
    )
    (dirs["noise"] / "whitenoise_not_found.md").write_text(
        "# Whitenoise\n\nNot run.\n",
        encoding="utf-8",
    )
    (dirs["postfit"] / "variant_postfit_summary.tsv").write_text(
        (
            "variant\tn_rows\tn_postfit\tn_bad_toa\tn_event\tn_review_event\tmax_abs_sigma\tmax_abs_sigma_keep\t"
            "worst_backend_by_flagged_fraction\tworst_backend_flagged_fraction\n"
            "combined\t10\t10\t2\t1\t0\t6.2\t2.1\tNRT.NUPPI.1484\t0.3\n"
        ),
        encoding="utf-8",
    )
    (dirs["qc"] / "backend_flag_summary.tsv").write_text(
        (
            "variant\tbackend\tn_rows\tn_flagged\tflagged_fraction\tmedian_abs_postfit_us\tmax_abs_postfit_us\n"
            "combined\tNRT.NUPPI.1484\t10\t3\t0.3\t0.5\t4.2\n"
        ),
        encoding="utf-8",
    )
    fig, ax = mod.plt.subplots()
    ax.plot([0, 1], [0, 1])
    mod.savefig(fig, dirs["postfit"] / "plots" / "combined_overview.png")

    args = mod.argparse.Namespace(
        psr="J1909-3744",
        pdf_name="",
        pandoc="pandoc",
        pdf_engine="xelatex",
    )
    pdf_path = mod.build_pdf_with_matplotlib(
        args=args,
        out_root=out_root,
        decision="ACCEPT_CANDIDATE_FOR_HUMAN_SIGNOFF",
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
