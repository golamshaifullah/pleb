from __future__ import annotations

import json

from pleb.run_report import generate_run_report


def test_generate_run_report_writes_pdf_for_pipeline_run(tmp_path) -> None:
    run_dir = tmp_path / "results" / "wf_demo" / "main"
    (run_dir / "run_settings").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_settings" / "command.txt").write_text(
        "pleb --config demo.toml\n", encoding="utf-8"
    )
    (run_dir / "run_settings" / "pipeline_config.resolved.toml").write_text(
        'home_dir = "/repo"\n', encoding="utf-8"
    )
    fix_dir = run_dir / "fix_dataset" / "main"
    fix_dir.mkdir(parents=True, exist_ok=True)
    (fix_dir / "fix_dataset_report.json").write_text(
        json.dumps(
            [
                {
                    "psr": "J0000+0001",
                    "steps": [{"update_alltim_includes": {"added": 1}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    (fix_dir / "fix_dataset_summary.tsv").write_text(
        "pulsar\tadded_includes\tmissing_jumps\tremoved_lines\n"
        "J0000+0001\t1\t0\t0\n",
        encoding="utf-8",
    )
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "qc_summary.tsv").write_text(
        "pulsar\tvariant\tbranch\tqc_status\tqc_csv\tqc_error\n"
        "J0000+0001\tnew\tmain\tsuccess\t/path/J0000+0001.new_qc.csv\t\n",
        encoding="utf-8",
    )

    pdf_path = generate_run_report(run_dir)

    assert pdf_path is not None
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_generate_run_report_writes_workflow_page_when_steps_supplied(tmp_path) -> None:
    run_dir = tmp_path / "results" / "wf_demo" / "main"
    run_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = generate_run_report(
        run_dir,
        workflow_steps=[
            {
                "step": "pipeline",
                "kind": "pipeline",
                "run_dir": str(run_dir),
                "fix_summary": "",
                "qc_summary": "",
            }
        ],
        output_name="workflow_report.pdf",
    )

    assert pdf_path is not None
    assert pdf_path.exists()
    assert pdf_path.name == "workflow_report.pdf"
