import json
import shutil

import pytest

from pleb.observations import read_observations, source_id, write_observations
from pleb.release import (
    ReleaseError,
    apply_review_catalog,
    digest,
    input_inventory,
    load_release_config,
    run_release,
    publish_release,
    attach_qc_identities,
)


def project(root):
    raw = root / "raw/J0000+0000"
    raw.mkdir(parents=True)
    (raw / "J0000+0000.par").write_text("PSRJ J0000+0000\nF0 1 1\nDM 1\n")
    (raw / "TEST.A.1400.tim").write_text(
        "FORMAT 1\n"
        "obs1 1400 58000 1 @ -sys TEST.A.1400\n"
        "obs2 1400 58001 1 @ -sys TEST.A.1400\n"
        "C obs3 1400 58002 1 @ -sys TEST.A.1400\n"
    )
    config = root / "release.toml"
    config.write_text("""release_version = 1
[run]
mode = "release"
[source]
raw = "raw"
[ingest]
sources = ["."]
par_roots = ["."]
pulsars = ["J0000+0000"]
[ingest.backends."TEST.A.1400"]
root = "."
tim_glob = "TEST.A.1400.tim"
[processing]
remove_patterns = []
remove_overlaps_exact = false
[qc]
mode = "disabled"
[review]
mode = "none"
[timing]
fit = false
validate = false
weighted = true
ephem = "DE440"
clock = "TT(BIPM2024)"
[variants]
combined = ["*"]
[output]
directory = "result"
""")
    return config


@pytest.fixture(autouse=True)
def no_optional_plotting(monkeypatch):
    # Plot reports are unrelated to selection correctness and expensive here.
    monkeypatch.setattr("pleb.ingest._write_ingest_upset_plots", lambda *a: [])


def test_raw_to_release_and_verified_resume(tmp_path):
    config = project(tmp_path)
    result = run_release(config)
    assert result["selection"]["active"] == 2
    assert run_release(config, resume=True)["selection"] == result["selection"]
    assert (
        "MODE 1"
        in (tmp_path / "result/dataset/J0000+0000/J0000+0000_combined.par").read_text()
    )
    assert (
        "TT(BIPM2024)"
        in (tmp_path / "result/dataset/J0000+0000/J0000+0000_combined.par").read_text()
    )
    assert not (tmp_path / "raw/J0000+0000/J0000+0000_combined.par").exists()


def test_source_decisions_survive_moving_checkout(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    config = project(first)
    path, cfg = load_release_config(config)
    _, _, records, inventory = input_inventory(path, cfg)
    catalog = dict(
        format_version=1,
        identities_sha256=inventory["identities_sha256"],
        decisions=[
            dict(source_id=records[0]["source_id"], action="comment", reason="expert"),
            dict(source_id=records[2]["source_id"], action="keep", reason="expert"),
        ],
    )
    (first / "decisions.json").write_text(json.dumps(catalog))
    text = config.read_text().replace(
        'mode = "none"', 'mode = "catalog"\ncatalog = "decisions.json"'
    )
    config.write_text(text)
    run = run_release(config)
    shutil.copytree(first, tmp_path / "moved", ignore=shutil.ignore_patterns("result"))
    moved = run_release(tmp_path / "moved/release.toml")
    assert run["selection"] == moved["selection"]
    rows = read_observations(
        tmp_path / "moved/result/dataset/J0000+0000/tims/TEST.A.1400.tim"
    )
    assert [r.state for r in rows] == ["commented", "active", "active"]


def test_unknown_or_changed_decisions_fail_before_output(tmp_path):
    config = project(tmp_path)
    catalog = dict(format_version=1, identities_sha256="wrong", decisions=[])
    (tmp_path / "decisions.json").write_text(json.dumps(catalog))
    config.write_text(
        config.read_text().replace(
            'mode = "none"', 'mode = "catalog"\ncatalog = "decisions.json"'
        )
    )
    with pytest.raises(ReleaseError, match="does not match"):
        run_release(config)
    assert not (tmp_path / "result").exists()


def test_resume_rejects_external_dataset_edits(tmp_path):
    config = project(tmp_path)
    run_release(config)
    p = tmp_path / "result/dataset/J0000+0000/tims/TEST.A.1400.tim"
    p.write_text(p.read_text() + "unexpected 1400 59000 1 @\n")
    with pytest.raises(ReleaseError, match="changed outside"):
        run_release(config, resume=True)


def test_duplicate_copy_ids_and_comment_only_action(tmp_path):
    p = tmp_path / "a.tim"
    p.write_text("a 1400 58000 1 @\na 1400 58000 1 @\n")
    rows = read_observations(p)
    ids = [source_id("a.tim", "abc", row.line) for row in rows]
    assert ids[0] != ids[1]
    write_observations(
        p, [(ids[0], rows[0], True, "keep"), (ids[1], rows[1], False, "duplicate")]
    )
    assert len(read_observations(p)) == 2
    assert [r.state for r in read_observations(p)] == ["active", "commented"]


def test_merging_time_and_weight_contexts_does_not_leak(tmp_path):
    a, b = tmp_path / "a.tim", tmp_path / "b.tim"
    a.write_text("TIME 2\nEFAC 3\nEQUAD 4\na 1400 58000 1 @ -addsat -1\n")
    b.write_text("b 1400 58001 1 @\n")
    rows = [*read_observations(a), *read_observations(b)]
    write_observations(
        tmp_path / "merged.tim", [(str(i), row, True, "") for i, row in enumerate(rows)]
    )
    merged = read_observations(tmp_path / "merged.tim")
    assert [(r.epoch, r.efac, r.equad) for r in merged] == [
        (r.epoch, r.efac, r.equad) for r in rows
    ]


def test_unmatched_or_ambiguous_review_is_not_silently_ignored(tmp_path):
    p = tmp_path / "a.tim"
    p.write_text("a 1400 58000 1 @\n")
    bindings = [dict(file="a.tim", source_id="a", pulsar="J0000+0000")]
    with pytest.raises(ReleaseError, match="Unmatched"):
        apply_review_catalog(
            tmp_path,
            bindings,
            dict(
                format_version=1, decisions=[dict(source_id="missing", action="keep")]
            ),
        )
    with pytest.raises(ReleaseError, match="Repeated"):
        apply_review_catalog(
            tmp_path,
            bindings,
            dict(format_version=1, decisions=[dict(source_id="a", action="keep")] * 2),
        )


def test_rejects_typo_and_destructive_processing(tmp_path):
    config = project(tmp_path)
    config.write_text(
        config.read_text().replace("remove_patterns = []", 'dedupe_action = "delete"')
    )
    with pytest.raises(ReleaseError, match="requires"):
        load_release_config(config)
    config.write_text(
        config.read_text().replace(
            'dedupe_action = "delete"', 'dedupe_acton = "comment"'
        )
    )
    with pytest.raises(ReleaseError, match="Unknown processing"):
        load_release_config(config)


def test_historical_comment_styles_stay_inactive(tmp_path):
    p = tmp_path / "legacy.tim"
    p.write_text(
        "CC ?c001 1400 58000 1 @\nC?? c002 1400 58001 1 @\nCc003 1400 58002 1 @\n#c004 1400 58003 1 @\nc005 1400 58004 1 @\n"
    )
    rows = read_observations(p)
    assert [r.filename for r in rows] == ["?c001", "c002", "c003", "c004", "c005"]
    assert [r.state for r in rows] == ["commented"] * 4 + ["active"]


def test_source_model_change_invalidates_input_lock(tmp_path):
    config = project(tmp_path)
    path, cfg = load_release_config(config)
    _, _, _, inventory = input_inventory(path, cfg)
    config.write_text(
        config.read_text().replace(
            'raw = "raw"', 'raw = "raw"\ninventory_sha256 = "' + digest(inventory) + '"'
        )
    )
    par = tmp_path / "raw/J0000+0000/J0000+0000.par"
    par.write_text(par.read_text().replace("F0 1", "F0 2"))
    with pytest.raises(ReleaseError, match="inventory differs"):
        run_release(config)
    assert not (tmp_path / "result").exists()


def test_interactive_review_resumes_with_recorded_source_decision(tmp_path):
    from pleb.qc_review import load_qc_frames, make_override_rows, write_overrides

    config = project(tmp_path)
    config.write_text(
        config.read_text().replace('mode = "none"', 'mode = "interactive"')
    )
    first = run_release(config)
    assert first["status"] == "awaiting_review"
    review = tmp_path / "result/review"
    frame = load_qc_frames(review)
    assert frame["review_id"].tolist() == frame["source_id"].tolist()
    overrides = make_override_rows(
        frame.iloc[[0]], action="mark_bad", reason="manual check"
    )
    write_overrides(overrides, review / "manual_qc_overrides.csv")
    final = run_release(config, resume=True)
    assert final["selection"]["active"] == 1
    saved = json.loads((tmp_path / "result/reports/review_decisions.json").read_text())
    assert saved["decisions"][0]["source_id"] == frame.iloc[0]["source_id"]


def test_publish_replaces_dataset_without_touching_checkout(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        return subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    old = repo / "release"
    old.mkdir()
    (old / "old.tim").write_text("old")
    (repo / "check.txt").write_text("unchanged")
    git("add", ".")
    git("commit", "-qm", "initial")
    head = git("rev-parse", "HEAD")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "new.tim").write_text("new")
    (repo / "untracked").write_text("preserve")
    settings = dict(repository="repo", branch="rebuilt", dataset_path="release")
    result = publish_release(dataset, tmp_path / "config.toml", settings)
    assert git("rev-parse", "HEAD") == head
    assert (
        git("ls-tree", "-r", "--name-only", result["commit"], "--", "release")
        == "release/new.tim"
    )
    assert (old / "old.tim").read_text() == "old"
    assert git("show", result["commit"] + ":check.txt") == "unchanged"
    assert (repo / "untracked").read_text() == "preserve"
    assert publish_release(dataset, tmp_path / "config.toml", settings) == result


def test_qc_source_match_rejects_identical_active_candidates(tmp_path):
    import pandas as pd

    p = tmp_path / "a.tim"
    p.write_text("a 1400 58000 1 @\na 1400 58000 1 @\n")
    bindings = [
        dict(file="a.tim", source_id=str(i), pulsar="J0000+0000") for i in range(2)
    ]
    frame = pd.DataFrame([dict(_timfile="a.tim", mjd=58000, freq=1400)])
    with pytest.raises(ReleaseError, match="2 source matches"):
        attach_qc_identities(frame, tmp_path, bindings, "J0000+0000")


def test_reference_detects_timing_correction_even_when_count_matches(tmp_path):
    config = project(tmp_path)
    result = run_release(config)
    (tmp_path / "expected.json").write_text(json.dumps(result["selection"]))
    config.write_text(
        config.read_text()
        .replace(
            '[output]\ndirectory = "result"',
            '[verify]\nreference = "expected.json"\n[output]\ndirectory = "corrected"',
        )
        .replace("[qc]", '[[processing.time_rules]]\nseconds = "1"\n[qc]')
    )
    with pytest.raises(ReleaseError, match="differs from the approved"):
        run_release(config)


def test_release_timing_runs_only_the_requested_fit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = project(tmp_path)
    config.write_text(
        config.read_text().replace("fit = false", "fit = true\nnative = true")
    )
    calls = []

    def execute(command, **kwargs):
        folder = kwargs["cwd"]
        assert "-newpar" in command
        assert "-nofit" not in command
        calls.append(command)
        text = (folder / "input.par").read_text()
        assert "NITS 1" in text and "MODE 1" in text
        (folder / "new.par").write_text(text)
        return SimpleNamespace(returncode=0, stdout="Timing fit completed", stderr="")

    # Restrict the mock to the timing stage; ingestion itself uses no subprocess.
    monkeypatch.setattr("pleb.release.subprocess.run", execute)
    run_release(config)
    assert len(calls) == 1


def test_migration_uses_raw_and_rejects_timing_context_changes(tmp_path):
    from pleb.review_migration import import_selection

    config = project(tmp_path)
    run_release(config)
    reference = tmp_path / "result/dataset"
    output, expected = tmp_path / "decisions.json", tmp_path / "expected.json"
    result = import_selection(
        config, reference, output, expected, tmp_path / "migration"
    )
    assert result["active"] == 2
    assert json.loads(expected.read_text())["measurements_sha256"]
    p = reference / "J0000+0000/tims/TEST.A.1400.tim"
    p.write_text(p.read_text().replace("58000 1 @", "58000 1 @ -padd 0.1"))
    with pytest.raises(ReleaseError, match="timing-context"):
        import_selection(
            config,
            reference,
            tmp_path / "bad.json",
            tmp_path / "bad-reference.json",
            tmp_path / "bad-migration",
        )
    assert not (tmp_path / "bad.json").exists()


def test_include_count_preserves_inline_and_child_selection(tmp_path):
    from pleb.tim_utils import active_count

    (tmp_path / "child.tim").write_text(
        "c001 1400 58000 1 @\nEND\nc002 1400 58001 1 @\n"
    )
    p = tmp_path / "wrapper.tim"
    p.write_text(
        "c003 1400 58002 1 @\nINCLUDE child.tim\nSKIP\nINCLUDE absent.tim\nNOSKIP\nC c004 1400 58003 1 @\n"
    )
    assert active_count(p) == 2


def test_release_rejects_destructive_nested_qc_action(tmp_path):
    config = project(tmp_path)
    config.write_text(config.read_text() + '\n[qc.apply]\nqc_solar_action = "delete"\n')
    with pytest.raises(ReleaseError, match="Release QC requires"):
        run_release(config)
    assert not (tmp_path / "result").exists()


def test_unsupported_timing_directive_is_not_silently_lost(tmp_path):
    p = tmp_path / "source.tim"
    p.write_text("PHASE 1\nobs 1400 58000 1 @\n")
    with pytest.raises(ValueError, match="Unsupported timing directive"):
        read_observations(p)


def test_release_policy_does_not_depend_on_checkout_tables(tmp_path, monkeypatch):
    from pleb import dataset_fix
    from pleb.release import _processing_config

    config = project(tmp_path)
    path, cfg = load_release_config(config)
    # Poison legacy ambient defaults: native mode must use only declared maps.
    monkeypatch.setattr(dataset_fix, "_BACKEND_BW_TABLE", {"TEST.A.1400.tim": 999})
    native = _processing_config(path, cfg)
    assert native.backend_bandwidths == {}
    assert native.overlap_exact_map == {}
    cfg["processing"].update(
        backend_bandwidths={"TEST.A.1400.tim": 128},
        overlap_exact_map={"TEST.A.1400.tim": ["TEST.B.1400.tim"]},
    )
    native = _processing_config(path, cfg)
    assert native.backend_bandwidths == {"TEST.A.1400.tim": 128}
    parent = tmp_path / "overlap/J0000+0000"
    (parent / "tims").mkdir(parents=True)
    a = parent / "tims/TEST.A.1400.tim"
    a.write_text("obs 1400 58000 1 @\nobs 1400 58000.000001 2 @\n")
    dataset_fix.dedupe_timfile_toas(
        a,
        apply=True,
        backup=False,
        same_obs_same_bw_globs=["TEST.*.tim"],
        backend_bandwidths=native.backend_bandwidths,
    )
    assert [r.state for r in read_observations(a)] == ["active", "commented"]
    (parent / "tims/TEST.B.1400.tim").write_text("obs 1400 58000 1 @\n")
    report = dataset_fix.remove_overlaps_exact(
        parent, overlap_map=native.overlap_exact_map, apply=True, backup=False
    )
    assert report["commented"] == 1


def test_qc_worker_uses_run_local_logs_and_cache(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    from pleb.release import _stage_qc

    config = project(tmp_path)
    run_release(config, stop_after="initial_models")
    path, cfg = load_release_config(config)
    cfg["qc"] = {"mode": "detect"}
    cfg["timing"]["native"] = False
    cfg["container"] = {"image": "example.sif"}
    monkeypatch.setenv("PLEB_LOG_DIR", "/unwritable/shared-logs")
    monkeypatch.setenv("MPLCONFIGDIR", "/unwritable/shared-mpl")

    def worker(command, **kwargs):
        root = tmp_path / "result"
        environment = kwargs["env"]
        for name, directory in [
            ("PLEB_LOG_DIR", "logs"),
            ("MPLCONFIGDIR", "matplotlib"),
        ]:
            assert environment[name] == str(root / "runtime" / directory)
            assert environment["APPTAINERENV_" + name] == environment[name]
            assert (root / "runtime" / directory).is_dir()
        pd.DataFrame(
            [
                dict(_timfile="TEST.A.1400.tim", mjd=58000, freq=1400),
                dict(_timfile="TEST.A.1400.tim", mjd=58001, freq=1400),
            ]
        ).to_csv(root / "qc/J0000+0000_qc.csv", index=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    (tmp_path / "result/qc").mkdir()
    monkeypatch.setattr("pleb.release.subprocess.run", worker)
    monkeypatch.setattr("pleb.dataset_fix.fix_pulsar_dataset", lambda *a: {})
    monkeypatch.setattr("pleb.tempo2.container_runtime", lambda: "apptainer")
    _stage_qc(tmp_path / "result", path, cfg, ["J0000+0000"])
