import pytest

from pleb.timing_selection import fit_statistics, select_timing_model


def statistics(n, k, chi):
    return f"Number of points in fit = {n}\nNumber of fit parameters: {k}\nChisqr/nfree = {chi} / {n-k} = {chi/(n-k)}\n"


def test_selects_supported_candidate_and_keeps_baseline_untouched(tmp_path):
    base = tmp_path / "baseline.par"
    base.write_text("F0 100 1 0.1\n")
    calls = []

    def execute(source, folder):
        calls.append(source.read_text())
        output = folder / "new.par"
        output.write_text(source.read_text())
        return output, statistics(100, 3, 140)

    chosen, report = select_timing_model(
        base,
        fit_statistics(statistics(100, 2, 200)),
        ["F2"],
        tmp_path / "scan",
        execute,
        min_delta_bic=10,
    )
    assert chosen != base and "F2" in chosen.read_text()
    assert base.read_text() == "F0 100 1 0.1\n"
    assert len(calls) == 1
    assert report["trials"][0]["delta_bic"] == pytest.approx(55.394829814)
    assert report["trials"][0]["conditional_white_noise"]


def test_search_rejects_changed_observation_selection(tmp_path):
    base = tmp_path / "baseline.par"
    base.write_text("F0 100 1\n")

    def execute(source, folder):
        return source, statistics(99, 3, 100)

    with pytest.raises(ValueError, match="changed the selected TOAs"):
        select_timing_model(
            base,
            fit_statistics(statistics(100, 2, 200)),
            ["F2"],
            tmp_path / "scan",
            execute,
        )


def test_fit_statistics_rejects_invalid_degrees_of_freedom():
    with pytest.raises(ValueError, match="Invalid"):
        fit_statistics(statistics(2, 3, 10))


def test_weak_candidate_keeps_original_model(tmp_path):
    base = tmp_path / "baseline.par"
    base.write_text("F0 100 1\n")

    def execute(source, folder):
        return source, statistics(100, 3, 199)

    chosen, _ = select_timing_model(
        base,
        fit_statistics(statistics(100, 2, 200)),
        ["F2"],
        tmp_path / "scan",
        execute,
    )
    assert chosen == base
