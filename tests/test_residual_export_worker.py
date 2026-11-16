"""Fit/export controls with synthetic inputs; no telescope data or external fit."""

import pytest

from pleb import residual_export_worker as worker


def test_saved_model_export_creates_its_work_directory(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    class ReachedModelLoad(Exception):
        pass

    pulsar = "J0000+0000"
    data = tmp_path / "data"
    folder = data / pulsar
    folder.mkdir(parents=True)
    (folder / f"{pulsar}_combined.par").write_text("PSRJ J0000+0000\n")
    (folder / f"{pulsar}_combined_all.tim").write_text("FORMAT 1\n")
    model = tmp_path / "saved/new.par"
    model.parent.mkdir()
    model.write_text("PSRJ J0000+0000\n")
    (model.parent / "tempo2.log").write_text("saved fit")
    monkeypatch.setattr(worker, "stats", lambda text: {"n": 1})

    def load_model(**kwargs):
        assert (tmp_path / "fits" / pulsar / "combined").is_dir()
        assert kwargs["parfile"] == str(model)
        raise ReachedModelLoad

    libstempo = ModuleType("libstempo")
    libstempo.tempopulsar = load_model
    monkeypatch.setitem(sys.modules, "libstempo", libstempo)
    for package, attribute in [("enterprise", "Tempo2Pulsar"), ("discovery", "Pulsar")]:
        parent = ModuleType(package)
        parent.__path__ = []
        child = ModuleType(package + ".pulsar")
        setattr(child, attribute, object)
        monkeypatch.setitem(sys.modules, package, parent)
        monkeypatch.setitem(sys.modules, package + ".pulsar", child)
    with pytest.raises(ReachedModelLoad):
        worker.process(data, tmp_path, pulsar, model_file=model)


def test_convergence_preserves_small_decimal_parameter_steps():
    old = {
        "F0": dict(
            value="100.00000000000000000000",
            uncertainty="1e-18",
            unit="Hz",
            kind="timing",
        ),
        "JUMP": dict(value="0", uncertainty="1", unit="s", kind="jump"),
    }
    new = {
        "F0": dict(
            value="100.00000000000000000001",
            uncertainty="1e-18",
            unit="Hz",
            kind="timing",
        ),
        "JUMP": dict(value="1000000", uncertainty="1", unit="s", kind="jump"),
    }
    assert worker.maximum_step(old, new) == pytest.approx(0.01)


def test_fixed_model_fit_requires_three_stable_passes(tmp_path, monkeypatch):
    seen = []

    def fit(par, tim, folder, root):
        folder.mkdir(parents=True)
        p = folder / "new.par"
        p.write_text("F0 100 1 .1\n")
        (folder / "tempo2.log").write_text("RMS post-fit residual = 1.0 (us)\n")
        seen.append(folder)
        return p, dict(n=100, k=1, weighted_fit=True)

    monkeypatch.setattr(worker, "tempo", fit)
    chosen, stats, history = worker.converged_fit(
        tmp_path / "input.par", tmp_path / "input.tim", tmp_path / "fits", tmp_path
    )
    assert len(seen) == 3 and len(history) == 3 and chosen == seen[-1] / "new.par"


def test_unstable_fit_is_not_exported(tmp_path, monkeypatch):
    seen = []

    def fit(par, tim, folder, root):
        folder.mkdir(parents=True)
        seen.append(folder)
        p = folder / "new.par"
        p.write_text(f"F0 {100+len(seen)} 1 .1\n")
        (folder / "tempo2.log").write_text("RMS post-fit residual = 1.0 (us)\n")
        return p, dict(n=100, k=1, weighted_fit=True)

    monkeypatch.setattr(worker, "tempo", fit)
    with pytest.raises(RuntimeError, match="did not converge"):
        worker.converged_fit(
            tmp_path / "input.par", tmp_path / "input.tim", tmp_path / "fits", tmp_path
        )
    assert len(seen) == 8


def test_longer_fit_budget_keeps_convergence_checks(tmp_path, monkeypatch):
    seen = []

    def fit(par, tim, folder, root):
        folder.mkdir(parents=True)
        seen.append(folder)
        fitted = folder / "new.par"
        fitted.write_text(f"F0 {100 + min(len(seen), 10)} 1 .1\n")
        (folder / "tempo2.log").write_text("RMS post-fit residual = 1.0 (us)\n")
        return fitted, dict(n=100, k=1, weighted_fit=True)

    monkeypatch.setattr(worker, "tempo", fit)
    chosen, _, history = worker.converged_fit(
        tmp_path / "input.par",
        tmp_path / "input.tim",
        tmp_path / "fits",
        tmp_path,
        max_iterations=16,
    )
    assert len(history) == 11 and chosen == seen[-1] / "new.par"
    assert history[-1]["max_parameter_step_sigma"] == 0


def test_fit_budget_cannot_disable_minimum_passes(tmp_path):
    with pytest.raises(ValueError, match="three fit iterations"):
        worker.converged_fit(
            tmp_path / "p",
            tmp_path / "t",
            tmp_path / "fits",
            tmp_path,
            max_iterations=2,
        )


def test_export_metadata_preserves_time_addsat_and_end(tmp_path):
    p = tmp_path / "input.tim"
    p.write_text(
        "FORMAT 1\na 1400 59000 1 jbdfb -sys A\nTIME .001\nb 1400 59001 1 jbdfb -sys A -addsat .002\nEND\nc 1400 59002 1 jbdfb -sys A\n"
    )
    rows = worker.tim_metadata(p)
    assert len(rows) == 2 and rows[0]["mjd"] == 59000
    assert (rows[1]["mjd"] - 59001) * 86400 == pytest.approx(0.003, abs=1e-6)
    assert all(r["telescope"] == "jbdfb" for r in rows)
