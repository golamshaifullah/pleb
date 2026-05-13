"""Tests for public release comparison utilities."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd
from astropy import units as u
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

from pleb.public_release_compare import (
    ProviderSpec,
    _build_comparison,
    _download_and_extract_assets,
    _find_par_files,
    _parse_parfile,
    _parse_zenodo_record_id,
    ReleaseAsset,
    compare_public_releases,
    load_provider_specs,
    normalize_astrometry,
    resolve_latest_assets,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )


def test_parse_parfile_extracts_value_and_error(tmp_path: Path) -> None:
    p = tmp_path / "J0000+0000.par"
    _write(
        p,
        "\n".join(
            [
                "RAJ 12:00:00 1 0.0001",
                "DECJ +01:00:00 1 0.0002",
                "F0 100.0 1 1e-9",
                "JUMP -sys EFF.P200.1380 0 1",
            ]
        )
        + "\n",
    )
    df = _parse_parfile(p)
    assert "JUMP" not in set(df["param"].tolist())
    assert "RAJ" in set(df["param"].tolist())
    f0 = df[df["param"] == "F0"].iloc[0]
    assert str(f0["value_raw"]) == "100.0"
    assert float(f0["error_raw"]) == 1e-9


def test_parse_parfile_canonicalizes_pulsar_name_from_provider_suffixes(
    tmp_path: Path,
) -> None:
    p = tmp_path / "J1909-3744.nb.par"
    _write(p, "F0 100.0 1 1e-9\n")
    df = _parse_parfile(p)
    assert set(df["pulsar"].tolist()) == {"J1909-3744"}


def test_parse_parfile_uses_aliases_for_b_name(tmp_path: Path) -> None:
    p = tmp_path / "B1855+09.IPTADR2.par"
    _write(p, "PSRB B1855+09\nF0 100.0 1 1e-9\n")
    df = _parse_parfile(p, aliases={"B1855+09": "J1857+0943"})
    assert set(df["pulsar"].tolist()) == {"J1857+0943"}


def test_normalize_astrometry_converts_to_icrs_degrees() -> None:
    df = pd.DataFrame(
        [
            {
                "provider": "TEST",
                "pulsar": "J0000+0000",
                "param": "RAJ",
                "value_raw": "12:00:00",
                "error_raw": 0.1,
                "parfile": "/tmp/a.par",
            },
            {
                "provider": "TEST",
                "pulsar": "J0000+0000",
                "param": "DECJ",
                "value_raw": "+00:00:00",
                "error_raw": 0.1,
                "parfile": "/tmp/a.par",
            },
        ]
    )
    out = normalize_astrometry(df)
    raj = out[out["param"] == "RAJ"].iloc[0]
    decj = out[out["param"] == "DECJ"].iloc[0]
    assert abs(float(raj["value_num"]) - 180.0) < 1e-8
    assert abs(float(decj["value_num"]) - 0.0) < 1e-8
    assert raj["frame_norm"] == "icrs"
    assert raj["compare_param"] == "RA_ICRS_DEG"
    assert decj["compare_param"] == "DEC_ICRS_DEG"
    assert abs(float(raj["error_num"]) - (0.1 / 240.0)) < 1e-12
    assert abs(float(decj["error_num"]) - (0.1 / 3600.0)) < 1e-12


def test_build_comparison_unifies_equatorial_and_ecliptic_coordinates() -> None:
    icrs = SkyCoord(ra=180.0 * u.deg, dec=10.0 * u.deg, frame="icrs")
    ecl = icrs.transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    df = pd.DataFrame(
        [
            {
                "provider": "EQ",
                "pulsar": "J1909-3744",
                "param": "RAJ",
                "value_raw": "12:00:00",
                "error_raw": 0.0001,
                "parfile": "/tmp/eq.par",
            },
            {
                "provider": "EQ",
                "pulsar": "J1909-3744",
                "param": "DECJ",
                "value_raw": "+10:00:00",
                "error_raw": 0.0001,
                "parfile": "/tmp/eq.par",
            },
            {
                "provider": "ECL",
                "pulsar": "J1909-3744",
                "param": "ELONG",
                "value_raw": f"{float(ecl.lon.deg):.15f}",
                "error_raw": 1e-6,
                "parfile": "/tmp/ecl.par",
            },
            {
                "provider": "ECL",
                "pulsar": "J1909-3744",
                "param": "ELAT",
                "value_raw": f"{float(ecl.lat.deg):.15f}",
                "error_raw": 1e-6,
                "parfile": "/tmp/ecl.par",
            },
        ]
    )
    norm = normalize_astrometry(df)
    cmp = _build_comparison(norm)
    ra = cmp[cmp["param"] == "RA_ICRS_DEG"].iloc[0]
    dec = cmp[cmp["param"] == "DEC_ICRS_DEG"].iloc[0]
    assert int(ra["n_providers"]) == 2
    assert int(dec["n_providers"]) == 2
    assert str(ra["source_params"]) == "ELONG,RAJ"
    assert str(dec["source_params"]) == "DECJ,ELAT"
    assert abs(float(ra["span"])) < 1e-9
    assert abs(float(dec["span"])) < 1e-9
    assert str(ra["agreement_class"]) == "consistent"
    assert str(dec["agreement_class"]) == "consistent"


def test_build_comparison_aggregates_provider_spread() -> None:
    norm = pd.DataFrame(
        [
            {
                "provider": "A",
                "pulsar": "J1",
                "param": "F0",
                "value_num": 1.0,
                "error_raw": 0.1,
            },
            {
                "provider": "B",
                "pulsar": "J1",
                "param": "F0",
                "value_num": 1.2,
                "error_raw": 0.2,
            },
        ]
    )
    cmp = _build_comparison(norm)
    row = cmp.iloc[0]
    assert int(row["n_providers"]) == 2
    assert abs(float(row["span"]) - 0.2) < 1e-12
    assert str(row["source_params"]) == "F0"
    assert int(row["n_numeric_values"]) == 2
    assert int(row["n_error_values"]) == 2
    assert abs(float(row["sigma_tension_max"]) - 0.8944271909999157) < 1e-12
    assert abs(float(row["sigma_tension_mean"]) - 0.8944271909999157) < 1e-12
    assert abs(float(row["reduced_chi2"]) - 0.8) < 1e-12
    assert str(row["worst_provider_pair"]) == "A vs B"
    assert str(row["agreement_class"]) == "consistent"


def test_load_provider_specs_reads_exact_source_fields(tmp_path: Path) -> None:
    providers = tmp_path / "providers.toml"
    _write(
        providers,
        "\n".join(
            [
                "[providers.NANOGrav]",
                'type = "zenodo_record"',
                'url = "https://zenodo.org/records/8423265"',
                'asset_regex = "^15yr_BayesianTiming\\\\.tgz$"',
                'par_glob = "narrowband/par/*.nb.par"',
                "",
                "[providers.IPTA]",
                'type = "git_clone"',
                'url = "https://gitlab.com/IPTA/DR2.git"',
                'ref = "main"',
                'par_glob = "IPTADR2/release/VersionA/J*/J*.IPTADR2.par"',
            ]
        )
        + "\n",
    )
    specs = load_provider_specs(providers)
    assert len(specs) == 2
    nanograv = next(s for s in specs if s.name == "NANOGrav")
    assert nanograv.type == "zenodo_record"
    assert nanograv.url == "https://zenodo.org/records/8423265"
    assert nanograv.par_glob == "narrowband/par/*.nb.par"
    ipta = next(s for s in specs if s.name == "IPTA")
    assert ipta.type == "git_clone"
    assert ipta.url == "https://gitlab.com/IPTA/DR2.git"
    assert ipta.ref == "main"
    assert ipta.par_glob == "IPTADR2/release/VersionA/J*/J*.IPTADR2.par"


def test_parse_zenodo_record_id_accepts_record_urls() -> None:
    assert _parse_zenodo_record_id("8423265") == "8423265"
    assert _parse_zenodo_record_id("https://zenodo.org/records/8423265") == "8423265"
    assert _parse_zenodo_record_id("doi:10.5281/zenodo.8300645") == "8300645"
    assert _parse_zenodo_record_id("") is None


def test_resolve_latest_assets_for_zenodo_record(monkeypatch) -> None:
    def fake_get_json(url: str) -> dict:
        assert url == "https://zenodo.org/api/records/8423265"
        return {
            "id": 8423265,
            "metadata": {"title": "NANOGrav 15-year"},
            "files": [
                {
                    "key": "README.txt",
                    "links": {"self": "https://example.test/readme"},
                },
                {
                    "key": "15yr_BayesianTiming.tgz",
                    "links": {"self": "https://example.test/15yr_BayesianTiming.tgz"},
                },
            ],
        }

    monkeypatch.setattr("pleb.public_release_compare._http_get_json", fake_get_json)
    assets = resolve_latest_assets(
        ProviderSpec(
            name="NANOGrav",
            type="zenodo_record",
            url="https://zenodo.org/records/8423265",
            asset_regex=r"^15yr_BayesianTiming\.tgz$",
        )
    )
    assert len(assets) == 1
    asset = assets[0]
    assert asset.provider == "NANOGrav"
    assert asset.release_id == "8423265"
    assert asset.filename == "15yr_BayesianTiming.tgz"
    assert asset.url == "https://example.test/15yr_BayesianTiming.tgz"
    assert asset.par_glob == "**/*.par"


def test_resolve_latest_assets_for_git_clone() -> None:
    assets = resolve_latest_assets(
        ProviderSpec(
            name="IPTA",
            type="git_clone",
            url="https://gitlab.com/IPTA/DR2.git",
            ref="main",
        )
    )
    assert len(assets) == 1
    asset = assets[0]
    assert asset.provider == "IPTA"
    assert asset.asset_type == "git_clone"
    assert asset.url == "https://gitlab.com/IPTA/DR2.git"
    assert asset.ref == "main"
    assert asset.release_id == "main"
    assert asset.par_glob == "**/*.par"


def test_resolve_latest_assets_for_direct_url() -> None:
    assets = resolve_latest_assets(
        ProviderSpec(
            name="NANOGrav",
            type="direct_url",
            url=(
                "https://zenodo.org/records/16051178/files/"
                "NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz?download=1"
            ),
            asset_regex=r"^NANOGrav15yr_PulsarTiming_v2\.1\.0\.tar\.gz$",
        )
    )
    assert len(assets) == 1
    asset = assets[0]
    assert asset.provider == "NANOGrav"
    assert asset.asset_type == "download"
    assert asset.filename == "NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz"
    assert asset.release_id == "16051178"


def test_find_par_files_uses_provider_specific_glob(tmp_path: Path) -> None:
    keep = (
        tmp_path
        / "NANOGrav15yr_PulsarTiming_v2.1.0"
        / "narrowband"
        / "par"
        / "J1909-3744.nb.par"
    )
    skip = (
        tmp_path
        / "NANOGrav15yr_PulsarTiming_v2.1.0"
        / "wideband"
        / "par"
        / "J1909-3744.wb.par"
    )
    _write(keep, "F0 1\n")
    _write(skip, "F0 2\n")
    found = _find_par_files(
        tmp_path, "NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/par/*.nb.par"
    )
    assert found == [keep]


def test_download_and_extract_assets_uses_git_clone_par_glob(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "fake_repo"
    keep = (
        repo_root
        / "IPTADR2"
        / "release"
        / "VersionA"
        / "J1909-3744"
        / "J1909-3744.IPTADR2.par"
    )
    skip = repo_root / "other" / "J1909-3744.par"
    _write(keep, "F0 1\n")
    _write(skip, "F0 2\n")

    def fake_clone(url: str, dst: Path, *, ref: str | None = None) -> Path:
        return repo_root

    monkeypatch.setattr("pleb.public_release_compare._clone_git_repo", fake_clone)
    out = _download_and_extract_assets(
        [
            ReleaseAsset(
                provider="IPTA",
                release_id="main",
                title="DR2",
                url="https://gitlab.com/IPTA/DR2.git",
                filename="DR2",
                asset_type="git_clone",
                ref="main",
                par_glob="IPTADR2/release/VersionA/J*/J*.IPTADR2.par",
            )
        ],
        tmp_path / "cache",
    )
    assert out["IPTA"] == [keep]


def test_download_and_extract_assets_reuses_cached_downloads_and_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache"
    extracted_par = (
        cache_root / "TEST" / "test-release" / "extracted" / "pars" / "J1909-3744.par"
    )
    archive = cache_root / "TEST" / "test-release" / "release.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"cached")
    _write(extracted_par, "F0 1\n")

    def fail_download(url: str, dst: Path) -> Path:
        raise AssertionError("download should not run when cached asset exists")

    def fail_extract(asset_file: Path, out_dir: Path) -> Path:
        raise AssertionError("extract should not run when extracted cache exists")

    monkeypatch.setattr("pleb.public_release_compare._http_download", fail_download)
    monkeypatch.setattr("pleb.public_release_compare._extract_archive", fail_extract)
    out = _download_and_extract_assets(
        [
            ReleaseAsset(
                provider="TEST",
                release_id="test-release",
                title="Test Release",
                url="https://example.test/release.tar.gz",
                filename="release.tar.gz",
                par_glob="pars/*.par",
            )
        ],
        cache_root,
    )
    assert out["TEST"] == [extracted_par]


def test_compare_public_releases_writes_assets_tsv_for_slots_dataclass(
    tmp_path: Path, monkeypatch
) -> None:
    providers = tmp_path / "providers.toml"
    _write(
        providers,
        "\n".join(
            [
                "[providers.TEST]",
                'type = "direct_url"',
                'url = "https://example.test/release.tar.gz"',
            ]
        )
        + "\n",
    )
    par = tmp_path / "pars_a" / "J1909-3744.par"
    _write(par, "RAJ 12:00:00 1 0.1\nDECJ -37:44:00 1 0.2\nF0 100.0 1 1e-9\n")

    par_b = tmp_path / "pars_b" / "J1909-3744.par"
    _write(par_b, "RAJ 12:00:00 1 0.1\nDECJ -37:44:00 1 0.2\nF0 100.2 1 1e-9\n")

    def fake_resolve(spec: ProviderSpec) -> list[ReleaseAsset]:
        return [
            ReleaseAsset(
                provider="TEST_A",
                release_id="test-release-a",
                title="Test Release A",
                url="https://example.test/release_a.tar.gz",
                filename="release_a.tar.gz",
                par_glob="pars/*.par",
            ),
            ReleaseAsset(
                provider="TEST_B",
                release_id="test-release-b",
                title="Test Release B",
                url="https://example.test/release_b.tar.gz",
                filename="release_b.tar.gz",
                par_glob="pars/*.par",
            ),
        ]

    def fake_download_extract(
        assets: list[ReleaseAsset], cache_root: Path
    ) -> dict[str, list[Path]]:
        return {"TEST_A": [par], "TEST_B": [par_b]}

    monkeypatch.setattr(
        "pleb.public_release_compare.resolve_latest_assets", fake_resolve
    )
    monkeypatch.setattr(
        "pleb.public_release_compare._download_and_extract_assets",
        fake_download_extract,
    )

    out = compare_public_releases(tmp_path / "out", providers)
    assets_df = pd.read_csv(out["assets"], sep="\t")
    assert list(assets_df["provider"]) == ["TEST_A", "TEST_B"]
    assert list(assets_df["release_id"]) == ["test-release-a", "test-release-b"]
    assert list(assets_df["par_glob"]) == ["pars/*.par", "pars/*.par"]
    cmp_df = pd.read_csv(out["comparison"], sep="\t")
    f0 = cmp_df[cmp_df["param"] == "F0"].iloc[0]
    assert float(f0["sigma_tension_max"]) > 1.0e7
    assert str(f0["agreement_class"]) == "strong_tension"
    assert set(cmp_df["param"].tolist()) >= {"RA_ICRS_DEG", "DEC_ICRS_DEG", "F0"}
    summary_df = pd.read_csv(out["summary"], sep="\t")
    assert "worst_sigma_tension" in summary_df.columns
    report_text = Path(out["report"]).read_text(encoding="utf-8")
    assert "Figure of merit" in report_text
    assert "Worst Disagreements Overall" in report_text
    assert "RAJ/ELONG -> RA_ICRS_DEG" in report_text


def test_compare_public_releases_uses_shared_cache_dir(
    tmp_path: Path, monkeypatch
) -> None:
    providers = tmp_path / "providers.toml"
    _write(
        providers,
        "\n".join(
            [
                "[providers.TEST]",
                'type = "direct_url"',
                'url = "https://example.test/release.tar.gz"',
            ]
        )
        + "\n",
    )
    par = tmp_path / "pars" / "J1909-3744.par"
    _write(par, "F0 100.0 1 1e-9\n")
    shared_cache = tmp_path / "shared_cache"
    captured: dict[str, Path] = {}

    def fake_resolve(spec: ProviderSpec) -> list[ReleaseAsset]:
        return [
            ReleaseAsset(
                provider="TEST",
                release_id="test-release",
                title="Test Release",
                url="https://example.test/release.tar.gz",
                filename="release.tar.gz",
                par_glob="pars/*.par",
            )
        ]

    def fake_download_extract(
        assets: list[ReleaseAsset], cache_root: Path
    ) -> dict[str, list[Path]]:
        captured["cache_root"] = Path(cache_root)
        return {"TEST": [par]}

    monkeypatch.setattr(
        "pleb.public_release_compare.resolve_latest_assets", fake_resolve
    )
    monkeypatch.setattr(
        "pleb.public_release_compare._download_and_extract_assets",
        fake_download_extract,
    )
    monkeypatch.setattr(
        "pleb.public_release_compare._default_public_release_cache_dir",
        lambda out_dir: shared_cache,
    )

    out = compare_public_releases(tmp_path / "out", providers)
    assert captured["cache_root"] == shared_cache.resolve()
    assert out["cache_dir"] == shared_cache.resolve()


def test_compare_public_releases_writes_local_variant_vs_public_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    providers = tmp_path / "providers.toml"
    _write(
        providers,
        "\n".join(
            [
                "[providers.TEST]",
                'type = "direct_url"',
                'url = "https://example.test/release.tar.gz"',
            ]
        )
        + "\n",
    )
    public_a = tmp_path / "pars_public_a" / "J1909-3744.nb.par"
    _write(public_a, "F0 100.0 1 1e-9\n")
    public_b = tmp_path / "pars_public_b" / "J1909-3744.nb.par"
    _write(public_b, "F0 100.1 1 1e-9\n")

    ds_root = tmp_path / "dataset"
    psr_dir = ds_root / "J1909-3744"
    psr_dir.mkdir(parents=True)
    _write(psr_dir / "J1909-3744.par", "F0 100.05 1 1e-9\n")
    _write(psr_dir / "J1909-3744_new.par", "F0 100.2 1 1e-9\n")

    def fake_resolve(spec: ProviderSpec) -> list[ReleaseAsset]:
        return [
            ReleaseAsset(
                provider="TEST_A",
                release_id="test-release-a",
                title="Test Release A",
                url="https://example.test/release_a.tar.gz",
                filename="release_a.tar.gz",
                par_glob="pars/*.par",
            ),
            ReleaseAsset(
                provider="TEST_B",
                release_id="test-release-b",
                title="Test Release B",
                url="https://example.test/release_b.tar.gz",
                filename="release_b.tar.gz",
                par_glob="pars/*.par",
            ),
        ]

    def fake_download_extract(
        assets: list[ReleaseAsset], cache_root: Path
    ) -> dict[str, list[Path]]:
        return {"TEST_A": [public_a], "TEST_B": [public_b]}

    monkeypatch.setattr(
        "pleb.public_release_compare.resolve_latest_assets", fake_resolve
    )
    monkeypatch.setattr(
        "pleb.public_release_compare._download_and_extract_assets",
        fake_download_extract,
    )

    out = compare_public_releases(
        tmp_path / "out",
        providers,
        local_dataset_root=ds_root,
        local_pulsars=["J1909-3744"],
    )

    local_pair_df = pd.read_csv(out["local_vs_public"], sep="\t")
    assert set(local_pair_df["local_variant"].astype(str)) == {"base", "new"}
    assert set(local_pair_df["public_provider"].astype(str)) == {"TEST_A", "TEST_B"}
    assert set(local_pair_df["local_source"].astype(str)) == {
        "PLEB_LOCAL[base]",
        "PLEB_LOCAL[new]",
    }

    local_summary_df = pd.read_csv(out["local_vs_public_summary"], sep="\t")
    assert set(local_summary_df["local_variant"].astype(str)) == {"base", "new"}
    assert set(local_summary_df["public_provider"].astype(str)) == {"TEST_A", "TEST_B"}

    report_text = Path(out["report"]).read_text(encoding="utf-8")
    assert "Local Variants vs Public Releases" in report_text
    assert "Local Variant Summary" in report_text


def test_compare_public_releases_reads_local_variants_from_branch(
    monkeypatch, tmp_path: Path
) -> None:
    providers = tmp_path / "providers.toml"
    _write(
        providers,
        """
[providers.TEST_A]
type = "direct_url"
url = "https://example.test/release_a.tar.gz"
asset_regex = ".*"
par_glob = "pars/*.par"
""".strip() + "\n",
    )

    public_a = tmp_path / "pars_public_a" / "J1909-3744.par"
    _write(public_a, "F0 100.0 1 1e-9\n")

    def fake_resolve(spec: ProviderSpec) -> list[ReleaseAsset]:
        return [
            ReleaseAsset(
                provider="TEST_A",
                release_id="test-release-a",
                title="Test Release A",
                url="https://example.test/release_a.tar.gz",
                filename="release_a.tar.gz",
                par_glob="pars/*.par",
            )
        ]

    def fake_download_extract(
        assets: list[ReleaseAsset], cache_root: Path
    ) -> dict[str, list[Path]]:
        return {"TEST_A": [public_a]}

    monkeypatch.setattr(
        "pleb.public_release_compare.resolve_latest_assets", fake_resolve
    )
    monkeypatch.setattr(
        "pleb.public_release_compare._download_and_extract_assets",
        fake_download_extract,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-b", "main")
    ds_root = repo_root / "dataset"
    psr_dir = ds_root / "J1909-3744"
    psr_dir.mkdir(parents=True)
    _write(psr_dir / "J1909-3744.par", "F0 9.0 1 1e-9\n")
    _git(repo_root, "add", ".")
    _git(
        repo_root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "main",
    )

    _git(repo_root, "checkout", "-b", "scanbranch")
    _write(psr_dir / "J1909-3744.par", "F0 100.05 1 1e-9\n")
    _write(psr_dir / "J1909-3744_new.par", "F0 100.2 1 1e-9\n")
    _git(repo_root, "add", ".")
    _git(
        repo_root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "scanbranch",
    )

    _git(repo_root, "checkout", "main")
    # Dirty live worktree with different values to prove local_branch ignores it.
    _write(psr_dir / "J1909-3744.par", "F0 999.0 1 1e-9\n")

    out = compare_public_releases(
        tmp_path / "out_branch",
        providers,
        local_dataset_root=ds_root,
        local_branch="scanbranch",
        local_pulsars=["J1909-3744"],
    )

    local_pair_df = pd.read_csv(out["local_vs_public"], sep="\t")
    assert set(local_pair_df["local_variant"].astype(str)) == {"base", "new"}
    assert set(local_pair_df["local_source"].astype(str)) == {
        "PLEB_LOCAL[base]",
        "PLEB_LOCAL[new]",
    }

    raw_df = pd.read_csv(out["raw"], sep="\t")
    local_rows = raw_df[raw_df["provider"] == "PLEB_LOCAL"].copy()
    vals = set(local_rows[local_rows["param"] == "F0"]["value_raw"].astype(str))
    assert "100.05" in vals
    assert "100.2" in vals
    assert "999.0" not in vals
    assert any(
        str(parfile).startswith("scanbranch:dataset/J1909-3744/")
        for parfile in local_rows["parfile"].astype(str)
    )
