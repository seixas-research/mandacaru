"""``mandacaru-build`` (:mod:`mandacaru.pseudopotentials.build_cli`)."""
import pytest

from mandacaru.pseudopotentials.build_cli import (
    FAMILIES, build_parser, main)
from mandacaru.pseudopotentials.io import load_pseudopotential


@pytest.fixture(autouse=True)
def _outside_the_repository(tmp_path, monkeypatch):
    """Every test runs from a temporary folder: without ``--output``,
    ``mandacaru-build`` writes into ``./<family>/``, and a test that
    generates by mistake must not leave datasets in the checkout."""
    monkeypatch.chdir(tmp_path)


def test_the_example_command_parses():
    args = build_parser().parse_args(
        ["--pp", "PAW", "--relativistic", "--xc", "LDA", "--element", "Fe"])
    assert FAMILIES[args.pp.lower()] == "paw-lcao"
    assert args.relativity == "scalar" and args.xc == "lda"
    assert args.element == ["Fe"]


@pytest.mark.parametrize("spelling, family", [
    ("PAW", "paw-lcao"), ("paw-lcao", "paw-lcao"), ("UPAW", "upaw-lcao"),
    ("upaw-lcao", "upaw-lcao")])
def test_family_spellings(spelling, family):
    assert FAMILIES[spelling.lower()] == family


def test_an_unknown_family_is_a_usage_error(capsys):
    with pytest.raises(SystemExit):
        main(["--pp", "GTH", "--element", "H"])
    assert "--pp must be one of PAW, UPAW" in capsys.readouterr().err


def test_an_unknown_element_is_a_usage_error(capsys):
    with pytest.raises(SystemExit):
        main(["--pp", "PAW", "--element", "Xx"])
    assert "unknown element" in capsys.readouterr().err


def test_install_and_output_are_exclusive(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--element", "H", "--install", "--output", str(tmp_path)])
    assert "exclusive" in capsys.readouterr().err


@pytest.mark.parametrize("pp, family", [("PAW", "paw-lcao"),
                                        ("UPAW", "upaw-lcao")])
def test_it_writes_a_loadable_dataset(tmp_path, capsys, pp, family):
    code = main(["--pp", pp, "--relativistic", "--xc", "LDA", "--element",
                 "H", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "no ghost" in out and "1 of 1 dataset(s) written" in out
    pp = load_pseudopotential(str(tmp_path / family / "H.parquet"))
    assert pp.symbol == "H" and pp.relativity == "scalar"


def test_a_failed_element_is_reported_and_sets_the_exit_code(tmp_path, capsys,
                                                             monkeypatch):
    from mandacaru import _native

    def broken(context, symbol, options, path, **keywords):
        raise RuntimeError("no bound state")
    monkeypatch.setattr(_native, "paw_build", broken)
    code = main(["--pp", "PAW", "--element", "H", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "H  FAILED" in out and "RuntimeError: no bound state" in out
    assert "0 of 1 dataset(s) written, 1 failed" in out


def test_a_generator_note_reaches_the_report(tmp_path, capsys, monkeypatch):
    """The generator's observations (``Generated::notes``) are listed as warnings."""
    from mandacaru import _native
    real = _native.paw_build

    def noted(*arguments, **keywords):
        answer = real(*arguments, **keywords)
        answer["notes"] = [*answer["notes"], "cutoff beyond the trustworthy range"]
        return answer
    monkeypatch.setattr(_native, "paw_build", noted)
    code = main(["--pp", "PAW", "--element", "H", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "warning: cutoff beyond the trustworthy range" in out


def test_install_needs_the_library_variable(monkeypatch, capsys):
    monkeypatch.delenv("MANDACARU_PAW_PATH", raising=False)
    assert main(["--pp", "PAW", "--element", "H", "--install"]) == 2
    assert "mandacaru-paw --set" in capsys.readouterr().err


def test_install_writes_into_the_functional_folder(tmp_path, monkeypatch,
                                                   capsys):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(tmp_path))
    assert main(["--pp", "PAW", "--element", "H", "--install"]) == 0
    assert (tmp_path / "lda-sr" / "H.parquet").is_file()


def test_a_pbe_install_writes_into_the_pbe_set(tmp_path, monkeypatch,
                                               capsys):
    """``--xc pbe --install`` fills ``pbe-sr/``, the folder the PBE
    datasets are read from, and the file says it is PBE."""
    from mandacaru.pseudopotentials.io import load_pseudopotential

    monkeypatch.setenv("MANDACARU_PAW_PATH", str(tmp_path))
    assert main(["--pp", "PAW", "--element", "H", "--xc", "pbe",
                 "--install"]) == 0
    path = tmp_path / "pbe-sr" / "H.parquet"
    assert path.is_file() and not (tmp_path / "lda-sr").exists()
    assert load_pseudopotential(str(path)).xc == "pbe"


def _env_threads():
    import os
    return int(os.environ.get("OMP_NUM_THREADS", "0"))


def test_a_worker_shares_the_cores_out():
    """Each build worker gets its share of the physical cores, not a pool as
    wide as the machine (seven workers ran 48 threads each on 12 cores)."""
    from concurrent.futures import ProcessPoolExecutor

    from mandacaru import _native
    from mandacaru.pseudopotentials.build_cli import _limit_threads

    assert _native.worker_threads(1) == _native.grid_blas_threads() >= 1
    assert _native.worker_threads(10_000) == 1

    with ProcessPoolExecutor(max_workers=1, initializer=_limit_threads,
                             initargs=(1,)) as pool:
        assert pool.submit(_env_threads).result() == 1



def test_a_file_that_does_not_reproduce_its_dataset_is_reported_and_not_written(
        tmp_path, capsys, monkeypatch):
    """The file is read back in Rust before it is released (``paw::generate::build``; its
    removal on failure is gated in ``crates/mandacaru-core/tests/paw_generate_build.rs``): a
    read-back that moved the levels comes back as data and is reported as a failure."""
    from mandacaru import _native

    def moved(context, symbol, options, path, **keywords):
        return {"kind": "reload", "refusal": "the file read back does not reproduce the "
                "dataset: l=0 levels moved 1.0e-02 Ha (tolerance 1e-05 Ha); not written"}
    monkeypatch.setattr(_native, "paw_build", moved)
    code = main(["--pp", "PAW", "--element", "H", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "H  FAILED" in out and "does not reproduce the dataset" in out
    assert not list((tmp_path / "paw-lcao").iterdir())


def test_the_build_writes_the_record_rust_generated(tmp_path, capsys):
    """Nothing crosses the Python view on the way to the file: a cored element's file carries
    its core states (dataset format 3), which the Python view does not model."""
    from mandacaru import _native
    from mandacaru.pseudopotentials.native_context import context

    code = main(["--pp", "PAW", "--element", "Li", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    payload, tables = _native.paw_record_read(context(), str(tmp_path / "paw-lcao" / "Li.parquet"))
    import json
    meta = json.loads(payload)
    assert meta["version"] == 3
    assert [(s["n"], s["l"]) for s in meta["core_states"]] == [(1, 0)]
    assert "core_wave_n1_l0" in tables


def test_the_report_is_of_the_file_read_back(tmp_path, capsys):
    code = main(["--pp", "PAW", "--element", "H", "--output", str(tmp_path),
                 "--check"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "read back: levels within" in out and "phase" in out
    assert "gates: G1 pass, G2 pass" in out and "G13 n/a" in out
    assert [p.name for p in (tmp_path / "paw-lcao").iterdir()] == [
        "H.parquet"]


def test_a_file_that_fails_a_gate_is_refused_with_its_rows_and_not_written(
        tmp_path, capsys, monkeypatch):
    """The gates run in Rust (``paw::validate``) on the file read back, before it is moved into place
    (the refusal itself is gated in ``crates/mandacaru-core/tests/paw_validate_release.rs``): the build
    returns the refusal as data and this module prints its rows."""
    from mandacaru import _native

    def refused(context, symbol, options, path, **keywords):
        return {"kind": "gate", "gates": {"passed": False},
                "refusal": f"{symbol} (lda scalar) fails 1 gate row(s); not written:\n"
                           "  G9  FAIL         smooth-core electrons against partial_core_electrons (e): "
                           "2.137e-4 <= 1.000e-4"}
    monkeypatch.setattr(_native, "paw_build", refused)
    code = main(["--pp", "PAW", "--element", "H", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "H  FAILED" in out and "G9  FAIL" in out and "not written" in out
    assert "1 of 1 dataset(s) written" not in out
    assert not list((tmp_path / "paw-lcao").iterdir())
