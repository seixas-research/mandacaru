"""``mandacaru-build`` (:mod:`mandacaru.pseudopotentials.build_cli`)."""
import pytest

from mandacaru.pseudopotentials.build_cli import FAMILIES, build_parser, main
from mandacaru.pseudopotentials.io import load_pseudopotential


@pytest.fixture(autouse=True)
def _outside_the_repository(tmp_path, monkeypatch):
    """Every test runs from a temporary directory: without ``--output``,
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
    ("ONCV", "oncvpsp"), ("ONCVPSP", "oncvpsp"), ("NCPP", "ncpp"), ("TM", "ncpp")])
def test_family_spellings(spelling, family):
    assert FAMILIES[spelling.lower()] == family


def test_ncpp_refuses_what_its_generator_cannot_do(capsys):
    with pytest.raises(SystemExit):
        main(["--pp", "NCPP", "--dirac", "--element", "H"])
    assert "no spin-orbit term" in capsys.readouterr().err


def test_an_unknown_element_is_a_usage_error(capsys):
    with pytest.raises(SystemExit):
        main(["--pp", "PAW", "--element", "Xx"])
    assert "unknown element" in capsys.readouterr().err


def test_install_and_output_are_exclusive(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--element", "H", "--install", "--output", str(tmp_path)])
    assert "exclusive" in capsys.readouterr().err


@pytest.mark.parametrize("pp, family", [("PAW", "paw-lcao"),
                                        ("ONCV", "oncvpsp")])
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
    from mandacaru.pseudopotentials import build_cli

    def broken(family, symbol, options):
        raise RuntimeError("no bound state")
    monkeypatch.setattr(build_cli, "_generate", broken)
    code = main(["--pp", "PAW", "--element", "H", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "H  FAILED" in out and "RuntimeError: no bound state" in out
    assert "0 of 1 dataset(s) written, 1 failed" in out


def test_a_generator_warning_reaches_the_report(tmp_path, capsys, monkeypatch):
    import warnings

    from mandacaru.pseudopotentials import build_cli
    real = build_cli._generate

    def noisy(family, symbol, options):
        warnings.warn("cutoff beyond the trustworthy range")
        warnings.warn("overflow", RuntimeWarning)
        return real(family, symbol, options)
    monkeypatch.setattr(build_cli, "_generate", noisy)
    code = main(["--pp", "PAW", "--element", "H", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "warning: cutoff beyond the trustworthy range" in out
    assert "overflow" not in out


def test_install_needs_the_library_variable(monkeypatch, capsys):
    monkeypatch.delenv("MANDACARU_PAW_PATH", raising=False)
    assert main(["--pp", "PAW", "--element", "H", "--install"]) == 2
    assert "mandacaru --set-paw" in capsys.readouterr().err


def test_install_writes_into_the_functional_folder(tmp_path, monkeypatch,
                                                   capsys):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(tmp_path))
    assert main(["--pp", "PAW", "--element", "H", "--install"]) == 0
    assert (tmp_path / "lda" / "H.parquet").is_file()
