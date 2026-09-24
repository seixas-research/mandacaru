"""``mandacaru-build`` (:mod:`mandacaru.pseudopotentials.build_cli`)."""
import pytest

from mandacaru.pseudopotentials.build_cli import FAMILIES, build_parser, main
from mandacaru.pseudopotentials.io import load_pseudopotential


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
        main(["--pp", "NCPP", "--relativistic", "--element", "H"])
    assert "non-relativistic LDA only" in capsys.readouterr().err


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
