# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_environment.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Library locations from ``MANDACARU_{NCPP,ONCVPSP,PAW}_PATH``
(:mod:`mandacaru.pseudopotentials.environment`)."""

import os

import pytest

from mandacaru.pseudopotentials import environment
from mandacaru.pseudopotentials.environment import (
    FAMILY_VARIABLES, LibraryPathError, library_directory, repository_path,
    status_lines, upaw_directory)


@pytest.fixture
def checkout(tmp_path):
    """A fake repository checkout with an ``lda/`` folder holding H."""
    (tmp_path / "lda").mkdir()
    (tmp_path / "lda" / "H.parquet").write_bytes(b"")
    return tmp_path


@pytest.mark.parametrize("family, flag", [("paw-lcao", "--set-paw"),
                                          ("oncvpsp", "--set-oncvpsp"),
                                          ("ncpp", "--set-ncpp"),
                                          ("upaw-lcao", "--set-upaw")])
def test_an_unset_variable_names_the_command_that_sets_it(monkeypatch, family,
                                                          flag):
    monkeypatch.delenv(FAMILY_VARIABLES[family], raising=False)
    with pytest.raises(LibraryPathError, match=f"mandacaru {flag}") as error:
        library_directory(family)
    assert f"{FAMILY_VARIABLES[family]} is not set" in str(error.value)


def test_an_alias_resolves_to_its_family(monkeypatch, checkout):
    monkeypatch.setenv("MANDACARU_ONCVPSP_PATH", str(checkout))
    assert library_directory("oncv") == os.path.join(str(checkout), "lda")


def test_a_path_that_is_not_a_directory_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(tmp_path / "nowhere"))
    with pytest.raises(LibraryPathError, match="is not a directory"):
        library_directory("paw-lcao")


def test_the_functional_picks_the_subdirectory(monkeypatch, checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(checkout))
    assert library_directory("paw-lcao") == os.path.join(str(checkout), "lda")
    with pytest.raises(LibraryPathError, match="has no pbe/ directory"):
        library_directory("paw-lcao", "pbe")
    # For writing, a missing functional folder is simply where to create it.
    assert library_directory("paw-lcao", "PBE", must_exist=False) == \
        os.path.join(str(checkout), "pbe")
    with pytest.raises(ValueError, match="xc must be one of"):
        library_directory("paw-lcao", "b3lyp")


def test_the_variable_is_read_when_it_is_needed(monkeypatch, checkout,
                                                tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    (other / "lda").mkdir()
    monkeypatch.setenv("MANDACARU_NCPP_PATH", str(checkout))
    first = library_directory("ncpp")
    monkeypatch.setenv("MANDACARU_NCPP_PATH", str(other))
    assert library_directory("ncpp") != first


def test_a_directory_option_bypasses_the_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("MANDACARU_PAW_PATH", raising=False)
    assert library_directory("paw-lcao", directory=tmp_path) == str(tmp_path)


def test_a_user_path_is_expanded(monkeypatch, checkout):
    monkeypatch.setenv("HOME", str(checkout.parent))
    monkeypatch.setenv("MANDACARU_PAW_PATH", f"~/{checkout.name}")
    assert repository_path("paw-lcao") == str(checkout)


def test_upaw_is_generated_on_demand_without_the_variable(monkeypatch,
                                                         checkout, tmp_path_factory):
    monkeypatch.delenv("MANDACARU_UPAW_PATH", raising=False)
    assert upaw_directory() is None
    monkeypatch.setenv("MANDACARU_UPAW_PATH", str(checkout))
    assert upaw_directory() == os.path.join(str(checkout), "lda")
    # Optional does not mean unchecked: a variable that names no directory
    # is still an error.
    monkeypatch.setenv("MANDACARU_UPAW_PATH",
                       str(tmp_path_factory.mktemp("u") / "missing"))
    with pytest.raises(LibraryPathError, match="--set-upaw"):
        upaw_directory()


def test_status_reports_every_family(monkeypatch, checkout, tmp_path_factory):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(checkout))
    monkeypatch.delenv("MANDACARU_NCPP_PATH", raising=False)
    monkeypatch.setenv("MANDACARU_ONCVPSP_PATH",
                       str(tmp_path_factory.mktemp("x") / "missing"))
    lines = {line.split()[0]: line for line in status_lines()}
    assert "lda/: 1 datasets" in lines["paw-lcao"]
    assert "not set" in lines["ncpp"] and "--set-ncpp" in lines["ncpp"]
    assert "NOT A DIRECTORY" in lines["oncvpsp"]


def test_a_loader_refuses_before_any_file_is_read(monkeypatch):
    from mandacaru.pseudopotentials import get_oncv, get_paw, get_pseudopotential

    for variable in FAMILY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    # UPAW-LCAO is optional: without its variable it is generated on demand,
    # so only the three required families are asked to refuse here.
    for load, flag in ((get_paw, "--set-paw"), (get_oncv, "--set-oncvpsp"),
                       (get_pseudopotential, "--set-ncpp")):
        with pytest.raises(LibraryPathError, match=flag):
            load("H")


def test_a_calculation_stops_before_it_starts(monkeypatch):
    """The dataset is loaded when the Hamiltonian is built, before any
    integral: an unset variable stops the run there, with the fix."""
    from ase import Atoms

    from mandacaru import Mandacaru

    monkeypatch.delenv("MANDACARU_PAW_PATH", raising=False)
    atoms = Atoms("H2", positions=[(3.0, 3.0, 2.63), (3.0, 3.0, 3.37)],
                  cell=[6.0, 6.0, 6.0])
    atoms.calc = Mandacaru(basis={"name": "PAW-LCAO"})
    with pytest.raises(LibraryPathError, match="--set-paw"):
        atoms.get_potential_energy()


def test_no_library_ships_inside_the_package():
    package = os.path.dirname(environment.__file__)
    assert not os.path.exists(os.path.join(package, "library"))
