# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_environment.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Library locations from ``MANDACARU_{PAW,UPAW}_PATH``
(:mod:`mandacaru.pseudopotentials.environment`)."""

import os

import pytest

from mandacaru.pseudopotentials import environment
from mandacaru.pseudopotentials.environment import (
    FAMILY_VARIABLES, LibraryPathError, library_directory, library_folder,
    repository_path, status_lines, upaw_directory)


@pytest.fixture
def checkout(tmp_path):
    """A fake repository checkout with an ``lda-sr/`` folder holding H."""
    (tmp_path / "lda-sr").mkdir()
    (tmp_path / "lda-sr" / "H.parquet").write_bytes(b"")
    return tmp_path


@pytest.mark.parametrize("family, flag", [("paw-lcao", "--set-paw"),
                                          ("upaw-lcao", "--set-upaw")])
def test_an_unset_variable_names_the_command_that_sets_it(monkeypatch, family,
                                                          flag):
    monkeypatch.delenv(FAMILY_VARIABLES[family], raising=False)
    with pytest.raises(LibraryPathError, match=f"mandacaru {flag}") as error:
        library_directory(family)
    assert f"{FAMILY_VARIABLES[family]} is not set" in str(error.value)


def test_names_are_case_insensitive(monkeypatch, checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(checkout))
    assert library_directory("PAW-LCAO") == os.path.join(str(checkout), "lda-sr")


def test_a_path_that_is_not_a_directory_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(tmp_path / "nowhere"))
    with pytest.raises(LibraryPathError, match="is not a directory"):
        library_directory("paw-lcao")


def test_the_functional_picks_the_subdirectory(monkeypatch, checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(checkout))
    assert library_directory("paw-lcao") == os.path.join(str(checkout),
                                                         "lda-sr")
    with pytest.raises(LibraryPathError,
                       match=r"has no pbe/ folder \(it has: lda-sr\)"):
        library_directory("paw-lcao", "pbe")
    # For writing, a missing functional folder is simply where to create it.
    assert library_directory("paw-lcao", "PBE", must_exist=False) == \
        os.path.join(str(checkout), "pbe")
    with pytest.raises(ValueError, match="xc must be one of"):
        library_directory("paw-lcao", "b3lyp")


def test_paw_ships_a_scalar_and_a_dirac_lda_set(monkeypatch, paw_checkout):
    """PAW-LCAO's folders are named for the set, not the functional: the
    scalar-relativistic ``lda-sr/`` is the default and ``lda-dirac/`` carries
    the spin-orbit term.  There is no ``lda/`` alias."""
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    assert library_directory("paw-lcao") == str(paw_checkout / "lda-sr")
    assert library_directory("paw-lcao", relativity="dirac") == \
        str(paw_checkout / "lda-dirac")
    (paw_checkout / "lda").mkdir()
    assert library_directory("paw-lcao") == str(paw_checkout / "lda-sr")


def test_the_variable_is_read_when_it_is_needed(monkeypatch, checkout,
                                                tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    (other / "lda-sr").mkdir()
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(checkout))
    first = library_directory("paw-lcao")
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(other))
    assert library_directory("paw-lcao") != first


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
    # UPAW-LCAO keeps one folder per functional; the scalar-relativistic
    # ``lda-sr/`` naming is for the shipped library.
    # Optional does not mean unchecked: a variable that names no directory
    # is still an error.
    monkeypatch.setenv("MANDACARU_UPAW_PATH",
                       str(tmp_path_factory.mktemp("u") / "missing"))
    with pytest.raises(LibraryPathError, match="--set-upaw"):
        upaw_directory()


def test_status_reports_every_family(monkeypatch, paw_checkout,
                                    tmp_path_factory):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    monkeypatch.setenv("MANDACARU_UPAW_PATH",
                       str(tmp_path_factory.mktemp("x") / "missing"))
    lines = {line.split()[0]: line for line in status_lines()}
    assert set(lines) == {"paw-lcao", "upaw-lcao"}
    assert "lda-sr/: 1 datasets" in lines["paw-lcao"]
    assert "lda-dirac/: 1 datasets" in lines["paw-lcao"]
    assert "NOT A DIRECTORY" in lines["upaw-lcao"]
    monkeypatch.delenv("MANDACARU_PAW_PATH")
    lines = {line.split()[0]: line for line in status_lines()}
    assert "not set" in lines["paw-lcao"] and "--set-paw" in lines["paw-lcao"]


def test_a_loader_refuses_before_any_file_is_read(monkeypatch):
    from mandacaru.pseudopotentials import get_paw

    for variable in FAMILY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    # UPAW-LCAO is optional: without its variable it is generated on demand,
    # so only PAW-LCAO is asked to refuse here.
    with pytest.raises(LibraryPathError, match="--set-paw"):
        get_paw("H")


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


# --------------------------------------------------------------------------- #
# Mandacaru(directory=...): the library folder a calculation reads.
# --------------------------------------------------------------------------- #

@pytest.fixture
def paw_checkout(tmp_path):
    """A fake PAW-LCAO checkout: ``lda-sr/`` and ``lda-dirac/``, each with H."""
    for name in ("lda-sr", "lda-dirac"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "H.parquet").write_bytes(b"")
    return tmp_path


def test_a_folder_is_resolved_inside_the_checkout(monkeypatch, paw_checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    assert library_folder("PAW-LCAO", "lda-dirac") == \
        str(paw_checkout / "lda-dirac")
    assert library_folder("paw-lcao", "lda-sr") == str(paw_checkout / "lda-sr")


@pytest.mark.parametrize("folder", ["../lda-sr", "/tmp", "lda-sr/lda-dirac", "..", ""])
def test_a_folder_is_a_name_not_a_path(monkeypatch, paw_checkout, folder):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    with pytest.raises(ValueError, match="name of one folder"):
        library_folder("paw-lcao", folder)


def test_a_missing_folder_lists_the_ones_present(monkeypatch, paw_checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    with pytest.raises(LibraryPathError, match=r"no dirac/ folder \(it has: lda-dirac, lda-sr\)"):
        library_folder("paw-lcao", "dirac")


def test_the_calculator_points_the_basis_at_the_folder(monkeypatch,
                                                       paw_checkout):
    from mandacaru import Mandacaru

    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                     directory="lda-dirac")
    assert calc.basis == {"name": "PAW-LCAO", "size": "SZ",
                          "directory": str(paw_checkout / "lda-dirac")}
    assert calc.library_folder == "lda-dirac"
    assert calc.directory == "."        # ASE's working directory, untouched


def test_the_default_folder_leaves_the_basis_alone(monkeypatch, paw_checkout):
    from mandacaru import Mandacaru

    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    for kwargs in ({}, {"directory": "lda-sr"}):
        calc = Mandacaru(method="rhf", basis="PAW-LCAO", **kwargs)
        assert calc.basis == "PAW-LCAO"


def test_an_explicit_basis_directory_wins(monkeypatch, paw_checkout, tmp_path):
    from mandacaru import Mandacaru

    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    own = {"name": "PAW-LCAO", "directory": str(tmp_path)}
    assert Mandacaru(method="rhf", basis=own, directory="lda-dirac").basis == own


def test_a_per_element_basis_rewrites_only_its_library_entries(
        monkeypatch, paw_checkout):
    from mandacaru import Mandacaru

    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    calc = Mandacaru(method="rhf", basis={"H": "PAW-LCAO", "*": "HAO"},
                     directory="lda-dirac")
    assert calc.basis == {"H": {"name": "PAW-LCAO",
                                "directory": str(paw_checkout / "lda-dirac")},
                          "*": "HAO"}


def test_a_folder_needs_a_library_basis():
    from mandacaru import Mandacaru

    with pytest.raises(ValueError, match="reads none"):
        Mandacaru(method="rhf", basis="HAO", directory="lda-dirac")


def test_a_dataset_in_the_wrong_functional_folder_is_refused(tmp_path):
    """An LDA file copied into ``pbe/`` would otherwise run silently as the
    PBE dataset ``directory="pbe"`` asked for."""
    import shutil

    from mandacaru.pseudopotentials.paw import get_paw

    try:
        source = get_paw("H", xc="lda").source
    except (LibraryPathError, FileNotFoundError):
        pytest.skip("no PAW-LCAO library with an lda-sr/ folder")
    folder = tmp_path / "pbe"
    folder.mkdir()
    shutil.copy(source, folder / "H.parquet")
    with pytest.raises(ValueError, match="LDA dataset in the pbe/ folder"):
        get_paw("H", directory=str(folder))
