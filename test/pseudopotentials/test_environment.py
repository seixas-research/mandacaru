# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_environment.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Library locations from ``MANDACARU_{PAW,UPAW}_PATH``
(:mod:`mandacaru.pseudopotentials.environment`)."""

import os
import re

import pytest

from mandacaru.pseudopotentials import environment
from mandacaru.pseudopotentials.environment import (
    FAMILY_VARIABLES, LibraryPathError, paw_dataset_folder, library_folder,
    library_status, repository_path, upaw_dataset_folder)


@pytest.fixture
def checkout(tmp_path):
    """A fake repository checkout with an ``lda-sr/`` folder holding H."""
    (tmp_path / "lda-sr").mkdir()
    (tmp_path / "lda-sr" / "H.parquet").write_bytes(b"")
    return tmp_path


@pytest.mark.parametrize("family, command", [
    ("paw-lcao", "mandacaru-paw --set"),
    ("upaw-lcao", "mandacaru-paw --set-upaw")])
def test_an_unset_variable_names_the_command_that_sets_it(monkeypatch, family,
                                                          command):
    monkeypatch.delenv(FAMILY_VARIABLES[family], raising=False)
    with pytest.raises(LibraryPathError, match=re.escape(command)) as error:
        paw_dataset_folder(family)
    assert f"{FAMILY_VARIABLES[family]} is not set" in str(error.value)


def test_names_are_case_insensitive(monkeypatch, checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(checkout))
    assert paw_dataset_folder("PAW-LCAO") == os.path.join(str(checkout), "lda-sr")


def test_a_path_that_is_not_a_folder_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(tmp_path / "nowhere"))
    with pytest.raises(LibraryPathError, match="is not a folder"):
        paw_dataset_folder("paw-lcao")


def test_the_functional_picks_the_subfolder(monkeypatch, checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(checkout))
    assert paw_dataset_folder("paw-lcao") == os.path.join(str(checkout),
                                                         "lda-sr")
    with pytest.raises(LibraryPathError,
                       match=r"has no pbe-sr/ folder \(it has: lda-sr\)"):
        paw_dataset_folder("paw-lcao", "pbe")
    # For writing, a missing set folder is simply where to create it.
    assert paw_dataset_folder("paw-lcao", "PBE", must_exist=False) == \
        os.path.join(str(checkout), "pbe-sr")
    assert paw_dataset_folder("paw-lcao", "pbe", must_exist=False,
                             relativity="dirac") == \
        os.path.join(str(checkout), "pbe-fr")
    with pytest.raises(ValueError, match="xc must be one of"):
        paw_dataset_folder("paw-lcao", "b3lyp")


def test_paw_ships_scalar_and_dirac_lda_and_pbe(monkeypatch, paw_checkout):
    """PAW-LCAO's folders are named for the set, not the functional: the
    scalar-relativistic ``lda-sr/`` is the default, ``lda-fr/`` carries
    the spin-orbit term, and ``pbe-sr/`` and ``pbe-fr/`` are the PBE sets
    without and with it.  There is no ``lda/`` or ``pbe/`` alias."""
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    assert paw_dataset_folder("paw-lcao") == str(paw_checkout / "lda-sr")
    assert paw_dataset_folder("paw-lcao", relativity="dirac") == \
        str(paw_checkout / "lda-fr")
    (paw_checkout / "pbe-sr").mkdir()
    assert paw_dataset_folder("paw-lcao", "pbe") == str(paw_checkout / "pbe-sr")
    (paw_checkout / "pbe-fr").mkdir()
    assert paw_dataset_folder("paw-lcao", "pbe", relativity="dirac") == \
        str(paw_checkout / "pbe-fr")
    (paw_checkout / "lda").mkdir()
    (paw_checkout / "pbe").mkdir()
    assert paw_dataset_folder("paw-lcao") == str(paw_checkout / "lda-sr")
    assert paw_dataset_folder("paw-lcao", "pbe") == str(paw_checkout / "pbe-sr")


def test_the_variable_is_read_when_it_is_needed(monkeypatch, checkout,
                                                tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    (other / "lda-sr").mkdir()
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(checkout))
    first = paw_dataset_folder("paw-lcao")
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(other))
    assert paw_dataset_folder("paw-lcao") != first


def test_a_paw_dataset_option_bypasses_the_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("MANDACARU_PAW_PATH", raising=False)
    assert paw_dataset_folder("paw-lcao", paw_dataset=tmp_path) == str(tmp_path)


def test_a_user_path_is_expanded(monkeypatch, checkout):
    monkeypatch.setenv("HOME", str(checkout.parent))
    monkeypatch.setenv("MANDACARU_PAW_PATH", f"~/{checkout.name}")
    assert repository_path("paw-lcao") == str(checkout)


def test_upaw_is_generated_on_demand_without_the_variable(monkeypatch,
                                                         checkout, tmp_path_factory):
    monkeypatch.delenv("MANDACARU_UPAW_PATH", raising=False)
    assert upaw_dataset_folder() is None
    monkeypatch.setenv("MANDACARU_UPAW_PATH", str(checkout))
    assert upaw_dataset_folder() == os.path.join(str(checkout), "lda-sr")
    # UPAW-LCAO keeps one folder per functional; the scalar-relativistic
    # ``lda-sr/`` naming is for the shipped library.
    # Optional does not mean unchecked: a variable that names no folder
    # is still an error.
    monkeypatch.setenv("MANDACARU_UPAW_PATH",
                       str(tmp_path_factory.mktemp("u") / "missing"))
    with pytest.raises(LibraryPathError, match="mandacaru-paw --set-upaw /path/to/mandacaru-upaw"):
        upaw_dataset_folder()


def test_status_reports_every_family(monkeypatch, paw_checkout,
                                    tmp_path_factory):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    monkeypatch.setenv("MANDACARU_UPAW_PATH",
                       str(tmp_path_factory.mktemp("x") / "missing"))
    report = {status.family: status for status in library_status()}
    assert set(report) == {"paw-lcao", "upaw-lcao"}
    paw = report["paw-lcao"]
    assert paw.state == "ok" and paw.ok
    sets = {item.name: item for item in paw.sets}
    assert sets["lda-sr"].count == 1 and sets["lda-fr"].count == 1
    assert (sets["lda-fr"].functional, sets["lda-fr"].relativity) == ("lda",
                                                                      "dirac")
    assert report["upaw-lcao"].state == "invalid"
    assert not report["upaw-lcao"].ok
    monkeypatch.delenv("MANDACARU_PAW_PATH")
    paw = {s.family: s for s in library_status()}["paw-lcao"]
    assert paw.state == "unset" and not paw.ok and not paw.optional


def test_an_unset_optional_family_is_fine(monkeypatch):
    monkeypatch.delenv("MANDACARU_UPAW_PATH", raising=False)
    upaw = {s.family: s for s in library_status()}["upaw-lcao"]
    assert upaw.state == "unset" and upaw.optional and upaw.ok


def test_a_loader_refuses_before_any_file_is_read(monkeypatch):
    from mandacaru.pseudopotentials import get_paw

    for variable in FAMILY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    # UPAW-LCAO is optional: without its variable it is generated on demand,
    # so only PAW-LCAO is asked to refuse here.
    with pytest.raises(LibraryPathError, match="mandacaru-paw --set"):
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
    with pytest.raises(LibraryPathError, match="mandacaru-paw --set"):
        atoms.get_potential_energy()


def test_no_library_ships_inside_the_package():
    package = os.path.dirname(environment.__file__)
    assert not os.path.exists(os.path.join(package, "library"))


# --------------------------------------------------------------------------- #
# Mandacaru(paw_dataset=...): the library folder a calculation reads.
# --------------------------------------------------------------------------- #

@pytest.fixture
def paw_checkout(tmp_path):
    """A fake PAW-LCAO checkout: ``lda-sr/`` and ``lda-fr/``, each with H."""
    for name in ("lda-sr", "lda-fr"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "H.parquet").write_bytes(b"")
    return tmp_path


def test_a_folder_is_resolved_inside_the_checkout(monkeypatch, paw_checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    assert library_folder("PAW-LCAO", "lda-fr") == \
        str(paw_checkout / "lda-fr")
    assert library_folder("paw-lcao", "lda-sr") == str(paw_checkout / "lda-sr")


@pytest.mark.parametrize("folder", ["../lda-sr", "/tmp", "lda-sr/lda-fr", "..", ""])
def test_a_folder_is_a_name_not_a_path(monkeypatch, paw_checkout, folder):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    with pytest.raises(ValueError, match="name of one folder"):
        library_folder("paw-lcao", folder)


def test_a_missing_folder_lists_the_ones_present(monkeypatch, paw_checkout):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw_checkout))
    with pytest.raises(LibraryPathError, match=r"no fr/ folder \(it has: lda-fr, lda-sr\)"):
        library_folder("paw-lcao", "fr")


def test_the_calculator_names_the_folder_in_the_basis(paw_checkout):
    """``Mandacaru(paw_dataset=...)`` merges the folder name into the basis;
    the loaders resolve it inside the checkout when a dataset is read."""
    from mandacaru import Mandacaru

    calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                     paw_dataset="lda-fr")
    assert calc.basis == {"name": "PAW-LCAO", "size": "SZ",
                          "paw_dataset": "lda-fr"}
    assert Mandacaru(method="rhf", basis="PAW-LCAO").basis == "PAW-LCAO"


def test_a_basis_and_a_calculator_cannot_name_two_folders():
    from mandacaru import Mandacaru

    own = {"name": "PAW-LCAO", "paw_dataset": "lda-sr"}
    assert Mandacaru(method="rhf", basis=own, paw_dataset="lda-sr").basis == own
    with pytest.raises(ValueError, match="different PAW datasets"):
        Mandacaru(method="rhf", basis=own, paw_dataset="lda-fr")


def test_a_folder_needs_a_library_basis():
    from mandacaru import Mandacaru

    with pytest.raises(ValueError, match="applies to a PAW-LCAO or UPAW-LCAO"):
        Mandacaru(method="rhf", basis="HAO", paw_dataset="lda-fr")
    # The functional's own set is chosen only where a library is read, so an
    # all-electron basis with xc="pbe" is not an error.
    assert Mandacaru(method="dft", xc="pbe", basis="HAO").basis == "HAO"


@pytest.mark.parametrize("xc, folder", [
    (None, "lda-sr"), ("lda", "lda-sr"), ("pbe", "pbe-sr"),
    ("hse06", "pbe-sr"), ("r2scan", "pbe-sr"), ("pbe+rvv10", "pbe-sr"),
    ("r2scan+rvv10", "pbe-sr")])
def test_the_functional_picks_its_own_set(xc, folder):
    """The shared selection (molecule and crystal alike): without a named
    folder, Kohn-Sham reads the set of its functional; a named folder wins;
    UPAW-LCAO has no functional sets."""
    from mandacaru.pseudopotentials.environment import (
        default_library_folder, with_functional_set)

    assert default_library_folder(xc) == folder
    assert with_functional_set("PAW-LCAO", {"size": "SZ"}, xc or "lda") == \
        {"size": "SZ", "paw_dataset": folder}
    assert with_functional_set("paw-lcao", {"paw_dataset": None},
                               xc or "lda") == {"paw_dataset": folder}
    assert with_functional_set("paw-lcao", {"paw_dataset": "lda-fr"},
                               xc or "lda") == {"paw_dataset": "lda-fr"}
    assert with_functional_set("upaw-lcao", {}, xc or "lda") == {}


# --------------------------------------------------------------------------- #
# The folder a calculation actually reads, molecule and crystal: the run is
# stopped where the datasets have just been loaded, before any SCF.
# --------------------------------------------------------------------------- #

class _Stopped(Exception):
    pass


@pytest.fixture
def real_library():
    """The shipped PAW-LCAO checkout named by ``MANDACARU_PAW_PATH``."""
    try:
        for name in ("lda-sr", "pbe-sr"):
            library_folder("paw-lcao", name)
    except LibraryPathError:
        pytest.skip("no PAW-LCAO library with lda-sr/ and pbe-sr/ folders")


def _molecule_folder(monkeypatch, **options):
    """The folder of the datasets a molecular ``Mandacaru`` loads."""
    from ase import Atoms

    from mandacaru import Mandacaru
    from mandacaru.pseudopotentials import orbitals

    seen = {}

    def stop(symbols, positions, species, **kwargs):
        seen["sources"] = {os.path.basename(os.path.dirname(d.source))
                           for d in species.values()}
        raise _Stopped

    monkeypatch.setattr(orbitals, "pseudo_basis", stop)
    atoms = Atoms("H2", positions=[(3.0, 3.0, 2.63), (3.0, 3.0, 3.37)],
                  cell=[6.0, 6.0, 6.0])
    atoms.calc = Mandacaru(basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.4, trace=False, **options)
    with pytest.raises(_Stopped):
        atoms.get_potential_energy()
    (folder,) = seen["sources"]
    return folder


def _crystal_folder(monkeypatch, **options):
    """The folder of the datasets a periodic ``Mandacaru`` hands the crystal
    setup (:func:`~mandacaru.pseudopotentials.periodic_paw.build_crystal`)."""
    from ase.build import bulk

    from mandacaru import Mandacaru
    from mandacaru.pseudopotentials import periodic_paw
    from mandacaru.pseudopotentials.paw import get_paw

    seen = {}

    def stop(atoms, grid_spacing, basis_options, **kwargs):
        seen["source"] = get_paw("Si", basis_options["paw_dataset"]).source
        raise _Stopped

    monkeypatch.setattr(periodic_paw, "build_crystal", stop)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = Mandacaru(method="dft",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.4, trace=False,
                           kpts={"size": (1, 1, 1), "gamma": True}, **options)
    with pytest.raises(_Stopped):
        atoms.get_potential_energy()
    return os.path.basename(os.path.dirname(seen["source"]))


@pytest.mark.parametrize("options, folder", [
    ({"method": "dft", "xc": "lda"}, "lda-sr"),
    ({"method": "dft", "xc": "pbe"}, "pbe-sr"),
    ({"method": "dft", "xc": "hse06"}, "pbe-sr"),
    ({"method": "dft", "xc": "r2scan"}, "pbe-sr"),
    ({"method": "dft", "xc": "pbe", "paw_dataset": "lda-sr"}, "lda-sr"),
    ({"method": "rhf"}, "lda-sr")])
@pytest.mark.filterwarnings("ignore:xc=:RuntimeWarning")  # a named set may mismatch
def test_a_molecule_reads_the_set_of_its_functional(monkeypatch, real_library,
                                                    options, folder):
    assert _molecule_folder(monkeypatch, **options) == folder


@pytest.mark.parametrize("options, folder", [
    ({"xc": "lda"}, "lda-sr"),
    ({"xc": "pbe"}, "pbe-sr"),
    ({"xc": "hse06"}, "pbe-sr"),
    ({"xc": "r2scan"}, "pbe-sr"),
    ({"xc": "pbe", "paw_dataset": "lda-sr"}, "lda-sr"),
    ({"xc": "lda", "paw_dataset": "pbe-sr"}, "pbe-sr")])
@pytest.mark.filterwarnings("ignore:xc=:RuntimeWarning")  # a named set may mismatch
def test_a_crystal_reads_the_set_of_its_functional(monkeypatch, real_library,
                                                   options, folder):
    """A crystal used to read ``lda-sr`` whatever its functional: PBE and HSE06
    energies were those of datasets generated with another functional."""
    assert _crystal_folder(monkeypatch, **options) == folder


def test_a_dataset_in_the_wrong_functional_folder_is_refused(tmp_path):
    """An LDA file copied into ``pbe/`` would otherwise run silently as the
    PBE dataset ``paw_dataset="pbe"`` asked for."""
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
        get_paw("H", paw_dataset=str(folder))
