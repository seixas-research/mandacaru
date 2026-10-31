"""DFT+U through ``Mandacaru(hubbard_u=...)`` against the frozen numbers of the deleted route.

The references are ``reference/hubbard/frozen.json`` (HU-0, the old Python driving the same
extension) and the crystal fixtures of ``export_hubbard_crystal.py``. Tolerances are those of the
Rust gates (``crates/mandacaru-core/tests/periodic_hubbard.rs``, HU-3): the energy is variational
(1e-8 eV), the DFT+U energy, the gap and the occupations are first order in the density error
(1e-5). Each SCF here is one to two seconds of setup plus a few seconds of iterations, under
1 GB. Needs ``MANDACARU_PAW_PATH``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, make_supercell

from mandacaru import Mandacaru
from mandacaru.integrations import paw
from mandacaru.units import HARTREE_TO_EV
from mandacaru.utils import parse_output

HERE = Path(__file__).resolve().parent
FROZEN = json.loads((HERE / "reference" / "hubbard" / "frozen.json").read_text())["cases"]
FIXTURE = HERE / "reference" / "hubbard" / "crystals"
SZ = {"name": "PAW-LCAO", "size": "SZ"}
#: Loading the CuH crystal file takes 738 MB of the calculation budget (the default is 512 MiB).
LOAD = dict(method="dft", trace=False, resources={"memory_limit_bytes": 1 << 30})
CRYSTAL_INPUT = paw.crystal_input

ENERGY_TOLERANCE_EV = 1e-8
FIRST_ORDER_EV = 1e-5
N_TOLERANCE = 1e-5
NI_D = {"Ni": {"l": "d", "U": 3.0, "J": 0.0}}
CU_D = {"Cu": {"l": "d", "U": 4.0}}


def nio():
    """NiO AFM-II: rocksalt a = 4.17 A in the two-formula-unit cell, moments +-2 on Ni."""
    atoms = make_supercell(bulk("NiO", "rocksalt", a=4.17), [[1, 0, 0], [0, 1, 0], [-1, -1, 2]])
    ni = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Ni"]
    moments = np.zeros(len(atoms))
    moments[ni] = [2.0 * (-1) ** k for k in range(len(ni))]
    atoms.set_initial_magnetic_moments(moments)
    return atoms


NIO_OPTIONS = dict(
    method="dft", xc="lda", grid_spacing=0.25, trace=False, basis=SZ, spin=True,
    kpts={"size": (2, 2, 2), "gamma": True}, smearing={"method": "fermi-dirac", "width": 0.05},
)


def cuh():
    """CuH in a 7 A periodic box (Gamma only), restricted."""
    atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[7.0] * 3, pbc=True)
    atoms.center()
    return atoms


CUH_OPTIONS = dict(
    method="dft", xc="lda", grid_spacing=0.3, trace=False, basis=SZ, symmetry=False,
    smearing={"method": "fermi-dirac", "width": 0.01},
)


def ni_atom():
    atoms = Atoms("Ni", positions=[[0, 0, 0]], cell=[7.0] * 3, pbc=True)
    atoms.center()
    atoms.set_initial_magnetic_moments([2.0])
    return atoms


def gap_ev(result):
    raw = result.periodic.snapshot(include_matrices=False)
    levels = np.asarray(raw["eigenvalues"]).ravel()
    fermi = raw["fermi_level"]
    return (levels[levels > fermi].min() - levels[levels <= fermi].max()) * HARTREE_TO_EV


def run(atoms, **options):
    atoms.calc = Mandacaru(**options)
    atoms.get_potential_energy()
    return atoms.calc.result


class _Captured(BaseException):
    """Stops ``configure`` after the snapshot is written, before the native problem is built."""


def snapshot_of(monkeypatch, atoms, **options):
    taken = {}

    def hook(*args, **kwargs):
        metadata, _ = CRYSTAL_INPUT(*args, **kwargs)
        taken.update(json.loads(metadata))
        raise _Captured

    monkeypatch.setattr(paw, "crystal_input", hook)
    atoms.calc = Mandacaru(**options)
    with pytest.raises(_Captured):
        atoms.get_potential_energy()
    return taken


# --------------------------------------------------------------------------- #
# The shell table Python hands to Rust.
# --------------------------------------------------------------------------- #


def test_the_shell_table_is_the_one_the_frozen_weights_define(monkeypatch):
    """Projectors, basis and seed equal the fixture's, and the Rust weights equal the frozen
    ``w_raw`` of the deleted ``shell_weights`` (the fixture's weights come from ``frozen.json``)."""
    taken = snapshot_of(monkeypatch, nio(), hubbard_u=NI_D, **NIO_OPTIONS)
    fixture = json.loads((FIXTURE / "nio_afm2.json").read_text())["hubbard"]
    table = taken["hubbard"]
    assert table["form"] == fixture["form"] == "dudarev-fll"
    assert len(table["shells"]) == len(fixture["shells"]) == 2
    for ours, theirs in zip(table["shells"], fixture["shells"]):
        for key in ("atom", "symbol", "l", "projectors", "projector_basis", "seed"):
            assert ours[key] == theirs[key], key
        assert np.allclose(ours["weights"], theirs["weights"], rtol=0, atol=1e-10)
        assert ours["u"] == 3.0 / HARTREE_TO_EV and ours["j"] == 0.0
    moments = [s["seed"]["moment"] for s in table["shells"]]
    assert moments == [2.0, -2.0]
    assert snapshot_of(monkeypatch, nio(), **NIO_OPTIONS)["hubbard"] is None
    # the form field is part of the contract: naming the one form changes nothing
    named = {"Ni": {**NI_D["Ni"], "form": "dudarev-fll"}}
    assert snapshot_of(monkeypatch, nio(), hubbard_u=named, **NIO_OPTIONS)["hubbard"] == table


# --------------------------------------------------------------------------- #
# CuH in a box: U = 0 is plain DFT, U = 4 is the frozen shift, and the file round trip.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cuh_runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cuh")
    plain = run(cuh(), save_crystal=folder / "plain.json", **CUH_OPTIONS)
    inert = run(cuh(), hubbard_u={"Cu": {"l": "d", "U": 0.0}}, **CUH_OPTIONS)
    atoms = cuh()
    four = run(atoms, hubbard_u=CU_D, save_crystal=folder / "four.json", **CUH_OPTIONS)
    return dict(folder=folder, plain=plain, inert=inert, four=four, atoms=atoms)


def test_zero_u_is_plain_dft_through_the_public_api(cuh_runs):
    plain, inert = cuh_runs["plain"], cuh_runs["inert"]
    assert abs(inert.in_units("eV") - plain.in_units("eV")) < ENERGY_TOLERANCE_EV
    assert abs(plain.in_units("eV") - FROZEN["cuh_crystal_L7_U0"]["energy_eV"]) < ENERGY_TOLERANCE_EV
    assert plain.hubbard is None and plain.hubbard_energy is None
    assert "hubbard" not in plain.periodic.snapshot(include_matrices=False)["terms"]
    assert inert.hubbard_energy == 0.0
    assert inert.periodic.snapshot(include_matrices=False)["terms"]["hubbard"] == 0.0
    assert abs(gap_ev(inert) - gap_ev(plain)) < FIRST_ORDER_EV


def test_cuh_at_u_4_shifts_by_the_frozen_amount_and_exposes_its_occupations(cuh_runs):
    plain, four = cuh_runs["plain"], cuh_runs["four"]
    frozen = FROZEN["cuh_crystal_L7_U4"]
    shift = four.in_units("eV") - plain.in_units("eV")
    expected = frozen["energy_eV"] - FROZEN["cuh_crystal_L7_U0"]["energy_eV"]
    assert abs(shift - expected) < ENERGY_TOLERANCE_EV
    assert abs(gap_ev(four) - frozen["gap_from_levels_eV"]) < FIRST_ORDER_EV
    assert abs(four.hubbard_energy - frozen["hubbard_eV"]) < FIRST_ORDER_EV
    (shell,) = four.hubbard
    assert (shell["symbol"], shell["l"], shell["atom"]) == ("Cu", "d", 0)
    assert abs(shell["U"] - 4.0) < 1e-12 and shell["J"] == 0.0
    assert abs(shell["U_eff"] - 4.0) < 1e-12
    assert abs(shell["energy"] - four.hubbard_energy) < 1e-12
    n = shell["n"]
    assert n.shape == (1, 5, 5) and np.allclose(n[0], n[0].conj().T, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(n[0])
    assert np.allclose(np.sort(eigenvalues), np.sort(frozen["n_eigenvalues"][0][0]), atol=N_TOLERANCE)
    assert abs(shell["trace"][0] - eigenvalues.sum()) < 1e-12


def test_off_mesh_bands_carry_the_converged_correction(cuh_runs):
    """The band path diagonalizes with the SCF potential and the DFT+U operator: at Gamma, a
    mesh point, it returns the SCF spectrum (without the operator the d levels move by 0.07 Ha)."""
    four = cuh_runs["four"]
    band = cuh_runs["atoms"].calc.band_structure(path="GX", npoints=3)
    scf = four.periodic.snapshot(include_matrices=False)["eigenvalues"][:, 0, :]
    assert np.allclose(band.energies[:, 0, :], scf * HARTREE_TO_EV, rtol=0, atol=1e-8)


def test_a_saved_crystal_carries_its_correction_and_reproduces_the_energy(cuh_runs):
    folder = cuh_runs["folder"]
    loaded = Mandacaru(load_crystal=folder / "four.json",
                       smearing={"method": "fermi-dirac", "width": 0.01}, **LOAD)
    result = loaded.run()
    assert abs(result.in_units("eV") - cuh_runs["four"].in_units("eV")) < ENERGY_TOLERANCE_EV
    assert abs(result.hubbard_energy - cuh_runs["four"].hubbard_energy) < FIRST_ORDER_EV
    assert loaded.periodic_problem.hubbard == [(0, "Cu", 2, 4.0 / HARTREE_TO_EV, 0.0)]


def test_the_option_against_a_loaded_crystal_follows_the_mismatch_rule(cuh_runs):
    folder = cuh_runs["folder"]
    short = dict(periodic_scf_options={"max_iterations": 1}, **LOAD)
    # the file's own definition is accepted (one iteration is enough to pass the check)
    Mandacaru(load_crystal=folder / "four.json", hubbard_u=CU_D, **short).run()
    for option in (
        {"Cu": {"l": "d", "U": 5.0}},
        {"Cu": {"l": "d", "U": 4.0, "J": 1.0}},
        {"Cu": {"l": "f", "U": 4.0}},
        {"Cu": {"l": "d", "U": 4.0}, "Ni": {"l": "d", "U": 4.0}},
    ):
        with pytest.raises(ValueError, match="disagrees with the crystal file"):
            Mandacaru(load_crystal=folder / "four.json", hubbard_u=option, **short).run()
    with pytest.raises(ValueError, match="disagrees with the crystal file"):
        Mandacaru(load_crystal=folder / "plain.json", hubbard_u=CU_D, **short).run()


# --------------------------------------------------------------------------- #
# NiO AFM-II at U = 3 eV: the frozen energy, gap, DFT+U energy and occupations, and the log.
# --------------------------------------------------------------------------- #


def test_nio_afm2_at_u_3_matches_the_frozen_numbers_and_the_log_states_them_once(tmp_path):
    frozen = FROZEN["nio_U3_A"]
    atoms = nio()
    log = tmp_path / "nio.log"
    result = run(atoms, hubbard_u=NI_D, txt=str(log), **NIO_OPTIONS)
    assert result.success
    assert abs(atoms.get_potential_energy() - frozen["energy_eV"]) < ENERGY_TOLERANCE_EV
    assert abs(gap_ev(result) - frozen["gap_from_levels_eV"]) < FIRST_ORDER_EV
    assert abs(result.hubbard_energy - frozen["hubbard_eV"]) < FIRST_ORDER_EV
    shells = result.hubbard
    assert len(shells) == 2
    assert sum(s["energy"] for s in shells) == pytest.approx(result.hubbard_energy, abs=1e-12)
    spectrum = np.sort(np.concatenate([np.linalg.eigvalsh(s["n"][spin]) for s in shells for spin in (0, 1)]))
    assert abs(spectrum.min() - frozen["n_eigenvalue_min"]) < N_TOLERANCE
    assert abs(spectrum.max() - frozen["n_eigenvalue_max"]) < N_TOLERANCE
    # antiferromagnetic: the two Ni sites are spin mirrors, and the d moment is positive on site 0
    up0, down0 = shells[0]["trace"]
    assert shells[1]["trace"] == pytest.approx([down0, up0], abs=1e-6)
    assert up0 - down0 > 1.0
    # the log: the correction's energy and definition are written once, in [SCF SUMMARY]
    text = log.read_text()
    assert text.count("hubbard_energy_eV") == 1 and text.count("U_eff") == 1
    summary = parse_output(str(log))["summary"]
    assert abs(float(summary["hubbard_energy_eV"]) - result.hubbard_energy) < 1e-8
    assert summary["hubbard_Ni_d"].startswith("U 3.000000 eV, J 0.000000 eV, U_eff 3.000000 eV")


def test_a_run_without_the_correction_writes_no_hubbard_lines(tmp_path):
    log = tmp_path / "plain.log"
    run(cuh(), txt=str(log), **CUH_OPTIONS)
    assert "hubbard" not in log.read_text()
    assert "[SCF SUMMARY]" in log.read_text()


# --------------------------------------------------------------------------- #
# Refusals: unsupported input raises, with the reason, and never selects another calculation.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec, message",
    [
        ({}, "must map an element symbol"),
        ("Ni", "must map an element symbol"),
        ({"Xx": {"l": "d", "U": 1.0}}, "not an element symbol"),
        ({"X": {"l": "d", "U": 1.0}}, "not an element symbol"),
        ({"Ni": 6.0}, "needs the keys"),
        ({"Ni": {"U": 6.0}}, "needs the keys"),
        ({"Ni": {"l": "d"}}, "needs the keys"),
        ({"Ni": {"l": "d", "U": 6.0, "weight": 1}}, "needs the keys"),
        ({"Ni": {"l": "d", "U": 6.0, "form": "x"}}, "unknown Hubbard form"),
        ({"Ni": {"l": "d", "U": 6.0, "form": 1}}, "form must be a string"),
        ({"Ni": {"l": "p", "U": 6.0}}, "'d' or 'f'"),
        ({"Ni": {"l": 2, "U": 6.0}}, "'d' or 'f'"),
        ({"Ni": {"l": "d", "U": -1.0}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": True}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": float("nan")}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": float("inf")}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": "6"}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": 1.0, "J": -0.1}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": 1.0, "J": 2.0}}, "exceeds U"),
    ],
)
def test_a_malformed_option_is_refused_at_construction(spec, message):
    with pytest.raises(ValueError, match=message):
        Mandacaru(method="dft", hubbard_u=spec, trace=False)


def test_numpy_scalars_are_numbers_and_numpy_booleans_are_not():
    Mandacaru(method="dft", hubbard_u={"Ni": {"l": "d", "U": np.float32(3.0), "J": np.int64(0)}},
              trace=False)
    with pytest.raises(ValueError, match="finite number >= 0"):
        Mandacaru(method="dft", hubbard_u={"Ni": {"l": "d", "U": np.bool_(True)}}, trace=False)
    with pytest.raises(ValueError, match="must map an element symbol"):
        Mandacaru(method="dft", hubbard_u={("Ni",): {"l": "d", "U": 1.0}}, trace=False)


def test_combinations_the_engine_does_not_implement_are_refused_at_construction():
    with pytest.raises(NotImplementedError, match="hybrid functional 'hse06'"):
        Mandacaru(method="dft", xc="hse06", hubbard_u=NI_D, trace=False)
    with pytest.raises(ValueError, match="precision='float64'"):
        Mandacaru(method="dft", precision="float32", hubbard_u=NI_D, trace=False)
    # a geometry given with the calculator: a molecule has no periodic direction
    molecule = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[8.0] * 3)
    with pytest.raises(ValueError, match="needs a periodic cell"):
        Mandacaru(method="dft", atoms=molecule, hubbard_u=CU_D, trace=False)


def test_a_molecule_is_refused_before_any_calculation():
    molecule = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[8.0] * 3)
    molecule.center()
    molecule.calc = Mandacaru(method="dft", hubbard_u=CU_D, trace=False, basis=SZ)
    with pytest.raises(ValueError, match="needs a periodic cell"):
        molecule.get_potential_energy()


def test_an_element_the_system_lacks_is_refused(monkeypatch):
    silicon = bulk("Si", "diamond", a=5.43)
    silicon.calc = Mandacaru(method="dft", hubbard_u=NI_D, trace=False, basis=SZ,
                             grid_spacing=0.4)
    with pytest.raises(ValueError, match="Ni, which is not in the system"):
        silicon.get_potential_energy()


def test_a_dataset_without_the_shell_is_refused():
    atom = ni_atom()
    atom.calc = Mandacaru(method="dft", hubbard_u={"Ni": {"l": "f", "U": 3.0}}, trace=False,
                          basis=SZ, grid_spacing=0.4, spin=True)
    with pytest.raises(ValueError, match="Ni dataset has no 'f' channel"):
        atom.get_potential_energy()


def test_a_spin_orbit_dataset_is_refused():
    atom = ni_atom()
    atom.calc = Mandacaru(method="dft", hubbard_u=NI_D, trace=False, grid_spacing=0.4,
                          basis={**SZ, "paw_dataset": "lda-fr"}, spin=True)
    with pytest.raises(NotImplementedError, match="spin-orbit"):
        atom.get_potential_energy()


def test_a_crystal_whose_mesh_is_reduced_by_symmetry_is_refused_naming_the_option():
    nickel = bulk("Ni", "fcc", a=3.52)
    nickel.set_initial_magnetic_moments([0.0])        # ASE's default moment would keep the full mesh
    nickel.calc = Mandacaru(method="dft", hubbard_u=NI_D, trace=False, basis=SZ,
                            grid_spacing=0.4, kpts={"size": (2, 2, 2), "gamma": True})
    with pytest.raises(ValueError, match="symmetry=False"):
        nickel.get_potential_energy()
