"""MG-Q7: Z2 tapering through ``Mandacaru(taper=True)`` against the untapered calculation.

The molecules are the ones the frozen reference (``reference/tapering.json``) was built
from: RHF on PAW-LCAO SZ at a coarse grid. Every comparison is the same calculation
twice, with and without ``taper=True``: the energy, the density matrices (natural
occupations, the density field and the dipole moment it carries) and the register size.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from mandacaru import Mandacaru, _native

FROZEN = Path(__file__).parent / "reference/tapering.json"
#: Qubits Z2 tapering removes from the Jordan-Wigner register of each molecule.
REMOVED = {"H2": 3, "LiH": 2, "BeH2": 3}
OPTIMIZER = {"method": "BFGS", "maxiter": 1000, "tol": 1e-12}
CONVERGENCE = {"gradient": 1e-9, "energy": 1e-10}


def geometry(name):
    """The molecule; linear BeH2 (Be-H 1.33 Angstrom) is not in the G2 set."""
    if name == "BeH2":
        return Atoms("BeH2", positions=[(0., 0., 0.), (0., 0., 1.33), (0., 0., -1.33)])
    return molecule(name)


def calculate(name, method, taper, **options):
    atoms = geometry(name)
    atoms.center(vacuum=3.0)
    calc = Mandacaru(method=method, taper=taper, grid_spacing=0.45, trace=False,
                     basis={"name": "PAW-LCAO", "size": "SZ"}, atomic_units=True,
                     optimizer=OPTIMIZER, convergence=CONVERGENCE, max_iterations=100,
                     parallelization={"threads": 1}, **options)
    atoms.calc = calc
    atoms.get_potential_energy()
    return calc


def dipole(field):
    """The electronic dipole of a sampled density, in e Bohr (its own origin)."""
    grid = field.grid
    weights = field.data * grid.dV
    return np.array([np.sum(weights * axis) for axis in (grid.X, grid.Y, grid.Z)])


@pytest.fixture(scope="module", params=sorted(REMOVED))
def pair(request):
    return request.param, {
        method: (calculate(request.param, method, False), calculate(request.param, method, True))
        for method in ("vqe", "adapt-vqe")}


class TestTaperedMolecules:
    def test_the_energy_equals_the_untapered_one(self, pair):
        name, runs = pair
        for method, (plain, tapered) in runs.items():
            e_plain, e_tapered = plain.result.energy_hartree, tapered.result.energy_hartree
            print(f"{name} {method}: {e_plain:+.12f} {e_tapered:+.12f} ({abs(e_plain - e_tapered):.1e})")
            assert e_tapered == pytest.approx(e_plain, abs=1e-8)
            assert tapered.result.reference_energy == pytest.approx(
                plain.result.reference_energy, abs=1e-10)

    def test_the_register_loses_one_qubit_per_generator(self, pair):
        name, runs = pair
        for plain, tapered in runs.values():
            info = tapered.problem.taper
            assert plain.problem.taper is None
            assert info["removed"] == len(info["generators"]) == len(info["anchors"]) == len(info["signs"])
            assert tapered.n_qubits == plain.n_qubits - info["removed"] == info["qubits"]
            assert info["qubits_before"] == plain.n_qubits
            assert info["removed"] == REMOVED[name]
            assert tapered.problem.dimension == 2 ** tapered.n_qubits

    def test_the_density_matrices_natural_occupations_and_dipole_agree(self, pair):
        name, runs = pair
        for plain, tapered in runs.values():
            a, b = plain.result.natural_orbitals(), tapered.result.natural_orbitals()
            assert np.max(np.abs(a.occupations - b.occupations)) < 1e-8
            assert b.n_electrons == pytest.approx(a.n_electrons, abs=1e-10)
            one_a, two_a = plain.problem.rdms(np.ascontiguousarray(plain.result._raw["state"]))
            one_b, two_b = tapered.problem.rdms(np.ascontiguousarray(tapered.result._raw["state"]))
            assert np.max(np.abs(one_a - one_b)) < 1e-8
            assert np.max(np.abs(two_a - two_b)) < 1e-8
            density_a = plain.result.volumetric_field("density")
            density_b = tapered.result.volumetric_field("density")
            assert np.max(np.abs(density_a.data - density_b.data)) < 1e-8
            assert np.max(np.abs(dipole(density_a) - dipole(density_b))) < 1e-8

    def test_the_dropped_pool_generators_are_reported(self, pair):
        name, runs = pair
        plain, tapered = runs["adapt-vqe"]
        labels = tapered.problem.taper["dropped"]
        assert len(tapered.problem.pool_labels) + len(labels) == len(plain.problem.pool_labels)
        assert not set(labels) & set(tapered.problem.pool_labels)
        # H2 and BeH2 have excitations that change the spatial parity; LiH keeps its whole pool.
        assert bool(labels) == (name != "LiH"), f"{name}: dropped {labels}"

    def test_the_state_is_the_tapered_register(self, pair):
        _, runs = pair
        for _, tapered in runs.values():
            assert tapered.result.state().shape == (2 ** tapered.n_qubits,)
            checkpoint = tapered.solver.checkpoint
            assert checkpoint.n_qubits == tapered.n_qubits
            assert list(checkpoint.reference_qubits) == tapered.problem.reference_qubits


def test_every_shipped_frozen_hamiltonian_is_taperable_through_the_option():
    """The Hamiltonians of the frozen cases, supplied directly: energy and register."""
    cases = json.loads(FROZEN.read_text())["cases"]
    context = _native.Context(threads=1)
    for case in cases:
        if case["molecule"] != "H2" or case["mapping"] == "bravyi_kitaev":
            continue
        n = case["n_qubits"]
        operator = context.pauli_operator(n, [tuple(t) for t in case["terms"]])
        energies = []
        for taper in (False, True):
            with Mandacaru(method="vqe", hamiltonian=operator, n_spatial_orbitals=2,
                           num_particles=(1, 1), mapping=case["mapping"], taper=taper,
                           atomic_units=True, optimizer={"method": "BFGS", "tol": 1e-12},
                           parallelization={"threads": 1}, trace=False) as calc:
                result = calc.run()
                energies.append(result.energy_hartree)
                assert calc.n_qubits == (case["tapered_qubits"] if taper else n)
        assert energies[1] == pytest.approx(energies[0], abs=1e-9)


def h2_calculator(**options):
    case = next(c for c in json.loads(FROZEN.read_text())["cases"]
                if c["molecule"] == "H2" and c["mapping"] == "jordan_wigner")
    context = _native.Context(threads=1)
    operator = context.pauli_operator(4, [tuple(t) for t in case["terms"]])
    options = {"method": "adapt-vqe", "atomic_units": True, "parallelization": {"threads": 1},
               "trace": False, **options}
    return Mandacaru(hamiltonian=operator, n_spatial_orbitals=2, num_particles=(1, 1), **options)


class TestRefusals:
    def test_a_leaking_pool_generator_is_dropped_and_reported(self):
        # The minimal-basis singles change the spatial parity; the double excitation keeps it.
        with h2_calculator(taper=True, pool="fermionic") as calc:
            calc.run()
            info = calc.problem.taper
            assert info["dropped"] == ["S(0->1)", "S(2->3)"]
            assert calc.problem.pool_labels == ["D(0,2->1,3)"]
            assert info["removed"] == 3 and len(info["signs"]) == 3

    def test_the_default_ceo_pool_drops_the_same_singles(self):
        with h2_calculator(taper=True) as calc:
            calc.run()
            assert calc.problem.taper["dropped"] == ["CEO[o0,o1]{S(0->1)}", "CEO[o2,o3]{S(2->3)}"]
            assert calc.problem.pool_labels[0] == "D(0,2->1,3)"

    @pytest.mark.parametrize("pool", ["qubit", "spin-orbit"])
    def test_a_register_pool_is_refused_with_the_parity_reduced_wording(self, pool):
        with h2_calculator(taper=True, pool=pool) as calc:
            with pytest.raises(ValueError, match="untapered register"):
                calc.run()

    @pytest.mark.parametrize("value", [1, 0, "yes", None, np.bool_(True)])
    def test_a_non_bool_taper_raises_type_error(self, value):
        with pytest.raises(TypeError, match="taper must be True or False"):
            h2_calculator(taper=value)

    def test_incompatible_options_are_refused_in_the_constructor(self):
        with pytest.raises(ValueError, match="sector=True"):
            h2_calculator(taper=True, sector=True)
        with pytest.raises(ValueError, match="save_hamiltonian"):
            h2_calculator(taper=True, save_hamiltonian="never_written.json")
        with pytest.raises(ValueError, match="hva"):
            h2_calculator(taper=True, method="vqe", ansatz={"name": "hva"})
        with pytest.raises(ValueError, match="does not use: taper"):
            h2_calculator(taper=True, method="vasqa")
        with pytest.raises(ValueError, match="does not use: taper"):
            Mandacaru(method="rhf", taper=True)

    def test_a_register_without_symmetry_runs_untapered_and_says_so(self, tmp_path):
        case = next(c for c in json.loads(FROZEN.read_text())["cases"]
                    if c["molecule"] == "LiH" and c["mapping"] == "parity_reduced")
        context = _native.Context(threads=1)
        operator = context.pauli_operator(2, [tuple(t) for t in case["terms"]])
        log = tmp_path / "output.txt"
        with Mandacaru(method="vqe", hamiltonian=operator, n_spatial_orbitals=2,
                       num_particles=(1, 1), mapping="parity_reduced", taper=True,
                       atomic_units=True, parallelization={"threads": 1}, txt=str(log)) as calc:
            calc.run()
            assert calc.problem.taper is None and calc.n_qubits == 2
        assert "Z2 tapering: none found" in log.read_text()

    def test_the_run_log_owns_the_taper_line(self, tmp_path):
        log = tmp_path / "output.txt"
        with h2_calculator(taper=True, txt=str(log)) as calc:
            calc.run()
        text = log.read_text()
        assert "Z2 tapering: 3 qubit(s) removed from a 4-qubit register" in text
        assert "pool generator(s) dropped as symmetry-changing" in text
        with h2_calculator(txt=str(tmp_path / "plain.txt")) as calc:
            calc.run()
        assert "Z2 tapering: none" in (tmp_path / "plain.txt").read_text()

    def test_a_shot_run_refuses_the_density_matrices_of_a_tapered_register(self):
        with h2_calculator(taper=True, shots=2048, method="vqe",
                           optimizer={"method": "COBYLA", "maxiter": 3}) as calc:
            result = calc.run()
            with pytest.raises(ValueError, match="shot-based run"):
                result.natural_orbitals()
