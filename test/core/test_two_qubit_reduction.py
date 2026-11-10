# -*- coding: utf-8 -*-
# file: test/test_two_qubit_reduction.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The parity mapping's two-qubit reduction, end to end.

``mapping="parity_reduced"`` tapers the two parity
qubits fixed by the particle numbers from the Hamiltonian, the pool / UCCSD
generators and the reference determinant alike, so ADAPT-VQE and VQE run on
``2M - 2`` qubits and reach exactly the energies of the untapered register.
"""

import numpy as np
from mandacaru.units import HARTREE_TO_EV
import pytest

from mandacaru import Mandacaru
from mandacaru import _native
from mandacaru.core import Fermion, PauliSum
from mandacaru.specifications import estimate_qubits

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
LBFGS = {"method": "L-BFGS", "maxiter": 2000, "tol": 1e-12}


def _h2():
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.center(vacuum=3.0)
    return atoms


@pytest.fixture(scope="module")
def reference():
    atoms = _h2()
    atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO", grid_spacing=0.4,
                           trace=False, profile=False, max_iterations=4)
    atoms.get_total_energy()
    return atoms.calc.result.optimal_energy, atoms.calc.result.reference_energy


class TestMapping:
    @staticmethod
    def _problem(mapping):
        """A 2-orbital, (1, 1) problem whose Hamiltonian is the number operator."""
        number = Fermion.from_integrals(np.eye(4))
        operator = number.map_to_qubits(
            mapping, num_particles=(1, 1) if mapping == "parity_reduced" else None)
        return _native.Context(threads=1).quantum_problem(
            operator._native_operator(), 2, [1, 1], mapping=mapping)

    def test_the_reference_determinant_in_each_encoding(self):
        # The determinant 1010 (modes 0 and 2): parity sums 1100, with the
        # alpha and total parity qubits (1 and 3) dropped by the reduction.
        assert self._problem("parity").reference_qubits == [0, 1]
        assert self._problem("parity_reduced").reference_qubits == [0]
        assert self._problem("jordan_wigner").reference_qubits == [0, 2]
        assert self._problem("parity_reduced").num_qubits == 2

    def test_the_tapered_qubits_take_the_sector_sign(self):
        """Z on the alpha-parity and total-parity qubits is the sector's parity:
        in the (1, 1) sector the reduced register's four states are the four
        determinants, and the number operators are constants on them."""
        alpha = Fermion.from_integrals(np.diag([1.0, 1.0, 0.0, 0.0]))
        total = Fermion.from_integrals(np.eye(4))
        reduced = {name: op.map_to_qubits("parity_reduced", num_particles=(1, 1))
                   for name, op in (("alpha", alpha), ("total", total))}
        assert reduced["alpha"].num_qubits == 2
        assert np.allclose(reduced["alpha"].to_matrix(), np.eye(4))
        assert np.allclose(reduced["total"].to_matrix(), 2 * np.eye(4))


class TestPoolsOnTheReducedRegister:
    @pytest.mark.parametrize("pool", ["qeb", "ceo"])
    def test_qubit_excitation_pools_are_tapered(self, pool):
        """They commute with both tapered symmetries, so the taper is exact."""
        atoms = _h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool=pool, basis="HAO",
                               grid_spacing=0.4, mapping="parity_reduced",
                               trace=False, max_iterations=1)
        atoms.get_total_energy()
        assert atoms.calc.n_qubits == 2

    def test_the_qubit_pool_still_refuses(self):
        """Individual Pauli strings do not commute with the symmetries."""
        atoms = _h2()
        with pytest.raises(ValueError, match="untapered register"):
            atoms.calc = Mandacaru(method="adapt-vqe", pool="qubit",
                                   basis="HAO", grid_spacing=0.4,
                                   mapping="parity_reduced", trace=False,
                                   max_iterations=1)
            atoms.get_total_energy()


class TestDrivers:
    def test_adapt_vqe_reaches_the_untapered_energy(self, reference):
        exact, hf = reference
        atoms = _h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="HAO", grid_spacing=0.4,
                               mapping="parity_reduced", trace=False,
                               profile=True, max_iterations=4)
        atoms.get_total_energy()
        calc = atoms.calc
        assert calc.n_qubits == 2
        terms = {label for label, _re, _im in calc.problem.hamiltonian_terms}
        assert len(PauliSum({label: 1.0 for label in terms}).simplify().terms) == 5
        assert calc.result.metrics.cnot_count is not None   # profiled on 2 qubits
        # Both sides are eV (1e-8 / 1e-7 Ha scaled by the conversion factor).
        assert calc.result.reference_energy == pytest.approx(
            hf, abs=1e-8 * HARTREE_TO_EV)
        assert calc.result.optimal_energy == pytest.approx(
            exact, abs=1e-7 * HARTREE_TO_EV)

    def test_vqe_reaches_the_untapered_energy(self, reference):
        exact, _hf = reference
        atoms = _h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4,
                               mapping="parity_reduced",
                               optimizer=LBFGS, trace=False)
        atoms.get_total_energy()
        assert atoms.calc.n_qubits == 2
        assert atoms.calc.result.optimal_energy == pytest.approx(
            exact, abs=1e-6 * HARTREE_TO_EV)

    def test_the_old_constructor_argument_is_removed(self):
        with pytest.raises(TypeError, match="two_qubit_reduction"):
            Mandacaru(method="adapt-vqe", pool="fermionic", basis="HAO",
                      two_qubit_reduction=True)

    def test_subspace_solvers_run_on_the_reduced_register(self):
        energies = {}
        for mapping in ("jordan_wigner", "parity_reduced"):
            atoms = _h2()
            atoms.calc = Mandacaru(method="subspace-vqe", basis="HAO",
                                   grid_spacing=0.4, mapping=mapping,
                                   trace=False)
            atoms.get_total_energy()
            energies[mapping] = (atoms.calc.n_qubits,
                                 np.asarray(atoms.calc.result.energies))
        assert energies["jordan_wigner"][0] == 4
        assert energies["parity_reduced"][0] == 2
        assert np.allclose(energies["parity_reduced"][1],
                           energies["jordan_wigner"][1], atol=1e-6)

    def test_the_dry_run_counts_the_reduced_register(self):
        estimate = estimate_qubits(_h2(), basis="HAO", mapping="parity_reduced")
        assert estimate.n_qubits == 2

    def test_measured_energy_on_the_reduced_register(self, reference):
        exact, _hf = reference
        atoms = _h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="HAO", grid_spacing=0.4,
                               mapping="parity_reduced", trace=False,
                               profile=False, max_iterations=4,
                               execute_circuits=True)
        atoms.get_total_energy()
        assert atoms.calc.result.optimal_energy == pytest.approx(
            exact, abs=1e-8 * HARTREE_TO_EV)
        problem = atoms.calc.problem
        qc = atoms.calc.provider.build(
            problem.num_qubits, problem.reference_qubits,
            atoms.calc.solver.ansatz.pauli_generators,
            atoms.calc.result.optimal_parameters)
        assert qc.num_qubits == 2
