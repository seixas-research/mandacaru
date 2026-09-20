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

from mandacaru.algorithms import Mandacaru
from mandacaru.circuits import AdaptAnsatz, UCCSD
from mandacaru.circuits.pools import build_pool
from mandacaru.core.mapping import (parity_tapered_qubits, reference_qubit_bits,
                                    two_qubit_reduce)
from mandacaru.core import PauliSum


def _h2():
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.center(vacuum=3.0)
    return atoms


@pytest.fixture(scope="module")
def reference():
    atoms = _h2()
    atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO", h=0.4,
                           trace=False, profile=False, max_iterations=4)
    atoms.get_total_energy()
    return atoms.calc.result.optimal_energy, atoms.calc.result.reference_energy


class TestMapping:
    def test_tapered_positions_and_reference_bits(self):
        assert parity_tapered_qubits(4) == (1, 3)
        full = reference_qubit_bits("parity", 4, [0, 2])
        reduced = reference_qubit_bits("parity_reduced", 4, [0, 2])
        assert list(full) == [1, 1, 0, 0]       # parity sums of |1010>
        assert list(reduced) == [1, 0]           # positions 1 and 3 dropped
        assert len(reference_qubit_bits("jordan_wigner", 4, [0, 2])) == 4

    def test_reduce_drops_z_on_tapered_qubits_with_the_sector_sign(self):
        op = PauliSum({"IZIZ": 1.0, "ZIII": 0.5, "XIXI": 0.25})
        red = two_qubit_reduce(op, 4, (1, 1))
        # Z on both tapered qubits: (-1)^1 * (-1)^2 = -1 times the identity.
        assert red.terms["II"] == pytest.approx(-1.0)
        assert red.terms["ZI"] == pytest.approx(0.5)
        assert red.terms["XX"] == pytest.approx(0.25)


class TestPoolsAndAnsatze:
    def test_fermionic_pool_is_tapered(self):
        pool = build_pool("fermionic", 2, (1, 1),
                          mapping="parity_reduced")
        assert pool.n_qubits == 2 and pool.n_modes == 4
        assert all(op.generator.num_qubits == 2 for op in pool.operators())
        assert pool.occupied_orbitals == (0, 2)

    @pytest.mark.parametrize("name", ["qeb", "ceo"])
    def test_qubit_excitation_pools_are_tapered(self, name):
        """They commute with both tapered symmetries, so the taper is exact."""
        pool = build_pool(name, 2, (1, 1), mapping="parity_reduced")
        assert pool.n_qubits == 2 and pool.n_modes == 4
        assert all(op.generator.num_qubits == 2 for op in pool.operators())
        assert pool.operators()

    def test_the_qubit_pool_still_refuses(self):
        """Individual Pauli strings do not commute with the symmetries."""
        with pytest.raises(ValueError, match="do not commute"):
            build_pool("qubit", 2, (1, 1), mapping="parity_reduced")

    def test_adapt_ansatz_reference_is_tapered(self):
        ansatz = AdaptAnsatz(2, (0, 2), "parity_reduced",
                             num_particles=(1, 1))
        assert ansatz.reference_qubits() == [0]
        assert np.isclose(np.abs(ansatz.reference_state()[0b10]), 1.0)

    def test_uccsd_is_tapered(self):
        uccsd = UCCSD(2, (1, 1), mapping="parity_reduced")
        assert uccsd.n_qubits == 2
        assert all(g.num_qubits == 2 for g in uccsd.pauli_generators)
        assert np.isclose(np.abs(uccsd.reference_state()[0b10]), 1.0)


class TestDrivers:
    def test_adapt_vqe_reaches_the_untapered_energy(self, reference):
        exact, hf = reference
        atoms = _h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="FAO", h=0.4,
                               mapping="parity_reduced", trace=False,
                               profile=True, max_iterations=4)
        atoms.get_total_energy()
        calc = atoms.calc
        assert calc.n_qubits == 2
        assert len(calc.hamiltonian.simplify().terms) == 5
        assert calc.result.metrics.cnot_count is not None   # profiled on 2 qubits
        # Both sides are eV (1e-8 / 1e-7 Ha scaled by the conversion factor).
        assert calc.result.reference_energy == pytest.approx(
            hf, abs=1e-8 * HARTREE_TO_EV)
        assert calc.result.optimal_energy == pytest.approx(
            exact, abs=1e-7 * HARTREE_TO_EV)

    def test_vqe_reaches_the_untapered_energy(self, reference):
        exact, _hf = reference
        atoms = _h2()
        atoms.calc = Mandacaru(method="vqe", basis="FAO", h=0.4,
                               mapping="parity_reduced",
                               optimizer="L-BFGS-B", trace=False)
        atoms.get_total_energy()
        assert atoms.calc.n_qubits == 2
        assert atoms.calc.result.optimal_energy == pytest.approx(
            exact, abs=1e-6 * HARTREE_TO_EV)

    def test_the_old_constructor_argument_is_removed(self):
        with pytest.raises(TypeError, match="two_qubit_reduction"):
            Mandacaru(method="adapt-vqe", pool="fermionic", basis="FAO",
                      two_qubit_reduction=True)

    def test_subspace_solvers_refuse(self):
        with pytest.raises(NotImplementedError):
            Mandacaru(method="subspace-vqe", basis="FAO",
                      mapping="parity_reduced", trace=False)

    def test_dry_run_counts_the_reduced_register(self):
        atoms = _h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="FAO", mapping="parity_reduced",
                               dry_run=True,
                               trace=False)
        assert np.isnan(atoms.get_potential_energy())
        assert atoms.calc.dry_run_result.n_qubits == 2

    def test_measured_energy_on_the_reduced_register(self, reference):
        from mandacaru.backends.providers import QiskitProvider
        exact, _hf = reference
        atoms = _h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="FAO", h=0.4,
                               mapping="parity_reduced", trace=False,
                               profile=False, max_iterations=4)
        atoms.get_total_energy()
        provider = QiskitProvider()
        assert atoms.calc.measured_energy(provider) == pytest.approx(
            exact, abs=1e-8 * HARTREE_TO_EV)
        qc = provider.build(*atoms.calc.ansatz_problem()[:4])
        assert qc.num_qubits == 2
