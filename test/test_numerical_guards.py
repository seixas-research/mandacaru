# -*- coding: utf-8 -*-
# file: test/test_numerical_guards.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Invalid numerics are reported, not silently propagated.

Three ways a wrong answer used to look like a valid one: an overlap matrix too
singular to invert, a C kernel that returned without writing its output, and a
Pauli operator whose register width was inferred from whatever term survived.
"""

import warnings

import numpy as np
import pytest

from carcara.basis import FullAtomicOrbital
from carcara.core.hamiltonian import (OVERLAP_EIGENVALUE_FLOOR,
                                      OVERLAP_EIGENVALUE_WARN,
                                      MolecularIntegrals)
from carcara.core.mapping import PauliSum
from carcara.integrals import Grid
from carcara.integrals._backend import (_check_filled, _check_samples,
                                        one_body_matrices)


def hydrogen_integrals(basis_size=1):
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=3.0, h=0.6, units="bohr")
    basis = [FullAtomicOrbital(1, 0, 0, Z=1.0, center=[0.0, 0.0, 0.0],
                               units="bohr") for _ in range(basis_size)]
    return MolecularIntegrals([(1.0, [0.0, 0.0, 0.0])], basis, grid,
                              units="bohr")


class TestOverlapConditioning:
    def test_a_duplicated_basis_function_is_refused(self):
        """Two copies of one orbital make the overlap exactly singular."""
        integrals = hydrogen_integrals(basis_size=2)
        with pytest.raises(ValueError, match="linearly dependent"):
            integrals.one_body()

    def test_a_nearly_dependent_basis_warns(self):
        integrals = hydrogen_integrals()
        off = 1.0 - 1e-8                      # smallest eigenvalue ~ 5e-9
        integrals.overlap = lambda: np.array([[1.0, off], [off, 1.0]])
        with pytest.warns(RuntimeWarning, match="nearly linearly dependent"):
            integrals._lowdin_x()

    def test_a_well_conditioned_basis_is_silent(self):
        integrals = hydrogen_integrals()
        integrals.overlap = lambda: np.array([[1.0, 0.2], [0.2, 1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            integrals._lowdin_x()

    def test_thresholds_are_ordered(self):
        assert OVERLAP_EIGENVALUE_FLOOR < OVERLAP_EIGENVALUE_WARN


class TestBackendBuffers:
    def test_sample_shape_must_match_the_grid(self):
        with pytest.raises(ValueError, match="expected samples"):
            _check_samples(np.ones((2, 4), dtype=complex), 5, "one_body_matrices")

    def test_non_finite_samples_are_refused(self):
        bad = np.array([[np.nan, 1.0]], dtype=complex)
        with pytest.raises(ValueError, match="non-finite"):
            _check_samples(bad, 2, "one_body_matrices")

    def test_an_unwritten_output_is_an_allocation_failure(self):
        """The C kernels are void and bail out on a failed malloc."""
        with pytest.raises(MemoryError, match="all-zero"):
            _check_filled(np.zeros((2, 2), dtype=complex),
                          np.ones((2, 5), dtype=complex), "kinetic matrix")

    def test_a_filled_output_passes_through(self):
        out = np.eye(2, dtype=complex)
        assert _check_filled(out, np.ones((2, 5), dtype=complex), "x") is out

    def test_the_wrapper_validates_before_calling_the_kernel(self):
        grid = Grid(center=[0.0, 0.0, 0.0], box_size=1.0, h=0.5, units="bohr")
        psi = np.ones((2, grid.size + 3), dtype=complex)
        potential = np.zeros(grid.size, dtype=float)
        with pytest.raises(ValueError, match="expected samples"):
            one_body_matrices(psi, potential, grid)


class TestPauliSumRegister:
    def test_a_zero_operator_keeps_its_register(self):
        zero = PauliSum({"XYZI": 0.0}).simplify()
        assert not zero.terms
        assert zero.num_qubits == 4
        assert zero.to_sparse_matrix().shape == (16, 16)

    def test_mixed_widths_are_refused(self):
        two, three = PauliSum({"XX": 1.0}), PauliSum({"XXX": 1.0})
        with pytest.raises(ValueError, match="2-qubit and a 3-qubit"):
            two + three
        with pytest.raises(ValueError, match="2-qubit and a 3-qubit"):
            two.compose(three)

    def test_invalid_labels_and_coefficients_are_refused(self):
        with pytest.raises(ValueError, match="invalid Pauli string"):
            PauliSum({"XQ": 1.0})
        with pytest.raises(ValueError, match="non-finite"):
            PauliSum({"XX": complex(np.inf, 0.0)})
        with pytest.raises(ValueError, match="equal length"):
            PauliSum({"XX": 1.0, "XXX": 1.0})

    def test_declared_width_must_match_the_labels(self):
        with pytest.raises(ValueError, match="register"):
            PauliSum({"XX": 1.0}, num_qubits=3)

    def test_algebra_preserves_the_width(self):
        op = PauliSum({"XY": 1.0}) + PauliSum({"YX": 1.0})
        assert op.num_qubits == 2
        assert (2.0 * op).num_qubits == 2
        assert op.compose(PauliSum.identity(2)).num_qubits == 2


class TestForceCapabilityGate:
    """Unsupported force configurations fail instead of returning a number."""

    @staticmethod
    def gate(**context):
        """Run the calculator's check against a stubbed solver."""
        from types import SimpleNamespace

        from carcara import Carcara

        calc = Carcara(method="adapt-vqe", basis="FAO",
                       force_method=context.pop("force_method", "rdm"),
                       measurement_provider=context.pop("provider", None))
        solver = SimpleNamespace(
            shots=context.pop("shots", 0),
            num_particles=context.pop("num_particles", (1, 1)),
            _gradient_context={
                "family": context.pop("family", "paw"),
                "integrals": SimpleNamespace(
                    kinetic=context.pop("kinetic", "fd"),
                    basis=context.pop("basis", []))})
        return calc._check_force_support(solver)

    def test_spectral_kinetic_is_refused(self):
        with pytest.raises(NotImplementedError, match="finite-difference"):
            self.gate(kinetic="spectral")

    def test_measured_energy_without_measured_rdms_is_refused(self):
        with pytest.raises(NotImplementedError, match="shots"):
            self.gate(shots=4096)

    def test_open_shell_is_refused_by_the_legacy_path(self):
        with pytest.raises(NotImplementedError, match="closed-shell"):
            self.gate(force_method="scf-response", num_particles=(2, 1))

    def test_the_rdm_gradient_handles_open_shells(self):
        assert self.gate(num_particles=(2, 1)) is None

    def test_complex_orbitals_warn_only_on_the_legacy_path(self):
        from types import SimpleNamespace

        p_function = SimpleNamespace(l=1)
        with pytest.warns(RuntimeWarning, match="real-arithmetic"):
            self.gate(force_method="scf-response", basis=[p_function])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert self.gate(basis=[p_function]) is None

    def test_a_supported_configuration_passes(self):
        from types import SimpleNamespace

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert self.gate(force_method="scf-response",
                             basis=[SimpleNamespace(l=0)]) is None

    def test_spectral_kinetic_is_refused_end_to_end(self):
        from ase import Atoms

        from carcara import Carcara

        atoms = Atoms("H2", positions=[[4, 4, 3.63], [4, 4, 4.37]],
                      cell=[8.0] * 3)
        atoms.calc = Carcara(method="adapt-vqe", basis="FAO", h=0.5,
                             kinetic="spectral", verbose=False, profile=False,
                             max_iterations=1)
        with pytest.raises(NotImplementedError, match="finite-difference"):
            atoms.get_forces()


class TestStaleResultsAreCleared:
    def test_an_energy_only_step_clears_the_force_breakdown(self):
        from ase import Atoms

        from carcara import Carcara

        atoms = Atoms("H2", positions=[[4, 4, 3.63], [4, 4, 4.37]],
                      cell=[8.0] * 3)
        atoms.calc = Carcara(method="adapt-vqe", basis="FAO", h=0.5,
                             verbose=False, profile=False, max_iterations=2)
        atoms.get_forces()
        assert atoms.calc.force_result is not None
        moved = atoms.copy()
        moved.positions[1, 2] += 0.05
        moved.calc = atoms.calc
        moved.get_potential_energy()
        assert atoms.calc.force_result is None


class TestHermiticityCheck:
    def test_a_non_hermitian_hamiltonian_is_refused(self):
        from carcara.algorithms import ADAPTVQE
        from carcara.core.mapping import PauliSum

        bad = PauliSum({"ZZ": 1.0, "XX": 0.5j})
        with pytest.raises(ValueError, match="not Hermitian"):
            ADAPTVQE(hamiltonian=bad, num_particles=(1, 1),
                     n_spatial_orbitals=1, verbose=False, profile=False)


class TestPseudoBasisOptions:
    def test_split_norm_may_differ_per_element(self):
        from carcara.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        name, options = resolve_basis({"Li": {"name": "PAW", "split_norm": 0.3},
                                       "H": {"name": "PAW",
                                             "split_norm": 0.15}})
        _family, merged = resolve_pseudo_basis(name, options, ["Li", "H"])
        assert merged["split_norm"] == {"Li": 0.3, "H": 0.15}

    def test_one_shared_value_stays_a_scalar(self):
        from carcara.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        name, options = resolve_basis({"Li": {"name": "PAW", "split_norm": 0.2},
                                       "H": {"name": "PAW",
                                             "split_norm": 0.2}})
        _family, merged = resolve_pseudo_basis(name, options, ["Li", "H"])
        assert merged["split_norm"] == 0.2

    def test_per_element_split_norm_reaches_the_basis(self):
        from carcara.pseudopotentials import pseudo_basis
        from carcara.pseudopotentials.paw import get_paw

        potentials = {"H": get_paw("H")}
        shared = pseudo_basis(["H"], [[0.0, 0.0, 0.0]], potentials, size="DZ",
                              split_norm=0.15)[0]
        split = pseudo_basis(["H"], [[0.0, 0.0, 0.0]], potentials, size="DZ",
                             split_norm={"H": 0.45})[0]
        # The second zeta is built from the split radius, so it must differ.
        assert not np.allclose(shared[1].table.values, split[1].table.values)

    def test_n_electrons_is_refused_with_a_pseudopotential_basis(self):
        from ase import Atoms

        from carcara.algorithms._hamiltonian_from_atoms import (
            build_basis_hamiltonian)

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[8.0] * 3)
        atoms.center()
        with pytest.raises(ValueError, match="n_electrons is not accepted"):
            build_basis_hamiltonian(atoms, "PAW", None, 0.4, 0, 2)
