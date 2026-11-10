# -*- coding: utf-8 -*-
# file: test/algorithms/test_finite_size_vqe.py

"""Comprehensive test suite for finite-size corrections (KZK, MPC, CCMH) with VQE and ADAPT-VQE."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from mandacaru import Mandacaru
from mandacaru.core.ccmh import CCMHCorrection
from mandacaru.core.kzk import KZKCorrection
from mandacaru.core.mpc import FiniteSizeCorrection, spatial_rdms
from mandacaru.optimizers import Optimizer

SLSQP = Optimizer(method="SLSQP", maxiter=500, tol=1e-10)


@pytest.fixture(scope="module")
def periodic_h2_vqe():
    """Periodic H2 in a 6x6x6 Angstrom supercell solved with VQE."""
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=np.diag([6.0, 6.0, 6.0]),
        pbc=[True, True, True],
    )
    atoms.calc = Mandacaru(
        method="vqe",
        crystal_orbitals="real-space",
        basis="sto-3g",
        grid_spacing=0.4,
        trace=False,
        optimizer=SLSQP,
    )
    # Using molecular integration with box supercell
    atoms.calc.calculate(atoms, properties=("energy",))
    return atoms


@pytest.fixture(scope="module")
def periodic_h2_adapt():
    """Periodic H2 in a 6x6x6 Angstrom supercell solved with ADAPT-VQE."""
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=np.diag([6.0, 6.0, 6.0]),
        pbc=[True, True, True],
    )
    atoms.calc = Mandacaru(
        method="adapt-vqe",
        crystal_orbitals="real-space",
        basis="sto-3g",
        grid_spacing=0.4,
        trace=False,
        optimizer=SLSQP,
    )
    atoms.calc.calculate(atoms, properties=("energy",))
    return atoms


class TestFiniteSizeVQE:
    """Tests for VQE finite-size corrections."""

    def test_vqe_rdms_satisfy_exact_hole_sum_rule(self, periodic_h2_vqe):
        calc = periodic_h2_vqe.calc
        state = np.asarray(calc.result.state(), dtype=np.complex128)
        gamma, gamma2 = calc.problem.rdms(state, two_body=True)
        D, Gamma = spatial_rdms(gamma, gamma2, calc.problem.n_spatial_orbitals)
        assert np.isclose(np.trace(D).real, 2.0, atol=1e-6)
        hole = Gamma - np.einsum("pr,qs->pqrs", D, D)
        contracted = np.einsum("pqrq->pr", hole)
        diff = np.max(np.abs(contracted - (-D)))
        assert diff < 1e-12, f"sum_q (Gamma - D (x) D)_pqrq != -D_pr; max error {diff}"

    def test_vqe_kzk_correction(self, periodic_h2_vqe):
        calc = periodic_h2_vqe.calc
        corr = calc.finite_size_correction(scheme="kzk")
        assert isinstance(corr, KZKCorrection)
        assert np.isclose(corr.n_electrons, 2.0, atol=1e-4)
        assert corr.length > 0.0
        assert np.isfinite(corr.correction)
        assert corr.correction_per_cell == corr.correction

    def test_vqe_mpc_correction(self, periodic_h2_vqe):
        calc = periodic_h2_vqe.calc
        corr = calc.finite_size_correction(scheme="mpc")
        assert isinstance(corr, FiniteSizeCorrection)
        assert np.isclose(corr.n_electrons, 2.0, atol=1e-4)
        assert corr.volume > 0.0
        assert np.isfinite(corr.correction)
        assert "per supercell" in corr.summary()

    def test_vqe_ccmh_correction(self, periodic_h2_vqe):
        calc = periodic_h2_vqe.calc
        corr = calc.finite_size_correction(scheme="ccmh", shells=3)
        assert isinstance(corr, CCMHCorrection)
        assert np.isclose(corr.n_electrons, 2.0, atol=1e-4)
        assert np.isfinite(corr.correction)
        assert np.isfinite(corr.limit)


class TestFiniteSizeADAPTVQE:
    """Tests for ADAPT-VQE finite-size corrections."""

    def test_adapt_vqe_rdms_satisfy_exact_hole_sum_rule(self, periodic_h2_adapt):
        calc = periodic_h2_adapt.calc
        state = np.asarray(calc.result.state(), dtype=np.complex128)
        gamma, gamma2 = calc.problem.rdms(state, two_body=True)
        D, Gamma = spatial_rdms(gamma, gamma2, calc.problem.n_spatial_orbitals)
        assert np.isclose(np.trace(D).real, 2.0, atol=1e-6)
        hole = Gamma - np.einsum("pr,qs->pqrs", D, D)
        contracted = np.einsum("pqrq->pr", hole)
        diff = np.max(np.abs(contracted - (-D)))
        assert diff < 1e-12, f"sum_q (Gamma - D (x) D)_pqrq != -D_pr; max error {diff}"

    def test_adapt_vqe_all_three_schemes(self, periodic_h2_adapt):
        calc = periodic_h2_adapt.calc
        kzk = calc.finite_size_correction("kzk")
        assert isinstance(kzk, KZKCorrection)
        assert np.isclose(kzk.n_electrons, 2.0, atol=1e-4)

        mpc = calc.finite_size_correction("mpc")
        assert isinstance(mpc, FiniteSizeCorrection)
        assert np.isclose(mpc.n_electrons, 2.0, atol=1e-4)

        ccmh = calc.finite_size_correction("ccmh")
        assert isinstance(ccmh, CCMHCorrection)
        assert np.isclose(ccmh.n_electrons, 2.0, atol=1e-4)


class TestFiniteSizeValidation:
    """Validation and boundary tests."""

    def test_non_periodic_system_raises_value_error(self):
        atoms = molecule("H2")
        atoms.cell = [6.0, 6.0, 6.0]
        calc = Mandacaru(method="vqe", basis="sto-3g", grid_spacing=0.4, trace=False, optimizer=SLSQP)
        atoms.calc = calc
        atoms.get_potential_energy()
        with pytest.raises(ValueError, match="not periodic"):
            calc.finite_size_correction("kzk")

    def test_dft_method_raises_value_error(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([4.0] * 3), pbc=True)
        calc = Mandacaru(method="dft", basis="HAO")
        atoms.calc = calc
        with pytest.raises(ValueError, match="not supported for method='dft'"):
            calc.finite_size_correction("kzk")

    def test_unknown_scheme_raises_value_error(self, periodic_h2_vqe):
        with pytest.raises(ValueError, match="unknown finite-size scheme"):
            periodic_h2_vqe.calc.finite_size_correction("unknown_scheme")
