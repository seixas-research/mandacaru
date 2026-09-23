# -*- coding: utf-8 -*-
# file: test/core/test_ccmh.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The structure-factor finite-size correction: :mod:`mandacaru.core.ccmh`.

Three checks carry this file.

* **``M(0)`` is the identity.**  The Fourier matrix elements at ``k = 0`` are
  the overlap, and the RDMs live in an orthonormal basis, so ``M(0)`` must be
  ``I`` to round-off.  It is the cheapest test that the AO-to-MO chain used for
  the structure factor is the same one the Hamiltonian was written in; get the
  Loewdin or the molecular-orbital rotation wrong and this fails immediately
  while every other number still looks plausible.
* **The contraction matches direct operator algebra.**  ``N S(k)`` is also
  ``|| rho_k^dagger |Psi> ||^2`` with ``rho_k`` built as a ``Fermion``, mapped
  to qubits and applied to the state vector.  That route shares no code with
  the RDM contraction, and the two agree to exactly ``0.0``.  Note ``S -> 1``
  at large ``k`` is *not* tested: it needs a complete basis, and a minimal one
  decays instead (0.105 at 1.91 Bohr^-1 here).
* **The quadratic regime is not assumed.**  The whole correction is the
  ``k -> 0`` limit of ``S(k)/k^2``, which only exists where ``S ~ k^2`` has
  set in.  A cell surrounded by vacuum never gets there, and the code has to
  say so rather than extrapolate a trend that has not started.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms.pseudo_forces import spatial_rdms
from mandacaru.core.ccmh import (QUADRATIC_REGIME_SPREAD, CCMHCorrection,
                                 ccmh_correction, structure_factor)
from mandacaru.core.mpc import FiniteSizeCorrection
from mandacaru.optimizers import Optimizer

SLSQP = Optimizer(method="SLSQP", maxiter=1000, tol=1e-12)


@pytest.fixture(scope="module")
def solved_chain():
    """A solved 4-qubit periodic H chain, shared by every test here."""
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=np.diag([2.0, 10.0, 10.0]), pbc=[True, False, False])
    atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                           kpts={"size": (2, 1, 1), "gamma": True},
                           basis="HAO", h=0.35, trace=False, optimizer=SLSQP)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def chain_structure(solved_chain):
    solver = solved_chain.calc.solver
    integrals = solver._gradient_context["integrals"]
    gamma, gamma2 = solved_chain.calc._state_rdms(solver, two_body=True)
    D, Gamma = spatial_rdms(gamma, gamma2, len(integrals.basis))
    return structure_factor(integrals, D, Gamma, limit=2)


class TestTheStructureFactor:
    def test_the_transform_chain_matches_the_hamiltonian(self, chain_structure):
        """``M(0)`` is the overlap, and the RDM basis is orthonormal."""
        assert chain_structure.overlap_error < 1e-12

    def test_it_is_real_and_non_negative(self, chain_structure):
        assert np.all(np.isfinite(chain_structure.values))
        assert chain_structure.values.min() > -1e-10

    def test_it_matches_direct_operator_algebra(self, solved_chain):
        r"""The contraction against an independent route through the state.

        ``N S(k) = || rho_k^dagger |Psi> ||^2`` with
        ``rho_k = sum_pq M_pq a^dag_p a_q`` built as a ``Fermion``, mapped to
        qubits and applied to the state vector.  That shares no code with the
        RDM contraction, so it catches a wrong index pairing or a dropped
        exchange term.  Measured agreement: exactly 0.0.
        """
        from scipy import fft as sfft

        from mandacaru.core.ccmh import _reciprocal_vectors
        from mandacaru.core.mapping import Fermion

        solver = solved_chain.calc.solver
        integrals = solver._gradient_context["integrals"]
        state = solver.ansatz.state(solver.result.optimal_parameters)
        if state.shape[0] != 2 ** solver.n_qubits:
            pytest.skip("the ansatz works in a particle-number sector here")

        # Its own structure factor at the same `limit`: _reciprocal_vectors
        # enumerates a different first triple for a different span, so the
        # index only lines up when both use the same one.
        gamma, gamma2 = solved_chain.calc._state_rdms(solver, two_body=True)
        M = len(integrals.basis)
        D, Gamma = spatial_rdms(gamma, gamma2, M)
        reference = structure_factor(integrals, D, Gamma, limit=1)
        A = integrals._lowdin_x() @ integrals.mo_coefficients
        grid, psi = integrals.grid, integrals._engine._psi
        shape = tuple(grid.shape)
        triples, kvectors = _reciprocal_vectors(integrals.cell, 1)

        node = tuple(int(m) % n for m, n in zip(triples[0], shape))
        Mao = np.empty((len(psi), len(psi)), dtype=complex)
        for a in range(len(psi)):
            for b in range(len(psi)):
                spectrum = sfft.fftn((np.conj(psi[a]) * psi[b]).reshape(shape))
                Mao[a, b] = spectrum[node] * grid.dV
        Mk = A.conj().T @ Mao @ A

        operator = Fermion()
        for p in range(M):
            for q in range(M):
                for spin in (0, 1):
                    operator = operator + (Fermion.creation(p + spin * M)
                                           * Fermion.annihilation(q + spin * M)
                                           * Mk[p, q])
        rho = operator.map_to_qubits(method=solver.mapping,
                                     n_modes=2 * M).to_sparse_matrix()
        applied = rho.conj().T @ state
        direct = float(np.real(np.vdot(applied, applied)))
        direct /= reference.n_electrons

        assert reference.kvectors[0] == pytest.approx(kvectors[0], abs=1e-12)
        assert reference.values[0] == pytest.approx(direct, abs=1e-12)

    def test_a_finite_basis_decays_at_large_k(self, chain_structure):
        """``S -> 1`` needs a complete basis; a minimal one cannot get there.

        The matrix elements die once ``k`` exceeds the inverse orbital size,
        so ``S`` dies with them -- 0.105 at 1.91 Bohr^-1 here.  Documented
        rather than asserted as a bug, because the correction reads only the
        small-``k`` end, which a localized basis does represent.
        """
        order = np.argsort(chain_structure.magnitudes)
        assert chain_structure.values[order][-1] < chain_structure.values[order][0]

    def test_it_normalizes_on_the_electron_count(self, chain_structure):
        assert chain_structure.n_electrons == pytest.approx(2.0)

    def test_gamma_is_excluded(self, chain_structure):
        """``k = 0`` is the term the discrete sum omits; it is not a datum."""
        assert np.all(chain_structure.magnitudes > 1e-12)

    def test_shells_group_symmetry_equivalent_vectors(self, chain_structure):
        k, s = chain_structure.shells(3)
        assert k.size == s.size <= 3
        assert np.all(np.diff(k) > 0)      # strictly increasing shells


class TestTheQuadraticRegimeIsChecked:
    """The limit exists only where ``S ~ k^2``; assuming it is the failure."""

    def test_a_chain_in_vacuum_is_flagged(self, chain_structure):
        """Vacuum does not suppress long-wavelength density fluctuations.

        Measured on this cell: the three smallest shells give ``S`` of about
        1.8, 1.6 and 1.3, nowhere near ``k^2``, and ``S/k^2`` swings from 16
        to 2.8.  Extrapolating that intercept is meaningless, so it warns.
        """
        with pytest.warns(RuntimeWarning, match="quadratic regime"):
            result = ccmh_correction(chain_structure, volume=2699.3, n_cells=2)
        assert not result.reliable
        assert result.quadratic_spread > QUADRATIC_REGIME_SPREAD

    def test_a_flat_ratio_is_not_flagged(self):
        """A synthetic ``S = c k^2`` must pass without a warning."""
        from mandacaru.core.ccmh import StructureFactor

        k = np.array([0.3, 0.4, 0.5, 0.6])
        vectors = np.column_stack([k, np.zeros_like(k), np.zeros_like(k)])
        structure = StructureFactor(
            kvectors=vectors, values=0.25 * k ** 2, magnitudes=k,
            n_electrons=4.0, overlap_error=0.0)
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ccmh_correction(structure, volume=1000.0, n_cells=1)
        assert [w for w in caught if "quadratic" in str(w.message)] == []
        assert result.reliable
        assert result.limit == pytest.approx(0.25, rel=1e-8)

    def test_an_empty_structure_is_refused(self):
        from mandacaru.core.ccmh import StructureFactor

        empty = StructureFactor(kvectors=np.empty((0, 3)),
                                values=np.empty(0), magnitudes=np.empty(0),
                                n_electrons=1.0, overlap_error=0.0)
        with pytest.raises(ValueError, match="no reciprocal-lattice shells"):
            ccmh_correction(empty, volume=100.0)


class TestTheCorrectionRecord:
    def test_the_rpa_limit_is_the_plasma_formula(self):
        """``rpa_limit = 1 / (2 omega_p)`` with ``omega_p = sqrt(4 pi n)``."""
        from mandacaru.core.ccmh import StructureFactor

        k = np.array([0.3, 0.4, 0.5])
        structure = StructureFactor(
            kvectors=np.column_stack([k, 0 * k, 0 * k]),
            values=0.1 * k ** 2, magnitudes=k, n_electrons=8.0,
            overlap_error=0.0)
        result = ccmh_correction(structure, volume=1000.0, n_cells=8)
        density = 8.0 / 1000.0
        omega_p = np.sqrt(4 * np.pi * density)
        assert result.density == pytest.approx(density)
        assert result.plasma_frequency == pytest.approx(omega_p)
        assert result.rpa_limit == pytest.approx(1.0 / (2 * omega_p))

    def test_the_scopes_are_kept_apart(self):
        from mandacaru.core.ccmh import StructureFactor

        k = np.array([0.3, 0.4, 0.5])
        structure = StructureFactor(
            kvectors=np.column_stack([k, 0 * k, 0 * k]),
            values=0.1 * k ** 2, magnitudes=k, n_electrons=8.0,
            overlap_error=0.0)
        result = ccmh_correction(structure, volume=1000.0, n_cells=8,
                                 energy_per_cell=-1.5)
        assert result.correction_per_cell == pytest.approx(result.correction / 8)
        assert result.corrected_energy_per_cell == pytest.approx(
            -1.5 + result.correction_per_cell)

    def test_the_summary_names_both_limits(self, chain_structure):
        with pytest.warns(RuntimeWarning):
            result = ccmh_correction(chain_structure, volume=2699.3, n_cells=2)
        text = result.summary()
        assert "lim S/k^2" in text and "RPA" in text


class TestTheSchemesAreBothReachable:
    """Both corrections come from one entry point, so they can be compared."""

    def test_mpc_is_the_current_default(self, solved_chain):
        assert isinstance(solved_chain.calc.finite_size_correction(),
                          FiniteSizeCorrection)

    def test_ccmh_is_selectable(self, solved_chain):
        with pytest.warns(RuntimeWarning, match="quadratic regime"):
            result = solved_chain.calc.finite_size_correction("ccmh")
        assert isinstance(result, CCMHCorrection)

    def test_they_disagree_which_is_the_point(self, solved_chain):
        """Different physics, different numbers -- that is why both exist."""
        mpc = solved_chain.calc.finite_size_correction("mpc")
        with pytest.warns(RuntimeWarning):
            ccmh = solved_chain.calc.finite_size_correction("ccmh")
        assert mpc.correction_per_cell != pytest.approx(
            ccmh.correction_per_cell, rel=1e-3)

    def test_an_unknown_scheme_is_refused_by_name(self, solved_chain):
        # This used to pass "kzk", which was then unimplemented.  It is a
        # scheme now, so the example had to become one that is not.
        with pytest.raises(ValueError, match="unknown finite-size scheme"):
            solved_chain.calc.finite_size_correction("nonesuch")

    def test_both_carry_the_same_reported_energy(self, solved_chain):
        mpc = solved_chain.calc.finite_size_correction("mpc")
        with pytest.warns(RuntimeWarning):
            ccmh = solved_chain.calc.finite_size_correction("ccmh")
        assert mpc.energy_per_cell == pytest.approx(ccmh.energy_per_cell)
        assert mpc.n_cells == ccmh.n_cells == 2
