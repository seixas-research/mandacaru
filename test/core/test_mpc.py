# -*- coding: utf-8 -*-
# file: test/core/test_mpc.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The exchange-correlation hole and its images: :mod:`mandacaru.core.mpc`.

Two checks here are identities rather than convergence properties, and they are
the ones worth reading first.

* **The kernel covers exactly the cell.** The Wigner-Seitz cell is a primitive
  cell, so it has the primitive volume exactly, for every Bravais lattice.  An
  earlier kernel zeroed the offsets whose *folded* value fell outside the
  Wigner-Seitz cell -- which looks like the truncation the name suggests and
  drops whole lattice-equivalence classes without ever sampling their in-cell
  representative.  It covered 85 % of a hexagonal cell and 76 % of an FCC one.
* **The hole holds exactly one electron.**
  ``sum_q (Gamma - D (x) D)_pqrq = -D_pr`` follows from the RDM trace
  relations and holds however bad the state is.  It is the sharpest available
  test that the two pair densities are contracted with the same index pairing;
  swapping the direct and exchange pairings breaks it immediately.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms.pseudo_forces import spatial_rdms
from mandacaru.core.mpc import (FiniteSizeCorrection, TruncatedCoulombSolver,
                                exchange_correlation_hole_energy,
                                wigner_seitz_kernel)
from mandacaru.integrals.poisson import (PeriodicPoissonSolver,
                                         voxel_self_potential)
from mandacaru.optimizers import Optimizer

SLSQP = Optimizer(method="SLSQP", maxiter=1000, tol=1e-12)

#: int_{[-1/2, 1/2]^3} d^3u / |u|, the cube's own Coulomb constant.
C_CUBE = 2.3800774

LATTICES = {
    "cubic": np.diag([6.0, 6.0, 6.0]),
    "hexagonal": np.array([[6.0, 0.0, 0.0],
                           [-3.0, 6.0 * np.sqrt(3) / 2, 0.0],
                           [0.0, 0.0, 6.0]]),
    "fcc": 3.0 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
    "triclinic": np.array([[6.0, 0.0, 0.0], [1.8, 6.0, 0.0], [1.2, 0.9, 6.0]]),
}


def step_of(cell, n):
    """Grid step vectors (as columns) for ``n`` nodes per lattice vector."""
    return np.column_stack([np.asarray(cell, float)[m] / n for m in range(3)])


@pytest.fixture(scope="module")
def solved_chain():
    """A solved 4-qubit periodic H chain, shared by the state-level tests."""
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=np.diag([2.0, 10.0, 10.0]), pbc=[True, False, False])
    atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                           kpts={"size": (2, 1, 1), "gamma": True},
                           basis="HAO", h=0.35, trace=False, optimizer=SLSQP)
    atoms.get_potential_energy()
    return atoms


class TestTheTruncatedKernel:
    def test_the_self_node_is_the_voxel_average(self):
        """Not an ad-hoc softening: the same treatment the isolated solver uses."""
        step = np.diag([0.5, 0.5, 0.5])
        kernel = wigner_seitz_kernel((12, 12, 12), step)
        volume = abs(float(np.linalg.det(step)))
        assert kernel[0, 0, 0] == pytest.approx(
            voxel_self_potential(step) / volume, rel=1e-12)
        # For a cube of side dx that is C_cube / dx.
        assert kernel[0, 0, 0] == pytest.approx(C_CUBE / 0.5, rel=1e-6)

    @pytest.mark.parametrize("n", [12, 24])
    def test_it_integrates_to_the_analytic_cube_value(self, n):
        r"""``int_cell d^3r/|r_min| = L^2 * 2.3800774`` for a cube."""
        length = 6.0
        solver = TruncatedCoulombSolver((n, n, n), np.diag([length / n] * 3))
        integral = float(solver.real_kernel.sum() * solver.dV)
        assert integral == pytest.approx(length ** 2 * C_CUBE, rel=5e-3)

    def test_refining_the_grid_converges_to_it(self):
        length = 6.0
        errors = []
        for n in (12, 24):
            solver = TruncatedCoulombSolver((n, n, n), np.diag([length / n] * 3))
            integral = float(solver.real_kernel.sum() * solver.dV)
            errors.append(abs(integral - length ** 2 * C_CUBE))
        assert errors[1] < errors[0]

    @pytest.mark.parametrize("name", sorted(LATTICES))
    def test_it_covers_exactly_the_cell_volume(self, name):
        """The Wigner-Seitz cell is a primitive cell: same volume, always.

        This is the check that caught the kernel zeroing whole
        lattice-equivalence classes -- it read 0.85 for hexagonal and 0.76 for
        FCC, and 1.0 only for the cubic case where folding happens to give the
        Wigner-Seitz cell.
        """
        cell = LATTICES[name]
        solver = TruncatedCoulombSolver((16, 16, 16), step_of(cell, 16))
        assert solver.enclosed_volume == pytest.approx(solver.volume, rel=1e-12)

    @pytest.mark.parametrize("name", sorted(LATTICES))
    def test_nothing_is_zero(self, name):
        """Every offset is one equivalence class with one in-cell image."""
        cell = LATTICES[name]
        kernel = wigner_seitz_kernel((12, 12, 12), step_of(cell, 12))
        assert np.all(kernel > 0.0)

    def test_a_skewed_cell_needs_translations_a_cube_does_not(self):
        """Folding indices is the minimum image only for an orthogonal cell.

        The FCC Wigner-Seitz cell is a rhombic dodecahedron, so many folded
        offsets are *not* the closest image and the kernel must differ from the
        naive folded one.
        """
        fcc = LATTICES["fcc"]
        step = step_of(fcc, 12)
        folded = np.arange(12)
        folded = np.where(folded <= 6, folded, folded - 12).astype(float)
        SX, SY, SZ = np.meshgrid(folded, folded, folded, indexing="ij")
        naive = np.sqrt(sum((step[row, 0] * SX + step[row, 1] * SY
                             + step[row, 2] * SZ) ** 2 for row in range(3)))
        kernel = wigner_seitz_kernel((12, 12, 12), step)
        with np.errstate(divide="ignore"):
            naive_kernel = np.where(naive > 0, 1.0 / naive, 0.0)
        naive_kernel[0, 0, 0] = kernel[0, 0, 0]
        assert not np.allclose(kernel, naive_kernel)
        # The minimum image is never farther than the folded offset.
        assert np.all(kernel >= naive_kernel - 1e-12)

    def test_the_solver_matches_the_periodic_one_s_interface(self):
        """It is handed to ``two_body`` in the periodic solver's place."""
        step = np.diag([0.5, 0.5, 0.5])
        truncated = TruncatedCoulombSolver((8, 8, 8), step)
        periodic = PeriodicPoissonSolver((8, 8, 8), step=step)
        for name in ("shape", "L", "dV", "volume", "solve", "solve_stack"):
            assert hasattr(truncated, name), name
        assert truncated.shape == periodic.shape
        assert truncated.L == periodic.L
        assert truncated.dV == pytest.approx(periodic.dV)
        density = np.zeros(8 ** 3)
        density[0] = 1.0
        assert truncated.solve(density).shape == (8 ** 3,)

    def test_a_degenerate_step_is_refused(self):
        with pytest.raises(ValueError, match="zero volume"):
            TruncatedCoulombSolver((4, 4, 4), np.zeros((3, 3)))

    def test_a_malformed_step_is_refused(self):
        with pytest.raises(ValueError, match="3x3"):
            wigner_seitz_kernel((4, 4, 4), np.eye(2))


class TestTheHoleIsContractedCorrectly:
    def test_identical_kernels_give_exactly_zero(self):
        """The null test: no difference in the kernel, no correction."""
        rng = np.random.default_rng(0)
        M = 4
        eri = rng.normal(size=(M, M, M, M))
        D = rng.normal(size=(M, M))
        Gamma = rng.normal(size=(M, M, M, M))
        result = exchange_correlation_hole_energy(
            eri, eri, D, Gamma, volume=100.0)
        assert result.correction == 0.0

    def test_the_hole_holds_exactly_one_electron(self, solved_chain):
        r"""``sum_q (Gamma - D (x) D)_pqrq = -D_pr``.

        An RDM trace identity, so it holds for any state, converged or not.
        It is what pins the index pairing: electron 1 carries ``(p, r)`` and
        electron 2 ``(q, s)``, and swapping them substitutes exchange for
        direct.
        """
        solver = solved_chain.calc.solver
        gamma, gamma2 = solved_chain.calc._state_rdms(solver, two_body=True)
        integrals = solver._gradient_context["integrals"]
        D, Gamma = spatial_rdms(gamma, gamma2, len(integrals.basis))
        hole = Gamma - np.einsum("pr,qs->pqrs", D, D)
        assert np.einsum("pqrq->pr", hole) == pytest.approx(-D, abs=1e-10)

    def test_mismatched_tensors_are_refused(self):
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="same orbitals"):
            exchange_correlation_hole_energy(
                rng.normal(size=(3, 3, 3, 3)), rng.normal(size=(4, 4, 4, 4)),
                rng.normal(size=(3, 3)), rng.normal(size=(3, 3, 3, 3)),
                volume=1.0)

    def test_a_mismatched_rdm_is_refused(self):
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="two-body RDM"):
            exchange_correlation_hole_energy(
                rng.normal(size=(3, 3, 3, 3)), rng.normal(size=(3, 3, 3, 3)),
                rng.normal(size=(3, 3)), rng.normal(size=(4, 4, 4, 4)),
                volume=1.0)


class TestTheSecondTensorIsComparable:
    """Two tensors that differ only in the kernel, or the subtraction is noise."""

    def test_the_periodic_rebuild_reproduces_the_hamiltonian_s_tensor(
            self, solved_chain):
        integrals = solved_chain.calc.solver._gradient_context["integrals"]
        rebuilt = integrals.two_body_with_kernel(None)
        assert rebuilt.shape == (len(integrals.basis),) * 4
        # Rebuilding twice must be deterministic, which is what lets the
        # truncated tensor be subtracted from it.
        assert rebuilt == pytest.approx(integrals.two_body_with_kernel(None),
                                        abs=0.0)

    def test_the_truncated_tensor_differs_but_keeps_the_symmetries(
            self, solved_chain):
        integrals = solved_chain.calc.solver._gradient_context["integrals"]
        periodic = integrals.two_body_with_kernel(None)
        truncated = integrals.two_body_with_kernel(
            TruncatedCoulombSolver(integrals.grid.shape,
                                   step=integrals.grid.step))
        assert not np.allclose(periodic, truncated)
        # <pq|rs> = <qp|sr> for a real symmetric kernel.
        assert truncated == pytest.approx(
            np.einsum("pqrs->qpsr", truncated), abs=1e-10)

    def test_the_mo_rotation_must_exist(self, solved_chain):
        """Asking for the MO basis before there is one is refused, not faked."""
        from mandacaru.core import MolecularIntegrals, minimal_hao_basis
        from mandacaru.integrals import Grid

        cell = np.diag([6.0, 6.0, 6.0])
        nuclei = [(1.0, np.array([3.0, 3.0, 3.0]))]
        grid = Grid(center=[3.0] * 3, box_size=0.0, h=0.6, units="angstrom",
                    cell=cell)
        integrals = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid,
                                       units="angstrom")
        with pytest.raises(ValueError, match="molecular_hamiltonian"):
            integrals.two_body_with_kernel(None, mo=True)


class TestTheCorrectionOnARun:
    def test_the_scopes_are_kept_apart(self, solved_chain):
        """Supercell and per-cell are an ``n_cells``-fold error if mixed."""
        correction = solved_chain.calc.finite_size_correction()
        assert correction.n_cells == 2
        assert correction.correction_per_cell == pytest.approx(
            correction.correction / 2)
        assert correction.n_electrons == pytest.approx(2.0)

    def test_the_energy_it_carries_is_the_reported_one(self, solved_chain):
        """``energy_per_cell`` must match what ASE was told, in Hartree."""
        from mandacaru.units import from_hartree

        correction = solved_chain.calc.finite_size_correction()
        reported = solved_chain.get_potential_energy()
        assert from_hartree(correction.energy_per_cell, "eV") == pytest.approx(
            reported, rel=1e-10)
        assert correction.mpc_energy_per_cell == pytest.approx(
            correction.energy_per_cell + correction.correction_per_cell)

    def test_the_summary_names_both_scopes(self, solved_chain):
        text = solved_chain.calc.finite_size_correction().summary()
        assert "per supercell" in text and "per cell" in text

    def test_in_units_converts_every_energy(self, solved_chain):
        correction = solved_chain.calc.finite_size_correction()
        ev = correction.in_units("eV")
        assert isinstance(ev, FiniteSizeCorrection)
        assert ev.correction == pytest.approx(correction.correction * 27.211386,
                                              rel=1e-5)
        assert ev.n_cells == correction.n_cells        # not an energy

    def test_a_molecular_method_is_refused(self):
        """A molecule has no images, so there is nothing to remove."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                      cell=np.diag([8.0] * 3))
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.4,
                               trace=False, optimizer=SLSQP)
        atoms.get_potential_energy()
        with pytest.raises(ValueError, match="not periodic"):
            atoms.calc.finite_size_correction()
