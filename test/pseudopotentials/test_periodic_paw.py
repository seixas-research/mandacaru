# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_periodic_paw.py

# This code is part of Mandacaru.
# MIT License

"""PAW-LCAO in a crystal: Bloch sums, k-point reduction, lattice energies."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mandacaru.algorithms.periodic_dft import PeriodicKohnSham
from mandacaru.integrals import exchange_correlation as xc_grid
from mandacaru.integrals import reciprocal as rc
from mandacaru.pseudopotentials import periodic_paw as pp

#: Diamond silicon on a grid commensurate with both the primitive cell and
#: its 2x1x1 supercell (10 nodes per primitive lattice vector).
SILICON = bulk("Si", "diamond", a=5.43)
SILICON_H = float(np.linalg.norm(SILICON.cell[0])) / 10
SILICON_BASIS = {"size": "SZ", "filter": 200}


def _energy(atoms, h, options, kpts=None):
    """Extrapolated energy (Hartree per cell) of a periodic ``atoms``."""
    crystal, context = pp.build_crystal(atoms, h, options, kpts=kpts)
    constant = (sum(d.one_center_energy for d in crystal.datasets)
                + sum(xc_grid.core_correction_offset(d)
                      for d in crystal.datasets))
    result = PeriodicKohnSham(crystal, context["n_electrons"], "lda",
                              smearing={"method": "fermi-dirac",
                                        "width": 0.01},
                              constant=constant).run()
    assert result.converged
    return result


class TestKPoints:
    def test_a_gamma_centered_even_mesh_is_its_own_time_reverse(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            monkhorst_pack_kpts
        _size, _gamma, mesh = monkhorst_pack_kpts({"size": (2, 2, 2),
                                                   "gamma": True})
        reduced, weights = pp.time_reversal_reduce(mesh)
        assert len(reduced) == 8 and np.allclose(weights, 1 / 8)

    def test_an_odd_mesh_pairs_k_with_minus_k(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            monkhorst_pack_kpts
        _size, _gamma, mesh = monkhorst_pack_kpts((3, 3, 3))
        reduced, weights = pp.time_reversal_reduce(mesh)
        assert len(reduced) == 14
        assert weights.sum() == pytest.approx(1.0)
        gamma = np.argmin(np.linalg.norm(reduced, axis=1))
        assert np.allclose(reduced[gamma], 0.0)
        assert weights[gamma] == pytest.approx(1 / 27)       # its own partner


class TestBlochSums:
    def test_a_lattice_translation_multiplies_by_the_phase(self):
        crystal, _context = pp.build_crystal(SILICON, SILICON_H,
                                             SILICON_BASIS)
        A = crystal.lattice
        k = rc.reciprocal_vectors(A) @ np.array([0.25, -0.5, 0.125])
        points = (np.array([0.3, 1.1]), np.array([0.7, -0.4]),
                  np.array([1.9, 0.2]))
        shifted = tuple(points[i] + A[i, 0] for i in range(3))
        center = np.zeros(3)
        here = pp.bloch_values(crystal.basis, A, k[None], points, center, 12.0)
        there = pp.bloch_values(crystal.basis, A, k[None], shifted, center,
                                12.0)
        assert np.allclose(there, np.exp(1j * k @ A[:, 0]) * here, atol=1e-12)

    def test_a_support_is_where_the_table_vanishes_not_where_it_ends(self):
        crystal, _context = pp.build_crystal(SILICON, SILICON_H,
                                             SILICON_BASIS)
        for function in crystal.basis:
            assert pp._support(function) < 20.0


class TestEnergies:
    def test_a_mesh_equals_its_supercell(self):
        """Born-von Karman: a 2x1x1 mesh is the 2x1x1 supercell at Gamma.

        Every term -- Bloch phases, projections, short-range spheres,
        lattice electrostatics -- has to agree for this to hold per cell.
        """
        mesh = _energy(SILICON, SILICON_H, SILICON_BASIS,
                       kpts={"size": (2, 1, 1), "gamma": True})
        supercell = _energy(SILICON.repeat((2, 1, 1)), SILICON_H,
                            SILICON_BASIS)
        assert mesh.extrapolated_energy == pytest.approx(
            supercell.extrapolated_energy / 2, abs=1e-7)

    def test_a_molecule_in_a_box_is_the_molecular_limit(self):
        """H2 at Gamma in an 8 Angstrom cell: the converged molecular energy.

        The molecular PAW-LCAO-DZP LDA energy, with the spectral kinetic
        operator, extrapolates as h^2 from h = 0.25 ... 0.10 Angstrom to
        -1.1269 Ha; the periodic, Fourier-filtered one gives it at h = 0.25
        (HISTORY.md, 2026-10-01).
        """
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[8.0] * 3, pbc=True)
        atoms.center()
        result = _energy(atoms, 0.25, {"size": "DZP"})
        assert result.extrapolated_energy == pytest.approx(-1.1269, abs=5e-4)


def _grid(atoms, nodes):
    from mandacaru.integrals import Grid
    cell = np.asarray(atoms.cell)
    return Grid(center=0.5 * cell.sum(axis=0), box_size=0.0,
                h=float(np.linalg.norm(cell[0])) / nodes, units="angstrom",
                cell=cell, periodic=True)


class TestGridSymmetry:
    def test_the_grid_decides_which_operations_survive(self):
        """Diamond's non-symmorphic quarter translations land on 8 or 12
        nodes per lattice vector, not on 10: there only the 24 operations
        without them are kept."""
        assert pp.grid_symmetry(SILICON, _grid(SILICON, 8)).n_operations == 48
        kept = pp.grid_symmetry(SILICON, _grid(SILICON, 10))
        assert kept.n_operations == 24 and kept.n_space_group == 48
        assert np.allclose(kept.info.translations % 1.0, 0.0)

    def test_the_kept_operations_form_a_group(self):
        symmetry = pp.grid_symmetry(SILICON, _grid(SILICON, 10))
        perms = {tuple(p) for p in symmetry.permutations}
        for a in symmetry.permutations[:6]:
            for b in symmetry.permutations[:6]:
                assert tuple(a[b]) in perms

    def test_harmonic_rotations_are_unitary_and_compose(self):
        symmetry = pp.grid_symmetry(SILICON, _grid(SILICON, 8))
        for rotation in symmetry.rotations:
            for L, T in rotation.items():
                assert np.allclose(T.conj().T @ T, np.eye(2 * L + 1))

    def test_symmetrizing_is_a_projection(self):
        symmetry = pp.grid_symmetry(SILICON, _grid(SILICON, 8))
        rng = np.random.default_rng(2)
        field = rng.normal(size=8 ** 3)
        once = symmetry.field(field)
        assert np.allclose(symmetry.field(once), once)
        assert once.sum() == pytest.approx(field.sum())
        q = {(a, L, M): complex(rng.normal(), rng.normal())
             for a in range(2) for L in range(3) for M in range(-L, L + 1)}
        q_once = symmetry.moments(q)
        q_twice = symmetry.moments(q_once)
        assert all(abs(q_twice[k] - q_once[k]) < 1e-12 for k in q)

    def test_aluminum_has_sixteen_irreducible_points_on_six_cubed(self):
        from ase.build import bulk as ase_bulk
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            monkhorst_pack_kpts
        from mandacaru.core.symmetry import irreducible_kpoints
        al = ase_bulk("Al", "fcc", a=4.05)
        symmetry = pp.grid_symmetry(al, _grid(al, 10))
        _size, _gamma, mesh = monkhorst_pack_kpts({"size": (6, 6, 6),
                                                   "gamma": True})
        zone = irreducible_kpoints(mesh, symmetry.info)
        assert len(zone.points) == 16 and zone.weights.sum() == 216


class TestTheIrreducibleWedge:
    @pytest.mark.parametrize("atoms, nodes, kpts, options, functional", [
        (SILICON, 12, (2, 2, 2), {"size": "SZ", "filter": 200}, "lda"),
        (bulk("Mg", "hcp", a=3.21, c=5.21), 10, (3, 3, 2),
         {"size": "SZP", "filter": 200}, "lda"),
        (SILICON, 12, (2, 2, 2), {"size": "SZ", "filter": 200}, "r2scan"),
    ], ids=["diamond-Si", "hcp-Mg", "diamond-Si-r2scan"])
    def test_it_reproduces_the_time_reversed_mesh(self, atoms, nodes, kpts,
                                                  options, functional):
        """The wedge, symmetrized, is the full mesh: to 1e-8 Ha per cell.

        hcp Mg is the case where the operations mix two axes of a grid whose
        third axis has a different node count.  The grid must resolve the
        basis: on 8 nodes the 200 eV silicon basis reaches past the Nyquist
        wave-vector, where an even FFT is not symmetric, and the discretized
        problem itself is no longer (9 uHa off).  r2SCAN checks that the
        kinetic-energy density is symmetrized like the density: summed over
        the wedge alone it was 2.9e-4 Ha off.
        """
        h = float(np.linalg.norm(np.asarray(atoms.cell)[0])) / nodes
        energies = []
        for symmetry in (True, False):
            crystal, context = pp.build_crystal(
                atoms, h, options, kpts={"size": kpts, "gamma": True},
                symmetry=symmetry)
            result = PeriodicKohnSham(
                crystal, context["n_electrons"], functional,
                smearing={"method": "fermi-dirac", "width": 0.1}).run(
                    tol=1e-10, density_tol=1e-8)
            energies.append((result.extrapolated_energy,
                             len(crystal.kpoints)))
        (wedge, n_wedge), (full, n_full) = energies
        assert n_wedge < n_full
        # A gradient-dependent functional also carries the grid's own
        # rotation residual -- spectral gradients see an FFT box the
        # operations do not map onto itself -- which falls steeply with h
        # (1.6e-8 Ha here; HISTORY 2026-10-03).
        tolerance = 1e-8 if functional == "lda" else 5e-8
        assert wedge == pytest.approx(full, abs=tolerance)


def test_bloch_sums_are_the_image_sums():
    """``bloch_values`` against the plain sum over lattice images of
    ``exp(i k.R) chi(r - R)``, at random points and k."""
    crystal, _context = pp.build_crystal(SILICON, SILICON_H, SILICON_BASIS)
    rng = np.random.default_rng(5)
    center, radius = crystal._cell_region()
    points = center + rng.uniform(-0.5, 0.5, size=(40, 3)) * radius
    kpoints = crystal.cartesian_kpoints(rng.random((3, 3)))
    values = pp.bloch_values(crystal.basis, crystal.lattice, kpoints,
                             tuple(points.T), center, radius)
    reach = max(pp._support(f) for f in crystal.basis) + 2 * radius
    expected = np.zeros_like(values)
    for R in rc.lattice_translations(crystal.lattice, reach + radius):
        phases = np.exp(1j * (kpoints @ R))
        for mu, function in enumerate(crystal.basis):
            chi = function.evaluate(*(points - R).T)
            expected[:, mu, :] += phases[:, None] * chi[None, :]
    assert np.abs(values - expected).max() < 1e-12 * np.abs(expected).max()


class TestTheProjectorSpaceMoments:
    r"""``Q^A_LM(k) = C_A blk C_A^\dagger`` is never formed: the Hamiltonian
    term and the moments go through the ``(P, P)`` projector space."""

    @pytest.fixture(scope="class")
    def solver(self):
        crystal, context = pp.build_crystal(
            SILICON, SILICON_H, SILICON_BASIS,
            kpts={"size": (2, 1, 1), "gamma": True})
        solver = PeriodicKohnSham(crystal, context["n_electrons"], "lda")
        solver.run()
        return solver

    @staticmethod
    def _dense(crystal, data):
        C = data.projections
        return {ch: C[:, crystal.projector_columns(ch[0])] @ blk
                @ C[:, crystal.projector_columns(ch[0])].conj().T
                for ch, blk in crystal.multipole_blocks.items()}

    def test_the_operator_is_the_weighted_sum_of_the_dense_ones(self, solver):
        crystal = solver.crystal
        _V, _v_tau, w = solver.potentials
        D = crystal.moment_operator(w)
        for data in crystal.kpoint_data:
            dense = sum(w[ch] * Q for ch, Q in self._dense(crystal,
                                                           data).items())
            C = data.projections
            assert np.allclose(C @ D @ C.conj().T, dense, atol=1e-13)

    def test_the_traces_are_the_dense_moments(self, solver):
        crystal = solver.crystal
        expected = {ch: 0.0j for ch in crystal.channels}
        R = 0.0
        for data, P in zip(crystal.kpoint_data, solver.density_matrices):
            for ch, Q in self._dense(crystal, data).items():
                expected[ch] += data.weight * np.sum(P * Q.T)
            C = data.projections
            R = R + data.weight * (C.conj().T @ P @ C)
        q = crystal.moment_traces(R)
        for ch in crystal.channels:
            assert q[ch] == pytest.approx(expected[ch], abs=1e-13)


def test_the_crystal_build_runs_single_threaded_blas():
    """BLAS on one thread beside the OpenMP kernels
    (:func:`~mandacaru.integrals._backend.single_threaded_blas`)."""
    for function in (pp.build_crystal, pp.PeriodicPAW.kpoint_matrices):
        assert getattr(function, "__wrapped__", None) is not None
