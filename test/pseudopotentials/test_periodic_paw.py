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
SILICON_BASIS = {"size": "SZ", "filter_cutoff": 200}


def _energy(atoms, grid_spacing, options, kpts=None):
    """Extrapolated energy (Hartree per cell) of a periodic ``atoms``."""
    crystal, context = pp.build_crystal(atoms, grid_spacing, options, kpts=kpts)
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
        operator, extrapolates as h^2 from grid_spacing = 0.25 ... 0.10 Angstrom to
        -1.1269 Ha; the periodic, Fourier-filtered one gives it at grid_spacing = 0.25
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
                grid_spacing=float(np.linalg.norm(cell[0])) / nodes, units="angstrom",
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

    def test_a_crystal_with_an_f_channel_symmetrizes_its_moments(self):
        """An f channel puts compensation multipoles up to L = 6 on its
        atom; the rotation tables stopped at L = 4 and fcc Au (or bcc W)
        stopped in its first SCF step with ``KeyError: 5``."""
        from mandacaru import Mandacaru

        atoms = bulk("Au", "fcc", a=4.08)
        atoms.calc = Mandacaru(
            method="dft", xc="lda", grid_spacing=0.3, trace=False,
            basis={"name": "PAW-LCAO", "size": "SZ"},
            kpts={"size": (2, 2, 2), "gamma": True},
            smearing={"method": "fermi-dirac", "width": 0.1})
        assert np.isfinite(atoms.get_potential_energy())

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


def _equivalent(a, b, rotations):
    """Whether fractional k-points ``a`` and ``b`` are images under one of
    ``rotations`` (``k -> k W``, the reduction's convention) or its time
    reverse, up to a reciprocal-lattice vector."""
    for W in rotations:
        image = np.asarray(a) @ np.asarray(W, dtype=float)
        for sign in (1.0, -1.0):
            delta = sign * image - np.asarray(b)
            if np.all(np.abs(delta - np.round(delta)) < 1e-8):
                return True
    return False


class TestTheMeshWedge:
    """:func:`~mandacaru.pseudopotentials.periodic_paw.mesh_wedge`: the map
    from every full-mesh point to its wedge representative, which the
    tetrahedron DOS and the Fermi surface unfold through."""

    @staticmethod
    def check(mesh, size, operations):
        points, mapping, kept = pp.mesh_wedge(mesh, size, operations)
        rotations = ([np.eye(3)] if kept is None
                     else list(kept.info.rotations))
        assert mapping.shape == (len(mesh),)
        # Every mesh point is an image of its representative.
        for j, k in enumerate(mesh):
            assert _equivalent(points[mapping[j]], k, rotations), j
        # Every representative is a mesh point, standing for itself, and no
        # two are images of each other.
        for i, k in enumerate(points):
            delta = mesh - k
            here = np.flatnonzero(np.all(np.abs(delta - np.round(delta))
                                         < 1e-8, axis=1))
            assert len(here) == 1 and mapping[here[0]] == i
            for other in points[:i]:
                assert not _equivalent(other, k, rotations)
        # The weights are the orbit sizes over N_k: reduce_mesh's.
        weights = np.bincount(mapping, minlength=len(points)) / len(mesh)
        assert np.all(weights > 0) and weights.sum() == pytest.approx(1.0)
        reduced, reference, _ops = pp.reduce_mesh(mesh, operations)
        assert len(reduced) == len(points)
        for k, w in zip(reduced, reference):
            delta = points - k
            match = np.flatnonzero(np.all(np.abs(delta - np.round(delta))
                                          < 1e-8, axis=1))
            assert len(match) == 1
            assert weights[match[0]] == pytest.approx(w, abs=1e-14)
        return points, kept

    @pytest.mark.parametrize("kpts", [
        {"size": (4, 4, 4), "gamma": True}, (4, 4, 4), (3, 3, 3),
        {"size": (3, 4, 2), "gamma": True}, (3, 4, 2), (2, 2, 2)])
    def test_time_reversal_alone(self, kpts):
        """No point-group operation: k and -k, keyed on the lattice 2 n k
        (a shifted mesh's coordinates are odd multiples of 1/(2n))."""
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            monkhorst_pack_kpts
        size, _gamma, mesh = monkhorst_pack_kpts(kpts)
        points, kept = self.check(mesh, size, None)
        assert kept is None
        # Half the points, plus the self-reverse ones (2k on the lattice).
        own = np.sum(np.all(np.abs(2 * mesh - np.round(2 * mesh)) < 1e-8,
                            axis=1))
        assert len(points) == (len(mesh) + own) // 2

    @pytest.mark.parametrize("atoms, nodes, kpts, expected", [
        (SILICON, 8, {"size": (4, 4, 4), "gamma": True}, 8),
        (SILICON, 8, (4, 4, 4), 10),
        (SILICON, 8, (3, 3, 3), 4),
        (SILICON, 10, (4, 4, 4), None),
        (bulk("Mg", "hcp", a=3.21, c=5.21), 10, (4, 4, 2), None),
        (bulk("Mg", "hcp", a=3.21, c=5.21), 10,
         {"size": (3, 3, 2), "gamma": True}, None)],
        ids=["Si-gamma-4", "Si-shifted-4", "Si-3", "Si-24-ops",
             "Mg-shifted", "Mg-gamma"])
    def test_with_the_point_group(self, atoms, nodes, kpts, expected):
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            monkhorst_pack_kpts
        size, _gamma, mesh = monkhorst_pack_kpts(kpts)
        symmetry = pp.grid_symmetry(atoms, _grid(atoms, nodes))
        points, kept = self.check(mesh, size, symmetry)
        assert kept is not None and len(points) < len(mesh) // 2
        if expected is not None:
            assert len(points) == expected

    def test_a_mesh_without_its_time_reverse_is_refused(self):
        mesh = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
        with pytest.raises(ValueError, match="k -> -k"):
            pp.mesh_wedge(mesh, (4, 1, 1))

    def test_without_time_reversal_the_mesh_is_kept_whole(self):
        """A magnetic spinor crystal's bands are not even in k: every point
        stands for itself, with weight 1/N_k, and no operation is taken."""
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            monkhorst_pack_kpts
        size, _gamma, mesh = monkhorst_pack_kpts((3, 4, 2))
        points, mapping, kept = pp.mesh_wedge(mesh, size,
                                              time_reversal=False)
        assert kept is None and np.array_equal(points, mesh)
        assert np.array_equal(mapping, np.arange(len(mesh)))
        reduced, weights, kept = pp.reduce_mesh(mesh, time_reversal=False)
        assert kept is None and np.array_equal(reduced, mesh)
        assert np.allclose(weights, 1.0 / len(mesh))
        symmetry = pp.grid_symmetry(SILICON, _grid(SILICON, 8))
        with pytest.raises(ValueError, match="time reversal"):
            pp.reduce_mesh(mesh, symmetry, time_reversal=False)


class TestTheIrreducibleWedge:
    @pytest.mark.parametrize("atoms, nodes, kpts, options, functional", [
        (SILICON, 12, (2, 2, 2), {"size": "SZ", "filter_cutoff": 200}, "lda"),
        (bulk("Mg", "hcp", a=3.21, c=5.21), 10, (3, 3, 2),
         {"size": "SZP", "filter_cutoff": 200}, "lda"),
        (bulk("Mg", "hcp", a=3.21, c=5.21), 10, (3, 3, 2),
         {"size": "SZP", "filter_cutoff": 200}, "pbe"),
        (SILICON, 12, (2, 2, 2), {"size": "SZ", "filter_cutoff": 200}, "r2scan"),
    ], ids=["diamond-Si", "hcp-Mg", "hcp-Mg-pbe", "diamond-Si-r2scan"])
    def test_it_reproduces_the_time_reversed_mesh(self, atoms, nodes, kpts,
                                                  options, functional):
        """The wedge, symmetrized, is the full mesh: to 1e-8 Ha per cell.

        hcp Mg is the case where the operations mix two axes of a grid whose
        third axis has a different node count.  The grid must resolve the
        basis: on 8 nodes the 200 eV silicon basis reaches past the Nyquist
        wave-vector, where an even FFT is not symmetric, and the discretized
        problem itself is no longer (9 uHa off).  r2SCAN checks that the
        kinetic-energy density is symmetrized like the density: summed over
        the wedge alone it was 2.9e-4 Ha off.  PBE on hcp Mg: the density
        gradients commute with the hexagonal operations (wave-vectors
        averaged over the lattice's operations, K13).  Semicore hcp Mg
        missed by 1.6e-4 Ha while the kinetic matrix took the FFT box's
        alias of G + k, and still by 2.7e-8 while the Coulomb kernel, the grid
        charges and the short-range sphere rule did not commute with the
        operations either (K13b, :class:`TestTheOperationsCommute`).
        """
        grid_spacing = float(np.linalg.norm(np.asarray(atoms.cell)[0])) / nodes
        energies = []
        for symmetry in (True, False):
            crystal, context = pp.build_crystal(
                atoms, grid_spacing, options, kpts={"size": kpts, "gamma": True},
                symmetry=symmetry)
            result = PeriodicKohnSham(
                crystal, context["n_electrons"], functional,
                smearing={"method": "fermi-dirac", "width": 0.1}).run(
                    tol=1e-10, density_tol=1e-8)
            energies.append((result.extrapolated_energy,
                             len(crystal.kpoints)))
        (wedge, n_wedge), (full, n_full) = energies
        assert n_wedge < n_full
        assert wedge == pytest.approx(full, abs=1e-8)


#: Semicore hcp Mg on a grid that does not resolve its 2p functions (10
#: nodes along a, 16 along c): the case where the FFT box's aliases are
#: farthest from commuting with the hexagonal operations.
MAGNESIUM = bulk("Mg", "hcp", a=3.21, c=5.21)


class TestTheOperationsCommute:
    """Every operator of the crystal commutes with the space group the
    k-mesh is reduced by, so a wedge symmetrized to the full mesh is the full
    mesh (HISTORY 2026-10-11, K13b)."""

    @pytest.fixture(scope="class")
    def magnesium(self):
        grid_spacing = float(np.linalg.norm(MAGNESIUM.cell[0])) / 10
        crystal, context = pp.build_crystal(
            MAGNESIUM, grid_spacing, {"size": "SZP", "filter_cutoff": 200},
            kpts={"size": (3, 3, 2), "gamma": True}, symmetry=False)
        return crystal, context, pp.grid_symmetry(MAGNESIUM, crystal.grid)

    def test_a_star_of_k_points_has_one_spectrum_per_matrix(self, magnesium):
        """The six k-points of the star of (0, 1/3, 0) under the operations
        and time reversal, each its own representative, give one spectrum
        of the kinetic and short-range local matrices.  With the box's
        alias of G + k the kinetic spectra split by 3.4e-4 Ha; with 32
        azimuths the short-range ones by 2.7e-8."""
        crystal, _context, symmetry = magnesium
        star = []
        for W in symmetry.info.rotations:
            image = np.array([0.0, 1.0 / 3.0, 0.0]) @ np.asarray(W, float)
            for point in (image, -image):
                if not any(np.allclose(point, x) for x in star):
                    star.append(point)
        assert len(star) == 6
        data = crystal.kpoint_matrices(crystal.cartesian_kpoints(star))
        spectra = []
        for d in data:
            kinetic = crystal._kinetic(d.psi, d.k)
            local = crystal._short_range_local(d.k[None])[0]
            spectra.append([np.linalg.eigvalsh(0.5 * (A + A.conj().T))
                            for A in (kinetic, local)])
        for other in spectra[1:]:
            for a, b in zip(spectra[0], other):
                assert np.max(np.abs(a - b)) < 1e-12

    def test_a_symmetric_density_has_a_symmetric_potential(self, magnesium):
        """The Coulomb kernel and the compensation and ion charges on the
        grid: the electrostatic potential of a symmetric density and
        symmetric multipoles is symmetric, and so are the moments' energy
        derivatives.  At the box's aliases the potential was 6.5e-3 of its
        peak off, the derivatives 9e-4."""
        crystal, context, symmetry = magnesium
        solver = PeriodicKohnSham(crystal, context["n_electrons"], "lda")
        rho, q = crystal.initial_density()
        rho, q = symmetry.field(rho), symmetry.moments(q)
        potential, w = solver._electrostatics(rho, q)
        assert np.max(np.abs(symmetry.field(potential) - potential)) \
            < 1e-12 * np.max(np.abs(potential))
        symmetric_w = symmetry.moments(w)
        assert max(abs(symmetric_w[key] - w[key]) for key in w) < 1e-12


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


class TestTheLatticeSumDerivatives:
    """The ion-pair and ion-compensation lattice sums have analytic
    derivatives (the crystal force no longer differences them)."""

    @pytest.fixture(scope="class")
    def crystal(self):
        atoms = SILICON.copy()
        atoms.positions[1] += [0.05, 0.02, -0.03]
        crystal, _context = pp.build_crystal(atoms, SILICON_H, SILICON_BASIS)
        return crystal

    @staticmethod
    def _central(function, centers, atom, d, step):
        values = []
        for sign in (+1.0, -1.0):
            moved = [np.array(x, dtype=float) for x in centers]
            moved[atom][d] += sign * step
            values.append(function(moved))
        return values

    def test_the_ion_pair_gradient_is_the_difference_of_the_sum(self, crystal):
        analytic = crystal.ion_constants_gradient()
        for atom in range(len(crystal.centers)):
            for d in range(3):
                plus, minus = self._central(crystal.ion_constants,
                                            crystal.centers, atom, d, 1e-4)
                assert analytic[atom, d] == pytest.approx(
                    (plus - minus) / 2e-4, abs=1e-8)
        assert np.abs(analytic).max() > 1e-4

    def test_the_ion_compensation_gradient_is_the_difference_of_the_sum(
            self, crystal):
        analytic = crystal.short_range_ion_compensation_gradient()
        assert set(analytic) == set(crystal.channels)
        for atom, d in ((0, 0), (1, 2)):
            plus, minus = self._central(
                crystal.short_range_ion_compensation, crystal.centers, atom,
                d, 1e-4)
            for channel in crystal.channels:
                numeric = (plus[channel] - minus[channel]) / 2e-4
                assert analytic[channel][atom, d] == pytest.approx(
                    numeric, abs=1e-7)

    def test_both_gradients_sum_to_zero_over_the_atoms(self, crystal):
        """Translating every atom together changes neither sum."""
        assert crystal.ion_constants_gradient().sum(axis=0) == pytest.approx(
            0.0, abs=1e-13)
        for gradient in crystal.short_range_ion_compensation_gradient(
                ).values():
            assert np.abs(gradient.sum(axis=0)).max() < 1e-12

    def test_the_local_potential_derivative_is_the_slope(self, crystal):
        dataset = crystal.datasets[0]
        radius = np.concatenate([
            np.linspace(0.5 * dataset.r[0], 0.9 * dataset.r[0], 3),
            np.linspace(1.05 * dataset.r[0], 0.97 * dataset.r[-1], 40),
            np.linspace(1.05 * dataset.r[-1], 3.0 * dataset.r[-1], 5)])
        step = 1e-6
        numeric = (dataset.local_potential(radius + step)
                   - dataset.local_potential(radius - step)) / (2 * step)
        assert dataset.local_potential_derivative(radius) == pytest.approx(
            numeric, abs=2e-6)


class TestTheSphereBlochSums:
    """The Bloch sums of the projector and short-range spheres: contracted
    image by image, evaluated for any number of k-points at once, and only in
    the blocks a force needs."""

    @pytest.fixture(scope="class")
    def crystal(self):
        crystal, _context = pp.build_crystal(
            SILICON, SILICON_H, SILICON_BASIS,
            kpts={"size": (2, 2, 1), "gamma": True})
        return crystal

    def test_the_image_resolved_projection_is_the_contracted_sum(self,
                                                                 crystal):
        """Basis functions and their derivatives alike."""
        rng = np.random.default_rng(3)
        center, radius = crystal._cell_region()
        points = center + rng.uniform(-0.5, 0.5, size=(900, 3)) * radius
        order = np.lexsort((np.floor(points[:, 2]), np.floor(points[:, 1]),
                            np.floor(points[:, 0])))
        points = tuple(points[order].T)
        columns = rng.normal(size=(3, 900)) + 1j * rng.normal(size=(3, 900))
        functions = list(crystal.basis) + [
            pp.BasisGradient(f, axis) for axis in range(3)
            for f in crystal.basis[:4]]
        kpoints = crystal.kpoints
        values = pp.bloch_values(functions, crystal.lattice, kpoints, points,
                                 center, radius)
        expected = np.einsum("kmx,px->kmp", values.conj(), columns)
        got = pp.bloch_projections(functions, crystal.lattice, kpoints,
                                   points, columns, center, radius)
        assert got == pytest.approx(expected, abs=1e-12 * np.abs(expected).max())

    def test_k_points_evaluated_together_equal_one_by_one(self, crystal):
        kpoints = crystal.cartesian_kpoints(
            np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.5, 0.0]]))
        together = crystal.kpoint_matrices(kpoints)
        for i, k in enumerate(kpoints):
            alone = crystal.kpoint_matrices(k[None, :])[0]
            for name in ("overlap", "fixed", "projections"):
                assert np.allclose(getattr(together[i], name),
                                   getattr(alone, name), atol=1e-12, rtol=0)

    def test_the_cross_block_is_the_block_of_the_full_matrix(self, crystal):
        kpoints = crystal.kpoints
        full = crystal.short_range_atom(0, kpoints)
        rows = list(range(0, crystal.M, 2))
        columns = list(range(1, crystal.M, 2))
        cross = crystal.short_range_cross(
            0, kpoints, [crystal.basis[mu] for mu in rows],
            [crystal.basis[mu] for mu in columns])
        assert np.allclose(cross, full[:, rows][:, :, columns], atol=1e-12,
                           rtol=0)

    def test_the_chunk_holds_its_bytes_whatever_the_k_points(self):
        assert pp._sphere_chunk_size(1, 20) == pp.SPHERE_CHUNK
        for nk, rows in ((10, 200), (100, 400), (1000, 1000)):
            size = pp._sphere_chunk_size(nk, rows)
            assert 4096 <= size <= pp.SPHERE_CHUNK
            assert size == 4096 or 32 * nk * rows * size <= pp.SPHERE_BYTES


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


class TestTheAugmentationRemainder:
    """What the compensation charges leave of the augmentation pair densities
    (:meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.remainder_transforms`)
    and of their Coulomb energy (:meth:`onsite_remainder`)."""

    @pytest.fixture(scope="class")
    def crystal(self):
        crystal, _context = pp.build_crystal(
            SILICON, SILICON_H, SILICON_BASIS,
            kpts={"size": (1, 1, 1), "gamma": True})
        return crystal

    def test_it_moves_with_the_atom(self, crystal):
        """A pure phase from one atom to the other: the two silicon atoms'
        tables are the same function times their structure factors."""
        vectors = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, -0.7],
                            [0.4, 2.1, 1.1]]).T
        first = crystal.remainder_transforms(0, vectors)
        second = crystal.remainder_transforms(1, vectors)
        shift = np.exp(-1j * (vectors.T @ (crystal.centers[1]
                                           - crystal.centers[0])))
        assert np.allclose(second, first * shift[None, None, :], atol=1e-9)

    def test_it_is_hermitian_in_the_pair(self, crystal):
        """``E_ab(g) = conj(E_ba(-g))`` for the pair density of a function
        and its conjugate partner."""
        vectors = np.array([[1.3, 0.2, -0.7], [0.4, 2.1, 1.1]]).T
        plus = crystal.remainder_transforms(0, vectors)
        minus = crystal.remainder_transforms(0, -vectors)
        assert np.allclose(plus, np.conj(minus).transpose(1, 0, 2),
                           atol=1e-9)

    def test_the_onsite_energy_is_a_hermitian_pair_tensor(self, crystal):
        """``conj(DeltaW[a, b, c, d]) = DeltaW[b, a, d, c]`` (the pair
        density of ``b, a`` is the conjugate of ``a, b``) and it is
        symmetric under the exchange of the two pairs."""
        W = crystal.onsite_remainder(0)
        assert np.allclose(W, np.conj(W.transpose(1, 0, 3, 2)), atol=1e-10)
        # raw-basis entries are large (tens of Hartree) and the pair-exchange
        # symmetry holds to the quadrature of the moments, 5e-5 relative
        assert np.allclose(W, W.transpose(2, 3, 0, 1), rtol=1e-3, atol=1e-3)


def test_the_crystal_build_runs_single_threaded_blas():
    """BLAS on one thread beside the OpenMP kernels
    (:func:`~mandacaru.integrals._backend.single_threaded_blas`)."""
    for function in (pp.build_crystal, pp.PeriodicPAW.kpoint_matrices):
        assert getattr(function, "__wrapped__", None) is not None


class TestTheSpinOrbitTerm:
    """The spin-orbit term of a spin-orbit set, Bloch summed per k-point
    (:meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.spin_orbit_matrix`)."""

    @staticmethod
    def options(xc="lda"):
        from mandacaru.pseudopotentials.paw import paw_library_path
        return {**SILICON_BASIS,
                "paw_dataset": paw_library_path(None, xc,
                                                relativity="dirac")}

    @pytest.fixture(scope="class")
    def crystal(self):
        crystal, _context = pp.build_crystal(
            SILICON, SILICON_H, self.options(),
            kpts={"size": (2, 1, 1), "gamma": True}, spin_orbit=True)
        return crystal

    def test_a_spin_orbit_set_needs_spinor_states(self):
        """Without ``spin_orbit=True`` the set is refused, not read as a
        scalar-relativistic one."""
        with pytest.raises(NotImplementedError, match="spin-orbit"):
            pp.build_crystal(SILICON, SILICON_H, self.options(),
                             kpts={"size": (1, 1, 1), "gamma": True})
        assert pp.spin_orbit_symbols(SILICON, self.options()) == ["Si"]
        assert pp.spin_orbit_symbols(SILICON, SILICON_BASIS) == []

    def test_it_couples_the_spins_and_is_hermitian(self, crystal):
        M = crystal.M
        assert crystal.has_spin_orbit
        for data in crystal.kpoint_data:
            h = data.spin_orbit
            assert h.shape == (2 * M, 2 * M)
            assert np.allclose(h, h.conj().T, atol=1e-15)
            assert np.max(np.abs(h[:M, M:])) > 1e-4
            # L.S is traceless over each shell's spin-orbitals.
            assert abs(np.trace(np.linalg.solve(
                np.kron(np.eye(2), data.overlap), h))) < 1e-10

    def test_it_is_periodic_in_the_reciprocal_lattice(self, crystal):
        """The Bloch sums at k + G are those at k, and so is the term."""
        k = crystal.cartesian_kpoints([[0.1, 0.2, -0.3]])
        G = crystal.cartesian_kpoints([[1.0, -1.0, 2.0]])
        assert np.allclose(crystal.spin_orbit_matrix(k),
                           crystal.spin_orbit_matrix(k + G), atol=1e-12)
        assert np.allclose(crystal.spin_orbit_matrix(crystal.kpoints),
                           np.stack([d.spin_orbit
                                     for d in crystal.kpoint_data]),
                           atol=1e-14)

    def test_time_reversal_maps_k_to_minus_k(self, crystal):
        """Its spectrum in the metric S(k) x 1 is even in k (Kramers), and
        at Gamma it comes in pairs."""
        from scipy.linalg import eigh

        k = crystal.cartesian_kpoints([[0.1, 0.2, -0.3], [0.0, 0.0, 0.0]])
        data = crystal.kpoint_matrices(np.vstack([k, -k[:1]]))
        spectra = [eigh(d.spin_orbit, np.kron(np.eye(2), d.overlap),
                        eigvals_only=True) for d in data]
        assert np.allclose(spectra[0], spectra[2], atol=1e-12)
        assert np.allclose(spectra[1][0::2], spectra[1][1::2], atol=1e-12)

    def test_a_scalar_crystal_has_none(self):
        crystal, _context = pp.build_crystal(
            SILICON, SILICON_H, SILICON_BASIS,
            kpts={"size": (1, 1, 1), "gamma": True}, spin_orbit=True)
        assert not crystal.has_spin_orbit
        assert crystal.kpoint_data[0].spin_orbit is None
        assert not np.any(crystal.spin_orbit_matrix(crystal.kpoints))

    def test_a_magnetization_is_refused_on_a_reduced_mesh(self, crystal):
        """An axial vector the scalar averages would not rotate."""
        assert crystal.symmetry is not None
        M = crystal.M
        P = [np.eye(2 * M, dtype=complex)] * len(crystal.kpoint_data)
        with pytest.raises(NotImplementedError, match="axial"):
            crystal.magnetization(P)


class TestDiffuseIsMolecular:
    def test_a_crystal_with_diffuse_shells_is_refused(self):
        # Refused before anything is sampled: in a solid a diffuse shell is
        # nearly dependent on the neighbors' valence functions.
        with pytest.raises(ValueError, match="for molecules"):
            pp.build_crystal(SILICON, SILICON_H,
                             {**SILICON_BASIS, "diffuse": True})
