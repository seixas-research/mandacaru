# -*- coding: utf-8 -*-
# file: test/algorithms/test_periodic_volumetric.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Real-space outputs of a crystal: densities, Bloch orbitals and natural
orbitals on the cell grid, written as periodic ``.cube`` / ``.xsf`` files.

Identities and sum rules only (the datasets are being rebuilt): the smooth
density plus its augmentation is the valence count, the spin density plus
its augmentation the moment, the reconstructed density integrates to the
valence count by itself, files read back as written, repeats tile exactly,
Bloch orbitals at Gamma are real up to a phase.  The molecule-in-a-box
comparison lives with the rest of K20's in ``test_periodic_dft.py``.
"""

import numpy as np
import pytest
from ase.build import bulk, molecule
from ase.io.cube import read_cube_data
from ase.io.xsf import read_xsf

from mandacaru import Mandacaru
from mandacaru.algorithms import periodic_volumetric as pv
from mandacaru.units import ANGSTROM_TO_BOHR


@pytest.fixture(scope="module")
def silicon():
    """Diamond Si, SZ, coarse grid, 2x2x2 mesh (Gamma, X-type and L-type
    points in the wedge)."""
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.34,
                           kpts={"size": (2, 2, 2), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.01},
                           basis={"name": "PAW-LCAO", "size": "SZ",
                                  "filter": 200}, trace=False)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def oxygen():
    """Triplet O2 in a periodic box at Gamma: a spin-polarized crystal."""
    atoms = molecule("O2")
    atoms.set_cell([7.0, 7.0, 7.5])
    atoms.center()
    atoms.pbc = True
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.3, trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZ",
                                  "filter": 300},
                           smearing={"method": "fermi-dirac", "width": 0.01})
    atoms.get_potential_energy()
    return atoms


class TestTheDensities:
    def test_the_smooth_density_and_its_augmentation_hold_the_valence(
            self, silicon):
        field = silicon.calc.volumetric_field("density")
        assert field.data.shape == field.grid.shape
        assert field.integral + field.augmentation_charge == pytest.approx(
            8.0, abs=1e-10)
        assert field.n_electrons == pytest.approx(8.0, abs=1e-10)
        assert field.data.min() > 0.0

    def test_it_is_the_density_the_populations_partition(self, silicon):
        """The same smooth density as the crystal partition's."""
        field = silicon.calc.volumetric_field("density")
        partition = silicon.calc.atomic_partition("voronoi")
        assert field.integral == pytest.approx(
            partition.grid_electrons - np.sum(partition.augmentation),
            abs=1e-10)

    def test_the_reconstructed_density_is_positive_and_holds_the_valence(
            self, silicon):
        """Smooth density plus the spherical one-center correction, voxel
        averaged: the file integrates to the valence count by itself (the
        correction's radial integral is the augmentation charge; the
        voxel rule and the radial table agree to 4e-5 e here)."""
        field = silicon.calc.volumetric_field("reconstructed_density")
        smooth = silicon.calc.volumetric_field("density")
        assert field.integral == pytest.approx(8.0, abs=1e-3)
        assert field.augmentation_charge == 0.0
        assert field.data.min() > 0.0
        # The correction lives inside the spheres: it raises the peaks at
        # the nuclei and leaves the interstitial density alone.
        assert field.data.max() > 2.0 * smooth.data.max()
        assert np.isclose(field.data, smooth.data).mean() > 0.3

    def test_the_spin_density_holds_the_moment(self, oxygen):
        calc = oxygen.calc
        spin = calc.volumetric_field("spin_density")
        assert spin.integral + spin.augmentation_charge == pytest.approx(
            calc.get_total_magnetic_moment(), abs=1e-10)
        assert spin.n_electrons == pytest.approx(2.0, abs=1e-8)
        up = calc.volumetric_field("alpha_density")
        down = calc.volumetric_field("beta_density")
        total = calc.volumetric_field("density")
        assert np.allclose(up.data + down.data, total.data, atol=1e-14)
        assert np.allclose(up.data - down.data, spin.data, atol=1e-14)
        assert total.integral + total.augmentation_charge == pytest.approx(
            12.0, abs=1e-10)
        reconstructed = calc.volumetric_field("reconstructed_density")
        assert reconstructed.integral == pytest.approx(12.0, abs=1e-3)
        assert reconstructed.data.min() >= 0.0

    def test_a_restricted_crystal_has_no_spin_density(self, silicon):
        with pytest.raises(ValueError, match="spin-polarized"):
            silicon.calc.volumetric_field("spin_density")


class TestTheFiles:
    def test_the_cube_file_reads_back(self, silicon, tmp_path):
        """The cell's nodes once; ASE's reader sees the cell and the atoms
        where they are."""
        field = silicon.calc.write_cube(tmp_path / "si.cube")
        data, atoms = read_cube_data(str(tmp_path / "si.cube"))
        assert data.shape == field.data.shape
        assert np.abs(data - field.data).max() < 1e-6 * field.data.max()
        assert np.allclose(atoms.cell.array, silicon.cell.array, atol=1e-5)
        assert np.allclose(atoms.get_scaled_positions(),
                           silicon.get_scaled_positions(), atol=1e-6)

    def test_the_xsf_file_carries_the_lattice(self, silicon, tmp_path):
        field = silicon.calc.write_cube(tmp_path / "si.xsf")
        with open(tmp_path / "si.xsf") as handle:
            data, origin, spans, atoms = read_xsf(handle, read_data=True)
        assert atoms.pbc.all()
        assert np.allclose(atoms.cell.array, silicon.cell.array, atol=1e-6)
        assert np.allclose(spans, silicon.cell.array, atol=1e-5)
        assert data.shape == tuple(n + 1 for n in field.data.shape)
        assert np.abs(data[:-1, :-1, :-1] - field.data).max() < 2e-6
        assert np.allclose(data[-1], data[0])

    def test_repeats_tile_exactly(self, silicon, tmp_path):
        calc = silicon.calc
        cell = calc.volumetric_field("density")
        tiled = calc.write_cube(tmp_path / "si222.cube", repeat=(2, 1, 3))
        n = cell.data.shape
        assert tiled.data.shape == (2 * n[0], n[1], 3 * n[2])
        for i in range(2):
            for k in range(3):
                assert np.array_equal(
                    tiled.data[i * n[0]:(i + 1) * n[0], :,
                               k * n[2]:(k + 1) * n[2]], cell.data)
        assert len(tiled.numbers) == 6 * len(silicon)
        assert tiled.integral == pytest.approx(cell.integral, abs=1e-12)
        data, atoms = read_cube_data(str(tmp_path / "si222.cube"))
        assert data.shape == tiled.data.shape
        assert np.allclose(atoms.cell.array, (np.diag([2, 1, 3])
                                              @ silicon.cell.array),
                           atol=1e-5)
        with pytest.raises(ValueError, match="positive integers"):
            calc.volumetric_field("density", repeat=(0, 1, 1))


class TestTheOrbitals:
    def test_bloch_orbitals_at_gamma_are_real_up_to_a_phase(self, silicon):
        """Diamond is centrosymmetric and Gamma time-reversal invariant: each
        band -- the degenerate valence and conduction triplets included --
        is real once its global phase is fixed."""
        calc = silicon.calc
        for band in range(8):
            real = calc.volumetric_field("bloch_orbital", band,
                                         kpoint=(0, 0, 0), component="real")
            imag = calc.volumetric_field("bloch_orbital", band,
                                         kpoint=(0, 0, 0), component="imag")
            assert np.linalg.norm(imag.data) < 1e-10 * np.linalg.norm(
                real.data)
            # The smooth part of a PAW-normalized orbital.
            assert real.norm == pytest.approx(1.0, abs=0.01)

    def test_the_real_gauge_rotates_within_degenerate_sets(self, silicon):
        """The eigensolver leaves the triplets real only to ~1e-8; the gauge
        makes every orbital real to rounding by a unitary rotation inside
        each degenerate set, which keeps the states and their projector."""
        solver = silicon.calc.solver._periodic_solver
        crystal = solver.crystal
        V, v_tau, w, channel = solver.channel_potentials()[0]
        data, eps, vectors = next(solver._diagonalized(
            np.zeros((1, 3)), V, v_tau, w, channel))
        psi, S, C = data[0].psi, data[0].overlap, vectors[0]
        gauged = pv.real_gauge(C, eps[0], psi, S, crystal.grid.dV)
        assert max(pv.imaginary_ratio(gauged[:, n] @ psi)
                   for n in range(8)) < 1e-12
        assert np.allclose(gauged.conj().T @ S @ gauged, np.eye(8),
                           atol=1e-10)
        for members in ([0], [1, 2, 3], [4, 5, 6], [7]):
            before = C[:, members] @ C[:, members].conj().T
            after = gauged[:, members] @ gauged[:, members].conj().T
            assert np.allclose(before, after, atol=1e-10)

    def test_a_general_k_point_needs_a_component(self, silicon):
        calc = silicon.calc
        with pytest.raises(ValueError, match="genuinely complex"):
            calc.volumetric_field("bloch_orbital", 0, kpoint=(0.1, 0.2, 0.3))
        real = calc.volumetric_field("bloch_orbital", 0,
                                     kpoint=(0.1, 0.2, 0.3), component="real")
        imag = calc.volumetric_field("bloch_orbital", 0,
                                     kpoint=(0.1, 0.2, 0.3), component="imag")
        modulus = calc.volumetric_field("bloch_orbital", 0,
                                        kpoint=(0.1, 0.2, 0.3),
                                        component="modulus")
        assert np.allclose(real.data ** 2 + imag.data ** 2,
                           modulus.data ** 2, atol=1e-12)

    def test_repeated_cells_carry_the_bloch_phase(self, silicon):
        """psi(r + R) = exp(i k.R) psi(r): at the zone-boundary point
        k = b3 / 2 the next cell along a3 has the opposite sign."""
        calc = silicon.calc
        one = calc.volumetric_field("bloch_orbital", 0, kpoint=(0, 0, 0.5),
                                    component="real")
        two = calc.volumetric_field("bloch_orbital", 0, kpoint=(0, 0, 0.5),
                                    component="real", repeat=(1, 1, 2))
        n = one.data.shape[2]
        assert np.allclose(two.data[:, :, n:], -two.data[:, :, :n],
                           atol=1e-12)
        assert np.allclose(np.abs(two.data[:, :, :n]), np.abs(one.data),
                           atol=1e-12)

    def test_a_spin_crystal_names_its_channel(self, oxygen):
        calc = oxygen.calc
        with pytest.raises(ValueError, match="spin=0 or 1"):
            calc.volumetric_field("bloch_orbital", 5)
        up = calc.volumetric_field("bloch_orbital", 5, spin=0)
        down = calc.volumetric_field("bloch_orbital", 5, spin=1)
        assert "spin up" in up.notes and "spin down" in down.notes
        # The occupied pi* of the majority channel lies below the minority's.
        level = [float(f.notes[0].split()[4]) for f in (up, down)]
        assert level[0] < level[1]


class TestTheNaturalOrbitals:
    def test_a_restricted_crystal_has_its_occupations(self, silicon):
        orbitals = silicon.calc.natural_orbitals()
        assert orbitals.n_spins == 1
        assert orbitals.n_electrons == pytest.approx(8.0, abs=1e-10)
        assert np.allclose(orbitals.kpoints, silicon.calc.get_ibz_k_points())
        assert np.all(np.diff(orbitals.occupations, axis=1) <= 1e-12)
        occupations = np.sort(np.concatenate(
            silicon.calc.result.scf.occupations))
        assert np.allclose(np.sort(orbitals.occupations.ravel()),
                           occupations, atol=1e-10)

    def test_a_spin_crystal_has_two_singly_occupied_orbitals(self, oxygen):
        """Triplet O2, spin-summed: the two pi* hold one electron each."""
        orbitals = oxygen.calc.natural_orbitals()
        assert orbitals.n_spins == 2
        n = orbitals.occupations[0]
        assert orbitals.n_electrons == pytest.approx(12.0, abs=1e-10)
        assert np.sum(np.abs(n - 1.0) < 1e-6) == 2
        assert n.min() >= 0.0 and n.max() <= 2.0 + 1e-10
        field = oxygen.calc.volumetric_field("natural_orbital", 5)
        assert field.occupation == pytest.approx(1.0, abs=1e-6)

    def test_the_orbital_matches_an_scf_k_point(self, silicon):
        calc = silicon.calc
        by_index = calc.volumetric_field("natural_orbital", 0, kpoint=1)
        by_point = calc.volumetric_field(
            "natural_orbital", 0, kpoint=calc.get_ibz_k_points()[1] + 1.0)
        assert np.array_equal(by_index.data, by_point.data)
        with pytest.raises(ValueError, match="not an SCF k-point"):
            calc.volumetric_field("natural_orbital", 0,
                                  kpoint=(0.1, 0.2, 0.3))

    def test_the_rpa_ones_are_still_reached_by_method(self, silicon):
        with pytest.raises(TypeError, match="method='rpa'"):
            silicon.calc.natural_orbitals(cutoff=2.0)
        with pytest.raises(ValueError, match="'rpa'"):
            silicon.calc.natural_orbitals(method="mp2")


class TestTheRefusals:
    def test_the_dipole_points_to_the_polarization(self, silicon):
        with pytest.raises(NotImplementedError, match="get_polarization"):
            silicon.calc.get_dipole_moment()

    def test_grid_and_state_belong_to_molecules(self, silicon):
        calc = silicon.calc
        with pytest.raises(NotImplementedError, match="grid="):
            calc.volumetric_field("density", state=1)
        with pytest.raises(NotImplementedError, match="grid="):
            calc.natural_orbitals(grid=object())

    def test_options_that_do_not_apply_are_refused(self, silicon):
        calc = silicon.calc
        with pytest.raises(ValueError, match="summed over the whole mesh"):
            calc.volumetric_field("density", kpoint=0)
        with pytest.raises(ValueError, match="spin="):
            calc.volumetric_field("density", spin=0)
        with pytest.raises(ValueError, match="unknown quantity"):
            calc.volumetric_field("molecular_orbital")

    def test_a_molecule_takes_no_crystal_options(self):
        atoms = molecule("H2")
        atoms.center(vacuum=2.5)
        atoms.calc = Mandacaru(method="dft", xc="lda", h=0.3, trace=False,
                               basis="HAO")
        atoms.get_potential_energy()
        with pytest.raises(ValueError, match="crystal"):
            atoms.calc.volumetric_field("density", repeat=(2, 1, 1))


def test_the_cell_atoms_are_wrapped_into_the_written_cell(silicon):
    """An atom given outside the cell is drawn at its periodic image
    inside the box the data spans."""
    crystal = silicon.calc.solver._periodic_solver.crystal
    _numbers, _charges, positions = pv.cell_atoms(crystal, (1, 1, 1))
    lattice = np.asarray(crystal.lattice)
    fractional = np.linalg.solve(lattice, positions.T).T
    assert np.all(fractional >= -1e-9) and np.all(fractional < 1.0)
    assert np.allclose(positions / ANGSTROM_TO_BOHR, silicon.positions,
                       atol=1e-6)
