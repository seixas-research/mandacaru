# -*- coding: utf-8 -*-
# file: test/test_volumetric.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Volumetric output: densities, natural orbitals and the ``.cube`` file.

Three things are pinned here.  The **physics**: what the file holds is a
one-particle reduction of the many-body state, so it must integrate to the
electron count (exactly, on the calculation's own grid), reproduce the explicit
determinant density for a Hartree-Fock state, and give the same answer whatever
register representation the run used.  The **format**: the header has to carry
the grid's own origin and step vectors, so that the atoms and the data line up
in a viewer -- the classic cube-file bug, tested with a molecule deliberately
*not* centered in its cell and with a skewed grid.  And the **contracts**: a
run with no basis functions to draw is refused by name rather than producing an
empty file.
"""

import os

import numpy as np
import pytest
from ase import Atoms
from ase.io.cube import read_cube

from mandacaru import Mandacaru
from mandacaru.algorithms.volumetric import (OrbitalExpansion, QUANTITIES,
                                             reference_occupations,
                                             spin_resolved_rdm,
                                             volumetric_field)
from mandacaru.integrals import Grid
from mandacaru.units import BOHR_TO_ANGSTROM
from mandacaru.utils.cube import (CUBE_AXIS_COMMENT, detect_format, write_cube,
                                  write_xsf)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

#: Coarse grid everywhere: these tests check identities, not accuracy, and the
#: identities below are exact on any grid.
H = 0.30


def _paw_available() -> bool:
    from mandacaru.pseudopotentials import get_paw
    try:
        get_paw("H")
    except (FileNotFoundError, ValueError):
        return False
    return True


needs_paw = pytest.mark.skipif(not _paw_available(),
                               reason="MANDACARU_PAW_PATH is not configured")


def _molecule(symbols, positions, cell, center=True, **info):
    atoms = Atoms(symbols, positions=positions, cell=cell, **info)
    if center:
        atoms.center()
    return atoms


def _solved(atoms, **options):
    """Attach a calculator, evaluate the energy, return the calculator."""
    atoms.calc = Mandacaru(method="adapt-vqe", basis=options.pop("basis", "HAO"),
                           h=options.pop("h", H), trace=False, **options)
    atoms.get_potential_energy()
    return atoms.calc


@pytest.fixture(scope="module")
def h2():
    """H2 at equilibrium, HAO, Jordan-Wigner."""
    atoms = _molecule("H2", [[0, 0, 0], [0, 0, 0.74]], [6, 6, 6])
    return atoms, _solved(atoms)


@pytest.fixture(scope="module")
def h2_offset():
    """The same molecule, deliberately *not* centered in its cell."""
    atoms = _molecule("H2", [[1.0, 1.2, 1.4], [1.0, 1.2, 2.14]], [6, 6, 6],
                      center=False)
    return atoms, _solved(atoms)


# --------------------------------------------------------------------------- #
# The file format, with no solver involved.
# --------------------------------------------------------------------------- #

class TestCubeFormat:
    """``utils/cube.py`` alone: the header must describe *this* grid."""

    @staticmethod
    def _field(step=None):
        shape = (4, 5, 3)
        data = np.arange(np.prod(shape), dtype=float).reshape(shape) * 1e-3
        origin = np.array([-1.5, 0.25, 3.0])
        if step is None:
            step = np.diag([0.4, 0.5, 0.6])
        numbers = np.array([1, 8])
        positions = np.array([[0.0, 0.0, 3.5], [1.0, 0.5, 3.5]])
        return data, origin, step, numbers, positions

    def test_header_and_data_round_trip(self, tmp_path):
        data, origin, step, numbers, positions = self._field()
        path = write_cube(tmp_path / "f.cube", data, origin, step, numbers,
                          positions, comment="unit test")
        with open(path) as handle:
            back = read_cube(handle)
        assert back["data"].shape == data.shape
        # %13.5E keeps five significant digits; the values here are ~1e-3.
        assert np.abs(back["data"] - data).max() < 1e-8
        assert np.allclose(back["origin"], origin * BOHR_TO_ANGSTROM, atol=1e-5)
        # ASE returns the voxel vectors as *rows*; ours are the columns of step.
        assert np.allclose(back["spacing"], step.T * BOHR_TO_ANGSTROM, atol=1e-5)
        assert np.allclose(back["atoms"].get_positions(),
                           positions * BOHR_TO_ANGSTROM, atol=1e-5)
        assert list(back["atoms"].numbers) == [1, 8]

    def test_skewed_step_vectors_survive(self, tmp_path):
        """A non-orthogonal grid is expressible: the step vectors are vectors."""
        step = np.array([[0.4, 0.1, 0.0],
                         [0.0, 0.5, 0.08],
                         [0.0, 0.0, 0.6]])
        data, origin, _step, numbers, positions = self._field()
        path = write_cube(tmp_path / "skew.cube", data, origin, step, numbers,
                          positions)
        with open(path) as handle:
            back = read_cube(handle)
        assert np.allclose(back["spacing"], step.T * BOHR_TO_ANGSTROM, atol=1e-5)

    def test_second_line_is_the_one_ase_parses(self, tmp_path):
        data, origin, step, numbers, positions = self._field()
        path = write_cube(tmp_path / "f.cube", data, origin, step, numbers,
                          positions, comment="a comment\nwith a newline")
        lines = open(path).read().splitlines()
        assert lines[0] == "a comment with a newline"   # never two lines
        assert lines[1] == CUBE_AXIS_COMMENT

    def test_charges_column_defaults_to_the_atomic_numbers(self, tmp_path):
        data, origin, step, numbers, positions = self._field()
        path = write_cube(tmp_path / "f.cube", data, origin, step, numbers,
                          positions, charges=[1.0, 6.0])
        rows = open(path).read().splitlines()[6:8]
        assert [float(row.split()[1]) for row in rows] == [1.0, 6.0]

    def test_complex_data_is_refused(self, tmp_path):
        data, origin, step, numbers, positions = self._field()
        with pytest.raises(TypeError, match="real numbers"):
            write_cube(tmp_path / "f.cube", data.astype(complex) * 1j, origin,
                       step, numbers, positions)

    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path):
        data, origin, step, numbers, positions = self._field()
        path = write_cube(tmp_path / "f.cube", data, origin, step, numbers,
                          positions, comment="first")
        with pytest.raises(ValueError):
            write_cube(path, data, origin, step, numbers, positions[:1])
        assert open(path).read().splitlines()[0] == "first"

    def test_format_detection(self, tmp_path):
        assert detect_format("x.cube") == "cube"
        assert detect_format("x.cub") == "cube"
        assert detect_format("x.xsf") == "xsf"
        assert detect_format("x.dat", format="cube") == "cube"
        with pytest.raises(ValueError, match="cannot tell"):
            detect_format("x.dat")
        with pytest.raises(ValueError, match="unknown volumetric format"):
            detect_format("x.cube", format="vtk")

    def test_xsf_is_readable_by_ase(self, tmp_path):
        import ase.io

        data, origin, step, numbers, positions = self._field()
        path = write_xsf(tmp_path / "f.xsf", data, origin, step, numbers,
                         positions)
        back = ase.io.read(path)
        assert list(back.numbers) == [1, 8]
        assert np.allclose(back.get_positions(),
                           positions * BOHR_TO_ANGSTROM, atol=1e-8)


# --------------------------------------------------------------------------- #
# The physics: what the numbers on the grid must add up to.
# --------------------------------------------------------------------------- #

class TestElectronCount:
    """A density integrates to the electron count -- exactly, on its own grid."""

    def test_h2_all_electron(self, h2):
        _atoms, calc = h2
        field = calc.volumetric_field("density")
        # The grid quadrature *is* what defines the overlap the molecular
        # orbitals are orthonormal under, so this identity is exact.
        assert field.integral == pytest.approx(2.0, abs=1e-10)
        assert field.augmentation_charge == 0.0
        assert field.data.min() >= 0.0

    def test_frozen_core_is_refilled(self):
        atoms = _molecule("LiH", [[0, 0, 0], [0, 0, 1.6]], [8, 8, 8])
        calc = _solved(atoms, frozen_core=True)
        assert calc.solver._gradient_context["frozen"] == (0,)
        assert calc.num_particles == (1, 1)          # the *active* space
        field = calc.volumetric_field("density")
        assert field.integral == pytest.approx(4.0, abs=1e-10)

    def test_closed_shell_spin_density_vanishes(self, h2):
        _atoms, calc = h2
        field = calc.volumetric_field("spin_density")
        assert np.abs(field.data).max() < 1e-12
        assert field.integral == pytest.approx(0.0, abs=1e-12)

    def test_doublet_spin_density_integrates_to_one(self):
        atoms = _molecule("H3", [[0, 0, 0], [0, 0, 0.9], [0, 0.95, 0.45]],
                          [7, 7, 7])
        calc = _solved(atoms)
        assert calc.num_particles == (2, 1)
        field = calc.volumetric_field("spin_density")
        assert field.integral == pytest.approx(1.0, abs=1e-10)
        assert np.abs(field.data).max() > 1e-3          # and it is not flat
        alpha = calc.volumetric_field("alpha_density")
        beta = calc.volumetric_field("beta_density")
        assert alpha.integral == pytest.approx(2.0, abs=1e-10)
        assert beta.integral == pytest.approx(1.0, abs=1e-10)
        assert np.allclose(alpha.data - beta.data, field.data, atol=1e-14)

    def test_difference_density_integrates_to_zero(self, h2):
        _atoms, calc = h2
        field = calc.volumetric_field("difference_density")
        assert field.integral == pytest.approx(0.0, abs=1e-10)
        # H2 at equilibrium is weakly correlated, but not uncorrelated.
        assert np.abs(field.data).max() > 1e-4


class TestDensityFromTheDeterminant:
    """The RDM route must reproduce the textbook ``sum_occ |phi_i|^2``."""

    def test_hartree_fock_density_matches_the_explicit_sum(self, h2):
        _atoms, calc = h2
        integrals = calc.solver._gradient_context["integrals"]
        M = len(integrals.basis)
        gamma = np.zeros((2 * M, 2 * M), dtype=complex)
        gamma[0, 0] = gamma[M, M] = 1.0          # the HF determinant, (1, 1)
        field = volumetric_field(integrals, gamma, quantity="density",
                                 num_particles=(1, 1))
        expansion = OrbitalExpansion(integrals)
        explicit = 2.0 * np.abs(expansion.orbital(expansion.mo[:, 0])) ** 2
        assert np.abs(field.data.ravel() - explicit).max() < 1e-14
        assert field.integral == pytest.approx(2.0, abs=1e-10)

    def test_hartree_fock_difference_density_is_zero(self, h2):
        _atoms, calc = h2
        integrals = calc.solver._gradient_context["integrals"]
        M = len(integrals.basis)
        gamma = np.zeros((2 * M, 2 * M), dtype=complex)
        gamma[0, 0] = gamma[M, M] = 1.0
        field = volumetric_field(integrals, gamma,
                                 quantity="difference_density",
                                 num_particles=(1, 1))
        assert np.abs(field.data).max() < 1e-14

    def test_the_frozen_core_reference_is_the_ansatz_reference(self):
        """With a core, the reference is core + lowest *active* of each spin."""
        atoms = _molecule("LiH", [[0, 0, 0], [0, 0, 1.6]], [8, 8, 8])
        calc = _solved(atoms, frozen_core=True)
        integrals = calc.solver._gradient_context["integrals"]
        frozen = calc.solver._gradient_context["frozen"]
        n_act = len(integrals.basis) - len(frozen)
        gamma = np.zeros((2 * n_act, 2 * n_act), dtype=complex)
        gamma[0, 0] = gamma[n_act, n_act] = 1.0      # active HF determinant
        common = dict(frozen=frozen, num_particles=(1, 1),
                      n_spatial_orbitals=len(integrals.basis))
        density = volumetric_field(integrals, gamma, quantity="density",
                                   **common)
        assert density.integral == pytest.approx(4.0, abs=1e-10)
        difference = volumetric_field(integrals, gamma,
                                      quantity="difference_density", **common)
        assert np.abs(difference.data).max() < 1e-14

    def test_reference_occupations_follow_the_frozen_core(self):
        alpha, beta = reference_occupations(5, (1, 1), frozen=(0,))
        assert list(alpha) == [1.0, 1.0, 0.0, 0.0, 0.0]
        assert list(beta) == [1.0, 1.0, 0.0, 0.0, 0.0]
        alpha, beta = reference_occupations(4, (2, 1))
        assert list(alpha) == [1.0, 1.0, 0.0, 0.0]
        assert list(beta) == [1.0, 0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="do not fit"):
            reference_occupations(2, (3, 1))

    def test_spin_resolved_rdm_rejects_the_wrong_shape(self):
        with pytest.raises(ValueError, match="spin-orbital RDM"):
            spin_resolved_rdm(np.zeros((4, 4)), 3, frozen=())

    def test_a_non_hermitian_rdm_is_refused(self):
        """What makes the density real is the Hermiticity of the RDM."""
        gamma = np.eye(4, dtype=complex)
        gamma[0, 1] = 0.3
        gamma[1, 0] = 0.1                    # not the conjugate
        with pytest.raises(ValueError, match="not Hermitian"):
            spin_resolved_rdm(gamma, 2)
        gamma[1, 0] = 0.3
        alpha, beta = spin_resolved_rdm(gamma, 2)
        assert np.allclose(alpha, np.array([[1.0, 0.3], [0.3, 1.0]]))
        assert np.allclose(beta, np.eye(2))


class TestNaturalOrbitals:
    """Occupations are the correlation; the orbitals are an orthonormal set."""

    def test_occupations_are_physical_and_sum_to_n(self, h2):
        _atoms, calc = h2
        orbitals = calc.natural_orbitals()
        assert orbitals.n_electrons == pytest.approx(2.0, abs=1e-10)
        assert orbitals.occupations.min() >= -1e-12
        assert orbitals.occupations.max() <= 2.0 + 1e-12
        # Descending, which is what `index=0` means.
        assert np.all(np.diff(orbitals.occupations) <= 1e-14)

    def test_orthonormal_under_the_overlap(self, h2):
        _atoms, calc = h2
        integrals = calc.solver._gradient_context["integrals"]
        c = calc.natural_orbitals().coefficients
        gram = c.conj().T @ integrals.overlap() @ c
        assert np.abs(gram - np.eye(c.shape[1])).max() < 1e-10

    def test_a_determinant_gives_exactly_two_and_zero(self, h2):
        from mandacaru.algorithms.volumetric import state_natural_orbitals

        _atoms, calc = h2
        integrals = calc.solver._gradient_context["integrals"]
        M = len(integrals.basis)
        gamma = np.zeros((2 * M, 2 * M), dtype=complex)
        gamma[0, 0] = gamma[M, M] = 1.0
        orbitals = state_natural_orbitals(integrals, gamma)
        assert orbitals.occupations == pytest.approx([2.0, 0.0], abs=1e-12)

    def test_stretching_h2_opens_the_occupations(self, h2):
        """The picture the difference density draws, as numbers."""
        _atoms, tight = h2
        close = tight.natural_orbitals().occupations
        atoms = _molecule("H2", [[0, 0, 0], [0, 0, 2.5]], [8, 8, 8])
        far = _solved(atoms).natural_orbitals().occupations
        assert close[0] == pytest.approx(1.9749, abs=2e-3)
        assert close[1] == pytest.approx(0.0251, abs=2e-3)
        # At 2.5 Angstrom the bond is broken: the two orbitals share the pair.
        assert far[1] > 0.5
        assert far[0] - far[1] < close[0] - close[1]

    def test_the_orbital_file_carries_its_occupation(self, h2, tmp_path):
        _atoms, calc = h2
        field = calc.write_cube(tmp_path / "no0.cube",
                                quantity="natural_orbital", index=0)
        assert field.occupation == pytest.approx(
            calc.natural_orbitals().occupations[0])
        assert field.norm == pytest.approx(1.0, abs=1e-8)
        assert "occupation" in open(tmp_path / "no0.cube").readline()

    def test_molecular_orbital_is_the_reference_orbital(self, h2, tmp_path):
        _atoms, calc = h2
        field = calc.write_cube(tmp_path / "mo0.cube",
                                quantity="molecular_orbital", index=0)
        integrals = calc.solver._gradient_context["integrals"]
        expansion = OrbitalExpansion(integrals)
        expected = np.real(expansion.orbital(expansion.mo[:, 0]))
        assert np.abs(np.abs(field.data.ravel()) - np.abs(expected)).max() < 1e-14


# --------------------------------------------------------------------------- #
# The grid: atoms and data must line up.
# --------------------------------------------------------------------------- #

class TestGridAlignment:
    """Where the density sits, relative to where the viewer draws the nuclei."""

    def test_uncentered_molecule_lines_up(self, h2_offset, tmp_path):
        atoms, calc = h2_offset
        calc.write_cube(tmp_path / "d.cube")
        with open(tmp_path / "d.cube") as handle:
            back = read_cube(handle)
        # The nuclei come back where the geometry put them, not at the cell
        # origin and not at the grid center.
        assert np.allclose(back["atoms"].get_positions(),
                           atoms.get_positions(), atol=1e-5)
        # And the density peaks on a nucleus: the node nearest one of them.
        node = np.unravel_index(np.argmax(back["data"]), back["data"].shape)
        position = back["origin"] + back["spacing"].T @ np.array(node)
        distance = np.linalg.norm(atoms.get_positions() - position,
                                  axis=1).min()
        assert distance < H            # within one grid step of a nucleus

    def test_origin_is_the_first_node_not_the_cell_corner(self, h2_offset):
        atoms, calc = h2_offset
        grid = calc.solver._gradient_context["integrals"].grid
        field = calc.volumetric_field("density")
        assert np.allclose(field.origin,
                           [grid.X.flat[0], grid.Y.flat[0], grid.Z.flat[0]])
        # The box is centered on the molecule, so the origin is not the corner
        # of the cell -- which is exactly why ASE's own writer cannot be used.
        assert np.linalg.norm(field.origin) > 1e-6
        assert not np.allclose(field.origin * BOHR_TO_ANGSTROM,
                               np.zeros(3), atol=1e-6)

    def test_non_orthogonal_cell_runs_and_writes(self, tmp_path):
        cell = np.array([[6.0, 0.0, 0.0], [1.5, 6.0, 0.0], [0.0, 1.0, 6.0]])
        atoms = _molecule("H2", [[0, 0, 0], [0, 0, 0.74]], cell)
        calc = _solved(atoms)
        field = calc.write_cube(tmp_path / "skew.cube")
        assert field.integral == pytest.approx(2.0, abs=1e-10)
        with open(tmp_path / "skew.cube") as handle:
            back = read_cube(handle)
        assert np.allclose(back["spacing"],
                           field.grid.step.T * BOHR_TO_ANGSTROM, atol=1e-5)

    def test_a_skewed_output_grid_is_written_as_step_vectors(self, h2,
                                                             tmp_path):
        atoms, calc = h2
        skew = Grid(center=atoms.get_positions().mean(axis=0), box_size=0.0,
                    h=0.5, units="angstrom",
                    cell=np.array([[6.0, 0.0, 0.0], [1.5, 6.0, 0.0],
                                   [0.0, 1.2, 6.0]]), skew=True)
        assert not skew.is_orthogonal
        field = calc.write_cube(tmp_path / "skew.cube", grid=skew)
        with open(tmp_path / "skew.cube") as handle:
            back = read_cube(handle)
        assert np.allclose(back["spacing"], skew.step.T * BOHR_TO_ANGSTROM,
                           atol=1e-5)
        assert np.abs(back["data"] - field.data).max() < 1e-6

    def test_a_finer_grid_resamples_the_basis(self, h2):
        atoms, calc = h2
        fine = Grid(center=atoms.get_positions().mean(axis=0), box_size=0.0,
                    h=0.5 * H, units="angstrom", cell=atoms.get_cell())
        field = calc.volumetric_field("density", grid=fine)
        assert field.data.shape == fine.shape
        assert field.data.shape != calc.volumetric_field("density").data.shape
        # Off the calculation's own grid the two quadratures differ; that gap
        # is the discretization error of the run, not a bug.
        assert field.integral == pytest.approx(2.0, abs=0.05)


# --------------------------------------------------------------------------- #
# The register representation must not change the picture.
# --------------------------------------------------------------------------- #

class TestRegisterRepresentations:
    """The density is a property of the state, not of how it was encoded."""

    def test_parity_reduced_matches_jordan_wigner(self, h2):
        _atoms, jw = h2
        reference = jw.volumetric_field("density").data
        atoms = _molecule("H2", [[0, 0, 0], [0, 0, 0.74]], [6, 6, 6])
        tapered = _solved(atoms, mapping="parity_reduced")
        assert tapered.n_qubits == 2
        assert np.abs(tapered.volumetric_field("density").data
                      - reference).max() < 1e-12

    def test_particle_number_sector_matches_the_full_register(self, h2):
        _atoms, full = h2
        reference = full.volumetric_field("density").data
        atoms = _molecule("H2", [[0, 0, 0], [0, 0, 0.74]], [6, 6, 6])
        sector = _solved(atoms, sector=True)
        assert sector.solver._sector is not None
        # Two independent ADAPT runs, so they stop at very slightly different
        # points; what is pinned is that the sector representation changes
        # nothing about the density, not that an optimizer is deterministic.
        assert np.abs(sector.volumetric_field("density").data
                      - reference).max() < 1e-9


class TestExcitedStates:
    """``state=`` selects which level the picture is drawn from."""

    def test_subspace_levels_have_their_own_density(self):
        atoms = _molecule("H2", [[0, 0, 0], [0, 0, 0.74]], [6, 6, 6])
        atoms.calc = Mandacaru(method="subspace-vqe", basis="HAO", h=H,
                               num_states=2, trace=False)
        atoms.get_potential_energy()
        ground = atoms.calc.volumetric_field("density", state=0)
        excited = atoms.calc.volumetric_field("density", state=1)
        assert ground.integral == pytest.approx(2.0, abs=1e-10)
        assert excited.integral == pytest.approx(2.0, abs=1e-10)
        assert np.abs(ground.data - excited.data).max() > 1e-3
        # The first excited level of H2 in this space is the open-shell one:
        # its natural occupations are both close to one.
        occupations = atoms.calc.natural_orbitals(state=1).occupations
        assert occupations == pytest.approx([1.0, 1.0], abs=1e-2)

    def test_an_explicit_state_vector_is_accepted(self, h2):
        _atoms, calc = h2
        levels = calc.energy_levels(2)
        field = calc.volumetric_field("density", state=levels.states[1])
        assert field.integral == pytest.approx(2.0, abs=1e-10)
        assert np.abs(field.data
                      - calc.volumetric_field("density").data).max() > 1e-3


# --------------------------------------------------------------------------- #
# Pseudopotentials: the smooth valence density, and the charge it misses.
# --------------------------------------------------------------------------- #

@needs_paw
class TestPseudoValenceDensity:

    def test_augmentation_completes_the_electron_count(self):
        atoms = _molecule("H2", [[0, 0, 0], [0, 0, 0.74]], [6, 6, 6])
        calc = _solved(atoms, basis="PAW-LCAO")
        field = calc.volumetric_field("density")
        # The grid holds the *smooth* density; the augmentation charge is the
        # rest, and the two must add up to the valence electron count.
        assert field.augmentation_charge > 1e-3
        assert field.integral < 2.0 - 1e-3
        assert (field.integral + field.augmentation_charge
                == pytest.approx(2.0, abs=1e-10))
        assert field.n_electrons == pytest.approx(2.0, abs=1e-10)

    def test_the_file_says_it_is_a_pseudo_density(self, tmp_path):
        atoms = _molecule("H2", [[0, 0, 0], [0, 0, 0.74]], [6, 6, 6])
        calc = _solved(atoms, basis="PAW-LCAO")
        calc.write_cube(tmp_path / "paw.cube")
        comment = open(tmp_path / "paw.cube").readline()
        assert "augmentation" in comment
        assert "pseudo valence density" in comment

    def test_real_atomic_numbers_are_written_not_valence_charges(self,
                                                                 tmp_path):
        atoms = _molecule("LiH", [[0, 0, 0], [0, 0, 1.6]], [8, 8, 8])
        calc = _solved(atoms, basis="PAW-LCAO")
        field = calc.write_cube(tmp_path / "lih.cube")
        assert list(field.numbers) == [3, 1]        # lithium, not its ion
        with open(tmp_path / "lih.cube") as handle:
            back = read_cube(handle)
        assert back["atoms"].get_chemical_symbols() == ["Li", "H"]


# --------------------------------------------------------------------------- #
# Contracts.
# --------------------------------------------------------------------------- #

class TestContracts:

    def test_every_quantity_builds(self, h2, tmp_path):
        _atoms, calc = h2
        for quantity in QUANTITIES:
            field = calc.write_cube(tmp_path / f"{quantity}.cube",
                                    quantity=quantity)
            assert field.quantity == quantity
            assert field.output_path == str(tmp_path / f"{quantity}.cube")
            assert os.path.getsize(field.output_path) > 0
            assert field.summary()

    def test_an_unwritten_field_has_no_path(self, h2):
        _atoms, calc = h2
        assert calc.volumetric_field("density").output_path is None

    def test_unknown_quantity(self, h2):
        _atoms, calc = h2
        with pytest.raises(ValueError, match="unknown quantity"):
            calc.volumetric_field("charge_density")

    def test_orbital_index_out_of_range(self, h2):
        _atoms, calc = h2
        with pytest.raises(ValueError, match="out of range"):
            calc.volumetric_field("natural_orbital", index=7)

    def test_unavailable_state_index(self, h2):
        _atoms, calc = h2
        with pytest.raises(ValueError, match="state=2 is not available"):
            calc.volumetric_field("density", state=2)

    def test_before_a_run(self):
        calc = Mandacaru(method="adapt-vqe", basis="HAO", h=H, trace=False)
        with pytest.raises(RuntimeError, match="nothing has been solved yet"):
            calc.volumetric_field("density")

    def test_direct_mode_has_no_basis_to_draw(self, h2):
        """A qubit Hamiltonian with no geometry cannot produce a picture."""
        _atoms, calc = h2
        direct = Mandacaru(method="adapt-vqe", hamiltonian=calc.hamiltonian,
                           num_particles=calc.num_particles,
                           n_spatial_orbitals=2, trace=False)
        direct.run()
        with pytest.raises(NotImplementedError, match="needs the basis"):
            direct.volumetric_field("density")

    def test_unknown_component(self, h2):
        _atoms, calc = h2
        with pytest.raises(ValueError, match="component must be one of"):
            calc.volumetric_field("natural_orbital", component="phase")

    def test_component_choices_agree_on_a_real_orbital(self, h2):
        _atoms, calc = h2
        auto = calc.volumetric_field("natural_orbital", index=0)
        modulus = calc.volumetric_field("natural_orbital", index=0,
                                        component="modulus")
        imaginary = calc.volumetric_field("natural_orbital", index=0,
                                          component="imag")
        assert np.allclose(np.abs(auto.data), modulus.data, atol=1e-14)
        assert np.abs(imaginary.data).max() < 1e-12

    def test_wrong_number_of_atomic_numbers(self, h2):
        _atoms, calc = h2
        integrals = calc.solver._gradient_context["integrals"]
        gamma, _ = calc._state_rdms(calc.solver, two_body=False)
        with pytest.raises(ValueError, match="atomic numbers"):
            volumetric_field(integrals, gamma, numbers=[1, 1, 1])
