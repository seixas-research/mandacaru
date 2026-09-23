# -*- coding: utf-8 -*-
# file: test/core/test_periodic.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""A crystal rather than a molecule in a box: :class:`PeriodicIntegrals`.

The electrostatic primitives are pinned elsewhere -- the periodic Coulomb
kernel in ``test/integrals/test_periodic_poisson.py``, the Ewald potential and
Madelung constant in ``test/core/test_ewald.py``.  What is checked here is that
they **assemble** into an energy, which is where two defects hid that every
individual primitive passed:

* the periodic image sum translated by ``grid.cell``, which is stored in the
  Grid's user units, while the sampled coordinates are in Bohr -- so the images
  sat 1.89x too close and several copies of every orbital landed inside the
  cell.  The orbitals still looked normalized one at a time; only the sampled
  stack gave it away (norm 4.8 instead of 1).  The Bohr/Angstrom trap, hiding
  in the basis rather than in a gradient.
* the overlap was taken over the *bare* functions while the kinetic, potential
  and two-body terms used the image-summed stack, so ``S`` described a
  different basis from ``h`` and ``g``.

Both are cheap to catch and neither was visible in a single-primitive test,
which is why the first three assertions below are about self-consistency of the
sampled basis rather than about any energy.
"""

import functools

import numpy as np
import pytest

from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.core.ewald import madelung_constant
from mandacaru.core.periodic import PeriodicIntegrals
from mandacaru.integrals import Grid
from mandacaru.units import from_hartree, to_bohr

SPACING = 0.25          # Angstrom
BOND = 0.74             # H2 bond length, Angstrom


@functools.lru_cache(maxsize=8)
def hydrogen_molecule(length, periodic=True):
    """H2 centered in a cubic cell of side ``length`` (Angstrom).

    Cached: the tests below want the same few cells, and building one is
    seconds of integral work that does not depend on which assertion reads it.
    Nothing here may mutate the returned object -- see :func:`energy_parts`,
    which exists so the Madelung test can drop a term without doing so.
    """
    cell = np.diag([length] * 3)
    nuclei = [(1.0, np.array([length / 2 - BOND / 2, length / 2, length / 2])),
              (1.0, np.array([length / 2 + BOND / 2, length / 2, length / 2]))]
    basis = minimal_hao_basis(nuclei)
    grid = Grid(center=[length / 2] * 3, box_size=0.0, h=SPACING,
                units="angstrom", cell=cell, periodic=periodic)
    softening = 0.5 * min(grid.dx, grid.dy, grid.dz)
    if not periodic:
        return MolecularIntegrals(nuclei, basis, grid, units="angstrom",
                                  softening=softening)
    return PeriodicIntegrals(nuclei, basis, grid, cell, n_electrons=2,
                             units="angstrom", softening=softening)


@functools.lru_cache(maxsize=8)
def energy_parts(length, periodic=True):
    """``(electronic, ion-ion, Madelung)`` of that cell, in Hartree.

    Split rather than summed so a caller can leave a term out **without
    mutating the integrals**: the Madelung test used to build a second copy and
    set ``constant_energy = 0`` on it, which both doubled the work and would
    poison a cache.
    """
    integrals = hydrogen_molecule(length, periodic)
    result = integrals.hartree_fock(2)
    return (float(result.electronic_energy), float(integrals.nuclear_repulsion),
            float(integrals.constant_energy))


def total_energy(length, periodic=True, madelung=True):
    """RHF total energy in eV, with or without the Madelung constant."""
    electronic, ion_ion, constant = energy_parts(length, periodic)
    total = electronic + ion_ion + (constant if madelung else 0.0)
    return from_hartree(total, "eV")


class TestTheSampledBasisIsPeriodic:
    """The checks that would have caught the unit bug immediately."""

    @pytest.mark.parametrize("length", [8.0, 12.0])
    def test_each_periodic_function_is_still_normalized(self, length):
        """``phi^per`` is a sum of images, not a sum of copies in one cell.

        With the translations taken in the wrong unit the images fall inside
        the cell and the norm jumps to ~4.8.  Genuine image overlap at these
        cell sizes is under a percent.
        """
        integrals = hydrogen_molecule(length)
        stack = integrals._engine._psi
        norms = np.real(np.einsum("ag,ag->a", np.conj(stack), stack))
        norms = norms * integrals.grid.dV
        assert norms == pytest.approx(np.ones(norms.size), abs=2e-2)

    def test_the_overlap_describes_the_functions_the_rest_is_built_from(self):
        """``S`` must come from the same sampled stack as ``T``, ``V`` and ``g``."""
        integrals = hydrogen_molecule(10.0)
        stack = integrals._engine._psi
        expected = (np.conj(stack) @ stack.T) * integrals.grid.dV
        assert integrals.bare_overlap() == pytest.approx(
            0.5 * (expected + expected.conj().T), abs=1e-12)

    def test_a_grid_that_is_not_one_period_is_refused(self):
        """``shape @ step`` must reproduce the cell, or two lattices coexist."""
        length = 10.0
        cell = np.diag([length] * 3)
        nuclei = [(1.0, np.array([length / 2] * 3))]
        # periodic=False keeps the molecular convention: a repeated endpoint.
        grid = Grid(center=[length / 2] * 3, box_size=0.0, h=SPACING,
                    units="angstrom", cell=cell, periodic=False)
        with pytest.raises(ValueError, match="not commensurate"):
            PeriodicIntegrals(nuclei, minimal_hao_basis(nuclei), grid, cell,
                              n_electrons=1, units="angstrom")


class TestTheGridIsOnePeriod:
    @pytest.mark.parametrize("length", [8.0, 10.0, 12.0])
    def test_shape_times_step_is_the_cell(self, length):
        cell = np.diag([length] * 3)
        grid = Grid(center=[length / 2] * 3, box_size=0.0, h=SPACING,
                    units="angstrom", cell=cell, periodic=True)
        spanned = np.asarray(grid.step) @ np.diag(grid.shape)
        assert spanned == pytest.approx(to_bohr(cell, "angstrom"), abs=1e-10)

    def test_the_molecular_grid_is_deliberately_not(self):
        """The molecular convention repeats the endpoint, and must keep doing so."""
        cell = np.diag([10.0] * 3)
        grid = Grid(center=[5.0] * 3, box_size=0.0, h=SPACING,
                    units="angstrom", cell=cell)
        spanned = np.asarray(grid.step) @ np.diag(grid.shape)
        assert not np.allclose(spanned, to_bohr(cell, "angstrom"), atol=1e-6)


class TestNonOrthogonalCells:
    """The commensurability check must compare like with like.

    ``grid.step`` holds its step vectors as **columns**, so ``shape @ step``
    does too, while ``PeriodicIntegrals.cell`` holds the lattice vectors as
    **rows** -- the ASE convention it documents.  Comparing the two without a
    transpose is invisible for a cubic, tetragonal or orthorhombic cell,
    because a diagonal matrix equals its own transpose, and refuses **every**
    hexagonal, monoclinic and triclinic cell, whose off-diagonal terms land on
    the wrong side.

    Every other periodic test in the suite uses a square lattice or a 1-D
    chain, which is exactly why none of them caught it.
    """

    @staticmethod
    def lattice(cell, h=0.4):
        cell = np.asarray(cell, dtype=float)
        center = 0.5 * cell.sum(axis=0)
        nuclei = [(1.0, center)]
        grid = Grid(center=center, box_size=0.0, h=h, units="angstrom",
                    cell=cell, periodic=True)
        return PeriodicIntegrals(
            nuclei, minimal_hao_basis(nuclei), grid, cell, n_electrons=1,
            units="angstrom",
            softening=0.5 * min(grid.dx, grid.dy, grid.dz))

    @pytest.mark.parametrize("name, cell", [
        ("hexagonal", [[2.6, 0.0, 0.0], [-1.3, 2.6 * np.sqrt(3) / 2, 0.0],
                       [0.0, 0.0, 9.0]]),
        ("monoclinic", [[2.6, 0.0, 0.0], [0.4, 2.6, 0.0], [0.0, 0.0, 9.0]]),
        ("triclinic", [[3.0, 0.0, 0.0], [0.9, 2.8, 0.0], [0.5, 0.4, 3.2]]),
    ])
    def test_a_skewed_cell_is_accepted(self, name, cell):
        """It used to raise `not commensurate` for all three."""
        integrals = self.lattice(cell)
        spanned = np.asarray(integrals.grid.step) @ np.diag(integrals.grid.shape)
        # The grid really does span the cell -- transposed, which is the point.
        assert spanned == pytest.approx(to_bohr(np.asarray(cell, float),
                                                "angstrom").T, abs=1e-10)

    def test_a_skewed_cell_is_still_checked(self):
        """The fix must not turn the check off: a wrong grid must still raise."""
        cell = np.array([[2.6, 0.0, 0.0], [-1.3, 2.6 * np.sqrt(3) / 2, 0.0],
                         [0.0, 0.0, 9.0]])
        center = 0.5 * cell.sum(axis=0)
        nuclei = [(1.0, center)]
        # periodic=False keeps the molecular convention: a repeated endpoint.
        grid = Grid(center=center, box_size=0.0, h=0.4, units="angstrom",
                    cell=cell, periodic=False)
        with pytest.raises(ValueError, match="not commensurate"):
            PeriodicIntegrals(nuclei, minimal_hao_basis(nuclei), grid, cell,
                              n_electrons=1, units="angstrom")

    def test_the_message_shows_the_vectors_not_the_diagonals(self):
        """A skewed cell can match on the diagonal and still be wrong.

        The old message printed only ``np.diag(...)`` of each, so it reported a
        mismatch while displaying two identical triples.
        """
        cell = np.array([[2.6, 0.0, 0.0], [0.9, 2.6, 0.0], [0.0, 0.0, 9.0]])
        center = 0.5 * cell.sum(axis=0)
        nuclei = [(1.0, center)]
        grid = Grid(center=center, box_size=0.0, h=0.4, units="angstrom",
                    cell=cell, periodic=False)
        with pytest.raises(ValueError) as excinfo:
            PeriodicIntegrals(nuclei, minimal_hao_basis(nuclei), grid, cell,
                              n_electrons=1, units="angstrom")
        message = str(excinfo.value)
        assert "largest difference" in message
        # Full 3x3 vectors, so the skew is visible: nested brackets.
        assert message.count("[[") == 2


class TestTheAssembledEnergy:
    """H2 is neutral and compact, so the periodic answer must be close."""

    def test_it_is_near_the_isolated_limit(self):
        """It was +74 eV out while the image lattice was in the wrong unit."""
        assert abs(total_energy(12.0) - total_energy(12.0, periodic=False)) < 1.5

    @pytest.mark.parametrize("length", [8.0, 12.0])
    def test_the_madelung_term_is_what_is_left_over_without_it(self, length):
        r"""Dropping ``1/2 N_e v_M`` must cost exactly that much.

        The pair-density ERI carries no diagonal, so neither an electron's
        self-energy nor its self-image is in it; the constant is the
        Gamma-point form of the Gygi-Baldereschi correction.  Its absence is a
        shift of ``-N_e v_M / 2``, which is what this measures.
        """
        with_term = total_energy(length)
        without = total_energy(length, madelung=False)

        cell = hydrogen_molecule(length).cell
        expected = from_hartree(-0.5 * 2 * madelung_constant(cell), "eV")
        assert without - with_term == pytest.approx(expected, rel=0.25)

    def test_the_remaining_drift_is_the_finite_size_tail(self):
        """What is left scales as ``1/Omega`` -- the quadrupole term.

        A neutral molecule's periodic images interact through their quadrupoles
        once the monopole and the self-image are accounted for, so the residual
        falls off as the cell volume grows rather than staying constant.
        """
        energies = [total_energy(length) for length in (8.0, 10.0, 12.0)]
        steps = np.diff(energies)
        assert np.all(np.abs(steps) < 0.15)
        # The tail shrinks: each step is smaller than the one before it.
        assert abs(steps[1]) < abs(steps[0])
