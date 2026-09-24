# -*- coding: utf-8 -*-
# file: core/periodic.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Integrals for a **crystal** rather than a molecule in a box.

:class:`~mandacaru.core.hamiltonian.MolecularIntegrals` describes an isolated
system: the Coulomb kernel is zero-padded so the tail cannot wrap the box, and
the external potential is the bare ``-Z/r`` of the nuclei that happen to be
inside it.  Both are right for a molecule and wrong for a solid.
:class:`PeriodicIntegrals` replaces the three electrostatic terms with their
lattice sums, leaving everything downstream -- the orthonormalization, RHF, the
second-quantized Hamiltonian, the mappings, the solvers -- untouched.

What changes, and why each one has to:

* **Hartree / ERI** -- the reciprocal-space kernel ``4 pi / G^2`` with
  ``G = 0`` dropped (:class:`~mandacaru.integrals.poisson.PeriodicPoissonSolver`).
  The grid *is* the cell here, so the circular convolution the isolated solver
  pads against is exactly the physics.
* **External potential** -- the zero-mean Ewald potential of the whole ion
  lattice (:func:`~mandacaru.core.ewald.ewald_potential`), not a finite window
  of bare ``-Z/r``.  A finite window has a mean that moves with the window,
  which shows up as an arbitrary shift per electron.
* **Ion-ion** -- the Ewald energy (:func:`~mandacaru.core.ewald.ewald_energy`),
  not a pairwise sum over the cell.
* **Kinetic** -- the spectral (FFT) operator, which is periodic on the cell;
  the finite-difference stencil has a hard wall at the box edge.

**The three share one convention and must.** Each drops its own ``G = 0`` term
against a uniform neutralizing background.  For a neutral cell the three
divergences cancel exactly -- but only because all three are built the same
way.

**The Madelung term.** Dropping ``G = 0`` from the Coulomb kernel removes more
than the Hartree divergence: it also removes each electron's interaction with
its own periodic images and the background, which nothing cancels.  Its value
is ``1/2 N_e v_M``, with ``v_M`` the lattice's Madelung potential
(:func:`~mandacaru.core.ewald.madelung_constant`).  That is the
Gygi-Baldereschi correction, which for a Gamma-point supercell is this single
constant rather than an auxiliary-function construction, and it is carried in
:attr:`constant_energy`.  Without it the energy per cell does not converge with
supercell size -- it wanders by electronvolts and, for a chain in a
transversely padded cell, eventually grows.

**Limitations.** The kernel is periodic in all three directions.  For a chain
or a slab the transverse images are spurious; a neutral chain's images interact
only through quadrupoles (``L^-5``) and are negligible, but a **slab carrying a
dipole needs a 2-D-truncated kernel**.  That kernel now exists and is verified
-- :func:`~mandacaru.integrals.poisson.slab_truncated_kernel`, reached as
``PeriodicPoissonSolver(..., truncation="slab", axis=2)`` -- but it is **not
wired into this class**, and the reason is worth stating rather than leaving as
an omission.

The Hartree term is one of *three* electrostatic terms here, and they share a
convention: the electron-ion potential is a 3-D Ewald lattice sum
(:func:`~mandacaru.core.ewald.ewald_potential`), the ion-ion energy is the 3-D
Ewald energy, and :attr:`constant_energy` carries the 3-D Madelung constant.
Truncating only the electron-electron kernel would give a total energy assembled
from two different boundary conditions -- each term finite and plausible, the sum
meaningless -- and for a dipolar slab the ion terms carry exactly the image
interaction the truncation was meant to remove.  Making the option available
here therefore needs the matching 2-D forms of all three (an Ewald split against
the truncated kernel, with the real-space sum restricted to in-plane lattice
vectors and the ``G = 0`` background re-derived), plus their derivatives in
:mod:`mandacaru.algorithms.periodic_forces`.  Until then the combination is
refused by name rather than silently mixed.

Forces and stress *are* differentiated through this construction, in
:mod:`mandacaru.algorithms.periodic_forces`: the Ewald potential is rebuilt with
one ion displaced, the ion-ion Ewald gradient is analytic, and the Madelung term
-- a function of the cell alone -- drops out of the force and reappears in the
stress.
"""

from __future__ import annotations

import numpy as np

from .ewald import ewald_energy, ewald_potential, madelung_constant
from .hamiltonian import MolecularIntegrals
from ..units import to_bohr


class PeriodicIntegrals(MolecularIntegrals):
    r"""Integrals over a periodic cell, with lattice-summed electrostatics.

    Parameters
    ----------
    nuclei : sequence of (Z, position)
        The nuclei of **one** cell, in ``units``.
    basis : sequence of BasisFunction
        The basis of one cell.  Its functions are sampled on the cell grid; a
        function wider than the cell is not represented correctly, so keep the
        cell larger than the orbital support.
    grid : Grid
        The cell grid.  It *is* the periodic cell: node ``n`` and node
        ``n + shape`` are the same point.
    cell : (3, 3) array_like
        Lattice vectors as **rows** (the ASE convention), in ``units``.
    n_electrons : int
        Electrons per cell, needed for the Madelung term.  ``None`` leaves that
        term out, which is only correct if the caller adds it.
    kinetic : {"spectral", "fd"}
        Defaults to ``"spectral"``, the operator that is periodic on the cell.
    **kwargs
        Passed through to :class:`MolecularIntegrals`.
    """

    def __init__(self, nuclei, basis, grid, cell, n_electrons: int | None = None,
                 units: str = "angstrom", kinetic: str = "spectral",
                 truncation: str = "none", **kwargs):
        if str(truncation).strip().lower() != "none":
            raise NotImplementedError(
                f"truncation={truncation!r} is not available on the periodic "
                f"Hamiltonian yet.  The two-dimensionally truncated kernel "
                f"itself is implemented and verified "
                f"(mandacaru.integrals.poisson.slab_truncated_kernel), but this "
                f"class assembles three electrostatic terms that must share one "
                f"boundary condition -- the Hartree kernel, the electron-ion "
                f"Ewald potential and the ion-ion Ewald energy, plus the "
                f"Madelung constant in constant_energy.  Truncating only the "
                f"first would give a total energy built from two different "
                f"boundary conditions: every term finite and plausible, the sum "
                f"meaningless, and for a dipolar slab the ion terms would still "
                f"carry the image interaction the truncation removes.  Use the "
                f"solver directly for a Hartree-only study, and see this "
                f"module's docstring for what wiring it up requires.")
        super().__init__(nuclei, basis, grid, units=units, kinetic=kinetic,
                         periodic=True, **kwargs)
        self.cell = to_bohr(np.asarray(cell, dtype=float).reshape(3, 3), units)
        if abs(float(np.linalg.det(self.cell))) <= 0.0:
            raise ValueError("the periodic cell has zero volume")
        self._check_commensurate(grid)
        self.n_electrons = None if n_electrons is None else int(n_electrons)
        self.constant_energy = self._madelung_energy()

    def _check_commensurate(self, grid) -> None:
        r"""The grid must reproduce the cell exactly: ``shape @ step == cell``.

        Everything periodic here assumes one period.  The FFT Coulomb kernel
        takes it from ``step @ diag(shape)``; the image sum and the Ewald terms
        take it from the cell.  A grid built with a repeated endpoint -- the
        molecular convention, ``step = a / (n - 1)`` -- makes the first longer
        than the second by one step, so the Hamiltonian describes two different
        lattices at once.  Build the grid with ``Grid(..., periodic=True)``.
        """
        # `grid.step` holds the step vectors as COLUMNS, so `spanned` does too,
        # while `self.cell` holds the lattice vectors as ROWS (the ASE
        # convention this class documents).  Compare like with like.  A
        # diagonal matrix equals its own transpose, so omitting this is
        # invisible for a cubic, tetragonal or orthorhombic cell -- and rejects
        # every hexagonal, monoclinic and triclinic one, whose off-diagonal
        # terms then sit on the wrong side.
        spanned = np.asarray(grid.step, dtype=float) @ np.diag(grid.shape)
        expected = self.cell.T
        error = float(np.abs(spanned - expected).max())
        scale = float(np.abs(self.cell).max()) or 1.0
        if error > 1e-8 * scale:
            # Report the lattice vectors themselves, not the diagonals: for a
            # skewed cell the diagonals can agree while the cell does not, and
            # a message showing two identical triples explains nothing.
            raise ValueError(
                "the grid is not commensurate with the periodic cell: "
                f"shape @ step spans {np.round(spanned.T, 5).tolist()} Bohr "
                f"but the cell is {np.round(self.cell, 5).tolist()} Bohr "
                f"(largest difference {error:.3e} Bohr).  A periodic "
                "calculation needs exactly one period on the grid -- build it "
                "with Grid(..., cell=cell, periodic=True), which drops the "
                "repeated endpoint.")

    # -- the electron's potential energy in the ion lattice ---------------- #
    def external_potential(self):
        """The Ewald potential of the ion lattice, as the engine's callable.

        Zero-mean, like the Coulomb kernel that accompanies it.  The charges
        are ``-Z`` because this is what an *electron* feels.
        """
        if self.uses_pseudopotentials:
            raise NotImplementedError(
                "a pseudopotential has not been given a periodic lattice sum "
                "yet; use an all-electron basis for a periodic calculation.")
        sites = np.array([to_bohr(position, self.units)
                          for _charge, position in self.nuclei], dtype=float)
        charges = np.array([-float(charge) for charge, _p in self.nuclei])

        def potential(x, y, z):
            points = np.stack([np.asarray(x, dtype=float).ravel(),
                               np.asarray(y, dtype=float).ravel(),
                               np.asarray(z, dtype=float).ravel()], axis=1)
            values = ewald_potential(sites, charges, self.cell, points,
                                     softening=self._potentials.softening)
            return values.reshape(np.asarray(x).shape)

        return potential

    # -- the constants the lattice contributes ----------------------------- #
    @property
    def nuclear_repulsion(self) -> float:
        """Ion-ion energy per cell by Ewald summation (Hartree)."""
        if not self.nuclei:
            return 0.0
        sites = np.array([to_bohr(position, self.units)
                          for _charge, position in self.nuclei], dtype=float)
        charges = np.array([float(charge) for charge, _p in self.nuclei])
        return float(ewald_energy(sites, charges, self.cell))

    def madelung_constant(self) -> float:
        """``v_M`` of this cell (Hartree per unit charge)."""
        return float(madelung_constant(self.cell))

    def _madelung_energy(self) -> float:
        r"""``1/2 N_e v_M`` -- the term the dropped ``G = 0`` leaves out.

        Constant in a fixed particle-number sector, because the operator it
        regularizes is the number operator, so it does not depend on the
        correlated state at all.
        """
        if self.n_electrons is None:
            return 0.0
        return 0.5 * self.n_electrons * self.madelung_constant()
