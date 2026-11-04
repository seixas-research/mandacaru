# -*- coding: utf-8 -*-
# file: core/mpc.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The exchange-correlation hole in a periodic cell, and the MPC correction.

A Born-von Karman supercell gives every electron a periodic array of images of
itself.  The **Hartree** part of that is physical -- the charge density really
is periodic, and the Ewald kernel
(:class:`~mandacaru.integrals.poisson.PeriodicPoissonSolver`) is the right way
to sum it.  The **exchange-correlation hole** is not.  The hole is a localized
object, roughly the size of the interparticle spacing, that follows one
electron around; in the real crystal it has no images, and in the supercell it
interacts with ``N_c - 1`` copies of itself.  That spurious interaction is the
leading finite-size error of a neutral cell once the Madelung term
(:mod:`mandacaru.core.ewald`) has been accounted for, and it falls off only as
:math:`1/\Omega`.

The **model periodic Coulomb** (MPC) construction of Fraser *et al.* and
Williamson *et al.* removes it by using a different kernel for each part:

.. math::

    E_{ee}^{\text{MPC}}
      = \tfrac{1}{2}\iint \rho(\mathbf r)\,\rho(\mathbf r')\,
        v_{\text{E}}(\mathbf r - \mathbf r')
      + \tfrac{1}{2}\iint \bigl[\rho_2(\mathbf r, \mathbf r')
        - \rho(\mathbf r)\rho(\mathbf r')\bigr]\,
        f(\mathbf r - \mathbf r') ,

the Hartree term with the periodic :math:`v_{\text{E}}` and the hole with
:math:`f`, the **bare** :math:`1/r` truncated at the Wigner-Seitz cell of the
lattice.  Subtracting the plain Ewald energy
:math:`\tfrac12 \iint \rho_2\, v_{\text{E}}` leaves a correction that involves
*only* the hole,

.. math::

    \Delta E = \tfrac{1}{2} \iint
      \bigl[\rho_2(\mathbf r, \mathbf r') - \rho(\mathbf r)\rho(\mathbf r')\bigr]
      \bigl[f(\mathbf r - \mathbf r') - v_{\text{E}}(\mathbf r - \mathbf r')\bigr] ,
    \label{eq-delta}

which is this module's central quantity.  In the second-quantized basis the two
pair densities are the two-body RDM and the direct product of the one-body one,

.. math::

    \Delta E = \tfrac{1}{2} \sum_{pqrs}
      \bigl(f_{pqrs} - g^{\text{E}}_{pqrs}\bigr)
      \bigl(\Gamma_{pqrs} - D_{pr} D_{qs}\bigr) ,

so it needs a **second two-electron tensor**, built with :math:`f` instead of
the periodic kernel, and the RDMs of the converged state.  Both are things
Mandacaru already has: the tensor through
:meth:`~mandacaru.integrals.engine.IntegralEngine.two_body` with an injected
solver, the RDMs through :mod:`mandacaru.algorithms.rdm`.

Two things this module is careful about
---------------------------------------

**"Truncated at the Wigner-Seitz cell" means the minimum image, and nothing
is set to zero.**  The kernel lives on the torus of grid index offsets, and
each offset is one lattice-equivalence class with exactly **one**
representative inside the Wigner-Seitz cell -- the minimum image.  So
:math:`f(\mathbf d) = 1/|\mathbf d_{\min}|` everywhere, where
:math:`\mathbf d_{\min}` minimizes :math:`|\mathbf d + \mathbf R|` over
lattice translations.  For a cubic cell the minimum image is just the offset
folded to :math:`[-n/2, n/2)`; for a hexagonal, FCC or triclinic cell it is
not, so the minimum is searched over a shell of translations.

Zeroing the offsets whose *folded* value falls outside the Wigner-Seitz cell
looks like the truncation the name suggests and is wrong: it discards those
classes without ever sampling their in-cell representative, so the kernel ends
up covering only the intersection of the parallelepiped with the Wigner-Seitz
cell.  Measured before the fix -- 85 % of a hexagonal cell's volume and 76 %
of an FCC one, when the Wigner-Seitz cell is a primitive cell and has the
primitive volume **exactly**.  :attr:`TruncatedCoulombSolver.enclosed_volume`
is the check that caught it and is kept for that reason.

**The self node keeps the voxel average.**  The :math:`\mathbf d = 0` node of
the kernel carries :math:`\frac{1}{dV}\int_{\text{voxel}} d^3r / |r|`
(:func:`~mandacaru.integrals.poisson.voxel_self_potential`) rather than an
ad-hoc softening.  The same node is treated the same way in
:math:`g^{\text{E}}`, so it cancels in :math:`f - g^{\text{E}}` to the
accuracy of that treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft

from .._native import voxel_self_potential, wigner_seitz_kernel as _native_wigner_seitz_kernel

__all__ = ["TruncatedCoulombSolver", "wigner_seitz_kernel",
           "FiniteSizeCorrection", "exchange_correlation_hole_energy",
           "voxel_self_potential", "PeriodicPoissonSolver", "spatial_rdms",
           "PeriodicIntegralsAdapter", "MolecularIntegrals", "minimal_hao_basis"]


class PeriodicPoissonSolver:
    """Fourier-space Poisson solver on a periodic grid."""

    def __init__(self, shape, step=None, spacing=None, workers: int = -1):
        self.shape = tuple(int(s) for s in shape)
        if step is not None:
            self.step = np.asarray(step, dtype=float)
            if self.step.shape != (3, 3):
                raise ValueError(f"step must be a 3x3 matrix, got {self.step.shape}")
        elif spacing is not None:
            self.step = np.diag([float(spacing)] * 3)
        else:
            raise ValueError("give step or spacing")
        self.dV = abs(float(np.linalg.det(self.step)))
        if self.dV <= 0.0:
            raise ValueError("the voxel has zero volume (degenerate step)")
        self.cell = self.step @ np.diag(self.shape)
        self.volume = abs(float(np.linalg.det(self.cell)))
        self.L = self.shape
        self.workers = workers

        reciprocal = 2.0 * np.pi * np.linalg.inv(self.cell).T
        kx = np.fft.fftfreq(self.shape[0]) * self.shape[0]
        ky = np.fft.fftfreq(self.shape[1]) * self.shape[1]
        kz = np.fft.fftfreq(self.shape[2]) * self.shape[2]

        K = np.zeros(self.shape, dtype=float)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    m = np.array([kx[i], ky[j], kz[k]])
                    kvec = reciprocal @ m
                    g2 = float(np.dot(kvec, kvec))
                    if g2 > 0:
                        K[i, j, k] = (4.0 * np.pi / g2) / self.dV
        self._kernel = K

    def solve(self, rho_flat: np.ndarray) -> np.ndarray:
        return self.solve_stack(rho_flat[None, :])[0]

    def solve_stack(self, rho_stack: np.ndarray,
                    out: np.ndarray | None = None) -> np.ndarray:
        rho_stack = np.asarray(rho_stack, dtype=float)
        P = rho_stack.shape[0]
        N = int(np.prod(self.shape))
        if out is None:
            out = np.empty((P, N), dtype=float)
        for p in range(P):
            rho_3d = rho_stack[p].reshape(self.shape)
            r_fft = sfft.fftn(rho_3d, workers=self.workers)
            pot_3d = sfft.ifftn(r_fft * self._kernel, workers=self.workers).real * self.dV
            out[p] = pot_3d.ravel()
        return out


#: Lattice shells searched when deciding which translation minimizes the
#: distance.  One shell is enough for any Bravais lattice whose grid spans a
#: single period: the competing images of a point inside the cell are its
#: nearest neighbors.  Two is cheap and leaves margin for a very skewed cell.
MINIMUM_IMAGE_SHELLS = 2

#: Two distances closer than this (in Bohr) count as tied, and a tie is
#: resolved in favor of the zero translation -- a node exactly on the
#: Wigner-Seitz boundary is counted as inside, once.
TIE_TOLERANCE = 1e-10


def wigner_seitz_kernel(shape, step, shells: int = MINIMUM_IMAGE_SHELLS):
    r"""The bare ``1/r`` truncated at the Wigner-Seitz cell, on the grid.

    Parameters
    ----------
    shape : (int, int, int)
        Nodes per lattice vector.  The grid **is** the period.
    step : (3, 3) array_like
        Step vectors as columns, in Bohr (``Grid.step``).
    shells : int
        Lattice shells searched for the minimum image.

    Returns
    -------
    ndarray of shape ``shape``
        :math:`f(\mathbf d)` at every index offset, with the offsets laid out
        in FFT order so the array can be transformed directly.

    Notes
    -----
    The value at every offset is ``1/|d_min|``, the minimum image; nothing is
    zeroed.  See the module docstring for why the zeroing this function's name
    suggests would be wrong.  Validated against the analytic
    :math:`\int_{\text{cube}} d^3r/|r| = L^2 \times 2.3800774`, which the
    grid sum approaches as **1/n^2** (log-log slope -1.999 over ``n = 8..96``;
    an earlier note here said ``1/n``, which the same data already refuted --
    4.38e-3, 1.10e-3 and 6.17e-4 at ``n = 12, 24, 32`` fall by 3.98 and 1.78,
    against 4 and 1.78 for a quadratic rate).
    """
    shape = tuple(int(n) for n in shape)
    step = np.ascontiguousarray(np.asarray(step, dtype=float))
    if step.shape != (3, 3):
        raise ValueError(f"step must be a 3x3 matrix, got {step.shape}")
    return np.ascontiguousarray(_native_wigner_seitz_kernel(shape, step, shells=int(shells)))


class TruncatedCoulombSolver:
    """Circular convolution with the Wigner-Seitz-truncated bare ``1/r``.

    The interface matches
    :class:`~mandacaru.integrals.poisson.PeriodicPoissonSolver` -- ``shape``,
    ``step``, ``L``, ``dV``, ``solve``, ``solve_stack`` -- so it can be handed
    to :meth:`~mandacaru.integrals.engine.IntegralEngine.two_body` in its
    place and produce the matching two-electron tensor.

    No zero-padding: the kernel is already confined to one cell by
    construction, so the circular convolution the FFT performs is the intended
    one rather than an artifact to pad against.
    """

    def __init__(self, shape, step, workers: int = -1,
                 shells: int = MINIMUM_IMAGE_SHELLS):
        self.shape = tuple(int(s) for s in shape)
        self.step = np.asarray(step, dtype=float)
        if self.step.shape != (3, 3):
            raise ValueError(f"step must be a 3x3 matrix, got {self.step.shape}")
        self.dV = abs(float(np.linalg.det(self.step)))
        if self.dV <= 0.0:
            raise ValueError("the voxel has zero volume (degenerate step)")
        self.cell = self.step @ np.diag(self.shape)
        self.volume = abs(float(np.linalg.det(self.cell)))
        #: Transform shape: the grid itself, as for the periodic solver.
        self.L = self.shape
        self.workers = workers
        self.real_kernel = wigner_seitz_kernel(self.shape, self.step, shells)
        transform = sfft.fftn(self.real_kernel, workers=self.workers)
        #: True when the kernel is even (a minimum-image kernel is, up to
        #: ties on the Wigner-Seitz boundary), so its transform is real.
        self.even = bool(np.array_equal(self.real_kernel,
                                        self.real_kernel[::-1, ::-1, ::-1]))
        self._kernel = transform.real.copy() if self.even else transform

    @property
    def enclosed_volume(self) -> float:
        """Volume the kernel covers, in Bohr^3.

        The minimum-image kernel is non-zero at every node, so this equals the
        cell volume by construction -- the Wigner-Seitz cell is a primitive
        cell and has the primitive volume exactly.  Kept as the check that
        first caught the opposite: an earlier kernel zeroed the nodes whose
        folded offset lay outside the Wigner-Seitz cell and covered only 85 %
        of a hexagonal cell and 76 % of an FCC one.
        """
        return float(np.count_nonzero(self.real_kernel) * self.dV)

    @property
    def minimum_image_radius(self) -> float:
        """Largest minimum-image distance on the grid, in Bohr.

        The radius of the Wigner-Seitz cell's farthest corner: the range over
        which the truncated kernel acts, and hence the size of the largest
        exchange-correlation hole this cell can hold without its own images.
        """
        return float(np.max(1.0 / self.real_kernel[self.real_kernel > 0]))

    def solve(self, rho_flat: np.ndarray) -> np.ndarray:
        """Truncated-Coulomb potential of one density on the grid."""
        return self.solve_stack(rho_flat[None, :])[0]

    def solve_stack(self, rho_stack: np.ndarray,
                    out: np.ndarray | None = None) -> np.ndarray:
        """Truncated-Coulomb potentials of a stack of ``P`` densities."""
        rho_stack = np.asarray(rho_stack, dtype=float)
        P = rho_stack.shape[0]
        N = int(np.prod(self.shape))
        if out is None:
            out = np.empty((P, N), dtype=float)
        for p in range(P):
            rho_3d = rho_stack[p].reshape(self.shape)
            r_fft = sfft.fftn(rho_3d, workers=self.workers)
            pot_3d = sfft.ifftn(r_fft * self._kernel, workers=self.workers).real * self.dV
            out[p] = pot_3d.ravel()
        return out


@dataclass
class FiniteSizeCorrection:
    r"""What the exchange-correlation hole's own images cost.

    **Scope matters here and every field says which it is.**  The RDMs and the
    two tensors describe the Born-von Karman *supercell*, so that is what the
    contraction produces; the periodic drivers report energies per *primitive
    cell*.  Mixing the two silently is an ``n_cells``-fold error, so the
    supercell quantities keep the plain names and the per-cell ones say so.

    Attributes
    ----------
    hole_energy_periodic : float
        :math:`\tfrac12 \sum g^{\text{E}} (\Gamma - D \otimes D)` in Hartree,
        **per supercell**: the hole's interaction under the periodic kernel,
        images included.
    hole_energy_truncated : float
        The same contraction with the Wigner-Seitz minimum-image bare kernel:
        the hole interacting with itself alone.  Hartree, per supercell.
    correction : float
        ``hole_energy_truncated - hole_energy_periodic``, Hartree per
        supercell.
    correction_per_cell : float
        The same divided by :attr:`n_cells` -- the number to add to a reported
        per-cell energy.
    n_cells : int
        Primitive cells in the supercell.
    n_electrons : float
        ``tr D``, per supercell, carried as a check that the RDMs describe the
        system the tensors do.
    volume : float
        **Supercell** volume in Bohr^3.  This is the volume the ``1/Omega``
        finite-size law refers to, because it is the supercell periodicity that
        creates the images.
    energy_per_cell : float or None
        The uncorrected energy per primitive cell, in Hartree, when supplied.
    mpc_energy_per_cell : float or None
        ``energy_per_cell + correction_per_cell``.
    """

    hole_energy_periodic: float
    hole_energy_truncated: float
    correction: float
    n_cells: int
    n_electrons: float
    volume: float
    energy_per_cell: float | None = None

    @property
    def correction_per_cell(self) -> float:
        """The correction to add to a per-primitive-cell energy, in Hartree."""
        return self.correction / self.n_cells

    @property
    def mpc_energy_per_cell(self) -> float | None:
        """The MPC energy per primitive cell, in Hartree."""
        if self.energy_per_cell is None:
            return None
        return self.energy_per_cell + self.correction_per_cell

    @property
    def per_electron(self) -> float:
        """The correction per electron, in Hartree -- the comparable figure."""
        return self.correction / self.n_electrons if self.n_electrons else 0.0

    def in_units(self, units: str = "eV") -> "FiniteSizeCorrection":
        """The same record with every energy converted out of Hartree."""
        from ..units import from_hartree

        def convert(value):
            return None if value is None else float(from_hartree(value, units))

        return FiniteSizeCorrection(
            hole_energy_periodic=convert(self.hole_energy_periodic),
            hole_energy_truncated=convert(self.hole_energy_truncated),
            correction=convert(self.correction),
            n_cells=self.n_cells,
            n_electrons=self.n_electrons,
            volume=self.volume,
            energy_per_cell=convert(self.energy_per_cell))

    def summary(self) -> str:
        """One line, in eV, for a run log or a console."""
        ev = self.in_units("eV")
        text = (f"XC-hole image interaction: {ev.correction:+.6f} eV per "
                f"supercell = {ev.correction_per_cell:+.6f} eV per cell "
                f"({ev.per_electron:+.6f} eV/electron; supercell volume "
                f"{self.volume:.1f} Bohr^3, {self.n_cells} cells)")
        if ev.mpc_energy_per_cell is not None:
            text += (f"; MPC energy {ev.mpc_energy_per_cell:.6f} eV/cell "
                     f"against {ev.energy_per_cell:.6f}")
        return text


def exchange_correlation_hole_energy(eri_periodic, eri_truncated, D, Gamma,
                                     *, volume, n_cells: int = 1,
                                     energy_per_cell=None):
    r"""Contract the two kernels against the hole.

    Parameters
    ----------
    eri_periodic, eri_truncated : (M, M, M, M) array_like
        Two-electron tensors in physicists' notation over the **same** spatial
        orbitals, built with the periodic and the truncated kernel.
    D : (M, M) array_like
        Spin-summed one-body RDM.
    Gamma : (M, M, M, M) array_like
        Spin-summed two-body RDM, same index convention as the tensors.
    volume : float
        **Supercell** volume in Bohr^3.
    n_cells : int
        Primitive cells in the supercell, so the supercell correction can be
        brought back to the per-cell convention the drivers report in.
    energy_per_cell : float, optional
        The uncorrected energy **per primitive cell**, in Hartree.

    Returns
    -------
    FiniteSizeCorrection

    Notes
    -----
    The hole is :math:`\Gamma - D \otimes D` with
    :math:`(D \otimes D)_{pqrs} = D_{pr} D_{qs}` -- electron 1 holds the pair
    ``(p, r)`` and electron 2 the pair ``(q, s)``, which is the convention
    ``two_body`` returns and the one
    :func:`~mandacaru.algorithms.pseudo_forces.spatial_rdms` produces.  Getting
    that pairing wrong silently substitutes the exchange term for the direct
    one and changes the sign of the answer.
    """
    eri_periodic = np.asarray(eri_periodic)
    eri_truncated = np.asarray(eri_truncated)
    D = np.asarray(D)
    Gamma = np.asarray(Gamma)
    if eri_periodic.shape != eri_truncated.shape:
        raise ValueError(
            f"the two tensors must span the same orbitals; got "
            f"{eri_periodic.shape} and {eri_truncated.shape}")
    if Gamma.shape != eri_periodic.shape:
        raise ValueError(
            f"the two-body RDM {Gamma.shape} does not match the tensors "
            f"{eri_periodic.shape}")

    hole = Gamma - np.einsum("pr,qs->pqrs", D, D)
    periodic = 0.5 * float(np.real(np.einsum("pqrs,pqrs->",
                                             eri_periodic, hole)))
    truncated = 0.5 * float(np.real(np.einsum("pqrs,pqrs->",
                                              eri_truncated, hole)))
    correction = truncated - periodic
    return FiniteSizeCorrection(
        hole_energy_periodic=periodic,
        hole_energy_truncated=truncated,
        correction=correction,
        n_cells=int(n_cells),
        n_electrons=float(np.real(np.trace(D))),
        volume=float(volume),
        energy_per_cell=(None if energy_per_cell is None
                         else float(energy_per_cell)))


def spatial_rdms(gamma, gamma2, n_orbitals: int):
    r"""Spin-summed spatial RDMs from spin-blocked spin-orbital ones.

    Spin-orbital ``P = p + sigma * M`` (alpha block first).  The one-body RDM
    sums the two spin blocks; the two-body RDM sums the four blocks in which
    electron 1 keeps its spin and electron 2 keeps its spin.
    """
    m = int(n_orbitals)
    if gamma.shape != (2 * m, 2 * m):
        raise ValueError(
            f"expected a {2 * m}x{2 * m} one-body RDM "
            f"(frozen orbitals are not supported here), got {gamma.shape}"
        )
    a, b = slice(0, m), slice(m, 2 * m)
    d = gamma[a, a] + gamma[b, b]
    if gamma2 is None:
        return d, None
    g = (
        gamma2[a, a, a, a]
        + gamma2[a, b, a, b]
        + gamma2[b, a, b, a]
        + gamma2[b, b, b, b]
    )
    return d, g


class PeriodicIntegralsAdapter:
    """Adapts native MolecularIntegrals to the interface expected by finite-size corrections."""

    def __init__(self, integrals, reference, geometry, context=None):
        from types import SimpleNamespace
        from ..units import ANGSTROM_TO_BOHR

        self._integrals = integrals
        self._reference = reference
        self._geometry = geometry
        self.context = context

        snap = integrals.snapshot(include_samples=True)
        self._psi = snap["samples"]  # (M, ngrid)
        self._lowdin = snap["lowdin"]  # (M, M)
        self.orthogonalize = True
        self.basis = list(range(snap["samples"].shape[0]))

        ref_snap = reference.snapshot()
        self.mo_coefficients = ref_snap["coefficients"][0]  # (M, M)

        native_grid = integrals.grid

        class GridAdapter:
            def __init__(self, g):
                self.shape = tuple(g.shape)
                self.step = np.asarray(g.step_bohr, dtype=float)
                self.dV = float(g.dv_bohr3)

        self.grid = GridAdapter(native_grid)
        self.cell = np.asarray(geometry.cell, dtype=float) * ANGSTROM_TO_BOHR
        self._engine = SimpleNamespace(_psi=self._psi)
        self._cached_periodic = None

    def _lowdin_x(self):
        return self._lowdin

    def two_body_with_kernel(self, solver=None, *, mo: bool = True) -> np.ndarray:
        if mo and self.mo_coefficients is None:
            raise ValueError(
                "no molecular-orbital rotation is available yet: call "
                "molecular_hamiltonian() first, or pass mo=False to get the "
                "tensor in the Loewdin-orthonormalized basis."
            )
        if solver is None and self._cached_periodic is not None:
            return self._cached_periodic.copy()

        if solver is None:
            solver = PeriodicPoissonSolver(self.grid.shape, step=self.grid.step)

        psi = self._psi
        M, ngrid = psi.shape
        dV = self.grid.dV

        rho_stack = np.empty((M * M, ngrid), dtype=float)
        for b in range(M):
            for d in range(M):
                rho_stack[b * M + d] = np.real(np.conj(psi[b]) * psi[d])

        phi_stack = solver.solve_stack(rho_stack)
        eri_raw = (rho_stack @ phi_stack.T) * dV
        eri_4d = eri_raw.reshape(M, M, M, M)
        eri = np.transpose(eri_4d, (0, 2, 1, 3))

        if self.orthogonalize:
            X = self._lowdin_x()
            eri = np.einsum(
                "ap,bq,cr,ds,abcd->pqrs",
                X.conj(), X.conj(), X, X, eri, optimize=True
            )
        if mo and self.mo_coefficients is not None:
            V = self.mo_coefficients
            eri = np.einsum(
                "ap,bq,cr,ds,abcd->pqrs",
                V.conj(), V.conj(), V, V, eri, optimize=True
            )

        if solver is None or isinstance(solver, PeriodicPoissonSolver):
            self._cached_periodic = eri.copy()

        return eri


class MolecularIntegrals:
    """Mock/wrapper for legacy MolecularIntegrals interface."""

    def __init__(self, nuclei, basis, grid, units="angstrom", softening=1e-12, orthogonalize=True, **kwargs):
        self.nuclei = nuclei
        self.basis = list(basis)
        self.grid = grid
        self.units = units
        self.softening = softening
        self.orthogonalize = orthogonalize
        self.mo_coefficients = None
        self._kwargs = kwargs

    def two_body_with_kernel(self, solver=None, *, mo: bool = True):
        if mo and self.mo_coefficients is None:
            raise ValueError(
                "no molecular-orbital rotation is available yet: call "
                "molecular_hamiltonian() first, or pass mo=False to get the "
                "tensor in the Loewdin-orthonormalized basis."
            )
        raise NotImplementedError("two_body_with_kernel without mo is not implemented for standalone MolecularIntegrals")


def minimal_hao_basis(nuclei, grid_units: str = "angstrom"):
    from ..basis import HydrogenicAtomicOrbital

    basis = []
    for Z, R in nuclei:
        z_eff = HydrogenicAtomicOrbital.slater_effective_charge(int(round(Z)), 1, 0)
        basis.append(
            HydrogenicAtomicOrbital(1, 0, 0, Z=z_eff, center=R, units=grid_units)
        )
    return basis


