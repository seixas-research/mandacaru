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
(the Ewald sum of the Rust engine) has been accounted for, and it falls off only as
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
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .. import _native
from .._native import voxel_self_potential, wigner_seitz_kernel as _native_wigner_seitz_kernel

__all__ = ["TruncatedCoulombSolver", "wigner_seitz_kernel",
           "FiniteSizeCorrection", "exchange_correlation_hole_energy",
           "voxel_self_potential", "PeriodicPoissonSolver", "spatial_rdms",
           "PeriodicIntegralsAdapter"]


class PeriodicPoissonSolver:
    """Fourier-space Poisson solver on a periodic grid."""

    def __init__(self, shape, step=None, spacing=None, workers: int = -1):
        self.shape = tuple(int(s) for s in shape)
        if step is not None:
            self.step = np.ascontiguousarray(step, dtype=float)
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
        self._kernel = _native.periodic_poisson_kernel(self.shape, self.step, self.dV)

    def solve(self, rho_flat: np.ndarray) -> np.ndarray:
        return self.solve_stack(rho_flat[None, :])[0]

    def solve_stack(self, rho_stack: np.ndarray,
                    out: np.ndarray | None = None) -> np.ndarray:
        res = _native.poisson_solve_stack(
            self._kernel, self.shape, np.ascontiguousarray(rho_stack, dtype=float), self.dV
        )
        if out is not None:
            out[:] = res
            return out
        return res


#: Lattice shells searched when deciding which translation minimizes the distance.
MINIMUM_IMAGE_SHELLS = 2

#: Two distances closer than this (in Bohr) count as tied.
TIE_TOLERANCE = 1e-10


def wigner_seitz_kernel(shape, step, shells: int = MINIMUM_IMAGE_SHELLS):
    r"""The bare ``1/r`` truncated at the Wigner-Seitz cell, on the grid."""
    shape = tuple(int(n) for n in shape)
    step = np.ascontiguousarray(np.asarray(step, dtype=float))
    if step.shape != (3, 3):
        raise ValueError(f"step must be a 3x3 matrix, got {step.shape}")
    return np.ascontiguousarray(_native_wigner_seitz_kernel(shape, step, shells=int(shells)))


class TruncatedCoulombSolver:
    """Circular convolution with the Wigner-Seitz-truncated bare ``1/r``."""

    def __init__(self, shape, step, workers: int = -1,
                 shells: int = MINIMUM_IMAGE_SHELLS):
        self.shape = tuple(int(s) for s in shape)
        self.step = np.ascontiguousarray(step, dtype=float)
        if self.step.shape != (3, 3):
            raise ValueError(f"step must be a 3x3 matrix, got {self.step.shape}")
        self.dV = abs(float(np.linalg.det(self.step)))
        if self.dV <= 0.0:
            raise ValueError("the voxel has zero volume (degenerate step)")
        self.cell = self.step @ np.diag(self.shape)
        self.volume = abs(float(np.linalg.det(self.cell)))
        self.L = self.shape
        self.workers = workers
        self.real_kernel = wigner_seitz_kernel(self.shape, self.step, shells)
        self._kernel = _native.truncated_coulomb_kernel(self.real_kernel, self.shape)
        self.even = True

    @property
    def enclosed_volume(self) -> float:
        """Volume the kernel covers, in Bohr^3."""
        vol, _ = _native.wigner_seitz_metrics(self.real_kernel, self.dV)
        return float(vol)

    @property
    def minimum_image_radius(self) -> float:
        """Largest minimum-image distance on the grid, in Bohr."""
        _, rad = _native.wigner_seitz_metrics(self.real_kernel, self.dV)
        return float(rad)

    def solve(self, rho_flat: np.ndarray) -> np.ndarray:
        """Truncated-Coulomb potential of one density on the grid."""
        return self.solve_stack(rho_flat[None, :])[0]

    def solve_stack(self, rho_stack: np.ndarray,
                    out: np.ndarray | None = None) -> np.ndarray:
        """Truncated-Coulomb potentials of a stack of ``P`` densities."""
        res = _native.poisson_solve_stack(
            self._kernel, self.shape, np.ascontiguousarray(rho_stack, dtype=float), self.dV
        )
        if out is not None:
            out[:] = res
            return out
        return res


@dataclass
class FiniteSizeCorrection:
    r"""What the exchange-correlation hole's own images cost."""

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
    r"""Contract the two kernels against the hole."""
    eri_p = np.ascontiguousarray(eri_periodic, dtype=float)
    eri_t = np.ascontiguousarray(eri_truncated, dtype=float)
    d = np.ascontiguousarray(D, dtype=float)
    g = np.ascontiguousarray(Gamma, dtype=float)

    if eri_p.shape != eri_t.shape:
        raise ValueError(
            f"the two tensors must span the same orbitals; got "
            f"{eri_p.shape} and {eri_t.shape}")
    if g.shape != eri_p.shape:
        raise ValueError(
            f"the two-body RDM {g.shape} does not match the tensors "
            f"{eri_p.shape}")

    periodic, truncated, n_electrons = _native.mpc_hole_energies(eri_p, eri_t, d, g)
    correction = truncated - periodic
    return FiniteSizeCorrection(
        hole_energy_periodic=float(periodic),
        hole_energy_truncated=float(truncated),
        correction=float(correction),
        n_cells=int(n_cells),
        n_electrons=float(n_electrons),
        volume=float(volume),
        energy_per_cell=(None if energy_per_cell is None
                         else float(energy_per_cell)))


def spatial_rdms(gamma, gamma2, n_orbitals: int):
    r"""Spin-summed spatial RDMs from spin-blocked spin-orbital ones."""
    g1 = np.ascontiguousarray(gamma.real, dtype=float)
    g2 = np.ascontiguousarray(gamma2.real, dtype=float) if gamma2 is not None else None
    return _native.spatial_rdms(g1, g2, int(n_orbitals))


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

        psi = np.ascontiguousarray(self._psi, dtype=float)
        M = psi.shape[0]

        rho_stack = _native.compute_pair_densities(psi)
        phi_stack = np.ascontiguousarray(solver.solve_stack(rho_stack), dtype=float)

        X = np.ascontiguousarray(self._lowdin_x().real, dtype=float) if self.orthogonalize else None
        V = np.ascontiguousarray(self.mo_coefficients.real, dtype=float) if (mo and self.mo_coefficients is not None) else None

        context = self.context if self.context is not None else _native.Context(threads=1)
        eri = _native.pair_two_body(context, np.ascontiguousarray(rho_stack, dtype=float),
                                    phi_stack, self.grid.dV, X, V, M)

        if solver is None or isinstance(solver, PeriodicPoissonSolver):
            self._cached_periodic = eri.copy()

        return eri
