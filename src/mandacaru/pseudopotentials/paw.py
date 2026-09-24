# -*- coding: utf-8 -*-
# file: pseudopotentials/paw.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Projector augmented-wave datasets (PAW-LCAO).

The family ``"paw-lcao"`` implements P. E. Blöchl's projector augmented-wave method,
Phys. Rev. B **50**, 17953 (1994), in its frozen-core, one-center-expansion
form, with the one-center energies **linearized around the reference atom**
(a fixed coupling matrix :math:`D^0` per species, the "frozen augmentation"
that makes a PAW-LCAO dataset behave like an ultrasoft pseudopotential).  Written
from scratch on the same LDA radial atom as the Troullier-Martins and ONCVPSP
families (:mod:`mandacaru.basis.atomic_solver`), reusing the Numerov partial
waves, the spherical-Bessel machinery and the polynomial local potential of
:mod:`.oncv`.  Nothing is read from tables.

The transformation
------------------
Blöchl's linear map between the smooth (pseudo) wave functions the grid
sees and the all-electron ones,

.. math::

    |\psi\rangle = \mathcal T|\tilde\psi\rangle, \qquad
    \mathcal T = 1 + \sum_i \big(|\varphi_i\rangle - |\tilde\varphi_i\rangle\big)
    \langle\tilde p_i| ,

needs, per atom and angular momentum, **all-electron partial waves**
:math:`\varphi_i`, **smooth partial waves** :math:`\tilde\varphi_i` equal to
them beyond the augmentation radius :math:`r_c`, and **projectors**
:math:`\tilde p_i` localized inside :math:`r_c` and dual to the smooth waves,
:math:`\langle\tilde p_i|\tilde\varphi_j\rangle = \delta_{ij}`.  Because the
smooth waves are *not* norm-conserving, the transformation carries an overlap
operator :math:`S = 1 + \sum_{ij}|\tilde p_i\rangle q_{ij}\langle\tilde p_j|`
with :math:`q_{ij} = \langle\varphi_i|\varphi_j\rangle_{r<r_c} -
\langle\tilde\varphi_i|\tilde\varphi_j\rangle_{r<r_c}`, and the valence
problem is the generalized eigenproblem

.. math::

    \Big[T + \tilde v_{loc} + \sum_{ij}|\tilde p_i\rangle D_{ij}\langle\tilde p_j|\Big]
    \tilde\psi = \varepsilon\, S\,\tilde\psi .

Construction (per species, :func:`generate_paw`)
------------------------------------------------
1. **Reference atom.**  Self-consistent spherical LDA atom; the frozen core
   density :math:`n_c` (all subshells below the valence) and, for every
   valence :math:`l`, two all-electron partial waves: the bound valence state
   (Numerov, :func:`~.oncv.bound_state`) and the scattering state at
   :math:`\varepsilon_1 + \Delta` (:data:`DEFAULT_ENERGY_OFFSET` = 1 Ha,
   normalized to one inside :math:`r_c`) -- the same pair ONCVPSP uses.
2. **Smooth partial waves.**  Inside :math:`r_c`,
   :math:`\tilde\varphi_i = \sum_{n=1}^{8} c_{in} j_l(q_n r)` at the
   interleaved zeros of :math:`j_l` and :math:`j_l'`
   (:func:`~.oncv.bessel_wavevectors`), matched in value and first three
   derivatives to the all-electron wave (:func:`~.oncv.matching_targets`) and,
   in the four remaining degrees of freedom, with the residual kinetic energy
   beyond :math:`q_c` = 5 Bohr⁻¹ minimized (:func:`smooth_partial_waves`).
   **No norm condition is imposed** -- that is the point of PAW-LCAO; the norm
   deficit becomes :math:`q_{ij}`.
3. **Local potential.**  The even-polynomial continuation of the screened
   all-electron potential inside :math:`r_{cl}` (:func:`~.oncv.polynomial_local_potential`,
   :math:`r_{cl}` = :data:`DEFAULT_LOCAL_FACTOR` × the smallest :math:`r_c`),
   the "screened" :math:`\tilde v^{scr}`.  Any smooth continuation would do;
   this one needs no zero potential :math:`\bar v`.
4. **Projectors.**  :math:`\chi_i = (\varepsilon_i - T - \tilde v^{scr})
   \tilde\varphi_i` inside :math:`r_c` (analytic through
   :math:`T j_l(qr) = \tfrac12 q^2 j_l(qr)`), :math:`B_{ij} =
   \langle\tilde\varphi_i|\chi_j\rangle`, and the dual set
   :math:`\tilde p_i = \sum_k (B^{-1})_{ki}\chi_k` -- the Blöchl/Vanderbilt
   construction; the generator asserts
   :math:`\langle\tilde p_i|\tilde\varphi_j\rangle = \delta_{ij}` to
   :data:`DUALITY_TOLERANCE`.
5. **One-center matrices.**  On the fine inner grid: :math:`q_{ij}`; the
   kinetic difference :math:`\Delta T_{ij} = \langle\varphi_i|T|\varphi_j\rangle
   - \langle\tilde\varphi_i|T|\tilde\varphi_j\rangle` (all-electron side from
   the radial equation, smooth side analytic); the potential difference
   :math:`\Delta V^{scr}_{ij} = \langle\varphi_i|v^{AE}|\varphi_j\rangle -
   \langle\tilde\varphi_i|\tilde v^{scr}|\tilde\varphi_j\rangle`.  The screened
   coupling is :math:`D^{scr}_{ij} = B_{ij} + \varepsilon_j q_{ij}`, which
   equals :math:`\Delta T + \Delta V^{scr}` (checked, ``consistency_error``)
   and is symmetric by the generalized Wronskian identity
   :math:`B_{ij} - B_{ji} = (\varepsilon_i - \varepsilon_j) q_{ij}`
   (``asymmetry``); it reproduces every reference energy exactly in the
   generalized eigenproblem.
6. **Compensation charge and unscreening.**  The smooth reference valence
   density :math:`\tilde n_v` misses :math:`Q = \sum_l f_l\, q^l_{11}`
   electrons; the reference atom is spherical, so its compensation charge
   is the **monopole** :math:`\hat n = Q\,g(r)`
   with the shape :math:`g \propto (1 - r^2/r_g^2)^3` inside
   :math:`r_g = \min_l r_c` (:func:`compensation_shape`) restores neutrality
   with the ion outside the sphere.  Unscreening follows the norm-conserving
   families: :math:`\tilde v^{ion} = \tilde v^{scr} - v_H[\tilde n_v + \hat n]
   - v_{xc}[\tilde n_v]`, and the coupling loses the Hartree screening of the
   augmentation, :math:`D^{ion}_{ij} = D^{scr}_{ij} - q_{ij}\int v_H[\tilde n_v
   + \hat n]\,g`.
7. **Frozen one-center constant.**  With :math:`D` linearized, the double
   counting of the one-center Hartree and xc energies at the reference is a
   per-species constant (:attr:`PAWDataset.one_center_energy`), fixed so
   that the LDA reference atom evaluated with the molecular machinery
   (smooth kinetic + local + nonlocal, Hartree of :math:`\tilde n_v + \hat n`)
   has exactly the all-electron valence energy in the norm-conserving
   convention, :math:`E^{ref}_{val} = \sum_v f_v\varepsilon_v - E_H[n_v] -
   \int n_v v_{xc}[n_v] + E_{xc}[n_v]`.  It is added to every molecular
   Hamiltonian as :attr:`mandacaru.core.hamiltonian.MolecularIntegrals.constant_energy`.

In a molecule (:func:`build_paw`, :class:`PAWIntegrals`)
-------------------------------------------------------
The basis is the bound smooth partial waves (with the usual size hierarchy),
the external potential :math:`\sum_A \tilde v^{ion}_A`, the nonlocal term
:math:`C D^{ion} C^\dagger` and the overlap :math:`\tilde S + C q C^\dagger`
through the general separable form of
:class:`~mandacaru.core.hamiltonian.MolecularIntegrals`.  The two-body tensor is
built from **augmented pair densities**
:math:`\rho_{pr} = \tilde\phi_p^*\tilde\phi_r + \sum_A (C q C^\dagger)^A_{pr}\, g_A`
(:meth:`PAWIntegrals.two_body_augmentation`), so Hartree, exchange and
correlation all see neutral atoms; the compensation-compensation Coulomb
integrals are evaluated radially (exactly :math:`1/R_{AB}` for disjoint
spheres).

What is frozen or omitted relative to Blöchl's full method (see the guide):
the one-center Hartree and xc terms are linearized at the reference (fixed
:math:`D^0`, no self-consistent :math:`D_{ij}[\rho_{ij}]`); the
compensation multipoles stop at :math:`L \le 2 l_{max}` of the valence
channels (:mod:`.multipoles`); the core
is frozen; the one-center terms are those of the functional the reference
atom was solved with (``xc``); relativity enters through the scalar-relativistic
(or Dirac, with a spin-orbit term) reference atom and partial waves, not through
the molecular Hamiltonian; no projectors above the valence :math:`l`.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import simpson

from ..basis.xc import xc_potential
from ..basis.atomic_solver import (AtomicResult, hartree_potential,
                                    solve_atom)
from ..core.hamiltonian import MolecularIntegrals, projector_blocks
from .confinement import DEFAULT_ENERGY_SHIFT
from .generation import Channel, PseudoPotential, _valence_configuration
from .oncv import (Q_MAX, Q_STEP, PseudoWaves, _bessel_table,
                   _bessel_transform_table, _inner_grid, _log_derivative_of_u,
                   _pseudo_waves_record, _radial_f, _resample, _snap,
                   _spectrum_extent, _tail_transform, _with_origin,
                   bessel_derivatives, bessel_wavevectors, generation_points,
                   DEFAULT_EXTRA_L, DEFAULT_NLCC, DEFAULT_RELATIVITY,
                   DEFAULT_XC, log_derivative_errors, matching_targets,
                   ghost_free, numerov_outward, optimize_pseudo_waves,
                   polynomial_local_potential, reference_waves)

#: Registry name of the family (no aliases).  The **-LCAO** is not decoration:
#: this is Bloechl's projector-augmented-wave transformation carried on a
#: *localized* basis -- the bound smooth partial waves and their multiple-zeta
#: hierarchy -- rather than on plane waves, which is what the name of a basis
#: set has to say.  The method it implements is still PAW-LCAO.
FAMILY = "paw-lcao"
#: Subdirectory of the library holding the PAW-LCAO datasets.
LIBRARY_SUBDIR = "paw-lcao"

#: Registry name of the **unitary** variant (``basis="UPAW-LCAO"``).
UPAW_FAMILY = "upaw-lcao"
#: Subdirectory holding UPAW-LCAO datasets, when one has been built.
UPAW_LIBRARY_SUBDIR = "upaw-lcao"

#: Spherical Bessel functions per smooth partial wave.
DEFAULT_N_BESSEL = 8
#: Wave-vector cutoff of the residual kinetic energy (Bohr^-1).
DEFAULT_Q_CUT = 5.0
#: Second reference energy above the bound state (Hartree).
DEFAULT_ENERGY_OFFSET = 1.0
#: Fraction of the all-electron inner norm the smooth partial waves give up:
#: ``<phi~_i|phi~_j>_rc = (1 - DEFAULT_NORM_DEFICIT) <phi_i|phi_j>_rc``, so the
#: overlap correction ``q = deficit * <phi_i|phi_j>_rc`` is positive definite
#: and the PAW-LCAO overlap operator ``1 + sum |p> q <p|`` is bounded below by 1.
#: ``None`` drops the norm conditions altogether (free minimization of the
#: residual kinetic energy), which leaves the sign of ``q`` uncontrolled.
#: The value an element without an entry in :data:`DEFAULT_NORM_DEFICITS`
#: gets; the derivative matching alone fixes most of the inner norm, so a
#: deficit above ~0.2 (0.1 for Li) is not reachable by the Bessel expansion.
DEFAULT_NORM_DEFICIT = 0.1
#: Per-element norm deficits (H, Li kept small: a larger deficit grows a
#: ghost state in their s channel).
DEFAULT_NORM_DEFICITS = {"H": 0.05, "Li": 0.02, "C": 0.10, "N": 0.15,
                         "O": 0.15, "F": 0.15}
#: Per-element second reference energy above the bound state (Hartree);
#: elements without an entry use :data:`DEFAULT_ENERGY_OFFSET`.  Lithium's
#: scattering wave at +1 Ha sits at a pole of the logarithmic derivative
#: (L = +24 at r_c) and the smooth pair then grows nodes and a ghost; for
#: hydrogen +0.5 Ha gives a softer second projector (grid norm ratio 0.78
#: instead of 0.75 at 0.25 Angstrom) and better logarithmic derivatives.
DEFAULT_ENERGY_OFFSETS = {"H": 0.5, "Li": 0.5}
#: Default augmentation radius as a multiple of the outermost maximum of rR.
DEFAULT_RC_FACTOR = 1.3
#: Local-potential radius as a multiple of the smallest augmentation radius.
DEFAULT_LOCAL_FACTOR = 0.9
#: Per-element augmentation radii (Bohr); the ONCVPSP values, which the
#: molecular tests showed to be resolvable on a 0.25-0.30 Angstrom grid.
DEFAULT_CUTOFFS = {
    "H": {0: 1.30},
    "Li": {0: 2.60},
    "C": {0: 1.50, 1: 1.50},
    "N": {0: 1.45, 1: 1.45},
    "O": {0: 1.45, 1: 1.45},
    "F": {0: 1.40, 1: 1.40},
}
#: Per-element raise of the local potential at the origin (Hartree, Hamann's
#: ``dvloc0``).  The polynomial continuation of the screened all-electron
#: potential is deep enough (O: -5.8 Ha at the origin) to bind a spurious
#: 1s-like state of its own in the s channel of the first row; unlike the
#: near-singular ONCVPSP coupling, the PAW-LCAO projector term does not push it
#: away, so the local potential is raised until the s spectrum has nothing
#: between the bound state and the box states.
DEFAULT_LOCAL_SHIFTS = {"C": 12.0, "N": 10.0, "O": 6.0, "F": 8.0}
#: Largest tolerated deviation of ``<p_i|phi_j>`` from the identity.
DUALITY_TOLERANCE = 1e-8
#: Largest tolerated asymmetry of the screened coupling matrix (Hartree).
COUPLING_ASYMMETRY_TOLERANCE = 1e-4
#: A reference energy below this cannot be divided out when recovering the
#: norm correction of a dataset written before that correction was stored.
NORM_RECOVERY_FLOOR = 1e-6
#: Below this, a dataset's stored overlap correction and its recovered norm
#: correction are the *same array* rather than two O(c^-2)-separated matrices,
#: which is how a file written before the two were distinguished is detected.
#: It sits between the two scales involved and is not delicate: the recovery
#: carries about 1e-9 of symmetrization noise, while a relativistic dataset
#: that really does keep the two apart separates them by about 2e-4.
NORM_SPLIT_FLOOR = 1e-6
#: Smallest tolerated eigenvalue of the overlap operator ``1 + sum |p> q <p|``
#: restricted to the channel (``1 + lambda_min(q G_p)``, ``G_p`` the projector
#: Gram matrix): below it the PAW transformation is (nearly) singular.
OVERLAP_MINIMUM = 0.1
#: Radial points of the compensation charge's atom-centered quadrature.
COMPENSATION_RADIAL_POINTS = 48
#: Points of the radial quadrature grids used for the compensation charge.
COMPENSATION_POINTS = 2001
#: Projector basis sampled on the molecular grid (:meth:`PAWDataset.projector_set`).
DEFAULT_PROJECTOR_BASIS = "raw"
#: Spherical product quadrature of the atom-centered projections: Gauss-Legendre
#: in radius and cos(theta), uniform in phi.
PROJECTION_RADIAL_POINTS = 64
PROJECTION_POLAR_POINTS = 24
PROJECTION_AZIMUTHAL_POINTS = 48


# --------------------------------------------------------------------------- #
# Smooth partial waves: matched, no norm condition, minimal residual energy.
# --------------------------------------------------------------------------- #

def smooth_partial_waves(r: np.ndarray, v_ae: np.ndarray, l: int, waves: list,
                         energies: list, r_cut: float,
                         q_cut: float = DEFAULT_Q_CUT,
                         n_bessel: int = DEFAULT_N_BESSEL,
                         norm_deficit=DEFAULT_NORM_DEFICIT,
                         treatment: str = "none", kappa: int | None = None,
                         z_eff: float = 0.0) -> PseudoWaves:
    r"""Bessel expansions of the smooth partial waves of one channel.

    Each wave is matched in value and first three derivatives at ``r_cut``
    and its residual kinetic energy beyond ``q_cut`` is minimized in the
    remaining freedom.  **The norm is not conserved.**  With a
    ``norm_deficit`` :math:`s` (the default, :data:`DEFAULT_NORM_DEFICIT`)
    the inner-norm matrix is set to :math:`(1-s)` times the all-electron one
    (:func:`~.oncv.optimize_pseudo_waves` with ``norm_factor = 1 - s``), so
    the overlap correction :math:`q_{ij} = s\,\langle\varphi_i|\varphi_j
    \rangle_{r<r_c}` is positive definite and the PAW-LCAO overlap operator is
    bounded below by one -- a dataset built with free (unconstrained) waves,
    ``norm_deficit=None``, has no such guarantee: the two reference waves
    are nearly proportional in the core, their dual projectors are large,
    and an indefinite :math:`q` makes :math:`1 + \sum|\tilde p\rangle q
    \langle\tilde p|` singular (Li, O at every cutoff tried).  The returned
    :attr:`~.oncv.PseudoWaves.achieved` inner norms differ from the
    all-electron :attr:`~.oncv.PseudoWaves.norms` by exactly :math:`q`.

    ``norm_deficit=0`` is the **unitary PAW-LCAO** (UPAW-LCAO) of Ivanov *et al.*
    (arXiv:2408.03159): :math:`q \equiv 0` makes :math:`T^\dagger T = I`, so the
    pseudo states are orthonormal and the overlap operator is the identity.  It
    is a working option -- datasets for H, Li, C and O build with
    :math:`|q| \le 3\times10^{-14}`, an overlap minimum of exactly 1, no ghost
    states and the same eigenvalue reproduction -- but it is **not** the default,
    for three measured reasons.  (1) Mandacaru does not need it: the augmented
    overlap is Löwdin-orthogonalized (:func:`~mandacaru.core.hamiltonian._lowdin_x`)
    before the many-body Hamiltonian is built, so the second-quantized problem is
    already in an orthonormal basis -- the non-orthogonality UPAW-LCAO exists to cure
    is a plane-wave-basis problem.  (2) It does not remove the augmentation:
    the constraint fixes only the **norm**, i.e. the :math:`L = 0` moment, so the
    higher compensation multipoles survive and on oxygen :math:`L = 2` *grows*
    (water, h = 0.25: :math:`|Q^{L=0}|` 3.2e-2 -> 8e-15 but :math:`|Q^{L=2}|`
    1.4e-3 -> 2.2e-2).  (3) The waves are harder, which costs exactly what a
    real-space grid is most sensitive to: water's net force (the egg-box) is
    **5x larger** at h = 0.25 and 0.20 (0.38 -> 1.92 and 0.042 -> 0.223 eV/A),
    and the energy converges ~19 % more slowly in h -- the same ordering the
    paper reports (PAW-LCAO converged at 400 eV, UPAW-LCAO at 600).  The physics agrees
    where it should: LiH's bond length differs by 0.002 A and its binding energy
    by 0.098 eV.
    """
    if norm_deficit is not None:
        # The deficit scales whatever the correct norm target is, so a
        # relativistic reference atom needs nothing extra here: the Wronskian
        # form is already what `optimize_pseudo_waves` conserves
        # (:func:`~.oncv.norm_targets`), and `norm_factor` multiplies it.
        return optimize_pseudo_waves(r, v_ae, l, waves, energies, r_cut,
                                     q_cut=q_cut, n_bessel=n_bessel,
                                     norm_factor=1.0 - float(norm_deficit),
                                     treatment=treatment, kappa=kappa,
                                     z_eff=z_eff)
    r_cut = _snap(r, r_cut)
    r_in = _inner_grid(r_cut)
    q_grid = np.arange(0.0, Q_MAX + 0.5 * Q_STEP, Q_STEP)
    weight = 0.5 * q_grid ** 4 * (q_grid >= q_cut)

    waves_in = [_resample(r, w, r_in) for w in waves]
    norms = np.array([[simpson(a * b * r_in * r_in, x=r_in) for b in waves_in]
                      for a in waves_in])

    qs = bessel_wavevectors(l, r_cut, n_bessel)
    j, dj, d2j, d3j = bessel_derivatives(l, qs * r_cut)
    A = np.array([j, qs * dj, qs ** 2 * d2j, qs ** 3 * d3j])
    basis_in = _bessel_table(l, qs, r_in)
    G = simpson(basis_in[:, None, :] * basis_in[None, :, :]
                * (r_in * r_in)[None, None, :], x=r_in, axis=-1)
    transform = _bessel_transform_table(l, qs, r_in, q_grid)          # (N, Q)
    K = (transform * weight) @ transform.T * Q_STEP
    # Null space of the matching rows: c = c0 + Z y.
    _u, sigma, vt = np.linalg.svd(A)
    rank = int(np.sum(sigma > 1e-12 * sigma.max()))
    Z = vt[rank:].T
    Kz = Z.T @ K @ Z + 1e-10 * np.trace(K) / K.shape[0] * (Z.T @ G @ Z)

    coefficients, wavevectors, pseudo_in, residuals = [], [], [], []
    for wave, energy in zip(waves, energies):
        target = matching_targets(r, wave * r, v_ae, l, energy, r_cut,
                                  treatment, kappa, z_eff)
        c0, *_ = np.linalg.lstsq(A, target, rcond=None)
        bound = energy < 0 and abs(wave[-1] * r[-1]) < 1e-6
        tail = _tail_transform(l, r, wave, r_cut, bound, q_grid)
        kvec = (transform * weight) @ tail * Q_STEP
        k0 = float(np.sum(weight * tail * tail) * Q_STEP)
        y = np.linalg.solve(Kz, -Z.T @ (K @ c0 + kvec))
        c = c0 + Z @ y
        coefficients.append(c)
        wavevectors.append(qs)
        pseudo_in.append(c @ basis_in)
        residuals.append(float(c @ K @ c + 2.0 * kvec @ c + k0))

    return _pseudo_waves_record(l, r_cut, energies, waves, wavevectors,
                                coefficients, residuals, norms, pseudo_in,
                                r_in, q_cut)


# --------------------------------------------------------------------------- #
# One channel: projectors, overlap correction, one-center matrices.
# --------------------------------------------------------------------------- #

@dataclass
class PAWChannel(Channel):
    """One PAW-LCAO channel: the TM :class:`Channel` plus the partial-wave sets.

    ``pseudo_radial``/``eigenvalue``/``coefficients`` describe the first
    (bound) smooth partial wave, which is also the first-zeta basis function.
    All matrices are ``(n, n)`` over the radial partial waves of the channel.
    """

    reference_energies: list = field(default_factory=list)   # Hartree
    wavevectors: list = field(default_factory=list)          # per wave: q_n
    wave_coefficients: list = field(default_factory=list)    # per wave: c_n
    ae_waves: list = field(default_factory=list)             # per wave: phi(r)
    pseudo_waves: list = field(default_factory=list)         # per wave: phi~(r)
    projectors: list = field(default_factory=list)           # per wave: p~(r), dual
    raw_projectors: list = field(default_factory=list)       # per wave: chi(r)
    overlap_correction: np.ndarray = None                    # q_ij (charge)
    norm_correction: np.ndarray = None                       # q^norm_ij (energy)
    kinetic_difference: np.ndarray = None                    # Delta T_ij
    potential_difference: np.ndarray = None                  # Delta V^scr_ij
    coupling_screened: np.ndarray = None                     # D^scr_ij
    coupling: np.ndarray = None                              # D^ion_ij
    vanderbilt: np.ndarray = None                            # B_ij
    duality_error: float = 0.0                               # max |<p|phi~> - 1|
    overlap_minimum: float = 1.0                             # min eig of S in channel
    asymmetry: float = 0.0                                   # max |D^scr - D^scr^T|
    consistency_error: float = 0.0                           # |D^scr - dT - dV|
    residual_kinetic: list = field(default_factory=list)     # Hartree per wave
    q_cut: float = DEFAULT_Q_CUT

    def __repr__(self) -> str:
        energies = ", ".join(f"{e:+.4f}" for e in self.reference_energies)
        q = (np.diag(self.overlap_correction).tolist()
             if self.overlap_correction is not None else [])
        return (f"PAWChannel(l={self.l}, n={self.n}, eps=[{energies}] Ha, "
                f"rc={self.r_cut:.3f} Bohr, {len(self.projectors)} projectors, "
                f"q_ii={np.round(q, 4).tolist()})")


def assemble_paw_channel(r: np.ndarray, pw: PseudoWaves, v_ae: np.ndarray,
                         v_loc: np.ndarray, n: int, occupation: float = 0.0,
                         strict: bool = True) -> PAWChannel:
    r"""Projectors and one-center matrices of a channel for a given
    screened ``v_loc``.

    :math:`\chi_i = \sum_n c_{in}(\varepsilon_i - q_n^2/2 - \tilde v^{scr})
    j_l(q_n r)` inside ``r_cut`` (zero beyond), :math:`B_{ij} =
    \langle\tilde\varphi_i|\chi_j\rangle`, projectors
    :math:`\tilde p_i = \sum_k (B^{-1})_{ki}\chi_k` (dual to the smooth
    waves), :math:`q_{ij}`, :math:`\Delta T_{ij}`, :math:`\Delta V^{scr}_{ij}`
    and :math:`D^{scr}_{ij} = B_{ij} + \varepsilon_j q_{ij}` symmetrized.
    With ``strict`` a duality error above :data:`DUALITY_TOLERANCE` or an
    asymmetry above :data:`COUPLING_ASYMMETRY_TOLERANCE` raises.
    """
    l, r_cut = pw.l, pw.r_cut
    r_in = _inner_grid(r_cut)
    inside = r <= r_cut
    w = r_in * r_in
    v_loc_in = _resample(r, v_loc, r_in)
    v_ae_in = _resample(r, v_ae, r_in)
    energies = np.asarray(pw.energies, dtype=float)

    ae_in = [_resample(r, wave, r_in) for wave in pw.waves]
    pseudo_in, chi_in, kin_in = [], [], []
    for c, qs in zip(pw.coefficients, pw.wavevectors):
        basis_in = _bessel_table(l, qs, r_in)
        pseudo_in.append(c @ basis_in)
        kin_in.append((0.5 * qs ** 2 * c) @ basis_in)          # T phi~
    for c, qs, energy in zip(pw.coefficients, pw.wavevectors, energies):
        basis_in = _bessel_table(l, qs, r_in)
        chi_in.append(((energy - 0.5 * qs ** 2)[:, None] * basis_in
                       - v_loc_in[None, :] * basis_in).T @ c)

    def inner(a, b):
        return float(simpson(a * b * w, x=r_in))

    n_waves = len(pseudo_in)
    B = np.array([[inner(p, x) for x in chi_in] for p in pseudo_in])
    B_inv = np.linalg.inv(B)
    projectors_in = [sum(B_inv[k, i] * chi_in[k] for k in range(n_waves))
                     for i in range(n_waves)]
    duality = np.array([[inner(p, f) for f in pseudo_in]
                        for p in projectors_in])
    duality_error = float(np.max(np.abs(duality - np.eye(n_waves))))
    if strict and duality_error > DUALITY_TOLERANCE:
        raise RuntimeError(
            f"PAW-LCAO projectors of l={l} are not dual to the smooth partial "
            f"waves (error {duality_error:.2e}); B is ill-conditioned")

    # `q` is the charge the smooth density is missing, and it has exactly one
    # job to do in three places: the overlap operator
    # `S = 1 + sum |p~> q <p~|`, whose expectation is the electron count; the
    # compensation multipoles, whose L = 0 moment *is* that charge; and the
    # augmented density.  All three are statements about how many electrons
    # there are, so all three take the plain inner product.
    q = pw.norms - pw.achieved
    # The *coupling* is a different question -- it is an energy -- and it
    # takes the norm the Vanderbilt condition actually conserved.  The two
    # matrices are identical non-relativistically and differ at O(c^-2) once
    # the all-electron waves are relativistic; using `q` here instead leaves
    # D^scr asymmetric by 1.3e-4 Ha, and using `q_norm` in S instead leaves
    # the compensated density 4.8e-4 electrons short.
    targets = pw.norms if pw.targets is None else pw.targets
    q_norm = targets - pw.achieved
    # `q_norm` has to be *stored*, not just used here.  D^scr is defined as
    # `B + q_norm eps`, so anything that later runs the definition backwards --
    # the energy-dependent coupling `D(E) = D^scr - E q` of the log-derivative
    # diagnostic -- has to divide by the same matrix it was multiplied by.
    # Reconstructing with the charge `q` instead returns `B + eps (q_norm - q)`
    # rather than `B` at the reference energy: zero non-relativistically, 2.3e-4
    # under the scalar-relativistic default, and amplified about a hundredfold
    # where the reference energy sits near a node of the wave.
    # The overlap operator 1 + sum |p_i> q_ij <p_j| restricted to the channel
    # has the eigenvalues 1 + eig(q G_p); it must stay positive definite.
    gram = np.array([[inner(a, b) for b in projectors_in] for a in projectors_in])
    overlap_minimum = float(1.0 + np.linalg.eigvals(q @ gram).real.min())
    if strict and overlap_minimum < OVERLAP_MINIMUM:
        raise RuntimeError(
            f"the PAW-LCAO overlap operator of l={l} has an eigenvalue "
            f"{overlap_minimum:.3f} (the smooth partial waves carry too little "
            "or too much norm for their projectors); change r_cut, n_bessel "
            "or energy_offset")
    # Kinetic energies inside r_c.  The all-electron side is *defined* by the
    # radial equation rather than differentiated: `T phi_j = (eps_j - v_AE)
    # phi_j`.  That is what makes D^scr = dT + dV an identity rather than a
    # measurement, and it holds for a relativistic partial wave too, because
    # the relativistic kinetic operator is exactly what is left when V is
    # moved to the other side.  The norm paired with `eps_j` here is
    # `targets` -- the one the radial equation puts on its right-hand side --
    # which is why `D^scr` below is built from `q_norm` and not from the
    # charge `q`.
    # <phi_i|T|phi_j> = eps_j <phi_i|phi_j> - <phi_i|V|phi_j>, with the *same*
    # norm the condition conserves.  That is not a bookkeeping convenience: it
    # is the relativistic kinetic matrix element, because `targets` is the
    # M-weighted norm the relativistic radial equation puts on the right-hand
    # side.  With M = 1 it is the old arithmetic exactly.
    V_ae = np.array([[inner(ae_in[i], v_ae_in * ae_in[j])
                      for j in range(n_waves)] for i in range(n_waves)])
    T_ae = energies[None, :] * targets - V_ae
    T_ps = np.array([[inner(pseudo_in[i], kin_in[j]) for j in range(n_waves)]
                     for i in range(n_waves)])
    dT = T_ae - T_ps
    dT = 0.5 * (dT + dT.T)
    dV = V_ae - np.array([[inner(pseudo_in[i], v_loc_in * pseudo_in[j])
                           for j in range(n_waves)] for i in range(n_waves)])
    dV = 0.5 * (dV + dV.T)
    D_raw = B + q_norm * energies[None, :]
    asymmetry = float(np.max(np.abs(D_raw - D_raw.T)))
    if strict and asymmetry > COUPLING_ASYMMETRY_TOLERANCE:
        raise RuntimeError(
            f"the screened PAW-LCAO coupling of l={l} is asymmetric by "
            f"{asymmetry:.2e} Ha (the partial waves do not satisfy the "
            "radial equation or the matching failed)")
    D_scr = 0.5 * (D_raw + D_raw.T)
    consistency = float(np.max(np.abs(D_scr - (dT + dV))))

    # Full-grid tables.
    pseudo_waves, projectors = [], []
    for c, qs, wave in zip(pw.coefficients, pw.wavevectors, pw.waves):
        basis = _bessel_table(l, qs, r)
        pseudo_waves.append(np.where(inside, c @ basis, wave))
    chi_full = []
    for c, qs, energy in zip(pw.coefficients, pw.wavevectors, energies):
        basis = _bessel_table(l, qs, r)
        chi = ((energy - 0.5 * qs ** 2)[:, None] * basis
               - v_loc[None, :] * basis).T @ c
        chi_full.append(np.where(inside, chi, 0.0))
    for i in range(n_waves):
        projectors.append(sum(B_inv[k, i] * chi_full[k] for k in range(n_waves)))

    return PAWChannel(
        l=l, n=n, eigenvalue=float(energies[0]), r_cut=float(r_cut),
        coefficients=pw.coefficients[0], pseudo_radial=pseudo_waves[0],
        v_screened=v_loc, v_ionic=None, occupation=float(occupation),
        norm_error=float(abs(q[0, 0])),
        reference_energies=[float(e) for e in energies],
        wavevectors=list(pw.wavevectors), wave_coefficients=list(pw.coefficients),
        ae_waves=list(pw.waves), pseudo_waves=pseudo_waves,
        projectors=projectors, raw_projectors=chi_full,
        overlap_correction=q, norm_correction=q_norm, kinetic_difference=dT,
        potential_difference=dV, coupling_screened=D_scr, coupling=None,
        vanderbilt=B, duality_error=duality_error,
        overlap_minimum=overlap_minimum, asymmetry=asymmetry,
        consistency_error=consistency,
        residual_kinetic=list(pw.residual_kinetic), q_cut=pw.q_cut)


# --------------------------------------------------------------------------- #
# Compensation charge (monopole) and smooth core density.
# --------------------------------------------------------------------------- #

def compensation_shape(radius, r_g: float) -> np.ndarray:
    r"""Normalized shape :math:`g(r) = \frac{315}{64\pi r_g^3}(1 - r^2/r_g^2)^3`
    inside ``r_g``, zero beyond (``int g d^3r = 1``; :math:`C^2` at ``r_g``)."""
    radius = np.asarray(radius, dtype=float)
    x = radius / float(r_g)
    return np.where(x < 1.0, 315.0 / (64.0 * np.pi * r_g ** 3)
                    * np.clip(1.0 - x * x, 0.0, None) ** 3, 0.0)


def compensation_potential(radius, r_g: float) -> np.ndarray:
    r"""Coulomb potential of :func:`compensation_shape` (unit charge):
    :math:`[Q(x)/x + \tfrac{315}{128}(1-x^2)^4]/r_g` inside with the enclosed
    charge :math:`Q(x) = \tfrac{315}{16}(x^3/3 - 3x^5/5 + 3x^7/7 - x^9/9)`,
    :math:`1/r` outside -- finite at the origin (:math:`2.46/r_g`)."""
    radius = np.asarray(radius, dtype=float)
    x = np.clip(radius / float(r_g), 1e-12, None)
    enclosed = 315.0 / 16.0 * (x ** 3 / 3.0 - 3.0 * x ** 5 / 5.0
                               + 3.0 * x ** 7 / 7.0 - x ** 9 / 9.0)
    inner = (enclosed / x + 315.0 / 128.0 * np.clip(1.0 - x * x, 0.0, None) ** 4) \
        / float(r_g)
    with np.errstate(divide="ignore"):
        outer = 1.0 / np.maximum(radius, 1e-12)
    return np.where(x < 1.0, inner, outer)


def compensation_coulomb(r_a: float, r_b: float, distance: float,
                         points: int = 400, angles: int = 64) -> float:
    r"""Coulomb interaction of two unit compensation charges of radii
    ``r_a``/``r_b`` a ``distance`` apart: exactly :math:`1/R` for disjoint
    spheres, otherwise :math:`2\pi\int r'^2 g_B(r')\int_{-1}^{1}
    V_{g_A}(|r' + R|)\,d\mu\,dr'` by Gauss-Legendre quadrature."""
    R = float(distance)
    if R >= r_a + r_b:
        return 1.0 / R
    if R < 1e-12:
        rr = np.linspace(0.0, max(r_a, r_b), COMPENSATION_POINTS)
        return float(simpson(4.0 * np.pi * rr * rr * compensation_shape(rr, r_b)
                             * compensation_potential(rr, r_a), x=rr))
    x, wx = np.polynomial.legendre.leggauss(points)
    rp = 0.5 * r_b * (x + 1.0)
    wr = 0.5 * r_b * wx
    mu, wmu = np.polynomial.legendre.leggauss(angles)
    dist = np.sqrt(rp[:, None] ** 2 + R * R + 2.0 * rp[:, None] * R * mu[None, :])
    angular = compensation_potential(dist, r_a) @ wmu
    return float(2.0 * np.pi * np.sum(wr * rp * rp * compensation_shape(rp, r_b)
                                      * angular))


def pseudize_density(r: np.ndarray, density: np.ndarray, r_cut: float
                     ) -> np.ndarray:
    r"""Smooth counterpart of a spherical density: :math:`a + b r^2 + c r^4`
    inside ``r_cut`` matched in value, first and second derivative."""
    from .generation import _local_derivatives
    r_cut = _snap(r, r_cut)
    d = _local_derivatives(r, density, r_cut, order=2)
    # value, slope, curvature of a + b r^2 + c r^4 at r_cut
    M = np.array([[1.0, r_cut ** 2, r_cut ** 4],
                  [0.0, 2.0 * r_cut, 4.0 * r_cut ** 3],
                  [0.0, 2.0, 12.0 * r_cut ** 2]])
    a, b, c = np.linalg.solve(M, d[:3])
    smooth = a + b * r * r + c * r ** 4
    return np.where(r <= r_cut, smooth, density)


def _hartree_energy(r, rho) -> float:
    return float(0.5 * np.trapezoid(hartree_potential(r, rho) * rho
                                    * 4.0 * np.pi * r * r, r))


def _xc_energies(r, rho, xc: str = DEFAULT_XC):
    """``(E_xc[rho], int rho v_xc[rho])`` in the functional ``xc``.

    It has to be the functional the reference atom was solved with: the
    one-center constant subtracts these double-counting terms from that atom's
    band energy, and a PBE band energy corrected with LDA terms is neither.
    """
    e_xc, v_xc = xc_potential(r, rho, xc)
    shell = 4.0 * np.pi * r * r
    return (float(np.trapezoid(e_xc * rho * shell, r)),
            float(np.trapezoid(v_xc * rho * shell, r)))


# --------------------------------------------------------------------------- #
# The dataset record.
# --------------------------------------------------------------------------- #

_LOCAL_SPLINES: dict = {}


def _deficit_label(norm_deficit) -> str:
    """``norm_deficit`` as printed: a zero deficit is a unitary dataset."""
    if norm_deficit is None:
        return "free"
    return f"{float(norm_deficit):g}" + (" (unitary)" if norm_deficit == 0 else "")


def _local_spline(dataset):
    """Cubic spline of ``dataset.v_local`` (cached per table)."""
    cached = _LOCAL_SPLINES.get(id(dataset))
    if cached is not None and cached[0] is dataset.r and cached[1] is dataset.v_local:
        return cached[2]
    from scipy.interpolate import CubicSpline
    spline = CubicSpline(np.asarray(dataset.r, dtype=float),
                         np.asarray(dataset.v_local, dtype=float))
    _LOCAL_SPLINES[id(dataset)] = (dataset.r, dataset.v_local, spline)
    return spline


@dataclass
class PAWDataset(PseudoPotential):
    r"""A PAW-LCAO dataset: local potential, projectors, one-center matrices.

    Inherits the :class:`~.generation.PseudoPotential` layout so the valence
    basis (:func:`~.orbitals.pseudo_basis`, first zeta = the bound smooth
    partial wave), the local-potential sampler (:meth:`local_potential`,
    the **ionic** :math:`\tilde v^{ion}`) and the multiple-zeta hierarchy work
    unchanged; every ``channels[l]`` is a :class:`PAWChannel`.  ``projectors``
    maps ``l -> [p~_1(r), p~_2(r)]``, ``coupling`` maps ``l -> D^ion``, and
    ``overlap_correction`` maps ``l -> q`` -- the two block matrices the
    molecular path hands to :class:`~mandacaru.core.hamiltonian.MolecularIntegrals`.
    """

    coupling: dict = field(default_factory=dict)             # l -> D^ion (n, n)
    coupling_screened: dict = field(default_factory=dict)    # l -> D^scr
    overlap_correction: dict = field(default_factory=dict)   # l -> q (charge)
    norm_correction: dict = field(default_factory=dict)      # l -> q^norm
    kinetic_difference: dict = field(default_factory=dict)   # l -> Delta T
    v_local_screened: np.ndarray = None
    core_density: np.ndarray = None                          # n_c(r)
    smooth_core_density: np.ndarray = None                   # n~_c(r)
    r_cut_local: float = 0.0
    local_shift: float = 0.0
    compensation_radius: float = 0.0                         # r_g (Bohr)
    compensation_charge: float = 0.0                         # Q of n^ (electrons)
    hartree_screening: float = 0.0                           # int v_H[n~+n^] g
    one_center_energy: float = 0.0                           # frozen constant
    energies: dict = field(default_factory=dict)             # bookkeeping (Ha)
    q_cut: float = DEFAULT_Q_CUT
    energy_offset: float = DEFAULT_ENERGY_OFFSET
    norm_deficit: float | None = DEFAULT_NORM_DEFICIT
    #: How the reference atom was solved.  Defaults describe the *pre-
    #: relativistic* construction, so a dataset that does not say carries no
    #: false provenance; :func:`generate_paw` always passes the real values.
    xc: str = "lda"
    relativity: str = "none"
    #: Record of the nonlinear core correction, and channels added above the
    #: highest valence l.
    nlcc: dict = field(default_factory=dict)
    extra_l: int = 0
    #: ``l -> D_SO``, the one-center spin-orbit difference of the channel.
    #: Empty unless the dataset was generated with ``relativity="dirac"``.
    #: Unlike the ONCVPSP family, these multiply the dataset's **own**
    #: projectors -- there is one set per l, not one per j -- so the overlap
    #: operator is untouched and stays diagonal in spin.
    spin_orbit: dict = field(default_factory=dict)

    @property
    def has_spin_orbit(self) -> bool:
        """Whether this dataset carries a spin-orbit term."""
        return bool(self.spin_orbit)

    def local_potential(self, radius) -> np.ndarray:
        r"""The ionic local potential at arbitrary radii (Bohr), **C\ :sup:`2`**.

        Interpolated with a cubic spline rather than linearly: the PAW-LCAO force is
        the derivative of the energy, and a piecewise-linear potential puts a
        kink at every table point, which left the analytic and finite-difference
        forces depending on their step at the 1e-2 eV/Angstrom level.  Beyond
        the table the potential is the ionic tail :math:`-Z_{ion}/r`; below its
        first point it is flat.
        """
        spline = _local_spline(self)
        radius = np.asarray(radius, dtype=float)
        inside = radius <= self.r[-1]
        values = spline(np.clip(radius, self.r[0], self.r[-1]))
        return np.where(inside, values,
                        -self.valence_charge / np.maximum(radius, 1e-12))

    def projector(self, l: int, radius, index: int = 0) -> np.ndarray:
        """Interpolate projector ``index`` of channel ``l`` onto ``radius``."""
        radius = np.asarray(radius, dtype=float)
        p = self.projectors[int(l)][int(index)]
        return np.where(radius <= self.r[-1],
                        np.interp(np.clip(radius, self.r[0], self.r[-1]),
                                  self.r, p), 0.0)

    def compensation_shape(self, radius) -> np.ndarray:
        """Unit monopole compensation charge of this species at ``radius``."""
        return compensation_shape(radius, self.compensation_radius)

    def compensation_potential(self, radius) -> np.ndarray:
        """Coulomb potential of the unit compensation charge at ``radius``."""
        return compensation_potential(radius, self.compensation_radius)

    def reference_energies(self, l: int) -> list:
        return list(self.channels[int(l)].reference_energies)

    def projector_set(self, l: int, basis: str = "dual"):
        r"""``(functions, D, q)`` of channel ``l`` in one of the two
        equivalent projector bases.

        ``"dual"``: the PAW-LCAO projectors :math:`\tilde p_i` with the blocks
        :math:`D^{ion}` and :math:`q` as stored.  ``"raw"``: the smooth
        :math:`\chi_k = (\varepsilon_k - T - \tilde v^{scr})\tilde\varphi_k`
        they were built from, with the transformed blocks
        :math:`B^{-1} D B^{-T}` and :math:`B^{-1} q B^{-T}` -- the same
        operators :math:`\sum|\tilde p\rangle D\langle\tilde p|` and
        :math:`\sum|\tilde p\rangle q\langle\tilde p|` exactly, but the
        functions the grid samples are much softer (the dual projectors mix
        two nearly parallel :math:`\chi` with large coefficients, so their
        smooth parts cancel; on a 0.25-0.30 Angstrom grid their norm ratios
        fall to 0.76 while the raw ones stay within 10 %).
        """
        channel = self.channels[int(l)]
        D = np.asarray(channel.coupling, dtype=float)
        q = np.asarray(channel.overlap_correction, dtype=float)
        if basis == "dual":
            return list(channel.projectors), D, q
        if basis == "raw":
            B_inv = np.linalg.inv(np.asarray(channel.vanderbilt, dtype=float))
            return (list(channel.raw_projectors), B_inv @ D @ B_inv.T,
                    B_inv @ q @ B_inv.T)
        raise ValueError(f"unknown projector basis {basis!r}; use 'dual' or "
                         "'raw'")

    def multipole_moments(self, l1: int, l2: int, L: int,
                          basis: str = DEFAULT_PROJECTOR_BASIS) -> np.ndarray:
        r"""``Delta^{(L)}_{ij}`` between channels ``l1`` and ``l2``, in the
        projectors' basis.

        The radial half of the compensation multipole (see
        :mod:`mandacaru.pseudopotentials.multipoles`); ``l1 == l2`` and ``L == 0``
        gives back :attr:`PAWChannel.overlap_correction`.  ``"raw"`` applies the
        same :math:`B^{-1}\cdot B^{-T}` transformation the coupling and overlap
        blocks get, once per side.
        """
        from .multipoles import radial_moments

        key = (int(l1), int(l2), int(L), str(basis))
        cache = getattr(self, "_multipole_cache", None)
        if cache is None:
            cache = self._multipole_cache = {}
        if key not in cache:
            if int(l1) == int(l2) and int(L) == 0:
                # The monopole is stored exactly; reconstructing it from the
                # tabulated waves agrees only to ~1e-8, which the raw basis's
                # large B^{-1} amplifies to ~1e-4.  Use the stored value so the
                # L = 0 term stays bit-for-bit what the monopole-only code had.
                delta = np.asarray(self.channels[int(l1)].overlap_correction,
                                   dtype=float)
            else:
                delta = radial_moments(self, l1, l2, L)
            if basis == "raw":
                b1 = np.linalg.inv(np.asarray(
                    self.channels[int(l1)].vanderbilt, dtype=float))
                b2 = np.linalg.inv(np.asarray(
                    self.channels[int(l2)].vanderbilt, dtype=float))
                delta = b1 @ delta @ b2.T
            elif basis != "dual":
                raise ValueError(f"unknown projector basis {basis!r}")
            cache[key] = delta
        return cache[key]

    def __repr__(self) -> str:
        channels = ", ".join(f"l={l}x{len(self.projectors.get(l, []))}"
                             for l in sorted(self.channels))
        return (f"PAWDataset({self.symbol}, Z_ion={self.valence_charge:g}, "
                f"[{channels}], rg={self.compensation_radius:.2f}, "
                f"Q^={self.compensation_charge:+.4f}, "
                f"E1c={self.one_center_energy:+.4f} Ha, "
                f"deficit={_deficit_label(self.norm_deficit)})")


# --------------------------------------------------------------------------- #
# Generation.
# --------------------------------------------------------------------------- #

def spin_orbit_blocks(r, channels, v_ae, v_smooth, atomic_number,
                      mass_corrected: bool = True):
    r"""``{l: D_SO}`` -- the one-center spin-orbit difference of each channel.

    Spin-orbit coupling enters a PAW-LCAO dataset exactly the way every other
    one-center term does: as the difference between what the all-electron
    system has inside the augmentation sphere and what the smooth system has
    there,

    .. math::

        D^{SO}_{ij} = \int_0^{r_c}\Big[\xi(r)\varphi_i\varphi_j
            - \tilde\xi(r)\tilde\varphi_i\tilde\varphi_j\Big] r^2\,dr ,

    with :math:`\xi = \frac{1}{2c^2M^2 r}\frac{dV}{dr}` from the respective
    potentials (:func:`~mandacaru.basis.relativity.spin_orbit_radial`).  The
    operator it multiplies is :math:`\mathbf{L}\cdot\mathbf{S}`, so an
    ``l = 0`` channel has none.

    The smooth term is not a rounding detail to be dropped: without it the
    correction would double-count whatever spin-orbit coupling the smooth
    Hamiltonian already carries through :math:`\tilde V`.  It is small,
    because :math:`dV/dr` is where the nucleus is and the smooth potential has
    no nucleus -- but "small" is measured, not assumed.
    """
    from ..basis.relativity import spin_orbit_radial

    r = np.asarray(r, dtype=float)
    xi_ae = spin_orbit_radial(r, v_ae, atomic_number=float(atomic_number),
                              mass_corrected=mass_corrected)
    xi_ps = spin_orbit_radial(r, v_smooth, atomic_number=0.0,
                              mass_corrected=mass_corrected)
    blocks = {}
    for l, channel in channels.items():
        if int(l) == 0:
            continue
        inside = r <= channel.r_cut
        n = len(channel.ae_waves)
        D = np.zeros((n, n), dtype=float)
        weight = r * r
        for i in range(n):
            for j in range(n):
                ae = (xi_ae * channel.ae_waves[i] * channel.ae_waves[j]
                      * weight)[inside]
                ps = (xi_ps * channel.pseudo_waves[i] * channel.pseudo_waves[j]
                      * weight)[inside]
                D[i, j] = float(np.trapezoid(ae - ps, r[inside]))
        blocks[int(l)] = 0.5 * (D + D.T)
    return blocks


def generate_paw(symbol: str, *, r_cut=None, rc_factor: float = DEFAULT_RC_FACTOR,
                 r_cut_local: float | None = None,
                 local_factor: float = DEFAULT_LOCAL_FACTOR,
                 local_shift: float | None = None,
                 q_cut: float = DEFAULT_Q_CUT,
                 energy_offset: float | None = None,
                 n_bessel: int = DEFAULT_N_BESSEL,
                 norm_deficit="default",
                 points: int | None = None, r_max: float = 30.0,
                 atom: AtomicResult | None = None,
                 xc: str = DEFAULT_XC,
                 relativity: str = DEFAULT_RELATIVITY,
                 nlcc: bool | float = DEFAULT_NLCC,
                 extra_l: int = DEFAULT_EXTRA_L,
                 ghosts: str = "repair") -> PAWDataset:
    r"""Generate a PAW-LCAO dataset for ``symbol`` (see the module docstring).

    Parameters
    ----------
    r_cut : float or dict, optional
        Augmentation radii (Bohr), one value or ``{l: r_c}``; defaults to
        :data:`DEFAULT_CUTOFFS`, else ``rc_factor`` times the outermost
        maximum of the bound partial wave.
    r_cut_local : float, optional
        Radius of the polynomial local potential (default ``local_factor``
        times the smallest augmentation radius).
    local_shift : float, optional
        Raise of the local potential at the origin above its five-coefficient
        polynomial continuation (Hartree; Hamann's ``dvloc0``), the knob
        against ghost states of a deep local potential.  Defaults to
        :data:`DEFAULT_LOCAL_SHIFTS` for the element, else 0.
    norm_deficit : float or None
        Fraction of the all-electron inner norm the smooth waves give up
        (:func:`smooth_partial_waves`); defaults to
        :data:`DEFAULT_NORM_DEFICITS` for the element, else
        :data:`DEFAULT_NORM_DEFICIT`; ``None`` for free waves.
    energy_offset : float, optional
        Second reference energy above the bound state (Hartree); defaults to
        :data:`DEFAULT_ENERGY_OFFSETS` for the element, else
        :data:`DEFAULT_ENERGY_OFFSET`.
    q_cut, n_bessel, points, r_max, atom, ghosts
        As in :func:`~.oncv.generate_oncv`; the ghost search is
        :func:`~.oncv.ghost_free`, with the spectrum of the generalized
        problem (:func:`paw_spectrum`).
    """
    if ghosts != "keep":
        options = {k: v for k, v in locals().items()
                   if k not in ("symbol", "ghosts")}
        # Every repair is built at zero norm deficit.  Alone it removes the
        # ghost the deficit itself causes (iron, cutoffs untouched).  And
        # when the cutoffs have to be balanced, a positive deficit with those
        # large augmentation spheres is ghost-free but does not bind: CuH at
        # deficit 0.1 has its minimum at 1.8 Angstrom and disagrees with the
        # ONCVPSP curve by 1-2 eV; at deficit 0 both minima are at 1.46 and
        # the curves agree to 0.07-0.3 eV (HISTORY.md, 2026-09-24).
        return ghost_free(generate_paw, _paw_levels, log_derivative_paw,
                          symbol, options, ghosts,
                          overrides={"norm_deficit": 0.0})

    from ase.data import atomic_numbers

    from ..basis.relativity import _resolve as _resolve_relativity
    from .core_correction import partial_core_density

    atomic_number = int(atomic_numbers[symbol])
    relativity = _resolve_relativity(relativity)
    # A Dirac PAW-LCAO dataset is **scalar-relativistic plus a spin-orbit term**,
    # not a j-resolved augmentation sphere.  The distinction matters and is
    # not a shortcut: the PAW-LCAO overlap operator is 1 + sum |p~> q <p~|, so a
    # j-dependent q would give the *metric* of the generalized eigenproblem an
    # L.S structure, and every consumer of S -- the Loewdin orthogonalization
    # above all -- would have to learn about spin.  Keeping one set of partial
    # waves per l (the j average, which is what the Koelling-Harmon equation
    # solves) leaves q, Delta T and the compensation charges exactly as they
    # are, and carries spin-orbit coupling where it belongs: as a one-center
    # difference in the *Hamiltonian*, built the same way D and q are.
    partial_wave_treatment = "scalar" if relativity == "dirac" else relativity
    if atom is None:
        atom = solve_atom(atomic_number,
                          points=(generation_points(atomic_number)
                                  if points is None else int(points)),
                          r_max=r_max, tolerance=1e-7, mixing=0.25,
                          xc=xc, relativity=relativity)
    valence_config, core_config = _valence_configuration(
        atomic_number, configuration=atom.occupations)
    if not valence_config:
        raise ValueError(f"{symbol} has no valence subshells to pseudize")
    valence_charge = float(sum(valence_config.values()))
    r, v_ae = atom.r, atom.v_effective
    z_eff = float(atomic_number)
    shell = 4.0 * np.pi * r * r
    if isinstance(norm_deficit, str):
        norm_deficit = DEFAULT_NORM_DEFICITS.get(symbol, DEFAULT_NORM_DEFICIT)
    energy_offset = float(DEFAULT_ENERGY_OFFSETS.get(symbol, DEFAULT_ENERGY_OFFSET)
                          if energy_offset is None else energy_offset)

    per_l, cutoffs, references = reference_waves(
        symbol, atom, valence_config, z_eff, r_cut, rc_factor, energy_offset,
        defaults=DEFAULT_CUTOFFS, treatment=partial_wave_treatment,
        extra_l=extra_l)
    r_local = _snap(r, float(r_cut_local) if r_cut_local is not None
                    else float(local_factor * min(cutoffs.values())))
    r_g = float(min(cutoffs.values()))

    # An `extra_l` channel has no bound state: both its references are
    # scattering waves, normalized inside r_c by convention rather than by
    # physics.  Asking such a channel to *also* give up a fraction of that
    # norm over-constrains the Bessel expansion -- the matching conditions
    # alone already require more norm than (1 - s) allows, and the
    # minimization has no feasible point.  It is conserved exactly instead,
    # which makes its overlap correction q identically zero: the channel
    # holds no charge, so it has nothing to correct.
    bound_ls = {l for _n, l in valence_config}
    waves = {
        l: smooth_partial_waves(r, v_ae, l, ae, energies, cutoffs[l],
                                treatment=partial_wave_treatment,
                                z_eff=z_eff,
                                q_cut=q_cut, n_bessel=n_bessel,
                                norm_deficit=(norm_deficit if l in bound_ls
                                              else 0.0))
        for l, (ae, energies) in references.items()}

    shift = float(DEFAULT_LOCAL_SHIFTS.get(symbol, 0.0) if local_shift is None
                  else local_shift)
    v_loc = polynomial_local_potential(r, v_ae, r_local, shift)

    channels: dict = {}
    for l, states in per_l.items():
        channels[l] = assemble_paw_channel(
            r, waves[l], v_ae, v_loc, n=states[0][0],
            occupation=float(sum(o for _n, _e, _w, o in states)))

    # Densities of the reference atom: frozen core, all-electron valence
    # (Numerov bound states), smooth valence and its compensation charge.
    core_density = np.zeros_like(r)
    for (n, l), occupancy in core_config.items():
        core_density += occupancy * atom.orbitals[(n, l)] ** 2 / shell
    smooth_core = (pseudize_density(r, core_density, r_g)
                   if core_config else np.zeros_like(r))
    ae_valence = np.zeros_like(r)
    smooth_valence = np.zeros_like(r)
    compensation_charge = 0.0
    band = 0.0
    for l, channel in channels.items():
        ae_valence += channel.occupation * channel.ae_waves[0] ** 2 / (4.0 * np.pi)
        smooth_valence += channel.occupation * channel.pseudo_radial ** 2 \
            / (4.0 * np.pi)
        compensation_charge += channel.occupation * channel.overlap_correction[0, 0]
        band += channel.occupation * channel.eigenvalue
    g = compensation_shape(r, r_g)
    augmented = smooth_valence + compensation_charge * g

    # Unscreening: Hartree of the neutral smooth density, xc of the smooth
    # valence density -- plus the smooth core when the nonlinear core
    # correction is on.  PAW-LCAO already carries `smooth_core`, the pseudized
    # frozen core it needs for its one-center energies, so the correction
    # here is a matter of *including* that density in v_xc rather than
    # building a second one: what the Louie-Froyen-Cohen sin(Br)/r form does
    # for a norm-conserving family, `pseudize_density` has already done.
    nlcc_details = {"applied": False, "r_nlcc": None,
                    "reason": ("not requested" if nlcc is False
                               else "the atom has no core to correct for")}
    xc_core = np.zeros_like(r)
    if nlcc is not False and core_config:
        xc_core = smooth_core
        if nlcc is not True:
            # An explicit radius re-pseudizes the true core at that radius
            # instead of at the compensation radius r_g.
            xc_core, nlcc_details = partial_core_density(
                r, core_density, smooth_valence, r_nlcc=float(nlcc))
        else:
            nlcc_details = {
                "applied": True, "r_nlcc": float(r_g), "source": "smooth_core",
                "core_electrons": float(np.trapezoid(core_density * shell, r)),
                "partial_core_electrons": float(
                    np.trapezoid(smooth_core * shell, r))}
    v_hartree = hartree_potential(r, augmented)
    _e_xc, v_xc = xc_potential(r, smooth_valence + xc_core, xc)
    v_local_ionic = v_loc - v_hartree - v_xc
    hartree_screening = float(np.trapezoid(v_hartree * g * shell, r))
    # Spin-orbit coupling, when asked for: a one-center difference like every
    # other PAW-LCAO matrix.  `v_loc` is the *screened* smooth potential, which is
    # what the smooth Hamiltonian actually carries inside the sphere.
    spin_orbit = (spin_orbit_blocks(r, channels, v_ae, v_loc, atomic_number)
                  if relativity == "dirac" else {})
    for channel in channels.values():
        channel.v_ionic = v_local_ionic
        channel.coupling = (channel.coupling_screened
                            - channel.overlap_correction * hartree_screening)

    # Frozen one-center constant: the reference atom evaluated with the
    # molecular machinery must give the all-electron valence energy in the
    # norm-conserving convention.
    e_h_ae = _hartree_energy(r, ae_valence)
    e_h_ps = _hartree_energy(r, augmented)
    e_xc_ae, v_xc_ae = _xc_energies(r, ae_valence, xc)
    e_xc_ps, v_xc_ps = _xc_energies(r, smooth_valence, xc)
    reference_valence = band - e_h_ae - v_xc_ae + e_xc_ae
    pseudo_atom = band - e_h_ps - v_xc_ps + e_xc_ps
    one_center = reference_valence - pseudo_atom
    e_xc_full_ae, _v = _xc_energies(r, ae_valence + core_density, xc)
    e_xc_full_ps, _v = _xc_energies(r, smooth_valence + smooth_core, xc)
    energies = {
        "band": float(band),
        "reference_valence": float(reference_valence),
        "pseudo_atom": float(pseudo_atom),
        "one_center": float(one_center),
        "hartree_ae": e_h_ae, "hartree_ps": e_h_ps,
        "xc_ae": e_xc_ae, "xc_ps": e_xc_ps,
        "core_valence_xc_omitted": float((e_xc_full_ae - e_xc_ae)
                                         - (e_xc_full_ps - e_xc_ps)),
        "atom_total": float(atom.total_energy),
    }

    return PAWDataset(
        symbol=symbol, atomic_number=atomic_number,
        valence_charge=valence_charge, r=r, channels=channels,
        v_local=v_local_ionic, local_l=-1,
        projectors={l: list(c.projectors) for l, c in channels.items()},
        kb_energies={}, valence_density=smooth_valence, atom=atom,
        family=FAMILY,
        coupling={l: np.array(c.coupling) for l, c in channels.items()},
        coupling_screened={l: np.array(c.coupling_screened)
                           for l, c in channels.items()},
        overlap_correction={l: np.array(c.overlap_correction)
                            for l, c in channels.items()},
        norm_correction={l: np.array(c.norm_correction)
                         for l, c in channels.items()},
        kinetic_difference={l: np.array(c.kinetic_difference)
                            for l, c in channels.items()},
        v_local_screened=v_loc, core_density=core_density,
        smooth_core_density=smooth_core, r_cut_local=r_local,
        local_shift=shift, compensation_radius=r_g, compensation_charge=float(compensation_charge),
        hartree_screening=hartree_screening, one_center_energy=float(one_center),
        energies=energies, q_cut=float(q_cut),
        energy_offset=float(energy_offset),
        norm_deficit=None if norm_deficit is None else float(norm_deficit),
        xc=str(xc), relativity=relativity, nlcc=dict(nlcc_details),
        extra_l=int(extra_l), spin_orbit=spin_orbit)


# --------------------------------------------------------------------------- #
# Diagnostics: generalized spectrum, reconstruction, logarithmic derivatives.
# --------------------------------------------------------------------------- #

#: Spacing (Bohr) of the uniform grid the radial spectrum is solved on.
SPECTRUM_SPACING = 0.005


def _channel_operator(pp: PAWDataset, l: int, r_max: float, stride: int):
    """Grid, local potential, ``u``-form projectors, ``D^scr`` and ``q``."""
    h = stride * SPECTRUM_SPACING
    return _channel_operator_on(pp, l, np.arange(1, int(r_max / h) + 1) * h)


def _channel_operator_on(pp: PAWDataset, l: int, r):
    """:func:`_channel_operator` on a caller's uniform grid ``r`` (the confined
    orbitals of :mod:`~.confinement` put a node exactly on their wall)."""
    from scipy.interpolate import CubicSpline

    v = CubicSpline(pp.r, pp.v_local_screened)(r) + l * (l + 1) / (2.0 * r * r)
    p_u = [CubicSpline(pp.r, np.asarray(p))(r) * r for p in pp.projectors[l]]
    return (r, v, p_u, np.asarray(pp.coupling_screened[l], dtype=float),
            np.asarray(pp.overlap_correction[l], dtype=float))


def _generalized_matrices(r, v, p_u, D, q):
    h = r[1] - r[0]
    n = r.size
    H = np.diag(1.0 / h ** 2 + v)
    idx = np.arange(n - 1)
    H[idx, idx + 1] = H[idx + 1, idx] = -0.5 / h ** 2
    S = np.eye(n)
    if p_u:
        X = np.array(p_u)
        H += h * X.T @ D @ X
        S += h * X.T @ q @ X
    return H, S


def _generalized_spectrum(r, v, p_u, D, q, n_states):
    from scipy.linalg import eigh
    H, S = _generalized_matrices(r, v, p_u, D, q)
    return eigh(H, S, eigvals_only=True, subset_by_index=[0, n_states - 1])


def paw_spectrum(pp: PAWDataset, l: int, n_states: int = 3,
                 r_max: float | None = None, stride: int = 2) -> np.ndarray:
    r"""Lowest eigenvalues of the generalized problem :math:`(T_l +
    \tilde v^{scr} + \sum|\tilde p_i\rangle D^{scr}_{ij}\langle\tilde p_j|)c =
    \varepsilon(1 + \sum|\tilde p_i\rangle q_{ij}\langle\tilde p_j|)c` on the
    radial grid (3-point Laplacian at ``stride`` and ``2 * stride`` times
    :data:`SPECTRUM_SPACING`, Richardson-extrapolated).  A ghost state shows
    up as an eigenvalue below the bound reference energy.
    """
    r_max = _spectrum_extent(pp, l) if r_max is None else float(r_max)
    fine = _generalized_spectrum(*_channel_operator(pp, l, r_max, stride),
                                 n_states)
    coarse = _generalized_spectrum(*_channel_operator(pp, l, r_max, 2 * stride),
                                   n_states)
    return (4.0 * fine - coarse) / 3.0


def _paw_levels(pp, l: int) -> np.ndarray:
    """The two lowest eigenvalues of one PAW-LCAO channel
    (:func:`~.oncv.ghost_free`)."""
    return paw_spectrum(pp, l, n_states=2)


def paw_eigenstate(pp: PAWDataset, l: int, r_max: float | None = None,
                   stride: int = 2):
    """Lowest smooth eigenfunction ``(r, u, energy)`` of the generalized
    problem, Richardson-extrapolated pointwise from the ``stride`` and
    ``2 * stride`` grids (returned on the coarser one), ``S``-normalized
    and positive near the origin."""
    from scipy.linalg import eigh

    from scipy.interpolate import CubicSpline

    r_max = _spectrum_extent(pp, l) if r_max is None else float(r_max)
    bound = CubicSpline(pp.r, pp.channels[int(l)].pseudo_radial)
    out = []
    for s in (stride, 2 * stride):
        r, v, p_u, D, q = _channel_operator(pp, l, r_max, s)
        H, S = _generalized_matrices(r, v, p_u, D, q)
        values, vectors = eigh(H, S, subset_by_index=[0, 0])
        u = vectors[:, 0]
        h = r[1] - r[0]
        u = u / np.sqrt(h * u @ S @ u)                 # int u S u = 1
        # Sign: the stored bound smooth wave (a nodeless 2s is negative
        # near the origin, so "positive at the origin" is the wrong rule).
        if np.trapezoid(u * bound(r) * r, dx=h) < 0:
            u = -u
        out.append((r, u, float(values[0])))
    (r_f, u_f, e_f), (r_c, u_c, e_c) = out
    u = (4.0 * u_f[1::2] - u_c) / 3.0
    return r_c, u, (4.0 * e_f - e_c) / 3.0


def reconstruct_ae(pp: PAWDataset, l: int, r: np.ndarray, u: np.ndarray
                   ) -> np.ndarray:
    r"""Blöchl's transformation of a smooth radial function ``u = r R``:
    :math:`u = \tilde u + \sum_i (u_i - \tilde u_i)\langle\tilde p_i|\tilde u\rangle`."""
    from scipy.interpolate import CubicSpline
    channel = pp.channels[int(l)]
    out = np.array(u, dtype=float)
    for ae, ps, p in zip(channel.ae_waves, channel.pseudo_waves,
                         channel.projectors):
        p_u = CubicSpline(pp.r, p)(r) * r
        weight = float(simpson(p_u * u, x=r))
        out += weight * (CubicSpline(pp.r, ae)(r) - CubicSpline(pp.r, ps)(r)) * r
    return out


def log_derivative_paw(pp: PAWDataset, l: int, energy: float,
                       r_cut: float | None = None) -> float:
    r"""Smooth logarithmic derivative :math:`\tilde R'/\tilde R` at ``r_cut``
    of the generalized problem at ``energy``: :func:`~.oncv.log_derivative_ps`
    with the projectors :math:`\tilde p_i` and the energy-dependent coupling
    :math:`D^{scr} - E\,q`.  Equals the all-electron one at the reference
    energies (and, for a good dataset, in between)."""
    l = int(l)
    r_cut = pp.channels[l].r_cut if r_cut is None else float(r_cut)
    r = pp.r
    r0 = _with_origin(r)
    v0 = np.concatenate([[pp.v_local_screened[0]], pp.v_local_screened])
    f = _radial_f(r0, v0, l, energy)
    p_u = [np.concatenate([[0.0], np.asarray(p) * r]) for p in pp.projectors[l]]
    D = (np.asarray(pp.coupling_screened[l], dtype=float)
         - float(energy) * np.asarray(pp.norm_correction[l], dtype=float))
    seed = (r0[1] ** (l + 1), r0[2] ** (l + 1))
    u0 = numerov_outward(r0, f, np.zeros_like(r0), seed, start=1)
    uj = [numerov_outward(r0, f, 2.0 * p, (0.0, 0.0), start=1) for p in p_u]
    dr = r0[1] - r0[0]
    m0 = np.array([np.trapezoid(p * u0, dx=dr) for p in p_u])
    M = np.array([[np.trapezoid(p * u, dx=dr) for u in uj] for p in p_u])
    a = np.linalg.solve(np.eye(len(p_u)) - D @ M, D @ m0)
    u = u0 + sum(ai * ui for ai, ui in zip(a, uj))
    return _log_derivative_of_u(r0, u, r_cut)


def check_paw_channel(pp: PAWDataset, l: int, midpoint: bool = True) -> dict:
    """Validation numbers of one channel.

    ``duality_error``, ``asymmetry``, ``consistency_error``, ``residual_kinetic``, ``overlap_correction``
        Stored construction diagnostics (the last is the ``q`` block).

    ``eigenvalue_error``, ``spectrum``
        Lowest eigenvalue of the generalized atomic problem minus the bound
        reference energy (a ghost makes it negative), and the spectrum.

    ``reconstruction_error``
        Largest deviation of the all-electron wave reconstructed from the
        lowest smooth eigenfunction, :func:`reconstruct_ae`, from the stored
        all-electron partial wave (``u`` form, unit norm).
    ``tail_error``
        Largest deviation of the bound smooth wave from the all-electron one
        beyond ``r_c``.
    ``log_derivative_errors``
        ``{energy: (|L_paw - L_ae|, L_ae)}`` at the two reference energies
        and (when ``midpoint``) halfway between; needs ``pp.atom``.
    """
    from scipy.interpolate import CubicSpline

    channel = pp.channels[int(l)]
    r = pp.r
    energies = list(channel.reference_energies)
    spectrum = paw_spectrum(pp, l)
    # The inner lobe of a nodal 2s needs the 0.005 Bohr grid to reconstruct
    # to 1e-5 (0.01 Bohr leaves 1e-4); a diffuse wave with a 24 Bohr box is
    # smooth enough at 0.01 Bohr and would cost a 4800-point dense solve.
    extent = _spectrum_extent(pp, l)
    r_u, u, _e = paw_eigenstate(pp, l, stride=1 if extent <= 15.0 else 2)
    reconstructed = reconstruct_ae(pp, l, r_u, u)
    ae_u = CubicSpline(r, channel.ae_waves[0])(r_u) * r_u
    # The radial solver has a Dirichlet wall at its box edge, where a diffuse
    # tail (Li 2s at 24 Bohr: 1e-4) is forced to zero; compare inside it.
    core = r_u <= 0.75 * r_u[-1]
    outside = r > channel.r_cut
    out = {
        "eigenvalue_error": float(spectrum[0] - energies[0]),
        "spectrum": spectrum,
        "duality_error": float(channel.duality_error),
        "overlap_minimum": float(channel.overlap_minimum),
        "asymmetry": float(channel.asymmetry),
        "consistency_error": float(channel.consistency_error),
        "overlap_correction": np.asarray(channel.overlap_correction),
        "residual_kinetic": list(channel.residual_kinetic),
        "reconstruction_error": float(np.max(np.abs(reconstructed - ae_u)[core])),
        "smooth_deviation": float(np.max(np.abs(u - ae_u)[core])),
        "tail_error": float(np.max(np.abs(channel.pseudo_radial[outside]
                                          - channel.ae_waves[0][outside]))),
        "nodes": int(np.sum(np.diff(np.sign(
            channel.pseudo_radial[(r > 0.05) & (r < 5.0)])) != 0)),
        "log_derivative_errors": None,
    }
    if pp.atom is None:
        return out
    out["log_derivative_errors"] = log_derivative_errors(
        pp, l, energies, channel.r_cut, log_derivative_paw, midpoint)
    return out


def report_paw(pp: PAWDataset) -> str:
    """Human-readable validation summary for every channel."""
    lines = [f"{pp!r}",
             f"  valence charge  : {pp.valence_charge:g}",
             f"  norm deficit    : {_deficit_label(pp.norm_deficit)}",
             f"  local potential : polynomial inside rcl = {pp.r_cut_local:.3f} "
             f"Bohr, V_loc(0) = {pp.v_local_screened[0]:+.4f} Ha (screened)",
             f"  compensation    : Q^ = {pp.compensation_charge:+.5f} e in "
             f"rg = {pp.compensation_radius:.3f} Bohr, int v_H g = "
             f"{pp.hartree_screening:+.5f} Ha",
             f"  one-center      : E_1c = {pp.one_center_energy:+.6f} Ha "
             f"(reference valence {pp.energies.get('reference_valence', 0.0):+.6f}, "
             f"pseudo atom {pp.energies.get('pseudo_atom', 0.0):+.6f})"]
    for l in sorted(pp.channels):
        channel = pp.channels[l]
        checks = check_paw_channel(pp, l)
        lines.append(f"  l={l}: rc={channel.r_cut:.3f}  eps="
                     + ", ".join(f"{e:+.4f}" for e in channel.reference_energies)
                     + f"  q={np.round(channel.overlap_correction, 5).tolist()}"
                     f"  duality={checks['duality_error']:.1e}"
                     f"  eps err={checks['eigenvalue_error']:+.1e}"
                     f"  |dphi|={checks['reconstruction_error']:.1e}"
                     f"  nodes={checks['nodes']}")
        for energy, (error, l_ae) in (checks["log_derivative_errors"]
                                      or {}).items():
            lines.append(f"        L(E={energy:+.4f}) = {l_ae:+.5f}  "
                         f"|dL| = {error:.1e}")
        lines.append("        D^ion = " + np.array2string(
            channel.coupling, precision=5, suppress_small=True).replace("\n", ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Molecule: projectors, blocks, augmented integrals, the family builder.
# --------------------------------------------------------------------------- #

def paw_projectors(symbols, positions, datasets, units: str = "angstrom",
                   projector_basis: str = DEFAULT_PROJECTOR_BASIS):
    """The :class:`~.orbitals.KBProjector` set: every radial projector of
    every ``(atom, l, m)``, in the ``projector_basis`` of
    :meth:`PAWDataset.projector_set` (each projector records it as
    ``projector_basis`` so the block builders transform consistently)."""
    from .orbitals import KBProjector

    projectors = []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = datasets[symbol]
        for l in sorted(pp.projectors):
            functions, D, _q = pp.projector_set(l, projector_basis)
            for m in range(-l, l + 1):
                for i, p in enumerate(functions):
                    projector = KBProjector(
                        pp, l, m, center=position, units=units,
                        atom_index=index, index=i, radial=p,
                        kb_energy=float(D[i, i]))
                    projector.projector_basis = projector_basis
                    projectors.append(projector)
    return projectors


def _blocks(projectors, symbols, datasets, which) -> dict:
    blocks = {}
    for projector in projectors:
        key = projector.block_key
        if key not in blocks:
            pp = datasets[symbols[projector.atom_index]]
            basis = getattr(projector, "projector_basis",
                            DEFAULT_PROJECTOR_BASIS)
            _f, D, q = pp.projector_set(projector.l, basis)
            blocks[key] = np.asarray(D if which == "coupling" else q,
                                     dtype=complex)
    return blocks


def paw_spin_orbit_blocks(projectors, symbols, datasets) -> dict:
    """``{(atom, l): D_SO}`` -- the spin-orbit blocks, empty without them.

    Keyed by ``(atom, l)`` and not ``(atom, l, m)``: the spin-orbit term is
    the one part of the nonlocal potential that couples different ``m``, so it
    cannot be a block of the same block-diagonal matrix
    (:func:`mandacaru.core.spin_orbit.spin_orbit_one_body` consumes it).
    """
    blocks: dict = {}
    for projector in projectors:
        dataset = datasets[symbols[projector.atom_index]]
        table = getattr(dataset, "spin_orbit", None)
        if not table or projector.l not in table:
            continue
        key = (projector.atom_index, projector.l)
        if key not in blocks:
            blocks[key] = np.asarray(table[projector.l], dtype=complex)
    return blocks


def paw_coupling_blocks(projectors, symbols, datasets) -> dict:
    """``{(atom, l, m): D_l}`` for ``nonlocal_coupling`` (in the
    projectors' basis)."""
    return _blocks(projectors, symbols, datasets, "coupling")


def _multipole_grid_potential(radius, dx, dy, dz, r_g, L, M):
    """``v_L(r) Y_LM`` sampled on grid offsets from one center."""
    from .multipoles import shape_potential
    from ..basis._angular import spherical_harmonic

    v = shape_potential(radius, r_g, L)
    if int(L) == 0:
        return v * spherical_harmonic(0, 0, 0.0, 0.0)
    safe = np.maximum(radius, 1e-300)
    theta = np.arccos(np.clip(dz / safe, -1.0, 1.0))
    phi = np.arctan2(dy, dx)
    return v * spherical_harmonic(int(L), int(M), theta, phi)


def paw_multipole_blocks(projectors, datasets_by_atom) -> dict:
    r"""``{(atom, L, M): matrix}`` -- the compensation multipole moments.

    ``matrix[a, b]`` is the ``LM`` moment carried by the pair density of this
    atom's projectors ``a`` and ``b``, i.e. the radial moment
    :math:`\Delta^{(L)}` times the angular coupling
    :math:`\int Y^*_{LM}Y^*_{l_am_a}Y_{l_bm_b}d\Omega`.  Rows and columns are
    numbered within the atom, in the order the projectors appear;
    ``datasets_by_atom[atom]`` is that atom's dataset.

    ``L = 0`` reproduces ``paw_overlap_blocks`` scaled by
    :math:`1/\sqrt{4\pi}`, which is the normalization that makes the
    :math:`L = 0` term of the augmentation identical to the monopole-only code
    it replaces.  Channels that carry no moment are omitted.
    """
    from .multipoles import gaunt, multipole_range

    per_atom: dict = {}
    for position, projector in enumerate(projectors):
        per_atom.setdefault(projector.atom_index, []).append(
            (position, projector))

    blocks: dict = {}
    for atom, entries in per_atom.items():
        dataset = datasets_by_atom[atom]
        basis = getattr(entries[0][1], "projector_basis",
                        DEFAULT_PROJECTOR_BASIS)
        n = len(entries)
        levels = sorted({L for _p, a in entries for _q, b in entries
                         for L in multipole_range(a.l, b.l)})
        for L in levels:
            for M in range(-L, L + 1):
                matrix = np.zeros((n, n), dtype=complex)
                for a, (_pa, pa) in enumerate(entries):
                    for b, (_pb, pb) in enumerate(entries):
                        angular = gaunt(L, M, pa.l, pa.m, pb.l, pb.m)
                        if angular == 0:
                            continue
                        delta = dataset.multipole_moments(pa.l, pb.l, L, basis)
                        matrix[a, b] = delta[pa.index, pb.index] * angular
                if np.any(np.abs(matrix) > 1e-14):
                    blocks[(atom, L, M)] = matrix
    return blocks


def paw_overlap_blocks(projectors, symbols, datasets) -> dict:
    """``{(atom, l, m): q_l}`` for ``nonlocal_overlap`` (in the
    projectors' basis)."""
    return _blocks(projectors, symbols, datasets, "overlap")


def _projector_sphere(projector):
    """Quadrature points (Bohr, a coordinate triple) and weights over the
    sphere of radius ``projector.r_cut`` around ``projector.center``."""
    x, wx = np.polynomial.legendre.leggauss(PROJECTION_RADIAL_POINTS)
    t, wt = np.polynomial.legendre.leggauss(PROJECTION_POLAR_POINTS)
    n_phi = PROJECTION_AZIMUTHAL_POINTS
    azimuth = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    sin_t = np.sqrt(1.0 - t * t)
    directions = np.stack([(sin_t[:, None] * np.cos(azimuth)[None, :]).ravel(),
                           (sin_t[:, None] * np.sin(azimuth)[None, :]).ravel(),
                           np.repeat(t, n_phi)])
    angular = np.repeat(wt, n_phi) * (2.0 * np.pi / n_phi)
    r_cut = float(projector.r_cut)
    radii = 0.5 * r_cut * (x + 1.0)
    weights = (0.5 * r_cut * wx * radii * radii)[:, None] * angular[None, :]
    center = np.asarray(projector.center, dtype=float)
    points = tuple(center[i] + radii[:, None] * directions[i][None, :]
                   for i in range(3))
    return points, weights


def atom_centered_projections(basis, projectors) -> np.ndarray:
    r"""``C[mu, p] = <phi_mu|p_p>`` by quadrature over each projector's sphere.

    See :meth:`PAWIntegrals.projections`.  Returns an ``(M, P)`` array.
    """
    C = np.zeros((len(basis), len(projectors)), dtype=complex)
    for p, projector in enumerate(projectors):
        points, weights = _projector_sphere(projector)
        values = projector.evaluate(*points) * weights
        for mu, fn in enumerate(basis):
            C[mu, p] = np.sum(np.conj(fn.evaluate(*points)) * values)
    return C


def atom_centered_projection_gradients(basis, projectors,
                                       delta: float = 1e-3) -> np.ndarray:
    r"""``G[mu, p, k] = <d phi_mu / d R_k | p_p>``, the basis function moving.

    Same quadrature as :func:`atom_centered_projections`; the orbital
    derivative with respect to its center is a central difference of the
    analytic function (step ``delta``, Bohr).  Because the projection depends
    only on the separation, moving the projector instead gives ``-G``.
    Returns an ``(M, P, 3)`` array.
    """
    G = np.zeros((len(basis), len(projectors), 3), dtype=complex)
    for p, projector in enumerate(projectors):
        points, weights = _projector_sphere(projector)
        values = projector.evaluate(*points) * weights
        for mu, fn in enumerate(basis):
            for k in range(3):
                shift = [0.0, 0.0, 0.0]
                shift[k] = delta
                plus = fn.evaluate(*(points[i] - shift[i] for i in range(3)))
                minus = fn.evaluate(*(points[i] + shift[i] for i in range(3)))
                G[mu, p, k] = np.sum(np.conj((plus - minus) / (2.0 * delta))
                                     * values)
    return G


class PAWIntegrals(MolecularIntegrals):
    r"""Molecular integrals with PAW-LCAO compensation charges.

    A :class:`~mandacaru.core.hamiltonian.MolecularIntegrals` whose
    two-body tensor is built from the augmented pair densities
    :math:`\rho_{pr} = \tilde\phi_p^*\tilde\phi_r + \sum_A Q^A_{pr}\,g_A`,
    :math:`Q^A = (C q C^\dagger)^A` being the augmentation moments of atom
    ``A`` -- one per multipole channel :math:`(L, M)`, the :math:`L = 0` one
    being the block that also augments the overlap:

    .. math::

        \langle pq|rs\rangle = \langle pq|rs\rangle_{grid}
          + \sum_A \big(Q^A_{pr} W^A_{qs} + W^A_{pr} Q^A_{qs}\big)
          + \sum_{AB} Q^A_{pr}\, U_{AB}\, Q^B_{qs},

    with :math:`W^A_{qs} = \int \tilde\phi_q^*\tilde\phi_s V_{g_A}` (grid)
    and :math:`U_{AB}` the compensation-compensation Coulomb integrals
    (:func:`compensation_coulomb`).  ``constant_energy`` carries the sum
    of the species' frozen one-center energies.
    """

    def __init__(self, nuclei, basis, grid, *, datasets, **kwargs):
        super().__init__(nuclei, basis, grid, pseudos=datasets,
                         **kwargs)
        self.datasets = list(datasets)
        self.constant_energy = float(sum(d.one_center_energy
                                         for d in self.datasets))
        self._Q = None
        self._W = None
        self._U = None
        self._Vion = None
        self._Vsr = None

    #: The projections are exact atom-centered integrals (:meth:`projections`),
    #: which the force code differentiates accordingly.
    exact_projections = True

    #: Integrate the local potential's short-range part on atom-centered
    #: spheres instead of the grid (:meth:`short_range_local`).  A pure
    #: quadrature-accuracy improvement, on by default; set ``False`` on an
    #: instance or a subclass to put the whole local potential back on the
    #: grid, which is what the comparison tests do.
    exact_local_potential = True

    # -- the range-separated local potential ------------------------------- #

    @property
    def split_local_potential(self) -> bool:
        """Whether the local potential is range-separated on this instance.

        :attr:`exact_local_potential` and at least one dataset to separate.
        """
        return bool(self.exact_local_potential) and bool(self.datasets)

    def local_split_width(self) -> float:
        """Gaussian width ``sigma`` (Bohr) of the range separation on this grid."""
        from .local_split import split_width
        return split_width(self.grid)

    def external_potential(self):
        r"""The callable the engine samples.

        With :attr:`exact_local_potential` on this is only the **long-range**
        half of the local channel, :math:`-Z^{ion}_A\,
        \mathrm{erf}(r/\sqrt2\sigma)/r` per atom: the potential of a Gaussian
        ion wide enough for the grid to resolve
        (:data:`~.local_split.SIGMA_FACTOR`).  The rest arrives through
        :meth:`short_range_local`.  Off, it is the full local channel as
        :class:`~mandacaru.core.hamiltonian.MolecularIntegrals` samples it.
        """
        if not self.split_local_potential:
            return super().external_potential()
        from .local_split import long_range_sampler
        return long_range_sampler(self._potentials.nuclei, self.datasets,
                                  self.local_split_width())

    def short_range_local_matrices(self, *, gradients: bool = False,
                                   delta=None):
        """``({atom: I_A}, {atom: G_A})`` of the short-range local term.

        ``I_A`` is atom ``A``'s sphere integral and ``G_A[p, q, k]`` the same
        integral with :math:`\\phi_p`'s own center displaced -- what the force
        needs for both of its halves (see
        :func:`~.local_split.short_range_matrices`).  ``G`` is ``None`` unless
        ``gradients`` is set; the matrices themselves are **not** cached here
        (:meth:`short_range_local` caches the assembled sum).
        """
        from .local_split import DEFAULT_DELTA, short_range_matrices

        if not self.split_local_potential:
            return {}, (None if not gradients else {})
        centers = [np.asarray(center, dtype=float)
                   for _z, center in self._potentials.nuclei]
        return short_range_matrices(
            self.basis, centers, self.datasets, self.local_split_width(),
            gradients=gradients,
            delta=DEFAULT_DELTA if delta is None else float(delta))

    def short_range_local(self):
        r"""``sum_A int phi_p^* phi_q v^{sr}_A``, by atom-centered quadrature.

        The local potential decays as :math:`-Z^{ion}/r` and so cannot be
        integrated in a finite sphere; it is split into the potential of a
        Gaussian ion (long ranged but smooth -- kept on the grid, see
        :meth:`external_potential`) and the remainder, which vanishes beyond a
        few Gaussian widths and is integrated here on a spherical product
        quadrature centered on each atom, exactly as the projections are
        (:meth:`projections`).  The result depends only on the separations of
        the basis functions from the sphere's center, so it is **exactly
        translation invariant** -- a rigid shift moves it by 3e-16.

        The two halves sum to the potential the un-split calculation used, so
        the difference is quadrature accuracy alone: ~3e-7 Hartree on the
        hardest case measured (water PAW-LCAO-DZ) and better than 1e-9 on H2.

        **What this is not:** a cure for the egg-box.  The *long-range* half
        stays on the grid and keeps a ripple of its own, comparable to (and at
        h >= 0.25 Angstrom larger than) the un-split potential's, because the
        artifact is dominated by the sampling of the pair density rather than
        of the potential.  :mod:`.local_split` carries the measured per-term
        table and the width scan behind that statement.
        """
        if not self.split_local_potential:
            return None
        if self._Vsr is None:
            matrices, _ = self.short_range_local_matrices()
            total = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
            for matrix in matrices.values():
                total = total + matrix
            self._Vsr = total
        return self._Vsr

    def local_potential_functions(self):
        """Per-atom radial callables of whatever :meth:`external_potential` samples.

        The long-range Gaussian-ion potentials when the split is active, else
        the datasets' own local channels.  The nuclear gradient's
        Hellmann-Feynman term differentiates the potential *on the grid*, so it
        has to be handed the same half the grid was given -- see
        :func:`mandacaru.algorithms.pseudo_forces._atom_potentials`.
        """
        if not self.split_local_potential:
            return [dataset.local_potential for dataset in self.datasets]
        from .local_split import long_range_potential

        sigma = self.local_split_width()
        return [(lambda radius, Z=float(dataset.valence_charge), s=sigma:
                 long_range_potential(Z, s, radius))
                for dataset in self.datasets]

    def projections(self) -> np.ndarray:
        r"""``C[mu, p] = <phi_mu|p_p>``, integrated over each projector's sphere.

        A projector vanishes beyond its cutoff, so
        :math:`\langle\phi_\mu|p_p\rangle = \int_{|\mathbf x| < r_c}
        \phi^*_\mu(\mathbf R_p + \mathbf x)\, p(\mathbf x)\, d^3x`, which is
        evaluated with a spherical product quadrature centered on the projector
        (:func:`atom_centered_projections`) instead of the real-space grid.  The
        result depends only on the relative position of the basis function and
        the projector, so it is **exactly translation invariant** and the
        on-site entries :math:`\delta_{ll'}\delta_{mm'}\int R_\mu p\, r^2dr`
        come out right by construction.

        The grid quadrature this replaces is poor for these functions: the dual
        projectors are sharp (the two reference waves are nearly parallel in
        the core, so :math:`B^{-1}` is large), and on an H\ :sub:`2` DZP basis
        at h = 0.25 Angstrom rigidly translating the molecule by one grid step
        moved the nonlocal energy by 3.9 eV, the augmented overlap by 1.0 eV
        and the compensation Coulomb term by 1.4 eV -- an egg-box that swamps
        any force.  With the quadrature those ripples vanish; the remaining
        grid ripple (kinetic, local potential, Hartree) is 18 / 13 / 5 meV at
        h = 0.25 / 0.20 / 0.15 Angstrom.  The quadrature is converged to
        4e-7 and agrees with an h = 0.10 Angstrom grid to 1e-4.  The grid
        values are still computed once, for :attr:`kb_resolution_ratios`.
        """
        if self._C is not None:
            return self._C
        super().projections()               # grid values: resolution ratios
        self._C = atom_centered_projections(self.basis, self.kb_projectors)
        return self._C

    def _atom_positions(self):
        groups = projector_blocks(self.kb_projectors)
        per_atom: dict = {}
        for (atom, _l, _m), positions in groups.items():
            per_atom.setdefault(atom, []).extend(positions)
        return per_atom

    def multipole_channels(self) -> list:
        """``[(atom, L, M), ...]`` carrying a compensation multipole, ordered.

        An atom with only an ``s`` channel contributes ``(atom, 0, 0)`` alone;
        one with ``p`` valence adds the ``L = 1`` dipoles and ``L = 2``
        quadrupoles its pair densities carry.
        """
        return sorted(self.compensation_moments())

    def compensation_moments(self) -> dict:
        r"""``{(atom, L, M): Q}`` -- the augmentation multipole moments
        :math:`Q^{A,LM}_{pr} = \sum_{ij\in A} C_{pi}\,\Delta^{LM}_{ij}\,
        C^*_{rj}` over the orbital pairs.

        The ``L = 0`` entry is the old monopole exactly (the shape and the
        moment each pick up a compensating :math:`\sqrt{4\pi}`); every higher
        ``L`` is a term the monopole-only code dropped.
        """
        if self._Q is None:
            C = self.projections()
            blocks = paw_multipole_blocks(self.kb_projectors, self.datasets)
            positions = self._atom_positions()
            self._Q = {}
            for (atom, L, M), matrix in blocks.items():
                idx = np.asarray(positions[atom])
                Ca = C[:, idx]
                self._Q[(atom, L, M)] = Ca @ matrix @ Ca.conj().T
        return self._Q

    def compensation_potentials(self) -> dict:
        r"""``{(atom, L, M): W}`` -- :math:`W^{A,LM}_{qs} = \int
        \tilde\phi_q^*\tilde\phi_s\,v_L(r_A)\,Y_{LM}(\hat r_A)\,d^3r`
        on the grid."""
        if self._W is None:
            psi = self._engine._psi
            X, Y, Z = (self.grid.X.ravel(), self.grid.Y.ravel(),
                       self.grid.Z.ravel())
            self._W = {}
            for atom, L, M in sorted(self.compensation_moments()):
                _z, center = self._potentials.nuclei[atom]
                dx, dy, dz = X - center[0], Y - center[1], Z - center[2]
                radius = np.sqrt(dx * dx + dy * dy + dz * dz)
                v = _multipole_grid_potential(
                    radius, dx, dy, dz,
                    self.datasets[atom].compensation_radius, L, M)
                self._W[(atom, L, M)] = (np.conj(psi) * v) @ psi.T * self.grid.dV
        return self._W

    def compensation_coulomb(self) -> np.ndarray:
        """``U[a, b]`` over :meth:`multipole_channels` -- the Coulomb energy of
        the unit compensation multipoles.

        Unlike the monopole case this depends on the *direction* between the
        centers, not only their separation, which is exactly the physics the
        dipole terms carry.
        """
        from .multipoles import multipole_coulomb_matrix

        if self._U is None:
            channels = self.multipole_channels()
            index = {c: i for i, c in enumerate(channels)}
            per_atom: dict = {}
            for atom, L, M in channels:
                per_atom.setdefault(atom, []).append((L, M))
            U = np.zeros((len(channels), len(channels)), dtype=complex)
            for a, levels_a in per_atom.items():
                for b, levels_b in per_atom.items():
                    ra = self.datasets[a].compensation_radius
                    rb = self.datasets[b].compensation_radius
                    displacement = (np.asarray(self._potentials.nuclei[b][1])
                                    - np.asarray(self._potentials.nuclei[a][1]))
                    block = multipole_coulomb_matrix(ra, levels_a, rb,
                                                     levels_b, displacement)
                    for i, la in enumerate(levels_a):
                        for j, lb in enumerate(levels_b):
                            U[index[(a,) + la], index[(b,) + lb]] = block[i, j]
            self._U = U
        return self._U

    def compensation_ionic_at(self, centers) -> dict:
        r"""``{(atom, L, M): int ghat_{A,LM} sum_{B != A} v^ion_B}`` (Hartree)
        for atoms at ``centers`` (Bohr).

        Integrated on an **atom-centered** spherical quadrature, not the grid:
        the shape spans only a few grid points at h = 0.20 Angstrom, so a grid
        sum would be inaccurate and would put an egg-box straight into the
        force.  The on-site term (``B = A``) is left out -- the isolated atom's
        own compensation-ion interaction is already inside the dataset, which
        is calibrated to reproduce the reference atom's energy.

        Taking ``centers`` as an argument is what lets the force re-evaluate it
        at displaced positions (:func:`~mandacaru.algorithms.pseudo_forces._ionic_shift`).
        """
        from .multipoles import shape_function
        from ..basis._angular import spherical_harmonic

        x, wx = np.polynomial.legendre.leggauss(COMPENSATION_RADIAL_POINTS)
        t, wt = np.polynomial.legendre.leggauss(PROJECTION_POLAR_POINTS)
        n_phi = PROJECTION_AZIMUTHAL_POINTS
        azimuth = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
        sin_t = np.sqrt(1.0 - t * t)
        direction = np.stack([
            (sin_t[:, None] * np.cos(azimuth)[None, :]).ravel(),
            (sin_t[:, None] * np.sin(azimuth)[None, :]).ravel(),
            np.repeat(t, n_phi)], axis=1)
        w_ang = np.repeat(wt, n_phi) * (2.0 * np.pi / n_phi)
        theta = np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
        phi = np.arctan2(direction[:, 1], direction[:, 0])

        out = {}
        for atom, L, M in sorted(self.compensation_moments()):
            dataset = self.datasets[atom]
            r_g = float(dataset.compensation_radius)
            radius = 0.5 * r_g * (x + 1.0)
            weight = 0.5 * r_g * wx * radius * radius
            points = (np.asarray(centers[atom], dtype=float)[None, None, :]
                      + radius[:, None, None] * direction[None, :, :])
            shape = shape_function(radius, r_g, L)
            harmonic = spherical_harmonic(int(L), int(M), theta, phi)
            total = 0.0 + 0.0j
            for other, neighbor in enumerate(self.datasets):
                if other == atom:
                    continue
                center = np.asarray(centers[other], dtype=float)
                distance = np.linalg.norm(points - center[None, None, :], axis=2)
                v = neighbor.local_potential(distance)
                total += complex(np.sum(weight[:, None] * w_ang[None, :]
                                        * shape[:, None] * harmonic[None, :] * v))
            out[(atom, L, M)] = total
        return out

    def compensation_ionic(self) -> dict:
        """:meth:`compensation_ionic_at` at the actual nuclear positions."""
        if self._Vion is None:
            centers = [np.asarray(self._potentials.nuclei[a][1], dtype=float)
                       for a in range(len(self.datasets))]
            self._Vion = self.compensation_ionic_at(centers)
        return self._Vion

    def one_body_augmentation(self):
        r"""``sum_{A,LM} Q^{A,LM}_{pr} \int \hat g_{A,LM} v^{ion}``.

        The compensation charge is a real piece of electron density, so it is
        attracted to the other nuclei exactly as the smooth density is.  The
        two-body augmentation already gives it its Hartree *repulsion*; without
        this term it has no attraction at all, which is why the augmentation
        grew by ~8.7 eV on forming an O-H bond and the molecule would not bind.
        """
        moments = self.compensation_moments()
        if not moments:
            return None
        potentials = self.compensation_ionic()
        out = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        for channel, Q in moments.items():
            out = out + Q * potentials[channel]
        return out

    def two_body_augmentation(self):
        Qs = self.compensation_moments()
        if not Qs:
            return None
        Ws = self.compensation_potentials()
        U = self.compensation_coulomb()
        channels = self.multipole_channels()
        M = self.n_orbitals
        aug = np.zeros((M, M, M, M), dtype=complex)
        for i, ca in enumerate(channels):
            Qa, Wa = Qs[ca], Ws[ca]
            aug += np.einsum("pr,qs->pqrs", Qa, Wa)
            aug += np.einsum("pr,qs->pqrs", Wa, Qa)
            for j, cb in enumerate(channels):
                aug += U[i, j] * np.einsum("pr,qs->pqrs", Qa, Qs[cb])
        return aug



def paw_library_path(directory=None) -> str:
    """The PAW-LCAO library directory (``library/paw-lcao`` by default)."""
    from .io import library_root
    if directory is not None:
        return os.fspath(directory)
    return os.path.join(library_root(), LIBRARY_SUBDIR)


_CACHE: dict = {}


def get_paw(symbol: str, directory=None) -> PAWDataset:
    """Load ``symbol`` from the PAW-LCAO library (cached)."""
    from .io import load_library_dataset

    return load_library_dataset(
        symbol, paw_library_path(directory), FAMILY, _CACHE,
        label="PAW-LCAO", noun="dataset",
        repository="mandacaru-paw", link_flag="--link-paw-lcao",
        builder="build_paw_library")


def build_paw_library(elements=("H", "Li", "C", "N", "O", "F"),
                      directory=None, *, verbose: bool = True,
                      format: str | None = None, stride: int | None = None,
                      **generation_options):
    """Generate and save PAW-LCAO datasets for ``elements``; returns the paths."""
    from .io import DEFAULT_FORMAT, STRIDE, library_file, save_pseudopotential

    folder = paw_library_path(directory)
    format = DEFAULT_FORMAT if format is None else format
    stride = STRIDE if stride is None else int(stride)
    written = []
    for symbol in elements:
        pp = generate_paw(symbol, **generation_options)
        path = save_pseudopotential(pp, library_file(symbol, folder, format),
                                    format=format, stride=stride)
        written.append(path)
        if verbose:
            print(f"  {symbol:>2}  Z_ion={pp.valence_charge:>4.0f}  "
                  + "  ".join(f"l{l}: rc={c.r_cut:.2f} q11="
                              f"{c.overlap_correction[0, 0]:+.4f}"
                              for l, c in sorted(pp.channels.items()))
                  + f"  E1c={pp.one_center_energy:+.4f}  -> "
                  f"{os.path.basename(path)}")
    _CACHE.clear()
    return written


def generate_upaw(symbol: str, **options) -> PAWDataset:
    r"""Generate a **unitary** PAW-LCAO (UPAW-LCAO) dataset: the ``norm_deficit = 0`` PAW-LCAO.

    The smooth partial waves carry the full all-electron inner norm, so
    :math:`q_{ij} = 0`, the transformation satisfies :math:`T^\dagger T = I` and
    the overlap operator is the identity -- the construction of Ivanov *et al.*
    (arXiv:2408.03159).  Everything else is :func:`generate_paw`.

    ``norm_deficit`` is not accepted: it *is* what distinguishes the two
    families, and a UPAW-LCAO dataset with a nonzero deficit would be a PAW-LCAO dataset
    wearing the wrong name.
    """
    if "norm_deficit" in options:
        raise TypeError(
            "generate_upaw does not take `norm_deficit`: UPAW-LCAO is defined by "
            "norm_deficit = 0 (use generate_paw for any other value)")
    dataset = generate_paw(symbol, norm_deficit=0.0, **options)
    return replace(dataset, family=UPAW_FAMILY)


def upaw_library_path(directory=None) -> str:
    """The UPAW-LCAO library directory (``library/upaw-lcao`` by default)."""
    from .io import library_root
    if directory is not None:
        return os.fspath(directory)
    return os.path.join(library_root(), UPAW_LIBRARY_SUBDIR)


def get_upaw(symbol: str, directory=None) -> PAWDataset:
    """Load ``symbol`` from the UPAW-LCAO library, **or generate it** (cached).

    Unlike the other families, UPAW-LCAO has no shipped library: there is no
    sibling data repository for it, and requiring a 92-element build before the
    option can be tried at all would make it unusable.  Generation is a few
    seconds per element (H 0.4 s, O 2.2 s, measured) and the result is cached
    for the process, so a small molecule pays that once.  Build a library with
    :func:`build_upaw_library` to skip it, and it is used whenever it exists.
    """
    from .io import library_file, load_pseudopotential

    folder = upaw_library_path(directory)
    key = f"{symbol}@{folder}@{UPAW_FAMILY}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    path = library_file(symbol, folder)
    if os.path.exists(path):
        pp = load_pseudopotential(path)
        family = str(getattr(pp, "family", "")).lower()
        if family != UPAW_FAMILY:
            raise ValueError(f"{path!r} belongs to family {family!r}, not "
                             f"{UPAW_FAMILY!r}")
    else:
        if directory is not None:
            raise FileNotFoundError(
                f"no UPAW-LCAO dataset for {symbol!r} at {path!r}; build one with "
                f"build_upaw_library([{symbol!r}], directory={directory!r})")
        warnings.warn(
            f"generating a UPAW-LCAO dataset for {symbol} (no library at "
            f"{folder!r}); it is cached for this process.  Build one once with "
            f"mandacaru.pseudopotentials.paw.build_upaw_library([...]) to skip "
            f"this.", RuntimeWarning, stacklevel=2)
        pp = generate_upaw(symbol)
    _CACHE[key] = pp
    return pp


def build_upaw_library(elements=("H", "Li", "C", "N", "O", "F"),
                       directory=None, *, verbose: bool = True,
                       format: str | None = None, stride: int | None = None,
                       **generation_options):
    """Generate and save UPAW-LCAO datasets for ``elements``; returns the paths."""
    from .io import DEFAULT_FORMAT, STRIDE, library_file, save_pseudopotential

    folder = upaw_library_path(directory)
    os.makedirs(folder, exist_ok=True)
    format = DEFAULT_FORMAT if format is None else format
    stride = STRIDE if stride is None else int(stride)
    written = []
    for symbol in elements:
        pp = generate_upaw(symbol, **generation_options)
        path = save_pseudopotential(pp, library_file(symbol, folder, format),
                                    format=format, stride=stride)
        written.append(path)
        if verbose:
            print(f"  {symbol:>2}  Z_ion={pp.valence_charge:>4.0f}  "
                  + "  ".join(f"l{l}: rc={c.r_cut:.2f} |q|="
                              f"{abs(c.overlap_correction[0, 0]):.1e}"
                              for l, c in sorted(pp.channels.items()))
                  + f"  E1c={pp.one_center_energy:+.4f}  -> "
                  f"{os.path.basename(path)}")
    _CACHE.clear()
    return written


def build_upaw(atoms, grid, h, charge, spin, options, kinetic=None, **active):
    """Valence-only Hamiltonian from UPAW-LCAO datasets (see :func:`build_paw`).

    The whole molecular path is PAW-LCAO's; only the loader and the recorded
    family name differ.  The name matters because the family registry is
    where per-family option defaults live (``default_options``), so a UPAW-LCAO
    run must resolve *its own* spec, not PAW-LCAO's.
    """
    return build_paw(atoms, grid, h, charge, spin, options, kinetic=kinetic,
                     loader=get_upaw, family=UPAW_FAMILY, **active)


def build_paw(atoms, grid, h, charge, spin, options, kinetic=None,
              loader=None, family=None, **active):
    r"""Valence-only Hamiltonian from PAW-LCAO datasets.

    Same 5-tuple as the other families: the basis is the bound smooth
    partial waves (with the ``size`` hierarchy), the external potential the
    ionic local potential, the nonlocal term :math:`C D^{ion} C^\dagger`, the
    overlap :math:`\tilde S + C q C^\dagger`, the two-body tensor augmented
    by the compensation multipoles (:math:`L = 0 \ldots 2 l_{max}`), and the
    constant the ion-ion
    repulsion plus the frozen one-center energies.
    """
    from .families import build_valence_hamiltonian

    return build_valence_hamiltonian(
        atoms, grid, h, charge, spin, options, kinetic, **active,
        family=FAMILY if family is None else family,
        load=get_paw if loader is None else loader,
        projectors=lambda symbols, positions, datasets, opts: paw_projectors(
            symbols, positions, datasets,
            projector_basis=opts.get("projector_basis",
                                     DEFAULT_PROJECTOR_BASIS)),
        coupling=paw_coupling_blocks, overlap=paw_overlap_blocks,
        spin_orbit=paw_spin_orbit_blocks,
        integrals_class=PAWIntegrals, potentials_keyword="datasets")


# --------------------------------------------------------------------------- #
# On-disk payload (used by io.py for family "paw-lcao").
# --------------------------------------------------------------------------- #

def to_payload(pp: PAWDataset, stride: int = 1) -> dict:
    """JSON-shaped payload; radial tables live under ``"radial_tables"``."""
    from .io import _table

    tables = {"r": _table(pp.r, stride),
              "v_local": _table(pp.v_local, stride),
              "v_local_screened": _table(pp.v_local_screened, stride),
              "valence_density": _table(pp.valence_density, stride),
              "core_density": _table(pp.core_density, stride),
              "smooth_core_density": _table(pp.smooth_core_density, stride)}
    channels = {}
    for l, channel in pp.channels.items():
        key = str(l)
        for i, (ae, ps, p, chi) in enumerate(zip(channel.ae_waves,
                                                 channel.pseudo_waves,
                                                 channel.projectors,
                                                 channel.raw_projectors)):
            tables[f"ae_wave_l{key}_{i}"] = _table(ae, stride)
            tables[f"pseudo_wave_l{key}_{i}"] = _table(ps, stride)
            tables[f"projector_l{key}_{i}"] = _table(p, stride)
            tables[f"raw_projector_l{key}_{i}"] = _table(chi, stride)
        channels[key] = {
            "n": int(channel.n),
            "r_cut": float(channel.r_cut),
            "occupation": float(channel.occupation),
            "reference_energies": [float(e) for e in channel.reference_energies],
            "wavevectors": [np.asarray(q).tolist() for q in channel.wavevectors],
            "wave_coefficients": [np.asarray(c).tolist()
                                  for c in channel.wave_coefficients],
            "coupling": np.asarray(channel.coupling, dtype=float).tolist(),
            "coupling_screened": np.asarray(channel.coupling_screened,
                                            dtype=float).tolist(),
            "overlap_correction": np.asarray(channel.overlap_correction,
                                             dtype=float).tolist(),
            "norm_correction": np.asarray(channel.norm_correction,
                                          dtype=float).tolist(),
            "kinetic_difference": np.asarray(channel.kinetic_difference,
                                             dtype=float).tolist(),
            "potential_difference": np.asarray(channel.potential_difference,
                                               dtype=float).tolist(),
            "vanderbilt": np.asarray(channel.vanderbilt, dtype=float).tolist(),
            "duality_error": float(channel.duality_error),
            "overlap_minimum": float(channel.overlap_minimum),
            "asymmetry": float(channel.asymmetry),
            "consistency_error": float(channel.consistency_error),
            "residual_kinetic": [float(e) for e in channel.residual_kinetic],
            "q_cut": float(channel.q_cut),
        }
    return {"symbol": pp.symbol, "atomic_number": int(pp.atomic_number),
            "valence_charge": float(pp.valence_charge),
            "r_cut_local": float(pp.r_cut_local),
            "local_shift": float(pp.local_shift),
            "compensation_radius": float(pp.compensation_radius),
            "compensation_charge": float(pp.compensation_charge),
            "hartree_screening": float(pp.hartree_screening),
            "one_center_energy": float(pp.one_center_energy),
            "energies": {k: float(v) for k, v in pp.energies.items()},
            "q_cut": float(pp.q_cut), "energy_offset": float(pp.energy_offset),
            "norm_deficit": pp.norm_deficit,
            "xc": str(pp.xc), "relativity": str(pp.relativity),
            "extra_l": int(pp.extra_l), "nlcc": dict(pp.nlcc or {}),
            "spin_orbit": {str(l): np.asarray(D).real.tolist()
                           for l, D in (pp.spin_orbit or {}).items()},
            "channels": channels, "radial_tables": tables}


def from_payload(payload: dict) -> PAWDataset:
    """Rebuild the record written by :func:`to_payload`."""
    tables = payload["radial_tables"]
    r = np.asarray(tables["r"], dtype=float)
    v_local = np.asarray(tables["v_local"], dtype=float)
    v_screened = np.asarray(tables["v_local_screened"], dtype=float)
    channels, projectors = {}, {}
    coupling, coupling_screened, overlap, kinetic = {}, {}, {}, {}
    norm = {}
    for key, entry in payload["channels"].items():
        l = int(key)
        ae, ps, ps_p, raw = [], [], [], []
        i = 0
        while f"pseudo_wave_l{key}_{i}" in tables:
            ae.append(np.asarray(tables[f"ae_wave_l{key}_{i}"], float))
            ps.append(np.asarray(tables[f"pseudo_wave_l{key}_{i}"], float))
            ps_p.append(np.asarray(tables[f"projector_l{key}_{i}"], float))
            raw.append(np.asarray(tables[f"raw_projector_l{key}_{i}"], float))
            i += 1
        D = np.asarray(entry["coupling"], dtype=float)
        D_scr = np.asarray(entry["coupling_screened"], dtype=float)
        q = np.asarray(entry["overlap_correction"], dtype=float)
        # `norm_correction` post-dates the relativistic partial waves, but a
        # file written without it is not lost: D^scr is *defined* as
        # `sym(B + q_norm eps)`, and B and the reference energies are both on
        # disk, so the definition inverts.  The only inexactness is the
        # symmetrization, and the generator refuses to write a channel whose
        # D_raw is asymmetric beyond COUPLING_ASYMMETRY_TOLERANCE -- on the
        # shipped datasets the recovery is good to 6e-11, against the 6e-5 the
        # charge `q` would be wrong by.  Non-relativistically it is exact
        # anyway, since `targets` is unset and q_norm is q by construction.
        if "norm_correction" in entry:
            q_norm = np.asarray(entry["norm_correction"], dtype=float)
        else:
            eps_ref = np.asarray(entry["reference_energies"], dtype=float)
            B_ref = np.asarray(entry["vanderbilt"], dtype=float)
            if np.min(np.abs(eps_ref)) < NORM_RECOVERY_FLOOR:
                raise ValueError(
                    f"the PAW-LCAO dataset for {payload['symbol']} predates "
                    "the stored norm correction and has a reference energy at "
                    "zero, so the correction cannot be recovered from its "
                    "coupling; regenerate it with the current code")
            q_norm = (D_scr - B_ref) / eps_ref[None, :]
            # Recovering the norm exposes something worse in these files.  A
            # dataset written before the split had only *one* matrix, and it
            # was the M-weighted norm -- so the field named
            # `overlap_correction` holds the metric where the **charge**
            # belongs.  That field is not a diagnostic: `compensation_charge`,
            # `multipole_moments` and the ionic screening all read it, so the
            # augmented charge of such a dataset is wrong by the O(c^-2) gap
            # (2.3e-4 for oxygen, 2.1e-4 for nitrogen).  It cannot be repaired
            # on load -- the plain overlap is not a function of anything
            # stored -- so it is reported rather than hidden.  The test is
            # exact, not heuristic: only a pre-split file can have these two
            # agree to machine precision while the atom is relativistic.
            if (str(payload.get("relativity", "none")) != "none"
                    and np.max(np.abs(q - q_norm)) < NORM_SPLIT_FLOOR):
                warnings.warn(
                    f"the PAW-LCAO dataset for {payload['symbol']} stores the "
                    "M-weighted norm in its overlap correction, where the "
                    "charge belongs: it was generated before the two were "
                    "distinguished, so its compensation charge and multipole "
                    "moments are off by order 1e-4 electrons.  Energies from "
                    "it are usable but not converged in that respect; "
                    f"regenerate it with build_paw_library((\"{payload['symbol']}\",)).",
                    RuntimeWarning, stacklevel=2)
        dT = np.asarray(entry["kinetic_difference"], dtype=float)
        channels[l] = PAWChannel(
            l=l, n=int(entry["n"]),
            eigenvalue=float(entry["reference_energies"][0]),
            r_cut=float(entry["r_cut"]),
            coefficients=np.asarray(entry["wave_coefficients"][0], float),
            pseudo_radial=ps[0], v_screened=v_screened, v_ionic=v_local,
            occupation=float(entry["occupation"]),
            norm_error=float(abs(q[0, 0])),
            reference_energies=[float(e) for e in entry["reference_energies"]],
            wavevectors=[np.asarray(v, float) for v in entry["wavevectors"]],
            wave_coefficients=[np.asarray(c, float)
                               for c in entry["wave_coefficients"]],
            ae_waves=ae, pseudo_waves=ps, projectors=ps_p, raw_projectors=raw,
            overlap_correction=q, norm_correction=q_norm,
            kinetic_difference=dT,
            potential_difference=np.asarray(entry["potential_difference"],
                                            dtype=float),
            coupling_screened=D_scr, coupling=D,
            vanderbilt=np.asarray(entry["vanderbilt"], dtype=float),
            duality_error=float(entry.get("duality_error", 0.0)),
            overlap_minimum=float(entry.get("overlap_minimum", 1.0)),
            asymmetry=float(entry.get("asymmetry", 0.0)),
            consistency_error=float(entry.get("consistency_error", 0.0)),
            residual_kinetic=[float(e) for e in entry["residual_kinetic"]],
            q_cut=float(entry.get("q_cut", payload.get("q_cut", DEFAULT_Q_CUT))))
        projectors[l] = ps_p
        coupling[l], coupling_screened[l] = D, D_scr
        overlap[l], kinetic[l], norm[l] = q, dT, q_norm
    return PAWDataset(
        symbol=payload["symbol"], atomic_number=int(payload["atomic_number"]),
        valence_charge=float(payload["valence_charge"]), r=r,
        channels=channels, v_local=v_local, local_l=-1, projectors=projectors,
        kb_energies={},
        valence_density=np.asarray(tables["valence_density"], dtype=float),
        # The layout is shared with UPAW-LCAO, so the family is read, not assumed.
        atom=None, family=str(payload.get("family", FAMILY)), coupling=coupling,
        coupling_screened=coupling_screened, overlap_correction=overlap,
        norm_correction=norm, kinetic_difference=kinetic,
        v_local_screened=v_screened,
        core_density=np.asarray(tables["core_density"], dtype=float),
        smooth_core_density=np.asarray(tables["smooth_core_density"],
                                       dtype=float),
        r_cut_local=float(payload["r_cut_local"]),
        local_shift=float(payload.get("local_shift", 0.0)),
        compensation_radius=float(payload["compensation_radius"]),
        compensation_charge=float(payload["compensation_charge"]),
        hartree_screening=float(payload["hartree_screening"]),
        one_center_energy=float(payload["one_center_energy"]),
        energies={k: float(v) for k, v in payload.get("energies", {}).items()},
        q_cut=float(payload.get("q_cut", DEFAULT_Q_CUT)),
        energy_offset=float(payload.get("energy_offset",
                                        DEFAULT_ENERGY_OFFSET)),
        norm_deficit=payload.get("norm_deficit", DEFAULT_NORM_DEFICIT),
        # A payload without these keys predates them, and a dataset written
        # before relativity was an option is non-relativistic with no core
        # correction.  Defaulting to the *current* defaults would label every
        # file already in the library as something it is not.
        xc=str(payload.get("xc", "lda")),
        relativity=str(payload.get("relativity", "none")),
        extra_l=int(payload.get("extra_l", 0)),
        nlcc=dict(payload.get("nlcc") or {"applied": False, "r_nlcc": None,
                                          "reason": "written before the "
                                                    "core correction"}),
        spin_orbit={int(l): np.asarray(D, dtype=float)
                    for l, D in (payload.get("spin_orbit") or {}).items()})


# --------------------------------------------------------------------------- #
# Registration.
# --------------------------------------------------------------------------- #

#: Both PAW-LCAO families filter their basis by default (see
#: :mod:`mandacaru.basis.filtering`).  A PAW-LCAO smooth partial wave is already
#: *built* to be band-limited -- :func:`~.oncv.optimize_pseudo_waves` minimizes
#: the kinetic energy beyond ``q_cut`` -- so removing what is left above the
#: grid's Nyquist wave-vector costs little and takes most of the egg-box with
#: it; the norm-conserving families keep it opt-in because their orbitals are
#: not optimized that way.  ``basis={"name": "PAW-LCAO", "filter": False}`` restores
#: the unfiltered basis exactly.
#:
#: ``energy_shift = 0.1`` eV makes the default PAW-LCAO basis a
#: **confined** one, GPAW's default recipe; the polarization shell then
#: defaults to GPAW's quasi-Gaussian (:func:`~.confinement.
#: resolve_polarization` -- derived from the confinement, so it is not listed
#: here).  ``{"energy_shift": None}`` restores the free-atom orbitals and, with
#: them, the ``"orbital"`` polarization shell.
PAW_DEFAULT_OPTIONS = {"filter": True, "energy_shift": DEFAULT_ENERGY_SHIFT}

#: What a PAW-LCAO / UPAW-LCAO basis dict may say beyond the family-independent options:
#: which projector set is sampled; the ``energy_shift`` (eV) that confines the
#: first zeta and the ``confinement`` potential's ``(amplitude, r_i / r_c)``
#: (:mod:`~.confinement`; off, and GPAW's, by default); and the
#: ``polarization`` shell, ``"orbital"`` (default) or GPAW's ``"gaussian"``.
PAW_EXTRA_OPTIONS = ("projector_basis", "energy_shift", "confinement",
                     "polarization")


def _register():
    from .families import (COMMON_OPTIONS, FamilySpec, PSEUDO_FAMILIES,
                           register_family)
    PAW_OPTIONS = COMMON_OPTIONS + PAW_EXTRA_OPTIONS
    if FAMILY in PSEUDO_FAMILIES:
        return PSEUDO_FAMILIES[FAMILY]
    return register_family(FamilySpec(
        name=FAMILY,
        description="projector augmented wave (Bloechl 1994), frozen core, "
                    "linearized one-center terms, multipole compensation",
        generate=lambda symbol, **options: generate_paw(symbol, **options),
        get=get_paw,
        build=build_paw,
        norm_conserving=False,
        aliases=(),
        options=PAW_OPTIONS,
        default_options=dict(PAW_DEFAULT_OPTIONS),
    ))


def _register_upaw():
    """Register the **unitary** PAW-LCAO family (``basis="UPAW-LCAO"``).

    Same machinery as PAW-LCAO with :math:`q = 0`, so the transformation is unitary
    and the overlap operator is the identity.  Kept as an option rather than the
    default: Mandacaru orthogonalizes the augmented overlap anyway, the constraint
    fixes only the monopole (the higher compensation multipoles survive), and the
    harder waves cost about five times the grid egg-box on oxygen -- see
    :func:`smooth_partial_waves` for the measurements.
    """
    from .families import (COMMON_OPTIONS, PSEUDO_FAMILIES, FamilySpec,
                           register_family)
    PAW_OPTIONS = COMMON_OPTIONS + PAW_EXTRA_OPTIONS
    if UPAW_FAMILY in PSEUDO_FAMILIES:
        return PSEUDO_FAMILIES[UPAW_FAMILY]
    return register_family(FamilySpec(
        name=UPAW_FAMILY,
        description="unitary projector augmented wave (Ivanov 2024, "
                    "arXiv:2408.03159): PAW-LCAO with q = 0, so T is unitary and "
                    "the pseudo states are orthonormal",
        generate=lambda symbol, **options: generate_upaw(symbol, **options),
        get=get_upaw,
        build=build_upaw,
        norm_conserving=False,
        aliases=("unitary-paw-lcao",),
        options=PAW_OPTIONS,
        # UPAW-LCAO's partial waves are the harder ones (five times water's grid
        # egg-box at h = 0.25), so if the filter earns its place for PAW-LCAO it
        # earns it for UPAW-LCAO a fortiori.
        default_options=dict(PAW_DEFAULT_OPTIONS),
    ))


PAW_FAMILY = _register()
UPAW_FAMILY_SPEC = _register_upaw()
