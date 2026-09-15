# -*- coding: utf-8 -*-
# file: pseudopotentials/paw.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Projector augmented-wave datasets (PAW).

The family ``"paw"`` implements P. E. Blöchl's projector augmented-wave method,
Phys. Rev. B **50**, 17953 (1994), in its frozen-core, one-center-expansion
form, with the one-center energies **linearized around the reference atom**
(a fixed coupling matrix :math:`D^0` per species, the "frozen augmentation"
that makes a PAW dataset behave like an ultrasoft pseudopotential).  Written
from scratch on the same LDA radial atom as the Troullier-Martins and ONCVPSP
families (:mod:`carcara.basis.atomic_solver`), reusing the Numerov partial
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
   **No norm condition is imposed** -- that is the point of PAW; the norm
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
   electrons; the **monopole compensation charge** :math:`\hat n = Q\,g(r)`
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
   Hamiltonian as :attr:`carcara.core.hamiltonian.MolecularIntegrals.constant_energy`.

In a molecule (:func:`build_paw`, :class:`PAWIntegrals`)
-------------------------------------------------------
The basis is the bound smooth partial waves (with the usual size hierarchy),
the external potential :math:`\sum_A \tilde v^{ion}_A`, the nonlocal term
:math:`C D^{ion} C^\dagger` and the overlap :math:`\tilde S + C q C^\dagger`
through the general separable form of
:class:`~carcara.core.hamiltonian.MolecularIntegrals`.  The two-body tensor is
built from **augmented pair densities**
:math:`\rho_{pr} = \tilde\phi_p^*\tilde\phi_r + \sum_A (C q C^\dagger)^A_{pr}\, g_A`
(:meth:`PAWIntegrals.two_body_augmentation`), so Hartree, exchange and
correlation all see neutral atoms; the compensation-compensation Coulomb
integrals are evaluated radially (exactly :math:`1/R_{AB}` for disjoint
spheres).

What is frozen or omitted relative to Blöchl's full method (see the guide):
the one-center Hartree and xc terms are linearized at the reference (fixed
:math:`D^0`, no self-consistent :math:`D_{ij}[\rho_{ij}]`); only the
:math:`l = 0` compensation moment is built (no higher multipoles); the core
is frozen with no nonlinear core correction (the core-valence xc of the
reference atom stays in :math:`\tilde v^{ion}` / :math:`D^{ion}`, the smooth
core density is stored but not used); LDA only; no relativistic terms; no
projectors above the valence :math:`l`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import simpson

from ..basis.atomic_solver import (AtomicResult, hartree_potential, lda_xc,
                                    solve_atom)
from ..core.hamiltonian import MolecularIntegrals, projector_blocks
from .generation import Channel, PseudoPotential, _valence_configuration
from .oncv import (Q_MAX, Q_STEP, PseudoWaves, _bessel_table,
                   _bessel_transform_table, _inner_grid, _log_derivative_of_u,
                   _radial_f, _resample, _snap, _spectrum_extent,
                   _tail_transform, _with_origin, bessel_derivatives,
                   bessel_wavevectors, bound_state, generation_points,
                   log_derivative_ae, matching_targets, numerov_outward,
                   optimize_pseudo_waves, polynomial_local_potential,
                   scattering_wave)

#: Registry name of the family (no aliases).
FAMILY = "paw"
#: Subdirectory of the pseudopotential library holding the PAW datasets.
LIBRARY_SUBDIR = "paw"

#: Spherical Bessel functions per smooth partial wave.
DEFAULT_N_BESSEL = 8
#: Wave-vector cutoff of the residual kinetic energy (Bohr^-1).
DEFAULT_Q_CUT = 5.0
#: Second reference energy above the bound state (Hartree).
DEFAULT_ENERGY_OFFSET = 1.0
#: Fraction of the all-electron inner norm the smooth partial waves give up:
#: ``<phi~_i|phi~_j>_rc = (1 - DEFAULT_NORM_DEFICIT) <phi_i|phi_j>_rc``, so the
#: overlap correction ``q = deficit * <phi_i|phi_j>_rc`` is positive definite
#: and the PAW overlap operator ``1 + sum |p> q <p|`` is bounded below by 1.
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
#: near-singular ONCVPSP coupling, the PAW projector term does not push it
#: away, so the local potential is raised until the s spectrum has nothing
#: between the bound state and the box states.
DEFAULT_LOCAL_SHIFTS = {"C": 12.0, "N": 10.0, "O": 6.0, "F": 8.0}
#: Largest tolerated deviation of ``<p_i|phi_j>`` from the identity.
DUALITY_TOLERANCE = 1e-8
#: Largest tolerated asymmetry of the screened coupling matrix (Hartree).
COUPLING_ASYMMETRY_TOLERANCE = 1e-4
#: Smallest tolerated eigenvalue of the overlap operator ``1 + sum |p> q <p|``
#: restricted to the channel (``1 + lambda_min(q G_p)``, ``G_p`` the projector
#: Gram matrix): below it the PAW transformation is (nearly) singular.
OVERLAP_MINIMUM = 0.1
#: Points of the radial quadrature grids used for the compensation charge.
COMPENSATION_POINTS = 2001
#: Projector basis sampled on the molecular grid (:meth:`PAWDataset.projector_set`).
DEFAULT_PROJECTOR_BASIS = "raw"


# --------------------------------------------------------------------------- #
# Smooth partial waves: matched, no norm condition, minimal residual energy.
# --------------------------------------------------------------------------- #

def smooth_partial_waves(r: np.ndarray, v_ae: np.ndarray, l: int, waves: list,
                         energies: list, r_cut: float,
                         q_cut: float = DEFAULT_Q_CUT,
                         n_bessel: int = DEFAULT_N_BESSEL,
                         norm_deficit=DEFAULT_NORM_DEFICIT) -> PseudoWaves:
    r"""Bessel expansions of the smooth partial waves of one channel.

    Each wave is matched in value and first three derivatives at ``r_cut``
    and its residual kinetic energy beyond ``q_cut`` is minimized in the
    remaining freedom.  **The norm is not conserved.**  With a
    ``norm_deficit`` :math:`s` (the default, :data:`DEFAULT_NORM_DEFICIT`)
    the inner-norm matrix is set to :math:`(1-s)` times the all-electron one
    (:func:`~.oncv.optimize_pseudo_waves` with ``norm_factor = 1 - s``), so
    the overlap correction :math:`q_{ij} = s\,\langle\varphi_i|\varphi_j
    \rangle_{r<r_c}` is positive definite and the PAW overlap operator is
    bounded below by one -- a dataset built with free (unconstrained) waves,
    ``norm_deficit=None``, has no such guarantee: the two reference waves
    are nearly proportional in the core, their dual projectors are large,
    and an indefinite :math:`q` makes :math:`1 + \sum|\tilde p\rangle q
    \langle\tilde p|` singular (Li, O at every cutoff tried).  The returned
    :attr:`~.oncv.PseudoWaves.achieved` inner norms differ from the
    all-electron :attr:`~.oncv.PseudoWaves.norms` by exactly :math:`q`.
    """
    if norm_deficit is not None:
        return optimize_pseudo_waves(r, v_ae, l, waves, energies, r_cut,
                                     q_cut=q_cut, n_bessel=n_bessel,
                                     norm_factor=1.0 - float(norm_deficit))
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
        target = matching_targets(r, wave * r, v_ae, l, energy, r_cut)
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

    achieved = np.array([[simpson(a * b * r_in * r_in, x=r_in) for b in pseudo_in]
                         for a in pseudo_in])
    return PseudoWaves(l=l, r_cut=float(r_cut), energies=list(energies),
                       waves=list(waves), wavevectors=wavevectors,
                       coefficients=coefficients, residual_kinetic=residuals,
                       norms=norms, achieved=achieved, q_cut=float(q_cut))


# --------------------------------------------------------------------------- #
# One channel: projectors, overlap correction, one-center matrices.
# --------------------------------------------------------------------------- #

@dataclass
class PAWChannel(Channel):
    """One PAW channel: the TM :class:`Channel` plus the partial-wave sets.

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
    overlap_correction: np.ndarray = None                    # q_ij
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
            f"PAW projectors of l={l} are not dual to the smooth partial "
            f"waves (error {duality_error:.2e}); B is ill-conditioned")

    q = pw.norms - pw.achieved
    # The overlap operator 1 + sum |p_i> q_ij <p_j| restricted to the channel
    # has the eigenvalues 1 + eig(q G_p); it must stay positive definite.
    gram = np.array([[inner(a, b) for b in projectors_in] for a in projectors_in])
    overlap_minimum = float(1.0 + np.linalg.eigvals(q @ gram).real.min())
    if strict and overlap_minimum < OVERLAP_MINIMUM:
        raise RuntimeError(
            f"the PAW overlap operator of l={l} has an eigenvalue "
            f"{overlap_minimum:.3f} (the smooth partial waves carry too little "
            "or too much norm for their projectors); change r_cut, n_bessel "
            "or energy_offset")
    # Kinetic energies inside r_c: all-electron from the radial equation
    # T phi_j = (eps_j - v_AE) phi_j, smooth side analytic.
    T_ae = np.array([[inner(ae_in[i], (energies[j] - v_ae_in) * ae_in[j])
                      for j in range(n_waves)] for i in range(n_waves)])
    T_ps = np.array([[inner(pseudo_in[i], kin_in[j]) for j in range(n_waves)]
                     for i in range(n_waves)])
    dT = T_ae - T_ps
    dT = 0.5 * (dT + dT.T)
    dV = np.array([[inner(ae_in[i], v_ae_in * ae_in[j])
                    - inner(pseudo_in[i], v_loc_in * pseudo_in[j])
                    for j in range(n_waves)] for i in range(n_waves)])
    dV = 0.5 * (dV + dV.T)
    D_raw = B + q * energies[None, :]
    asymmetry = float(np.max(np.abs(D_raw - D_raw.T)))
    if strict and asymmetry > COUPLING_ASYMMETRY_TOLERANCE:
        raise RuntimeError(
            f"the screened PAW coupling of l={l} is asymmetric by "
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
        overlap_correction=q, kinetic_difference=dT,
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


def _xc_energies(r, rho):
    """``(E_xc[rho], int rho v_xc[rho])``."""
    e_xc, v_xc = lda_xc(rho)
    shell = 4.0 * np.pi * r * r
    return (float(np.trapezoid(e_xc * rho * shell, r)),
            float(np.trapezoid(v_xc * rho * shell, r)))


# --------------------------------------------------------------------------- #
# The dataset record.
# --------------------------------------------------------------------------- #

@dataclass
class PAWDataset(PseudoPotential):
    r"""A PAW dataset: local potential, projectors, one-center matrices.

    Inherits the :class:`~.generation.PseudoPotential` layout so the valence
    basis (:func:`~.orbitals.pseudo_basis`, first zeta = the bound smooth
    partial wave), the local-potential sampler (:meth:`local_potential`,
    the **ionic** :math:`\tilde v^{ion}`) and the multiple-zeta hierarchy work
    unchanged; every ``channels[l]`` is a :class:`PAWChannel`.  ``projectors``
    maps ``l -> [p~_1(r), p~_2(r)]``, ``coupling`` maps ``l -> D^ion``, and
    ``overlap_correction`` maps ``l -> q`` -- the two block matrices the
    molecular path hands to :class:`~carcara.core.hamiltonian.MolecularIntegrals`.
    """

    coupling: dict = field(default_factory=dict)             # l -> D^ion (n, n)
    coupling_screened: dict = field(default_factory=dict)    # l -> D^scr
    overlap_correction: dict = field(default_factory=dict)   # l -> q
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

    @property
    def nonlocal_channels(self) -> list:
        return sorted(self.projectors)

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

        ``"dual"``: the PAW projectors :math:`\tilde p_i` with the blocks
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

    def __repr__(self) -> str:
        channels = ", ".join(f"l={l}x{len(self.projectors.get(l, []))}"
                             for l in sorted(self.channels))
        return (f"PAWDataset({self.symbol}, Z_ion={self.valence_charge:g}, "
                f"[{channels}], rg={self.compensation_radius:.2f}, "
                f"Q^={self.compensation_charge:+.4f}, "
                f"E1c={self.one_center_energy:+.4f} Ha)")


# --------------------------------------------------------------------------- #
# Generation.
# --------------------------------------------------------------------------- #

def _cutoff_for(symbol, l, r, radial, r_cut, rc_factor):
    if isinstance(r_cut, dict):
        return float(r_cut[l])
    if r_cut is not None:
        return float(r_cut)
    table = DEFAULT_CUTOFFS.get(symbol)
    if table is not None and l in table:
        return float(table[l])
    peak = r[int(np.argmax(np.abs(radial * r)))]
    return float(rc_factor * peak)


def generate_paw(symbol: str, *, r_cut=None, rc_factor: float = DEFAULT_RC_FACTOR,
                 r_cut_local: float | None = None,
                 local_factor: float = DEFAULT_LOCAL_FACTOR,
                 local_shift: float | None = None,
                 q_cut: float = DEFAULT_Q_CUT,
                 energy_offset: float | None = None,
                 n_bessel: int = DEFAULT_N_BESSEL,
                 norm_deficit="default",
                 points: int | None = None, r_max: float = 30.0,
                 atom: AtomicResult | None = None) -> PAWDataset:
    r"""Generate a PAW dataset for ``symbol`` (see the module docstring).

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
    q_cut, n_bessel, points, r_max, atom
        As in :func:`~.oncv.generate_oncv`.
    """
    from ase.data import atomic_numbers

    atomic_number = int(atomic_numbers[symbol])
    if atom is None:
        atom = solve_atom(atomic_number,
                          points=(generation_points(atomic_number)
                                  if points is None else int(points)),
                          r_max=r_max, tolerance=1e-7, mixing=0.25)
    valence_config, core_config = _valence_configuration(atomic_number)
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

    per_l: dict = {}
    for (n, l), occupancy in sorted(valence_config.items()):
        u, energy = bound_state(r, v_ae, l, atom.eigenvalues[(n, l)], z_eff)
        per_l.setdefault(l, []).append((n, energy, u / r, occupancy))

    cutoffs = {l: _snap(r, _cutoff_for(symbol, l, r, states[0][2], r_cut,
                                        rc_factor))
               for l, states in per_l.items()}
    r_local = _snap(r, float(r_cut_local) if r_cut_local is not None
                    else float(local_factor * min(cutoffs.values())))
    r_g = float(min(cutoffs.values()))

    waves: dict = {}
    for l, states in per_l.items():
        ae = [w for _n, _e, w, _o in states]
        energies = [e for _n, e, _w, _o in states]
        if len(states) == 1:
            energy_2 = energies[0] + float(energy_offset)
            u2 = scattering_wave(r, v_ae, l, energy_2, z_eff)
            inside = r <= cutoffs[l]
            u2 = u2 / np.sqrt(np.trapezoid(u2[inside] ** 2, r[inside]))
            ae.append(u2 / r)
            energies.append(energy_2)
        waves[l] = smooth_partial_waves(r, v_ae, l, ae[:2], energies[:2],
                                        cutoffs[l], q_cut=q_cut,
                                        n_bessel=n_bessel,
                                        norm_deficit=norm_deficit)

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

    # Unscreening: Hartree of the neutral smooth density, LDA xc of the
    # smooth valence density (as the norm-conserving families do).
    v_hartree = hartree_potential(r, augmented)
    _e_xc, v_xc = lda_xc(smooth_valence)
    v_local_ionic = v_loc - v_hartree - v_xc
    hartree_screening = float(np.trapezoid(v_hartree * g * shell, r))
    for channel in channels.values():
        channel.v_ionic = v_local_ionic
        channel.coupling = (channel.coupling_screened
                            - channel.overlap_correction * hartree_screening)

    # Frozen one-center constant: the reference atom evaluated with the
    # molecular machinery must give the all-electron valence energy in the
    # norm-conserving convention.
    e_h_ae = _hartree_energy(r, ae_valence)
    e_h_ps = _hartree_energy(r, augmented)
    e_xc_ae, v_xc_ae = _xc_energies(r, ae_valence)
    e_xc_ps, v_xc_ps = _xc_energies(r, smooth_valence)
    reference_valence = band - e_h_ae - v_xc_ae + e_xc_ae
    pseudo_atom = band - e_h_ps - v_xc_ps + e_xc_ps
    one_center = reference_valence - pseudo_atom
    e_xc_full_ae, _v = _xc_energies(r, ae_valence + core_density)
    e_xc_full_ps, _v = _xc_energies(r, smooth_valence + smooth_core)
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
        kinetic_difference={l: np.array(c.kinetic_difference)
                            for l, c in channels.items()},
        v_local_screened=v_loc, core_density=core_density,
        smooth_core_density=smooth_core, r_cut_local=r_local,
        local_shift=shift, compensation_radius=r_g, compensation_charge=float(compensation_charge),
        hartree_screening=hartree_screening, one_center_energy=float(one_center),
        energies=energies, q_cut=float(q_cut),
        energy_offset=float(energy_offset),
        norm_deficit=None if norm_deficit is None else float(norm_deficit))


# --------------------------------------------------------------------------- #
# Diagnostics: generalized spectrum, reconstruction, logarithmic derivatives.
# --------------------------------------------------------------------------- #

#: Spacing (Bohr) of the uniform grid the radial spectrum is solved on.
SPECTRUM_SPACING = 0.005


def _channel_operator(pp: PAWDataset, l: int, r_max: float, stride: int):
    """Grid, local potential, ``u``-form projectors, ``D^scr`` and ``q``."""
    from scipy.interpolate import CubicSpline

    h = stride * SPECTRUM_SPACING
    r = np.arange(1, int(r_max / h) + 1) * h
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
         - float(energy) * np.asarray(pp.overlap_correction[l], dtype=float))
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
    ae = pp.atom
    if ae is None:
        return out
    z = float(pp.atomic_number)
    probes = list(energies)
    if midpoint:
        probes.append(0.5 * (energies[0] + energies[1]))
    errors = {}
    for energy in probes:
        l_ae = log_derivative_ae(r, ae.v_effective, l, energy, channel.r_cut, z)
        l_ps = log_derivative_paw(pp, l, energy)
        errors[float(energy)] = (float(abs(l_ps - l_ae)), float(l_ae))
    out["log_derivative_errors"] = errors
    return out


def report_paw(pp: PAWDataset) -> str:
    """Human-readable validation summary for every channel."""
    lines = [f"{pp!r}",
             f"  valence charge  : {pp.valence_charge:g}",
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


def paw_coupling_blocks(projectors, symbols, datasets) -> dict:
    """``{(atom, l, m): D_l}`` for ``nonlocal_coupling`` (in the
    projectors' basis)."""
    return _blocks(projectors, symbols, datasets, "coupling")


def paw_overlap_blocks(projectors, symbols, datasets) -> dict:
    """``{(atom, l, m): q_l}`` for ``nonlocal_overlap`` (in the
    projectors' basis)."""
    return _blocks(projectors, symbols, datasets, "overlap")


class PAWIntegrals(MolecularIntegrals):
    r"""Molecular integrals with PAW compensation charges.

    A :class:`~carcara.core.hamiltonian.MolecularIntegrals` whose
    two-body tensor is built from the augmented pair densities
    :math:`\rho_{pr} = \tilde\phi_p^*\tilde\phi_r + \sum_A Q^A_{pr}\,g_A`,
    :math:`Q^A = (C q C^\dagger)^A` being the monopole augmentation
    moments of atom ``A`` (the same blocks that augment the overlap):

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

    def projections(self) -> np.ndarray:
        r"""``C[mu, p] = <phi_mu|p_p>`` with the **on-site** entries exact.

        The grid quadrature of the parent class is kept for projections of
        a basis function on the projectors of *another* atom, but a basis
        function and a projector on the **same** center share the spherical
        harmonic, so their projection is the one-dimensional radial integral
        :math:`\delta_{ll'}\delta_{mm'}\int R_\mu(r)\,p(r)\,r^2 dr`, evaluated
        here on a fine radial grid.  This matters: the dual projectors are
        sharp (the two reference waves are nearly parallel in the core, so
        :math:`B^{-1}` is large) and the grid value of
        :math:`\langle\tilde\varphi_1|\tilde p_i\rangle`, exactly
        :math:`\delta_{i1}`, came out anywhere between 0.7 and 3 times that
        on 0.25-0.30 Angstrom grids depending on where the nucleus sat between
        nodes -- an error the overlap correction turns into a 0.2-0.6 Ha
        shift of LiH.  The off-site projections are small and enter only
        through :math:`q` (never through :math:`B^{-1}`), so the grid is
        adequate for them.  The resolution ratios still report the grid.
        """
        if self._C is not None:
            return self._C
        C = np.array(super().projections(), dtype=complex)
        for p, projector in enumerate(self.kb_projectors):
            r = np.linspace(0.0, float(projector.r_cut), COMPENSATION_POINTS)
            radial_p = np.asarray(projector.radial(r), dtype=float)
            for mu, fn in enumerate(self.basis):
                center = getattr(fn, "center", None)
                if center is None or np.linalg.norm(
                        np.asarray(center, dtype=float) - projector.center) > 1e-8:
                    continue
                if (int(getattr(fn, "l", -1)), int(getattr(fn, "m", 0))) != \
                        (projector.l, projector.m):
                    C[mu, p] = 0.0
                    continue
                radial_mu = np.asarray(fn.radial(r), dtype=float)
                C[mu, p] = simpson(radial_mu * radial_p * r * r, x=r)
        self._C = C
        return C

    def _atom_positions(self):
        groups = projector_blocks(self.kb_projectors)
        per_atom: dict = {}
        for (atom, _l, _m), positions in groups.items():
            per_atom.setdefault(atom, []).extend(positions)
        return per_atom

    def compensation_moments(self) -> dict:
        r"""``{atom: Q^A}`` -- the ``(M, M)`` monopole augmentation
        moments :math:`Q^A_{pr} = \sum_{ij\in A} C_{pi} q_{ij} C^*_{rj}`."""
        if self._Q is None:
            C = self.projections()
            Q = self.nonlocal_overlap_matrix()
            self._Q = {}
            if Q is not None:
                for atom, positions in self._atom_positions().items():
                    idx = np.asarray(positions)
                    Ca = C[:, idx]
                    self._Q[atom] = Ca @ Q[np.ix_(idx, idx)] @ Ca.conj().T
        return self._Q

    def compensation_potentials(self) -> dict:
        r"""``{atom: W^A}`` -- :math:`W^A_{qs} = \int\tilde\phi_q^*
        \tilde\phi_s\,V_{g_A}\,d^3r` on the grid."""
        if self._W is None:
            psi = self._engine._psi
            X, Y, Z = (self.grid.X.ravel(), self.grid.Y.ravel(),
                       self.grid.Z.ravel())
            self._W = {}
            for atom in self.compensation_moments():
                _z, center = self._potentials.nuclei[atom]
                radius = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2
                                 + (Z - center[2]) ** 2)
                v = self.datasets[atom].compensation_potential(radius)
                self._W[atom] = (np.conj(psi) * v) @ psi.T * self.grid.dV
        return self._W

    def compensation_coulomb(self) -> np.ndarray:
        """``U[A, B]`` -- Coulomb energy of unit compensation charges."""
        if self._U is None:
            atoms = sorted(self.compensation_moments())
            n = len(self.datasets)
            U = np.zeros((n, n))
            for a in atoms:
                for b in atoms:
                    ra = self.datasets[a].compensation_radius
                    rb = self.datasets[b].compensation_radius
                    d = np.linalg.norm(self._potentials.nuclei[a][1]
                                       - self._potentials.nuclei[b][1])
                    U[a, b] = compensation_coulomb(ra, rb, d)
            self._U = U
        return self._U

    def two_body_augmentation(self):
        Qs = self.compensation_moments()
        if not Qs:
            return None
        Ws = self.compensation_potentials()
        U = self.compensation_coulomb()
        M = self.n_orbitals
        aug = np.zeros((M, M, M, M), dtype=complex)
        for a, Qa in Qs.items():
            Wa = Ws[a]
            aug += np.einsum("pr,qs->pqrs", Qa, Wa)
            aug += np.einsum("pr,qs->pqrs", Wa, Qa)
            for b, Qb in Qs.items():
                aug += U[a, b] * np.einsum("pr,qs->pqrs", Qa, Qb)
        return aug



def paw_library_path(directory=None) -> str:
    """The PAW library directory (``library/paw`` by default)."""
    from .io import library_root
    if directory is not None:
        return os.fspath(directory)
    return os.path.join(library_root(), LIBRARY_SUBDIR)


_CACHE: dict = {}


def get_paw(symbol: str, directory=None) -> PAWDataset:
    """Load ``symbol`` from the PAW library (cached)."""
    from .io import available_elements, library_file, load_pseudopotential

    folder = paw_library_path(directory)
    key = f"{symbol}@{folder}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    path = library_file(symbol, folder)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no PAW dataset for {symbol!r} at {path!r}. Available: "
            f"{', '.join(available_elements(folder)) or '(none)'}. "
            "Generate it with build_paw_library([symbol]).")
    pp = load_pseudopotential(path)
    if str(getattr(pp, "family", "")).lower() != FAMILY:
        raise ValueError(f"{path!r} belongs to family {pp.family!r}, not "
                         f"{FAMILY!r}")
    _CACHE[key] = pp
    return pp


def build_paw_library(elements=("H", "Li", "C", "N", "O", "F"),
                      directory=None, *, verbose: bool = True,
                      format: str | None = None, stride: int | None = None,
                      **generation_options):
    """Generate and save PAW datasets for ``elements``; returns the paths."""
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


def build_paw(atoms, grid, h, charge, spin, options, kinetic=None):
    r"""Valence-only Hamiltonian from PAW datasets.

    Same 5-tuple as the other families: the basis is the bound smooth
    partial waves (with the ``size`` hierarchy), the external potential the
    ionic local potential, the nonlocal term :math:`C D^{ion} C^\dagger`, the
    overlap :math:`\tilde S + C q C^\dagger`, the two-body tensor augmented
    by the monopole compensation charges, and the constant the ion-ion
    repulsion plus the frozen one-center energies.
    """
    from ..algorithms._hamiltonian_from_atoms import (
        DEFAULT_KINETIC, _num_particles, _warn_unresolved, coherent_positions,
        grid_from_cell, resolve_num_unpaired)
    from .orbitals import pseudo_basis, valence_electrons

    directory = options.get("directory")
    symbols = atoms.get_chemical_symbols()
    positions = coherent_positions(atoms)
    datasets = {symbol: get_paw(symbol, directory) for symbol in set(symbols)}

    basis_fns, atom_of_orbital = pseudo_basis(
        symbols, positions, datasets, size=options.get("size", "SZ"),
        split_norm=options.get("split_norm"))
    projectors = paw_projectors(
        symbols, positions, datasets,
        projector_basis=options.get("projector_basis", DEFAULT_PROJECTOR_BASIS))
    coupling = paw_coupling_blocks(projectors, symbols, datasets)
    overlap = paw_overlap_blocks(projectors, symbols, datasets)
    nuclei = [(datasets[symbol].valence_charge, position)
              for symbol, position in zip(symbols, positions)]

    n_el = int(round(valence_electrons(symbols, datasets))) - int(charge)
    g = (grid if grid is not None
         else grid_from_cell(atoms, h, center=positions.mean(axis=0)))
    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    num_particles = _num_particles(n_el, n_unpaired, FAMILY.upper())
    integrals = PAWIntegrals(
        nuclei, basis_fns, g, softening=0.0,
        datasets=[datasets[s] for s in symbols],
        kb_projectors=projectors, nonlocal_coupling=coupling,
        nonlocal_overlap=overlap,
        kinetic=kinetic or DEFAULT_KINETIC["pseudopotentials"])
    hamiltonian = integrals.molecular_hamiltonian(mo_basis=True,
                                                  n_electrons=n_el,
                                                  num_particles=num_particles)
    _warn_unresolved(integrals, basis_fns, h)

    context = {"integrals": integrals, "atom_of_orbital": atom_of_orbital,
               "frozen": (), "n_electrons": n_el,
               "pseudopotentials": datasets, "kb_projectors": projectors,
               "nonlocal_coupling": coupling, "nonlocal_overlap": overlap,
               "family": FAMILY}
    return (hamiltonian, num_particles, len(basis_fns),
            integrals.integration_profile(), context)


# --------------------------------------------------------------------------- #
# On-disk payload (used by io.py for family "paw").
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
            "channels": channels, "radial_tables": tables}


def from_payload(payload: dict) -> PAWDataset:
    """Rebuild the record written by :func:`to_payload`."""
    tables = payload["radial_tables"]
    r = np.asarray(tables["r"], dtype=float)
    v_local = np.asarray(tables["v_local"], dtype=float)
    v_screened = np.asarray(tables["v_local_screened"], dtype=float)
    channels, projectors = {}, {}
    coupling, coupling_screened, overlap, kinetic = {}, {}, {}, {}
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
            overlap_correction=q, kinetic_difference=dT,
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
        overlap[l], kinetic[l] = q, dT
    return PAWDataset(
        symbol=payload["symbol"], atomic_number=int(payload["atomic_number"]),
        valence_charge=float(payload["valence_charge"]), r=r,
        channels=channels, v_local=v_local, local_l=-1, projectors=projectors,
        kb_energies={},
        valence_density=np.asarray(tables["valence_density"], dtype=float),
        atom=None, family=FAMILY, coupling=coupling,
        coupling_screened=coupling_screened, overlap_correction=overlap,
        kinetic_difference=kinetic, v_local_screened=v_screened,
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
        norm_deficit=payload.get("norm_deficit", DEFAULT_NORM_DEFICIT))


# --------------------------------------------------------------------------- #
# Registration.
# --------------------------------------------------------------------------- #

def _register():
    from .families import (COMMON_OPTIONS, FamilySpec, PSEUDO_FAMILIES,
                           register_family)
    if FAMILY in PSEUDO_FAMILIES:
        return PSEUDO_FAMILIES[FAMILY]
    return register_family(FamilySpec(
        name=FAMILY,
        description="projector augmented wave (Bloechl 1994), frozen core, "
                    "linearized one-center terms, monopole compensation",
        generate=lambda symbol, **options: generate_paw(symbol, **options),
        get=get_paw,
        build=build_paw,
        norm_conserving=False,
        aliases=(),
        options=COMMON_OPTIONS + ("projector_basis",),
    ))


PAW_FAMILY = _register()
