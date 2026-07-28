# -*- coding: utf-8 -*-
# file: algorithms/forces.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Nuclear gradients: Hellmann-Feynman and Pulay forces.

The force on nucleus :math:`A` is :math:`\mathbf F_A = -\,dE/d\mathbf R_A`.  At a
converged variational minimum the *parameter* response vanishes
(:math:`\partial E/\partial\vec\theta = 0`), so the reduced density matrices can
be held fixed and only the **integrals** have to be differentiated:

.. math::

    \frac{dE}{d\mathbf R_A}
      = \underbrace{\sum_{\mu\nu} D_{\mu\nu}
          \Big\langle \mu \Big| \frac{\partial \hat V_{ne}}{\partial \mathbf R_A}
          \Big| \nu \Big\rangle
        + \frac{\partial V_{nn}}{\partial \mathbf R_A}}_{\text{Hellmann-Feynman}}
      \; + \; \underbrace{\Big\langle \frac{\partial E}{\partial S},
                                      \frac{\partial S}{\partial \mathbf R_A}\Big\rangle
        + \Big\langle \frac{\partial E}{\partial h},
                      \frac{\partial h}{\partial \mathbf R_A}\Big\rangle_{\text{basis}}
        + \Big\langle \frac{\partial E}{\partial g},
                      \frac{\partial g}{\partial \mathbf R_A}\Big\rangle}_{\text{Pulay}} .

Why Pulay terms are **not** optional here
-----------------------------------------
Carcará's orbitals are atom-centered: when a nucleus moves, its basis functions
move with it, so the basis itself depends on :math:`\mathbf R`.  The
Hellmann-Feynman theorem assumes a fixed basis and is therefore *incomplete*.
The size of the omission is not academic -- for H\ :sub:`2` in the FAO basis the
Hellmann-Feynman force is only ~40 % of the true gradient and never crosses zero,
so a relaxation driven by it alone would not find an equilibrium at all.  Both
contributions are computed here, and :func:`nuclear_gradient` returns them
separately so the split can be inspected.

How each piece is obtained
--------------------------
* **Orbital derivatives.** For an atom-centered function
  :math:`\phi_\mu(\mathbf r) = f(\mathbf r - \mathbf R_A)`, moving the center is
  the same as shifting the sampling point:
  :math:`\partial\phi_\mu/\partial R_{A,k}` is evaluated by central differences of
  the analytic orbital on a displaced grid.  No integral is recomputed.
* **Integral derivatives.** :math:`\partial S`, :math:`\partial T` and
  :math:`\partial V` follow from one call to the *same* one-body kernel on the
  stacked ``[phi; dphi]`` basis, so the finite-difference Laplacian matches the
  one the energy used.  The two-body derivative never builds a
  :math:`\partial g` tensor: it is contracted on the fly against the Coulomb
  potentials of the pair densities (see :func:`_two_body_pulay`).
* **Algebraic response.** :math:`\partial E/\partial S`, :math:`\partial E/\partial h`
  and :math:`\partial E/\partial g` come from automatic differentiation of the
  Löwdin / SCF / transform stack -- see :mod:`carcara.algorithms._jax_energy` for
  why AD is the right tool for that layer and not for the grid kernels.

Accuracy caveat
---------------
Holding the RDMs fixed is exact when the wavefunction is variationally complete
in its orbital space (full CI).  With a **frozen core** or a truncated ansatz an
orbital-relaxation (coupled-perturbed) term is neglected;
:func:`check_forces_against_finite_difference` measures the residual for a given
system so the approximation can be quantified rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..integrals.poisson import PoissonFFTSolver
from ..units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, HARTREE_TO_EV

#: Central-difference step for the orbital derivatives, in Bohr.
DEFAULT_ORBITAL_DELTA = 1e-3

#: Ways of evaluating the electron-nucleus term of the Hellmann-Feynman force.
HELLMANN_FEYNMAN_FORMS = ("analytic", "by-parts")

#: Convert a gradient in Hartree/Bohr to ASE's eV/Angstrom.
HA_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


@dataclass
class ForceResult:
    """Nuclear gradient of a converged variational calculation.

    All arrays are ``(n_atoms, 3)``.  :attr:`forces` is what ASE consumes
    (**eV/Angstrom**, already negated: it is :math:`-dE/d\\mathbf R`); the
    component breakdown is in Hartree/Bohr *gradient* convention
    (:math:`+dE/d\\mathbf R`) so the physics can be inspected term by term.
    """

    forces: np.ndarray                                  # eV/Angstrom
    hellmann_feynman: np.ndarray                        # dE/dR, Ha/Bohr
    pulay: np.ndarray                                   # dE/dR, Ha/Bohr
    gradient: np.ndarray                                # total dE/dR, Ha/Bohr
    n_electrons: float = 0.0                            # tr(gamma), a sanity check
    details: dict = field(default_factory=dict)

    @property
    def max_force(self) -> float:
        """Largest force magnitude on any atom (eV/Angstrom)."""
        return float(np.max(np.linalg.norm(self.forces, axis=1)))

    @property
    def pulay_fraction(self) -> float:
        """‖Pulay‖ / ‖total gradient‖ -- how badly Hellmann-Feynman alone fails."""
        total = float(np.linalg.norm(self.gradient))
        return float(np.linalg.norm(self.pulay) / total) if total > 0 else 0.0

    def __repr__(self) -> str:
        return (f"ForceResult(max|F| = {self.max_force:.4f} eV/A, "
                f"Pulay fraction = {self.pulay_fraction:.2f})")


# --------------------------------------------------------------------------- #
# Orbital derivatives on the grid.
# --------------------------------------------------------------------------- #

def orbital_derivatives(basis, grid, atom_of_orbital, atom, delta=None):
    r"""``d phi_mu / d R_{A,k}`` sampled on the grid, for one atom.

    Returns a ``(3, M, ngrid)`` array; row ``k`` is the derivative with respect to
    Cartesian direction ``k``.  Rows for orbitals *not* centered on ``atom`` are
    zero, since only that atom's functions move.

    Because :math:`\phi_\mu(\mathbf r) = f(\mathbf r - \mathbf R_A)`, displacing
    the center by :math:`+\delta` equals evaluating the unchanged function at
    :math:`\mathbf r - \delta`, so this needs no rebuilt basis objects -- only two
    extra samplings per direction.
    """
    delta = DEFAULT_ORBITAL_DELTA if delta is None else float(delta)
    X, Y, Z = grid.X, grid.Y, grid.Z
    M = len(basis)
    out = np.zeros((3, M, grid.size), dtype=complex)
    for k in range(3):
        shift = [np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)]
        shift[k] = np.full_like(X, delta)
        for mu, fn in enumerate(basis):
            if atom_of_orbital[mu] != atom:
                continue
            plus = fn.evaluate(X - shift[0], Y - shift[1], Z - shift[2])
            minus = fn.evaluate(X + shift[0], Y + shift[1], Z + shift[2])
            out[k, mu] = ((plus - minus) / (2.0 * delta)).ravel()
    return out


# --------------------------------------------------------------------------- #
# Hellmann-Feynman term.
# --------------------------------------------------------------------------- #

def _nuclear_repulsion_gradient(nuclei_bohr, charges):
    r"""``dV_nn/dR_A = -sum_{B!=A} Z_A Z_B (R_A-R_B)/R_AB^3`` -- purely analytic."""
    gradient = np.zeros((len(charges), 3))
    for a, (Za, Ra) in enumerate(zip(charges, nuclei_bohr)):
        for b, (Zb, Rb) in enumerate(zip(charges, nuclei_bohr)):
            if b == a:
                continue
            sep = Ra - Rb
            gradient[a] -= Za * Zb * sep / np.linalg.norm(sep) ** 3
    return gradient


def orbital_gradients(basis, grid, delta=None):
    r"""``grad phi_mu`` sampled on the grid, for **every** orbital.

    Returns ``(3, M, ngrid)``.  Related to :func:`orbital_derivatives` by
    :math:`\nabla\phi_\mu = -\partial\phi_\mu/\partial\mathbf R_A` -- moving the
    center one way is the same as moving the sampling point the other -- but
    computed for all orbitals at once, since the density gradient needs them all
    regardless of which atom they sit on.
    """
    delta = DEFAULT_ORBITAL_DELTA if delta is None else float(delta)
    X, Y, Z = grid.X, grid.Y, grid.Z
    out = np.zeros((3, len(basis), grid.size), dtype=complex)
    for k in range(3):
        shift = [np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)]
        shift[k] = np.full_like(X, delta)
        for mu, fn in enumerate(basis):
            plus = fn.evaluate(X + shift[0], Y + shift[1], Z + shift[2])
            minus = fn.evaluate(X - shift[0], Y - shift[1], Z - shift[2])
            out[k, mu] = ((plus - minus) / (2.0 * delta)).ravel()
    return out


def density_gradient(density_ao, psi_grid, grad_psi):
    r"""``grad rho`` from the analytic orbital gradients.

    :math:`\rho = \sum_{\mu\nu} D_{\mu\nu}\phi^*_\mu\phi_\nu`, so

    .. math::

        \nabla\rho = \sum_{\mu\nu} D_{\mu\nu}
            \big[(\nabla\phi^*_\mu)\phi_\nu + \phi^*_\mu(\nabla\phi_\nu)\big].

    Using this instead of a finite-difference gradient of the sampled density
    matters: the density has a **cusp** at every nucleus, exactly where the
    potential it is integrated against is largest, and a finite-difference
    stencil across that cusp is badly wrong.  The orbital gradients are analytic
    (evaluated from the closed-form basis functions), so no stencil ever
    straddles the singularity.
    """
    out = np.zeros((3, psi_grid.shape[1]))
    for k in range(3):
        term = np.einsum("mn,mg,ng->g", density_ao, grad_psi[k].conj(),
                         psi_grid, optimize=True)
        out[k] = 2.0 * np.real(term)          # D Hermitian, so the two terms pair
    return out


def _electronic_by_parts(rho, grid, nuclei_bohr, charges, softening,
                         grad_rho=None):
    r"""Electron-nucleus gradient with the derivative moved onto the density.

    Integrating by parts turns

    .. math::

        \frac{\partial E_{ne}}{\partial \mathbf R_A}
          = -\int \rho\, \nabla V_A
          = +\int (\nabla \rho)\, V_A ,

    which is mathematically identical in the continuum but *numerically much
    better behaved on a grid*.  The standard form integrates
    :math:`\rho\,(\mathbf r - \mathbf R_A)/r^3`: a near-singular integrand whose
    physical value is a tiny anisotropy left over from an almost perfect
    cancellation of enormous core contributions.  The by-parts form instead
    integrates :math:`(\nabla\rho)\,V_A` against the *same* softened
    :math:`-Z/\max(r,s)` the energy already uses, so the worst singularity the
    quadrature ever sees is :math:`1/r` rather than :math:`1/r^2`.

    .. warning::

       **This does not fix the heavy-atom problem.**  Measured on isolated atoms,
       where the exact force is zero, the by-parts form with a *faithful*
       (analytic) density gradient is no better than the standard one, and often
       worse:

       ===========  ==========  =========================  =============================
       isolated Be  analytic    by-parts (finite-diff rho) by-parts (analytic grad rho)
       ===========  ==========  =========================  =============================
       h = 0.20     1167        157                        4427
       h = 0.10     1593        50                         1600
       ===========  ==========  =========================  =============================

       The apparently excellent finite-difference column is a **smoothing
       artifact**, not accuracy: differencing the sampled density across the
       nuclear cusp acts as a low-pass filter that happens to damp the spurious
       force.  It is not a method -- it also gets H\ :sub:`2`, where the standard
       form is verified correct to 0.04 %, wrong by a factor of two.  This
       implementation therefore uses the analytic orbital gradients by default and
       makes no accuracy claim for the result.

       The two forms are also equal only in the continuum: the standard form is
       exactly the derivative of the energy Carcará computes, whereas this one is
       not, so an optimizer using it will not reach a stationary point of the
       reported energy.

       The mode is kept as an independent formulation for diagnosis -- the
       agreement or disagreement between the two is itself a useful measure of
       how badly the grid resolves the core.  The real fix is to remove the core
       cusp (pseudopotentials) or the grid (analytic Gaussian integrals).
    """
    if grad_rho is None:
        # Fallback: finite-difference the sampled density.  Less accurate near
        # the nuclear cusp, so callers should pass the analytic `grad_rho`.
        if not grid.is_orthorhombic:
            raise NotImplementedError(
                "the by-parts Hellmann-Feynman form needs a density gradient; "
                "without analytic orbital gradients it falls back to a "
                "finite-difference stencil, which assumes an orthorhombic grid")
        dlist = np.gradient(rho.reshape(grid.shape), grid.dx, grid.dy, grid.dz)
        grad_rho = np.stack([d.ravel() for d in dlist])

    coords = np.stack(grid.flat_coords())
    gradient = np.zeros((len(charges), 3))

    for a, (Za, Ra) in enumerate(zip(charges, nuclei_bohr)):
        diff = coords - Ra[:, None]
        r = np.sqrt(np.sum(diff * diff, axis=0))
        potential = -Za / np.maximum(r, softening)          # V_A on the grid
        for k in range(3):
            gradient[a, k] = np.sum(grad_rho[k] * potential) * grid.dV
    return gradient


def pseudopotential_local_gradient(density_ao, psi_grid, grid, nuclei_bohr,
                                   potentials, grad_psi=None):
    r"""``dE/dR_A`` from the **local** channel of a pseudopotential.

    The same Hellmann-Feynman idea, but with a potential that is bounded rather
    than singular:

    .. math::

        rac{\partial E}{\partial \mathbf R_A}
          = \int 
ho(\mathbf r)\,
            rac{\partial V^A_{	ext{loc}}(|\mathbf r-\mathbf R_A|)}
                 {\partial \mathbf R_A}\, d^3r
          = -\int 
ho(\mathbf r)\, V^{A\prime}_{	ext{loc}}(r)\,
            \hat{\mathbf r}_A \, d^3r .

    :math:`V'_{	ext{loc}}` is obtained by differentiating the radial table, and
    is finite everywhere -- which is exactly why the pseudopotential force does
    not suffer the core pathology of the all-electron one.
    """
    rho = np.real(np.einsum("mn,mg,ng->g", density_ao,
                            psi_grid.conj(), psi_grid, optimize=True))
    coords = np.stack(grid.flat_coords())
    gradient = np.zeros((len(potentials), 3))

    for a, (pp, Ra) in enumerate(zip(potentials, nuclei_bohr)):
        diff = coords - Ra[:, None]
        r = np.sqrt(np.sum(diff * diff, axis=0))

        # The quantity actually needed is (dV/dr)/r, because
        #   dV/dR_A = -(dV/dr) * (r - R_A)/r ,
        # and it is *finite* at the origin: a Troullier-Martins potential is flat
        # there, so dV/dr ~ c r.  Interpolating dV/dr and then dividing by r
        # instead would manufacture a 1/r spike at any grid node that lands near
        # the nucleus -- which is precisely the pathology pseudopotentials exist
        # to remove.  So the ratio is formed on the radial table first.
        dv_dr = np.gradient(pp.v_local, pp.r)
        ratio = dv_dr / pp.r
        ratio[0] = ratio[1]                     # constant limit as r -> 0
        # Beyond the table V = -Z_ion/r, so (dV/dr)/r = Z_ion/r^3.
        outside = pp.valence_charge / pp.r[-1] ** 3
        value = np.interp(r, pp.r, ratio, left=ratio[0], right=outside)
        far = r > pp.r[-1]
        if np.any(far):
            value[far] = pp.valence_charge / r[far] ** 3

        weight = -rho * value * grid.dV
        gradient[a] = np.sum(weight * diff, axis=1)
    return gradient


def kb_nonlocal_gradient(density_ao, basis, grid, projectors, atom_of_orbital,
                         n_atoms, delta=None):
    r"""``dE/dR_A`` from the Kleinman-Bylander **nonlocal** term.

    The nonlocal energy is
    :math:`E_{NL} = \sum_p E^{KB}_p \sum_{\mu
u} D_{\mu
u}
    \langle\phi_\mu|\chi_p
angle\langle\chi_p|\phi_
u
angle`, and *two* things
    move when atom :math:`A` shifts: the basis functions centered on it (a Pulay
    term) and the projectors belonging to it.  Both derivatives are taken by the
    same displaced-sampling trick used for the orbitals, so no integral is
    recomputed from scratch.
    """
    from ..integrals import _backend

    delta = DEFAULT_ORBITAL_DELTA if delta is None else float(delta)
    X, Y, Z = grid.X, grid.Y, grid.Z
    psi = np.stack([f.evaluate(X, Y, Z).ravel() for f in basis])
    chi = np.stack([p.evaluate(X, Y, Z).ravel() for p in projectors])         if projectors else np.zeros((0, grid.size), dtype=complex)
    gradient = np.zeros((n_atoms, 3))
    if not projectors:
        return gradient

    energies = np.array([p.kb_energy for p in projectors])
    overlaps = _backend.kb_projections(psi, chi, grid.dV)        # (M, P)

    for atom in range(n_atoms):
        for k in range(3):
            shift = [np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)]
            shift[k] = np.full_like(X, delta)

            # d(basis)/dR_A: only functions centered on this atom move.
            dpsi = np.zeros_like(psi)
            for mu, fn in enumerate(basis):
                if atom_of_orbital[mu] != atom:
                    continue
                plus = fn.evaluate(X - shift[0], Y - shift[1], Z - shift[2])
                minus = fn.evaluate(X + shift[0], Y + shift[1], Z + shift[2])
                dpsi[mu] = ((plus - minus) / (2.0 * delta)).ravel()

            # d(projector)/dR_A: only projectors belonging to this atom move.
            dchi = np.zeros_like(chi)
            for index, projector in enumerate(projectors):
                if projector.atom_index != atom:
                    continue
                plus = projector.evaluate(X - shift[0], Y - shift[1],
                                          Z - shift[2])
                minus = projector.evaluate(X + shift[0], Y + shift[1],
                                           Z + shift[2])
                dchi[index] = ((plus - minus) / (2.0 * delta)).ravel()

            d_overlap = (_backend.kb_projections(dpsi, chi, grid.dV)
                         + _backend.kb_projections(psi, dchi, grid.dV))
            # dE = 2 Re sum_p E_p sum_mn D_mn d<mu|chi_p> <chi_p|nu>
            term = np.einsum("mn,mp,np->", density_ao, d_overlap,
                             (overlaps * energies).conj(), optimize=True)
            gradient[atom, k] = 2.0 * float(np.real(term))
    return gradient


def hellmann_feynman_gradient(density_ao, psi_grid, grid, nuclei_bohr, charges,
                              softening, form: str = "analytic",
                              grad_psi=None):
    r"""``dE/dR_A`` from the explicit nuclear dependence of the Hamiltonian.

    Two contributions, both analytic:

    * electron-nucleus attraction,
      :math:`\partial\langle \hat V_{ne}\rangle/\partial R_{A,k}
      = -Z_A \int \rho(\mathbf r)\,(r-R_A)_k/|\mathbf r - \mathbf R_A|^3`;
    * nuclear repulsion,
      :math:`\partial V_{nn}/\partial R_{A,k}
      = -\sum_{B\neq A} Z_A Z_B (R_A-R_B)_k/R_{AB}^3`.

    ``density_ao`` is the AO-basis one-particle density matrix
    (:math:`\partial E/\partial h`), which is what makes this the *correlated*
    Hellmann-Feynman force rather than a mean-field one.

    Parameters
    ----------
    form : {"analytic", "by-parts"}
        How the electron-nucleus term is evaluated.

        ``"analytic"`` (default) differentiates the potential directly.  It is
        exactly the derivative of the energy Carcará computes -- the right choice
        when force and energy must be consistent -- but its integrand is
        near-singular and, for heavy atoms on a practical grid, dominated by
        core-region noise.

        ``"by-parts"`` moves the derivative onto the density instead (see
        :func:`_electronic_by_parts`).  Far better conditioned and convergent,
        at the cost of no longer being exactly ``dE/dR`` of the discrete energy.

    .. note::

       For the analytic form the engine regularizes the Coulomb singularity as
       :math:`-Z/\max(r, s)`, so **inside** the softening radius the potential is
       *constant* and its nuclear derivative is exactly zero.  Differentiating the
       clamped expression instead would give a spurious :math:`1/s^3` spike: a
       single grid node closer than :math:`s` to a nucleus is enough to corrupt
       the force by ~20 %, which is why that region is masked out explicitly.
    """
    if form not in HELLMANN_FEYNMAN_FORMS:
        raise ValueError(
            f"unknown Hellmann-Feynman form {form!r}; use one of "
            f"{HELLMANN_FEYNMAN_FORMS}")

    rho = np.real(np.einsum("mn,mg,ng->g", density_ao,
                            psi_grid.conj(), psi_grid, optimize=True))

    if form == "by-parts":
        grad_rho = (density_gradient(density_ao, psi_grid, grad_psi)
                    if grad_psi is not None else None)
        electronic = _electronic_by_parts(rho, grid, nuclei_bohr, charges,
                                          softening, grad_rho=grad_rho)
    else:
        coords = np.stack(grid.flat_coords())                   # (3, ngrid)
        electronic = np.zeros((len(charges), 3))
        for a, (Za, Ra) in enumerate(zip(charges, nuclei_bohr)):
            diff = coords - Ra[:, None]                         # (3, ngrid)
            r = np.sqrt(np.sum(diff * diff, axis=0))
            # V = -Z/max(r, s) is flat inside the softening radius, so its
            # nuclear derivative vanishes there.
            weight = np.where(r >= softening,
                              -Za * rho / np.maximum(r, softening) ** 3,
                              0.0) * grid.dV
            electronic[a] = np.sum(weight * diff, axis=1)

    return electronic + _nuclear_repulsion_gradient(nuclei_bohr, charges)


# --------------------------------------------------------------------------- #
# Pulay terms.
# --------------------------------------------------------------------------- #

def _one_body_pulay(engine, psi_grid, dpsi, vext, grid, de_dh, de_ds):
    r"""Basis-motion part of ``<dE/dh, dh/dR> + <dE/dS, dS/dR>``.

    ``dh/dR`` and ``dS/dR`` are evaluated by running the *same* one-body kernel
    on the stacked basis ``[phi; dphi]`` and reading the cross blocks, so the
    finite-difference Laplacian is identical to the one the energy used.
    """
    from ..integrals import _backend

    M = psi_grid.shape[0]
    stacked = np.ascontiguousarray(np.vstack([psi_grid, dpsi]))
    T_full, V_full = _backend.one_body_matrices(stacked, vext, grid)
    h_full = T_full + V_full

    # <dphi_mu|h|phi_nu> and its transpose partner.
    cross_h = h_full[M:, :M]
    dh = np.real(cross_h + cross_h.conj().T)
    # Overlap derivative from the same stacked sampling.
    cross_s = (dpsi.conj() @ psi_grid.T) * grid.dV
    ds = np.real(cross_s + cross_s.conj().T)

    return (float(np.sum(de_dh * dh)), float(np.sum(de_ds * ds)))


def _pair_potentials(psi_grid, grid):
    """Coulomb potentials ``Phi_ij`` of every pair density ``conj(phi_i) phi_j``."""
    M, ngrid = psi_grid.shape
    pairs = (np.conj(psi_grid)[:, None, :]
             * psi_grid[None, :, :]).reshape(M * M, ngrid)
    solver = PoissonFFTSolver(grid.shape, grid.dx)
    return solver.solve_stack(pairs).reshape(M, M, ngrid)


def _two_body_pulay(psi_grid, dpsi, phi_pairs, grid, de_dg):
    r"""``<dE/dg, dg/dR>`` without ever forming ``dg``.

    With ``g[m,n,l,s] = <mn|ls> = int rho_ml(1) Phi_ns(1)``, the derivative has
    four terms; two act on electron 1 and two on electron 2.  Using the symmetry
    of the Coulomb kernel, both reduce to contracting a *derivative pair density*
    against a potential that is already available:

    ``A_ml(r) = sum_ns dE/dg[m,n,l,s] Phi_ns(r)``  (electron 1) and
    ``B_ns(r) = sum_ml dE/dg[m,n,l,s] Phi_ml(r)``  (electron 2).

    Cost is one ``M^4 x ngrid`` contraction and no extra Poisson solves.
    """
    # Derivative of the pair density: d(conj(phi_m) phi_l).
    drho = (dpsi.conj()[:, None, :] * psi_grid[None, :, :]
            + psi_grid.conj()[:, None, :] * dpsi[None, :, :])       # (M, M, g)

    A = np.einsum("mnls,nsg->mlg", de_dg, phi_pairs, optimize=True)
    B = np.einsum("mnls,mlg->nsg", de_dg, phi_pairs, optimize=True)
    value = (np.einsum("mlg,mlg->", drho, A, optimize=True)
             + np.einsum("nsg,nsg->", drho, B, optimize=True))
    return float(np.real(value)) * grid.dV


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def nuclear_gradient(integrals, gamma, gamma2, *, n_electrons, atom_of_orbital,
                     frozen=(), orbital_delta=None, scf_iterations=40,
                     include_pulay=True,
                     hellmann_feynman: str = "analytic") -> ForceResult:
    r"""Total nuclear gradient of a converged variational calculation.

    Parameters
    ----------
    integrals : MolecularIntegrals
        The engine that built the Hamiltonian -- it carries the basis, the grid,
        the nuclei and the Coulomb softening, so the gradient is evaluated with
        exactly the quantities the energy used.
    gamma, gamma2 : ndarray
        One- and two-particle RDMs of the converged state, in the *active
        spin-orbital* basis (see :mod:`carcara.algorithms.rdm`).
    n_electrons : int
        Total electron count (including any frozen core).
    atom_of_orbital : sequence of int
        Which atom each basis function is centered on.
    frozen : sequence of int
        Frozen spatial molecular orbitals, if the frozen-core approximation was
        used.
    include_pulay : bool
        Compute the Pulay terms (default ``True``).  ``False`` returns the bare
        Hellmann-Feynman force -- much cheaper, but for an atom-centered basis it
        is *not* the physical gradient (see the module docstring).
    hellmann_feynman : {"analytic", "by-parts"}
        How the electron-nucleus term is evaluated (default ``"analytic"``).
        ``"by-parts"`` is far better conditioned for heavy atoms but is no longer
        exactly the derivative of the reported energy -- see
        :func:`hellmann_feynman_gradient`.

    Returns
    -------
    ForceResult
        Forces in eV/Angstrom plus the Hellmann-Feynman / Pulay breakdown.
    """
    from ._jax_energy import integral_gradients

    grid = integrals.grid
    basis = integrals.basis
    M = len(basis)

    psi_grid = np.stack([b.evaluate(grid.X, grid.Y, grid.Z).ravel()
                         for b in basis])

    # The *same* external potential the energy was built from.  Going straight
    # to `_potentials.nuclear_potential` would silently use the bare -Z/r even
    # on the pseudopotential path -- which is both the wrong operator and, since
    # that path sets softening = 0, divergent wherever a grid node lands near a
    # nucleus.  Route through the engine's own accessor instead.
    external = integrals.external_potential()
    vext = np.ascontiguousarray(
        np.real(external(grid.X, grid.Y, grid.Z)).reshape(-1),
        dtype=np.float64)

    # Raw AO integrals -- the inputs the algebraic layer differentiates.  This
    # must reproduce MolecularIntegrals._compute() exactly, Kleinman-Bylander
    # term included: `de_dh` and friends are derivatives of *this* h_ao, so any
    # term missing here makes them derivatives of an energy nobody evaluated.
    S = np.real(integrals.overlap())
    T, V = integrals._engine.one_body(external, energy_units="Ha")
    one_body = T + V + integrals.kb_nonlocal()
    h_ao = np.real(0.5 * (one_body + one_body.conj().T))
    eri_ao = np.real(integrals._engine.two_body(method="fft", energy_units="Ha"))

    de_ds, de_dh, de_dg = integral_gradients(
        S, h_ao, eri_ao, gamma, gamma2, n_electrons=n_electrons,
        frozen=tuple(frozen), nuclear_repulsion=0.0,
        scf_iterations=scf_iterations)

    charges = [Z for Z, _R in integrals.nuclei]
    nuclei_bohr = [np.asarray(R, dtype=float) * ANGSTROM_TO_BOHR
                   for _Z, R in integrals.nuclei]
    softening = integrals._potentials.softening

    if getattr(integrals, "uses_pseudopotentials", False):
        # Pseudopotential path: a smooth local channel plus the Kleinman-Bylander
        # nonlocal term, and ion-ion rather than nucleus-nucleus repulsion.
        hf = pseudopotential_local_gradient(de_dh, psi_grid, grid, nuclei_bohr,
                                            integrals.pseudopotentials)
        hf = hf + _nuclear_repulsion_gradient(nuclei_bohr, charges)
        hf = hf + kb_nonlocal_gradient(
            de_dh, basis, grid, integrals.kb_projectors, atom_of_orbital,
            len(charges), delta=orbital_delta)
    else:
        grad_psi = (orbital_gradients(basis, grid, orbital_delta)
                    if hellmann_feynman == "by-parts" else None)
        hf = hellmann_feynman_gradient(de_dh, psi_grid, grid, nuclei_bohr,
                                       charges, softening,
                                       form=hellmann_feynman,
                                       grad_psi=grad_psi)

    pulay = np.zeros_like(hf)
    if include_pulay:
        phi_pairs = _pair_potentials(psi_grid, grid)
        for atom in range(len(charges)):
            dstack = orbital_derivatives(basis, grid, atom_of_orbital, atom,
                                         orbital_delta)
            if not np.any(dstack):
                continue                        # no basis function on this atom
            for k in range(3):
                dpsi = dstack[k]
                one_h, one_s = _one_body_pulay(integrals._engine, psi_grid,
                                               dpsi, vext, grid, de_dh, de_ds)
                two = _two_body_pulay(psi_grid, dpsi, phi_pairs, grid, de_dg)
                pulay[atom, k] = one_h + one_s + two

    gradient = hf + pulay
    forces = -gradient * HA_BOHR_TO_EV_ANGSTROM
    return ForceResult(
        forces=forces, hellmann_feynman=hf, pulay=pulay, gradient=gradient,
        n_electrons=float(np.real(np.trace(gamma))),
        details={"n_orbitals": M, "frozen": tuple(frozen),
                 "include_pulay": bool(include_pulay),
                 "hellmann_feynman": hellmann_feynman})
