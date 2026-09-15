# -*- coding: utf-8 -*-
# file: algorithms/pseudo_forces.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Hellmann-Feynman and Pulay forces for the PAW and ONCVPSP families.

The energy these families report is

.. math::

    E(\mathbf R) = \sum_{pq} D_{pq}\, h^{\rm MO}_{pq}
      + \tfrac12 \sum_{pqrs} \Gamma_{pqrs}\, g^{\rm MO}_{pqrs}
      + E_{\rm ion}(\mathbf R) + E_{\rm 1c},
    \qquad
    h^{\rm MO} = A^\dagger h A, \quad A = S^{-1/2} V ,

with :math:`D` and :math:`\Gamma` the spin-summed reduced density matrices of
the converged state, :math:`V` its molecular orbitals in the Löwdin basis, and
the atomic-orbital matrices

* :math:`S = \tilde S + C q C^\dagger` (the PAW-augmented overlap),
* :math:`h = T + V_{\rm loc} + C D^{\rm ion} C^\dagger` (kinetic, local and
  nonlocal parts),
* :math:`g = g_{\rm grid} + \sum_A (Q^A\otimes W^A + W^A\otimes Q^A)
  + \sum_{AB} U_{AB}\, Q^A\otimes Q^B` (the compensation charges),

where :math:`C_{\mu p} = \langle\phi_\mu|p_p\rangle` are the projections of the
basis on the projectors.  Holding :math:`D`, :math:`\Gamma` and :math:`V` fixed,

.. math::

    \frac{dE}{d\mathbf R_A} =
      \Big\langle \frac{\partial E}{\partial S}, \frac{dS}{d\mathbf R_A}\Big\rangle
    + \Big\langle \frac{\partial E}{\partial h}, \frac{dh}{d\mathbf R_A}\Big\rangle
    + \Big\langle \frac{\partial E}{\partial g}, \frac{dg}{d\mathbf R_A}\Big\rangle
    + \frac{\partial E_{\rm ion}}{\partial \mathbf R_A},

and every matrix derivative splits into two physically distinct parts:

**Hellmann-Feynman** -- the *operators* of atom :math:`A` move with its
nucleus while the basis stays put: the local potential
:math:`V^A_{\rm loc}`, the projectors :math:`p^A` (through :math:`C`, in both
the nonlocal term and the augmented overlap), the compensation charge
(:math:`W^A` and :math:`U_{AB}`), and the ion-ion repulsion.

**Pulay** -- the *basis functions* centered on :math:`A` move while the
operators stay put: :math:`\tilde S`, :math:`T`, :math:`V_{\rm loc}`,
:math:`C`, the grid two-electron tensor and :math:`W`.  Atom-centered bases
make this term large; it is not optional.

Holding the RDMs and :math:`V` fixed is exact when the state is **stationary
with respect to orbital rotations** -- an exact eigenstate of the Hamiltonian
in its orbital space, which is what a converged ADAPT-VQE run over all
orbitals approaches (generalized Brillouin theorem).  The residual
:math:`\langle\Psi|[H,\hat\kappa]|\Psi\rangle` is reported as
``details["orbital_gradient"]`` so the approximation is measured rather than
assumed.  RHF MOs are complex once the basis carries :math:`l>0` functions
(DZP polarization), and nothing here assumes real arithmetic.

How the pieces are evaluated
----------------------------
* Matrix derivatives on the **frozen grid** by displaced sampling: a function
  centered at :math:`\mathbf R` moved by :math:`\delta` is the same function
  sampled at :math:`\mathbf r - \delta` -- for orbitals, projectors, the local
  potential and the compensation potential alike.  The kinetic cross terms
  reuse the energy's own finite-difference kernel on the stacked
  ``[phi; dphi]`` basis, and the two-electron derivative is contracted from
  the pair densities against the Coulomb potentials the energy uses.
* The PAW **projections** :math:`C_{\mu p} = \langle\phi_\mu|p_p\rangle` are
  atom-centered quadratures over each projector's sphere
  (:meth:`~carcara.pseudopotentials.paw.PAWIntegrals.projections`), so they
  depend only on the separation of the function and the projector.  Their
  derivative is the same quadrature of :math:`\partial\phi_\mu/\partial\mathbf R`,
  :math:`G_{\mu p}`: the Pulay part (basis function moving) is :math:`+G` and
  the Hellmann-Feynman part (projector moving) is :math:`-G`, so the total is
  exactly translation invariant.  ONCVPSP keeps grid projections, whose
  derivatives are sampled on the grid like everything else.
* The contraction with :math:`\partial E/\partial(S,h,g)` is the
  **directional derivative** of the small algebraic energy
  :math:`E(S, h, g)` along :math:`(dS, dh, dg)`, by a central difference
  (the algebra is :math:`O(M^4)` with :math:`M` basis functions).  That covers
  the derivative of :math:`S^{-1/2}` and the complex MO transform without
  hand-derived energy-weighted density matrices.
"""

from __future__ import annotations

import numpy as np

#: Central-difference step of the algebraic directional derivative.
DEFAULT_ALGEBRAIC_STEP = 1e-4
#: Step (Bohr) of the compensation-Coulomb distance derivative.
COMPENSATION_DISTANCE_STEP = 1e-4
#: Tolerance (Hartree) of the reconstructed-energy consistency check.
ENERGY_CHECK_TOLERANCE = 1e-6


# --------------------------------------------------------------------------- #
# The algebraic energy.
# --------------------------------------------------------------------------- #

def spatial_rdms(gamma, gamma2, n_orbitals: int):
    r"""Spin-summed spatial RDMs from spin-blocked spin-orbital ones.

    Spin-orbital ``P = p + sigma * M`` (alpha block first).  The one-body RDM
    sums the two spin blocks; the two-body RDM sums the four blocks in which
    electron 1 keeps its spin and electron 2 keeps its spin, matching
    :func:`~carcara.core.hamiltonian.spin_block_integrals`.
    """
    M = int(n_orbitals)
    gamma = np.asarray(gamma, dtype=complex)
    gamma2 = np.asarray(gamma2, dtype=complex)
    if gamma.shape != (2 * M, 2 * M):
        raise ValueError(f"expected a {2 * M}x{2 * M} one-body RDM "
                         f"(frozen orbitals are not supported here), got "
                         f"{gamma.shape}")
    a, b = slice(0, M), slice(M, 2 * M)
    D = gamma[a, a] + gamma[b, b]
    G = (gamma2[a, a, a, a] + gamma2[a, b, a, b]
         + gamma2[b, a, b, a] + gamma2[b, b, b, b])
    return D, G


class AlgebraicEnergy:
    r"""Electronic energy as a function of the AO matrices, RDMs and MOs fixed.

    ``energy(S, h, g)`` orthonormalizes with :math:`X = S^{-1/2}`, transforms
    with :math:`A = X V` and contracts with the spatial RDMs -- the same
    pipeline :class:`~carcara.core.hamiltonian.MolecularIntegrals` and
    :class:`~carcara.algorithms.hartree_fock.RHF` follow to build the
    Hamiltonian the state was optimized on.
    """

    def __init__(self, mo_coefficients, D, G):
        self.V = np.asarray(mo_coefficients, dtype=complex)
        self.D = np.asarray(D, dtype=complex)
        self.G = np.asarray(G, dtype=complex)

    @staticmethod
    def lowdin(S):
        S = 0.5 * (S + S.conj().T)
        w, U = np.linalg.eigh(S)
        return (U * (1.0 / np.sqrt(w))) @ U.conj().T

    def with_orbitals(self, A, h, g) -> float:
        h_mo = A.conj().T @ h @ A
        g_mo = np.einsum("ap,bq,cr,ds,abcd->pqrs", A.conj(), A.conj(), A, A, g,
                         optimize=True)
        return float(np.real(np.einsum("pq,pq->", self.D, h_mo)
                             + 0.5 * np.einsum("pqrs,pqrs->", self.G, g_mo)))

    def __call__(self, S, h, g) -> float:
        return self.with_orbitals(self.lowdin(S) @ self.V, h, g)

    def directional(self, S, h, g, dS, dh, dg,
                    step: float = DEFAULT_ALGEBRAIC_STEP) -> float:
        """``<dE/dS, dS> + <dE/dh, dh> + <dE/dg, dg>`` by a central difference."""
        def shifted(sign):
            return self(S + sign * step * dS if dS is not None else S,
                        h + sign * step * dh if dh is not None else h,
                        g + sign * step * dg if dg is not None else g)
        return (shifted(+1.0) - shifted(-1.0)) / (2.0 * step)

    def orbital_gradient(self, S, h, g, step: float = 1e-5) -> float:
        r"""Largest :math:`|\partial E/\partial\kappa_{pq}|` over orbital rotations.

        Zero for a state that is an eigenstate of the Hamiltonian in its
        orbital space; the size of the neglected orbital-response term.
        """
        from scipy.linalg import expm

        A = self.lowdin(S) @ self.V
        M = A.shape[1]
        largest = 0.0
        for p in range(M):
            for q in range(p + 1, M):
                for kind in (1.0, 1j):
                    K = np.zeros((M, M), dtype=complex)
                    K[p, q] = kind
                    K[q, p] = -np.conj(kind)
                    plus = self.with_orbitals(A @ expm(step * K), h, g)
                    minus = self.with_orbitals(A @ expm(-step * K), h, g)
                    largest = max(largest, abs(plus - minus) / (2.0 * step))
        return largest


# --------------------------------------------------------------------------- #
# Displaced sampling on the grid.
# --------------------------------------------------------------------------- #

def _moved_function(function, grid, k: int, delta: float) -> np.ndarray:
    """``d f / d R_k`` of a centered function, on the flattened grid."""
    shift = [0.0, 0.0, 0.0]
    shift[k] = delta
    plus = function.evaluate(grid.X - shift[0], grid.Y - shift[1],
                             grid.Z - shift[2])
    minus = function.evaluate(grid.X + shift[0], grid.Y + shift[1],
                              grid.Z + shift[2])
    return ((plus - minus) / (2.0 * delta)).ravel()


def _moved_radial(radial, center, grid, k: int, delta: float) -> np.ndarray:
    """``d v(|r - R|) / d R_k`` of a spherical potential centered at ``center``."""
    out = []
    for sign in (+1.0, -1.0):
        c = np.array(center, dtype=float)
        c[k] += sign * delta
        radius = np.sqrt((grid.X - c[0]) ** 2 + (grid.Y - c[1]) ** 2
                         + (grid.Z - c[2]) ** 2)
        out.append(np.asarray(radial(radius), dtype=float).ravel())
    return (out[0] - out[1]) / (2.0 * delta)


def _sampled_radial(radial, center, grid) -> np.ndarray:
    radius = np.sqrt((grid.X - center[0]) ** 2 + (grid.Y - center[1]) ** 2
                     + (grid.Z - center[2]) ** 2)
    return np.asarray(radial(radius), dtype=float).ravel()


# --------------------------------------------------------------------------- #
# The gradient.
# --------------------------------------------------------------------------- #

def _outer(X, Y):
    """``(X (x) Y)_pqrs = X_pr Y_qs`` -- the layout of the augmented tensor."""
    return np.einsum("pr,qs->pqrs", X, Y)


def _hermitian(X):
    return 0.5 * (X + X.conj().T)


def pseudo_nuclear_gradient(integrals, gamma, gamma2, *, atom_of_orbital,
                            orbital_delta=None, include_pulay: bool = True,
                            algebraic_step: float = DEFAULT_ALGEBRAIC_STEP,
                            orbital_gradient: bool = True):
    r"""Hellmann-Feynman + Pulay gradient of a pseudopotential calculation.

    Parameters
    ----------
    integrals : MolecularIntegrals or PAWIntegrals
        The engine that built the Hamiltonian (after
        ``molecular_hamiltonian(mo_basis=True)``, so its
        :attr:`mo_coefficients` are set).
    gamma, gamma2 : ndarray
        Spin-orbital one- and two-body RDMs of the converged state, over all
        orbitals (no frozen core).
    atom_of_orbital : sequence of int
        Which atom each basis function is centered on.
    orbital_delta : float, optional
        Displacement step (Bohr) of the sampled derivatives.
    include_pulay : bool
        Compute the Pulay part (default ``True``).
    algebraic_step : float
        Step of the directional derivative of the algebraic energy.
    orbital_gradient : bool
        Measure the orbital-rotation residual (``details["orbital_gradient"]``).

    Returns
    -------
    ForceResult
        Forces, Hellmann-Feynman and Pulay gradients in eV/Angstrom; ``details``
        carries the reconstructed energy (Hartree), the orbital gradient and the
        number of on-site projection pairs.
    """
    from ..integrals import _backend
    from .forces import (DEFAULT_ORBITAL_DELTA, HA_BOHR_TO_EV_ANGSTROM,
                         ForceResult, _nuclear_repulsion_gradient,
                         _pair_potentials, orbital_derivatives)

    if integrals.mo_coefficients is None:
        raise RuntimeError("the integrals carry no molecular orbitals; build the "
                           "Hamiltonian with molecular_hamiltonian(mo_basis=True)")
    if integrals.kinetic != "fd":
        raise NotImplementedError(
            "pseudopotential forces differentiate the finite-difference kinetic "
            f"operator; kinetic={integrals.kinetic!r} is not supported")

    delta = DEFAULT_ORBITAL_DELTA if orbital_delta is None else float(orbital_delta)
    grid = integrals.grid
    dV = grid.dV
    basis = integrals.basis
    M = len(basis)
    ngrid = grid.size
    psi = np.ascontiguousarray(integrals._engine._psi)
    projectors = list(integrals.kb_projectors)
    P = len(projectors)
    chi = (np.stack([p.evaluate(grid.X, grid.Y, grid.Z).ravel()
                     for p in projectors])
           if P else np.zeros((0, ngrid), dtype=complex))
    centers = [np.asarray(c, dtype=float) for _z, c in integrals._potentials.nuclei]
    charges = [float(z) for z, _c in integrals.nuclei]
    n_atoms = len(centers)
    datasets = integrals.pseudopotentials

    D1, G2 = spatial_rdms(gamma, gamma2, M)
    energy = AlgebraicEnergy(integrals.mo_coefficients, D1, G2)

    # -- the reference AO matrices, exactly as MolecularIntegrals._compute --
    external = integrals.external_potential()
    vext = np.ascontiguousarray(
        np.real(external(grid.X, grid.Y, grid.Z)).reshape(-1), dtype=np.float64)
    T, V_loc = integrals._engine.one_body(external, energy_units="Ha",
                                          kinetic=integrals.kinetic)
    C = integrals.projections() if P else np.zeros((M, 0), dtype=complex)
    D_nl = integrals.nonlocal_coupling_matrix() if P else None
    Q_nl = integrals.nonlocal_overlap_matrix()
    one = T + V_loc + (C @ D_nl @ C.conj().T if P else 0.0)
    h0 = _hermitian(one)
    S0 = integrals.overlap()
    g0 = integrals._engine.two_body(method="fft", energy_units="Ha")
    augmentation = integrals.two_body_augmentation()
    if augmentation is not None:
        g0 = g0 + np.asarray(augmentation)
    electronic = energy(S0, h0, g0)
    total = electronic + integrals.nuclear_repulsion + integrals.constant_energy

    # -- PAW compensation charges --
    paw = augmentation is not None and hasattr(integrals, "compensation_moments")
    if paw:
        Q_mom = integrals.compensation_moments()
        W_mom = integrals.compensation_potentials()
        U = integrals.compensation_coulomb()
        comp_atoms = sorted(Q_mom)
        blocks = integrals._atom_positions()
        q_blocks = {A: Q_nl[np.ix_(blocks[A], blocks[A])] for A in comp_atoms}
        v_comp = {A: _sampled_radial(datasets[A].compensation_potential,
                                     centers[A], grid) for A in comp_atoms}
    exact_projections = bool(getattr(integrals, "exact_projections", False)) and P > 0
    if exact_projections:
        from ..pseudopotentials.paw import atom_centered_projection_gradients
        projection_gradients = atom_centered_projection_gradients(
            basis, projectors, delta)
    orbital_atoms = np.asarray(atom_of_orbital)

    def nonlocal_terms(dC):
        dh = dC @ D_nl @ C.conj().T + C @ D_nl @ dC.conj().T
        dS = (dC @ Q_nl @ C.conj().T + C @ Q_nl @ dC.conj().T
              if Q_nl is not None else None)
        return dh, dS

    def augmentation_derivative(dC, dW, dU):
        dQ = {}
        for A in comp_atoms:
            idx = blocks[A]
            Ca, dCa = C[:, idx], dC[:, idx]
            dQ[A] = dCa @ q_blocks[A] @ Ca.conj().T + Ca @ q_blocks[A] @ dCa.conj().T
        out = np.zeros((M, M, M, M), dtype=complex)
        for A in comp_atoms:
            out += _outer(dQ[A], W_mom[A]) + _outer(W_mom[A], dQ[A])
            if A in dW:
                out += _outer(Q_mom[A], dW[A]) + _outer(dW[A], Q_mom[A])
            for B in comp_atoms:
                out += U[A, B] * (_outer(dQ[A], Q_mom[B]) + _outer(Q_mom[A], dQ[B]))
                if dU is not None and dU[A, B] != 0.0:
                    out += dU[A, B] * _outer(Q_mom[A], Q_mom[B])
        return out

    phi = (_pair_potentials(psi, grid).reshape(M * M, ngrid)
           if include_pulay else None)

    hf = np.zeros((n_atoms, 3))
    pulay = np.zeros((n_atoms, 3))
    for atom in range(n_atoms):
        own_projectors = [p for p, proj in enumerate(projectors)
                          if proj.atom_index == atom]
        orbital_stack = (orbital_derivatives(basis, grid, atom_of_orbital, atom,
                                             delta)
                         if include_pulay else None)
        for k in range(3):
            # ---------------- Pulay: the basis functions of `atom` move ----
            if include_pulay and np.any(orbital_stack[k]):
                dpsi = orbital_stack[k]
                cross = (dpsi.conj() @ psi.T) * dV
                dS = cross + cross.conj().T
                T_full, V_full = _backend.one_body_matrices(
                    np.ascontiguousarray(np.vstack([psi, dpsi])), vext, grid)
                cross_h = (T_full + V_full)[M:, :M]
                dh = cross_h + cross_h.conj().T
                if P:
                    if exact_projections:
                        dC = np.zeros((M, P), dtype=complex)
                        own = orbital_atoms == atom
                        dC[own] = projection_gradients[own, :, k]
                    else:
                        dC = _backend.kb_projections(dpsi, chi, dV)
                    dh_nl, dS_aug = nonlocal_terms(dC)
                    dh = dh + dh_nl
                    if dS_aug is not None:
                        dS = dS + dS_aug
                drho = (dpsi.conj()[:, None, :] * psi[None, :, :]
                        + psi.conj()[:, None, :] * dpsi[None, :, :]
                        ).reshape(M * M, ngrid)
                R = (drho @ phi.T + phi @ drho.T) * dV
                dg = R.reshape(M, M, M, M).transpose(0, 2, 1, 3)
                if paw:
                    dW = {}
                    for A in comp_atoms:
                        cross_w = ((dpsi.conj() * v_comp[A]) @ psi.T) * dV
                        dW[A] = cross_w + cross_w.conj().T
                    dg = dg + augmentation_derivative(dC, dW, None)
                pulay[atom, k] = energy.directional(
                    S0, h0, g0, _hermitian(dS), _hermitian(dh), dg,
                    algebraic_step)

            # ---------------- Hellmann-Feynman: the operators of `atom` move --
            w_loc = _moved_radial(datasets[atom].local_potential, centers[atom],
                                  grid, k, delta)
            dh = ((psi.conj() * w_loc) @ psi.T) * dV
            dS = None
            dg = None
            if P:
                if exact_projections:
                    dC = np.zeros((M, P), dtype=complex)
                    dC[:, own_projectors] = -projection_gradients[:, own_projectors, k]
                else:
                    dchi = np.zeros_like(chi)
                    for p in own_projectors:
                        dchi[p] = _moved_function(projectors[p], grid, k, delta)
                    dC = _backend.kb_projections(psi, dchi, dV)
                dh_nl, dS = nonlocal_terms(dC)
                dh = dh + dh_nl
                if paw:
                    dW = {}
                    if atom in comp_atoms:
                        w_comp = _moved_radial(datasets[atom].compensation_potential,
                                               centers[atom], grid, k, delta)
                        dW[atom] = ((psi.conj() * w_comp) @ psi.T) * dV
                    dU = np.zeros_like(U)
                    for B in comp_atoms:
                        if B == atom or atom not in comp_atoms:
                            continue
                        sep = centers[atom] - centers[B]
                        distance = float(np.linalg.norm(sep))
                        from ..pseudopotentials.paw import compensation_coulomb
                        ra = datasets[atom].compensation_radius
                        rb = datasets[B].compensation_radius
                        step = COMPENSATION_DISTANCE_STEP
                        slope = (compensation_coulomb(ra, rb, distance + step)
                                 - compensation_coulomb(ra, rb, distance - step)) \
                            / (2.0 * step)
                        dU[atom, B] = dU[B, atom] = slope * sep[k] / distance
                    dg = augmentation_derivative(dC, dW, dU)
            hf[atom, k] = energy.directional(
                S0, h0, g0, _hermitian(dS) if dS is not None else None,
                _hermitian(dh), dg, algebraic_step)

    hf = hf + _nuclear_repulsion_gradient(centers, charges)

    details = {"method": "pseudopotential", "energy_hartree": float(total),
               "electronic_energy_hartree": float(electronic),
               "exact_projections": exact_projections,
               "include_pulay": bool(include_pulay),
               "n_orbitals": M}
    if orbital_gradient:
        details["orbital_gradient"] = energy.orbital_gradient(S0, h0, g0)

    hf = hf * HA_BOHR_TO_EV_ANGSTROM
    pulay = pulay * HA_BOHR_TO_EV_ANGSTROM
    gradient = hf + pulay
    return ForceResult(forces=-gradient, hellmann_feynman=hf, pulay=pulay,
                       gradient=gradient,
                       n_electrons=float(np.real(np.trace(D1))), details=details)
