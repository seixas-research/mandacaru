# -*- coding: utf-8 -*-
# file: algorithms/orbital_response.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Orbital response of an active-space energy: the z-vector (Lagrangian).

An active-space VQE or ADAPT-VQE energy is written in the canonical
Hartree-Fock orbitals :math:`V(\mathbf R)` of each geometry (a frozen core, a
set of active orbitals, deleted virtuals), so its derivative has two parts:

.. math::

    \frac{dE}{dx} = \Big.\frac{\partial E}{\partial x}\Big|_{V}
    + \sum_{pq} \Omega_{pq}\,U^x_{pq},
    \qquad dV = V\,U^x,

with :math:`U^x` the first-order change of the orbitals.  The first term is
what :func:`~mandacaru.algorithms.pseudo_forces.pseudo_nuclear_gradient`
computes (Hellmann-Feynman and Pulay with the orbitals fixed in the Löwdin
frame).  The second is zero only when the state is stationary with respect to
orbital rotations -- a Hartree-Fock determinant or the exact eigenstate of the
orbital space -- and is *not* small for a truncated ADAPT-VQE state or whenever
virtual orbitals are deleted.

Work in the orthonormal (Löwdin) atomic-orbital frame with integrals
:math:`h', g'` (:math:`h'=X h X`, :math:`X=S^{-1/2}`) and the spin-summed
spatial RDMs :math:`D, G` of the state in the molecular orbitals.

**Orbital gradient.**  For :math:`V\to V(1+U)` with :math:`U` anti-Hermitian,
:math:`dE = 2\,\mathrm{Re}\sum_{tq} U_{tq}Y_{tq}`, where

.. math::

    Y_{tr} = \sum_p h_{pt}D_{pr} + \sum_{pqs} G_{pqrs}\langle pq|ts\rangle ,

and with the anti-Hermitian :math:`\Omega_{tq} = Y_{tq} - \overline{Y_{qt}}` the
response is :math:`2\,\mathrm{Re}\sum_{t<q}\Omega_{tq}U_{tq}`.

**Canonical orbitals.**  The orbitals diagonalize the Fock matrix, so
:math:`U_{pq} = F^x_{pq}/(\epsilon_q-\epsilon_p)` for :math:`p\ne q`, with the
Fock derivative :math:`F^x = F^{x}_{0} + G[P^x]` containing the response of
the density, :math:`P^x_{ai}=2U_{ai}`.  The occupied-virtual block is the
coupled-perturbed Hartree-Fock (CPHF) equation
:math:`L(U) = -F^x_{0,{\rm ov}}`, :math:`L(U)_{ai} = (\epsilon_a-\epsilon_i)U_{ai}
+ G[P^x(U)]_{ai}`.

**z-vector** (Handy and Schaefer 1984).  Define the Hermitian matrix
:math:`\Lambda'_{tq} = \Omega_{tq}/(\epsilon_q-\epsilon_t)` (zero on the
diagonal), the one-electron response :math:`q = (V^\dagger G[\tilde\Lambda']V)_{\rm ov}`
and solve :math:`L(z) = q` *once*; then

.. math::

    \sum_{pq}\Omega_{pq}U^x_{pq} = \sum_{pq}\Lambda_{pq}F^x_{0,pq},
    \qquad \Lambda = \Lambda' + \Lambda^{z},\quad
    \Lambda^z_{ai} = -2\bar z_{ai},\ \Lambda^z_{ia} = -2 z_{ai},

for every nuclear coordinate :math:`x` -- no CPHF per displacement.  Because
:math:`F_0` is linear in :math:`(h',g')`, the response is the *same* algebraic
energy with effective density matrices,

.. math::

    D^{\rm eff} = D + \Lambda ,\qquad
    \tfrac12\sum G^{\rm eff}_{pqrs}g_{pqrs} = \tfrac12\sum G_{pqrs}g_{pqrs}
    + \sum_{pq,i}\Lambda_{pq}\,(2\langle pi|qi\rangle-\langle pi|iq\rangle),

so the existing streamed derivative contraction of
:func:`~mandacaru.algorithms.pseudo_forces.pseudo_nuclear_gradient` (one pass
over the derivative integrals, never an :math:`M^4` array per atom) delivers
the whole gradient.  The cost beyond the unrelaxed force is one
:math:`O(M^5)` integral transformation and one CPHF solve, each
:math:`O(M^4)` per iteration; the RDMs are those of the active space only.

Canonical orbitals of a symmetric molecule can be degenerate; the response of
a degenerate pair is undefined (the canonical gauge is arbitrary) and is
harmless exactly when the energy does not depend on that rotation
(:math:`\Omega_{tq}=0`).  A nonzero :math:`\Omega` between degenerate orbitals
raises :class:`DegenerateOrbitalError`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Orbital-energy difference below which two canonical orbitals are degenerate.
DEGENERACY_TOLERANCE = 1e-6
#: Largest ``|Omega|`` tolerated between degenerate orbitals.
DEGENERATE_GRADIENT_TOLERANCE = 1e-6
#: Orbital-energy difference (Ha) below which a pair that carries density is
#: reported as nearly degenerate: its mixing under a finite nuclear step is not
#: perturbative, so the derivative holds only for steps much smaller than
#: ``|F^x_pq| / gap`` and a finite displacement sees something else.
NEAR_DEGENERACY = 2e-3
#: Largest off-diagonal Fock element (Ha) of the reference orbitals.
SCF_TOLERANCE = 1e-5
#: Convergence of the CPHF (z-vector) solve, on the residual norm.
CPHF_TOLERANCE = 1e-10


class OrbitalResponseUnavailable(RuntimeError):
    """The z-vector cannot be formed for this reference (the caller falls
    back to the unrelaxed gradient and a finite-difference response)."""


class DegenerateOrbitalError(OrbitalResponseUnavailable):
    """The state depends on the rotation of degenerate canonical orbitals."""


@dataclass
class OrbitalResponse:
    """The effective RDMs of the relaxed energy and what was measured."""

    D: np.ndarray                     # (M, M) effective spatial one-RDM
    G: np.ndarray                     # (M, M, M, M) effective two-RDM
    orbital_gradient: float           # max |Omega|, Ha
    scf_residual: float               # max off-diagonal Fock element, Ha
    cphf_iterations: int
    cphf_residual: float
    degenerate_residual: float        # max |Omega| across degenerate pairs
    near_degenerate: tuple = ()       # (p, q, gap) of nearly degenerate pairs


def lowdin_frame(S: np.ndarray, h: np.ndarray, g: np.ndarray):
    r"""``(h', g')`` in the Löwdin frame, :math:`X=S^{-1/2}` on every index.

    ``g`` is the dense physicist tensor :math:`g_{abcd}=\langle ab|cd\rangle`
    of :class:`~mandacaru.algorithms.pseudo_forces.AlgebraicEnergy`.
    """
    S = 0.5 * (S + S.conj().T)
    w, U = np.linalg.eigh(S)
    X = (U * (1.0 / np.sqrt(w))) @ U.conj().T
    return transform(h, g, X)


def transform(h: np.ndarray, g: np.ndarray, A: np.ndarray):
    """``h_mo = A^H h A`` and ``g_mo`` with ``A^*`` on the bra indices."""
    h_mo = A.conj().T @ h @ A
    g_mo = np.einsum("ap,abcd->pbcd", A.conj(), g, optimize=True)
    g_mo = np.einsum("bq,pbcd->pqcd", A.conj(), g_mo, optimize=True)
    g_mo = np.einsum("cr,pqcd->pqrd", A, g_mo, optimize=True)
    g_mo = np.einsum("ds,pqrd->pqrs", A, g_mo, optimize=True)
    return h_mo, g_mo


def _fock_response(g: np.ndarray, P: np.ndarray) -> np.ndarray:
    r""":math:`G[P] = J[P] - \tfrac12 K[P]` of a Hermitian matrix ``P``
    (spin-summed density convention :math:`P=2\sum_i v_i v_i^\dagger`)."""
    J = np.einsum("acbd,dc->ab", g, P, optimize=True)
    K = np.einsum("acdb,dc->ab", g, P, optimize=True)
    return J - 0.5 * K


def _partial(V: np.ndarray, g: np.ndarray, rows, b, c, d) -> np.ndarray:
    """``g`` transformed with ``V[:, rows]`` on the first index and the
    index sets ``b, c, d`` on the others (bra indices conjugated)."""
    Vc = V.conj()
    out = np.einsum("ap,abcd->pbcd", Vc[:, rows], g, optimize=True)
    out = np.einsum("bq,pbcd->pqcd", Vc[:, b], out, optimize=True)
    out = np.einsum("cr,pqcd->pqrd", V[:, c], out, optimize=True)
    return np.einsum("ds,pqrd->pqrs", V[:, d], out, optimize=True)


def orbital_gradient_matrix(h_mo: np.ndarray, D: np.ndarray, G: np.ndarray,
                            V: np.ndarray, g: np.ndarray, support
                            ) -> np.ndarray:
    r""":math:`\Omega` (anti-Hermitian, ``(M, M)``) of the energy
    :math:`\sum D h + \tfrac12\sum G g` for rotations of the orbitals ``V``.

    ``g`` is the Löwdin-frame tensor; ``support`` the orbitals on which the
    RDMs live (frozen and active), so that the transformed tensor is
    ``(|support|, |support|, M, |support|)``.
    """
    M = h_mo.shape[0]
    support = list(support)
    Y = np.einsum("pt,pr->tr", h_mo, D)
    # g_{pqts}: p, q, s in the support, t free.
    g4 = _partial(V, g, support, support, list(range(M)), support)
    Gs = G[np.ix_(support, support, support, support)]
    Y2 = np.einsum("pqrs,pqts->tr", Gs, g4, optimize=True)
    Y[:, support] += Y2
    return Y - Y.conj().T


def response_densities(h: np.ndarray, g: np.ndarray, V: np.ndarray,
                       D: np.ndarray, G: np.ndarray, *, n_occ: int,
                       support, frame: bool = True,
                       S: np.ndarray | None = None) -> OrbitalResponse:
    r"""Effective RDMs :math:`D+\Lambda`, :math:`G+G^\Lambda` of the relaxed energy.

    Parameters
    ----------
    h, g : ndarray
        Atomic-orbital one-body matrix and dense two-body tensor
        (:math:`g_{abcd}=\langle ab|cd\rangle`); with ``frame=True`` they are
        orthonormalized with ``S`` here, else they already are in the Löwdin
        frame.
    V : ndarray
        Canonical molecular orbitals in the Löwdin frame, columns ordered
        as the RDMs.
    D, G : ndarray
        Spin-summed spatial RDMs of the state in those orbitals over all
        ``M`` orbitals (frozen core filled, deleted virtuals empty).
    n_occ : int
        Number of doubly occupied orbitals of the Hartree-Fock reference.
    support : sequence of int
        Orbitals carrying density (frozen and active).
    """
    if frame:
        if S is None:
            raise ValueError("an atomic-orbital frame needs the overlap S")
        h, g = lowdin_frame(S, h, g)
    h = 0.5 * (h + h.conj().T)
    M = h.shape[0]
    occ = list(range(n_occ))
    virt = list(range(n_occ, M))
    h_mo = V.conj().T @ h @ V

    # The Fock matrix of the reference determinant in its own orbitals.
    coulomb = _partial(V, g, list(range(M)), occ, list(range(M)), occ)
    exchange = _partial(V, g, list(range(M)), occ, occ, list(range(M)))
    F = (h_mo + 2.0 * np.einsum("piqi->pq", coulomb)
         - np.einsum("piiq->pq", exchange))
    del coulomb, exchange
    eps = np.real(np.diag(F)).copy()
    scf_residual = float(np.abs(F - np.diag(np.diag(F))).max())
    if scf_residual > SCF_TOLERANCE:
        raise OrbitalResponseUnavailable(
            f"the reference orbitals are not canonical Hartree-Fock orbitals "
            f"(largest off-diagonal Fock element {scf_residual:.1e} Ha); the "
            "orbital response of the active-space force would be wrong")

    Omega = orbital_gradient_matrix(h_mo, D, G, V, g, support)
    delta = eps[None, :] - eps[:, None]                      # eps_q - eps_t
    degenerate = np.abs(delta) < DEGENERACY_TOLERANCE
    np.fill_diagonal(degenerate, True)
    stuck = float(np.abs(Omega[degenerate]).max(initial=0.0))
    if stuck > DEGENERATE_GRADIENT_TOLERANCE:
        raise DegenerateOrbitalError(
            "the state depends on the rotation of degenerate canonical "
            f"orbitals (|dE/dkappa| = {stuck:.1e} Ha between orbitals of "
            "equal energy): their canonical gauge is undefined, so the "
            "response of the orbitals has no unique value")
    Lam = np.where(degenerate, 0.0,
                   Omega / np.where(degenerate, 1.0, delta))

    # CPHF operator on the occupied-virtual block.
    Vo, Va = V[:, occ], V[:, virt]
    gap = eps[virt][:, None] - eps[occ][None, :]

    def response(U):
        X = Va @ U @ Vo.conj().T
        Pdx = 2.0 * (X + X.conj().T)
        Gm = _fock_response(g, Pdx)
        return Va.conj().T @ Gm @ Vo

    def operator(U):
        return gap * U + response(U)

    Vlam = V @ Lam.T @ V.conj().T                             # tilde Lambda
    Hq = V.conj().T @ _fock_response(g, Vlam) @ V
    q = Hq[np.ix_(virt, occ)]
    z, iterations, residual = _conjugate_gradient(operator, q, gap)

    full = Lam.copy()
    full[np.ix_(virt, occ)] += -2.0 * z.conj()
    full[np.ix_(occ, virt)] += -2.0 * z.T

    D_eff = D + full
    G_eff = np.array(G, dtype=complex if np.iscomplexobj(G) or
                     np.iscomplexobj(full) else float, copy=True)
    Gl = np.zeros_like(G_eff)
    for i in occ:
        Gl[:, i, :, i] += 4.0 * full
        Gl[:, i, i, :] -= 2.0 * full
    G_eff += 0.5 * (Gl + Gl.transpose(1, 0, 3, 2))
    near = []
    carriers = {p for p in range(M) if abs(D[p, p] - 2.0) > 1e-9
                and abs(D[p, p]) > 1e-9}
    for p in range(M):
        for q in range(p + 1, M):
            gap = abs(eps[q] - eps[p])
            if gap < NEAR_DEGENERACY and (p in carriers or q in carriers):
                near.append((p, q, float(gap)))
    return OrbitalResponse(
        D=_real_if_possible(D_eff), G=_real_if_possible(G_eff),
        near_degenerate=tuple(near),
        orbital_gradient=float(np.abs(Omega).max(initial=0.0)),
        scf_residual=scf_residual, cphf_iterations=iterations,
        cphf_residual=residual, degenerate_residual=stuck)


def _real_if_possible(x: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(x) and np.abs(x.imag).max(initial=0.0) <= 1e-13 * max(
            np.abs(x.real).max(initial=0.0), 1.0):
        return np.ascontiguousarray(x.real)
    return x


def _conjugate_gradient(operator, rhs, gap, tol: float = CPHF_TOLERANCE,
                        max_iter: int = 200):
    """Solve ``operator(z) = rhs`` for a real-linear, self-adjoint,
    positive-definite operator on complex arrays (real inner product),
    preconditioned by the orbital-energy gap."""
    def dot(a, b):
        return float(np.real(np.vdot(a, b)))

    x = rhs / gap
    r = rhs - operator(x)
    zv = r / gap
    p = zv.copy()
    rz = dot(r, zv)
    norm = np.sqrt(dot(rhs, rhs)) or 1.0
    residual = np.sqrt(dot(r, r)) / norm
    iteration = 0
    while residual > tol and iteration < max_iter:
        Ap = operator(p)
        alpha = rz / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        residual = np.sqrt(dot(r, r)) / norm
        iteration += 1
        zv = r / gap
        rz_new = dot(r, zv)
        p = zv + (rz_new / rz) * p
        rz = rz_new
    if residual > max(tol * 1e3, 1e-8):
        raise OrbitalResponseUnavailable(
            f"the z-vector (CPHF) solve did not converge: relative residual "
            f"{residual:.1e} after {iteration} iterations")
    return x, iteration, float(residual)
