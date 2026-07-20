# -*- coding: utf-8 -*-
# file: algorithms/_jax_energy.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Differentiable algebraic layer: energy as a function of the raw AO integrals.

This module answers one question with automatic differentiation:

.. math::

    \frac{\partial E}{\partial S_{\mu\nu}}, \quad
    \frac{\partial E}{\partial h_{\mu\nu}}, \quad
    \frac{\partial E}{\partial \langle\mu\nu|\lambda\sigma\rangle}

for the energy the driver actually reports.  Those three derivatives are the
missing half of the nuclear gradient: combined with the *integral* derivatives
:math:`\partial S/\partial\mathbf R` etc. (which come analytically from the
grid-sampled orbital derivatives, see :mod:`carcara.algorithms.forces`), the
chain rule gives the full force including **Pulay** terms.

Why automatic differentiation here, and only here
-------------------------------------------------
Between the raw atomic-orbital integrals and the energy sits a stack of dense
linear algebra: Löwdin symmetric orthogonalization :math:`X = S^{-1/2}`, the
restricted Hartree-Fock SCF that defines the molecular orbitals, the four-index
integral transform, the frozen-core reduction, and the spin-blocking.  Every one
of those steps *responds* when the nuclei move, and hand-deriving that response
(the energy-weighted density matrix, the derivative of a matrix inverse square
root, the coupled-perturbed SCF equations) is the classic error-prone part of an
analytic-gradient implementation.

It is also **small**: these tensors are :math:`M\times M` and :math:`M^4` for a
handful of orbitals, not grid-sized.  Differentiating them costs microseconds and
no meaningful memory.  So this layer -- and deliberately *not* the integral
engine, which is an OpenMP C library behind ``ctypes`` and inherently opaque to a
tracer -- is where JAX belongs.

The function mirrors :class:`~carcara.core.hamiltonian.MolecularIntegrals`
step for step; :func:`energy_from_integrals` is checked against the driver's own
energy in the test suite, which is what guarantees the derivatives describe the
right function.

Requires ``jax``; :func:`jax_available` reports whether it can be imported.
"""

from __future__ import annotations

import numpy as np


def jax_available() -> bool:
    """True when JAX can be imported (the differentiable layer is usable)."""
    try:
        import jax  # noqa: F401
    except Exception:
        return False
    return True


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "analytic nuclear gradients use JAX to differentiate the algebraic "
            "layer (Loewdin orthogonalization, SCF, integral transforms); "
            "install it with `pip install jax`") from exc
    # Gradients of a quantum-chemical energy are meaningless in float32.
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


# --------------------------------------------------------------------------- #
# The energy functional, written in JAX.
# --------------------------------------------------------------------------- #

#: Newton-Schulz iterations for the inverse square root (see `_inverse_sqrt`).
NEWTON_SCHULZ_STEPS = 60


def _inverse_sqrt(jnp, S, steps: int = NEWTON_SCHULZ_STEPS):
    r"""Löwdin :math:`S^{-1/2}`, computed **without** an eigendecomposition.

    The obvious implementation diagonalizes ``S`` and inverts the square roots of
    its eigenvalues.  That is fine for the *value* but breaks for the
    *derivative*: the gradient of ``eigh`` contains factors
    :math:`1/(\lambda_i - \lambda_j)`, so a degenerate overlap eigenvalue makes it
    ``NaN``.  Degeneracies are not exotic here -- they are forced by symmetry.
    Water in a minimal basis has an exactly doubly-degenerate overlap eigenvalue
    from the oxygen :math:`2p_x`/:math:`2p_y` pair, which is enough to poison the
    whole gradient.

    The degeneracy is an artefact of the eigenvector parameterization, not of the
    matrix function: :math:`S \mapsto S^{-1/2}` is perfectly smooth there.  So we
    compute it with the coupled Newton-Schulz iteration

    .. math::

        T_k = \tfrac12\,(3I - Z_k Y_k), \qquad
        Y_{k+1} = Y_k T_k, \qquad
        Z_{k+1} = T_k Z_k ,

    which uses only matrix products and therefore differentiates cleanly.
    ``S`` is pre-scaled by its Frobenius norm so the iteration's convergence
    condition holds, and the scaling is undone at the end.
    """
    scale = jnp.sqrt(jnp.sum(S * S))                 # Frobenius norm
    A = S / scale
    identity = jnp.eye(S.shape[0], dtype=S.dtype)
    Y, Z = A, identity
    for _ in range(steps):
        T = 0.5 * (3.0 * identity - Z @ Y)
        Y, Z = Y @ T, T @ Z
    # Z -> A^{-1/2}; undo the scaling: S^{-1/2} = A^{-1/2} / sqrt(scale).
    return Z / jnp.sqrt(scale)


def _transform_eri(jnp, eri, C):
    """``eri_pqrs -> sum C*_ap C*_bq C_cr C_ds <ab|cd>`` (physicists')."""
    return jnp.einsum("ap,bq,cr,ds,abcd->pqrs", C, C, C, C, eri,
                      optimize=True)


#: Density-matrix convergence threshold for the SCF probe below.
SCF_DENSITY_TOL = 1e-10
#: Hard cap on the probe's iteration search.
SCF_ITERATION_CAP = 2000
#: Extra iterations unrolled past the probe's answer, so the differentiated
#: iterations are unambiguously no-ops rather than merely nearly so.
SCF_ITERATION_MARGIN = 20


def scf_iterations_required(h, eri, n_occ, tol: float = SCF_DENSITY_TOL,
                            cap: int = SCF_ITERATION_CAP):
    """Iterations this RHF needs to converge, in plain NumPy; ``None`` if it can't.

    The unrolled :func:`_scf` is only a valid *derivative* if the last iterations
    are genuinely no-ops -- see the warning there.  Whether that holds is a
    property of the integrals, not something the caller can know in advance, so
    it is measured here on the same fixed-point iteration.  This runs outside the
    JAX trace (it is a plain loop with a data-dependent break), which is exactly
    why it cannot live inside ``_scf`` itself.
    """
    h = np.asarray(np.real(h), dtype=float)
    eri = np.asarray(np.real(eri), dtype=float)
    _w, C = np.linalg.eigh(h)
    for iteration in range(int(cap)):
        occ = C[:, :n_occ]
        D = 2.0 * (occ @ occ.T)
        J = np.einsum("rs,prqs->pq", D, eri, optimize=True)
        K = np.einsum("rs,prsq->pq", D, eri, optimize=True)
        F = h + J - 0.5 * K
        F = 0.5 * (F + F.T)
        _e, C = np.linalg.eigh(F)
        D_new = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
        if iteration > 2 and np.abs(D_new - D).max() < tol:
            return iteration + 1
    return None


def _scf(jnp, h, eri, n_occ, n_iter):
    """Unrolled closed-shell RHF; returns the molecular-orbital coefficients.

    A fixed iteration count keeps the computation a straight-line function of the
    integrals, so it differentiates without an implicit-function rule.  At
    convergence the extra iterations are no-ops and the derivative is the
    converged (coupled-perturbed) response.

    .. warning::

       That last sentence is a *premise*, not a guarantee, and it fails silently.
       If ``n_iter`` stops the iteration while the density is still moving, the
       derivative is taken through a transient and can be enormous -- H2O on a
       pseudopotential grid sat on a metastable plateau for ~19 iterations, began
       escaping around iteration 20, and was still moving at 45; differentiating
       the 40-iteration unroll gave gradients of order 1e8 Ha/Bohr, while 10 or
       80 iterations both gave sane O(1) answers.  Nothing about the result looked
       wrong -- the forces were finite, and the spurious component sat in a
       direction that symmetry forced to zero.

       Callers must therefore size ``n_iter`` with
       :func:`scf_iterations_required` rather than trusting a fixed default.
    """
    _w, C = jnp.linalg.eigh(h)                      # core-Hamiltonian guess
    for _ in range(n_iter):
        occ = C[:, :n_occ]
        D = 2.0 * (occ @ occ.T)
        J = jnp.einsum("rs,prqs->pq", D, eri, optimize=True)
        K = jnp.einsum("rs,prsq->pq", D, eri, optimize=True)
        F = h + J - 0.5 * K
        F = 0.5 * (F + F.T)
        _w, C = jnp.linalg.eigh(F)
    return C


def _freeze_core(jnp, h_mo, eri_mo, frozen, active):
    """Frozen-core reduction: constant core energy + effective one-body field.

    Mirrors :func:`carcara.core.hamiltonian.freeze_core_integrals`.
    """
    if not frozen:
        return h_mo, eri_mo, 0.0
    frozen = list(frozen)
    active = list(active)

    core_energy = 0.0
    for i in frozen:
        core_energy = core_energy + 2.0 * h_mo[i, i]
    for i in frozen:
        for j in frozen:
            core_energy = core_energy + (2.0 * eri_mo[i, j, i, j]
                                         - eri_mo[i, j, j, i])

    idx = jnp.array(active)
    h_eff = h_mo[jnp.ix_(idx, idx)]
    for i in frozen:
        j_term = eri_mo[jnp.ix_(idx, jnp.array([i]), idx, jnp.array([i]))]
        k_term = eri_mo[jnp.ix_(idx, jnp.array([i]), jnp.array([i]), idx)]
        h_eff = h_eff + 2.0 * j_term[:, 0, :, 0] - k_term[:, 0, 0, :]
    eri_act = eri_mo[jnp.ix_(idx, idx, idx, idx)]
    return h_eff, eri_act, core_energy


def _spin_block(jnp, h, eri):
    """Spatial -> spin-orbital integrals, alpha block then beta block."""
    M = h.shape[0]
    zeros = jnp.zeros((M, M), dtype=h.dtype)
    h_so = jnp.block([[h, zeros], [zeros, h]])

    # <pq|rs> is non-zero only when spin(p)==spin(r) and spin(q)==spin(s).
    same = jnp.eye(2)
    # g_so[(sp,p),(sq,q),(sr,r),(ss,s)] = delta_{sp,sr} delta_{sq,ss} eri[p,q,r,s]
    g_so = jnp.einsum("pqrs,ac,bd->apbqcrds", eri, same, same,
                      optimize=True)
    n = 2 * M
    return h_so, g_so.reshape(n, n, n, n)


def energy_from_integrals(S, h_ao, eri_ao, gamma, gamma2, *, n_electrons,
                          frozen=(), nuclear_repulsion=0.0, scf_iterations=40,
                          orthogonalize=True, mo_basis=True):
    r"""Total energy from the **raw AO integrals**, with the RDMs held fixed.

    Reproduces the driver's pipeline: Löwdin orthogonalization, RHF molecular
    orbitals, the integral transform, the frozen-core reduction, spin-blocking,
    and finally the RDM contraction
    :math:`E = \sum \gamma_{pq}h_{pq} + \tfrac12\sum\Gamma_{pqrs}\langle pq|rs\rangle`
    plus the constant core and nuclear-repulsion shifts.

    Written entirely in ``jax.numpy``, so :func:`integral_gradients` can
    differentiate it with respect to ``S``, ``h_ao`` and ``eri_ao``.
    """
    _jax, jnp = _require_jax()
    S = jnp.asarray(np.real(S))
    h = jnp.asarray(np.real(h_ao))
    eri = jnp.asarray(np.real(eri_ao))
    gamma = jnp.asarray(np.real(gamma))
    gamma2 = jnp.asarray(np.real(gamma2))

    if orthogonalize:
        X = _inverse_sqrt(jnp, S)
        h = X.T @ h @ X
        eri = _transform_eri(jnp, eri, X)

    M = h.shape[0]
    if mo_basis:
        C = _scf(jnp, h, eri, n_electrons // 2, scf_iterations)
        h = C.T @ h @ C
        eri = _transform_eri(jnp, eri, C)

    frozen = sorted(int(i) for i in frozen)
    active = [p for p in range(M) if p not in frozen]
    h, eri, core_energy = _freeze_core(jnp, h, eri, frozen, active)

    h_so, g_so = _spin_block(jnp, h, eri)
    one = jnp.einsum("pq,pq->", gamma, h_so)
    two = 0.5 * jnp.einsum("pqrs,pqrs->", gamma2, g_so)
    return one + two + core_energy + nuclear_repulsion


def _resolve_scf_iterations(S, h_ao, eri_ao, kwargs) -> int:
    """Unroll length long enough that the differentiated SCF has converged.

    Measures the requirement with :func:`scf_iterations_required` on the same
    Löwdin-orthogonalized integrals :func:`_scf` will see, and returns the larger
    of that (plus a margin) and whatever the caller asked for.  The caller's
    ``scf_iterations`` is therefore a *floor*, not a ceiling -- silently
    differentiating an unconverged SCF is the failure this exists to prevent.
    """
    requested = int(kwargs.get("scf_iterations", 40))
    if not kwargs.get("mo_basis", True):
        return requested                        # no SCF to converge

    h = np.real(np.asarray(h_ao, dtype=float))
    eri = np.real(np.asarray(eri_ao, dtype=float))
    if kwargs.get("orthogonalize", True):
        w, U = np.linalg.eigh(np.real(np.asarray(S, dtype=float)))
        X = (U * (1.0 / np.sqrt(w))) @ U.T
        h = X.T @ h @ X
        eri = np.einsum("ap,bq,cr,ds,abcd->pqrs", X, X, X, X, eri,
                        optimize=True)

    n_occ = int(kwargs["n_electrons"]) // 2
    needed = scf_iterations_required(h, eri, n_occ)
    if needed is None:
        raise RuntimeError(
            "the RHF reference used by the nuclear gradient did not converge "
            f"within {SCF_ITERATION_CAP} iterations, so the gradient through it "
            "would be meaningless. This usually means the geometry is far from "
            "any bound minimum or the grid is too coarse to represent it.")
    return max(requested, needed + SCF_ITERATION_MARGIN)


def integral_gradients(S, h_ao, eri_ao, gamma, gamma2, **kwargs):
    r"""``(dE/dS, dE/dh, dE/d<..>)`` at fixed RDMs, by automatic differentiation.

    Returns three NumPy arrays shaped like ``S``, ``h_ao`` and ``eri_ao``.

    * ``dE/dh`` is the AO-basis one-particle density matrix;
    * ``dE/d<..>`` is (half) the AO-basis two-particle density matrix;
    * ``dE/dS`` is the **orthonormality-constraint** term -- the derivative that
      flows through :math:`S^{-1/2}` and the SCF.  It is precisely the piece a
      hand-written gradient has to reconstruct as an energy-weighted density
      matrix, and getting it from AD is why this layer exists.

    Contracted with the corresponding integral derivatives with respect to the
    nuclear coordinates, these give the Pulay contribution to the force.
    """
    jax, _jnp = _require_jax()
    kwargs = dict(kwargs)
    kwargs["scf_iterations"] = _resolve_scf_iterations(S, h_ao, eri_ao, kwargs)

    def energy(S_, h_, eri_):
        return energy_from_integrals(S_, h_, eri_, gamma, gamma2, **kwargs)

    grads = jax.grad(energy, argnums=(0, 1, 2))(
        np.real(np.asarray(S)), np.real(np.asarray(h_ao)),
        np.real(np.asarray(eri_ao)))
    return tuple(np.asarray(g, dtype=float) for g in grads)
