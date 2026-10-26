r"""Stability of a converged spin-unrestricted Kohn-Sham state.

A converged SCF is a stationary point, not necessarily a minimum.  At integer
occupations the state is a minimum when the real orbital-rotation Hessian
:math:`H = \partial^2 E/\partial\kappa\,\partial\kappa`
(occupied-to-virtual rotations of both spins, with the full density
response) has no negative eigenvalue.  Its lowest eigenpair comes from a
Davidson iteration on Hessian-vector products, each one finite difference of
the orbital gradient :math:`g_{ai} = 2\,\mathrm{Re}\,F_{ai}` (one Fock
build, no Hessian matrix, no two-body tensor).  A negative eigenvalue is
followed: the orbitals are rotated along the mode, the best step kept, and the
SCF re-converged from there, until the Hessian is positive or the rounds run
out.  The exactly equilateral H3 doublet is the case: its integer-occupation
SCF stopped at a symmetric saddle 650 meV above the doublet and reported
convergence.

A degenerate shell at a Fermi level that the SCF fills equally
(:func:`~mandacaru.algorithms.dft.shell_occupations`) is an ensemble by
design (the OH radical's beta pi has the symmetric density of its
degenerate state); the check leaves it alone and says so.

The same Hessian products drive :func:`second_order`, the Newton-CG
rescue of an integer-occupation SCF whose plain loop has stalled.

The check costs a few tens of Fock builds, a fraction of the SCF itself.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..utils.logging import scf_record

#: Hessian eigenvalue (Hartree per radian squared) below which a state counts
#: as unstable.  The Hessian's entries are of the order of the orbital gaps
#: (0.1 to 1 Hartree); the finite-difference products carry ~1e-5.
STABILITY_THRESHOLD = -2e-4

#: Hessian-vector products one check may spend (one Fock build each).
STABILITY_MAX_PRODUCTS = 30

#: Re-optimizations along unstable modes before the state is reported
#: unstable.
STABILITY_MAX_ROUNDS = 4

#: Orbital-gradient finite-difference step (radians along a unit mode).
STABILITY_STEP = 2e-4

#: Rotation angles (radians along the unit mode) tried before re-converging.
DESCENT_ANGLES = (0.15, -0.15, 0.4, -0.4)

#: Energy (Hartree) a re-converged state must lie below the old one to
#: replace it.
LOWERING = 1e-7


@dataclass
class SCFState:
    """One spin-unrestricted SCF state: densities, energy and the iteration
    record."""

    Da: np.ndarray
    Db: np.ndarray
    energy: float
    converged: bool
    iterations: int
    history: list
    #: Fractional occupations of each channel's canonical orbitals, or None.
    fa: np.ndarray | None = None
    fb: np.ndarray | None = None

    @property
    def integer(self) -> bool:
        return self.fa is None and self.fb is None


@dataclass
class StabilityReport:
    """What the stability check found and did (reported once, in
    ``[SCF SUMMARY]``)."""

    #: ``"stable"``, ``"ensemble"`` (a partly filled degenerate shell,
    #: not checked), ``"lowered"`` (a lower state replaced the converged
    #: one) or ``"unstable"`` (a negative mode no re-optimization removed).
    outcome: str
    #: Lowest Hessian eigenvalue of the final state (Hartree per radian
    #: squared), ``None`` for an ensemble (not defined).
    lowest_eigenvalue: float | None = None
    products: int = 0
    rounds: int = 0
    #: Hartree the check lowered the energy by (zero when it stayed).
    energy_lowering: float = 0.0
    #: Free text: what was split or followed.
    detail: str = ""

    def __str__(self) -> str:
        text = {"stable": "stable", "ensemble": "ensemble kept",
                "lowered": "lower state found", "unstable": "unstable"}[
                    self.outcome]
        if self.lowest_eigenvalue is not None:
            text += (f" (lowest orbital-rotation Hessian eigenvalue "
                     f"{self.lowest_eigenvalue:.3e} Hartree/rad^2, "
                     f"{self.products} products)")
        if self.detail:
            text += "; " + self.detail
        return text


def _adapted_orbitals(F, D, n):
    """Orthonormal orbitals whose first ``n`` span the occupied space of the
    projector ``D``, each block diagonalizing ``F``; and their energies.
    For a converged state these are the canonical orbitals."""
    weights, U = np.linalg.eigh(D)
    C = U[:, np.argsort(-weights)]
    mo = C.conj().T @ F @ C
    eo, Vo = np.linalg.eigh(mo[:n, :n])
    ev, Vv = np.linalg.eigh(mo[n:, n:])
    return (np.concatenate([np.real(eo), np.real(ev)]),
            np.hstack([C[:, :n] @ Vo, C[:, n:] @ Vv]))


class _Rotations:
    """Occupied-to-virtual rotations of the canonical orbitals of a state."""

    def __init__(self, solver, state: SCFState):
        self.solver = solver
        self.n = (solver.n_alpha, solver.n_beta)
        self.M = solver.M
        Fa, Fb = solver._fock_pair(state.Da, state.Db)[:2]
        self.eps, self.C = [], []
        for F, D, n in zip((Fa, Fb), (state.Da, state.Db), self.n):
            eps, C = _adapted_orbitals(F, D, n)
            self.eps.append(eps)
            self.C.append(C)
        self.sizes = [n * (self.M - n) for n in self.n]
        self.dim = sum(self.sizes)

    def split(self, x):
        out, start = [], 0
        for n, size in zip(self.n, self.sizes):
            out.append(x[start:start + size].reshape(self.M - n, n))
            start += size
        return out

    def diagonal(self):
        """Hessian diagonal estimate ``2 (eps_a - eps_i)``, spin by spin."""
        parts = []
        for n, eps in zip(self.n, self.eps):
            parts.append((2.0 * (eps[n:, None] - eps[None, :n])).ravel())
        return np.concatenate(parts) if parts else np.zeros(0)

    def orbitals(self, x):
        """Rotated orbitals ``C exp(K(x))`` of both spins."""
        rotated = []
        for C, n, kappa in zip(self.C, self.n, self.split(x)):
            if n == 0 or n == self.M:
                rotated.append(C)
                continue
            K = np.zeros((self.M, self.M))
            K[n:, :n] = kappa
            K[:n, n:] = -kappa.T
            rotated.append(C @ expm(K))
        return rotated

    def densities(self, x):
        Ca, Cb = self.orbitals(x)
        return (self.solver._spin_density(Ca, self.n[0]),
                self.solver._spin_density(Cb, self.n[1]), Ca, Cb)

    def gradient(self, x):
        """``(g, energy)``: the orbital gradient ``2 Re F_ai`` in the frame
        of the rotated orbitals, and the energy there."""
        Da, Db, Ca, Cb = self.densities(x)
        Fa, Fb, energy = self.solver._fock_pair(Da, Db)[:3]
        parts = []
        for F, C, n in zip((Fa, Fb), (Ca, Cb), self.n):
            mo = C.conj().T @ F @ C
            parts.append((2.0 * np.real(mo[n:, :n])).ravel())
        return np.concatenate(parts), float(energy)


def lowest_mode(rotations: _Rotations, g0: np.ndarray,
                max_products: int = STABILITY_MAX_PRODUCTS,
                threshold: float = STABILITY_THRESHOLD):
    """Lowest eigenpair of the orbital Hessian by Davidson's method on
    finite-difference Hessian-vector products.

    ``g0`` is the orbital gradient of the converged state (zero up to the
    SCF tolerance); products are ``(g(eps v) - g0) / eps``.  Returns
    ``(eigenvalue, mode, products)``; the iteration stops at a converged
    Ritz pair, as soon as the Ritz value lies below ``threshold`` (it is an
    upper bound of the lowest eigenvalue, so the instability is proven), or
    when the products run out.
    """
    dim = rotations.dim
    diag = rotations.diagonal()
    step = STABILITY_STEP

    def product(v):
        return (rotations.gradient(step * v)[0] - g0) / step

    # Start with the unit vectors of the smallest diagonal entries.
    k = min(4, dim)
    seeds = np.argsort(diag)[:k]
    basis, images = [], []
    for index in seeds:
        v = np.zeros(dim)
        v[index] = 1.0
        _append(basis, images, v, product)
    products = len(basis)
    value, mode = np.inf, None
    while True:
        V, HV = np.array(basis).T, np.array(images).T
        T = V.T @ HV
        w, y = np.linalg.eigh(0.5 * (T + T.T))
        value = float(w[0])
        mode = V @ y[:, 0]
        residual = HV @ y[:, 0] - value * mode
        norm = float(np.linalg.norm(residual))
        if (value < threshold or norm < 1e-3 * max(1.0, abs(value))
                or products >= max_products or len(basis) >= dim):
            return value, mode / np.linalg.norm(mode), products
        denominator = diag - value
        denominator = np.where(np.abs(denominator) < 1e-2,
                               np.where(denominator < 0, -1e-2, 1e-2),
                               denominator)
        if not _append(basis, images, residual / denominator, product):
            return value, mode / np.linalg.norm(mode), products
        products += 1


def _append(basis, images, v, product) -> bool:
    """Orthonormalize ``v`` against ``basis`` and add it with its image;
    ``False`` when nothing new is left."""
    for _ in range(2):
        for b in basis:
            v = v - (b @ v) * b
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return False
    v = v / norm
    basis.append(v)
    images.append(product(v))
    return True


def stabilize(solver, state: SCFState, *, max_iter: int, tol: float,
              diis: bool) -> tuple[SCFState, StabilityReport]:
    """Check ``state`` (converged) and re-optimize it while it is not a
    minimum; returns the final state and the report."""
    start_energy = state.energy
    history = list(state.history)           # every iteration run, kept or not
    total = state.iterations
    report_detail = ""
    if not state.integer:
        return state, StabilityReport(
            "ensemble", detail="a partly filled degenerate shell is an "
            "ensemble by design; the orbital Hessian is not applied")
    rounds, products, eigenvalue = 0, 0, None
    while True:
        rotations = _Rotations(solver, state)
        if rotations.dim == 0:
            eigenvalue = None
            break
        g0, _ = rotations.gradient(np.zeros(rotations.dim))
        eigenvalue, mode, used = lowest_mode(rotations, g0)
        products += used
        if eigenvalue >= STABILITY_THRESHOLD:
            break
        if rounds >= STABILITY_MAX_ROUNDS:
            break
        rounds += 1
        candidate = _follow(solver, rotations, mode, max_iter=max_iter,
                            tol=tol, diis=diis, offset=total)
        history += candidate.history
        total = candidate.iterations
        if not candidate.converged \
                or candidate.energy >= state.energy - LOWERING:
            report_detail = (report_detail + "; " if report_detail else "") \
                + "following the unstable mode found no lower converged state"
            break
        state = candidate
    lowering = start_energy - state.energy
    if eigenvalue is not None and eigenvalue < STABILITY_THRESHOLD:
        outcome = "unstable"
        warnings.warn(
            f"the Kohn-Sham state is a saddle point: the lowest "
            f"orbital-rotation Hessian eigenvalue is {eigenvalue:.3e} "
            f"Hartree/rad^2 and re-optimization along the mode found no "
            f"lower converged state", RuntimeWarning, stacklevel=4)
    else:
        outcome = "lowered" if lowering > LOWERING else "stable"
    if rounds and outcome == "lowered":
        report_detail = (report_detail + "; " if report_detail else "") \
            + (f"{rounds} re-optimization{'s' if rounds > 1 else ''} along "
               f"unstable orbital-rotation modes")
    state.history, state.iterations = history, total
    return state, StabilityReport(outcome, eigenvalue, products, rounds,
                                  max(lowering, 0.0), report_detail)


def _follow(solver, rotations: _Rotations, mode, *, max_iter: int,
            tol: float, diis: bool, offset: int) -> SCFState:
    """Rotate along ``mode`` by the best of :data:`DESCENT_ANGLES` and
    re-converge."""
    best = None
    for angle in DESCENT_ANGLES:
        _g, energy = rotations.gradient(angle * mode)
        if best is None or energy < best[0]:
            best = (energy, angle)
    Da, Db, _Ca, _Cb = rotations.densities(best[1] * mode)
    return solver._iterate(Da, Db, None, None, max_iter=max_iter, tol=tol,
                           diis=diis, offset=offset, fractional=False)


#: Plain SCF iterations before a stalled integer-occupation run switches to
#: second-order steps (:func:`second_order`).
SECOND_ORDER_AFTER = 80

#: Conjugate-gradient iterations (one Fock build each) of a Newton step.
SECOND_ORDER_CG = 30

#: Longest rotation (radians, largest element) of one Newton step.
SECOND_ORDER_TRUST = 0.5


def second_order(solver, state: SCFState, *, tol: float, offset: int,
                 max_iter: int) -> SCFState:
    """Newton-CG orbital optimization of a stalled integer-occupation state.

    The plain loop (DIIS and level shift) crawls on a flat energy surface:
    open-shell transition-metal oxides hold the energy to 1e-9 Hartree while
    the density wanders at 1e-4 for hundreds of iterations.  Each step here
    solves :math:`H p = -g` by preconditioned conjugate gradients on
    finite-difference Hessian-vector products (a Fock build each), limits
    the rotation to :data:`SECOND_ORDER_TRUST` radians and backtracks until
    the energy does not rise.  One record per step joins the iteration log
    (its residual is the largest orbital-gradient element); the state is
    converged when that element and the energy change are below ``tol``.
    A step counts as one iteration of ``max_iter`` (it costs up to
    :data:`SECOND_ORDER_CG` plus a few Fock builds).
    """
    history, previous = [], np.inf
    start = time.perf_counter()
    steps = max_iter
    for taken in range(steps + 1):
        rotations = _Rotations(solver, state)
        if rotations.dim == 0:
            break
        g0, energy0 = rotations.gradient(np.zeros(rotations.dim))
        residual = float(np.max(np.abs(g0)))
        history.append(scf_record(offset + len(history) + 1, start, energy0,
                                  previous, residual))
        converged = residual < tol and abs(energy0 - previous) < tol
        state = SCFState(state.Da, state.Db, energy0, converged,
                         offset + len(history), history)
        if converged or taken == steps:
            break
        previous = energy0
        step, _used = _newton_direction(rotations, g0, SECOND_ORDER_CG)
        peak = float(np.max(np.abs(step))) if step.size else 0.0
        if peak > SECOND_ORDER_TRUST:
            step *= SECOND_ORDER_TRUST / peak
        scale = 1.0
        for _try in range(6):
            _g, energy = rotations.gradient(scale * step)
            if energy <= energy0 + 1e-10:
                break
            scale *= 0.5
        else:
            break                    # no downhill step: the plain loop resumes
        Da, Db, _Ca, _Cb = rotations.densities(scale * step)
        state = SCFState(Da, Db, energy0, False, offset + len(history),
                         history)
    return state


def _newton_direction(rotations: _Rotations, g0, max_cg: int):
    """Inexact Newton step ``p``: preconditioned CG on ``H p = -g0``,
    stopped at a residual a tenth of the gradient, at negative curvature
    (the step then follows it downhill) or after ``max_cg`` products."""
    diag = rotations.diagonal()
    diag = np.where(np.abs(diag) < 0.05, 0.05, np.abs(diag))
    unit = STABILITY_STEP

    def product(v):
        norm = np.linalg.norm(v)
        if norm == 0.0:
            return np.zeros_like(v)
        return norm * (rotations.gradient(unit * v / norm)[0] - g0) / unit

    x = np.zeros_like(g0)
    r = -g0.copy()
    z = r / diag
    d = z.copy()
    rz = float(r @ z)
    target = 0.1 * np.linalg.norm(g0)
    used = 0
    for _ in range(max_cg):
        Hd = product(d)
        used += 1
        curvature = float(d @ Hd)
        if curvature <= 1e-3 * float(d @ (diag * d)):
            if not np.any(x):
                x = d.copy()          # non-positive curvature: go downhill
            break
        alpha = rz / curvature
        x += alpha * d
        r -= alpha * Hd
        if np.linalg.norm(r) < target:
            break
        z = r / diag
        rz_new = float(r @ z)
        d = z + (rz_new / rz) * d
        rz = rz_new
    return x, used
