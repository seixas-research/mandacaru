# -*- coding: utf-8 -*-
# file: algorithms/periodic_dft.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Kohn-Sham DFT in a crystal: Bloch states, k-point sampling and smearing.

The periodic counterpart of :mod:`mandacaru.algorithms.dft`, behind the same
``Mandacaru(method="dft")`` whenever the geometry is periodic.  The
k-dependent matrices and the lattice electrostatics come from
:class:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW`; this module
solves the self-consistency problem on them:

* at every k-point the generalized eigenproblem
  :math:`H(\mathbf k)\,c = \varepsilon\,S(\mathbf k)\,c`;
* occupations from a **smearing** function around a Fermi level found by
  bisection on the electron count (:data:`SMEARING_METHODS`), so metals and
  insulators are handled alike;
* the density, its compensation moments and (for r2SCAN) its kinetic-energy
  density summed over the k-point weights;
* Pulay (DIIS) mixing of the input density and moments, with a Kerker
  preconditioner on the density's long wavelengths -- the charge sloshing a
  metal would otherwise show.

The energy is evaluated as a functional of the *output* density matrices
(every term recomputed from them, no double counting), and the reported
quantities follow the usual smearing conventions: the free energy
:math:`F = E - \sigma S` (the variational quantity, what forces are
derivatives of), and the :math:`\sigma \to 0` estimate
:math:`E_0 = \tfrac12(E + F)` for Fermi-Dirac and Gaussian smearing, or
:math:`F` itself for Methfessel-Paxton, whose free energy is already
correct to :math:`O(\sigma^3)`.

After convergence the potential is frozen and :meth:`PeriodicKohnSham.bands`
diagonalizes :math:`H(\mathbf k)` at any k-point -- a band path, or a dense
mesh for the density of states -- with, on request, each state's Loewdin
weights on the atomic orbitals (:func:`loewdin_weights`), from which the
projected density of states and fat bands follow (:func:`broadened_dos`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.special import erfc

from ..integrals import exchange_correlation as xc_grid
from ..integrals import reciprocal as rc

#: Smearing functions ``smearing={"method": ..., "width": eV}`` accepts.
SMEARING_METHODS = ("fermi-dirac", "gaussian", "methfessel-paxton")

#: Default smearing: Fermi-Dirac at 0.1 eV.
DEFAULT_SMEARING = {"method": "fermi-dirac", "width": 0.1}

#: Pulay history and mixing fraction.
MIXING_HISTORY = 7
MIXING_BETA = 0.25
#: Kerker wave-vector (Bohr^-1): long-wavelength density residuals are damped
#: by ``G^2 / (G^2 + q0^2)``.
KERKER_Q0 = 1.0


# --------------------------------------------------------------------------- #
# Densities of states.
# --------------------------------------------------------------------------- #

def loewdin_weights(S, C) -> np.ndarray:
    r"""``|S^{1/2} C|^2``: each state's weight on each orbital, ``(M, n)``.

    The states are :math:`S`-orthonormal, so the columns of
    :math:`S^{1/2}C` are orthonormal and every column of the result is
    non-negative and sums to one -- a partition of each state over the
    symmetrically orthogonalized atomic orbitals.
    """
    s, U = np.linalg.eigh(S)
    root = (U * np.sqrt(np.clip(s, 0.0, None))) @ U.conj().T
    return np.abs(root @ C) ** 2


def broadened_dos(energies, eigenvalues, weights, width: float,
                  state_weights=None) -> np.ndarray:
    r"""Gaussian-broadened density of states, both spins, per cell.

    :math:`g(E) = 2\sum_k w_k \sum_n \delta_\sigma(E - \varepsilon_{nk})`
    with :math:`\delta_\sigma` a normalized Gaussian of standard deviation
    ``width`` -- so :math:`\int g = 2 n_{bands}`.  ``energies``,
    ``eigenvalues`` (``(nk, n)``) and ``width`` share one unit; ``weights``
    (``(nk,)``) sum to one.  With ``state_weights`` (``(nk, M, n)``) the
    result is ``(len(energies), M)``, one column per orbital, and its rows sum
    to the total.
    """
    energies = np.asarray(energies, dtype=float)
    width = float(width)
    if width <= 0.0:
        raise ValueError(f"the broadening width must be positive, got {width}")
    norm = 2.0 / (width * np.sqrt(2.0 * np.pi))
    out = None
    for k, (eps, w) in enumerate(zip(eigenvalues, weights)):
        x = (energies[:, None] - np.asarray(eps, dtype=float)[None, :]) / width
        peaks = norm * float(w) * np.exp(-0.5 * x * x)        # (nE, n)
        term = (peaks.sum(axis=1) if state_weights is None
                else peaks @ np.asarray(state_weights[k]).T)
        out = term if out is None else out + term
    return out


# --------------------------------------------------------------------------- #
# Smearing.
# --------------------------------------------------------------------------- #

def resolve_smearing(spec) -> tuple[str, float]:
    """``(method, width in Hartree)`` of a smearing spec (width given in eV)."""
    from ..units import EV_TO_HARTREE

    if spec is None:
        spec = DEFAULT_SMEARING
    if isinstance(spec, (int, float)):
        spec = {"method": DEFAULT_SMEARING["method"], "width": float(spec)}
    if not isinstance(spec, dict):
        raise ValueError(f"smearing must be a dict, a width in eV or None; "
                         f"got {spec!r}")
    unknown = set(spec) - {"method", "width"}
    if unknown:
        raise ValueError(f"unknown smearing option(s) {sorted(unknown)}; use "
                         "'method' and 'width' (eV)")
    method = str(spec.get("method", DEFAULT_SMEARING["method"])).strip().lower()
    method = {"fd": "fermi-dirac", "fermi": "fermi-dirac",
              "mp": "methfessel-paxton"}.get(method, method)
    if method not in SMEARING_METHODS:
        raise ValueError(f"unknown smearing method {spec.get('method')!r}; "
                         f"use one of {SMEARING_METHODS}")
    width = float(spec.get("width", DEFAULT_SMEARING["width"]))
    if width <= 0.0:
        raise ValueError(f"the smearing width must be positive, got {width}")
    return method, width * EV_TO_HARTREE


def occupation(x, method: str) -> np.ndarray:
    r"""Occupation in :math:`[0, 1]` (per spin) of :math:`x = (\varepsilon-\mu)/\sigma`."""
    x = np.asarray(x, dtype=float)
    if method == "fermi-dirac":
        return 0.5 * (1.0 - np.tanh(0.5 * x))
    gauss = 0.5 * erfc(x)
    if method == "gaussian":
        return gauss
    return gauss - x * np.exp(-x * x) / (2.0 * np.sqrt(np.pi))


def entropy(x, method: str) -> np.ndarray:
    r"""Entropy per state (per spin, in units of :math:`k_B`) at ``x``.

    The :math:`-\sigma S` term makes :math:`F` variational in the occupations.
    """
    x = np.asarray(x, dtype=float)
    if method == "fermi-dirac":
        f = occupation(x, method)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = (np.where(f > 0, f * np.log(np.where(f > 0, f, 1.0)), 0.0)
                     + np.where(f < 1, (1 - f) * np.log(np.where(f < 1, 1 - f,
                                                                1.0)), 0.0))
        return -terms
    if method == "gaussian":
        return np.exp(-x * x) / (2.0 * np.sqrt(np.pi))
    # Methfessel-Paxton, first order: S = (1/2) A_1 H_2(x) e^{-x^2}.
    return -(2.0 * x * x - 1.0) * np.exp(-x * x) / (4.0 * np.sqrt(np.pi))


def fermi_level(eigenvalues, weights, n_electrons: float, method: str,
                width: float) -> float:
    """The chemical potential that holds ``n_electrons`` (spin-degenerate)."""
    eps = np.concatenate([np.ravel(e) for e in eigenvalues])
    w = np.concatenate([np.full(len(e), wk)
                        for e, wk in zip(eigenvalues, weights)])

    def count(mu):
        return 2.0 * float(np.sum(w * occupation((eps - mu) / width, method)))

    capacity = 2.0 * float(np.sum(w))
    if capacity < n_electrons - 1e-9:
        raise ValueError(f"the basis holds only {capacity:.3f} electrons per "
                         f"cell, fewer than the {n_electrons:g} required")
    # 60 widths: every smearing function is 1 to machine precision there, so
    # a basis that holds exactly the electron count still brackets mu.
    lo, hi = eps.min() - 60.0 * width, eps.max() + 60.0 * width
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if count(mid) < n_electrons:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Mixing.
# --------------------------------------------------------------------------- #

class PulayMixer:
    """DIIS on ``(n~, q)`` with a Kerker-preconditioned linear step."""

    def __init__(self, crystal, history: int = MIXING_HISTORY,
                 beta: float = MIXING_BETA, q0: float = KERKER_Q0):
        self.crystal = crystal
        self.history = int(history)
        self.beta = float(beta)
        G2 = np.sum(crystal.G * crystal.G, axis=0)
        self.kerker = G2 / (G2 + q0 * q0)
        # Not zero at G = 0, as it would be with a fixed valence charge: the
        # smooth density's charge is *not* fixed -- it trades with the
        # compensation monopoles -- and a damped G = 0 would freeze it.
        self.kerker.flat[0] = 1.0
        self._inputs: list[np.ndarray] = []
        self._residuals: list[np.ndarray] = []
        self.n_grid = crystal.grid.size

    def _pack(self, rho, q) -> np.ndarray:
        values = [q[c] for c in self.crystal.channels]
        return np.concatenate([rho.astype(complex), np.asarray(values,
                                                              dtype=complex)])

    def _unpack(self, x):
        rho = np.real(x[:self.n_grid])
        q = {c: x[self.n_grid + i] for i, c in enumerate(self.crystal.channels)}
        return rho, q

    def _precondition(self, residual) -> np.ndarray:
        rho = residual[:self.n_grid]
        g = self.crystal.grid
        smooth = np.real(np.fft.ifftn(np.fft.fftn(rho.reshape(g.shape))
                                      * self.kerker)).reshape(-1)
        out = residual.copy()
        out[:self.n_grid] = smooth
        return out

    def mix(self, rho_in, q_in, rho_out, q_out):
        """The next input ``(n~, q)``."""
        x_in = self._pack(rho_in, q_in)
        residual = self._pack(rho_out, q_out) - x_in
        self._inputs.append(x_in)
        self._residuals.append(residual)
        if len(self._inputs) > self.history:
            self._inputs.pop(0)
            self._residuals.pop(0)
        n = len(self._inputs)
        dV = self.crystal.grid.dV
        metric = np.ones(len(residual))
        metric[:self.n_grid] = dV
        B = np.empty((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                B[i, j] = float(np.real(np.vdot(self._residuals[i] * metric,
                                                self._residuals[j])))
        B[n, :n] = B[:n, n] = -1.0
        B[n, n] = 0.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        try:
            alpha = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            alpha = np.zeros(n)
            alpha[-1] = 1.0
        x_opt = sum(a * x for a, x in zip(alpha, self._inputs))
        r_opt = sum(a * r for a, r in zip(alpha, self._residuals))
        x_next = x_opt + self.beta * self._precondition(r_opt)
        return self._unpack(x_next)

    @staticmethod
    def residual_norm(rho_in, rho_out, dV) -> float:
        return float(np.sum(np.abs(rho_out - rho_in)) * dV)


# --------------------------------------------------------------------------- #
# The SCF.
# --------------------------------------------------------------------------- #

@dataclass
class PeriodicKohnShamResult:
    """A converged periodic Kohn-Sham calculation (Hartree, per cell).

    ``eigenvalues`` and ``fermi_level`` are measured from the cell average of
    the electrostatic potential of the electrons and the *point* ions, each
    atom's non-Coulomb local potential included -- the usual plane-wave
    reference (:meth:`PeriodicKohnSham.eigenvalue_reference`).  Without it the
    zero would move with the grid spacing.
    """

    functional: str
    free_energy: float                  # F = E - sigma S
    energy: float                       # E (internal)
    extrapolated_energy: float          # sigma -> 0 estimate
    fermi_level: float
    kpoints: np.ndarray                 # Cartesian, Bohr^-1
    weights: np.ndarray
    eigenvalues: list                   # per k-point, ascending (Hartree)
    occupations: list                   # per k-point, 0..2
    converged: bool
    n_iterations: int
    smearing: tuple
    terms: dict = field(default_factory=dict)

    @property
    def band_gap(self) -> float | None:
        """Conduction-band minimum minus valence-band maximum **over the
        mesh** (Hartree), or ``None`` for a metal -- when the Fermi level lies
        inside the range some band spans across the k-points.

        Only the k-points of the mesh are seen, so this is an upper bound on
        the true gap whenever a band edge lies between them (silicon's
        conduction-band minimum, ~85 % of the way to X, is on no small
        Gamma-centered mesh).
        """
        bands = np.array([np.asarray(e) for e in self.eigenvalues])
        lowest, highest = bands.min(axis=0), bands.max(axis=0)
        mu = self.fermi_level
        if np.any((lowest < mu) & (highest > mu)):
            return None
        below = highest[highest <= mu]
        above = lowest[lowest > mu]
        if not below.size or not above.size:
            return None
        return float(above.min() - below.max())


class PeriodicKohnSham:
    r"""Self-consistent Kohn-Sham equations on a :class:`PeriodicPAW` crystal."""

    def __init__(self, crystal, n_electrons: float, functional: str = "lda",
                 smearing=None, relativistic: bool = False,
                 constant: float = 0.0):
        self.crystal = crystal
        self.n_electrons = float(n_electrons)
        self.functional = xc_grid.resolve_functional(functional)
        self.meta = xc_grid.is_meta_gga(self.functional)
        self.method, self.width = resolve_smearing(smearing)
        self.relativistic = bool(relativistic)
        self.constant = float(constant)
        self.core = crystal.core_density()
        self._core_tau = crystal.core_tau() if self.meta else None
        self.U, self.U_ion, self.E_ion = crystal.dense_coulomb()
        self.ion_constant = crystal.ion_constants()
        self.ion_grid = crystal.ion_charge()
        self.g_hat = crystal.compensation_grid()
        self.onsite = crystal.onsite_ion_compensation()
        self.short_range = crystal.short_range_ion_compensation()
        self._gradients = ([crystal.bloch_gradients(d.psi, d.k)
                            for d in crystal.kpoint_data] if self.meta
                           else None)
        #: ``(V, v_tau, w)`` that produced the converged eigenvalues -- set by
        #: :meth:`run`, read by :meth:`bands`.
        self.potentials = None

    # -- the effective Hamiltonian ------------------------------------------ #

    def _vector(self, q) -> np.ndarray:
        return np.array([q[c] for c in self.crystal.channels], dtype=complex)

    def _potentials(self, rho, q, tau):
        """``(V_grid, v_tau, w, xc_terms)`` of an input ``(n~, q, tau)``."""
        c = self.crystal
        rho_G = rc.to_reciprocal(c.grid, rho)
        hat_G = sum((q[ch] * self.g_hat[ch] for ch in c.channels),
                    np.zeros_like(rho_G))
        potential_G = c.kernel * (rho_G + hat_G + self.ion_grid)
        V_es = np.real(rc.to_real(c.grid, potential_G))
        xc_rho = rho + (self.core if self.core is not None else 0.0)
        xc_tau = None
        if self.meta:
            xc_tau = tau + (self._core_tau if self._core_tau is not None
                            else 0.0)
        terms = xc_grid.evaluate(c.grid, xc_rho, self.functional,
                                 relativistic=self.relativistic, tau=xc_tau)
        # dE/dq_a = int V g_a: the smooth density's potential on the grid set,
        # the compact charges' on the dense set.
        qv = self._vector(q)
        w = {}
        for a, ch in enumerate(c.channels):
            smooth = np.sum(c.kernel * np.conj(rho_G) * self.g_hat[ch]) \
                / c.volume
            dense = np.sum(np.conj(qv) * self.U[:, a]) + np.conj(self.U_ion[a])
            w[ch] = (smooth + dense + self.short_range[ch]
                     - self.onsite.get(ch, 0.0))
        V, v_tau = V_es + terms.potential, terms.tau_potential
        symmetry = c.symmetry
        if symmetry is not None:
            # The energy sees the density only through its symmetrized form,
            # so its exact derivative is the symmetrized potential.  The two
            # differ only where the spectral gradients do not commute with
            # the operations -- the Nyquist plane of an even grid -- but a
            # Kohn-Sham matrix that is not the derivative of the energy is a
            # wrong one there.
            V = symmetry.field(V)
            if v_tau is not None:
                v_tau = symmetry.field(v_tau)
        return V, v_tau, w, terms

    def _hamiltonian(self, data, V, v_tau, w, gradients=None) -> np.ndarray:
        psi, dV = data.psi, self.crystal.grid.dV
        H = data.fixed + ((psi.conj() * V) @ psi.T) * dV
        if v_tau is not None:
            for d in gradients:
                H = H + 0.5 * ((d.conj() * v_tau) @ d.T) * dV
        for ch, Q in data.moments.items():
            H = H + w[ch] * Q
        return 0.5 * (H + H.conj().T)

    # -- the energy --------------------------------------------------------- #

    def energy_terms(self, matrices, rho, q, tau) -> dict:
        """Every term of the Kohn-Sham energy of the density matrices (Ha)."""
        c = self.crystal
        band = sum(d.weight * float(np.real(np.sum(P * d.fixed.T)))
                   for d, P in zip(c.kpoint_data, matrices))
        rho_G = rc.to_reciprocal(c.grid, rho)
        hat_G = sum((q[ch] * self.g_hat[ch] for ch in c.channels),
                    np.zeros_like(rho_G))
        smooth = float(np.real(np.sum(
            c.kernel * (0.5 * np.abs(rho_G) ** 2
                        + np.conj(rho_G) * (hat_G + self.ion_grid))))) \
            / c.volume
        qv = self._vector(q)
        compact = (0.5 * float(np.real(np.conj(qv) @ self.U @ qv))
                   + float(np.real(np.sum(np.conj(qv) * self.U_ion)))
                   + self.E_ion)
        ionic = float(np.real(sum(q[ch] * self.short_range[ch]
                                  for ch in c.channels))) \
            - float(np.real(sum(q[ch] * value
                                for ch, value in self.onsite.items())))
        xc_rho = rho + (self.core if self.core is not None else 0.0)
        xc_tau = None
        if self.meta:
            xc_tau = tau + (self._core_tau if self._core_tau is not None
                            else 0.0)
        e_xc = xc_grid.evaluate(c.grid, xc_rho, self.functional,
                                relativistic=self.relativistic,
                                tau=xc_tau).energy
        return {"band_fixed": band, "electrostatic": smooth + compact + ionic
                + self.ion_constant, "xc": e_xc, "constant": self.constant}

    # -- the loop ----------------------------------------------------------- #

    def run(self, max_iter: int = 300, tol: float = 1e-7,
            density_tol: float = 1e-5) -> PeriodicKohnShamResult:
        """Iterate to self-consistency from superposed atomic densities."""
        c = self.crystal
        rho, q = c.initial_density()
        tau = np.zeros(c.grid.size) if self.meta else None
        mixer = PulayMixer(c)
        previous = np.inf
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            V, v_tau, w, _terms = self._potentials(rho, q, tau)
            eigenvalues, vectors = [], []
            for i, data in enumerate(c.kpoint_data):
                H = self._hamiltonian(data, V, v_tau, w,
                                      self._gradients[i] if self.meta else None)
                eps, C = eigh(H, data.overlap)
                eigenvalues.append(eps)
                vectors.append(C)
            mu = fermi_level(eigenvalues, c.weights, self.n_electrons,
                             self.method, self.width)
            occupations = [2.0 * occupation((e - mu) / self.width, self.method)
                           for e in eigenvalues]
            matrices = [(C * f) @ C.conj().T
                        for C, f in zip(vectors, occupations)]
            rho_out, q_out = c.density(matrices)
            tau_out = self._tau(matrices) if self.meta else None
            terms = self.energy_terms(matrices, rho_out, q_out, tau_out)
            energy = sum(terms.values())
            entropy_term = -self.width * sum(
                d.weight * 2.0 * float(np.sum(entropy((e - mu) / self.width,
                                                      self.method)))
                for d, e in zip(c.kpoint_data, eigenvalues))
            free = energy + entropy_term
            residual = PulayMixer.residual_norm(rho, rho_out, c.grid.dV)
            if abs(free - previous) < tol and residual < density_tol:
                converged = True
                break
            previous = free
            rho, q = mixer.mix(rho, q, rho_out, q_out)
            if self.meta:
                tau = tau_out
        # The potentials of the last diagonalization, not of the output
        # density: those are the ones the reported eigenvalues belong to.
        self.potentials = (V, v_tau, w)
        extrapolated = (free if self.method == "methfessel-paxton"
                        else 0.5 * (energy + free))
        terms["entropy"] = entropy_term
        reference = self.eigenvalue_reference()
        return PeriodicKohnShamResult(
            functional=self.functional, free_energy=free, energy=energy,
            extrapolated_energy=extrapolated, fermi_level=mu + reference,
            kpoints=c.kpoints, weights=c.weights,
            eigenvalues=[e + reference for e in eigenvalues],
            occupations=occupations,
            converged=converged, n_iterations=it,
            smearing=(self.method, self.width), terms=terms)

    def bands(self, kpoints, projections: bool = False):
        r"""Non-self-consistent eigenvalues at Cartesian ``kpoints`` (Bohr^-1).

        The converged potential is frozen and :math:`H(\mathbf k)` built and
        diagonalized at each point, a block of k-points at a time so the
        Bloch sums of a long path or a dense mesh are never all held.  At a
        k-point of the SCF mesh it returns the SCF eigenvalues.

        Returns ``(eigenvalues, weights)``: ``(nk, M)`` in Hartree, on the
        same reference as the result's (:meth:`eigenvalue_reference`), and
        with ``projections`` the ``(nk, M, M)`` Loewdin weights
        (:func:`loewdin_weights`; ``weights[k, mu, n]`` of state ``n`` on
        orbital ``mu``), else ``None``.
        """
        if self.potentials is None:
            raise RuntimeError("bands() needs a converged run() first")
        V, v_tau, w = self.potentials
        c = self.crystal
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        reference = self.eigenvalue_reference()
        block = c.kpoint_block()
        eigenvalues, weights = [], []
        for start in range(0, len(kpoints), block):
            for data in c.kpoint_matrices(kpoints[start:start + block]):
                gradients = (c.bloch_gradients(data.psi, data.k)
                             if self.meta else None)
                H = self._hamiltonian(data, V, v_tau, w, gradients)
                eps, C = eigh(H, data.overlap)
                eigenvalues.append(eps + reference)
                if projections:
                    weights.append(loewdin_weights(data.overlap, C))
        return (np.array(eigenvalues),
                np.array(weights) if projections else None)

    def eigenvalue_reference(self) -> float:
        r"""Constant (Hartree) that moves the eigenvalues to the plane-wave zero.

        The grid carries the long-range half of each local potential, the
        potential of a Gaussian ion of width :math:`\sigma`, with
        :math:`\mathbf G = 0` dropped; the short-range half is integrated on
        spheres and keeps its cell average.  That average holds
        :math:`-\int Z\,\mathrm{erfc}(r/\sqrt2\sigma)/r\,d^3r / \Omega =
        -2\pi Z\sigma^2/\Omega` per atom, the difference between a Gaussian
        and a point ion -- and :math:`\sigma` is tied to the grid spacing, so
        it moved every eigenvalue with ``h`` (silicon's lowest level by
        1.7 eV between h = 0.30 and 0.20 Angstrom).  Adding it back measures
        the eigenvalues from the point-ion average; the occupations, which
        depend only on :math:`\varepsilon - \mu`, are unaffected.
        """
        c = self.crystal
        return float(2.0 * np.pi * c.sigma ** 2 * np.sum(c.charges)
                     / c.volume)

    def _tau(self, matrices) -> np.ndarray:
        """The crystal's (symmetrized) tau, from the cached Bloch gradients."""
        return self.crystal.kinetic_energy_density(matrices, self._gradients)
