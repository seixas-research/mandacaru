# -*- coding: utf-8 -*-
# file: algorithms/expressivity.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Expressibility of parameterized quantum circuits (PQCs).

The *expressibility* of a PQC measures how uniformly its output states cover the
accessible Hilbert space as its parameters are sampled at random -- a proxy for
how much of the space the ansatz can represent (Sim, Johnson & Aspuru-Guzik,
2019).  It is quantified by comparing the distribution of state fidelities

.. math::

    F = \bigl|\langle\psi(\vec\theta_a)|\psi(\vec\theta_b)\rangle\bigr|^2

between random parameter pairs to the fidelity distribution of Haar-random states.
For a :math:`d`-dimensional space the Haar fidelity density is

.. math::

    P_{\text{Haar}}(F) = (d-1)(1-F)^{d-2},

and the expressibility score is the Kullback-Leibler divergence of the sampled
distribution from the Haar one, over discretized fidelity bins:

.. math::

    E = D_{\mathrm{KL}}\!\bigl(P_{\text{PQC}} \,\|\, P_{\text{Haar}}\bigr)
      = \sum_i P_{\text{PQC}}(F_i)\,\ln\frac{P_{\text{PQC}}(F_i)}{P_{\text{Haar}}(F_i)}.

A **smaller** :math:`E` means a more expressive circuit (its states look
Haar-random); :math:`E = 0` is the maximally expressive limit.

Effective dimension (a physical subtlety)
------------------------------------------
The dimension :math:`d` is **not** always :math:`2^N`.  Carcará's fermionic
ansätze (:class:`~carcara.circuits.ansatz.UCCSD` and the fermionic / QEB / CEO
ADAPT ansätze) conserve particle number *and* the spin projection :math:`S_z`, so
their states never leave the symmetry sector of the Hartree-Fock reference.  That
sector has dimension

.. math::

    d = \binom{M}{n_\alpha}\binom{M}{n_\beta}, \qquad M = N/2,

not :math:`2^N` (e.g. H\ :sub:`2`: :math:`d = \binom{2}{1}\binom{2}{1} = 4`, not
:math:`16`).  Comparing such an ansatz to a full :math:`2^N`-dimensional Haar
distribution would label *every* number-conserving circuit inexpressive.  Use
:func:`active_space_dimension` (from a known particle number) or
:func:`estimate_effective_dimension` (empirically, as the rank of the span of
sampled states) to obtain the correct :math:`d`.  The qubit-ADAPT pool, whose
individual Pauli generators do **not** conserve particle number, can explore a
larger sector -- there the empirical estimate is the safe choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import comb
from scipy.stats import entropy

EPS = 1e-10   # guards log(0) / division by zero in the KL divergence


# --------------------------------------------------------------------------- #
# Generator interface.
# --------------------------------------------------------------------------- #

def _resolve_generator(circuit_generator):
    """Return ``(num_parameters, state_fn, n_qubits)`` for a Carcará ansatz.

    Accepts any object exposing ``num_parameters`` and ``state(theta)`` -- i.e.
    :class:`~carcara.circuits.ansatz.UCCSD`,
    :class:`~carcara.algorithms.adapt_vqe.AdaptAnsatz`, or a custom generator with
    the same interface.
    """
    if hasattr(circuit_generator, "state") and \
            hasattr(circuit_generator, "num_parameters"):
        return (int(circuit_generator.num_parameters),
                circuit_generator.state,
                getattr(circuit_generator, "n_qubits", None))
    raise TypeError(
        "circuit_generator must expose `num_parameters` and `state(theta)` "
        "(e.g. a UCCSD or AdaptAnsatz)")


def _sample_states(state_fn, num_parameters: int, num: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Stack ``num`` output state vectors for random ``theta`` in ``[0, 2*pi]^M``."""
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(num, num_parameters))
    states = [np.asarray(state_fn(theta), dtype=complex) for theta in thetas]
    return np.array(states)


# --------------------------------------------------------------------------- #
# Effective Hilbert-space dimension.
# --------------------------------------------------------------------------- #

def active_space_dimension(n_qubits: int,
                           num_particles: tuple[int, int] | None = None) -> int:
    r"""Dimension of the accessible Hilbert space.

    With ``num_particles = (n_alpha, n_beta)`` this returns the dimension of the
    particle- and :math:`S_z`-conserving sector,
    :math:`\binom{M}{n_\alpha}\binom{M}{n_\beta}` with ``M = n_qubits / 2`` -- the
    space a fermionic (UCCSD / fermionic-, QEB-, CEO-ADAPT) ansatz actually
    explores.  Without ``num_particles`` it falls back to the full space
    ``2**n_qubits``.
    """
    if num_particles is None:
        return 2 ** int(n_qubits)
    n_alpha, n_beta = num_particles
    M = int(n_qubits) // 2
    return int(comb(M, n_alpha, exact=True) * comb(M, n_beta, exact=True))


def estimate_effective_dimension(circuit_generator, num_probe: int = 256,
                                 tol: float = 1e-9,
                                 rng: np.random.Generator | None = None) -> int:
    """Empirical effective dimension: the rank of the span of sampled states.

    Samples ``num_probe`` random output states and counts the singular values
    above ``tol`` (relative to the largest).  This captures the true dimension of
    the subspace the ansatz reaches, whatever symmetries it does or does not
    respect -- the robust choice for the qubit-ADAPT pool.
    """
    rng = np.random.default_rng(rng)
    M, state_fn, _ = _resolve_generator(circuit_generator)
    states = _sample_states(state_fn, M, num_probe, rng)
    sv = np.linalg.svd(states, compute_uv=False)
    if sv.size == 0 or sv[0] == 0:
        return 1
    return int(np.count_nonzero(sv > tol * sv[0]))


# --------------------------------------------------------------------------- #
# Haar distribution and fidelity sampling.
# --------------------------------------------------------------------------- #

def calculate_haar_distribution(d: int, bins: int) -> np.ndarray:
    r"""Haar fidelity probability *mass* per bin for a ``d``-dimensional space.

    Integrates the Haar density :math:`P_{\text{Haar}}(F) = (d-1)(1-F)^{d-2}` over
    each of ``bins`` equal-width bins on ``[0, 1]`` using its closed-form CDF
    :math:`\mathrm{CDF}(F) = 1 - (1-F)^{d-1}`, so the returned array is exact and
    sums to 1 (more accurate than sampling the density at bin centres).
    """
    if d < 2:
        raise ValueError(f"Haar distribution needs dimension d >= 2, got {d}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    cdf = 1.0 - (1.0 - edges) ** (d - 1)
    return np.diff(cdf)


def haar_density(F, d: int) -> np.ndarray:
    r"""Haar fidelity *density* :math:`(d-1)(1-F)^{d-2}` (for plotting a curve)."""
    F = np.asarray(F, dtype=float)
    return (d - 1) * (1.0 - F) ** (d - 2)


def sample_pqc_fidelities(circuit_generator, num_samples: int = 1000,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    r"""Fidelities ``F = |<psi(theta_a)|psi(theta_b)>|^2`` for random parameter pairs.

    Draws ``num_samples`` independent pairs of parameter vectors uniformly from
    :math:`[0, 2\pi]^M` (``M`` = the generator's current parameter count), prepares
    the two states with Carcará's exact state-vector backend, and returns the
    ``num_samples`` fidelities.  With ``M = 0`` (e.g. an ADAPT ansatz before any
    operator is added) every state is the fixed reference, so all fidelities are 1.
    """
    rng = np.random.default_rng(rng)
    M, state_fn, _ = _resolve_generator(circuit_generator)
    a = _sample_states(state_fn, M, num_samples, rng)
    b = _sample_states(state_fn, M, num_samples, rng)
    overlaps = np.sum(np.conj(a) * b, axis=1)
    return np.abs(overlaps) ** 2


def calculate_kl_divergence(fidelities, num_qubits: int, num_bins: int = 75,
                            dim: int | None = None) -> float:
    r"""Expressibility score ``E = D_KL(P_PQC || P_Haar)`` (natural log).

    Bins the sampled ``fidelities`` into ``num_bins`` bins on ``[0, 1]`` to form
    :math:`P_{\text{PQC}}`, builds the analytical :math:`P_{\text{Haar}}` for
    dimension ``dim`` (default ``2**num_qubits`` -- pass the effective sector
    dimension for a number-conserving ansatz, see the module docstring), adds
    ``EPS`` to both to avoid ``log(0)``/division by zero, and returns the KL
    divergence.  Lower is more expressive.
    """
    d = int(dim) if dim is not None else 2 ** int(num_qubits)
    counts, _ = np.histogram(np.asarray(fidelities), bins=num_bins, range=(0.0, 1.0))
    p_pqc = counts / max(counts.sum(), 1)
    p_haar = calculate_haar_distribution(d, num_bins)
    # scipy.stats.entropy(pk, qk) = sum pk * log(pk/qk); it renormalizes both.
    return float(entropy(p_pqc + EPS, p_haar + EPS))


# --------------------------------------------------------------------------- #
# High-level driver.
# --------------------------------------------------------------------------- #

@dataclass
class ExpressibilityResult:
    """Outcome of an expressibility measurement."""

    kl_divergence: float              # the expressibility score E (lower = better)
    fidelities: np.ndarray            # sampled |<a|b>|^2 values
    num_qubits: int
    dimension: int                    # effective Hilbert-space dimension used
    num_samples: int
    num_bins: int

    @property
    def expressibility(self) -> float:
        """Alias for :attr:`kl_divergence` (the score ``E``)."""
        return self.kl_divergence

    def __repr__(self) -> str:
        return (f"ExpressibilityResult(E={self.kl_divergence:.4f}, "
                f"d={self.dimension}, n_samples={self.num_samples})")


def compute_expressibility(circuit, num_qubits: int | None = None,
                           num_samples: int = 1000, bins: int = 50,
                           dim: int | None = None,
                           num_particles: tuple[int, int] | None = None,
                           rng: np.random.Generator | None = None
                           ) -> ExpressibilityResult:
    """Measure the expressibility of a Carcará ansatz.

    Parameters
    ----------
    circuit : ansatz
        Any generator exposing ``num_parameters`` and ``state(theta)`` (e.g.
        :class:`~carcara.circuits.ansatz.UCCSD` or
        :class:`~carcara.algorithms.adapt_vqe.AdaptAnsatz`).
    num_qubits : int, optional
        Qubit count; taken from ``circuit.n_qubits`` if omitted.
    num_samples : int
        Number of random fidelity samples (default ``1000``).
    bins : int
        Number of fidelity bins for the KL divergence (default ``50``).
    dim : int, optional
        Effective Hilbert-space dimension for the Haar reference.  If omitted it is
        derived from ``num_particles`` (:func:`active_space_dimension`) when given,
        else defaults to ``2**num_qubits``.
    num_particles : (int, int), optional
        ``(n_alpha, n_beta)``; used to pick the number-conserving sector dimension
        when ``dim`` is not supplied.
    rng : numpy Generator, optional
        Seedable RNG for reproducibility.
    """
    rng = np.random.default_rng(rng)
    _, _, n_from_gen = _resolve_generator(circuit)
    if num_qubits is None:
        if n_from_gen is None:
            raise ValueError("num_qubits is required when the generator has no "
                             "`n_qubits` attribute")
        num_qubits = n_from_gen

    if dim is None:
        dim = active_space_dimension(num_qubits, num_particles)

    fidelities = sample_pqc_fidelities(circuit, num_samples, rng)
    kl = calculate_kl_divergence(fidelities, num_qubits, bins, dim)
    return ExpressibilityResult(kl_divergence=kl, fidelities=fidelities,
                                num_qubits=int(num_qubits), dimension=int(dim),
                                num_samples=int(num_samples), num_bins=int(bins))


# --------------------------------------------------------------------------- #
# ADAPT-VQE expressibility tracking.
# --------------------------------------------------------------------------- #

@dataclass
class ExpressibilityStep:
    """Expressibility recorded at one ADAPT-VQE iteration."""

    num_operators: int
    num_parameters: int
    kl_divergence: float
    energy: float | None = None
    operator_label: str | None = None


class ADAPTExpressivityTracker:
    """Callback that records ansatz expressibility as ADAPT-VQE grows the circuit.

    Pass an instance as the ``callback`` of :meth:`ADAPTVQE.run
    <carcara.algorithms.adapt_vqe.ADAPTVQE.run>`: after each accepted operator it
    samples the current ansatz's expressibility against a **fixed** Haar reference
    (so the scores are comparable across steps) and appends an
    :class:`ExpressibilityStep`.  The resulting :attr:`history` shows how
    expressibility improves (KL decreases) and saturates as operators accumulate.

    Parameters
    ----------
    num_qubits : int
        Qubit count of the ansatz.
    dim : int, optional
        Fixed effective dimension for the Haar reference.  Defaults to the
        number-conserving sector from ``num_particles``, else ``2**num_qubits``.
        (Fixing it -- rather than re-estimating per step -- is what makes the
        growth curve meaningful.)
    num_particles : (int, int), optional
        Used to derive ``dim`` when it is not given.
    num_samples : int
        Fidelity samples per step (default ``500``; fewer than a one-off
        measurement since it runs every iteration).
    bins : int
        Fidelity bins for the KL divergence (default ``75``).
    rng : numpy Generator, optional
        Seedable RNG.
    """

    def __init__(self, num_qubits: int, dim: int | None = None,
                 num_particles: tuple[int, int] | None = None,
                 num_samples: int = 500, bins: int = 75,
                 rng: np.random.Generator | None = None):
        self.num_qubits = int(num_qubits)
        self.dimension = int(dim) if dim is not None \
            else active_space_dimension(num_qubits, num_particles)
        self.num_samples = int(num_samples)
        self.bins = int(bins)
        self.rng = np.random.default_rng(rng)
        self.history: list[ExpressibilityStep] = []

    def __call__(self, info: dict) -> None:
        """ADAPT-VQE callback: measure and log the current ansatz."""
        ansatz = info["ansatz"]
        fidelities = sample_pqc_fidelities(ansatz, self.num_samples, self.rng)
        kl = calculate_kl_divergence(fidelities, self.num_qubits, self.bins,
                                     self.dimension)
        self.history.append(ExpressibilityStep(
            num_operators=int(info["num_operators"]),
            num_parameters=int(info["num_operators"]),
            kl_divergence=kl,
            energy=info.get("energy"),
            operator_label=info.get("operator_label")))

    # -- convenience views ------------------------------------------------ #

    @property
    def num_operators(self) -> list[int]:
        return [s.num_operators for s in self.history]

    @property
    def scores(self) -> list[float]:
        return [s.kl_divergence for s in self.history]


def track_adapt_expressivity(adapt_vqe, dim: int | None = None,
                             num_samples: int = 500, bins: int = 75,
                             rng: np.random.Generator | None = None,
                             **run_kwargs):
    """Run an :class:`ADAPTVQE` while tracking expressibility per ADAPT step.

    Returns ``(adapt_result, history)`` where ``history`` is the tracker's list of
    :class:`ExpressibilityStep`.  The Haar reference dimension is fixed to
    ``dim`` (or the number-conserving sector inferred from
    ``adapt_vqe.num_particles``).  Extra keyword arguments are forwarded to
    :meth:`ADAPTVQE.run <carcara.algorithms.adapt_vqe.ADAPTVQE.run>`.
    """
    tracker = ADAPTExpressivityTracker(
        adapt_vqe.n_qubits, dim=dim, num_particles=adapt_vqe.num_particles,
        num_samples=num_samples, bins=bins, rng=rng)
    result = adapt_vqe.run(callback=tracker, **run_kwargs)
    return result, tracker.history


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #

def plot_fidelity_distribution(result, ax=None, bins: int | None = None,
                               color: str = "#0072B2", label: str = "PQC"):
    """Plot the sampled fidelity histogram against the analytical Haar density.

    ``result`` is an :class:`ExpressibilityResult` (or a raw fidelity array, in
    which case pass ``dim`` via a result is not possible -- use the result form).
    Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    if not isinstance(result, ExpressibilityResult):
        raise TypeError("pass an ExpressibilityResult (from compute_expressibility)")
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))
    nbins = bins or result.num_bins

    ax.hist(result.fidelities, bins=nbins, range=(0.0, 1.0), density=True,
            color=color, alpha=0.55, edgecolor="white", linewidth=0.5,
            label=f"{label}  (E = {result.kl_divergence:.3f})", zorder=2)
    grid = np.linspace(0.0, 1.0, 400)
    ax.plot(grid, haar_density(grid, result.dimension), color="#D55E00", lw=2.2,
            label=f"Haar  (d = {result.dimension})", zorder=3)
    ax.set_xlabel("fidelity  F = |⟨ψ(θₐ)|ψ(θ_b)⟩|²")
    ax.set_ylabel("probability density")
    ax.set_title("PQC fidelity distribution vs Haar")
    ax.grid(True, color="0.9", lw=0.8, zorder=0)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def plot_expressivity_growth(history, ax=None, color: str = "#009E73",
                             label: str | None = None):
    """Plot the expressibility score ``E`` versus the number of ADAPT operators.

    ``history`` is a list of :class:`ExpressibilityStep` (e.g. from
    :func:`track_adapt_expressivity`).  Lower ``E`` is more expressive, so the
    curve typically decreases and then saturates.  Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))
    x = [s.num_operators for s in history]
    y = [s.kl_divergence for s in history]
    ax.plot(x, y, color=color, lw=2.0, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)
    ax.set_xlabel("number of ADAPT operators")
    ax.set_ylabel("expressibility  E = D_KL(P_PQC ‖ P_Haar)")
    ax.set_title("Expressibility growth during ADAPT-VQE")
    ax.grid(True, color="0.9", lw=0.8, zorder=0)
    if label:
        ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax
