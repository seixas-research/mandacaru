# -*- coding: utf-8 -*-
# file: algorithms/rpa_density.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The one-particle density matrix of the direct random-phase
approximation, and its natural orbitals, for a crystal.

**Amplitudes.**  Direct RPA is ring coupled-cluster doubles with the direct
(Coulomb) rings only (Scuseria, Henderson and Sorensen 2008): with the
particle-hole pairs :math:`p = (n\to m)`, their gaps :math:`\Delta_p` and
the spin-singlet matrices :math:`A = \Delta + 2K`, :math:`B = 2K'` built
from the Coulomb integrals :math:`K_{pr} = (nm|r)`, the amplitudes
:math:`t = Y X^{-1}` follow from the positive-frequency solutions of the
RPA eigenproblem, and

.. math::

    E_c = \tfrac12\sum_{pr} B_{pr}\,t_{pr}
        = \tfrac12\Big(\sum_\nu\Omega_\nu - \operatorname{tr}A\Big).

**Crystal.**  The pairs are those of the Born-von Karman supercell of the
k-mesh, :math:`(n\mathbf k\to m,\mathbf k+\mathbf q)`, which momentum
conservation splits into blocks: an excitation of momentum
:math:`\mathbf q` pairs with one of :math:`-\mathbf q`, so each
:math:`\mathbf q` is one problem of dimension
:math:`n_{\mathbf q} + n_{-\mathbf q}` -- the Gamma-point supercell problem,
block-diagonalized (the supercell itself would be one problem :math:`N_k`
times larger).  The integrals are :math:`L L^\dagger` products of the
transition densities on the vectors :math:`\mathbf q+\mathbf G` within a
cutoff, compensation charges included (as for the screened interaction,
:func:`~mandacaru.algorithms.wannier.polarizability`),
:math:`L_p(\mathbf G) = \sqrt{w_p/2}\,\rho_p(\mathbf q+\mathbf G)
\sqrt{4\pi/|\mathbf q+\mathbf G|^2/N\Omega}`; the divergent
:math:`\mathbf q = \mathbf G = 0` term is left out (its pair densities
vanish).  A metal's smeared occupations enter as pair weights
:math:`w_p = F_n - F_m`, the ensemble form of :math:`\chi_0`; an insulator
has :math:`w_p = 2` on every occupied-to-empty pair.  Each block is solved
through the Cholesky factor of :math:`\begin{pmatrix}A & B\\ B^\dagger &
A'^*\end{pmatrix}` (positive definite), and the block of :math:`-\mathbf q`
is the transpose of that of :math:`\mathbf q`.

**Density.**  The amplitudes are contracted like the unrelaxed MP2 density,
direct terms only (there are no exchange rings to pair with):

.. math::

    \gamma_{ab} \mathrel{+}= \sum_{n,r} t_{(na),r}\,t^*_{(nb),r},\qquad
    \gamma_{ij} \mathrel{-}= \sum_{m,r} t_{(jm),r}\,t^*_{(im),r},

on top of the reference occupations.  Both act within one k-point, so
:math:`\gamma(\mathbf k)` is block diagonal; its trace stays the electron
count exactly, and its eigenvectors are the Bloch natural orbitals, ranked
by their occupation's distance from 2 or 0 as the molecular selector does
(:mod:`~mandacaru.algorithms.active_space`).  At second order the density
is the direct part of the MP2 one; unlike MP2 it stays finite when a gap
closes, since the rings screen the interaction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..integrals import reciprocal as rc
from ..units import HARTREE_TO_EV

#: Radius (Bohr^-1) of the vectors q + G the Coulomb integrals are summed
#: on.  Measured on Si (HISTORY, "Choosing the target space"): occupations
#: move by < 2 % of their deviation between 3 and 4 Bohr^-1.
RPA_CUTOFF = 3.0

#: Convergence of the iterated amplitudes (:func:`ring_amplitudes_iterative`):
#: the largest change of an amplitude per step.  The eigenproblem agrees
#: with them to 3e-13 on Si.
AMPLITUDE_TOLERANCE = 1e-11

#: A pair whose weight F_n - F_m (electrons) is below this does not enter.
PAIR_THRESHOLD = 1e-6

#: A natural orbital whose occupation deviates from 2 or 0 by at least this
#: much is selected (:meth:`CrystalNaturalOrbitals.n_selected`), as the
#: molecular selector's occupation threshold.  Measured on Si, 3^3: it keeps
#: 10 orbitals at every k-point -- the four bonding (deviations 0.04-0.12),
#: the four antibonding (0.034-0.087), both with sp3 projectability 0.89-1,
#: and two d-like ones (0.022-0.033) -- above a gap down to 0.013.
DEFAULT_DEVIATION_THRESHOLD = 0.02


# --------------------------------------------------------------------------- #
# The amplitudes of one momentum block.
# --------------------------------------------------------------------------- #

@dataclass
class RingBlock:
    """The solution of one momentum block (:func:`ring_amplitudes`):
    ``amplitudes[r, p]`` = :math:`t_{pr}` (rows the partner block's pairs,
    columns this block's), ``omega`` and ``partner_omega`` the excitation
    energies of the two momenta, ``energy`` the amplitude formula's share of
    :math:`E_c` (Hartree, supercell) and ``plasmon`` (``plasmon_partner``)
    the plasmon formula's for each momentum."""

    amplitudes: np.ndarray
    omega: np.ndarray
    partner_omega: np.ndarray
    energy: float
    plasmon: float
    plasmon_partner: float


def ring_amplitudes(delta, L, delta_partner, L_partner) -> RingBlock:
    r"""Direct-RPA amplitudes of one momentum block.

    ``delta`` (``(n,)``, Hartree) are the gaps of the block's pairs and
    ``L`` (``(n, n_G)``) their scaled transition densities, so that
    :math:`A = \operatorname{diag}\Delta + 2LL^\dagger`; ``delta_partner``
    and ``L_partner`` those of the opposite momentum, ``L_partner`` at the
    opposite vectors, so that :math:`B = 2L L_\text{partner}^T`.  For real
    orbitals and one block, ``L_partner = L``.  The Hermitian matrix
    :math:`M = \begin{pmatrix}A & B\\ B^\dagger & A'^*\end{pmatrix}` is
    positive definite; with :math:`M = CC^\dagger`, the eigenvectors
    :math:`w` of :math:`C^\dagger J C` (:math:`J = \operatorname{diag}(1,
    -1)`) give :math:`(X, Y) = C^{-\dagger}w` with :math:`|w|^2 = \Omega`.
    """
    from scipy.linalg import cholesky, solve_triangular

    delta = np.asarray(delta, dtype=float)
    delta_partner = np.asarray(delta_partner, dtype=float)
    n, m = len(delta), len(delta_partner)
    V = np.concatenate([np.asarray(L, dtype=complex),
                        np.conj(np.asarray(L_partner, dtype=complex))])
    M = 2.0 * (V @ V.conj().T)
    M[np.diag_indices(n + m)] += np.concatenate([delta, delta_partner])
    C = cholesky(M, lower=True)
    signs = np.concatenate([np.ones(n), -np.ones(m)])
    H = C.conj().T @ (signs[:, None] * C)
    values, w = np.linalg.eigh(0.5 * (H + H.conj().T))
    positive = values[m:]
    if positive.min() <= 0.0 or values[:m].max() >= 0.0:
        raise RuntimeError("the RPA problem is unstable (an imaginary "
                           "frequency): the reference is not a minimum")
    w = w[:, m:] * np.sqrt(positive)[None, :]
    v = solve_triangular(C, w, lower=True, trans=2)
    X, Y = v[:n], v[n:]
    T = np.linalg.solve(X.T, Y.T).T                       # Y X^{-1}
    B = 2.0 * (L @ np.asarray(L_partner).T)
    energy = 0.5 * float(np.real(np.sum(B.T * T)))
    trace_a = float(np.sum(delta) + 2.0 * np.sum(np.abs(L) ** 2))
    trace_partner = float(np.sum(delta_partner)
                          + 2.0 * np.sum(np.abs(L_partner) ** 2))
    return RingBlock(amplitudes=T, omega=positive,
                     partner_omega=-values[:m][::-1],
                     energy=energy,
                     plasmon=0.5 * (float(np.sum(positive)) - trace_a),
                     plasmon_partner=0.5 * (float(np.sum(-values[:m]))
                                            - trace_partner))


def ring_amplitudes_iterative(delta, L, delta_partner, L_partner, *,
                              tol: float = AMPLITUDE_TOLERANCE,
                              max_iter: int = 200, history: int = 8):
    r"""``(T, energy, iterations)``: the amplitudes of
    :func:`ring_amplitudes` from the Riccati equation itself,
    :math:`\Delta'T + T\Delta + 2(L'^* + TL)(L^\dagger + L'^TT) = 0`,
    iterated as :math:`T \leftarrow -2SR/(\Delta'_r + \Delta_p)` with
    DIIS -- products through the :math:`n_G` vectors only, so a step costs
    :math:`O(nm\,n_G)` against the eigenproblem's :math:`O((n+m)^3)`.
    ``None`` for ``T`` when it does not converge to ``tol`` (the largest
    residual, Hartree-free) within ``max_iter``."""
    delta = np.asarray(delta, dtype=float)
    delta_partner = np.asarray(delta_partner, dtype=float)
    L = np.asarray(L, dtype=complex)
    Lp = np.asarray(L_partner, dtype=complex)
    denominator = delta_partner[:, None] + delta[None, :]
    left, right = np.conj(Lp), L.conj().T
    T = np.zeros((len(delta_partner), len(delta)), dtype=complex)
    trials, residuals = [], []
    for iteration in range(1, int(max_iter) + 1):
        new = -2.0 * ((left + T @ L) @ (right + Lp.T @ T)) / denominator
        residual = new - T
        size = float(np.abs(residual).max())
        if size < tol:
            T = new
            break
        if not np.isfinite(size) or size > 1e3:
            return None, None, iteration
        trials.append(new)
        residuals.append(residual)
        if len(trials) > history:
            trials.pop(0)
            residuals.pop(0)
        k = len(residuals)
        overlap = np.empty((k + 1, k + 1))
        overlap[:k, :k] = [[np.real(np.vdot(a, b)) for b in residuals]
                           for a in residuals]
        overlap[k, :k] = overlap[:k, k] = -1.0
        overlap[k, k] = 0.0
        rhs = np.zeros(k + 1)
        rhs[k] = -1.0
        try:
            weights = np.linalg.solve(overlap, rhs)[:k]
            T = sum(w * t for w, t in zip(weights, trials))
        except np.linalg.LinAlgError:
            T = new
    else:
        return None, None, iteration
    energy = float(np.real(np.sum((Lp.T @ T) * L.T)))
    return T, energy, iteration


def density_corrections(T, holes, particles, keys, n_bands: int) -> dict:
    r"""The amplitudes' corrections to the density, per k-point.

    ``T`` is ``amplitudes`` of a :class:`RingBlock` (rows the partner
    block's pairs, columns this block's); the columns' pairs are ``holes``
    (band at the hole's k-point), ``particles`` (band at the particle's)
    and ``keys`` ``(hole k-point, particle k-point)`` per column.  Returns
    ``{k: (n_bands, n_bands)}`` with, at the holes' k-point,
    :math:`\Delta\gamma_{ij} = -\sum_{m,r}t_{(jm)r}t^*_{(im)r}` and at the
    particles' :math:`\Delta\gamma_{ab} = \sum_{n,r}t_{(na)r}t^*_{(nb)r}`
    (a state partly occupied in a metal collects both)."""
    out: dict = {}
    holes = np.asarray(holes)
    particles = np.asarray(particles)
    keys = np.asarray(keys)
    eye = np.eye(int(n_bands))

    def add(k, matrix):
        k = int(k)
        out[k] = out[k] + matrix if k in out else matrix

    for key in np.unique(keys, axis=0):
        columns = np.flatnonzero(np.all(keys == key[None, :], axis=1))
        Tk = T[:, columns]
        W = Tk.T @ np.conj(Tk)              # W[c, c'] = sum_r t_cr t*_c'r
        h, a = holes[columns], particles[columns]
        Eh, Ea = eye[:, h], eye[:, a]       # one-hot, (n_bands, n_c)
        add(key[1], Ea @ (W * (h[:, None] == h[None, :])) @ Ea.T)
        add(key[0], -Eh @ (W * (a[:, None] == a[None, :])).T @ Eh.T)
    return out


def molecular_ring_density(energies, n_occupied: int, L):
    r"""``(gamma, energy, plasmon)``: the direct-RPA density matrix (spin
    summed, in the orbitals' basis) and correlation energy (both formulas,
    Hartree) of a closed shell with orbital ``energies``, the lowest
    ``n_occupied`` doubly occupied, and real factors ``L`` (``(M, M,
    n_aux)``) of the Coulomb integrals, :math:`(pq|rs) = \sum_Q L_{pqQ}
    L_{rsQ}` -- a density-fitted or Cholesky-decomposed molecule."""
    e = np.asarray(energies, dtype=float)
    o = int(n_occupied)
    M = len(e)
    i, a = np.meshgrid(np.arange(o), np.arange(o, M), indexing="ij")
    i, a = i.ravel(), a.ravel()
    delta = e[a] - e[i]
    Lp = np.asarray(L)[i, a, :]
    block = ring_amplitudes(delta, Lp, delta, Lp)
    gamma = np.zeros((M, M), dtype=complex)
    gamma[np.arange(o), np.arange(o)] = 2.0
    zeros = np.zeros_like(i)
    for matrix in density_corrections(block.amplitudes, i, a,
                                      np.stack([zeros, zeros], axis=1),
                                      M).values():
        gamma += matrix
    return gamma, block.energy, block.plasmon


# --------------------------------------------------------------------------- #
# The crystal.
# --------------------------------------------------------------------------- #

@dataclass
class CrystalNaturalOrbitals:
    r"""The Bloch natural orbitals of a crystal's RPA density matrix.

    On the full Gamma-centered ``size`` mesh (``kpoints`` fractional, C
    order) and the ``bands`` of the basis: ``energies`` (eV) and
    ``reference`` occupations (0..2) of the Kohn-Sham states,
    ``occupations[k]`` the natural occupations at each k-point, largest
    first, and ``vectors[k]`` (``(n_bands, n_bands)``) the natural orbitals
    as columns in the band basis.  ``correlation_energy`` is the direct-RPA
    :math:`E_c` per cell (eV) from the amplitudes, ``plasmon_energy`` the
    same from the excitation energies (a check; ``None`` when the
    amplitudes were iterated).  ``threshold`` is the
    deviation from 2 or 0 that selects an orbital
    (:meth:`n_selected`); ``timings`` in seconds.
    """

    size: tuple
    kpoints: np.ndarray
    bands: tuple
    energies: np.ndarray
    reference: np.ndarray
    occupations: np.ndarray
    vectors: np.ndarray
    correlation_energy: float
    plasmon_energy: float | None
    cutoff: float
    threshold: float = DEFAULT_DEVIATION_THRESHOLD
    fermi_level: float = 0.0
    n_pairs: int = 0
    timings: dict = field(default_factory=dict)

    @property
    def deviations(self) -> np.ndarray:
        """``(nk, n_bands)``: each natural occupation's distance from 2 or
        0, the measure of how correlated the orbital is."""
        n = self.occupations
        return np.minimum(n, 2.0 - n)

    @property
    def electron_count(self) -> float:
        """The trace of the density matrix, averaged over the mesh."""
        return float(np.mean(np.sum(self.occupations, axis=1)))

    @property
    def idempotency(self) -> float:
        r""":math:`\operatorname{tr}[\gamma/2 - (\gamma/2)^2]` per cell, zero
        for a single determinant."""
        x = 0.5 * self.occupations
        return float(np.mean(np.sum(x - x * x, axis=1)))

    def n_selected(self, threshold=None) -> int:
        """The number of natural orbitals per cell whose deviation reaches
        ``threshold`` (default :attr:`threshold`), averaged over the mesh
        and rounded: the number of functions a ``windows="rpa"``
        Wannierization builds."""
        threshold = self.threshold if threshold is None else float(threshold)
        return int(round(float(np.mean(np.sum(self.deviations >= threshold,
                                              axis=1)))))

    @property
    def n_functions(self) -> int:
        return self.n_selected()

    def selected(self, n_functions=None) -> np.ndarray:
        """``(nk, J)``: at each k-point the indices of the ``J`` most
        correlated natural orbitals (largest deviation first)."""
        J = self.n_functions if n_functions is None else int(n_functions)
        return np.argsort(-self.deviations, axis=1, kind="stable")[:, :J]

    def band_projectabilities(self, n_functions, bands=None,
                              n_kpoints=None) -> np.ndarray:
        r"""``(nk, len(bands))``: each band's weight in the span of the
        ``n_functions`` most correlated natural orbitals at its k-point,
        :math:`p_{n\mathbf k} = \sum_{\nu}|U_{n\nu}(\mathbf k)|^2` -- the
        projectability of ``windows="rpa"``."""
        if n_kpoints is not None and int(n_kpoints) != len(self.kpoints):
            raise ValueError(
                f"the natural orbitals are on a {self.size} mesh "
                f"({len(self.kpoints)} points), the Wannier functions on "
                f"one of {n_kpoints}: use the same kpts")
        chosen = self.selected(n_functions)
        p = np.array([np.sum(np.abs(U[:, c]) ** 2, axis=1)
                      for U, c in zip(self.vectors, chosen)])
        if bands is None:
            return p
        index = {b: i for i, b in enumerate(self.bands)}
        missing = [b for b in bands if b not in index]
        if missing:
            raise ValueError(f"bands {missing} are not among the natural "
                             f"orbitals' bands {self.bands}")
        return p[:, [index[b] for b in bands]]

    def summary(self) -> str:
        """The natural-occupation spectrum: the strongly occupied orbitals
        (occupation above 1) and the weakly occupied ones, each by rank at
        every k-point, with the occupation's range over the mesh; ``*``
        marks the ranks the threshold selects."""
        J = self.n_functions
        chosen = self.selected(J)
        lines = [
            "RPA natural orbitals (direct ring CCD density)",
            f"  mesh {self.size}, bands {self.bands[0]}..{self.bands[-1]}, "
            f"cutoff {self.cutoff:g} Bohr^-1, largest block "
            f"{self.n_pairs} pairs",
            f"  E_c {self.correlation_energy:.6f} eV/cell"
            + ("" if self.plasmon_energy is None else
               f" (plasmon {self.plasmon_energy:.6f})"),
            f"  trace {self.electron_count:.8f}, tr[g/2 - (g/2)^2] "
            f"{self.idempotency:.3e}",
            f"  {J} per k-point selected (deviation >= "
            f"{self.threshold:g})"]
        n = self.occupations
        for name, upper in (("strongly occupied", True),
                            ("weakly occupied", False)):
            side = n >= 1.0 if upper else n < 1.0
            counts = side.sum(axis=1)
            if counts.max() == 0:
                continue
            lines.append(f"  {name}: rank  occupation min .. max"
                         "     selected at")
            for rank in range(int(counts.min())):
                values, picked = [], 0
                for k in range(len(n)):
                    index = np.flatnonzero(side[k])
                    order = index[np.argsort(-self.deviations[k, index],
                                             kind="stable")]
                    values.append(n[k, order[rank]])
                    picked += int(order[rank] in chosen[k])
                lines.append(f"    {rank:<4d} {min(values):.6f} .. "
                             f"{max(values):.6f}   {picked}/{len(n)} "
                             "k-points")
        return "\n".join(lines)


def _mesh(size):
    size = tuple(int(n) for n in size)
    cells = np.indices(size).reshape(3, -1).T
    return size, cells


def gather_states(solver, size, bands, fermi_level):
    """``(fractional, states)``: per mesh point ``(psi, a, e, F)`` -- the
    smooth parts on the cell grid, the projector coefficients, the energies
    (Hartree) and electrons per state at the SCF's Fermi level and
    smearing."""
    from .periodic_dft import occupation
    from .wannier import mesh_states

    fractional, data, vectors, energies = mesh_states(solver, size, bands, 0)
    states = []
    for d, v, e in zip(data, vectors, energies):
        F = 2.0 * occupation((e - fermi_level) / solver.width, solver.method)
        states.append((v.T @ d.psi, d.projections.conj().T @ v, e, F))
    return fractional, states


def _vectors_within(crystal, q, cutoff):
    """Integer ``g`` ``(3, n)`` of the vectors q + G within ``cutoff``, the
    zero vector left out."""
    shape = tuple(int(n) for n in crystal.grid.shape)
    B = rc.reciprocal_vectors(crystal.lattice)
    g = np.stack(np.meshgrid(*[np.fft.fftfreq(n, d=1.0 / n).astype(int)
                               for n in shape], indexing="ij")).reshape(3, -1)
    shifted = np.asarray(q)[:, None] + B @ g
    length = np.sum(shifted * shifted, axis=0)
    return g[:, (length <= cutoff ** 2) & (length > 1e-20)]


def transition_densities(crystal, first, second, holes, particles, q, g):
    r"""``(n_pairs, n_G)``: :math:`\rho_p(\mathbf q+\mathbf G) = \int
    \psi^*_{n\mathbf k}\psi_{m,\mathbf k+\mathbf q}e^{-i(\mathbf q+\mathbf
    G)\cdot\mathbf r}` with the compensation charges, for the pairs
    (``holes[i]`` of state set ``first``, ``particles[i]`` of ``second``,
    each ``(psi, a, ...)``) on the vectors ``g`` (integer, ``(3, n)``) --
    the transition densities of
    :func:`~mandacaru.algorithms.wannier.polarizability`."""
    grid = crystal.grid
    shape = tuple(int(n) for n in grid.shape)
    B = rc.reciprocal_vectors(crystal.lattice)
    r = np.stack([np.ravel(grid.X), np.ravel(grid.Y), np.ravel(grid.Z)])
    qG = np.asarray(q)[:, None] + B @ g
    fft_index = np.ravel_multi_index(tuple(np.mod(g, np.asarray(
        shape)[:, None])), shape)
    to_G = grid.dV * np.exp(-1j * (rc.grid_origin(grid) @ (B @ g)))
    psi1, a1 = first[0], first[1]
    psi2, a2 = second[0], second[1]
    product = np.conj(psi1[holes]) * psi2[particles] * np.exp(
        -1j * (np.asarray(q) @ r))[None, :]
    transform = np.fft.fftn(product.reshape(-1, *shape),
                            axes=(1, 2, 3)).reshape(len(holes), -1)
    rho = transform[:, fft_index] * to_G[None, :]
    transforms = crystal.compensation_transforms(qG)
    for c, (atom, L, M) in enumerate(crystal.channels):
        own = crystal.projector_columns(atom)
        blk = crystal.multipole_blocks[(atom, L, M)]
        Q = np.einsum("ip,ij,jp->p", np.conj(a1[own][:, holes]), blk,
                      a2[own][:, particles])
        rho += Q[:, None] * transforms[c][None, :]
    return rho


@dataclass
class _Block:
    """The pairs of one momentum block and their scaled densities."""

    q_cell: np.ndarray
    g: np.ndarray
    holes: np.ndarray
    particles: np.ndarray
    keys: np.ndarray               # (hole k-point, particle k-point)
    delta: np.ndarray
    weight: np.ndarray
    L: np.ndarray


def momentum_blocks(crystal, fractional, states, size, cutoff,
                    pair_threshold: float = PAIR_THRESHOLD) -> list:
    """One :class:`_Block` per mesh vector q: the pairs ``(n k -> m, k+q)``
    with weight ``F_n - F_m`` above ``pair_threshold`` and their scaled
    transition densities on q + G within ``cutoff``."""
    size, cells = _mesh(size)
    N = int(np.prod(size))
    B = rc.reciprocal_vectors(crystal.lattice)
    volume = abs(float(np.linalg.det(crystal.lattice)))
    blocks = []
    for q_cell in cells:
        q = B @ (q_cell / np.asarray(size, dtype=float))
        g = _vectors_within(crystal, q, cutoff)
        qG = q[:, None] + B @ g
        kernel = np.sqrt(4.0 * np.pi / np.sum(qG * qG, axis=0)
                         / (N * volume))
        partner = np.ravel_multi_index(tuple(((cells + q_cell)
                                              % np.asarray(size)).T), size)
        holes, particles, keys, delta, weight, L = [], [], [], [], [], []
        for k, k2 in enumerate(partner):
            _p1, _a1, e1, F1 = states[k]
            _p2, _a2, e2, F2 = states[k2]
            w = F1[:, None] - F2[None, :]
            n, m = np.nonzero(w > pair_threshold)
            if n.size == 0:
                continue
            rho = transition_densities(crystal, states[k], states[k2], n, m,
                                       q, g)
            scale = np.sqrt(0.5 * w[n, m])
            holes.append(n)
            particles.append(m)
            keys.append(np.stack([np.full(n.size, k), np.full(n.size, k2)],
                                 axis=1))
            delta.append(e2[m] - e1[n])
            weight.append(w[n, m])
            L.append(scale[:, None] * rho * kernel[None, :])
        if not holes:
            raise ValueError("no particle-hole pair carries weight: the "
                             "crystal has no empty band in the basis")
        blocks.append(_Block(q_cell=q_cell, g=g,
                             holes=np.concatenate(holes),
                             particles=np.concatenate(particles),
                             keys=np.concatenate(keys),
                             delta=np.concatenate(delta),
                             weight=np.concatenate(weight),
                             L=np.concatenate(L)))
    if min(float(b.delta.min()) for b in blocks) <= 0.0:
        raise ValueError("a weighted pair has no gap: the occupations are "
                         "not a function of the energy")
    return blocks


def partner_columns(blocks, size) -> list:
    """``[(partner block, column index), ...]``: for block q, the block of
    -q and, for each of q's vectors q + G, the column of -(q + G) among the
    partner's."""
    size, cells = _mesh(size)
    out = []
    for block in blocks:
        minus = (-block.q_cell) % np.asarray(size)
        index = int(np.ravel_multi_index(tuple(minus), size))
        shift = (minus + block.q_cell) // np.asarray(size)
        other = blocks[index]
        lookup = {tuple(col): c for c, col in enumerate(other.g.T)}
        columns = np.array([lookup[tuple(-col - shift)] for col in
                            block.g.T])
        out.append((index, columns))
    return out


def crystal_rpa(solver, size, fermi_level, *, bands=None, cutoff=None,
                threshold=DEFAULT_DEVIATION_THRESHOLD,
                pair_threshold: float = PAIR_THRESHOLD,
                solver_method: str = "iterative"
                ) -> CrystalNaturalOrbitals:
    """:class:`CrystalNaturalOrbitals` of a converged, spin-restricted
    crystal on the full Gamma-centered ``size`` mesh: the direct-RPA
    density matrix from the ``bands`` (default every band of the basis)
    at the SCF's ``fermi_level`` (Hartree) and smearing, its Coulomb
    integrals on the vectors within ``cutoff`` (Bohr^-1, default
    :data:`RPA_CUTOFF`).  ``solver_method`` is ``"iterative"`` (the Riccati
    equation with DIIS, :func:`ring_amplitudes_iterative`, the eigenproblem
    where it does not converge) or ``"eigen"`` (:func:`ring_amplitudes`,
    which also gives the plasmon formula's energy)."""
    if solver_method not in ("iterative", "eigen"):
        raise ValueError(f"solver_method must be 'iterative' or 'eigen'; got "
                         f"{solver_method!r}")
    if int(solver.n_spins) != 1:
        raise NotImplementedError("the RPA density matrix is implemented for "
                                  "a spin-restricted crystal")
    crystal = solver.crystal
    cutoff = RPA_CUTOFF if cutoff is None else float(cutoff)
    bands = tuple(range(crystal.M)) if bands is None else tuple(
        int(b) for b in bands)
    size, _cells = _mesh(size)
    N = int(np.prod(size))
    timings = {}
    t0 = time.perf_counter()
    fractional, states = gather_states(solver, size, bands, fermi_level)
    timings["states"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    blocks = momentum_blocks(crystal, fractional, states, size, cutoff,
                             pair_threshold)
    partners = partner_columns(blocks, size)
    timings["integrals"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    solved: dict = {}
    energy = plasmon = 0.0
    nb = len(bands)
    gamma = np.zeros((N, nb, nb), dtype=complex)
    for k, (_psi, _a, _e, F) in enumerate(states):
        gamma[k][np.diag_indices(nb)] = F
    for q, block in enumerate(blocks):
        index, columns = partners[q]
        if index in solved and index != q:
            T = solved.pop(index).T
            ring_energy, ring_plasmon = None, None
        else:
            other = blocks[index]
            T = None
            if solver_method == "iterative":
                T, ring_energy, _steps = ring_amplitudes_iterative(
                    block.delta, block.L, other.delta, other.L[:, columns])
                ring_plasmon = np.nan
            if T is None:
                ring = ring_amplitudes(block.delta, block.L, other.delta,
                                       other.L[:, columns])
                T = ring.amplitudes
                ring_energy = ring.energy
                ring_plasmon = ring.plasmon + (ring.plasmon_partner
                                               if index != q else 0.0)
            if index != q:
                solved[q] = T
                ring_energy *= 2.0           # the -q block's equal share
        if ring_energy is not None:
            energy += ring_energy
            plasmon += ring_plasmon
        for kk, matrix in density_corrections(
                T, block.holes, block.particles, block.keys, nb).items():
            gamma[kk] += matrix
    timings["amplitudes"] = time.perf_counter() - t0
    occupations, vectors = [], []
    for g in gamma:
        values, U = np.linalg.eigh(0.5 * (g + g.conj().T))
        occupations.append(values[::-1])
        vectors.append(U[:, ::-1])
    return CrystalNaturalOrbitals(
        size=size, kpoints=fractional, bands=bands,
        energies=np.array([s[2] for s in states]) * HARTREE_TO_EV,
        reference=np.array([s[3] for s in states]),
        occupations=np.array(occupations), vectors=np.array(vectors),
        correlation_energy=energy / N * HARTREE_TO_EV,
        plasmon_energy=(None if np.isnan(plasmon)
                        else plasmon / N * HARTREE_TO_EV), cutoff=cutoff,
        threshold=float(threshold), fermi_level=float(fermi_level)
        * HARTREE_TO_EV,
        n_pairs=max(len(b.delta) for b in blocks), timings=timings)


def chi_correlation_energy(blocks, partners, n_frequencies: int = 48,
                           scale: float = 0.5) -> float:
    r"""The direct-RPA correlation energy (Hartree, supercell) from the
    polarizability on the imaginary axis,

    .. math::

        E_c = \frac1{2\pi}\int_0^\infty d\omega\sum_{\mathbf q}
            \operatorname{Re}\operatorname{tr}\big[\ln(1 - \Pi_{\mathbf
            q}(i\omega)) + \Pi_{\mathbf q}(i\omega)\big],

    :math:`\Pi = v^{1/2}\chi_0 v^{1/2}` assembled from the blocks' scaled
    transition densities, :math:`\Pi = 2L^T(-\Delta + i\omega)^{-1}L^* -
    2L'^\dagger(\Delta' + i\omega)^{-1}L'` -- a check on the amplitudes
    independent of them.  Gauss-Legendre in :math:`\omega = s(1+x)/(1-x)`.
    """
    x, wx = np.polynomial.legendre.leggauss(int(n_frequencies))
    omega = scale * (1.0 + x) / (1.0 - x)
    jacobian = wx * 2.0 * scale / (1.0 - x) ** 2
    total = 0.0
    for q, block in enumerate(blocks):
        index, columns = partners[q]
        L = block.L
        Lp = blocks[index].L[:, columns]
        dp = blocks[index].delta
        for w, dw in zip(omega, jacobian):
            Pi = (2.0 * (L.T * (1.0 / (-block.delta + 1j * w))[None, :])
                  @ np.conj(L)
                  - 2.0 * (Lp.conj().T * (1.0 / (dp + 1j * w))[None, :])
                  @ Lp)
            _sign, logdet = np.linalg.slogdet(np.eye(len(Pi)) - Pi)
            total += dw * (float(np.real(logdet))
                           + float(np.real(np.trace(Pi))))
    return total / (2.0 * np.pi)
