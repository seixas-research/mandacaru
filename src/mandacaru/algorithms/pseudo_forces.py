# -*- coding: utf-8 -*-
# file: algorithms/pseudo_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Hellmann-Feynman and Pulay forces for every atom-centered basis.

Pseudopotential families (PAW-LCAO, UPAW-LCAO) and all-electron bases share one
derivative: the difference is only which terms exist.  An all-electron
Hamiltonian has no projectors, no augmented overlap and no compensation
charges, and its per-atom local potential is the bare (softened) ``-Z/r``;
every one of those pieces is detected from the integrals themselves.

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

* :math:`S = \tilde S + C q C^\dagger` (the PAW-LCAO-augmented overlap),
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
  use the energy's own Laplacian
  (:func:`~mandacaru.algorithms.forces.one_body_cross`): the
  finite-difference kernel on the stacked ``[phi; dphi]`` basis, or the moved
  functions against the FFT Laplacian of the unmoved ones for
  ``kinetic="spectral"``.  The two-electron derivative is contracted from
  the pair densities against the Coulomb potentials the energy uses.
* The PAW-LCAO **projections** :math:`C_{\mu p} = \langle\phi_\mu|p_p\rangle` are
  atom-centered quadratures over each projector's sphere
  (:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.projections`), so they
  depend only on the separation of the function and the projector.  Their
  derivative is the same quadrature of :math:`\partial\phi_\mu/\partial\mathbf R`,
  :math:`G_{\mu p}`: the Pulay part (basis function moving) is :math:`+G` and
  the Hellmann-Feynman part (projector moving) is :math:`-G`, so the total is
  exactly translation invariant.
* PAW-LCAO's **local potential** is range-separated
  (:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.short_range_local`): the
  grid samples only the long-range Gaussian-ion half, and the short-range half
  is an atom-centered sphere integral :math:`I^A_{pq}`.  It is differentiated
  exactly like the projections, through one array
  :math:`G^A_{pqk} = \int(\partial\phi_p/\partial R_{p,k})^*\phi_q v^{sr}_A`:
  Pulay for atom :math:`B` is
  :math:`\sum_A([p\in B]\,G^A_{pqk} + [q\in B]\,\overline{G^A_{qpk}})` and
  Hellmann-Feynman for atom :math:`A` is
  :math:`-(G^A_{pqk} + \overline{G^A_{qpk}})` over its own sphere, so a rigid
  translation of everything cancels identically.
* The contraction with :math:`\partial E/\partial(S,h)` is the
  **directional derivative** of the small algebraic energy
  :math:`E(S, h, g)` along :math:`(dS, dh)`, by a central difference
  extrapolated from two steps (Richardson, order :math:`\epsilon^4`; the
  algebra is :math:`O(M^5)` with :math:`M` basis functions).  That covers
  the derivative of :math:`S^{-1/2}` and the complex MO transform without
  hand-derived energy-weighted density matrices.
* The energy is **linear in** :math:`g`, so its part along :math:`dg` is
  :math:`\langle\partial E/\partial g, dg\rangle` exactly, with
  :math:`\partial E/\partial g` the two-body RDM taken back to the atomic
  orbitals once for the whole gradient (``pair_derivative``).  No
  :math:`dg` is ever formed: the grid pairs' part streams the pair
  potentials once into every atom and direction
  (:func:`~mandacaru.algorithms.forces._two_body_pulay`), and the
  compensation charges' and spheres' low-rank parts contract the
  derivative once with each fixed factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..integrals.packed import dense_tensor, is_packed

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
    :func:`~mandacaru.core.hamiltonian.spin_block_integrals`.
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
    pipeline :class:`~mandacaru.core.hamiltonian.MolecularIntegrals` and
    :class:`~mandacaru.algorithms.hartree_fock.RHF` follow to build the
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
        w = np.maximum(w, 1e-14)
        return (U * (1.0 / np.sqrt(w))) @ U.conj().T

    def with_orbitals(self, A, h, g) -> float:
        h_mo = A.conj().T @ h @ A
        if is_packed(g):
            # Packed to packed (real orbitals), contracted with the RDM.
            two = g.transform(A).contract(self.G)
        else:
            g_mo = np.einsum("ap,bq,cr,ds,abcd->pqrs", A.conj(), A.conj(), A,
                             A, g, optimize=True)
            two = np.einsum("pqrs,pqrs->", self.G, g_mo)
        return float(np.real(np.einsum("pq,pq->", self.D, h_mo) + 0.5 * two))

    def __call__(self, S, h, g) -> float:
        return self.with_orbitals(self.lowdin(S) @ self.V, h, g)

    def pair_derivative(self, S) -> np.ndarray:
        r""":math:`\partial E/\partial g_{abcd}` at overlap ``S``, as the
        ``(M^2, M^2)`` matrix :math:`P_{(ac),(bd)}`: the energy is linear in
        ``g``, :math:`E_g = \tfrac12\sum\Gamma_{pqrs} g^{\rm MO}_{pqrs}`,
        so this is the two-body RDM taken back to the atomic orbitals,
        :math:`\tfrac12\sum_{pqrs}\Gamma_{pqrs} A^*_{ap} A^*_{bq} A_{cr}
        A_{ds}`.  Contracted with a change of ``g`` it is the exact
        directional derivative of the two-body term (no difference step):
        the gradient never forms that change."""
        A = self.lowdin(S) @ self.V
        M = A.shape[0]
        P = np.einsum("ap,bq,cr,ds,pqrs->acbd", A.conj(), A.conj(), A, A,
                      self.G, optimize=True)
        return 0.5 * np.ascontiguousarray(P).reshape(M * M, M * M)

    def directional(self, S, h, g, dS, dh, dg,
                    step: float = DEFAULT_ALGEBRAIC_STEP, dh_so=None) -> float:
        """``<dE/dS, dS> + <dE/dh, dh> + <dE/dg, dg>`` by a central difference."""
        if dh_so is not None:
            raise ValueError("a spin-orbit derivative needs SpinorAlgebraicEnergy")

        def shifted(sign):
            return self(S + sign * step * dS if dS is not None else S,
                        h + sign * step * dh if dh is not None else h,
                        g + sign * step * dg if dg is not None else g)
        return (shifted(+1.0) - shifted(-1.0)) / (2.0 * step)

    def orbital_gradient(self, S, h, g, step: float = 1e-5) -> float | None:
        r"""Largest :math:`|\partial E/\partial\kappa_{pq}|` over orbital rotations.

        The size of the neglected orbital-response term.  It vanishes at both
        ends of the correlation range -- for the Hartree-Fock determinant
        (a converged RHF is orbital-stationary) and for the exact ground state
        of this orbital space (a full CI there is invariant under any rotation
        of those orbitals) -- so a nonzero value means the state is neither:
        the ansatz stopped short of that exact state, and this measures it to
        *first* order in the state error where the energy shows it to second.
        """
        from scipy.linalg import expm

        # Complex rotations leave the packed tensor's symmetry: it is read
        # in full here.
        g = dense_tensor(g)
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

def _moved_function(function, grid, k: int, delta: float,
                    cache: dict) -> np.ndarray:
    """``d f / d R_k`` of a centered function, on the flattened grid: minus
    its gradient (:meth:`~mandacaru.basis.base.BasisFunction.sample_gradient`),
    sampled once for the three directions (kept in ``cache``)."""
    if id(function) not in cache:
        cache[id(function)] = function.sample_gradient(grid, delta)
    return -cache[id(function)][k]


def _moved_radial(radial, center, grid, k: int, delta: float,
                  atom_geom=None) -> np.ndarray:
    """``d v(|r - R|) / d R_k`` of a spherical potential centered at ``center``."""
    delta = float(delta)
    if atom_geom is not None:
        dx, dy, dz, r2 = atom_geom
        dk = (dx, dy, dz)[k]
        d2 = delta * delta
        r_plus = np.sqrt(np.maximum(r2 - (2.0 * delta) * dk + d2, 0.0))
        r_minus = np.sqrt(np.maximum(r2 + (2.0 * delta) * dk + d2, 0.0))
        return (np.asarray(radial(r_plus), dtype=float).ravel()
                - np.asarray(radial(r_minus), dtype=float).ravel()) / (2.0 * delta)
    out = []
    for sign in (+1.0, -1.0):
        c = np.array(center, dtype=float)
        c[k] += sign * delta
        radius = np.sqrt((grid.X - c[0]) ** 2 + (grid.Y - c[1]) ** 2
                         + (grid.Z - c[2]) ** 2)
        out.append(np.asarray(radial(radius), dtype=float).ravel())
    return (out[0] - out[1]) / (2.0 * delta)


def _ionic_shift(integrals, centers, atom, k, step):
    """``{channel: dV}`` -- how the compensation-ion integrals change when
    ``atom`` moves along ``k``.

    Both roles matter: the moving atom carries its own compensation shape
    through its neighbors' potentials, and carries its ionic potential under
    everyone else's shapes.  Differentiating the quadrature directly would mean
    two derivative kernels, so the integral is simply re-evaluated at displaced
    centers -- it is smooth and deterministic, so a central difference is
    clean.
    """
    plus_centers = [np.array(c, dtype=float) for c in centers]
    minus_centers = [np.array(c, dtype=float) for c in centers]
    plus_centers[atom][k] += float(step)
    minus_centers[atom][k] -= float(step)
    plus = integrals.compensation_ionic_at(plus_centers)
    minus = integrals.compensation_ionic_at(minus_centers)
    return {c: (plus[c] - minus[c]) / (2.0 * float(step)) for c in plus}


def _sampled_multipole(dataset, center, grid, L, M, potential=None, *,
                       real: bool):
    """``v_L(|r - center|) Y_LM`` of a compensation multipole, on the grid
    (``potential(radius, r_g, L)`` in place of the Coulomb one), in the
    harmonics ``real`` states."""
    from ..basis._angular import spherical_harmonic
    from ..pseudopotentials.multipoles import shape_potential

    dx = grid.X.ravel() - center[0]
    dy = grid.Y.ravel() - center[1]
    dz = grid.Z.ravel() - center[2]
    radius = np.sqrt(dx * dx + dy * dy + dz * dz)
    value = (shape_potential if potential is None else potential)(
        radius, float(dataset.compensation_radius), int(L))
    if int(L) == 0:
        return value * spherical_harmonic(0, 0, 0.0, 0.0, real=real)
    safe = np.maximum(radius, 1e-300)
    theta = np.arccos(np.clip(dz / safe, -1.0, 1.0))
    phi = np.arctan2(dy, dx)
    return value * spherical_harmonic(int(L), int(M), theta, phi, real=real)


def _moved_multipole(dataset, center, grid, L, M, k, delta, potential=None,
                     *, real: bool):
    """Derivative of :func:`_sampled_multipole` as its center moves along ``k``."""
    step = np.zeros(3)
    step[k] = float(delta)
    return ((_sampled_multipole(dataset, center + step, grid, L, M, potential,
                                real=real)
             - _sampled_multipole(dataset, center - step, grid, L, M,
                                  potential, real=real))
            / (2.0 * float(delta)))


def _sampled_multipoles_atom(dataset, center, grid, levels, potential=None, *,
                             real: bool, geom=None) -> dict:
    """All ``{(L, M): v_L(|r - center|) Y_LM}`` of one atom, sharing geometry."""
    from ..pseudopotentials.multipoles import multipole_potentials_cartesian

    if not levels:
        return {}
    if geom is None:
        dx = grid.X.ravel() - center[0]
        dy = grid.Y.ravel() - center[1]
        dz = grid.Z.ravel() - center[2]
        radius = np.sqrt(dx * dx + dy * dy + dz * dz)
    else:
        dx, dy, dz, radius = geom
    return multipole_potentials_cartesian(
        dx, dy, dz, radius, float(dataset.compensation_radius), levels,
        potential=potential, real=real)


def _moved_multipoles_atom(dataset, center, grid, levels, k, delta,
                           potential=None, *, real: bool,
                           atom_geom=None) -> dict:
    """Derivative of :func:`_sampled_multipoles_atom` along ``k``."""
    delta = float(delta)
    if atom_geom is None:
        dx = grid.X.ravel() - center[0]
        dy = grid.Y.ravel() - center[1]
        dz = grid.Z.ravel() - center[2]
        r2 = dx * dx + dy * dy + dz * dz
    else:
        dx, dy, dz, r2 = atom_geom
    coords = [dx, dy, dz]
    dk = coords[k]
    d2 = delta * delta
    r_plus = np.sqrt(np.maximum(r2 - (2.0 * delta) * dk + d2, 0.0))
    r_minus = np.sqrt(np.maximum(r2 + (2.0 * delta) * dk + d2, 0.0))
    coords_plus = list(coords)
    coords_minus = list(coords)
    coords_plus[k] = dk - delta
    coords_minus[k] = dk + delta
    plus = _sampled_multipoles_atom(
        dataset, center, grid, levels, potential, real=real,
        geom=(coords_plus[0], coords_plus[1], coords_plus[2], r_plus))
    minus = _sampled_multipoles_atom(
        dataset, center, grid, levels, potential, real=real,
        geom=(coords_minus[0], coords_minus[1], coords_minus[2], r_minus))
    inv = 1.0 / (2.0 * delta)
    return {lm: (plus[lm] - minus[lm]) * inv for lm in plus}


# --------------------------------------------------------------------------- #
# The gradient.
# --------------------------------------------------------------------------- #

def _atom_potentials(integrals):
    """The per-atom local potential callables ``v_A(r)`` of this Hamiltonian.

    A pseudopotential family supplies its own local channel; an all-electron
    Hamiltonian's atom contributes the bare (softened) ``-Z/r`` -- exactly what
    :meth:`~mandacaru.integrals.potentials.Potentials.nuclear_potential` samples,
    so the Hellmann-Feynman term differentiates the same operator the energy
    was built from.

    The engine may sample only *part* of its local potential on the grid -- PAW-LCAO
    keeps the long-range Gaussian-ion half there and integrates the rest on
    atom-centered spheres (``short_range_local``).  It then says so through
    ``local_potential_functions``, and that is what this returns: the grid half
    of the derivative has to differentiate the grid half of the operator.
    """
    named = getattr(integrals, "local_potential_functions", None)
    if named is not None:
        return list(named())
    datasets = getattr(integrals, "pseudopotentials", None)
    if datasets:
        return [dataset.local_potential for dataset in datasets]
    softening = float(getattr(integrals._potentials, "softening", 1e-12))
    potentials = []
    for charge, _center in integrals._potentials.nuclei:
        def coulomb(radius, Z=float(charge), floor=softening):
            return -Z / np.maximum(np.asarray(radius, dtype=float), floor)
        potentials.append(coulomb)
    return potentials


def _hermitian(X):
    return 0.5 * (X + X.conj().T)


def _displacement_derivative(slope, own):
    r"""The derivative of a two-center matrix when the functions flagged by
    ``own`` (an atom's) move with it.

    ``slope[a, b]`` is the derivative of :math:`M_{ab}` with respect to the
    displacement :math:`\mathbf R_b - \mathbf R_a` of its two centers
    (:meth:`~mandacaru.integrals.two_center.TwoCenter.gradient`): moving
    the bra's center is minus that and moving the ket's plus it, so a pair
    on the same atom does not change.
    """
    return (np.where(own[:, None], -slope, 0.0)
            + np.where(own[None, :], slope, 0.0))


@dataclass
class ScreenedExchange:
    r"""A screened hybrid's exact exchange, for :func:`pseudo_nuclear_gradient`.

    The energy is :math:`\sum_\sigma w_\sigma \sum P^\sigma_{pq}
    P^\sigma_{sr}\,g^{\rm SR}_{qrsp}` over the AO density matrices
    ``densities = [(w, P), ...]`` (a closed shell: one entry, :math:`w =
    -a/4`, both spins in :math:`P`; unrestricted: one per spin, :math:`w =
    -a/2`) plus, on PAW-LCAO, the one-center terms :math:`E^1(C^\dagger P
    C)` whose ``(P, P)`` operators :math:`\partial E^1/\partial(C^\dagger
    P^\sigma C)^T` are ``operators`` (one per entry, or ``None``).

    Only what moves with the *integrals* is differentiated here -- the
    short-range tensor :math:`g^{\rm SR}` (the grid pairs under the
    screened kernel,
    :class:`~mandacaru.integrals.poisson.PoissonFFTSolver` with ``omega``, through
    the moving basis functions, and on PAW-LCAO the augmentation minus its
    long-range part, through the moving compensation charges) and the
    projections ``C`` of the
    one-center terms.  The dependence on the overlap through
    :math:`P = S^{-1/2} D S^{-1/2}` is the caller's, folded into the
    energy's fixed one-body matrix as for the semilocal potential.
    """

    omega: float
    densities: list
    operators: list | None = None


class SpinorAlgebraicEnergy(AlgebraicEnergy):
    r""":class:`AlgebraicEnergy` for a Hamiltonian written in GHF spinors.

    The orbitals are ``(2M, 2M)`` two-component spinors in mode order, the
    RDMs are the full spin-orbital ones (their spin-off-diagonal blocks are
    what spin-orbit coupling populates), and the one-body matrix gains the
    atomic-orbital spin-orbit matrix ``h_so`` (not Loewdin-transformed):
    :math:`E = \sum\gamma_{PQ}(A^\dagger H A)_{PQ}
    + \tfrac12\sum\Gamma_{PQRS}\langle PQ|RS\rangle_A` with
    :math:`A = (X \oplus X)V`, :math:`X = S^{-1/2}` and
    :math:`H = (h \oplus h) + h_{SO}`.
    """

    def __init__(self, spinors, gamma, gamma2, h_so):
        self.V = np.asarray(spinors, dtype=complex)
        self.D = np.asarray(gamma, dtype=complex)
        self.G = np.asarray(gamma2, dtype=complex)
        self.h_so = np.asarray(h_so, dtype=complex)

    def with_orbitals(self, A, h, g, h_so=None) -> float:
        from .hartree_fock import spinor_integrals

        M = h.shape[0]
        H = np.zeros((2 * M, 2 * M), dtype=complex)
        H[:M, :M] = H[M:, M:] = h
        H = H + (self.h_so if h_so is None else h_so)
        h_mo, g_mo = spinor_integrals(H, g, A)
        return float(np.real(np.einsum("pq,pq->", self.D, h_mo)
                             + 0.5 * np.einsum("pqrs,pqrs->", self.G, g_mo)))

    def __call__(self, S, h, g, h_so=None) -> float:
        X = self.lowdin(S)
        M = X.shape[0]
        XX = np.zeros((2 * M, 2 * M), dtype=complex)
        XX[:M, :M] = XX[M:, M:] = X
        return self.with_orbitals(XX @ self.V, h, g, h_so)

    def pair_derivative(self, S) -> np.ndarray:
        r""":meth:`AlgebraicEnergy.pair_derivative` for spinors: electron 1
        keeps its spin between ``P`` and ``R``, electron 2 between ``Q`` and
        ``S``, so the spin-free tensor's derivative sums the four spin
        blocks of the spinor coefficients."""
        X = self.lowdin(S)
        M = X.shape[0]
        XX = np.zeros((2 * M, 2 * M), dtype=complex)
        XX[:M, :M] = XX[M:, M:] = X
        A = XX @ self.V
        parts = (A[:M], A[M:])
        P = 0
        for Cs in parts:
            for Ct in parts:
                P = P + np.einsum("ap,bq,cr,ds,pqrs->acbd", Cs.conj(),
                                  Ct.conj(), Cs, Ct, self.G, optimize=True)
        return 0.5 * np.ascontiguousarray(P).reshape(M * M, M * M)

    def directional(self, S, h, g, dS, dh, dg,
                    step: float = DEFAULT_ALGEBRAIC_STEP, dh_so=None) -> float:
        """The same central difference, the spin-orbit matrix moving too."""
        def shifted(sign):
            return self(S + sign * step * dS if dS is not None else S,
                        h + sign * step * dh if dh is not None else h,
                        g + sign * step * dg if dg is not None else g,
                        self.h_so + sign * step * dh_so
                        if dh_so is not None else None)
        return (shifted(+1.0) - shifted(-1.0)) / (2.0 * step)

    def orbital_gradient(self, S, h, g, step: float = 1e-5) -> None:
        """Not measured for spinors: the (2M)^2 rotations each need a
        (2M)^4 transformation.  A full CI in the spinor space, which is what
        the spin-orbit runs reach, has none to measure."""
        return None


def pseudo_nuclear_gradient(integrals, gamma, gamma2, *, atom_of_orbital,
                            orbital_delta=None, include_pulay: bool = True,
                            algebraic_step: float = DEFAULT_ALGEBRAIC_STEP,
                            orbital_gradient: bool = True, energy=None,
                            field=None, exchange=None,
                            orbital_response: dict | None = None):
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
    energy : object, optional
        The algebraic energy :math:`E(S, h, g)` to differentiate, in place of
        the one built from ``gamma``, ``gamma2`` and the integrals' molecular
        orbitals (which are then not needed): any object with
        ``__call__(S, h, g)``, ``directional(...)`` like
        :class:`AlgebraicEnergy` and a spatial one-body matrix ``D`` (the
        electron count is its trace).  The Kohn-Sham gradient
        (:mod:`mandacaru.algorithms.dft`) supplies its own.  An energy whose
        ``two_body`` attribute is false carries the Hartree term in its
        one-body matrix (linearized about the converged density, with the
        potential in ``field``), so the gradient never forms the two-body
        tensor, its pair-density derivatives or any ``M^4`` array: the
        compensation-charge terms are contracted with the fixed density
        matrix to scalars.
    field : object, optional
        Terms of the energy that are not functions of ``(S, h, g)``.
        ``field.pulay(dpsi)`` returns the change of an extra one-body matrix
        (already folded into ``energy``) when the basis functions move by
        ``dpsi``, added to ``dh``; ``field.hellmann_feynman(atom, k)`` returns
        a gradient component (Hartree/Bohr) for whatever else moves with
        ``atom``.
    exchange : ScreenedExchange, optional
        A screened hybrid's exact exchange, which is not a function of
        ``(S, h, g)`` alone: the derivative of its short-range tensor and of
        its PAW-LCAO one-center terms through the projections.
    orbital_response : dict, optional
        ``{"n_occ": int, "support": sequence}`` -- differentiate the energy of
        the *canonical* Hartree-Fock orbitals of each geometry instead of
        holding the orbitals fixed: the RDMs are replaced by the effective
        ones of the z-vector (:mod:`~mandacaru.algorithms.orbital_response`)
        before the derivative integrals are contracted, so the response of the
        orbitals costs one coupled-perturbed solve and no extra pass.
        ``n_occ`` is the number of doubly occupied orbitals of the reference,
        ``support`` the frozen and active orbitals that carry density.
        ``details["orbital_response"]`` reports the residuals.

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
                         _two_body_pulay, atom_rows, one_body_cross,
                         orbital_derivatives)

    if energy is None and integrals.mo_coefficients is None:
        raise RuntimeError("the integrals carry no molecular orbitals; build the "
                           "Hamiltonian with molecular_hamiltonian(mo_basis=True)")

    delta = DEFAULT_ORBITAL_DELTA if orbital_delta is None else float(orbital_delta)
    two_body = bool(getattr(energy, "two_body", True))
    if exchange is not None and not two_body:
        raise ValueError("a hybrid's exchange differentiates the two-body "
                         "tensor: the energy must keep it (two_body=True)")
    grid = integrals.grid
    dV = grid.dV
    basis = integrals.basis
    M = len(basis)
    ngrid = grid.size
    psi = np.ascontiguousarray(integrals._engine._psi)
    projectors = list(integrals.kb_projectors)
    P = len(projectors)
    exact_projections = bool(getattr(integrals, "exact_projections", False)) and P > 0
    chi = (np.stack([p.evaluate(grid.X, grid.Y, grid.Z).ravel()
                     for p in projectors])
           if P and not exact_projections else np.zeros((0, ngrid), dtype=complex))
    centers = [np.asarray(c, dtype=float) for _z, c in integrals._potentials.nuclei]
    charges = [float(z) for z, _c in integrals.nuclei]
    n_atoms = len(centers)
    datasets = getattr(integrals, "pseudopotentials", None) or []
    atom_potentials = _atom_potentials(integrals)

    # A spin-orbit Hamiltonian is written in GHF spinors: the energy keeps
    # the full spin-orbital RDMs and the spin-orbit matrix, whose derivative
    # comes from its own projections (`so_terms` below).
    spinor = energy is None and bool(getattr(integrals, "spinor_basis", False))
    if energy is not None:
        D1 = np.asarray(energy.D, dtype=complex)
    elif spinor:
        from ..core.spin_orbit import spin_orbit_pair

        so_projectors = list(integrals.spin_orbit_projectors)
        so_blocks = integrals.spin_orbit_coupling
        chi_so = np.stack([p.evaluate(grid.X, grid.Y, grid.Z).ravel()
                           for p in so_projectors])
        C_so = integrals.spin_orbit_projections()
        h_so_ao = spin_orbit_pair(C_so, C_so, so_projectors, so_blocks)
        h_so_ao = 0.5 * (h_so_ao + h_so_ao.conj().T)
        energy = SpinorAlgebraicEnergy(integrals.mo_coefficients, gamma,
                                       gamma2, h_so_ao)
        D1 = np.asarray(gamma, dtype=complex)

        def so_terms(dC_so):
            return (spin_orbit_pair(dC_so, C_so, so_projectors, so_blocks)
                    + spin_orbit_pair(C_so, dC_so, so_projectors, so_blocks))
    else:
        D1, G2 = spatial_rdms(gamma, gamma2, M)
        energy = AlgebraicEnergy(integrals.mo_coefficients, D1, G2)

    # -- the reference AO matrices, exactly as MolecularIntegrals._compute --
    external = integrals.external_potential()
    vext = np.ascontiguousarray(
        np.real(external(grid.X, grid.Y, grid.Z)).reshape(-1), dtype=np.float64)
    kinetic = integrals.kinetic
    T, V_loc = integrals.kinetic_and_local()
    # With the analytic two-center tables the overlap and the kinetic energy
    # differentiate through them, atom pair by atom pair, and only the
    # potential block of the Pulay term stays on the grid.
    analytic = integrals.two_center == "analytic"
    if analytic and include_pulay:
        tables = integrals.two_center_tables()
        overlap_slope = tables.gradient("overlap")              # (3, M, M)
        kinetic_slope = tables.gradient("kinetic")
    # The spectral Pulay term contracts the moved functions against the FFT
    # Laplacian of the unmoved ones: one transform per basis function for the
    # whole gradient (`one_body_cross`).
    laplacian = (integrals._engine.spectral_laplacian(psi)
                 if include_pulay and kinetic == "spectral" else None)
    C = integrals.projections() if P else np.zeros((M, 0), dtype=complex)
    D_nl = integrals.nonlocal_coupling_matrix() if P else None
    Q_nl = integrals.nonlocal_overlap_matrix()
    one = T + V_loc + (C @ D_nl @ C.conj().T if P else 0.0)
    # With the local potential range-separated, `V_loc` above is only its
    # long-range half (what the grid sampled); the short-range half is an
    # atom-centered quadrature and is differentiated analytically below.
    split_local = bool(getattr(integrals, "split_local_potential", False))
    if split_local:
        one = one + np.asarray(integrals.short_range_local())
    one_body_aug = integrals.one_body_augmentation()
    if one_body_aug is not None:
        # The compensation charges' electron-ion attraction is part of the
        # reported energy, so the gradient has to differentiate it too.
        one = one + np.asarray(one_body_aug)
    h0 = _hermitian(one)
    S0 = integrals.overlap()
    # The Hamiltonian build already computed it; the FFT build is the most
    # expensive step of the whole pipeline.
    if two_body:
        g0, augmentation = integrals._ao_two_body_terms()
        if spinor:
            # The spinor transform reads the full tensor.
            g0 = dense_tensor(g0)
    else:
        g0, augmentation = None, integrals.two_body_augmentation()
    electronic = energy(S0, h0, g0)
    total = electronic + integrals.nuclear_repulsion + integrals.constant_energy
    response = None
    if orbital_response is not None:
        from .orbital_response import response_densities

        if not isinstance(energy, AlgebraicEnergy) or not two_body \
                or spinor or exchange is not None:
            raise ValueError("the orbital response needs the plain algebraic "
                             "energy of spatial RDMs")
        response = response_densities(
            h0, dense_tensor(g0), energy.V, energy.D, energy.G,
            n_occ=int(orbital_response["n_occ"]),
            support=orbital_response["support"], S=S0)
        energy = AlgebraicEnergy(energy.V, response.D, response.G)
        orbital_gradient = False

    # -- PAW-LCAO compensation charges --
    paw = augmentation is not None and hasattr(integrals, "compensation_moments")
    if paw:
        from ..pseudopotentials.paw import paw_multipole_blocks

        # Everything is indexed by a compensation channel (atom, L, M): an
        # s-valence atom has only (A, 0, 0), a p-valence one also carries the
        # dipoles and quadrupoles its pair densities produce.
        Q_mom = integrals.compensation_moments()
        W_mom = integrals.compensation_potentials()
        U = integrals.compensation_coulomb()
        U_default = U
        V_ion = integrals.compensation_ionic()
        channels = integrals.multipole_channels()
        channel_index = {c: i for i, c in enumerate(channels)}
        comp_atoms = sorted({A for A, _L, _M in channels})
        channels_by_atom = {
            A: [c[1:] for c in channels if c[0] == A] for A in comp_atoms
        }
        blocks = integrals._atom_positions()
        moment_blocks = paw_multipole_blocks(projectors, integrals.datasets)
        v_comp = {}
        for A in comp_atoms:
            for (L, M_val), vals in _sampled_multipoles_atom(
                    datasets[A], centers[A], grid, channels_by_atom[A],
                    real=integrals.real_harmonics).items():
                v_comp[(A, L, M_val)] = vals
        ionic_gradients = (
            integrals.compensation_ionic_gradients(
                centers, COMPENSATION_DISTANCE_STEP)
            if hasattr(integrals, "compensation_ionic_gradients")
            else None
        )
    # -- PAW-LCAO exact spheres: the one-center Coulomb tensor --
    # sum_k lambda_k G^k (x) G^k with G^k = C_A U^k C_A^T moves only with the
    # projections; the one-body part (frozen functional out, core exchange
    # in) is already inside the nonlocal coupling D_nl.
    factors = getattr(integrals, "one_center_factors", None)
    spheres = factors() if factors is not None and P else None
    if spheres is not None and spheres.rank:
        G_sph, lam_sph = integrals.one_center_channels()
    else:
        spheres = None

    def sphere_terms(dC):
        r"""``dg`` of the spheres' Coulomb tensor for a change of the
        projections, as ``(c, X, Y, side)`` terms of ``sum c X (x) Y``,
        :math:`(X\otimes Y)_{pqrs} = X_{pr}Y_{qs}` (the layout of the
        augmented tensor), with the fixed factor given by its key and
        ``side`` naming which of the two moves."""
        terms = []
        dG = spheres.channel_derivatives(C, dC)
        for k, value in enumerate(lam_sph):
            if np.any(dG[k]):
                terms.append((value, dG[k], ("sphere", k), "right"))
                terms.append((value, ("sphere", k), dG[k], "left"))
        return terms

    # -- the Hartree terms of a linearized energy (`two_body` false) --
    # With the density matrix P0 held fixed, the Hartree energy's change under
    # a change dg of the two-body tensor is (1/2) P0 P0 dg.  For the low-rank
    # terms dg = sum X (x) Y, (X (x) Y)_pqrs = X_pr Y_qs, that is
    # (1/2) tr(P0 X) tr(P0 Y): scalars, no M^4 array.  The grid pairs'
    # part is the potential V_H of `field`, and the dependence on P (through
    # S) is the fixed one-body matrix.
    if not two_body:
        if spheres is not None:
            raise ValueError("the exact one-center Coulomb tensor is a "
                             "two-body term: keep the tensor (two_body=True)")
        P0 = energy.density(S0)
        if not np.iscomplexobj(psi):
            P0 = np.real(P0)                  # real orbitals, real density
        rho0 = np.real(np.sum((P0.T @ psi) * psi.conj(), axis=0))
        P_psi = P0.T @ psi
        P_psi_conj = P0 @ psi.conj()

        def trace(X):
            """``sum_ac P0_ca X_ac``."""
            return np.sum(P0.T * X)

        if paw:
            q_mom = np.array([trace(Q_mom[c]) for c in channels])
            w_mom = np.array([trace(W_mom[c]) for c in channels])
            coupling = 0.5 * (U + U.T) @ q_mom
            v_comp_tot = (sum(q_mom[a] * v_comp[ca]
                              for a, ca in enumerate(channels))
                          if channels else np.zeros(ngrid, dtype=float))

        def augmentation_contracted(dQ, dwd, dU):
            """``(1/2) P0 P0 dg_aug`` for the moments' change ``dQ``, the
            traces ``dwd[channel] = tr(P0 dW)`` and the coupling's change
            ``dU``."""
            value = 0.0
            for a, ca in enumerate(channels):
                value += trace(dQ[ca]) * (w_mom[a] + coupling[a])
                if ca in dwd:
                    value += q_mom[a] * dwd[ca]
            if dU is not None:
                value += 0.5 * (q_mom @ dU @ q_mom)
            return float(np.real(value))

        def pair_change(dpsi_k, moved):
            """``sum_ac P0_ca d(psi_a^* psi_c)`` on the grid, for the
            functions ``dpsi`` of one atom (the rows that moved)."""
            dpsi_moved = dpsi_k if dpsi_k.shape[0] != M else dpsi_k[moved]
            return (np.sum(dpsi_moved.conj() * P_psi[moved], axis=0)
                    + np.sum(dpsi_moved * P_psi_conj[moved], axis=0))

    if exact_projections:
        projection_gradients = integrals.projection_gradients(delta)
    orbital_atoms = np.asarray(atom_of_orbital)

    # `sr_gradients[A][p, q, k]` = the integral of phi_p^* phi_q v^sr_A with
    # phi_p's *own* center displaced along k.  The same array serves both
    # halves of the derivative, because the integral depends only on the
    # separations: moving a basis function contributes +G, moving the sphere
    # (with its potential) -G.  Their sum over a rigid translation is zero.
    sr_gradients = (integrals.short_range_local_matrices(gradients=True,
                                                         delta=delta,
                                                         values=False)[1]
                    if split_local else None)

    def short_range_pulay(atom, k):
        """``dV^sr`` when the basis functions of ``atom`` move along ``k``."""
        own = orbital_atoms == atom
        out = np.zeros((M, M), dtype=complex)
        for G in sr_gradients.values():
            Gk = G[:, :, k]
            out[own, :] += Gk[own, :]                       # p on this atom
            out[:, own] += Gk.conj().T[:, own]              # q on this atom
        return out

    def short_range_hellmann_feynman(atom, k):
        """``dV^sr`` when ``atom``'s sphere and potential move along ``k``."""
        Gk = sr_gradients[atom][:, :, k]
        return -(Gk + Gk.conj().T)

    def nonlocal_terms(dC):
        dh = dC @ D_nl @ C.conj().T + C @ D_nl @ dC.conj().T
        dS = (dC @ Q_nl @ C.conj().T + C @ Q_nl @ dC.conj().T
              if Q_nl is not None else None)
        return dh, dS

    def moment_derivatives(dC):
        """``{channel: dQ}`` from a change in the projections."""
        out = {}
        for channel, block in moment_blocks.items():
            idx = blocks[channel[0]]
            Ca, dCa = C[:, idx], dC[:, idx]
            out[channel] = (dCa @ block @ Ca.conj().T
                            + Ca @ block @ dCa.conj().T)
        return out

    def augmentation_terms(dQ, dW, dU, W=None, U=None):
        """The change of the compensation charges' tensor ``sum_ab
        (Q_a (x) W_a + W_a (x) Q_a) + U_ab Q_a (x) Q_b`` for changes ``dQ``,
        ``dW`` and ``dU`` of its pieces, as ``(c, X, Y)`` terms of ``sum c
        X (x) Y`` -- one or both of ``X``, ``Y`` arrays, the other the key
        of a fixed factor (``("W", a)``, ``("Q", a)``, ``("UQ", a) = sum_b
        U_ab Q_b``, ``("QU", b) = sum_a U_ab Q_a``) or, with ``W``/``U``
        given (the long-range part), arrays too."""
        bare = W is None
        W = W_mom if W is None else W
        U = U_default if U is None else U
        terms = []
        for a, ca in enumerate(channels):
            dQa = dQ[ca]
            Wa = ("W", a) if bare else W[ca]
            terms.append((1.0, dQa, Wa, "right"))
            terms.append((1.0, Wa, dQa, "left"))
            if ca in dW:
                terms.append((1.0, ("Q", a), dW[ca], "left"))
                terms.append((1.0, dW[ca], ("Q", a), "right"))
            if bare:
                terms.append((1.0, dQa, ("UQ", a), "right"))
                terms.append((1.0, ("QU", a), dQa, "left"))
            else:
                UQ = sum(U[a, b] * Q_mom[cb] for b, cb in enumerate(channels))
                QU = sum(U[b, a] * Q_mom[cb] for b, cb in enumerate(channels))
                terms.append((1.0, dQa, UQ, "right"))
                terms.append((1.0, QU, dQa, "left"))
            if dU is not None:
                for b, cb in enumerate(channels):
                    if dU[a, b] != 0.0:
                        terms.append((dU[a, b], ("Q", a), ("Q", b), "fixed"))
        return terms

    def ionic_derivative(dQ, dV):
        """Derivative of the compensation charges' electron-ion attraction.

        ``sum_{A,LM} Q^{A,LM} V^{A,LM}`` is a one-body operator, so both its
        pieces -- the moments moving with the projections and the shape sliding
        through the neighbors' ionic potentials -- land in ``dh``.
        """
        out = np.zeros((M, M), dtype=complex)
        for channel in channels:
            out = out + dQ[channel] * V_ion[channel]
            if dV is not None and channel in dV:
                out = out + Q_mom[channel] * dV[channel]
        return out

    # The two-body tensor enters the energy linearly, so the derivative
    # along a change dg is <dE/dg, dg>, with dE/dg the RDM taken back to the
    # atomic orbitals (`pair_derivative`, one M^4 array for the whole
    # gradient).  No dg is formed: the grid pairs' part comes from one
    # stream of pair potentials contracted into every atom and direction at
    # once (`_two_body_pulay`), and the low-rank parts (compensation charges,
    # spheres) through the derivative contracted once with each fixed
    # factor.  For real functions only the real part of dE/dg reaches a real
    # dg, so it is kept real.
    stream_derivatives = (include_pulay and exchange is None
                          and kinetic != "fd")
    derivatives = (orbital_derivatives(basis, grid, atom_of_orbital, None,
                                       delta) if (include_pulay and not stream_derivatives)
                   else None)
    pulay_pairs = None
    # The fixed factors of the low-rank terms (spheres, compensation
    # charges), by key.
    fixed = {}
    if spheres is not None:
        for k in range(len(lam_sph)):
            fixed[("sphere", k)] = np.asarray(G_sph[k])
    if paw and two_body:
        for a, ca in enumerate(channels):
            fixed[("Q", a)] = np.asarray(Q_mom[ca])
            fixed[("W", a)] = np.asarray(W_mom[ca])
            fixed[("UQ", a)] = sum(U_default[a, b] * np.asarray(Q_mom[cb])
                                   for b, cb in enumerate(channels))
            fixed[("QU", a)] = sum(U_default[b, a] * np.asarray(Q_mom[cb])
                                   for b, cb in enumerate(channels))
    fixed_first, fixed_second = {}, {}
    if two_body:
        pair = energy.pair_derivative(S0)                       # (M^2, M^2)
        if not np.iscomplexobj(psi) and np.iscomplexobj(pair):
            pair = np.ascontiguousarray(pair.real)
        if include_pulay:
            pulay_pairs = _two_body_pulay(psi, derivatives, grid, pair,
                                          atom_of_orbital, n_atoms,
                                          pair_layout=True,
                                          basis=basis, delta=delta)
        if fixed:
            keys = list(fixed)
            stack = np.stack([fixed[key] for key in keys]).reshape(len(keys),
                                                                   M * M)
            first = (pair @ stack.T).T.reshape(len(keys), M, M)
            second = (pair.T @ stack.T).T.reshape(len(keys), M, M)
            fixed_first = dict(zip(keys, first))
            fixed_second = dict(zip(keys, second))
        del pair

    W_S_T = None
    if not two_body and not spinor and hasattr(energy, "V_xc"):
        if getattr(energy, "E_occ", None) is not None:
            W_S_T = -np.ascontiguousarray(energy.E_occ.T)
        else:
            evals_s, U_s = np.linalg.eigh(_hermitian(S0))
            root_s = np.sqrt(np.maximum(evals_s, 1e-12))
            X0 = (U_s / root_s) @ U_s.conj().T
            K_loewner = -1.0 / (root_s[:, None] * root_s[None, :]
                                * (root_s[:, None] + root_s[None, :]))
            H_eff = h0 + energy.V_xc
            M0 = U_s.conj().T @ (energy.D @ X0 @ H_eff
                                 + H_eff @ X0 @ energy.D) @ U_s
            W_S_T = np.ascontiguousarray(
                (U_s @ (K_loewner * M0) @ U_s.conj().T).T)

    def overlap_directional(dS, dh, dh_so) -> float:
        """``<dE/dS, dS> + <dE/dh, dh>`` at the fixed tensor: the energy's
        directional derivative by a central difference, which is exact in
        ``h`` (linear) and of order ``step^2`` in ``S`` (through
        ``S^{-1/2}``) -- extrapolated from two steps (Richardson), so of
        order ``step^4``.  The two-body part, linear in ``g``, is contracted
        exactly beside it.  A central difference along ``(dS, dh, dg)``
        together carried a ``step^2`` error, partly cancelled between the
        ``S`` and ``g`` parts; split, the ``S`` part alone was 5x worse at
        the same step (water SZP: 7.9e-6 against 1.6e-6 eV/Angstrom from the
        zero-step limit), extrapolated it is 1e-7 (round-off)."""
        if W_S_T is not None and dh_so is None:
            val = float(np.real(np.sum(P0.T * dh)))
            if dS is not None:
                val += float(np.real(np.sum(W_S_T * dS)))
            return val

        def central(step):
            return energy.directional(S0, h0, g0, dS, dh, None, step,
                                      dh_so=dh_so)

        if dS is None:
            return central(algebraic_step)
        return (4.0 * central(0.5 * algebraic_step)
                - central(algebraic_step)) / 3.0

    def two_body_contracted(terms) -> float:
        """``<dE/dg, sum c X (x) Y>`` of :func:`augmentation_terms` /
        :func:`sphere_terms` terms: ``sum_ac X_ac (dE/dg . Y)_ac`` against
        the fixed factor's precomputed contraction."""
        value = 0.0
        for c, X, Y, side in terms:
            if side == "right":                  # X moves, Y fixed
                value += c * np.sum(X * fixed_first[Y])
            elif side == "left":                 # X fixed, Y moves
                value += c * np.sum(Y * fixed_second[X])
            else:                                # both fixed
                value += c * np.sum(fixed[X] * fixed_first[Y])
        return float(np.real(value))

    # -- a screened hybrid's exact exchange --
    if exchange is not None:
        from ..integrals.poisson import PoissonFFTSolver

        short_range = PoissonFFTSolver(grid.shape, step=grid.step,
                                       omega=float(exchange.omega))
        pulay_pairs_sr = None
        if include_pulay:
            # dE_x/dg_abcd = sum_w weight Pw_da Pw_cb, at (ac), (bd).
            pair_sr = sum(weight * np.einsum("da,cb->acbd", Pw, Pw)
                          for weight, Pw in exchange.densities)
            pair_sr = np.ascontiguousarray(pair_sr).reshape(M * M, M * M)
            if not np.iscomplexobj(psi) and np.iscomplexobj(pair_sr):
                pair_sr = np.ascontiguousarray(pair_sr.real)
            pulay_pairs_sr = _two_body_pulay(psi, derivatives, grid, pair_sr,
                                             atom_of_orbital, n_atoms,
                                             short_range, pair_layout=True,
                                             basis=basis, delta=delta)
            del pair_sr
        if paw:
            from ..pseudopotentials.paw import long_range_multipole_potential

            lr_potential = long_range_multipole_potential(exchange.omega)
            W_lr, U_lr = integrals.long_range_compensation(exchange.omega)
            v_comp_lr = {}
            for A in comp_atoms:
                for (L, M_val), vals in _sampled_multipoles_atom(
                        datasets[A], centers[A], grid, channels_by_atom[A],
                        lr_potential, real=integrals.real_harmonics).items():
                    v_comp_lr[(A, L, M_val)] = vals

    def resolved(X):
        """A term's factor as an array (a fixed factor's key resolved)."""
        return fixed[X] if isinstance(X, tuple) else X

    def exchange_term(terms_sr, dC):
        """``dE_x`` from a change of the short-range tensor, given as
        ``(c, X, Y)`` low-rank terms (its grid pairs' part is streamed
        separately), and of the projections (the one-center terms), at
        fixed AO density.  ``sum Pw_pq Pw_sr (X (x) Y)_qrsp = tr(Pw X Pw
        Y)``: no tensor."""
        value = 0.0
        for index, (weight, Pw) in enumerate(exchange.densities):
            for c, X, Y, _side in terms_sr:
                value += weight * c * np.trace(Pw @ resolved(X) @ Pw
                                               @ resolved(Y))
            if exchange.operators is not None and dC is not None and P:
                O = exchange.operators[index]
                value += np.trace(O @ (dC.conj().T @ Pw @ C
                                       + C.conj().T @ Pw @ dC))
        return float(np.real(value))

    def long_range_augmentation_derivative(atom, k, dQ, atom_geom=None):
        """The long-range augmentation's change when ``atom``'s operators
        (projections, compensation shapes) move along ``k``, as
        :func:`augmentation_terms` terms."""
        from ..pseudopotentials.multipoles import multipole_coulomb_matrix

        dW_lr = {}
        levels_a = channels_by_atom.get(atom, [])
        if levels_a:
            moved_lr = _moved_multipoles_atom(
                datasets[atom], centers[atom], grid, levels_a, k, delta,
                lr_potential, real=integrals.real_harmonics,
                atom_geom=atom_geom)
            for (L, M_val), w_comp in moved_lr.items():
                dW_lr[(atom, L, M_val)] = ((psi.conj() * w_comp) @ psi.T) * dV
        dU_lr = np.zeros_like(U_lr)
        step = COMPENSATION_DISTANCE_STEP
        for B in comp_atoms:
            if B == atom or not levels_a:
                continue
            levels_b = channels_by_atom[B]
            shift = np.zeros(3)
            shift[k] = step
            base = centers[B] - centers[atom]
            ra = datasets[atom].compensation_radius
            rb = datasets[B].compensation_radius
            slope = (multipole_coulomb_matrix(ra, levels_a, rb, levels_b,
                                              base - shift,
                                              potential=lr_potential,
                                              real=integrals.real_harmonics)
                     - multipole_coulomb_matrix(ra, levels_a, rb, levels_b,
                                                base + shift,
                                                potential=lr_potential,
                                                real=integrals.real_harmonics)
                     ) / (2.0 * step)
            for i, la in enumerate(levels_a):
                for j, lb in enumerate(levels_b):
                    u = channel_index[(atom,) + la]
                    v = channel_index[(B,) + lb]
                    dU_lr[u, v] = slope[i, j]
                    dU_lr[v, u] = slope[i, j]
        return augmentation_terms(dQ, dW_lr, dU_lr, W_lr, U_lr)

    cross_h_tables = None
    overlap_cross_tables = None
    field_pulay_tables = None
    if include_pulay and not stream_derivatives:
        if kinetic != "fd":
            cross_h_tables = [
                one_body_cross(psi, derivatives[k], vext, grid, kinetic,
                               laplacian)
                for k in range(3)
            ]
        if not analytic:
            overlap_cross_tables = [
                (derivatives[k].conj() @ psi.T) * dV for k in range(3)
            ]
        if field is not None and hasattr(field, "pulay_tables"):
            field_pulay_tables = field.pulay_tables(derivatives)

    need_dpsi_stack = (
        stream_derivatives
        or (cross_h_tables is None)
        or (P and not exact_projections)
        or (paw and two_body)
        or spinor
        or (field is not None and field_pulay_tables is None)
    )
    X_flat = grid.X.ravel()
    Y_flat = grid.Y.ravel()
    Z_flat = grid.Z.ravel()

    hf = np.zeros((n_atoms, 3))
    pulay = np.zeros((n_atoms, 3))
    moved_cache: dict = {}
    for atom in range(n_atoms):
        moved_cache.clear()
        own_projectors = [p for p, proj in enumerate(projectors)
                          if proj.atom_index == atom]
        own_mask = orbital_atoms == atom
        has_own = bool(np.any(own_mask))
        if stream_derivatives and include_pulay and has_own:
            d_atom = orbital_derivatives(basis, grid, atom_of_orbital, atom, delta)
            orbital_stack = atom_rows(d_atom, atom_of_orbital, atom)
            del d_atom
        else:
            orbital_stack = (atom_rows(derivatives, atom_of_orbital, atom)
                             if include_pulay and has_own and need_dpsi_stack
                             else None)
        dx_a = X_flat - centers[atom][0]
        dy_a = Y_flat - centers[atom][1]
        dz_a = Z_flat - centers[atom][2]
        atom_geom = (dx_a, dy_a, dz_a, dx_a * dx_a + dy_a * dy_a + dz_a * dz_a)

        for k in range(3):
            # ---------------- Pulay: the basis functions of `atom` move ----
            if include_pulay and has_own and (stream_derivatives or np.any(derivatives[k, own_mask])):
                dpsi = orbital_stack[k] if orbital_stack is not None else None
                if cross_h_tables is not None:
                    cross_h = np.zeros((M, M), dtype=complex)
                    cross_h[own_mask] = cross_h_tables[k][own_mask]
                else:
                    cross_h = one_body_cross(psi, dpsi, vext, grid, kinetic,
                                             laplacian)
                dh = cross_h + cross_h.conj().T
                if analytic:
                    dS = _displacement_derivative(overlap_slope[k], own_mask)
                    dh = dh + _displacement_derivative(kinetic_slope[k],
                                                       own_mask)
                else:
                    if overlap_cross_tables is not None:
                        cross = np.zeros((M, M), dtype=complex)
                        cross[own_mask] = overlap_cross_tables[k][own_mask]
                    else:
                        cross = (dpsi.conj() @ psi.T) * dV
                    dS = cross + cross.conj().T
                if split_local:
                    dh = dh + short_range_pulay(atom, k)
                if P:
                    if exact_projections:
                        dC = np.zeros((M, P), dtype=complex)
                        dC[own_mask] = projection_gradients[own_mask, :, k]
                    else:
                        dC = _backend.kb_projections(dpsi, chi, dV)
                    dh_nl, dS_aug = nonlocal_terms(dC)
                    dh = dh + dh_nl
                    if dS_aug is not None:
                        dS = dS + dS_aug
                # The two-body change, as the streamed grid pairs' part
                # and low-rank terms (`dg` is never formed).
                terms, terms_sr, extra = [], [], 0.0
                if two_body:
                    extra += float(pulay_pairs[atom, k])
                    if spheres is not None:
                        terms += sphere_terms(dC)
                if exchange is not None:
                    extra_sr = float(pulay_pairs_sr[atom, k])
                if paw and not two_body:
                    # tr(P0 dW) is the compensation potential against the
                    # change of the density the moved functions make.
                    drho = (pair_change(dpsi, own_mask) if stream_derivatives
                            else pair_change(derivatives[k], own_mask))
                    dQ = moment_derivatives(dC)
                    extra += (augmentation_contracted(dQ, {}, None)
                              + dV * float(np.real(v_comp_tot @ drho)))
                    dh = dh + ionic_derivative(dQ, None)
                elif paw:
                    dW = {}
                    for channel in channels:
                        # W_qs = int conj(phi_q) v_LM phi_s is Hermitian only
                        # where the weight is real, i.e. M = 0.  The complex
                        # Y_LM channels are not, so both halves of the product
                        # rule are written out -- `cross + cross^H` would put
                        # conj(v) in the second one and silently symmetrize an
                        # operator that is not symmetric.
                        v = v_comp[channel]
                        dW[channel] = (((dpsi.conj() * v) @ psi.T)
                                       + ((psi.conj() * v) @ dpsi.T)) * dV
                    dQ = moment_derivatives(dC)
                    aug_terms = augmentation_terms(dQ, dW, None)
                    terms += aug_terms
                    if exchange is not None:
                        dW_lr = {}
                        for channel in channels:
                            v = v_comp_lr[channel]
                            dW_lr[channel] = (((dpsi.conj() * v) @ psi.T)
                                              + ((psi.conj() * v) @ dpsi.T)
                                              ) * dV
                        terms_sr += aug_terms + [
                            (-c, X, Y, side) for c, X, Y, side in
                            augmentation_terms(dQ, dW_lr, None, W_lr, U_lr)]
                    # The compensation shapes do not move with the basis, so
                    # only the moments change here.
                    dh = dh + ionic_derivative(dQ, None)
                dh_so = (so_terms(_backend.kb_projections(dpsi, chi_so, dV))
                         if spinor else None)
                if field_pulay_tables is not None:
                    dh = dh + field.pulay_from_table(
                        field_pulay_tables[k], own_mask)
                elif field is not None:
                    cross_f = ((dpsi.conj() * field.v) @ psi.T) * dV
                    if getattr(field, "w", None) is not None or getattr(field, "v_tau", None) is not None:
                        from ..integrals import exchange_correlation as xc_grid
                        grad_dpsi = xc_grid.gradient(grid, dpsi)
                        if field.w is not None:
                            for component, dphi, dmoved in zip(field.w, field.grad_psi, grad_dpsi):
                                cross_f = cross_f + ((dpsi.conj() * component) @ dphi.T
                                                     + (dmoved.conj() * component) @ psi.T) * dV
                        if field.v_tau is not None:
                            w_tau = 0.5 * field.v_tau
                            for d, dd in zip(field.grad_psi, grad_dpsi):
                                cross_f = cross_f + ((dd.conj() * w_tau) @ d.T) * dV
                    dh = dh + (cross_f + cross_f.conj().T)
                if terms:
                    extra += two_body_contracted(terms)
                pulay[atom, k] = extra + overlap_directional(
                    _hermitian(dS), _hermitian(dh), dh_so)
                if field is not None and hasattr(field, "spin_term"):
                    # A spin-resolved potential: the magnetization part the
                    # spin-averaged energy does not see. When E_occ is supplied,
                    # the overlap derivative dS is already fully accounted for in
                    # W_S_T = -E_occ.T for both spin channels.
                    dS_spin = None if getattr(energy, "E_occ", None) is not None else _hermitian(dS)
                    if field_pulay_tables is not None:
                        pulay[atom, k] += field.spin_term(
                            S0, dS_spin, None, algebraic_step,
                            k=k, own=own_mask)
                    elif hasattr(field, "v_half"):
                        if getattr(field, "_pm_cache", None) is None or field._pm_cache[0] is not S0:
                            X_mag = AlgebraicEnergy.lowdin(S0)
                            field._pm_cache = (S0, X_mag @ field.Dm @ X_mag.conj().T)
                        val_spin = 0.0
                        if dS_spin is not None:
                            def mag(S):
                                Xm = AlgebraicEnergy.lowdin(S)
                                return Xm @ field.Dm @ Xm.conj().T
                            dPm = (mag(S0 + algebraic_step * dS_spin)
                                   - mag(S0 - algebraic_step * dS_spin)) / (2.0 * algebraic_step)
                            val_spin += float(np.real(np.sum(dPm * field.delta_matrix.T)))
                        cross_half = ((dpsi.conj() * field.v_half) @ psi.T) * dV
                        if getattr(field, "w_half", None) is not None or getattr(field, "v_tau_half", None) is not None:
                            from ..integrals import exchange_correlation as xc_grid
                            grad_dpsi = xc_grid.gradient(grid, dpsi)
                            if field.w_half is not None:
                                for comp, dphi, dmoved in zip(field.w_half, field.grad_psi, grad_dpsi):
                                    cross_half = cross_half + ((dpsi.conj() * comp) @ dphi.T
                                                               + (dmoved.conj() * comp) @ psi.T) * dV
                            if field.v_tau_half is not None:
                                w_tau = 0.5 * field.v_tau_half
                                for d, dd in zip(field.grad_psi, grad_dpsi):
                                    cross_half = cross_half + ((dd.conj() * w_tau) @ d.T) * dV
                        dDelta = cross_half + cross_half.conj().T
                        val_spin += float(np.real(np.sum(field._pm_cache[1] * dDelta.T)))
                        pulay[atom, k] += val_spin
                    else:
                        pulay[atom, k] += field.spin_term(
                            S0, dS_spin, dpsi, algebraic_step)
                if exchange is not None:
                    pulay[atom, k] += extra_sr + exchange_term(
                        terms_sr, dC if P else None)

            # ---------------- Hellmann-Feynman: the operators of `atom` move --
            w_loc = _moved_radial(atom_potentials[atom], centers[atom],
                                  grid, k, delta, atom_geom=atom_geom)
            if not two_body and not spinor:
                dh = np.zeros((M, M), dtype=complex)
                extra = dV * float(w_loc @ rho0)
            else:
                dh = ((psi.conj() * w_loc) @ psi.T) * dV
                extra = 0.0
            if split_local:
                dh = dh + short_range_hellmann_feynman(atom, k)
            dS = None
            terms, terms_sr = [], None
            if P:
                if exact_projections:
                    dC = np.zeros((M, P), dtype=complex)
                    dC[:, own_projectors] = -projection_gradients[:, own_projectors, k]
                else:
                    dchi = np.zeros_like(chi)
                    for p in own_projectors:
                        dchi[p] = _moved_function(projectors[p], grid, k, delta,
                                                  moved_cache)
                    dC = _backend.kb_projections(psi, dchi, dV)
                dh_nl, dS = nonlocal_terms(dC)
                dh = dh + dh_nl
                if spheres is not None:
                    terms += sphere_terms(dC)
                if paw:
                    from ..pseudopotentials.multipoles import (
                        multipole_coulomb_matrix)

                    dW = {}
                    levels_a = channels_by_atom.get(atom, [])
                    if levels_a:
                        moved_atom = _moved_multipoles_atom(
                            datasets[atom], centers[atom], grid, levels_a, k,
                            delta, real=integrals.real_harmonics,
                            atom_geom=atom_geom)
                        for (L, M_val), w_comp in moved_atom.items():
                            channel = (atom, L, M_val)
                            if two_body:
                                dW[channel] = ((psi.conj() * w_comp) @ psi.T) * dV
                            else:
                                dW[channel] = dV * (w_comp @ rho0)

                    # Multipole-multipole Coulomb depends on the *direction*
                    # between the centers, not only the distance as the
                    # monopole did, so the derivative is taken with respect to
                    # the displacement vector itself.
                    dU = np.zeros_like(U)
                    step = COMPENSATION_DISTANCE_STEP
                    for B in comp_atoms:
                        if B == atom or not levels_a:
                            continue
                        levels_b = channels_by_atom[B]
                        ra = datasets[atom].compensation_radius
                        rb = datasets[B].compensation_radius
                        shift = np.zeros(3)
                        shift[k] = step
                        base = centers[B] - centers[atom]
                        plus = multipole_coulomb_matrix(
                            ra, levels_a, rb, levels_b, base - shift,
                            real=integrals.real_harmonics)
                        minus = multipole_coulomb_matrix(
                            ra, levels_a, rb, levels_b, base + shift,
                            real=integrals.real_harmonics)
                        slope = (plus - minus) / (2.0 * step)
                        for i, la in enumerate(levels_a):
                            for j, lb in enumerate(levels_b):
                                u = channel_index[(atom,) + la]
                                v = channel_index[(B,) + lb]
                                dU[u, v] = slope[i, j]
                                dU[v, u] = slope[i, j]

                    # The electron-ion attraction of the compensation charges
                    # changes both because this atom's shape moves and because
                    # its ionic potential moves under everyone else's shape.
                    dV_ion = (ionic_gradients[(atom, k)]
                              if ionic_gradients is not None
                              else _ionic_shift(integrals, centers, atom, k, step))
                    dQ = moment_derivatives(dC)
                    if two_body:
                        terms += augmentation_terms(dQ, dW, dU)
                    else:
                        extra += augmentation_contracted(dQ, dW, dU)
                    dh = dh + ionic_derivative(dQ, dV_ion)
                    if exchange is not None:
                        terms_sr = terms + [
                            (-c, X, Y, side) for c, X, Y, side in
                            long_range_augmentation_derivative(
                                atom, k, dQ, atom_geom=atom_geom)]
            dh_so = None
            if spinor:
                dchi_so = np.zeros_like(chi_so)
                for p, projector in enumerate(so_projectors):
                    if projector.atom_index == atom:
                        dchi_so[p] = _moved_function(projector, grid, k, delta,
                                                     moved_cache)
                dh_so = so_terms(_backend.kb_projections(psi, dchi_so, dV))
            if terms:
                extra += two_body_contracted(terms)
            hf[atom, k] = extra + overlap_directional(
                _hermitian(dS) if dS is not None else None, _hermitian(dh),
                dh_so)
            if field is not None and hasattr(field, "spin_term") \
                    and dS is not None and getattr(energy, "E_occ", None) is None:
                hf[atom, k] += field.spin_term(S0, _hermitian(dS), None,
                                               algebraic_step)
            if field is not None:
                hf[atom, k] += field.hellmann_feynman(atom, k)
            if exchange is not None:
                hf[atom, k] += exchange_term(terms_sr or [],
                                             dC if P else None)

    hf = hf + _nuclear_repulsion_gradient(centers, charges)

    details = {"method": "pseudopotential", "energy_hartree": float(total),
               "electronic_energy_hartree": float(electronic),
               "exact_projections": exact_projections,
               "split_local_potential": split_local,
               "include_pulay": bool(include_pulay),
               "n_orbitals": M}
    if response is not None:
        details["orbital_gradient"] = response.orbital_gradient
        details["orbital_response"] = {
            "cphf_iterations": response.cphf_iterations,
            "cphf_residual": response.cphf_residual,
            "scf_residual": response.scf_residual,
            "near_degenerate": response.near_degenerate}
    if orbital_gradient:
        details["orbital_gradient"] = energy.orbital_gradient(S0, h0, g0)

    hf = hf * HA_BOHR_TO_EV_ANGSTROM
    pulay = pulay * HA_BOHR_TO_EV_ANGSTROM
    gradient = hf + pulay
    return ForceResult(forces=-gradient, hellmann_feynman=hf, pulay=pulay,
                       gradient=gradient,
                       n_electrons=float(np.real(np.trace(D1))), details=details)
