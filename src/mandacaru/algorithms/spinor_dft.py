# -*- coding: utf-8 -*-
# file: algorithms/spinor_dft.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Two-component (spinor) Kohn-Sham for molecules and crystals, spin-orbit
included.

A PAW-LCAO dataset of a spin-orbit set (``lda-fr/``, ``pbe-fr/``) carries a
spin-orbit term (:mod:`mandacaru.core.spin_orbit`) that couples the two spin
blocks of the one-body Hamiltonian.  Its Kohn-Sham orbitals are therefore
spinors, :math:`\psi_i = (\psi_{i\uparrow}, \psi_{i\downarrow})`, and the
Kohn-Sham matrix is :math:`2M \times 2M` over the Loewdin basis of the shared
builder, alpha block first (the layout of
:func:`~mandacaru.core.hamiltonian.spin_block_integrals`):

.. math::

    F = \begin{pmatrix} h & 0\\ 0 & h\end{pmatrix} + h^{SO}
        + \begin{pmatrix} J & 0\\ 0 & J\end{pmatrix}
        + \int \phi_p^*\big(v_{xc}\,\mathbb 1
          + \mathbf B_{xc}\cdot\boldsymbol\sigma\big)\phi_q ,

with :math:`h^{SO}` exactly the term ``method="ghf"`` adds
(:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.spin_orbit_matrix`),
:math:`J` the Hartree matrix of the total density (the same Poisson solve, or
tensor contraction, as the collinear solvers) and the exchange-correlation
potential and field of the noncollinear functional
(:func:`~mandacaru.integrals.noncollinear.evaluate_noncollinear`, local spin
axis; LDA and PBE).  The spin density matrix of the spinor density matrix
:math:`D_{p\sigma,q\tau} = \sum_i f_i C_{p\sigma,i}C^*_{q\tau,i}` is
:math:`\rho_{\sigma\tau} = \sum_{pq}\phi_p D_{p\sigma,q\tau}\phi_q^*`, so
:math:`n = \rho_{\uparrow\uparrow} + \rho_{\downarrow\downarrow}`,
:math:`m_x = 2\,\mathrm{Re}\,\rho_{\uparrow\downarrow}`,
:math:`m_y = -2\,\mathrm{Im}\,\rho_{\uparrow\downarrow}` and
:math:`m_z = \rho_{\uparrow\uparrow} - \rho_{\downarrow\downarrow}`.

Occupations follow the collinear solvers: aufbau over the :math:`2M` spinor
levels, except a degenerate shell at the Fermi level, whose electrons are
shared equally (:func:`~.dft.shell_occupations`) -- a Kramers pair holding one
electron, or the :math:`p_{3/2}` quartet of an open :math:`p` shell.  The SCF
starts from the collinear Kohn-Sham solution of the same integrals without
the spin-orbit term (:func:`collinear_spinor_density`): a spin-polarized open
shell then stays polarized, which a start from the Kramers-degenerate core
guess, filled equally, never would.  That collinear energy is kept
(``scalar_energy``), so the result also reports what the spin-orbit term
changed.

Without a spin-orbit term the collinear solution is a fixed point of the
spinor SCF, and a stable one, so a noncollinear ground state is reached only
from a noncollinear start: per-atom moment vectors (``(N, 3)`` initial
magnetic moments) turn each atom's share of the collinear magnetization to
its own direction (:func:`noncollinear_spinor_density`), and the same SCF
then runs with or without the spin-orbit term.

A crystal (:class:`SpinorPeriodicKohnSham`) is the same problem in Bloch
spinors: the spin-orbit term of the molecule summed over the lattice per
k-point, the noncollinear functional on the cell grid, one electron per
spinor band around one Fermi level, and the k-point mesh reduced by the
space group and time reversal only for a crystal solved as non-magnetic, its
magnetization set to zero by time-reversal symmetry (a magnetic spinor crystal
is solved on the whole mesh).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..integrals.exchange_correlation import gradient_term_matrix
from ..integrals.noncollinear import (NONCOLLINEAR_FUNCTIONALS,
                                      evaluate_noncollinear,
                                      supports_noncollinear)
from .dft import (DEFAULT_XC, DIIS_RESTART, LEVEL_SHIFT_TRIGGER, KohnSham,
                  KohnShamUResult, MOTensor, _DeferredTensor,
                  _IndependentDIIS, _stalled, shell_occupations)
from .hartree_fock import (GHF, LEVEL_SHIFT, GHFResult, _level_shifted,
                           kramers_pairing, spinor_integrals)
from ..integrals import reciprocal as rc
from ..integrals._backend import single_threaded_blas
from .periodic_device import make_device
from . import periodic_exchange as exact_exchange
from .periodic_dft import (HYBRID_TOL, PeriodicKohnSham,
                           PeriodicKohnShamResult, PulayMixer, entropy,
                           fermi_level, occupation)
from ..utils.logging import scf_record


class SpinorTensor(MOTensor):
    """The spin-orbital two-body tensor in a set of spinors, transformed
    when first read (:func:`~.hartree_fock.spinor_integrals`), like
    :class:`~.dft.MOTensor` for spatial orbitals.  ``orbitals`` are
    ``(2M, K)`` spinor columns over the Loewdin basis, alpha rows first."""

    def build(self) -> np.ndarray:
        C = self.orbitals
        one = np.zeros((C.shape[0], C.shape[0]), dtype=complex)
        return spinor_integrals(one, self.integrals.two_body(), C)[1]


@dataclass
class KohnShamGResult(_DeferredTensor, GHFResult):
    r"""A converged two-component (spinor) Kohn-Sham determinant.

    The fields of :class:`~mandacaru.algorithms.hartree_fock.GHFResult` --
    ``(2M, 2M)`` spinor coefficients, the spinor Kohn-Sham eigenvalues,
    ``h_mo``/``eri_mo`` in the spinors in the exported **mode order**
    (:meth:`~.hartree_fock.GHF.mode_order`), the Kramers pairing of the
    occupied levels -- plus the Kohn-Sham energy decomposition (Hartree).
    ``eri_mo`` is transformed the first time it is read (:class:`SpinorTensor`)
    on the Poisson route.

    ``occupations`` is ``None`` for the aufbau determinant, else the
    occupations (0 to 1) of the energy-ordered spinors when a degenerate shell
    at the Fermi level is shared.  ``scalar_energy`` is the electronic energy
    of the collinear Kohn-Sham solution of the same integrals without the
    spin-orbit term, the SCF's starting point (``None`` when it was started
    otherwise).  ``spin_moment`` is the spin moment vector
    :math:`2\langle\mathbf S\rangle = (M_x, M_y, M_z)` in Bohr magnetons,
    from the traces of the spin blocks: the spin part only, not the total
    :math:`\langle\mathbf L + 2\mathbf S\rangle` (in a :math:`p_{1/2}`
    spinor the two have opposite signs).  ``spin_orbit`` says whether the
    spin-orbit term was in the operator (a spinor run without it is a
    noncollinear start on a scalar set).
    """

    functional: str = DEFAULT_XC
    hartree_energy: float = 0.0
    xc_energy: float = 0.0
    #: A hybrid's exact-exchange share of ``xc_energy`` (zero otherwise).
    exact_exchange_energy: float = 0.0
    core_correction_energy: float = 0.0
    dispersion_energy: float = 0.0
    #: One record per iteration (:func:`~mandacaru.utils.logging.scf_record`).
    history: list = field(default_factory=list)
    occupations: np.ndarray | None = None
    #: :attr:`determinant_energy`, computed by the SCF.
    reference_energy: float = 0.0
    scalar_energy: float | None = None
    spin_moment: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    spin_orbit: bool = True

    @property
    def determinant_energy(self) -> float:
        r"""Energy of the determinant of the lowest ``n_electrons`` spinors in
        the many-body Hamiltonian (Hartree, no constant): the Hartree-Fock
        functional of the Kohn-Sham spinors, the exported problem's reference.
        """
        return float(self.reference_energy)

    @property
    def spin_orbit_energy(self) -> float | None:
        """``electronic_energy - scalar_energy``: what the spin-orbit term
        (and the freedom of the spinors) lowered the collinear Kohn-Sham
        energy by, in the same basis and grid (Hartree); ``None`` without the
        collinear run or without a spin-orbit term (:attr:`noncollinear_energy`
        is then the change)."""
        if self.scalar_energy is None or not self.spin_orbit:
            return None
        return float(self.electronic_energy - self.scalar_energy)

    @property
    def noncollinear_energy(self) -> float | None:
        """``electronic_energy - scalar_energy`` of a spinor run without a
        spin-orbit term: what the noncollinear magnetization lowered the
        collinear Kohn-Sham energy by (Hartree); ``None`` with a spin-orbit
        term (:attr:`spin_orbit_energy`) or without the collinear run."""
        if self.scalar_energy is None or self.spin_orbit:
            return None
        return float(self.electronic_energy - self.scalar_energy)

    def __repr__(self) -> str:
        return (f"KohnShamGResult({self.functional}, "
                f"E_elec={self.electronic_energy:.6f}, "
                f"n_electrons={self.n_electrons}, "
                f"converged={self.converged})")


def collinear_spinor_density(result) -> np.ndarray:
    r"""The ``(2M, 2M)`` spinor density matrix of a collinear Kohn-Sham
    result: :math:`D_\alpha` and :math:`D_\beta` on the diagonal blocks,
    fractional shell occupations included -- the restricted result's
    ``occupations`` hold both spins (0 to 2)."""
    if isinstance(result, KohnShamUResult):
        channels = [(result.mo_coefficients_alpha, result.n_alpha,
                     result.occupations_alpha),
                    (result.mo_coefficients_beta, result.n_beta,
                     result.occupations_beta)]
    else:
        half = (None if result.occupations is None
                else 0.5 * np.asarray(result.occupations))
        channels = [(result.mo_coefficients, result.n_occupied, half)] * 2
    blocks = []
    for C, n, f in channels:
        C = np.asarray(C)
        blocks.append((C * f) @ C.conj().T if f is not None
                      else C[:, :n] @ C[:, :n].conj().T)
    M = blocks[0].shape[0]
    D = np.zeros((2 * M, 2 * M), dtype=complex)
    D[:M, :M], D[M:, M:] = blocks
    return D


def noncollinear_spinor_density(result, directions) -> np.ndarray:
    r"""The collinear result's spinor density matrix with each basis
    function's share of the magnetization turned to its own direction.

    ``directions`` is ``(M, 3)``: the unit spin direction of each Loewdin
    basis function (its atom's initial moment vector), or zero to drop its
    share.  With :math:`P = D_\alpha + D_\beta`, :math:`P_m = D_\alpha -
    D_\beta` and :math:`W_k = \mathrm{diag}(d_{pk})`, the magnetization
    matrices are :math:`m_k = \tfrac12(W_k P_m + P_m W_k)` -- an atom's share
    is the Loewdin-population split of :math:`P_m` -- and

    .. math::

        D = \tfrac12\begin{pmatrix} P + m_z & m_x - i m_y\\
                                 m_x + i m_y & P - m_z\end{pmatrix}.

    The charge density is the collinear one; every direction along
    :math:`+z` gives back :func:`collinear_spinor_density`, and a common
    direction a globally rotated copy of it.  It is a starting point, not a
    projector: the SCF's first diagonalization makes it one.
    """
    D = collinear_spinor_density(result)
    M = D.shape[0] // 2
    d = np.asarray(directions, dtype=float)
    if d.shape != (M, 3):
        raise ValueError(f"directions must be ({M}, 3), one per basis "
                         f"function, got {d.shape}")
    P = D[:M, :M] + D[M:, M:]
    Pm = D[:M, :M] - D[M:, M:]
    mx, my, mz = (0.5 * (d[:, k, None] * Pm + Pm * d[None, :, k])
                  for k in range(3))
    out = np.empty_like(D)
    out[:M, :M] = 0.5 * (P + mz)
    out[M:, M:] = 0.5 * (P - mz)
    out[:M, M:] = 0.5 * (mx - 1j * my)
    out[M:, :M] = out[:M, M:].conj().T
    return out


class SpinorKohnSham(KohnSham):
    r"""Two-component Kohn-Sham SCF of a molecule in an orthonormal basis.

    Parameters
    ----------
    h, eri, orbitals, grid
        As for :class:`~.dft.KohnSham`: the spin-free core Hamiltonian
        ``(M, M)``, the two-body tensor or the
        :class:`~.dft.PoissonHartree`, the orthonormal functions on the grid.
    n_electrons : int
        Total electron count (any parity).
    spin_orbit : (2M, 2M) array, optional
        The spin-orbit term in the same basis, alpha block first
        (:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.spin_orbit_matrix`);
        ``None`` solves the spinor problem without it, whose solutions include
        the collinear ones.
    functional, core_density, relativistic, core_tau, short_range_eri, \
    screening, one_center
        As for :class:`~.dft.KohnSham`, restricted to
        :data:`~mandacaru.integrals.noncollinear.NONCOLLINEAR_FUNCTIONALS`.
        r2SCAN takes the kinetic-energy density matrix of the spinors along
        the local axis; HSE06 the short-range exchange of every spin block
        of the spinor density matrix (generalized Hartree-Fock's), and its
        one-center terms (:meth:`~mandacaru.pseudopotentials.onecenter.
        OneCenterHybrid.evaluate_spinor`).
    """

    def __init__(self, h, eri, orbitals, grid, n_electrons: int,
                 spin_orbit=None, functional: str = DEFAULT_XC,
                 core_density=None, relativistic: bool = False,
                 core_tau=None, short_range_eri=None, screening=None,
                 one_center=None):
        if not supports_noncollinear(functional):
            raise NotImplementedError(spinor_functional_refusal(functional))
        super().__init__(h, eri, orbitals, grid, n_electrons,
                         functional=functional, core_density=core_density,
                         relativistic=relativistic, core_tau=core_tau,
                         short_range_eri=short_range_eri,
                         screening=screening, one_center=one_center)
        M = self.M
        h_spin = np.zeros((2 * M, 2 * M), dtype=complex)
        h_spin[:M, :M] = h_spin[M:, M:] = self.h
        if spin_orbit is not None:
            spin_orbit = np.asarray(spin_orbit, dtype=complex)
            if spin_orbit.shape != h_spin.shape:
                raise ValueError(
                    f"spin_orbit must be ({2 * M}, {2 * M}) for {M} spatial "
                    f"orbitals, got {spin_orbit.shape}")
            h_spin = h_spin + spin_orbit
        self.h_spin = 0.5 * (h_spin + h_spin.conj().T)
        #: Whether the operator carries the spin-orbit term.
        self.spin_orbit = spin_orbit is not None

    @classmethod
    def from_solver(cls, solver: KohnSham, spin_orbit=None):
        """The spinor problem of a collinear :class:`~.dft.KohnSham` (or
        unrestricted) solver: its core Hamiltonian, Hartree builder, grid
        samples, partial core (and its tau), functional and a hybrid's
        short-range tensor and spheres, plus ``spin_orbit``."""
        return cls(solver.h, (solver.poisson if solver.poisson is not None
                              else solver.eri),
                   solver.orbitals, solver.grid, solver.n_electrons,
                   spin_orbit=spin_orbit, functional=solver.functional,
                   core_density=solver.core_density,
                   relativistic=solver.relativistic,
                   core_tau=solver._core_tau,
                   short_range_eri=solver.short_range_eri,
                   screening=solver.screening, one_center=solver.one_center)

    def _set_electrons(self, n_electrons) -> None:
        self.n_electrons = int(n_electrons)
        if not 0 < self.n_electrons <= 2 * self.M:
            raise ValueError(f"{n_electrons} electrons do not fit "
                             f"{2 * self.M} spinors")

    # -- the density and its matrices ------------------------------------- #

    def _density_matrix(self, C, occupations=None) -> np.ndarray:
        """``D = sum_i f_i C_i C_i^dagger`` over the spinor columns: the
        lowest ``n_electrons`` filled, or the fractional ``occupations``."""
        if occupations is not None:
            return (C * occupations) @ C.conj().T
        occupied = C[:, :self.n_electrons]
        return occupied @ occupied.conj().T

    def _blocks(self, D):
        M = self.M
        return D[:M, :M], D[M:, M:], D[:M, M:]

    def _field(self, X) -> np.ndarray:
        r""":math:`\sum_{pq}\phi_p X_{pq}\phi_q^*` on the grid (complex)."""
        phi = self.orbitals
        return np.sum(phi * (X @ phi.conj()), axis=0)

    def spin_densities(self, D):
        r"""``(n, m)``: the valence density and the ``(3, ngrid)``
        magnetization of the spinor density matrix ``D``."""
        up, down, mixed = self._blocks(D)
        uu, dd, ud = self._field(up), self._field(down), self._field(mixed)
        n = np.real(uu + dd)
        m = np.stack([2.0 * np.real(ud), -2.0 * np.imag(ud), np.real(uu - dd)])
        return n, m

    def spin_moment(self, D) -> np.ndarray:
        r"""Spin moment :math:`2\langle\mathbf S\rangle = (M_x, M_y, M_z)`
        (Bohr magnetons) from the traces of the spin blocks -- the basis is
        orthonormal, so the trace is the integral, augmentation included."""
        up, down, mixed = self._blocks(D)
        t = np.trace(mixed)
        return np.array([2.0 * np.real(t), -2.0 * np.imag(t),
                         np.real(np.trace(up) - np.trace(down))])

    def _gradient_field(self, X) -> np.ndarray:
        r""":math:`\nabla\sum_{pq}\phi_p X_{pq}\phi_q^*` on the grid,
        ``(3, ngrid)`` complex, by the product rule on the orbitals' own
        gradients."""
        phi = self.orbitals
        return np.stack([np.sum(dphi * (X @ phi.conj()), axis=0)
                         + np.sum(phi * (X @ dphi.conj()), axis=0)
                         for dphi in self._orbital_gradients])

    def spin_gradients(self, D):
        r"""``(grad_n, grad_m)``: the gradients of the valence density and
        of the magnetization (:meth:`spin_densities`), ``(3, ngrid)`` and
        ``(3, 3, ngrid)`` with the derivative index first, by the product
        rule on the orbitals, so that a meta-GGA's :math:`\tau_W` is the
        one its :math:`\tau` is measured against."""
        up, down, mixed = self._blocks(D)
        uu, dd, ud = (self._gradient_field(x) for x in (up, down, mixed))
        grad_n = np.real(uu + dd)
        grad_m = np.stack([2.0 * np.real(ud), -2.0 * np.imag(ud),
                           np.real(uu - dd)], axis=1)
        return grad_n, grad_m

    def _matrix(self, v) -> np.ndarray:
        r""":math:`\int\phi_p^* v\,\phi_q`."""
        phi = self.orbitals
        return (phi.conj() * v) @ phi.T * self.grid.dV

    def _tau_matrix(self, X) -> np.ndarray:
        r""":math:`\tfrac12\sum_{pq}X_{pq}\nabla\phi_p\cdot\nabla\phi_q^*`
        on the grid (complex)."""
        out = 0.0
        for dphi in self._orbital_gradients:
            out = out + np.sum(dphi * (X @ dphi.conj()), axis=0)
        return 0.5 * out

    def spinor_kinetic_energy_density(self, D):
        r"""``(tau, tau_m)``: the trace and the ``(3, ngrid)`` vector of the
        spinors' kinetic-energy density matrix, built as the magnetization
        is from the spin blocks (:meth:`spin_densities`)."""
        up, down, mixed = self._blocks(D)
        uu, dd, ud = (self._tau_matrix(x) for x in (up, down, mixed))
        return (np.real(uu + dd),
                np.stack([2.0 * np.real(ud), -2.0 * np.imag(ud),
                          np.real(uu - dd)]))

    def _gradient_matrix(self, v) -> np.ndarray:
        r""":math:`\tfrac12\int v\,\nabla\phi_p^*\cdot\nabla\phi_q`."""
        out = 0.0
        for dphi in self._orbital_gradients:
            out = out + (dphi.conj() * (0.5 * v)) @ dphi.T * self.grid.dV
        return out

    @staticmethod
    def _pauli(M, identity, x, y, z) -> np.ndarray:
        """The ``(2M, 2M)`` matrix ``identity 1 + (x, y, z) . sigma`` of
        ``(M, M)`` parts, alpha block first."""
        V = np.zeros((2 * M, 2 * M), dtype=complex)
        V[:M, :M] = identity + z
        V[M:, M:] = identity - z
        V[:M, M:] = x - 1j * y
        V[M:, :M] = x + 1j * y
        return V

    def _xc(self, D):
        r"""``(E_xc, V_xc)``: the noncollinear functional's energy and its
        ``(2M, 2M)`` matrix, :math:`v\,\mathbb 1 + \mathbf B\cdot
        \boldsymbol\sigma` in each spin block (and r2SCAN's
        :math:`\tfrac12\nabla\phi^*\cdot(v_\tau + \mathbf b_\tau\cdot
        \boldsymbol\sigma)\nabla\phi`).  A hybrid adds the short-range
        exact exchange of the four spin blocks,
        :math:`-\tfrac a2\sum_{\sigma\tau}\mathrm{tr}(D^{\tau\sigma}
        K[D^{\sigma\tau}])` and :math:`-aK[D^{\sigma\tau}]`, and the
        spheres' terms."""
        n, m = self.spin_densities(D)
        valence = n
        if self.core_density is not None:
            n = n + self.core_density
        tau = tau_m = gradients = None
        if self.meta:
            tau, tau_m = self.spinor_kinetic_energy_density(D)
            if self._core_tau is not None:
                tau = tau + self._core_tau
            grad_n, grad_m = self.spin_gradients(D)
            if self.core_density is not None:
                grad_n = grad_n + self._core_gradient()
            gradients = (grad_n, grad_m)
        terms = evaluate_noncollinear(self.grid, n, m, self.functional,
                                      relativistic=self.relativistic,
                                      tau=tau, tau_magnetization=tau_m,
                                      screening=self.screening,
                                      exchange_density=valence,
                                      gradients=gradients)
        M = self.M
        identity = self._matrix(terms.potential)
        paulis = [self._matrix(b) for b in terms.field]
        if terms.gradient_potential is not None:
            def term(vector):
                return gradient_term_matrix(
                    self.orbitals, self._orbital_gradients, vector,
                    self.grid.dV)
            identity = identity + term(terms.gradient_potential)
            paulis = [b + term(terms.gradient_field[:, a])
                      for a, b in enumerate(paulis)]
        V = self._pauli(M, identity, *paulis)
        if terms.tau_potential is not None:
            V = V + self._pauli(M, self._gradient_matrix(terms.tau_potential),
                                *(self._gradient_matrix(b)
                                  for b in terms.tau_field))
        energy = terms.energy
        self.exact_exchange_energy = 0.0
        if self.screening is not None:
            a = self.screening[1]
            blocks = [[D[s * M:(s + 1) * M, t * M:(t + 1) * M]
                       for t in range(2)] for s in range(2)]
            exact = 0.0
            for s in range(2):
                for t in range(2):
                    K = self._exchange_matrix(blocks[s][t])
                    exact -= 0.5 * a * float(np.real(np.sum(
                        blocks[t][s] * K.T)))
                    V[s * M:(s + 1) * M, t * M:(t + 1) * M] -= a * K
            energy += exact
            self.exact_exchange_energy = exact
            if self.one_center is not None:
                C, hybrid = self.one_center
                spheres = hybrid.evaluate_spinor(
                    [[C.conj().T @ blocks[s][t] @ C for t in range(2)]
                     for s in range(2)])
                self.exact_exchange_energy += spheres.exact_exchange
                energy += spheres.energy
                P = C.shape[1]
                O = spheres.operators[0]
                for s in range(2):
                    for t in range(2):
                        V[s * M:(s + 1) * M, t * M:(t + 1) * M] += (
                            C @ O[s * P:(s + 1) * P, t * P:(t + 1) * P]
                            @ C.conj().T)
        return energy, 0.5 * (V + V.conj().T)

    def _fock(self, D):
        """``(F, E_elec, E_H, E_xc)`` at the spinor density matrix ``D``."""
        up, down, _mixed = self._blocks(D)
        total = up + down
        J = self._hartree(total)
        e_xc, V_xc = self._xc(D)
        M = self.M
        F = self.h_spin + V_xc
        F[:M, :M] += J
        F[M:, M:] += J
        e_one = float(np.real(np.sum(D * self.h_spin.T)))
        e_hartree = 0.5 * float(np.real(np.sum(total * J.T)))
        return (0.5 * (F + F.conj().T), e_one + e_hartree + e_xc, e_hartree,
                e_xc)

    def determinant_energy(self, C) -> float:
        r"""Hartree-Fock energy of the determinant of the spinor columns ``C``
        in the many-body Hamiltonian (Hartree, no constant):
        :math:`\sum_i h_{ii} + \tfrac12\sum_{ij}(\langle ij|ij\rangle -
        \langle ij|ji\rangle)`.  The tensor route evaluates it as
        :class:`~.hartree_fock.GHF` does; the Poisson route takes the Coulomb
        sum from one solve of the determinant's density and the exchange from
        one solve per spinor pair
        (:meth:`~.dft.PoissonHartree.spinor_exchange`)."""
        C = np.asarray(C)
        n = C.shape[1]
        if self.poisson is None:
            return GHF(self.h_spin, self.eri, n).energy_of(C)
        D = C @ C.conj().T
        up, down, _mixed = self._blocks(D)
        total = up + down
        J = self._hartree(total)
        one = float(np.real(np.sum(D * self.h_spin.T)))
        coulomb = 0.5 * float(np.real(np.sum(total * J.T)))
        exchange = 0.5 * float(np.sum(self.poisson.spinor_exchange(C)))
        return one + coulomb - exchange

    # -- the SCF ---------------------------------------------------------- #

    def _mo_problem(self, V):
        """``(h_mo, eri_mo)`` in the spinor columns ``V`` (mode order)."""
        h_mo = V.conj().T @ self.h_spin @ V
        if self.poisson is not None:
            return h_mo, SpinorTensor(self.poisson.integrals, V)
        return h_mo, spinor_integrals(self.h_spin, self.eri, V)[1]

    def run(self, guess, max_iter: int = 200, tol: float = 1e-8,
            diis: bool = True) -> KohnShamGResult:
        """Iterate to self-consistency from ``guess``.

        ``guess`` is the starting ``(2M, 2M)`` spinor density matrix: the
        collinear solution (:func:`collinear_spinor_density`), or its
        noncollinear turn (:func:`noncollinear_spinor_density`).  The loop is
        :class:`~.dft.KohnSham`'s: DIIS restarted when it stalls, the level
        shift while the energy rises, the degenerate shell at the Fermi level
        shared equally, converged when both the energy change and the
        largest density-matrix change are below ``tol``.
        """
        D = np.asarray(guess, dtype=complex)
        occupations = None
        energy = np.inf
        converged = False
        mixer = _IndependentDIIS() if diis else None
        shift = 0.0
        it = 0
        history, start = [], time.perf_counter()
        for it in range(1, max_iter + 1):
            F, current, _e_h, _e_xc = self._fock(D)
            if mixer is not None and it % DIIS_RESTART == 0 \
                    and _stalled(history):
                mixer = _IndependentDIIS()
            if current > energy + LEVEL_SHIFT_TRIGGER and shift == 0.0:
                shift = LEVEL_SHIFT
            elif shift and abs(current - energy) < LEVEL_SHIFT_TRIGGER:
                shift = 0.0
            F_iter = mixer.extrapolate(F, D) if mixer else F
            if shift:
                F_iter = _level_shifted(F_iter, D, shift)
            _eps, C = np.linalg.eigh(F_iter)
            occupations = shell_occupations(_eps, self.n_electrons)
            D_new = self._density_matrix(C, occupations)
            change = float(np.max(np.abs(D_new - D)))
            history.append(scf_record(it, start, current, energy, change))
            if abs(current - energy) < tol and change < tol:
                D, energy = D_new, current
                converged = True
                break
            D, energy = D_new, current

        F, energy, e_hartree, e_xc = self._fock(D)
        eps, C = np.linalg.eigh(F)                  # canonical spinors
        V = C[:, GHF.mode_order(self.M)]
        h_mo, eri_mo = self._mo_problem(V)
        return KohnShamGResult(
            electronic_energy=energy, n_electrons=self.n_electrons,
            mo_energies=np.real(eps), mo_coefficients=C, h_mo=h_mo,
            eri_mo=eri_mo, converged=converged, n_iterations=it,
            mode_coefficients=V,
            kramers_pairing=kramers_pairing(eps, self.n_electrons),
            functional=self.functional, hartree_energy=e_hartree,
            xc_energy=e_xc, exact_exchange_energy=self.exact_exchange_energy,
            history=history, occupations=occupations,
            reference_energy=self.determinant_energy(
                C[:, :self.n_electrons]),
            spin_moment=self.spin_moment(D), spin_orbit=self.spin_orbit)


def spin_rotation(direction) -> np.ndarray:
    r"""The ``(2, 2)`` SU(2) matrix that turns spin up along :math:`z` to
    spin up along the unit ``direction`` :math:`(\sin\theta\cos\phi,
    \sin\theta\sin\phi, \cos\theta)`: columns the two spinors
    :math:`(\cos\tfrac\theta2, e^{i\phi}\sin\tfrac\theta2)` and
    :math:`(-e^{-i\phi}\sin\tfrac\theta2, \cos\tfrac\theta2)`."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    theta = float(np.arccos(np.clip(d[2], -1.0, 1.0)))
    phi = float(np.arctan2(d[1], d[0]))
    c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
    return np.array([[c, -np.exp(-1j * phi) * s],
                     [np.exp(1j * phi) * s, c]])


def _rotated_blocks(U, up, down) -> np.ndarray:
    r"""The spinor operator :math:`(U\otimes 1)\,\mathrm{diag}(up,
    down)\,(U\otimes 1)^\dagger` of spin-diagonal blocks ``up`` and
    ``down`` (``(..., n, n)`` each), alpha block first."""
    up, down = np.asarray(up), np.asarray(down)
    n = up.shape[-1]
    out = np.zeros(up.shape[:-2] + (2 * n, 2 * n), dtype=complex)
    for a in range(2):
        for b in range(2):
            out[..., a * n:(a + 1) * n, b * n:(b + 1) * n] = (
                U[a, 0] * np.conj(U[b, 0]) * up
                + U[a, 1] * np.conj(U[b, 1]) * down)
    return out


def spinor_functional_refusal(functional: str) -> str:
    """The message refusing a functional without a spinor form."""
    return (f"two-component (spin-orbit) Kohn-Sham supports "
            f"xc={' or '.join(repr(f) for f in NONCOLLINEAR_FUNCTIONALS)} "
            f"(noncollinear, local spin axis); xc={functional!r} has no "
            f"spinor form here.  Use a scalar-relativistic set "
            f"(paw_dataset='lda-sr' or 'pbe-sr') for it, or method='ghf' for "
            f"the spin-orbit mean field")


# --------------------------------------------------------------------------- #
# Crystals: two-component Bloch states.
# --------------------------------------------------------------------------- #

#: Why a consumer of a crystal's collinear Bloch states refuses a
#: two-component one.
TWO_COMPONENT_CRYSTAL_REFUSAL = (
    "the crystal was solved in two-component (spinor) Bloch states, which "
    "{what} does not take: it reads collinear states (one spin channel, or "
    "two).  The energy, the bands, the densities of states, the Fermi "
    "surface and the charges of a spinor crystal are available")


@dataclass
class PeriodicKohnShamGResult(PeriodicKohnShamResult):
    r"""A converged two-component (spinor) crystal (Hartree, per cell).

    The fields of :class:`~.periodic_dft.PeriodicKohnShamResult`, with one
    set of levels: ``eigenvalues[k]`` and ``occupations[k]`` are ``(2M,)``,
    each spinor band holding one electron (occupations 0 to 1), and
    ``n_spins`` is 1.  ``terms`` carries ``spin_orbit``, the expectation
    value of the spin-orbit term (part of the energy).

    ``scalar_energy`` is the :math:`\sigma \to 0` energy of the collinear
    solution of the same crystal without the spin-orbit term, the SCF's
    starting point.  ``spin_moment`` is :math:`2\langle\mathbf S\rangle =
    (M_x, M_y, M_z)` per cell in Bohr magnetons (augmentation included), the
    spin part only; ``magnetic_moment`` is its :math:`z` component.
    ``time_reversal`` says time-reversal symmetry was imposed: the crystal
    was solved as non-magnetic, its magnetization set to zero and not
    computed (no initial moments: the irreducible wedge holds);
    otherwise the crystal was solved on the whole mesh with the
    magnetization a vector field.  ``spin_orbit`` says whether the operator
    carried the spin-orbit term (a spinor run without it is a noncollinear
    start on a scalar set).
    """

    scalar_energy: float | None = None
    spin_orbit: bool = True
    spin_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time_reversal: bool = True

    @property
    def spin_orbit_energy(self) -> float | None:
        """``extrapolated_energy - scalar_energy`` with the spin-orbit term:
        what it (and the freedom of the spinors) changed the collinear
        crystal's energy by, in the same basis, grid and mesh (Hartree);
        ``None`` without the term (:attr:`noncollinear_energy`)."""
        if self.scalar_energy is None or not self.spin_orbit:
            return None
        return float(self.extrapolated_energy - self.scalar_energy)

    @property
    def noncollinear_energy(self) -> float | None:
        """``extrapolated_energy - scalar_energy`` of a spinor crystal
        without a spin-orbit term: what the noncollinear magnetization
        changed the collinear energy by (Hartree); ``None`` with the term."""
        if self.scalar_energy is None or self.spin_orbit:
            return None
        return float(self.extrapolated_energy - self.scalar_energy)


@dataclass
class SpinorPotentials:
    r"""The density-dependent parts of a two-component crystal's operator.

    ``V`` the local potential (identity in spin), ``w`` the compensation
    moments' derivatives, ``B`` the exchange-correlation field ``(3,
    ngrid)`` (``None`` when time reversal is imposed), ``v_tau`` and
    ``b_tau`` a meta-GGA's identity and Pauli parts on the kinetic-energy
    density matrix, ``one_center`` a hybrid's ``(2P, 2P)`` sphere operator
    and ``states`` the occupied spinor states (``MeshStates``) its exact
    exchange is taken over.
    """

    V: np.ndarray
    w: dict
    B: np.ndarray | None = None
    v_tau: np.ndarray | None = None
    b_tau: np.ndarray | None = None
    one_center: np.ndarray | None = None
    states: object = None


class SpinorPeriodicKohnSham(PeriodicKohnSham):
    r"""Two-component (spinor) Kohn-Sham equations of a crystal.

    The Bloch basis is :math:`\chi_{\mu\mathbf k}\otimes|\sigma\rangle`,
    alpha block first, with the spin-free metric :math:`S(\mathbf k)\otimes
    \mathbb 1`; at every k-point

    .. math::

        H(\mathbf k) = H^{0}(\mathbf k)\otimes\mathbb 1 + h^{SO}(\mathbf k)
            + \int\chi^*_{\mu\mathbf k}\,\mathbf B_{xc}\cdot
              \boldsymbol\sigma\,\chi_{\nu\mathbf k},

    with :math:`H^0` the collinear crystal's matrix of the total density
    (kinetic, local, nonlocal, Hartree, the compensation moments and
    :math:`v_{xc}`), :math:`h^{SO}` the molecule's spin-orbit term Bloch
    summed (:meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.spin_orbit_matrix`)
    and the field of the noncollinear functional
    (:func:`~mandacaru.integrals.noncollinear.evaluate_noncollinear`).
    Occupations are one electron per spinor band around one Fermi level;
    the smearing, the energy conventions and the mixing are the collinear
    crystal's.

    **The k-point mesh.**  The spin is not a separate quantum number, but
    time reversal :math:`\Theta = i\sigma_y K` still maps the states at
    :math:`\mathbf k` onto those at :math:`-\mathbf k` with the same energy
    and density and the *opposite* magnetization, and a space-group
    operation maps them, spin rotated, onto those at :math:`R\mathbf k`
    with the density moved and the magnetization rotated as an axial vector.
    So two routes are exact:

    * a crystal given **no initial moments** is solved as a non-magnetic
      one: time-reversal symmetry is imposed, so the magnetization is zero
      and not computed (``B_xc = 0``, the unpolarized functional), and the
      density is a scalar, so the wedge of the space group and time
      reversal reduces the mesh as for a collinear crystal -- the spin
      rotations do not change a density.  This is an assumption, as a
      collinear crystal without moments is restricted: a crystal that would
      magnetize (an open shell, a ferromagnet) must be given moments, or it
      lands on the time-reversal symmetric state (a free Tl atom at Gamma:
      a half-filled Kramers pair, 122 meV above its magnetic
      :math:`6p_{1/2}` state);
    * a **magnetic** one (``(N,)`` moments with the spin-orbit term, or
      ``(N, 3)`` moment vectors): time reversal and the operations both
      move the magnetization, so the crystal must be built on the **whole
      mesh** (``build_crystal(..., full_mesh=True)``) and the magnetization
      is the vector field of every k-point.

    The SCF continues the collinear one: :meth:`run` solves the collinear
    problem first (restricted, or spin-polarized with the moments' lengths
    along :math:`z`) and starts from the density its last diagonalization
    was built from -- with the magnetization along :math:`z`, or with each
    atom's Hirshfeld share of it turned to that atom's moment vector
    (``directions``).  Without the spin-orbit term and with every direction
    along :math:`z` the first spinor step is the collinear step itself, so
    the SCF stops there, on the collinear energy to round-off.

    **r2SCAN** adds the kinetic-energy density matrix
    (:meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.spinor_kinetic_energy_density`)
    and the operator :math:`\tfrac12\int\nabla\chi^*\cdot(v_\tau\mathbb 1 +
    \mathbf b_\tau\cdot\boldsymbol\sigma)\nabla\chi`; **HSE06** the
    spinor short-range exact exchange
    (:meth:`~.periodic_exchange.PeriodicExchange.spinor_matrices`: every spin
    block of the spinor density matrix, as in generalized Hartree-Fock) and
    its one-center terms
    (:meth:`~mandacaru.pseudopotentials.onecenter.OneCenterHybrid.evaluate_spinor`),
    mixed with the density as the collinear crystal's are.  The exchange
    needs every occupied state of the mesh, and the images of a wedge's
    spinors would need their spin rotations: a hybrid spinor crystal is
    solved on the whole mesh.

    Parameters
    ----------
    crystal, n_electrons, functional, smearing, relativistic, constant, screening
        As for :class:`~.periodic_dft.PeriodicKohnSham`; the functional is
        one of :data:`~mandacaru.integrals.noncollinear.NONCOLLINEAR_FUNCTIONALS`.
    magnetic_moments : (N,), optional
        The collinear start's moments (the lengths of moment vectors).
    directions : (N, 3), optional
        Each atom's unit moment direction (zero drops its share), or ``None``
        to keep the collinear axis; given, the crystal is magnetic.
    """

    two_component = True

    def __init__(self, crystal, n_electrons: float, functional: str = "lda",
                 smearing=None, relativistic: bool = False,
                 constant: float = 0.0, magnetic_moments=None,
                 directions=None, screening=None):
        if not supports_noncollinear(functional):
            raise NotImplementedError(spinor_functional_refusal(functional))
        super().__init__(crystal, n_electrons, functional, smearing=smearing,
                         relativistic=relativistic, constant=constant,
                         magnetic_moments=magnetic_moments,
                         screening=screening)
        self.directions = (None if directions is None
                           else np.asarray(directions, dtype=float))
        if self.directions is not None and \
                self.directions.shape != (len(crystal.datasets), 3):
            raise ValueError(f"directions must be ({len(crystal.datasets)}, "
                             f"3), one per atom, got {self.directions.shape}")
        #: Whether the operator carries the spin-orbit term.
        self.spin_orbit = bool(crystal.has_spin_orbit)
        #: Whether time-reversal symmetry is imposed (no initial moments):
        #: the magnetization is zero and not computed, and the wedge holds.
        self.time_reversal = self.directions is None and self.n_spins == 1
        if not self.time_reversal or self.screening is not None:
            self._require_full_mesh()
        if self.screening is not None:
            # Every mesh point's own states, for the collinear start too: no
            # time-reversal images (a spinor's is i sigma_y K, not K).
            self.exchange = exact_exchange.PeriodicExchange(
                crystal, self.screening[0], time_reversal=False)
        #: The collinear start (:class:`~.periodic_dft.PeriodicKohnShamResult`).
        self.scalar_result = None

    def _require_full_mesh(self) -> None:
        """A magnetic (or hybrid) spinor crystal on a reduced mesh would be
        silently wrong: refuse anything but the whole mesh."""
        c = self.crystal
        if c.symmetry is not None:
            raise ValueError("a magnetic or hybrid two-component crystal "
                             "needs the whole k-point mesh: build it with "
                             "full_mesh=True (no symmetry reduction)")
        B = rc.reciprocal_vectors(c.lattice)
        frac = c.kpoints @ np.linalg.inv(B.T)
        for k in frac:
            delta = frac + k[None, :]
            delta -= np.round(delta)
            if not np.any(np.all(np.abs(delta) < 1e-8, axis=1)):
                raise ValueError(
                    "a magnetic or hybrid two-component crystal needs the "
                    "whole k-point mesh, and this one lacks -k for some k "
                    "(it was reduced by time reversal): build it with "
                    "full_mesh=True")

    # -- the operator ------------------------------------------------------- #

    def _xc_tau(self, tau):
        """The kinetic-energy density the functional sees: ``tau`` plus the
        partial core's."""
        if tau is None:
            return None
        return tau + (self._core_tau if self._core_tau is not None else 0.0)

    def _noncollinear_terms(self, rho, m, tau=None, tau_m=None):
        """The noncollinear functional of ``(n~ + core, m)`` (and the
        kinetic-energy density matrix), a hybrid's short-range exchange
        taken on the valence ``rho``."""
        n = rho + (self.core if self.core is not None else 0.0)
        return evaluate_noncollinear(
            self.crystal.grid, n, m, self.functional,
            relativistic=self.relativistic, tau=self._xc_tau(tau),
            tau_magnetization=tau_m, screening=self.screening,
            exchange_density=rho)

    def _spinor_potentials(self, rho, q, m, tau=None,
                           tau_m=None) -> SpinorPotentials:
        """The operator's parts of an input ``(n~, q, m)`` (and ``tau``,
        ``tau_m``): the collinear crystal's potentials and no field when
        time reversal is imposed (``m`` is ``None``), else the
        noncollinear functional's."""
        if m is None:
            V, v_tau, w, _terms = self._potentials(rho, q, tau)
            return SpinorPotentials(V=V, w=w, v_tau=v_tau)
        V_es, w = self._electrostatics(rho, q)
        terms = self._noncollinear_terms(rho, m, tau, tau_m)
        return SpinorPotentials(V=V_es + terms.potential, w=w, B=terms.field,
                                v_tau=terms.tau_potential,
                                b_tau=terms.tau_field)

    def spinor_hamiltonians(self, device, potentials: SpinorPotentials,
                            K=None) -> np.ndarray:
        """``(nk, 2M, 2M)``: :math:`H(\\mathbf k)` at ``device``'s k-points
        (with its Bloch gradients for a meta-GGA) for ``potentials``, minus
        the hybrid's fraction of the exchange ``K`` (``(nk, 2M, 2M)``)."""
        p = potentials
        H0 = device.hamiltonians(p.V, p.v_tau, p.w)
        nk, M = H0.shape[0], H0.shape[1]
        H = np.zeros((nk, 2 * M, 2 * M), dtype=complex)
        H[:, :M, :M] = H0
        H[:, M:, M:] = H0
        for pauli, matrices in ((p.B, device.grid_matrices),
                                (p.b_tau, device.gradient_matrices)):
            if pauli is None:
                continue
            bx, by, bz = np.moveaxis(matrices(pauli), 1, 0)
            H[:, :M, :M] += bz
            H[:, M:, M:] -= bz
            H[:, :M, M:] += bx - 1j * by
            H[:, M:, :M] += bx + 1j * by
        if p.one_center is not None:
            P = len(self.crystal.projectors)
            for i, d in enumerate(device.kpoint_data):
                C = d.projections
                for a in range(2):
                    for b in range(2):
                        H[i, a * M:(a + 1) * M, b * M:(b + 1) * M] += (
                            C @ p.one_center[a * P:(a + 1) * P,
                                             b * P:(b + 1) * P]
                            @ C.conj().T)
        if self.spin_orbit:
            H = H + np.stack([d.spin_orbit for d in device.kpoint_data])
        if K is not None:
            H = H - self.screening[1] * K
        return 0.5 * (H + np.swapaxes(H, -1, -2).conj())

    # -- the density and the energy ---------------------------------------- #

    def _traced(self, P) -> np.ndarray:
        M = self.crystal.M
        return P[:, :M, :M] + P[:, M:, M:]

    def _spinor_density(self, device, P):
        """``(n~, q, m, tau, tau_m)`` of the spinor density matrices ``P``:
        ``m`` (and ``tau_m``) ``None`` when time reversal is imposed,
        ``tau`` ``None`` without a meta-GGA."""
        traced = list(self._traced(P))
        rho, q = device.density(traced)
        m = tau = tau_m = None
        if self.time_reversal:
            if self.meta:
                tau = self._tau(traced)
        else:
            m = device.magnetization(list(P))
            if self.meta:
                tau, tau_m = self.crystal.spinor_kinetic_energy_density(
                    list(P), self._gradients)
        return rho, q, m, tau, tau_m

    def spinor_energy_terms(self, P, rho, q, m, tau=None,
                            tau_m=None) -> dict:
        """Every term of the energy of the spinor density matrices ``P`` and
        their densities (Hartree): the collinear crystal's terms of the
        traced matrices, ``spin_orbit`` (:math:`\\sum_k w_k\\,
        \\mathrm{tr}\\,P_k h^{SO}_k`) and, for a magnetization, the
        noncollinear exchange-correlation energy.  A hybrid's exact exchange
        is the caller's (:meth:`spinor_hybrid_terms`)."""
        c = self.crystal
        if m is None:
            terms = self.energy_terms(list(self._traced(P)), rho, q, tau)
        else:
            band = sum(d.weight * float(np.real(np.sum(T * d.fixed.T)))
                       for d, T in zip(c.kpoint_data, self._traced(P)))
            terms = {"band_fixed": band,
                     "electrostatic": self.electrostatic_energy(rho, q),
                     "xc": self._noncollinear_terms(rho, m, tau,
                                                    tau_m).energy,
                     "constant": self.constant}
        terms["spin_orbit"] = (sum(
            d.weight * float(np.real(np.sum(Pk * d.spin_orbit.T)))
            for d, Pk in zip(c.kpoint_data, P)) if self.spin_orbit else 0.0)
        return terms

    def spinor_hybrid_terms(self, vectors, occupations, P):
        """``(energy, states, K, O)``: the spinor exact exchange of the
        mesh's occupied states (the grid part and the spheres' one-center
        terms, Hartree), the states, their exchange matrices ``(nk, 2M,
        2M)`` at the SCF's k-points and the spheres' ``(2P, 2P)``
        operator."""
        data = self.crystal.kpoint_data
        states = self.exchange.spinor_mesh_states(data, vectors, occupations)
        K = self.exchange.spinor_matrices(data, states)
        one = self.spheres.evaluate_spinor(states.projected_density_blocks())
        grid = exact_exchange.exchange_energy(data, list(P), K,
                                              0.5 * self.screening[1])
        return grid + one.energy, states, K, one.operators[0]

    def spin_moment(self, P) -> np.ndarray:
        r""":math:`2\langle\mathbf S\rangle = (M_x, M_y, M_z)` per cell
        (Bohr magnetons): :math:`\sum_k w_k\,\mathrm{tr}(P^k_{\sigma\tau}
        S_k)` of the spin blocks, the augmented overlap carrying the spheres'
        share.  Zero when time-reversal symmetry is imposed."""
        if self.time_reversal:
            return np.zeros(3)
        M = self.crystal.M
        out = np.zeros(3)
        for d, Pk in zip(self.crystal.kpoint_data, P):
            S = d.overlap.T
            ud = np.sum(Pk[:M, M:] * S)
            out += d.weight * np.array([
                2.0 * np.real(ud), -2.0 * np.imag(ud),
                np.real(np.sum((Pk[:M, :M] - Pk[M:, M:]) * S))])
        return out

    # -- the start ---------------------------------------------------------- #

    def _turned(self, field_z) -> np.ndarray:
        """A collinear ``(ngrid,)`` field along :math:`z` as ``(3, ngrid)``:
        along :math:`z`, or each atom's Hirshfeld share turned to its
        direction."""
        if self.directions is None:
            out = np.zeros((3, field_z.size))
            out[2] = field_z
            return out
        axis = np.einsum("ag,ac->cg", self.crystal.valence_shares(),
                         self.directions)
        return field_z[None, :] * axis

    def _start(self, start: dict) -> dict:
        """The spinor SCF's first input from the collinear start's last one
        (``scf_input``): the density and moments, the magnetization (and
        r2SCAN's tau) along :math:`z` or turned, and a hybrid's operators
        as spin-diagonal blocks -- the collinear operator itself."""
        rho, q, m_z, tau = (start["rho"], start["q"], start["m"],
                            start["tau"])
        spin = m_z is not None
        out = {"rho": rho, "q": q, "m": None, "tau": None, "tau_m": None,
               "operators": None}
        if self.meta:
            if spin:
                tau_up, tau_dn = tau
                out["tau"] = tau_up + tau_dn
                tau_z = tau_up - tau_dn
            else:
                out["tau"], tau_z = tau, np.zeros_like(tau)
            if not self.time_reversal:
                out["tau_m"] = self._turned(tau_z)
        if not self.time_reversal:
            out["m"] = self._turned(m_z if spin
                                    else np.zeros(self.crystal.grid.size))
        operators = start["operators"]
        if operators is not None:
            # The collinear H = H0 - (a/2) K (restricted, both spins) or
            # H0 - a K_s (a channel); the spinor one is H0 - a K_spinor.
            if spin:
                K_up, K_dn, O_up, O_dn = operators
            else:
                K_up = K_dn = 0.5 * operators[0]
                O_up = O_dn = operators[1]
            # Turned with the spin rotation that takes z to the moments'
            # mean direction: exact for a common direction, a start
            # otherwise (as the turned shares of m are).
            U = spin_rotation(self._mean_direction())
            out["operators"] = [_rotated_blocks(U, K_up, K_dn),
                                _rotated_blocks(U, O_up, O_dn)]
        return out

    def _mean_direction(self) -> np.ndarray:
        """The moment-weighted mean of the atoms' directions (``z`` without
        directions, or when they cancel)."""
        if self.directions is None:
            return np.array([0.0, 0.0, 1.0])
        lengths = (np.ones(len(self.directions))
                   if self.initial_moments is None
                   else np.abs(np.asarray(self.initial_moments, float)))
        mean = lengths @ self.directions
        size = float(np.linalg.norm(mean))
        return mean / size if size > 1e-12 else np.array([0.0, 0.0, 1.0])

    # -- the loop ----------------------------------------------------------- #

    @single_threaded_blas
    def run(self, max_iter: int = 300, tol: float = 1e-7,
            density_tol: float = 1e-5) -> PeriodicKohnShamGResult:
        """Solve the collinear crystal, then the spinor one from it (the
        class docstring); the tolerances are the collinear SCF's."""
        c, device = self.crystal, self.device
        collinear = PeriodicKohnSham.run(self, max_iter, tol, density_tol)
        self.scalar_result = collinear
        start = self._start(self.scf_input)
        rho, q, m = start["rho"], start["q"], start["m"]
        tau, tau_m = start["tau"], start["tau_m"]
        operators_in = start["operators"]
        mixer = PulayMixer(c, spin=m is not None)
        weights = list(c.weights)
        # The spinor SCF continues the collinear one: its first step is
        # compared with the collinear energy.
        previous = float(collinear.free_energy)
        converged, it = False, 0
        history, begin = [], time.perf_counter()
        hybrid, change = None, 0.0
        for it in range(1, max_iter + 1):
            inputs = (rho, q, m, tau, tau_m, operators_in)
            potentials = self._spinor_potentials(rho, q, m, tau, tau_m)
            if operators_in is not None:
                potentials.one_center = operators_in[1]
            H = self.spinor_hamiltonians(
                device, potentials,
                None if operators_in is None else operators_in[0])
            eps, vectors = device.spinor_eigensolve(H)
            eigenvalues = list(eps)
            mu = fermi_level(eigenvalues, weights, self.n_electrons,
                             self.method, self.width, degeneracy=1.0)
            f = occupation((eps - mu) / self.width, self.method)
            P = device.density_matrices(vectors, f)
            rho_out, q_out, m_out, tau_out, tau_m_out = \
                self._spinor_density(device, P)
            terms = self.spinor_energy_terms(P, rho_out, q_out, m_out,
                                             tau_out, tau_m_out)
            if self.screening is not None:
                energy_x, states, K, O = self.spinor_hybrid_terms(
                    vectors, f, P)
                terms["exact_exchange"] = energy_x
                hybrid = (states, O)
                operators_out = [K, O]
                change = self._operator_change(operators_in, operators_out)
            energy = sum(terms.values())
            entropy_term = -self.width * sum(
                d.weight * float(np.sum(entropy((e - mu) / self.width,
                                                self.method)))
                for d, e in zip(c.kpoint_data, eigenvalues))
            free = energy + entropy_term
            residual = PulayMixer.residual_norm(rho, rho_out, c.grid.dV)
            extra = {}
            if m is not None:
                residual += PulayMixer.residual_norm(m, m_out, c.grid.dV)
                extra["moment"] = float(np.linalg.norm(self.spin_moment(P)))
            history.append(scf_record(it, begin, free, previous, residual,
                                      **extra))
            if (abs(free - previous) < tol and residual < density_tol
                    and change < HYBRID_TOL):
                converged = True
                break
            previous = free
            extra_in = extra_out = None
            if self.screening is not None:
                extra_in = (operators_in if operators_in is not None else
                            [np.zeros_like(x) for x in operators_out])
                extra_out = operators_out
            if m is None:
                mixed = mixer.mix(rho, q, rho_out, q_out, extra_in=extra_in,
                                  extra_out=extra_out)
                rho, q = mixed[0], mixed[1]
            else:
                mixed = mixer.mix(rho, q, rho_out, q_out, m, m_out,
                                  extra_in=extra_in, extra_out=extra_out)
                rho, q, m = mixed[0], mixed[1], mixed[2]
            if self.screening is not None:
                operators_in = mixed[-1]
            tau, tau_m = tau_out, tau_m_out
        extrapolated = (free if self.method == "methfessel-paxton"
                        else 0.5 * (energy + free))
        terms["entropy"] = entropy_term
        if hybrid is not None:
            # The bands take the exchange of the output states, as the
            # collinear crystal's do (equal to the inputs' to HYBRID_TOL).
            potentials.states, potentials.one_center = hybrid
        self.potentials = potentials
        self.scf_input = dict(zip(("rho", "q", "m", "tau", "tau_m",
                                   "operators"), inputs))
        self.density_matrices = list(P)
        self.output_density = (rho_out, q_out, m_out)
        reference = self.eigenvalue_reference()
        moment = self.spin_moment(P)
        return PeriodicKohnShamGResult(
            functional=self.functional, free_energy=free, energy=energy,
            extrapolated_energy=extrapolated, fermi_level=mu + reference,
            kpoints=c.kpoints, weights=c.weights,
            eigenvalues=[e + reference for e in eigenvalues],
            occupations=list(f), converged=converged, n_iterations=it,
            smearing=(self.method, self.width), terms=terms, n_spins=1,
            magnetic_moment=float(moment[2]), history=history,
            scalar_energy=float(collinear.extrapolated_energy),
            spin_orbit=self.spin_orbit, spin_moment=moment,
            time_reversal=self.time_reversal)

    # -- after convergence -------------------------------------------------- #

    def _diagonalized(self, kpoints, potentials=None, _unused=None,
                      _unused_too=None, channel=0, data=None):
        """Yield ``(data, eigenvalues, vectors)`` for blocks of
        Cartesian ``kpoints`` at the frozen operator (``potentials``,
        default the converged one): the k-point matrices, the eigenvalues
        (Hartree, on the SCF's own reference) and the ``(2M, 2M)`` spinor
        eigenvectors, alpha rows first, orthonormal in the metric
        :math:`S\\otimes\\mathbb 1`.  The signature is the collinear
        crystal's (:meth:`channel_potentials` returns ``(potentials, None,
        None, 0)``), so the Berry phase and the Wannier functions take the
        same route."""
        c = self.crystal
        p = self.potentials if potentials is None else potentials
        if p is None:
            raise RuntimeError("the crystal has no converged operator: "
                               "run() first")
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        given = data
        block = c.kpoint_block() if given is None else len(kpoints)
        for start in range(0, len(kpoints), block):
            data = (list(given) if given is not None
                    else c.kpoint_matrices(kpoints[start:start + block]))
            gradients = ([c.bloch_gradients(d.psi, d.k) for d in data]
                         if self.meta else None)
            device = make_device(c, data, gradients)
            K = (None if p.states is None
                 else self.exchange.spinor_matrices(data, p.states))
            eps, vectors = device.spinor_eigensolve(
                self.spinor_hamiltonians(device, p, K))
            yield data, eps, vectors

    def channel_potentials(self):
        """``[(potentials, None, None, 0)]``: the one two-component
        "channel", in the shape of the collinear crystal's, the arguments of
        :meth:`_diagonalized`."""
        if self.potentials is None:
            raise RuntimeError("the crystal has no converged potential: "
                               "run() first")
        return [(self.potentials, None, None, 0)]

    @single_threaded_blas
    def bands(self, kpoints, projections: bool = False):
        r"""Non-self-consistent spinor bands at Cartesian ``kpoints``
        (Bohr^-1), the converged operator frozen.

        Returns ``(eigenvalues, weights)``: ``(nk, 2M)`` in Hartree on the
        result's reference, and with ``projections`` the ``(nk, M, 2M)``
        Loewdin weights of each spinor band on each orbital, its two spin
        components summed (each band's weights sum to one), else ``None``.
        """
        if self.potentials is None:
            raise RuntimeError("bands() needs a converged run() first")
        M = self.crystal.M
        reference = self.eigenvalue_reference()
        eigenvalues, weights = [], []
        for data, eps, vectors in self._diagonalized(kpoints):
            eigenvalues.extend(eps + reference)
            if projections:
                for d, C in zip(data, vectors):
                    s, U = np.linalg.eigh(d.overlap)
                    root = (U * np.sqrt(np.clip(s, 0.0, None))) @ U.conj().T
                    weights.append(np.abs(root @ C[:M]) ** 2
                                   + np.abs(root @ C[M:]) ** 2)
        return (np.array(eigenvalues),
                np.array(weights) if projections else None)
