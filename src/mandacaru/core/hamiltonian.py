# -*- coding: utf-8 -*-
# file: core/hamiltonian.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Electronic-structure integrals and the molecular Hamiltonian.

:class:`MolecularIntegrals` computes the one- and two-body integrals over a
localized Hydrogenic Atomic Orbitals (HAO) basis by driving the real-space
:class:`~mandacaru.integrals.IntegralEngine`, and assembles the second-quantized
molecular Hamiltonian as a :class:`~mandacaru.core.mapping.Fermion`.

Conventions (atomic units, Hartree):

* one-body ``h_pq = <p| -1/2 nabla^2 - sum_I Z_I/|r-R_I| |q>``;
* two-body in **physicists' notation**
  ``<pq|rs> = int int p*(1) q*(2) r(1) s(2)/r12`` -- electron 1 carries the
  orbital pair ``(p, r)`` and electron 2 the pair ``(q, s)`` (this is exactly
  what :meth:`IntegralEngine.two_body` returns);
* the Hamiltonian uses the ordering of :meth:`Fermion.from_integrals`,

  .. math::

      H = \sum_{PQ} h_{PQ}\, a^\dagger_P a_Q
        + \tfrac12 \sum_{PQRS} \langle PQ|RS\rangle\, a^\dagger_P a^\dagger_Q a_S a_R,

  over spin-orbitals, with the spin-blocked expansion of the spatial integrals
  (the standard physicists'-notation second-quantized electronic Hamiltonian).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import warnings

import numpy as np

from ..basis import HydrogenicAtomicOrbital
from ..integrals import Grid, IntegralEngine, Potentials
from ..units import to_bohr
from .mapping import Fermion

if TYPE_CHECKING:
    from ..algorithms.hartree_fock import RHFResult, UHFResult

#: Smallest overlap eigenvalue the Loewdin transform accepts.
OVERLAP_EIGENVALUE_FLOOR = 1e-10
#: Below this the basis is reported as nearly linearly dependent.
OVERLAP_EIGENVALUE_WARN = 1e-6


def _radial_norm(fn) -> float:
    """``int R^2 r^2 dr`` of a radially tabulated function (``nan`` if none)."""
    from ..integrals.engine import radial_norm

    radial = getattr(fn, "radial", None)
    return float("nan") if radial is None else radial_norm(radial)


#: Resolution ratios outside ``[1 - RESOLUTION_TOLERANCE, 1 + RESOLUTION_TOLERANCE]``
#: mark a basis function (or projector) the grid does not resolve.
RESOLUTION_TOLERANCE = 0.25


class MeanFieldMixin:
    """Hartree-Fock on the ``one_body()`` / ``two_body()`` of an integral class
    whose basis is orthonormal (Löwdin-orthogonalized orbitals, plane waves)."""

    def hartree_fock(self, n_electrons: int) -> RHFResult:
        """Restricted Hartree-Fock in the orthonormal spatial basis.

        Returns an :class:`~mandacaru.algorithms.hartree_fock.RHFResult` with the
        MO coefficients, orbital energies and MO-basis integrals.
        """
        from ..algorithms.hartree_fock import RHF
        cached = getattr(self, "_last_rhf_result", None)
        if cached is not None and cached[0] == int(n_electrons):
            return cached[1]
        result = RHF(self.one_body(), self.two_body(), n_electrons).run()
        self._last_rhf_result = (int(n_electrons), result)
        return result

    def open_shell_hartree_fock(self, n_alpha: int,
                               n_beta: int) -> UHFResult:
        """Unrestricted Hartree-Fock for ``(n_alpha, n_beta)`` electrons.

        Returns an :class:`~mandacaru.algorithms.hartree_fock.UHFResult` whose
        ``h_mo`` / ``eri_mo`` are in the **natural-orbital** basis of the UHF
        total density -- the single spatial basis an open-shell (odd-electron or
        spin-polarized) Hamiltonian is written in.  Complex integrals are
        handled.
        """
        from ..algorithms.hartree_fock import UHF
        particles = (int(n_alpha), int(n_beta))
        cached = getattr(self, "_last_uhf_result", None)
        if cached is not None and cached[0] == particles:
            return cached[1]
        result = UHF(self.one_body(), self.two_body(), n_alpha, n_beta).solve()
        self._last_uhf_result = ((int(n_alpha), int(n_beta)), result)
        return result


class MolecularIntegrals(MeanFieldMixin):
    r"""One- and two-body integrals over a localized basis for a molecule.

    Parameters
    ----------
    nuclei : sequence of ``(Z, position)``
        Nuclear charges and Cartesian positions (in ``units``) defining the
        electron-nuclear potential.
    basis : sequence of BasisFunction
        Spatial orbitals spanning the active space (e.g. ``HydrogenicAtomicOrbital``).
    grid : Grid
        Real-space integration grid.
    units : {"angstrom", "bohr"}
        Unit of the nuclear positions (default ``"angstrom"``).
    softening : float
        Coulomb softening length in **Bohr** (default ``1e-12``, effectively a bare
        ``-Z/r``): the electron-nuclear potential is ``-Z/max(r, softening)``, which
        bounds the singularity a nucleus samples when it sits on (or very near) a
        grid node.  The ASE-calculator path sets it to a fraction of the grid step
        so heavier-atom cores stay numerically finite on a coarse grid.
    kinetic : {"fd", "spectral"}
        Discretization of the Laplacian (default ``"fd"``, the 3-point
        finite-difference stencil).  ``"spectral"`` uses the FFT Laplacian,
        which never underestimates a function's kinetic energy and so prevents
        the collapse of compact functions into deep potentials; see
        :meth:`mandacaru.integrals.IntegralEngine.one_body`.
    kb_projectors : sequence, optional
        Nonlocal projector functions :math:`\chi_p` (sampled like basis
        functions), each carrying ``atom_index``, ``channel = (l, m)`` and a
        radial ``index`` within that channel -- see
        :class:`mandacaru.pseudopotentials.orbitals.KBProjector`.
    nonlocal_coupling : dict, optional
        The blocks of the coupling matrix :math:`D` of the general separable
        form :math:`H^{NL} = C D C^\dagger` (see :meth:`kb_nonlocal`),
        ``{(atom_index, l, m): (n, n) array}`` with ``n`` the number of radial
        projectors in that channel.  ``None`` takes each projector's own
        ``kb_energy`` on the diagonal -- the Kleinman-Bylander form.
    nonlocal_overlap : dict, optional
        Blocks of an overlap correction :math:`Q` in the same layout.  When
        given, the basis overlap used for the Loewdin orthogonalization becomes
        :math:`S + C Q C^\dagger` (PAW-LCAO-type augmented overlap).  ``None`` (the
        norm-conserving case) leaves :math:`S` alone.
    """

    def __init__(self, nuclei: Sequence[tuple[float, np.ndarray]],
                 basis, grid: Grid, units: str = "angstrom",
                 orthogonalize: bool = True, softening: float = 1e-12,
                 pseudos=None, kb_projectors=None,
                 kinetic: str = "fd", nonlocal_coupling=None,
                 nonlocal_overlap=None, periodic: bool = False,
                 spin_orbit_coupling=None):
        if kinetic not in ("fd", "spectral"):
            raise ValueError(f"unknown kinetic operator {kinetic!r}; use "
                             "'fd' or 'spectral'")
        self.kinetic = kinetic
        #: Whether the Coulomb kernel and the external potential are
        #: periodic (see :class:`~mandacaru.core.periodic.PeriodicIntegrals`).
        self.periodic = bool(periodic)
        #: ``T_grid / T_exact`` per basis function, filled by the integrals.
        self.resolution_ratios: np.ndarray | None = None
        #: ``<chi|chi>_grid / <chi|chi>_radial`` per KB projector, likewise.
        self.kb_resolution_ratios: np.ndarray | None = None
        self.nuclei = [(float(Z), np.asarray(R, dtype=float)) for Z, R in nuclei]
        self.basis = list(basis)
        self.grid = grid
        self.units = units
        self.orthogonalize = orthogonalize
        self._engine = IntegralEngine(self.basis, grid, periodic=periodic)
        # With pseudopotentials the "nuclei" carry the *ionic* charges Z_ion, so
        # the nuclear repulsion below is already the ion-ion term.
        self.pseudopotentials = (list(pseudos)
                                 if pseudos is not None else None)
        self.kb_projectors = list(kb_projectors) if kb_projectors else []
        #: Blocks of the nonlocal coupling matrix ``D`` (``None``: KB diagonal).
        self.nonlocal_coupling = (dict(nonlocal_coupling)
                                  if nonlocal_coupling is not None else None)
        #: Blocks of the overlap correction ``Q`` (``None``: norm-conserving).
        self.nonlocal_overlap = (dict(nonlocal_overlap)
                                 if nonlocal_overlap is not None else None)
        #: Blocks of the spin-orbit coupling, ``{(atom, l): D_SO}``.  Empty
        #: unless the pseudopotentials were generated with
        #: ``relativity="dirac"`` -- see :mod:`mandacaru.core.spin_orbit`.
        self.spin_orbit_coupling = (dict(spin_orbit_coupling)
                                    if spin_orbit_coupling else {})
        if self.nonlocal_overlap is not None and not self.kb_projectors:
            raise ValueError("nonlocal_overlap needs projectors to act on")
        if self.spin_orbit_coupling and not self.kb_projectors:
            raise ValueError("spin_orbit_coupling needs projectors to act on")
        self._potentials = Potentials(self.nuclei, softening=softening,
                                      units=units,
                                      pseudos=self.pseudopotentials)
        #: Additive constant (Hartree) carried into every Hamiltonian this
        #: object assembles, next to the nuclear repulsion -- e.g. the frozen
        #: one-center energies of a PAW-LCAO dataset.  Zero for a plain basis.
        self.constant_energy: float = 0.0
        self._S: np.ndarray | None = None
        self._S_bare: np.ndarray | None = None
        self._C: np.ndarray | None = None
        self._h1: np.ndarray | None = None
        self._eri: np.ndarray | None = None
        #: Molecular-orbital coefficients (columns, in the Loewdin-orthonormal
        #: basis) of the last ``molecular_hamiltonian(mo_basis=True)``: the
        #: RHF orbitals, or the UHF natural orbitals for an open shell.
        self.mo_coefficients: np.ndarray | None = None
        #: :class:`~mandacaru.algorithms.active_space.ActiveSpace` of the last
        #: ``molecular_hamiltonian(mo_basis=True)`` -- which spatial orbitals
        #: were frozen, kept and deleted.  ``None`` until one has been built.
        self.active_space = None

    @property
    def n_orbitals(self) -> int:
        """Number of spatial orbitals."""
        return len(self.basis)

    def integration_profile(self) -> dict:
        """Timing / cores / peak-memory profile of the integral engine.

        Populated once the integrals have run (via :meth:`molecular_hamiltonian`,
        :meth:`one_body`, ...).  Keys: ``stages_s`` (per-stage wall times),
        ``total_s``, ``peak_memory_mb``, ``n_cores`` (OpenMP threads, ``None`` for
        the NumPy fallback) and ``backend``.
        """
        return self._engine.integration_profile()

    # -- overlap and orthogonalization ------------------------------------ #

    def bare_overlap(self) -> np.ndarray:
        r"""Grid overlap ``S_pq = <p|q>`` of the (generally non-orthogonal) basis."""
        if self._S_bare is None:
            if self.periodic:
                # The periodic basis is the image sum the engine sampled; the
                # bare functions are a different set, and an overlap taken over
                # them would not match the T, V and g built from the stack.
                psi = self._engine._psi
            else:
                psi = np.stack(
                    [b.evaluate(self.grid.X, self.grid.Y, self.grid.Z).ravel()
                     for b in self.basis])
            S = (np.conj(psi) @ psi.T) * self.grid.dV
            self._S_bare = 0.5 * (S + S.conj().T)
        return self._S_bare

    def overlap(self) -> np.ndarray:
        r"""The overlap the orthonormalization uses.

        The bare grid overlap :meth:`bare_overlap` for a norm-conserving basis;
        with an overlap correction (``nonlocal_overlap``, the PAW-LCAO-type
        :math:`Q` blocks) it is the augmented :math:`S + C Q C^\dagger`, where
        :math:`C` are the projections of :meth:`projections`.
        """
        if self._S is None:
            S = self.bare_overlap()
            Q = self.nonlocal_overlap_matrix()
            if Q is not None:
                C = self.projections()
                S = S + C @ Q @ C.conj().T
                S = 0.5 * (S + S.conj().T)
            self._S = S
        return self._S

    def _lowdin_x(self) -> np.ndarray:
        r"""Symmetric orthogonalization matrix :math:`X = S^{-1/2}`.

        :math:`S^{-1/2}` divides by :math:`\sqrt{\lambda}`, so a
        near-dependent basis (a tiny overlap eigenvalue) multiplies the grid's
        integration noise by :math:`\lambda^{-1/2}` and quietly corrupts every
        transformed integral.  The spectrum is therefore checked: below
        :data:`OVERLAP_EIGENVALUE_WARN` it warns, and at or below
        :data:`OVERLAP_EIGENVALUE_FLOOR` -- where the transform is numerical
        noise -- it refuses.
        """
        S = self.overlap()
        w, U = np.linalg.eigh(S)
        smallest = float(w.min())
        if smallest <= OVERLAP_EIGENVALUE_FLOOR:
            raise ValueError(
                f"the basis is linearly dependent on this grid: the smallest "
                f"overlap eigenvalue is {smallest:.3e} (floor "
                f"{OVERLAP_EIGENVALUE_FLOOR:.1e}), so S^(-1/2) would amplify "
                f"integration noise by {1.0 / np.sqrt(max(smallest, 1e-300)):.1e}. "
                f"Drop the redundant functions (a smaller basis `size`) or "
                f"refine the grid (smaller h).")
        if smallest < OVERLAP_EIGENVALUE_WARN:
            warnings.warn(
                f"nearly linearly dependent basis: smallest overlap eigenvalue "
                f"{smallest:.3e}; integrals are amplified by "
                f"{1.0 / np.sqrt(smallest):.1e} in the orthonormal basis",
                RuntimeWarning, stacklevel=2)
        return (U * (1.0 / np.sqrt(w))) @ U.conj().T

    def conjugation_matrix(self):
        r"""Complex conjugation of the orbital basis, :math:`\chi^* = \chi K`.

        Spherical harmonics satisfy :math:`Y_{lm}^* = (-1)^m Y_{l,-m}`, so a basis
        holding every :math:`m` of a shell is closed under conjugation and
        :math:`K` is unitary with :math:`K K^* = 1`.  Returned in the orthonormal
        (Löwdin) basis the integrals are expressed in, or ``None`` when the basis
        is not orthogonalized.  Every operator here is real, so
        :math:`A^* = K^\dagger A K` for each of them.
        """
        if not self.orthogonalize:
            return None
        self.one_body()                                   # samples the basis
        psi = self._engine._psi
        N = (np.conj(psi) @ np.conj(psi).T) * self.grid.dV
        K = np.linalg.solve(self.bare_overlap(), N)       # phi* = phi K
        X = self._lowdin_x()
        return np.linalg.solve(X, K @ X.conj())

    def real_orbitals(self, orbitals, boundaries=()):
        """``orbitals`` made conjugation-real without changing the determinant.

        See :func:`conjugation_real_orbitals`; the orbitals are rotated only
        within the blocks separated by ``boundaries``.
        """
        conjugation = self.conjugation_matrix()
        if conjugation is None:
            return np.asarray(orbitals)
        return conjugation_real_orbitals(orbitals, conjugation, boundaries)

    @property
    def uses_pseudopotentials(self) -> bool:
        """True when the external potential is a sum of pseudopotentials."""
        return bool(self.pseudopotentials)

    def external_potential(self):
        """The callable the engine samples: pseudopotential or bare ``-Z/r``."""
        return (self._potentials.pseudopotential if self.uses_pseudopotentials
                else self._potentials.nuclear_potential)

    def projections(self) -> np.ndarray:
        r"""Projections ``C[mu, p] = <phi_mu|chi_p>`` of the basis on the projectors.

        The only grid work the nonlocal term needs (``(M, P)``, via
        :func:`mandacaru.integrals._backend.kb_projections`, C-accelerated);
        cached.  Also fills :attr:`kb_resolution_ratios` -- each projector's
        grid norm against its exact radial norm, since a projector the grid
        cannot resolve makes the nonlocal energy of its channel meaningless.
        """
        if self._C is None:
            M = len(self.basis)
            if not self.kb_projectors:
                self._C = np.zeros((M, 0), dtype=complex)
                return self._C
            from ..integrals import _backend

            chi = np.stack([p.evaluate(self.grid.X, self.grid.Y,
                                       self.grid.Z).ravel()
                            for p in self.kb_projectors])
            grid_norm = np.real(np.einsum("pg,pg->p", np.conj(chi), chi)) \
                * self.grid.dV
            radial_norm = np.array([_radial_norm(p) for p in self.kb_projectors])
            with np.errstate(divide="ignore", invalid="ignore"):
                self.kb_resolution_ratios = grid_norm / radial_norm
            self._C = _backend.kb_projections(self._engine._psi, chi,
                                              self.grid.dV)
        return self._C

    def nonlocal_coupling_matrix(self) -> np.ndarray:
        r"""The ``(P, P)`` block-diagonal coupling matrix ``D``.

        Assembled from the ``nonlocal_coupling`` blocks keyed by
        ``(atom_index, l, m)``; without them, the Kleinman-Bylander diagonal
        of the projectors' ``kb_energy``.
        """
        return assemble_block_matrix(
            self.kb_projectors, self.nonlocal_coupling,
            diagonal=[p.kb_energy for p in self.kb_projectors])

    @property
    def has_spin_orbit(self) -> bool:
        """Whether a spin-orbit term will be added to the Hamiltonian."""
        return bool(self.spin_orbit_coupling)

    def spin_orbit_matrix(self) -> np.ndarray:
        r"""The ``(2M, 2M)`` spin-orbital matrix of the spin-orbit term.

        Zero (and still ``(2M, 2M)``) when no channel carries one.  See
        :func:`mandacaru.core.spin_orbit.spin_orbit_one_body`; the result is
        complex Hermitian and, unlike every other one-body term here, is
        **not** block-diagonal in spin.
        """
        from .spin_orbit import spin_orbit_one_body

        M = len(self.basis)
        if not self.spin_orbit_coupling:
            return np.zeros((2 * M, 2 * M), dtype=complex)
        return spin_orbit_one_body(self.projections(), self.kb_projectors,
                                   self.spin_orbit_coupling)

    def _spin_orbit_in_mo_basis(self) -> np.ndarray:
        r"""The spin-orbit matrix rotated into the molecular-orbital basis.

        The rotation is spin-independent -- the same spatial
        :attr:`mo_coefficients` for both spins -- so each of the four spin
        quadrants transforms on its own as :math:`C^\dagger h C`.  The
        molecular orbitals themselves come from a Hartree-Fock solution of the
        **scalar** Hamiltonian; that is the usual and correct order, since the
        reference determinant only has to span the space, not diagonalize the
        operator that is about to be added to it.
        """
        C = np.asarray(self.mo_coefficients)
        M = C.shape[0]
        h = self.spin_orbit_matrix()
        out = np.zeros_like(h)
        for sigma in (0, 1):
            for tau in (0, 1):
                block = h[sigma * M:(sigma + 1) * M, tau * M:(tau + 1) * M]
                out[sigma * M:(sigma + 1) * M,
                    tau * M:(tau + 1) * M] = C.conj().T @ block @ C
        return out

    def nonlocal_overlap_matrix(self):
        """The ``(P, P)`` overlap-correction matrix ``Q``, or ``None``."""
        if self.nonlocal_overlap is None:
            return None
        return assemble_block_matrix(self.kb_projectors, self.nonlocal_overlap)

    def kb_nonlocal(self) -> np.ndarray:
        r"""Nonlocal pseudopotential matrix in the basis, general separable form.

        .. math::

            H^{NL} = C\,D\,C^\dagger, \qquad
            C_{\mu p} = \langle\phi_\mu|\chi_p\rangle ,

        with :math:`D` block-diagonal over ``(atom, l, m)``
        (:meth:`nonlocal_coupling_matrix`).  For the Kleinman-Bylander form
        every block is :math:`[E^{KB}_l]` and this reduces to the familiar sum
        of rank-one terms :math:`\sum_p |\chi_p\rangle E^{KB}_p\langle\chi_p|`;
        a family with several radial projectors per channel supplies the
        full block instead.  Only :math:`C` touches the grid; the rest is a
        small matrix product.  Returns zeros when there are no projectors.
        """
        M = len(self.basis)
        if not self.kb_projectors:
            return np.zeros((M, M), dtype=complex)
        C = self.projections()
        D = self.nonlocal_coupling_matrix()
        return C @ D @ C.conj().T

    #: Backward-compatible alias of :meth:`kb_nonlocal`.
    nonlocal_matrix = kb_nonlocal

    def short_range_local(self):
        """Local-potential matrix evaluated *off* the grid, or ``None``.

        A family may integrate part of its local potential more accurately than
        the grid can: :meth:`external_potential` then samples only the
        remainder and this hook supplies the missing ``(M, M)`` block, which is
        added to ``T + V + C D C^dagger`` exactly like
        :meth:`one_body_augmentation`.  PAW-LCAO splits its local channel into the
        long-range potential of a Gaussian ion (smooth, so it stays on the
        grid) plus a short-range remainder of compact support, integrated on an
        atom-centered spherical quadrature that is exactly translation invariant
        -- see :class:`mandacaru.pseudopotentials.paw.PAWIntegrals` and
        :mod:`mandacaru.pseudopotentials.local_split`.  A plain basis samples
        its whole potential on the grid and returns ``None``.

        Whatever returns something here must also change what
        :meth:`external_potential` samples, or the two halves of the potential
        would both be counted.
        """
        return None

    def one_body_augmentation(self):
        """One-body correction added to ``T + V + C D C^dagger``, or ``None``.

        The counterpart of :meth:`two_body_augmentation` for terms that are
        linear in the density.  A plain basis has none; PAW-LCAO uses it for the
        electron-ion attraction of its compensation charges, which the grid
        integral of the external potential cannot see (that integral weights
        only the smooth pair density, and the compensation charge is an extra
        density on top of it).
        """
        return None

    def two_body_augmentation(self):
        r"""Correction added to the grid two-body tensor, or ``None``.

        A hook for families whose pair densities carry more than the product
        of two basis functions -- the PAW-LCAO compensation charges
        (:class:`mandacaru.pseudopotentials.paw.PAWIntegrals`)
        return the ``(M, M, M, M)`` tensor of the extra Coulomb terms in the
        same physicists' layout as :meth:`two_body`.  The plain basis has
        nothing to add.
        """
        return None

    def _compute(self):
        """Build both integral blocks (kept for callers that need both)."""
        self._compute_one_body()
        self._compute_two_body()

    def _compute_one_body(self):
        """The one-body block only -- no electron-repulsion tensor.

        Single-particle work (the Bloch band Hamiltonian, an overlap or kinetic
        inspection) must not pay for the O(M^4 G) ERI contraction.
        """
        if self._h1 is not None:
            return
        T, V = self._engine.one_body(self.external_potential(),
                                     energy_units="Ha", kinetic=self.kinetic)
        self.resolution_ratios = self._engine.resolution(T, kinetic=self.kinetic)
        one = T + V + self.kb_nonlocal()
        short_range = self.short_range_local()
        if short_range is not None:
            # `V` above is then only the long-range half of the local potential
            # (see `short_range_local`); this is the rest of it.
            one = one + np.asarray(short_range)
        augmentation = self.one_body_augmentation()
        if augmentation is not None:
            one = one + np.asarray(augmentation)
        h = 0.5 * (one + one.conj().T)           # symmetrize away grid noise
        if self.orthogonalize:
            X = self._lowdin_x()
            h = X.conj().T @ h @ X
        self._h1 = h

    def _compute_two_body(self):
        """The electron-repulsion tensor only."""
        if self._eri is not None:
            return
        eri = self._engine.two_body(method="fft", energy_units="Ha")
        augmentation = self.two_body_augmentation()
        if augmentation is not None:
            eri = eri + np.asarray(augmentation)
        if self.orthogonalize:
            # Lowdin-orthonormalize the basis; the second-quantized Hamiltonian
            # requires an orthonormal orbital set.
            X = self._lowdin_x()
            # Physicists' <pq|rs> = int p*(1) q*(2) r(1) s(2): the bra indices
            # (p, q) take X* and the ket indices (r, s) take X.  The overlap --
            # hence X -- is complex whenever the basis carries l > 0 functions
            # off a symmetry plane, so the conjugation pattern matters: the
            # former (X*, X, X*, X) broke the tensor's symmetries and shifted
            # water's Hartree-Fock energy by 1.2 Ha.
            eri = np.einsum("ap,bq,cr,ds,abcd->pqrs",
                            X.conj(), X.conj(), X, X, eri, optimize=True)
        self._eri = eri

    def two_body_with_kernel(self, solver, *, mo: bool = True) -> np.ndarray:
        r"""The two-body tensor rebuilt with a **different** Coulomb kernel.

        Everything but the kernel is held fixed -- the same sampled orbitals,
        the same augmentation, the same Loewdin ``X`` and the same molecular
        orbitals ``mo_coefficients`` -- so the result can be subtracted from
        :meth:`two_body` term by term.  That is what the
        exchange-correlation-hole correction of
        :mod:`mandacaru.core.mpc` needs: two tensors that differ only in the
        physics being tested.

        Parameters
        ----------
        solver : object
            A Coulomb solver with ``solve_stack``, ``L`` and ``dV`` -- e.g.
            :class:`~mandacaru.core.mpc.TruncatedCoulombSolver`.
        mo : bool
            Rotate into the molecular-orbital basis the Hamiltonian uses
            (the default).  ``False`` stops after the Loewdin step, which is
            what a caller comparing raw orbital-basis tensors wants.

        Raises
        ------
        ValueError
            With ``mo=True`` before :meth:`molecular_hamiltonian` has run, so
            there is no rotation to reuse.  Asking for a tensor in a basis that
            does not exist yet would otherwise silently return the Loewdin one.
        """
        eri = self._engine.two_body(method="fft", energy_units="Ha",
                                    solver=solver)
        augmentation = self.two_body_augmentation()
        if augmentation is not None:
            eri = eri + np.asarray(augmentation)
        if self.orthogonalize:
            X = self._lowdin_x()
            # The same conjugation pattern as _compute_two_body: bra indices
            # (p, q) take X*, ket indices (r, s) take X.
            eri = np.einsum("ap,bq,cr,ds,abcd->pqrs",
                            X.conj(), X.conj(), X, X, eri, optimize=True)
        if not mo:
            return eri
        V = self.mo_coefficients
        if V is None:
            raise ValueError(
                "no molecular-orbital rotation is available yet: call "
                "molecular_hamiltonian() first, or pass mo=False to get the "
                "tensor in the Loewdin-orthonormalized basis.")
        return np.einsum("ap,bq,cr,ds,abcd->pqrs",
                         V.conj(), V.conj(), V, V, eri, optimize=True)

    def unresolved(self, tolerance: float = RESOLUTION_TOLERANCE):
        """Indices of basis functions and projectors the grid does not resolve.

        Returns ``(functions, projectors)``: the basis-function indices whose
        kinetic-energy ratio :attr:`resolution_ratios` and the projector
        indices whose norm ratio :attr:`kb_resolution_ratios` fall outside
        ``1 +/- tolerance``.  Runs the integrals if they have not been run.
        """
        if self.resolution_ratios is None:
            self._compute_one_body()
        def outside(ratios):
            if ratios is None:
                return []
            ratios = np.asarray(ratios, dtype=float)
            bad = np.isfinite(ratios) & (np.abs(ratios - 1.0) > tolerance)
            return [int(i) for i in np.nonzero(bad)[0]]
        return outside(self.resolution_ratios), outside(self.kb_resolution_ratios)

    # -- spatial integrals (Hartree) -------------------------------------- #

    def one_body(self) -> np.ndarray:
        r"""Spatial one-body core Hamiltonian ``h_pq = T_pq + V_pq`` (Hartree).

        In the orthonormalized orbital basis when ``orthogonalize=True``.
        """
        if self._h1 is None:
            self._compute_one_body()
        return self._h1

    def two_body(self) -> np.ndarray:
        r"""Spatial two-body tensor ``<pq|rs>`` in physicists' notation (Hartree)."""
        if self._eri is None:
            self._compute_two_body()
        return self._eri

    @property
    def nuclear_repulsion(self) -> float:
        r"""Nuclear repulsion energy ``sum_{I<J} Z_I Z_J/|R_I-R_J|`` (Hartree)."""
        e = 0.0
        pos = [to_bohr(R, self.units) for _Z, R in self.nuclei]
        Zs = [Z for Z, _R in self.nuclei]
        for i in range(len(self.nuclei)):
            for j in range(i + 1, len(self.nuclei)):
                e += Zs[i] * Zs[j] / np.linalg.norm(pos[i] - pos[j])
        return float(e)

    # -- spin-orbital integrals ------------------------------------------- #

    def spin_orbital_integrals(self, spin_orbit: bool = True
                               ) -> tuple[np.ndarray, np.ndarray]:
        r"""Spin-orbital ``(h_so, g_so)`` for the Hamiltonian (spin-blocked).

        Spin-orbital ``P = p + sigma * M`` (``M`` spatial orbitals; ``sigma = 0``
        alpha for the first block, ``1`` beta for the second).  ``g_so`` is the
        two-electron integral in physicists' notation ``<PQ|RS>``, non-zero only
        when ``spin(P) == spin(R)`` (electron 1) and ``spin(Q) == spin(S)``
        (electron 2).  The returned tensors feed :meth:`Fermion.from_integrals`
        directly and yield a Hermitian, spin- and particle-number-conserving
        Hamiltonian -- unless a spin-orbit term is present and ``spin_orbit``
        is left on, in which case it conserves particle number and
        :math:`J_z` but not :math:`S_z`.
        """
        h_spin = (self.spin_orbit_matrix()
                  if spin_orbit and self.spin_orbit_coupling else None)
        return spin_block_integrals(self.one_body(), self.two_body(), h_spin)

    # -- molecular Hamiltonian -------------------------------------------- #

    def molecular_hamiltonian(self, include_nuclear_repulsion: bool = True,
                              mo_basis: bool = False,
                              n_electrons: int | None = None,
                              frozen_orbitals=None, num_particles=None,
                              open_shell: bool | None = None,
                              active_orbitals=None,
                              active_selection: str = "energy",
                              active_threshold=None) -> Fermion:
        """Assemble the second-quantized :class:`Fermion` Hamiltonian.

        Spin-orbitals are ordered alpha-block then beta-block, so the parity
        mapping's two-qubit reduction (which taper the alpha- and total-parity
        qubits) applies directly.

        With ``mo_basis=True`` the spatial integrals are first transformed to a
        Hartree-Fock molecular-orbital basis (``n_electrons`` required), so the
        reference determinant is the HF ground state -- the basis expected by
        ADAPT-VQE and by variational algorithms in general.  For an **even**
        electron count that is the closed-shell RHF basis.  For an **odd** count
        -- or whenever ``open_shell=True`` -- it is the **natural-orbital basis
        of the unrestricted (UHF) solution** for ``num_particles =
        (n_alpha, n_beta)`` (default: the lowest spin state, one unpaired
        electron), one spatial basis shared by both spins, so the Hamiltonian
        keeps the same alpha/beta-block form; see
        :mod:`mandacaru.algorithms.hartree_fock`.  ``open_shell=False`` forces
        RHF (and rejects an odd count).

        ``frozen_orbitals`` applies the **frozen-core approximation**: the given
        (doubly occupied) spatial MO indices are removed from the active space and
        replaced by their mean-field contribution -- a constant core energy plus an
        effective one-body potential on the remaining orbitals (see
        :func:`freeze_core_integrals`).  It requires ``mo_basis=True``; the returned
        Hamiltonian acts only on the active spin-orbitals.

        ``active_orbitals`` truncates the **virtual** space as well, which is
        what makes a large basis affordable on a qubit register: a virtual
        orbital that is dropped needs no mean field folded in, because it is
        empty in the reference, so it costs exactly the correlation it would
        have carried.  ``active_selection`` chooses how the virtuals are ranked
        (``"energy"``, ``"mp2"``, ``"natural"``).  Both are resolved by
        :func:`~mandacaru.algorithms.active_space.resolve_active_space`, whose
        result is left on :attr:`active_space` for the density and the run log.
        """
        frozen = sorted({int(i) for i in frozen_orbitals}) if frozen_orbitals \
            else []
        core_energy = 0.0
        self.active_space = None
        if mo_basis:
            h_mo, eri_mo, self.mo_coefficients = molecular_orbital_integrals(
                self, n_electrons, num_particles, open_shell)
            space = self._resolve_active_space(
                h_mo, eri_mo, n_electrons, num_particles, open_shell,
                frozen, active_orbitals, active_selection, active_threshold)
            if space is not None and space.rotation is not None:
                from ..algorithms.mp2 import rotate_integrals

                h_mo, eri_mo = rotate_integrals(h_mo, eri_mo, space.rotation)
                h_mo, eri_mo = np.real_if_close(h_mo), np.real_if_close(eri_mo)
                self.mo_coefficients = np.asarray(
                    self.mo_coefficients) @ space.rotation
            if space is not None:
                frozen, active = list(space.frozen), list(space.active)
            else:
                active = [p for p in range(self.n_orbitals) if p not in frozen]
            if frozen or len(active) != self.n_orbitals:
                h_mo, eri_mo, core_energy = freeze_core_integrals(
                    h_mo, eri_mo, frozen, active)
            h_spin = None
            if self.spin_orbit_coupling:
                _refuse_spin_orbit_without_sz(num_particles, frozen,
                                              deleted=getattr(space, "deleted",
                                                              ()))
                h_spin = self._spin_orbit_in_mo_basis()
            h_so, g_so = spin_block_integrals(h_mo, eri_mo, h_spin)
        else:
            if frozen:
                raise ValueError(
                    "frozen_orbitals requires mo_basis=True (the frozen-core "
                    "approximation freezes canonical molecular orbitals)")
            if active_orbitals is not None or active_threshold is not None:
                raise ValueError(
                    "active_orbitals requires mo_basis=True: an active space is "
                    "a choice among molecular orbitals, and the atomic-orbital "
                    "Hamiltonian has none to choose from")
            h_so, g_so = self.spin_orbital_integrals()
        H = Fermion.from_integrals(h_so, g_so)
        const = core_energy + self.constant_energy + (
            self.nuclear_repulsion if include_nuclear_repulsion else 0.0)
        if mo_basis and space is not None and space.deleted:
            # Reuse the *reduced* integrals for finite-difference forces.  They
            # already contain the frozen-core potential and the selected virtual
            # space, before mapping and optional Z2 tapering.
            self.active_h_so = h_so
            self.active_g_so = g_so
            self.active_constant = float(np.real(const))
        if abs(const) > 1e-14:
            H = H + Fermion({(): complex(const)}, n_modes=h_so.shape[0])
        return H

    def _resolve_active_space(self, h_mo, eri_mo, n_electrons, num_particles,
                              open_shell, frozen, active_orbitals,
                              active_selection, active_threshold=None):
        """The :class:`ActiveSpace` this Hamiltonian is built in, or ``None``.

        ``None`` only when nothing is frozen and nothing is truncated, so the
        old path (the whole orbital set, no partition object) is untouched.
        """
        from ..algorithms.active_space import resolve_active_space

        if active_orbitals is None and active_threshold is None and not frozen:
            return None
        n_el, n_alpha, n_beta, open_shell = resolve_reference(
            n_electrons, num_particles, open_shell)
        occupations = None
        if ((active_orbitals is not None or active_threshold is not None)
                and str(active_selection).strip().lower() == "natural"):
            # Only this selector needs them, and it needs the open-shell
            # reference, so the solve is paid for exactly when it is used.
            occupations = np.real(np.asarray(
                self.open_shell_hartree_fock(n_alpha, n_beta)
                .natural_occupations, dtype=float))
        space = resolve_active_space(
            h_mo, eri_mo, n_orbitals=self.n_orbitals,
            num_particles=(n_alpha, n_beta), active_orbitals=active_orbitals,
            selection=active_selection, frozen=frozen,
            reference_occupations=occupations, open_shell=open_shell,
            threshold=active_threshold)
        self.active_space = space
        return space

    def hartree_fock_hamiltonian(self, n_electrons: int,
                                 include_nuclear_repulsion: bool = True) -> Fermion:
        r"""Reduce the molecular Hamiltonian to its Hartree-Fock (mean-field) form.

        Returns the diagonal one-body :class:`Fermion`

        .. math::

            H_{\mathrm{HF}} = \sum_P \varepsilon_P\, a^\dagger_P a_P + c ,

        where :math:`\varepsilon_P` are the (spin-orbital) RHF orbital energies and
        the constant :math:`c` is fixed so that, in the ``n_electrons`` sector, the
        aufbau (Hartree-Fock) determinant is the ground state with exactly the RHF
        total energy.  This is the mean-field Hamiltonian whose ground state is the
        HF determinant -- e.g. a cheap reference for the variational drivers.
        """
        rhf = self.hartree_fock(n_electrons)
        eps = np.real(np.asarray(rhf.mo_energies, dtype=float))
        M = len(eps)
        eps_so = np.concatenate([eps, eps])          # alpha block, then beta block
        n_so = 2 * M
        n_occ = n_electrons // 2
        occupied = list(range(n_occ)) + list(range(M, M + n_occ))

        terms = {((P, True), (P, False)): complex(eps_so[P]) for P in range(n_so)}
        H = Fermion(terms, n_modes=n_so)

        e_reference = float(sum(eps_so[P] for P in occupied))
        e_hf = rhf.electronic_energy + self.constant_energy + (
            self.nuclear_repulsion if include_nuclear_repulsion else 0.0)
        const = complex(e_hf - e_reference)
        if abs(const) > 1e-14:
            H = H + Fermion({(): const}, n_modes=n_so)
        return H


def _refuse_spin_orbit_without_sz(num_particles, frozen_orbitals,
                                  deleted=()) -> None:
    r"""Refuse the machinery that assumes ``S_z`` is a good quantum number.

    Spin-orbit coupling gives the one-body matrix an alpha-beta block, so the
    Hamiltonian conserves :math:`J_z` and particle number but **not**
    :math:`S_z`.  Everything downstream that was written against a fixed
    ``(n_alpha, n_beta)`` -- the particle-number sector reduction
    (:mod:`mandacaru.core.sector`), the parity mapping's two-qubit taper, and
    every excitation pool that pairs an alpha excitation with a beta one --
    would then be reducing or exciting in a space the Hamiltonian does not
    preserve, and would return a number that looks plausible.

    Nothing ships a Dirac dataset today, so no ordinary run reaches this.  It
    is here so that the day one does, it fails instead of being believed.
    """
    if num_particles is not None:
        raise NotImplementedError(
            "spin-orbit coupling and an explicit num_particles are "
            "incompatible: the alpha-beta block of the one-body matrix means "
            "S_z is not conserved, so there is no (n_alpha, n_beta) sector "
            "for the Hamiltonian to act within -- only a total particle "
            "number.  Drop num_particles, or generate the pseudopotential "
            "with relativity='scalar', which carries the mass-velocity and "
            "Darwin terms without the spin-orbit one.")
    if frozen_orbitals:
        raise NotImplementedError(
            "a frozen core and spin-orbit coupling are incompatible: "
            "freezing replaces doubly occupied spatial orbitals by a mean "
            "field, and the spin-orbit term is what stops the two spins of "
            "an orbital from being occupied together.")
    if len(deleted):
        raise NotImplementedError(
            "a truncated virtual space and spin-orbit coupling are "
            "incompatible: the selector ranks spatial orbitals, and with "
            "spin-orbit coupling the two spins of a spatial orbital are not "
            "degenerate partners -- keeping or deleting them together is not "
            "a choice the Hamiltonian permits.")


def projector_blocks(projectors) -> dict:
    """Group projector positions by channel: ``{(atom, l, m): [p, ...]}``.

    Positions within a block are ordered by the projector's radial ``index``
    (0 for a single Kleinman-Bylander projector); a duplicate index inside a
    channel is an error, since the block's rows would be ambiguous.
    """
    groups: dict = {}
    for position, projector in enumerate(projectors):
        key = (int(projector.atom_index), *(int(v) for v in projector.channel))
        groups.setdefault(key, []).append(
            (int(getattr(projector, "index", 0)), position))
    ordered = {}
    for key, entries in groups.items():
        indices = [i for i, _p in entries]
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"projectors of channel (atom, l, m)={key} carry duplicate "
                f"radial indices {sorted(indices)}")
        ordered[key] = [p for _i, p in sorted(entries)]
    return ordered


def assemble_block_matrix(projectors, blocks, diagonal=None) -> np.ndarray:
    r"""Assemble a ``(P, P)`` block-diagonal matrix over the projector channels.

    ``blocks`` maps ``(atom_index, l, m)`` to an ``(n, n)`` array for the ``n``
    radial projectors of that channel (every channel must be supplied, and no
    key may name a channel without projectors).  With ``blocks=None`` the
    matrix is ``diag(diagonal)`` -- the Kleinman-Bylander case of one energy
    per projector.
    """
    P = len(projectors)
    out = np.zeros((P, P), dtype=complex)
    if blocks is None:
        if diagonal is None:
            raise ValueError("either blocks or a diagonal must be given")
        diagonal = np.asarray(list(diagonal), dtype=complex)
        if diagonal.shape != (P,):
            raise ValueError(f"expected {P} diagonal entries, got "
                             f"{diagonal.shape}")
        out[np.diag_indices(P)] = diagonal
        return out
    groups = projector_blocks(projectors)
    normalized = {tuple(int(v) for v in key): value
                  for key, value in blocks.items()}
    missing = sorted(set(groups) - set(normalized))
    extra = sorted(set(normalized) - set(groups))
    if missing or extra:
        raise ValueError(
            "nonlocal blocks do not match the projector channels: "
            f"missing {missing}, unmatched {extra}")
    for key, positions in groups.items():
        block = np.asarray(normalized[key], dtype=complex)
        n = len(positions)
        if block.shape != (n, n):
            raise ValueError(
                f"block for channel {key} must be ({n}, {n}), got {block.shape}")
        out[np.ix_(positions, positions)] = block
    return out


def resolve_reference(n_electrons, num_particles=None, open_shell=None):
    """``(n_el, n_alpha, n_beta, open_shell)`` of a mean-field reference.

    ``num_particles`` defaults to the lowest spin state (one unpaired electron
    for an odd count) and ``open_shell`` to "any unpaired electron", i.e. an odd
    count **or** ``n_alpha != n_beta``: a triplet O2 has an even count and still
    needs the UHF natural orbitals, because RHF orbitals are optimized for the
    singlet and put the (7, 5) determinant eV above its open-shell reference.
    ``open_shell=False`` with an odd count is refused.  The one place this is
    decided, for every integral class.
    """
    if n_electrons is None:
        raise ValueError("mo_basis=True requires n_electrons")
    n_el = int(n_electrons)
    if num_particles is None:
        n_unpaired = n_el % 2
        num_particles = ((n_el + n_unpaired) // 2, (n_el - n_unpaired) // 2)
    na, nb = (int(v) for v in num_particles)
    if na + nb != n_el:
        raise ValueError(
            f"num_particles {num_particles} does not sum to "
            f"n_electrons={n_el}")
    if open_shell is None:
        open_shell = n_el % 2 == 1 or na != nb
    if not open_shell and n_el % 2:
        raise ValueError(
            "open_shell=False (closed-shell RHF) needs an even "
            f"electron count; got {n_el}")
    return n_el, na, nb, bool(open_shell)


def molecular_orbital_integrals(integrals, n_electrons, num_particles=None,
                                open_shell=None):
    """``(h_mo, eri_mo, orbitals)`` in the conjugation-real mean-field basis.

    ``integrals`` is any integral class offering ``hartree_fock``,
    ``open_shell_hartree_fock``, ``real_orbitals``, ``one_body`` and
    ``two_body`` (:class:`MolecularIntegrals`,
    :class:`~mandacaru.core.planewave.PlaneWaveIntegrals`).  The basis is
    closed-shell RHF, or the UHF natural orbitals for an open shell
    (:func:`resolve_reference`).

    Complex orbitals (``l > 0`` harmonics, plane waves) leave the SCF with
    arbitrary phases and degenerate-pair mixing, which makes the MO Hamiltonian
    complex while every operator pool is real: ADAPT then stalls above the
    ground state (62 meV on H2 PAW-LCAO-DZP, 39 meV on H2 in plane waves).  The
    orbitals are therefore rotated to conjugation-real form inside the
    occupied / virtual blocks -- same determinant, real Hamiltonian.
    """
    n_el, na, nb, open_shell = resolve_reference(n_electrons, num_particles,
                                                 open_shell)
    if open_shell:
        uhf = integrals.open_shell_hartree_fock(na, nb)
        h_mo, eri_mo = uhf.h_mo, uhf.eri_mo
        orbitals, blocks = uhf.natural_orbitals, (min(na, nb), max(na, nb))
    else:
        rhf = integrals.hartree_fock(n_el)
        h_mo, eri_mo = rhf.h_mo, rhf.eri_mo
        orbitals, blocks = rhf.mo_coefficients, (n_el // 2,)
    real = integrals.real_orbitals(orbitals, blocks)
    if real is not orbitals:
        from ..algorithms.hartree_fock import transform_integrals
        h_mo, eri_mo = transform_integrals(integrals.one_body(),
                                           integrals.two_body(), real)
        h_mo, eri_mo = np.real_if_close(h_mo), np.real_if_close(eri_mo)
    return h_mo, eri_mo, real


#: Largest residual (unitarity, block coupling, reality) :func:`conjugation_real_orbitals` accepts.
REALITY_TOLERANCE = 1e-8


def conjugation_real_orbitals(orbitals, conjugation, boundaries=(),
                              tolerance: float = REALITY_TOLERANCE):
    r"""Rotate orthonormal orbitals so each is invariant under conjugation.

    An orbital with coefficients :math:`v` (in a basis whose conjugation is
    :math:`K`, see :meth:`MolecularIntegrals.conjugation_matrix`) is *real* when
    :math:`K v^* = v`.  In a basis of real orbitals every integral of a real
    operator is real, so the qubit Hamiltonian is real and its ground state has
    real amplitudes -- reachable by the real excitation generators of the
    operator pools.  For orthonormal orbitals :math:`V` spanning a
    conjugation-invariant space, :math:`B = V^\dagger K V^*` is unitary and
    symmetric, its principal square root :math:`W` satisfies :math:`W W^T = B`,
    and :math:`V W` is real: the phase of each non-degenerate orbital is fixed
    and degenerate partners are recombined (:math:`p_{\pm 1}` into
    :math:`p_x, p_y`).

    The rotation stays inside the blocks separated by ``boundaries`` (e.g. the
    occupied / virtual split), so the reference determinant and the orbital
    energies are unchanged.  Returns ``orbitals`` itself (the same object)
    whenever the construction does not apply -- a basis not closed under
    conjugation, or a block that is not conjugation-invariant.
    """
    from scipy.linalg import sqrtm

    V = np.asarray(orbitals)
    K = np.asarray(conjugation, dtype=complex)
    n = V.shape[1]
    if V.shape[0] != K.shape[0] or \
            np.abs(K @ K.conj() - np.eye(K.shape[0])).max() > tolerance:
        return orbitals
    B = V.conj().T @ K @ V.conj()
    if np.abs(B @ B.conj().T - np.eye(n)).max() > tolerance:
        return orbitals
    edges = [0] + sorted({int(b) for b in boundaries if 0 < int(b) < n}) + [n]
    W = np.zeros((n, n), dtype=complex)
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = np.zeros(n, dtype=bool)
        inside[lo:hi] = True
        if np.abs(B[np.ix_(inside, ~inside)]).max(initial=0.0) > tolerance:
            return orbitals
        block = B[lo:hi, lo:hi]
        W[lo:hi, lo:hi] = sqrtm(0.5 * (block + block.T))
    real = V @ W
    if np.abs(K @ real.conj() - real).max() > tolerance or \
            np.abs(real.conj().T @ real - V.conj().T @ V).max() > tolerance:
        return orbitals
    return real


def spin_block_integrals(h: np.ndarray, eri: np.ndarray,
                         h_spin: np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    r"""Expand spatial integrals ``(h, <pq|rs>)`` to spin-orbitals (spin-blocked).

    Spin-orbital ``P = p + sigma * M`` (``sigma = 0`` alpha for the first ``M``,
    ``1`` beta for the second).  The one-body block is diagonal in spin; the
    physicists'-notation two-body tensor is non-zero only when
    ``spin(P) == spin(R)`` (electron 1) and ``spin(Q) == spin(S)`` (electron 2).

    ``h_spin`` is an already-spin-orbital ``(2M, 2M)`` term **added after** the
    spin-blocking, for an operator that does not commute with :math:`S_z` --
    which in this code means spin-orbit coupling and nothing else
    (:meth:`MolecularIntegrals.spin_orbit_matrix`).  It is the one way the
    one-body matrix acquires an alpha-beta block.
    """
    h = np.asarray(h)
    eri = np.asarray(eri)
    M = h.shape[0]
    n_so = 2 * M

    def spin(P):
        return P // M

    def orb(P):
        return P % M

    h_so = np.zeros((n_so, n_so), dtype=complex)
    for P in range(n_so):
        for Q in range(n_so):
            if spin(P) == spin(Q):
                h_so[P, Q] = h[orb(P), orb(Q)]

    g_so = np.zeros((n_so,) * 4, dtype=complex)
    for P in range(n_so):
        for Q in range(n_so):
            for R in range(n_so):
                for S in range(n_so):
                    if spin(P) == spin(R) and spin(Q) == spin(S):
                        g_so[P, Q, R, S] = eri[orb(P), orb(Q), orb(R), orb(S)]
    if h_spin is not None:
        h_spin = np.asarray(h_spin)
        if h_spin.shape != (n_so, n_so):
            raise ValueError(
                f"h_spin must be ({n_so}, {n_so}) for {M} spatial orbitals, "
                f"got {h_spin.shape}")
        h_so = h_so + h_spin
    return h_so, g_so


def freeze_core_integrals(h_mo: np.ndarray, eri_mo: np.ndarray,
                          frozen: Sequence[int], active: Sequence[int]
                          ) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Frozen-core reduction of the MO-basis spatial integrals.

    Given the molecular-orbital one-body ``h_mo`` and physicists'-notation
    two-body ``<pq|rs>`` (``eri_mo``), a set of doubly occupied ``frozen`` (core)
    spatial orbitals and the complementary ``active`` orbitals, returns
    ``(h_active, eri_active, core_energy)`` for the reduced active-space
    Hamiltonian:

    .. math::

        E_{\text{core}} &= 2\sum_{i}h_{ii}
            + \sum_{ij}\bigl(2\langle ij|ij\rangle-\langle ij|ji\rangle\bigr), \\
        h^{\text{eff}}_{pq} &= h_{pq}
            + \sum_{i}\bigl(2\langle pi|qi\rangle-\langle pi|iq\rangle\bigr),

    with ``i, j`` over ``frozen`` and ``p, q`` over ``active``; ``eri_active`` is
    the ``active`` sub-block of ``eri_mo``.  Adding ``core_energy`` as a constant
    and using ``(h_active, eri_active)`` in the active space reproduces the full
    energy exactly for a determinant that keeps every frozen orbital doubly
    occupied (the frozen-core approximation).
    """
    h_mo = np.asarray(h_mo)
    eri_mo = np.asarray(eri_mo)
    frozen = list(frozen)
    active = list(active)

    core_energy = 0.0
    for i in frozen:
        core_energy += 2.0 * h_mo[i, i]
    for i in frozen:
        for j in frozen:
            core_energy += 2.0 * eri_mo[i, j, i, j] - eri_mo[i, j, j, i]

    n_act = len(active)
    h_eff = np.zeros((n_act, n_act), dtype=complex)
    for a, p in enumerate(active):
        for b, q in enumerate(active):
            val = h_mo[p, q]
            for i in frozen:
                val += 2.0 * eri_mo[p, i, q, i] - eri_mo[p, i, i, q]
            h_eff[a, b] = val

    eri_active = eri_mo[np.ix_(active, active, active, active)]
    return h_eff, eri_active, float(np.real(core_energy))


def minimal_hao_basis(nuclei, grid_units: str = "angstrom"):
    """Build one Slater-screened HAO (Hydrogenic Atomic Orbital) 1s orbital per atom (minimal basis).

    ``nuclei`` is a sequence of ``(Z, position)``; returns a list of
    :class:`~mandacaru.basis.HydrogenicAtomicOrbital`, one 1s per center with the Slater
    effective charge for that atom's 1s.
    """
    basis = []
    for Z, R in nuclei:
        z_eff = HydrogenicAtomicOrbital.slater_effective_charge(int(round(Z)), 1, 0)
        basis.append(HydrogenicAtomicOrbital(1, 0, 0, Z=z_eff, center=R,
                                       units=grid_units))
    return basis
