# -*- coding: utf-8 -*-
# file: core/ccmh.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Finite-size correction from the structure factor (Chiesa *et al.*).

A periodic calculation evaluates the electron-electron energy as a **discrete
sum** over the reciprocal lattice of the supercell,

.. math::

    V_N = \frac{1}{2\Omega} \sum_{\mathbf k \neq 0}
          v(\mathbf k)\,\bigl[S(\mathbf k) - 1\bigr] ,
    \qquad v(\mathbf k) = \frac{4\pi}{k^2},

where the thermodynamic limit wants the **integral**
:math:`\tfrac12 \int \frac{d^3k}{(2\pi)^3} v(k)[S(k)-1]`.  For a smooth
integrand the two agree to exponential accuracy -- except that the sum has no
:math:`\mathbf k = 0` term.  Each k-point of the supercell carries reciprocal
volume :math:`(2\pi)^3/\Omega`, so the piece the sum omits is

.. math::

    \Delta V = \frac{1}{2\Omega}\,
      \lim_{\mathbf k \to 0} \bigl[v(\mathbf k) S(\mathbf k)\bigr] .
    \label{eq-ccmh}

That limit is finite, which is the whole point: :math:`S(\mathbf k)` vanishes
as :math:`k^2` at small :math:`k` -- the long-wavelength density fluctuations a
Coulomb system suppresses -- so :math:`v(k)S(k) \to 4\pi \lim_{k\to0} S(k)/k^2`
is a constant, and the finite-size error is one missing constant rather than a
divergence.  This is the Chiesa-Ceperley-Martin-Holzmann correction.

**No fitted parameters enter.** Chiesa *et al.* evaluate the limit with the
random-phase form :math:`S(k) \to k^2 / (2\omega_p)`,
:math:`\omega_p = \sqrt{4\pi n}`, which gives
:math:`\Delta V = \pi / (\Omega\,\omega_p)`.  Here the limit is taken from the
**computed** :math:`S(\mathbf k)` instead, by extrapolating the smallest
reciprocal-lattice shells, and the RPA value is reported alongside as a
diagnostic rather than used.  A system that is not well described by the RPA --
which is most of what a correlated solver is run on -- then gets its own
answer, and the gap between the two is visible.

This is deliberately **not** the model-periodic-Coulomb (MPC) route of
:mod:`mandacaru.core.mpc`.  MPC swaps the interaction kernel, which for this
code double-counts: measured on a simple-cubic hydrogen lattice, the MPC hole
correction came out at 86 % (1x1x1) and 92 % (2x2x2) of the Madelung term
``constant_energy`` already subtracts, and it scaled as
:math:`\Omega^{-1/3}` rather than :math:`1/\Omega` -- the signature of a
Madelung-like term.  The correction here is additive and orthogonal to that
constant.

The structure factor
--------------------

:math:`S(\mathbf k)` is a property of the pair density, so it comes from the
two-body RDM.  With :math:`\hat\rho_{\mathbf k} = \sum_i e^{-i\mathbf k \cdot
\mathbf r_i} = \sum_{pq} M_{pq}(\mathbf k)\, a^\dagger_p a_q` and
:math:`a^\dagger_p a_q a^\dagger_r a_s = a^\dagger_p a^\dagger_r a_s a_q
+ \delta_{qr} a^\dagger_p a_s`,

.. math::

    N S(\mathbf k) = \sum_{pqrs} M_{pq} M^{*}_{rs}\,\Gamma_{prsq}
                   + \sum_{pqs} M_{pq} M^{*}_{qs}\, D_{ps} ,

the second term being the exchange-like contraction the operator ordering
leaves behind.  The one-body matrix elements
:math:`M_{pq}(\mathbf k) = \int \phi^{*}_p e^{-i\mathbf k \cdot \mathbf r}
\phi_q` are the Fourier coefficients of the pair densities, which on a
periodic grid are exactly a DFT -- so they cost one FFT per pair and need no
new quadrature.  ``M(0)`` is the overlap, which in the orthonormal basis the
Hamiltonian uses is the identity; that is the cheapest available check that
the transform chain is right.  The contraction itself was verified against
direct operator algebra -- building :math:`\hat\rho_{\mathbf k}` as a
``Fermion``, mapping it to qubits and taking
:math:`\|\hat\rho^{\dagger}_{\mathbf k}|\Psi\rangle\|^2 / N` -- which
agrees to ``0.0``.

**The computed ``S(k)`` is basis-limited at large ``k``.**  For point particles
:math:`S(\mathbf k) \to 1` as :math:`k \to \infty`; in a finite basis the
matrix elements :math:`M_{pq}(\mathbf k)` decay once :math:`k` exceeds the
inverse orbital size, and :math:`S` decays with them -- measured at ``0.105``
at :math:`k = 1.91\,a_0^{-1}` for a minimal two-orbital basis.  That does not
affect the correction, which reads only the *small*-``k`` end, where long
wavelengths are exactly what a localized basis does represent.  It does mean
``S(k)`` from this routine is not a substitute for a converged structure
factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft

__all__ = ["StructureFactor", "CCMHCorrection", "structure_factor",
           "ccmh_correction"]

#: Reciprocal-lattice shells used to extrapolate ``S(k)/k^2`` to ``k = 0``.
#: The fit needs the small-k region where ``S ~ k^2`` holds; too many shells
#: and the quartic term contaminates it, too few and there is nothing to fit.
DEFAULT_SHELLS = 3

#: ``|m|`` of the integer reciprocal-lattice triples enumerated.  Shell 3 of a
#: cubic lattice already needs ``|m| = 2``, so this is comfortable.
GVECTOR_RANGE = 3

#: Largest relative spread of ``S(k)/k^2`` across the fitted shells that still
#: counts as "inside the quadratic regime".  The correction rests entirely on
#: ``S -> k^2`` holding where the smallest reciprocal-lattice vectors sit; if
#: the ratio is still varying strongly the extrapolation is reading a trend
#: that has not started, and the intercept means nothing.
QUADRATIC_REGIME_SPREAD = 0.5


@dataclass
class StructureFactor:
    """``S(k)`` of the correlated state at the supercell's reciprocal lattice.

    Attributes
    ----------
    kvectors : ndarray, shape (nk, 3)
        Cartesian reciprocal vectors in Bohr^-1, excluding ``k = 0``.
    values : ndarray, shape (nk,)
        ``S(k)``, real and non-negative for a physical state.
    magnitudes : ndarray, shape (nk,)
        ``|k|`` in Bohr^-1.
    n_electrons : float
        ``tr D``, the normalization used.
    overlap_error : float
        ``max |M(0) - I|``.  ``M(0)`` is the overlap in the basis the RDMs live
        in, which is orthonormal, so this must be at round-off; anything larger
        means the AO-to-MO chain used here does not match the Hamiltonian's.
    """

    kvectors: np.ndarray
    values: np.ndarray
    magnitudes: np.ndarray
    n_electrons: float
    overlap_error: float

    def shells(self, count: int = DEFAULT_SHELLS):
        """``(|k|, S)`` averaged over the ``count`` smallest shells.

        Symmetry-equivalent vectors carry the same ``S`` for a symmetric
        state, so averaging a shell both reduces noise and makes the
        extrapolation isotropic.
        """
        if self.magnitudes.size == 0:
            return np.empty(0), np.empty(0)
        order = np.argsort(self.magnitudes)
        k = self.magnitudes[order]
        s = self.values[order]
        edges = np.flatnonzero(np.diff(k) > 1e-8 * max(k[-1], 1.0)) + 1
        groups = np.split(np.arange(len(k)), edges)[:int(count)]
        return (np.array([k[g].mean() for g in groups]),
                np.array([s[g].mean() for g in groups]))


@dataclass
class CCMHCorrection:
    r"""The leading finite-size error of the electron-electron energy.

    Attributes
    ----------
    correction : float
        :math:`\Delta V` of Eq. (ccmh), Hartree **per supercell**.  Add it to
        the supercell energy.
    correction_per_cell : float
        The same divided by :attr:`n_cells`, to add to a per-cell energy.
    limit : float
        :math:`\lim_{k \to 0} S(k)/k^2` in Bohr^2, extrapolated from the
        computed shells.
    rpa_limit : float
        :math:`1/(2\omega_p)`, the random-phase value, for comparison only.
    plasma_frequency : float
        :math:`\omega_p = \sqrt{4\pi n}` in Hartree.
    density : float
        :math:`n = N/\Omega` in Bohr^-3.
    volume : float
        Supercell volume in Bohr^3.
    n_cells : int
        Primitive cells in the supercell.
    quadratic_spread : float
        Relative spread of ``S(k)/k^2`` over the fitted shells.  Small means
        the quadratic regime is resolved and the limit is meaningful; see
        :attr:`reliable`.
    n_electrons : float
        Electrons in the supercell.
    energy_per_cell : float or None
        The uncorrected per-cell energy in Hartree, when supplied.
    structure : StructureFactor or None
        The structure factor the limit was taken from.
    """

    correction: float
    limit: float
    rpa_limit: float
    plasma_frequency: float
    density: float
    volume: float
    n_cells: int
    n_electrons: float
    quadratic_spread: float = 0.0
    energy_per_cell: float | None = None
    structure: StructureFactor | None = field(default=None, repr=False)

    @property
    def reliable(self) -> bool:
        """Whether ``S(k)/k^2`` was flat enough for the limit to mean anything."""
        return self.quadratic_spread <= QUADRATIC_REGIME_SPREAD

    @property
    def correction_per_cell(self) -> float:
        """Hartree to add to a per-primitive-cell energy."""
        return self.correction / self.n_cells

    @property
    def corrected_energy_per_cell(self) -> float | None:
        """The finite-size-corrected energy per primitive cell, in Hartree."""
        if self.energy_per_cell is None:
            return None
        return self.energy_per_cell + self.correction_per_cell

    @property
    def rpa_correction(self) -> float:
        r""":math:`\pi / (\Omega \omega_p)` -- what the RPA limit would give."""
        return 4.0 * np.pi * self.rpa_limit / (2.0 * self.volume)

    def summary(self) -> str:
        """One line, in eV."""
        from ..units import from_hartree

        return (f"CCMH finite-size correction: "
                f"{from_hartree(self.correction_per_cell, 'eV'):+.6f} eV/cell "
                f"(lim S/k^2 = {self.limit:.5f} Bohr^2 against RPA "
                f"{self.rpa_limit:.5f}; omega_p = "
                f"{from_hartree(self.plasma_frequency, 'eV'):.3f} eV, "
                f"n = {self.density:.6f} Bohr^-3)")


def _reciprocal_vectors(cell, limit: int = GVECTOR_RANGE):
    """Integer triples and their Cartesian reciprocal vectors, without ``0``.

    ``cell`` holds the lattice vectors as rows, so the reciprocal vectors as
    rows are ``2 pi inv(cell).T`` and ``k = m @ B``.
    """
    cell = np.asarray(cell, dtype=float)
    B = 2.0 * np.pi * np.linalg.inv(cell).T
    span = range(-int(limit), int(limit) + 1)
    triples = np.array([[i, j, k] for i in span for j in span for k in span
                        if (i, j, k) != (0, 0, 0)], dtype=int)
    return triples, triples.astype(float) @ B


def structure_factor(integrals, D, Gamma, *, limit: int = GVECTOR_RANGE):
    r"""``S(k)`` of the state described by ``(D, Gamma)``.

    Parameters
    ----------
    integrals : PeriodicIntegrals
        The integrals the Hamiltonian was built from; supplies the sampled
        orbitals, the grid and the AO-to-MO chain.
    D, Gamma : array_like
        Spin-summed spatial RDMs, as
        :func:`~mandacaru.algorithms.pseudo_forces.spatial_rdms` returns them.
    limit : int
        Range of integer reciprocal-lattice triples.

    Returns
    -------
    StructureFactor
    """
    D = np.asarray(D)
    Gamma = np.asarray(Gamma)
    grid = integrals.grid
    shape = tuple(grid.shape)
    psi = integrals._engine._psi                      # (M_ao, ngrid)

    # The RDMs live in the basis the Hamiltonian was written in: Loewdin
    # orthonormalized, then rotated to molecular orbitals.  A one-body operator
    # transforms as A^dagger M A with A the same product.
    A = integrals._lowdin_x() if integrals.orthogonalize else np.eye(len(psi))
    V = integrals.mo_coefficients
    if V is not None:
        A = A @ V

    triples, kvectors = _reciprocal_vectors(integrals.cell, limit)

    # M_ab(k) = integral conj(phi_a) e^{-ikr} phi_b.  On a grid spanning one
    # period and reciprocal-lattice k, the phase is exactly the DFT kernel, so
    # one FFT per pair density gives every k at once.
    n_ao = psi.shape[0]
    wanted = [tuple(int(m) % n for m, n in zip(triple, shape))
              for triple in triples]
    M = np.empty((len(triples) + 1, n_ao, n_ao), dtype=complex)
    for a in range(n_ao):
        for b in range(a, n_ao):
            pair = (np.conj(psi[a]) * psi[b]).reshape(shape)
            spectrum = sfft.fftn(pair) * grid.dV
            M[0, a, b] = spectrum[0, 0, 0]
            M[0, b, a] = np.conj(spectrum[0, 0, 0])
            for index, node in enumerate(wanted, start=1):
                value = spectrum[node]
                M[index, a, b] = value
                # phi_b* phi_a is the conjugate of phi_a* phi_b, so its
                # transform at the same k is the conjugate at -k; the array is
                # Hermitian in (a, b) only at k = 0.  Sample it directly.
                M[index, b, a] = np.conj(
                    spectrum[tuple((-m) % n for m, n in zip(node, shape))])

    # Rotate every k into the RDM basis in one pass.
    M = np.einsum("ap,kab,bq->kpq", A.conj(), M, A, optimize=True)

    overlap_error = float(np.abs(M[0] - np.eye(M.shape[1])).max())
    n_electrons = float(np.real(np.trace(D)))

    # N S(k) = sum M_pq M*_rs Gamma_prsq + sum M_pq M*_qs D_ps
    values = np.empty(len(triples), dtype=float)
    for index in range(1, len(triples) + 1):
        Mk = M[index]
        pair = np.einsum("pq,rs,prsq->", Mk, np.conj(Mk), Gamma, optimize=True)
        exchange = np.einsum("pq,qs,ps->", Mk, np.conj(Mk), D, optimize=True)
        values[index - 1] = float(np.real(pair + exchange)) / n_electrons

    return StructureFactor(
        kvectors=kvectors,
        values=values,
        magnitudes=np.linalg.norm(kvectors, axis=1),
        n_electrons=n_electrons,
        overlap_error=overlap_error)


def ccmh_correction(structure: StructureFactor, *, volume, n_cells: int = 1,
                    shells: int = DEFAULT_SHELLS, energy_per_cell=None):
    r"""Take :math:`\lim_{k\to0} S(k)/k^2` and turn it into :math:`\Delta V`.

    The limit is extrapolated from the smallest shells by a linear fit of
    :math:`S(k)/k^2` against :math:`k^2` -- the next term in the small-``k``
    expansion of :math:`S` is quartic, so that is the right variable and the
    intercept is the limit.  With a single shell the fit degenerates to that
    shell's value, which is reported rather than refused.
    """
    k, s = structure.shells(shells)
    if k.size == 0:
        raise ValueError("no reciprocal-lattice shells to extrapolate from")
    ratio = s / k ** 2
    if k.size == 1:
        value = float(ratio[0])
    else:
        slope, intercept = np.polyfit(k ** 2, ratio, 1)
        value = float(intercept)

    # Is the quadratic regime actually resolved?  S(k)/k^2 must be flat there.
    spread = 0.0
    if ratio.size > 1 and abs(ratio.mean()) > 0:
        spread = float((ratio.max() - ratio.min()) / abs(ratio.mean()))
    if spread > QUADRATIC_REGIME_SPREAD:
        import warnings

        warnings.warn(
            f"the CCMH extrapolation is outside the quadratic regime: "
            f"S(k)/k^2 varies by {spread:.0%} across the {k.size} fitted "
            f"shells ({np.array2string(ratio, precision=3)}), so S(k) has not "
            f"reached its small-k form where the smallest reciprocal-lattice "
            f"vector sits (|k| = {k[0]:.4f} Bohr^-1, S = {s[0]:.4f}).  The "
            "correction below is an extrapolation of a trend that has not "
            "started.  CCMH assumes an extended system whose long-wavelength "
            "density fluctuations are Coulomb-suppressed; a molecule or a "
            "chain surrounded by vacuum is not one, and neither is a cell too "
            "small to resolve k -> 0.  Use a larger supercell, or treat this "
            "number as a diagnostic only.",
            RuntimeWarning, stacklevel=2)

    volume = float(volume)
    density = structure.n_electrons / volume
    omega_p = float(np.sqrt(4.0 * np.pi * density))
    correction = 4.0 * np.pi * value / (2.0 * volume)
    return CCMHCorrection(
        correction=correction,
        limit=value,
        quadratic_spread=spread,
        rpa_limit=1.0 / (2.0 * omega_p),
        plasma_frequency=omega_p,
        density=density,
        volume=volume,
        n_cells=int(n_cells),
        n_electrons=structure.n_electrons,
        energy_per_cell=(None if energy_per_cell is None
                         else float(energy_per_cell)),
        structure=structure)
