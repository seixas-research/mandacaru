"""Unit-aware view of a native Lehmann spectral function on the Brillouin-zone mesh."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SpectralFunction:
    r"""``A(k, omega)`` of the correlated supercell state at the commensurate k-points.

    For every mesh point ``k`` and primitive Lowdin orbital ``nu``, the Bloch operator
    :math:`c^\dagger_{k\nu\sigma} = N^{-1/2}\sum_R e^{2\pi i k\cdot R} c^\dagger_{R\nu\sigma}`
    gives the diagonal spectral function

    .. math::

        A_{\nu\nu}(k,\omega) = \sum_n |\langle n, N+1| c^\dagger_{k\nu\sigma}|\Psi\rangle|^2
            \delta(\omega - (E_n - E_0))
          + \sum_n |\langle n, N-1| c_{k\nu\sigma}|\Psi\rangle|^2 \delta(\omega - (E_0 - E_n)),

    with :math:`|\Psi\rangle` the variational state of the run, :math:`E_0` its energy and
    :math:`E_n` the exact eigenvalues of the same Hamiltonian in the ``N +- 1`` sectors (a
    Lanczos recursion per part, exact once it spans the sector). ``values`` is the trace over
    orbitals and spins broadened by a Lorentzian of half-width ``eta``, in states per eV per
    cell. Along a band path (``kpath`` is set) every row of ``values`` is a point of the path.
    Energies are excitation energies in eV relative to :math:`E_0`: removal poles sit at
    :math:`E_0 - E_n^{N-1}`, addition poles at :math:`E_n^{N+1} - E_0`, so the gap of the state
    is ``addition_edge - removal_edge``.
    """

    kpoints: np.ndarray
    energies: np.ndarray
    values: np.ndarray
    eta: float
    poles: dict = field(repr=False)
    #: ``(nk, n_orbitals, 2)``: total weight of each orbital channel and spin; 1 by the sum rule.
    sum_rule: np.ndarray = field(repr=False)
    #: ``(nk, n_orbitals, 2)``: energy (eV) and weight of the strongest pole of each channel.
    dominant_energies: np.ndarray = field(repr=False)
    dominant_weights: np.ndarray = field(repr=False)
    removal_edge: float = 0.0
    addition_edge: float = 0.0
    ground_energy: float = 0.0
    mesh: tuple = (1, 1, 1)
    #: The band path ``kpoints`` and ``values`` are along (``kpoints`` then are its points).
    kpath: object = None
    #: With a reduction to the irreducible zone: the symmetry audit of the state (a dict).
    audit: dict | None = field(default=None, repr=False)

    @property
    def chemical_potential(self):
        """Midpoint between the highest removal and the lowest addition pole (eV)."""
        return 0.5 * (self.removal_edge + self.addition_edge)

    @property
    def gap(self):
        """``addition_edge - removal_edge`` (eV): the fundamental gap of the state."""
        return self.addition_edge - self.removal_edge

    def bands(self, spin=0):
        """``(nk, n_orbitals)``: the strongest pole of each orbital channel (eV)."""
        return self.dominant_energies[:, :, spin].copy()

    def band_weights(self, spin=0):
        """``(nk, n_orbitals)``: the weight that pole carries (1 for a determinant)."""
        return self.dominant_weights[:, :, spin].copy()
