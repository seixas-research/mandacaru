# -*- coding: utf-8 -*-
# file: core/planewave.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Plane-wave basis: one- and two-body integrals with periodic boundary conditions.

A plane-wave basis is intrinsically periodic: the orbitals are the delocalized
Bloch waves of the simulation cell,

.. math::

    \varphi_{\mathbf G}(\mathbf r) = \frac{1}{\sqrt\Omega}\,
        e^{i\,\mathbf G\cdot\mathbf r},

for every reciprocal-lattice vector :math:`\mathbf G` below the kinetic-energy
cutoff, :math:`\tfrac12|\mathbf G|^2 \le E_{\mathrm{cut}}` (``energy_cutoff``,
default ``300`` eV).  The integrals then have closed reciprocal-space forms and
the kinetic energy is diagonal:

.. math::

    T_{pq} &= \tfrac12 |\mathbf G_p|^2\, \delta_{pq}, \\
    V_{pq} &= -\frac{4\pi}{\Omega} \sum_I
        \frac{Z_I\, e^{-i(\mathbf G_p-\mathbf G_q)\cdot\mathbf R_I}}
             {|\mathbf G_p-\mathbf G_q|^2}\quad (p\neq q), \\
    \langle pq|rs\rangle &= \frac{4\pi}{\Omega}\,
        \frac{\delta_{\mathbf G_p+\mathbf G_q,\,\mathbf G_r+\mathbf G_s}}
             {|\mathbf G_p-\mathbf G_r|^2}\quad (\mathbf G_p\neq\mathbf G_r).

The divergent :math:`\mathbf G = 0` components (the average potential and the
Hartree self-energy) are removed by the usual neutralizing background (jellium):
the :math:`p=q` external term and the :math:`p=r` electron-electron term are set
to zero.  :class:`PlaneWaveIntegrals` mirrors
:class:`~carcara.core.hamiltonian.MolecularIntegrals` so the rest of the pipeline
(RHF molecular-orbital basis, fermion-to-qubit mapping, VQE/ADAPT-VQE) is
unchanged.

**Practical note.**  The number of plane waves grows as
:math:`E_{\mathrm{cut}}^{3/2}\,\Omega`; the dense two-body tensor is
:math:`O(N_{\mathrm{pw}}^4)`, so a physically realistic cutoff on a molecule-sized
cell yields far too many plane waves for an exact state-vector simulation.  The
class raises above ``max_plane_waves`` (default ``60``); use a small cell and/or a
low cutoff for tractable examples.
"""

from __future__ import annotations

import numpy as np

from ..units import EV_TO_HARTREE, to_bohr

DEFAULT_ENERGY_CUTOFF_EV = 300.0


def reciprocal_lattice(cell_bohr: np.ndarray) -> np.ndarray:
    r"""Reciprocal-lattice vectors (rows) for real lattice ``cell_bohr`` (rows).

    Uses the crystallographic :math:`2\pi` convention, so
    :math:`\mathbf a_i\cdot\mathbf b_j = 2\pi\,\delta_{ij}`.
    """
    A = np.asarray(cell_bohr, dtype=float)
    return 2.0 * np.pi * np.linalg.inv(A).T


def plane_wave_vectors(cell_bohr: np.ndarray, energy_cutoff_ha: float):
    r"""Reciprocal vectors ``G`` with :math:`\tfrac12|G|^2 \le E_{\mathrm{cut}}`.

    Returns ``(G, miller)``: the Cartesian reciprocal vectors (Bohr\ :sup:`-1`)
    and their integer Miller indices, both sorted by kinetic energy so the first
    entry is ``G = 0``.
    """
    A = np.asarray(cell_bohr, dtype=float)
    B = reciprocal_lattice(A)
    gmax = np.sqrt(2.0 * energy_cutoff_ha)
    # Rigorous bound on the Miller index: |m_i| = |a_i . G| / 2pi <= |a_i| gmax / 2pi.
    alen = np.linalg.norm(A, axis=1)
    nmax = np.floor(alen * gmax / (2.0 * np.pi)).astype(int) + 1

    gs, millers = [], []
    for i in range(-nmax[0], nmax[0] + 1):
        for j in range(-nmax[1], nmax[1] + 1):
            for k in range(-nmax[2], nmax[2] + 1):
                G = i * B[0] + j * B[1] + k * B[2]
                if 0.5 * float(G @ G) <= energy_cutoff_ha + 1e-10:
                    gs.append(G)
                    millers.append((i, j, k))
    G = np.asarray(gs, dtype=float).reshape(-1, 3)
    miller = np.asarray(millers, dtype=int).reshape(-1, 3)
    order = np.argsort(0.5 * np.sum(G * G, axis=1))
    return G[order], miller[order]


class PlaneWaveIntegrals:
    """Plane-wave one-/two-body integrals and the molecular Hamiltonian (PBC).

    Parameters
    ----------
    nuclei : sequence of ``(Z, position)``
        Nuclear charges and Cartesian positions (in ``units``).
    cell : (3, 3) array_like
        Real-space lattice tensor (rows are lattice vectors, ASE convention), in
        ``units`` -- the periodic simulation cell.
    energy_cutoff : float
        Plane-wave kinetic-energy cutoff (default ``300`` eV): all ``G`` with
        ``|G|^2/2 <= energy_cutoff`` are included.
    units : {"angstrom", "bohr"}
        Unit of ``cell`` and the nuclear positions (default ``"angstrom"``).
    energy_units : {"eV", "Ha"}
        Unit of ``energy_cutoff`` (default ``"eV"``).
    max_plane_waves : int
        Guard against an intractable ``O(N^4)`` two-body tensor (default ``60``).
    """

    def __init__(self, nuclei, cell, energy_cutoff: float = DEFAULT_ENERGY_CUTOFF_EV,
                 units: str = "angstrom", energy_units: str = "eV",
                 max_plane_waves: int = 60):
        self.cell_bohr = np.asarray(to_bohr(cell, units), dtype=float)
        if self.cell_bohr.shape != (3, 3) or not np.any(self.cell_bohr):
            raise ValueError("the plane-wave basis requires a (3, 3) unit cell")
        self.volume = float(abs(np.linalg.det(self.cell_bohr)))
        self.energy_cutoff_ev = float(energy_cutoff) * (
            1.0 if energy_units.lower() in ("ev",) else 1.0 / EV_TO_HARTREE)
        self.energy_cutoff_ha = (float(energy_cutoff) * EV_TO_HARTREE
                                 if energy_units.lower() in ("ev",)
                                 else float(energy_cutoff))
        self.nuclei = [(float(Z), np.asarray(to_bohr(R, units), dtype=float))
                       for Z, R in nuclei]

        self.G, self.miller = plane_wave_vectors(self.cell_bohr,
                                                 self.energy_cutoff_ha)
        self.npw = int(len(self.G))
        if self.npw > int(max_plane_waves):
            raise ValueError(
                f"the plane-wave cutoff yields {self.npw} plane waves, above the "
                f"limit max_plane_waves={max_plane_waves}: the O(N^4) two-body "
                "tensor would be intractable.  Reduce energy_cutoff or the cell "
                "size (or raise max_plane_waves for the integrals alone).")

        # Miller-index -> plane-wave index, for momentum conservation lookups.
        self._index = {tuple(m): p for p, m in enumerate(self.miller)}
        self._h: np.ndarray | None = None
        self._eri: np.ndarray | None = None

        from ..utils.profiling import Timings
        from ..integrals import _backend
        self.timings = Timings(
            n_cores=_backend.num_threads(),
            backend="C (OpenMP)" if _backend.HAS_C_BACKEND else "NumPy")

    # -- basic properties ------------------------------------------------- #

    @property
    def n_orbitals(self) -> int:
        return self.npw

    def integration_profile(self) -> dict:
        return self.timings.as_dict()

    @property
    def nuclear_repulsion(self) -> float:
        r"""Pairwise nuclear repulsion ``sum_{I<J} Z_I Z_J / |R_I - R_J|`` (Hartree)."""
        e = 0.0
        for i in range(len(self.nuclei)):
            for j in range(i + 1, len(self.nuclei)):
                e += (self.nuclei[i][0] * self.nuclei[j][0]
                      / np.linalg.norm(self.nuclei[i][1] - self.nuclei[j][1]))
        return float(e)

    def overlap(self) -> np.ndarray:
        """Overlap matrix -- the identity (plane waves are orthonormal)."""
        return np.eye(self.npw, dtype=complex)

    # -- integrals -------------------------------------------------------- #

    def _compute(self) -> None:
        npw, G = self.npw, self.G
        prefac = 4.0 * np.pi / self.volume

        with self.timings.time("one-body integrals (PW)"):
            # Kinetic energy is diagonal in the plane-wave basis.
            h = np.diag(0.5 * np.sum(G * G, axis=1)).astype(complex)
            # External (electron-nuclear) potential; G=0 (p==q) dropped (jellium).
            Zs = np.array([Z for Z, _R in self.nuclei])
            Rs = np.array([R for _Z, R in self.nuclei])
            for p in range(npw):
                for q in range(npw):
                    if p == q:
                        continue
                    dG = G[p] - G[q]
                    g2 = float(dG @ dG)
                    structure = np.sum(Zs * np.exp(-1j * (Rs @ dG)))
                    h[p, q] += -prefac * structure / g2
            self._h = h

        with self.timings.time("two-body integrals (PW)"):
            eri = np.zeros((npw, npw, npw, npw), dtype=complex)
            mill = self.miller
            for p in range(npw):
                for r in range(npw):
                    if p == r:
                        continue                      # G=0 Hartree term (jellium)
                    dG = G[p] - G[r]
                    value = prefac / float(dG @ dG)
                    for q in range(npw):
                        # Momentum conservation: G_s = G_p + G_q - G_r.
                        ms = tuple(mill[p] + mill[q] - mill[r])
                        s = self._index.get(ms)
                        if s is not None:
                            eri[p, q, r, s] = value
            self._eri = eri

    def one_body(self) -> np.ndarray:
        r"""Spatial one-body Hamiltonian ``h = T + V`` (Hartree), plane-wave basis."""
        if self._h is None:
            self._compute()
        return self._h

    def two_body(self) -> np.ndarray:
        r"""Two-body tensor ``<pq|rs>`` (physicists', Hartree), plane-wave basis."""
        if self._eri is None:
            self._compute()
        return self._eri

    # -- Hartree-Fock and the molecular Hamiltonian ----------------------- #

    def hartree_fock(self, n_electrons: int):
        """Restricted Hartree-Fock in the (orthonormal) plane-wave basis."""
        from ..algorithms.hartree_fock import RHF
        return RHF(self.one_body(), self.two_body(), n_electrons).run()

    def molecular_hamiltonian(self, include_nuclear_repulsion: bool = True,
                              mo_basis: bool = False,
                              n_electrons: int | None = None):
        """Assemble the second-quantized plane-wave :class:`Fermion` Hamiltonian.

        Mirrors :meth:`carcara.core.MolecularIntegrals.molecular_hamiltonian`:
        ``mo_basis=True`` first rotates the integrals into the RHF molecular-orbital
        basis (``n_electrons`` required).
        """
        from .hamiltonian import spin_block_integrals
        from .mapping import Fermion

        if mo_basis:
            if n_electrons is None:
                raise ValueError("mo_basis=True requires n_electrons")
            rhf = self.hartree_fock(n_electrons)
            h_so, g_so = spin_block_integrals(rhf.h_mo, rhf.eri_mo)
        else:
            h_so, g_so = spin_block_integrals(self.one_body(), self.two_body())
        H = Fermion.from_integrals(h_so, g_so)
        if include_nuclear_repulsion:
            H = H + Fermion({(): complex(self.nuclear_repulsion)},
                            n_modes=2 * self.n_orbitals)
        return H
