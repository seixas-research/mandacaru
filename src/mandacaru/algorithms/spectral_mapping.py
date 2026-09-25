# -*- coding: utf-8 -*-
# file: algorithms/spectral_mapping.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Mapping-aware sector transitions for the Lehmann spectral function.

A ladder operator changes particle number, so it connects two sectors rather
than acting within either one.  Jordan-Wigner, parity and Bravyi-Kitaev use
their ordinary full-register Pauli images.  For ``parity_reduced`` the source
state is lifted by restoring its two fixed parity bits; the ladder operator is
applied in the full parity encoding; and the resulting state is projected onto
the reduced register with the *target* particle-number parities.  The target
Hamiltonian is reduced with those same target parities.  Thus no ladder
operator is incorrectly tapered as a number-conserving observable.
"""

from __future__ import annotations

import numpy as np

from ..core.mapping import Fermion, PauliSum, resolve_mapping
from ..core.sector import (MAX_SECTOR_QUBITS, ParticleSector,
                           apply_pauli_sum)


class SpectralRegister:
    r"""The reference register and its mapped :math:`N\pm1` sectors.

    Parameters
    ----------
    mapping : str
        The encoding used by the converged solver.
    n_modes : int
        Number of original spin-orbitals, before parity reduction.
    num_particles : (int, int)
        Alpha and beta electron counts of the reference state.
    state : ndarray
        Optimized state in the solver's full or sector register.
    hamiltonian : PauliSum
        Solver Hamiltonian on its actual register.
    fermion_hamiltonian : Fermion, optional
        Original operator, required for ``parity_reduced`` so each charged
        sector can use its own exact parity reduction.
    """

    def __init__(self, mapping: str, n_modes: int,
                 num_particles: tuple[int, int], state: np.ndarray,
                 hamiltonian: PauliSum,
                 fermion_hamiltonian: Fermion | None = None) -> None:
        self.mapping = resolve_mapping(mapping)
        self.n_modes = int(n_modes)
        self.num_particles = tuple(int(n) for n in num_particles)
        self.reduced = self.mapping == "parity_reduced"
        self.register_qubits = self.n_modes - (2 if self.reduced else 0)
        if self.n_modes < 2 or self.n_modes % 2:
            raise ValueError("the spectral register needs an even number of "
                             "spin-orbitals")
        if self.n_modes > MAX_SECTOR_QUBITS:
            raise ValueError(
                f"spectral transitions need at most {MAX_SECTOR_QUBITS} "
                "spin-orbitals for 64-bit basis indices")
        if hamiltonian.num_qubits != self.register_qubits:
            raise ValueError(
                f"the {self.mapping} Hamiltonian has {hamiltonian.num_qubits} "
                f"qubits; expected {self.register_qubits}")
        if self.reduced and (fermion_hamiltonian is None or
                             fermion_hamiltonian.n_modes() != self.n_modes):
            raise ValueError(
                "parity_reduced spectra need the original fermionic "
                "Hamiltonian to build the N+1 and N-1 sectors")
        self._fermion_hamiltonian = fermion_hamiltonian
        self._hamiltonians = {self.num_particles: hamiltonian}
        self._sectors: dict[tuple[int, int], ParticleSector] = {}

        sector = self.sector(self.num_particles)
        psi = np.asarray(state, dtype=complex).ravel()
        if psi.size == 1 << self.register_qubits:
            amplitudes = sector.project(psi)
            outside_norm = np.linalg.norm(psi)**2 - np.linalg.norm(amplitudes)**2
            if outside_norm > 1e-8:
                raise ValueError("the converged state leaves its declared "
                                 "particle-number sector")
        elif psi.size == sector.dim:
            amplitudes = psi
        else:
            raise ValueError(
                f"the optimized state has {psi.size} amplitudes; expected "
                f"{1 << self.register_qubits} on the full register or "
                f"{sector.dim} in its particle-number sector")
        nonzero = np.abs(amplitudes) > 1e-14
        indices = sector.indices[nonzero]
        self._source_indices = (self._lift(indices, self.num_particles)
                                if self.reduced else indices)
        self._source_amplitudes = amplitudes[nonzero]

    def sector(self, particles: tuple[int, int]) -> ParticleSector:
        """Cached sector on the solver's register for ``particles``."""
        particles = tuple(int(n) for n in particles)
        if particles not in self._sectors:
            self._sectors[particles] = ParticleSector(
                self.register_qubits, particles, self.mapping)
        return self._sectors[particles]

    def hamiltonian(self, particles: tuple[int, int]) -> PauliSum:
        """Hamiltonian reduced with the target sector's parity eigenvalues."""
        particles = tuple(int(n) for n in particles)
        if not self.reduced:
            return self._hamiltonians[self.num_particles]
        if particles not in self._hamiltonians:
            self._hamiltonians[particles] = \
                self._fermion_hamiltonian.map_to_qubits(
                    "parity_reduced", n_modes=self.n_modes,
                    num_particles=particles)
        return self._hamiltonians[particles]

    def _lift(self, indices: np.ndarray,
              particles: tuple[int, int]) -> np.ndarray:
        """Restore the alpha and total parity bits to reduced basis indices."""
        m = self.n_modes // 2
        low_mask = (1 << (m - 1)) - 1
        alpha = indices >> np.int64(m - 1)
        beta = indices & np.int64(low_mask)
        return ((alpha << np.int64(m + 1))
                | (np.int64(particles[0] & 1) << np.int64(m))
                | (beta << np.int64(1))
                | np.int64((particles[0] + particles[1]) & 1))

    def _drop(self, indices: np.ndarray,
              particles: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Remove target parity bits and report which images had correct signs."""
        m = self.n_modes // 2
        low_mask = (1 << (m - 1)) - 1
        correct = (((indices >> np.int64(m)) & 1) == (particles[0] & 1))
        correct &= (indices & 1) == ((particles[0] + particles[1]) & 1)
        reduced = (((indices >> np.int64(m + 1)) << np.int64(m - 1))
                   | ((indices >> np.int64(1)) & np.int64(low_mask)))
        return reduced, correct

    def transition(self, creation: Fermion,
                   particles: tuple[int, int], *, addition: bool
                   ) -> np.ndarray:
        r"""Apply a creation or annihilation operator into a charged sector.

        Returns amplitudes in :meth:`sector` order.  The mapped Pauli operator
        is applied once to the sparse reference state, with no full-register
        matrix and no repeated VQE state preparation.  A failed sector or
        parity projection raises, rather than silently losing spectral weight.
        """
        particles = tuple(int(n) for n in particles)
        m = self.n_modes // 2
        if any(n < 0 or n > m for n in particles):
            return np.empty(0, dtype=complex)
        base = "parity" if self.reduced else self.mapping
        operator = creation.map_to_qubits(base, n_modes=self.n_modes)
        if not addition:
            operator = PauliSum(
                {label: np.conj(coeff) for label, coeff in operator.terms.items()},
                num_qubits=self.n_modes)
        images, amplitudes = apply_pauli_sum(
            operator, self._source_indices, self._source_amplitudes)
        if self.reduced:
            images, correct = self._drop(images, particles)
        else:
            correct = np.ones(images.size, dtype=bool)
        sector = self.sector(particles)
        positions = sector.positions(images)
        kept = correct & (positions >= 0)
        if np.linalg.norm(amplitudes[~kept]) > 1e-8:
            raise RuntimeError(
                "a mapped ladder operator left its N+1 or N-1 sector; "
                "the operator and Hamiltonian use inconsistent mappings")
        out = np.zeros(sector.dim, dtype=complex)
        np.add.at(out, positions[kept], amplitudes[kept])
        return out
