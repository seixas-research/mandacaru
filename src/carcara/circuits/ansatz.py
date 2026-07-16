# -*- coding: utf-8 -*-
# file: circuits/ansatz.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Parameterized ansätze for variational quantum algorithms.

:class:`UCCSD` implements the Unitary Coupled-Cluster with Singles and Doubles
ansatz as a state-vector generator:

.. math::

    |\psi(\vec\theta)\rangle
        = e^{\sum_k \theta_k (\hat T_k - \hat T_k^\dagger)}\,|\text{HF}\rangle,

the exponential of the full anti-Hermitian cluster operator built from the single
and double excitation generators (:mod:`carcara.circuits.gates`), acting on the
Hartree-Fock reference determinant.  Each generator is mapped to qubits with the
same fermion-to-qubit mapping as the Hamiltonian (Jordan-Wigner by default) and
the exponential is evaluated exactly on the :math:`2^N` state vector -- an exact,
simulator-oriented reference implementation for small systems.

With ``trotter=True`` the state is instead the first-order Trotter product
:math:`\prod_k e^{\theta_k (\hat T_k - \hat T_k^\dagger)}|\text{HF}\rangle`, which
is the form realized as a quantum circuit but only approximates the exact UCC
unitary.

Spin-orbitals follow the spin-blocked ordering used throughout Carcará: the
first ``n_spatial_orbitals`` indices are the :math:`\alpha` orbitals, the next
``n_spatial_orbitals`` the :math:`\beta` orbitals.  The reference fills the
lowest ``n_alpha`` / ``n_beta`` orbitals of each block.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..core.mapping import Fermion
from .gates import double_excitation, single_excitation


class UCCSD:
    """UCCSD ansatz state-vector generator over ``2 * n_spatial_orbitals`` qubits.

    Parameters
    ----------
    n_spatial_orbitals : int
        Number of spatial orbitals ``M``; the ansatz acts on ``N = 2M`` qubits.
    num_particles : (int, int)
        ``(n_alpha, n_beta)`` electrons defining the Hartree-Fock reference.
    mapping : str
        Fermion-to-qubit mapping for the generators (default ``"jordan_wigner"``);
        must match the Hamiltonian's mapping.
    include_singles : bool
        Include single excitations (default ``True``).
    trotter : bool
        If ``True``, prepare the state as the first-order Trotter product of
        single-generator exponentials (circuit-faithful but approximate); if
        ``False`` (default), exponentiate the full cluster operator exactly.
    """

    def __init__(self, n_spatial_orbitals: int, num_particles: tuple[int, int],
                 mapping: str = "jordan_wigner", include_singles: bool = True,
                 trotter: bool = False):
        self.n_spatial_orbitals = int(n_spatial_orbitals)
        self.num_particles = (int(num_particles[0]), int(num_particles[1]))
        self.mapping = mapping
        self.include_singles = include_singles
        self.trotter = trotter
        self.n_qubits = 2 * self.n_spatial_orbitals

        self._occupied, self._virtual = self._reference_partition()
        self.excitations = self._build_excitations()
        # Pre-map and pre-materialize each anti-Hermitian generator matrix once.
        self._generators = [
            g.map_to_qubits(self.mapping, n_modes=self.n_qubits).to_matrix()
            for g in self.excitations]
        self._hf = self._reference_vector()

    # -- reference determinant -------------------------------------------- #

    def _reference_partition(self) -> tuple[list[int], list[int]]:
        """Occupied and virtual spin-orbital indices of the HF reference."""
        M = self.n_spatial_orbitals
        na, nb = self.num_particles
        occ = list(range(na)) + list(range(M, M + nb))
        virt = ([a for a in range(M) if a not in occ]
                + [a for a in range(M, 2 * M) if a not in occ])
        return occ, virt

    def _reference_vector(self) -> np.ndarray:
        """Hartree-Fock computational-basis state (qubit 0 is the MSB)."""
        index = 0
        for j in self._occupied:
            index |= 1 << (self.n_qubits - 1 - j)
        vec = np.zeros(2 ** self.n_qubits, dtype=complex)
        vec[index] = 1.0
        return vec

    # -- excitation pool -------------------------------------------------- #

    def _spin(self, p: int) -> int:
        return p // self.n_spatial_orbitals

    def _build_excitations(self) -> list[Fermion]:
        """Spin-conserving single and double excitation generators."""
        occ, virt = self._occupied, self._virtual
        gens: list[Fermion] = []

        if self.include_singles:
            for i in occ:
                for a in virt:
                    if self._spin(i) == self._spin(a):
                        gens.append(single_excitation(i, a))

        for x in range(len(occ)):
            for y in range(x + 1, len(occ)):
                i, j = occ[x], occ[y]
                for p in range(len(virt)):
                    for q in range(p + 1, len(virt)):
                        a, b = virt[p], virt[q]
                        # Conserve total spin projection Sz.
                        if self._spin(i) + self._spin(j) == self._spin(a) + self._spin(b):
                            gens.append(double_excitation(i, j, a, b))
        return gens

    # -- public interface ------------------------------------------------- #

    @property
    def num_parameters(self) -> int:
        return len(self.excitations)

    def reference_state(self) -> np.ndarray:
        """The Hartree-Fock reference state vector (a copy)."""
        return self._hf.copy()

    def state(self, theta) -> np.ndarray:
        r"""Prepared state ``|psi(theta)>`` for parameters ``theta``.

        Exact UCC (default): ``exp(sum_k theta_k G_k) |HF>``.  Trotter mode:
        the product ``prod_k exp(theta_k G_k) |HF>``.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.num_parameters:
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {theta.size}")
        if self.trotter:
            psi = self._hf.copy()
            for angle, gen in zip(theta, self._generators):
                psi = expm(angle * gen) @ psi
            return psi
        cluster = sum((angle * gen for angle, gen in zip(theta, self._generators)),
                      np.zeros((2 ** self.n_qubits,) * 2, dtype=complex))
        return expm(cluster) @ self._hf

    def __repr__(self) -> str:
        return (f"UCCSD(n_qubits={self.n_qubits}, "
                f"num_particles={self.num_particles}, "
                f"num_parameters={self.num_parameters})")
