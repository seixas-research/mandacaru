# -*- coding: utf-8 -*-
# file: algorithms/hva.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Fixed Hamiltonian variational ansatz through the ordinary VQE optimizer."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from ..circuits.hva import HamiltonianVariationalAnsatz
from ..core.mapping import Fermion
from .dry_run import QubitEstimate
from .vqe import VQE, VQEResult


class HVA(VQE):
    """Optimize ordered Hamiltonian layers with the shared VQE machinery.

    Use through ``Mandacaru(method='hva', layers=...)``.  Geometry mode builds
    the normal MO-basis fermionic Hamiltonian.  Direct mode takes that
    ``Fermion`` together with ``num_particles`` and optionally
    ``n_spatial_orbitals``.  ``hva_groups`` selects a physical decomposition;
    the default is one-body and two-body Hamiltonian parts.

    This initial implementation evaluates exact sparse exponential actions on
    local state vectors.  It refuses circuit execution and checkpoints, which
    would require a specified gate-level decomposition of each noncommuting
    Hamiltonian group.
    """

    citation_method = "hva"
    solver_label = "HVA"
    _default_sparse = "auto"
    _supports_tapering = False

    def __init__(self, hamiltonian: Fermion | None = None,
                 num_particles: tuple[int, int] | None = None,
                 n_spatial_orbitals: int | None = None, *, layers: int = 2,
                 hva_groups: tuple[Fermion, ...] | None = None,
                 **driver_kwargs: object) -> None:
        if isinstance(layers, bool) or not isinstance(layers, Integral) or layers < 1:
            raise ValueError("HVA layers must be a positive integer")
        unsupported = ("load_hamiltonian", "ansatz_builder", "checkpoint",
                       "resume", "execute_circuits", "shots")
        used = [name for name in unsupported if driver_kwargs.get(name)]
        if used:
            raise ValueError(
                "HVA needs its fermionic group decomposition and local exact "
                "state evolution; it does not take "
                + ", ".join(f"{name}=" for name in used))
        if driver_kwargs.get("backend_provider") not in (None, "qiskit"):
            raise ValueError("HVA currently uses local exact state evolution; "
                             "set backend_provider='qiskit' or leave it out")
        self.layers = int(layers)
        self.hva_groups = None if hva_groups is None else tuple(hva_groups)
        super().__init__(hamiltonian=None, ansatz=None, **driver_kwargs)
        if hamiltonian is not None:
            if num_particles is None:
                raise ValueError("direct HVA needs num_particles=(n_alpha, n_beta)")
            orbitals = (hamiltonian.n_modes() // 2 if n_spatial_orbitals is None
                        else int(n_spatial_orbitals))
            if self.dry_run:
                self._dry_run_problem = (hamiltonian, num_particles, orbitals)
            else:
                self._configure(hamiltonian, num_particles, orbitals)
                self._built_from_hamiltonian = True
                self._maybe_write_references()
        elif num_particles is not None or n_spatial_orbitals is not None:
            raise ValueError("num_particles and n_spatial_orbitals need a direct "
                             "fermionic hamiltonian=")

    def _configure(self, hamiltonian: Fermion,
                   num_particles: tuple[int, int], n_orbitals: int) -> None:
        """Build the fixed HVA ansatz and let VQE map the same Hamiltonian."""
        if not isinstance(hamiltonian, Fermion):
            raise TypeError("HVA needs a Fermion Hamiltonian so its physical "
                            "one- and two-body groups are available")
        if hamiltonian.n_modes() != 2 * n_orbitals:
            raise ValueError("HVA Hamiltonian modes and spatial orbitals disagree")
        self._preset_ansatz = HamiltonianVariationalAnsatz(
            hamiltonian, num_particles, mapping=self.mapping,
            layers=self.layers, groups=self.hva_groups)
        super()._configure(hamiltonian, num_particles, n_orbitals)

    def run(self, initial_parameters: np.ndarray | None = None
            ) -> VQEResult | QubitEstimate:
        """Optimize from nonzero seed angles unless angles were supplied.

        For real Hamiltonians and a real Hartree-Fock determinant, the energy
        derivatives of ``exp(-i theta H_j)`` vanish at all-zero angles.  A
        small deterministic seed moves the fixed HVA off that stationary
        point while preserving reproducibility.
        """
        if initial_parameters is None and self._configured:
            initial_parameters = np.full(self.ansatz.num_parameters, 0.1)
        return super().run(initial_parameters=initial_parameters)
