# -*- coding: utf-8 -*-
# file: algorithms/sector_support.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The particle-number sector option shared by the state-vector drivers.

A molecular Hamiltonian and every excitation generator conserve
``(n_alpha, n_beta)``, so a state vector needs only the
``C(M, n_alpha) C(M, n_beta)`` determinants of that sector, not the ``2^(2M)``
amplitudes of the register.  :class:`SectorMixin` holds the option
(``sector=True | False | "auto"``) and the safety check behind it: a sector is
used only when the Hamiltonian and every generator keep the state inside it.
ADAPT-VQE and MCAS-VQE (through :class:`~mandacaru.algorithms.pool_driver.PoolDriver`)
and the fixed-ansatz VQE use it.
"""

from __future__ import annotations

import warnings


class SectorMixin:
    """The ``sector=`` option of a driver whose states may live in a sector."""

    #: Registers at least this wide use the particle-number sector by default.
    SECTOR_AUTO_QUBITS = 16
    #: Drivers whose states cannot live in a sector override this.
    _supports_sector = True
    #: Largest ``sector.dim * len(terms)`` product worth checking for leakage.
    SECTOR_GUARD_WORK = 20_000_000

    def _set_sector_option(self, sector) -> None:
        """Validate and store ``sector`` (``True``, ``False`` or ``"auto"``)."""
        if not (isinstance(sector, bool)
                or (isinstance(sector, str) and sector.strip().lower() == "auto")):
            raise ValueError(f"unknown sector spec {sector!r}; use True, False "
                             "or 'auto'")
        self.sector = sector

    def _resolve_sector(self, n_qubits: int, qubit_h=None, operators=()):
        """The :class:`~mandacaru.core.sector.ParticleSector` to simulate, or ``None``.

        A sector is only safe when the Hamiltonian *and* every generator
        keep the ansatz inside it: :meth:`~mandacaru.core.sector.ParticleSector.restrict`
        drops whatever leaves, which would quietly replace a generator by its
        projection (``exp(PAP) != P exp(A) P``).  The qubit pool's individual
        Pauli strings do leak by construction.  An explicit ``sector=True``
        raises for such a pool; the automatic choice falls back to the full
        register and says so.
        """
        spec = self.sector
        internal = (self.ansatz_provider() is None
                    and self._sector_simulates_states())
        if isinstance(spec, str):
            use = (int(n_qubits) >= self.SECTOR_AUTO_QUBITS
                   and self._supports_sector and internal)
        else:
            use = bool(spec)
        if not use:
            return None
        if not self._supports_sector:
            raise NotImplementedError(
                f"{type(self).__name__} builds full-register reference states "
                "and cannot run in a particle-number sector (sector=True)")
        if not internal:
            raise ValueError(
                "sector=True needs the internal state-vector backend; "
                + ("a shot-based run (shots > 0) simulates no state at all"
                   if getattr(self, "shots", 0) else
                   "executing the ansatz as a circuit prepares full-register "
                   "states"))
        from ..core.sector import ParticleSector
        try:
            sector = ParticleSector(n_qubits, self.num_particles, self.mapping,
                                    spin_conserving=getattr(
                                        self, "spin_conserving", True))
        except ValueError as error:
            # Too many determinants to enumerate: the automatic choice
            # leaves the decision (and its refusal) to the register path.
            if not isinstance(spec, str):
                raise
            warnings.warn(f"simulating the full register: {error}",
                          RuntimeWarning, stacklevel=3)
            return None

        leaking = self._sector_leak(sector, qubit_h, operators)
        if leaking is not None:
            message = (f"{leaking} does not conserve the "
                       f"{self.num_particles} particle-number sector, so "
                       f"restricting it would change the operator "
                       f"(exp(PAP) != P exp(A) P)")
            if not isinstance(spec, str):
                raise ValueError(
                    f"sector=True was requested but {message}; use the "
                    f"'fermionic', 'qeb' or 'ceo' pool, or sector=False")
            warnings.warn(f"simulating the full register: {message}",
                          RuntimeWarning, stacklevel=3)
            return None
        return sector

    def _sector_simulates_states(self) -> bool:
        """Whether this run simulates states locally, where a sector helps.

        Not with ``shots > 0``: every energy *and* every pool gradient of a
        shot-based run is measured on the device, and no state is simulated,
        so a sector would be enumerated (and checked for leaks, at the cost
        of its dimension) for nothing.
        """
        return not getattr(self, "shots", 0)

    def _sector_leak(self, sector, qubit_h, operators) -> str | None:
        """Name of the first operator that leaves ``sector``, or ``None``."""
        def affordable(operator):
            return (operator is not None
                    and sector.dim * max(len(operator.terms), 1)
                    <= self.SECTOR_GUARD_WORK)

        if affordable(qubit_h) and not sector.conserves(qubit_h):
            return "the Hamiltonian"
        for index, op in enumerate(operators):
            generator = getattr(op, "generator", op)
            if affordable(generator) and not sector.conserves(generator):
                return (f"operator {getattr(op, 'label', index)!r}")
        return None
