# -*- coding: utf-8 -*-
# file: backends/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Execution backends: devices, circuit providers (SDKs) and error mitigation.

* :mod:`~mandacaru.backends.hardware` -- the *device* registry (which machine the
  variational loop runs on);
* :mod:`~mandacaru.backends.providers` -- the *circuit provider* registry (which
  SDK builds and executes the ansatz circuits: Qiskit, Amazon Braket or Cirq);
* :mod:`~mandacaru.backends.measurement` -- shot-based energies, qubit-wise
  commuting groups and the measurement budget that refuses an oversized job;
* :mod:`~mandacaru.backends.factorization` -- the double-factorized Hamiltonian
  and the Givens decomposition of its orbital rotations.
"""

from .factorization import (DoubleFactorization, double_factorization,
                            factorized_measurement_bases,
                            spatial_double_factorization)
from .hardware import (available_devices, is_simulator, normalize_device,
                       require_runnable)
from .measurement import (DEFAULT_MEASUREMENT_BUDGET, chunk_labels_by_basis,
                          qubit_wise_commuting_groups,
                          resolve_measurement_budget, shot_noise_estimate)
from .providers import (BACKEND_PROVIDERS, BraketProvider, CircuitProvider,
                        CirqProvider, QiskitProvider, build_provider,
                        normalize_provider, provider_available)

__all__ = [
    "DEFAULT_MEASUREMENT_BUDGET",
    "chunk_labels_by_basis",
    "resolve_measurement_budget",
    "qubit_wise_commuting_groups",
    "shot_noise_estimate",
    "DoubleFactorization",
    "double_factorization",
    "spatial_double_factorization",
    "factorized_measurement_bases",
    "available_devices",
    "normalize_device",
    "is_simulator",
    "require_runnable",
    "BACKEND_PROVIDERS",
    "CircuitProvider",
    "QiskitProvider",
    "BraketProvider",
    "CirqProvider",
    "build_provider",
    "normalize_provider",
    "provider_available",
]
