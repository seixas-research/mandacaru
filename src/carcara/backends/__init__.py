# -*- coding: utf-8 -*-
# file: backends/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Execution backends: devices, circuit providers (SDKs) and error mitigation.

* :mod:`~carcara.backends.hardware` -- the *device* registry (which machine the
  variational loop runs on);
* :mod:`~carcara.backends.providers` -- the *circuit provider* registry (which
  SDK builds and executes the ansatz circuits: Qiskit, Amazon Braket or Cirq).
"""

from .hardware import (available_devices, is_simulator, normalize_device,
                       require_runnable)
from .providers import (BACKEND_PROVIDERS, BraketProvider, CircuitProvider,
                        CirqProvider, QiskitProvider, build_provider,
                        normalize_provider, provider_available)

__all__ = [
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
