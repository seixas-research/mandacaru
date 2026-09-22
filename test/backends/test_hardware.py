# -*- coding: utf-8 -*-
# file: test/backends/test_hardware.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The device registry: which machine a run is allowed to name.

:mod:`mandacaru.backends.hardware` maps a device name to its qubit capacity
and its simulator/QPU status, and refuses a name it does not know rather than
falling back to a simulator.
"""

import pytest

from mandacaru.algorithms import Mandacaru
from mandacaru.backends import (available_devices, is_simulator,
                                normalize_device)


class TestDeviceRegistry:
    def test_aer_is_default_and_simulator(self):
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO")
        assert adapt.device == "AER_simulator"
        assert is_simulator("AER_simulator")

    def test_aliases_normalize(self):
        assert normalize_device("aer") == "AER_simulator"
        assert normalize_device("statevector") == "AER_simulator"
        assert normalize_device("ibmq") == "ibm-quantum"

    def test_unknown_device_rejected(self):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", pool="ceo", device="quantum-thing")

    def test_ibm_quantum_listed_but_not_simulator(self):
        assert "ibm-quantum" in available_devices()
        assert not is_simulator("ibm-quantum")
