# -*- coding: utf-8 -*-
# file: backends/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Execution backends (devices) and error mitigation."""

from .hardware import (available_devices, is_simulator, normalize_device,
                       require_runnable)

__all__ = [
    "available_devices",
    "normalize_device",
    "is_simulator",
    "require_runnable",
]
