# -*- coding: utf-8 -*-
# file: backends/hardware.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Execution devices for the variational loop.

Carcará computes energies on an **exact state-vector backend** today; this module
is the small registry that lets the user *name* the device an
:class:`~carcara.algorithms.ADAPTVQE` should run on, so the same driver can later
dispatch to a shot-based simulator or real quantum hardware without changing its
public API.

Supported names
---------------
``"AER_simulator"``
    The default: an ideal (noiseless) simulator.  In the current build this is
    the exact state-vector backend; it is the device used by the tests and
    examples.
``"ibm-quantum"``
    Real IBM Quantum hardware (via Qiskit Runtime).  **Reserved** -- the
    dispatch is not implemented yet; naming it is accepted so code can be written
    against the final API, but running raises :class:`NotImplementedError`.
"""

from __future__ import annotations

# Canonical device name -> whether it is a (classically simulable) simulator.
_DEVICES: dict[str, bool] = {
    "AER_simulator": True,
    "ibm-quantum": False,
}

# Accepted spellings mapped to the canonical name.
_ALIASES: dict[str, str] = {
    "aer": "AER_simulator",
    "aer_simulator": "AER_simulator",
    "aer-simulator": "AER_simulator",
    "statevector": "AER_simulator",
    "simulator": "AER_simulator",
    "ibm": "ibm-quantum",
    "ibm_quantum": "ibm-quantum",
    "ibmq": "ibm-quantum",
}


def available_devices() -> list[str]:
    """Canonical device names understood by :func:`normalize_device`."""
    return list(_DEVICES)


def normalize_device(name: str) -> str:
    """Resolve a device name/alias to its canonical form.

    Raises
    ------
    ValueError
        If ``name`` is not a known device.
    """
    key = str(name).strip()
    if key in _DEVICES:
        return key
    canon = _ALIASES.get(key.lower())
    if canon is not None:
        return canon
    raise ValueError(
        f"unknown device {name!r}; available: {', '.join(_DEVICES)} "
        "(more will be added later)")


def is_simulator(name: str) -> bool:
    """True when the (normalized) device is a classically simulable simulator."""
    return _DEVICES[normalize_device(name)]


def require_runnable(name: str) -> str:
    """Return the canonical device, or raise if it cannot be executed yet.

    ``"ibm-quantum"`` (and any future real-hardware device) is accepted as a
    label but not yet wired to an execution path, so attempting to *run* on it
    raises :class:`NotImplementedError` with a clear message.
    """
    canon = normalize_device(name)
    if not is_simulator(canon):
        raise NotImplementedError(
            f"device {canon!r} is not implemented yet; real-hardware execution "
            "(IBM Quantum) is planned for a later release. Use 'AER_simulator'.")
    return canon
