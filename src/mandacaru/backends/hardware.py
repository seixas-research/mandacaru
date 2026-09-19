# -*- coding: utf-8 -*-
# file: backends/hardware.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Execution devices for the variational loop.

Mandacaru computes energies on an **exact state-vector backend** today; this module
is the small registry that lets the user *name* the device an
:class:`~mandacaru.algorithms.ADAPTVQE` should run on, so the same driver can later
dispatch to a shot-based simulator or real quantum hardware without changing its
public API.

Supported names
---------------
``"AER_simulator"``
    The default: an ideal (noiseless) simulator.  In the current build this is
    the exact state-vector backend; it is the device used by the tests and
    examples.
``"ibm-quantum"``
    Real IBM Quantum hardware via **Qiskit Runtime**: the least-busy
    operational QPU of your account (or of the ``instance`` named in the
    driver's ``backend_options``).  Runnable with ``shots > 0`` through
    :class:`~mandacaru.backends.providers.QiskitProvider`; a specific processor
    is named directly, e.g. ``"ibm_torino"`` (any ``ibm_*`` name is accepted
    verbatim).  Billed to your IBM Quantum account.
``"fake_*"``
    A Qiskit Runtime **fake backend** (``qiskit_ibm_runtime.fake_provider``,
    e.g. ``"fake_manila"``): the real IBM execution path -- transpilation to
    that processor's gate set and coupling map, the Runtime ``SamplerV2`` --
    run locally.  The way to rehearse a hardware run without a queue or a bill.
``"braket-local"``
    Amazon Braket's local state-vector simulator, driven by
    :class:`~mandacaru.backends.providers.BraketProvider`.
``"braket-sv1"`` / ``"braket-dm1"`` / ``"braket-tn1"``
    Amazon Braket's **managed** on-demand simulators, run through the AWS
    service.  Runnable, billed to your AWS account.
``"braket-ionq-aria"`` / ``"braket-ionq-forte"`` / ``"braket-iqm-garnet"`` / ``"braket-rigetti-ankaa"``
    Real QPUs on Amazon Braket.  Runnable via
    :class:`~mandacaru.backends.providers.BraketProvider` with ``shots > 0`` --
    see :func:`device_arn` and :mod:`mandacaru.backends.measurement`.

Any string starting with ``arn:aws:braket``, ``ibm_`` or ``fake_`` is also
accepted verbatim, so a device that post-dates this release can still be named.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Prefix identifying an Amazon Braket device ARN.
AWS_ARN_PREFIX = "arn:aws:braket"
#: Prefix of a named IBM Quantum processor (``ibm_torino``, ``ibm_fez``, ...).
IBM_PREFIX = "ibm_"
#: Prefix of a Qiskit Runtime fake backend (``fake_manila``, ``fake_torino``).
FAKE_PREFIX = "fake_"


@dataclass(frozen=True)
class Device:
    """One entry of the device registry."""

    name: str
    simulator: bool          # classically simulable (vs. real QPU)
    runnable: bool           # wired to an execution path in this build
    provider: str | None     # circuit provider that drives it
    arn: str | None = None   # Amazon Braket ARN, for AWS devices
    description: str = ""
    #: Qubit capacity of a real QPU (``None`` for simulators and for devices
    #: whose size is not fixed by the name, e.g. the least-busy IBM entry).
    qubits: int | None = None

    @property
    def is_aws(self) -> bool:
        return self.arn is not None

    @property
    def is_ibm(self) -> bool:
        """True for IBM Quantum hardware (the least-busy entry or a named QPU)."""
        return self.provider == "qiskit" and not self.simulator

    @property
    def is_fake(self) -> bool:
        """True for a Qiskit Runtime fake backend (local rehearsal of the IBM path)."""
        return self.name.startswith(FAKE_PREFIX)


# Canonical device name -> Device.
_DEVICES: dict[str, Device] = {
    d.name: d for d in (
        Device("AER_simulator", True, True, None,
               description="ideal state-vector simulator (default)"),
        Device("ibm-quantum", False, True, "qiskit",
               description="IBM Quantum hardware via Qiskit Runtime "
                           "(least-busy operational QPU of your account)"),
        # -- Amazon Braket --------------------------------------------------- #
        Device("braket-local", True, True, "braket",
               description="Braket local state-vector simulator"),
        Device("braket-sv1", True, True, "braket",
               arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
               description="Braket managed state-vector simulator (SV1)"),
        Device("braket-dm1", True, True, "braket",
               arn="arn:aws:braket:::device/quantum-simulator/amazon/dm1",
               description="Braket managed density-matrix simulator (DM1)"),
        Device("braket-tn1", True, True, "braket",
               arn="arn:aws:braket:::device/quantum-simulator/amazon/tn1",
               description="Braket managed tensor-network simulator (TN1)"),
        Device("braket-ionq-aria", False, True, "braket",
               arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
               description="IonQ Aria-1 trapped-ion QPU (25 qubits)",
               qubits=25),
        Device("braket-ionq-forte", False, True, "braket",
               arn="arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
               description="IonQ Forte-1 trapped-ion QPU (36 qubits)",
               qubits=36),
        Device("braket-iqm-garnet", False, True, "braket",
               arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
               description="IQM Garnet superconducting QPU (20 qubits)",
               qubits=20),
        Device("braket-rigetti-ankaa", False, True, "braket",
               arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
               description="Rigetti Ankaa-3 superconducting QPU (84 qubits)",
               qubits=84),
    )
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
    "braket": "braket-local",
    "braket_local": "braket-local",
    "sv1": "braket-sv1",
    "braket_sv1": "braket-sv1",
    "dm1": "braket-dm1",
    "tn1": "braket-tn1",
    "ionq": "braket-ionq-aria",
    "braket-ionq": "braket-ionq-aria",
    "iqm": "braket-iqm-garnet",
    "braket-iqm": "braket-iqm-garnet",
    "rigetti": "braket-rigetti-ankaa",
    "braket-rigetti": "braket-rigetti-ankaa",
}


def available_devices() -> list[str]:
    """Canonical device names understood by :func:`normalize_device`."""
    return list(_DEVICES)


def describe_devices() -> list[Device]:
    """The full registry, for listing what a build can target."""
    return list(_DEVICES.values())


def normalize_device(name: str) -> str:
    """Resolve a device name/alias to its canonical form.

    A raw Amazon Braket ARN is returned unchanged (mapped to its registered name
    when it is one this build knows), so devices added by AWS after this release
    can still be named.

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
    if key.startswith(AWS_ARN_PREFIX):
        for device in _DEVICES.values():
            if device.arn == key:
                return device.name
        return key                      # an ARN this build does not know by name
    if key.lower().startswith((IBM_PREFIX, FAKE_PREFIX)):
        return key.lower()              # a named IBM processor / fake backend
    raise ValueError(
        f"unknown device {name!r}; available: {', '.join(_DEVICES)} "
        "(or any 'arn:aws:braket...' device ARN, an IBM processor name such as "
        "'ibm_torino', or a Qiskit Runtime fake backend such as 'fake_manila')")


def get_device(name: str) -> Device:
    """The :class:`Device` record for ``name`` (synthesized for unknown ARNs)."""
    canon = normalize_device(name)
    device = _DEVICES.get(canon)
    if device is not None:
        return device
    if canon.startswith(FAKE_PREFIX):
        return Device(canon, simulator=True, runnable=True, provider="qiskit",
                      description="Qiskit Runtime fake backend (local rehearsal "
                                  "of the IBM execution path)")
    if canon.startswith(IBM_PREFIX):
        return Device(canon, simulator=False, runnable=True, provider="qiskit",
                      description="IBM Quantum processor via Qiskit Runtime "
                                  "(by name)")
    # An Amazon Braket ARN this build has no entry for: assume a QPU (the
    # conservative reading -- it is billed and shot-based either way).
    return Device(canon, simulator=False, runnable=True, provider="braket",
                  arn=canon, description="Amazon Braket device (by ARN)")


def device_arn(name: str) -> str | None:
    """The Amazon Braket ARN of ``name``, or ``None`` if it is not an AWS device."""
    return get_device(name).arn


def device_qubits(name: str) -> int | None:
    """Qubit capacity of ``name``, or ``None`` when it is not fixed by the name.

    Simulators are bounded by memory rather than a qubit register, and the
    ``"ibm-quantum"`` label (least-busy) does not pin one processor, so both
    return ``None``; the registered Braket QPUs report their register size.
    Used by the dry-run qubit estimate to say whether a problem fits a device.
    """
    return get_device(name).qubits


def device_provider(name: str) -> str | None:
    """The circuit provider that drives ``name`` (``None`` = the internal one)."""
    return get_device(name).provider


def is_simulator(name: str) -> bool:
    """True when the (normalized) device is a classically simulable simulator."""
    return get_device(name).simulator


def is_aws_device(name: str) -> bool:
    """True when the device runs through the Amazon Braket service."""
    return get_device(name).is_aws


def is_ibm_device(name: str) -> bool:
    """True when the device is IBM Quantum hardware (Qiskit Runtime)."""
    return get_device(name).is_ibm


def is_fake_device(name: str) -> bool:
    """True when the device is a Qiskit Runtime fake backend (runs locally)."""
    return get_device(name).is_fake


def requires_shots(name: str) -> bool:
    """True when the device can only be run with ``shots > 0`` (every QPU).

    Real hardware never returns a state vector, so an energy has to be estimated
    from measurements -- see :mod:`mandacaru.backends.measurement`.
    """
    return not get_device(name).simulator and get_device(name).runnable


def require_runnable(name: str) -> str:
    """Return the canonical device, or raise if it cannot be executed.

    Every registered device is wired to an execution path in this build; the
    check is kept so a future reserved entry fails at *run* time with a clear
    message rather than deep inside a provider.
    """
    device = get_device(name)
    if not device.runnable:
        raise NotImplementedError(
            f"device {device.name!r} is not implemented yet; execution on "
            f"{device.description} is planned for a later release. Use "
            "'AER_simulator', an IBM Quantum device ('ibm-quantum', "
            "'ibm_torino', ...) or an Amazon Braket device "
            "(e.g. 'braket-sv1', 'braket-ionq-aria').")
    return device.name
