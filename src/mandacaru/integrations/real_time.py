"""The calculator's real-time request: finding the converged reference, the hand-off to the
native propagator and the run-log block.

The numbers are Rust's (``molecular::realtime``); the option names are parsed there, so an unknown
``kick``, ``gauge`` or ``propagator`` raises ``ValueError`` naming the accepted values, a step above
the propagator's ``dt W`` bound raises ``ValueError`` naming ``time_step``, and an unsupported
reference raises ``NotImplementedError`` naming what is missing.
"""

from __future__ import annotations

import time

from .. import _native
from ..analysis.real_time import RealTime

_CRYSTAL = (
    "real-time propagation of a crystal is not implemented yet: it runs on molecules (the "
    "crystal route, the velocity gauge with the macroscopic current, is planned)"
)

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _reference(calculation):
    """The native mean-field state of a converged molecule, running the calculation if needed."""
    if calculation.capabilities.quantum:
        raise ValueError(
            "get_real_time propagates a mean-field state: it needs method='dft', 'rhf' or 'uhf', "
            f"not the correlated quantum state of method={calculation.method!r}"
        )
    if calculation.result is None:
        atoms = calculation._attached_atoms
        if atoms is None:
            raise RuntimeError("run a calculation before asking for a real-time propagation")
        if any(atoms.pbc) or calculation._load_crystal is not None:
            raise NotImplementedError(_CRYSTAL)
        calculation.calculate(atoms, properties=("energy",))
    if calculation.result.periodic is not None:
        raise NotImplementedError(_CRYSTAL)
    reference = calculation.result.reference
    if reference is None:
        raise ValueError("this calculation kept no mean-field state to propagate")
    return reference


def _direction(direction):
    if isinstance(direction, str):
        if direction not in _AXES:
            raise ValueError(
                f"direction must be 'x', 'y', 'z' or a 3-vector, not {direction!r}")
        return _AXES[direction]
    values = tuple(float(v) for v in direction)
    if len(values) != 3:
        raise ValueError("direction must be 'x', 'y', 'z' or a 3-vector")
    return values


def real_time(calculation, duration=40.0, time_step=0.005, *, direction="x", strength=0.002,
              kick="density", gauge=None, propagator="etrs", corrector_tolerance=1e-6,
              max_energy_drift=0.05):
    reference = _reference(calculation)
    started = time.perf_counter()
    data = _native.molecular_real_time(
        reference, direction=_direction(direction), strength=strength, time_step=time_step,
        duration=duration, atomic_units=calculation._atomic_units, kick=kick,
        gauge="length" if gauge is None else gauge, propagator=propagator,
        corrector_tolerance=corrector_tolerance, max_energy_drift=max_energy_drift,
    )
    seconds = time.perf_counter() - started
    view = RealTime(data, atomic_units=calculation._atomic_units,
                    context=calculation.result.context)
    from .reporting import write_real_time

    write_real_time(calculation, view, seconds=seconds)
    return view
