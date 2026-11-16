"""The calculator's real-time and dielectric-function requests: finding the converged reference,
the hand-off to the native propagator or response and the run-log block.

The numbers are Rust's (``molecular::realtime`` for a molecule, ``periodic::optics::realtime``
and ``periodic::optics::dielectric`` for a crystal); the option names are parsed there, so an
unknown ``direction``, ``kick``, ``gauge``, ``propagator`` or ``kernel`` raises ``ValueError`` naming the accepted values, a step above the
propagator's ``dt W`` bound raises ``ValueError`` naming ``time_step``, and an unsupported
reference raises ``NotImplementedError`` naming what is missing.
"""

from __future__ import annotations

import time

from .. import _native
from ..analysis.real_time import DielectricFunction, RealTime


def _reference(calculation):
    """The converged mean-field state, running the calculation if needed: a molecule's native
    mean field, or a crystal's native periodic result (``periodic=True``)."""
    if calculation.capabilities.quantum:
        raise ValueError(
            "get_real_time propagates a mean-field state: it needs method='dft', 'rhf' or 'uhf', "
            f"not the correlated quantum state of method={calculation.method!r}"
        )
    if calculation.result is None:
        atoms = calculation._attached_atoms
        if atoms is None:
            raise RuntimeError("run a calculation before asking for a real-time propagation")
        calculation.calculate(atoms, properties=("energy",))
    if calculation.result.periodic is not None:
        return calculation.result.periodic, True
    reference = calculation.result.reference
    if reference is None:
        raise ValueError("this calculation kept no mean-field state to propagate")
    return reference, False


def real_time(calculation, duration=40.0, time_step=0.005, *, direction="x", strength=0.002,
              kick="density", gauge=None, propagator="etrs", corrector_tolerance=1e-6,
              max_energy_drift=0.05, kernel="adiabatic", kpts=None):
    reference, periodic = _reference(calculation)
    started = time.perf_counter()
    run = _native.crystal_real_time if periodic else _native.molecular_real_time
    data = run(reference, direction=direction, strength=strength, time_step=time_step,
               duration=duration, atomic_units=calculation._atomic_units, kick=kick, gauge=gauge,
               kernel=kernel, propagator=propagator, corrector_tolerance=corrector_tolerance,
               max_energy_drift=max_energy_drift, kpts=None if kpts is None else tuple(kpts))
    seconds = time.perf_counter() - started
    view = RealTime(data, atomic_units=calculation._atomic_units,
                    context=calculation.result.context)
    from .reporting import write_real_time

    write_real_time(calculation, view, seconds=seconds)
    return view


def dielectric_function(calculation, energies=None, damping=0.1, **options):
    reference, periodic = _reference(calculation)
    if not periodic:
        raise ValueError("get_dielectric_function needs a crystal; a molecule has get_excitations")
    view = DielectricFunction(_native.crystal_dielectric_function(
        reference, energies=energies, damping=damping, atomic_units=calculation._atomic_units,
        **options))
    from .reporting import write_record

    write_record(calculation, "[DIELECTRIC FUNCTION]", view)
    return view
