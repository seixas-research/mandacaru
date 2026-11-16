"""The calculator's excited-state request: finding the converged reference, the hand-off to the
native linear-response solver and the run-log block.

The numbers are Rust's (``molecular::excitations`` for a molecule, ``periodic::optics::bse`` for a
crystal); the option names are parsed there too, so an unknown ``approximation``, ``spin`` or
``kernel`` raises ``ValueError`` naming the accepted values, and an unsupported reference raises
``NotImplementedError`` naming what is missing.
"""

from __future__ import annotations

from .. import _native
from ..analysis.excitations import CrystalExcitations, Excitations


def _reference(calculation):
    """The native mean-field state of a converged molecule, or a crystal's native periodic result,
    running the calculation if needed."""
    if calculation.capabilities.quantum:
        raise ValueError(
            "get_excitations is the linear response of a mean-field state: it needs method='dft', "
            f"'rhf' or 'uhf' (closed or open shell), not the correlated quantum state of method={calculation.method!r} "
            "(energy_levels and the subspace methods give the excited states of a quantum run)"
        )
    if calculation.result is None:
        atoms = calculation._attached_atoms
        if atoms is None:
            raise RuntimeError("run a calculation before asking for excitations")
        calculation.calculate(atoms, properties=("energy",))
    if calculation.result.periodic is not None:
        return calculation.result.periodic
    reference = calculation.result.reference
    if reference is None:
        raise ValueError("this calculation kept no mean-field state to respond")
    return reference


def excitations(calculation, num_states=5, *, approximation="tda", spin=None,
                kernel="adiabatic", tolerance=1e-5, max_iterations=200, **crystal):
    from .reporting import write_excitations, write_record

    reference = _reference(calculation)
    if calculation.result.periodic is not None:
        view = CrystalExcitations(_native.crystal_excitations(
            reference, num_states=num_states, approximation=approximation, spin=spin, kernel=kernel,
            tolerance=tolerance, max_iterations=max_iterations, atomic_units=calculation._atomic_units, **crystal))
        write_record(calculation, "[EXCITED STATES]", view)
        return view
    native = _native.molecular_excitations(
        reference, num_states=num_states, approximation=approximation, spin=spin,
        kernel=kernel or "adiabatic", tolerance=tolerance, max_iterations=max_iterations, **crystal,
    )
    view = Excitations(native, atomic_units=calculation._atomic_units,
                       context=calculation.result.context)
    write_excitations(calculation, view, tolerance=tolerance)
    return view

