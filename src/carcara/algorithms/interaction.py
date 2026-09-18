# -*- coding: utf-8 -*-
# file: algorithms/interaction.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Interaction energies between fragments, on one shared grid.

The interaction energy of a complex is the energy difference

.. math::

    E_\text{int} = E(\text{AB}) - E(\text{A}) - E(\text{B}),

and on a real-space grid the one thing that decides whether that difference
means anything is that **all three energies are evaluated on the same grid,
with every atom at the same position relative to the grid nodes**.  The
calculator normally re-centres its box on whatever geometry it is given, so a
fragment computed on its own sits differently on the grid than it does inside
the complex; for an atom with a sharp core that alone can shift the energy by
electronvolts (hundreds, for a sodium ion at a 0.3 Angstrom spacing), and the
"interaction energy" then measures the grid, not the chemistry.

:func:`interaction_energy` therefore builds the grid **once**, from the
complex, and evaluates the complex and every fragment on it, each fragment
being the complex with the other atoms deleted -- same coordinates, same box,
same spacing, same Coulomb softening.  Fragment charges and spin states are
given per fragment (the total charge must add up); the same variational
method and options are used throughout.  ``method="rhf"`` gives the
mean-field (Hartree-Fock) interaction energy without any circuit, a cheap way
to check the setup before the variational runs.

.. code-block:: python

    from carcara.algorithms import interaction_energy

    result = interaction_energy(complex_atoms, fragments=[[0, 1, 2], [3]],
                                charges=[0, 1], method="adapt-vqe",
                                basis="FAO", frozen_core=True, h=0.25)
    result.energy            # eV (Hartree with atomic_units=True)
    result.in_units("Ha")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import convert_energy, energy_unit_label, from_hartree


@dataclass
class InteractionEnergy:
    """``E(complex) - sum E(fragments)`` and everything that went into it.

    Every energy is in :attr:`energy_unit` -- **eV** by default, Hartree when
    ``atomic_units=True`` was passed to the solver; :meth:`in_units` converts.
    """

    energy: float                       # E(complex) - sum E(fragments)
    complex_energy: float
    fragment_energies: list[float]      # one per fragment
    fragments: list[list[int]]          # atom indices of each fragment
    charges: list[int]
    method: str
    grid: object                        # the shared Grid
    results: list = field(default_factory=list)   # per-run result objects
    complex_result: object = None
    energy_unit: str = "eV"             # unit of every energy above

    def in_units(self, units: str = "eV") -> float:
        """The interaction energy converted to ``units`` (``"eV"`` or ``"Ha"``)."""
        return float(convert_energy(self.energy, self.energy_unit, units))

    def __repr__(self) -> str:
        return (f"InteractionEnergy(E_int={self.energy:+.6f} {self.energy_unit}, "
                f"method={self.method!r}, fragments={self.fragments})")


def _check_fragments(atoms, fragments, charges, charge):
    n = len(atoms)
    frags = [sorted(int(i) for i in f) for f in fragments]
    if len(frags) < 2:
        raise ValueError("an interaction energy needs at least two fragments")
    seen: list[int] = []
    for f in frags:
        if not f:
            raise ValueError("a fragment cannot be empty")
        if any(i < 0 or i >= n for i in f):
            raise ValueError(f"fragment {f} has atom indices outside 0..{n - 1}")
        seen.extend(f)
    if sorted(seen) != list(range(n)):
        raise ValueError(
            "the fragments must partition the complex: every atom exactly "
            f"once (got {sorted(seen)} for {n} atoms)")
    if charges is None:
        charges = [0] * len(frags)
        charges[0] = int(charge)
    charges = [int(c) for c in charges]
    if len(charges) != len(frags):
        raise ValueError("one charge per fragment is required")
    if sum(charges) != int(charge):
        raise ValueError(
            f"fragment charges {charges} sum to {sum(charges)}, not the "
            f"complex charge {charge}")
    return frags, charges


def _shared_grid(atoms, h, grid):
    if grid is not None:
        return grid
    from ._hamiltonian_from_atoms import grid_from_cell
    return grid_from_cell(atoms, h)


def _rhf_energy(atoms, charge, grid, h, basis, frozen_core, frozen_orbitals,
                spin, kinetic=None):
    """Hartree-Fock total energy of ``atoms`` on ``grid`` (Hartree)."""
    from ._hamiltonian_from_atoms import build_basis_hamiltonian
    _H, particles, _n, _profile, context = build_basis_hamiltonian(
        atoms, basis, grid, h, charge, None, spin=spin,
        frozen_core=frozen_core, frozen_orbitals=frozen_orbitals,
        kinetic=kinetic)
    if context is None:
        raise NotImplementedError(
            "interaction energies need an atom-centered basis")
    integrals = context["integrals"]
    n_el = int(context["n_electrons"])
    if n_el == 0:
        electronic = 0.0
    elif n_el % 2 == 0 and particles[0] == particles[1]:
        electronic = integrals.hartree_fock(n_el).electronic_energy
    else:
        electronic = integrals.open_shell_hartree_fock(
            *particles).electronic_energy
    return float(electronic + integrals.nuclear_repulsion)


def interaction_energy(atoms, fragments, charges=None, *, charge: int = 0,
                       method: str = "adapt-vqe", basis="FAO",
                       h: float = 0.20, grid=None,
                       **solver_kwargs) -> InteractionEnergy:
    """``E(complex) - sum_i E(fragment_i)`` with every energy on one grid.

    Parameters
    ----------
    atoms : ase.Atoms
        The complex.  Its cell sizes the shared grid (define it in the
        geometry, e.g. ``atoms.center(vacuum=4.0)``); the magnetic moments carried by the atoms select the spin
        state of the complex and of each fragment.
    fragments : sequence of sequences of int
        Atom indices of each fragment; together they must cover every atom of
        the complex exactly once.
    charges : sequence of int, optional
        Charge of each fragment, summing to ``charge``.  Default: the complex
        charge on the first fragment, zero on the others.
    charge : int
        Charge of the complex.
    method : str
        Any :class:`~carcara.algorithms.calculator.Carcara` method, or
        ``"rhf"`` for the mean-field (Hartree-Fock) interaction energy with no
        circuit at all.
    basis, h, grid, **solver_kwargs :
        Forwarded to every run (``basis`` may be a pseudopotential family --
        ``"NCPP"`` / ``"ONCVPSP"`` / ``"PAW"`` -- like anywhere else).  ``grid``
        overrides the shared grid built from the complex.

    Returns
    -------
    InteractionEnergy
    """
    from ..integrals import Grid  # noqa: F401  (documents the shared object)

    frags, frag_charges = _check_fragments(atoms, fragments, charges, charge)
    shared = _shared_grid(atoms, h, grid)
    method_key = str(method).strip().lower()
    # Output units follow the solver's convention (eV unless atomic_units).
    unit = energy_unit_label("Ha" if solver_kwargs.get("atomic_units") else "eV")

    pieces = [(list(range(len(atoms))), int(charge))] + list(zip(frags, frag_charges))
    energies: list[float] = []
    results: list = []
    for indices, q in pieces:
        sub = atoms[indices]
        sub.set_cell(atoms.get_cell())
        sub.set_pbc(atoms.get_pbc())
        if method_key in ("rhf", "hf", "hartree-fock"):
            energy = _rhf_energy(
                sub, q, shared, h, basis,
                solver_kwargs.get("frozen_core", False),
                solver_kwargs.get("frozen_orbitals"),
                solver_kwargs.get("spin", False),
                solver_kwargs.get("kinetic"))
            energy = float(from_hartree(energy, unit))   # RHF works in Hartree
            results.append(None)
        else:
            from .calculator import Carcara
            calc = Carcara(method=method,
                           basis=basis,
                           h=h,
                           grid=shared,
                           charge=q,
                           **solver_kwargs)
            sub.calc = calc
            sub.get_potential_energy()
            energy = float(calc.result.optimal_energy)   # already in `unit`
            results.append(calc.result)
        energies.append(energy)

    e_complex, e_frags = energies[0], energies[1:]
    return InteractionEnergy(
        energy=float(e_complex - sum(e_frags)), complex_energy=e_complex,
        fragment_energies=e_frags, fragments=frags, charges=frag_charges,
        method=method_key, grid=shared, results=results[1:],
        complex_result=results[0], energy_unit=unit)
