"""Geometry values independent of ASE and any execution service."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import _native


@dataclass(frozen=True, eq=False)
class Geometry:
    numbers: tuple[int, ...]
    positions: np.ndarray
    cell: np.ndarray
    pbc: tuple[bool, bool, bool] = (False, False, False)
    units: str = "angstrom"
    magnetic_moments: np.ndarray | None = None
    _native_value: object = field(init=False, repr=False)

    def __post_init__(self):
        positions = np.array(self.positions, dtype=float, copy=True, order="C")
        cell = np.array(self.cell, dtype=float, copy=True, order="C")
        numbers = tuple(self.numbers)
        native = _native.Geometry(numbers, positions, cell, self.pbc, self.units)
        moments = (
            np.zeros(len(numbers))
            if self.magnetic_moments is None
            else np.array(self.magnetic_moments, dtype=float, copy=True)
        )
        if moments.shape != (len(numbers),) or not np.isfinite(moments).all():
            raise ValueError("magnetic_moments must be one finite value per atom")
        for value in (positions, cell, moments):
            value.flags.writeable = False
        object.__setattr__(self, "numbers", numbers)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "pbc", tuple(native.pbc))
        object.__setattr__(self, "magnetic_moments", moments)
        object.__setattr__(self, "_native_value", native)

    def native(self):
        return self._native_value

    def same_as(self, other):
        return (
            isinstance(other, Geometry)
            and self.numbers == other.numbers
            and self.units == other.units
            and self.pbc == other.pbc
            and all(
                np.array_equal(a, b)
                for a, b in [
                    (self.positions, other.positions),
                    (self.cell, other.cell),
                    (self.magnetic_moments, other.magnetic_moments),
                ]
            )
        )


def geometry_value(value):
    if isinstance(value, Geometry):
        return value
    if isinstance(value, _native.Geometry):
        return Geometry(
            tuple(value.numbers),
            value.positions_bohr,
            value.cell_bohr,
            tuple(value.pbc),
            "bohr",
        )
    # Geometry protocol: accessing an ASE object does not import ASE.
    if all(
        hasattr(value, k)
        for k in ("get_atomic_numbers", "get_positions", "get_cell", "get_pbc")
    ):
        return Geometry(
            tuple(map(int, value.get_atomic_numbers())),
            value.get_positions(),
            value.get_cell(),
            tuple(map(bool, value.get_pbc())),
            magnetic_moments=value.get_initial_magnetic_moments(),
        )
    raise TypeError(
        "geometry must be a Geometry, native Geometry, or an object implementing the geometry protocol"
    )
