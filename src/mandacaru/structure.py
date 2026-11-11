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
    #: One initial moment vector per atom (``(N, 3)``, Bohr magneton) when the moments were
    #: given as vectors (a noncollinear start), else ``None``; ``magnetic_moments`` is then zero.
    moment_vectors: np.ndarray | None = field(init=False, default=None)
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
        vectors = None
        if moments.ndim == 2 and moments.shape == (len(numbers), 3) and np.isfinite(moments).all():
            vectors, moments = moments, np.zeros(len(numbers))
        elif moments.shape != (len(numbers),) or not np.isfinite(moments).all():
            raise ValueError(
                "magnetic_moments must be one finite value, or one finite 3-vector, per atom"
            )
        for value in (positions, cell, moments) + ((vectors,) if vectors is not None else ()):
            value.flags.writeable = False
        object.__setattr__(self, "numbers", numbers)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "pbc", tuple(native.pbc))
        object.__setattr__(self, "magnetic_moments", moments)
        object.__setattr__(self, "moment_vectors", vectors)
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
            and (self.moment_vectors is None) == (other.moment_vectors is None)
            and (
                self.moment_vectors is None
                or np.array_equal(self.moment_vectors, other.moment_vectors)
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


def initial_moments(atoms, *, spinor=False, unrestricted=False):
    """The collinear initial magnetic moments of a crystal, one number per atom.

    A scalar-relativistic crystal with nonzero initial magnetic moments automatically
    calculates with spin polarization (``spin=True`` / ``unrestricted=True``). A spin-orbit crystal takes
    its moments as the start of the noncollinear magnetization and needs neither option.
    """
    moments = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    if moments.ndim != 1:
        raise NotImplementedError("native periodic SCF supports collinear initial moments")
    return moments
