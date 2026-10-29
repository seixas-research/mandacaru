"""Display and file-format views of molecular fields calculated in Rust."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NaturalOrbitals:
    occupations: np.ndarray
    coefficients: np.ndarray
    mo_coefficients: np.ndarray | None
    n_electrons: float


@dataclass(frozen=True)
class SamplingGrid:
    native: object
    units: str = "bohr"

    @property
    def shape(self):
        return tuple(self.native.shape)

    @property
    def step(self):
        return self.native.step_bohr

    @property
    def dV(self):
        return self.native.dv_bohr3

    @property
    def periodic(self):
        return self.native.periodic

    @cached_property
    def _coordinates(self):
        # Coordinate layout for plotting; field values are sampled in Rust.
        indices = np.indices(self.shape, dtype=float).reshape(3, -1)
        return (
            np.asarray(self.native.origin_bohr)[:, None] + self.step @ indices
        ).reshape((3,) + self.shape)

    @property
    def X(self):
        return self._coordinates[0]

    @property
    def Y(self):
        return self._coordinates[1]

    @property
    def Z(self):
        return self._coordinates[2]


@dataclass
class VolumetricField:
    quantity: str
    data: np.ndarray
    grid: SamplingGrid
    numbers: np.ndarray
    positions: np.ndarray
    charges: np.ndarray
    units: str
    integral: float
    norm: float
    n_electrons: float
    context: object
    index: int | None = None
    occupation: float | None = None
    component: str = "real"
    augmentation_charge: float = 0.0
    cell: object = None
    repeat: tuple = (1, 1, 1)
    output_path: str | None = None

    @property
    def origin(self):
        return np.asarray(self.grid.native.origin_bohr)

    def label(self):
        name = self.quantity.replace("_", " ")
        return name if self.index is None else f"{name} #{self.index}"

    def comment(self):
        text = f"Mandacaru | {self.label()} [{self.units}] | integral={self.integral:.8f} | norm={self.norm:.8f}"
        if self.augmentation_charge:
            text += f" | PAW augmentation={self.augmentation_charge:.8f} electrons (not in this smooth field)"
        return text

    def summary(self):
        return f"{self.label()} [{self.units}]\nintegral: {self.integral:.8f}\nnorm: {self.norm:.8f}"

    def write(self, path, format=None, comment=None):
        from ..utils.cube import detect_format

        self.context.write_field(
            str(path),
            np.ascontiguousarray(self.data),
            self.origin.tolist(),
            self.grid.step.tolist(),
            self.numbers.tolist(),
            self.positions.tolist(),
            format=detect_format(path, format),
            charges=self.charges.tolist(),
            comment=self.comment() if comment is None else str(comment),
        )
        self.output_path = str(Path(path))
        return self.output_path
