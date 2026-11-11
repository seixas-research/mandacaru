"""Views of the crystal symmetry, the irreducible zone and a band path (all computed in Rust)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CrystalSymmetry:
    """The space group of a structure: its type and its operations ``x -> W x + t``.

    ``rotations`` (``(n, 3, 3)``, integers) and ``translations`` (``(n, 3)``) are in fractional
    coordinates of the cell. A cell with a non-periodic axis is read as the crystal that repeats
    along it, so a slab in vacuum reports the group of the stacked crystal.
    """

    number: int
    international: str
    point_group: str
    rotations: np.ndarray = field(repr=False)
    translations: np.ndarray = field(repr=False)
    n_operations: int
    n_pure_translations: int
    symprec: float

    def summary(self):
        """One line: the space group, its point group and the number of operations."""
        pure = "" if self.n_pure_translations == 1 else (
            f", {self.n_pure_translations} pure translations (a supercell)")
        return (f"Space group {self.international} (No. {self.number}), point group "
                f"{self.point_group}: {self.n_operations} operations{pure}, symprec "
                f"{self.symprec:g}")


@dataclass(frozen=True)
class IrreducibleZone:
    """A k-mesh reduced by the point group of the crystal and time reversal.

    ``kpoints`` are the irreducible points (fractional) and ``weights`` the share of the full
    mesh each one stands for (they add up to 1); ``full`` is the whole mesh and ``mapping`` the
    irreducible point of each of its points. ``rotations`` are the rotation parts of the space
    group that map the mesh onto itself, the group the reduction used; ``space_group_operations``
    and ``distinct_rotations`` count the operations of the space group before that filter.
    """

    size: tuple
    gamma: bool
    kpoints: np.ndarray
    weights: np.ndarray
    full: np.ndarray = field(repr=False)
    mapping: np.ndarray = field(repr=False)
    rotations: np.ndarray = field(repr=False)
    space_group_operations: int = 0
    distinct_rotations: int = 0

    def summary(self):
        """The reduction and, one line per point, the irreducible k-points with their weights."""
        n1, n2, n3 = self.size
        kind = "Gamma-centered" if self.gamma else "Monkhorst-Pack"
        lines = [
            f"Irreducible zone of the {n1}x{n2}x{n3} {kind} mesh: {len(self.kpoints)} of "
            f"{len(self.full)} points; {len(self.rotations)} of {self.distinct_rotations} "
            "rotations preserve the mesh, with time reversal"
        ]
        for k, w in zip(self.kpoints, self.weights):
            lines.append(f"  k = ({k[0]:+.4f}, {k[1]:+.4f}, {k[2]:+.4f})  weight {w:.6f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class KPath:
    """The mesh points on a band path, in path order.

    ``kpoints`` are fractional, ``distances`` the position of each along the path (1/Angstrom,
    ``2 pi`` included). ``labels`` and ``label_distances`` are the path's vertices, a jump
    written ``"X|M"``; ``missing`` lists the vertices that are no mesh point and ``complete``
    says there are none. ``mesh_index`` is the point of the calculation's mesh each one is an
    image of, and ``convention`` the lattice class the named points came from.
    """

    convention: str
    kpoints: np.ndarray
    mesh_index: np.ndarray = field(repr=False)
    distances: np.ndarray
    labels: tuple
    label_distances: np.ndarray
    missing: tuple
    complete: bool

    @classmethod
    def from_native(cls, raw):
        return cls(
            convention=str(raw["convention"]),
            kpoints=np.asarray(raw["kpoints"]),
            mesh_index=np.asarray(raw["mesh_index"]),
            distances=np.asarray(raw["distances"]),
            labels=tuple(raw["labels"]),
            label_distances=np.asarray(raw["label_distances"]),
            missing=tuple(raw["missing"]),
            complete=bool(raw["complete"]),
        )

    def summary(self):
        text = (f"Band path {'-'.join(self.labels)} ({self.convention} lattice): "
                f"{len(self.kpoints)} mesh points")
        if self.missing:
            text += f"; not on the mesh: {', '.join(self.missing)}"
        return text
