"""The calculator side of the crystal symmetry, the irreducible zone and band paths.

The space group, the reduction of the k-mesh, the band-path conventions and the selection of mesh
points are computed in Rust (``periodic::space_group``, ``periodic::kpath``). This module reads
the structure and the mesh off the calculator, hands arrays across and wraps the results.
"""

from __future__ import annotations

import numpy as np

from .. import _native
from ..analysis.kspace import CrystalSymmetry, IrreducibleZone, KPath


def structure(calculation):
    """``(cell, pbc, scaled positions, atomic numbers)`` of the calculation's structure."""
    atoms = calculation.atoms
    if atoms is None:
        atoms = getattr(calculation, "_attached_atoms", None)
    if atoms is None:
        raise RuntimeError(
            "the calculation has no structure: attach it with `atoms.calc = calculation` or run it"
        )
    return (
        np.ascontiguousarray(atoms.cell.array, dtype=float),
        tuple(bool(p) for p in atoms.pbc),
        np.ascontiguousarray(atoms.get_scaled_positions(wrap=False), dtype=float),
        np.ascontiguousarray(atoms.numbers, dtype=np.int64),
    )


def mesh(calculation, pbc):
    """``(size, gamma)`` of the calculation's ``kpts``, checked against the periodic axes."""
    spec = calculation._kpts_spec
    size = tuple(int(n) for n in spec.get("size", (1, 1, 1)))
    for axis, (periodic, n) in enumerate(zip(pbc, size)):
        if not periodic and n != 1:
            raise ValueError(
                f"kpts has {n} points along lattice vector {'abc'[axis]}, which is not "
                "periodic: use 1 there"
            )
    return size, bool(spec.get("gamma", False))


def symmetry(calculation, symprec=1e-5):
    """The space group of the calculation's structure."""
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be positive and finite")
    cell, _, positions, numbers = structure(calculation)
    raw = _native.crystal_symmetry(cell, positions, numbers, float(symprec),
                                   calculation.context)
    return CrystalSymmetry(
        number=int(raw["number"]),
        international=str(raw["international"]),
        point_group=str(raw["point_group"]),
        rotations=np.asarray(raw["rotations"]),
        translations=np.asarray(raw["translations"]),
        n_operations=int(raw["n_operations"]),
        n_pure_translations=int(raw["n_pure_translations"]),
        symprec=float(raw["symprec"]),
    )


def irreducible_zone(calculation, symprec=1e-5):
    """The calculation's k-mesh reduced by the crystal's point group and time reversal."""
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be positive and finite")
    cell, pbc, positions, numbers = structure(calculation)
    size, gamma = mesh(calculation, pbc)
    raw = _native.irreducible_zone(
        calculation.context, cell, positions, numbers, list(size), gamma=gamma,
        symprec=float(symprec))
    return IrreducibleZone(
        size=size,
        gamma=gamma,
        kpoints=np.asarray(raw["points"]),
        weights=np.asarray(raw["weights"]),
        full=np.asarray(raw["full"]),
        mapping=np.asarray(raw["mapping"]),
        rotations=np.asarray(raw["rotations"]),
        space_group_operations=int(raw["space_group_operations"]),
        distinct_rotations=int(raw["distinct_rotations"]),
    )


def path_arguments(path):
    """``(text, points)`` for the native path builders: a string of labels, or a sequence of
    fractional points, each optionally paired with a label as ``(label, point)``."""
    if path is None or isinstance(path, str):
        return path, None
    points = []
    for entry in path:
        if (len(entry) == 2 and isinstance(entry[0], str)):
            label, point = entry
        else:
            label, point = "", entry
        point = [float(x) for x in point]
        if len(point) != 3:
            raise ValueError("a band-path point is a fractional (k1, k2, k3)")
        points.append((label, point))
    return None, points


def band_path(calculation, path=None):
    """The mesh points of the calculation's ``kpts`` on a band path.

    ``path`` is a string of the lattice's high-symmetry labels (``"GXMG"``; a comma jumps), a
    sequence of fractional points, or ``None`` for the lattice's default path.
    """
    cell, pbc, _, _ = structure(calculation)
    size, gamma = mesh(calculation, pbc)
    text, points = path_arguments(path)
    raw = _native.band_path(
        calculation.context, cell, list(pbc), list(size), gamma=gamma, path=text, points=points)
    return KPath.from_native(raw)
