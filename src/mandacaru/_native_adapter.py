"""ASE-to-engine conversion at the Python boundary."""

from __future__ import annotations


def geometry_from_atoms(atoms):
    """Snapshot an ASE geometry into Rust-owned storage, normalized to Bohr.

    No calculator, cache, or ASE object is retained by the native geometry.
    Dummy atoms are refused; the scientific ghost-basis specification remains
    a separate solver option.
    """
    import numpy as np
    from ._native import Geometry

    return Geometry(atoms.get_atomic_numbers().tolist(),
                    np.ascontiguousarray(atoms.get_positions(), dtype=np.float64),
                    np.ascontiguousarray(atoms.get_cell(), dtype=np.float64),
                    atoms.get_pbc().tolist(), "angstrom")
