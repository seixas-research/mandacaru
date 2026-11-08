"""Test helper: the native space-group detection (``_native.crystal_symmetry``) of an ASE ``Atoms``.

The detection is Rust (``periodic::space_group``); this only hands the cell, fractional positions and
atomic numbers over and wraps the dictionary it returns, for ``test_space_group.py`` and
``test_space_group_number.py``.
"""
from types import SimpleNamespace

import numpy as np

from mandacaru import _native

DEFAULT_SYMPREC = 1e-5


def crystal_symmetry(atoms, symprec: float = DEFAULT_SYMPREC):
    raw = _native.crystal_symmetry(
        np.ascontiguousarray(atoms.cell[:], dtype=np.float64),
        np.ascontiguousarray(atoms.get_scaled_positions(), dtype=np.float64),
        np.ascontiguousarray(atoms.get_atomic_numbers(), dtype=np.int64), float(symprec))
    rotations = np.asarray(raw["rotations"], dtype=int)
    return SimpleNamespace(
        number=int(raw["number"]), international=str(raw["international"]),
        point_group=str(raw["point_group"]), rotations=rotations,
        translations=np.asarray(raw["translations"], dtype=float),
        symprec=float(raw["symprec"]), symprec_used=float(raw["symprec_used"]),
        n_pure_translations=int(raw["n_pure_translations"]), n_operations=len(rotations))
