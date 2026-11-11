"""The views of the symmetry, the irreducible zone and the band path (``analysis/kspace.py``)."""

import numpy as np

from mandacaru.analysis.kspace import CrystalSymmetry, IrreducibleZone, KPath


def test_the_symmetry_summary_names_the_group_and_its_operations():
    symmetry = CrystalSymmetry(
        number=123, international="P4/mmm", point_group="4/mmm", rotations=np.zeros((16, 3, 3)),
        translations=np.zeros((16, 3)), n_operations=16, n_pure_translations=1, symprec=1e-5)
    text = symmetry.summary()
    assert "P4/mmm" in text and "No. 123" in text and "4/mmm" in text and "16 operations" in text
    assert "pure translations" not in text
    supercell = CrystalSymmetry(
        number=123, international="P4/mmm", point_group="4/mmm", rotations=np.zeros((64, 3, 3)),
        translations=np.zeros((64, 3)), n_operations=64, n_pure_translations=4, symprec=1e-5)
    assert "4 pure translations" in supercell.summary()


def test_the_zone_summary_lists_every_irreducible_point_with_its_weight():
    zone = IrreducibleZone(
        size=(2, 2, 1), gamma=True, kpoints=np.array([[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]]),
        weights=np.array([0.25, 0.5, 0.25]), full=np.zeros((4, 3)), mapping=np.zeros(4, dtype=int),
        rotations=np.zeros((16, 3, 3)), space_group_operations=16, distinct_rotations=16)
    lines = zone.summary().splitlines()
    assert "2x2x1 Gamma-centered" in lines[0] and "3 of 4 points" in lines[0]
    assert len(lines) == 4 and "weight 0.500000" in lines[2]


def test_the_path_summary_reports_what_the_mesh_misses():
    raw = {"convention": "square", "kpoints": np.zeros((4, 3)), "mesh_index": [0, 1, 2, 0],
           "distances": np.arange(4.0), "labels": ["G", "X", "M", "G"],
           "label_distances": np.arange(4.0), "missing": ["X", "M"], "complete": False}
    path = KPath.from_native(raw)
    assert path.labels == ("G", "X", "M", "G") and not path.complete
    text = path.summary()
    assert "G-X-M-G" in text and "square" in text and "not on the mesh: X, M" in text
