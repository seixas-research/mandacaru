"""The ``setup`` block of a crystal snapshot (``integrations/setup_block.py``): what the Rust derivative engines rebuild from.

The block is written by the Python setup next to the ``sites`` block and parsed and checked in Rust
(``periodic::description``); here is its content, read from a snapshot a one-iteration run saved (the SCF is not needed).
Needs ``MANDACARU_PAW_PATH``.
"""
import json
import os

import pytest
from ase import Atoms
from ase.build import bulk

from mandacaru import Mandacaru

pytestmark = pytest.mark.skipif(not os.environ.get("MANDACARU_PAW_PATH"), reason="MANDACARU_PAW_PATH is not set")

BASIS = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}


def snapshot(atoms, path, **options):
    atoms.calc = Mandacaru(method="dft", basis=BASIS, trace=False, save_crystal=path,
                           periodic_scf_options={"max_iterations": 1}, **options)
    try:
        atoms.get_potential_energy()
    except Exception:  # noqa: BLE001 - one iteration does not converge; the snapshot is saved before the SCF
        if not path.exists():
            raise
    return json.loads(path.read_text())


def test_the_block_describes_the_basis_the_projectors_and_the_ion_width(tmp_path):
    atoms = Atoms("LiH", cell=[6.0, 6.2, 6.4], pbc=True, positions=[[2.0, 2.1, 2.2], [3.5, 2.2, 2.0]])
    payload = snapshot(atoms, tmp_path / "lih.json", grid_spacing=0.45, symmetry=False)
    block = payload["setup"]
    assert block["version"] == 1 and block["operations"] is None
    functions = block["functions"]
    assert len(functions["atom"]) == len(functions["m"]) == len(functions["table"]) == payload["m"]
    assert set(functions["atom"]) == {0, 1}
    assert max(functions["table"]) < len(block["tables"])
    for table in block["tables"]:
        assert len(table["r"]) == len(table["values"]) >= 4
        assert all(b > a for a, b in zip(table["r"], table["r"][1:]))
    assert len(block["projectors"]) == payload["p"]
    assert all(len(row) == 4 for row in block["projectors"])
    assert block["sigma_bohr"] > 0
    # The datasets, their digests and the centers are the sites block's, not repeated here.
    assert {"centers", "datasets", "projector_basis"} <= set(payload["sites"])
    assert not {"datasets", "centers"} & set(block)


def test_a_reduced_mesh_lists_its_operations_with_their_atom_maps(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    payload = snapshot(atoms, tmp_path / "al.json", grid_spacing=0.45, kpts={"size": (3, 3, 3), "gamma": True},
                       smearing={"method": "fermi-dirac", "width": 0.3}, symmetry=True)
    operations = payload["setup"]["operations"]
    count = len(payload["permutations"])
    assert count == 48
    assert len(operations["rotations"]) == len(operations["translations"]) == len(operations["atom_maps"]) == count
    assert all(len(w) == 9 and all(isinstance(x, int) for x in w) for w in operations["rotations"])
    assert all(sorted(m) == [0] for m in operations["atom_maps"])
