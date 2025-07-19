import pytest
import torch
import tempfile
import os
from ase import Atoms
from ase.io import write
from carcara.atoms_dataset import AtomsDataset

def create_xyz_file(atoms_list, filename):
    """Helper to write a list of ASE Atoms objects to an XYZ file."""
    write(filename, atoms_list)

def test_init_raises_value_error_on_none():
    with pytest.raises(ValueError, match="Dataset cannot be None."):
        AtomsDataset(dataset=None)

def test_init_reads_single_structure(tmp_path):
    # Create a simple H2 molecule
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    xyz_file = tmp_path / "h2.xyz"
    create_xyz_file([atoms], xyz_file)
    ds = AtomsDataset(str(xyz_file))
    assert len(ds.dataset) == 1
    assert len(ds.dataset_proc) == 1
    proc = ds.dataset_proc[0]
    assert torch.allclose(proc['positions'], torch.tensor([[0, 0, 0], [0, 0, 0.74]], dtype=torch.float32))
    assert torch.equal(proc['atomic_numbers'], torch.tensor([1, 1], dtype=torch.int64))

def test_init_reads_multiple_structures(tmp_path):
    atoms1 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms2 = Atoms('CO', positions=[[0, 0, 0], [1.13, 0, 0]])
    xyz_file = tmp_path / "multi.xyz"
    create_xyz_file([atoms1, atoms2], xyz_file)
    ds = AtomsDataset(str(xyz_file))
    assert len(ds.dataset) == 2
    assert len(ds.dataset_proc) == 2
    # Check atomic numbers for both structures
    assert torch.equal(ds.dataset_proc[0]['atomic_numbers'], torch.tensor([1, 1]))
    assert torch.equal(ds.dataset_proc[1]['atomic_numbers'], torch.tensor([6, 8]))

def test_init_raises_on_empty_structure(tmp_path):
    # Create an empty Atoms object
    atoms = Atoms()
    xyz_file = tmp_path / "empty.xyz"
    create_xyz_file([atoms], xyz_file)
    with pytest.raises(ValueError, match="Structure must contain at least one atom."):
        AtomsDataset(str(xyz_file))