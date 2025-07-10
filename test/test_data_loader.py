import os
import tempfile
import pytest
from ase import Atoms
from ase.io import write
from carcara.data_loader import DataLoader

def create_xyz_file(path, symbol='H', positions=[[0, 0, 0]]):
    atoms = Atoms(symbols=symbol, positions=positions)
    write(path, atoms, format='xyz')

def test_init_with_empty_dataset():
    loader = DataLoader()
    assert loader.dataset_files == []
    assert loader.dataset == []
    assert len(loader) == 0

def test_init_with_single_xyz_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test1.xyz')
        create_xyz_file(file_path)
        loader = DataLoader([file_path])
        assert loader.dataset_files == [file_path]
        assert len(loader.dataset) == 1
        assert isinstance(loader.dataset[0], Atoms)
        assert loader.dataset[0].symbols == 'H'

def test_init_with_multiple_xyz_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, 'test1.xyz')
        file2 = os.path.join(tmpdir, 'test2.xyz')
        create_xyz_file(file1, symbol='H')
        create_xyz_file(file2, symbol='O', positions=[[0,0,0],[0,0,1]])
        loader = DataLoader([file1, file2])
        assert loader.dataset_files == [file1, file2]
        assert len(loader.dataset) == 2
        assert loader.dataset[0].symbols == 'H'
        assert loader.dataset[1].symbols == 'O2'
        assert len(loader) == 2

def test_init_with_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        DataLoader(['nonexistent.xyz'])