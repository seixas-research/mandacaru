import numpy as np
from carcara.atoms_noise import AtomsNoiseGenerator
from ase.build import bulk
from ase.io import read, write
from ase.calculators.emt import EMT
import pytest


@pytest.fixture
def setup_data():
    atoms = bulk("Au", "fcc", a=4.08, cubic=True).repeat((2, 2, 2))
    return atoms

@pytest.fixture
def setup_calculator():
    return EMT()

def test_atoms_noise_generator_initialization(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, noise_type=noise_type, seed=seed)
    assert generator.atoms is not None, "Atoms object should be initialized"
    assert generator.rng is not None, "Random generator should be initialized"
    assert generator.noise_type == noise_type, "Noise type should be set correctly"


def test_relax_structure(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    relaxed_atoms = generator.relax_structure()
    assert relaxed_atoms is not None, "Relaxed structure should not be None"
    assert len(relaxed_atoms) == len(setup_data), "Relaxed structure should have the same number of atoms as the original"


def test_relax_structure_with_cell_relaxation(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    relaxed_atoms = generator.relax_structure(relax_cell=True)
    assert relaxed_atoms is not None, "Relaxed structure should not be None"
    assert len(relaxed_atoms) == len(setup_data), "Relaxed structure should have the same number of atoms as the original"
    assert not np.array_equal(relaxed_atoms.get_cell(), setup_data.get_cell()), "Cell should be relaxed and differ from original cell"


def test_generate_samples(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    num_samples = 10
    generator.generate_samples(num_samples=num_samples, noise_type=noise_type)
    assert len(generator.samples) == num_samples, f"Should generate {num_samples} samples"
    for sample in generator.samples:
        assert len(sample) == len(setup_data), "Each sample should have the same number of atoms as the original"


def test_noise_application(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    original_positions = generator.atoms.get_positions().copy()
    generator.generate_samples(num_samples=1, noise_type=noise_type)
    noisy_positions = generator.samples[0].get_positions()
    assert not np.array_equal(original_positions, noisy_positions), "Noisy positions should differ from original positions"


def test_cell_noise_application(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    original_cell = generator.atoms.get_cell().copy()
    generator.generate_samples(num_samples=1, noise_type=noise_type, cell_mode='all')
    noisy_cell = generator.samples[0].get_cell()
    assert not np.array_equal(original_cell, noisy_cell), "Noisy cell should differ from original cell"


def test_different_noise_types(setup_data, setup_calculator, seed=42):
    for noise_type in ['normal', 'uniform']:
        generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
        generator.relax_structure()
        generator.generate_samples(num_samples=1, noise_type=noise_type)
        assert len(generator.samples) == 1, f"Should generate 1 sample for noise type {noise_type}"


def test_reproducibility(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator1 = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator1.relax_structure()
    generator1.generate_samples(num_samples=1, noise_type=noise_type)

    generator2 = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator2.relax_structure()
    generator2.generate_samples(num_samples=1, noise_type=noise_type)

    assert np.array_equal(generator1.samples[0].get_positions(), generator2.samples[0].get_positions()), "Samples should be identical for the same seed"


def test_invalid_noise_type(setup_data, setup_calculator, seed=42):
    with pytest.raises(ValueError):
        AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type='invalid_noise_type', seed=seed)


def test_invalid_cell_mode(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    with pytest.raises(ValueError):
        generator.generate_samples(num_samples=1, noise_type=noise_type, cell_mode='invalid_cell_mode')


def test_zero_noise_uniform(setup_data, setup_calculator, seed=42, noise_type='uniform'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    original_positions = generator.atoms.get_positions().copy()
    generator.generate_samples(num_samples=1, noise_type=noise_type, noise_level_pos=0.0, noise_level_cell=0.0)
    noisy_positions = generator.samples[0].get_positions()
    assert np.array_equal(original_positions, noisy_positions), "Noisy positions should be identical to original positions with zero noise (uniform noise)"


def test_zero_noise_normal(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    original_positions = generator.atoms.get_positions().copy()
    generator.generate_samples(num_samples=1, noise_type=noise_type, noise_level_pos=0.0, noise_level_cell=0.0)
    noisy_positions = generator.samples[0].get_positions()
    assert np.array_equal(original_positions, noisy_positions), "Noisy positions should be identical to original positions with zero noise (normal noise)"


def test_large_noise(setup_data, setup_calculator, seed=42, noise_type='uniform'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    original_positions = generator.atoms.get_positions().copy()
    generator.generate_samples(num_samples=1, noise_type=noise_type, noise_level_pos=0.3, noise_level_cell=0.3)
    noisy_positions = generator.samples[0].get_positions()
    distance = np.linalg.norm(noisy_positions - original_positions)
    assert distance > 0.1, "Noisy positions should be significantly different from original positions with large noise"


def test_cell_scaling(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    original_cell = generator.atoms.get_cell().copy()
    generator.generate_samples(num_samples=1, noise_type=noise_type, scale_cell=1.5)
    scaled_cell = generator.samples[0].get_cell()
    assert not np.array_equal(original_cell, scaled_cell), "Scaled cell should differ from original cell"


def test_cell_scaling_xy(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    original_cell = generator.atoms.get_cell().copy()
    generator.generate_samples(num_samples=1, noise_type=noise_type, scale_cell=1.5, cell_mode='xy')
    scaled_cell = generator.samples[0].get_cell()
    assert not np.array_equal(original_cell[:2, :2], scaled_cell[:2, :2]), "Scaled xy cell should differ from original xy cell"
    assert np.array_equal(original_cell[2], scaled_cell[2]), "z component of the cell should remain unchanged in xy mode"


def test_relax_structure_without_calculator(setup_data, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, noise_type=noise_type, seed=seed)
    with pytest.raises(ValueError):
        generator.relax_structure()


def test_relax_structure_with_invalid_algorithm(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    with pytest.raises(ValueError):
        generator.relax_structure(algorithm='invalid_algorithm')

def test_generate_samples_with_invalid_noise_type(setup_data, setup_calculator, seed=42):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, seed=seed)
    generator.relax_structure()
    with pytest.raises(ValueError):
        generator.generate_samples(num_samples=1, noise_type='invalid_noise_type')

def test_generate_samples_with_invalid_cell_mode(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    with pytest.raises(ValueError):
        generator.generate_samples(num_samples=1, noise_type=noise_type, cell_mode='invalid_cell_mode')


def test_relax_structure_with_invalid_mask(setup_data, setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    with pytest.raises(ValueError):
        generator.relax_structure(relax_cell=True, cell_mask=[True, False])


def test_save_xyz(setup_data, setup_calculator, tmp_path, seed=42, noise_type='uniform'):
    filename = str(tmp_path / "test_sample.xyz")
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    generator.generate_samples(num_samples=1, noise_type=noise_type)
    sample = generator.samples[0]
    generator.save_to_xyz(filename=filename, compute_ref=True)
    loaded_sample = read(filename)
    assert len(sample) == len(loaded_sample), "Number of atoms should be the same after saving and loading"
    assert np.allclose(sample.get_positions(), loaded_sample.get_positions()), "Positions should be close after saving and loading"
    assert np.allclose(sample.get_cell(), loaded_sample.get_cell()), "Cell should be close after saving and loading"


def test_save_xyz_with_multiple_samples(setup_data, setup_calculator, tmp_path, seed=42, noise_type='normal'):
    filename = str(tmp_path / "test_samples.xyz")
    generator = AtomsNoiseGenerator(setup_data, calculator=setup_calculator, noise_type=noise_type, seed=seed)
    generator.relax_structure()
    num_samples = 5
    generator.generate_samples(num_samples=num_samples, noise_type=noise_type)
    generator.save_to_xyz(filename=filename, compute_ref=True)
    loaded_samples = read(filename, index=':')
    assert len(loaded_samples) == num_samples, f"Should load {num_samples} samples from the file"
    for original, loaded in zip(generator.samples, loaded_samples):
        assert len(original) == len(loaded), "Number of atoms should be the same for each sample after saving and loading"
        assert np.allclose(original.get_positions(), loaded.get_positions()), "Positions should be close for each sample after saving and loading"
        assert np.allclose(original.get_cell(), loaded.get_cell()), "Cell should be close for each sample after saving and loading"
