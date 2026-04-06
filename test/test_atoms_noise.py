import numpy as np
from carcara.atoms_noise import AtomsNoiseGenerator
from ase.build import bulk
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