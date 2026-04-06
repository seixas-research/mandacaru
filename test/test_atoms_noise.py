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

def test_atoms_noise_generator_initialization(atoms=setup_data, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(atoms, noise_type=noise_type, seed=seed)
    assert generator.atoms is not None, "Atoms object should be initialized"
    assert generator.rng is not None, "Random generator should be initialized"
    assert generator.noise_type == noise_type, "Noise type should be set correctly"


def test_relax_structure(atoms=setup_data, calculator=setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(atoms, noise_type=noise_type, seed=seed)
    generator.calculator = calculator
    relaxed_atoms = generator.relax_structure()
    assert relaxed_atoms is not None, "Relaxed structure should not be None"
    assert len(relaxed_atoms) == len(atoms), "Relaxed structure should have the same number of atoms as the original"


def test_generate_samples(atoms=setup_data, calculator=setup_calculator, seed=42, noise_type='normal'):
    generator = AtomsNoiseGenerator(atoms, noise_type=noise_type, seed=seed)
    generator.calculator = calculator
    generator.relax_structure()
    num_samples = 10
    generator.generate_samples(num_samples=num_samples, noise_type=noise_type)
    assert len(generator.samples) == num_samples, f"Should generate {num_samples} samples"
    for sample in generator.samples:
        assert len(sample) == len(atoms), "Each sample should have the same number of atoms as the original"
