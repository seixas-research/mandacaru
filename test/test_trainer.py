import numpy as np
import pytest
from carcara.core.trainer import MACETrainer
from ase.build import bulk
from ase.calculators.emt import EMT

@pytest.fixture
def setup_samples():

    atoms = bulk("Au", "fcc", a=4.08, cubic=True).repeat((2, 2, 2))
    calculator = EMT()
    atoms.set_calculator(calculator)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return [(atoms, energy, forces)]
