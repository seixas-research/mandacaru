"""PAW Kohn–Sham energy conventions and separation from the HF model."""
from ase import Atoms
import numpy as np
import pytest

from mandacaru import Mandacaru
from mandacaru.pseudopotentials import get_paw
from mandacaru.pseudopotentials.environment import default_library_folder


def test_unconfined_beryllium_recovers_existing_atomic_valence_reference():
    # The tolerance and atomic energy reference predate the Rust migration.
    atoms = Atoms("Be", positions=[[4.] * 3], cell=[8.] * 3)
    dataset = get_paw("Be", paw_dataset=default_library_folder("lda"))
    with Mandacaru(method="dft", xc="lda", atoms=atoms, atomic_units=True,
                   basis={"name": "PAW-LCAO", "size": "SZ", "energy_shift": None},
                   grid_spacing=.2, resources={"threads": 1}) as calc:
        result = calc.run()
        assert result.success
        assert result.energy_hartree == pytest.approx(dataset.energies["reference_valence"], abs=3e-3)
        assert not calc.native_reference.has_model


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_paw_ks_spectrum_and_quantum_export_are_distinct_energy_functionals(functional):
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5.] * 3)
    with Mandacaru(method="dft", xc=functional, atoms=atoms, atomic_units=True,
                   basis={"name": "PAW-LCAO", "size": "DZ"}, grid_spacing=.4,
                   resources={"threads": 1}) as calc:
        result = calc.run()
        assert result.success
        assert not calc.native_reference.has_model
        assert np.all(np.diff(result.scf.mo_energies) >= 0.)
        assert result.scf.n_occupied == 1
        model = result.hamiltonian
        problem = model.problem()
        determinant = problem.evaluate([], indices=[])["energy_ha"]
        assert np.isfinite(determinant)
        assert abs(determinant - result.energy_hartree) > 1e-3
