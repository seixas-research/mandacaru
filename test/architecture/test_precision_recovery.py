"""Molecular precision controls must select owned native storage."""
from ase import Atoms
from ase.build import bulk
import numpy as np
import pytest

from mandacaru import Mandacaru


def test_native_precision_changes_sample_storage_and_preserves_total_energy():
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    energies = []
    samples = []
    for precision, dtype in (("double", np.float64), ("single", np.float32)):
        with Mandacaru(method="rhf", atoms=atoms, basis="HAO", precision=precision,
                       grid_spacing=.5, resources={"threads": 1}) as calc:
            energies.append(atoms.get_potential_energy())
            snapshot = calc.integrals.snapshot(include_samples=True)
            assert snapshot["precision"] == precision
            assert snapshot["samples"].dtype == dtype
            assert snapshot["orthonormal_samples"].dtype == dtype
            assert snapshot["one_body"].dtype == np.float64
            samples.append(snapshot["samples"])
            assert calc.result.success
    assert samples[1].nbytes * 2 == samples[0].nbytes
    assert energies[1] == pytest.approx(energies[0], abs=1e-5)
    assert np.array_equal(samples[1], samples[0].astype(np.float32))


def test_invalid_precision_is_rejected_before_setup():
    with pytest.raises(ValueError, match="precision"):
        Mandacaru(precision="quadruple")


def test_periodic_single_precision_and_repeated_ase_cell_updates():
    atoms = bulk("Al", "fcc", a=4.05)
    with Mandacaru(method="dft", xc="lda", atoms=atoms, basis="PAW-LCAO",
                   grid_spacing=.8, precision="single",
                   resources={"threads": 1}) as calc:
        original = atoms.get_potential_energy()
        assert calc.result.success
        assert calc.periodic_problem.precision == "single"
        assert calc.result.scf.n_iterations > 0
        assert atoms.get_magnetic_moment() == pytest.approx(0.)
        atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
        updated = atoms.get_potential_energy()
        assert calc.result.success
        assert updated != pytest.approx(original, abs=1e-5)
        assert not calc.calculation_required(atoms, ["energy"])
