"""Geometry handover must change the native start and preserve physical results."""
from ase import Atoms
import numpy as np
import pytest

from mandacaru import Mandacaru


@pytest.mark.parametrize("method", ["adapt-vqe", "mcas-vqe", "valqa"])
def test_transfer_reoptimizes_the_previous_ansatz_in_the_new_geometry(method):
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    options = {} if method == "adapt-vqe" else dict(max_steps=20, transfer_steps=2, seed=7, record=False)
    with Mandacaru(method=method, atoms=atoms, basis="HAO", pool="qeb",
                   grid_spacing=.5, transfer=True, resources={"threads": 1}, **options) as calc:
        atoms.get_potential_energy()
        assert not calc.result.transfer["accepted"]
        previous = calc.result
        atoms.positions[1, 2] += .001
        energy = atoms.get_potential_energy()
        result = calc.result
        assert result is not previous
        assert result.transfer["accepted"], result.transfer
        assert result.transfer["confidence"] > .99
        assert result._raw["seed_relaxation"]["kept"]
        assert result._raw["seed_relaxation"]["success"]
        assert np.allclose(result.state(), calc.checkpoint.state(), atol=1e-12)
        if method != "adapt-vqe":
            assert len(result.steps) == 2
        grid = calc.integrals.grid
    with Mandacaru(method="adapt-vqe", atoms=atoms, basis="HAO", pool="qeb", grid=grid,
                   grid_spacing=.5, resources={"threads": 1}) as fresh:
        assert energy == pytest.approx(atoms.get_potential_energy(), abs=1e-8)


def test_transfer_rejects_incompatible_controls_before_running():
    for options in (dict(transfer=1), dict(transfer=True, transfer_threshold=-1.),
                    dict(transfer=True, transfer_steps=-1), dict(transfer_steps=10),
                    dict(transfer=True, resume="state.json"),
                    dict(update_between_geometries=True)):
        with pytest.raises((ValueError, TypeError)):
            Mandacaru(method="adapt-vqe", **options)
