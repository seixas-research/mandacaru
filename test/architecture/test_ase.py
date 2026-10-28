"""ASE is an optional adapter over the same native calculation."""

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from mandacaru import Geometry, Mandacaru
from mandacaru.integrations.ase import ASECalculator


def test_ase_adapter_preserves_energy_force_and_displacement_response():
    atoms = ase.Atoms(
        "H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5] * 3
    )
    options = dict(method="dft", xc="pbe", grid_spacing=0.5, resources={"threads": 1})
    with Mandacaru(**options) as calculation:
        atoms.calc = ASECalculator(calculation)
        force = atoms.get_forces()
        energy = atoms.get_potential_energy()
        snapshot = Geometry((1, 1), atoms.positions, atoms.cell.array)
        with Mandacaru(**options, grid=calculation.integrals.grid) as standalone:
            result = standalone.run(snapshot)
            assert result.in_units("eV") == pytest.approx(energy, abs=1e-10)
            assert np.allclose(result.forces(), force, atol=1e-10)
        step = 1e-5
        values = []
        for offset in (step, -step):
            atoms.positions[1, 2] = 2.87 + offset
            values.append(atoms.get_potential_energy())
        assert force[1, 2] == pytest.approx(
            -(values[0] - values[1]) / (2 * step), abs=2e-5
        )
