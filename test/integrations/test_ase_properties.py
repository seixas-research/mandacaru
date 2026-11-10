"""The tetrahedron density of states of a DFT crystal through the ASE views.

The numerical gates of the method (analytic free-electron and tight-binding densities, the
projection sum) are in ``crates/mandacaru-core/tests/tetrahedron.rs``; here the calculator hands
it a crystal's bands on the full mesh.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru


@pytest.fixture(scope="module")
def crystal():
    from mandacaru.pseudopotentials.paw import get_paw

    try:
        get_paw("H", paw_dataset="lda-sr")
    except FileNotFoundError:
        pytest.skip("the lda-sr PAW-LCAO set is not installed")
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.75, 0, 0]], cell=np.diag([2.0, 5.0, 5.0]),
                  pbc=True)
    atoms.calc = Mandacaru(method="dft", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.3, kpts={"size": (4, 1, 1), "gamma": True},
                           trace=False)
    atoms.get_potential_energy()
    return atoms


def test_the_tetrahedron_density_holds_every_band(crystal):
    calc = crystal.calc
    axis = np.linspace(-30.0, 60.0, 90001)
    energies, dos = calc.dos(energies=axis, kpts=(16, 1, 1), integration="tetrahedron")
    # Two bands, two spins; the 1-D steps leave 1e-3 to a trapezoid rule on a 1 meV axis.
    assert np.trapezoid(dos, energies) == pytest.approx(2 * calc.get_number_of_bands(), abs=2e-3)
    assert np.all(dos >= 0)


def test_the_projected_tetrahedron_density_adds_up_to_the_total(crystal):
    calc = crystal.calc
    axis = np.linspace(-30.0, 60.0, 2001)
    _, dos = calc.dos(energies=axis, integration="tetrahedron")
    _, pdos = calc.pdos(energies=axis, integration="tetrahedron")
    assert sorted(pdos) == [(0, 0), (1, 0)]
    assert np.allclose(sum(pdos.values()), dos, atol=1e-12)


def test_bad_integration_requests_raise(crystal):
    calc = crystal.calc
    with pytest.raises(ValueError, match="integration"):
        calc.dos(integration="simpson")
    with pytest.raises(ValueError, match="Gamma-centered"):
        calc.dos(kpts={"size": (4, 1, 1), "gamma": False}, integration="tetrahedron")
