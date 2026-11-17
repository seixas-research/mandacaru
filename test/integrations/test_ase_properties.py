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


def test_a_polar_molecule_has_its_dipole_moment_and_a_crystal_raises(crystal):
    """``get_dipole_moment`` of LiH (PAW-LCAO DZP, LDA): along the bond, from H to Li
    (Li+ H-), unchanged by a rigid translation of the neutral molecule to the grid's
    egg-box level, and the same in e Bohr with ``atomic_units=True``. A crystal raises."""
    def lithium_hydride(**options):
        atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[6, 6, 6])
        atoms.center()
        atoms.calc = Mandacaru(method="dft", basis={"name": "PAW-LCAO", "size": "DZP"},
                               grid_spacing=0.3, trace=False, **options)
        return atoms

    atoms = lithium_hydride()
    moment = atoms.get_dipole_moment()
    assert moment.shape == (3,)
    assert moment[2] < -0.5 and max(abs(moment[0]), abs(moment[1])) < 1e-3 * abs(moment[2])
    atoms.positions += [0.4, -0.3, 0.2]
    assert np.allclose(atoms.get_dipole_moment(), moment, atol=1e-2 * abs(moment[2]))
    atomic = lithium_hydride(atomic_units=True).get_dipole_moment()
    assert np.allclose(atomic * 0.529177210903, moment, rtol=1e-9, atol=1e-12)
    with pytest.raises(NotImplementedError, match="get_polarization"):
        crystal.calc.get_dipole_moment()
