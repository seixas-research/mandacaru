"""The ``Excitations`` view: native broadening, natural transition orbitals and units.

H2 on a PAW-LCAO DZP basis; needs ``MANDACARU_PAW_PATH``."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru


@pytest.fixture(scope="module")
def excitations():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5])
    atoms.center()
    atoms.calc = Mandacaru(method="dft", xc="pbe", basis={"name": "PAW-LCAO", "size": "DZP"},
                           grid_spacing=0.3)
    return atoms.calc.get_excitations(4, approximation="casida")


def test_the_spectrum_has_the_total_oscillator_strength(excitations):
    axis = np.linspace(0, 80, 40001)
    for shape in ("gaussian", "lorentzian"):
        spectrum = excitations.spectrum(axis, width=0.2, shape=shape)
        area = np.trapezoid(spectrum, axis)
        assert abs(area - excitations.oscillator_strengths.sum()) < 0.02 * excitations.oscillator_strengths.sum()
    with pytest.raises(ValueError, match="lorentzian"):
        excitations.spectrum(axis, shape="voigt")


def test_natural_transition_orbitals_are_weighted_pairs(excitations):
    weights, occupied, empty = excitations.natural_transition_orbitals(0)
    assert abs(weights.sum() - 1) < 1e-12 and np.all(np.diff(weights) <= 1e-15)
    assert occupied.shape == empty.shape
    assert excitations.nto_weights(0)[0] == pytest.approx(weights[0])
    density = excitations.transition_density_matrix(0)
    assert np.allclose(density, density.T)


def test_transitions_name_reference_orbitals(excitations):
    first = excitations.transitions(0)[0]
    assert first.spin is None
    assert first.occupied < excitations.occupied[0] <= first.empty and 0 < first.weight <= 1 + 1e-12
