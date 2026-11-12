"""The water and NH2 excitations of the native linear-response route against their frozen values
(``reference/excitations.json``, written by ``freeze_excitations.py``). Needs
``MANDACARU_PAW_PATH``."""

import json
from pathlib import Path

import pytest

from .freeze_excitations import NUM_STATES, SETTINGS, multiplets, system
from mandacaru import Mandacaru

REFERENCE = json.loads((Path(__file__).parent / "reference" / "excitations.json").read_text())
# The SCF replays to about 1e-9 eV; the roots are converged to 1e-8 in the residual.
ENERGY_TOLERANCE_EV = 1e-6
STRENGTH_TOLERANCE = 1e-7


@pytest.mark.parametrize("key", sorted(REFERENCE["cases"]))
def test_water_excitations_match_the_frozen_values(key):
    atoms, spin = system(key)
    xc, approximation = key.removeprefix("NH2/").split("/")[:2]
    frozen = REFERENCE["cases"][key]
    atoms.calc = Mandacaru(xc=xc, **SETTINGS)
    found = atoms.calc.get_excitations(NUM_STATES, approximation=approximation, spin=spin,
                                       tolerance=1e-8)
    assert len(found.energies) == len(frozen["energies_eV"])
    for value, reference in zip(found.energies, frozen["energies_eV"]):
        assert abs(value - reference) < ENERGY_TOLERANCE_EV, key
    groups = multiplets(found.energies, found.oscillator_strengths, found.degenerate_tolerance)
    assert [g[:2] for g in groups] == [g[:2] for g in frozen["multiplets"]]
    for (_, _, value), (_, _, reference) in zip(groups, frozen["multiplets"]):
        assert abs(value - reference) < STRENGTH_TOLERANCE, key
    if "velocity_multiplets" in frozen:
        velocity = multiplets(found.energies, found.velocity_oscillator_strengths,
                              found.degenerate_tolerance)
        for (_, _, value), (_, _, reference) in zip(velocity, frozen["velocity_multiplets"]):
            assert abs(value - reference) < STRENGTH_TOLERANCE, key
    if frozen.get("spin_squared") is not None:
        for value, reference in zip(found.spin_squared, frozen["spin_squared"]):
            assert abs(value - reference) < STRENGTH_TOLERANCE, key
    if frozen["one_center_hartree_eV"] is not None:
        for value, reference in zip(found.one_center_hartree, frozen["one_center_hartree_eV"]):
            assert abs(value - reference) < ENERGY_TOLERANCE_EV, key
    atoms.calc.close()
