"""The analytic PAW-LCAO molecular forces as a user gets them: ``Mandacaru`` + ASE.

``reference/molecular_forces.json`` holds, per molecule, the atoms, the options and the analytic forces,
re-frozen on 2026-11-16 against the regenerated PAW library (the first freeze, 2026-11-03, was the retired
finite-difference route, which the analytic gradient reproduced to 2.0e-7 eV/Angstrom). The force's own oracle,
reconverged finite differences at a fixed grid, is the Rust slow tier ``paw_forces_dataset``. Each case rebuilds
its atoms from the file and checks the live analytic force against the frozen number. Needs ``MANDACARU_PAW_PATH``.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

REFERENCE = json.loads((Path(__file__).parent / "reference" / "molecular_forces.json").read_text())
CASES = REFERENCE["cases"]
COMMON = REFERENCE["options_common"]

pytestmark = pytest.mark.skipif(not os.environ.get("MANDACARU_PAW_PATH"), reason="MANDACARU_PAW_PATH is not set")

#: eV/Angstrom. Two independent runs of the 2026-11-16 freeze give the same bits. The tolerance is the one the
#: retired route set: its Richardson step error was 1e-5 at most, 25 times the analytic force's miss against it.
TOLERANCE = 5e-6


def molecule(case, **overrides):
    record = CASES[case]
    spec = record["atoms"]
    atoms = Atoms(spec["symbols"], positions=spec["positions_angstrom"], cell=spec["cell_angstrom"])
    if spec["magmoms"]:
        atoms.set_initial_magnetic_moments(spec["magmoms"])
    atoms.calc = Mandacaru(**{**record["options"], **COMMON, **overrides})
    return atoms, record


@pytest.mark.parametrize("case", ["h2_pbe", "h2_rhf", "lih_rhf", "lih_pbe", "h2o_pbe", "h2o_rhf", "oh_upbe", "h2o_r2scan", "h2o_hse06"])
def test_the_analytic_force_reproduces_the_frozen_force(case):
    atoms, record = molecule(case)
    forces = atoms.get_forces()
    miss = np.abs(forces - np.array(record["forces_ev_angstrom"])).max()
    assert miss < TOLERANCE, (case, miss)
    details = atoms.calc.result.force_details
    assert details["scheme"] == "analytic"
    assert details["stationarity"] < 1e-6
    assert "step_bohr" not in details and "step_error_estimate" not in details
    # The frozen coordinate is a signal, not a zero the engine could return by accident.
    assert np.abs(record["forces_ev_angstrom"]).max() > 1e3 * TOLERANCE


def test_float32_forces_follow_the_float64_ones_at_the_float32_floor():
    """The float32 gradient is analytic: no step turns rounding into force noise. Its floor is the float32 SCF's."""
    atoms, record = molecule("h2o_pbe", precision="float32", scf_options={"tolerance": 1e-6})
    forces = atoms.get_forces()
    miss = np.abs(forces - np.array(record["forces_ev_angstrom"])).max()
    assert atoms.calc.result.force_details["scheme"] == "analytic"
    # Measured 1.6e-6 (tolerance 1e-6) and 2.1e-6 (default tolerance) eV/Angstrom; the gate is 100 times that.
    assert miss < 2e-4, miss
