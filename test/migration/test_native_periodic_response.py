"""The analytic crystal forces and stress as a user gets them: ``Mandacaru(method="dft")`` + ASE.

``reference/periodic_response.json`` (``freeze_periodic_response.py``) holds, per crystal, the atoms and the options, the force and
the stress the Rust engines gave through ``atoms.get_forces()`` / ``atoms.get_stress()`` when they were frozen, and the
reconverged-finite-difference oracles of the same discretized energy (``freeze_periodic_gradients.py`` and
``freeze_periodic_strain.py``, the Richardson protocol of ``PAW_COMPLETION_PLAN.md`` 14.4) with the gate of every component.
Each case rebuilds its atoms from the file and checks the live number against the engine's frozen one (tight: the same
code on the same SCF) and against the oracle (at the gate). Needs ``MANDACARU_PAW_PATH``.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

REFERENCE = json.loads((Path(__file__).parent / "reference" / "periodic_response.json").read_text())["cases"]

pytestmark = pytest.mark.skipif(not os.environ.get("MANDACARU_PAW_PATH"), reason="MANDACARU_PAW_PATH is not set")

#: The same code on the same SCF reproduces the frozen engine number to the SCF noise (1e-9 Ha over a Bohr).
ENGINE_FORCE = 1e-6
ENGINE_STRESS = 1e-6


def crystal(case):
    record = REFERENCE[case]
    a = record["atoms"]
    atoms = Atoms(a["symbols"], cell=a["cell_angstrom"], positions=a["positions_angstrom"], pbc=a["pbc"])
    if any(a["magmoms"]):
        atoms.set_initial_magnetic_moments(a["magmoms"])
    atoms.calc = Mandacaru(method="dft", **record["options"])
    return atoms, record


def check_forces(case):
    atoms, record = crystal(case)
    forces = atoms.get_forces()
    assert np.allclose(forces, record["forces_ev_angstrom"], atol=ENGINE_FORCE), (
        np.abs(forces - record["forces_ev_angstrom"]).max())
    for entry in record["force_oracle"]:
        miss = abs(forces[entry["atom"], entry["axis"]] - entry["force_ev_angstrom"])
        assert miss < entry["gate_ev_angstrom"], (case, entry, miss)
        # The coordinate is a signal, not a zero the engine could return by accident.
        assert abs(entry["force_ev_angstrom"]) > 100 * entry["gate_ev_angstrom"]
    return atoms


def check_stress(case):
    atoms, record = crystal(case)
    stress = atoms.get_stress()
    assert stress.shape == (6,)
    assert np.allclose(stress, record["stress_ev_angstrom3"], atol=ENGINE_STRESS)
    for voigt, entry in enumerate(record["stress_oracle"]):
        miss = abs(stress[voigt] - entry["stress_ev_angstrom3"])
        assert miss < entry["gate_ev_angstrom3"], (case, voigt, miss, entry["gate_ev_angstrom3"])
    full = atoms.get_stress(voigt=False)
    assert np.allclose(full, full.T)
    assert np.allclose(full[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]], stress)
    return atoms


def test_lih_forces_and_stress_follow_the_reconverged_finite_difference():
    atoms = check_forces("lih")
    # One SCF serves both properties (and get_forces after get_stress): the native result is reused.
    result = atoms.calc.result
    atoms.get_stress()
    assert atoms.calc.result is result
    assert np.allclose(atoms.calc.result.forces(), atoms.get_forces())
    check_stress("lih")


def test_hse06_forces_and_stress_follow_the_reconverged_finite_difference():
    check_forces("lih_hse06")
    check_stress("lih_hse06")


#: sigma_xx of the spin-polarized ``beh_lda`` crystal (eV/Angstrom^3): the Richardson extrapolation of
#: central strain differences of the free energy (steps 1e-3 and 2e-3, every SCF reconverged) with the
#: ion width held at the unstrained cell's value (the pin of the analytic stress). Measured
#: 2026-11-08 at 0.000755343; the analytic stress then agreed to 4.5e-9, as the unpolarized cell did.
BEH_SPIN_SIGMA_XX = 7.55343e-4
BEH_SPIN_GATE = 5e-8


def test_spin_polarized_forces_and_stress_follow_the_reconverged_finite_difference():
    atoms = check_forces("beh_lda")
    stress = atoms.get_stress()
    assert abs(stress[0] - BEH_SPIN_SIGMA_XX) < BEH_SPIN_GATE, stress[0]


@pytest.mark.slow
def test_si_forces_follow_the_reconverged_finite_difference():
    """Displaced diamond Si on the 2x2x2 mesh (measured 80-130 s for the SCF and the forces)."""
    check_forces("si")


@pytest.mark.slow
def test_si_stress_follows_the_reconverged_finite_difference():
    """The same crystal's stress (measured 120-200 s)."""
    check_stress("si")


@pytest.mark.slow
def test_aluminum_metal_forces_and_stress_follow_the_reconverged_finite_difference():
    check_forces("al2")
    check_stress("al_small")


@pytest.mark.slow
def test_spinor_forces_follow_the_reconverged_finite_difference():
    """Diamond Si with spin-orbit coupling (the lda-fr datasets), time-reversal symmetric spinor."""
    check_forces("si_ldafr")


@pytest.mark.slow
def test_spinor_stress_follows_the_reconverged_finite_difference():
    """Measured 173 s on a loaded machine (four threads), the longest test of the suite: 7 s under the 180 s budget,
    which a busier machine breaches. The frozen crystal (2x2x2, 16 spinor bands, grid 0.5) cannot be shrunk without
    re-freezing its oracle (368 s); the shear and isotropy of the same engine on a Gamma-only crystal are in
    ``test/algorithms/test_spinor_dft.py::TestASpinOrbitCrystal::test_spinor_crystal_stress``."""
    check_stress("si_ldafr")
