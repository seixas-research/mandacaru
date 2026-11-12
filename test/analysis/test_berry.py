"""The Berry result views convert units and nothing else: hand-made native records, no crystal."""
import numpy as np
import pytest
from mandacaru.analysis.berry import AnomalousHall, BerryCurvature, BerryPhase, Polarization

BOHR = 0.529177210903
E_PER_BOHR2 = 1.602176634e-19 / (BOHR * 1e-10) ** 2


def test_the_polarization_view_converts_e_per_bohr2_to_coulomb_per_m2_or_keeps_atomic_units():
    raw = dict(vector=[0.01, 0.0, -0.02], quanta=np.eye(3) * 0.1,
               electronic_vector=[0.5, 0.0, 0.0], ionic_vector=[-0.49, 0.0, -0.02],
               fractional=[0.1, 0.0, -0.2], raw=[1.1, 0.0, -0.2], phases=np.zeros((1, 3)),
               spread=[0.0, 0.0, 0.0], size=[4, 4, 4], quantum=2, degeneracy=2.0,
               minimum_singular_value=0.9)
    p = Polarization.from_native(raw)
    np.testing.assert_allclose(p.vector, np.array(raw["vector"]) * E_PER_BOHR2, rtol=1e-12)
    np.testing.assert_allclose(p.quanta, raw["quanta"] * E_PER_BOHR2, rtol=1e-12)
    np.testing.assert_allclose(p.fractional, raw["fractional"])
    assert p.strings == (4, 4, 4) and p.quantum == 2
    atomic = Polarization.from_native(raw, atomic=True)
    np.testing.assert_allclose(atomic.vector, raw["vector"])


def test_the_phase_view_converts_the_connection_to_angstrom_and_keeps_radians():
    raw = dict(direction=1, size=[2, 5, 3], transverse_axes=[0, 2], starts=np.zeros((2, 3, 3)),
               phases=np.full((1, 2, 3), 0.3), mean=[0.3], spread=[0.0],
               centers=[np.full((2, 3, 2), 0.25)], connection=[np.full((2, 3, 2), 2.0)],
               bands=[(0, 2)], minimum_singular_value=0.8)
    phase = BerryPhase.from_native(raw)
    np.testing.assert_allclose(phase.connection[0], 2.0 * BOHR)
    np.testing.assert_allclose(phase.phases, 0.3)
    assert phase.bands == ((0, 2),) and phase.transverse_axes == (0, 2)


def test_the_curvature_view_converts_bohr2_to_angstrom2_and_keeps_the_integers():
    raw = dict(normal=2, axes=[0, 1], size=[2, 2], offset=0.0, orientation=[0.0, 0.0, 1.0],
               flux=np.full((2, 2, 2), 0.1), curvature=np.full((2, 2, 2), 3.0),
               centers=np.zeros((2, 2, 3)), chern=[1, -1], chern_total=0, bands=[(0, 1), (0, 1)],
               max_flux=0.1, minimum_singular_value=0.9)
    plane = BerryCurvature.from_native(raw)
    np.testing.assert_allclose(plane.curvature, 3.0 * BOHR ** 2)
    assert plane.chern_numbers == (1, -1) and plane.chern_number == 0
    assert isinstance(plane.chern_number, int)


def test_the_polarization_view_issues_each_native_warning_and_still_returns_the_value():
    raw = dict(vector=[0.0, 0.0, 0.0], quanta=np.eye(3), electronic_vector=[0.0] * 3,
               ionic_vector=[0.0] * 3, fractional=[0.0] * 3, raw=[0.0] * 3,
               phases=np.zeros((1, 3)), spread=[2.0, 0.0, 0.0], size=[2, 2, 2], quantum=1,
               degeneracy=2.0, minimum_singular_value=0.9,
               warnings=["the Berry phases of the strings along G_0 spread by 2.000 rad"])
    with pytest.warns(UserWarning, match="spread by 2.000 rad"):
        p = Polarization.from_native(raw)
    np.testing.assert_allclose(p.phase_spread, [2.0, 0.0, 0.0])


def test_the_hall_view_converts_to_siemens_per_cm_and_issues_the_native_warnings():
    sigma = np.zeros((3, 3))
    sigma[0, 1], sigma[1, 0] = 1e-3, -1e-3
    raw = dict(sigma=sigma, sigma_coarse=sigma * 0.5, metallic=True, numbers=[0.0, 0.0, 0.2],
               layers=[[[0.0]], [[0.0]], [[0.2, 0.2]]], chern=[[], [], []], size=[4, 4, 4],
               max_flux=0.1, cap=3, merged=2, warnings=["the intrinsic Hall conductivity of this "
                                                        "metal is not converged"])
    with pytest.warns(UserWarning, match="not converged"):
        hall = AnomalousHall.from_native(raw)
    factor = 2 * np.pi * 1.602176634e-19 ** 2 / 6.62607015e-34 / (BOHR * 1e-8)
    np.testing.assert_allclose(hall.sigma[0, 1], 1e-3 * factor, rtol=1e-12)
    np.testing.assert_allclose(hall.sigma_coarse[0, 1], 0.5e-3 * factor, rtol=1e-12)
    assert hall.metallic and hall.merged == 2 and hall.strings == (4, 4, 4)
