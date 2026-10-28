"""Native canonical XC point API, frozen numerical references and derivatives."""
import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native

REFERENCE = Path(__file__).parent / "native_cutover/xc_reference.json"


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_native_spin_energy_and_seven_partials_against_frozen_reference(functional):
    data = json.loads(REFERENCE.read_text())
    variables = np.asarray(data["points"], dtype=float)
    expected = np.asarray(data["expected"][functional], dtype=float)
    actual = _native.Context(threads=1).xc_points(functional, variables)
    assert actual.shape == (len(variables), 8)
    assert np.allclose(actual, expected, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan", "hse06"])
def test_seven_partials_are_derivatives_of_the_native_energy(functional):
    ctx = _native.Context(threads=1)
    for point in [[.12, .08, .004, -.001, .003, .08, .06],
                  [.2, .1, .04, .005, .02, .001, .001]]:
        x = np.array([point], dtype=float)
        result = ctx.xc_points(functional, x)[0]
        for axis in range(7):
            delta = min(1e-6 * max(abs(x[0, axis]), .001), abs(x[0, axis])/4)
            plus, minus = x.copy(), x.copy()
            plus[0, axis] += delta
            minus[0, axis] -= delta
            difference = (ctx.xc_points(functional, plus)[0, 0] -
                          ctx.xc_points(functional, minus)[0, 0]) / (2*delta)
            assert difference == pytest.approx(result[axis+1], rel=3e-5, abs=2e-7)


def test_screened_hybrid_semilocal_energy_against_independent_quadrature():
    data = json.loads(REFERENCE.read_text())
    actual = _native.Context(threads=1).xc_points("hse06", np.asarray(data["points"]))
    assert np.allclose(actual[:, 0], data["expected"]["hse06_energy"], atol=2e-11, rtol=2e-10)
    assert np.all(actual[:, 6:] == 0.)


def test_spin_exchange_symmetry_and_ownership():
    ctx = _native.Context(threads=1)
    values = np.array([[.2, .1, .004, -.001, .003, .07, .09]])
    original = values.copy()
    actual = ctx.xc_points("r2scan", values)
    switched = ctx.xc_points("r2scan", values[:, [1, 0, 4, 3, 2, 6, 5]].copy())
    assert np.allclose(actual, switched[:, [0, 2, 1, 5, 4, 3, 7, 6]], atol=1e-13)
    assert np.array_equal(values, original)
    assert ctx.memory_used_bytes == 0


def test_canonical_names_validation_and_budget():
    ctx = _native.Context(threads=1)
    point = np.ones((1, 7))*.1
    for alias in ["PBE", "gga", "pz81", "r²scan"]:
        with pytest.raises(ValueError):
            ctx.xc_points(alias, point)
    with pytest.raises(ValueError):
        ctx.xc_points("r2scan", point, relativistic=True)
    with pytest.raises(ValueError):
        ctx.xc_points("r2scan", point[:, :5].copy())
    with pytest.raises(ValueError):
        ctx.xc_points("r2scan", np.full((1, 7), np.nan))
    with pytest.raises(MemoryError):
        _native.Context(memory_limit_bytes=64).xc_points("r2scan", np.ones((10, 7)))
    assert ctx.memory_used_bytes == 0


@pytest.mark.parametrize("functional", ["r2scan", "hse06", "pbe+rvv10", "r2scan+rvv10"])
def test_native_molecular_stationary_force_follows_total_energy(functional):
    from ase import Atoms
    from mandacaru._native_adapter import geometry_from_atoms
    from mandacaru.units import ANGSTROM_TO_BOHR
    ctx = _native.Context(threads=1)
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5.]*3)
    base = ctx.molecular_integrals(geometry_from_atoms(atoms), spacing_bohr=.5*ANGSTROM_TO_BOHR)
    def solve(integrals):
        state = integrals.kohn_sham(functional, [1, 1], tolerance=1e-12, guesses=1)
        assert state.converged
        return state
    state = solve(base)
    analytic = state.gradient()["gradient"][1, 2]
    step = 1e-5
    energies = []
    for delta in [step, -step]:
        moved = atoms.copy()
        moved.positions[1, 2] += delta
        integrals = ctx.molecular_integrals(geometry_from_atoms(moved), grid=base.grid)
        snapshot = solve(integrals).snapshot()
        energies.append(snapshot["electronic_energy"] + snapshot["nuclear_repulsion"])
    finite_difference = (energies[0]-energies[1])/(2*step*ANGSTROM_TO_BOHR)
    assert analytic == pytest.approx(finite_difference, abs=2e-7)
