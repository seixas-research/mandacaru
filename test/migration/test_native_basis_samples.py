"""MG-6: the all-electron basis families are built and sampled in Rust and reproduce the frozen
values of the Python families they replaced (``reference/basis_samples.json``,
``freeze_basis_samples.py``)."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mandacaru import _native
from mandacaru.integrations.ase import geometry_from_atoms
from mandacaru.specifications import basis_specification

FROZEN = json.loads((Path(__file__).parent / "reference" / "basis_samples.json").read_text())


def sample(atoms, points, spec):
    return _native.Context(threads=1).sample_basis(
        geometry_from_atoms(atoms), np.ascontiguousarray(points),
        basis_spec=json.dumps(basis_specification(spec)))


def ids(case):
    return case["name"]


@pytest.mark.parametrize("case", FROZEN["molecular"], ids=ids)
def test_the_generated_families_match_the_frozen_python_samples(case):
    atoms = Atoms(numbers=case["numbers"], positions=case["positions"])
    points, expected = np.asarray(case["points"]), np.asarray(case["values"])
    actual = sample(atoms, points, case["basis"])
    assert actual.shape == expected.shape
    # Point 0 is exactly on the nucleus, where R(r) Y_lm ~ r^l Y_lm is exactly zero for l > 0. The
    # Python sampler floored r at 1e-15 Bohr and picked a direction, so it returned the shell's
    # slope times 1e-15 there (1.5e-11 for the Og 2p); the analytic limit replaces that artifact.
    # The three points 1e-9 Bohr away, where every direction is defined, keep the full tolerance
    # and so pin the slope at the nucleus.
    expected[np.asarray(case["l"]) > 0, 0] = 0.
    assert np.allclose(actual, expected, atol=3e-12, rtol=1e-12)


def test_a_per_element_basis_samples_each_selected_family():
    case = FROZEN["per_element"]
    atoms = Atoms(case["symbols"], positions=case["positions"])
    actual = sample(atoms, case["points"], case["spec"])
    assert np.allclose(actual, case["values"], atol=1e-13)


@pytest.mark.parametrize("case", FROZEN["numerical"], ids=ids)
def test_numerical_atomic_orbitals_match_the_frozen_splines_and_shell_order(case):
    atoms = Atoms(numbers=[case["number"]], positions=[[0, 0, 0]])
    points, expected = np.asarray(case["points"]), np.asarray(case["values"])
    options = case["options"]
    actual = sample(atoms, points, dict(name="NAO", **options))
    assert actual.shape == expected.shape
    # Point 0 is the nucleus, where the direction of a shell with l > 0 is undefined. The tabulated
    # radial function starts at r = 0 with a small nonzero R(0) for those shells (1.6e-4 for the
    # hydrogen polarization p, 2e-2 for the oxygen p), so the value there is R(0) times a harmonic
    # of an arbitrary direction: the Python sampler used theta = pi/2, phi = 0, the native one the
    # zero vector. Only l = 0 is well defined there and is compared at the full tolerance; for
    # l > 0 the sample must stay within the harmonic's bound times R(0).
    for i, (l, r0) in enumerate(zip(case["l"], case["r0"])):
        if l > 0:
            assert abs(actual[i, 0]) <= r0 * np.sqrt(2 * (2 * l + 1) / (4 * np.pi)) + 1e-15
            expected[i, 0] = actual[i, 0]
    assert np.allclose(actual, expected, atol=2e-10, rtol=2e-9)
    assert np.all(actual[:, 3] == 0.)


@pytest.mark.parametrize("case", FROZEN["all_electron"], ids=ids)
def test_all_electron_nao_matches_the_frozen_species_and_irregular_wall_spline(case):
    atoms = Atoms(numbers=[case["number"]], positions=[[0., 0., 0.]])
    points, expected = np.asarray(case["points"]), np.asarray(case["values"])
    actual = sample(atoms, points, dict(name="NAO-AE", **case["options"]))
    assert actual.shape == expected.shape
    custom = "extra" in case["options"]
    assert np.allclose(actual, expected, atol=1e-6 if custom else 8e-7, rtol=3e-6)
    if not custom:
        assert np.all(actual[:, 4] == 0.)


def test_failed_all_electron_species_are_refused_and_release_their_memory():
    options = dict(tier=0, onset=1.5, width=.8, points=400, r_max=12., tail_norm=None)
    atom = Atoms("He", positions=[[0., 0., 0.]])
    points = np.ascontiguousarray(np.random.default_rng(28).normal(size=(41, 3)))
    spec = basis_specification(dict(name="NAO-AE", **options))
    ctx = _native.Context()
    for invalid in (dict(spec, tier=3), dict(spec, r_max=1.), dict(spec, width=0.)):
        with pytest.raises(ValueError):
            ctx.sample_basis(geometry_from_atoms(atom), points, basis_spec=json.dumps(invalid))
        assert ctx.memory_used_bytes == 0
