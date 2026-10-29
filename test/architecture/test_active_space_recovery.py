"""Native selection must retain its orbital gauge and avoid a full MO tensor."""

import json

from ase import Atoms
import numpy as np
import pytest

from mandacaru import Mandacaru


def setup():
    return Mandacaru(
        method="rhf",
        basis={"name": "HAO", "virtual_orbitals": 1},
        grid_spacing=0.5,
        resources={"threads": 1},
        atoms=Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5] * 3),
    )


def test_mp2_selection_matches_dense_reference_and_preserves_rotated_density():
    with setup() as calc:
        calc.get_potential_energy()
        reference = calc.native_reference
        assert not reference.has_model
        selected = reference.active_model(json.dumps({"method": "mp2", "orbitals": 3}))
        assert selected.n_spatial_orbitals == 3
        assert not reference.has_model
        rotation = selected.orbital_rotation
        m = calc.integrals.n_orbitals
        assert rotation.shape == (m, m)
        assert np.allclose(rotation.conj().T @ rotation, np.eye(m), atol=1e-12)
        dense = reference.model
        mp2 = dense.mp2()
        assert selected.mp2_correlation_energy == pytest.approx(
            mp2["correlation_energy"], abs=1e-10
        )
        assert np.sum(selected.selection_occupations) == pytest.approx(2.0, abs=1e-12)
        # Independent full-source contractions in the same selected orbital gauge.
        c = rotation[:, selected.orbital_indices]
        expected_h = c.conj().T @ dense.one_body @ c
        expected_g = np.einsum(
            "ap,bq,cr,ds,abcd->pqrs", c.conj(), c.conj(), c, c, dense.two_body
        )
        assert np.allclose(selected.one_body, expected_h, atol=1e-11)
        assert np.allclose(selected.two_body, expected_g, atol=1e-11)
        # A complex one-electron orbital has known density after the FNO rotation.
        amplitude = np.array([1.0, 1j, -0.3]) / np.sqrt(2.09)
        gamma = np.zeros((6, 6), complex)
        gamma[:3, :3] = np.outer(amplitude.conj(), amplitude)
        view = reference.observables(gamma=gamma, model=selected)
        lowdin = calc.integrals.snapshot()["lowdin"]
        source = reference.snapshot()["model_orbitals"]
        expected = lowdin @ source @ c @ amplitude
        actual = view["coefficients"][:, 0]
        assert np.allclose(
            np.outer(actual, actual.conj()),
            np.outer(expected, expected.conj()),
            atol=1e-11,
        )


def test_energy_selection_frozen_core_matches_dense_reduction_without_eager_model():
    with setup() as calc:
        calc.get_potential_energy()
        reference = calc.native_reference
        selected = reference.active_model(json.dumps({"frozen": 1, "orbitals": 2}))
        assert not reference.has_model
        assert selected.num_particles == [0, 0]
        dense = reference.model.reduce([0], [1, 2])
        assert selected.constant_energy == pytest.approx(
            dense.constant_energy, abs=1e-12
        )
        assert np.allclose(selected.one_body, dense.one_body, atol=1e-12)
        assert np.allclose(selected.two_body, dense.two_body, atol=1e-12)


def test_selection_refuses_unported_constraints_and_invalid_counts_before_building():
    with setup() as calc:
        calc.get_potential_energy()
        for spec in (
            {"method": "unsupported-method"},
            {"method": "mp2", "orbitals": [0, 1]},
            {"orbitals": 0},
            {"frozen": [2]},
            {"threshold": 0.1},
            {"unrecognized": 1},
        ):
            with pytest.raises(ValueError):
                calc.native_reference.active_model(json.dumps(spec))
            assert not calc.native_reference.has_model


def test_symmetry_selection_keeps_atomic_p_multiplets_whole():
    atoms = Atoms("He", positions=[[2.5, 2.5, 2.5]], cell=[5] * 3)
    with Mandacaru(
        method="rhf",
        atoms=atoms,
        basis={"name": "HAO", "virtual_orbitals": 2},
        grid_spacing=0.3,
        resources={"threads": 1},
    ) as calc:
        atoms.get_potential_energy()
        for method in ("energy", "mp2"):
            selected = calc.native_reference.active_model(
                json.dumps({"method": method, "orbitals": 3, "symmetry": True})
            )
            assert not calc.native_reference.has_model
            sets = selected.orbital_symmetry_sets
            assert any(len(group) == 3 for group in sets)
            active = set(selected.orbital_indices)
            for group in sets:
                assert not active.intersection(group) or set(group) <= active
            assert selected.n_spatial_orbitals == 2


def test_threshold_and_occupied_count_preserve_frozen_density_and_hartree_fock_energy():
    with setup() as calc:
        energy = calc.get_potential_energy()
        reference = calc.native_reference
        threshold = reference.active_model(
            json.dumps({"method": "mp2", "threshold": 1e-6})
        )
        assert not reference.has_model
        assert all(
            threshold.selection_occupations[i] >= 1e-6
            for i in threshold.orbital_indices
            if i >= 1
        )
        frozen = reference.active_model(
            json.dumps({"method": "mp2", "orbitals": {"occupied": 0, "virtual": 2}})
        )
        assert frozen.num_particles == [0, 0]
        assert frozen.constant_energy == pytest.approx(
            energy / 27.211386245988, abs=1e-10
        )
        values = reference.observables(
            gamma=np.zeros((4, 4), complex), model=frozen, quantity="density"
        )
        assert sum(values["occupations"]) == pytest.approx(2.0, abs=1e-12)
        assert values["integral"] == pytest.approx(2.0, abs=1e-11)


@pytest.mark.parametrize("method", ["adapt-vqe", "subspace-adapt-vqe"])
def test_quantum_result_retains_selected_orbitals_for_every_state_after_close(method):
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5] * 3)
    with Mandacaru(
        method=method,
        atoms=atoms,
        basis={"name": "HAO", "virtual_orbitals": 1},
        grid_spacing=0.5,
        active_space={"method": "mp2", "orbitals": 3},
        resources={"threads": 1},
    ) as calc:
        atoms.get_potential_energy()
        result = calc.result
        model = result.hamiltonian
        assert not calc.native_reference.has_model
        source = calc.native_reference.snapshot()["model_orbitals"]
        lowdin = calc.integrals.snapshot()["lowdin"]
        coefficients = (
            lowdin @ source @ model.orbital_rotation[:, model.orbital_indices]
        )
        states = result.snapshot().get("state", result.snapshot().get("states"))
        states = states[None, :] if states.ndim == 1 else states
        expected = []
        for state in states:
            gamma, _ = calc.problem.rdms(state, two_body=False)
            spin_sum = gamma[:3, :3].T + gamma[3:, 3:].T
            expected.append(coefficients @ spin_sum @ coefficients.conj().T)
    for i, density in enumerate(expected):
        natural = result.natural_orbitals(state=i)
        actual = (
            natural.coefficients * natural.occupations
        ) @ natural.coefficients.conj().T
        assert np.allclose(actual, density, atol=1e-11)
        assert sum(natural.occupations) == pytest.approx(2.0, abs=1e-11)
