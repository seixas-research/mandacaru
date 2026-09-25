"""Length-gauge sign, spin ordering, mapping, and transition matrix elements."""

import numpy as np
import pytest

from mandacaru.core import Fermion, electric_dipole_potential, electric_field_vector


def test_spatial_dipole_sign_spin_blocks_and_nuclear_phase():
    positions = np.zeros((3, 2, 2), dtype=complex)
    positions[2] = [[0.5, 0.3j], [-0.3j, -0.2]]
    field = [0, 0, 0.7]
    dipole = electric_dipole_potential(positions, field, nuclear_dipole=[0, 0, 0.1])
    expected = Fermion.from_integrals(np.kron(np.eye(2), 0.7*positions[2])).to_matrix()
    expected -= 0.07*np.eye(16)
    np.testing.assert_allclose(dipole.to_matrix(), expected, atol=1e-14)
    assert dipole.is_hermitian()


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev",
                                     "parity_reduced"])
def test_particle_conservation_and_encodings(mapping):
    positions = np.zeros((3, 2, 2))
    positions[0] = [[0.2, 0.8], [0.8, -0.3]]
    v = electric_dipole_potential(positions, [0.1, 0, 0], mapping=mapping,
                                  num_particles=(1, 1))
    number = Fermion.from_integrals(np.eye(4)).map_to_qubits(mapping, num_particles=(1, 1))
    np.testing.assert_allclose(v.to_matrix() @ number.to_matrix(),
                               number.to_matrix() @ v.to_matrix(), atol=1e-14)
    expected = Fermion.from_integrals(np.kron(np.eye(2), 0.1*positions[0]))
    np.testing.assert_allclose(v.to_matrix(),
                               expected.map_to_qubits(mapping, num_particles=(1, 1)).to_matrix())


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev",
                                     "parity_reduced"])
def test_zero_field_keeps_width(mapping):
    v = electric_dipole_potential(np.zeros((3, 2, 2)), [0, 0, 0],
                                  mapping=mapping, num_particles=(1, 1))
    assert v.num_qubits == (2 if mapping == "parity_reduced" else 4)
    assert not v.terms


def test_spin_orbital_input_and_transition():
    positions = np.zeros((3, 2, 2))
    positions[1] = [[0, 1], [1, 0]]
    v = electric_dipole_potential(positions, [0, 0.2, 0], spatial_orbitals=False)
    assert v.num_qubits == 2
    # One electron moves between the two orbitals; no vacuum transition.
    assert v.to_matrix()[1, 2] == pytest.approx(0.2)
    assert v.to_matrix()[0, 2] == 0


@pytest.mark.parametrize("positions", [np.zeros((2, 2)), np.zeros((3, 0, 0)),
                                       np.zeros((3, 2, 3)), np.full((3, 2, 2), np.nan),
                                       np.full((3, 2, 2), 1j)])
def test_invalid_positions(positions):
    with pytest.raises(ValueError, match="position_integrals"):
        electric_dipole_potential(positions)


@pytest.mark.parametrize("name", ["field", "nuclear_dipole"])
@pytest.mark.parametrize("value", [[1, 2], [0, 0, np.inf], [0, 0, 1j]])
def test_invalid_cartesian_vectors(name, value):
    with pytest.raises(ValueError, match=name):
        electric_dipole_potential(np.zeros((3, 2, 2)), **{name: value})


def test_direction_and_strength_match_cartesian_coupling():
    positions = np.array([[[1, 0], [0, -1]], [[0, 1], [1, 0]],
                          [[0, -1j], [1j, 0]]])
    direction = np.array([1, 2, -2])/3
    potential = electric_dipole_potential(
        positions, field_direction=direction, field_strength=0.02,
        nuclear_dipole=[0.1, 0.3, -0.2])
    expected = electric_dipole_potential(
        positions, field=0.02*direction, nuclear_dipole=[0.1, 0.3, -0.2])
    np.testing.assert_allclose(potential.to_matrix(), expected.to_matrix())
    opposite = electric_dipole_potential(
        positions, field_direction=-direction, field_strength=0.02,
        nuclear_dipole=[0.1, 0.3, -0.2])
    np.testing.assert_allclose(opposite.to_matrix(), -expected.to_matrix())
    np.testing.assert_array_equal(electric_field_vector([1, 0, 0], 0), [0, 0, 0])


@pytest.mark.parametrize("direction", [[0, 0, 0], [1, 1, 1], [2, 0, 0],
                                       [1, 0], [np.nan, 0, 1], [0, 0, 1j]])
def test_invalid_unit_direction(direction):
    with pytest.raises(ValueError, match="field_direction"):
        electric_field_vector(direction, 0.01)


@pytest.mark.parametrize("strength", [-0.1, np.nan, np.inf, 1j, [0.1]])
def test_invalid_field_strength(strength):
    with pytest.raises(ValueError, match="field_strength"):
        electric_field_vector([0, 0, 1], strength)


@pytest.mark.parametrize("options", [
    {"field_direction": [1, 0, 0]}, {"field_strength": 0.01},
    {"field": [0, 0, 1], "field_direction": [1, 0, 0], "field_strength": 0.01},
])
def test_ambiguous_or_incomplete_field_input(options):
    with pytest.raises(ValueError, match="supply both"):
        electric_dipole_potential(np.zeros((3, 2, 2)), **options)
