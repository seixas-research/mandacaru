# -*- coding: utf-8 -*-
# file: test/algorithms/test_spectral_mapping.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Charged-sector ladder transitions in every fermion-to-qubit encoding."""

import numpy as np
import pytest

from mandacaru.algorithms.spectral_mapping import SpectralRegister
from mandacaru.core.mapping import (Fermion, PauliSum,
                                    reference_qubit_bits)
from mandacaru.core.sector import ParticleSector


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                     "parity_reduced", "bravyi_kitaev"])
def test_sector_sized_reference_obeys_ladder_anticommutation(mapping):
    """A 16-mode sector vector has the right addition and removal norms."""
    n_modes = 16
    particles = (1, 1)
    width = n_modes - (2 if mapping == "parity_reduced" else 0)
    sector = ParticleSector(width, particles, mapping)
    bits = reference_qubit_bits(mapping, n_modes, occupied=(0, 8))
    index = int(sum(int(bit) << (width - 1 - q)
                    for q, bit in enumerate(bits)))
    state = sector.basis_vector(index)
    fermion_h = Fermion({(): 1.0}, n_modes=n_modes)
    register = SpectralRegister(
        mapping, n_modes, particles, state,
        PauliSum.identity(width), fermion_hamiltonian=fermion_h)

    for orbital in (0, 2):
        creation = Fermion.creation(orbital)
        added = register.transition(creation, (2, 1), addition=True)
        removed = register.transition(creation, (0, 1), addition=False)
        assert np.linalg.norm(added)**2 + np.linalg.norm(removed)**2 == \
            pytest.approx(1.0, abs=1e-12)
        assert added.size == register.sector((2, 1)).dim
        assert removed.size == register.sector((0, 1)).dim

    if mapping == "parity_reduced":
        assert register.hamiltonian((2, 1)).num_qubits == width
        assert register.hamiltonian((0, 1)).num_qubits == width


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                     "parity_reduced", "bravyi_kitaev"])
def test_complex_transition_matches_fock_space_matrix(mapping):
    """Charged amplitudes, including phases, match an independent Fock oracle."""
    n_modes = 4
    particles = (1, 1)
    fock_sector = ParticleSector(n_modes, particles, "jordan_wigner")
    amplitudes = np.array([0.5, 0.5j, -0.5, -0.5j])
    physical = {tuple(occupation): amplitude
                for occupation, amplitude in zip(fock_sector.occupations,
                                                 amplitudes)}
    width = n_modes - (2 if mapping == "parity_reduced" else 0)
    source_sector = ParticleSector(width, particles, mapping)
    state = np.array([physical[tuple(occupation)]
                      for occupation in source_sector.occupations])
    creation = (Fermion.creation(0) * (0.6 + 0.2j)
                + Fermion.creation(1) * (-0.3 + 0.7j))
    register = SpectralRegister(
        mapping, n_modes, particles, state, PauliSum.identity(width),
        fermion_hamiltonian=Fermion({(): 1.0}, n_modes=n_modes))
    matrix = creation.to_matrix(n_modes=n_modes)
    fock_state = fock_sector.embed(amplitudes)

    for target, addition, expected_full in (
            ((2, 1), True, matrix @ fock_state),
            ((0, 1), False, matrix.conj().T @ fock_state)):
        actual_sector = register.sector(target)
        actual = register.transition(creation, target, addition=addition)
        expected_sector = ParticleSector(n_modes, target, "jordan_wigner")
        expected = expected_sector.project(expected_full)
        expected_by_occupation = {
            tuple(occupation): amplitude
            for occupation, amplitude in zip(expected_sector.occupations,
                                             expected)}
        ordered_expected = np.array([
            expected_by_occupation[tuple(occupation)]
            for occupation in actual_sector.occupations])
        assert actual == pytest.approx(ordered_expected, abs=1e-12)
