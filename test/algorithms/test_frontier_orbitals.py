"""Canonical RHF HOMO/LUMO energies and well-defined occupied/virtual edges."""

import numpy as np
import pytest

from mandacaru.algorithms import RHFResult


def reference(energies, occupied=1, converged=True):
    """Minimal RHF result for independently specified orbital energies."""
    n = len(energies)
    return RHFResult(electronic_energy=-1.0, mo_energies=np.array(energies),
                     mo_coefficients=np.eye(n), n_occupied=occupied,
                     converged=converged, h_mo=np.zeros((n, n)),
                     eri_mo=np.zeros((n, n, n, n)))


def test_homo_lumo_gap_uses_occupied_boundary_not_total_energies():
    rhf = reference([-1.5, -0.7, 0.2, 0.4], occupied=2)
    assert rhf.homo_energy == pytest.approx(-0.7)
    assert rhf.lumo_energy == pytest.approx(0.2)
    assert rhf.homo_lumo_gap == pytest.approx(0.9)
    rhf.electronic_energy = -100
    assert rhf.homo_lumo_gap == pytest.approx(0.9)


def test_degenerate_frontier_has_zero_gap():
    assert reference([-0.1, -0.1]).homo_lumo_gap == 0


@pytest.mark.parametrize("occupied", [0, 2, -1, 1.5, True])
def test_missing_or_invalid_occupied_virtual_partition(occupied):
    with pytest.raises(ValueError, match="occupied and unoccupied"):
        _ = reference([-1, 0.2], occupied=occupied).homo_lumo_gap


@pytest.mark.parametrize("energies", [[0.3, -1], [-1, np.nan], [-1, np.inf], [-1, 1j]])
def test_invalid_orbital_energies(energies):
    with pytest.raises(ValueError, match="mo_energies"):
        _ = reference(energies).homo_lumo_gap


def test_unconverged_reference_is_rejected():
    with pytest.raises(ValueError, match="converged RHF"):
        _ = reference([-1, 0.2], converged=False).homo_lumo_gap
