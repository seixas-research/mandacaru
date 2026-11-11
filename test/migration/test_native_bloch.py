"""The native Bloch supercell pins, and why they no longer match the removed periodic engine.

The Python periodic engine (``core/periodic.py`` with the ``_BlochMixin`` drivers, removed in the
Rust cutover; recoverable with ``git show 241d688^:src/mandacaru/algorithms/bloch.py``) pinned
energies per primitive cell (its methods were ``bloch-vqe`` and ``bloch-adapt-vqe``; today ``vqe``
and ``adapt-vqe`` take that route on any structure with a periodic axis). The first native
supercell (2026-11-09) reproduced them: -12.4222428812 against -12.4222428989 eV, and the H2-chain
table -29.31494 / -28.14154 against -29.315 / -28.142.

**That parity is given up on purpose (2026-11-10).** Both engines point-sampled the hydrogen 1s
cusp and a softened -Z/r on a grid that does not move with the atoms, so their energy depended on
where the crystal sat on the grid: a rigid translation of an H2 chain changed it by up to 0.65 eV
per cell. The supercell now uses band-limited orbitals (truncated at pi / (2 max h)) and the ion
potential as the grid's own Fourier series. The independent checks of the new numbers are exact
translation invariance (1e-12 eV per cell under sub-grid shifts) and monotone convergence with the
grid (``crates/mandacaru-core/tests/molecular_periodic.rs``). Every pin, old -> new, eV per cell:

| case | old (and old-engine) pin | new pin |
|---|---|---|
| H chain a = 2.0 A, (2,1,1), HAO, 0.35, JW, SLSQP | -12.4222428812 (old engine -12.4222428989) | -13.382786723210 |
| same, (4,1,1), fermionic pool, both methods | -11.2629466539 | -12.382599787022 |
| same, (4,1,1), ``adapt-vqe`` at its default ``ceo`` (exact) | -11.28253862 | -12.399508550047 |
| H2 chain a = 3.0 A, d = 0.74 A, 10 A vacuum, 0.30, (1,1,1) | -29.31494 (old engine -29.315) | -29.101976418500 |
| same, (2,1,1) | -28.14154 (old engine -28.142) | -27.979588028527 |

The fermionic pool still stops above the exact 4-cell energy, by 0.0169 eV (0.0196 before).
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

SLSQP = {"method": "SLSQP", "maxiter": 1000, "tol": 1e-12}


def test_the_two_cell_hydrogen_chain_pin():
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([2.0, 10.0, 10.0]),
                  pbc=[True, False, False])
    atoms.calc = Mandacaru(method="vqe", crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.35,
                           mapping="jordan_wigner", kpts={"size": (2, 1, 1), "gamma": True},
                           optimizer=SLSQP, trace=False)
    assert atoms.get_potential_energy() == pytest.approx(-13.382786723210, abs=1e-7)


@pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
def test_the_metallic_four_cell_chain_pin_with_the_fermionic_pool(method):
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([2.0, 10.0, 10.0]),
                  pbc=[True, False, False])
    # The fermionic pool's stalled energy: ``adapt-vqe`` defaults to ``ceo``, so this pin names
    # its pool.
    atoms.calc = Mandacaru(method=method, pool="fermionic", crystal_orbitals="real-space",
                           basis="HAO", grid_spacing=0.35,
                           mapping="jordan_wigner", kpts={"size": (4, 1, 1), "gamma": True},
                           optimizer=SLSQP, trace=False)
    assert atoms.get_potential_energy() == pytest.approx(-12.382599787022, abs=1e-6)


def test_the_default_pool_of_adapt_vqe_reaches_the_exact_four_cell_energy():
    """``adapt-vqe`` on a crystal defaults to ``ceo`` (owner decision 2026-11-09), which reaches
    the exact ground state of the supercell Hamiltonian, -12.399508550047 eV per cell, where the
    fermionic pool stops 0.0169 eV above it."""
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([2.0, 10.0, 10.0]),
                  pbc=[True, False, False])
    atoms.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.35,
                           mapping="jordan_wigner", kpts={"size": (4, 1, 1), "gamma": True},
                           optimizer=SLSQP, trace=False)
    assert atoms.calc._pool == "ceo"
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(-12.399508550047, abs=1e-6)   # measured 2026-11-10


@pytest.mark.parametrize("size, pin", [((1, 1, 1), -29.101976418500),
                                       ((2, 1, 1), -27.979588028527)])
def test_the_insulating_h2_chain_pins(size, pin):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=np.diag([3.0, 10.0, 10.0]),
                  pbc=[True, False, False])
    atoms.calc = Mandacaru(method="vqe", crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.30,
                           kpts={"size": size, "gamma": True}, optimizer=SLSQP, trace=False)
    assert atoms.get_potential_energy() == pytest.approx(pin, abs=1e-6)
