"""Parity of the native Bloch supercell Hamiltonian with the removed periodic engine.

The Python periodic engine (``core/periodic.py`` with the ``_BlochMixin`` drivers, removed in the
Rust cutover; recoverable with ``git show 241d688^:src/mandacaru/algorithms/bloch.py``) pinned
these energies per primitive cell (its methods were ``bloch-vqe`` and ``bloch-adapt-vqe``; today
``vqe`` and ``adapt-vqe`` take that route on any structure with a periodic axis). The cutover had
left those methods running an isolated
cluster (-12.0319 eV on the first case); the native supercell (``molecular::periodic``)
reproduces the old lattice-summed values:

- H chain, a = 2.0 A, HAO, grid_spacing 0.35, Jordan-Wigner, SLSQP tol 1e-12, kpts (2,1,1): old
  pin -12.422242898937355 eV, native -12.422242881249 eV (1.8e-8 eV apart, measured 2026-11-09);
- insulating H2 chain, a = 3.0 A, d = 0.74 A, 10 A vacuum, grid_spacing 0.30, SLSQP: the old
  module docstring quoted -29.315 and -28.142 eV per cell at 1 and 2 k-points; native
  -29.31494 and -28.14154.

- metallic H chain, a = 2.0 A, kpts (4,1,1): old pin -11.2629466539 eV. The native RHF used to stop
  on a level-shifted, non-aufbau saddle (-9.3217 eV per cell; the occupied orbital above an empty
  one) whose number-parity sector excludes the ground state, so VQE stalled at -10.241 and ADAPT
  at -10.809. The SCF's aufbau gate (``molecular/scf.rs``) restarts from it with the degenerate
  k = +-1/4 pair mixed, reaching the RHF minimum (-9.7814 eV per cell); from there both methods
  reproduce the old pin. The exact ground state of the same Hamiltonian is -11.2825 eV: the
  fermionic pool stops 0.0196 eV above it, in both engines (diagnosed 2026-11-09).
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

SLSQP = {"method": "SLSQP", "maxiter": 1000, "tol": 1e-12}


def test_the_hydrogen_chain_reproduces_the_old_periodic_pin():
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([2.0, 10.0, 10.0]),
                  pbc=[True, False, False])
    atoms.calc = Mandacaru(method="vqe", crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.35,
                           mapping="jordan_wigner", kpts={"size": (2, 1, 1), "gamma": True},
                           optimizer=SLSQP, trace=False)
    assert atoms.get_potential_energy() == pytest.approx(-12.422242898937355, abs=5e-8)


@pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
def test_the_metallic_four_cell_chain_reproduces_the_old_pin(method):
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([2.0, 10.0, 10.0]),
                  pbc=[True, False, False])
    # The old pin is the fermionic pool's stalled energy: ``adapt-vqe`` defaults to ``ceo``,
    # so this frozen reference names its pool.
    atoms.calc = Mandacaru(method=method, pool="fermionic", crystal_orbitals="real-space",
                           basis="HAO", grid_spacing=0.35,
                           mapping="jordan_wigner", kpts={"size": (4, 1, 1), "gamma": True},
                           optimizer=SLSQP, trace=False)
    assert atoms.get_potential_energy() == pytest.approx(-11.2629466539, abs=1e-6)


def test_the_default_pool_of_adapt_vqe_reaches_the_exact_four_cell_energy():
    """``adapt-vqe`` on a crystal defaults to ``ceo`` (owner decision 2026-11-09), which reaches the exact
    ground state of the supercell Hamiltonian, -11.2825 eV, where the fermionic pool stops
    0.0196 eV above it."""
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([2.0, 10.0, 10.0]),
                  pbc=[True, False, False])
    atoms.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.35,
                           mapping="jordan_wigner", kpts={"size": (4, 1, 1), "gamma": True},
                           optimizer=SLSQP, trace=False)
    assert atoms.calc._pool == "ceo"
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(-11.2825, abs=1e-4)
    assert energy == pytest.approx(-11.28253862, abs=1e-6)   # measured 2026-11-09


@pytest.mark.parametrize("size, old", [((1, 1, 1), -29.315), ((2, 1, 1), -28.142)])
def test_the_insulating_h2_chain_reproduces_the_old_table(size, old):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=np.diag([3.0, 10.0, 10.0]),
                  pbc=[True, False, False])
    atoms.calc = Mandacaru(method="vqe", crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.30,
                           kpts={"size": size, "gamma": True}, optimizer=SLSQP, trace=False)
    # The old values were quoted to the meV.
    assert atoms.get_potential_energy() == pytest.approx(old, abs=1e-3)
