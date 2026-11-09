"""MG-F4: every cell of the forces table reaches the Rust analytic route.

The 30 cells are {molecule, crystal} x {LDA, PBE, r2SCAN, rVV10, HSE06} x {unpolarized,
spin-polarized, spin-orbit}. Their accuracy is gated in Rust (``dev_plan/STATUS.md`` section 2a
names each gate); this test pins only the route from ``Mandacaru(...).get_forces()``. A
molecule's force reports the analytic scheme of its stationary state, and a crystal's comes from
the periodic engine, with the finite-difference driver of the quantum methods made to fail, so no
cell can reach a difference of energies.
"""
import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

FUNCTIONALS = ("lda", "pbe", "r2scan", "pbe+rvv10", "hse06")
SPINS = ("unpolarized", "spin-polarized", "spin-orbit")
BASIS = {"name": "PAW-LCAO", "size": "SZ"}


def _dataset(xc, spin):
    if spin != "spin-orbit":
        return {}
    return {"paw_dataset": "lda-fr" if xc == "lda" else "pbe-fr"}


def _molecule(spin):
    if spin == "spin-polarized":
        atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]], cell=[6, 6, 7])
    else:
        atoms = Atoms("HF", positions=[[0, 0, 0], [0.08, 0, 0.95]], cell=[6, 6, 7])
    atoms.center()
    return atoms


def _crystal(spin):
    if spin == "spin-polarized":
        atoms = Atoms("HH", cell=[6.2] * 3, pbc=True, positions=[[3.1, 3.3, 2.0], [3.62, 3.0, 4.0]])
        atoms.set_initial_magnetic_moments([1.0, 1.0])
        return atoms
    return Atoms("LiH", cell=[[6.0, 0, 0], [0.4, 5.8, 0], [0.3, 0.2, 5.6]], pbc=True,
                 positions=[[2.0, 2.1, 2.2], [3.5, 2.2, 2.0]])


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("xc", FUNCTIONALS)
def test_a_molecular_force_is_the_analytic_gradient(xc, spin):
    atoms = _molecule(spin)
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.4, trace=False, basis=BASIS,
                           spin=spin == "spin-polarized", **_dataset(xc, spin))
    forces = atoms.get_forces()
    assert forces.shape == (2, 3) and np.all(np.isfinite(forces))
    assert atoms.calc.result.force_details["scheme"] == "analytic"
    snapshot = atoms.calc.native_reference.snapshot()
    expected = {"unpolarized": "rhf", "spin-polarized": "uhf", "spin-orbit": "ghf"}[spin]
    assert snapshot["method"] == expected and snapshot["functional"] == xc


@pytest.mark.slow
@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("xc", FUNCTIONALS)
def test_a_crystal_force_comes_from_the_periodic_engine(xc, spin, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a crystal force reached the finite-difference driver")

    monkeypatch.setattr(Mandacaru, "_periodic_quantum_forces", refuse)
    atoms = _crystal(spin)
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.45, trace=False,
                           basis={**BASIS, "filter_cutoff": 200},
                           spin=spin == "spin-polarized",
                           kpts={"size": (1, 1, 1), "gamma": True}, symmetry=False,
                           periodic_scf_options={"max_iterations": 200, "tolerance": 1e-9},
                           **_dataset(xc, spin))
    forces = atoms.get_forces()
    assert forces.shape == (2, 3) and np.all(np.isfinite(forces))
    assert atoms.calc.result.periodic is not None


def test_a_magnetic_spinor_hybrid_crystal_force_comes_from_the_periodic_engine(monkeypatch):
    """F2b: HSE06 on a spin-orbit dataset with nonzero moments is a magnetic spinor hybrid (not
    time-reversal symmetric); its force is analytic (gated in Rust by
    ``periodic_gradient_spinor_hybrid_magnetic``). Measured 4 s."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("a crystal force reached the finite-difference driver")

    monkeypatch.setattr(Mandacaru, "_periodic_quantum_forces", refuse)
    atoms = _crystal("spin-polarized")
    atoms.calc = Mandacaru(method="dft", xc="hse06", paw_dataset="pbe-fr", grid_spacing=0.45,
                           trace=False, basis={**BASIS, "filter_cutoff": 200},
                           kpts={"size": (1, 1, 1), "gamma": True}, symmetry=False,
                           periodic_scf_options={"max_iterations": 200, "tolerance": 1e-9})
    forces = atoms.get_forces()
    assert atoms.calc.periodic_problem.spinor
    assert forces.shape == (2, 3) and np.all(np.isfinite(forces))
    assert np.abs(forces).max() > 1e-3
