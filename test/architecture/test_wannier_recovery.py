"""Public ASE/native Wannier behavior, independent band and ownership checks."""
import gc

import numpy as np
from ase.build import bulk
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV


def test_wannier_full_space_recovers_ase_bands_and_releases_owned_frames():
    atoms = bulk("Si", "diamond", a=5.43)
    calc = Mandacaru(method="dft", xc="lda",
                     basis={"name": "PAW-LCAO", "size": "SZ"},
                     grid_spacing=0.8, kpts={"size": (2, 2, 2), "gamma": True},
                     resources={"threads": 1})
    atoms.calc = calc
    atoms.get_potential_energy()
    before = calc.context.memory_used_bytes
    wannier = calc.wannier(8, guess="sp3", max_iterations=250)
    assert wannier.n_functions == 8
    assert np.isfinite(wannier.centers).all()
    assert np.isfinite(wannier.spreads).all()
    assert np.min(wannier.spreads) > -1e-9
    np.testing.assert_allclose(wannier.total_spread, np.sum(wannier.spreads), atol=1e-12)
    assert len(set(wannier.labels)) == 8
    # Same complete band space, different gauges and Fourier representation.
    expected = calc._bands_at(wannier.kpoints)[0] * HARTREE_TO_EV
    np.testing.assert_allclose(wannier.interpolate(wannier.kpoints), expected,
                               atol=2e-8, rtol=0)
    off_mesh = wannier.interpolate([[0.13, 0.27, 0.11], [0.91, 0.04, 0.38]])
    assert off_mesh.shape == (2, 8)
    assert np.isfinite(off_mesh).all()
    centers = wannier.centers
    centers[:] = 123
    assert not np.all(wannier.centers == 123)
    assert calc.context.memory_used_bytes > before
    del wannier
    gc.collect()
    assert calc.context.memory_used_bytes == before


def test_bond_trials_and_frozen_window_keep_the_occupied_band_spectrum():
    atoms = bulk("Si", "diamond", a=5.43)
    calc = Mandacaru(method="dft", xc="lda",
                     basis={"name": "PAW-LCAO", "size": "SZ"},
                     grid_spacing=0.8, kpts={"size": (1, 1, 1), "gamma": True},
                     resources={"threads": 1})
    atoms.calc = calc
    atoms.get_potential_energy()
    fermi = calc.get_fermi_level()
    wannier = calc.wannier(4, guess="bonds",
                          windows={"outer": (-1000., 1000.),
                                   "frozen": (-1000., fermi)},
                          max_iterations=100)
    assert wannier.n_functions == 4
    assert wannier._snapshot["disentanglement_converged"]
    np.testing.assert_allclose(wannier.interpolate([[0., 0., 0.]])[0],
                               calc.get_eigenvalues()[:4], atol=2e-8, rtol=0)


def test_hybrid_wannier_frames_carry_the_exact_exchange():
    """A hybrid Wannier frame is the hybrid Hamiltonian, not the local potential alone.

    The full-space Wannier bands must equal the band path at the mesh (which applies K(k)) and the
    SCF eigenvalues. Without the exchange term in the frame they differ by several eV (7.7 eV for
    this cell), with no error raised.
    """
    atoms = bulk("Si", "diamond", a=5.43)
    calc = Mandacaru(method="dft", xc="hse06",
                     basis={"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200},
                     grid_spacing=0.8, kpts={"size": (1, 1, 1), "gamma": True},
                     resources={"threads": 1}, trace=False)
    atoms.calc = calc
    atoms.get_potential_energy()
    wannier = calc.wannier(8, guess="sp3", max_iterations=250)
    expected = calc._bands_at(wannier.kpoints)[0] * HARTREE_TO_EV
    # Measured 1.6e-12 eV: the same Hamiltonian in another gauge.
    np.testing.assert_allclose(wannier.interpolate(wannier.kpoints), expected,
                               atol=1e-9, rtol=0)
    # Independent of the band path: the eigenvalues of the SCF itself. The exact exchange that
    # defines them is part of the SCF's stopping rule (its change is below ``tolerance``, 1e-7 Ha),
    # so they are the bands of the converged density up to the last change of K. Measured 4.9e-8 eV
    # here, identical in 10 of 10 fresh runs; the bound is 20x that, and the error this test guards
    # against (a frame without K) is 7.7 eV. Before K entered the stopping rule the spread across
    # identical runs was 2.9e-7 .. 3.5e-3 eV.
    np.testing.assert_allclose(np.sort(wannier.interpolate([[0., 0., 0.]])[0])[:8],
                               calc.get_eigenvalues()[:8], atol=1e-6, rtol=0)
