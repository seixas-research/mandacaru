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
