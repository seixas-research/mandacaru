# -*- coding: utf-8 -*-
# file: test/algorithms/test_spinor_dft.py

# This code is part of Mandacaru.
# MIT License

"""Two-component (spinor) Kohn-Sham behind ``Mandacaru(method="dft")`` with a
spin-orbit dataset (``paw_dataset="lda-fr"`` / ``"pbe-fr"``), for crystals.

Molecular spin-orbit datasets, noncollinear moment vectors and the spinor
meta-GGA/hybrid crystal are not reachable through ``Mandacaru`` yet
(``docs/source/development/native_cutover.md``, "Remaining cutover work")."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

SZ = {"name": "PAW-LCAO", "size": "SZ"}


def _oh():
    atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]],
                  cell=[6, 6, 7])
    atoms.center()
    return atoms


class TestMomentVectors:
    """A moment is one number per atom: a noncollinear (N, 3) start is
    refused for every method instead of being collapsed to a collinear one."""

    @pytest.mark.parametrize("method", ["uhf", "dft"])
    def test_every_method_refuses_moment_vectors(self, method):
        atoms = _oh()
        atoms.set_initial_magnetic_moments([[1.0, 0.0, 0.0], [0.0] * 3])
        atoms.calc = Mandacaru(method=method, grid_spacing=0.4, basis=SZ,
                               trace=False)
        with pytest.raises(ValueError, match="one finite value per atom"):
            atoms.get_potential_energy()

    def test_forces_are_refused(self):
        atoms = _oh()
        atoms.set_initial_magnetic_moments([[0.0, 0.0, 1.0], [0.0] * 3])
        atoms.calc = Mandacaru(method="dft", grid_spacing=0.3, basis=SZ,
                               trace=False)
        with pytest.raises(ValueError, match="one finite value per atom"):
            atoms.get_forces()


# --------------------------------------------------------------------------- #
# Crystals: two-component Bloch states.
# --------------------------------------------------------------------------- #

#: Diamond Si's basis for the crystal tests: SZ, filtered at 200 eV for the
#: coarse grid (h = 0.34 Angstrom), as test_periodic_dft.py's silicon.
SILICON_BASIS = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}


def _silicon(xc="lda", kpts=(2, 2, 2), **options):
    """Diamond Si on the spin-orbit set of ``xc`` (by default), 2x2x2."""
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43)
    options.setdefault("paw_dataset", "lda-fr" if xc == "lda" else "pbe-fr")
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.34,
                           kpts={"size": kpts, "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.01},
                           basis=SILICON_BASIS, trace=False, **options)
    atoms.get_potential_energy()
    return atoms


def _oxygen_box(moments, xc="lda", **options):
    """O2 in a periodic box at Gamma (a spin crystal), on the scalar set of
    ``xc`` unless ``options`` name another."""
    from ase.build import molecule

    atoms = molecule("O2")
    atoms.set_cell([7.0, 7.0, 7.5])
    atoms.center()
    atoms.pbc = True
    atoms.set_initial_magnetic_moments(None)
    atoms.set_initial_magnetic_moments(moments)
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.34,
                           smearing={"method": "fermi-dirac", "width": 0.01},
                           basis=SILICON_BASIS, trace=False, **options)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def spinor_silicon():
    """Diamond Si on ``lda-fr``."""
    return _silicon()


class TestASpinOrbitCrystal:
    """Diamond Si on the LDA spin-orbit set: a non-magnetic insulator in
    Bloch spinors, time reversal imposed."""

    def test_it_is_solved_in_spinors(self, spinor_silicon):
        calc = spinor_silicon.calc
        snapshot = calc.result.snapshot()
        scf = calc.result.scf
        assert scf.converged and snapshot["spinor"]
        assert snapshot["time_reversal"]
        assert calc.get_number_of_spins() == 1
        # 8 spatial functions: 16 spinor bands of one electron each.
        assert calc.get_eigenvalues(kpt=0).shape == (16,)
        # Point-group averaging is off for spinors, so the 2x2x2 Gamma mesh
        # (all time-reversal-invariant points) keeps its eight points.
        assert len(calc.get_ibz_k_points()) == 8
        for f in snapshot["occupations"][0]:
            assert np.max(f) <= 1.0 + 1e-12
            assert np.sum(f) == pytest.approx(8.0, abs=1e-8)
        assert calc.get_magnetic_moment() == 0.0
        # Second order for a closed shell: a fraction of a meV for Si.
        assert -1e-3 < scf.spin_orbit_energy * HARTREE_TO_EV < 0.0

    def test_bands_at_a_time_reversal_point_are_kramers_pairs(
            self, spinor_silicon):
        gamma = spinor_silicon.calc.get_eigenvalues(kpt=0)
        assert np.allclose(gamma[0::2], gamma[1::2], atol=1e-10)

    def test_bands_are_even_in_k(self, spinor_silicon):
        k = np.array([[0.13, 0.27, -0.31]])
        plus = spinor_silicon.calc._bands_at(k)
        minus = spinor_silicon.calc._bands_at(-k)
        assert np.allclose(plus, minus, atol=1e-12)

    def test_the_valence_top_splits_off_a_kramers_pair(self, spinor_silicon):
        """At Gamma the p-like valence top (the scalar triplet) is the
        quartet Gamma8 above the pair Gamma7: 58 meV for this coarse SZ
        silicon (DZP h 0.15 A: HISTORY "Phase P step 2")."""
        top = np.sort(spinor_silicon.calc.get_eigenvalues(kpt=0))[2:8]
        assert np.ptp(top[2:]) < 1e-5 and np.ptp(top[:2]) < 1e-10
        assert 0.04 < top[2] - top[1] < 0.08

    def test_the_spectra_count_one_electron_per_band(self, spinor_silicon):
        calc = spinor_silicon.calc
        energies, dos = calc.dos(width=0.05)
        assert np.trapezoid(dos, energies) == pytest.approx(16.0, abs=1e-6)
        _e, pdos = calc.pdos(width=0.05, energies=energies)
        assert np.allclose(sum(pdos.values()), dos, atol=1e-10)

    def test_what_it_cannot_do_is_refused(self, spinor_silicon):
        calc = spinor_silicon.calc
        with pytest.raises(NotImplementedError, match="not implemented"):
            calc.get_forces(spinor_silicon)
        with pytest.raises(NotImplementedError, match="not implemented"):
            calc.get_stress(spinor_silicon)

    def test_wannier_functions_with_the_term(self, spinor_silicon):
        """The SOC crystal's valence: eight spinor functions whose
        Hamiltonian gives back the spinor bands on the mesh -- the split-off
        pair at Gamma included."""
        calc = spinor_silicon.calc
        result = calc.wannier()
        assert result.n_functions == 8
        for kpt, k in enumerate(calc.get_ibz_k_points()):
            assert np.allclose(result.interpolate([k])[0],
                               calc.get_eigenvalues(kpt=kpt)[:8], atol=1e-8)
        top = result.interpolate([[0.0, 0.0, 0.0]])[0][2:]
        assert top[2] - top[1] == pytest.approx(
            np.sort(calc.get_eigenvalues(kpt=0))[4]
            - np.sort(calc.get_eigenvalues(kpt=0))[3], abs=1e-8)
        assert top[2] - top[1] > 0.04


@pytest.mark.slow
def test_a_magnetic_crystal_with_spin_orbit_coupling():
    """The O2 triplet in a box on ``lda-fr`` with collinear moments: the
    spin-orbit term lowers the energy, and the crystal is solved on the
    whole mesh."""
    atoms = _oxygen_box([1.0, 1.0], paw_dataset="lda-fr")
    snapshot = atoms.calc.result.snapshot()
    assert snapshot["converged"] and snapshot["spinor"]
    assert not snapshot["time_reversal"]
    assert snapshot["magnetic_moment"] == pytest.approx(2.0, abs=0.01)
    assert atoms.calc.result.scf.spin_orbit_energy < 0.0
    assert atoms.calc.get_magnetic_moment() == pytest.approx(
        snapshot["magnetic_moment"])


def test_bi2te3_spin_orbit_coupling_opens_band_gap():
    """Bulk Bi2Te3 on pbe-fr: strong spin-orbit coupling of Bi (Z=83)
    and Te (Z=52) yields a substantial negative spin-orbit energy and
    exact Kramers degeneracy at time-reversal invariant points."""
    from ase.spacegroup import crystal

    atoms = crystal(
        symbols=["Te", "Bi", "Te"],
        basis=[(0, 0, 0.2117), (0, 0, 0.3990), (0, 0, 0.0)],
        spacegroup=166,
        cellpar=[4.383, 4.383, 30.487, 90, 90, 120],
        primitive_cell=True,
    )
    atoms.calc = Mandacaru(
        method="dft",
        xc="pbe",
        paw_dataset="pbe-fr",
        basis={"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": True},
        kpts={"size": (2, 2, 2), "gamma": True},
        grid_spacing=0.25,
        trace=False,
        periodic_scf_options={"max_iterations": 30},
    )
    atoms.get_potential_energy()
    snapshot = atoms.calc.result.snapshot()
    assert snapshot["converged"] and snapshot["spinor"]
    assert snapshot["time_reversal"]
    assert atoms.calc.result.scf.spin_orbit_energy * HARTREE_TO_EV < -0.5
    gamma = atoms.calc.get_eigenvalues(kpt=0)
    assert np.allclose(gamma[0::2], gamma[1::2], atol=1e-10)
