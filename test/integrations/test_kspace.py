"""Symmetry, the irreducible zone and band paths through ``Mandacaru``.

The numerical gates (the zone-boundary geometry of every named lattice, the orbits and weights
of the zone, the tight-binding dispersion and the point-group and time-reversal invariance of
``A(k, omega)``) are in ``crates/mandacaru-core/tests/{kpath,bloch_path}.rs``; here the
calculator hands its structure and mesh across and the views come back. The runs are one
hydrogen atom per cell, 4 to 8 qubits.
"""

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

SLSQP = {"method": "SLSQP", "maxiter": 1000, "tol": 1e-12}


def square(size=(2, 2, 1), a=2.6, **options):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[a, a, 9.0], pbc=[True, True, False])
    atoms.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.35, kpts={"size": size, "gamma": True},
                           optimizer=SLSQP, trace=False, **options)
    return atoms


def chain(size=(4, 1, 1), a=2.0):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[a, 10.0, 10.0],
                  pbc=[True, False, False])
    atoms.calc = Mandacaru(method="vqe", crystal_orbitals="real-space", basis="HAO",
                           grid_spacing=0.35, mapping="jordan_wigner",
                           kpts={"size": size, "gamma": True}, optimizer=SLSQP, trace=False)
    return atoms


class TestTheStructureBeforeAnyRun:
    def test_the_space_group_of_the_square_lattice(self):
        symmetry = square().calc.symmetry()
        assert (symmetry.number, symmetry.international, symmetry.point_group) == (
            123, "P4/mmm", "4/mmm")
        assert symmetry.n_operations == 16 and symmetry.rotations.shape == (16, 3, 3)
        assert "P4/mmm" in symmetry.summary()

    def test_the_irreducible_zone_of_the_two_by_two_mesh(self):
        zone = square().calc.irreducible_zone()
        assert zone.size == (2, 2, 1) and len(zone.full) == 4
        assert sorted(zone.weights) == [0.25, 0.25, 0.5]
        assert zone.weights.sum() == pytest.approx(1.0, abs=1e-15)
        assert zone.space_group_operations == 16 and len(zone.rotations) == 16
        assert "3 of 4 points" in zone.summary()

    def test_a_mesh_the_lattice_does_not_fit_keeps_the_subgroup_that_does(self):
        zone = square(size=(2, 1, 1)).calc.irreducible_zone()
        assert zone.distinct_rotations == 16 and len(zone.rotations) < 16
        assert zone.weights.sum() == pytest.approx(1.0, abs=1e-15)

    def test_a_second_species_lowers_the_symmetry(self):
        atoms = Atoms("HHe", positions=[[0, 0, 0], [1.3, 0, 0]], cell=[2.6, 2.6, 9.0],
                      pbc=[True, True, False])
        atoms.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space",
                               basis="HAO", grid_spacing=0.35, trace=False,
                               kpts={"size": (2, 2, 1), "gamma": True})
        assert atoms.calc.symmetry().n_operations < 16

    def test_the_square_band_path_has_the_distances_of_its_zone(self):
        path = square(size=(4, 4, 1)).calc.band_path("GXMG")
        unit = np.pi / 2.6
        assert path.complete and path.convention == "square"
        assert path.labels == ("G", "X", "M", "G")
        assert np.allclose(path.label_distances, np.array([0, 1, 2, 2 + 2 ** 0.5]) * unit)
        assert np.allclose(path.distances[:5], np.array([0, 0.5, 1, 1.5, 2]) * unit)

    def test_the_default_path_is_the_lattices_own(self):
        assert square(size=(4, 4, 1)).calc.band_path().labels == ("G", "X", "M", "G")

    def test_a_mesh_without_a_vertex_lists_it_as_missing(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[2.6, 2.6, 9.0], pbc=[True, True, False])
        atoms.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", basis="HAO",
                               grid_spacing=0.35, kpts=(3, 3, 1), trace=False)
        path = atoms.calc.band_path("GXMG")
        assert not path.complete and path.missing == ("X", "M") and len(path.kpoints) == 4

    def test_explicit_points_make_a_path_on_any_lattice(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[[2.0, 0, 0], [0.7, 2.9, 0], [0, 0, 9.0]],
                      pbc=[True, True, False])
        atoms.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", basis="HAO",
                               grid_spacing=0.35, kpts={"size": (4, 4, 1), "gamma": True},
                               trace=False)
        with pytest.raises(ValueError, match="explicit points"):
            atoms.calc.band_path("GX")
        path = atoms.calc.band_path([("A", (0, 0, 0)), ("B", (0.5, 0, 0))])
        assert path.convention == "explicit" and path.labels == ("A", "B")
        assert len(path.kpoints) == 3

    @pytest.mark.parametrize("bad, message", [
        ("GXQ", "no high-symmetry point"), ("G", "at least two"), ("GX,", "empty run"),
        ("gx", "capital letter")])
    def test_a_path_that_cannot_be_named_raises(self, bad, message):
        with pytest.raises(ValueError, match=message):
            square(size=(4, 4, 1)).calc.band_path(bad)

    def test_a_calculation_without_a_structure_raises(self):
        calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", basis="HAO",
                         grid_spacing=0.35, trace=False)
        with pytest.raises(RuntimeError, match="no structure"):
            calc.symmetry()

    def test_a_mesh_along_a_vacuum_axis_raises(self):
        atoms = square(size=(2, 2, 2))
        with pytest.raises(ValueError, match="not periodic"):
            atoms.calc.irreducible_zone()


class TestTheSpectralFunctionAlongAPath:
    @pytest.fixture(scope="class")
    def solved(self):
        atoms = chain()
        atoms.get_potential_energy()
        return atoms.calc

    def test_the_rows_are_the_mesh_points_on_the_path_in_order(self, solved):
        spectral = solved.get_spectral_function(path="GX", eta=0.2)
        assert spectral.kpath.complete and spectral.kpath.convention == "chain"
        assert np.allclose(spectral.kpoints[:, 0], [0.0, 0.25, 0.5])
        assert np.allclose(spectral.kpath.distances, np.array([0, 0.5, 1.0]) * np.pi / 2.0)
        assert spectral.values.shape == (3, len(spectral.energies))
        assert np.allclose(spectral.sum_rule, 1.0, atol=1e-12)

    def test_each_row_is_the_spectral_function_at_its_own_k(self, solved):
        axis = np.linspace(-30.0, 30.0, 200)
        along = solved.get_spectral_function(energies=axis, eta=0.2, path="GX")
        for row, k in enumerate(along.kpoints):
            alone = solved.get_spectral_function(energies=axis, eta=0.2, kpoints=[k])
            assert np.allclose(along.values[row], alone.values[0], atol=1e-12)

    def test_a_path_and_a_list_of_points_exclude_each_other(self, solved):
        with pytest.raises(ValueError, match="either kpoints or a band path"):
            solved.get_spectral_function(path="GX", kpoints=[[0, 0, 0]])

    def test_the_reduction_needs_a_path(self, solved):
        with pytest.raises(ValueError, match="pass path="):
            solved.get_spectral_function(irreducible=True)

    def test_a_vertex_the_mesh_misses_warns_and_keeps_the_points_it_has(self):
        atoms = chain(size=(3, 1, 1))
        atoms.get_potential_energy()
        with pytest.warns(RuntimeWarning, match="does not pass through X"):
            spectral = atoms.calc.get_spectral_function(path="GX", eta=0.2)
        assert np.allclose(spectral.kpoints[:, 0], [0.0, 1 / 3])
        assert spectral.kpath.missing == ("X",)

    def test_the_reduction_reproduces_the_rows_of_a_symmetric_state(self, solved):
        axis = np.linspace(-30.0, 30.0, 200)
        full = solved.get_spectral_function(energies=axis, eta=0.2, path="GX")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reduced = solved.get_spectral_function(energies=axis, eta=0.2, path="GX",
                                                   irreducible=True)
        audit = reduced.audit
        # k = +-1/4 are one star: the audit resolves both and finds the same spectrum.
        assert audit["star_size"] == 2 and audit["symmetric"] and audit["residual"] < 1e-6
        assert np.allclose(reduced.values, full.values, atol=1e-8)
        assert full.audit is None

    def test_time_reversal_gives_the_same_spectrum_at_k_and_minus_k(self, solved):
        axis = np.linspace(-30.0, 30.0, 200)
        plus = solved.get_spectral_function(energies=axis, eta=0.2, kpoints=[[0.25, 0, 0]])
        minus = solved.get_spectral_function(energies=axis, eta=0.2, kpoints=[[-0.25, 0, 0]])
        assert np.allclose(plus.values, minus.values, atol=1e-9)

    def test_a_broken_symmetry_warns_and_says_where(self, solved, monkeypatch):
        """The audit's verdict reaches the user. The native audit finds this state symmetric,
        so the verdict is overwritten; the arithmetic of the residual is gated in Rust."""
        from mandacaru import _native

        real = _native.bloch_path_spectral_function

        def broken(*args, **kwargs):
            raw = real(*args, **kwargs)
            raw["audit"] = dict(raw["audit"], residual=1.2, symmetric=False)
            return raw

        monkeypatch.setattr(_native, "bloch_path_spectral_function", broken)
        with pytest.warns(RuntimeWarning, match="not symmetric under the point group"):
            reduced = solved.get_spectral_function(path="GX", irreducible=True)
        assert reduced.audit["residual"] == 1.2

    def test_the_square_lattice_state_is_symmetric_under_the_fourfold_axis(self):
        atoms = square()
        atoms.get_potential_energy()
        axis = np.linspace(-20.0, 20.0, 300)
        calc = atoms.calc
        x = calc.get_spectral_function(energies=axis, eta=0.3, kpoints=[[0.5, 0, 0]]).values
        y = calc.get_spectral_function(energies=axis, eta=0.3, kpoints=[[0, 0.5, 0]]).values
        assert np.abs(x - y).max() < 1e-6 * x.max()
        path = calc.get_spectral_function(energies=axis, eta=0.3, path="GXMG")
        assert path.kpath.labels == ("G", "X", "M", "G") and path.values.shape == (4, 300)
        assert np.allclose(path.values[0], path.values[3], atol=1e-12)
