# -*- coding: utf-8 -*-
# file: test/test_charges.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Partial charges and magnetic moments of a converged state.

Every partial charge is a *convention*, so most of what can be pinned here is
what must hold for **all** of them: the charges sum to the system's charge,
the weights partition unity, a symmetric molecule comes out neutral, and the
moments sum to ``N_alpha - N_beta``.  The one thing that is not a convention
is the total magnetic moment, which takes no ``method`` at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms.charges import (PARTITION_METHODS, WEIGHT_METHODS,
                                          atomic_weights, free_atom_density,
                                          partition_crystal,
                                          reference_subshells)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def h2(distance=0.74, cell=8.0, **kwargs):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, distance]],
                  cell=[cell] * 3, **kwargs)
    atoms.center()
    return atoms


def lih(distance=1.6, cell=10.0):
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, distance]],
                  cell=[cell] * 3)
    atoms.center()
    return atoms


def h3(cell=8.0, **kwargs):
    atoms = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.9], [0, 0.95, 0.45]],
                  cell=[cell] * 3, **kwargs)
    atoms.center()
    return atoms


def solved(atoms, **options):
    options.setdefault("basis", "HAO")
    options.setdefault("h", 0.25)
    atoms.calc = Mandacaru(method="adapt-vqe", trace=False, profile=False,
                           **options)
    atoms.get_potential_energy()
    return atoms.calc


@pytest.fixture(scope="module")
def water_free_atoms():
    """Warm the reference cache once; solving an atom is the slow part."""
    free_atom_density(1)
    free_atom_density(3)


# --------------------------------------------------------------------------- #

class TestTheReferenceAtoms:
    def test_a_free_atom_holds_its_electrons(self):
        for Z in (1, 3, 8):
            r, rho = free_atom_density(Z)
            electrons = np.trapezoid(4.0 * np.pi * r ** 2 * rho, r)
            assert electrons == pytest.approx(Z, abs=2e-3)

    def test_the_valence_reference_holds_the_valence_electrons(self):
        # Li is 1s2 2s1, so its valence reference must hold exactly one.
        r, rho = free_atom_density(3, reference_subshells(3, valence=True))
        electrons = np.trapezoid(4.0 * np.pi * r ** 2 * rho, r)
        assert electrons == pytest.approx(1.0, abs=2e-3)

    def test_it_is_solved_once(self):
        first = free_atom_density(1)
        assert free_atom_density(1) is first


class TestTheWeights:
    """Whatever the convention, the weights are a partition of unity."""

    @pytest.mark.parametrize("method", WEIGHT_METHODS)
    def test_they_sum_to_one_everywhere(self, method):
        calc = solved(lih(), h=0.30)
        integrals = calc.solver._gradient_context["integrals"]
        grid = integrals.grid
        positions = np.array([R for _Z, R in integrals._potentials.nuclei])
        weights = atomic_weights(method, grid, positions, numbers=[3, 1])
        assert weights.shape == (2,) + tuple(grid.shape)
        assert np.allclose(weights.sum(axis=0), 1.0)
        assert np.all(weights >= -1e-12)

    def test_an_unknown_partition_is_refused(self):
        with pytest.raises(ValueError, match="unknown weight partition"):
            atomic_weights("mulliken", None, [[0.0, 0.0, 0.0]])
        # Bader's basins are traced, not a weight on the nodes.
        with pytest.raises(ValueError, match="bader_populations"):
            atomic_weights("bader", None, [[0.0, 0.0, 0.0]])


class TestChargesSumToTheSystemCharge:
    """Not a convention: the total is fixed however the atoms split it."""

    @pytest.mark.parametrize("method", PARTITION_METHODS)
    def test_a_neutral_molecule(self, method):
        calc = solved(lih(), h=0.25)
        assert calc.atomic_partition(method).total_charge == \
            pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize("method", PARTITION_METHODS)
    def test_a_cation(self, method):
        calc = solved(h3(), charge=1)
        partition = calc.atomic_partition(method)
        assert partition.total_charge == pytest.approx(1.0, abs=1e-8)
        assert partition.grid_electrons == pytest.approx(2.0, abs=1e-8)


class TestSymmetry:
    """A homonuclear dimer is neutral on both atoms, by symmetry."""

    @pytest.mark.parametrize("method", ["hirshfeld", "voronoi"])
    @pytest.mark.parametrize("h", [0.25, 0.20])
    def test_the_smooth_and_the_geometric_partition_are_exact(self, method, h):
        charges = solved(h2(), h=h).atomic_partition(method).charges
        assert np.allclose(charges, 0.0, atol=1e-9)

    def test_the_voronoi_tie_is_split_not_given_away(self):
        """The dividing plane lands on a node layer at some spacings.

        Handing that layer to the lower atom index charges H2 by 0.16 e at
        h = 0.25 and by nothing at h = 0.15 -- a pure discretization artifact,
        which splitting the tie removes.
        """
        for h in (0.30, 0.25, 0.20, 0.15):
            charges = solved(h2(), h=h).atomic_partition("voronoi").charges
            assert np.allclose(charges, 0.0, atol=1e-9), h

    @pytest.mark.parametrize("h", [0.30, 0.25])
    def test_bader_is_neutral_too(self, h):
        """The continuous ascent with boundary refinement: the bond midplane
        is a whole plane of points exactly on the zero-flux surface, split
        evenly.  The on-grid ascent this replaced charged H2 by 0.1 e at
        h = 0.25 (and by 0.08 e before the refinement)."""
        charges = solved(h2(), h=h).atomic_partition("bader").charges
        assert np.allclose(charges, 0.0, atol=1e-6)


class TestPolarity:
    def test_lithium_hydride_is_lithium_positive(self):
        """Li(+)H(-): every convention has to agree on the sign."""
        for method in PARTITION_METHODS:
            charges = solved(lih(), h=0.20).atomic_partition(method).charges
            assert charges[0] > 0.1, method            # Li
            assert charges[1] < -0.1, method           # H

    def test_hirshfeld_is_the_smallest_of_the_three(self):
        """Its reference is a neutral atom, so it famously understates."""
        calc = solved(lih(), h=0.20)
        smooth = abs(calc.atomic_partition("hirshfeld").charges[0])
        basin = abs(calc.atomic_partition("bader").charges[0])
        assert smooth < basin


class TestMagneticMoments:
    def test_a_closed_shell_has_none(self):
        partition = solved(h2()).atomic_partition("hirshfeld")
        assert np.allclose(partition.magnetic_moments, 0.0, atol=1e-6)
        assert partition.total_magnetic_moment == pytest.approx(0.0, abs=1e-9)

    def test_a_triplet_carries_two(self):
        calc = solved(h2(distance=1.2, magmoms=[1, 1]))
        partition = calc.atomic_partition("hirshfeld")
        assert calc.get_total_magnetic_moment() == pytest.approx(2.0, abs=1e-9)
        assert np.allclose(partition.magnetic_moments, 1.0, atol=1e-6)

    def test_a_doublet_carries_one(self):
        calc = solved(h3(magmoms=[1, 0, 0]))
        assert calc.get_total_magnetic_moment() == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("method", PARTITION_METHODS)
    def test_the_moments_sum_to_the_total_whatever_the_partition(self, method):
        calc = solved(h3(magmoms=[1, 0, 0]))
        partition = calc.atomic_partition(method)
        assert np.sum(partition.magnetic_moments) == \
            pytest.approx(calc.get_total_magnetic_moment(), abs=1e-6)

    def test_the_total_takes_no_partition(self):
        """It is an integral over all space, so there is nothing to choose."""
        import inspect
        signature = inspect.signature(Mandacaru.get_total_magnetic_moment)
        assert "method" not in signature.parameters

    def test_it_is_measured_not_repeated_back(self):
        """N_alpha - N_beta of the *state*, not the magmoms it was given."""
        calc = solved(h3(magmoms=[1, 0, 0]))
        gamma, _ = calc._state_rdms(calc.solver, two_body=False)
        n_alpha, n_beta = calc.solver.num_particles
        assert calc.get_total_magnetic_moment() == \
            pytest.approx(n_alpha - n_beta, abs=1e-9)


class TestElectronsAreAllAccountedFor:
    def test_a_frozen_core_is_refilled(self):
        """The core is not in the RDM but is in the molecule."""
        frozen = solved(lih(), h=0.20, active_space={"frozen": "auto"})
        assert frozen.atomic_partition("hirshfeld").grid_electrons == \
            pytest.approx(4.0, abs=1e-8)

    def test_freezing_barely_moves_the_charges(self):
        full = solved(lih(), h=0.20).atomic_partition("hirshfeld").charges
        frozen = solved(lih(), h=0.20, active_space={"frozen": "auto"})
        assert frozen.atomic_partition("hirshfeld").charges == \
            pytest.approx(full, abs=0.05)

    def test_the_paw_augmentation_is_added_per_atom(self):
        """The smooth grid density does not integrate to N; the spheres hold
        the rest, and they are on-site so the split is exact."""
        calc = solved(h2(), basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.20)
        partition = calc.atomic_partition("hirshfeld")
        assert np.all(partition.augmentation > 0.0)
        assert partition.grid_electrons == pytest.approx(2.0, abs=1e-8)
        assert any("augmentation" in note for note in partition.notes)

    def test_a_pseudopotential_run_is_measured_against_its_valence(self):
        calc = solved(lih(), basis={"name": "PAW-LCAO", "size": "DZP"}, h=0.20)
        partition = calc.atomic_partition("hirshfeld")
        # Li contributes one valence electron, H one.
        assert np.allclose(partition.reference_charges, [1.0, 1.0])
        assert partition.grid_electrons == pytest.approx(2.0, abs=1e-8)
        assert any("valence" in note for note in partition.notes)


class TestTheASESurface:
    def test_the_three_accessors_and_what_they_record(self):
        atoms = h3(magmoms=[1, 0, 0])
        calc = solved(atoms)
        charges = atoms.get_charges()
        moments = atoms.get_magnetic_moments()
        total = atoms.get_magnetic_moment()
        assert charges.shape == (3,) and moments.shape == (3,)
        assert total == pytest.approx(calc.get_total_magnetic_moment())
        assert set(calc.results) >= {"charges", "magmoms", "magmom"}

    def test_the_method_is_selectable_on_the_getters(self):
        calc = solved(lih(), h=0.20)
        assert not np.allclose(calc.get_charges(method="hirshfeld"),
                               calc.get_charges(method="bader"))

    def test_an_unknown_method_is_refused(self):
        calc = solved(h2())
        with pytest.raises(ValueError, match="unknown partition"):
            calc.get_charges(method="mulliken")

    def test_direct_mode_has_no_density_to_partition(self):
        """A qubit operator carries no basis functions and no nuclei."""
        from mandacaru.core import PauliSum
        calc = Mandacaru(method="adapt-vqe", hamiltonian=PauliSum({"ZI": 1.0}),
                         pool="qubit", num_particles=(1, 1),
                         n_spatial_orbitals=1, trace=False)
        calc.run()
        with pytest.raises(NotImplementedError, match="real-space picture"):
            calc.get_charges()


class TestTheCoarseGrid:
    def test_bader_still_separates_two_atoms(self):
        """H2 at h = 0.30: the on-grid ascent merged the two basins and
        handed one atom no electrons; the continuous one, on an
        interpolation of the density, separates them."""
        calc = solved(h2(), h=0.30)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            partition = calc.atomic_partition("bader")
        assert not any("no basin" in str(w.message) for w in caught)
        assert np.allclose(partition.populations, 1.0, atol=1e-6)


class TestWhatGetsWrittenToAFile:
    """``population=`` is what puts the analysis in an extxyz file.

    ASE's extxyz writer turns the per-atom entries of ``calc.results`` into
    columns and the scalars into header keys, so the whole feature is: have
    them in ``results`` by the time the file is written.
    """

    def written(self, atoms):
        import io

        from ase.io import write
        buffer = io.StringIO()
        write(buffer, atoms, format="extxyz")
        return buffer.getvalue()

    def test_one_energy_call_is_enough(self):
        atoms = h3(magmoms=[1, 0, 0])
        calc = solved(atoms, h=0.30, population="hirshfeld")
        assert set(calc.results) >= {"charges", "magmoms", "magmom"}

    def test_the_columns_and_the_header(self):
        atoms = h3(magmoms=[1, 0, 0])
        solved(atoms, h=0.30, population="hirshfeld")
        text = self.written(atoms)
        header = text.splitlines()[1]
        # ASE names the per-atom charge column `charge`, singular.
        assert ":charge:R:1" in header and ":magmoms:R:1" in header
        assert "magmom=" in header and "energy=" in header

    def test_it_reads_back(self):
        import io

        from ase.io import read
        atoms = h3(magmoms=[1, 0, 0])
        calc = solved(atoms, h=0.30, population="hirshfeld")
        back = read(io.StringIO(self.written(atoms)), format="extxyz")
        assert back.get_charges() == pytest.approx(calc.results["charges"])
        assert back.get_magnetic_moments() == \
            pytest.approx(calc.results["magmoms"])
        assert back.calc.results["magmom"] == \
            pytest.approx(calc.results["magmom"])

    def test_off_by_default(self):
        """The partition costs a grid pass a relaxation has no use for."""
        atoms = h3(magmoms=[1, 0, 0])
        calc = solved(atoms, h=0.30)
        assert calc.population is None
        assert not {"charges", "magmoms", "magmom"} & set(calc.results)
        assert ":charge:" not in self.written(atoms)

    def test_the_analysis_is_declared_to_ase(self):
        for name in ("charges", "magmoms", "magmom"):
            assert name in Mandacaru.implemented_properties

    @pytest.mark.parametrize("spec,expected",
                             [(True, "hirshfeld"), ("bader", "bader"),
                              ("Voronoi", "voronoi"), (None, None),
                              (False, None)])
    def test_the_option_is_normalized(self, spec, expected):
        calc = Mandacaru(method="adapt-vqe", basis="HAO", population=spec)
        assert calc.population == expected

    def test_an_unknown_one_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown population analysis"):
            Mandacaru(method="adapt-vqe", basis="HAO", population="mulliken")

    def test_the_getters_follow_it(self):
        """A run configured for Bader must not report Hirshfeld."""
        atoms = lih()
        calc = solved(atoms, h=0.20, population="bader")
        assert calc.get_charges() == pytest.approx(
            calc.atomic_partition("bader").charges)
        # ... and an explicit method still wins.
        assert calc.get_charges(method="hirshfeld") == pytest.approx(
            calc.atomic_partition("hirshfeld").charges)


def test_the_summary_reports_every_atom():
    calc = solved(lih(), h=0.25)
    text = calc.atomic_partition("hirshfeld").summary()
    assert "hirshfeld" in text and "total" in text
    assert len(text.splitlines()) >= 4


class TestCrystals:
    """Partitions of a periodic Kohn-Sham density (``partition_crystal``)."""

    @staticmethod
    def _crystal(atoms, nodes):
        from mandacaru.pseudopotentials.periodic_paw import build_crystal
        from mandacaru.algorithms.periodic_dft import PeriodicKohnSham
        h = float(np.linalg.norm(np.asarray(atoms.cell)[0])) / nodes
        crystal, context = build_crystal(atoms, h, {"size": "SZ",
                                                    "filter": 400})
        solver = PeriodicKohnSham(crystal, context["n_electrons"], "lda")
        solver.run()
        return crystal, solver.density_matrices, atoms.get_atomic_numbers()

    def test_equivalent_atoms_are_neutral_in_silicon(self):
        """On a grid that keeps all 48 operations the two atoms are
        equivalent, so every partition gives zero.  Bader needs the frozen
        core: silicon's valence density peaks at the bond centers, and a
        valence-only ascent handed those basins to one atom (+-1 e)."""
        from ase.build import bulk
        from mandacaru.algorithms.charges import partition_crystal
        crystal, matrices, numbers = self._crystal(bulk("Si", "diamond",
                                                        a=5.43), 12)
        for method in PARTITION_METHODS:
            partition = partition_crystal(crystal, matrices, method=method,
                                          numbers=numbers)
            assert partition.grid_electrons == pytest.approx(8.0, abs=1e-8)
            # Bader's precision is its trajectories' (step, gradient): 2e-3.
            tolerance = 5e-3 if method == "bader" else 1e-3
            assert np.allclose(partition.charges, 0.0, atol=tolerance)

    def test_rocksalt_lih_is_ionic_in_the_expected_order(self):
        """Li gives its electron to H in all three, and the conventions rank
        as they do for molecules: Hirshfeld < Voronoi < Bader."""
        from ase.build import bulk
        from mandacaru.algorithms.charges import partition_crystal
        crystal, matrices, numbers = self._crystal(bulk("LiH", "rocksalt",
                                                        a=4.0), 12)
        charge = {method: partition_crystal(crystal, matrices, method=method,
                                            numbers=numbers).charges
                  for method in PARTITION_METHODS}
        for values in charge.values():
            assert values.sum() == pytest.approx(0.0, abs=1e-8)
            assert values[0] > 0.0 > values[1]
        assert 0.0 < charge["hirshfeld"][0] < charge["voronoi"][0] \
            < charge["bader"][0] <= 1.0 + 1e-3


def test_crystal_bader_is_converged_and_shift_invariant():
    """Rocksalt LiH (SZ, 4x4x4, 10 nodes): Li +0.843 against the continuous
    reference +0.847 (Vera, 2026-10-03; the on-grid ascent gave 0.75-0.81
    and moved 0.01 e with a shift of the atoms), and a rigid shift of the
    atoms by 0.3 of a grid step moves it by about 1e-3 e."""
    from ase.build import bulk
    from mandacaru.algorithms.charges import partition_crystal
    from mandacaru.algorithms.periodic_dft import PeriodicKohnSham
    from mandacaru.pseudopotentials.periodic_paw import build_crystal

    def lithium_charge(shift):
        atoms = bulk("LiH", "rocksalt", a=4.0)
        h = float(np.linalg.norm(atoms.cell[0])) / 10
        direction = np.array([1.0, 0.7, 0.4]) / np.linalg.norm([1.0, 0.7, 0.4])
        atoms.positions += shift * h * direction
        crystal, context = build_crystal(atoms, h, {"size": "SZ",
                                                    "filter": 400},
                                         kpts=(4, 4, 4))
        solver = PeriodicKohnSham(crystal, context["n_electrons"], "lda")
        solver.run()
        return partition_crystal(crystal, solver.density_matrices,
                                 method="bader",
                                 numbers=atoms.get_atomic_numbers()).charges[0]

    plain, shifted = lithium_charge(0.0), lithium_charge(0.3)
    assert plain == pytest.approx(0.847, abs=0.01)
    assert shifted == pytest.approx(plain, abs=3e-3)


def test_the_crystal_partition_runs_single_threaded_blas():
    """BLAS on one thread beside the OpenMP kernels
    (:func:`~mandacaru.integrals._backend.single_threaded_blas`)."""
    for function in (partition_crystal,):
        assert getattr(function, "__wrapped__", None) is not None
