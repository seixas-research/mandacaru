# -*- coding: utf-8 -*-
# file: test/algorithms/test_bloch.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The periodic methods, reached through ``Mandacaru`` like every other one.

``method="bloch-vqe"`` and ``method="bloch-adapt-vqe"`` solve the Born-von
Karman supercell and report the energy **per primitive cell**.  Covered here:
method resolution, the option checks the constructor can make and the geometry
checks it cannot, the regression pins recorded before the driver was moved
behind the entry point, the supercell identity the method rests on, the bands
and Fermi level, the dry run, and the refusal of forces.

Everything runs on a 1-D hydrogen chain at a coarse grid: the whole file is a
couple of seconds, and the suite has no budget for more.
"""

import functools
import pathlib

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import ADAPTVQEResult, Mandacaru, VQEResult
from mandacaru.optimizers import Optimizer

# Written out rather than left to the library default: a test that pins an
# energy should say what optimized it.  The dict spelling is the *examples*'
# convention; a test reusing one configuration across a file uses the object.
SLSQP = Optimizer(method="SLSQP", maxiter=1000, tol=1e-12)

#: The Gamma-centered two-cell mesh most tests here use.
GAMMA2 = {"size": (2, 1, 1), "gamma": True}

METHODS = ["bloch-vqe", "bloch-adapt-vqe"]
_RESULT_TYPES = {"bloch-vqe": VQEResult, "bloch-adapt-vqe": ADAPTVQEResult}
#: ADAPT has adaptive controls VQE does not take.
_EXTRA = {"bloch-vqe": {}, "bloch-adapt-vqe": dict(max_iterations=10,
                                                   gradient_tolerance=1e-3)}

# ---- the reference values ------------------------------------------------- #
# 1-D H chain, a = 2.0 A, HAO, h = 0.35, SLSQP(tol=1e-12), in eV per primitive
# cell.  Measured against the **periodic** engine: the electrostatics are
# lattice sums (periodic Coulomb kernel, Ewald external potential and ion-ion,
# Madelung constant, periodic basis images), so these are not the values the
# earlier box-cluster build produced and must not be compared with them.
#
# a = 2.0 A rather than 1.0: at 1.0 A the cell is smaller than an H 1s and the
# periodic image sum does not converge, which the engine warns about.
#
# Re-recorded when the periodic grid became **commensurate** with the primitive
# cell (``Grid(commensurate=)``, driven by ``_BlochMixin._grid_commensurate``).
# The old values were measured on grids of 5.5 and 5.75 nodes per primitive
# cell, where a primitive translation is not a whole number of grid steps and
# Bloch's theorem does not hold on the grid; they were 0.64 to 0.98 eV lower.
# Two signs that the new numbers are the right ones: the two solvers now agree
# to 1e-12 eV at four cells, where the broken-symmetry Hamiltonian had them
# 0.34 eV apart, and a degenerate band pair of a square lattice closes from
# 0.087 Ha to 5e-16.  The pre-commensurate values are kept below so a future
# shift can be told apart from this one.
PIN_ENERGY = {
    ("bloch-vqe", (2, 1, 1)): -12.422242898937355,
    ("bloch-vqe", (4, 1, 1)): -11.262946653871158,
    ("bloch-adapt-vqe", (2, 1, 1)): -12.422242898937347,
    ("bloch-adapt-vqe", (4, 1, 1)): -11.262946653870129,
}
#: What the same runs gave on the incommensurate grid, for provenance only.
PIN_BEFORE_COMMENSURATE_GRID = {
    ("bloch-vqe", (2, 1, 1)): -13.28657394944987,
    ("bloch-vqe", (4, 1, 1)): -12.241550533427803,
    ("bloch-adapt-vqe", (2, 1, 1)): -13.286573949452588,
    ("bloch-adapt-vqe", (4, 1, 1)): -11.90056922993599,
}
#: Reproduced bit-for-bit, so 1e-8 eV leaves orders of margin over
#: summation-order noise while still catching a changed cell, grid or kernel.
PIN_TOLERANCE = 1e-8


def chain(**cell_kwargs):
    """1-D hydrogen chain, one atom per primitive cell."""
    return Atoms("H", positions=[[0.0, 0.0, 0.0]],
                 cell=cell_kwargs.pop(
                     "cell", [[2.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                              [0.0, 0.0, 10.0]]),
                 pbc=cell_kwargs.pop("pbc", [True, False, False]))


def periodic(method="bloch-vqe", mesh=GAMMA2, **kwargs):
    """A chain with the periodic calculator attached (coarse, fast settings)."""
    atoms = chain()
    kwargs.setdefault("basis", "HAO")
    kwargs.setdefault("mapping", "jordan_wigner")
    kwargs.setdefault("h", 0.35)
    kwargs.setdefault("trace", False)
    atoms.calc = Mandacaru(method=method, kpts=mesh, **kwargs)
    return atoms


# --------------------------------------------------------------------------- #
# The methods are reached like every other one.
# --------------------------------------------------------------------------- #
class TestMethodResolution:
    @pytest.mark.parametrize("method", METHODS)
    def test_the_method_is_stable_and_resolves(self, method):
        from mandacaru.algorithms.calculator import STABLE_METHODS

        assert method in STABLE_METHODS
        assert periodic(method).calc.method == method

    @pytest.mark.parametrize("spelling", [
        "bloch-adapt-vqe", "bloch_adapt_vqe", "BlochADAPTVQE", "Bloch-Adapt-VQE",
    ])
    def test_spelling_is_insensitive(self, spelling):
        assert periodic(spelling).calc.method == "bloch-adapt-vqe"

    def test_the_default_method_is_unchanged(self):
        from mandacaru.algorithms.calculator import DEFAULT_METHOD

        assert DEFAULT_METHOD == "adapt-vqe"

    @pytest.mark.parametrize("method", METHODS)
    def test_the_solver_is_the_molecular_one_over_a_supercell(self, method):
        """The periodic driver *is* the molecular solver, plus the BvK layer.

        Asserted on the MRO by name rather than with ``isinstance``: importing
        a solver class anywhere under ``test/`` is exactly what
        ``test_single_entry_point`` forbids.
        """
        solver = periodic(method).calc.solver
        ancestors = [klass.__name__ for klass in type(solver).__mro__]
        assert "_BlochMixin" in ancestors
        assert ("ADAPTVQE" if "adapt" in method else "VQE") in ancestors
        assert solver.n_supercells == 2


# --------------------------------------------------------------------------- #
# Option checks: everything the constructor can know without a geometry.
# --------------------------------------------------------------------------- #
class TestConstructorRefusesBadOptions:
    def test_kpts_is_required(self):
        with pytest.raises(ValueError, match="kpts"):
            Mandacaru(method="bloch-vqe", basis="HAO")

    def test_a_shifted_mesh_is_refused(self):
        """A bare triple is an ASE Monkhorst-Pack mesh, which for an even size
        does not contain Gamma -- and the Born-von Karman equivalence is an
        identity only for a Gamma-centered mesh."""
        with pytest.raises(ValueError, match="Gamma"):
            Mandacaru(method="bloch-vqe", kpts=(2, 1, 1), basis="HAO")

    @pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
    def test_a_mesh_that_already_contains_gamma_is_accepted(self, mesh):
        """An odd Monkhorst-Pack mesh is Gamma-centered on its own."""
        assert Mandacaru(method="bloch-vqe", kpts=mesh,
                         basis="HAO").kpts == mesh

    def test_a_malformed_kpts_is_refused(self):
        with pytest.raises(ValueError):
            Mandacaru(method="bloch-vqe", kpts=(2, 1), basis="HAO")

    def test_direct_mode_is_refused(self):
        with pytest.raises(ValueError, match="direct mode"):
            Mandacaru(method="bloch-vqe", kpts=GAMMA2, hamiltonian=object())


class TestUnsupportedCombinations:
    def test_a_pseudopotential_basis_is_refused(self):
        """It would otherwise solve an isolated cluster, silently.

        The pseudopotential path builds its own integrals and never sees the
        periodic flag, so it cannot be left to the guard inside
        ``PeriodicIntegrals`` -- that code is never reached from here.
        """
        atoms = chain()
        atoms.calc = Mandacaru(method="bloch-vqe", kpts=GAMMA2,
                               basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.35,
                               trace=False)
        with pytest.raises(NotImplementedError, match="periodic lattice sum"):
            atoms.get_potential_energy()


class TestMolecularKPointsAreUnchanged:
    """A non-periodic method must keep exactly its old ``kpts`` behavior."""

    def test_a_denser_mesh_still_raises_at_run(self):
        from mandacaru.core import PauliSum

        calc = Mandacaru(method="adapt-vqe", kpts=(2, 1, 1),
                         hamiltonian=PauliSum({"ZZ": 1.0}), num_particles=(1, 1),
                         n_spatial_orbitals=1, trace=False)
        with pytest.raises(NotImplementedError, match="Monkhorst-Pack"):
            calc.run()

    def test_a_bare_triple_is_still_accepted(self):
        """The Gamma-centering rule is the periodic methods' own."""
        assert Mandacaru(method="vqe", kpts=(2, 2, 1), basis="HAO").kpts == (2, 2, 1)


# --------------------------------------------------------------------------- #
# Geometry checks: everything that needs the Atoms, and nothing before it.
# --------------------------------------------------------------------------- #
class TestGeometryIsCheckedBeforeAnyWork:
    def test_a_non_periodic_geometry_is_refused(self):
        atoms = chain(pbc=[False, False, False])
        atoms.calc = Mandacaru(method="bloch-vqe", kpts=GAMMA2, basis="HAO",
                               h=0.35, trace=False)
        with pytest.raises(ValueError, match="periodic boundary conditions"):
            atoms.get_potential_energy()

    def test_a_non_periodic_geometry_is_refused_in_the_dry_run_too(self):
        calc = Mandacaru(method="bloch-vqe", kpts=GAMMA2, basis="HAO", h=0.35,
                         trace=False)
        with pytest.raises(ValueError, match="periodic boundary conditions"):
            calc.dry_run(chain(pbc=[False, False, False]))

    def test_a_one_dimensional_chain_works(self):
        atoms = periodic("bloch-vqe", optimizer=SLSQP)
        assert tuple(atoms.pbc) == (True, False, False)
        assert np.isfinite(atoms.get_potential_energy())

    def test_k_points_along_an_open_direction_name_that_direction(self):
        atoms = chain()
        atoms.calc = Mandacaru(method="bloch-vqe",
                               kpts={"size": (1, 2, 1), "gamma": True},
                               basis="HAO", h=0.35, trace=False)
        with pytest.raises(ValueError, match="direction b"):
            atoms.get_potential_energy()

    def test_a_degenerate_cell_is_refused(self):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                      cell=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
                      pbc=[True, True, False])
        atoms.calc = Mandacaru(method="bloch-vqe",
                               kpts={"size": (2, 2, 1), "gamma": True},
                               basis="HAO", h=0.35, trace=False)
        with pytest.raises(ValueError, match="degenerate"):
            atoms.get_potential_energy()

    def test_nothing_expensive_runs_before_the_geometry_is_judged(self,
                                                                  monkeypatch):
        """The same guarantee ``test_dry_run`` makes: no Hamiltonian is built.

        A geometry check that ran after the integrals would still be correct
        and would still cost the user the whole build.
        """
        import mandacaru.algorithms._hamiltonian_from_atoms as builder

        monkeypatch.setattr(
            builder, "build_basis_hamiltonian",
            lambda *a, **k: pytest.fail("the Hamiltonian builder was reached"))
        atoms = chain(pbc=[False, False, False])
        atoms.calc = Mandacaru(method="bloch-vqe", kpts=GAMMA2, basis="HAO",
                               h=0.35, trace=False)
        with pytest.raises(ValueError, match="periodic boundary conditions"):
            atoms.get_potential_energy()


# --------------------------------------------------------------------------- #
# The energy: the pins, and the identity the whole method rests on.
# --------------------------------------------------------------------------- #
class TestTotalEnergy:
    @pytest.mark.parametrize("method, size", sorted(PIN_ENERGY))
    def test_it_reproduces_the_recorded_energy(self, method, size):
        atoms = periodic(method, mesh={"size": size, "gamma": True},
                         optimizer=SLSQP, **_EXTRA[method])
        assert atoms.get_potential_energy() == pytest.approx(
            PIN_ENERGY[(method, size)], abs=PIN_TOLERANCE)

    @pytest.mark.parametrize("method", METHODS)
    def test_the_result_lands_on_calc_result(self, method):
        atoms = periodic(method, optimizer=SLSQP, **_EXTRA[method])
        atoms.get_potential_energy()
        assert isinstance(atoms.calc.result, _RESULT_TYPES[method])

    @pytest.mark.parametrize("method", METHODS)
    def test_it_is_the_supercell_energy_divided_by_the_cells(self, method):
        """What ASE reports is the solver's own total, per primitive cell.

        This used to compare against the molecular method on
        ``atoms.repeat(...)``.  That comparison is no longer meaningful: the
        periodic driver builds lattice-summed electrostatics while the
        molecular one builds an isolated cluster, so the two are different
        physical quantities and agreeing would be the bug.  What remains worth
        pinning is the division itself.
        """
        atoms = periodic(method, optimizer=SLSQP, **_EXTRA[method])
        per_cell = atoms.get_potential_energy()
        assert atoms.calc.solver.n_supercells == 2
        assert per_cell == pytest.approx(
            atoms.calc.result.optimal_energy / 2, abs=PIN_TOLERANCE)

    @pytest.mark.parametrize("method", METHODS)
    def test_the_hamiltonian_is_built_with_lattice_sums(self, method):
        """A periodic method must not be solving a molecule in a box."""
        atoms = periodic(method, optimizer=SLSQP, **_EXTRA[method])
        atoms.get_potential_energy()
        integrals = atoms.calc.solver._gradient_context["integrals"]
        assert type(integrals).__name__ == "PeriodicIntegrals"
        assert integrals.constant_energy != 0.0          # the Madelung term


# --------------------------------------------------------------------------- #
# Bands and the Fermi level (single-particle: see the module docstring).
# --------------------------------------------------------------------------- #
class TestBands:
    """Bands are now the quasiparticle peaks of ``A(E, k)``."""

    def test_they_have_one_entry_per_k_point_and_orbital(self):
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        assert atoms.calc.bands().shape == (2, 1)

    def test_they_are_the_dominant_pole_of_the_spectral_function(self):
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        spectral = atoms.calc.get_spectral_function()
        bands = atoms.calc.bands(spectral=spectral)
        for ik, per_orbital in enumerate(spectral.orbital_poles):
            for nu, (poles, weights) in enumerate(per_orbital):
                assert bands[ik, nu] == pytest.approx(poles[np.argmax(weights)])

    def test_the_weights_say_how_band_like_each_peak_is(self):
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        weights = atoms.calc.band_weights()
        assert weights.shape == (2, 1)
        assert np.all((weights > 0.0) & (weights <= 1.0))

    def test_they_need_the_correlated_state(self):
        """Interacting bands cannot exist before the state does."""
        calc = Mandacaru(method="bloch-vqe", kpts=GAMMA2, basis="HAO", h=0.35,
                         trace=False)
        with pytest.raises(ValueError, match="ground state"):
            calc.bands()

    def test_a_continuous_path_is_refused(self):
        """A finite supercell has Bloch operators only on its own mesh."""
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        with pytest.raises(NotImplementedError, match="continuous band path"):
            atoms.calc.band_structure("GX", npoints=201)


class TestSpectralFunction:
    def test_the_sum_rule_is_exact(self):
        """``{c, c+} = 1``, so removal and addition weights must add to 1.

        This is an identity, not a convergence property: it catches a wrong
        orbital rotation, a missing branch or a bad spin convention at once.
        """
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        assert atoms.calc.get_spectral_function().sum_rule < 1e-9

    def test_it_integrates_to_one_state_per_spin_and_orbital(self):
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        spectral = atoms.calc.get_spectral_function(
            energies=np.linspace(-120.0, 120.0, 4000), eta=0.2)
        area = np.trapezoid(spectral.weights, spectral.energies, axis=1)
        assert area == pytest.approx(np.full(area.shape, 2.0), abs=2e-2)

    def test_the_weight_splits_at_the_fermi_momentum(self):
        """Half filling puts Gamma below mu and X above it."""
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        spectral = atoms.calc.get_spectral_function()
        removal = [poles[weights.argmax()] < 0.0
                   for poles, weights in
                   (spectral.orbital_poles[ik][0] for ik in range(2))]
        assert removal == [True, False]

    @pytest.mark.parametrize("mapping", ["parity", "parity_reduced",
                                         "bravyi_kitaev"])
    def test_every_mapping_has_the_same_poles_and_weights(self, mapping):
        """Ladder transitions and charged-sector Hamiltonians share a map."""
        baseline = periodic(optimizer=SLSQP)
        alternative = periodic(mapping=mapping, optimizer=SLSQP)
        baseline.get_potential_energy()
        alternative.get_potential_energy()
        reference = baseline.calc.get_spectral_function(points=100)
        spectral = alternative.calc.get_spectral_function(
            energies=reference.energies)
        assert spectral.sum_rule < 1e-9
        assert np.allclose(spectral.weights, reference.weights, atol=1e-8)
        for (poles, weights), (expected_poles, expected_weights) in zip(
                spectral.poles, reference.poles):
            assert np.allclose(poles, expected_poles, atol=1e-8)
            assert np.allclose(weights, expected_weights, atol=1e-8)
        if mapping == "parity_reduced":
            assert alternative.calc.n_qubits == baseline.calc.n_qubits - 2

    def test_adapt_uses_the_charged_reduced_sectors(self):
        """The ADAPT ansatz can feed a parity-reduced Lehmann calculation."""
        atoms = periodic("bloch-adapt-vqe", mapping="parity_reduced",
                         optimizer=SLSQP, **_EXTRA["bloch-adapt-vqe"])
        atoms.get_potential_energy()
        spectral = atoms.calc.get_spectral_function(points=100)
        assert spectral.sum_rule < 1e-9
        assert all(np.isfinite(poles).all() and np.isfinite(weights).all()
                   for poles, weights in spectral.poles)

    def test_it_writes_a_csv(self, tmp_path):
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        path = atoms.calc.get_spectral_function().write(tmp_path / "a.csv")
        assert pathlib.Path(path).read_text().startswith("k_index,")


class TestFermiLevel:
    def test_it_is_the_interacting_chemical_potential(self):
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        spectral = atoms.calc.get_spectral_function()
        assert (atoms.calc.get_fermi_level(spectral=spectral)
                == pytest.approx(spectral.chemical_potential))

    def test_it_lies_between_the_branches(self):
        atoms = periodic(optimizer=SLSQP)
        atoms.get_potential_energy()
        spectral = atoms.calc.get_spectral_function()
        poles = np.concatenate([p for p, _w in spectral.poles])
        weights = np.concatenate([w for _p, w in spectral.poles])
        keep = weights > 1e-6
        assert poles[keep].min() < spectral.chemical_potential < poles[keep].max()


class TestTwoDimensional:
    def test_a_slab_is_supported(self):
        """Full 3-D periodicity is not required."""
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]],
                      pbc=[True, True, False])
        atoms.calc = Mandacaru(method="bloch-vqe",
                               kpts={"size": (2, 2, 1), "gamma": True},
                               basis="HAO", h=0.40, trace=False,
                               optimizer=SLSQP)
        assert atoms.calc.dimension == 2
        assert atoms.calc.n_supercells == 4
        atoms.get_potential_energy()
        # Four commensurate k-points, one orbital per primitive cell.
        assert atoms.calc.bands().shape == (4, 1)


# --------------------------------------------------------------------------- #
# Bloch's theorem holds on the grid only if a primitive translation is a whole
# number of grid steps.
# --------------------------------------------------------------------------- #
def square_supercell_grid(h, size=(2, 2, 1), a=2.6, commensurate=True):
    """Just the grid of that supercell -- no sampling, no integrals.

    The node-count assertions need nothing else, and building the integrals for
    them costs seconds of work no assertion reads.
    """
    from mandacaru.integrals import Grid

    primitive = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, 9.0]],
                      pbc=[True, True, False])
    cell = np.asarray(primitive.repeat(size).cell[:], dtype=float)
    return Grid(center=0.5 * cell.sum(axis=0), box_size=0.0, h=h,
                units="angstrom", cell=cell, periodic=True,
                commensurate=size if commensurate else None)


@functools.lru_cache(maxsize=8)
def square_supercell_integrals(h, size=(2, 2, 1), a=2.6, commensurate=True):
    """Periodic integrals of an ``n x n`` square lattice of H atoms.

    Cached: the tests below want the same few grids, and building one is
    seconds of integral work that does not depend on which assertion reads it.
    """
    from mandacaru.core import minimal_hao_basis
    from mandacaru.core.periodic import PeriodicIntegrals
    from mandacaru.integrals import Grid

    primitive = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, 9.0]],
                      pbc=[True, True, False])
    supercell = primitive.repeat(size)
    cell = np.asarray(supercell.cell[:], dtype=float)
    nuclei = [(1.0, position) for position in supercell.positions]
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0, h=h,
                units="angstrom", cell=cell, periodic=True,
                commensurate=size if commensurate else None)
    integrals = PeriodicIntegrals(
        nuclei, minimal_hao_basis(nuclei), grid, cell,
        n_electrons=len(nuclei), units="angstrom",
        softening=0.5 * min(grid.dx, grid.dy, grid.dz))
    return grid, integrals


class TestTheGridIsCommensurateWithThePrimitiveCell:
    """The supercell's sites are copies of one atom, and must stay copies.

    A Born-von Karman supercell of an ``(n1, n2, n3)`` mesh is ``n_i``
    primitive cells long, so a primitive translation is ``N_i / n_i`` grid
    steps.  When that is not a whole number the grid is not invariant under the
    translation Bloch's theorem rests on: the repeated atoms sample it at
    different offsets and stop being equivalent.  Everything downstream -- the
    bands, the spectral function, the irreducible zone -- assumes they are.
    """

    @pytest.mark.parametrize("h, divisible", [(0.35, False), (0.325, True)])
    def test_the_node_count_is_rounded_up_to_a_multiple(self, h, divisible):
        bare = square_supercell_grid(h, commensurate=False)
        fixed = square_supercell_grid(h, commensurate=True)
        assert (bare.shape[0] % 2 == 0) is divisible
        assert fixed.shape[0] % 2 == 0
        # Rounded *up*: the requested spacing is an upper bound, so the grid
        # may only get finer.
        assert fixed.shape[0] >= bare.shape[0]

    def test_equivalent_sites_get_equal_on_site_energies(self):
        """The diagonal of ``h`` over four copies of one atom.

        Measured at ``h = 0.35`` (15 nodes, 7.5 steps per cell) the four spread
        by 0.10 Ha; on the commensurate grid they agree to round-off.
        """
        _grid, broken = square_supercell_integrals(0.35, commensurate=False)
        _grid, fixed = square_supercell_integrals(0.35, commensurate=True)

        def spread(integrals):
            one = integrals.one_body()
            one = one[0] if isinstance(one, tuple) else one
            diagonal = np.real(np.diag(one))
            return float(diagonal.max() - diagonal.min())

        assert spread(broken) > 1e-2
        assert spread(fixed) < 1e-10

    def test_the_degenerate_band_pair_of_a_square_lattice_closes(self):
        r"""``X`` and ``X'`` of a 2x2 mesh are related by C4 and must coincide.

        This is the single-particle problem ``h c = eps S c``, before any SCF,
        so it isolates the grid from the physics of the filling.  Measured at
        ``h = 0.35``: the commensurate grid gives an **exact** degeneracy
        (6e-16 Ha) and the incommensurate one gives 1.8e-3 Ha -- a near
        degeneracy, which is not the same thing and is what a plot would show
        as two bands where there is one.
        """
        import scipy.linalg as sla

        def levels(commensurate):
            _grid, integrals = square_supercell_integrals(
                0.35, commensurate=commensurate)
            one = integrals.one_body()
            one = one[0] if isinstance(one, tuple) else one
            return sla.eigh(np.real(one), np.real(integrals.overlap()),
                            eigvals_only=True)

        broken = levels(False)
        fixed = levels(True)
        # A degenerate pair, wherever it sits in the ordering, closed to
        # round-off on the commensurate grid and absent on the other.
        assert np.min(np.abs(np.diff(fixed))) < 1e-12
        assert np.min(np.abs(np.diff(broken))) > 1e-6

    def test_the_driver_asks_for_it(self):
        calc = Mandacaru(method="bloch-vqe",
                         kpts={"size": (3, 2, 1), "gamma": True},
                         basis="HAO", h=0.4, trace=False, optimizer=SLSQP)
        assert calc.solver._grid_commensurate() == (3, 2, 1)

    def test_a_molecular_driver_asks_for_nothing(self):
        """The requirement is Bloch's, so no molecular path may inherit it."""
        calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.4, trace=False,
                         optimizer=SLSQP)
        assert calc.solver._grid_commensurate() is None


# --------------------------------------------------------------------------- #
# 2-D and 3-D: the mesh is not a line, so a band plot needs a path.
# --------------------------------------------------------------------------- #
def square_lattice(size=(2, 2, 1), a=2.6, h=0.35, method="bloch-adapt-vqe"):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, 9.0]],
                  pbc=[True, True, False])
    atoms.calc = Mandacaru(method=method, kpts={"size": size, "gamma": True},
                           basis="HAO", h=h, trace=False, optimizer=SLSQP)
    return atoms


@pytest.fixture(scope="module")
def solved_square():
    """One solved half-filled square lattice, shared by every test below.

    Eight qubits is a few seconds, and six tests need the same state; re-running
    it per test is the difference between a cheap file and a slow one.  Nothing
    here mutates the calculator.
    """
    atoms = square_lattice()
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def square_spectrum(solved_square):
    """The mesh spectral function of that state, computed once."""
    return solved_square.calc.get_spectral_function(eta=0.25, points=120)


class TestBandPathsNeedNoRun:
    """The path and the symmetry are properties of the lattice and the mesh.

    Neither needs the correlated state, so both are available before anything
    is solved -- which is what lets a caller check that a mesh carries the
    points the plot will name.
    """

    def test_the_path_selects_mesh_points(self):
        calc = square_lattice().calc
        kpath = calc.band_path("GXMG")
        assert kpath.complete
        assert len(kpath) == 4
        assert kpath.labels == ["G", "X", "M", "G"]

    def test_a_mesh_without_x_says_so(self):
        calc = square_lattice(size=(3, 3, 1)).calc
        with pytest.warns(RuntimeWarning, match="not on this"):
            kpath = calc.band_path("GXMG")
        assert set(kpath.missing) == {"X", "M"}

    def test_the_default_path_is_the_lattice_s_own(self):
        assert square_lattice().calc.band_path().path == "MGXM"

    def test_the_space_group_is_reported(self):
        assert square_lattice().calc.symmetry().international == "P4/mmm"

    def test_the_irreducible_zone_reduces_the_mesh(self):
        zone = square_lattice().calc.irreducible_zone()
        assert len(zone.points) == 3
        assert int(zone.weights.sum()) == 4

    def test_the_ase_k_point_interface(self):
        calc = square_lattice().calc
        assert calc.get_bz_k_points().shape == (4, 3)
        assert calc.get_ibz_k_points().shape == (3, 3)
        assert calc.get_k_point_weights().sum() == pytest.approx(1.0)

    def test_the_wedge_does_not_shrink_the_register(self):
        """The saving is in A(E, k); the supercell is fixed by the full mesh."""
        calc = square_lattice().calc
        assert calc.n_supercells == 4
        assert len(calc.irreducible_zone().points) == 3
        assert calc.dry_run(square_lattice()).n_qubits == 8


def hexagonal_lattice(size=(2, 2, 1), a=2.6, h=0.45):
    """One H per primitive cell on a 2-D hexagonal lattice, in a tall box."""
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=[[a, 0.0, 0.0],
                        [-a / 2, a * np.sqrt(3) / 2, 0.0],
                        [0.0, 0.0, 9.0]],
                  pbc=[True, True, False])
    atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                           kpts={"size": size, "gamma": True},
                           basis="HAO", h=h, trace=False, optimizer=SLSQP)
    return atoms


@pytest.fixture(scope="module")
def solved_hexagonal():
    """One solved hexagonal lattice, shared: 8 qubits, ~1.5 s."""
    atoms = hexagonal_lattice()
    atoms.get_potential_energy()
    return atoms


class TestANonOrthogonalLatticeRunsEndToEnd:
    """A skewed cell must reach an energy through ``Mandacaru``, not just
    through :class:`PeriodicIntegrals`.

    ``test/core/test_periodic.py::TestNonOrthogonalCells`` pins the integrals
    layer.  This pins the whole path -- driver, grid, basis images, Ewald,
    spectral kinetic operator, symmetry and band path -- because the
    commensurability check that used to refuse every hexagonal, monoclinic and
    triclinic cell sat *between* the two, and a passing integrals test said
    nothing about whether a user could run one.

    Every other periodic test in this file uses a square lattice or a 1-D
    chain, which is exactly the family a transposed cell comparison cannot
    affect.
    """

    def test_it_reaches_an_energy(self, solved_hexagonal):
        """It used to raise ValueError('not commensurate') before any run."""
        energy = solved_hexagonal.get_potential_energy()
        assert np.isfinite(energy)
        assert solved_hexagonal.calc.n_qubits == 8

    def test_the_space_group_is_hexagonal(self, solved_hexagonal):
        symmetry = solved_hexagonal.calc.symmetry()
        assert symmetry.international == "P6/mmm"
        assert symmetry.n_operations == 24

    def test_the_wedge_is_smaller_than_the_square_one(self, solved_hexagonal):
        """24 operations against 16 reduce a 4-point mesh further."""
        zone = solved_hexagonal.calc.irreducible_zone()
        assert len(zone.points) == 2
        assert int(zone.weights.sum()) == 4
        assert zone.reduction == pytest.approx(2.0)

    def test_k_is_not_on_a_two_by_two_mesh(self, solved_hexagonal):
        """``K = (1/3, 1/3, 0)`` needs thirds; a 2x2x1 mesh has halves."""
        kpath = solved_hexagonal.calc.band_path("GMKG", warn=False)
        assert kpath.missing == ["K"]
        assert not kpath.complete

    def test_the_path_uses_the_hexagonal_metric(self, solved_hexagonal):
        r"""The Gamma-M-K triangle, in inverse Angstrom.

        ``b_1 . b_2 = -|b|^2 / 2`` here, so every one of these lengths comes
        out wrong if the path is measured on fractional coordinates.  The three
        points are not collinear -- the right angle is at M.
        """
        a = 2.6
        kpath = solved_hexagonal.calc.band_path("GMKG", warn=False)
        ticks = dict(zip(kpath.labels, kpath.label_distances))
        assert ticks["M"] == pytest.approx(2 * np.pi / (a * np.sqrt(3)), rel=1e-9)
        assert ticks["K"] - ticks["M"] == pytest.approx(2 * np.pi / (3 * a),
                                                        rel=1e-9)
        assert kpath.label_distances[-1] - ticks["K"] == pytest.approx(
            4 * np.pi / (3 * a), rel=1e-9)

    def test_the_sum_rule_holds_on_a_skewed_lattice(self, solved_hexagonal):
        """The anticommutator identity does not care about the cell shape."""
        spectral = solved_hexagonal.calc.get_spectral_function(
            eta=0.4, points=80, path="GMKG")
        assert spectral.sum_rule < 1e-9
        assert len(spectral.kpoints) == len(spectral.kpath)

    def test_the_grid_spans_the_skewed_cell_exactly(self, solved_hexagonal):
        """``shape @ step`` must reproduce the cell -- transposed.

        This is the shape of the bug itself: ``step`` holds its vectors as
        columns and ``cell`` as rows, so the two agree only after a transpose.
        """
        integrals = solved_hexagonal.calc.solver._gradient_context["integrals"]
        grid = integrals.grid
        spanned = np.asarray(grid.step) @ np.diag(grid.shape)
        assert spanned == pytest.approx(integrals.cell.T, abs=1e-10)
        assert not np.allclose(spanned, integrals.cell)   # genuinely skewed


class TestTheIrreducibleZoneIsAudited:
    """``irreducible=True`` verifies its assumption instead of trusting it.

    ``A(E, k)`` inherits the symmetry of the **state**, not of the lattice.  A
    partially filled degenerate manifold -- a metallic mesh -- makes the
    reference pick one member and break the point group, and then the wedge is
    not enough.  One extra Lehmann evaluation is what tells the two apart.
    """

    def test_a_symmetry_broken_state_is_caught_and_named(self, solved_square,
                                                         square_spectrum):
        """Half-filled square lattice: C4-equivalent k-points disagree."""
        with pytest.warns(RuntimeWarning, match="the state does not"):
            reduced = solved_square.calc.get_spectral_function(
                energies=square_spectrum.energies, eta=0.25, irreducible=True)
        assert reduced.irreducible.symmetry_residual > 1e-2

    def test_a_closed_shell_mesh_reduces_exactly(self):
        """Six k-points on a chain: Gamma plus the filled +-1/6 pair.

        The state carries the full symmetry, so the wedge reproduces the whole
        mesh -- not approximately, but to round-off.
        """
        atoms = chain()
        # max_iterations=2 rather than a converged run: what is under test is
        # whether the *state respects the symmetry*, and it does so from the
        # closed-shell reference onward.  A full growth is 20 s, four operators
        # 6.7 s and two ~4 s, and the residual is the same 1e-15 in every case.
        atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                               kpts={"size": (6, 1, 1), "gamma": True},
                               basis="HAO", h=0.35, trace=False,
                               max_iterations=2, optimizer=SLSQP)
        atoms.get_potential_energy()
        full = atoms.calc.get_spectral_function(eta=0.3, points=80)
        reduced = atoms.calc.get_spectral_function(
            energies=full.energies, eta=0.3, irreducible=True)
        assert reduced.irreducible.symmetry_residual < 1e-10
        assert reduced.weights == pytest.approx(full.weights, abs=1e-12)
        assert reduced.irreducible.reduction > 1.0


class TestTheSpectralFunctionOnAPath:
    def test_the_rows_are_the_full_mesh_rows(self, solved_square,
                                             square_spectrum):
        along = solved_square.calc.get_spectral_function(
            energies=square_spectrum.energies, eta=0.25, path="GXMG")
        assert along.weights == pytest.approx(
            square_spectrum.weights[along.kpath.indices], abs=0.0)

    def test_the_axis_is_the_path_length(self, solved_square, square_spectrum):
        along = solved_square.calc.get_spectral_function(
            energies=square_spectrum.energies, eta=0.25, path="GXMG")
        assert along.distances == pytest.approx(along.kpath.distances)
        assert np.all(np.diff(along.distances) > 0)

    def test_a_mesh_spectrum_falls_back_to_the_index(self, square_spectrum):
        """Without a path there is no length to measure along."""
        assert square_spectrum.kpath is None
        assert square_spectrum.distances == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_the_csv_carries_all_three_components(self, tmp_path,
                                                 solved_square):
        """Writing only ``k[0]`` loses k_y for every 2-D and 3-D lattice."""
        along = solved_square.calc.get_spectral_function(eta=0.25, points=20,
                                                        path="GXMG")
        target = along.write(tmp_path / "square.csv")
        header, *rows = pathlib.Path(target).read_text().splitlines()
        assert header.split(",")[:6] == ["k_index", "k1", "k2", "k3",
                                         "k_distance_inv_ang", "label"]
        # The X point of this path has k2 = 1/2, which a k[0]-only writer drops.
        assert any(row.split(",")[2] == "0.5000000000" for row in rows)
        assert any(row.split(",")[5] == "X" for row in rows)


class TestDryRun:
    @pytest.mark.parametrize("cells, qubits", [(1, 2), (2, 4), (4, 8)])
    def test_the_qubit_count_scales_with_the_cells(self, cells, qubits):
        calc = Mandacaru(method="bloch-vqe",
                         kpts={"size": (cells, 1, 1), "gamma": True},
                         basis="HAO", h=0.35, trace=False)
        assert calc.dry_run(chain()).n_qubits == qubits

    def test_the_label_names_the_supercell(self):
        calc = Mandacaru(method="bloch-vqe", kpts=GAMMA2, basis="HAO", h=0.35,
                         trace=False)
        assert "2x1x1 Born-von Karman supercell" in calc.dry_run(chain()).basis


@pytest.fixture(scope="module")
def forced_chain():
    """One periodic force run, shared and deliberately cheap.

    The assertions below are about the *driver* -- that it advertises forces
    and returns a gradient of the reported energy -- not about the value, which
    ``test/algorithms/test_periodic_forces.py`` pins against a finite
    difference.  So this uses a small transverse cell and a coarse grid: two
    full force calculations on the shared ``periodic()`` geometry cost 29 s
    between them, and this costs a fraction of that.
    """
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=np.diag([2.0, 6.0, 6.0]), pbc=[True, False, False])
    atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                           kpts={"size": (2, 1, 1), "gamma": True},
                           basis="HAO", h=0.45, trace=False, optimizer=SLSQP)
    atoms.get_forces()
    return atoms


class TestForcesAreImplemented:
    """This class used to be ``TestForcesAreRefused``.

    It asserted that ``get_forces()`` raised, on the grounds that there was no
    Born-von Karman analogue of the Hellmann-Feynman and Pulay terms.  There is
    one now (:mod:`mandacaru.algorithms.periodic_forces`), validated against a
    central difference of the same fixed-state energy, so the refusal is gone
    rather than merely bypassed -- the ``supports_forces`` gate it was built on
    is deleted, not set to ``True``.  The gradient itself is pinned in
    ``test/algorithms/test_periodic_forces.py``; what is kept here is that the
    periodic *driver* delivers it through ASE.
    """

    @pytest.mark.parametrize("method", METHODS)
    def test_the_driver_routes_to_the_periodic_gradient(self, method):
        calc = periodic(method).calc
        assert type(calc.solver).periodic_hamiltonian is True
        assert not hasattr(type(calc.solver), "supports_forces")

    def test_get_forces_returns_a_finite_gradient(self, forced_chain):
        forces = forced_chain.get_forces()
        assert forces.shape == (len(forced_chain), 3)
        assert np.all(np.isfinite(forces))

    def test_the_force_is_the_gradient_of_the_reported_energy(self,
                                                              forced_chain):
        """The rebuilt energy must match the solver's, or the force is refused.

        That check lives in ``Mandacaru._forces``; here it is exercised through
        the public path, where a mismatch would raise instead of returning a
        plausible array.
        """
        result = forced_chain.calc.force_result
        reported = float(forced_chain.calc.solver._from_energy_units(
            forced_chain.calc.solver.result.optimal_energy, "Ha"))
        assert result.details["energy_hartree"] == pytest.approx(reported,
                                                                 abs=1e-6)


class TestTheRunLog:
    def test_the_mesh_and_supercell_are_stated_once(self, tmp_path):
        """One owner per fact: the ``[ELECTRONS]`` k-points line carries both."""
        log = tmp_path / "bloch.txt"
        atoms = periodic("bloch-adapt-vqe", optimizer=SLSQP, txt=str(log),
                         **_EXTRA["bloch-adapt-vqe"])
        atoms.get_potential_energy()
        text = log.read_text()
        carrying = [line for line in text.splitlines()
                    if "supercell" in line.lower()]
        assert len(carrying) == 1, carrying
        assert "k-points:" in carrying[0]
        assert "2x1x1 Monkhorst-Pack" in carrying[0]
