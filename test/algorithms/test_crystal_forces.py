# -*- coding: utf-8 -*-
# file: test/algorithms/test_crystal_forces.py

# This code is part of Mandacaru.
# MIT License

"""Forces and stress of a periodic Kohn-Sham crystal.

Each is checked against a central difference of the **self-consistent** free
energy on the same grid -- the derivative the analytic gradient (and the
fixed-state strain) claims to be.  Each check solves several crystals at
tight tolerances, so those are marked ``slow`` -- but one spin check on a
coarse grid, a crystal that loses its moment against the restricted one:
run them with ``pytest -m slow test/algorithms/test_crystal_forces.py``.  The
cost of one gradient (a few seconds for two atoms) is bounded by the cheap
tests of :class:`TestTheCostOfAGradient`.
Spin-polarized crystals are checked on doublets: LiH with one electron
removed for the forces, the neutral Li2H for the stress.
"""

import numpy as np
import pytest
from ase.build import bulk, molecule

from mandacaru import Mandacaru
from mandacaru.algorithms.crystal_forces import (crystal_gradient,
                                                 crystal_stress)
from mandacaru.algorithms.periodic_dft import PeriodicKohnSham
from mandacaru.integrals import Grid
from mandacaru.integrals import exchange_correlation as xc_grid
from mandacaru.pseudopotentials.periodic_paw import build_crystal
from mandacaru.units import HARTREE_TO_EV, to_bohr

OPTIONS = {"size": "SZ", "filter_cutoff": 300}
SMEARING = {"method": "fermi-dirac", "width": 0.1}


def _lih(a=4.0):
    """Rocksalt LiH with H pushed off its site: every atom feels a force."""
    atoms = bulk("LiH", "rocksalt", a=a)
    atoms.positions[1] += [0.06, -0.03, 0.04]
    return atoms


def _doublet(a=4.0):
    """:func:`_lih` with an initial moment on H, solved with one electron
    removed (``charge=1``): the hole in the H 1s level is a spin doublet,
    one Bohr magneton per cell, in the cell whose restricted forces are
    checked above.  The cell's charge sits on the uniform background the
    reciprocal-space electrostatics implies, the same in every displaced and
    strained crystal, so the finite differences are of one functional."""
    atoms = _lih(a)
    atoms.set_initial_magnetic_moments([0.0, 1.0])
    return atoms


def _li2h():
    """Li2H in an orthorhombic cell at Gamma: three valence electrons, a
    neutral doublet (one Bohr magneton per cell) with Li's partial core."""
    from ase import Atoms
    atoms = Atoms("Li2H", positions=[[0.1, 0.0, 0.0], [0.0, 0.05, 3.9],
                                     [-0.05, 0.1, 1.8]],
                  cell=[3.0, 3.2, 6.0], pbc=True)
    atoms.set_initial_magnetic_moments([0.5, 0.5, 0.0])
    return atoms


def _solve(atoms, grid_spacing, kpts=None, functional="lda", grid=None,
           symmetry=False, charge=0):
    """The converged crystal: spin-polarized when ``atoms`` carry initial
    moments, as the calculator runs it; ``charge`` electrons removed."""
    crystal, context = build_crystal(atoms, grid_spacing, OPTIONS, kpts=kpts, grid=grid,
                                     symmetry=symmetry)
    constant = float(sum(d.one_center_energy for d in crystal.datasets)
                     + sum(xc_grid.core_correction_offset(d)
                           for d in crystal.datasets))
    solver = PeriodicKohnSham(
        crystal, context["n_electrons"] - charge, functional,
        smearing=SMEARING,
        constant=constant,
        magnetic_moments=atoms.get_initial_magnetic_moments())
    result = solver.run(tol=1e-11, density_tol=1e-9)
    assert result.converged
    return solver, result


def _strained(atoms, epsilon, shape):
    moved = atoms.copy()
    cell = atoms.cell.array @ (np.eye(3) + epsilon).T
    moved.set_cell(cell, scale_atoms=True)
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0,
                grid_spacing=np.linalg.norm(cell, axis=1) / np.asarray(shape),
                units="angstrom", cell=cell, periodic=True)
    return moved, grid


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan", "hse06",
                                        "pbe+rvv10", "r2scan+rvv10"])
def test_the_force_is_the_derivative_of_the_free_energy(functional):
    """Hellmann-Feynman + Pulay against a central difference of F (one
    component of each atom), on a fixed grid: 1e-5 eV/Angstrom.  Si on a
    2x2x2 mesh reached 8e-7 (LDA), 8e-7 (PBE) and 4e-6 (r2SCAN)."""
    atoms = _lih()
    grid_spacing = float(np.linalg.norm(atoms.cell[0])) / 10
    solver, _result = _solve(atoms, grid_spacing, functional=functional)
    gradient = crystal_gradient(solver)
    assert np.all(np.abs(gradient.hellmann_feynman) > 0.0)
    assert np.any(np.abs(gradient.pulay) > 1e-4)
    step = 1e-3                                                 # Angstrom
    for atom, d in ((1, 0), (0, 2)):
        values = []
        for sign in (+1.0, -1.0):
            moved = atoms.copy()
            moved.positions[atom, d] += sign * step
            values.append(_solve(moved, grid_spacing, functional=functional,
                                 grid=solver.crystal.grid)[1].free_energy)
        numeric = (values[0] - values[1]) / (2.0 * step) * HARTREE_TO_EV
        assert gradient.gradient[atom, d] == pytest.approx(numeric, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("functional, size", [("lda", (2, 2, 2)),
                                              ("hse06", (2, 1, 1))])
def test_a_k_mesh_force_is_the_derivative_of_the_free_energy(functional,
                                                            size):
    """Si on a k-mesh: the k-point phases of the Bloch sums and the
    time-reversed weights, which a Gamma-only check cannot see -- and, for
    the hybrid, the exchange between different k-points (Gamma and X are
    enough for that, and keep it inside the time budget)."""
    from ase.build import bulk as build
    atoms = build("Si", "diamond", a=5.43)
    atoms.positions[1] += [0.05, 0.02, -0.03]
    kpts = {"size": size, "gamma": True}
    grid_spacing = float(np.linalg.norm(atoms.cell[0])) / 10
    solver, _result = _solve(atoms, grid_spacing, kpts=kpts, functional=functional)
    gradient = crystal_gradient(solver)
    step = 1e-3
    values = []
    for sign in (+1.0, -1.0):
        moved = atoms.copy()
        moved.positions[1, 0] += sign * step
        values.append(_solve(moved, grid_spacing, kpts=kpts, functional=functional,
                             grid=solver.crystal.grid)[1].free_energy)
    numeric = (values[0] - values[1]) / (2.0 * step) * HARTREE_TO_EV
    assert gradient.gradient[1, 0] == pytest.approx(numeric, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "hse06", "pbe+rvv10",
                                         "r2scan+rvv10"])
def test_the_stress_is_the_strain_derivative_of_the_free_energy(functional):
    """The fixed-state strain derivative against self-consistent strained
    crystals (same node counts, pinned filter): a diagonal and a shear
    component to 1e-5 relative."""
    atoms = _lih()
    grid_spacing = float(np.linalg.norm(atoms.cell[0])) / 10
    solver, _result = _solve(atoms, grid_spacing, functional=functional)
    stress = crystal_stress(solver, atoms, OPTIONS, "paw-lcao", None)
    volume = abs(np.linalg.det(to_bohr(atoms.cell.array, "angstrom")))
    step = 1e-3
    for a, b in ((0, 0), (0, 1)):
        values = []
        for sign in (+1.0, -1.0):
            epsilon = np.zeros((3, 3))
            epsilon[a, b] += 0.5 * sign * step
            epsilon[b, a] += 0.5 * sign * step
            moved, grid = _strained(atoms, epsilon,
                                    solver.crystal.grid.shape)
            values.append(_solve(moved, grid_spacing, grid=grid,
                                 functional=functional)[1].free_energy)
        numeric = (values[0] - values[1]) / (2.0 * step) / volume
        assert stress[a, b] == pytest.approx(numeric, rel=1e-5, abs=1e-9)


class TestTheCalculator:
    @pytest.fixture(scope="class")
    def lih(self):
        atoms = _lih()
        atoms.calc = Mandacaru(method="dft", basis={"name": "PAW-LCAO",
                                                    **OPTIONS},
                               grid_spacing=float(np.linalg.norm(atoms.cell[0])) / 10,
                               smearing=SMEARING, trace=False)
        atoms.get_potential_energy()
        return atoms

    @pytest.mark.slow
    def test_forces_reach_ase_with_their_breakdown(self, lih):
        forces = lih.get_forces()
        assert forces.shape == (2, 3)
        result = lih.calc.force_result
        assert np.allclose(result.forces, forces)
        assert np.allclose(result.hellmann_feynman + result.pulay,
                           result.gradient)

    def test_the_free_energy_is_what_ase_relaxes(self, lih):
        """Forces and stress are derivatives of F, not of E(sigma -> 0)."""
        scf = lih.calc.result.scf
        assert lih.get_potential_energy(force_consistent=True) == \
            pytest.approx(scf.free_energy * HARTREE_TO_EV)
        assert lih.get_potential_energy() == pytest.approx(
            lih.calc.result.optimal_energy)

    def test_stress_reaches_ase_in_voigt_order(self, lih, monkeypatch):
        """The calculator's side of the stress: units (Ha/Bohr^3 to
        eV/Angstrom^3) and ASE's Voigt order, with the solver's tensor
        replaced by a known one -- the tensor itself is checked against
        strained crystals above (twelve rebuilds would not fit 180 s here).
        """
        from mandacaru.units import BOHR_TO_ANGSTROM
        tensor = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])
        monkeypatch.setattr(lih.calc.solver, "crystal_stress",
                            lambda strain=None: tensor * 1e-4)
        stress = lih.get_stress()
        scale = 1e-4 * HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3
        assert np.allclose(stress, np.array([1, 2, 3, 4, 5, 6]) * scale)
        assert np.allclose(lih.calc.get_stress(voigt=False), tensor * scale)


def _c3v_silicon(functional, h_divisor):
    """``(atoms, kpts, full, reduced)``: Si with one atom moved along [111]
    -- a C3v group, six operations, two invariant strains -- solved on the
    time-reversal mesh and on its irreducible wedge, one grid."""
    from mandacaru.algorithms.crystal_forces import (_operations,
                                                     invariant_strains)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1] += 0.04 * np.ones(3)
    kpts = {"size": (2, 2, 2), "gamma": True}
    grid_spacing = float(np.linalg.norm(atoms.cell[0])) / h_divisor
    full, _ = _solve(atoms, grid_spacing, kpts=kpts, functional=functional)
    reduced, _ = _solve(atoms, grid_spacing, kpts=kpts, grid=full.crystal.grid,
                        symmetry=True, functional=functional)
    assert len(_operations(reduced.crystal)) == 6
    assert len(invariant_strains(reduced.crystal)) == 2
    assert len(reduced.crystal.kpoints) < len(full.crystal.kpoints)
    return atoms, kpts, full, reduced


@pytest.mark.slow
def test_the_irreducible_k_points_give_the_full_mesh_result():
    """Forces and stress from the irreducible k-points, symmetrized, equal
    the time-reversal mesh's: the forces are nonzero and the stress is not
    hydrostatic."""
    atoms, kpts, full, reduced = _c3v_silicon("lda", 10)
    f_full = crystal_gradient(full).forces
    f_reduced = crystal_gradient(reduced).forces
    assert np.max(np.abs(f_full)) > 0.1
    assert f_reduced == pytest.approx(f_full, abs=1e-6)
    s_full = crystal_stress(full, atoms, OPTIONS, "paw-lcao", kpts)
    s_reduced = crystal_stress(reduced, atoms, OPTIONS, "paw-lcao", kpts)
    assert abs(s_full[0, 1]) > 1e-6                  # not hydrostatic
    assert s_reduced == pytest.approx(s_full, abs=1e-8)


@pytest.mark.slow
class TestTheIrreducibleKPointsOfAHybrid:
    """The same for HSE06, whose exchange takes the full mesh's states as
    images of the wedge's.  The two SCFs are shared, so forces and stress
    each fit the time budget.  An odd grid: an even grid's Nyquist plane
    breaks the symmetry the directly solved mesh sees and the images do not
    (8^3: stress 4e-8 apart); 7^3 is too coarse to tell them apart."""

    @pytest.fixture(scope="class")
    def silicon(self):
        return _c3v_silicon("hse06", 9)

    def test_the_forces(self, silicon):
        _atoms, _kpts, full, reduced = silicon
        f_full = crystal_gradient(full).forces
        assert np.max(np.abs(f_full)) > 0.1
        assert crystal_gradient(reduced).forces == pytest.approx(f_full,
                                                                 abs=1e-6)

    def test_the_stress(self, silicon):
        atoms, kpts, full, reduced = silicon
        s_full = crystal_stress(full, atoms, OPTIONS, "paw-lcao", kpts)
        s_reduced = crystal_stress(reduced, atoms, OPTIONS, "paw-lcao", kpts)
        assert abs(s_full[0, 1]) > 1e-6                  # not hydrostatic
        assert s_reduced == pytest.approx(s_full, abs=1e-8)


# --------------------------------------------------------------------------- #
# Spin-polarized crystals.
# --------------------------------------------------------------------------- #

def _doublet_gradient(atoms, grid_spacing, solver, atom, d, functional, kpts=None,
                      step=1e-3):
    """Central difference (eV/Angstrom) of the doublet's free energy on
    ``solver``'s grid; every displaced crystal keeps its moment."""
    values = []
    for sign in (+1.0, -1.0):
        moved = atoms.copy()
        moved.positions[atom, d] += sign * step
        _s, result = _solve(moved, grid_spacing, kpts=kpts, functional=functional,
                            grid=solver.crystal.grid, charge=1)
        assert result.magnetic_moment == pytest.approx(1.0, abs=1e-3)
        values.append(result.free_energy)
    return (values[0] - values[1]) / (2.0 * step) * HARTREE_TO_EV


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan", "hse06",
                                        "pbe+rvv10", "r2scan+rvv10"])
def test_a_spin_crystal_force_is_the_derivative_of_the_free_energy(
        functional):
    """The LiH+ doublet at Gamma: each channel's Pulay term at its own
    potential (and exchange), the core under the mean of the channels'
    potentials -- where the empty minority channel meets Li's partial core,
    only at the nodes the functional does not clip.  1e-5 eV/Angstrom;
    measured 3e-7 to 4e-7 for every functional."""
    atoms = _doublet()
    grid_spacing = float(np.linalg.norm(atoms.cell[0])) / 10
    solver, result = _solve(atoms, grid_spacing, functional=functional, charge=1)
    assert solver.n_spins == 2
    assert result.magnetic_moment == pytest.approx(1.0, abs=1e-3)
    gradient = crystal_gradient(solver)
    assert np.any(np.abs(gradient.pulay) > 1e-4)
    for atom, d in ((1, 0), (0, 2)):
        numeric = _doublet_gradient(atoms, grid_spacing, solver, atom, d, functional)
        assert gradient.gradient[atom, d] == pytest.approx(numeric, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "hse06"])
def test_a_spin_k_mesh_force_is_the_derivative_of_the_free_energy(
        functional):
    """The doublet on a 2x1x1 mesh: the k-point phases in both channels
    and, for the hybrid, each channel's exchange between Gamma and X.  An
    expanded cell (a = 6 Angstrom) keeps the moment on the mesh, where the
    hole's band at a = 4 is too wide for it."""
    atoms = _doublet(a=6.0)
    kpts = {"size": (2, 1, 1), "gamma": True}
    grid_spacing = float(np.linalg.norm(atoms.cell[0])) / 14
    solver, result = _solve(atoms, grid_spacing, kpts=kpts, functional=functional,
                            charge=1)
    assert len(solver.crystal.kpoints) == 2
    assert result.magnetic_moment == pytest.approx(1.0, abs=1e-3)
    gradient = crystal_gradient(solver)
    for atom, d in ((1, 0), (0, 2)):
        numeric = _doublet_gradient(atoms, grid_spacing, solver, atom, d, functional,
                                    kpts)
        assert gradient.gradient[atom, d] == pytest.approx(numeric, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "hse06"])
def test_a_spin_crystal_stress_is_the_strain_derivative(functional):
    """Both channels' carried states against self-consistent strained
    doublets: a diagonal and a shear component to 1e-5 relative.

    The neutral Li2H rather than the charged LiH+ of the force tests: a
    charged cell's energy is strongly curved in the strain, so the 1e-3
    finite difference itself carries a 1e-4 relative error there (the
    analytic stress matches its step -> 0 limit, HISTORY 2026-10-04)."""
    atoms = _li2h()
    grid_spacing = 0.33
    solver, _result = _solve(atoms, grid_spacing, functional=functional)
    stress = crystal_stress(solver, atoms, OPTIONS, "paw-lcao", None)
    volume = abs(np.linalg.det(to_bohr(atoms.cell.array, "angstrom")))
    step = 1e-3
    for a, b in ((0, 0), (0, 1)):
        values = []
        for sign in (+1.0, -1.0):
            epsilon = np.zeros((3, 3))
            epsilon[a, b] += 0.5 * sign * step
            epsilon[b, a] += 0.5 * sign * step
            moved, grid = _strained(atoms, epsilon,
                                    solver.crystal.grid.shape)
            _s, result = _solve(moved, grid_spacing, grid=grid, functional=functional)
            assert result.magnetic_moment == pytest.approx(1.0, abs=1e-2)
            values.append(result.free_energy)
        numeric = (values[0] - values[1]) / (2.0 * step) / volume
        assert stress[a, b] == pytest.approx(numeric, rel=1e-5, abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "hse06"])
def test_a_spin_crystal_that_loses_its_moment_has_the_restricted_derivatives(
        functional):
    """LiH from small moments converges to no moment: two channels of one
    electron per state, each at its own potential (and -a K), give the
    restricted force (one channel of two, -a/2 K), and the strained state of
    both channels the restricted strained energy.  The hybrid runs by
    default (about 25 s): it covers the semilocal terms too."""
    from mandacaru.algorithms.crystal_forces import (carried_energy,
                                                     kpoint_state)
    atoms = _lih()
    grid_spacing = float(np.linalg.norm(atoms.cell[0])) / 6
    restricted, plain = _solve(atoms, grid_spacing, functional=functional)
    polarized = atoms.copy()
    polarized.set_initial_magnetic_moments([0.2, 0.2])
    spin, result = _solve(polarized, grid_spacing, functional=functional,
                          grid=restricted.crystal.grid)
    assert spin.n_spins == 2
    assert abs(result.magnetic_moment) < 1e-6
    assert result.free_energy == pytest.approx(plain.free_energy, abs=1e-8)
    expected = crystal_gradient(restricted)
    gradient = crystal_gradient(spin)
    assert np.max(np.abs(expected.gradient)) > 0.1
    assert gradient.gradient == pytest.approx(expected.gradient, abs=1e-6)
    assert gradient.pulay == pytest.approx(expected.pulay, abs=1e-6)
    strain = np.eye(3) + 1e-3 * np.array([[1.0, 0.5, 0.0], [0.5, 0.0, 0.0],
                                          [0.0, 0.0, -0.5]])
    energies = [carried_energy(solver, kpoint_state(solver), solver_atoms,
                               OPTIONS, "paw-lcao", None, strain)
                for solver, solver_atoms in ((restricted, atoms),
                                             (spin, polarized))]
    assert energies[1] == pytest.approx(energies[0], abs=1e-8)


@pytest.mark.slow
def test_spin_crystal_forces_reach_ase():
    """A spin-polarized crystal's forces are not refused: triplet O2 in a
    periodic box through the calculator, its breakdown kept; the two atoms'
    forces are equal and opposite up to the grid's egg-box."""
    atoms = molecule("O2")                    # ASE's moments: 1 per atom
    atoms.set_cell([4.0, 4.0, 4.5])
    atoms.center()
    atoms.pbc = True
    atoms.positions[1] += [0.05, -0.03, 0.02]
    atoms.calc = Mandacaru(method="dft", basis={"name": "PAW-LCAO",
                                                **OPTIONS},
                           grid_spacing=4.0 / 14, smearing=SMEARING, trace=False)
    forces = atoms.get_forces()
    assert atoms.calc.get_number_of_spins() == 2
    result = atoms.calc.force_result
    assert result.details["n_spins"] == 2
    assert np.allclose(result.forces, forces)
    assert np.allclose(result.hellmann_feynman + result.pulay,
                       result.gradient)
    assert np.max(np.abs(forces)) > 1.0
    assert forces[0] == pytest.approx(-forces[1], abs=0.1)


def test_the_crystal_derivatives_runs_single_threaded_blas():
    """BLAS on one thread beside the OpenMP kernels
    (:func:`~mandacaru.integrals._backend.single_threaded_blas`)."""
    for function in (crystal_gradient, crystal_stress):
        assert getattr(function, "__wrapped__", None) is not None


# --------------------------------------------------------------------------- #
# The cost of a crystal gradient (cheap: no finite-difference energies).
# --------------------------------------------------------------------------- #

def _count_bloch_calls(monkeypatch):
    """``{kernel: calls}`` of the C Bloch kernels, counted from now on."""
    from collections import Counter

    from mandacaru.integrals import _backend

    calls = Counter()
    for name in ("bloch_shell", "bloch_project"):
        original = getattr(_backend, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(_backend, name, counted)
    return calls


class TestTheCostOfAGradient:
    """Redundancy that once cost minutes stays out: the lattice sums are
    analytic, the SCF's matrices are reused, and the Bloch sums of a sphere
    are evaluated once for all k-points, not once per block of them."""

    MESHES = (None, {"size": (2, 1, 1), "gamma": True})

    @pytest.fixture(scope="class")
    def solved(self):
        atoms = _lih()
        grid_spacing = float(np.linalg.norm(atoms.cell[0])) / 10
        return {i: _solve(atoms, grid_spacing, kpts=kpts)[0]
                for i, kpts in enumerate(self.MESHES)}

    def test_the_sums_are_the_difference_of_the_sums(self, solved):
        """The analytic ion-pair and ion-compensation derivatives against the
        Richardson difference of the sums they replaced (``DEFAULT_DELTA``)."""
        from mandacaru.algorithms.crystal_forces import (DEFAULT_DELTA,
                                                         _sum_terms)

        solver = solved[0]
        c = solver.crystal
        _rho, q, _tau = solver.output_density
        analytic = _sum_terms(solver, {"density": solver.output_density})

        def central(function, atom, d, h):
            values = []
            for sign in (+1.0, -1.0):
                moved = [np.array(x, dtype=float) for x in c.centers]
                moved[atom][d] += sign * h
                values.append(function(moved))
            return values

        def value(atom, d, h):
            plus, minus = central(c.short_range_ion_compensation, atom, d, h)
            total = float(np.real(sum(q[ch] * (plus[ch] - minus[ch])
                                      for ch in c.channels))) / (2.0 * h)
            plus, minus = central(c.ion_constants, atom, d, h)
            return total + (plus - minus) / (2.0 * h)

        for atom, d in ((0, 1), (1, 2)):
            numeric = (4.0 * value(atom, d, 0.5 * DEFAULT_DELTA)
                       - value(atom, d, DEFAULT_DELTA)) / 3.0
            assert analytic[atom, d] == pytest.approx(numeric, abs=1e-8)
        assert analytic.sum(axis=0) == pytest.approx(0.0, abs=1e-12)

    def test_the_scf_matrices_are_not_rebuilt(self, solved, monkeypatch):
        solver = solved[1]
        reference = crystal_gradient(solver).gradient

        def refused(*args, **kwargs):
            raise AssertionError("the SCF's k-point matrices were rebuilt")

        monkeypatch.setattr(solver.crystal, "kpoint_matrices", refused)
        assert crystal_gradient(solver).gradient == pytest.approx(
            reference, abs=1e-12)

    def test_the_gradient_does_not_depend_on_the_blocking(self, solved,
                                                          monkeypatch):
        """One k-point per block of grid sums and per wide chunk of spheres."""
        from mandacaru.pseudopotentials import periodic_paw

        solver = solved[1]
        reference = crystal_gradient(solver)
        monkeypatch.setattr(periodic_paw, "KPOINT_BLOCK_BYTES", 1)
        blocked = crystal_gradient(solver)
        assert blocked.hellmann_feynman == pytest.approx(
            reference.hellmann_feynman, abs=1e-10)
        assert blocked.pulay == pytest.approx(reference.pulay, abs=1e-10)

    def test_the_bloch_kernels_run_once_per_sphere_not_per_k_block(
            self, solved, monkeypatch):
        """LiH's gradient took 170 ``bloch_shell`` calls (136 of them its own
        extended basis, 34 rebuilding the SCF's matrices) at any mesh; the
        counts below hold at 1 and at 2 k-points, and a block of k-points
        no longer repeats the spheres."""
        counts = []
        for i in sorted(solved):
            calls = _count_bloch_calls(monkeypatch)
            crystal_gradient(solved[i])
            counts.append(dict(calls))
        assert counts[0] == counts[1]
        assert counts[0]["bloch_shell"] <= 70
        assert counts[0]["bloch_project"] <= 80

    def test_a_projection_does_not_change_when_k_points_join_the_call(
            self, solved):
        crystal = solved[1].crystal
        both = crystal.projections(crystal.kpoints)
        alone = np.stack([crystal.projections(k[None, :])[0]
                          for k in crystal.kpoints])
        assert np.allclose(both, alone, atol=1e-13, rtol=0)
