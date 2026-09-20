# -*- coding: utf-8 -*-
# file: test/test_paw_exact_local.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The range-separated local potential: exactness, invariance, forces.

PAW keeps only the long-range half of its local channel on the grid -- the
potential of a Gaussian ion, which the grid resolves -- and integrates the
short-range remainder on atom-centered spherical quadratures
(:mod:`mandacaru.pseudopotentials.local_split`).  The two halves add up to the
potential the all-grid form used, so the tests here are about *exactness*: the
sphere integral is translation invariant to machine precision, its derivative
sums to zero under a rigid translation, the split reproduces the all-grid
energy in the fine-grid limit, and the force is still the derivative of the
energy the solver reported.

``PAWIntegrals.exact_local_potential`` is the switch that recovers the old
all-grid behavior, which is what the comparisons here toggle.  Every
non-PAW basis is untouched: the hook on
:class:`~mandacaru.core.hamiltonian.MolecularIntegrals` returns ``None``.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms._hamiltonian_from_atoms import (build_basis_hamiltonian,
                                                          grid_from_cell)
from mandacaru.pseudopotentials import local_split as ls
from mandacaru.pseudopotentials.paw import PAWIntegrals
from mandacaru.units import HARTREE_TO_EV

SZ = {"name": "PAW", "size": "SZ"}


@pytest.fixture
def all_grid():
    """Run the body with the split off, then restore the default."""
    PAWIntegrals.exact_local_potential = False
    try:
        yield
    finally:
        PAWIntegrals.exact_local_potential = True


def h2(cell=8.0, distance=0.74):
    """Off the grid nodes on purpose: a symmetric molecule centered on a node
    is the easiest case, not the representative one."""
    center = np.array([cell / 2 + 0.013, cell / 2 - 0.021, cell / 2 + 0.007])
    half = np.array([0.0, 0.0, distance / 2])
    return Atoms("H2", positions=[center - half, center + half],
                 cell=[cell] * 3)


def water(cell=8.0):
    """C2v water with the two O-H bonds along +y and +z.

    Swapping y and z maps the molecule onto itself and exchanges the hydrogens,
    and the cubic grid shares that symmetry, so exact forces must obey it -- and
    the oxygen sphere (4.3 Bohr at h = 0.25 Angstrom) holds *both* hydrogens,
    which is what makes this the interesting case for a sphere quadrature: most
    of what it integrates is not centered on its own atom.
    """
    c = cell / 2
    return Atoms("OHH", positions=[[c, c, c], [c, c + 1.0, c], [c, c, c + 1.0]],
                 cell=[cell] * 3)


def context_of(atoms, h, basis=SZ, grid=None):
    grid = grid_from_cell(atoms, h) if grid is None else grid
    return build_basis_hamiltonian(atoms, basis, grid, h, 0, None)[4]


def integrals_of(atoms, h, basis=SZ, grid=None):
    return context_of(atoms, h, basis, grid)["integrals"]


def rhf_total(integrals, n_electrons=None):
    """RHF total energy: electronic + ion-ion + the frozen one-center constant."""
    if n_electrons is None:
        n_electrons = int(round(sum(d.valence_charge
                                    for d in integrals.pseudopotentials)))
    rhf = integrals.hartree_fock(n_electrons)
    return (rhf.electronic_energy + integrals.nuclear_repulsion
            + integrals.constant_energy)


# --------------------------------------------------------------------------- #
# The split itself.
# --------------------------------------------------------------------------- #

class TestRangeSeparation:
    def test_the_two_halves_add_up_to_the_local_potential(self):
        """``v^lr + v^sr == v`` by construction, at every radius.

        The short-range part is evaluated as a *difference* of the dataset's own
        ``local_potential`` and the Gaussian-ion potential, never from the
        asymptotic ``-Z erfc(r)/r``, so the identity is exact whatever the
        shipped table's tail actually does (it sits ~5e-7 Hartree off -Z/r).
        """
        from mandacaru.pseudopotentials.paw import get_paw

        radii = np.linspace(0.0, 12.0, 4001)
        for symbol in ("H", "O"):
            dataset = get_paw(symbol)
            sigma = 0.66
            total = (ls.long_range_potential(dataset.valence_charge, sigma, radii)
                     + ls.short_range_potential(dataset, sigma, radii))
            assert np.abs(total - dataset.local_potential(radii)).max() < 1e-12

    def test_the_long_range_part_is_finite_at_the_origin(self):
        """``-Z erf(r/sqrt(2) sigma)/r -> -Z sqrt(2/pi)/sigma``."""
        sigma = 0.5
        value = ls.long_range_potential(3.0, sigma, np.array([0.0]))[0]
        assert value == pytest.approx(-3.0 * np.sqrt(2.0 / np.pi) / sigma)
        # and it joins the tail smoothly: -Z/r far out.
        far = ls.long_range_potential(3.0, sigma, np.array([10.0]))[0]
        assert far == pytest.approx(-0.3, rel=1e-12)

    def test_the_short_range_part_dies_inside_its_sphere(self):
        """What the quadrature truncates, in its two separable pieces.

        Beyond the dataset's local cutoff the potential is the ionic tail, so
        ``v^sr = -Z erfc(r/sqrt2 sigma)/r`` plus whatever the *table* itself
        deviates from ``-Z/r``.  The first is the design's own quantity and is
        ``erfc(6.5/sqrt2) = 7.3e-11`` at the sphere's edge; the second is a
        property of the shipped, decimated library (measured at up to 4.4e-6
        Hartree near 2 Bohr for oxygen, falling to ~2e-7 by 8 Bohr) and is not
        something the split introduces -- the all-grid form integrates the same
        table.  Both are checked, separately, so a regression in either is
        attributable.
        """
        from scipy.special import erfc

        from mandacaru.pseudopotentials.paw import get_paw

        for symbol in ("H", "O"):
            dataset = get_paw(symbol)
            charge = dataset.valence_charge
            for sigma in (0.3, 0.5, 0.8):
                radius = ls.short_range_radius(dataset, sigma)
                assert radius >= ls.local_cutoff(dataset)
                outside = np.linspace(radius, radius + 6.0, 200)
                erfc_tail = (charge * erfc(outside / (np.sqrt(2.0) * sigma))
                             / outside)
                assert erfc_tail.max() < 1e-9
                table = (ls.short_range_potential(dataset, sigma, outside)
                         + erfc_tail)
                assert np.abs(table).max() < 1e-5

    def test_the_width_follows_the_grid(self):
        atoms = h2()
        for h in (0.30, 0.20, 0.10):
            grid = grid_from_cell(atoms, h)
            sigma = ls.split_width(grid)
            assert sigma == pytest.approx(ls.SIGMA_FACTOR * ls.grid_spacing(grid))
            # The Nyquist weight the factor was chosen for.
            nyquist = np.pi / ls.grid_spacing(grid)
            assert np.exp(-0.5 * (nyquist * sigma) ** 2) < 1e-4


def _pulay(gradients, owners, atom, k, M):
    """``dV^sr`` when the basis functions of ``atom`` move -- the force's formula."""
    own = owners == atom
    out = np.zeros((M, M), dtype=complex)
    for G in gradients.values():
        Gk = G[:, :, k]
        out[own, :] += Gk[own, :]
        out[:, own] += Gk.conj().T[:, own]
    return out


def _hellmann_feynman(gradients, atom, k):
    """``dV^sr`` when ``atom``'s own sphere and potential move."""
    Gk = gradients[atom][:, :, k]
    return -(Gk + Gk.conj().T)


# --------------------------------------------------------------------------- #
# Translation invariance of the quadrature.
# --------------------------------------------------------------------------- #

class TestTranslationInvariance:
    def test_the_short_range_matrix_does_not_see_the_grid(self):
        """A rigid shift by half a grid step leaves it bit-for-bit alone.

        This is the point of the whole construction: the integral depends only
        on the separations of the basis functions from the sphere's center, so
        no grid enters it.  The grid-sampled local potential moved by 24-634 meV
        over the same shifts (water PAW-SZ, h = 0.16-0.25 Angstrom).
        """
        h = 0.25
        atoms = water()
        grid = grid_from_cell(atoms, h)          # frozen: only the molecule moves
        reference = integrals_of(atoms, h, grid=grid).short_range_local()
        shifted = atoms.copy()
        shifted.positions += 0.5 * h * np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        moved = integrals_of(shifted, h, grid=grid).short_range_local()
        assert np.abs(moved - reference).max() < 1e-9
        assert np.abs(reference).max() > 0.1     # not trivially zero

    def test_the_matrix_is_hermitian(self):
        """``v^sr`` is real, so the same quadrature points give ``I = I^H``."""
        matrix = integrals_of(water(), 0.25).short_range_local()
        assert np.abs(matrix - matrix.conj().T).max() < 1e-14

    def test_a_rigid_translation_has_zero_derivative(self):
        """Pulay over every atom plus Hellmann-Feynman over every sphere.

        Moving all the basis functions and all the spheres together is a
        relabeling of the same integral, so the two halves of the derivative
        must cancel exactly -- the invariant that keeps the force free of any
        egg-box from this term.  Built from the *same* two formulas the
        gradient uses, with the real orbital-to-atom map.
        """
        context = context_of(water(), 0.25)
        integrals = context["integrals"]
        owners = np.asarray(context["atom_of_orbital"])
        gradients = integrals.short_range_local_matrices(gradients=True)[1]
        M = integrals.n_orbitals
        for k in range(3):
            total = np.zeros((M, M), dtype=complex)
            for atom in range(len(integrals.datasets)):
                total += _pulay(gradients, owners, atom, k, M)
                total += _hellmann_feynman(gradients, atom, k)
            assert np.abs(total).max() < 1e-14
        # And the gradient is not itself zero -- the cancellation is real.
        assert max(np.abs(G).max() for G in gradients.values()) > 1e-3

    def test_the_two_halves_reproduce_a_finite_difference(self):
        """Move one atom and difference the matrix the energy actually uses.

        ``short_range_local()`` at displaced geometries on a *frozen* grid is
        the ground truth for ``Pulay(B) + Hellmann-Feynman(B)``: moving atom B
        moves both its basis functions and its own sphere, which is exactly the
        sum of the two halves.  This is the check that would catch a sign, a
        conjugation or the Angstrom/Bohr factor -- the analytic derivative is
        per Bohr and an ASE displacement is in Angstrom.
        """
        from mandacaru.units import BOHR_TO_ANGSTROM

        h, atoms = 0.25, water()
        grid = grid_from_cell(atoms, h)
        context = context_of(atoms, h, grid=grid)
        integrals = context["integrals"]
        owners = np.asarray(context["atom_of_orbital"])
        gradients = integrals.short_range_local_matrices(gradients=True)[1]
        M = integrals.n_orbitals
        delta = 1e-3                                   # Bohr
        for atom in (0, 1):                            # oxygen and one hydrogen
            for k in (1, 2):
                moved = []
                for sign in (+1.0, -1.0):
                    shifted = atoms.copy()
                    shifted.positions[atom, k] += (sign * delta
                                                   * BOHR_TO_ANGSTROM)
                    moved.append(integrals_of(shifted, h, grid=grid)
                                 .short_range_local())
                numerical = (moved[0] - moved[1]) / (2.0 * delta)
                analytic = (_pulay(gradients, owners, atom, k, M)
                            + _hellmann_feynman(gradients, atom, k))
                assert np.abs(analytic - numerical).max() < 1e-6
                assert np.abs(numerical).max() > 1e-3


# --------------------------------------------------------------------------- #
# Agreement with the all-grid form.
# --------------------------------------------------------------------------- #

class TestAgreesWithTheGrid:
    @pytest.mark.parametrize("h, tolerance", [(0.16, 2e-4), (0.13, 1e-4)])
    def test_fine_grid_limit(self, h, tolerance):
        """Both forms integrate the same potential, so they must converge together.

        Measured on H2 PAW-SZ: ``split - grid`` is -2.35e-4 Hartree at
        h = 0.30 Angstrom, -4.1e-7 at 0.16 and +5.3e-8 at 0.09 -- the grid's own
        error on the short-range part, which is what the quadrature removes.
        """
        atoms = h2()
        PAWIntegrals.exact_local_potential = False
        try:
            on_grid = rhf_total(integrals_of(atoms, h))
        finally:
            PAWIntegrals.exact_local_potential = True
        split = rhf_total(integrals_of(atoms, h))
        assert split == pytest.approx(on_grid, abs=tolerance)

    def test_the_energy_moves_by_less_than_a_milli_hartree(self):
        """Water at a production spacing: a small, recorded shift.

        The split is not a model change -- it is a quadrature change -- so the
        energy must move by the grid's error on one term, not by a chemical
        amount.
        """
        atoms, h = water(), 0.20
        PAWIntegrals.exact_local_potential = False
        try:
            on_grid = rhf_total(integrals_of(atoms, h))
        finally:
            PAWIntegrals.exact_local_potential = True
        split = rhf_total(integrals_of(atoms, h))
        assert abs(split - on_grid) < 1e-3
        assert abs(split - on_grid) * HARTREE_TO_EV < 0.03


# --------------------------------------------------------------------------- #
# Everything else is untouched.
# --------------------------------------------------------------------------- #

class TestInertElsewhere:
    def test_a_plain_basis_has_no_short_range_hook(self):
        """The base hook returns ``None``, so ``_compute_one_body`` adds nothing."""
        atoms = Atoms("H2", positions=[[3, 3, 3], [3, 3, 3.74]], cell=[6.0] * 3)
        integrals = integrals_of(atoms, 0.3, basis="FAO")
        assert integrals.short_range_local() is None
        assert not getattr(integrals, "split_local_potential", False)

    def test_a_norm_conserving_family_is_untouched(self):
        """NCPP builds plain ``MolecularIntegrals``; nothing is separated."""
        atoms = Atoms("H2", positions=[[3, 3, 3], [3, 3, 3.74]], cell=[6.0] * 3)
        integrals = integrals_of(atoms, 0.3, basis="NCPP")
        assert integrals.short_range_local() is None

    @staticmethod
    def _core_slab(integrals):
        """Grid points inside the oxygen core, where the two halves differ."""
        center = np.asarray(integrals._potentials.nuclei[0][1], dtype=float)
        radius = np.sqrt(sum((a.ravel() - center[i]) ** 2 for i, a in
                             enumerate((integrals.grid.X, integrals.grid.Y,
                                        integrals.grid.Z))))
        inside = radius < 1.0                        # Bohr, well inside r_cut
        assert inside.sum() > 4
        return tuple(a.ravel()[inside] for a in (integrals.grid.X,
                                                 integrals.grid.Y,
                                                 integrals.grid.Z))

    def test_the_switch_restores_the_grid_potential_exactly(self, all_grid):
        """With the split off, ``external_potential`` is the full local channel."""
        integrals = integrals_of(water(), 0.25)
        assert integrals.short_range_local() is None
        assert integrals.split_local_potential is False
        slab = self._core_slab(integrals)
        sampled = integrals.external_potential()(*slab)
        expected = integrals._potentials.pseudopotential(*slab)
        assert np.abs(sampled - expected).max() == 0.0

    def test_the_grid_samples_only_the_long_range_half_when_split(self):
        integrals = integrals_of(water(), 0.25)
        assert integrals.split_local_potential is True
        slab = self._core_slab(integrals)
        sampled = integrals.external_potential()(*slab)
        full = integrals._potentials.pseudopotential(*slab)
        assert np.abs(sampled - full).max() > 0.1           # really different
        # ... and the difference is exactly what the quadrature integrates.
        sigma = integrals.local_split_width()
        rebuilt = np.zeros_like(sampled)
        for dataset, (_z, center) in zip(integrals.datasets,
                                         integrals._potentials.nuclei):
            radius = np.sqrt(sum((slab[i] - center[i]) ** 2 for i in range(3)))
            rebuilt += ls.short_range_potential(dataset, sigma, radius)
        assert np.abs(full - sampled - rebuilt).max() < 1e-12


# --------------------------------------------------------------------------- #
# Forces.
# --------------------------------------------------------------------------- #

def water_calculator(h=0.25):
    return Mandacaru(method="adapt-vqe", basis=SZ, h=h, pool="fermionic",
                     optimizer="L-BFGS-B", max_iterations=60,
                     gradient_tolerance=1e-4, profile=False)


@pytest.fixture(scope="module")
def h2o_forces():
    atoms = water()
    atoms.calc = water_calculator()
    forces = atoms.get_forces()
    return (atoms, forces, atoms.calc.force_result.unprojected,
            dict(atoms.calc.force_result.details))


class TestForces:
    def test_the_gradient_knows_about_the_split(self, h2o_forces):
        _atoms, _forces, _raw, details = h2o_forces
        assert details["split_local_potential"] is True

    def test_p_valence_force_is_the_derivative_of_the_energy(self, h2o_forces):
        """Central difference of the calculator's own energy on its frozen grid.

        The oxygen sphere holds both hydrogens, so this exercises the
        cross-center part of the quadrature derivative -- the part a
        single-atom test would miss.  Only one component is differenced, to keep
        the test inside the suite's time budget; the symmetry test below covers
        the others.
        """
        atoms, _forces, raw, _details = h2o_forces
        step, atom, axis = 0.004, 0, 1
        energies = []
        for sign in (1, -1):
            moved = atoms.copy()
            moved.positions[atom, axis] += sign * step
            moved.calc = atoms.calc
            energies.append(moved.get_potential_energy())
        numerical = -(energies[0] - energies[1]) / (2 * step)
        # The raw gradient: the finite difference is of the discretized energy,
        # which is not translation invariant, while `project_translation="auto"`
        # has taken that component out of the reported force.
        assert raw[atom, axis] == pytest.approx(numerical, abs=2e-2)

    def test_forces_respect_the_c2v_symmetry(self, h2o_forces):
        """The two O-H bonds lie along +y and +z and the cubic grid shares that
        symmetry, so a correct derivative of the sphere integrals keeps it."""
        _atoms, forces, _raw, _details = h2o_forces
        assert forces[0, 1] == pytest.approx(forces[0, 2], abs=5e-3)
        assert forces[1, 1] == pytest.approx(forces[2, 2], abs=5e-3)
        assert abs(forces[0, 1]) > 1.0
        assert np.abs(forces[:, 0]).max() < 1e-3

    def test_h2_force_agrees_with_the_all_grid_gradient(self):
        """The split moves the force by far less than the egg-box it lives in.

        Both are the gradient of their own energy, and the two energies differ
        only by the grid's error on the short-range term, so the forces have to
        track each other closely.
        """
        def run(split):
            PAWIntegrals.exact_local_potential = split
            try:
                atoms = h2(distance=0.80)
                atoms.calc = Mandacaru(method="adapt-vqe", basis=SZ, h=0.25,
                                       pool="fermionic", optimizer="L-BFGS-B",
                                       max_iterations=10,
                                       gradient_tolerance=1e-6,
                                       project_translation=False,
                                       profile=False)
                atoms.get_forces()
                return atoms.calc.force_result.unprojected
            finally:
                PAWIntegrals.exact_local_potential = True

        on_grid, split = run(False), run(True)
        assert np.abs(split - on_grid).max() < 0.05
        assert np.abs(on_grid).max() > 0.5              # a real force

    def test_the_legacy_force_path_refuses_the_split(self):
        """``scf-response`` rebuilds ``T + V_grid`` itself and knows nothing
        about the sphere integrals, so it must fail loudly rather than
        differentiate an energy nobody evaluated."""
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis=SZ, h=0.3,
                               force_method="scf-response", profile=False)
        with pytest.raises(NotImplementedError, match="range-separated"):
            atoms.get_forces()
