# -*- coding: utf-8 -*-
# file: test/algorithms/test_onsite.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The quadratic one-center Hartree term (``one_center_hartree="quadratic"``)
and the combination of on-site corrections (:mod:`mandacaru.algorithms.onsite`)."""

from __future__ import annotations

import numpy as np
import pytest

from mandacaru.algorithms import onsite
from mandacaru.units import HARTREE_TO_EV


@pytest.fixture(scope="module", autouse=True)
def _scratch_directory(tmp_path_factory):
    """A PAW molecule caches its two-center tables in the working directory."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("onsite"))
        yield


def _sphere(n=4, seed=0):
    """A random ``(positions, W, D0)`` with the symmetries of a Coulomb tensor,
    ``W[a,b,c,d] = W[b,a,c,d] = W[a,b,d,c] = W[c,d,a,b]``."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n,) * 4)
    W = W + W.transpose(1, 0, 2, 3)
    W = W + W.transpose(0, 1, 3, 2)
    W = W + W.transpose(2, 3, 0, 1)
    D0 = rng.normal(size=(n, n))
    return np.arange(n) + 1, W, 0.5 * (D0 + D0.T)


class TestOption:
    def test_the_default_is_the_datasets_own_term(self):
        assert onsite.resolve_one_center_hartree(None) == "linearized"
        assert onsite.resolve_one_center_hartree("quadratic") == "quadratic"

    @pytest.mark.parametrize("bad", ["exact", "", 1, True])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError, match="one_center_hartree"):
            onsite.resolve_one_center_hartree(bad)


class TestQuadraticHartree:
    def test_the_atom_s_own_density_costs_nothing(self):
        positions, W, D0 = _sphere()
        term = onsite.QuadraticHartree([(positions, W, D0)], 6)
        R = np.zeros((6, 6))
        R[np.ix_(positions, positions)] = D0.T
        energy, (operator,) = term.evaluate((R,), restricted=True)
        assert energy == pytest.approx(0.0, abs=1e-14)
        assert np.max(np.abs(operator)) < 1e-14

    @pytest.mark.parametrize("restricted", [True, False])
    def test_the_operator_is_the_derivative_of_the_energy(self, restricted):
        positions, W, D0 = _sphere(seed=3)
        term = onsite.QuadraticHartree([(positions, W, D0)], 6)
        rng = np.random.default_rng(5)
        spins = 1 if restricted else 2

        def draw():
            a = rng.normal(size=(6, 6))
            return 0.5 * (a + a.T)

        R = tuple(draw() for _ in range(spins))
        dR = tuple(draw() for _ in range(spins))
        eps = 1e-5
        up = term.evaluate(tuple(r + eps * d for r, d in zip(R, dR)),
                           restricted)[0]
        down = term.evaluate(tuple(r - eps * d for r, d in zip(R, dR)),
                             restricted)[0]
        operators = term.evaluate(R, restricted)[1]
        # dE = sum_s Tr[O_s dR_s^T]: the density enters through its total, so
        # a restricted R (both spins summed) has the one operator.
        analytic = sum(float(np.sum(o * d.T)) for o, d in zip(operators, dR))
        assert (up - down) / (2 * eps) == pytest.approx(analytic, rel=1e-7)

    def test_the_density_enters_through_its_total(self):
        positions, W, D0 = _sphere(seed=7)
        term = onsite.QuadraticHartree([(positions, W, D0)], 6)
        rng = np.random.default_rng(1)
        a, b = (0.5 * (m + m.T) for m in rng.normal(size=(2, 6, 6)))
        assert term.evaluate((a, b), False)[0] == pytest.approx(
            term.evaluate((a + b,), True)[0], rel=1e-12)

    def test_the_wrong_number_of_matrices_is_refused(self):
        term = onsite.QuadraticHartree([_sphere()], 6)
        with pytest.raises(ValueError, match="restricted"):
            term.evaluate((np.zeros((6, 6)),) * 2, restricted=True)


class TestCombination:
    def test_energies_and_operators_add(self):
        one = onsite.QuadraticHartree([_sphere(seed=1)], 6)
        two = onsite.QuadraticHartree([_sphere(seed=2)], 6)
        both = onsite.CombinedOnsite(first=one, second=two)
        R = (np.eye(6),)
        e1, (o1,) = one.evaluate(R, True)
        e2, (o2,) = two.evaluate(R, True)
        energy, (operator,) = both.evaluate(R, True)
        assert energy == pytest.approx(e1 + e2)
        assert np.allclose(operator, o1 + o2)
        assert both.energies == pytest.approx({"first": e1, "second": e2})

    def test_an_empty_combination_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            onsite.CombinedOnsite()


# --------------------------------------------------------------------------- #
# The option in the Kohn-Sham solvers.
# --------------------------------------------------------------------------- #

def _molecule(bond=1.5):
    from ase import Atoms

    atoms = Atoms("CuH", positions=[[0.0, 0.0, 0.0], [0.05, 0.0, bond]],
                  cell=[8.0] * 3)
    atoms.center()
    return atoms


def _run(atoms, **options):
    from mandacaru import Mandacaru

    xc = options.pop("xc", "lda")
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.3,
                           trace=False, stability=False,
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           **options)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def runs():
    return {"linearized": _run(_molecule()),
            "quadratic": _run(_molecule(), one_center_hartree="quadratic")}


class TestMolecule:
    def test_the_term_is_a_part_of_the_energy_of_its_own(self, runs):
        scf = runs["quadratic"].calc.result.scf
        assert scf.one_center_hartree_energy != 0.0
        assert runs["linearized"].calc.result.scf.one_center_hartree_energy \
            == 0.0

    def test_the_relaxed_energy_lies_between_the_two_bounds(self, runs):
        """E_quad[rho_q] <= E_quad[rho_lin] = E_lin + E2[rho_lin], and
        E_quad[rho_q] >= E_lin[rho_lin] + E2[rho_q] (E_lin is minimal at
        rho_lin)."""
        from mandacaru.algorithms.dft import kohn_sham_solver

        lin, quad = runs["linearized"], runs["quadratic"]
        na, nb = lin.calc.result.num_particles
        solver = kohn_sham_solver(
            quad.calc._solver._gradient_context["integrals"], na + nb, "lda")
        C = lin.calc.result.scf.mo_coefficients
        solver._fock(2 * C[:, :na] @ C[:, :na].T)
        upper = solver.one_center_hartree_energy * HARTREE_TO_EV
        lower = quad.calc.result.scf.one_center_hartree_energy * HARTREE_TO_EV
        shift = (quad.get_potential_energy() - lin.get_potential_energy())
        assert lower - 1e-6 <= shift <= upper + 1e-6

    def test_the_energy_and_the_fock_matrix_are_consistent(self, runs):
        """dE = Tr[F dD] for any change of the density matrix."""
        from mandacaru.algorithms.dft import kohn_sham_solver

        quad = runs["quadratic"]
        na, nb = quad.calc.result.num_particles
        solver = kohn_sham_solver(
            quad.calc._solver._gradient_context["integrals"], na + nb, "lda")
        C = quad.calc.result.scf.mo_coefficients
        D = 2 * C[:, :na] @ C[:, :na].T
        a = np.random.default_rng(2).normal(size=D.shape)
        d = 0.05 * 0.5 * (a + a.T)
        eps = 1e-4
        up = solver._fock(D + eps * d)[1]
        down = solver._fock(D - eps * d)[1]
        analytic = float(np.sum(solver._fock(D)[0] * d.T))
        assert (up - down) / (2 * eps) == pytest.approx(analytic, abs=1e-8)

    def test_the_forces_are_the_derivative_of_the_energy(self):
        """The unprojected analytic force against Richardson-extrapolated
        central differences of the energy on the frozen grid."""
        atoms = _run(_molecule(), one_center_hartree="quadratic")
        atoms.get_forces()
        analytic = atoms.calc.force_result.unprojected

        def energy(atom, axis, step):
            moved = atoms.copy()
            moved.positions[atom, axis] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        for atom, axis in ((1, 2), (1, 0)):
            coarse, fine = (-(energy(atom, axis, h) - energy(atom, axis, -h))
                            / (2 * h) for h in (0.004, 0.002))
            numerical = fine + (fine - coarse) / 3.0
            assert analytic[atom, axis] == pytest.approx(numerical, abs=1e-4)

    def test_it_combines_with_the_hubbard_correction(self, runs):
        both = _run(_molecule(), one_center_hartree="quadratic",
                    hubbard_u={"Cu": {"d": 4.0}})
        scf = both.calc.result.scf
        assert scf.hubbard_energy > 0.0
        assert scf.one_center_hartree_energy != 0.0

    def test_a_hybrid_is_refused(self):
        with pytest.raises(NotImplementedError, match="hybrid"):
            _run(_molecule(), one_center_hartree="quadratic", xc="hse06")

    def test_a_bad_value_is_refused(self):
        with pytest.raises(ValueError, match="one_center_hartree"):
            _run(_molecule(), one_center_hartree="exact")


class TestCrystal:
    def test_a_molecule_in_a_box_gets_the_molecule_s_shift(self):
        """The crystal and the molecular codes share no line of the on-site
        assembly: their energy shifts agree to the difference between a
        periodic and an isolated box."""
        from ase import Atoms
        from mandacaru import Mandacaru

        def box(pbc, **options):
            atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]],
                          cell=[9.0] * 3, pbc=pbc)
            atoms.center()
            extra = ({"kpts": {"size": (1, 1, 1), "gamma": True},
                      "smearing": {"method": "fermi-dirac", "width": 0.01}}
                     if pbc else {"stability": False})
            atoms.calc = Mandacaru(
                method="dft", xc="lda", grid_spacing=0.3, trace=False,
                basis={"name": "PAW-LCAO", "size": "SZ"}, **extra, **options)
            return atoms.get_potential_energy()

        shift = {label: box(pbc, one_center_hartree="quadratic") - box(pbc)
                 for label, pbc in (("molecule", False), ("crystal", True))}
        assert abs(shift["molecule"]) > 0.05
        assert shift["crystal"] == pytest.approx(shift["molecule"], abs=0.03)


# --------------------------------------------------------------------------- #
# The nonlinear one-center exchange-correlation (one_center_xc="nonlinear").
# --------------------------------------------------------------------------- #

def _feh(bond=1.6):
    from ase import Atoms

    atoms = Atoms("FeH", positions=[[0.0, 0.0, 0.0], [0.05, 0.0, bond]],
                  cell=[8.0] * 3)
    atoms.center()
    atoms.set_initial_magnetic_moments([3.0, 0.0])
    return atoms


@pytest.fixture(scope="module")
def spin_runs():
    return {"linearized": _run(_feh()),
            "nonlinear": _run(_feh(), one_center_xc="nonlinear")}


@pytest.fixture(scope="module")
def sphere(spin_runs):
    """The iron sphere's quadrature table, with its dataset and projectors."""
    from mandacaru.pseudopotentials.onecenter_xc import XCSphere

    context = spin_runs["nonlinear"].calc._solver._gradient_context
    projectors = [p for p in context["kb_projectors"] if p.atom_index == 0]
    datasets = context["pseudopotentials"]
    dataset = (datasets["Fe"] if isinstance(datasets, dict)
               else next(d for d in datasets if d.symbol == "Fe"))
    basis = projectors[0].projector_basis
    return XCSphere(dataset, projectors, basis), dataset, projectors, basis


class TestNonlinearOption:
    def test_the_default_is_the_datasets_own_term(self):
        assert onsite.resolve_one_center_xc(None) == "linearized"
        assert onsite.resolve_one_center_xc("nonlinear") == "nonlinear"

    @pytest.mark.parametrize("bad", ["exact", "", 1, True])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError, match="one_center_xc"):
            onsite.resolve_one_center_xc(bad)

    def test_a_gradient_functional_is_refused(self):
        with pytest.raises(NotImplementedError, match="local density"):
            _run(_feh(), one_center_xc="nonlinear", xc="pbe")


class TestXCSphere:
    def test_the_reference_atom_costs_nothing(self, sphere):
        table = sphere[0]
        half = 0.5 * table.reference
        energy, o_up, o_dn = table.remainder(half, half)
        assert energy == 0.0
        assert np.max(np.abs(o_up)) == 0.0 and np.max(np.abs(o_dn)) == 0.0

    def test_the_derivative_at_the_reference_is_the_datasets_potential(
            self, sphere):
        """V_self = dF/dD at the spherical, unpolarized reference equals the
        dataset's frozen V to the quadrature's accuracy."""
        from mandacaru.pseudopotentials.onecenter import exact_one_center_terms

        table, dataset, projectors, basis = sphere
        terms = exact_one_center_terms(dataset, basis)
        v_self = table.reference_potential()
        for l, (V, _K, _D0) in terms.items():
            idx = [k for k, p in enumerate(projectors)
                   if p.l == l and p.m == 0]
            block = v_self[np.ix_(idx, idx)]
            assert np.max(np.abs(block - V)) < 1e-3 * max(np.max(np.abs(V)),
                                                          1.0)

    def test_the_operator_is_the_derivative_of_the_energy(self, sphere):
        """Spin-polarized, strictly positive densities (a clipped channel
        makes the functional non-smooth and the difference quotient noisy)."""
        table = sphere[0]
        n = table.n
        rng = np.random.default_rng(4)
        half = 0.5 * table.reference
        R_up = half * 1.15 + 0.01 * np.eye(n) * (half != 0).any()
        R_dn = half * 0.85
        d_up = 0.02 * half * rng.uniform(0.5, 1.5, size=half.shape)
        d_dn = -0.02 * half * rng.uniform(0.5, 1.5, size=half.shape)
        d_up, d_dn = 0.5 * (d_up + d_up.T), 0.5 * (d_dn + d_dn.T)
        eps = 1e-4
        up = table.remainder(R_up + eps * d_up, R_dn + eps * d_dn)[0]
        down = table.remainder(R_up - eps * d_up, R_dn - eps * d_dn)[0]
        _e, o_up, o_dn = table.remainder(R_up, R_dn)
        analytic = float(np.trace(o_up @ d_up) + np.trace(o_dn @ d_dn))
        assert (up - down) / (2 * eps) == pytest.approx(analytic, rel=2e-3)

    def test_the_quadrature_is_converged(self, sphere):
        """A denser radial and angular rule moves the remainder of a
        polarized, non-spherical d shell by less than 1 meV."""
        from mandacaru.pseudopotentials.onecenter_xc import XCSphere

        table, dataset, projectors, basis = sphere
        dense = XCSphere(dataset, projectors, basis, radial=160, polar=32,
                         azimuthal=64)
        half = 0.5 * table.reference
        mask = np.array([[p.l == 2 and q.l == 2 and p.m == q.m
                          for q in projectors] for p in projectors])
        R_up = half + 0.5 * table.reference * mask
        R_up[np.ix_([k for k, p in enumerate(projectors) if p.l == 2 and p.m == 1],
                    [k for k, p in enumerate(projectors) if p.l == 2 and p.m == 1])] *= 0.5
        R_dn = half - 0.4 * table.reference * mask
        diff = abs(table.remainder(R_up, R_dn)[0]
                   - dense.remainder(R_up, R_dn)[0])
        assert diff * HARTREE_TO_EV < 1e-3


class TestNonlinearMolecule:
    def test_the_term_is_a_part_of_the_energy_of_its_own(self, spin_runs):
        scf = spin_runs["nonlinear"].calc.result.scf
        assert scf.one_center_xc_energy != 0.0
        assert spin_runs["linearized"].calc.result.scf.one_center_xc_energy \
            == 0.0

    def test_the_energy_and_the_fock_matrices_are_consistent(self, spin_runs):
        """dE = Tr[F_alpha dD_alpha] + Tr[F_beta dD_beta]."""
        from mandacaru.algorithms.dft import kohn_sham_solver

        run = spin_runs["nonlinear"]
        na, nb = run.calc.result.num_particles
        scf = run.calc.result.scf
        solver = kohn_sham_solver(
            run.calc._solver._gradient_context["integrals"], na + nb, "lda",
            spins=(na, nb))
        Da = scf.mo_coefficients_alpha[:, :na] @ scf.mo_coefficients_alpha[:, :na].T
        Db = scf.mo_coefficients_beta[:, :nb] @ scf.mo_coefficients_beta[:, :nb].T
        rng = np.random.default_rng(6)

        def symmetric():
            a = rng.normal(size=Da.shape)
            return 0.05 * 0.5 * (a + a.T)

        da, db = symmetric(), symmetric()
        eps = 1e-4
        up = solver._fock_pair(Da + eps * da, Db + eps * db)[2]
        down = solver._fock_pair(Da - eps * da, Db - eps * db)[2]
        Fa, Fb = solver._fock_pair(Da, Db)[:2]
        analytic = float(np.sum(Fa * da.T) + np.sum(Fb * db.T))
        assert solver.one_center_xc_energy != 0.0
        assert (up - down) / (2 * eps) == pytest.approx(analytic, abs=1e-7)

    def test_the_forces_are_the_derivative_of_the_energy(self):
        # 1.65 Angstrom: at 1.6 FeH has a second SCF solution close in energy
        # and the difference quotient can hop to it.
        atoms = _run(_feh(1.65), one_center_xc="nonlinear")
        atoms.get_forces()
        analytic = atoms.calc.force_result.unprojected

        def energy(atom, axis, step):
            moved = atoms.copy()
            moved.positions[atom, axis] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        for atom, axis in ((1, 2), (1, 0)):
            coarse, fine = (-(energy(atom, axis, h) - energy(atom, axis, -h))
                            / (2 * h) for h in (0.004, 0.002))
            numerical = fine + (fine - coarse) / 3.0
            assert analytic[atom, axis] == pytest.approx(numerical, abs=2e-4)

    def test_it_combines_with_the_other_corrections(self):
        both = _run(_feh(), one_center_xc="nonlinear",
                    one_center_hartree="quadratic")
        scf = both.calc.result.scf
        assert scf.one_center_xc_energy != 0.0
        assert scf.one_center_hartree_energy != 0.0


class TestNonlinearCrystal:
    @pytest.mark.parametrize("metal, bond, moment", [
        ("Cu", 1.46, None),             # closed shell, restricted
        ("Fe", 1.65, 3.0)])             # spin-polarized
    def test_a_molecule_in_a_box_gets_the_molecule_s_shift(
            self, metal, bond, moment):
        """The crystal uses complex harmonics and the full k-mesh, the
        molecule real ones: the shifts agree to the periodic-box difference."""
        from ase import Atoms
        from mandacaru import Mandacaru

        def box(pbc, **options):
            atoms = Atoms(f"{metal}H", positions=[[0, 0, 0], [0, 0, bond]],
                          cell=[9.0] * 3, pbc=pbc)
            atoms.center()
            if moment is not None:
                atoms.set_initial_magnetic_moments([moment, 0.0])
            extra = ({"kpts": {"size": (1, 1, 1), "gamma": True},
                      "smearing": {"method": "fermi-dirac", "width": 0.01}}
                     if pbc else {"stability": False})
            atoms.calc = Mandacaru(
                method="dft", xc="lda", grid_spacing=0.3, trace=False,
                basis={"name": "PAW-LCAO", "size": "SZ"}, **extra, **options)
            return atoms.get_potential_energy()

        shift = {label: box(pbc, one_center_xc="nonlinear") - box(pbc)
                 for label, pbc in (("molecule", False), ("crystal", True))}
        assert abs(shift["molecule"]) > 0.005
        # Each shift is a difference of two SCF runs, and an open d shell has
        # several close solutions: the spin-polarized pair agrees to 0.06 eV.
        assert shift["crystal"] == pytest.approx(shift["molecule"], abs=0.1)
