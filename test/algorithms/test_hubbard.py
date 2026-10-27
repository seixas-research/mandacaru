# -*- coding: utf-8 -*-
# file: test/algorithms/test_hubbard.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The on-site Hubbard correction: the ``hubbard_u`` option and the energy and
potential of one shell."""

from __future__ import annotations

import numpy as np
import pytest

from mandacaru.algorithms import hubbard
from mandacaru.units import EV_TO_HARTREE


@pytest.fixture(scope="module", autouse=True)
def _scratch_directory(tmp_path_factory):
    """Run from a scratch directory: a PAW molecule caches its two-center
    tables in the working directory."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("hubbard"))
        yield


def _occupation(size=7, seed=0):
    """A Hermitian occupation matrix with eigenvalues inside [0, 1]."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(size, size))
                        + 1j * rng.normal(size=(size, size)))
    return (q * rng.uniform(0.0, 1.0, size)) @ q.conj().T


class TestOption:
    def test_none_means_no_correction(self):
        assert hubbard.resolve_hubbard_u(None) == {}
        assert hubbard.resolve_hubbard_u({}) == {}

    def test_u_is_converted_from_electron_volt_to_hartree(self):
        out = hubbard.resolve_hubbard_u({"Gd": {"f": 6.0}, "Fe": {"d": 3.0}})
        assert out == {"Gd": {3: pytest.approx(6.0 * EV_TO_HARTREE)},
                       "Fe": {2: pytest.approx(3.0 * EV_TO_HARTREE)}}

    def test_a_zero_u_is_dropped(self):
        assert hubbard.resolve_hubbard_u({"Gd": {"f": 0.0}}) == {}

    @pytest.mark.parametrize("bad", [
        6.0, "Gd", [("Gd", 6.0)],                 # not a mapping of mappings
        {"Gd": 6.0},                              # a bare number: the shell is never guessed
        {"Gd": {}},
        {"Xx": {"f": 6.0}}, {1: {"f": 6.0}},      # not an element
        {"Gd": {"p": 6.0}}, {"Gd": {3: 6.0}},     # not a supported shell letter
        {"Gd": {"f": -1.0}}, {"Gd": {"f": float("nan")}},
        {"Gd": {"f": float("inf")}}, {"Gd": {"f": True}}, {"Gd": {"f": "6"}}])
    def test_bad_options_raise(self, bad):
        with pytest.raises(ValueError, match="hubbard_u"):
            hubbard.resolve_hubbard_u(bad)


class TestShell:
    def test_a_full_or_empty_shell_costs_nothing(self):
        for n in (np.eye(7), np.zeros((7, 7))):
            assert hubbard.onsite_energy(n, 0.2) == pytest.approx(0.0, abs=1e-14)

    def test_a_half_filled_shell_costs_the_most(self):
        n = 0.5 * np.eye(7)
        assert hubbard.onsite_energy(n, 0.2) == pytest.approx(
            0.5 * 0.2 * 7 * 0.25)
        assert hubbard.onsite_energy(n, 0.2) > hubbard.onsite_energy(
            0.9 * np.eye(7), 0.2)

    def test_the_energy_is_never_negative_for_physical_occupations(self):
        for seed in range(5):
            assert hubbard.onsite_energy(_occupation(seed=seed), 0.3) >= 0.0

    def test_the_potential_is_the_derivative_of_the_energy(self):
        """dE = Tr[V dn] for a Hermitian change of the occupation matrix."""
        n = _occupation()
        rng = np.random.default_rng(5)
        raw = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
        dn = 0.5 * (raw + raw.conj().T)
        step = 1e-6
        numerical = (hubbard.onsite_energy(n + step * dn, 0.3)
                     - hubbard.onsite_energy(n - step * dn, 0.3)) / (2 * step)
        analytic = float(np.real(np.trace(hubbard.onsite_potential(n, 0.3) @ dn)))
        assert analytic == pytest.approx(numerical, rel=1e-8, abs=1e-10)

    def test_the_potential_pushes_towards_integer_occupations(self):
        v = hubbard.onsite_potential(np.diag([0.0, 0.5, 1.0]), 0.4)
        assert np.allclose(np.diag(v), [0.2, 0.0, -0.2])


# --------------------------------------------------------------------------- #
# Weights, the occupation matrix and the operator, on a real PAW-LCAO dataset.
# --------------------------------------------------------------------------- #

def _projectors(symbol="Cu", positions=((0.0, 0.0, 0.0),)):
    from mandacaru.pseudopotentials import get_paw
    from mandacaru.pseudopotentials.paw import paw_projectors

    dataset = get_paw(symbol)
    projectors = paw_projectors([symbol] * len(positions), positions,
                                {symbol: dataset})
    return dataset, projectors


class TestShellWeights:
    def test_the_occupation_of_the_reference_orbital_is_the_same_in_both_bases(self):
        """A state that is the dataset's bound reference wave has the
        coefficients (1, 0) in the dual basis and ``B^-T (1, 0)`` in the raw
        one; its occupation ``(w . c)^2`` cannot depend on the choice."""
        from mandacaru.pseudopotentials.onecenter import _basis_transform

        dataset, _ = _projectors("Cu")
        dual = hubbard.shell_weights(dataset, 2, "dual")
        raw = hubbard.shell_weights(dataset, 2, "raw")
        transform = _basis_transform(dataset, 2, "raw")
        c_dual = np.array([1.0, 0.0])
        c_raw = np.linalg.inv(transform).T @ c_dual
        assert float(dual @ c_dual) ** 2 == pytest.approx(
            float(raw @ c_raw) ** 2, rel=1e-8)

    def test_a_shell_inside_its_sphere_holds_about_one_electron_per_orbital(self):
        dataset, _ = _projectors("Cu")
        n = float(hubbard.shell_weights(dataset, 2, "dual")[0]) ** 2
        assert 0.7 < n <= 1.0 + 1e-6


class TestBuilder:
    def test_the_selection_is_the_shell_of_the_named_atom_only(self):
        dataset, projectors = _projectors("Cu", ((0, 0, 0), (0, 0, 3.0)))
        onsite = hubbard.build_onsite(projectors, {"Cu": dataset},
                                      {"Cu": {2: 0.1}})
        assert len(onsite.shells) == 2                      # one d shell per atom
        for _u, positions, selection in onsite.shells:
            assert len(positions) == 10                     # 5 m x 2 projectors
            assert {projectors[p].l for p in positions} == {2}
            assert selection.shape == (10, 5)
        assert set(onsite.shells[0][1]).isdisjoint(onsite.shells[1][1])

    def test_an_element_that_is_not_in_the_system_is_refused(self):
        dataset, projectors = _projectors("Cu")
        with pytest.raises(ValueError, match="not in the system"):
            hubbard.build_onsite(projectors, {"Cu": dataset}, {"Gd": {3: 0.1}})

    def test_a_shell_the_dataset_lacks_is_refused(self):
        dataset, projectors = _projectors("Cu")
        with pytest.raises(ValueError, match="no l = 3 channel"):
            hubbard.build_onsite(projectors, {"Cu": dataset}, {"Cu": {3: 0.1}})

    def test_the_operator_is_the_derivative_of_the_energy_in_the_projected_density(self):
        dataset, projectors = _projectors("Cu")
        onsite = hubbard.build_onsite(projectors, {"Cu": dataset},
                                      {"Cu": {2: 0.1}})
        size = onsite.size
        rng = np.random.default_rng(2)
        a = rng.normal(size=(size, size))
        R = 0.05 * (a @ a.T) / size                         # a positive R
        b = rng.normal(size=(size, size))
        dR = 0.5 * (b + b.T)
        step = 1e-6
        [O, _other] = onsite.evaluate((R, R), restricted=False)[1]
        up = onsite.evaluate((R + step * dR, R), restricted=False)[0]
        down = onsite.evaluate((R - step * dR, R), restricted=False)[0]
        assert (up - down) / (2 * step) == pytest.approx(
            float(np.sum(O * dR.T)), rel=1e-7, abs=1e-12)

    def test_restricted_is_two_equal_spin_channels_of_half_the_density(self):
        dataset, projectors = _projectors("Cu")
        onsite = hubbard.build_onsite(projectors, {"Cu": dataset},
                                      {"Cu": {2: 0.1}})
        rng = np.random.default_rng(3)
        a = rng.normal(size=(onsite.size, onsite.size))
        R = 0.05 * (a @ a.T) / onsite.size
        e_restricted, [O] = onsite.evaluate((R,), restricted=True)
        e_spin, [Oa, Ob] = onsite.evaluate((0.5 * R, 0.5 * R), restricted=False)
        assert e_restricted == pytest.approx(e_spin, rel=1e-12)
        assert np.allclose(O, Oa) and np.allclose(Oa, Ob)


# --------------------------------------------------------------------------- #
# The option on a molecule.
# --------------------------------------------------------------------------- #

def _molecule(symbol, bond, moments=None):
    from ase import Atoms

    atoms = Atoms(f"{symbol}H", positions=[[0, 0, 0], [0, 0, bond]],
                  cell=[8.0] * 3)
    atoms.center()
    if moments is not None:
        atoms.set_initial_magnetic_moments(moments)
    return atoms


def _run(atoms, **options):
    from mandacaru import Mandacaru

    xc = options.pop("xc", "lda")
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.3,
                           trace=False, stability=False,
                           basis={"name": "PAW-LCAO", "size": "SZ"}, **options)
    atoms.get_potential_energy()
    return atoms


@pytest.mark.parametrize("symbol, bond, moments", [
    ("Cu", 1.46, None),                 # restricted, closed d shell
    ("Fe", 1.6, (3.0, 0.0))])           # unrestricted, open d shell
def test_the_energy_and_the_fock_matrix_are_consistent(symbol, bond, moments):
    """dE = Tr[F dD] for any change of the density matrix, with the on-site
    term on: its energy and its operator are the same functional."""
    from mandacaru.algorithms.dft import kohn_sham_solver

    atoms = _run(_molecule(symbol, bond, moments),
                 hubbard_u={symbol: {"d": 4.0}})
    scf = atoms.calc.result.scf
    integrals = atoms.calc._solver._gradient_context["integrals"]
    na, nb = atoms.calc.result.num_particles
    solver = kohn_sham_solver(integrals, na + nb, "lda",
                              spins=(na, nb) if moments else None)
    rng = np.random.default_rng(1)

    def symmetric():
        a = rng.normal(size=(solver.M, solver.M))
        return 0.05 * 0.5 * (a + a.T)

    def projector(coefficients, n):
        return coefficients[:, :n] @ coefficients[:, :n].T

    eps = 1e-4
    if moments:
        Da = projector(scf.mo_coefficients_alpha, na)
        Db = projector(scf.mo_coefficients_beta, nb)
        da, db = symmetric(), symmetric()
        up = solver._fock_pair(Da + eps * da, Db + eps * db)[2]
        down = solver._fock_pair(Da - eps * da, Db - eps * db)[2]
        Fa, Fb = solver._fock_pair(Da, Db)[:2]
        analytic = float(np.sum(Fa * da.T) + np.sum(Fb * db.T))
    else:
        D = 2 * projector(scf.mo_coefficients, na)
        d = symmetric()
        up = solver._fock(D + eps * d)[1]
        down = solver._fock(D - eps * d)[1]
        analytic = float(np.sum(solver._fock(D)[0] * d.T))
    assert solver.hubbard_energy > 0.0
    assert (up - down) / (2 * eps) == pytest.approx(analytic, abs=1e-8)


def test_the_correction_pushes_the_d_shell_towards_integer_occupations():
    """FeH: with U the d occupations are closer to 0 and 1 than without."""
    def shell_deviation(**options):
        atoms = _run(_molecule("Fe", 1.6, (3.0, 0.0)), **options)
        solver_scf = atoms.calc.result.scf
        from mandacaru.algorithms.dft import kohn_sham_solver

        integrals = atoms.calc._solver._gradient_context["integrals"]
        na, nb = atoms.calc.result.num_particles
        integrals.hubbard_u = {"Fe": {2: 1.0}}               # to measure only
        solver = kohn_sham_solver(integrals, na + nb, "lda", spins=(na, nb))
        C, combined = solver.onsite
        onsite = combined.components["hubbard"]
        total = 0.0
        for coefficients, n in ((solver_scf.mo_coefficients_alpha, na),
                                (solver_scf.mo_coefficients_beta, nb)):
            D = coefficients[:, :n] @ coefficients[:, :n].T
            [occupation] = onsite.occupations(C.conj().T @ D @ C)
            total += float(np.trace(occupation - occupation @ occupation))
        return total

    plain = shell_deviation()
    corrected = shell_deviation(hubbard_u={"Fe": {"d": 4.0}})
    assert corrected < plain


def test_a_zero_u_changes_nothing():
    plain = _run(_molecule("Cu", 1.46)).get_potential_energy()
    zero = _run(_molecule("Cu", 1.46),
                hubbard_u={"Cu": {"d": 0.0}}).get_potential_energy()
    assert zero == plain


class TestRefusals:
    def test_a_bad_option_is_refused_when_the_calculator_is_built(self):
        from mandacaru import Mandacaru

        with pytest.raises(ValueError, match="hubbard_u"):
            Mandacaru(method="dft", hubbard_u={"Gd": 6.0})

    def test_a_method_without_a_kohn_sham_problem_does_not_take_it(self):
        from mandacaru import Mandacaru

        with pytest.raises(TypeError, match="hubbard_u"):
            Mandacaru(method="uhf", hubbard_u={"Gd": {"f": 6.0}})

    def test_an_element_that_is_not_there_is_refused(self):
        with pytest.raises(ValueError, match="not in the system"):
            _run(_molecule("Cu", 1.46), hubbard_u={"Gd": {"f": 6.0}})

    def test_a_hybrid_is_refused(self):
        with pytest.raises(NotImplementedError, match="hybrid"):
            _run(_molecule("Cu", 1.46), hubbard_u={"Cu": {"d": 4.0}},
                 xc="hse06")


# --------------------------------------------------------------------------- #
# The option on a crystal (one Gamma point of a molecule in a box).
# --------------------------------------------------------------------------- #

def _box(pbc, moment=None, **options):
    """CuH in a 9 Angstrom cell, run as a molecule or as a Gamma-point
    crystal."""
    from ase import Atoms
    from mandacaru import Mandacaru

    atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[9.0] * 3,
                  pbc=pbc)
    atoms.center()
    if moment is not None:
        atoms.set_initial_magnetic_moments([moment, 0.0])
    extra = ({"kpts": {"size": (1, 1, 1), "gamma": True},
              "smearing": {"method": "fermi-dirac", "width": 0.01}}
             if pbc else {"stability": False})
    atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.3,
                           trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           **extra, **options)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def box():
    """Energies at U = 0 and 4 eV of CuH as a molecule and as a crystal."""
    out = {}
    for label, pbc in (("molecule", False), ("crystal", True)):
        out[label, 0.0] = _box(pbc)
        out[label, 4.0] = _box(pbc, hubbard_u={"Cu": {"d": 4.0}})
    return out


class TestCrystal:
    def test_a_molecule_in_a_box_gets_the_molecule_s_correction(self, box):
        """The crystal and the molecular codes share no line of the on-site
        term's assembly: their energy shifts agree to the difference between
        a periodic and an isolated box (9 meV here)."""
        shift = {label: box[label, 4.0].get_potential_energy()
                 - box[label, 0.0].get_potential_energy()
                 for label in ("molecule", "crystal")}
        assert shift["molecule"] > 0.3
        assert shift["crystal"] == pytest.approx(shift["molecule"], abs=0.03)

    def test_the_correction_is_a_term_of_the_crystal_energy(self, box):
        terms = box["crystal", 4.0].calc.result.scf.terms
        assert terms["hubbard"] > 0.0
        assert "hubbard" not in box["crystal", 0.0].calc.result.scf.terms

    def test_the_spin_polarized_loop_reproduces_the_restricted_one(self, box):
        """A closed-shell molecule started with a moment settles at zero:
        the two crystal loops then hold the same functional."""
        polarized = _box(True, moment=1.0, hubbard_u={"Cu": {"d": 4.0}})
        assert polarized.get_potential_energy() == pytest.approx(
            box["crystal", 4.0].get_potential_energy(), abs=1e-4)

    def test_forces_and_stress_exist_and_answer_to_the_correction(self, box):
        corrected, plain = box["crystal", 4.0], box["crystal", 0.0]
        for atoms in (corrected, plain):
            assert np.all(np.isfinite(atoms.get_forces()))
            assert np.all(np.isfinite(atoms.get_stress()))
        assert abs(corrected.get_forces() - plain.get_forces()).max() > 0.05
        assert abs(corrected.get_stress() - plain.get_stress()).max() > 1e-4

    def test_the_carried_energy_at_zero_strain_is_the_scf_energy(self, box):
        """The stress differentiates the converged state carried to a
        strained cell; at no strain that energy, on-site term included, is
        the SCF's."""
        from mandacaru.algorithms import crystal_forces
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            pseudopotential_family, resolve_basis)

        atoms = box["crystal", 4.0]
        driver = atoms.calc.solver
        solver = driver._periodic_solver
        name, spec = resolve_basis(driver.basis)
        options = {k: v for k, v in dict(spec or {}).items() if k != "name"}
        carried = crystal_forces.carried_energy(
            solver, crystal_forces.kpoint_state(solver), atoms, options,
            pseudopotential_family(name).name, driver._kpts_spec(),
            np.eye(3), symmetry=False)
        assert carried == pytest.approx(atoms.calc.result.scf.energy,
                                        abs=1e-8)

    @pytest.mark.slow
    def test_the_force_change_the_correction_makes_is_the_energy_derivative(self):
        """F(U) - F(0) of the moved atom against central differences of the
        energies at the two U (the crystal's own force error at this grid, a
        few meV/Angstrom, is the same in both and cancels)."""
        from ase import Atoms
        from mandacaru import Mandacaru

        def force_and_numeric(**options):
            atoms = Atoms("CuH", positions=[[0.0, 0.0, 0.0], [0.05, 0.0, 1.55]],
                          cell=[8.0] * 3, pbc=True)
            atoms.center()
            atoms.calc = Mandacaru(
                method="dft", xc="lda", grid_spacing=0.3, trace=False,
                basis={"name": "PAW-LCAO", "size": "SZ"},
                kpts={"size": (1, 1, 1), "gamma": True},
                smearing={"method": "fermi-dirac", "width": 0.2}, **options)
            analytic = atoms.get_forces()[0, 2]

            def energy(step):
                moved = atoms.copy()
                moved.positions[0, 2] += step
                moved.calc = atoms.calc
                return moved.get_potential_energy()

            fine = -(energy(0.004) - energy(-0.004)) / 0.008
            coarse = -(energy(0.008) - energy(-0.008)) / 0.016
            return analytic, fine + (fine - coarse) / 3.0

        with_u = force_and_numeric(hubbard_u={"Cu": {"d": 4.0}})
        without = force_and_numeric()
        assert with_u[0] - without[0] > 0.1                  # a real change
        assert (with_u[0] - without[0]) == pytest.approx(
            with_u[1] - without[1], abs=5e-3)

    def test_a_crystal_is_built_on_the_whole_mesh(self, box):
        """No symmetrization of the projected density is needed there."""
        crystal = box["crystal", 4.0].calc._solver._gradient_context[
            "integrals"]
        assert crystal.symmetry is None


class TestReporting:
    @pytest.fixture(scope="class")
    def run(self):
        return _run(_molecule("Cu", 1.46), hubbard_u={"Cu": {"d": 4.0}})

    def test_the_energy_is_a_term_of_its_own(self, run):
        scf = run.calc.result.scf
        assert scf.hubbard_energy > 0.0
        plain = _run(_molecule("Cu", 1.46))
        assert plain.calc.result.scf.hubbard_energy == 0.0
        # xc_energy is the functional's alone: the correction is not in it
        assert scf.xc_energy == pytest.approx(
            plain.calc.result.scf.xc_energy, abs=0.5)

    def test_the_run_log_names_the_option_once_and_the_energy_once(self, run):
        setup = run.calc._solver._scf_setup_fields()
        assert setup["hubbard_u"] == "Cu d 4 eV (Dudarev, simplified)"
        summary = run.calc._solver._scf_summary_fields(run.calc.result)
        assert any(key.startswith("hubbard_energy_") for key in summary)
        assert not any(key.startswith("hubbard_u") for key in summary)

    def test_the_paper_is_cited_only_when_the_correction_ran(self, run):
        assert "Dudarev1998" in run.calc.citation_keys()
        plain = _run(_molecule("Cu", 1.46))
        assert "Dudarev1998" not in plain.calc.citation_keys()


# --------------------------------------------------------------------------- #
# Forces of a molecule.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("symbol, bond, moments", [
    ("Cu", 1.50, None),                 # restricted
    ("Fe", 1.65, (3.0, 0.0))])          # unrestricted
def test_the_forces_are_the_derivative_of_the_energy(symbol, bond, moments):
    """The unprojected analytic force against Richardson-extrapolated central
    differences of the energy on the frozen grid, with the correction on --
    the way the other force tests compare (the net force and torque the
    calculator projects out are not part of an energy derivative)."""
    from ase import Atoms
    from mandacaru import Mandacaru

    atoms = Atoms(f"{symbol}H", positions=[[0.0, 0.0, 0.0], [0.05, 0.0, bond]],
                  cell=[8.0] * 3)
    atoms.center()
    if moments is not None:
        atoms.set_initial_magnetic_moments(moments)
    atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.3,
                           trace=False, stability=False,
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           hubbard_u={symbol: {"d": 4.0}})
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


def test_the_correction_changes_the_force():
    """A guard on the test above: the term is not negligible."""
    from ase import Atoms
    from mandacaru import Mandacaru

    def force(**options):
        atoms = Atoms("CuH", positions=[[0.0, 0.0, 0.0], [0.05, 0.0, 1.5]],
                      cell=[8.0] * 3)
        atoms.center()
        atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.3,
                               trace=False, stability=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               **options)
        atoms.get_forces()
        return atoms.calc.force_result.unprojected[1, 2]

    assert abs(force(hubbard_u={"Cu": {"d": 4.0}}) - force()) > 0.1
