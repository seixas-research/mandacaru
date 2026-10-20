# -*- coding: utf-8 -*-
# file: test/algorithms/test_dft.py

# This code is part of Mandacaru.
# MIT License

"""Closed-shell Kohn-Sham DFT behind ``Mandacaru(method="dft")``."""

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.units import ANGSTROM_TO_BOHR, from_hartree, to_hartree

NAO_DZ = {"name": "NAO", "size": "DZ"}
#: The unconfined single-zeta PAW-LCAO basis: the first function is the
#: dataset's own bound smooth partial wave, the reference atom's solution.
PAW_SZ_FREE = {"name": "PAW-LCAO", "size": "SZ", "energy_shift": None}


def h2():
    """Small closed-shell molecule on a fixed real-space box."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6] * 3)


def _run(atoms, **options):
    atoms.calc = Mandacaru(method="dft", trace=False, **options)
    return atoms.get_potential_energy()


@pytest.fixture(scope="module",
                params=["lda", "pbe", "r2scan", "hse06", "pbe+rvv10",
                                 "r2scan+rvv10"])
def converged(request):
    """An H2 Kohn-Sham run in a basis with more than one function per atom."""
    atoms = h2()
    energy = _run(atoms, xc=request.param, grid_spacing=0.3, basis=NAO_DZ)
    return request.param, atoms, energy


class TestTheRun:
    def test_it_converges_and_reports_a_total_energy(self, converged):
        _xc, atoms, energy = converged
        result = atoms.calc.result
        assert result.success
        assert result.method == "dft"
        assert energy == pytest.approx(result.optimal_energy)
        assert result.scf.xc_energy < 0.0 < result.scf.hartree_energy
        assert atoms.calc.solver.ansatz is None

    def test_the_energy_is_stationary_in_the_density_matrix(self, converged):
        r"""``F`` is the derivative of ``E``: ``dE = sum F^T dD``.

        This is what makes the Kohn-Sham potential *the* potential of the
        functional; a wrong ``v_xc``, ``tau`` term or Hartree term breaks it
        while still converging.  The step is positive semidefinite (a meta-GGA
        is only defined for a non-negative ``tau``) and the difference
        one-sided, second order.
        """
        from mandacaru.algorithms import dft
        _xc, atoms, _energy = converged
        integrals = atoms.calc.solver._gradient_context["integrals"]
        scf = atoms.calc.result.scf
        # The solver of the run: the grid samples, and for a hybrid the
        # short-range exchange tensor (its nonlocal potential is in F too).
        solver = dft.kohn_sham_solver(integrals, 2, scf.functional)
        C = scf.mo_coefficients
        D = 2.0 * np.outer(C[:, 0], C[:, 0].conj())
        rng = np.random.default_rng(7)
        B = rng.normal(size=D.shape) + 1j * rng.normal(size=D.shape)
        delta = 1e-2 * (B @ B.conj().T)
        eps = 1e-5            # the meta-GGA's quotient carries 2.5e-5 at 1e-4
        F, e0, _h, _x = solver._fock(D)
        e1 = solver._fock(D + eps * delta)[1]
        e2 = solver._fock(D + 2 * eps * delta)[1]
        directional = float(np.real(np.sum(F.T * delta)))
        assert (-3 * e0 + 4 * e1 - e2) / (2 * eps) == pytest.approx(
            directional, rel=1e-5)

    def test_a_one_orbital_density_is_iso_orbital_for_the_meta_gga(self):
        r"""``tau_W = |grad rho|^2/8 rho`` equals ``tau`` for H2's single
        occupied orbital: the density gradient r2SCAN sees is the orbitals'
        product rule, so the iso-orbital indicator is zero to round-off
        wherever the density is not.  The grid's own derivative of the
        sampled density is not (K1: that mismatch put a micro-Hartree ripple
        on the energy)."""
        from mandacaru.algorithms import dft
        from mandacaru.integrals import exchange_correlation as xc_grid
        atoms = h2()
        _run(atoms, xc="r2scan", grid_spacing=0.3, basis=NAO_DZ)
        integrals = atoms.calc.solver._gradient_context["integrals"]
        scf = atoms.calc.result.scf
        solver = dft.kohn_sham_solver(integrals, 2, scf.functional)
        C = scf.mo_coefficients
        D = 2.0 * np.outer(C[:, 0], C[:, 0].conj())
        rho, tau = solver.density(D), solver.kinetic_energy_density(D)
        live = rho > 1e-8

        def mismatch(gradient):
            weizsaecker = np.sum(gradient * gradient, axis=0) / (
                8.0 * np.where(live, rho, 1.0))
            return np.max(np.abs(weizsaecker - tau)[live]) / np.max(tau)

        assert mismatch(solver.density_gradient(D)) < 1e-12
        assert mismatch(xc_grid.gradient(solver.grid, rho)) > 1e-6

    def test_the_eigenvalues_are_ordered_and_the_gap_is_positive(self, converged):
        _xc, atoms, _energy = converged
        scf = atoms.calc.result.scf
        assert np.all(np.diff(scf.mo_energies) >= 0)
        assert scf.homo_lumo_gap > 0.0

    def test_the_kohn_sham_orbitals_export_a_quantum_problem(self, converged):
        """The exported Hamiltonian is written in the Kohn-Sham orbitals.

        Its reference determinant is the Kohn-Sham determinant, whose energy in
        the many-body Hamiltonian is the Hartree-Fock functional of those
        orbitals -- above the RHF minimum of the same basis -- and is what
        ``reference_energy`` reports.
        """
        _xc, atoms, _energy = converged
        result = atoms.calc.result
        state = result.reference_state()
        operator = result.qubit_hamiltonian()
        expectation = from_hartree(
            np.vdot(state, operator.to_sparse_matrix() @ state).real, "eV")
        assert expectation == pytest.approx(result.reference_energy, abs=1e-6)
        rhf = h2()
        rhf.calc = Mandacaru(method="rhf", grid_spacing=0.3, trace=False, basis=NAO_DZ)
        assert expectation >= rhf.get_potential_energy() - 1e-6
        post = Mandacaru(method="adapt-vqe", trace=False,
                         **result.as_quantum_problem())
        assert post.run().reference_energy == pytest.approx(
            result.reference_energy, abs=1e-7)

    def test_it_cites_kohn_sham_and_the_functional(self, converged):
        xc, atoms, _energy = converged
        keys = atoms.calc.citation_keys()
        assert {"HohenbergKohn1964", "KohnSham1965"} <= set(keys)
        expected = {"lda": {"PerdewZunger1981"}, "pbe": {"PBE1996"},
                    "r2scan": {"Furness2020"},
                    "hse06": {"Heyd2003", "Krukau2006", "PBE1996"},
                    "pbe+rvv10": {"PBE1996", "Vydrov2010", "Sabatini2013",
                                  "RomanPerez2009"},
                    "r2scan+rvv10": {"Furness2020", "Vydrov2010",
                                     "Sabatini2013", "RomanPerez2009",
                                     "Ning2022"}}[xc]
        assert expected <= set(keys)
        assert "Paier2005" not in keys          # no augmentation spheres

    def test_a_paw_hybrid_cites_the_one_center_exchange(self):
        atoms = h2()
        _run(atoms, xc="hse06", grid_spacing=0.35, basis={"name": "PAW-LCAO",
                                                "size": "SZ"})
        assert "Paier2005" in atoms.calc.citation_keys()


class TestConvergence:
    def test_stretched_h2_converges_with_every_functional(self):
        """A noise-level energy rise must not switch on the level shift.

        At 4 Angstrom the energy settles to 1e-9 Ha within a few iterations;
        a shift triggered by a 1e-10 Ha rise used to freeze the density change
        just above the tolerance for all 200 iterations.
        """
        for xc in ("lda", "pbe", "r2scan"):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 4.0]],
                          cell=[8, 8, 12])
            atoms.center()
            _run(atoms, xc=xc, grid_spacing=0.25, basis={"name": "PAW-LCAO"})
            assert atoms.calc.result.scf.n_iterations < 30

    def test_the_log_holds_every_iteration(self, tmp_path):
        """``[SCF ITERATIONS]``: one row per iteration, ending where the
        convergence test stopped and on the energy the summary reports."""
        from mandacaru.utils.logging import parse_output, reset_log
        path = str(tmp_path / "dft.txt")
        atoms = h2()
        atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.35, txt=path,
                               trace=False)
        energy = atoms.get_potential_energy()
        reset_log(path)
        parsed = parse_output(path)
        rows = parsed["scf_iterations"]
        scf = atoms.calc.result.scf
        assert [r["iter"] for r in rows] == list(range(1, scf.n_iterations
                                                       + 1))
        assert rows[0]["dE_eV"] is None
        for before, after in zip(rows, rows[1:]):
            assert after["dE_eV"] == pytest.approx(
                after["energy_eV"] - before["energy_eV"], abs=2e-10)
            assert after["time_s"] >= before["time_s"]
        assert rows[-1]["residual"] < 1e-8           # what stopped it
        # The last row is the converged electronic energy; the summary adds
        # the constants, so the two differ by exactly those.
        constant = float(parsed["summary"]["optimal_energy_eV"]) - \
            rows[-1]["energy_eV"]
        assert energy == pytest.approx(rows[-1]["energy_eV"] + constant,
                                       abs=1e-9)

    def test_an_r2scan_scf_does_not_stall(self):
        """DIIS restarts: this geometry oscillated at 1e-8 Ha for 200
        iterations with an uninterrupted history (HISTORY.md, 2026-10-02)."""
        from ase.build import molecule
        atoms = molecule("H2O")
        atoms.center(vacuum=3.0)
        atoms.positions[1] += [0.03, -0.02, 0.04]
        atoms.positions[0, 2] -= 0.004
        _run(atoms, xc="r2scan", grid_spacing=0.25, basis={"name": "PAW-LCAO",
                                                "size": "DZP"})
        assert atoms.calc.result.success


def test_a_degenerate_shell_at_the_fermi_level_is_filled_equally():
    """Aufbau unless the last filled and the first empty level are closer
    than the window; then their degenerate shell shares its electrons."""
    from mandacaru.algorithms.dft import shell_occupations
    assert shell_occupations([-2.0, -1.0, 0.0], 2) is None
    assert shell_occupations([-1.0, -0.5, -0.5, -0.5, 1.0], 2) == \
        pytest.approx([1, 1 / 3, 1 / 3, 1 / 3, 0])
    # A full degenerate shell is aufbau's; a slightly split one is shared.
    assert shell_occupations([-1.0, -0.5, -0.5 + 2e-4, 0.3], 3) is None
    assert shell_occupations([-1.0, -0.5, -0.5 + 2e-4, 0.3], 2) == \
        pytest.approx([1, 0.5, 0.5, 0])
    # A small real gap (stretched H2's 5.5 mHa) stays integer.
    assert shell_occupations([-0.2, -0.2 + 5.5e-3], 1) is None


class TestFunctionals:
    def test_the_gradient_corrections_lower_the_energy_of_h2(self):
        energies = {xc: _run(h2(), xc=xc, grid_spacing=0.35)
                    for xc in ("lda", "pbe", "r2scan")}
        assert energies["pbe"] < energies["lda"]
        assert energies["r2scan"] < energies["lda"]


class TestPAWLCAO:
    def test_the_reference_atom_recovers_its_valence_energy(self):
        """Be in its own smooth partial wave: the dataset's reference energy.

        With the unconfined single-zeta basis the Kohn-Sham solution *is* the
        reference atom, so the energy -- core correction and frozen one-center
        constant included -- is the dataset's ``reference_valence`` up to the
        grid error (about 1 mHa at grid_spacing = 0.2 Angstrom).
        """
        atoms = Atoms("Be", positions=[[4.0] * 3], cell=[8.0] * 3)
        energy = to_hartree(_run(atoms, xc="lda", grid_spacing=0.2, basis=PAW_SZ_FREE),
                            "eV")
        integrals = atoms.calc.solver._gradient_context["integrals"]
        dataset = integrals.pseudopotentials[0]
        assert dataset.nlcc.get("applied")
        assert atoms.calc.result.scf.core_correction_energy != 0.0
        assert energy == pytest.approx(
            dataset.energies["reference_valence"], abs=3e-3)

    def test_each_functional_on_its_own_set_is_silent(self):
        """By default LDA reads ``lda-sr`` and PBE and r2SCAN read
        ``pbe-sr`` (r2SCAN has no datasets of its own), with no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for xc in ("lda", "pbe", "r2scan"):
                atoms = h2()
                energy = _run(atoms, xc=xc, grid_spacing=0.3, basis={"name": "PAW-LCAO"})
                assert np.isfinite(energy)
                datasets = atoms.calc.solver._gradient_context[
                    "integrals"].pseudopotentials
                assert {d.xc for d in datasets} == {
                    "lda" if xc == "lda" else "pbe"}

    def test_another_set_warns_and_names_the_functional_s_own(self):
        # The warning names the folder of the PBE datasets.
        with pytest.warns(RuntimeWarning, match="different functionals; "
                          r"Mandacaru\(paw_dataset='pbe-sr'\)"):
            _run(h2(), xc="r2scan", grid_spacing=0.3, basis={"name": "PAW-LCAO"},
                 paw_dataset="lda-sr")
        with pytest.warns(RuntimeWarning, match="different functionals; "
                          r"Mandacaru\(paw_dataset='lda-sr'\)"):
            _run(h2(), xc="lda", grid_spacing=0.3, basis={"name": "PAW-LCAO"},
                 paw_dataset="pbe-sr")


def lih():
    """LiH off axis: lithium carries a core correction, and every PAW-LCAO
    term (projectors, compensation charges, split local potential) is live."""
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0.1, 0, 1.55]],
                  cell=[8, 8, 9])
    atoms.center()
    return atoms


@pytest.fixture(scope="module", params=[("lda", None), ("pbe", "d4")],
                ids=["lda", "pbe+d4"])
def lih_forces(request):
    xc, vdw = request.param
    if vdw == "d4":
        pytest.importorskip("dftd4")
    atoms = lih()
    atoms.calc = Mandacaru(method="dft", xc=xc, vdw=vdw, grid_spacing=0.25,
                           basis={"name": "PAW-LCAO", "size": "DZP"},
                           trace=False)
    forces = atoms.get_forces()
    return atoms, forces, atoms.calc.force_result


@pytest.mark.slow
@pytest.mark.parametrize("semilocal", ["pbe", "r2scan"])
def test_the_rvv10_force_is_the_derivative_of_its_energy(semilocal):
    """rVV10's own part of the force: (F[X+rVV10] - F[X]) against the
    central difference of the energy difference.  The difference isolates it
    from the semilocal functional's molecular force error; the part is one
    component, along the bond, where it is largest (two SCFs and eight
    energies are most of the time budget).  Asked as ``vdw="rvv10"``."""
    rvv10 = f"{semilocal}+rvv10"

    def run(vdw):
        atoms = lih()
        atoms.calc = Mandacaru(method="dft", xc=semilocal, vdw=vdw,
                               grid_spacing=0.25, trace=False,
                               basis={"name": "PAW-LCAO", "size": "DZP"})
        atoms.get_forces()
        # Kept now: moving the atoms below resets the calculator's result.
        return atoms, atoms.calc.force_result.unprojected.copy()

    runs = {semilocal: run(None), rvv10: run("rvv10")}
    assert runs[rvv10][0].calc.solver.xc == rvv10
    pair = {xc: atoms for xc, (atoms, _gradient) in runs.items()}

    def energy(xc, atom, k, step):
        moved = pair[xc].copy()
        moved.positions[atom, k] += step
        moved.calc = pair[xc].calc
        return moved.get_potential_energy()

    for atom, k in ((1, 2),):
        def part(step):
            return -((energy(rvv10, atom, k, step)
                      - energy(rvv10, atom, k, -step))
                     - (energy(semilocal, atom, k, step)
                        - energy(semilocal, atom, k, -step))) / (2 * step)
        coarse, fine = part(0.004), part(0.002)
        numerical = fine + (fine - coarse) / 3.0
        analytic = runs[rvv10][1][atom, k] - runs[semilocal][1][atom, k]
        assert analytic == pytest.approx(numerical, abs=1e-5)


class TestTheVdwOption:
    """``vdw=`` picks the van der Waals treatment; the constructor resolves
    and validates it before anything is built."""

    SZ = {"name": "PAW-LCAO", "size": "SZ"}

    def _solver(self, **options):
        return Mandacaru(method="dft", basis=self.SZ, **options).solver

    @pytest.mark.parametrize("xc, vdw, resolved, treatment", [
        ("pbe", "rvv10", "pbe+rvv10", "rvv10"),
        ("PBE", "RVV10", "pbe+rvv10", "rvv10"),
        ("r2scan", "rvv10", "r2scan+rvv10", "rvv10"),
        ("pbe+rvv10", None, "pbe+rvv10", "rvv10"),
        ("r2scan+rvv10", "rvv10", "r2scan+rvv10", "rvv10"),
        ("pbe", None, "pbe", None),
        ("pbe", "d4", "pbe", "d4"),
        ("r2scan", "d4", "r2scan", "d4"),
        ("hse06", "d4", "hse06", "d4")])
    def test_the_functional_and_the_treatment_are_resolved(
            self, xc, vdw, resolved, treatment):
        solver = self._solver(xc=xc, vdw=vdw)
        assert (solver.xc, solver.vdw) == (resolved, treatment)

    @pytest.mark.parametrize("xc", ["lda", "hse06"])
    def test_rvv10_pairs_with_pbe_and_r2scan_only(self, xc):
        with pytest.raises(ValueError, match="pairs with xc='pbe'"):
            self._solver(xc=xc, vdw="rvv10")

    @pytest.mark.parametrize("xc", ["pbe+rvv10", "r2scan+rvv10"])
    def test_rvv10_refuses_d4(self, xc):
        """The rVV10 forms carry their van der Waals attraction; D4 on top
        would count it twice."""
        with pytest.raises(ValueError, match="count it twice"):
            self._solver(xc=xc, vdw="d4")

    def test_d4_without_parameters_for_the_functional_is_refused(self):
        with pytest.raises(ValueError, match="no parameters"):
            self._solver(xc="lda", vdw="d4")

    def test_an_unknown_vdw_is_refused(self):
        with pytest.raises(ValueError, match="unknown vdw"):
            self._solver(xc="pbe", vdw="d3")

    @pytest.mark.parametrize("retired", ["r2scan-rvv10", "r2scan_rvv10"])
    def test_the_retired_functional_spelling_is_refused(self, retired):
        with pytest.raises(ValueError, match="unknown exchange-correlation"):
            self._solver(xc=retired)

    def test_the_retired_dispersion_option_is_refused(self):
        with pytest.raises((TypeError, ValueError)):
            Mandacaru(method="dft", xc="pbe", dispersion="d4",
                      basis=self.SZ)

    def test_the_rvv10_forms_read_the_pbe_dataset_set(self):
        for xc, vdw in (("pbe+rvv10", None), ("pbe", "rvv10"),
                        ("r2scan", "rvv10")):
            calc = Mandacaru(method="dft", xc=xc, vdw=vdw, basis=self.SZ)
            assert calc.library_folder == "pbe-sr"

    def test_d4_is_dispatched_to_the_energy_and_the_forces(self):
        """``vdw="d4"`` adds the D4 energy and gradient to the PBE run."""
        pytest.importorskip("dftd4")
        from mandacaru.algorithms.dft import (d4_dispersion_energy,
                                              d4_dispersion_gradient)
        from mandacaru.algorithms.forces import HA_BOHR_TO_EV_ANGSTROM
        atoms = lih()
        atoms.calc = Mandacaru(method="dft", xc="pbe", vdw="d4",
                               grid_spacing=0.25, trace=False,
                               basis={"name": "PAW-LCAO", "size": "DZP"})
        atoms.get_forces()
        result = atoms.calc.result
        assert result.scf.dispersion_energy == pytest.approx(
            d4_dispersion_energy(atoms, "pbe"), abs=1e-12)
        gradient = atoms.calc.force_result.details["dispersion_gradient"]
        assert np.allclose(gradient, d4_dispersion_gradient(atoms, "pbe")
                           * HA_BOHR_TO_EV_ANGSTROM, atol=1e-8)


class TestForces:
    def test_they_are_the_derivative_of_the_energy(self, lih_forces):
        """Richardson-extrapolated central differences on the frozen grid.

        The raw gradient is compared: the finite difference is of the
        discretized energy, which is not translation invariant.
        """
        atoms, _forces, result = lih_forces

        def energy(atom, k, step):
            moved = atoms.copy()
            moved.positions[atom, k] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        for atom, k in ((1, 2), (1, 0)):
            coarse, fine = (-(energy(atom, k, s) - energy(atom, k, -s)) / (2 * s)
                            for s in (0.004, 0.002))
            numerical = fine + (fine - coarse) / 3.0
            assert result.unprojected[atom, k] == pytest.approx(numerical,
                                                                abs=1e-4)

    def test_the_breakdown_and_the_dispersion_add_up(self, lih_forces):
        _atoms, forces, result = lih_forces
        assert np.allclose(result.hellmann_feynman + result.pulay,
                           result.gradient)
        assert result.details["method"] == "kohn-sham"
        assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-10)
        if "dispersion_gradient" in result.details:
            assert np.allclose(result.details["dispersion_gradient"].sum(axis=0),
                               0.0, atol=1e-10)

    def test_a_semilocal_gradient_never_forms_the_two_electron_tensor(
            self, lih_forces):
        """Hartree enters through its potential and the compensation
        charges' scalars: the SCF's Poisson route is all the gradient reads."""
        atoms, _forces, _result = lih_forces
        integrals = atoms.calc.solver._gradient_context["integrals"]
        assert integrals._eri_ao is None

    @pytest.mark.parametrize("xc, spacing, size", [
        ("pbe", 0.25, "DZP"), ("r2scan", 0.3, "SZP")])
    def test_it_is_the_gradient_through_the_tensor(self, xc, spacing, size):
        """The linearized Hartree term against ``hartree="tensor"``, which
        differentiates the dense tensor and its pair-density derivatives:
        the same forces, up to the dense route's finite-difference step.
        r2SCAN adds the moving core's share of the vector field of its
        density gradient on either route."""
        gradients = {}
        for route in ("poisson", "tensor"):
            atoms = lih()
            atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=spacing,
                                   hartree=route, trace=False,
                                   basis={"name": "PAW-LCAO", "size": size})
            atoms.get_forces()
            gradients[route] = atoms.calc.force_result
        linearized, dense = gradients["poisson"], gradients["tensor"]
        assert np.allclose(linearized.gradient, dense.gradient, atol=1e-5)
        assert np.allclose(linearized.hellmann_feynman,
                           dense.hellmann_feynman, atol=1e-5)
        assert linearized.details["energy_hartree"] == pytest.approx(
            dense.details["energy_hartree"], abs=1e-9)

    @pytest.mark.parametrize("kinetic", ["spectral", "fd"])
    def test_a_meta_gga_force_is_the_derivative_of_its_energy(self, kinetic):
        """r2SCAN: Richardson-extrapolated central differences on the
        frozen grid, to the bound of the LDA and PBE checks above.  Its
        energy surface used to be too rough for it (K1: 1.4 uHa, then the
        spectral gradient's ringing); the analytic force and the difference
        quotient now agree to 4e-5 eV/Angstrom, as PBE's do."""
        atoms = h2()
        atoms.calc = Mandacaru(method="dft", xc="r2scan", grid_spacing=0.3,
                               basis=NAO_DZ, kinetic=kinetic, trace=False)
        forces = atoms.get_forces()
        result = atoms.calc.force_result
        assert np.all(np.isfinite(forces))
        assert forces[1, 2] == pytest.approx(-forces[0, 2])

        def energy(step):
            moved = atoms.copy()
            moved.positions[1, 2] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        coarse, fine = (-(energy(s) - energy(-s)) / (2 * s)
                        for s in (0.004, 0.002))
        numerical = fine + (fine - coarse) / 3.0
        assert abs(numerical) > 1.0                  # a real force
        assert result.unprojected[1, 2] == pytest.approx(numerical, abs=1e-4)

    def test_the_force_on_water_s_oxygen_is_the_derivative_of_its_energy(self):
        """r2SCAN on water (PAW-LCAO SZ, ``grid_spacing`` 0.3), oxygen's z
        component: 4.5e-6 eV/Angstrom.  With the density gradient taken from
        the grid while tau comes from the orbitals' (the stencil alone) it was
        5.3e-4, the micro-Hartree roughness of the energy surface (K1) that
        the finite difference saw and the analytic force did not."""
        from ase.build import molecule

        atoms = molecule("H2O")
        atoms.center(vacuum=4.0)
        atoms.calc = Mandacaru(method="dft", xc="r2scan", grid_spacing=0.3,
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               trace=False)
        atoms.get_forces()
        result = atoms.calc.force_result

        def energy(step):
            moved = atoms.copy()
            moved.positions[0, 2] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        coarse, fine = (-(energy(s) - energy(-s)) / (2 * s)
                        for s in (0.004, 0.002))
        numerical = fine + (fine - coarse) / 3.0
        assert abs(numerical) > 1.0                  # a real force
        assert result.unprojected[0, 2] == pytest.approx(numerical, abs=1e-4)


class TestTheEnergySurface:
    """The energy of a molecule moved a few thousandths of an Angstrom is a
    smooth function of the displacement (K1)."""

    @staticmethod
    def _roughness(xc):
        """RMS residual (micro-Hartree) of a cubic fit to the energy of
        water with its oxygen moved along z, on a grid that follows the
        molecule: 7 points over +-0.0045 Angstrom, SZ, ``grid_spacing`` 0.3."""
        from ase.build import molecule

        base = molecule("H2O")
        base.center(vacuum=4.0)
        steps = np.arange(-3, 4) * 0.0015
        energies = []
        for step in steps:
            atoms = base.copy()
            atoms.positions[0, 2] += step
            atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.3,
                                   basis={"name": "PAW-LCAO", "size": "SZ"},
                                   trace=False)
            energies.append(to_hartree(atoms.get_potential_energy(), "eV"))
        energies = np.array(energies) * 1e6
        fit = np.polyval(np.polyfit(steps, energies, 3), steps)
        return float(np.sqrt(np.mean((energies - fit) ** 2)))

    def test_r2scan_is_as_smooth_as_pbe(self):
        """Measured (micro-Hartree rms): r2SCAN 0.0003, PBE 0.0001.  Before
        the density gradient came from the orbitals' product rule (the
        grid's derivative of the sampled density, while tau is the
        orbitals') r2SCAN gave 0.016, and with the spectral density
        gradient of v26.10.16 and earlier 0.49: the bounds sit above both
        controls and below both of those."""
        assert self._roughness("pbe") < 0.002
        assert self._roughness("r2scan") < 0.005


class TestForcesAfterTheEnergy:
    """ASE asks for the forces of a geometry whose energy it already has:
    the converged state serves them, a new geometry or a changed setting
    converges its own."""

    @staticmethod
    def _calculator(monkeypatch, calls, **options):
        calc = Mandacaru(method="dft", xc=options.pop("xc", "lda"),
                         grid_spacing=0.35, basis=NAO_DZ, trace=False,
                         **options)
        original = calc._single_point

        def counting(solver, atoms):
            calls.append(solver)
            return original(solver, atoms)

        monkeypatch.setattr(calc, "_single_point", counting)
        return calc

    def test_the_forces_do_not_repeat_the_scf(self, monkeypatch):
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(monkeypatch, calls)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        assert len(calls) == 1
        assert atoms.calc.solver is calls[0]
        assert atoms.get_potential_energy() == energy
        fresh = h2()
        fresh.calc = self._calculator(monkeypatch, calls)
        assert np.allclose(forces, fresh.get_forces(), atol=1e-9)
        assert len(calls) == 2                    # the fresh calculator's own

    def test_a_moved_atom_converges_again(self, monkeypatch):
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(monkeypatch, calls)
        atoms.get_potential_energy()
        before = atoms.get_forces().copy()
        atoms.positions[1, 2] += 0.05
        atoms.get_forces()
        assert len(calls) == 2
        assert not np.allclose(before, atoms.get_forces())
        atoms.get_potential_energy()
        assert len(calls) == 2

    def test_forces_first_then_energy_run_once(self, monkeypatch):
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(monkeypatch, calls)
        atoms.get_forces()
        atoms.get_potential_energy()
        assert len(calls) == 1

    @pytest.mark.parametrize("setting, value", [("xc", "pbe"), ("kinetic", "spectral")])
    def test_a_changed_setting_converges_again(self, monkeypatch, setting,
                                               value):
        """Assigning a setting does not tell ASE anything changed; the
        forces must not come from the state of the old one."""
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(monkeypatch, calls)
        atoms.get_potential_energy()
        atoms.calc.solver_kwargs[setting] = value
        forces = atoms.get_forces()
        assert len(calls) == 2
        reference = h2()
        reference.calc = self._calculator(monkeypatch, [], **{setting: value})
        assert np.allclose(forces, reference.get_forces(), atol=1e-9)

    def test_cleared_results_never_serve_forces(self, monkeypatch):
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(monkeypatch, calls)
        atoms.get_potential_energy()
        atoms.calc.reset()
        atoms.get_forces()
        assert len(calls) == 2
        atoms.calc.results.clear()
        atoms.get_forces()
        assert len(calls) == 3


class TestDispersion:
    def test_d4_adds_its_energy_to_the_total(self):
        pytest.importorskip("dftd4")
        from mandacaru.algorithms.dft import d4_dispersion_energy
        bare = h2()
        e_bare = _run(bare, xc="pbe", grid_spacing=0.35)
        dressed = h2()
        e_d4 = _run(dressed, xc="pbe", grid_spacing=0.35, vdw="d4")
        d4 = d4_dispersion_energy(dressed, "pbe")
        assert d4 < 0.0
        assert dressed.calc.result.scf.dispersion_energy == pytest.approx(d4)
        assert e_d4 - e_bare == pytest.approx(from_hartree(d4, "eV"), abs=1e-8)
        assert "Caldeweyher2019" in dressed.calc.citation_keys()

    def test_periodic_d4_tends_to_the_molecular_one(self):
        """The lattice sum of a molecule in a growing periodic box goes to
        its molecular D4 (-75 / -2.4 / 0.002 uHa at 12 / 20 / 45 A)."""
        pytest.importorskip("dftd4")
        from ase.build import molecule

        from mandacaru.algorithms.dft import d4_dispersion_energy
        isolated = molecule("C6H6")
        alone = d4_dispersion_energy(isolated, "pbe")
        boxed = isolated.copy()
        boxed.set_cell([45.0] * 3)
        boxed.center()
        boxed.pbc = True
        assert d4_dispersion_energy(boxed, "pbe") == pytest.approx(alone,
                                                                   abs=1e-8)
        boxed.set_cell([12.0] * 3)
        boxed.center()
        assert d4_dispersion_energy(boxed, "pbe") < alone - 1e-5

    def test_periodic_d4_gradient_and_virial_are_its_derivatives(self):
        """Distorted graphite: dftd4's gradient and virial against central
        differences of its lattice sum, in atoms and in a symmetric strain
        (the virial over the volume is the stress, ASE's sign)."""
        pytest.importorskip("dftd4")
        from mandacaru.algorithms.dft import d4_dispersion
        from mandacaru.units import ANGSTROM_TO_BOHR

        a, c = 2.46, 6.7
        graphite = Atoms(
            "C4", scaled_positions=[[0, 0, 0.25], [0, 0, 0.75],
                                    [1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]],
            cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]],
            pbc=True)
        graphite.positions[1, 2] += 0.07
        graphite.positions[2, 0] += 0.05
        result = d4_dispersion(graphite, "pbe", grad=True)
        step = 1e-4
        for atom, axis in ((1, 2), (2, 0)):
            energies = []
            for sign in (1, -1):
                moved = graphite.copy()
                moved.positions[atom, axis] += sign * step
                energies.append(d4_dispersion(moved, "pbe")["energy"])
            slope = (energies[0] - energies[1]) / (2 * step * ANGSTROM_TO_BOHR)
            assert result["gradient"][atom, axis] == pytest.approx(
                slope, rel=1e-5, abs=1e-11)
        volume = abs(np.linalg.det(graphite.get_cell())) * ANGSTROM_TO_BOHR ** 3
        for a_, b_ in ((0, 0), (2, 2), (0, 1)):
            energies = []
            for sign in (1, -1):
                strain = np.zeros((3, 3))
                strain[a_, b_] += 0.5 * sign * 1e-5
                strain[b_, a_] += 0.5 * sign * 1e-5
                strained = graphite.copy()
                strained.set_cell(graphite.get_cell() @ (np.eye(3) + strain),
                                  scale_atoms=True)
                energies.append(d4_dispersion(strained, "pbe")["energy"])
            stress = (energies[0] - energies[1]) / 2e-5 / volume
            assert result["virial"][a_, b_] / volume == pytest.approx(
                stress, rel=1e-5, abs=1e-12)

    @staticmethod
    def _distorted_silicon(**options):
        from ase.build import bulk

        atoms = bulk("Si", "diamond", a=5.43)
        atoms.positions[1] += [0.03, -0.02, 0.01]
        atoms.calc = Mandacaru(method="dft", xc="pbe", grid_spacing=0.35, trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               kpts={"size": (2, 2, 2), "gamma": True},
                               **options)
        return atoms

    def test_a_crystal_carries_d4_in_its_energy(self):
        """Distorted Si: PBE+D4 minus PBE is the D4 lattice sum, and the
        run log reports it."""
        pytest.importorskip("dftd4")
        from mandacaru.algorithms.dft import d4_dispersion

        bare = self._distorted_silicon()
        dressed = self._distorted_silicon(vdw="d4")
        difference = (dressed.get_potential_energy()
                      - bare.get_potential_energy())
        d4 = d4_dispersion(dressed, "pbe")["energy"]
        assert d4 < 0.0
        assert difference == pytest.approx(from_hartree(d4, "eV"), abs=1e-8)
        fields = dressed.calc.solver._scf_summary_fields(dressed.calc.result)
        assert any(key.startswith("dispersion_energy") for key in fields)
        assert "Caldeweyher2019" in dressed.calc.citation_keys()

    @pytest.mark.slow
    def test_a_crystal_carries_d4_in_its_forces(self):
        """The same Si: the force difference is minus dftd4's gradient
        (1.7e-12 eV/A when measured)."""
        pytest.importorskip("dftd4")
        from mandacaru.algorithms.dft import d4_dispersion

        bare = self._distorted_silicon()
        dressed = self._distorted_silicon(vdw="d4")
        difference = dressed.get_forces() - bare.get_forces()
        gradient = d4_dispersion(dressed, "pbe", grad=True)["gradient"]
        expected = -from_hartree(1.0, "eV") * ANGSTROM_TO_BOHR * gradient
        assert np.allclose(difference, expected, atol=1e-8)

    @pytest.mark.slow
    def test_a_crystal_carries_d4_in_its_stress(self):
        """The same Si: the stress difference is dftd4's virial over the
        volume (1e-18 eV/A^3 when measured)."""
        pytest.importorskip("dftd4")
        from ase.build import bulk

        from mandacaru.algorithms.dft import d4_dispersion

        stresses = []
        for options in ({}, {"vdw": "d4"}):
            atoms = bulk("Si", "diamond", a=5.43)
            atoms.calc = Mandacaru(method="dft", xc="pbe", grid_spacing=0.35, trace=False,
                                   basis={"name": "PAW-LCAO", "size": "SZ"},
                                   kpts={"size": (2, 2, 2), "gamma": True},
                                   **options)
            atoms.get_potential_energy()
            stresses.append(atoms.get_stress(voigt=False))
        volume = atoms.get_volume() * ANGSTROM_TO_BOHR ** 3
        expected = (d4_dispersion(atoms, "pbe", grad=True)["virial"] / volume
                    * from_hartree(1.0, "eV") * ANGSTROM_TO_BOHR ** 3)
        assert np.allclose(stresses[1] - stresses[0], expected, atol=1e-10)

    @pytest.mark.slow
    def test_a_crystal_with_ghosts_disperses_its_real_atoms(self):
        """A counterpoise fragment of a crystal: D4 is the lattice sum of
        the real atoms in the same cell, the ghost contributing nothing."""
        pytest.importorskip("dftd4")
        from mandacaru.algorithms.dft import d4_dispersion_energy

        bare = self._distorted_silicon(ghosts=[1])
        dressed = self._distorted_silicon(ghosts=[1], vdw="d4")
        difference = (dressed.get_potential_energy()
                      - bare.get_potential_energy())
        real = d4_dispersion_energy(dressed[[0]], "pbe")
        assert difference == pytest.approx(from_hartree(real, "eV"), abs=1e-8)
        assert real != pytest.approx(d4_dispersion_energy(dressed, "pbe"))


class TestOptions:
    def test_the_device_is_ignored(self):
        atoms = h2()
        energy = _run(atoms, grid_spacing=0.35, device="AER_simulator")
        assert np.isfinite(energy)

    def test_an_unknown_functional_is_refused_by_the_constructor(self):
        with pytest.raises(ValueError, match="unknown exchange-correlation"):
            Mandacaru(method="dft", xc="b3lyp")

    def test_a_hydrogen_atom_runs_fully_polarized(self):
        """One electron: an empty spin-down channel everywhere -- the
        full-polarization limit of the spin functionals, end to end."""
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[6] * 3)
        atoms.calc = Mandacaru(method="dft", grid_spacing=0.35, trace=False)
        energy = atoms.get_potential_energy()
        scf = atoms.calc.result.scf
        assert np.isfinite(energy) and atoms.calc.result.success
        assert (scf.n_alpha, scf.n_beta) == (1, 0)
        assert atoms.calc.get_total_magnetic_moment() == pytest.approx(1.0)

    def test_a_measurement_provider_is_refused(self):
        with pytest.raises(ValueError, match="no quantum state"):
            Mandacaru(method="dft", measurement_provider=object())


@pytest.fixture(scope="module")
def water_szp():
    """H2O in a small PAW-LCAO basis with polarization: s, p and d shells."""
    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.center(vacuum=2.5)
    _run(atoms, xc="lda", grid_spacing=0.3, basis={"name": "PAW-LCAO", "size": "SZP"})
    return atoms


class TestTheMolecularSpectrum:
    def test_the_dos_holds_every_level_and_the_electrons(self, water_szp):
        calc = water_szp.calc
        energies, dos = calc.dos(width=0.1)
        levels = calc.get_eigenvalues()
        assert np.trapezoid(dos, energies) == pytest.approx(2 * len(levels),
                                                            rel=1e-6)
        mu = calc.get_fermi_level()
        assert levels[3] < mu < levels[4]                  # 8 valence electrons
        below = energies < mu
        assert np.trapezoid(dos[below], energies[below]) == pytest.approx(
            8.0, abs=1e-6)

    def test_the_pdos_splits_the_dos_by_atom_and_shell(self, water_szp):
        calc = water_szp.calc
        _energies, dos = calc.dos(width=0.1)
        _energies, pdos = calc.pdos(width=0.1)
        assert np.abs(sum(pdos.values()) - dos).max() < 1e-8 * dos.max()
        assert (0, 2) in pdos and (1, 1) in pdos       # O d and H p shells
        assert all(np.all(v >= -1e-12) for v in pdos.values())

    def test_a_molecule_has_no_band_structure_or_k_mesh(self, water_szp):
        calc = water_szp.calc
        with pytest.raises(NotImplementedError, match="periodic"):
            calc.band_structure()
        with pytest.raises(ValueError, match="Brillouin"):
            calc.dos(kpts=(2, 2, 2))
        assert calc.get_ibz_k_points().shape == (1, 3)

    def test_a_molecule_refuses_the_tetrahedron_method(self, water_szp):
        """Discrete levels have no Brillouin zone to cut into tetrahedra."""
        calc = water_szp.calc
        for spectrum in (calc.dos, calc.pdos):
            with pytest.raises(NotImplementedError,
                               match="integration='gaussian'"):
                spectrum(integration="tetrahedron")

    def test_the_spectrum_needs_a_run(self):
        calc = Mandacaru(method="dft", trace=False)
        with pytest.raises(ValueError, match="get_potential_energy"):
            calc.dos()


def _oxygen(moments):
    from ase.build import molecule
    atoms = molecule("O2")
    atoms.center(vacuum=2.5)
    atoms.set_initial_magnetic_moments(moments)
    return atoms


@pytest.fixture(scope="module")
def oxygen_triplet():
    atoms = _oxygen([1.0, 1.0])
    _run(atoms, xc="lda", grid_spacing=0.3, basis={"name": "PAW-LCAO", "size": "SZ"},
         population="hirshfeld")
    return atoms


class TestUnrestricted:
    """Spin-unrestricted Kohn-Sham, selected by n_alpha != n_beta."""

    def test_triplet_oxygen_is_below_the_spin_restricted_ensemble(
            self, oxygen_triplet):
        """Spin-restricted O2 puts one electron in each of the two
        degenerate pi* orbitals (an ensemble).  Filling one of them
        breaks the symmetry, and the SCF then landed, depending on its
        path, 0.53 eV or 24 eV above the ensemble.  Hund's triplet lies
        0.78 eV below it (SZ, LDA), and the crystal path finds the same
        0.778 eV between its unpolarized and triplet O2 (Gamma, 1 mHa
        smearing)."""
        singlet = _oxygen([0.0, 0.0])
        e_singlet = _run(singlet, xc="lda", grid_spacing=0.3,
                         basis={"name": "PAW-LCAO", "size": "SZ"})
        e_triplet = oxygen_triplet.get_potential_energy()
        assert -0.85 < e_triplet - e_singlet < -0.7
        assert singlet.calc.get_number_of_spins() == 1
        scf = singlet.calc.result.scf
        assert np.allclose(scf.occupations, [2, 2, 2, 2, 2, 1, 1, 0])
        assert scf.mo_energies[6] - scf.mo_energies[5] < 1e-6
        # The density is the ensemble's: the pi* pair holds half an
        # electron of each spin in each member, whatever its gauge.
        gamma = singlet.calc.solver.mean_field_rdm()
        alpha = np.linalg.eigvalsh(gamma[:8, :8])
        assert np.allclose(alpha, [0, 0.5, 0.5, 1, 1, 1, 1, 1], atol=1e-8)

    def test_the_moment_and_the_spin_channels(self, oxygen_triplet):
        calc = oxygen_triplet.calc
        scf = calc.result.scf
        assert calc.result.success
        assert (scf.n_alpha, scf.n_beta) == (7, 5)
        assert calc.get_number_of_spins() == 2
        assert calc.get_total_magnetic_moment() == pytest.approx(2.0)
        assert np.allclose(calc.results["magmoms"], 1.0, atol=1e-3)
        assert scf.spin_contamination < 0.01
        assert len(calc.get_eigenvalues(spin=1)) == len(
            calc.get_eigenvalues(spin=0))
        energies, dos = calc.dos(width=0.1)
        below = energies < calc.get_fermi_level()
        assert np.trapezoid(dos[below], energies[below]) == pytest.approx(
            12.0, abs=1e-6)

    def test_the_many_body_problem_is_exported_in_natural_orbitals(
            self, oxygen_triplet):
        result = oxygen_triplet.calc.result
        assert np.allclose(result.model_orbitals,
                           result.scf.natural_orbitals)
        assert result.num_particles == (7, 5)
        problem = result.as_quantum_problem()
        assert problem["num_particles"] == (7, 5)

    @pytest.mark.parametrize("xc", ["lda", "pbe", "r2scan"])
    def test_unrestricted_forces_are_the_derivative_of_the_energy(self, xc):
        """The OH radical (seven valence electrons, so unrestricted): the
        spin-averaged gradient plus the magnetization term, against
        Richardson-extrapolated central differences on the frozen grid.

        Its one beta pi electron has no integer aufbau solution (whichever
        member holds it rises above the empty one): the SCF converges to
        the ensemble with half an electron in each, and the forces are
        that ensemble's."""
        atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]],
                      cell=[6, 6, 7])
        atoms.center()
        atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.3, trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        atoms.get_forces()
        result = atoms.calc.force_result
        assert atoms.calc.result.num_particles == (4, 3)
        scf = atoms.calc.result.scf
        assert scf.occupations_alpha is None
        assert np.allclose(scf.occupations_beta, [1, 1, 0.5, 0.5, 0])

        def energy(atom, k, step):
            moved = atoms.copy()
            moved.positions[atom, k] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        for atom, k in ((1, 2), (1, 0)):
            coarse, fine = (-(energy(atom, k, s) - energy(atom, k, -s))
                            / (2 * s) for s in (0.004, 0.002))
            numerical = fine + (fine - coarse) / 3.0
            assert result.unprojected[atom, k] == pytest.approx(numerical,
                                                                abs=1e-4)


@pytest.fixture(scope="module")
def h2_hybrid():
    """H2 with HSE06 in a basis with more than one function per atom."""
    atoms = h2()
    _run(atoms, xc="hse06", grid_spacing=0.3, basis=NAO_DZ)
    return atoms


@pytest.fixture(scope="module")
def h2_paw():
    """H2 on PAW-LCAO (LDA): the integrals a hybrid's terms are built from."""
    atoms = h2()
    _run(atoms, xc="lda", grid_spacing=0.35, basis={"name": "PAW-LCAO"})
    return atoms


class TestHybrid:
    """``xc="hse06"``: generalized Kohn-Sham with short-range exact exchange."""

    def test_the_exact_exchange_is_part_of_the_xc_energy(self, h2_hybrid):
        scf = h2_hybrid.calc.result.scf
        assert scf.functional == "hse06"
        assert scf.exact_exchange_energy < 0.0
        assert scf.xc_energy < scf.exact_exchange_energy
        log = h2_hybrid.calc.solver._scf_summary_fields(h2_hybrid.calc.result)
        assert float(log["exact_exchange_energy_eV"]) == pytest.approx(
            from_hartree(scf.exact_exchange_energy, "eV"))

    def test_the_hybrid_opens_the_gap_and_lowers_the_homo(self, h2_hybrid):
        """The textbook effect: HSE06 - PBE moves the H2 HOMO down by ~1.2
        eV (an independent all-electron aug-cc-pVQZ calculation: -1.219 eV)."""
        pbe = h2()
        _run(pbe, xc="pbe", grid_spacing=0.3, basis=NAO_DZ)
        shift = (h2_hybrid.calc.get_eigenvalues()[0]
                 - pbe.calc.get_eigenvalues()[0])
        assert -1.5 < shift < -0.9
        assert h2_hybrid.calc.result.scf.homo_lumo_gap > \
            pbe.calc.result.scf.homo_lumo_gap

    def test_the_two_screening_limits(self, h2_hybrid):
        r"""At fixed density: :math:`\omega \to 0` is the PBE0-like hybrid
        (a quarter of the full exact exchange for a quarter of the full
        semilocal one), :math:`\omega \to \infty` the unscreened semilocal
        functional alone."""
        from mandacaru.algorithms import dft
        from mandacaru.integrals import exchange_correlation as xc_grid
        integrals = h2_hybrid.calc.solver._gradient_context["integrals"]
        scf = h2_hybrid.calc.result.scf
        grid = integrals.grid
        reference = dft.kohn_sham_solver(integrals, 2, "hse06")
        D = reference._density_matrix(scf.mo_coefficients)
        rho = reference.density(D)
        semilocal = xc_grid.evaluate(grid, rho, "hse06",
                                     screening=(0.11, 0.0)).energy

        pbe0 = dft.kohn_sham_solver(integrals, 2, "hse06",
                                    screening=(0.0, 0.25))
        K = np.einsum("sr,prsq->pq", D, integrals.two_body())
        exact = -0.25 * float(np.real(np.sum(D * K.T)))     # E_x^HF
        without_exchange = xc_grid.evaluate(grid, rho, "hse06",
                                            screening=(0.0, 1.0)).energy
        expected = (semilocal - 0.25 * (semilocal - without_exchange)
                    + 0.25 * exact)
        assert pbe0._xc(D)[0] == pytest.approx(expected, abs=1e-10)

        unscreened = dft.kohn_sham_solver(integrals, 2, "hse06",
                                          screening=(1e3, 0.25))
        assert unscreened._xc(D)[0] == pytest.approx(semilocal, abs=1e-5)

    def test_full_exact_exchange_cancels_the_self_interaction(self):
        """One electron, fraction 1, no screening: the exact exchange is
        minus the Hartree energy -- the contraction and its spin factor."""
        from mandacaru.algorithms import dft
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[6] * 3)
        _run(atoms, grid_spacing=0.35)
        integrals = atoms.calc.solver._gradient_context["integrals"]
        scf = atoms.calc.result.scf
        solver = dft.kohn_sham_solver(integrals, 1, "hse06", spins=(1, 0),
                                      screening=(0.0, 1.0))
        Da = solver._spin_density(scf.mo_coefficients_alpha, 1)
        Db = np.zeros_like(Da)
        _fa, _fb, _e, hartree, _xc = solver._fock_pair(Da, Db)
        assert solver.exact_exchange_energy == pytest.approx(-hartree,
                                                             rel=1e-12)

    def test_the_unrestricted_energy_is_stationary(self):
        r"""``dE = sum_s tr(F_s dD_s)`` with the exact-exchange term in each
        channel's operator."""
        from mandacaru.algorithms import dft
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[6] * 3)
        _run(atoms, grid_spacing=0.35, xc="hse06", basis=NAO_DZ)
        integrals = atoms.calc.solver._gradient_context["integrals"]
        scf = atoms.calc.result.scf
        solver = dft.kohn_sham_solver(integrals, 1, "hse06", spins=(1, 0))
        Da = solver._spin_density(scf.mo_coefficients_alpha, 1)
        Db = 0.1 * Da
        rng = np.random.default_rng(5)
        B = rng.normal(size=Da.shape) + 1j * rng.normal(size=Da.shape)
        delta_a = 1e-2 * (B @ B.conj().T)
        delta_b = 0.5 * delta_a.T
        eps = 1e-4
        Fa, Fb, e0, _h, _x = solver._fock_pair(Da, Db)
        e1 = solver._fock_pair(Da + eps * delta_a, Db + eps * delta_b)[2]
        e2 = solver._fock_pair(Da + 2 * eps * delta_a,
                               Db + 2 * eps * delta_b)[2]
        directional = float(np.real(np.sum(Fa.T * delta_a)
                                    + np.sum(Fb.T * delta_b)))
        assert (-3 * e0 + 4 * e1 - e2) / (2 * eps) == pytest.approx(
            directional, rel=1e-5)

    def test_paw_exchange_sees_the_compensation_charges(self, h2_paw):
        r"""On PAW-LCAO the exchange tensor is the augmented pair densities':
        unscreened it is the full tensor, and its long-range part is
        :math:`2\omega/\sqrt\pi\,\delta_{pr}\delta_{qs}` at small omega --
        the augmented pair densities, not the smooth ones, carry the
        orthonormal basis's unit charges."""
        integrals = h2_paw.calc.solver._gradient_context["integrals"]
        assert integrals._ao_two_body_terms()[1] is not None
        assert np.allclose(integrals.short_range_two_body(0.0),
                           integrals.two_body(), atol=1e-12)
        w1, w2 = 0.005, 0.01
        long_range = (integrals.short_range_two_body(w1)
                      - integrals.short_range_two_body(w2)).real
        M = integrals.n_orbitals
        expected = (2.0 * (w2 - w1) / np.sqrt(np.pi)
                    * np.einsum("pr,qs->pqrs", np.eye(M), np.eye(M)))
        assert np.allclose(long_range, expected, atol=2e-5)

    def test_paw_one_center_terms_are_in_the_energy_and_the_operator(
            self, h2_paw):
        """The augmentation spheres' exact exchange and frozen semilocal
        exchange are part of ``E_xc``, and the operator is still its
        derivative."""
        from mandacaru.algorithms import dft
        integrals = h2_paw.calc.solver._gradient_context["integrals"]
        scf = h2_paw.calc.result.scf
        solver = dft.kohn_sham_solver(integrals, 2, "hse06")
        assert solver.one_center is not None
        D = solver._density_matrix(scf.mo_coefficients)
        C, spheres = solver.one_center
        terms = spheres.evaluate(C.conj().T @ D @ C)
        assert terms.exact_exchange < 0.0 and terms.semilocal != 0.0
        e_xc = solver._xc(D)[0]
        solver.one_center = None
        assert e_xc - solver._xc(D)[0] == pytest.approx(terms.energy,
                                                        abs=1e-12)
        solver.one_center = (C, spheres)
        rng = np.random.default_rng(7)
        B = rng.normal(size=D.shape) + 1j * rng.normal(size=D.shape)
        delta = 1e-2 * (B @ B.conj().T)
        eps = 1e-4
        F, e0, _h, _x = solver._fock(D)
        e1 = solver._fock(D + eps * delta)[1]
        e2 = solver._fock(D + 2 * eps * delta)[1]
        directional = float(np.real(np.sum(F.T * delta)))
        assert (-3 * e0 + 4 * e1 - e2) / (2 * eps) == pytest.approx(
            directional, rel=1e-5)

    def test_forces_are_the_derivative_of_the_energy(self):
        """The short-range tensor's derivative is the full one's minus the
        long-range one's; against Richardson-extrapolated central
        differences on the frozen grid (H2 tilted off its axis, so both
        components are nonzero)."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.05, 0, 0.74]],
                      cell=[6] * 3)
        _run(atoms, xc="hse06", grid_spacing=0.3, basis=NAO_DZ)
        atoms.get_forces()
        result = atoms.calc.force_result
        assert result.details["energy_hartree"] == pytest.approx(
            to_hartree(atoms.get_potential_energy(), "eV"), abs=1e-8)

        def energy(step):
            moved = atoms.copy()
            moved.positions[1, 2] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        coarse, fine = (-(energy(s) - energy(-s)) / (2 * s)
                        for s in (0.004, 0.002))
        numerical = fine + (fine - coarse) / 3.0
        assert result.unprojected[1, 2] == pytest.approx(numerical, abs=1e-4)


def _oh_radical():
    atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]],
                  cell=[6, 6, 7])
    atoms.center()
    return atoms


def _tilted_h2():
    return Atoms("H2", positions=[[0, 0, 0], [0.05, 0, 0.74]], cell=[6] * 3)


#: LiH in a single-zeta basis does not converge its SCF (with either
#: operator); SZP does.
PAW_SZP = {"name": "PAW-LCAO", "size": "SZP"}

#: Every Kohn-Sham route through the spectral Pulay term: the semilocal
#: functionals with and without D4, the screened hybrid, an unrestricted
#: run, the unitary family and an all-electron basis.
SPECTRAL_FORCE_CASES = {
    "lda/paw-lcao": (lih, dict(xc="lda", grid_spacing=0.3, basis=PAW_SZP)),
    "pbe+d4/paw-lcao": (lih, dict(xc="pbe", vdw="d4", grid_spacing=0.3,
                                  basis=PAW_SZP)),
    "lda/upaw-lcao": (_tilted_h2, dict(xc="lda", grid_spacing=0.25,
                                       basis={"name": "UPAW-LCAO",
                                              "size": "SZ"})),
    "hse06/nao": (_tilted_h2, dict(xc="hse06", grid_spacing=0.3, basis=NAO_DZ)),
    "r2scan/paw-lcao": (lih, dict(xc="r2scan", grid_spacing=0.3,
                                  basis=PAW_SZP)),
    "uks-lda/paw-lcao": (_oh_radical, dict(xc="lda", grid_spacing=0.3,
                                           basis={"name": "PAW-LCAO",
                                                  "size": "SZ"})),
}

#: 100 s under load; LDA on the same LiH covers its PAW-LCAO terms.
_SLOW_SPECTRAL_CASES = {"pbe+d4/paw-lcao"}


class TestSpectralKineticForces:
    """``kinetic="spectral"`` forces (Phase N): the Pulay term contracts the
    moved basis functions against the FFT Laplacian of the unmoved ones
    (``forces.one_body_cross``), so the force is the derivative of the
    energy the spectral operator gives.  Against Richardson-extrapolated
    central differences on the frozen grid, the standard of the stencil's
    tests above; HISTORY (2026-10-07, Phase N) has every component."""

    @pytest.mark.parametrize("case", [
        pytest.param(case, marks=pytest.mark.slow)
        if case in _SLOW_SPECTRAL_CASES else case
        for case in sorted(SPECTRAL_FORCE_CASES)])
    def test_they_are_the_derivative_of_the_energy(self, case):
        if "d4" in case:
            pytest.importorskip("dftd4")
        build, options = SPECTRAL_FORCE_CASES[case]
        atoms = build()
        atoms.calc = Mandacaru(method="dft", kinetic="spectral", trace=False,
                               **options)
        atoms.get_forces()
        result = atoms.calc.force_result
        assert atoms.calc.solver._gradient_context["integrals"].kinetic \
            == "spectral"

        def energy(step):
            moved = atoms.copy()
            moved.positions[1, 2] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        coarse, fine = (-(energy(s) - energy(-s)) / (2 * s)
                        for s in (0.004, 0.002))
        numerical = fine + (fine - coarse) / 3.0
        assert abs(numerical) > 0.1                  # a real force
        assert result.unprojected[1, 2] == pytest.approx(numerical, abs=1e-4)


class TestTheHartreeTerm:
    """``hartree="poisson"`` (the default) against ``hartree="tensor"``.

    Both integrate the spectral isolated kernel, the Poisson route with the
    PAW-LCAO compensation multipoles as low-rank matrices, so they agree to
    round-off: the Fock matrix, the energies and the reference energy of the
    exported Hamiltonian, on a basis whose compensation charges reach L = 2.
    """

    @staticmethod
    def _solvers(atoms, spins):
        from mandacaru.algorithms import dft
        integrals = atoms.calc.solver._gradient_context["integrals"]
        scf = atoms.calc.result.scf
        return [dft.kohn_sham_solver(integrals, sum(spins), scf.functional,
                                     spins=spins, hartree=route)
                for route in ("poisson", "tensor")], integrals, scf

    def test_the_closed_shell_fock_matrices_agree(self, water_szp):
        (poisson, tensor), integrals, scf = self._solvers(water_szp, (4, 4))
        assert max(L for _A, L, _M in integrals.multipole_channels()) == 2
        D = tensor._density_matrix(scf.mo_coefficients)
        F_p, e_p, h_p, _x = poisson._fock(D)
        F_t, e_t, h_t, _x = tensor._fock(D)
        assert np.abs(F_p - F_t).max() < 1e-10
        assert e_p == pytest.approx(e_t, abs=1e-10)
        assert h_p == pytest.approx(h_t, abs=1e-10)

    def test_the_reference_energy_is_the_tensor_one(self, water_szp):
        """The run (Poisson route) took the occupied block of the tensor
        from n^2/2 pair solves; the tensor gives the same determinant
        energy."""
        from mandacaru.algorithms import dft
        (_poisson, tensor), _integrals, scf = self._solvers(water_szp, (4, 4))
        h_mo, eri_mo = tensor._mo_integrals(scf.mo_coefficients, 4)[:2]
        assert dft.determinant_energy(h_mo, eri_mo, 4, 4) == pytest.approx(
            scf.determinant_energy, abs=1e-10)
        assert np.abs(scf.eri_mo - eri_mo).max() < 1e-10   # built on demand

    def test_the_unrestricted_routes_agree(self, oxygen_triplet):
        (poisson, tensor), _integrals, scf = self._solvers(oxygen_triplet,
                                                           (7, 5))
        Da = tensor._spin_density(scf.mo_coefficients_alpha, 7,
                                  scf.occupations_alpha)
        Db = tensor._spin_density(scf.mo_coefficients_beta, 5,
                                  scf.occupations_beta)
        one, other = poisson._fock_pair(Da, Db), tensor._fock_pair(Da, Db)
        for a, b in zip(one[:2], other[:2]):
            assert np.abs(a - b).max() < 1e-10
        assert one[2] == pytest.approx(other[2], abs=1e-10)

    def test_a_run_on_either_route_gives_the_same_energies(self):
        """And the Poisson route builds no two-body tensor at all."""
        energies, references = [], []
        for route in ("poisson", "tensor"):
            atoms = h2()
            energies.append(_run(atoms, grid_spacing=0.3, basis=NAO_DZ, hartree=route))
            references.append(atoms.calc.result.reference_energy)
            integrals = atoms.calc.solver._gradient_context["integrals"]
            built = (integrals._eri is not None, integrals._eri_ao is not None)
            assert built == ((False, False) if route == "poisson"
                             else (True, True))
        assert energies[0] == pytest.approx(energies[1], abs=1e-9)
        assert references[0] == pytest.approx(references[1], abs=1e-9)

    def test_an_unknown_route_is_refused(self):
        with pytest.raises(ValueError, match="unknown hartree"):
            Mandacaru(method="dft", hartree="multigrid")

    def test_a_crystal_refuses_the_tensor_route(self):
        from ase.build import bulk
        atoms = bulk("Al", "fcc", a=4.05)
        atoms.calc = Mandacaru(method="dft", hartree="tensor", trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        with pytest.raises(ValueError, match="molecular"):
            atoms.get_potential_energy()


class TestAnalyticTwoCenter:
    """``two_center="analytic"``: the overlap, the kinetic energy and the
    projections from the Fourier-Bessel tables instead of the grid
    (``integrals/two_center.py``); the matrices themselves are in
    ``test/integrals/test_two_center.py``."""

    PAW = {"name": "PAW-LCAO", "size": "DZP"}

    @staticmethod
    def _lih():
        atoms = Atoms("LiH", positions=[[0, 0, 0], [0.03, 0.02, 1.60]],
                      cell=[7, 7, 8])
        atoms.center()
        return atoms

    @pytest.fixture(scope="class", params=["lda", "pbe"])
    def runs(self, request):
        """LiH at h = 0.3 on both routes, forces taken."""
        out = {}
        for route in ("grid", "analytic"):
            atoms = self._lih()
            atoms.calc = Mandacaru(method="dft", xc=request.param,
                                   grid_spacing=0.3, basis=self.PAW,
                                   two_center=route, trace=False)
            atoms.get_forces()
            # An energy-only evaluation clears the result: keep the gradient.
            atoms.raw = atoms.calc.force_result.unprojected.copy()
            out[route] = atoms
        return request.param, out

    @staticmethod
    def _numerical(atoms, atom, k):
        def energy(step):
            moved = atoms.copy()
            moved.positions[atom, k] += step
            moved.calc = atoms.calc
            return moved.get_potential_energy()

        coarse, fine = (-(energy(s) - energy(-s)) / (2 * s)
                        for s in (0.004, 0.002))
        return fine + (fine - coarse) / 3.0

    def test_the_forces_are_the_derivative_of_the_energy(self, runs):
        """Richardson-extrapolated central differences of the same energy on
        the frozen grid: the analytic tables are differentiated exactly."""
        _xc, out = runs
        atoms = out["analytic"]
        for atom, k in ((1, 2), (1, 0), (0, 2)):
            assert atoms.raw[atom, k] == pytest.approx(
                self._numerical(atoms, atom, k), abs=2e-5)

    def test_the_energy_is_the_grids_to_the_grid_s_error(self, runs):
        _xc, out = runs
        grid, analytic = (out[r].calc.result.optimal_energy
                          for r in ("grid", "analytic"))
        assert abs(analytic - grid) < 5e-3                     # eV
        assert analytic != grid

    def test_the_log_and_the_integrals_name_the_route(self, runs):
        _xc, out = runs
        solver = out["analytic"].calc.solver
        integrals = solver._gradient_context["integrals"]
        assert (integrals.two_center, integrals.kinetic) == (
            "analytic", "analytic")
        assert solver._kinetic_label() == "analytic (two-center)"
        assert out["grid"].calc.solver._kinetic_label() == "spectral"

    def test_the_analytic_matrices_are_the_ones_the_energy_used(self, runs):
        """S is the tables', not the grid's: the overlap of the converged
        calculation is the analytic one to round-off, and differs from the
        grid's by the grid's own error."""
        _xc, out = runs
        integrals = out["analytic"].calc.solver._gradient_context["integrals"]
        tables = integrals.two_center_tables()
        assert np.allclose(integrals.bare_overlap(),
                           tables.matrix("overlap"), atol=1e-14)
        psi = integrals._engine._psi
        grid_overlap = psi @ psi.T * integrals.grid.dV
        assert 1e-8 < np.abs(integrals.bare_overlap() - grid_overlap).max() < 2e-2

    def test_the_net_force_shrinks(self, runs):
        """A free molecule's forces sum to zero; the grid's do not
        (egg-box).  The three-center terms stay on the grid, so a remainder
        is left, but the two-center part of it is gone."""
        _xc, out = runs
        net = {r: np.abs(out[r].raw.sum(axis=0)).max() for r in out}
        assert net["analytic"] < 0.5 * net["grid"]

    def test_the_unrestricted_forces_are_the_derivative_of_the_energy(self):
        """The OH radical (open shell, the magnetization term as well)."""
        atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]],
                      cell=[6, 6, 7])
        atoms.center()
        atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.3,
                               trace=False, two_center="analytic",
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        atoms.get_forces()
        assert atoms.calc.result.num_particles == (4, 3)
        result = atoms.calc.force_result
        for atom, k in ((1, 2), (1, 0)):
            assert result.unprojected[atom, k] == pytest.approx(
                self._numerical(atoms, atom, k), abs=2e-5)

    def test_the_run_log_states_it_once(self, tmp_path):
        """One owner block: the ``[ELECTRONS]`` line of the kinetic operator,
        which the analytic route owns; no second field repeats it."""
        from mandacaru.utils.logging import parse_output, reset_log
        path = str(tmp_path / "analytic.txt")
        atoms = h2()
        atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.35,
                               txt=path, trace=False, two_center="analytic",
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        atoms.get_potential_energy()
        reset_log(path)
        assert parse_output(path)["electrons"]["kinetic operator"] == \
            "analytic (two-center)"
        with open(path) as handle:
            text = handle.read()
        assert text.count("two-center") == 1
        assert "two_center" not in text         # not echoed by a second block

    def test_it_is_refused_with_a_kinetic_operator(self):
        with pytest.raises(ValueError, match="kinetic"):
            Mandacaru(method="dft", basis=self.PAW, two_center="analytic",
                      kinetic="spectral")

    @pytest.mark.parametrize("basis", ["NAO", "HAO", "6-31G", "PW"])
    def test_it_needs_a_pseudopotential_basis(self, basis):
        with pytest.raises(NotImplementedError, match="PAW-LCAO"):
            Mandacaru(method="dft", basis=basis, two_center="analytic")

    def test_an_unknown_route_is_refused(self):
        with pytest.raises(ValueError, match="two_center"):
            Mandacaru(method="dft", basis=self.PAW, two_center="exact")

    def test_scf_response_forces_are_refused_at_construction(self):
        """The legacy gradient differentiates the grid matrices; the
        combination fails in the constructor, before any SCF."""
        with pytest.raises(ValueError, match="scf-response"):
            Mandacaru(method="dft", basis=self.PAW, two_center="analytic",
                      force_method="scf-response")

    def test_spin_orbit_datasets_are_refused_before_the_grid_is_sampled(self):
        """Their projections are still quadratures.  The dataset is known
        only when the integrals are built, where the refusal comes first:
        before the basis is sampled on the grid (1 s, not an SCF)."""
        atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[6, 6, 7])
        atoms.calc = Mandacaru(method="rhf", grid_spacing=0.4, trace=False,
                               two_center="analytic", paw_dataset="lda-fr",
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        with pytest.raises(NotImplementedError, match="spin-orbit"):
            atoms.get_potential_energy()
        assert atoms.calc.solver._gradient_context is None

    def test_a_periodic_method_is_refused_at_construction(self):
        with pytest.raises(NotImplementedError, match="molecular"):
            Mandacaru(method="bloch-vqe", basis=self.PAW, two_center="analytic",
                      kpts={"size": (1, 1, 1), "gamma": True})

    def test_a_crystal_is_refused(self):
        from ase.build import bulk
        atoms = bulk("Al", "fcc", a=4.05)
        atoms.calc = Mandacaru(method="dft", trace=False,
                               two_center="analytic",
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        with pytest.raises(NotImplementedError, match="molecular"):
            atoms.get_potential_energy()

    def test_a_loaded_hamiltonian_would_ignore_it(self, tmp_path):
        with pytest.raises(ValueError, match="two_center"):
            Mandacaru(method="adapt-vqe", basis=self.PAW,
                      two_center="analytic",
                      load_hamiltonian=str(tmp_path / "missing.json"))
