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
from mandacaru.units import from_hartree, to_hartree

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


@pytest.fixture(scope="module", params=["lda", "pbe", "r2scan", "hse06"])
def converged(request):
    """An H2 Kohn-Sham run in a basis with more than one function per atom."""
    atoms = h2()
    energy = _run(atoms, xc=request.param, h=0.3, basis=NAO_DZ)
    return request.param, atoms, energy


class TestTheRun:
    def test_it_converges_and_reports_a_total_energy(self, converged):
        _xc, atoms, energy = converged
        result = atoms.calc.result
        assert result.success
        assert result.method == "dft"
        assert energy == pytest.approx(result.optimal_energy)
        assert result.scf.xc_energy < 0.0 < result.scf.hartree_energy
        assert atoms.calc.ansatz is None

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
        eps = 1e-4
        F, e0, _h, _x = solver._fock(D)
        e1 = solver._fock(D + eps * delta)[1]
        e2 = solver._fock(D + 2 * eps * delta)[1]
        directional = float(np.real(np.sum(F.T * delta)))
        assert (-3 * e0 + 4 * e1 - e2) / (2 * eps) == pytest.approx(
            directional, rel=1e-5)

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
        rhf.calc = Mandacaru(method="rhf", h=0.3, trace=False, basis=NAO_DZ)
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
                    "hse06": {"Heyd2003", "Krukau2006", "PBE1996"}}[xc]
        assert expected <= set(keys)
        assert "Paier2005" not in keys          # no augmentation spheres

    def test_a_paw_hybrid_cites_the_one_center_exchange(self):
        atoms = h2()
        _run(atoms, xc="hse06", h=0.35, basis={"name": "PAW-LCAO",
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
            _run(atoms, xc=xc, h=0.25, basis={"name": "PAW-LCAO"})
            assert atoms.calc.result.scf.n_iterations < 30

    def test_the_log_holds_every_iteration(self, tmp_path):
        """``[SCF ITERATIONS]``: one row per iteration, ending where the
        convergence test stopped and on the energy the summary reports."""
        from mandacaru.utils.logging import parse_output, reset_log
        path = str(tmp_path / "dft.txt")
        atoms = h2()
        atoms.calc = Mandacaru(method="dft", xc="lda", h=0.35, txt=path,
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
        _run(atoms, xc="r2scan", h=0.25, basis={"name": "PAW-LCAO",
                                                "size": "DZP"})
        assert atoms.calc.result.success


class TestFunctionals:
    def test_the_gradient_corrections_lower_the_energy_of_h2(self):
        energies = {xc: _run(h2(), xc=xc, h=0.35)
                    for xc in ("lda", "pbe", "r2scan")}
        assert energies["pbe"] < energies["lda"]
        assert energies["r2scan"] < energies["lda"]


class TestPAWLCAO:
    def test_the_reference_atom_recovers_its_valence_energy(self):
        """Be in its own smooth partial wave: the dataset's reference energy.

        With the unconfined single-zeta basis the Kohn-Sham solution *is* the
        reference atom, so the energy -- core correction and frozen one-center
        constant included -- is the dataset's ``reference_valence`` up to the
        grid error (about 1 mHa at h = 0.2 Angstrom).
        """
        atoms = Atoms("Be", positions=[[4.0] * 3], cell=[8.0] * 3)
        energy = to_hartree(_run(atoms, xc="lda", h=0.2, basis=PAW_SZ_FREE),
                            "eV")
        integrals = atoms.calc.solver._gradient_context["integrals"]
        dataset = integrals.pseudopotentials[0]
        assert dataset.nlcc.get("applied")
        assert atoms.calc.result.scf.core_correction_energy != 0.0
        assert energy == pytest.approx(
            dataset.energies["reference_valence"], abs=3e-3)

    def test_lda_on_lda_datasets_is_silent_and_pbe_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            lda = _run(h2(), xc="lda", h=0.3, basis={"name": "PAW-LCAO"})
        with pytest.warns(RuntimeWarning, match="different functionals"):
            pbe = _run(h2(), xc="pbe", h=0.3, basis={"name": "PAW-LCAO"})
        assert np.isfinite(lda) and np.isfinite(pbe)


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
    xc, dispersion = request.param
    if dispersion:
        pytest.importorskip("dftd4")
    atoms = lih()
    atoms.calc = Mandacaru(method="dft", xc=xc, dispersion=dispersion, h=0.25,
                           basis={"name": "PAW-LCAO", "size": "DZP"},
                           trace=False)
    forces = atoms.get_forces()
    return atoms, forces, atoms.calc.force_result


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

    def test_a_meta_gga_returns_forces_of_its_reported_energy(self):
        """r2SCAN: the rebuilt energy matches and the forces are finite.

        Its energy surface on a finite grid is too rough (~1 uHa) for a
        finite-difference check at the precision of the LDA and PBE ones.
        """
        atoms = h2()
        atoms.calc = Mandacaru(method="dft", xc="r2scan", h=0.3, basis=NAO_DZ,
                               trace=False)
        forces = atoms.get_forces()
        assert np.all(np.isfinite(forces))
        assert forces[1, 2] == pytest.approx(-forces[0, 2])


class TestDispersion:
    def test_d4_adds_its_energy_to_the_total(self):
        pytest.importorskip("dftd4")
        from mandacaru.algorithms.dft import d4_dispersion_energy
        bare = h2()
        e_bare = _run(bare, xc="pbe", h=0.35)
        dressed = h2()
        e_d4 = _run(dressed, xc="pbe", h=0.35, dispersion="d4")
        d4 = d4_dispersion_energy(dressed, "pbe")
        assert d4 < 0.0
        assert dressed.calc.result.scf.dispersion_energy == pytest.approx(d4)
        assert e_d4 - e_bare == pytest.approx(from_hartree(d4, "eV"), abs=1e-8)
        assert "Caldeweyher2019" in dressed.calc.citation_keys()

    def test_d4_without_parameters_for_the_functional_is_refused(self):
        with pytest.raises(ValueError, match="no parameters"):
            Mandacaru(method="dft", xc="lda", dispersion="d4")

    def test_an_unknown_dispersion_is_refused(self):
        with pytest.raises(ValueError, match="unknown dispersion"):
            Mandacaru(method="dft", xc="pbe", dispersion="d3")


class TestOptions:
    def test_the_device_is_ignored(self):
        atoms = h2()
        energy = _run(atoms, h=0.35, device="AER_simulator")
        assert np.isfinite(energy)

    def test_an_unknown_functional_is_refused_by_the_constructor(self):
        with pytest.raises(ValueError, match="unknown exchange-correlation"):
            Mandacaru(method="dft", xc="b3lyp")

    def test_a_hydrogen_atom_runs_fully_polarized(self):
        """One electron: an empty spin-down channel everywhere -- the
        full-polarization limit of the spin functionals, end to end."""
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[6] * 3)
        atoms.calc = Mandacaru(method="dft", h=0.35, trace=False)
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
    _run(atoms, xc="lda", h=0.3, basis={"name": "PAW-LCAO", "size": "SZP"})
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
    _run(atoms, xc="lda", h=0.3, basis={"name": "PAW-LCAO", "size": "SZ"},
         population="hirshfeld")
    return atoms


class TestUnrestricted:
    """Spin-unrestricted Kohn-Sham, selected by n_alpha != n_beta."""

    def test_triplet_oxygen_is_below_the_closed_shell_singlet(
            self, oxygen_triplet):
        """The closed-shell singlet puts two electrons in one of the two
        degenerate pi* orbitals; Hund's triplet is ~1.3 eV lower (SZ, LDA)."""
        singlet = _oxygen([0.0, 0.0])
        e_singlet = _run(singlet, xc="lda", h=0.3,
                         basis={"name": "PAW-LCAO", "size": "SZ"})
        e_triplet = oxygen_triplet.get_potential_energy()
        assert -1.6 < e_triplet - e_singlet < -0.9
        assert singlet.calc.get_number_of_spins() == 1

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

    @pytest.mark.parametrize("xc", ["lda", "pbe"])
    def test_unrestricted_forces_are_the_derivative_of_the_energy(self, xc):
        """The OH radical (seven valence electrons, so unrestricted): the
        spin-averaged gradient plus the magnetization term, against
        Richardson-extrapolated central differences on the frozen grid."""
        atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]],
                      cell=[6, 6, 7])
        atoms.center()
        atoms.calc = Mandacaru(method="dft", xc=xc, h=0.3, trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        atoms.get_forces()
        result = atoms.calc.force_result
        assert atoms.calc.result.num_particles == (4, 3)

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
    _run(atoms, xc="hse06", h=0.3, basis=NAO_DZ)
    return atoms


@pytest.fixture(scope="module")
def h2_paw():
    """H2 on PAW-LCAO (LDA): the integrals a hybrid's terms are built from."""
    atoms = h2()
    _run(atoms, xc="lda", h=0.35, basis={"name": "PAW-LCAO"})
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
        _run(pbe, xc="pbe", h=0.3, basis=NAO_DZ)
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
        _run(atoms, h=0.35)
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
        _run(atoms, h=0.35, xc="hse06", basis=NAO_DZ)
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
        _run(atoms, xc="hse06", h=0.3, basis=NAO_DZ)
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
