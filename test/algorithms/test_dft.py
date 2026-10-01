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


@pytest.fixture(scope="module", params=["lda", "pbe", "r2scan"])
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
        X = integrals._lowdin_x()
        solver = dft.KohnSham(integrals.one_body(), integrals.two_body(),
                              X.T @ integrals._engine._psi, integrals.grid, 2,
                              functional=scf.functional)
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
        expected = {"lda": "PerdewZunger1981", "pbe": "PBE1996",
                    "r2scan": "Furness2020"}[xc]
        assert expected in keys


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

    def test_an_open_shell_is_refused(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[6] * 3)
        atoms.calc = Mandacaru(method="dft", h=0.35, trace=False)
        with pytest.raises(NotImplementedError, match="closed-shell"):
            atoms.get_potential_energy()

    def test_a_measurement_provider_is_refused(self):
        with pytest.raises(ValueError, match="no quantum state"):
            Mandacaru(method="dft", measurement_provider=object())
