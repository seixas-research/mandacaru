# -*- coding: utf-8 -*-
# file: test/algorithms/test_dft.py

# This code is part of Mandacaru.
# MIT License

"""Closed-shell Kohn-Sham DFT behind ``Mandacaru(method="dft")``."""

import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

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


def _gap(scf):
    """HOMO-LUMO gap (Ha) of a spin-restricted Kohn-Sham state."""
    levels, occupied = scf.mo_energies, scf.n_occupied
    return levels[occupied] - levels[occupied - 1]


def _numerical_force(atoms, atom, k):
    """Richardson-extrapolated central difference of the energy (eV/Angstrom).

    The displaced geometries are run by the same calculator, which keeps the
    integration grid of the first force request (a frozen grid)."""
    def energy(step):
        moved = atoms.copy()
        moved.positions[atom, k] += step
        moved.calc = atoms.calc
        return moved.get_potential_energy()

    coarse, fine = (-(energy(s) - energy(-s)) / (2 * s)
                    for s in (0.004, 0.002))
    return fine + (fine - coarse) / 3.0


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
        xc, atoms, energy = converged
        result = atoms.calc.result
        assert result.success
        assert result.method == "dft"
        assert energy == pytest.approx(result.optimal_energy)
        energies = result.scf.snapshot()
        assert energies["functional"] == xc
        assert energies["xc_energy"] < 0.0 < energies["hartree_energy"]
        assert atoms.calc.solver is None

    def test_the_eigenvalues_are_ordered_and_the_gap_is_positive(self, converged):
        _xc, atoms, _energy = converged
        scf = atoms.calc.result.scf
        assert np.all(np.diff(scf.mo_energies) >= 0)
        assert _gap(scf) > 0.0

    def test_the_kohn_sham_orbitals_export_a_quantum_problem(self, converged):
        """The exported Hamiltonian is written in the Kohn-Sham orbitals.

        Its reference determinant is the Kohn-Sham determinant, whose energy in
        the many-body Hamiltonian is the Hartree-Fock functional of those
        orbitals -- above the RHF minimum of the same basis -- and is what the
        quantum run reports as its ``reference_energy``.
        """
        _xc, atoms, _energy = converged
        result = atoms.calc.result
        state = result.reference_state()
        operator = result.qubit_hamiltonian()
        expectation = from_hartree(
            np.vdot(state, operator.matrix() @ state).real, "eV")
        rhf = h2()
        rhf.calc = Mandacaru(method="rhf", grid_spacing=0.3, trace=False, basis=NAO_DZ)
        assert expectation >= rhf.get_potential_energy() - 1e-6
        post = Mandacaru(method="adapt-vqe", trace=False,
                         **result.as_quantum_problem())
        assert post.run().reference_energy == pytest.approx(expectation,
                                                            abs=1e-7)

    def test_it_cites_kohn_sham(self, converged):
        from mandacaru.utils.citations import citation_keys
        _xc, atoms, _energy = converged
        keys = citation_keys(method=atoms.calc.method)
        assert {"HohenbergKohn1964", "KohnSham1965"} <= set(keys)
        assert "Paier2005" not in keys          # no augmentation spheres


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

    @pytest.mark.parametrize("basis,atomic_units", [
        ("HAO", False),
        ({"name": "PAW-LCAO", "size": "SZ"}, False),
        ({"name": "PAW-LCAO", "size": "SZ"}, True),
    ])
    def test_the_history_holds_every_iteration(self, basis, atomic_units):
        """One record per iteration, ending where the convergence test
        stopped and on the electronic energy the state reports."""
        if basis == "HAO":
            atoms = h2()
        else:
            from ase.build import molecule
            atoms = molecule("H2O")
            atoms.center(vacuum=2.0)
        atoms.calc = Mandacaru(method="dft", xc="pbe", basis=basis,
                               grid_spacing=0.35, trace=False,
                               atomic_units=atomic_units)
        energy = atoms.get_potential_energy()
        result = atoms.calc.result
        scf = result.scf
        assert result.energy_unit == ("Ha" if atomic_units else "eV")
        assert to_hartree(energy, "eV") == pytest.approx(
            result.energy_hartree, abs=1e-8)       # ASE's energy is in eV
        rows = scf.snapshot()["history"]
        assert [row[0] for row in rows] == list(range(1, scf.n_iterations + 1))
        assert rows[0][2] is None
        for before, after in zip(rows, rows[1:]):
            assert after[2] == pytest.approx(after[1] - before[1], abs=2e-10)
        assert rows[-1][3] < 1e-8                    # what stopped it
        assert rows[-1][1] == pytest.approx(
            scf.snapshot()["electronic_energy"], abs=1e-9)

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
        from mandacaru.pseudopotentials.environment import default_library_folder
        from mandacaru.pseudopotentials.families import lookup_family
        atoms = Atoms("Be", positions=[[4.0] * 3], cell=[8.0] * 3)
        energy = to_hartree(_run(atoms, xc="lda", grid_spacing=0.2, basis=PAW_SZ_FREE),
                            "eV")
        dataset = lookup_family("PAW-LCAO").get("Be", default_library_folder("lda"))
        assert dataset.nlcc.get("applied")
        assert energy == pytest.approx(
            dataset.energies["reference_valence"], abs=3e-3)

    def test_each_functional_on_its_own_set_is_silent(self):
        """By default LDA reads ``lda-sr`` and PBE and r2SCAN read
        ``pbe-sr`` (r2SCAN has no datasets of its own), with no warning."""
        from mandacaru.pseudopotentials.environment import default_library_folder
        from mandacaru.pseudopotentials.families import lookup_family
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for xc in ("lda", "pbe", "r2scan"):
                atoms = h2()
                energy = _run(atoms, xc=xc, grid_spacing=0.3, basis={"name": "PAW-LCAO"})
                assert np.isfinite(energy)
                folder = default_library_folder(xc)
                assert folder == ("lda-sr" if xc == "lda" else "pbe-sr")
                dataset = lookup_family("PAW-LCAO").get("H", folder)
                assert dataset.xc == ("lda" if xc == "lda" else "pbe")

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

    def test_a_paw_hybrid_is_refused_not_approximated(self):
        """Screened exact exchange on PAW-LCAO needs the augmented
        short-range pair integrals; without them the run must raise."""
        atoms = h2()
        with pytest.raises(ValueError, match="augmented short-range"):
            _run(atoms, xc="hse06", grid_spacing=0.35,
                 basis={"name": "PAW-LCAO", "size": "SZ"})


def lih():
    """LiH off axis: lithium carries a core correction, and every PAW-LCAO
    term (projectors, compensation charges, split local potential) is live."""
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0.1, 0, 1.55]],
                  cell=[8, 8, 9])
    atoms.center()
    return atoms


@pytest.fixture(scope="module")
def lih_forces():
    atoms = lih()
    atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.25,
                           basis={"name": "PAW-LCAO", "size": "DZP"},
                           trace=False)
    forces = atoms.get_forces()
    result = atoms.calc.result
    # The raw (unprojected) Cartesian force: what the energy differentiates.
    return atoms, forces, result.forces(), result


def test_the_rvv10_force_is_the_derivative_of_its_energy_in_an_all_electron_basis():
    """rVV10's own part of the force: (F[X+rVV10] - F[X]) against the
    central difference of the energy difference.  The difference isolates it
    from the semilocal functional's force error.  Molecular PAW forces refuse
    nonlocal correlation, so this runs on the all-electron NAO basis: one
    component, along the bond."""
    for semilocal in ("pbe", "r2scan"):
        pair = {}
        for xc in (semilocal, f"{semilocal}+rvv10"):
            atoms = h2()
            atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.3,
                                   basis=NAO_DZ, trace=False,
                                   project_translation=False)
            pair[xc] = (atoms, atoms.get_forces()[1, 2])
        assert pair[f"{semilocal}+rvv10"][0].calc.xc == f"{semilocal}+rvv10"
        rvv10 = f"{semilocal}+rvv10"
        analytic = pair[rvv10][1] - pair[semilocal][1]
        numerical = (_numerical_force(pair[rvv10][0], 1, 2)
                     - _numerical_force(pair[semilocal][0], 1, 2))
        assert analytic == pytest.approx(numerical, abs=1e-5)


class TestTheFunctionalNames:
    """``xc=`` takes the canonical names only; the constructor validates them
    before anything is built."""

    SZ = {"name": "PAW-LCAO", "size": "SZ"}

    @pytest.mark.parametrize("xc", ["lda", "pbe", "r2scan", "hse06",
                                    "pbe+rvv10", "r2scan+rvv10"])
    def test_a_canonical_name_is_kept(self, xc):
        assert Mandacaru(method="dft", basis=self.SZ, xc=xc).xc == xc

    @pytest.mark.parametrize("xc", ["PBE", "lda+rvv10", "hse06+rvv10",
                                    "r2scan-rvv10", "r2scan_rvv10", "b3lyp"])
    def test_any_other_spelling_is_refused(self, xc):
        with pytest.raises(ValueError, match="canonical XC name"):
            Mandacaru(method="dft", basis=self.SZ, xc=xc)

    def test_the_retired_dispersion_option_is_refused(self):
        with pytest.raises((TypeError, ValueError)):
            Mandacaru(method="dft", xc="pbe", dispersion="d4",
                      basis=self.SZ)

    @pytest.mark.parametrize("xc", ["pbe+rvv10", "r2scan+rvv10"])
    def test_the_rvv10_forms_read_the_pbe_dataset_set(self, xc):
        from mandacaru.pseudopotentials.environment import default_library_folder
        assert default_library_folder(xc) == "pbe-sr"


class TestForces:
    def test_they_are_the_derivative_of_the_energy(self, lih_forces):
        """Richardson-extrapolated central differences on the frozen grid.

        The raw gradient is compared: the finite difference is of the
        discretized energy, which is not translation invariant.
        """
        atoms, _forces, raw, _result = lih_forces
        for atom, k in ((1, 2), (1, 0)):
            assert raw[atom, k] == pytest.approx(
                _numerical_force(atoms, atom, k), abs=1e-4)

    def test_the_projected_forces_sum_to_zero(self, lih_forces):
        _atoms, forces, raw, result = lih_forces
        assert result.force_details["scheme"] == "Richardson-fixed-density-PAW"
        assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-10)
        assert np.allclose(forces, raw - raw.mean(axis=0), atol=1e-10)

    def test_a_semilocal_gradient_never_forms_the_two_electron_tensor(
            self, lih_forces):
        """Hartree enters through its potential and the compensation
        charges' scalars: the SCF's Poisson route is all the gradient reads."""
        atoms, _forces, _raw, _result = lih_forces
        assert not atoms.calc.integrals.has_eri

    def test_a_meta_gga_force_is_the_derivative_of_its_energy(self):
        """r2SCAN: Richardson-extrapolated central differences on the
        frozen grid, to the bound of the LDA and PBE checks above.  Its
        energy surface used to be too rough for it (K1: 1.4 uHa, then the
        spectral gradient's ringing); the analytic force and the difference
        quotient now agree to 4e-5 eV/Angstrom, as PBE's do."""
        atoms = h2()
        atoms.calc = Mandacaru(method="dft", xc="r2scan", grid_spacing=0.3,
                               basis=NAO_DZ, kinetic="spectral", trace=False,
                               project_translation=False)
        forces = atoms.get_forces()
        assert np.all(np.isfinite(forces))
        assert forces[1, 2] == pytest.approx(-forces[0, 2])
        numerical = _numerical_force(atoms, 1, 2)
        assert abs(numerical) > 1.0                  # a real force
        assert forces[1, 2] == pytest.approx(numerical, abs=1e-4)

    def test_a_finite_difference_kinetic_operator_has_no_forces(self):
        """Forces differentiate the spectral kinetic operator only."""
        atoms = h2()
        atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.3,
                               basis=NAO_DZ, kinetic="fd", trace=False)
        with pytest.raises(PropertyNotImplementedError, match="forces"):
            atoms.get_forces()

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
                               trace=False, project_translation=False)
        forces = atoms.get_forces()
        numerical = _numerical_force(atoms, 0, 2)
        assert abs(numerical) > 1.0                  # a real force
        assert forces[0, 2] == pytest.approx(numerical, abs=1e-4)


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
    the converged state serves them, a new geometry converges its own."""

    @staticmethod
    def _calculator(calls, **options):
        """A calculator that records every SCF it runs (``run``)."""
        calc = Mandacaru(method="dft", xc=options.pop("xc", "lda"),
                         grid_spacing=0.35, basis=NAO_DZ, trace=False,
                         **options)
        original = calc.run

        def counting(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        calc.run = counting
        return calc

    def test_the_forces_do_not_repeat_the_scf(self):
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(calls)
        energy = atoms.get_potential_energy()
        result = atoms.calc.result
        forces = atoms.get_forces()
        assert len(calls) == 1
        assert atoms.calc.result is result
        assert atoms.get_potential_energy() == energy
        fresh = h2()
        fresh.calc = self._calculator(calls)
        assert np.allclose(forces, fresh.get_forces(), atol=1e-9)
        assert len(calls) == 2                    # the fresh calculator's own

    def test_a_moved_atom_converges_again(self):
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(calls)
        atoms.get_potential_energy()
        before = atoms.get_forces().copy()
        atoms.positions[1, 2] += 0.05
        atoms.get_forces()
        assert len(calls) == 2
        assert not np.allclose(before, atoms.get_forces())
        atoms.get_potential_energy()
        assert len(calls) == 2

    def test_forces_first_then_energy_run_once(self):
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(calls)
        atoms.get_forces()
        atoms.get_potential_energy()
        assert len(calls) == 1

    def test_cleared_results_never_serve_forces(self):
        calls = []
        atoms = h2()
        atoms.calc = self._calculator(calls)
        atoms.get_potential_energy()
        atoms.calc.reset()
        atoms.get_forces()
        assert len(calls) == 2
        atoms.calc.results.clear()
        atoms.get_forces()
        assert len(calls) == 3


class TestOptions:
    def test_the_device_is_ignored(self):
        atoms = h2()
        energy = _run(atoms, grid_spacing=0.35, device="AER_simulator")
        assert np.isfinite(energy)

    def test_an_unknown_functional_is_refused_by_the_constructor(self):
        with pytest.raises(ValueError, match="canonical XC name"):
            Mandacaru(method="dft", xc="b3lyp")

    def test_a_hydrogen_atom_runs_fully_polarized(self):
        """One electron: an empty spin-down channel everywhere -- the
        full-polarization limit of the spin functionals, end to end."""
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[6] * 3)
        atoms.calc = Mandacaru(method="dft", grid_spacing=0.35, trace=False)
        energy = atoms.get_potential_energy()
        scf = atoms.calc.result.scf
        assert np.isfinite(energy) and atoms.calc.result.success
        assert list(scf.num_particles) == [1, 0]
        assert atoms.get_magnetic_moment() == pytest.approx(1.0)


def test_a_shot_budget_is_refused_without_a_quantum_state():
    """A Kohn-Sham run has no quantum state to measure: asking for shots
    must raise instead of silently running the classical calculation."""
    with pytest.raises(ValueError, match="quantum"):
        Mandacaru(method="dft", shots=100)


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
        assert levels[3] <= mu < levels[4]                 # 8 valence electrons
        gap_center = 0.5 * (levels[3] + levels[4])
        below = energies < gap_center
        assert np.trapezoid(dos[below], energies[below]) == pytest.approx(
            8.0, abs=1e-6)

    def test_a_molecule_has_no_band_structure_or_k_mesh(self, water_szp):
        calc = water_szp.calc
        with pytest.raises(RuntimeError, match="periodic"):
            calc.band_structure()
        with pytest.raises(RuntimeError, match="periodic"):
            calc.dos(kpts=(2, 2, 2))
        assert calc.get_ibz_k_points().shape == (1, 3)

    def test_the_spectrum_needs_a_run(self):
        calc = Mandacaru(method="dft", trace=False)
        with pytest.raises(RuntimeError, match="run a calculation"):
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
    _run(atoms, xc="lda", grid_spacing=0.3, basis={"name": "PAW-LCAO", "size": "SZ"})
    return atoms


class TestUnrestricted:
    """Spin-unrestricted Kohn-Sham, selected by n_alpha != n_beta."""

    @pytest.mark.xfail(strict=True, reason=(
        "needs the half-and-half ensemble occupation of a degenerate shell, "
        "which the native engine does not have; on the exact axis the "
        "integer-occupation restricted SCF lands on a stationary point 36 eV "
        "high (HISTORY.md, 'Closed-shell O2 on the axis')"))
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
        assert np.allclose(singlet.calc.get_occupation_numbers(),
                           [2, 2, 2, 2, 2, 1, 1, 0])
        assert scf.mo_energies[6] - scf.mo_energies[5] < 1e-6
        # The density is the ensemble's: the pi* pair holds half an
        # electron of each spin in each member, whatever its gauge.
        gamma = scf.rdm()
        occupations = np.linalg.eigvalsh(gamma)
        assert np.allclose(occupations, np.sort([0, 0.5, 0.5, 1, 1, 1, 1, 1] * 2),
                           atol=1e-8)

    def test_hunds_triplet_oxygen_is_below_the_closed_shell_state(self):
        """With the bond tilted by 0.02 Angstrom the degeneracy of the pi*
        pair is lifted and the integer-occupation restricted SCF finds the
        physical closed-shell state; the triplet lies 1.31 eV below it
        (SZ, LDA)."""
        energies = {}
        for label, moments in (("closed", [0.0, 0.0]), ("triplet", [1.0, 1.0])):
            atoms = _oxygen(moments)
            atoms.positions[1, 0] += 0.02
            energies[label] = _run(atoms, xc="lda", grid_spacing=0.3,
                                   basis={"name": "PAW-LCAO", "size": "SZ"})
            assert atoms.calc.result.scf.converged
        assert -1.4 < energies["triplet"] - energies["closed"] < -1.2

    def test_the_moment_and_the_spin_channels(self, oxygen_triplet):
        calc = oxygen_triplet.calc
        scf = calc.result.scf
        assert calc.result.success
        assert list(scf.num_particles) == [7, 5]
        assert calc.get_number_of_spins() == 2
        assert oxygen_triplet.get_magnetic_moment() == pytest.approx(2.0)
        assert len(calc.get_eigenvalues(spin=1)) == len(
            calc.get_eigenvalues(spin=0))
        energies, dos = calc.dos(width=0.1)
        homo = max(calc.get_eigenvalues(spin=0)[6], calc.get_eigenvalues(spin=1)[4])
        lumo = min(calc.get_eigenvalues(spin=0)[7], calc.get_eigenvalues(spin=1)[5])
        below = energies < 0.5 * (homo + lumo)
        assert np.trapezoid(dos[below], energies[below]) == pytest.approx(
            12.0, abs=1e-6)

    def test_the_many_body_problem_is_exported_in_natural_orbitals(
            self, oxygen_triplet):
        result = oxygen_triplet.calc.result
        natural = result.natural_orbitals()
        model = result.scf.snapshot()["model_orbitals"]
        assert np.allclose(model.conj().T @ model, np.eye(len(model)), atol=1e-8)
        # The spin-summed one-body density matrix is diagonal in the exported
        # orbitals, its diagonal the natural occupations.
        gamma = np.asarray(result.scf.rdm())
        count = len(natural.occupations)
        assert np.allclose(gamma[:count, :count] + gamma[count:, count:],
                           np.diag(natural.occupations), atol=1e-7)
        assert result.num_particles == (7, 5)
        problem = result.as_quantum_problem()
        assert problem["num_particles"] == (7, 5)

    @pytest.mark.parametrize("xc", ["lda", "pbe", "r2scan"])
    def test_unrestricted_forces_are_the_derivative_of_the_energy(self, xc):
        """The OH radical (seven valence electrons, so unrestricted): the
        spin-averaged gradient plus the magnetization term, against
        Richardson-extrapolated central differences on the frozen grid.

        Its one beta pi electron has no integer aufbau solution (whichever
        member holds it rises above the empty one): the SCF was run to the
        ensemble with half an electron in each, and the forces are that
        state's."""
        atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]],
                      cell=[6, 6, 7])
        atoms.center()
        atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.3, trace=False,
                               project_translation=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        forces = atoms.get_forces()
        assert atoms.calc.result.num_particles == (4, 3)
        for atom, k in ((1, 2), (1, 0)):
            assert forces[atom, k] == pytest.approx(
                _numerical_force(atoms, atom, k), abs=1e-4)

    def test_one_diis_for_both_spins_converges_the_hydroxyl_radical(self):
        """K26 for the Kohn-Sham loop: OH (DZP, hydrogen 0.2 Angstrom off the
        axis) did not converge in 300 plain iterations with one DIIS per spin;
        one set of coefficients for both spins takes 54."""
        atoms = Atoms("OH", positions=[[0, 0, 0], [0.2, 0, 1.02]],
                      cell=[6, 6, 7])
        atoms.center()
        atoms.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.3,
                               trace=False, scf_options={"max_iterations": 100},
                               basis={"name": "PAW-LCAO", "size": "DZP"})
        atoms.get_potential_energy()
        scf = atoms.calc.result.scf
        assert scf.converged and scf.n_iterations < 100
        # The pin was -446.046312 eV, the Python engine's half-and-half pi
        # ensemble. The integer-occupation state this loop now reaches is
        # 3.77 meV lower; an all-electron LDA cross-check (PySCF, QZVP) puts the
        # integer state 36 meV below the ensemble, so the lower number is the
        # Kohn-Sham minimum and the old one was not.
        assert atoms.get_potential_energy() == pytest.approx(-446.050078,
                                                             abs=1e-4)


@pytest.fixture(scope="module")
def h2_hybrid():
    """H2 with HSE06 in a basis with more than one function per atom."""
    atoms = h2()
    _run(atoms, xc="hse06", grid_spacing=0.3, basis=NAO_DZ)
    return atoms


class TestHybrid:
    """``xc="hse06"``: generalized Kohn-Sham with short-range exact exchange."""

    def test_the_hybrid_opens_the_gap_and_lowers_the_homo(self, h2_hybrid):
        """The textbook effect: HSE06 - PBE moves the H2 HOMO down by ~1.2
        eV (an independent all-electron aug-cc-pVQZ calculation: -1.219 eV)."""
        assert h2_hybrid.calc.result.scf.snapshot()["functional"] == "hse06"
        pbe = h2()
        _run(pbe, xc="pbe", grid_spacing=0.3, basis=NAO_DZ)
        shift = (h2_hybrid.calc.get_eigenvalues()[0]
                 - pbe.calc.get_eigenvalues()[0])
        assert -1.5 < shift < -0.9
        assert _gap(h2_hybrid.calc.result.scf) > _gap(pbe.calc.result.scf)

    def test_forces_are_the_derivative_of_the_energy(self):
        """The short-range tensor's derivative is the full one's minus the
        long-range one's; against Richardson-extrapolated central
        differences on the frozen grid (H2 tilted off its axis, so both
        components are nonzero)."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.05, 0, 0.74]],
                      cell=[6] * 3)
        _run(atoms, xc="hse06", grid_spacing=0.3, basis=NAO_DZ,
             project_translation=False)
        forces = atoms.get_forces()
        result = atoms.calc.result
        assert result.energy_hartree == pytest.approx(
            to_hartree(atoms.get_potential_energy(), "eV"), abs=1e-8)
        assert forces[1, 2] == pytest.approx(_numerical_force(atoms, 1, 2),
                                             abs=1e-4)


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
#: functionals, the screened hybrid, an unrestricted run, the unitary family
#: and an all-electron basis.
SPECTRAL_FORCE_CASES = {
    "lda/paw-lcao": (lih, dict(xc="lda", grid_spacing=0.3, basis=PAW_SZP)),
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


class TestSpectralKineticForces:
    """``kinetic="spectral"`` forces (Phase N): the Pulay term contracts the
    moved basis functions against the FFT Laplacian of the unmoved ones, so
    the force is the derivative of the energy the spectral operator gives.
    Against Richardson-extrapolated central differences on the frozen grid,
    the standard of the stencil's tests above; HISTORY (2026-10-07, Phase N)
    has every component."""

    @pytest.mark.parametrize("case", sorted(SPECTRAL_FORCE_CASES))
    def test_they_are_the_derivative_of_the_energy(self, case):
        build, options = SPECTRAL_FORCE_CASES[case]
        atoms = build()
        atoms.calc = Mandacaru(method="dft", kinetic="spectral", trace=False,
                               project_translation=False, **options)
        forces = atoms.get_forces()
        numerical = _numerical_force(atoms, 1, 2)
        assert abs(numerical) > 0.1                  # a real force
        assert forces[1, 2] == pytest.approx(numerical, abs=1e-4)


class TestTheSpinOrbitDatasets:
    def test_spin_orbit_datasets_are_refused(self):
        """Their projector terms are not implemented in the molecular PAW
        engine: the run raises rather than dropping them."""
        atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[6, 6, 7])
        atoms.calc = Mandacaru(method="rhf", grid_spacing=0.4, trace=False,
                               paw_dataset="lda-fr",
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        with pytest.raises(NotImplementedError, match="spin-orbit"):
            atoms.get_potential_energy()
