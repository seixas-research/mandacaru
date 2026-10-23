# -*- coding: utf-8 -*-
# file: test/algorithms/test_spinor_dft.py

# This code is part of Mandacaru.
# MIT License

"""Two-component (spinor) Kohn-Sham behind ``Mandacaru(method="dft")`` with a
spin-orbit dataset (``paw_dataset="lda-fr"`` / ``"pbe-fr"``)."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

SZ = {"name": "PAW-LCAO", "size": "SZ"}


def _run(atoms, xc, moments, **options):
    """``atoms`` on the scalar set (no spin-orbit term) with the initial
    ``moments``: (N,) collinear, (N, 3) a noncollinear (spinor) start."""
    atoms = atoms.copy()
    atoms.set_initial_magnetic_moments(moments)
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.3,
                           trace=False, basis=SZ, **options)
    atoms.get_potential_energy()
    return atoms.calc.result.scf


def _oh():
    atoms = Atoms("OH", positions=[[0, 0, 0], [0.08, 0, 1.02]],
                  cell=[6, 6, 7])
    atoms.center()
    return atoms


class TestTheCollinearLimit:
    """Without a spin-orbit term the spinor problem contains the collinear
    one: started anywhere on its orbit under spin rotations it lands on the
    collinear energy."""

    @pytest.mark.parametrize("xc", ["lda", "pbe"])
    def test_an_open_shell_turned_to_x_is_the_unrestricted_energy(self, xc):
        """OH (seven valence electrons, its beta pi half filled in each
        member): the moment vector (1, 0, 0) on O turns the UKS density's
        moment from z to x."""
        uks = _run(_oh(), xc, [1.0, 0.0])
        spinor = _run(_oh(), xc, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert spinor.converged and not spinor.spin_orbit
        assert spinor.scalar_energy == pytest.approx(uks.electronic_energy,
                                                     abs=1e-10)
        assert spinor.electronic_energy == pytest.approx(
            uks.electronic_energy, abs=1e-8)
        assert spinor.spin_moment == pytest.approx([1.0, 0.0, 0.0],
                                                   abs=1e-6)
        assert spinor.spin_orbit_energy is None

    @pytest.mark.parametrize("xc", ["lda", "pbe"])
    def test_a_closed_shell_is_the_restricted_energy(self, xc):
        """Water with zero moment vectors: the spinor SCF from the RKS
        density stays on it -- Kramers-paired, no moment."""
        from ase.build import molecule

        atoms = molecule("H2O")
        atoms.center(vacuum=2.5)
        rks = _run(atoms, xc, [0.0] * 3)
        spinor = _run(atoms, xc, np.zeros((3, 3)))
        assert spinor.converged
        assert spinor.scalar_energy == pytest.approx(rks.electronic_energy,
                                                     abs=1e-10)
        assert spinor.electronic_energy == pytest.approx(
            rks.electronic_energy, abs=1e-8)
        assert spinor.kramers_pairing < 1e-8
        assert np.max(np.abs(spinor.spin_moment)) < 1e-8


def _triangle(R=2.5):
    """Equilateral H3 of side ``R`` (Angstrom) in the xy plane."""
    positions = R * np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    atoms = Atoms("H3", positions=positions, cell=[R + 6, R + 6, 6])
    atoms.center()
    return atoms


class TestANoncollinearStart:
    """(N, 3) initial moments turn each atom's share of the collinear
    magnetization to its vector: the only way to a noncollinear state, since
    without a spin-orbit term the collinear solution is a stable fixed point
    of the spinor SCF."""

    def test_a_frustrated_triangle_takes_the_120_degree_state(self):
        """Equilateral H3 at R = 2.5 Angstrom (LDA, SZ, grid 0.3): the
        coplanar 120-degree state is 37.83 meV below the UKS doublet, its
        site moments 0.912 Bohr magneton at 120 degrees, the total zero.
        The UKS start never leaves the collinear state.  The tolerance on
        the lowering (0.2 meV) is far above the SCF's (1e-8 Ha) and
        round-off; the total moment is 2e-4 (the cubic grid breaks the
        threefold axis), the site angles 120 to 0.01 degrees."""
        # This numerical pin describes the sampled operator and its SCF basin.
        uks = _run(_triangle(), "lda", [1.0, 0.0, 0.0], two_center="grid")
        angles = 2 * np.pi * np.arange(3) / 3
        spinor = _run(_triangle(), "lda",
                      np.c_[np.cos(angles), np.sin(angles), np.zeros(3)],
                      two_center="grid")
        assert spinor.converged
        lowering = (spinor.electronic_energy
                    - uks.electronic_energy) * HARTREE_TO_EV * 1e3
        assert lowering == pytest.approx(-37.83, abs=0.2)
        assert np.linalg.norm(spinor.spin_moment) < 1e-3
        # Site moments from the Loewdin-basis spin blocks (one function per
        # H in SZ).
        C = np.asarray(spinor.mo_coefficients)[:, :3]
        D = C @ C.conj().T
        sites = np.array([[2 * D[i, 3 + i].real, -2 * D[i, 3 + i].imag,
                           (D[i, i] - D[3 + i, 3 + i]).real]
                          for i in range(3)])
        assert np.linalg.norm(sites, axis=1) == pytest.approx([0.912] * 3,
                                                              abs=2e-3)
        assert np.max(np.abs(sites[:, 2])) < 1e-8
        for i in range(3):
            cosine = sites[i] @ sites[(i + 1) % 3] / np.linalg.norm(
                sites[i]) / np.linalg.norm(sites[(i + 1) % 3])
            assert np.degrees(np.arccos(cosine)) == pytest.approx(120.0,
                                                                  abs=0.05)
        assert spinor.noncollinear_energy < 0

    def test_the_run_log_says_how_it_started(self, tmp_path):
        import ast

        from mandacaru.utils.logging import parse_output

        path = str(tmp_path / "h3.txt")
        angles = 2 * np.pi * np.arange(3) / 3
        moments = np.c_[np.cos(angles), np.sin(angles), np.zeros(3)]
        spinor = _run(_triangle(), "lda", moments, txt=path)
        parsed = parse_output(path)
        setup, summary = parsed["setup"], parsed["summary"]
        assert "initial moment vector" in setup["scf_method"]
        assert setup["spin_orbit_coupling"] == "False"
        assert "spin_orbit_energy_eV" not in summary
        assert float(summary["noncollinear_energy_eV"]) == pytest.approx(
            spinor.noncollinear_energy * HARTREE_TO_EV, abs=1e-9)
        assert np.allclose(ast.literal_eval(
            parsed["system"]["initial_magnetic_moments"]), moments)

    def test_other_methods_refuse_moment_vectors(self):
        atoms = _triangle()
        atoms.set_initial_magnetic_moments(np.eye(3))
        atoms.calc = Mandacaru(method="uhf", grid_spacing=0.4, basis=SZ,
                               trace=False)
        with pytest.raises(NotImplementedError, match="only method='dft'"):
            atoms.get_potential_energy()

    def test_forces_are_refused(self):
        atoms = _oh()
        atoms.set_initial_magnetic_moments([[0.0, 0.0, 1.0], [0.0] * 3])
        atoms.calc = Mandacaru(method="dft", grid_spacing=0.3, basis=SZ,
                               trace=False)
        with pytest.raises(NotImplementedError, match="collinear"):
            atoms.get_forces()


def _hi(xc="lda", moments=None, **options):
    atoms = Atoms("HI", positions=[[0, 0, 0], [0, 0, 1.609]], cell=[9.0] * 3)
    atoms.center()
    if moments is not None:
        atoms.set_initial_magnetic_moments(moments)
    options.setdefault("paw_dataset", f"{xc}-fr")
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.25,
                           basis=SZ, trace=False, **options)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def hydrogen_iodide(tmp_path_factory):
    """HI in the LDA spin-orbit set (SZ: 18 electrons in 10 spatial
    functions, the iodine 4d semicore included), with its run log."""
    path = str(tmp_path_factory.mktemp("spinor") / "hi.txt")
    return _hi(txt=path), path


class TestHydrogenIodide:
    """A closed-shell molecule with spin-orbit coupling, end to end."""

    def test_the_occupied_spinors_come_in_kramers_pairs(self, hydrogen_iodide):
        atoms, _path = hydrogen_iodide
        scf = atoms.calc.result.scf
        assert atoms.calc.result.success
        assert scf.n_electrons == 18 and scf.mo_energies.shape == (20,)
        # Measured 1.8e-15 Ha: time reversal is not imposed anywhere.
        assert scf.kramers_pairing < 1e-8
        levels = scf.mo_energies
        assert np.max(np.abs(levels[1::2] - levels[0::2])) < 1e-8
        assert np.max(np.abs(scf.spin_moment)) < 1e-8
        assert atoms.calc.get_total_magnetic_moment() == pytest.approx(
            0.0, abs=1e-8)

    def test_without_the_spin_orbit_term_it_is_the_restricted_energy(
            self, hydrogen_iodide):
        """The spin-orbit run's ``scalar_energy`` is its collinear start;
        on the scalar set, where the spinor problem has no spin-orbit term,
        the spinor SCF from that RKS density (zero moment vectors) stays on
        the RKS energy."""
        atoms, _path = hydrogen_iodide
        scf = atoms.calc.result.scf
        assert scf.spin_orbit and scf.noncollinear_energy is None
        rks = _hi(paw_dataset="lda-sr").calc.result.scf
        free = _hi(paw_dataset="lda-sr", moments=np.zeros((2, 3))
                   ).calc.result.scf
        assert free.scalar_energy == pytest.approx(rks.electronic_energy,
                                                   abs=1e-10)
        assert free.electronic_energy == pytest.approx(free.scalar_energy,
                                                       abs=1e-8)

    @pytest.mark.parametrize("xc", ["lda", "pbe"])
    def test_the_spin_orbit_lowering_against_the_spinor_mean_field(
            self, xc, hydrogen_iodide):
        """The spin-orbit term lowers the closed shell in second order: by
        53.5 meV (LDA, lda-fr) and 50.5 meV (PBE, pbe-fr) in Kohn-Sham,
        against 46.6 and 42.8 meV from GHF+SO below the RHF determinant of
        the same integrals -- the same sign and size."""
        atoms = hydrogen_iodide[0] if xc == "lda" else _hi(xc)
        scf = atoms.calc.result.scf
        integrals = atoms.calc.solver._gradient_context["integrals"]
        lowering = scf.spin_orbit_energy * HARTREE_TO_EV
        assert -0.1 < lowering < -0.03
        mean_field = (integrals.generalized_hartree_fock(9, 9).electronic_energy
                      - integrals.hartree_fock(18).electronic_energy
                      ) * HARTREE_TO_EV
        assert 0.8 < lowering / mean_field < 1.4

    def test_rvv10_runs_with_the_spin_orbit_term(self):
        """``vdw="rvv10"`` on the spin-orbit set: the nonlocal correlation
        sees the total density, so it adds its energy to PBE's and leaves the
        spin-orbit lowering where PBE puts it (within a few meV)."""
        plain = _hi("pbe").calc.result.scf
        atoms = _hi("pbe", vdw="rvv10")
        scf = atoms.calc.result.scf
        assert atoms.calc.solver.xc == "pbe+rvv10"
        assert scf.spin_orbit and scf.converged
        assert scf.kramers_pairing < 1e-8
        assert scf.electronic_energy != pytest.approx(
            plain.electronic_energy, abs=1e-6)
        lowering = scf.spin_orbit_energy * HARTREE_TO_EV
        reference = plain.spin_orbit_energy * HARTREE_TO_EV
        assert -0.1 < lowering < -0.03
        assert lowering == pytest.approx(reference, abs=5e-3)

    def test_the_reference_determinant_is_the_exported_problem_s(
            self, hydrogen_iodide):
        """The Poisson route's determinant energy (one solve per spinor pair
        for the exchange) equals the spinor tensor's, and the exported
        integrals give it back in mode order."""
        from mandacaru.algorithms import GHF

        atoms, _path = hydrogen_iodide
        result = atoms.calc.result
        scf = result.scf
        integrals = atoms.calc.solver._gradient_context["integrals"]
        M = integrals.n_orbitals
        h = np.zeros((2 * M, 2 * M), dtype=complex)
        h[:M, :M] = h[M:, M:] = integrals.one_body()
        h += integrals.spin_orbit_matrix()
        tensor = GHF(h, integrals.two_body(), 18).energy_of(scf.mo_coefficients)
        assert scf.reference_energy == pytest.approx(tensor, abs=1e-9)
        # The reference fills the first 9 modes of each half.
        modes = list(range(9)) + list(range(M, M + 9))
        g = scf.eri_mo
        energy = (sum(np.real(scf.h_mo[p, p]) for p in modes)
                  + 0.5 * sum(np.real(g[p, q, p, q] - g[p, q, q, p])
                              for p in modes for q in modes))
        assert energy == pytest.approx(scf.reference_energy, abs=1e-9)
        assert result.num_particles == (9, 9)
        assert result.model_orbitals.shape == (2 * M, 2 * M)

    def test_one_particle_pictures_come_from_the_spinors(self,
                                                         hydrogen_iodide):
        atoms, _path = hydrogen_iodide
        calc = atoms.calc
        assert calc.get_number_of_spins() == 1
        assert len(calc.get_eigenvalues()) == 20
        energies, dos = calc.dos(width=0.1)
        below = energies < calc.get_fermi_level()
        assert np.trapezoid(dos[below], energies[below]) == pytest.approx(
            18.0, abs=1e-6)
        _axis, pdos = calc.pdos(width=0.1)
        assert np.allclose(sum(pdos.values()), dos, atol=1e-8)
        gamma = calc.solver.mean_field_rdm()
        assert np.real(np.trace(gamma)) == pytest.approx(18.0, abs=1e-10)
        assert sum(calc.get_charges(method="hirshfeld")) == pytest.approx(
            0.0, abs=1e-3)

    def test_the_run_log_and_the_citations(self, hydrogen_iodide):
        from mandacaru.utils.logging import parse_output

        atoms, path = hydrogen_iodide
        parsed = parse_output(path)
        setup, summary = parsed["setup"], parsed["summary"]
        assert "two-component" in setup["scf_method"]
        assert setup["spin_orbit_coupling"] == "True"
        scf = atoms.calc.result.scf
        assert float(summary["spin_orbit_energy_eV"]) == pytest.approx(
            scf.spin_orbit_energy * HARTREE_TO_EV, abs=1e-9)
        assert float(summary["kramers_pairing_Hartree"]) < 1e-8
        assert "spin_moment_bohr_magneton" in summary
        assert {"Kubler1988", "KohnSham1965"} <= set(
            atoms.calc.citation_keys())

    def test_forces_are_refused(self, hydrogen_iodide):
        atoms, _path = hydrogen_iodide
        with pytest.raises(NotImplementedError, match="two-component"):
            atoms.get_forces()


def test_an_open_shell_heavy_atom_holds_its_electron_in_p_one_half():
    """Tl 6p1 from the polarized UKS start: the one 6p electron settles in
    a 6p1/2 spinor, whose spin is -1/3 of its m_j = 1/2 -- the moment is
    1/3 Bohr magneton (0.33333), not the collinear 1 -- and the energy
    falls by about two thirds of the 6p splitting (633 meV, LDA, SZ).  That
    is the spin moment 2<S> only: the total <L + 2S> of p1/2 is -1/3."""
    atom = Atoms("Tl", positions=[[0, 0, 0]], cell=[9.0] * 3)
    atom.center()
    atom.calc = Mandacaru(method="dft", grid_spacing=0.25,
                          paw_dataset="lda-fr", basis=SZ, trace=False)
    atom.get_potential_energy()
    scf = atom.calc.result.scf
    assert atom.calc.result.success and scf.occupations is None
    assert np.linalg.norm(scf.spin_moment) == pytest.approx(1 / 3,
                                                              abs=1e-3)
    assert -0.75 < scf.spin_orbit_energy * HARTREE_TO_EV < -0.5


#: The 5p spin-orbit splitting eps(5p3/2) - eps(5p1/2) of the Dirac radial
#: Kohn-Sham atom of xenon (Hartree), as the spin-orbit sets' generator
#: solves it: ``solve_atom(54, points=generation_points(54), r_max=30.0,
#: tolerance=REFERENCE_TOLERANCE, mixing=REFERENCE_MIXING, xc=xc,
#: relativity="dirac").spin_orbit_splitting(5, 1)`` (about 100 s each, so
#: recomputed only by the slow test below).
DIRAC_XE_5P = {"lda": 0.046334333251477566, "pbe": 0.04499090050584015}


@pytest.mark.parametrize("xc", ["lda", "pbe"])
def test_the_xenon_5p_splitting_is_the_dirac_atom_s(xc):
    """Xe (closed 5p shell, so the 3D ensemble is the spherical Dirac atom's
    j-j configuration): -1.1 % (LDA) and -1.0 % (PBE) in SZ at
    grid_spacing 0.25 Angstrom; LDA SZ is -1.38/-1.10/-0.88 % at 0.3/0.25/
    0.2, extrapolating to -0.5 %, the 0.4 % the dataset's own molecular
    operator is short for the 5p (its j-averaged scalar channel).  DZP is
    -0.79 % at 0.2 (its +1.4 % at 0.25 is a grid artifact of the second
    zeta).  In the cubic
    box the p levels must stay degenerate (j = 1/2 and 3/2 are irreducible
    in the cubic double group), to round-off."""
    atom = Atoms("Xe", positions=[[0, 0, 0]], cell=[9.0] * 3)
    atom.center()
    atom.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.25,
                          paw_dataset=f"{xc}-fr", basis=SZ, trace=False)
    atom.get_potential_energy()
    scf = atom.calc.result.scf
    p = scf.mo_energies[:scf.n_electrons][-6:]
    assert p[1] - p[0] < 1e-10 and p[5] - p[2] < 1e-10
    splitting = p[2:].mean() - p[:2].mean()
    assert splitting == pytest.approx(DIRAC_XE_5P[xc], rel=0.02)


@pytest.mark.slow
def test_the_dirac_reference_splitting():
    from mandacaru.basis.atomic_solver import solve_atom
    from mandacaru.pseudopotentials.paw import generation_points
    from mandacaru.pseudopotentials.partial_waves import (REFERENCE_MIXING,
                                                          REFERENCE_TOLERANCE)

    atom = solve_atom(54, points=generation_points(54), r_max=30.0,
                      tolerance=REFERENCE_TOLERANCE, mixing=REFERENCE_MIXING,
                      xc="lda", relativity="dirac")
    assert atom.spin_orbit_splitting(5, 1) == pytest.approx(
        DIRAC_XE_5P["lda"], rel=1e-8)


@pytest.mark.parametrize("xc", [
    "r2scan", pytest.param("hse06", marks=pytest.mark.slow)])
def test_a_meta_gga_or_hybrid_molecule_has_the_collinear_limit(xc):
    """OH on the scalar set: its UKS moment turned to x or y by (N, 3)
    moments is the unrestricted energy -- r2SCAN with the spinors' kinetic-energy
    density matrix along the axis, HSE06 with the exchange of every spin
    block (LDA/PBE: TestTheCollinearLimit)."""
    uks = _run(_oh(), xc, [1.0, 0.0])
    for turn in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):     # real and complex U
        spinor = _run(_oh(), xc, [turn, [0.0, 0.0, 0.0]])
        assert spinor.converged
        assert spinor.electronic_energy == pytest.approx(
            uks.electronic_energy, abs=1e-8)
        assert spinor.spin_moment == pytest.approx(turn, abs=1e-6)
        if xc == "hse06":
            assert spinor.exact_exchange_energy == pytest.approx(
                uks.exact_exchange_energy, abs=1e-8)


class TestRefusals:
    def test_the_reduced_parity_mapping_is_refused(self):
        atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]],
                      cell=[6, 6, 7])
        atoms.calc = Mandacaru(method="dft", grid_spacing=0.4,
                               mapping="parity_reduced", paw_dataset="lda-fr",
                               basis=SZ, trace=False)
        with pytest.raises(ValueError, match="alpha-parity"):
            atoms.get_potential_energy()


# --------------------------------------------------------------------------- #
# Crystals: two-component Bloch states.
# --------------------------------------------------------------------------- #

#: Diamond Si's basis for the crystal tests: SZ, filtered at 200 eV for the
#: coarse grid (h = 0.34 Angstrom), as test_periodic_dft.py's silicon.
SILICON_BASIS = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}


def _silicon(xc="lda", moments=None, kpts=(2, 2, 2), **options):
    """Diamond Si on the spin-orbit set of ``xc`` (by default), 2x2x2."""
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43)
    if moments is not None:
        atoms.set_initial_magnetic_moments(moments)
    options.setdefault("paw_dataset", "lda-fr" if xc == "lda" else "pbe-fr")
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.34,
                           kpts={"size": kpts, "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.01},
                           basis=SILICON_BASIS, trace=False, **options)
    atoms.get_potential_energy()
    return atoms


def _oxygen_box(moments, xc="lda", **options):
    """O2 in a periodic box at Gamma (a spin crystal), on the scalar set of
    ``xc`` unless ``options`` name another."""
    from ase.build import molecule

    atoms = molecule("O2")
    atoms.set_cell([7.0, 7.0, 7.5])
    atoms.center()
    atoms.pbc = True
    atoms.set_initial_magnetic_moments(None)
    atoms.set_initial_magnetic_moments(moments)
    atoms.calc = Mandacaru(method="dft", xc=xc, grid_spacing=0.34,
                           smearing={"method": "fermi-dirac", "width": 0.01},
                           basis=SILICON_BASIS, trace=False, **options)
    atoms.get_potential_energy()
    return atoms


def _without_the_spin_orbit_term(monkeypatch):
    """Zero every spin-orbit coupling of the datasets: the spinor crystal
    keeps its two-component operator with a zero term."""
    from mandacaru.pseudopotentials import paw

    blocks = paw.paw_spin_orbit_blocks
    monkeypatch.setattr(paw, "paw_spin_orbit_blocks", lambda *args: {
        key: 0.0 * value for key, value in blocks(*args).items()})


@pytest.fixture(scope="module")
def spinor_silicon(tmp_path_factory):
    """Diamond Si on ``lda-fr``, with its run log."""
    path = str(tmp_path_factory.mktemp("spinor_crystal") / "si.txt")
    return _silicon(txt=path), path


class TestASpinOrbitCrystal:
    """Diamond Si on the LDA spin-orbit set: a non-magnetic insulator in
    Bloch spinors on the irreducible wedge."""

    def test_it_is_solved_in_spinors_on_the_wedge(self, spinor_silicon):
        atoms, _path = spinor_silicon
        calc, scf = atoms.calc, atoms.calc.result.scf
        assert scf.converged and scf.spin_orbit and scf.time_reversal
        assert calc.get_number_of_spins() == 1
        # 8 spatial functions: 16 spinor bands of one electron each.
        assert calc.get_eigenvalues(kpt=0).shape == (16,)
        assert len(calc.get_ibz_k_points()) < 8
        for f in scf.occupations:
            assert np.max(f) <= 1.0 + 1e-12
            assert np.sum(f) == pytest.approx(8.0, abs=1e-8)
        assert np.allclose(scf.spin_moment, 0.0)
        assert calc.get_total_magnetic_moment() == 0.0
        # Second order for a closed shell: a fraction of a meV for Si.
        assert -1e-3 < scf.spin_orbit_energy * HARTREE_TO_EV < 0.0
        assert scf.terms["spin_orbit"] < 0.0

    def test_bands_at_a_time_reversal_point_are_kramers_pairs(
            self, spinor_silicon):
        atoms, _path = spinor_silicon
        gamma = atoms.calc.get_eigenvalues(kpt=0)
        assert np.allclose(gamma[0::2], gamma[1::2], atol=1e-10)

    def test_bands_are_even_in_k(self, spinor_silicon):
        atoms, _path = spinor_silicon
        solver = atoms.calc.solver._periodic_solver
        k = atoms.calc.solver._gradient_context["integrals"] \
            .cartesian_kpoints([[0.13, 0.27, -0.31]])
        plus, _ = solver.bands(k)
        minus, _ = solver.bands(-k)
        assert np.allclose(plus, minus, atol=1e-12)

    def test_the_valence_top_splits_off_a_kramers_pair(self, spinor_silicon):
        """At Gamma the p-like valence top (the scalar triplet) is the
        quartet Gamma8 above the pair Gamma7: 58 meV for this coarse SZ
        silicon (DZP h 0.15 A: HISTORY "Phase P step 2")."""
        atoms, _path = spinor_silicon
        top = np.sort(atoms.calc.get_eigenvalues(kpt=0))[2:8]
        assert np.ptp(top[2:]) < 1e-5 and np.ptp(top[:2]) < 1e-10
        assert 0.04 < top[2] - top[1] < 0.08

    def test_the_spectra_count_one_electron_per_band(self, spinor_silicon):
        atoms, _path = spinor_silicon
        calc = atoms.calc
        energies, dos = calc.dos(width=0.05)
        assert np.trapezoid(dos, energies) == pytest.approx(16.0, abs=1e-6)
        _e, pdos = calc.pdos(width=0.05, energies=energies)
        assert np.allclose(sum(pdos.values()), dos, atol=1e-10)
        # The tetrahedron count: the eight valence electrons below the
        # gap, none in it (4x4x4, unfolded from the wedge).
        energies, dos = calc.dos(kpts=(4, 4, 4), integration="tetrahedron",
                                 npoints=4001)
        step = energies[1] - energies[0]
        below = energies <= calc.get_fermi_level()
        assert np.sum(dos) * step == pytest.approx(16.0, abs=1e-8)
        assert np.sum(dos[below]) * step == pytest.approx(8.0, abs=1e-8)

    def test_a_fermi_surface_holds_every_spinor_band(self, spinor_silicon):
        atoms, _path = spinor_silicon
        calc = atoms.calc
        energies = calc.fermi_surface((2, 2, 2))
        assert energies.shape == (16, 2, 2, 2)
        assert np.allclose(energies[:, 0, 0, 0], calc.get_eigenvalues(kpt=0),
                           atol=1e-9)

    def test_charges_come_from_the_traced_density(self, spinor_silicon):
        atoms, _path = spinor_silicon
        partition = atoms.calc.atomic_partition("hirshfeld")
        assert partition.grid_electrons == pytest.approx(8.0, abs=1e-8)
        assert partition.charges.sum() == pytest.approx(0.0, abs=1e-8)
        assert np.allclose(partition.magnetic_moments, 0.0)

    def test_the_run_log(self, spinor_silicon):
        from mandacaru.utils.logging import parse_output, reset_log

        atoms, path = spinor_silicon
        reset_log(path)
        parsed = parse_output(path)
        setup, summary = parsed["setup"], parsed["summary"]
        scf = atoms.calc.result.scf
        assert "two-component" in setup["scf_method"]
        assert setup["spin_orbit_coupling"] == "True"
        assert setup["spin"] == "two-component, non-magnetic"
        assert float(summary["spin_orbit_energy_eV"]) == pytest.approx(
            scf.spin_orbit_energy * HARTREE_TO_EV, abs=1e-9)
        assert summary["spin_moment_bohr_magneton"] == \
            "(0.000000, 0.000000, 0.000000)"
        assert "magnetic_moment_bohr_magneton" not in summary
        assert len(parsed["scf_iterations"]) == scf.n_iterations

    def test_what_it_cannot_do_is_refused(self, spinor_silicon, tmp_path):
        atoms, _path = spinor_silicon
        calc = atoms.calc
        with pytest.raises(NotImplementedError, match="two-component"):
            calc.get_forces(atoms)
        with pytest.raises(NotImplementedError, match="two-component"):
            calc.get_stress(atoms)
        with pytest.raises(NotImplementedError, match="spinor"):
            calc.write_cube(tmp_path / "density.cube")
        with pytest.raises(NotImplementedError, match="spinor"):
            calc.projectabilities({"Si": "s"})
        with pytest.raises(NotImplementedError, match="energy windows"):
            calc.wannier(windows="auto", guess={"Si": "sp3"})

    def test_the_polarization_counts_one_electron_per_band(
            self, spinor_silicon):
        """Centrosymmetric silicon: zero on the branch nearest zero (2e-6
        C/m^2: the grid is not quite inversion symmetric), its quantum
        e a_j / Omega (one electron per spinor band, not two)."""
        atoms, _path = spinor_silicon
        P = atoms.calc.get_polarization()
        assert np.allclose(P, 0.0, atol=1e-5)
        result = atoms.calc.polarization_result
        lattice = atoms.calc.solver._periodic_solver.crystal.lattice
        from mandacaru.algorithms.berry_phase import E_PER_BOHR2_IN_C_PER_M2
        volume = abs(np.linalg.det(lattice))
        assert np.allclose(result.quanta, lattice.T / volume
                           * E_PER_BOHR2_IN_C_PER_M2, atol=1e-12)


@pytest.fixture(scope="module")
def silicon_pair():
    """Si on ``lda-fr`` twice: as a collinear crystal (its spin-orbit
    couplings dropped -- the term absent) and as a spinor one with the term
    present but zero.  Their operators are the same; the spinor crystal
    stops at its first step."""
    from mandacaru.pseudopotentials import paw

    blocks = paw.paw_spin_orbit_blocks
    out = []
    for replacement in (lambda *args: {}, lambda *args: {
            key: 0.0 * value for key, value in blocks(*args).items()}):
        paw.paw_spin_orbit_blocks = replacement
        try:
            out.append(_silicon())
        finally:
            paw.paw_spin_orbit_blocks = blocks
    return tuple(out)


class TestSpinorAnalysesOfACrystal:
    """Wannier functions and the Berry-phase polarization of spinor Bloch
    states: without the spin-orbit term, two copies of the collinear ones."""

    def test_spinor_wannier_functions_are_two_collinear_copies(
            self, silicon_pair):
        collinear, spinor = silicon_pair
        assert not collinear.calc.solver._spinor_crystal()
        assert spinor.calc.result.scf.n_iterations == 1
        single = collinear.calc.wannier()
        double = spinor.calc.wannier()
        assert double.spinor and double.n_functions == 2 * single.n_functions
        assert double.labels[:2] == ("s@0:up", "s@0:down")
        assert np.allclose(double.spreads, np.repeat(single.spreads, 2),
                           atol=1e-9)
        assert np.allclose(double.centers, np.repeat(single.centers, 2,
                                                     axis=0), atol=1e-9)
        assert double.omega_invariant == pytest.approx(
            2.0 * single.omega_invariant, abs=1e-9)

    def test_the_berry_phase_is_twice_the_collinear_one(self, silicon_pair):
        """N spinor bands of one electron: the phase of each string is that
        of the doubly occupied bands counted twice (modulo 2 pi), and the
        polarization the same modulo the spinor quantum."""
        collinear, spinor = silicon_pair
        collinear.calc.get_polarization()
        spinor.calc.get_polarization()
        one = collinear.calc.polarization_result
        two = spinor.calc.polarization_result
        difference = np.angle(np.exp(1j * (two.phases - 2.0 * one.phases)))
        assert np.allclose(difference, 0.0, atol=1e-9)
        change = np.linalg.solve(two.quanta.T, (two.electronic + two.ionic)
                                 - (one.electronic + one.ionic))
        assert np.allclose(change - np.round(change), 0.0, atol=1e-9)

    def test_spinor_wannier_functions_with_the_term(self, spinor_silicon,
                                                    tmp_path):
        """The SOC crystal's valence: eight spinor functions whose
        Hamiltonian gives back the spinor bands on the mesh -- the split-off
        pair at Gamma included -- and whose two components and density are
        written as volumetric files."""
        atoms, _path = spinor_silicon
        calc = atoms.calc
        result = calc.wannier()
        assert result.n_functions == 8
        for kpt, k in enumerate(calc.get_ibz_k_points()):
            assert np.allclose(result.interpolate([k])[0],
                               calc.get_eigenvalues(kpt=kpt)[:8], atol=1e-8)
        top = result.interpolate([[0.0, 0.0, 0.0]])[0][2:]
        assert top[2] - top[1] == pytest.approx(
            np.sort(calc.get_eigenvalues(kpt=0))[4]
            - np.sort(calc.get_eigenvalues(kpt=0))[3], abs=1e-8)
        assert top[2] - top[1] > 0.04
        density = result.write(0, tmp_path / "w0.cube", part="density")
        up = result.write(0, tmp_path / "w0_up.cube", component="up")
        assert density.endswith(".cube") and up.endswith(".cube")
        with pytest.raises(ValueError, match="component"):
            result.write(0, tmp_path / "w0.xsf")
        with pytest.raises(NotImplementedError, match="spinor"):
            result.downfold()


@pytest.mark.slow
def test_a_polar_spin_orbit_crystal_polarizes_continuously():
    """Zinc-blende AlSb on ``lda-fr`` (SZ, h 0.3 A, 4^3): moving Al by
    +-0.01 A along x changes the polarization continuously, by the Born
    charge Z*_xx = 1.898 (1.8995 without the spin-orbit term, HISTORY
    "Phase P step 2b"; the term moves it by 0.1 %)."""
    from ase.build import bulk

    from mandacaru.algorithms.berry_phase import E_PER_BOHR2_IN_C_PER_M2
    from mandacaru.units import ANGSTROM_TO_BOHR

    raw = []
    for shift in (0.01, -0.01):
        atoms = bulk("AlSb", "zincblende", a=6.136)
        atoms.positions[0, 0] += shift
        atoms.calc = Mandacaru(method="dft", grid_spacing=0.3,
                               kpts={"size": (4, 4, 4), "gamma": True},
                               smearing={"method": "fermi-dirac",
                                         "width": 0.001},
                               paw_dataset="lda-fr", trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        atoms.get_potential_energy()
        atoms.calc.get_polarization()
        raw.append(atoms.calc.polarization_result)
    change = raw[0].raw - raw[1].raw
    change -= np.round(change)
    lattice = atoms.calc.solver._periodic_solver.crystal.lattice
    Z = ((change @ raw[0].quanta)[0] / (0.02 * ANGSTROM_TO_BOHR)
         * abs(np.linalg.det(lattice)) / E_PER_BOHR2_IN_C_PER_M2)
    assert Z == pytest.approx(1.8995, abs=0.01)


class TestTheCollinearLimitOfACrystal:
    """Without its spin-orbit term, or with every moment along one axis, a
    spinor crystal is the collinear one: its first step is the collinear
    step, so it stops there, to round-off."""

    @pytest.mark.parametrize("xc", [
        "lda", "r2scan", pytest.param("pbe", marks=pytest.mark.slow),
        pytest.param("hse06", marks=pytest.mark.slow),
        pytest.param("pbe+rvv10", marks=pytest.mark.slow),
        pytest.param("r2scan+rvv10", marks=pytest.mark.slow)])
    def test_without_its_term_it_is_the_restricted_crystal(self, xc,
                                                           monkeypatch):
        """Every functional: r2SCAN's kinetic-energy density, HSE06's
        exchange (on the whole mesh) seeded from the collinear one."""
        _without_the_spin_orbit_term(monkeypatch)
        scf = _silicon(xc).calc.result.scf
        assert scf.spin_orbit and scf.n_iterations == 1
        assert scf.extrapolated_energy == pytest.approx(scf.scalar_energy,
                                                        abs=1e-10)
        assert scf.spin_orbit_energy == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.slow
    def test_a_metal_without_its_term_is_the_restricted_crystal(
            self, monkeypatch):
        """fcc Al on a 4x4x4 mesh, 0.1 eV of Fermi-Dirac smearing."""
        from ase.build import bulk

        _without_the_spin_orbit_term(monkeypatch)
        atoms = bulk("Al")
        atoms.calc = Mandacaru(method="dft", grid_spacing=0.3,
                               kpts={"size": (4, 4, 4), "gamma": True},
                               paw_dataset="lda-fr", trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        atoms.get_potential_energy()
        scf = atoms.calc.result.scf
        assert scf.band_gap is None and scf.n_iterations == 1
        assert scf.extrapolated_energy == pytest.approx(scf.scalar_energy,
                                                        abs=1e-10)

    def test_the_whole_mesh_reproduces_the_wedge(self, spinor_silicon):
        """Zero moment vectors make the crystal magnetic in form: the whole
        mesh, the magnetization a vector field.  It stays zero, and the
        energy is the wedge's (to the collinear starts' convergence)."""
        wedge = spinor_silicon[0].calc.result.scf
        atoms = _silicon(moments=np.zeros((2, 3)))
        scf = atoms.calc.result.scf
        assert not scf.time_reversal and len(scf.kpoints) == 8
        assert np.allclose(scf.spin_moment, 0.0, atol=1e-10)
        assert scf.extrapolated_energy == pytest.approx(
            wedge.extrapolated_energy, abs=1e-7)
        gamma = atoms.calc.get_eigenvalues(kpt=0)
        assert np.allclose(gamma[0::2], gamma[1::2], atol=1e-8)

    @pytest.mark.parametrize("xc", [
        "lda", "r2scan", pytest.param("pbe", marks=pytest.mark.slow),
        pytest.param("hse06", marks=pytest.mark.slow),
        pytest.param("pbe+rvv10", marks=pytest.mark.slow),
        pytest.param("r2scan+rvv10", marks=pytest.mark.slow)])
    def test_moment_vectors_along_one_axis_are_the_unrestricted_crystal(
            self, xc):
        """The O2 triplet in a box on the scalar set: along z the moment
        vectors are the collinear start itself; turned to x or y, a global
        spin rotation, the energy cannot change (SU(2) invariance without
        the spin-orbit term).  The turn to y is a complex SU(2) matrix: it
        would expose a conjugation error in the off-diagonal spin block,
        which the real turn to x cannot."""
        for vector, moment in (([0, 0, 1.0], [0, 0, 2.0]),
                               ([1.0, 0, 0], [2.0, 0, 0]),
                               ([0, 1.0, 0], [0, 2.0, 0])):
            scf = _oxygen_box([vector, vector], xc).calc.result.scf
            assert scf.converged and not scf.spin_orbit
            assert not scf.time_reversal
            assert scf.noncollinear_energy == pytest.approx(0.0, abs=1e-10)
            assert scf.spin_moment == pytest.approx(moment, abs=1e-8)


@pytest.mark.slow
def test_a_magnetic_crystal_with_spin_orbit_coupling():
    """The O2 triplet in a box on ``lda-fr`` with collinear moments: the
    spin-orbit term turns the spin moment's length by a hair, lowers the
    energy, and the crystal is solved on the whole mesh."""
    atoms = _oxygen_box([1.0, 1.0], paw_dataset="lda-fr")
    scf = atoms.calc.result.scf
    assert scf.converged and scf.spin_orbit and not scf.time_reversal
    assert scf.spin_moment[2] == pytest.approx(2.0, abs=0.01)
    assert np.allclose(scf.spin_moment[:2], 0.0, atol=1e-6)
    assert scf.spin_orbit_energy < 0.0
    assert atoms.calc.get_total_magnetic_moment() == pytest.approx(
        scf.spin_moment[2])


def test_a_large_cell_at_gamma_is_the_molecule(hydrogen_iodide):
    """HI in its 9 Angstrom box, solved as a crystal at Gamma: the spin-orbit
    lowering is the molecule's to the grid's tolerance (h 0.25 A: 1.3 meV of
    53.5; h 0.18 A: 0.09 meV, HISTORY "Phase P step 2").  The molecule
    samples the iodine projectors on its grid, the crystal on spheres; the
    molecule's converges toward the crystal's."""
    molecule = hydrogen_iodide[0].calc.result.scf
    atoms = Atoms("HI", positions=[[0, 0, 0], [0, 0, 1.609]], cell=[9.0] * 3)
    atoms.center()
    atoms.pbc = True
    atoms.calc = Mandacaru(method="dft", grid_spacing=0.25, basis=SZ,
                           paw_dataset="lda-fr", trace=False,
                           kpts={"size": (1, 1, 1), "gamma": True},
                           smearing={"method": "fermi-dirac",
                                     "width": 0.001})
    atoms.get_potential_energy()
    crystal = atoms.calc.result.scf
    assert crystal.spin_orbit_energy * HARTREE_TO_EV == pytest.approx(
        molecule.spin_orbit_energy * HARTREE_TO_EV, abs=2e-3)


class TestTheSpinorKohnShamMatrixOfACrystal:
    @pytest.mark.parametrize("xc, rel", [
        ("lda", 1e-8), ("pbe", 1e-8), ("r2scan", 1e-8),
        pytest.param("hse06", 1e-6, marks=pytest.mark.slow)])
    def test_it_is_the_derivative_of_the_energy(self, xc, rel):
        r"""``dE = sum_k w_k tr(dP_k H_k)`` with the spin-orbit term and a
        noncollinear magnetization: the field ``B_xc``, r2SCAN's Pauli
        tau term, HSE06's spinor exchange and sphere operator and ``h_SO``
        enter the matrix as the derivative of the energy they add.  (The
        hybrid's grid exchange matches to 6e-8, as the collinear crystal's
        does, 1.1e-7: the compensation terms of the pair densities.)"""
        from ase.build import bulk

        from mandacaru.algorithms.spinor_dft import SpinorPeriodicKohnSham
        from mandacaru.pseudopotentials.paw import paw_library_path
        from mandacaru.pseudopotentials.periodic_paw import build_crystal

        si = bulk("Si", "diamond", a=5.43)
        folder = paw_library_path(None, "lda" if xc == "lda" else "pbe",
                                  relativity="dirac")
        crystal, context = build_crystal(
            si, float(np.linalg.norm(si.cell[0])) / 8,
            {"size": "SZ", "filter_cutoff": 200, "paw_dataset": folder},
            kpts={"size": (2, 1, 1), "gamma": True}, full_mesh=True,
            spin_orbit=True)
        ks = SpinorPeriodicKohnSham(
            crystal, context["n_electrons"], xc,
            directions=[[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
        device, M = ks.device, crystal.M
        rng = np.random.default_rng(3)
        P = []
        for data in crystal.kpoint_data:
            X = rng.normal(size=(2 * M, 8)) + 1j * rng.normal(size=(2 * M, 8))
            metric = np.kron(np.eye(2), data.overlap)
            L = np.linalg.cholesky(X.conj().T @ metric @ X)
            C = X @ np.linalg.inv(L).conj().T
            P.append(C @ C.conj().T)
        P = np.array(P)

        def states(matrices):
            # Any decomposition P = sum f c c^+ carries the exchange.
            pairs = [np.linalg.eigh(Pk) for Pk in matrices]
            return ([V for _l, V in pairs],
                    [np.clip(lam, 0.0, None) for lam, _V in pairs])

        def energy(matrices):
            rho, q, m, tau, tau_m = ks._spinor_density(device, matrices)
            total = sum(ks.spinor_energy_terms(matrices, rho, q, m, tau,
                                               tau_m).values())
            if ks.screening is not None:
                total += ks.spinor_hybrid_terms(*states(matrices),
                                                matrices)[0]
            return total

        rho, q, m, tau, tau_m = ks._spinor_density(device, P)
        assert np.max(np.abs(m[:2])) > 1e-3          # noncollinear
        potentials = ks._spinor_potentials(rho, q, m, tau, tau_m)
        K = None
        if ks.screening is not None:
            _e, _states, K, potentials.one_center = ks.spinor_hybrid_terms(
                *states(P), P)
        H = ks.spinor_hamiltonians(device, potentials, K)
        steps, directional = [], 0.0
        for data, H_k in zip(crystal.kpoint_data, H):
            B = (rng.normal(size=H_k.shape)
                 + 1j * rng.normal(size=H_k.shape))
            steps.append(1e-3 * (B @ B.conj().T))
            directional += data.weight * float(np.real(np.sum(steps[-1]
                                                              * H_k.T)))
        steps = np.array(steps)
        eps = 1e-4
        e0, e1, e2 = (energy(P + n * eps * steps) for n in range(3))
        assert (-3 * e0 + 4 * e1 - e2) / (2 * eps) == pytest.approx(
            directional, rel=rel)


class TestCrystalRefusals:
    def test_a_finite_field_is_refused(self):
        from ase.build import bulk

        atoms = bulk("Si", "diamond", a=5.43)
        atoms.calc = Mandacaru(method="dft", grid_spacing=0.4, basis=SZ,
                               paw_dataset="lda-fr", trace=False,
                               kpts={"size": (2, 1, 1), "gamma": True},
                               electric_field=[0.0, 0.0, 1e-3])
        with pytest.raises(NotImplementedError, match="collinear crystal"):
            atoms.get_potential_energy()
