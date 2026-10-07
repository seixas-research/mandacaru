# -*- coding: utf-8 -*-
# file: test/algorithms/test_wannier.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Maximally localized Wannier functions of a crystal: isolated and
entangled bands, spin-polarized crystals, interpolation, the functions on
the grid and the downfolded many-body problem, bare and screened."""

import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io.cube import read_cube_data

from mandacaru import Mandacaru
from mandacaru.algorithms.wannier import (SpinWannierResult, bond_centers,
                                          fragment_extent, gaussian_radial,
                                          is_orbital_spec, neighbor_shells,
                                          orbital_trials, real_harmonic,
                                          resolve_fragment, resolve_guess,
                                          resolve_windows, sp3_trials,
                                          trial_values)
from mandacaru.core.mapping import Fermion
from mandacaru.integrals import reciprocal as rc
from mandacaru.units import ANGSTROM_TO_BOHR, HARTREE_TO_EV


class TestTheShells:
    @pytest.mark.parametrize("atoms", [
        bulk("Si", "diamond", a=5.43),
        bulk("Mg", "hcp", a=3.21, c=5.21)], ids=["fcc", "hexagonal"])
    def test_the_weights_complete_the_identity(self, atoms):
        lattice = np.asarray(atoms.get_cell()).T * ANGSTROM_TO_BOHR
        B = rc.reciprocal_vectors(lattice)
        vectors, weights, steps = neighbor_shells(B, (4, 4, 3))
        assert np.allclose(np.einsum("b,bi,bj->ij", weights, vectors,
                                     vectors), np.eye(3), atol=1e-10)
        # Every b comes with -b.
        assert {tuple(s) for s in steps} == {tuple(-s) for s in steps}

    def test_silicon_has_four_bonds_per_cell(self):
        centers = bond_centers(bulk("Si", "diamond", a=5.43))
        assert len(centers) == 4


def perovskite(a=3.84):
    """Cubic SrVO3: V at the origin, its octahedron of O on the axes."""
    return Atoms("SrVO3", scaled_positions=[[0.5, 0.5, 0.5], [0, 0, 0],
                                            [0.5, 0, 0], [0, 0.5, 0],
                                            [0, 0, 0.5]],
                 cell=[a, a, a], pbc=True)


def sphere_rule(n=24):
    """Points on the unit sphere and weights, exact for harmonics of
    degree below ``n``."""
    t, wt = np.polynomial.legendre.leggauss(n)
    phi = np.linspace(0.0, 2.0 * np.pi, 2 * n, endpoint=False)
    T, P = np.meshgrid(t, phi, indexing="ij")
    s = np.sqrt(1.0 - T ** 2)
    points = np.stack([np.ravel(s * np.cos(P)), np.ravel(s * np.sin(P)),
                       np.ravel(T)])
    return points, np.ravel(np.outer(wt, np.full(2 * n, np.pi / n)))


class TestTrialOrbitals:
    """Real-harmonic trial orbitals named by atom and orbital."""

    def test_the_real_harmonics_are_the_cartesian_lobes(self):
        x, y, z = np.random.default_rng(0).normal(size=(3, 7))
        r2 = x * x + y * y + z * z
        d = np.sqrt(15.0 / (4.0 * np.pi))
        p = np.sqrt(3.0 / (4.0 * np.pi))
        expected = {(1, 1): p * x, (1, -1): p * y, (1, 0): p * z,
                    (2, -2): d * x * y / np.sqrt(r2),
                    (2, -1): d * y * z / np.sqrt(r2),
                    (2, 1): d * x * z / np.sqrt(r2),
                    (2, 2): 0.5 * d * (x * x - y * y) / np.sqrt(r2),
                    (2, 0): np.sqrt(5.0 / (16.0 * np.pi))
                    * (3.0 * z * z - r2) / np.sqrt(r2)}
        for (l, m), value in expected.items():
            assert np.allclose(real_harmonic(l, m, x, y, z),
                               value / np.sqrt(r2), atol=1e-14), (l, m)

    def test_the_real_harmonics_are_orthonormal(self):
        points, weights = sphere_rule()
        lm = [(l, m) for l in range(5) for m in range(-l, l + 1)]
        S = np.array([real_harmonic(l, m, *points) for l, m in lm])
        assert np.allclose((S * weights) @ S.T, np.eye(len(lm)), atol=1e-12)

    def test_the_gaussian_radials_are_normalized(self):
        r = np.linspace(0.0, 30.0, 60001)
        for l in range(4):
            norm = np.trapezoid(gaussian_radial(l, r, 1.3) ** 2 * r * r, r)
            assert norm == pytest.approx(1.0, abs=1e-10)

    def test_orbitals_are_named_by_atom(self):
        atoms = perovskite()
        t2g = orbital_trials(atoms, {"V": "t2g"})
        assert [t.label for t in t2g] == ["V1:dxy", "V1:dyz", "V1:dxz"]
        assert all(t.atom == 1 and t.center == (0.0, 0.0, 0.0) for t in t2g)
        p = orbital_trials(atoms, [("O", "p")])
        assert [t.atom for t in p] == [2, 2, 2, 3, 3, 3, 4, 4, 4]
        assert [t.label for t in p[:3]] == ["O2:py", "O2:pz", "O2:px"]
        shells = orbital_trials(atoms, [(1, ["s", "d", "eg"]), (0, "f")])
        assert len(shells) == 1 + 5 + 2 + 7
        assert {t.angular[0][0] for t in shells[-7:]} == {3}
        site = orbital_trials(atoms, [((1.92, 1.92, 0.0), "s")])
        assert site[0].atom is None
        assert np.allclose(site[0].center, np.array([1.92, 1.92, 0.0])
                           * ANGSTROM_TO_BOHR)

    def test_a_spec_is_told_from_centers(self):
        assert is_orbital_spec({"V": "d"})
        assert is_orbital_spec([("V", "d")])
        assert is_orbital_spec([((0.0, 0.0, 0.0), "s")])
        assert is_orbital_spec([("V", ["dxy", "dxz"], np.eye(3))])
        assert not is_orbital_spec([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert not is_orbital_spec(np.zeros((2, 3)))
        assert not is_orbital_spec("bonds")

    def test_a_rotated_frame_turns_t2g_into_eg(self):
        """Rotated 45 degrees about z, the frame's dxy is the Cartesian
        dx2-y2 (negated: x'y' = (y^2 - x^2) / 2)."""
        atoms = perovskite()
        frame = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
        rotated = orbital_trials(atoms, [("V", "dxy", frame)])[0]
        plain = orbital_trials(atoms, [("V", "dx2-y2")])[0]
        offsets = np.random.default_rng(1).normal(size=(3, 50))
        assert np.allclose(trial_values(rotated, offsets),
                           -trial_values(plain, offsets), atol=1e-13)
        with pytest.raises(ValueError, match="orthogonal"):
            orbital_trials(atoms, [("V", "d", [[1, 0, 0], [1, 1, 0],
                                               [0, 0, 1]])])

    def test_hybrids_point_along_the_frame(self):
        """The sp3 set in the default frame is the four hybrids toward the
        neighbors of diamond's atom at the origin."""
        si = bulk("Si", "diamond", a=5.43)
        named = orbital_trials(si, [(0, "sp3")])
        bonded = sp3_trials(si)[:4]
        offsets = np.random.default_rng(2).normal(size=(3, 40))
        values = np.array([trial_values(t, offsets) for t in named])
        for trial in bonded:
            own = trial_values(trial, offsets)
            assert np.min(np.abs(values - own).max(axis=1)) < 1e-12
        for name, count in (("sp", 2), ("sp2", 3), ("sp3", 4)):
            trials = orbital_trials(si, [(0, name)])
            assert len(trials) == count
            # Rows of (s, px, py, pz): orthonormal hybrids, equal s shares.
            rows = np.array([[dict(((l, m), c) for l, m, c in t.angular)
                              .get(lm, 0.0) for lm in
                              ((0, 0), (1, 1), (1, -1), (1, 0))]
                             for t in trials])
            assert np.allclose(rows @ rows.T, np.eye(count), atol=1e-12)
            assert np.allclose(rows[:, 0], rows[0, 0])
            directions = rows[:, 1:] / np.linalg.norm(rows[:, 1:], axis=1,
                                                      keepdims=True)
            cosine = -1.0 / (count - 1)      # 180, 120, 109.47 degrees
            off = directions @ directions.T - np.eye(count)
            assert np.allclose(off[~np.eye(count, dtype=bool)], cosine)

    def test_bad_guesses_are_refused(self):
        atoms = perovskite()
        with pytest.raises(ValueError, match="unknown orbital"):
            orbital_trials(atoms, {"V": "t2"})
        with pytest.raises(ValueError, match="no 'Fe' atom"):
            orbital_trials(atoms, {"Fe": "d"})
        with pytest.raises(ValueError, match="trial_radial"):
            orbital_trials(atoms, {"V": "d"}, radial="slater")
        with pytest.raises(ValueError, match="off the atoms"):
            resolve_guess("bonds", bulk("Si", "diamond", a=5.43),
                          radial="basis")
        with pytest.raises(ValueError, match="no trial orbital sits"):
            orbital_trials(atoms, [((1.0, 1.0, 1.0), "s")], radial="basis")
        with pytest.raises(ValueError, match="unknown guess"):
            resolve_guess("d", atoms)


@pytest.fixture(scope="module")
def silicon():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.35, trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZP"},
                           kpts={"size": (3, 3, 3), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.001})
    atoms.get_potential_energy()
    return atoms, atoms.calc.wannier()


class TestSilicon:
    def test_four_equivalent_functions_on_the_bonds(self, silicon):
        atoms, w = silicon
        cell = np.asarray(atoms.get_cell())
        bonds = bond_centers(atoms)
        for center in w.centers:
            shift = np.linalg.solve(cell.T, (bonds - center).T).T
            shift -= np.round(shift)
            assert np.abs(shift @ cell).sum(axis=1).min() < 1e-4
        assert np.allclose(w.spreads, w.spreads[0], rtol=1e-6)
        assert w.omega_diagonal < 1e-8

    def test_the_invariant_spread_does_not_depend_on_the_guess(self,
                                                                silicon):
        atoms, w = silicon
        other = atoms.calc.wannier(guess=[[0.3, 0.1, 0.2], [1.0, 0.5, -0.3],
                                          [-0.5, 0.9, 0.1],
                                          [0.2, -0.7, 0.8]])
        assert other.omega_invariant == pytest.approx(w.omega_invariant,
                                                      rel=1e-8)
        assert other.total_spread == pytest.approx(w.total_spread, rel=1e-6)

    def test_interpolation_is_exact_on_the_mesh(self, silicon):
        """With the hoppings on the nearest replicas (the default) or on
        the cell origins' Wigner-Seitz vectors."""
        atoms, w = silicon
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        direct, _ = solver.bands(w.kpoints @ B.T)
        direct = np.asarray(direct)[:, :4] * HARTREE_TO_EV
        cells = atoms.calc.wannier(replicas="cells")
        for result in (w, cells):
            assert np.abs(result.interpolate(w.kpoints)
                          - direct).max() < 1e-8

    def test_the_centers_sum_to_the_berry_phase(self, silicon):
        """The electrons' polarization is minus the sum of the Wannier
        centers (in quanta, modulo one)."""
        atoms, w = silicon
        solver = atoms.calc.solver._periodic_solver
        crystal = solver.crystal
        lattice = np.asarray(crystal.lattice)
        atoms.calc.get_polarization()
        result = atoms.calc.polarization_result
        ionic = (np.asarray(crystal.charges)
                 @ np.linalg.solve(lattice, np.asarray(crystal.centers).T).T)
        electronic = result.raw - ionic / 2.0        # quanta 2 e a / Omega
        from_centers = -np.linalg.solve(
            lattice, (w.centers * ANGSTROM_TO_BOHR).sum(axis=0))
        difference = from_centers - electronic
        assert np.abs(difference - np.round(difference)).max() < 1e-6

    def test_the_hoppings_are_hermitian(self, silicon):
        _atoms, w = silicon
        for R, block in zip(w.lattice_vectors, w.hamiltonian_R):
            assert np.allclose(w.hamiltonian(-R), block.conj().T, atol=1e-12)

    def test_the_functions_are_real_up_to_a_phase(self, silicon, tmp_path):
        """Diamond is centrosymmetric: each function is real once its
        global phase is fixed, and is written as a cube of the supercell."""
        atoms, w = silicon
        assert max(w.imaginary_ratio(n) for n in range(4)) < 1e-6
        values, _origin, _step = w.orbital(0)
        path = w.write(0, tmp_path / "w0.cube")
        data, read = read_cube_data(path)
        assert data.shape == values.shape
        assert np.allclose(data, values.real, atol=1e-4 * np.abs(
            values).max())
        assert len(read) == len(atoms) * int(np.prod(w.size))

    def test_the_valence_fragment_holds_the_band_energy(self, silicon):
        """The four valence functions of a cell are fully occupied, so the
        downfolded Kohn-Sham one-body part traces to the band energy per
        cell, the mean over the mesh of the occupied eigenvalues."""
        atoms, w = silicon
        problem = w.downfold(coulomb="isolated")
        assert problem.n_electrons == 8
        assert np.allclose(problem.density_matrix, 2.0 * np.eye(4),
                           atol=1e-8)
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        direct, _ = solver.bands(w.kpoints @ B.T)
        band_energy = 2.0 * np.mean(np.sum(np.asarray(direct)[:, :4], axis=1))
        assert 2.0 * np.trace(problem.kohn_sham).real == pytest.approx(
            band_energy, abs=1e-9)


#: Silicon's sp3 set: the valence and the four lowest conduction bands,
#: everything up to 1 eV above the Fermi level frozen.
FROZEN_ABOVE_FERMI = 1.0


@pytest.fixture(scope="module")
def silicon_sp3(silicon):
    atoms, _valence = silicon
    fermi = atoms.calc.get_fermi_level()
    windows = {"outer": (-20.0, fermi + 20.0),
               "frozen": (-20.0, fermi + FROZEN_ABOVE_FERMI)}
    return atoms, windows, atoms.calc.wannier(8, guess="sp3",
                                              windows=windows)


class TestDisentanglement:
    def test_windows_are_checked(self):
        with pytest.raises(ValueError, match="inside the outer"):
            resolve_windows({"outer": (0.0, 5.0), "frozen": (-1.0, 2.0)})
        with pytest.raises(ValueError, match="unknown window"):
            resolve_windows({"inner": (0.0, 5.0)})

    def test_more_functions_than_bands_need_windows(self, silicon):
        atoms, _w = silicon
        with pytest.raises(ValueError, match="windows"):
            atoms.calc.wannier(8, guess="sp3")

    def test_eight_hybrids_point_along_the_bonds(self, silicon_sp3):
        atoms, _windows, w = silicon_sp3
        assert w.unitary.shape[1:] == (atoms.calc.solver._periodic_solver
                                       .crystal.M, 8)
        assert np.allclose(w.spreads, w.spreads[0], rtol=1e-5)
        bond = atoms.positions[1] - atoms.positions[0]
        offsets = np.linalg.norm(w.centers - atoms.positions[0], axis=1)
        own = np.sort(offsets)[:4]
        # Four functions on the first atom, displaced toward its neighbors.
        assert np.allclose(own, own[0], atol=1e-5)
        assert 0.0 < own[0] < 0.5 * np.linalg.norm(bond)

    def test_the_frozen_bands_are_exact_on_the_mesh(self, silicon_sp3):
        atoms, windows, w = silicon_sp3
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        direct, _ = solver.bands(w.kpoints @ B.T)
        direct = np.asarray(direct) * HARTREE_TO_EV
        interpolated = w.interpolate(w.kpoints)
        lo, hi = windows["frozen"]
        for k in range(len(direct)):
            frozen = direct[k][(direct[k] >= lo) & (direct[k] <= hi)]
            assert len(frozen) >= 4
            assert np.abs(interpolated[k][:len(frozen)]
                          - frozen).max() < 1e-8

    @pytest.mark.slow
    def test_the_invariant_spread_does_not_depend_on_the_guess(
            self, silicon_sp3):
        atoms, windows, w = silicon_sp3
        rng = np.random.default_rng(1)
        centers = (np.repeat(atoms.positions, 4, axis=0)
                   + 0.6 * (rng.random((8, 3)) - 0.5))
        other = atoms.calc.wannier(8, guess=centers, windows=windows)
        assert other.omega_invariant == pytest.approx(w.omega_invariant,
                                                      rel=1e-7)
        assert "Souza2001" in atoms.calc.citation_keys()

    def test_the_hybrids_are_real_up_to_a_phase(self, silicon_sp3):
        _atoms, _windows, w = silicon_sp3
        assert max(w.imaginary_ratio(n) for n in range(8)) < 1e-6


def bond_pair(atoms, w) -> list:
    """The two hybrids facing each other across the bond between the
    cell's two atoms."""
    start, end = atoms.positions[0], atoms.positions[1]
    return [int(np.argmin(np.linalg.norm(w.centers - (a + 0.2 * (b - a)),
                                         axis=1)))
            for a, b in ((start, end), (end, start))]


@pytest.fixture(scope="module")
def bond(silicon_sp3):
    atoms, _windows, w = silicon_sp3
    return w.downfold(bond_pair(atoms, w))


class TestTheTwoSiteModel:
    """The two hybrids of one bond: a two-site, two-electron model."""

    def test_the_two_sites_are_equivalent(self, bond):
        h, g = bond.kohn_sham, bond.two_body
        assert bond.n_electrons == 2
        assert h[0, 0].real == pytest.approx(h[1, 1].real, abs=1e-6)
        assert g[0, 0, 0, 0].real == pytest.approx(g[1, 1, 1, 1].real,
                                                   rel=1e-4)
        # On-site repulsion above the inter-site one, both positive.
        assert g[0, 0, 0, 0].real > g[0, 1, 0, 1].real > 0.0
        assert np.allclose(g, np.conj(g.transpose(2, 3, 0, 1)), atol=1e-10)

    def test_the_double_counting_restores_the_kohn_sham_fock_matrix(
            self, bond):
        gamma, g = bond.density_matrix, bond.two_body
        fock = (bond.one_body
                + np.einsum("rs,prqs->pq", gamma, g)
                - 0.5 * np.einsum("rs,prsq->pq", gamma, g))
        assert np.allclose(fock, bond.kohn_sham, atol=1e-10)

    def test_adapt_vqe_solves_the_model_exactly(self, bond):
        """Through ``Mandacaru``, ADAPT-VQE reaches the exact two-electron
        singlet of the model diagonalized in the Wannier functions."""
        H = bond.fermion_hamiltonian().to_matrix(4)
        number = [Fermion({((i, True), (i, False)): 1.0},
                          n_modes=4).to_matrix(4) for i in range(4)]
        count = np.real(np.diag(sum(number)))
        spin = np.real(np.diag(number[0] + number[1] - number[2]
                               - number[3]))
        sector = np.flatnonzero((np.abs(count - 2) < 1e-9)
                                & (np.abs(spin) < 1e-9))
        exact = np.linalg.eigvalsh(H[np.ix_(sector, sector)])[0]
        calc = Mandacaru(method="adapt-vqe", trace=False,
                         **bond.as_quantum_problem())
        result = calc.run()
        assert result.optimal_energy == pytest.approx(
            exact * HARTREE_TO_EV, abs=1e-6)


def exact_ground_state(problem, num_particles):
    """Lowest eigenvalue (Hartree) of the model in the Wannier functions in
    the ``(up, down)`` sector, by exact diagonalization."""
    n = 2 * problem.n_spatial_orbitals
    H = problem.fermion_hamiltonian().to_matrix(n)
    number = [np.real(np.diag(Fermion({((i, True), (i, False)): 1.0},
                                      n_modes=n).to_matrix(n)))
              for i in range(n)]
    half = n // 2
    up, down = sum(number[:half]), sum(number[half:])
    sector = np.flatnonzero((np.abs(up - num_particles[0]) < 1e-9)
                            & (np.abs(down - num_particles[1]) < 1e-9))
    return np.linalg.eigvalsh(H[np.ix_(sector, sector)])[0]


#: A cheap screening for the plumbing: 12 of the 18 bands, |q + G| below
#: 2 Bohr^-1 (U moves by 0.12 eV and 11 meV from the full ones).
CHEAP_SCREENING = {"screening_bands": 12, "screening_cutoff": 2.0}


@pytest.fixture(scope="module")
def screened(silicon_sp3):
    atoms, _windows, w = silicon_sp3
    pair = bond_pair(atoms, w)
    return {"bare": w.downfold(pair),
            "crpa": w.downfold(pair, screening="crpa", **CHEAP_SCREENING),
            "rpa": w.downfold(pair, screening="rpa", **CHEAP_SCREENING)}


@pytest.mark.slow
class TestScreening:
    """The two hybrids of one bond, screened by constrained RPA."""

    def test_the_screened_interaction_is_much_weaker(self, screened):
        """U falls to well under half its bare value; J, an exchange
        density without charge, falls less."""
        bare, crpa = screened["bare"].two_body, screened["crpa"].two_body
        U, U0 = crpa[0, 0, 0, 0].real, bare[0, 0, 0, 0].real
        J, J0 = crpa[0, 1, 1, 0].real, bare[0, 1, 1, 0].real
        assert 0.0 < U < 0.5 * U0
        assert 0.0 < J < J0
        assert J / J0 > U / U0
        assert 0.0 < screened["crpa"].dielectric_head < 1.0
        assert screened["bare"].dielectric_head is None
        assert np.allclose(screened["crpa"].kohn_sham,
                           screened["bare"].kohn_sham)

    def test_excluding_nothing_screens_most(self, screened):
        """RPA, where every transition screens, lies below cRPA, which
        leaves out those inside the bond's Bloch subspace."""
        assert (screened["rpa"].two_body[0, 0, 0, 0].real
                < screened["crpa"].two_body[0, 0, 0, 0].real)

    def test_the_double_counting_uses_the_screened_interaction(
            self, screened):
        problem = screened["crpa"]
        gamma, g = problem.density_matrix, problem.two_body
        fock = (problem.one_body + np.einsum("rs,prqs->pq", gamma, g)
                - 0.5 * np.einsum("rs,prsq->pq", gamma, g))
        assert np.allclose(fock, problem.kohn_sham, atol=1e-10)

    def test_the_screening_and_the_kernel_are_cited(self, screened,
                                                     silicon_sp3):
        atoms = silicon_sp3[0]
        assert {"Aryasetiawan2004", "Sasioglu2011", "Spencer2008"} <= set(
            atoms.calc.citation_keys())

    def test_unknown_screening_is_refused(self, silicon_sp3):
        _atoms, _windows, w = silicon_sp3
        with pytest.raises(ValueError, match="screening"):
            w.downfold([0], screening="gw")


class TestImages:
    """The truncated interaction against the fragment's isolated images."""

    def test_the_isolated_interaction_agrees_within_the_radius(
            self, silicon_sp3):
        atoms, _windows, w = silicon_sp3
        pair = bond_pair(atoms, w)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            plain = w.downfold(pair)
            isolated = w.downfold(pair, coulomb="isolated")
        difference = np.abs(plain.two_body - isolated.two_body).max()
        assert difference * HARTREE_TO_EV < 0.01

    def test_a_fragment_meeting_its_copies_warns(self, silicon_sp3):
        """Two bonds of one atom on a 3^3 mesh reach past the truncation
        radius; isolated, they do not."""
        atoms, _windows, w = silicon_sp3
        cell = np.asarray(w.lattice).T
        images = [(m, (i, j, k)) for m in range(8) for i in (-1, 0, 1)
                  for j in (-1, 0, 1) for k in (-1, 0, 1)]
        where = np.array([w.centers[m] + cell @ np.array(R)
                          for m, R in images])
        start = atoms.positions[0]
        neighbors = [atoms.positions[1],
                     atoms.positions[1] - atoms.get_cell()[0]]
        modes = [images[int(np.argmin(np.linalg.norm(where - point,
                                                     axis=1)))]
                 for b in neighbors for point in (start + 0.2 * (b - start),
                                                  start + 0.8 * (b - start))]
        extent = fragment_extent(w, resolve_fragment(modes, 8))
        assert extent > w._coulomb_box().radius
        with pytest.warns(UserWarning, match="periodic copies"):
            w.downfold(modes)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            w.downfold(modes, coulomb="isolated")


#: Copper's s, p and d functions: everything up to 1 eV above the Fermi
#: level frozen.
COPPER_FROZEN = 1.0


def copper_windows(atoms):
    fermi = atoms.calc.get_fermi_level()
    return {"outer": (-30.0, fermi + 80.0),
            "frozen": (-30.0, fermi + COPPER_FROZEN)}


@pytest.fixture(scope="module")
def copper():
    """fcc Cu, a metal whose d bands cross its s band: the nine s, p and d
    functions of the atom from the bands of a single-zeta basis."""
    atoms = bulk("Cu", "fcc", a=3.61)
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.2, trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZP"},
                           kpts={"size": (4, 4, 4), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.05})
    atoms.get_potential_energy()
    return atoms, atoms.calc.wannier(guess={"Cu": ["s", "p", "d"]},
                                     windows=copper_windows(atoms))


def d_functions(w) -> list:
    return [n for n, label in enumerate(w.labels) if ":d" in label]


class TestCopperDShell:
    """Trial orbitals of every angular momentum on a transition metal."""

    def test_the_functions_sit_on_the_atom(self, copper):
        atoms, w = copper
        assert w.labels == ("Cu0:s", "Cu0:py", "Cu0:pz", "Cu0:px",
                            "Cu0:dxy", "Cu0:dyz", "Cu0:dz2", "Cu0:dxz",
                            "Cu0:dx2-y2")
        assert np.abs(w.centers - atoms.positions[0]).max() < 1e-4
        spreads = w.spreads
        t2g, eg = spreads[[4, 5, 7]], spreads[[6, 8]]
        assert np.allclose(t2g, t2g[0], rtol=1e-5)
        assert np.allclose(eg, eg[0], rtol=1e-5)
        assert np.allclose(spreads[1:4], spreads[1], rtol=1e-5)
        # The d functions are the most compact, well under an s or p one.
        assert 0.2 < spreads[4:].min() <= spreads[4:].max() < 0.6
        assert spreads[4:].max() < 0.5 * spreads[:4].min()

    def test_the_frozen_bands_are_exact_on_the_mesh(self, copper):
        atoms, w = copper
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        direct = np.asarray(solver.bands(w.kpoints @ B.T)[0]) * HARTREE_TO_EV
        interpolated = w.interpolate(w.kpoints)
        lo, hi = copper_windows(atoms)["frozen"]
        for k in range(len(direct)):
            frozen = direct[k][(direct[k] >= lo) & (direct[k] <= hi)]
            assert len(frozen) >= 5
            assert np.abs(interpolated[k][:len(frozen)]
                          - frozen).max() < 1e-8

    @pytest.mark.slow
    def test_the_basis_radials_give_the_same_subspace(self, copper):
        atoms, w = copper
        other = atoms.calc.wannier(guess={"Cu": ["s", "p", "d"]},
                                   windows=copper_windows(atoms),
                                   trial_radial="basis")
        assert other.omega_invariant == pytest.approx(w.omega_invariant,
                                                      rel=1e-8)
        assert np.abs(other.centers - w.centers).max() < 1e-4

    @pytest.mark.slow
    def test_the_cell_holds_the_valence(self, copper):
        """Every occupied state lies in the frozen window, so the nine
        functions of a cell hold every electron of the mesh's states --
        Cu's eleven, up to the few meV between these eigenvalues and the
        last SCF iteration's."""
        from mandacaru.algorithms.periodic_dft import occupation

        atoms, w = copper
        problem = w.downfold()
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        energies = np.asarray(solver.bands(w.kpoints @ B.T)[0])
        fermi = float(atoms.calc.solver._scf.fermi_level)
        electrons = 2.0 * np.mean(np.sum(occupation(
            (energies - fermi) / solver.width, solver.method), axis=1))
        assert np.trace(problem.density_matrix).real == pytest.approx(
            electrons, abs=1e-8)
        assert electrons == pytest.approx(11.0, abs=0.01)
        assert problem.num_particles == (6, 5)

    def test_the_d_model(self, copper):
        """The five d functions: real integrals, U equal within the
        t2g and within the eg set, and the Fock identity."""
        _atoms, w = copper
        d = d_functions(w)
        problem = w.downfold(d, num_particles=(4, 4))
        g = problem.two_body
        assert np.abs(g.imag).max() * HARTREE_TO_EV < 1e-6
        assert np.abs(problem.kohn_sham.imag).max() * HARTREE_TO_EV < 1e-6
        U = np.real([g[i, i, i, i] for i in range(5)]) * HARTREE_TO_EV
        assert np.allclose(U[[0, 1, 3]], U[0], atol=1e-3)
        assert np.allclose(U[[2, 4]], U[2], atol=1e-3)
        assert 15.0 < U.min()
        gamma = problem.density_matrix
        fock = (problem.one_body + np.einsum("rs,prqs->pq", gamma, g)
                - 0.5 * np.einsum("rs,prsq->pq", gamma, g))
        assert np.allclose(fock, problem.kohn_sham, atol=1e-10)

    @pytest.mark.slow
    def test_the_interpolation_is_within_10_mev_on_an_8_mesh(self, copper):
        """Between the points of an 8^3 mesh the interpolated bands stay
        within 10 meV of the true ones inside the frozen window, d bands
        and the s band crossing them alike."""
        atoms, _w = copper
        windows = copper_windows(atoms)
        w = atoms.calc.wannier(guess={"Cu": ["s", "p", "d"]},
                               windows=windows, kpts=(8, 8, 8))
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        kpoints = np.random.default_rng(3).random((30, 3))
        direct = np.asarray(solver.bands(kpoints @ B.T)[0]) * HARTREE_TO_EV
        interpolated = w.interpolate(kpoints)
        lo, hi = windows["frozen"]
        for k in range(len(kpoints)):
            frozen = direct[k][(direct[k] >= lo) & (direct[k] <= hi)]
            assert np.abs(interpolated[k][:len(frozen)]
                          - frozen).max() < 0.010

    def test_the_count_of_functions_follows_the_guess(self, copper):
        atoms, _w = copper
        with pytest.raises(ValueError, match="metal"):
            atoms.calc.wannier(guess={"Cu": "d"})
        with pytest.raises(ValueError, match="5 functions from 9 bands"):
            atoms.calc.wannier(guess={"Cu": "d"}, bands=range(9))
        with pytest.raises(ValueError, match="5 trial orbitals for 6"):
            atoms.calc.wannier(6, guess={"Cu": "d"},
                               windows=copper_windows(atoms))


@pytest.fixture(scope="module")
def vanadate_crystal():
    """Cubic SrVO3 (LDA, PAW-LCAO SZP, h 0.2 A, 3^3), converged."""
    atoms = perovskite()
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.2, trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZP"},
                           kpts={"size": (3, 3, 3), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.05})
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def vanadate(vanadate_crystal):
    """The V t2g model's functions: three disentangled from a window of
    2 eV either side of the Fermi level, frozen up to 0.5 eV above it --
    the basis' V p band crosses the t2g bands at R."""
    atoms = vanadate_crystal
    fermi = atoms.calc.get_fermi_level()
    w = atoms.calc.wannier(guess={"V": "t2g"},
                           windows={"outer": (fermi - 2.0, fermi + 2.0),
                                    "frozen": (fermi - 2.0, fermi + 0.5)})
    return atoms, w


@pytest.fixture(scope="module")
def t2g_model(vanadate):
    """The bare t2g model of the cell's V."""
    return vanadate[1].downfold()


@pytest.mark.slow
class TestTheT2gModel:
    """SrVO3's three t2g functions and their bare Hubbard-Kanamori
    interaction."""

    def test_the_fermi_level_cuts_the_t2g_bands(self, vanadate_crystal):
        """d1: twelve O bands full, the Fermi level inside the next three,
        the t2g, which lie within the model's 2 eV window."""
        atoms = vanadate_crystal
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        kpoints = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]) / 3.0
        bands = (np.asarray(solver.bands(kpoints @ B.T)[0]) * HARTREE_TO_EV
                 - atoms.calc.get_fermi_level())
        assert bands[:, 11].max() < -2.0
        assert bands[:, 12].min() < 0.0 < bands[:, 14].max() < 2.0

    def test_three_equivalent_functions_on_vanadium(self, vanadate):
        atoms, w = vanadate
        assert w.labels == ("V1:dxy", "V1:dyz", "V1:dxz")
        assert np.abs(w.centers - atoms.positions[1]).max() < 1e-4
        assert np.allclose(w.spreads, w.spreads[0], rtol=1e-5)
        assert w.omega_diagonal < 1e-8

    def test_the_interaction_has_the_cubic_symmetry(self, t2g_model):
        """One U, one U' and one J on the three orbitals; the exchange and
        pair-hopping integrals agree (real functions), and J lies near
        (U - U') / 2, the relation of a spherical atom."""
        g = t2g_model.two_body
        assert np.abs(g.imag).max() * HARTREE_TO_EV < 1e-8
        g = g.real * HARTREE_TO_EV
        pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
        U = np.array([g[i, i, i, i] for i in range(3)])
        Up = np.array([g[i, j, i, j] for i, j in pairs])
        J = np.array([g[i, j, j, i] for i, j in pairs])
        pair_hopping = np.array([g[i, i, j, j] for i, j in pairs])
        assert np.allclose(U, U[0], atol=1e-3)
        assert np.allclose(Up, Up[0], atol=1e-3)
        assert np.allclose(J, J[0], rtol=0.01)
        assert np.allclose(pair_hopping, J, atol=1e-8)
        assert U[0] > Up[0] > 0.0 and 0.0 < J[0] < 0.1 * U[0]
        assert 0.9 < (U[0] - Up[0]) / (2.0 * J.mean()) < 1.2

    def test_the_model_holds_the_d_electron(self, t2g_model):
        """d1: the three functions hold the one electron of the t2g
        bands, a third each, and the Fock identity holds."""
        problem = t2g_model
        gamma = problem.density_matrix
        assert np.trace(gamma).real == pytest.approx(1.0, abs=1e-3)
        assert np.allclose(np.diag(gamma).real, 1.0 / 3.0, atol=1e-3)
        assert problem.num_particles == (1, 0)
        g = problem.two_body
        fock = (problem.one_body + np.einsum("rs,prqs->pq", gamma, g)
                - 0.5 * np.einsum("rs,prsq->pq", gamma, g))
        assert np.allclose(fock, problem.kohn_sham, atol=1e-10)


@pytest.fixture(scope="module")
def iron():
    """bcc Fe, ferromagnetic: per spin, the projected s, p and d functions
    of the atom (the trial orbitals' Loewdin gauge, ``max_iter=0``; the
    spread minimization would turn them into hybrids off the atom)."""
    atoms = bulk("Fe", "bcc", a=2.87)
    atoms.set_initial_magnetic_moments([2.5])
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.2, trace=False,
                           basis={"name": "PAW-LCAO", "size": "DZP"},
                           kpts={"size": (4, 4, 4), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.05})
    atoms.get_potential_energy()
    fermi = atoms.calc.get_fermi_level()
    w = atoms.calc.wannier(guess={"Fe": ["s", "p", "d"]}, max_iter=0,
                           windows={"outer": (-30.0, fermi + 80.0),
                                    "frozen": (-30.0, fermi + 1.0)})
    return atoms, w


def mesh_occupations(atoms, kpoints, projections=False):
    """Electrons per state on the mesh ``kpoints`` (fractional) at the SCF
    Fermi level, per spin, and with ``projections`` the Loewdin weights."""
    from mandacaru.algorithms.periodic_dft import occupation

    solver = atoms.calc.solver._periodic_solver
    B = rc.reciprocal_vectors(solver.crystal.lattice)
    energies, weights = solver.bands(kpoints @ B.T, projections=projections)
    fermi = float(atoms.calc.solver._scf.fermi_level)
    f = (2.0 / solver.n_spins) * occupation(
        (np.asarray(energies) - fermi) / solver.width, solver.method)
    return f, weights


@pytest.mark.slow
class TestIronSpin:
    """A spin-polarized d shell: the moment and the d occupations."""

    def test_the_functions_sit_on_the_atom(self, iron):
        atoms, w = iron
        assert atoms.calc.get_total_magnetic_moment() > 1.8
        assert isinstance(w, SpinWannierResult)
        assert np.abs(w.centers - atoms.positions[0]).max() < 1e-6
        d = d_functions(w)
        for spreads in w.spreads:
            t2g, eg = spreads[[4, 5, 7]], spreads[[6, 8]]
            assert np.allclose(t2g, t2g[0], rtol=1e-5)
            assert np.allclose(eg, eg[0], rtol=1e-5)
            assert spreads[d].max() < spreads[0]

    def test_the_moment_and_the_d_occupations(self, iron):
        """The occupied states lie in the frozen window: the traces of the
        two channels' density matrices count the mesh's electrons, and
        their difference is the moment.  The d functions' occupations sit
        near the crystal's Loewdin d populations, and their spin
        difference, the d moment, agrees closely."""
        atoms, w = iron
        f, weights = mesh_occupations(atoms, w.up.kpoints, projections=True)
        crystal = atoms.calc.solver._periodic_solver.crystal
        d_orbitals = [mu for mu, function in enumerate(crystal.basis)
                      if function.l == 2]
        d = d_functions(w)
        wannier, loewdin = [], []
        for spin, channel in enumerate(w.channels):
            gamma = channel._density_bvk[0]
            electrons = np.mean(np.sum(f[spin], axis=1))
            assert np.trace(gamma).real == pytest.approx(electrons, abs=1e-8)
            wannier.append(np.trace(gamma[np.ix_(d, d)]).real)
            loewdin.append(np.mean(np.einsum(
                "kn,kmn->k", f[spin], weights[spin][:, d_orbitals])))
        moment = atoms.calc.get_total_magnetic_moment()
        traced = (np.trace(w.up._density_bvk[0])
                  - np.trace(w.down._density_bvk[0])).real
        assert traced == pytest.approx(moment, abs=5e-3)
        assert np.allclose(wannier, loewdin, atol=0.3)
        assert wannier[0] - wannier[1] == pytest.approx(
            loewdin[0] - loewdin[1], abs=0.02)

    def test_the_spin_resolved_d_model(self, iron):
        """The five d functions per spin: real integrals, the cubic
        symmetry of U, and the Fock identity in each channel."""
        _atoms, w = iron
        problem = w.downfold(d_functions(w), num_particles=(5, 3))
        g = problem.two_body
        assert g.shape == (2, 2, 5, 5, 5, 5)
        assert np.abs(g.imag).max() * HARTREE_TO_EV < 1e-8
        for s in range(2):
            U = np.real([g[s, s, i, i, i, i] for i in range(5)])
            assert np.allclose(U[[0, 1, 3]], U[0], rtol=1e-5)
            assert np.allclose(U[[2, 4]], U[2], rtol=1e-5)
        gamma = problem.density_matrix
        for s in range(2):
            fock = (problem.one_body[s]
                    + sum(np.einsum("rs,prqs->pq", gamma[t], g[s, t])
                          for t in range(2))
                    - np.einsum("rs,prsq->pq", gamma[s], g[s, s]))
            assert np.allclose(fock, problem.kohn_sham[s], atol=1e-8)


#: Two neighboring sites of the hydrogen chain.
SITES = [0, (0, (1, 0, 0))]


@pytest.fixture(scope="module")
def hydrogen_chain():
    """A ferromagnetic insulator: a stretched H chain (one atom per 2.6 A
    cell) with its electron spin up -- the up band full, the down band
    empty, one function per channel."""
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[2.6, 6.0, 6.0],
                  pbc=True)
    atoms.center(axis=(1, 2))
    atoms.set_initial_magnetic_moments([1.0])
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.3, trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZP"},
                           kpts={"size": (6, 1, 1), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.01})
    atoms.get_potential_energy()
    w = atoms.calc.wannier(guess=atoms.positions, bands=[0])
    return atoms, w, w.downfold(SITES, coulomb="isolated")


class TestSpinPolarized:
    def test_each_channel_has_its_own_functions(self, hydrogen_chain):
        atoms, w, _problem = hydrogen_chain
        assert atoms.calc.get_total_magnetic_moment() == pytest.approx(
            1.0, abs=1e-6)
        assert isinstance(w, SpinWannierResult)
        assert w.centers.shape == (2, 1, 3) and w.spreads.shape == (2, 1)
        assert np.abs(w.centers - atoms.positions[None]).max() < 1e-4
        # The empty down band is more diffuse than the full up one.
        assert w.omega_invariant[1] > w.omega_invariant[0]

    def test_each_channel_interpolates_its_own_band(self, hydrogen_chain):
        atoms, w, _problem = hydrogen_chain
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        path = np.zeros((7, 3))
        path[:, 0] = np.linspace(0.0, 0.5, 7)
        for kpoints, tolerance in ((w.up.kpoints, 1e-8), (path, 5e-3)):
            direct, _ = solver.bands(kpoints @ B.T)
            direct = np.asarray(direct)[:, :, :1] * HARTREE_TO_EV
            assert np.abs(w.interpolate(kpoints) - direct).max() < tolerance

    def test_the_trace_holds_the_band_energy(self, hydrogen_chain):
        """Up full, down empty: on one site, the sum over spins of
        tr(gamma h) is the band energy per cell."""
        atoms, w, problem = hydrogen_chain
        assert problem.spin_polarized
        assert problem.num_particles == (2, 0)
        assert np.allclose(problem.density_matrix[0], np.eye(2), atol=1e-8)
        assert np.allclose(problem.density_matrix[1], 0.0, atol=1e-8)
        site = w.downfold([0], coulomb="isolated")
        solver = atoms.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        direct, _ = solver.bands(w.up.kpoints @ B.T)
        band_energy = np.mean(np.asarray(direct)[0][:, 0])
        traced = sum(np.trace(g @ h).real for g, h in
                     zip(site.density_matrix, site.kohn_sham))
        assert traced == pytest.approx(band_energy, abs=1e-9)

    def test_the_fock_identity_holds_per_spin(self, hydrogen_chain):
        _atoms, _w, problem = hydrogen_chain
        g, gamma = problem.two_body, problem.density_matrix
        assert g.shape == (2, 2, 2, 2, 2, 2)
        for s in range(2):
            fock = (problem.one_body[s]
                    + sum(np.einsum("rs,prqs->pq", gamma[t], g[s, t])
                          for t in range(2))
                    - np.einsum("rs,prsq->pq", gamma[s], g[s, s]))
            assert np.allclose(fock, problem.kohn_sham[s], atol=1e-8)
        # The exchange splitting: the down level lies above the up one.
        assert problem.kohn_sham[1, 0, 0].real > problem.kohn_sham[0, 0,
                                                                   0].real
        # A real gauge: the model's integrals are real.
        assert np.abs(problem.two_body.imag).max() < 1e-8

    def test_both_channels_screen_one_interaction(self, hydrogen_chain):
        """cRPA sums the channels' polarizabilities into one W; with the up
        band full and the down band empty no transition lies inside the
        fragment's subspace, so it equals RPA."""
        _atoms, w, bare = hydrogen_chain
        crpa = w.downfold(SITES, coulomb="isolated", screening="crpa")
        rpa = w.downfold(SITES, coulomb="isolated", screening="rpa")
        assert crpa.two_body.shape == bare.two_body.shape
        assert np.allclose(crpa.two_body, rpa.two_body, atol=1e-8)
        U, U0 = crpa.two_body[0, 1, 0, 0, 0, 0], bare.two_body[0, 1, 0, 0,
                                                              0, 0]
        assert 0.0 < U.real < U0.real
        gamma, g = crpa.density_matrix, crpa.two_body
        for s in range(2):
            fock = (crpa.one_body[s]
                    + sum(np.einsum("rs,prqs->pq", gamma[t], g[s, t])
                          for t in range(2))
                    - np.einsum("rs,prsq->pq", gamma[s], g[s, s]))
            assert np.allclose(fock, crpa.kohn_sham[s], atol=1e-8)

    def test_the_small_box_needs_the_isolated_interaction(
            self, hydrogen_chain):
        """The chain's 6 A box puts its copies 6 A away: the plain
        truncated interaction warns and loses the inter-site repulsion."""
        _atoms, w, isolated = hydrogen_chain
        with pytest.warns(UserWarning, match="periodic copies"):
            plain = w.downfold(SITES)
        V = (plain.two_body[0, 0, 0, 1, 0, 1].real,
             isolated.two_body[0, 0, 0, 1, 0, 1].real)
        assert V[0] < V[1] - 0.02

    def test_adapt_vqe_solves_the_spin_dependent_model(self, hydrogen_chain):
        """In the (1, 1) sector, ADAPT-VQE through ``Mandacaru`` reaches
        the exact ground state of the model with different one-body parts
        per spin."""
        _atoms, w, _problem = hydrogen_chain
        problem = w.downfold(SITES, num_particles=(1, 1), coulomb="isolated")
        options = problem.as_quantum_problem()
        assert options["num_particles"] == (1, 1)
        result = Mandacaru(method="adapt-vqe", trace=False, **options).run()
        exact = exact_ground_state(problem, (1, 1))
        assert result.optimal_energy == pytest.approx(
            exact * HARTREE_TO_EV, abs=1e-6)


@pytest.mark.slow
def test_the_unpolarized_limit_is_the_restricted_result(silicon):
    """A spin-polarized run of silicon, which keeps no moment, gives both
    channels the restricted functions and the restricted model."""
    atoms, restricted = silicon
    spin = bulk("Si", "diamond", a=5.43)
    spin.set_initial_magnetic_moments([0.2, 0.2])
    spin.calc = Mandacaru(method="dft", xc="lda", h=0.35, trace=False,
                          basis={"name": "PAW-LCAO", "size": "SZP"},
                          kpts={"size": (3, 3, 3), "gamma": True},
                          smearing={"method": "fermi-dirac", "width": 0.001})
    spin.get_potential_energy()
    assert spin.calc.get_number_of_spins() == 2
    w = spin.calc.wannier()
    assert np.allclose(w.centers, restricted.centers[None], atol=1e-6)
    assert np.allclose(w.omega_invariant, restricted.omega_invariant,
                       atol=1e-5)
    split = w.downfold(coulomb="isolated")
    whole = restricted.downfold(coulomb="isolated")
    assert np.allclose(split.density_matrix.sum(axis=0),
                       whole.density_matrix, atol=1e-8)
    for a, b in zip(split.spin_orbital_integrals(),
                    whole.spin_orbital_integrals()):
        assert np.abs(a - b).max() * HARTREE_TO_EV < 1e-4
