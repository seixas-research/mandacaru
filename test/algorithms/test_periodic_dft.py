# -*- coding: utf-8 -*-
# file: test/algorithms/test_periodic_dft.py

# This code is part of Mandacaru.
# MIT License

"""Kohn-Sham DFT in a crystal: smearing, the SCF and the calculator path."""

import numpy as np
import pytest
from ase.build import bulk

from mandacaru import Mandacaru
from mandacaru.algorithms import periodic_dft as pd


class TestSmearing:
    @pytest.mark.parametrize("method", pd.SMEARING_METHODS)
    def test_occupations_fall_from_one_to_zero(self, method):
        x = np.linspace(-40.0, 40.0, 801)
        f = pd.occupation(x, method)
        assert f[0] == pytest.approx(1.0, abs=1e-6)
        assert f[-1] == pytest.approx(0.0, abs=1e-6)
        assert pd.occupation(0.0, method) == pytest.approx(0.5)

    @pytest.mark.parametrize("method", pd.SMEARING_METHODS)
    def test_the_fermi_level_holds_the_electron_count(self, method):
        rng = np.random.default_rng(3)
        eigenvalues = [np.sort(rng.normal(size=12)) for _ in range(5)]
        weights = np.full(5, 0.2)
        mu = pd.fermi_level(eigenvalues, weights, 9.0, method, 0.05)
        count = sum(w * 2.0 * np.sum(pd.occupation((e - mu) / 0.05, method))
                    for e, w in zip(eigenvalues, weights))
        assert count == pytest.approx(9.0, abs=1e-9)

    def test_the_fermi_dirac_entropy_is_the_mixing_entropy(self):
        x = np.array([-1.0, 0.0, 2.0])
        f = pd.occupation(x, "fermi-dirac")
        expected = -(f * np.log(f) + (1 - f) * np.log(1 - f))
        assert np.allclose(pd.entropy(x, "fermi-dirac"), expected)

    def test_the_free_energy_is_variational_in_the_occupations(self):
        r"""dF/df = 0 at the smeared occupations: eps - mu = -sigma dS/df."""
        eps, mu, sigma = 0.3, 0.1, 0.05
        x = (eps - mu) / sigma
        f = pd.occupation(x, "fermi-dirac")
        dS_df = np.log((1 - f) / f)
        assert eps - mu == pytest.approx(sigma * dS_df)

    @pytest.mark.parametrize("spec", [{"method": "cold", "width": 0.1},
                                      {"method": "gaussian", "width": -1},
                                      {"width": 0.1, "order": 2}])
    def test_invalid_smearing_is_refused(self, spec):
        with pytest.raises(ValueError):
            pd.resolve_smearing(spec)

    def test_a_bare_width_selects_fermi_dirac(self):
        method, width = pd.resolve_smearing(0.2)
        assert method == "fermi-dirac" and width == pytest.approx(0.2 / 27.211386245988)


class TestBandGap:
    def _result(self, bands, mu):
        bands = [np.asarray(b, float) for b in bands]
        return pd.PeriodicKohnShamResult(
            functional="lda", free_energy=0.0, energy=0.0,
            extrapolated_energy=0.0, fermi_level=mu, kpoints=np.zeros((2, 3)),
            weights=np.full(2, 0.5), eigenvalues=bands,
            occupations=[np.zeros(3)] * 2, converged=True, n_iterations=1,
            smearing=("fermi-dirac", 0.01))

    def test_an_insulator_has_the_direct_or_indirect_gap(self):
        result = self._result([[-1.0, -0.5, 0.5], [-1.1, -0.4, 0.3]], 0.0)
        assert result.band_gap == pytest.approx(0.7)

    def test_a_band_crossing_the_fermi_level_is_a_metal(self):
        result = self._result([[-1.0, -0.2, 0.5], [-1.1, 0.1, 0.6]], 0.0)
        assert result.band_gap is None


@pytest.fixture(scope="module")
def silicon():
    """Diamond Si, SZ, coarse grid and a 2x2x2 mesh: small but a crystal."""
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.34,
                           kpts={"size": (2, 2, 2), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.01},
                           basis={"name": "PAW-LCAO", "size": "SZ",
                                  "filter": 200}, trace=False)
    energy = atoms.get_potential_energy()
    return atoms, energy


class TestTheCalculator:
    def test_a_crystal_converges_with_a_gap(self, silicon):
        atoms, energy = silicon
        result = atoms.calc.result
        assert result.success and np.isfinite(energy)
        assert result.scf.band_gap is not None and result.scf.band_gap > 0.0
        assert len(result.scf.kpoints) == 3       # 2x2x2 -> irreducible wedge
        assert np.sum(result.scf.weights) == pytest.approx(1.0)

    def test_it_cites_the_crystal_machinery(self, silicon):
        atoms, _energy = silicon
        keys = set(atoms.calc.citation_keys())
        assert {"KohnSham1965", "MonkhorstPack1976", "Pulay1980",
                "Kerker1981", "Mermin1965", "Togo2018"} <= keys

    def test_the_valence_charge_is_conserved(self, silicon):
        atoms, _energy = silicon
        scf = atoms.calc.result.scf
        electrons = sum(w * np.sum(f) for w, f in zip(scf.weights,
                                                      scf.occupations))
        assert electrons == pytest.approx(8.0, abs=1e-8)

    def test_a_non_paw_basis_is_refused_for_a_crystal(self):
        atoms = bulk("Si", "diamond", a=5.43)
        atoms.calc = Mandacaru(method="dft", basis="HAO", trace=False)
        with pytest.raises(NotImplementedError, match="PAW-LCAO"):
            atoms.get_potential_energy()

    def test_crystal_forces_are_refused(self, silicon):
        atoms, _energy = silicon
        with pytest.raises(NotImplementedError, match="periodic"):
            atoms.get_forces()

    def test_an_invalid_smearing_is_refused_by_the_constructor(self):
        with pytest.raises(ValueError, match="smearing"):
            Mandacaru(method="dft", smearing={"method": "cold"})


#: Silicon on a grid commensurate with its primitive cell (8 nodes per
#: lattice vector): small enough for the engine-level tests below.
SILICON = bulk("Si", "diamond", a=5.43)
SILICON_BASIS = {"size": "SZ", "filter": 200}


def _crystal(h=None, kpts=None):
    from mandacaru.pseudopotentials.periodic_paw import build_crystal
    h = float(np.linalg.norm(SILICON.cell[0])) / 8 if h is None else h
    return build_crystal(SILICON, h, SILICON_BASIS,
                         kpts=kpts or {"size": (2, 1, 1), "gamma": True})


class TestTheKohnShamMatrix:
    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
    def test_it_is_the_derivative_of_the_energy(self, functional):
        r"""``dE = sum_k w_k tr(dP_k H_k)`` for every term of the functional.

        Checked at a non-self-consistent density (the identity holds at any
        density matrices), along a positive semidefinite step so a meta-GGA's
        kinetic-energy density stays non-negative; the difference is
        one-sided and second order.  This is what makes the r2SCAN tau term,
        built from the gradients of the Bloch sums, the derivative of the
        energy it adds.
        """
        from scipy.linalg import eigh
        crystal, context = _crystal()
        ks = pd.PeriodicKohnSham(crystal, context["n_electrons"], functional)
        rho, q = crystal.initial_density()
        tau = np.zeros(crystal.grid.size)
        V, v_tau, w, _terms = ks._potentials(rho, q, tau if ks.meta else None)
        matrices = []
        for i, data in enumerate(crystal.kpoint_data):
            H = ks._hamiltonian(data, V, v_tau, w,
                                ks._gradients[i] if ks.meta else None)
            _eps, C = eigh(H, data.overlap)
            occupied = C[:, :2]
            matrices.append(2.0 * occupied @ occupied.conj().T)

        def energy(mats):
            rho_k, q_k = crystal.density(mats)
            tau_k = ks._tau(mats) if ks.meta else None
            return sum(ks.energy_terms(mats, rho_k, q_k, tau_k).values())

        rho0, q0 = crystal.density(matrices)
        tau0 = ks._tau(matrices) if ks.meta else None
        V, v_tau, w, _terms = ks._potentials(rho0, q0, tau0)
        rng = np.random.default_rng(11)
        steps, directional = [], 0.0
        for i, data in enumerate(crystal.kpoint_data):
            B = (rng.normal(size=matrices[i].shape)
                 + 1j * rng.normal(size=matrices[i].shape))
            step = 1e-3 * (B @ B.conj().T)
            steps.append(step)
            H = ks._hamiltonian(data, V, v_tau, w,
                                ks._gradients[i] if ks.meta else None)
            directional += data.weight * float(np.real(np.sum(step * H.T)))
        eps = 1e-4
        e0 = energy(matrices)
        e1 = energy([P + eps * d for P, d in zip(matrices, steps)])
        e2 = energy([P + 2 * eps * d for P, d in zip(matrices, steps)])
        assert (-3 * e0 + 4 * e1 - e2) / (2 * eps) == pytest.approx(
            directional, rel=1e-5)


class TestTheEigenvalueReference:
    def test_eigenvalues_do_not_move_with_the_grid(self):
        """The plane-wave zero: silicon's levels at two grid spacings.

        Before the reference, the lowest level moved by 1.7 eV between
        h = 0.30 and 0.20 Angstrom while the energy stayed put (HISTORY.md,
        2026-10-02).
        """
        spacing = float(np.linalg.norm(SILICON.cell[0]))
        levels = []
        for n in (8, 10):
            crystal, context = _crystal(h=spacing / n)
            result = pd.PeriodicKohnSham(crystal, context["n_electrons"],
                                         "lda").run()
            levels.append((min(e.min() for e in result.eigenvalues),
                           result.fermi_level))
        assert levels[0][0] == pytest.approx(levels[1][0], abs=2e-4)
        assert levels[0][1] == pytest.approx(levels[1][1], abs=2e-4)


class TestTheSpectrum:
    """Non-self-consistent bands, densities of states and fat bands."""

    def test_the_bands_at_the_mesh_are_the_scf_eigenvalues(self, silicon):
        """The frozen potential is the one the SCF eigenvalues came from."""
        atoms, _energy = silicon
        solver = atoms.calc.solver
        crystal = solver._gradient_context["integrals"]
        bands, _ = solver._periodic_solver.bands(crystal.kpoints)
        assert np.abs(bands - np.array(atoms.calc.result.scf.eigenvalues)
                      ).max() < 1e-10

    def test_the_wedge_stands_for_the_whole_mesh(self, silicon):
        """Every point of the full 2x2x2 mesh has the levels of its wedge
        representative: the symmetrized potential is symmetric.  To the
        accuracy of the projector and short-range sphere quadratures, whose
        angular grids the point group does not map onto themselves (4e-8 Ha
        here)."""
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            monkhorst_pack_kpts)
        atoms, _energy = silicon
        solver = atoms.calc.solver
        crystal = solver._gradient_context["integrals"]
        _size, _gamma, mesh = monkhorst_pack_kpts({"size": (2, 2, 2),
                                                   "gamma": True})
        full, _ = solver._periodic_solver.bands(crystal.cartesian_kpoints(mesh))
        wedge = np.array(atoms.calc.result.scf.eigenvalues)
        for levels in full:
            assert np.abs(wedge - levels).max(axis=1).min() < 1e-6

    def test_loewdin_weights_partition_every_state(self, silicon):
        atoms, _energy = silicon
        solver = atoms.calc.solver
        crystal = solver._gradient_context["integrals"]
        _bands, weights = solver._periodic_solver.bands(
            crystal.cartesian_kpoints([[0.1, 0.2, 0.3]]), projections=True)
        assert weights.min() >= 0.0
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)

    def test_the_dos_counts_the_bands_and_the_electrons(self, silicon):
        atoms, _energy = silicon
        energies, dos = atoms.calc.dos(width=0.05)
        n_bands = len(atoms.calc.get_eigenvalues(0))
        assert np.trapezoid(dos, energies) == pytest.approx(2 * n_bands,
                                                            rel=1e-6)
        below = energies < atoms.calc.get_fermi_level()
        assert np.trapezoid(dos[below], energies[below]) == pytest.approx(
            8.0, abs=1e-3)

    def test_the_pdos_sums_to_the_dos(self, silicon):
        atoms, _energy = silicon
        energies, dos = atoms.calc.dos(width=0.1, npoints=801)
        _energies, pdos = atoms.calc.pdos(width=0.1, npoints=801)
        assert set(pdos) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert np.abs(sum(pdos.values()) - dos).max() < 1e-8 * dos.max()
        assert "Loewdin1950" in atoms.calc.citation_keys()

    def test_a_reduced_mesh_gives_the_full_mesh_pdos(self, silicon):
        """3x3x3 is reduced to its wedge, and each atom's share averaged over
        the atoms the operations map it to: the result is the full mesh's,
        computed here by brute force."""
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            monkhorst_pack_kpts)
        atoms, _energy = silicon
        calc, solver = atoms.calc, atoms.calc.solver
        crystal = solver._gradient_context["integrals"]
        axis, pdos = calc.pdos(width=0.1, npoints=401, kpts=(3, 3, 3))
        _size, _gamma, mesh = monkhorst_pack_kpts((3, 3, 3))
        levels, weights = solver._periodic_solver.bands(
            crystal.cartesian_kpoints(mesh), projections=True)
        per_orbital = pd.broadened_dos(
            axis, levels * 27.211386245988, np.full(len(mesh), 1 / len(mesh)),
            0.1, state_weights=weights)
        full = {}
        for mu, shell in enumerate(solver._orbital_shells()):
            full[shell] = full.get(shell, 0.0) + per_orbital[:, mu]
        # 1.3e-6 of the peak: the sphere quadratures' 4e-8 Ha asymmetry
        # moving 0.1 eV Gaussians.  Without the atom-map averaging the two
        # atoms differ by 1.3e-5 on this grid, which keeps quarter
        # translations off its nodes.
        scale = max(v.max() for v in full.values())
        for shell, values in full.items():
            assert np.abs(pdos[shell] - values).max() < 5e-6 * scale

    def test_the_band_structure_follows_the_path(self, silicon):
        from ase.spectrum.band_structure import BandStructure
        atoms, _energy = silicon
        bs = atoms.calc.band_structure(path="GXL", npoints=12)
        assert isinstance(bs, BandStructure)
        assert bs.energies.shape == (1, 12, len(atoms.calc.get_eigenvalues(0)))
        assert bs.reference == pytest.approx(atoms.calc.get_fermi_level())
        # Gamma is on the SCF mesh: the path's first point is the SCF's.
        assert np.allclose(bs.energies[0, 0], atoms.calc.get_eigenvalues(0),
                           atol=1e-8)
        assert "SetyawanCurtarolo2010" in atoms.calc.citation_keys()

    def test_path_points_on_the_mesh_have_the_scf_levels(self, silicon):
        """X and L of the fcc path lie on the Gamma-centered 2x2x2 mesh: the
        path's fractional k and the crystal's convention agree."""
        atoms, _energy = silicon
        bs = atoms.calc.band_structure(path="GXL", npoints=9)
        scf = np.array([atoms.calc.get_eigenvalues(k) for k in range(3)])
        special = bs.path.special_points
        for label in ("X", "L"):
            index = int(np.argmin(np.linalg.norm(
                bs.path.kpts - special[label], axis=1)))
            assert np.allclose(bs.path.kpts[index], special[label])
            assert np.abs(scf - bs.energies[0, index]).max(axis=1).min() \
                < 1e-5

    def test_fat_bands_weights_sum_to_one(self, silicon):
        atoms, _energy = silicon
        bs, weights = atoms.calc.fat_bands(path="GX", npoints=5)
        assert np.allclose(sum(weights.values()), 1.0, atol=1e-10)
        assert next(iter(weights.values())).shape == bs.energies.shape[1:]

    def test_the_ase_getters(self, silicon):
        atoms, _energy = silicon
        calc = atoms.calc
        assert calc.get_number_of_spins() == 1
        assert calc.get_ibz_k_points().shape == (3, 3)
        assert calc.get_k_point_weights().sum() == pytest.approx(1.0)
        with pytest.raises(ValueError, match="spin"):
            calc.get_eigenvalues(0, spin=1)


def test_crystal_populations_are_refused_with_a_pointer(silicon):
    atoms, _energy = silicon
    with pytest.raises(NotImplementedError, match="pdos"):
        atoms.calc.get_charges()
