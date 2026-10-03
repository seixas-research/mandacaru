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
