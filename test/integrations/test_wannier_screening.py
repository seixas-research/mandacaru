"""Screened Wannier downfolding of silicon's bond (the realistic case), through the public calculator.

Slow: a 4 x 4 x 4 SZ silicon, eight sp3 functions, and screened downfolds with `|q + G|` below
2 Bohr^-1 (the example's DZP crystal, its 26 states and 3 Bohr^-1 take a minute each). The Rust gates (the free-electron Adler-Wiser sum, the sum rules, the
limits) are in `crates/mandacaru-core/tests/{screening_medium,wannier_screening}.rs`; these are the
physical orderings and the isolated box against the supercell's.
"""
import itertools

import numpy as np
import pytest
from ase.build import bulk
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

pytestmark = pytest.mark.slow

CHEAP = {"screening_cutoff": 2.0}


@pytest.fixture(scope="module")
def silicon():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.25, kpts={"size": (4, 4, 4), "gamma": True},
                           parallelization={"threads": 2})
    atoms.get_potential_energy()
    fermi = atoms.calc.get_fermi_level()
    wannier = atoms.calc.wannier(8, guess="sp3",
                                 windows={"outer": (fermi - 20.0, fermi + 20.0),
                                          "frozen": (fermi - 20.0, fermi + 1.0)})
    start, end = atoms.positions
    bond = [int(np.argmin(np.linalg.norm(wannier.centers - (a + 0.2 * (b - a)), axis=1)))
            for a, b in ((start, end), (end, start))]
    return wannier, bond


@pytest.fixture(scope="module")
def models(silicon):
    wannier, bond = silicon
    return {"bare": wannier.downfold(bond),
            "crpa": wannier.downfold(bond, screening="crpa", **CHEAP),
            "rpa": wannier.downfold(bond, screening="rpa", **CHEAP)}


def test_the_screened_interaction_is_much_weaker_and_exchange_is_screened_less(models):
    bare, crpa = models["bare"].two_body, models["crpa"].two_body
    u, u0 = crpa[0, 0, 0, 0].real, bare[0, 0, 0, 0].real
    j, j0 = crpa[0, 1, 1, 0].real, bare[0, 1, 1, 0].real
    assert 0.0 < u < 0.5 * u0
    assert 0.0 < j < j0
    assert j / j0 > u / u0
    assert 0.0 < models["crpa"].dielectric_head < 1.0
    assert models["bare"].dielectric_head is None
    np.testing.assert_allclose(models["crpa"].kohn_sham, models["bare"].kohn_sham)


def test_excluding_nothing_screens_most(models):
    """Full RPA lies below cRPA, which leaves out the transitions inside the bond's Bloch
    subspace."""
    assert models["rpa"].two_body[0, 0, 0, 0].real < models["crpa"].two_body[0, 0, 0, 0].real


def test_the_double_counting_uses_the_screened_interaction(models):
    model = models["crpa"]
    gamma, g = model.density_matrix, model.two_body
    fock = model.one_body.copy()
    n = model.n_spatial_orbitals
    for p, q, r, s in itertools.product(range(n), repeat=4):
        fock[p, q] += gamma[r, s] * (g[p, r, q, s] - 0.5 * g[p, r, s, q])
    np.testing.assert_allclose(fock, model.kohn_sham, atol=1e-10)


def test_the_isolated_box_agrees_with_the_supercell_inside_the_radius(silicon):
    """A bond's charge fits inside the 4 x 4 x 4 supercell's truncation radius, so the fragment
    alone in a doubled box gives the same interaction (the Python era measured 0.8 meV on U)."""
    wannier, bond = silicon
    plain = wannier.downfold(bond)
    alone = wannier.downfold(bond, coulomb="isolated")
    assert alone.coulomb == "isolated"
    scale = HARTREE_TO_EV
    assert abs(alone.two_body[0, 0, 0, 0].real - plain.two_body[0, 0, 0, 0].real) * scale < 0.01
    assert abs(alone.two_body[0, 1, 1, 0].real - plain.two_body[0, 1, 1, 0].real) * scale < 0.005


def test_antiferromagnetic_nio_has_a_spin_resolved_model_with_the_fock_identity_per_spin():
    """Two spin channels with different functions (up and down Ni d sets): the double counting of
    each spin is built from both densities and its own exchange, so the per-spin Fock identity and
    the swap of the two electrons' spin blocks check the spin-polarized layout independently."""
    from ase.build import bulk, make_supercell

    prim = bulk("NiO", "rocksalt", a=4.17)
    atoms = make_supercell(prim, np.array([[1, 0, 0], [0, 1, 0], [-1, -1, 2]]))
    atoms.set_initial_magnetic_moments([2.0, 0.0, -2.0, 0.0])
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.35, kpts={"size": (2, 2, 1), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.01}, trace=False,
                           parallelization={"threads": 2})
    atoms.get_potential_energy()
    fermi = atoms.calc.get_fermi_level()
    wannier = atoms.calc.wannier(guess={"Ni": "d"}, windows={"outer": (fermi - 5.0, fermi + 5.0)})
    model = wannier.downfold(functions=[2, 4, 7, 9], num_particles=(2, 2), screening="crpa",
                             screening_cutoff=2.0, coulomb="isolated")
    assert model.spin_polarized and model.two_body.shape == (2, 2, 4, 4, 4, 4)
    n = model.n_spatial_orbitals
    gamma, g = model.density_matrix, model.two_body
    assert np.abs(gamma[0] - gamma[1]).max() > 1e-3, "the antiferromagnet's channels differ"
    for s in range(2):
        fock = model.one_body[s].copy()
        for p, q, r, u in itertools.product(range(n), repeat=4):
            fock[p, q] += sum(gamma[t][r, u] * g[s, t][p, r, q, u] for t in range(2))
            fock[p, q] -= gamma[s][r, u] * g[s, s][p, r, u, q]
        np.testing.assert_allclose(fock, model.kohn_sham[s], atol=1e-10)
    # exchanging the two electrons exchanges the spin blocks
    np.testing.assert_allclose(g[0, 1], g[1, 0].transpose(1, 0, 3, 2),
                               atol=1e-4 * abs(g).max())
    bare = wannier.downfold(functions=[2, 4, 7, 9], num_particles=(2, 2), coulomb="isolated")
    assert 0.0 < model.two_body[0, 1, 0, 0, 0, 0].real < bare.two_body[0, 1, 0, 0, 0, 0].real
