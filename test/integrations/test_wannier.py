"""Wannier downfolding through the public calculator: the bare fragment problem of an H2 chain.

The numbers are checked by identities and by an independent Fock build, not against frozen values;
the Rust gates (a real-space double sum, a two-Gaussian closed form, the sum rules) are in
``crates/mandacaru-core/tests/wannier_downfold.rs``.
"""
import itertools

import numpy as np
import pytest
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

BOHR = 0.529177210903
CELL = np.array([6.0, 8.0, 8.0]) * BOHR


pytestmark = pytest.mark.filterwarnings("ignore:the fragment spans")


@pytest.fixture(scope="module")
def chain():
    """H2 (1.4 Bohr bond) along a periodic chain axis; two H 1s functions on a 3-cell mesh."""
    atoms = Atoms("H2", positions=[[2.3 * BOHR, 4.0 * BOHR, 4.0 * BOHR],
                                   [3.7 * BOHR, 4.0 * BOHR, 4.0 * BOHR]],
                  cell=CELL, pbc=True)
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.4 * BOHR, kpts={"size": (1, 1, 1), "gamma": True},
                           parallelization={"threads": 2})
    atoms.get_potential_energy()
    wannier = atoms.calc.wannier(2, guess=atoms.positions, kpts={"size": (3, 1, 1), "gamma": True})
    assert wannier.converged
    return wannier


def test_the_bare_problem_has_a_hydrogenic_on_site_repulsion_and_the_fock_identity(chain):
    model = chain.downfold()
    n = model.n_spatial_orbitals
    assert n == 2
    assert model.num_particles == (1, 1)
    assert model.screening is None and model.dielectric_head is None
    assert model.two_body.shape == (n, n, n, n)
    u = model.two_body[0, 0, 0, 0].real * HARTREE_TO_EV
    j = model.two_body[0, 1, 1, 0].real * HARTREE_TO_EV
    # a hydrogen 1s has U = 5/8 Hartree = 17.0 eV; a bond-compressed function a little more
    assert 14.0 < u < 22.0, u
    assert 0.0 < j < 1.0, j
    # the Fock matrix of the density matrix is the Kohn-Sham Hamiltonian, built independently
    gamma, g = model.density_matrix, model.two_body
    fock = model.one_body.copy()
    for p, q, r, s in itertools.product(range(n), repeat=4):
        fock[p, q] += gamma[r, s] * (g[p, r, q, s] - 0.5 * g[p, r, s, q])
    np.testing.assert_allclose(fock, model.kohn_sham, atol=1e-10)
    assert abs(np.trace(gamma).real - 2.0) < 1e-3
    # electron exchange and Hermiticity of <pq|rs>
    np.testing.assert_allclose(g, g.transpose(1, 0, 3, 2), atol=1e-5 * abs(g).max())
    np.testing.assert_allclose(g, g.transpose(2, 3, 0, 1).conj(), atol=1e-5 * abs(g).max())


def test_the_natural_orbitals_are_the_same_model_most_occupied_first(chain):
    model = chain.downfold()
    occupations, rotation = model.natural_orbitals()
    np.testing.assert_allclose(occupations, np.sort(np.linalg.eigvalsh(model.density_matrix))[::-1],
                               atol=1e-12)
    assert occupations[0] > occupations[1]
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(2), atol=1e-12)
    h, g = model.integrals("natural")
    np.testing.assert_allclose(h, rotation.conj().T @ model.one_body @ rotation, atol=1e-12)
    expected = np.einsum("ap,bq,cr,ds,abcd->pqrs", rotation.conj(), rotation.conj(), rotation,
                         rotation, model.two_body, optimize=True)
    np.testing.assert_allclose(g, expected, atol=1e-12)


def test_a_fragment_moved_by_a_lattice_vector_is_the_same_problem(chain):
    home = chain.downfold()
    moved = chain.downfold([(0, (1, 0, 0)), (1, (1, 0, 0))])
    np.testing.assert_allclose(moved.kohn_sham, home.kohn_sham, atol=1e-9)
    np.testing.assert_allclose(moved.two_body, home.two_body, atol=1e-8 * abs(home.two_body).max())


def test_the_two_site_model_is_solved_by_adapt_vqe_to_its_exact_ground_state(chain):
    """The plan's own gate (CRPA_PLAN S1): ADAPT on the two-site model equals its exact ground
    state. The exact value is the lowest eigenvalue of the model's 4-qubit Fock matrix in the
    sector of one electron per spin (a 4 x 4 block), not a reference calculation on a molecule."""
    model = chain.downfold()
    problem = model.as_quantum_problem()
    assert problem["n_spatial_orbitals"] == 2 and problem["num_particles"] == (1, 1)
    result = Mandacaru(method="adapt-vqe", **problem, max_iterations=6, trace=False).run()
    matrix = model.fermion_hamiltonian("natural").to_matrix(4)
    numbers = np.array([[(state >> bit) & 1 for bit in range(4)] for state in range(16)])
    # spin-orbital P = p + 2 sigma: modes 0, 1 up and 2, 3 down; bit order is the mapping's
    sector = [s for s in range(16)
              if numbers[s, :2].sum() + numbers[s, 2:].sum() == 2 and numbers[s, :2].sum() == 1]
    exact = np.linalg.eigvalsh(matrix[np.ix_(sector, sector)])[0] * HARTREE_TO_EV
    assert abs(result.optimal_energy - exact) < 1e-6 * max(1.0, abs(exact))


def test_a_fragment_wider_than_the_truncation_radius_is_warned_about(chain):
    with pytest.warns(UserWarning, match="beyond the interaction's truncation radius"):
        chain.downfold()


def test_a_screened_fragment_is_ordered_between_the_rpa_and_the_bare_interaction(chain):
    one = [0]
    bare = chain.downfold(one)
    crpa = chain.downfold(one, screening="crpa", screening_cutoff=2.0)
    rpa = chain.downfold(one, screening="rpa", screening_cutoff=2.0)
    u = [m.two_body[0, 0, 0, 0].real * HARTREE_TO_EV for m in (rpa, crpa, bare)]
    assert u[0] < u[1] < u[2], u
    assert bare.dielectric_head is None
    for model in (crpa, rpa):
        assert 0.0 < model.dielectric_head <= 1.0
        assert model.screening in ("crpa", "rpa")
    # every state in the correlated subspace screens nothing: the same integrals as the bare ones
    full = chain.downfold(screening="crpa", screening_cutoff=2.0)
    np.testing.assert_allclose(full.two_body, chain.downfold().two_body, atol=1e-10)
    assert abs(full.dielectric_head - 1.0) < 1e-10


def test_the_isolated_box_is_the_fragment_alone_in_a_doubled_supercell(chain):
    alone = chain.downfold([0], coulomb="isolated")
    plain = chain.downfold([0])
    assert alone.coulomb == "isolated" and plain.coulomb == "supercell"
    # the same repulsion of one hydrogen-like function; the tiny test cell cuts the function, so
    # the two boxes differ by a few tenths of an eV (the silicon bond, which fits, agrees within
    # 10 meV in test_wannier_screening.py)
    u = [m.two_body[0, 0, 0, 0].real * HARTREE_TO_EV for m in (alone, plain)]
    assert abs(u[0] - u[1]) < 0.8 and 14.0 < u[0] < 22.0, u


def test_unavailable_choices_raise_by_name(chain):
    with pytest.raises(NotImplementedError, match="Kohn-Sham states per k-point"):
        chain.downfold(screening="rpa", screening_bands=9)
    with pytest.raises(ValueError, match="coulomb"):
        chain.downfold(coulomb="padded")
    with pytest.raises(ValueError, match="screening"):
        chain.downfold(screening="gw")
    with pytest.raises(ValueError, match="screening_cutoff"):
        chain.downfold(screening_cutoff=2.0)
    with pytest.raises(IndexError):
        chain.downfold([5])
    with pytest.raises(ValueError, match="each function once"):
        chain.downfold([0, 0])
