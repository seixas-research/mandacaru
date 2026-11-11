"""Tests for native quantum chemistry tensor factorizations."""
import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native
from .factorization_view import (
    double_factorization,
    givens_decomposition,
    spatial_double_factorization,
)
from mandacaru.core import PauliSum
from mandacaru.core.mapping import Fermion


def test_native_double_factorization_basic():
    rng = np.random.default_rng(42)
    m = 3
    h = rng.normal(size=(m, m))
    h = 0.5 * (h + h.T)
    g = rng.normal(size=(m, m, m, m))
    # Symmetrize g in chemists/physicists pairs
    g = 0.5 * (g + g.transpose(1, 0, 3, 2))

    n_modes, const, ob, ob_rot, ob_e, leaves, trunc = _native.double_factorization(
        np.ascontiguousarray(h, dtype=np.float64),
        np.ascontiguousarray(g, dtype=np.float64),
        1e-8,
        1e-10,
        None,
        1.23,
    )
    assert n_modes == m
    assert const == pytest.approx(1.23)
    assert ob.shape == (m, m)
    assert ob_rot.shape == (m, m)
    assert ob_e.shape == (m,)
    assert len(leaves) > 0
    for w, orb, coeff in leaves:
        assert isinstance(w, float)
        assert orb.shape == (m, m)
        assert coeff.shape == (m,)


def test_native_spatial_factorization_consistency():
    rng = np.random.default_rng(123)
    m, k = 4, 6
    F = rng.normal(size=(k, m, m))
    F = F + F.transpose(0, 2, 1)
    K = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    K[4, 5] = K[5, 4] = 0.2
    chemist = np.einsum("ij,ipr,jqs->prqs", K, F, F)
    eri = chemist.transpose(0, 2, 1, 3)
    h = rng.normal(size=(m, m))
    h = 0.5 * (h + h.T)

    df_dense = spatial_double_factorization(h, eri=eri)
    df_factored = spatial_double_factorization(h, factors=F, coupling=K)

    assert df_dense.n_modes == 2 * m
    assert df_factored.n_modes == 2 * m
    assert df_dense.measurement_bases == df_factored.measurement_bases
    assert np.allclose(df_dense.one_body, df_factored.one_body, atol=1e-12)
    assert np.allclose(
        sorted(leaf.weight for leaf in df_dense.leaves),
        sorted(leaf.weight for leaf in df_factored.leaves),
        atol=1e-10,
    )


def test_native_factorized_energy_rdm():
    rng = np.random.default_rng(99)
    m = 2
    h = rng.normal(size=(m, m))
    h = 0.5 * (h + h.T)
    g = rng.normal(size=(m, m, m, m))
    g = 0.5 * (g + g.transpose(1, 0, 3, 2))

    df = double_factorization(h, g, constant=0.5)

    one_rdm = rng.normal(size=(m, m))
    one_rdm = 0.5 * (one_rdm + one_rdm.T)
    two_rdm = rng.normal(size=(m, m, m, m))
    two_rdm = 0.5 * (two_rdm + two_rdm.transpose(1, 0, 3, 2))

    energy = df.energy(one_rdm, two_rdm)
    assert isinstance(energy, float)
    assert np.isfinite(energy)


FROZEN = Path(__file__).parent / "reference" / "factorization.json"


@pytest.fixture(scope="module")
def frozen():
    return json.loads(FROZEN.read_text())["cases"]


@pytest.mark.parametrize("name", ["H2", "LiH"])
def test_frozen_factorized_measurement_reference_describes_its_own_input(frozen, name):
    """The oracle of the Rust factorized measurement (frozen from the retired Python tail).

    The frozen integrals rebuild the frozen Hamiltonian, the native factorization of
    them reproduces the frozen leaves, and the frozen basis-rotation angles are the
    native Givens decomposition of the frozen orbitals.
    """
    case = frozen[name]
    n = case["n_modes"]
    h = np.array(case["one_body"])
    eri = np.array(case["two_body"])
    rebuilt = Fermion.from_spatial_integrals(h, eri).map_to_qubits(
        "jordan_wigner", n_modes=n) + PauliSum({"I" * n: case["constant"]}, num_qubits=n)
    frozen_h = PauliSum({label: complex(re, im) for label, re, im in case["hamiltonian_terms"]},
                        num_qubits=n)
    difference = (frozen_h + (-1.0) * rebuilt).simplify()
    assert max((abs(complex(c)) for c in difference.terms.values()), default=0.0) < 1e-9

    reference = case["factorization"]
    df = spatial_double_factorization(h, eri, constant=case["constant"])
    assert df.measurement_bases == reference["measurement_bases"]
    assert df.one_norm == pytest.approx(reference["one_norm"], rel=1e-9)
    assert sorted(leaf.weight for leaf in df.leaves) == pytest.approx(
        sorted(leaf["weight"] for leaf in reference["leaves"]), abs=1e-9)

    for key, orbitals in [("one_body", reference["one_body_orbitals"])] + [
            (f"leaf_{k}", leaf["orbitals"]) for k, leaf in enumerate(reference["leaves"])]:
        rotations, signs = givens_decomposition(np.array(orbitals))
        frozen_rotation = case["basis_rotations"][key]
        assert [p for p, _t in rotations] == [p for p, _t in frozen_rotation["rotations"]]
        assert [t for _p, t in rotations] == pytest.approx(
            [t for _p, t in frozen_rotation["rotations"]])
        assert list(signs) == pytest.approx(frozen_rotation["signs"])
        assert frozen_rotation["thetas"] == pytest.approx([t for _p, t in rotations])


@pytest.mark.parametrize("name", ["H2", "LiH"])
def test_frozen_factorized_energy_is_the_exact_and_the_commuting_group_energy(frozen, name):
    case = frozen[name]
    energies = case["energies"]
    problems = case["measurement_problems"]
    assert len(problems) == case["factorization"]["measurement_bases"] == len(energies["per_basis"])
    assert energies["factorized"] == pytest.approx(
        case["factorization"]["constant"] + sum(energies["per_basis"]), abs=1e-12)
    assert energies["factorized"] == pytest.approx(energies["exact"], abs=1e-8)
    assert energies["qwc"] == pytest.approx(energies["exact"], abs=1e-10)
    for problem in problems:
        # Only I and Z: one computational-basis measurement answers for the whole leaf.
        assert all(set(label) <= {"I", "Z"} for label, _re, _im in problem["observable"])
        assert problem["num_generators"] == len(problem["thetas"])
        assert problem["thetas"][:len(case["state"]["thetas"])] == case["state"]["thetas"]


def test_the_circuit_path_through_the_calculator_reproduces_the_frozen_h2_energies(frozen):
    """MG-Q6: ``measurement_scheme="double-factorized"`` on the state-vector circuit device.

    H2's UCCSD pool is the three generators the freeze used, so the frozen thetas are the point
    to evaluate and the frozen energies are the targets: the factorized energy is the frozen
    factorized one (the oracle's Python number), the exact one, and the commuting-group one.
    LiH's frozen numbers are pinned against the Rust plan in
    ``crates/mandacaru-core/tests/factorized_measurement.rs``.
    """
    from ase import Atoms

    from mandacaru import Mandacaru

    case = frozen["H2"]
    geometry = case["geometry"]
    atoms = Atoms(geometry["symbols"], positions=geometry["positions"], cell=geometry["cell"])
    atoms.calc = Mandacaru(trace=False, **{**geometry["options"], "method": "rhf"})
    atoms.get_potential_energy()
    model = atoms.calc.native_reference.model
    thetas = case["state"]["thetas"]
    energies = {}
    for scheme in ("double-factorized", "qwc"):
        calc = Mandacaru(
            method="vqe", native_model=model, trace=False, execute_circuits=True,
            atomic_units=True, optimizer={"method": "COBYLA", "maxiter": 0},
            measurement_scheme=scheme)
        energies[scheme] = calc.run(initial_parameters=thetas).energy_hartree
        assert len(calc.problem.pool_terms) == len(thetas)
    reference = case["energies"]
    assert energies["double-factorized"] == pytest.approx(reference["factorized"], abs=1e-10)
    assert energies["double-factorized"] == pytest.approx(reference["exact"], abs=1e-12)
    assert energies["qwc"] == pytest.approx(reference["qwc"], abs=1e-10)


# --------------------------------------------------------------------------- #
# 5. Double factorization: O(M) measurement bases.
# --------------------------------------------------------------------------- #

class TestDoubleFactorization:
    """``H`` as a one-body term plus squares of one-body terms.

    Motta *et al.*, npj Quantum Inf. **7**, 83 (2021); Huggins *et al.*, npj
    Quantum Inf. **7**, 23 (2021).  Each leaf is diagonal in an orbital basis
    of its own, so one measurement circuit per leaf yields every occupation
    correlator it needs -- ``L + 1`` bases instead of ``O(M^3)``.
    """








    def test_a_mismatched_tensor_is_refused(self):
        with pytest.raises(ValueError, match="does not match"):
            double_factorization(np.zeros((4, 4)), np.zeros((3, 3, 3, 3)))
        with pytest.raises(ValueError, match="does not match"):
            spatial_double_factorization(np.zeros((4, 4)),
                                         np.zeros((3, 3, 3, 3)))
        with pytest.raises(ValueError, match="do not match"):
            spatial_double_factorization(np.zeros((4, 4)),
                                         factors=np.zeros((2, 3, 3)))


class TestTheSpatialFactorization:
    """The factorization of spatial integrals (no ``(2M)^4`` tensor) is the
    spin-orbital one: the same operator, weights, one-norm and count."""



    def test_pair_factors_give_the_tensor_factorization(self):
        """From ``k < M^2`` factors and a coupling (Cholesky factors plus an
        augmentation's channels) the leaves come from a ``k x k`` problem,
        with the tensor's result."""
        rng = np.random.default_rng(4)
        m, k = 4, 6
        F = rng.normal(size=(k, m, m))
        F = F + F.transpose(0, 2, 1)
        K = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        K[4, 5] = K[5, 4] = 1.0
        K[4, 4] = 0.3
        chemist = np.einsum("ij,ipr,jqs->prqs", K, F, F)
        eri = chemist.transpose(0, 2, 1, 3)               # <pq|rs> = (pr|qs)
        h = rng.normal(size=(m, m))
        h = h + h.T
        dense = spatial_double_factorization(h, eri)
        factored = spatial_double_factorization(h, factors=F, coupling=K)
        assert factored.measurement_bases == dense.measurement_bases
        assert np.allclose(sorted(l.weight for l in factored.leaves),
                           sorted(l.weight for l in dense.leaves), atol=1e-10)
        assert np.allclose(factored.one_body, dense.one_body, atol=1e-12)
        difference = (dense.to_fermion().map_to_qubits()
                      + (-1.0) * factored.to_fermion().map_to_qubits()
                      ).simplify()
        assert max((abs(complex(c)) for c in difference.terms.values()),
                   default=0.0) < 1e-10


# --------------------------------------------------------------------------- #
# 5b. The Givens networks that make the factorized measurement runnable.
# --------------------------------------------------------------------------- #

class TestGivensRotations:
    """Any real orthogonal ``U`` is a product of nearest-neighbor rotations.

    Clements *et al.*, Optica **3**, 1460 (2016); Kivlichan *et al.*, Phys.
    Rev. Lett. **120**, 110501 (2018).  Under Jordan-Wigner each one is a
    two-qubit gate on adjacent wires, so the basis change a factorized
    measurement needs is a circuit an ordinary processor can run.
    """

    @staticmethod
    def orthogonal(m, seed=0):
        q, _r = np.linalg.qr(np.random.default_rng(seed).normal(size=(m, m)))
        return q

    @pytest.mark.parametrize("m", [2, 3, 4, 6])
    def test_the_decomposition_reproduces_the_matrix(self, m):
        from scipy.linalg import expm

        unitary = self.orthogonal(m, seed=m)
        rotations, signs = givens_decomposition(unitary)
        assert len(rotations) <= m * (m - 1) // 2
        assert all(abs(abs(s) - 1.0) < 1e-12 for s in signs)

        rebuilt = np.eye(m)
        for p, theta in rotations:
            block = np.zeros((m, m))
            block[p, p + 1], block[p + 1, p] = -theta, theta
            rebuilt = rebuilt @ expm(block)
        assert np.abs(rebuilt @ np.diag(signs) - unitary).max() < 1e-12

    @pytest.mark.parametrize("m", [2, 3, 4])
    def test_every_rotation_is_on_adjacent_modes(self, m):
        rotations, _signs = givens_decomposition(self.orthogonal(m, seed=m))
        assert all(0 <= p < m - 1 for p, _theta in rotations)

    def test_a_non_orthogonal_matrix_is_refused(self):
        with pytest.raises(ValueError, match="real orthogonal"):
            givens_decomposition(np.array([[1.0, 2.0], [3.0, 4.0]]))
