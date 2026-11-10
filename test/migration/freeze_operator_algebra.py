"""Freeze the operator algebra before it moves to Rust (MG-Q5).

Run once, on the Python route, from the repository root:
``python test/migration/freeze_operator_algebra.py``. The Python route it imports
is deleted by the change that added the reference, so the script is the record of
how the file was made and runs only at the parent of that change. It writes
``test/migration/reference/operator_algebra.json``, which
``test_native_operator_algebra.py`` reads after the Python algebra
(``core/pauli_algebra.py``, ``core/sector.py``, ``core/matrix_free.PauliOperator``
and the Python mappings of ``core/mapping.py``) is deleted.

Conventions of the file: a complex number is ``[re, im]``; a Pauli sum is
``{label: [re, im]}`` after ``simplify()`` (strings at or below 1e-12 dropped,
which is the threshold the native route applies); a matrix is a nested list of
complex numbers; qubit 0 is the leftmost letter and the most significant bit.
"""

import json
from pathlib import Path

import numpy as np

from mandacaru.core.mapping import (Fermion, PauliSum, qubit_excitation,
                                    reencode_pauli, reference_qubit_bits)
from mandacaru.core.matrix_free import PauliOperator
from mandacaru.core.sector import ParticleSector, apply_pauli_sum

OUT = Path(__file__).parent / "reference" / "operator_algebra.json"
MAPPINGS = ("jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced")


def z(value):
    value = complex(value)
    return [value.real, value.imag]


def pauli(op):
    return {label: z(c) for label, c in sorted(op.simplify().terms.items())}


def matrix(m):
    m = np.asarray(m, dtype=complex)
    return [[z(v) for v in row] for row in m]


def vector(v):
    return [z(c) for c in np.asarray(v, dtype=complex)]


def random_pauli(n, count, seed):
    rng = np.random.default_rng(seed)
    labels = {"".join(rng.choice(list("IXYZ"), n)) for _ in range(count)}
    return PauliSum({label: complex(rng.normal(), rng.normal())
                     for label in sorted(labels)}, num_qubits=n)


def term_list(operator):
    return [[[[m, bool(d)] for m, d in term], z(c)]
            for term, c in operator.terms.items()]


def hopping():
    t = Fermion.creation(0) * Fermion.annihilation(1)
    u = Fermion.creation(3) * Fermion.annihilation(4)
    return t + t.dagger() + u + u.dagger() \
        + 0.3 * (Fermion.creation(2) * Fermion.annihilation(2))


def complex_number_conserving():
    return Fermion({((0, True), (1, False)): .2 + .7j,
                    ((1, True), (0, False)): .2 - .7j,
                    ((4, True), (5, False)): -.3j,
                    ((5, True), (4, False)): .3j,
                    ((0, True), (3, True), (4, False), (1, False)): .31},
                   n_modes=6)


def molecular(m, seed):
    """A Hermitian spin-blocked one- plus two-body operator on ``m`` modes."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(m, m))
    h = 0.5 * (h + h.T)
    eri = rng.normal(size=(m,) * 4)
    eri = 0.25 * (eri + eri.transpose(1, 0, 2, 3) + eri.transpose(0, 1, 3, 2)
                  + eri.transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return Fermion.from_integrals(h, np.einsum("prqs->pqrs", eri))


def spin_blocked_hamiltonian():
    """Number- and spin-conserving, on 6 spin-orbitals (3 + 3)."""
    h = np.zeros((6, 6))
    for block in (0, 3):
        h[block, block + 1] = h[block + 1, block] = -0.5
        h[block + 1, block + 2] = h[block + 2, block + 1] = -0.35
        for k, e in enumerate((-1.2, -0.9, -0.4)):
            h[block + k, block + k] = e
    eri = np.zeros((6,) * 4)
    for p in range(6):
        eri[p, p, p, p] = 0.8
    eri[0, 3, 0, 3] = eri[3, 0, 3, 0] = 0.45
    return Fermion.from_integrals(h, eri)


def mapping_cases():
    cases = []
    operators = {
        "hopping": (hopping(), 6),
        "complex_number_conserving": (complex_number_conserving(), 6),
        "molecular_4": (molecular(4, 3), 4),
        "spin_blocked_hamiltonian": (spin_blocked_hamiltonian(), 6),
        "number_changing": (Fermion.creation(0) + Fermion.annihilation(2)
                            + 0.5 * (Fermion.creation(1) * Fermion.creation(3)),
                            4),
        "odd_modes": (Fermion.annihilation(1) * Fermion.creation(2)
                      + 0.25 * Fermion.creation(0) * Fermion.annihilation(0), 3),
        "identity_and_constant": (Fermion.identity() * 0.75
                                  + Fermion.creation(1) * Fermion.annihilation(1), 4),
    }
    for name, (op, n) in operators.items():
        for mapping in MAPPINGS:
            if mapping == "parity_reduced" and n % 2:
                continue            # the reduction is defined on spin-blocked registers
            case = {"operator": name, "mapping": mapping, "n_modes": n,
                    "terms": term_list(op)}
            if mapping == "parity_reduced":
                case["num_particles"] = [1, 1]
            kwargs = {"n_modes": n}
            if mapping == "parity_reduced":
                kwargs["num_particles"] = (1, 1)
            try:
                mapped = op.map_to_qubits(mapping, **kwargs)
            except ValueError as error:
                case["raises"] = str(error)
                cases.append(case)
                continue
            case["num_qubits"] = mapped.num_qubits
            case["result"] = pauli(mapped)
            cases.append(case)
    # The register width is inferred from the highest mode when not given.
    inferred = hopping().map_to_qubits("jordan_wigner")
    cases.append({"operator": "hopping", "mapping": "jordan_wigner", "n_modes": None,
                  "terms": term_list(hopping()), "num_qubits": inferred.num_qubits,
                  "result": pauli(inferred)})
    return cases


def algebra_cases():
    cases = []
    pairs = {
        "random_3": (random_pauli(3, 12, 1), random_pauli(3, 9, 2)),
        "random_5": (random_pauli(5, 30, 3), random_pauli(5, 8, 4)),
        "cancelling_product": (PauliSum({"XI": 1, "ZI": 1, "IZ": 0.5j}),
                               PauliSum({"XI": 1, "ZI": 1, "IY": -0.25})),
        "commuting": (PauliSum({"ZZ": 1.5, "ZI": -0.5, "II": 0.1}),
                      PauliSum({"IZ": 0.3j, "ZZ": 2.0})),
        "mixed_hermiticity": (PauliSum({"XY": 0.5 - 1j, "ZI": 1 + 0.3j, "YY": 2j}),
                              PauliSum({"YX": 2.0, "IZ": 0.7j, "XX": -1.1})),
    }
    for name, (a, b) in pairs.items():
        cases.append({
            "name": name, "num_qubits": a.num_qubits,
            "a": pauli(a), "b": pauli(b),
            "compose": pauli(a.compose(b)), "compose_reversed": pauli(b.compose(a)),
            "commutator": pauli(a.commutator(b)),
            "hermitian_part": pauli(a.hermitian_part()),
            "hermitian_part_b": pauli(b.hermitian_part()),
        })
    return cases


def matrix_cases():
    cases = []
    for n, count, seed in ((1, 3, 11), (3, 14, 12), (4, 40, 13), (5, 60, 14)):
        op = random_pauli(n, count, seed)
        cases.append({"num_qubits": n, "operator": pauli(op),
                      "dense": matrix(op.to_matrix()),
                      "sparse": matrix(op.to_sparse_matrix().toarray()),
                      "sparse_nnz": int(op.to_sparse_matrix().nnz)})
    empty = PauliSum({}, num_qubits=3)
    cases.append({"num_qubits": 3, "operator": {},
                  "dense": matrix(empty.to_matrix()),
                  "sparse": matrix(empty.to_sparse_matrix().toarray()),
                  "sparse_nnz": int(empty.to_sparse_matrix().nnz)})
    diagonal = PauliSum({"ZI": 0.5, "IZ": -0.25, "II": 1.0}, num_qubits=2)
    cases.append({"num_qubits": 2, "operator": pauli(diagonal),
                  "dense": matrix(diagonal.to_matrix()),
                  "sparse": matrix(diagonal.to_sparse_matrix().toarray()),
                  "sparse_nnz": int(diagonal.to_sparse_matrix().nnz)})
    return cases


def fermion_matrix_cases():
    cases = []
    ops = {"hopping_3": (Fermion.creation(0) * Fermion.annihilation(1)
                         + Fermion.creation(1) * Fermion.annihilation(0), 3),
           "anticommutator": (Fermion.annihilation(2) * Fermion.creation(1)
                              + Fermion.creation(1) * Fermion.annihilation(2), 3),
           "molecular_3": (molecular(3, 7), 3),
           "complex_4": (complex_number_conserving(), 6)}
    for name, (op, n) in ops.items():
        cases.append({"name": name, "n_modes": n, "terms": term_list(op),
                      "matrix": matrix(op.to_matrix(n_modes=n))})
    return cases


def integral_cases():
    """The term lists (in order) of the integral builders."""
    rng = np.random.default_rng(21)
    h = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    g = rng.normal(size=(3,) * 4)
    g[rng.random(g.shape) < 0.5] = 0.0
    spatial_h = rng.normal(size=(2, 2))
    spatial_eri = rng.normal(size=(2,) * 4)
    spatial_eri[rng.random(spatial_eri.shape) < 0.3] = 0.0
    spin = 0.1 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    resolved_h = rng.normal(size=(2, 2, 2))
    resolved_eri = rng.normal(size=(2, 2, 2, 2, 2, 2))
    resolved_eri[rng.random(resolved_eri.shape) < 0.5] = 0.0

    def entry(op, **inputs):
        return {"inputs": inputs, "n_modes": op.n_modes(), "terms": term_list(op)}

    def real(a):
        return [float(v) for v in np.ravel(a)]

    def cplx(a):
        return [z(v) for v in np.ravel(a)]

    return [
        {**entry(Fermion.from_integrals(h, g), kind="spin_orbital", tol=1e-12,
                 one=cplx(h), one_shape=list(h.shape), two=real(g), two_shape=list(g.shape))},
        {**entry(Fermion.from_integrals(real_part(h)), kind="spin_orbital", tol=1e-12,
                 one=real(real_part(h)), one_shape=[3, 3], two=None, two_shape=None)},
        {**entry(Fermion.from_integrals(0.5 * np.eye(2), tol=0), kind="spin_orbital", tol=0,
                 one=real(0.5 * np.eye(2)), one_shape=[2, 2], two=None, two_shape=None)},
        {**entry(Fermion.from_spatial_integrals(spatial_h, spatial_eri),
                 kind="spatial", tol=1e-12, one=real(spatial_h), one_shape=[2, 2],
                 two=real(spatial_eri), two_shape=[2] * 4, spin=None)},
        {**entry(Fermion.from_spatial_integrals(spatial_h, spatial_eri, spin),
                 kind="spatial", tol=1e-12, one=real(spatial_h), one_shape=[2, 2],
                 two=real(spatial_eri), two_shape=[2] * 4, spin=cplx(spin))},
        {**entry(Fermion.from_spatial_integrals(resolved_h, resolved_eri),
                 kind="spatial", tol=1e-12, one=real(resolved_h), one_shape=[2, 2, 2],
                 two=real(resolved_eri), two_shape=[2] * 2 + [2] * 4, spin=None)},
    ]


def real_part(a):
    return np.real(a)


def excitation_cases():
    cases = []
    shapes = [((3,), (0,)), ((4,), (1,)), ((2, 3), (0, 1)), ((4, 5), (1, 2)),
              ((3, 5), (0, 2))]
    for mapping in MAPPINGS:
        for out, into in shapes:
            kwargs = {"num_particles": (1, 1)} if mapping == "parity_reduced" else {}
            try:
                op = qubit_excitation(out, into, 6, method=mapping, **kwargs)
            except ValueError as error:
                cases.append({"out": list(out), "in": list(into), "mapping": mapping,
                              "raises": str(error)})
                continue
            cases.append({"out": list(out), "in": list(into), "mapping": mapping,
                          "num_qubits": op.num_qubits, "result": pauli(op)})
    return cases


def reencode_cases():
    cases = []
    jw_hopping = hopping().map_to_qubits("jordan_wigner", n_modes=6)
    jw_molecular = spin_blocked_hamiltonian().map_to_qubits("jordan_wigner", n_modes=6)
    sources = {"hopping": jw_hopping, "spin_blocked": jw_molecular,
               "random_6": random_pauli(6, 25, 31)}
    for name, source in sources.items():
        for mapping in MAPPINGS:
            kwargs = {"num_particles": (1, 1)} if mapping == "parity_reduced" else {}
            case = {"source": name, "terms": pauli(source), "mapping": mapping,
                    "num_particles": [1, 1] if kwargs else None}
            try:
                result = reencode_pauli(source, mapping, **kwargs)
            except ValueError as error:
                case["raises"] = str(error)
            else:
                case["num_qubits"] = result.num_qubits
                case["result"] = pauli(result)
            cases.append(case)
    return cases


def reference_bit_cases():
    """The qubit bit string of every determinant of 6 spin-orbitals."""
    out = {}
    for mapping in MAPPINGS:
        table = {}
        for mask in range(64):
            occupied = [j for j in range(6) if mask >> j & 1]
            bits = reference_qubit_bits(mapping, 6, occupied)
            table[str(mask)] = "".join(str(int(b)) for b in bits)
        out[mapping] = table
    return out


def sector_cases():
    cases = []
    hamiltonian = spin_blocked_hamiltonian()
    for mapping in MAPPINGS:
        kwargs = {"num_particles": (1, 1)} if mapping == "parity_reduced" else {}
        h = hamiltonian.map_to_qubits(mapping, n_modes=6, **kwargs)
        for particles in ((1, 1), (2, 1), (0, 2)):
            sector = ParticleSector(h.num_qubits, particles, mapping=mapping)
            case = {"mapping": mapping, "num_particles": list(particles),
                    "num_qubits": h.num_qubits, "indices": sector.indices.tolist()}
            if particles == (1, 1):
                restricted = sector.restrict(h)
                case["restricted"] = matrix(restricted.toarray())
                case["hamiltonian"] = pauli(h)
                case["conserves"] = bool(sector.conserves(h))
                if mapping != "parity_reduced":
                    leaking = leaking_operator(mapping)
                    case["leaking"] = pauli(leaking)
                    case["leaking_conserves"] = bool(sector.conserves(leaking))
            cases.append(case)
    for particles in ((1, 1), (2, 0), (1, 0)):
        sector = ParticleSector(6, particles, spin_conserving=False)
        cases.append({"mapping": "jordan_wigner", "spin_conserving": False,
                      "num_particles": list(particles), "num_qubits": 6,
                      "indices": sector.indices.tolist(), "dim": sector.dim})
    return cases


def leaking_operator(mapping):
    """A particle-number-changing operator in the given encoding."""
    op = Fermion.creation(0) + Fermion.annihilation(1) \
        + 0.5 * Fermion.creation(2) * Fermion.annihilation(0)
    return op.map_to_qubits(mapping, n_modes=6)


def apply_cases():
    cases = []
    op = random_pauli(4, 10, 41)
    rng = np.random.default_rng(42)
    for indices in ([0, 5, 9, 15], [3], list(range(16))):
        amplitudes = rng.normal(size=len(indices)) + 1j * rng.normal(size=len(indices))
        got_i, got_a = apply_pauli_sum(op, indices, amplitudes)
        cases.append({"operator": pauli(op), "num_qubits": 4, "indices": indices,
                      "amplitudes": vector(amplitudes),
                      "image_indices": got_i.tolist(), "image_amplitudes": vector(got_a)})
    # a cancelling image: the sum of two strings maps a state to nothing.
    cancel = PauliSum({"XI": 1.0, "YI": 1j}, num_qubits=2)
    got_i, got_a = apply_pauli_sum(cancel, [0, 1], np.array([1.0, 1.0], dtype=complex))
    cases.append({"operator": pauli(cancel), "num_qubits": 2, "indices": [0, 1],
                  "amplitudes": vector([1.0, 1.0]), "image_indices": got_i.tolist(),
                  "image_amplitudes": vector(got_a)})
    return cases


def matvec_cases():
    cases = []
    rng = np.random.default_rng(51)
    for n, count, seed in ((3, 10, 61), (6, 50, 62)):
        op = random_pauli(n, count, seed)
        state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        state /= np.linalg.norm(state)
        for cache in (0, 1 << 20):
            product = PauliOperator(op, cache_bytes=cache) @ state
            assert np.allclose(product, op.to_matrix() @ state, atol=1e-12)
        cases.append({"num_qubits": n, "operator": pauli(op), "state": vector(state),
                      "product": vector(op.to_matrix() @ state)})
    return cases


def main():
    out = {
        "conventions": "complex = [re, im]; Pauli sum = {label: complex} after simplify(1e-12)",
        "mappings": mapping_cases(),
        "algebra": algebra_cases(),
        "matrices": matrix_cases(),
        "fermion_matrices": fermion_matrix_cases(),
        "integrals": integral_cases(),
        "excitations": excitation_cases(),
        "reencoded": reencode_cases(),
        "reference_bits": reference_bit_cases(),
        "sectors": sector_cases(),
        "applied": apply_cases(),
        "matvec": matvec_cases(),
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
