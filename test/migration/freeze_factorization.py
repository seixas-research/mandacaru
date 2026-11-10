"""Freeze the factorized-measurement oracle before its Python tail is deleted (MG-Q1, MG-Q6).

The Python tail of ``backends/factorization.py`` -- ``givens_generator``,
``basis_rotation``, ``_diagonal_observable``, ``measurement_problems`` and
``factorized_energy`` -- was the only implementation of the ``O(M)`` basis-rotation
measurement. MG-Q6 ports it to Rust and needs these numbers as its oracle.

The tail no longer exists in the working tree. Run once, from the repository root,
against the last commit that still carried it::

    git show <commit>:src/mandacaru/backends/factorization.py > /tmp/factorization_oracle.py
    python test/migration/freeze_factorization.py /tmp/factorization_oracle.py

(without an argument the script reads ``git show HEAD:...``, which works only while HEAD
still has the tail). It writes ``test/migration/reference/factorization.json``, read by
``test_native_factorization.py``.

Input description. The molecular integrals are not exposed by the calculator, so they
are recovered from its qubit Hamiltonian (Jordan-Wigner, spin-block order, spin-free,
real): the vacuum element is the constant, the one-particle block is ``h``, and the
opposite-spin two-particle block is the physicists' spatial tensor ``<pq|rs>`` once the
one-body and constant parts are removed. The recovery is checked by rebuilding the qubit
operator with ``Fermion.from_spatial_integrals`` (maximum coefficient difference is
stored). A Rust test rebuilds the Hamiltonian from ``constant``, ``one_body`` and
``two_body``; ``geometry`` only says where they came from.

State. The Hartree-Fock determinant followed by the first three fermionic pool
generators at fixed angles, as a product of exponentials (what a circuit prepares).

Energies. ``exact`` is ``<psi|H|psi>`` on the state vector, ``qwc`` the provider's
energy of the same state through the qubit-wise commuting path, ``factorized`` the sum
of the constant and the ``L + 1`` basis energies of the oracle, ``per_basis`` those
``L + 1`` values.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.backends.providers import QiskitProvider
from mandacaru.backends.factorization import givens_decomposition, spatial_double_factorization
from mandacaru.core import PauliSum
from mandacaru.core.checkpoint import prepare_state
from mandacaru.core.mapping import Fermion

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "reference" / "factorization.json"
OPTIONS = {"method": "vqe", "basis": "HAO", "grid_spacing": 0.35}
CASES = {
    "H2": {"symbols": "H2", "distance": 0.74, "cell": 8.0},
    "LiH": {"symbols": "LiH", "distance": 1.60, "cell": 9.0},
}
THETAS = [0.21, -0.13, 0.08]


def load_oracle(path):
    """The deleted tail, loaded as ``mandacaru.backends._factorization_oracle``."""
    if path is None:
        source = subprocess.run(
            ["git", "show", "HEAD:src/mandacaru/backends/factorization.py"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
        path = Path(tempfile.mkdtemp()) / "factorization_oracle.py"
        path.write_text(source)
    name = "mandacaru.backends._factorization_oracle"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mandacaru.backends"
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "factorized_energy"):
        raise SystemExit(f"{path} has no factorized_energy: pass the pre-deletion source")
    return module


def terms(operator):
    """``[[label, re, im], ...]`` sorted by label."""
    return [[label, complex(c).real, complex(c).imag]
            for label, c in sorted(operator.simplify().terms.items())]


def pauli_sum(triples, n):
    return PauliSum({label: complex(re, im) for label, re, im in triples}, num_qubits=n)


def fock_index(n, occupied):
    """Qubit 0 is the most significant bit."""
    return sum(1 << (n - 1 - k) for k in occupied)


def recover_integrals(hamiltonian, n_spatial):
    """``(constant, h, <pq|rs>)`` of a spin-free Jordan-Wigner Hamiltonian, spin-block order."""
    n, m = 2 * n_spatial, n_spatial
    matrix = np.asarray(hamiltonian.to_matrix())
    element = lambda bra, ket: matrix[fock_index(n, bra), fock_index(n, ket)]
    constant = element([], [])
    assert abs(constant.imag) < 1e-12
    constant = float(constant.real)
    one = np.array([[element([p], [q]) - constant * (p == q) for q in range(n)]
                    for p in range(n)])
    assert np.abs(one.imag).max() < 1e-10
    h = one.real[:m, :m]
    assert np.abs(one.real[m:, m:] - h).max() < 1e-10 and np.abs(one.real[:m, m:]).max() < 1e-10
    eri = np.zeros((m, m, m, m))
    for p in range(m):
        for q in range(m):
            for r in range(m):
                for s in range(m):
                    value = element([p, m + q], [r, m + s]) - constant * (p == r) * (q == s)
                    value -= h[p, r] * (q == s) + h[q, s] * (p == r)
                    assert abs(value.imag) < 1e-10
                    eri[p, q, r, s] = value.real
    return constant, h, eri


def freeze_case(name, spec, oracle):
    atoms = Atoms(spec["symbols"], positions=[(0, 0, 0), (0, 0, spec["distance"])],
                  cell=[spec["cell"]] * 3)
    atoms.center()
    atoms.calc = Mandacaru(trace=False, **OPTIONS)
    atoms.get_potential_energy()
    problem = atoms.calc.problem
    n = problem.num_qubits
    n_spatial = n // 2
    hamiltonian = pauli_sum(problem.hamiltonian_terms, n)
    constant, h, eri = recover_integrals(hamiltonian, n_spatial)

    rebuilt = Fermion.from_spatial_integrals(h, eri).map_to_qubits(
        "jordan_wigner", n_modes=n) + PauliSum({"I" * n: constant}, num_qubits=n)
    difference = (hamiltonian + (-1.0) * rebuilt).simplify()
    recovery_error = max((abs(complex(c)) for c in difference.terms.values()), default=0.0)
    assert recovery_error < 1e-9, recovery_error

    factorization = spatial_double_factorization(h, eri, constant=constant)
    m_modes = factorization.n_modes
    assert m_modes == n

    occupied = [int(q) for q in problem.reference_qubits]
    generators = [pauli_sum(t, n) for t in problem.pool_terms[:len(THETAS)]]
    thetas = list(THETAS)

    def rotation(unitary):
        rotations, signs = givens_decomposition(unitary)
        generators_, thetas_ = oracle.basis_rotation(m_modes, unitary)
        assert [float(t) for _p, t in rotations] == [float(t) for t in thetas_]
        return {"rotations": [[int(p), float(t)] for p, t in rotations],
                "signs": [float(s) for s in signs],
                "generators": [terms(g) for g in generators_],
                "thetas": [float(t) for t in thetas_]}

    problems = oracle.measurement_problems(factorization, n, occupied, generators, thetas)
    provider = QiskitProvider(device="statevector")
    per_basis = [float(v) for v in provider.energies(problems)]
    factorized = float(oracle.factorized_energy(provider, (n, occupied, generators, thetas),
                                                factorization))
    assert abs(factorized - (factorization.constant + sum(per_basis))) < 1e-12

    psi = prepare_state(n, occupied, generators, thetas, "product")
    exact = float(np.vdot(psi, np.asarray(hamiltonian.to_matrix()) @ psi).real)
    qwc = float(provider.energy(n, occupied, generators, thetas, hamiltonian))

    return {
        "geometry": {"symbols": spec["symbols"],
                     "positions": [[float(x) for x in row] for row in atoms.get_positions()],
                     "cell": [spec["cell"]] * 3, "options": OPTIONS},
        "n_spatial_orbitals": n_spatial, "n_modes": n, "mapping": "jordan_wigner",
        "convention": {"one_body": "spatial h_pq", "two_body": "spatial <pq|rs>, physicists, "
                       "spin-free; spin orbital P = p + sigma * M",
                       "constant": "added to the energy; the qubit identity coefficient "
                                   "minus the integrals' own"},
        "constant": constant, "one_body": h.tolist(), "two_body": eri.tolist(),
        "hamiltonian_terms": terms(hamiltonian), "recovery_error": recovery_error,
        "factorization": {
            "call": "spatial_double_factorization(one_body, two_body, constant=constant)",
            "measurement_bases": factorization.measurement_bases,
            "one_norm": float(factorization.one_norm),
            "truncation_error": float(factorization.truncation_error),
            "constant": float(factorization.constant),
            "one_body_energies": factorization.one_body_energies.tolist(),
            "one_body_orbitals": factorization.one_body_orbitals.tolist(),
            "leaves": [{"weight": leaf.weight, "orbitals": leaf.orbitals.tolist(),
                        "coefficients": leaf.coefficients.tolist()}
                       for leaf in factorization.leaves]},
        "state": {"n_qubits": n, "occupied": occupied, "thetas": thetas,
                  "generators": [terms(g) for g in generators],
                  "preparation": "product"},
        "energies": {"exact": exact, "qwc": qwc, "factorized": factorized,
                     "per_basis": per_basis},
        "givens_generators": {str(p): terms(oracle.givens_generator(m_modes, p))
                              for p in range(m_modes - 1)},
        "basis_rotations": {"one_body": rotation(factorization.one_body_orbitals),
                            **{f"leaf_{k}": rotation(leaf.orbitals)
                               for k, leaf in enumerate(factorization.leaves)}},
        "measurement_problems": [
            {"n_qubits": int(p[0]), "occupied": [int(q) for q in p[1]],
             "num_generators": len(p[2]), "thetas": [float(t) for t in p[3]],
             "observable": terms(p[4])} for p in problems],
    }


def main():
    oracle = load_oracle(sys.argv[1] if len(sys.argv) > 1 else None)
    out = {"description": __doc__.strip().splitlines()[0], "thetas": THETAS, "cases": {}}
    for name, spec in CASES.items():
        out["cases"][name] = freeze_case(name, spec, oracle)
        energies = out["cases"][name]["energies"]
        print(name, "exact", energies["exact"], "qwc", energies["qwc"],
              "factorized", energies["factorized"])
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
