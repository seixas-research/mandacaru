# Backend Providers: Qiskit, Braket and Cirq

Carcará's ansätze are products of exponentials of anti-Hermitian generators,

$$|\psi(\vec\theta)\rangle = \prod_k e^{\theta_k A_k}\,|\mathrm{HF}\rangle .$$

The `backend_provider` argument chooses which quantum SDK **constructs** those
circuits — and, with `execute_circuits=True`, which SDK **runs** them.

```python
from carcara.algorithms import ADAPTVQE

ADAPTVQE(basis="FAO", backend_provider="qiskit")   # default
ADAPTVQE(basis="FAO", backend_provider="braket")   # amazon-braket-sdk
ADAPTVQE(basis="FAO", backend_provider="cirq")     # cirq
```

| provider | package | executes on |
|---|---|---|
| `"qiskit"` | `qiskit` | `qiskit.quantum_info.Statevector` |
| `"braket"` | `amazon-braket-sdk` | local simulator **or the AWS Braket service** |
| `"cirq"` | `cirq` | `cirq.Simulator` |

---

## Why the three agree exactly

Each generator is a qubit `PauliSum` whose terms are $A = \sum_j i\,c_j P_j$ with
**real** $c_j$ and **mutually commuting** Pauli strings $P_j$ — a property of the
fermionic and qubit excitation generators. The exponential therefore factorizes
*exactly*, with no Trotter error:

$$e^{\theta A} = \prod_j e^{i\,\theta c_j P_j},$$

and each factor is the textbook Pauli-rotation circuit: a basis change to the
$Z$ axis, a CNOT ladder accumulating the parity onto one qubit, an
$R_z(-2\theta c_j)$, then the ladder and basis change undone.

Because the decomposition is exact, all three providers emit the *same* unitary
from the *same* gate set (`X`, `H`, `S`, `S†`, `CNOT`, `R_z`) and must agree with
the internal NumPy state-vector backend to machine precision. That equivalence is
asserted in the test suite and demonstrated end-to-end on LiH:

```text
provider            E (Ha)   err vs FCI   ops  cnots  depth     time
(matrix)       -6.88824276     1.34e-07     8    208    273     0.3s
qiskit         -6.88824283     6.27e-08     8    208    273     4.1s
braket         -6.88824279     1.09e-07     8    208    359    43.9s
cirq           -6.88824281     8.17e-08     8    208    358     8.9s
```

:::{note}
**Endianness.** Carcará puts qubit 0 in the *most* significant position of the
amplitude index (the leftmost Kronecker factor), matching Braket and Cirq.
Qiskit is little-endian, so the Qiskit provider lays Carcará qubit `k` on Qiskit
wire `n-1-k`. Gate counts are unaffected — relabeling is an isomorphism.
:::

---

## Building versus executing

`execute_circuits` decides whether circuits are only *built* (for gate-count
profiling) or actually *run* to prepare each state:

```python
ADAPTVQE(backend_provider="qiskit")                        # execute_circuits=False
ADAPTVQE(backend_provider="braket")                        # execute_circuits=True
ADAPTVQE(backend_provider="qiskit", execute_circuits=True)  # opt in
ADAPTVQE(backend_provider="cirq",  execute_circuits=False)  # opt out
```

It defaults to `True` for `"braket"` and `"cirq"` — naming them is a request to
use them — and `False` for `"qiskit"`, which keeps the fast NumPy state-vector
numerics unless execution is asked for explicitly. **Circuit execution is orders
of magnitude slower** (one simulator invocation per energy evaluation) and gives
the same answer, so it is a verification and hardware path, not a performance
one.

Circuit *profiling* always uses the named SDK. Counts differ between providers
because only Qiskit re-optimizes during transpilation; the unitary does not.

---

## Using an ansatz directly

The providers are also usable below the driver level:

```python
from carcara.backends.providers import build_provider
from carcara.circuits import AdaptAnsatz
from carcara.circuits.pools import build_pool

pool = build_pool("qeb", 2, (1, 1))
provider = build_provider("cirq")

ansatz = AdaptAnsatz(4, pool.occupied_orbitals, "jordan_wigner",
                     provider=provider)
for op in pool.operators()[:3]:
    ansatz.append(op)

psi = ansatz.state([0.31, -0.72, 0.45])    # executed as a real circuit
circuit = provider.build(4, ansatz.reference_qubits(),
                         ansatz.pauli_generators, [0.31, -0.72, 0.45])
print(circuit)
```

{class}`~carcara.circuits.UCCSD` accepts `provider=` too, but a circuit realizes
the **Trotter product** form, so `trotter=True` is required:

```python
from carcara.circuits import UCCSD

UCCSD(2, (1, 1), trotter=True, provider=build_provider("braket"))
```

The VQE driver does this automatically when circuit execution is on.

:::{note}
A circuit can only be *initialized* in a computational basis state, so provider
execution accepts Slater-determinant references only — which is what the
Hartree-Fock reference and the SSVQE reference determinants are. A superposition
reference raises a clear `ValueError`.
:::

---

## Availability

Naming a provider never fails at import time; the SDK is imported on first use.

```python
from carcara.backends.providers import BACKEND_PROVIDERS, provider_available

BACKEND_PROVIDERS                 # ('qiskit', 'braket', 'cirq')
provider_available("braket")      # True if amazon-braket-sdk is importable
```

Aliases are accepted: `"ibm"` → `qiskit`, `"aws"` / `"amazon-braket"` →
`braket`, `"google"` → `cirq`.

---

## Next

Running on the **AWS Braket service**, including real QPUs, needs the shot-based
measurement path — see {doc}`aws_braket`.
