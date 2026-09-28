# Memory and scale: what fits, and what a 100-qubit problem can do

A state-vector simulation holds two kinds of objects: the **operators** (the
Hamiltonian and, for ADAPT-VQE, every pool generator) and the **state
vectors**. Mandacaru can shrink the first kind as far as you like. The second
kind is exponential in the register, and no representation removes that. This
guide covers both, and what is left for a register too large to simulate.

## The three operator forms

The `sparse` option chooses how the operators are held:

| `sparse=` | Operators | Memory | Product cost |
|---|---|---|---|
| `False` | dense `2^n x 2^n` matrices | `16 * 4^n` bytes each | fastest below ~10 qubits |
| `True` | CSR matrices | one entry per state per flip group | fast |
| `"matrix-free"` | Pauli strings (masks and coefficients) | polynomial, plus a bounded cache | slower |

In `"matrix-free"` form nothing of size `2^n x 2^n` is ever built, dense or
sparse. A Pauli string sends a basis state `|x>` to a sign times
`|x XOR f>`: its `2x2` factors reduce to one flip mask `f` and one phase mask
`z`. Applying the string to a vector is therefore an XOR, a popcount and a
gather per amplitude. Strings that share a flip mask share their images, so a
molecular Hamiltonian costs one gather per group, about nine times fewer than
its terms ({class}`~mandacaru.core.matrix_free.PauliOperator`).

With `sector=True`, or the default `sector="auto"` from 16 qubits, all three
forms act on the particle-number sector's `C(M, n_alpha) C(M, n_beta)` states
instead of `2^n`.

```python
from mandacaru.algorithms import Mandacaru

calc = Mandacaru(method="adapt-vqe", basis="HAO", pool="fermionic",
                 sparse="matrix-free")
```

**The default, `sparse="auto"`,** is dense below 10 qubits and CSR above. It
switches to matrix-free only when the stored matrices might not fit. The CSR
size is bounded by one entry per state per flip group; when that bound, summed
over the Hamiltonian and the pool, exceeds a quarter of the machine's physical
memory, the operators are applied matrix-free instead. The run log's
`state_vector_backend` line says which form ran.

**What the trade costs**, measured on a random 20-qubit molecular-type
Hamiltonian (14,251 terms, 2,536 flip groups) in its 14,400-state sector:

| Form | Held between products | First `H @ psi` | Each later one |
|---|---|---|---|
| CSR | 176 MB | 16.6 ms | 16.6 ms |
| matrix-free, every group cached | 176 MB | 3.1 s | 18.8 ms |
| matrix-free, nothing cached | a few vectors | 2.9 s | 2.9 s |

A matrix-free operator stores its cached groups as one CSR block, built on
the first product. When every group fits
({data}`~mandacaru.core.matrix_free.MATRIX_FREE_CACHE_BYTES`), it is the CSR
matrix under another name. Past that bound, the remaining groups are
recomputed on every product, which is where the memory saving and the cost
both come from. The pool is never cached, because it can hold millions of
generators: one screening pass over this problem's 609 generators takes 1.1 s,
against 0.03 s with stored matrices. Matrix-free is therefore the form for
problems whose matrices do not fit, not a default.

## The limit no operator form removes

A state vector has `2^n` complex amplitudes, 16 bytes each, and an adaptive
run holds several at once:

| Qubits | One state vector |
|---|---|
| 28 | 4 GiB |
| 30 | 16 GiB |
| 36 | 1 TiB |
| 100 | 2 x 10^31 bytes |

Exact full-register simulation ends near 30 qubits. A particle-number sector
moves the line but does not remove it: 50 spatial orbitals at half filling
have `C(50, 25)^2`, about `10^28`, states. A run whose state vectors cannot fit
is refused with the arithmetic before anything is allocated:

```text
MemoryError: a 100-qubit simulation cannot fit: its state vector has 2^100
amplitudes, ...
```

## What still works at 100 qubits

Everything before the simulation is polynomial and never builds a matrix:

- **the Hamiltonian:** {meth}`~mandacaru.core.mapping.Fermion.map_to_qubits`
  expands all terms of one length at once on bit tables. It costs about
  0.1 ms per two-body term on 100 qubits, so a Hamiltonian with `10^7` terms
  maps in about twenty minutes rather than hours;
- **the pool:** its generators, mapped the same way;
- **the gradient observables:** the ADAPT screening gradient of generator
  `A` is `g = <psi|[H, A]|psi>`, and
  {meth}`~mandacaru.core.mapping.PauliSum.commutator` forms `[H, A]` from the
  anticommutation rule on the same bit tables. Two strings contribute `2PQ`
  when they anticommute and nothing otherwise, and a 100-qubit commutator
  takes milliseconds;
- **measurement grouping and circuits** ({doc}`measurement_cost`).

```python
from mandacaru.circuits.gates import double_excitation
from mandacaru.core import Fermion

sites = 50                      # a Hubbard chain on 100 qubits
terms = {}
for spin in (0, sites):
    for i in range(sites - 1):
        p, q = i + spin, i + 1 + spin
        terms[((p, True), (q, False))] = -1.0
        terms[((q, True), (p, False))] = -1.0
for i in range(sites):
    terms[((i, True), (i, False), (i + sites, True), (i + sites, False))] = 4.0
H = Fermion(terms, n_modes=2 * sites).map_to_qubits("jordan_wigner")

A = double_excitation(10, 60, 11, 61).map_to_qubits("jordan_wigner",
                                                    n_modes=100)
observable = H.commutator(A)     # Hermitian; measure <psi|[H, A]|psi>
```

What such a register cannot do is *optimize*. Every screening pass and every
inner optimization would need `<psi|[H, A_i]|psi>` measured on a quantum
processor for every pool operator. Mandacaru optimizes on the local state
vector and sends only the optimized state to hardware for one measurement
({doc}`measurement_cost`), so a 100-qubit problem can be prepared, mapped and
costed here, but not solved.
