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

## The two-electron integrals

The classical preparation has a memory limit of its own, set by the basis
size `M` rather than by the register. The two-electron tensor
`<ab|cd>` holds `M^4` complex numbers, `16 M^4` bytes: 1.3 MB for water in
PAW-LCAO-SZP (`M = 17`), 2.2 GB for benzene in DZP (`M = 108`), 13 GB for
naphthalene in DZP (`M = 170`). It is built from the `M(M+1)/2` pair
densities `psi_b^* psi_d`, one Poisson solve each. Every solve's potential
gives a whole slab of the tensor, `eri[:, b, :, d]`, from one matrix product,
so the build needs the result plus a working set. The working set is bounded
by `MANDACARU_ERI_MEMORY_MB` (256 MB by default). Every grid convolution
(Poisson solves, the spectral Laplacian, pair transforms) runs on one shared
FFT workspace per grid ({mod}`mandacaru.integrals.fft`). The workspace
transforms in place and prunes the zero padding of an isolated solve. Real
densities go through real transforms.

A small active space does not need the whole tensor. A closed-shell energy or
MP2 selection goes integral-direct when the tensor route would not fit
({doc}`active_space`). When a reusable representation of all the integrals
is wanted, the **Cholesky option** factors them instead:

```python
from mandacaru.algorithms import Mandacaru

calc = Mandacaru(method="adapt-vqe", basis={"name": "PAW-LCAO", "size": "DZP"},
                 active_space={"orbitals": 8},
                 two_body={"method": "cholesky", "tolerance": 1e-7})
```

The command line spells it `--two-body cholesky --cholesky-tolerance 1e-7`.

The integrals are the Coulomb inner products of pair densities, a positive
semidefinite matrix of low numerical rank. A pivoted Cholesky decomposition
writes them as `(ac|bd) = sum_r B^r_ac B^r_bd`. The pivots are batched in
two steps (Folkestad, Kjonstad and Koch 2019), and the decomposition runs on
the grid without ever forming the matrix:

- the functions are made real by the unitary their conjugation closure
  allows, so the pair densities, their potentials and the factors are real;
- the exact diagonal comes from Parseval, forward transforms only;
- pairs whose sampled supports do not meet are skipped;
- each pivot's column is one matrix product.

The rank `R` is a few times `M`, so the factors take about `8 M^2 R` bytes
instead of `16 M^4`. The SCF builds its Coulomb and exchange from the
factors (RHF, or UHF for an open shell). The active-space integrals come from
the factors too, so the tensor is formed only when every orbital is active.
The PAW-LCAO compensation charges stay in their structured form beside the
factors, exactly.

**The tolerance bounds every integral, not the energy.** Each residual
Coulomb self-energy left out is below it, which bounds every integral's error
(Cauchy-Schwarz). An energy sums many integrals, though. Measured against the
dense tensor on the same grid (grid_spacing = 0.3 Angstrom), the largest energy error
over the RHF, MP2 and active-space FCI energies of water (DZP), N2 (SZP) and
LiH (DZP), and the RHF energy of benzene (DZP), was:

| `tolerance` (Ha) | Largest energy error (Ha) | `R / M`, water / benzene |
|---|---|---|
| 1e-6 | 3.1e-6 | 7.0 / 7.5 |
| 3e-7 | 1.1e-6 | 7.9 / 8.7 |
| **1e-7** (default) | 2.3e-7 | 8.4 / 9.9 |
| 1e-8 | 7.0e-8 | 9.9 / 12.5 |

The factors do not resolve an integral finer than the tolerance, so the ones
that vanish by symmetry come out as noise up to about the tolerance instead
of zero (ethanol, DZP: at most 1.3e-7 Ha at the default, against 1.6e-5 Ha
for the smallest genuine active integral). The second-quantized operator
leaves out every active integral below the tolerance. Kept, that noise
doubled the qubit Hamiltonian: ethanol in TZP on 20 qubits had 14,251 Pauli
terms against the dense tensor's 7,151, and every energy evaluation paid for
them. Left out, it does not change the active-space FCI energy of water
(DZP) at 1e-10 Ha.

The default, 1e-7 Ha, is the loosest tolerance that kept every one of these
energies within 1e-6 Ha. A larger molecule accumulates more: confirm the
tolerance on your own system. For benzene at 1e-7 Ha the factors and the
compensation terms take 73 MB, against 2.2 GB for the tensor.

The first version is for molecules, energies and active spaces. The cases
that need the dense tensor are refused with the reason: a crystal (a periodic
kernel and Bloch phases), `method="dft"` (its Hartree term is a Poisson
solve of the density), GHF and a spin-orbit (`lda-fr`) basis (a spinor
tensor), and the plane-wave basis.

### Forces from the factors

Forces with the factors need an `active_space`. A factorization's pivots are
chosen as it runs, so they can change when an atom moves: the truncated
integrals are accurate at every geometry but not a smooth function of it.
Their analytic gradient has to fix the pivots and differentiate the resulting
fitting (Aquilante, Lindh and Pedersen 2008), which Mandacaru does not
implement, so without an active space forces are refused.

With an active space, the force is the central difference of the reduced
Hamiltonian along each internal direction (the same route as `"dlpno-mp2"`,
see {doc}`active_space`), so only energies are computed and no factor is
differentiated. Each displaced geometry is factored on the **reference's
pivots**: with `C` the Coulomb columns of the pivot pair densities and `A`
their pivot rows, the integrals are `C A^-1 C^T`. That is density fitting in
the Coulomb metric whose auxiliary functions are the pivot pair densities
(Aquilante, Lindh and Pedersen 2008). At the reference geometry it is the
factorization itself, to round-off. At a displaced one it moves smoothly with
the nuclei, and it skips the diagonal, so a displaced build costs one Poisson
solve per pivot. The force is the gradient of the fitted energy, and like
every integral-direct active-space force it has no Hellmann-Feynman/Pulay
breakdown.

```python
calc = Mandacaru(method="adapt-vqe", basis={"name": "PAW-LCAO", "size": "TZP"},
                 active_space={"orbitals": {"occupied": 5, "virtual": 5},
                               "method": "mp2"},
                 two_body={"method": "cholesky", "tolerance": 1e-7})
```
