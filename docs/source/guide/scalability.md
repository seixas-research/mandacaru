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

With `sector=True`, or the default `sector="auto"` (from 16 qubits for
ADAPT-VQE, from 10 for `method="vqe"`), all three forms act on the
particle-number sector's `C(M, n_alpha) C(M, n_beta)` states instead of `2^n`.

```python
from mandacaru import Mandacaru

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

## The fixed-ansatz VQE in the sector

`method="vqe"` with the default UCCSD ansatz used to hold a dense `2^n x 2^n`
matrix for the Hamiltonian and one for every excitation generator: 16 * 4^n
bytes each, 4 GiB at 14 qubits, and a 12-qubit water or methanol active space
(92 to 117 generators) needed more than 25 GB. It now evolves the state in the
particle-number sector like ADAPT-VQE: sector vectors of
`C(M, n_alpha) C(M, n_beta)` amplitudes and sparse sector generators, with no
array of size `4^n` anywhere. `sector="auto"` (the default) turns it on from
10 qubits, `sector=True` / `False` force it, and a Hamiltonian or generator
that does not conserve `(n_alpha, n_beta)` keeps the full register with a
warning (`sector=True` raises instead). Only the UCCSD ansatz on an untapered
register prepares sector states; the HVA, a pre-built ansatz object and the
subspace solver keep the register.

The exact exponential `exp(sum_k theta_k G_k)|HF>` is applied with
`scipy.sparse.linalg.expm_multiply` to the summed sparse generator, whose
sparsity pattern is built once (each evaluation only reweights it). With
`ansatz={"name": "uccsd", "trotter": True}` the state is the ordered product of
closed-form exponentials, which also gives the optimizer the analytic adjoint
gradient (the `parameter_gradient` line of `[OPTIMIZATION SETUP]` says which
gradient ran): far fewer evaluations for a gradient method, so prefer it for
the hundreds of parameters of a 16- or 20-qubit active space.

Measured on a random molecular-type Hamiltonian (10 qubits, 54 parameters,
60 COBYLA evaluations, shared machine, indicative): 62.5 s and 1157 MiB with
the dense matrices, 0.6 s and 122 MiB in the sector, the same energy to
1e-10 Ha. Water in PAW-LCAO-DZ, 10 qubits: 52.8 s and 1374 MiB against 8.1 s
and 453 MiB, energy -458.808317 eV in both.

## Building the sector matrices

The Hamiltonian and the generators are restricted to the sector by flip mask
(a Pauli string sends `|x>` to a phase times `|x XOR f>`, and the strings that
share `f` share their images). Three costs were removed:

- the images are found in a table of `2^n` `int32` positions (64 MiB at 24
  qubits) instead of by binary search;
- the groups are written straight into the CSR arrays, counting each row's
  entries first. A flip mask is `x_row XOR x_col`, so no two groups reach the
  same entry and there is nothing to merge: the peak is the matrix plus a few
  vectors of `dim` entries, where the coordinate-format fold held 3.8 times
  the matrix (35 MiB against 129 MiB for a 16-qubit, 1.77-million-entry
  Hamiltonian);
- the Hermitian part `(H + H^dagger)/2` is taken on the coefficients (every
  Pauli string is Hermitian, so it is the real part of each coefficient)
  rather than on the matrix, which formed a transposed copy and a sum.

Methanol, PAW-LCAO-DZ, a CAS(10,10) MP2 active space on 20 qubits
(`pool="fermionic"`, 875 generators, 7,147 Pauli terms, 63,504 sector states,
6 ADAPT iterations, same energy -640.437776 eV before and after): peak
resident memory 2559 MiB -> 1578 MiB, the sparse restriction of the
Hamiltonian and the pool 4.8 s -> 3.6 s (shared machine, indicative). The
stored Hamiltonian is 532 MiB and the pool 401 MiB; what is left is those two
matrices. A random 20-qubit Hamiltonian of 14,251 terms had a 4.5 GB setup peak
and now peaks at 1.9 GB. `PauliSum.to_sparse_matrix` uses the same fill for
the full register: 12 qubits, 1,819 terms, 11.4 s -> 0.1 s.

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

What such a register cannot do is *simulate*: no local state vector exists at
this width. It can be **optimized on a device**. With `shots > 0` a run's
classical memory is polynomial in the qubit count -- no `2^n` array, no
sector-sized vector or matrix, no local screening: VQE measures every energy,
and ADAPT-VQE measures its energies and its pool gradients
`<psi|[H, A_i]|psi>` from the strings of these same commutators, grouped into
qubit-wise-commuting bases ({ref}`adapt-on-a-device`). The reference energy
is the determinant's, read off the `I`/`Z` strings. The test suite runs a
50-qubit ADAPT-VQE and a 50-qubit UCCSD-VQE this way, against a device double,
with every exponential constructor (`to_matrix`, `to_sparse_matrix`, the
sector, `Statevector`, `expm_multiply`, the `2^n` reference) made to raise:
the ADAPT run (624-operator fermionic pool, three growth steps) peaks at 24 MiB
of traced Python memory, the VQE run (624 parameters) at 4 MiB.

What it costs on the device is the other question, and the answer at 50
qubits is the measurement wall of {doc}`measurement_cost`, not memory.

## The two-electron integrals

The classical preparation has a memory limit of its own, set by the basis
size `M` rather than by the register. The two-electron tensor `<ab|cd>`
holds `M^4` numbers. A molecule's basis functions are real, so its integrals
have the eightfold symmetry `(ac|bd) = (ca|bd) = (bd|ac) = ...`, and
Mandacaru keeps only the distinct ones: the lower triangle of the symmetric
matrix over the `M(M+1)/2` pairs, about `M^4` bytes against the `8 M^4` of
the full real array. That is 0.09 MB for water in PAW-LCAO-SZP (`M = 17`),
139 MB for benzene in DZP (`M = 108`) and 845 MB for naphthalene in DZP
(`M = 170`), whose full array (6.7 GB real) did not fit in 8 GB at all. The
engine writes it packed as the potentials are solved, and the SCF (its
Coulomb and exchange), the orbital transforms, the active-space selectors,
the frozen core, the fermionic operator, the double factorization and the
nuclear gradient all read it packed
({class}`~mandacaru.integrals.packed.PackedERI`). The few consumers that
need the whole array -- a spin-orbit (spinor) or spin-orbital tensor, a
hybrid functional's exchange tensor, the Kohn-Sham handoff -- write it out
for themselves and drop it. A crystal keeps the full complex array (its
Bloch functions are complex). It is built from the `M(M+1)/2` pair
densities `psi_b^* psi_d`, one Poisson solve each. Every solve's potential
gives a whole slab of the tensor, `eri[:, b, :, d]` (the row of its pair in
the packed triangle), from one matrix product, so the build needs the
result plus a working set. The working set is bounded
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
from mandacaru import Mandacaru

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

- a molecule's basis functions are real (real spherical harmonics), so the
  pair densities, their potentials and the factors are real;
- the exact diagonal comes from Parseval, forward transforms only;
- pairs whose sampled supports do not meet are skipped;
- each pivot's column is one matrix product.

The rank `R` is a few times `M`, so the factors take about `4 M^2 R` bytes
(the upper triangles) instead of the packed tensor's `M^4` -- and they never
need the tensor's working set. The run log writes the
tolerance, the rank and the largest residual diagonal on one `[ELECTRONS]`
line ({doc}`run_output`). The SCF builds its Coulomb and exchange from the
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
compensation terms take 73 MB, against 139 MB for the packed tensor (and
the dense route's build, SCF and transforms around it).

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
