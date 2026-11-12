# Memory and scale: what fits, and what a 100-qubit problem can do

A state-vector simulation holds two kinds of objects: the **operators** (the
Hamiltonian and, for ADAPT-VQE, every pool generator) and the **state vectors**.
The operators can be shrunk. The state vectors are exponential in the register,
and no representation removes that. This guide covers both, the classical
preparation that comes before them, and what is left for a register too large to
simulate.

## Operators and the particle-number sector

Two options choose how the operators are held.

* `sector` restricts every operator and state to the particle-number sector, the
  `C(M, n_alpha) C(M, n_beta)` states with the right electron counts, instead of
  all $2^n$. It is `True`, `False` or `None`; `None` (the default) means on
  unless the pool is `"qubit"` or `"spin-orbit"`, whose operators do not conserve
  the particle number, or the ansatz is the HVA.
* `sparse` is `True` or `False` and holds the operators in compressed sparse
  form instead of dense matrices.

Both are booleans; any other value, including a string, is refused.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       grid_spacing=0.25,
                       pool="fermionic",
                       sector=True,
                       sparse=True,
                       parallelization={"threads": 2},
                       resources={"memory_limit_bytes": 2 * 1024**3})
atoms.get_potential_energy()
print(f"peak native memory: {atoms.calc.context.memory_peak_bytes / 2**20:.0f} MiB")
```

Two options configure the native context. `parallelization` sets the processors
(here two threads; [Parallel layout](#dft-parallel-layout) for a crystal's `kpts`
level). `resources` sets the memory: `memory_limit_bytes` (32 GiB by default),
`disk_offload_bytes` and `scratch_dir`. The context refuses an allocation that
would exceed the limit, rather than exhausting the machine, and reports its peak
as `calc.context.memory_peak_bytes` and `result.timings["memory_peak_bytes"]`.
Raise the limit for a larger problem.

## The limit no operator form removes

A state vector has $2^n$ complex amplitudes at 16 bytes each, and an adaptive run
holds several at once:

| Qubits | One state vector |
|---|---|
| 28 | 4 GiB |
| 30 | 16 GiB |
| 36 | 1 TiB |
| 100 | $2\times10^{31}$ bytes |

Exact full-register simulation ends near 30 qubits. The particle-number sector
moves the line but does not remove it: 50 spatial orbitals at half filling have
$\binom{50}{25}^2\approx10^{28}$ states.

## What a register that large can do

A register too wide to simulate can still be **optimized on a device**. With
`shots > 0` a run's classical memory is polynomial in the qubit count: no $2^n$
array, no sector-sized vector or matrix, and no local state-vector screening.
VQE measures every energy, and ADAPT-VQE measures its energies and its pool
gradients $\langle\psi|[H, A_k]|\psi\rangle$ from the Pauli strings of those
commutators, grouped into qubit-wise commuting bases
([ADAPT-VQE on a device](#adapt-on-a-device)). The reference energy is the
determinant's, read off the `I`/`Z` strings. The test suite runs a 50-qubit
shot-based ADAPT-VQE and VQE against a device double with every exponential
constructor made to raise.

What it costs on the device is the other question. At 50 qubits the answer is the
measurement wall of [what a hardware measurement costs](measurement_cost.md), not
memory. A smaller active space or a smaller pool shrinks each gradient step.

## The two-electron integrals

The classical preparation has a memory limit of its own, set by the basis size
$M$ rather than by the register. The Coulomb integrals $(ac|bd)$ would need
$M^4$ numbers as a tensor; Mandacaru holds them as **Cholesky factors**,
$(ac|bd)=\sum_r B^r_{ac}B^r_{bd}$. The mean-field
exchange is built from the factors, and the active-space integrals come from them
too, so the full tensor is not formed for a reduced register.

```python
from mandacaru import Mandacaru

calc = Mandacaru(method="adapt-vqe",
                 basis={"name": "PAW-LCAO", "size": "DZP"},
                 active_space={"orbitals": 8, "method": "mp2"},
                 two_body={"method": "cholesky", "tolerance": 1e-6})
```

`two_body={"method": "cholesky", "tolerance": t}` sets the truncation: the
factorization stops once the largest remaining diagonal integral is at or below
`t`, in eV (default `1e-8` eV). The tolerance bounds every integral, not the energy: an energy sums many integrals, so confirm the tolerance on your own
system by tightening it until the energy stops moving. A looser tolerance gives
fewer factors and less memory.

(double-factorized-representation)=
### The double-factorized representation of the active space

By default the Hamiltonian of the active space holds its $n^4$ two-electron tensor
(`"representation": "dense"`). With

```python
two_body={"method": "cholesky", "representation": "double-factorized",
          "truncation": 0.01}
```

it holds instead each Cholesky vector rotated into the $n$ active orbitals and that
vector's eigen decomposition, $L^r = \sum_t \lambda^r_t u^r_t u^{rT}_t$ (at most
$L n^2$ numbers, $L$ the number of vectors), and builds the qubit Hamiltonian from
them one vector at a time, never forming the tensor or the list of its $n^4$ fermion
terms. `truncation` (eV) is required: eigenpairs are dropped, smallest first, while
$2\sum_r \big[(\sum_t|\lambda^r_t|)^2 - (\sum_{t\,\mathrm{kept}}|\lambda^r_t|)^2\big]$
stays at or below it. That quantity bounds the norm of the change of the two-body
operator, so **no energy of the model moves by more than `truncation`**, plus the
roundoff of the Pauli coefficients at or below $10^{-12}$ Ha that the operator drops (as
the dense route drops tensor elements at that level). `0` drops only roundoff, and the Hamiltonian is then the dense one to $10^{-9}$ Ha per Pauli
coefficient. The run log's `two-body integrals` line states the truncation, the
eigenpairs kept and the bound reached, in the log's energy unit.

Measured on H₂O (PAW-LCAO DZP, `grid_spacing=0.25`, the oxygen 1s frozen), ADAPT-VQE
on 6 active orbitals moved from the dense energy by $2\times10^{-12}$, $-7\times10^{-5}$,
$6\times10^{-3}$ and $3\times10^{-2}$ eV at truncations of 0, 0.01, 0.1 and 1 eV. Building
the qubit Hamiltonian of 20 active orbitals (40 qubits) peaked at 160 MB of process
memory against 281 MB for the dense tensor. A truncation breaks the point-group zeros
of the Hamiltonian, so it can **raise** the number of Pauli terms (on 20 orbitals,
59,825 untruncated against 119,105 at 0.1 eV): it lowers memory, not the measurement
count.

A double-factorized model has no dense tensor, and what needs one raises
`NotImplementedError` naming the representation: the analytic forces,
`mp2_natural_orbitals()`, `measurement_scheme="double-factorized"` and the model's
`two_body`. Periodic DFT takes only `"dense"`.

`precision="float32"` stores the sampled orbitals in `float32`; the one-body
matrices, the integral accumulation and the FFT workspaces stay in `float64`.

A reduced register is the usual way to fit a large basis; see
[the active-space guide](active_space.md).
