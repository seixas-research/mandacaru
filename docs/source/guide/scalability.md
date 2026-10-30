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
                       resources={"threads": 2,
                                  "memory_limit_bytes": 2 * 1024**3})
atoms.get_potential_energy()
print(f"peak native memory: {atoms.calc.context.memory_peak_bytes / 2**20:.0f} MiB")
```

`resources` configures the native context: `threads`, `memory_limit_bytes`
(512 MiB by default) and `scratch_dir`. The context refuses an allocation that
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
                 two_body={"method": "cholesky", "tolerance": 1e-8})
```

`cholesky_tolerance` (default `1e-10`) and `two_body={"method": "cholesky",
"tolerance": ...}` set the same truncation; `two_body` accepts nothing but the
Cholesky method and its tolerance. The tolerance bounds every integral, not the
energy: an energy sums many integrals, so confirm the tolerance on your own
system by tightening it until the energy stops moving. A looser tolerance gives
fewer factors and less memory.

`precision="float32"` stores the sampled orbitals in `float32`; the one-body
matrices, the integral accumulation and the FFT workspaces stay in `float64`.

A reduced register is the usual way to fit a large basis; see
[the active-space guide](active_space.md).
