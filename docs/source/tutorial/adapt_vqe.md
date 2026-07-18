# ADAPT-VQE and operator pools

ADAPT-VQE (Grimsley *et al.*, 2019) builds a compact, problem-tailored ansatz
**one operator at a time**, instead of using a fixed template like UCCSD. Each
iteration it

1. evaluates the energy gradient of appending every operator $A_i$ in a pool,
   ```{math}
   g_i = \frac{\partial E}{\partial\theta_i}\Big|_{\theta_i=0}
       = \langle\psi(\vec\theta)|[H, A_i]|\psi(\vec\theta)\rangle,
   ```
2. stops if $\max_i |g_i| < \varepsilon$,
3. appends $e^{\theta_k A_{\text{opt}}}$ for the largest-gradient operator, and
4. re-optimizes **all** parameters (warm-started from the previous optimum).

Carcará's `ADAPTVQE` is an exact state-vector implementation, in the same spirit
as {doc}`VQE <vqe>`.

## The molecular-orbital basis matters

ADAPT must start from a **stationary** reference. Carcará's raw integrals are in
an orthogonalized atomic-orbital basis whose Hartree-Fock determinant is *not*
stationary, so single excitations would have large gradients and the method gets
stuck. Building the Hamiltonian in the restricted Hartree-Fock **molecular-orbital
basis** fixes this — by Brillouin's theorem single-excitation gradients vanish
there, so the physical double excitations are selected first:

```python
import numpy as np

from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid

R = 0.74
nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
          (1.0, np.array([0.0, 0.0, +R / 2]))]
grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.20)

integrals = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid)
# Transform to the RHF molecular-orbital basis (2 electrons -> 4 qubits).
H = integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
```

## Four operator pools

The pool supplies the candidate anti-Hermitian generators. All four are built
from the spin-conserving single and double excitations and selected by name:

| Pool | Generators | Circuits |
|------|------------|----------|
| `"fermionic"` | fermionic excitations, Jordan-Wigner mapped | deepest (JW `Z`-strings) |
| `"qubit"` | individual JW Pauli strings (qubit-ADAPT) | shallowest per operator |
| `"qeb"` | qubit excitations (JW `Z`-strings dropped) | distance-independent |
| `"ceo"` | coupled-exchange operators (QEBs sharing a support) | best accuracy per CNOT |

```python
from carcara.algorithms import ADAPTVQE

adapt = ADAPTVQE(H, pool="ceo", num_particles=(1, 1), n_spatial_orbitals=2)
result = adapt.run(max_iterations=15, gradient_tol=1e-6)

print(f"ADAPT-VQE energy = {result.optimal_energy:.8f} Ha")
print(f"operators added  = {result.operators}")
```

## Circuit profiling

Each grown ansatz is compiled to a native `{CNOT, U}` gate set with Qiskit, and
its **CNOT count** and **depth** are logged per iteration in the result's
`metrics` and `iterations`:

```python
print(f"CNOTs = {result.metrics.cnot_count}, depth = {result.metrics.depth}")
for step in result.iterations:
    print(step.operator_label, step.cnot_count, step.depth)
```

On H₂ every pool reaches the exact (FCI) ground state, but the hardware-minded
pools do so far more cheaply — the **qubit pool needs 6 CNOTs** versus **48** for
the fermionic pool. The complete comparison script is in
[`examples/run_adapt_vqe.py`](https://github.com/seixas-research/carcara/blob/main/examples/run_adapt_vqe.py).

```{note}
H₂ reaches FCI with a single operator, so it does not show the pools' relative
depth advantage for larger systems (where QEB/CEO drop long Jordan-Wigner
`Z`-strings). Linear H₄ is a better demonstration — see the
{doc}`expressibility tutorial <expressivity>`.
```
