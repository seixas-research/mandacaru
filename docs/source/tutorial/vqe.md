# Variational Quantum Eigensolver

The Variational Quantum Eigensolver (VQE) finds the ground-state energy of a
Hamiltonian by minimizing the Rayleigh quotient

```{math}
E(\vec\theta) = \langle \psi(\vec\theta) | H | \psi(\vec\theta) \rangle
```

over the parameters of a variational ansatz. This tutorial runs the full Carcará
pipeline on H₂:

```
geometry -> real-space integrals -> second-quantized Hamiltonian
         -> Jordan-Wigner qubit Hamiltonian
         -> UCCSD ansatz -> VQE optimization -> ground-state energy
```

Carcará's VQE is an **exact state-vector simulator**: the qubit Hamiltonian is
materialized as a dense matrix and the ansatz produces the exact `2^N` state
vector, so the energy is the noiseless expectation value. This validates the
whole pipeline against exact diagonalization on small molecules; a shot-based
estimator on real hardware is a drop-in replacement for the energy evaluation.

## The Hamiltonian

Build the second-quantized H₂ Hamiltonian from the real-space integral engine
over a minimal FAO 1s basis (one orbital per atom):

```python
import numpy as np

from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid

R = 0.74  # H2 bond length (Angstrom)
nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
          (1.0, np.array([0.0, 0.0, +R / 2]))]
basis = minimal_fao_basis(nuclei)
grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.15)

H = MolecularIntegrals(nuclei, basis, grid).molecular_hamiltonian()
# -> Fermion(73 terms, n_modes=4): 2 spatial orbitals -> 4 spin-orbitals
```

## The UCCSD ansatz

The Unitary Coupled-Cluster with Singles and Doubles ansatz prepares

```{math}
|\psi(\vec\theta)\rangle
    = e^{\sum_k \theta_k (\hat T_k - \hat T_k^\dagger)}\,|\text{HF}\rangle,
```

the exponential of the anti-Hermitian single and double excitation generators
acting on the Hartree-Fock reference. Each generator is mapped to qubits with the
same mapping as the Hamiltonian (Jordan-Wigner by default). For H₂ (4
spin-orbitals, 2 electrons) this is 2 singles + 1 double = **3 parameters**.

```python
from carcara.circuits import UCCSD

ansatz = UCCSD(n_spatial_orbitals=2, num_particles=(1, 1),
               mapping="jordan_wigner")
```

```{note}
The default evaluates the exact UCC unitary `exp(Σ θ_k G_k)`. Passing
`trotter=True` uses the first-order product `Π exp(θ_k G_k)` — the form realized
as a quantum circuit, which only *approximates* the exact UCC unitary (for H₂ it
does not quite reach the exact ground state).
```

## Running VQE

Hand the Hamiltonian, ansatz, and a classical optimizer to `VQE` and call
`run()`. Optimization starts from the Hartree-Fock reference (all parameters
zero):

```python
from carcara.algorithms import VQE
from carcara.optimizers import Optimizer

vqe = VQE(H, ansatz, optimizer=Optimizer(method="COBYLA", maxiter=2000))
result = vqe.run()

print(f"VQE ground-state energy = {result.optimal_energy:.6f} Ha")
print(f"optimal parameters      = {result.optimal_parameters}")
```

## Checking the result

Because the backend is exact, VQE must reproduce the exact ground state of the
qubit Hamiltonian:

```python
h_matrix = H.map_to_qubits("jordan_wigner").to_matrix()
exact = np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min()

assert abs(result.optimal_energy - exact) < 1.6e-3   # chemical accuracy
```

For this basis the ground-state energy is `-1.1154 Ha` (total, including nuclear
repulsion). The Hartree-Fock reference sits at `-0.3672 Ha`, so UCCSD recovers
about `0.75 Ha` of correlation energy and matches exact diagonalization to
`~10⁻⁹ Ha`. The complete runnable script is in
[`examples/H2_vqe.py`](https://github.com/seixas-research/carcara/blob/main/examples/H2_vqe.py).
