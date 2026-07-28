# Ground-State Search of H2 using VQE

This tutorial demonstrates the end-to-end simulation workflow of Carcará to compute the ground-state energy of the hydrogen molecule ($H_2$) using the Variational Quantum Eigensolver (VQE).

---

## Defining the Geometry and Potential

We start by placing two hydrogen nuclei along the z-axis, symmetrically about the origin at their equilibrium bond distance of $0.74$ Å. The external potential felt by the electrons is the sum of the nuclear Coulomb potentials:

$$V(\mathbf{r}) = -\sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}$$

In Carcará, this potential is set up using the `Potentials` class:

```python
import numpy as np
from carcara.integrals import Potentials

Z = 1.0
R = 0.74  # Angstrom
proton_a = np.array([0.0, 0.0, -R / 2])
proton_b = np.array([0.0, 0.0, +R / 2])

potentials = Potentials([(Z, proton_a), (Z, proton_b)])
```

---

## Setting up the Grid, Basis, and Integrals

Next, we establish a cubic real-space grid to evaluate the integrals numerically. The grid resolution is specified by the spacing $h = 0.10$ Å. The basis set consists of analytic hydrogenic 1s orbitals (`FullAtomicOrbital`) centered on each nucleus.

The `IntegralEngine` stack these orbitals and computes the one-body core Hamiltonian $h_{\text{core}} = T + V$ (where $T$ is the kinetic energy and $V$ is the nuclear attraction potential) and the two-body electron-repulsion integrals (ERI) $\langle ab|cd \rangle$ using an $O(N \log N)$ FFT Poisson solver:

```python
from carcara.basis import FullAtomicOrbital
from carcara.integrals import Grid, IntegralEngine

grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.10)
basis = [FullAtomicOrbital(1, 0, 0, Z=Z, center=proton_a),
         FullAtomicOrbital(1, 0, 0, Z=Z, center=proton_b)]

engine = IntegralEngine(basis, grid)

T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V
eri = engine.two_body(method="fft")
```

---

## Constructing the Hamiltonian and Mapping to Qubits

With the core and ERI integrals computed, we can construct the second-quantized fermionic Hamiltonian using the `MolecularIntegrals` class, which maps the integrals directly to a `Fermion` operator. 

We then map the fermionic Hamiltonian to qubit space using the Jordan-Wigner transformation, yielding a `PauliSum` operator:

```python
from carcara.core import MolecularIntegrals

# Create the MolecularIntegrals object
mol_integrals = MolecularIntegrals(
    nuclei=[(Z, proton_a), (Z, proton_b)],
    basis=basis,
    grid=grid
)

# Build the second-quantized Hamiltonian in the spin-orbital basis
H_fermion = mol_integrals.molecular_hamiltonian(mo_basis=False)

# Map to qubits using Jordan-Wigner
H_qubit = H_fermion.map_to_qubits(method="jordan_wigner")
print(H_qubit)
```

---

## Running VQE with UCCSD

We define the trial state using the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz, which acts on the Hartree-Fock reference state. We then minimize the energy expectation value:

$$E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})| H |\psi(\boldsymbol{\theta})\rangle$$

classically using the COBYLA optimizer:

```python
from carcara.circuits import UCCSD
from carcara.algorithms import QuantumCalculator

# 2 spatial orbitals -> 4 spin-orbitals; 2 electrons (1 alpha, 1 beta)
ansatz = UCCSD(n_spatial_orbitals=2, num_particles=(1, 1), mapping="jordan_wigner")

# Run VQE in direct mode (explicit Hamiltonian, no geometry)
calc = QuantumCalculator(method="vqe", hamiltonian=H_fermion, ansatz=ansatz,
                         optimizer="COBYLA")
result = calc.run()

print(f"VQE Ground-State Energy: {result.optimal_energy:.6f} Ha")
```

---

## Driving via the ASE Calculator Interface (Recommended)

Carcará provides a standard Atomic Simulation Environment (ASE) calculator. This interface simplifies the workflow by handling the geometry, grid creation, integral calculations, reference solver, mapping, and VQE loops automatically under the hood:

```python
from ase import Atoms
from carcara.algorithms import QuantumCalculator

# Define H2 molecule in a cubic unit cell
atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

# Attach the calculator, selecting the VQE method
atoms.calc = QuantumCalculator(method="vqe", basis="FAO", mapping="jordan_wigner",
                               optimizer="COBYLA", h=0.20)

# Execute calculation (energy returned in eV)
energy_ev = atoms.get_total_energy()
result = atoms.calc.result

print(f"Optimal Energy: {result.optimal_energy:.6f} Ha ({energy_ev:.6f} eV)")
```

A verbose run prints the compiled qubit Hamiltonian as Pauli strings, followed by a breakdown of wall times, thread counts, and memory consumption.
