# Ground-State Search of LiH using VQE

This tutorial scales the simulation workflow of Carcará to a larger, heteronuclear system: lithium hydride ($LiH$). This introduces different nuclear charges ($Z_{\text{Li}} = 3.0$ and $Z_{\text{H}} = 1.0$) and a multi-orbital basis set.

---

## Orbital Exponents and Slater's Rules

For heavy or core-electrons (like the Li 1s orbital), utilizing the bare nuclear charge ($Z = 3.0$) makes the orbital extremely contracted and difficult to resolve numerically on a standard grid. 

To overcome this, we use effective nuclear charges derived from **Slater's rules** via `FullAtomicOrbital.from_slater`. This assigns effective charges (e.g., $Z_{\text{eff}} = 2.70$ for Li 1s, $1.30$ for Li 2s/2p, and $1.00$ for H 1s), while the external potential retains the true nuclear charges:

```python
import numpy as np
from carcara.basis import FullAtomicOrbital
from carcara.integrals import Grid, IntegralEngine, Potentials

R = 1.595  # LiH equilibrium bond length (Angstrom)
li_pos = np.array([0.0, 0.0, -R / 2])
h_pos = np.array([0.0, 0.0, +R / 2])

# External potential (uses true nuclear charges)
potentials = Potentials([(3.0, li_pos), (1.0, h_pos)])

# Integrals grid
grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.10)

# Minimal Slater-basis: 3 orbitals on Li (1s, 2s, 2pz) and 1 orbital on H (1s)
basis = [FullAtomicOrbital.from_slater(1, 0, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(2, 0, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(2, 1, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(1, 0, 0, atomic_number=1, center=h_pos)]

engine = IntegralEngine(basis, grid)
```

---

## Integrals and Molecular-Orbital Transformation

Evaluating the integrals yields $4 \times 4$ one-body matrices and a $4 \times 4 \times 4 \times 4$ two-body ERI tensor. 

For systems with more than two electrons, running VQE directly in the atomic-orbital (AO) basis often slows down convergence. Instead, we transform the Hamiltonian to the restricted Hartree-Fock **molecular-orbital (MO) basis**. In this basis, the Hartree-Fock state becomes a stationary reference, and single-electron excitation gradients vanish:

```python
from carcara.core import MolecularIntegrals

# Compute integrals
T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V
eri = engine.two_body(method="fft")

# Re-package into MolecularIntegrals
mol_integrals = MolecularIntegrals(
    nuclei=[(3.0, li_pos), (1.0, h_pos)],
    basis=basis,
    grid=grid
)

# Build second-quantized Hamiltonian in the RHF molecular orbital basis (2 valence electrons)
H_mo = mol_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
```

---

## Executing VQE on LiH

We map the MO Hamiltonian to qubits and run VQE using the UCCSD ansatz. For LiH, with 4 spatial molecular orbitals (8 spin-orbitals) and 2 valence electrons, UCCSD has 22 parameters:

```python
from carcara.circuits import UCCSD
from carcara.algorithms import QuantumCalculator

# 4 spatial orbitals, 2 electrons (1 alpha, 1 beta)
ansatz = UCCSD(n_spatial_orbitals=4, num_particles=(1, 1), mapping="jordan_wigner")

# Execute VQE in direct mode (explicit Hamiltonian, no geometry)
calc = QuantumCalculator(method="vqe", hamiltonian=H_mo, ansatz=ansatz,
                         optimizer="COBYLA")
result = calc.run()

print(f"LiH VQE Ground-State Energy: {result.optimal_energy:.6f} Ha")
```

---

## Running with the ASE Interface

Just like $H_2$, $LiH$ can be simulated cleanly using the ASE calculator interface:

```python
from ase import Atoms
from carcara.algorithms import QuantumCalculator

# Define LiH molecule in a unit cell
atoms = Atoms("LiH", positions=[[4.0, 4.0, 3.20], [4.0, 4.0, 4.80]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

# Attach the calculator, selecting the VQE method
atoms.calc = QuantumCalculator(method="vqe", basis="FAO", mapping="jordan_wigner",
                               optimizer="COBYLA", h=0.15)

# Energy in eV
energy_ev = atoms.get_total_energy()
result = atoms.calc.result

print(f"LiH Ground-State Energy: {result.optimal_energy:.6f} Ha ({energy_ev:.6f} eV)")
```
