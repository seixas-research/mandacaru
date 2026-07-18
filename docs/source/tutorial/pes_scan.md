# Potential Energy Surface Scans

A potential-energy surface (PES) represents the ground-state energy of a molecule as a function of its nuclear coordinates. Scanning the bond length of diatomic molecules like $H_2$ and $LiH$ across varying distances allows us to observe chemical bonding wells, equilibrium distances, and dissociation limits.

This tutorial guides you through scanning potential-energy surfaces using VQE and ADAPT-VQE in Carcará, while highlighting how to manage numerical grid effects.

---

## Scanning H2 over Varying Distances

For a simple system like $H_2$, the electron clouds are relatively diffuse and contain no heavy core. We scan the distance from $0.4$ Å to $2.0$ Å using VQE with the UCCSD ansatz:

```python
import numpy as np
from ase import Atoms
from carcara.algorithms import VQE

# Range of distances (Angstrom)
distances = np.arange(0.4, 2.1, 0.1)
energies = []

for r in distances:
    # Setup H2 molecule centered in the box
    atoms = Atoms("H2", positions=[[4.0, 4.0, 4.0 - r/2], [4.0, 4.0, 4.0 + r/2]],
                  cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)
    
    # Attach VQE calculator
    atoms.calc = VQE(basis="FAO", mapping="jordan_wigner", optimizer="COBYLA", h=0.20, verbose=False)
    
    # Get total energy in eV
    energy_ev = atoms.get_total_energy()
    energies.append(energy_ev)
    print(f"R = {r:.2f} A -> Energy = {energy_ev:.4f} eV")
```

---

## Grid Alignment and the "Egg-Box" Effect in LiH

For systems with tight core orbitals, such as the Lithium 1s orbital in $LiH$, integrating on a uniform real-space grid introduces a numerical artifact known as the **egg-box effect**. 

As nuclei shift relative to the grid nodes, the sampled potential of the $-Z_A/|\mathbf{r}-\mathbf{R}_A|$ cusp varies slightly, introducing artificial ripples in the potential-energy surface.

### Mitigation Strategies
To eliminate these artificial oscillations and obtain smooth binding curves for $LiH$ in Carcará:
1. **Grid spacing step-matching:** Step the bond length by **exact multiples of the grid spacing** (or half-spacing, e.g., $\Delta R = 2h$ or $h$). This ensures that the nuclei maintain the same sub-node alignment along the coordinate axes for every point in the scan.
2. **Coordinated references:** Place isolated-atom references at the exact same sub-node grid coordinates to match numerical integration errors exactly when calculating binding energy:
   $$E_{\text{binding}} = E_{\text{molecule}} - \sum_i E_{\text{isolated}, i}$$

The following script scans the $LiH$ potential energy surface using ADAPT-VQE with a Coupled-Exchange Operator (`"ceo"`) pool:

```python
import numpy as np
from ase import Atoms
from carcara.algorithms import ADAPTVQE

# Set grid resolution h (Angstrom)
h_val = 0.15

# Step size is exactly 2 * h to maintain node alignment
distances = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
energies = []

for r in distances:
    # Place Li and H along the z-axis
    atoms = Atoms("LiH", positions=[[4.0, 4.0, 4.0 - r/2], [4.0, 4.0, 4.0 + r/2]],
                  cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)
    
    # Attach ADAPTVQE calculator
    atoms.calc = ADAPTVQE(
        pool="ceo",
        basis="FAO",
        optimizer="COBYLA",
        h=h_val,
        max_iterations=10,
        gradient_tolerance=1e-5,
        verbose=False
    )
    
    energy_ev = atoms.get_total_energy()
    energies.append(energy_ev)
    print(f"R = {r:.2f} A -> Energy = {energy_ev:.4f} eV")
```

Plotting these energies yields a smooth potential-energy curve showing a bound minimum around the experimental equilibrium distance of $1.6$ Å.
