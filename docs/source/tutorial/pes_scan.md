# Potential Energy Curve Scans

A potential-energy curve represents the ground-state energy of a molecule as a function of its nuclear coordinates. Scanning the bond length of diatomic molecules like $H_2$ and $LiH$ across varying distances allows us to observe chemical bonding wells, equilibrium distances, and dissociation limits.

This tutorial guides you through scanning potential-energy curves with the `QuantumCalculator`, referencing every curve to the **sum of isolated-atom energies**, and managing numerical grid effects.

---

## The absolute reference: the separated atoms

Every dissociation curve in Carcará is referenced to the sum of the isolated-atom energies,

$$E_{\text{binding}}(R) = E_{\text{molecule}}(R) - \sum_i E_{\text{atom},\,i},$$

so $E = 0$ is the separated-atom limit and the well depth is a binding energy. The atomic energies are computed with **unrestricted Hartree-Fock** (open-shell atoms like H and Li need it; for one-electron H it is exact within the basis and grid), *on the same grid, with the same Coulomb softening* as the molecular scan — see `examples/pes_utils.py`:

```python
from carcara.basis import BasisSet
from pes_utils import GridSpec, atomic_reference, commensurate_distances, molecule_positions

grid_spec = GridSpec(box_size=8.0, spacing=0.16)
grid = grid_spec.build()
distances = commensurate_distances(0.42, 3.0, grid_spec)

# Two isolated H atoms, placed at their first-geometry grid alignment.
e_atoms = atomic_reference(["H", "H"], BasisSet.build("FAO"), grid,
                           molecule_positions(float(distances[0])))
```

---

## Scanning H2 over varying distances

For a simple system like $H_2$, the electron clouds are relatively diffuse and contain no heavy core. We scan the distance with ADAPT-VQE, passing the **same fixed grid** to every geometry so molecule and atomic reference are integrated identically:

```python
from ase import Atoms
from carcara.algorithms import QuantumCalculator
from carcara.units import HARTREE_TO_EV

for r in distances:
    atoms = Atoms("H2", positions=molecule_positions(float(r)))
    atoms.calc = QuantumCalculator(method="adapt-vqe", pool="qeb", basis="FAO",
                                   grid=grid, optimizer="L-BFGS-B",
                                   verbose=False)
    atoms.get_total_energy()
    binding_ev = (atoms.calc.result.optimal_energy - e_atoms) * HARTREE_TO_EV
    print(f"R = {r:.2f} A -> E - E_atoms = {binding_ev:+.4f} eV")
```

The curve is bound near equilibrium and returns to $E = 0$ at large $R$. The complete script — including a VASQE sweep of the same curve and the CSV/plot output — is `examples/22_H2_dissociation.py`.

---

## Grid alignment and the "egg-box" effect in LiH

For systems with tight core orbitals, such as the lithium 1s orbital in $LiH$, integrating on a uniform real-space grid introduces a numerical artifact known as the **egg-box effect**.

As nuclei shift relative to the grid nodes, the sampled potential of the $-Z_A/|\mathbf{r}-\mathbf{R}_A|$ cusp varies slightly, introducing artificial ripples in the potential-energy curve.

### Mitigation strategies
To eliminate these artificial oscillations and obtain smooth binding curves for $LiH$ in Carcará:
1. **Grid spacing step-matching:** Step the bond length by **exact multiples of the grid spacing** (e.g., $\Delta R = 2h$, so each nucleus at $\pm R/2$ moves by whole grid nodes). This ensures that the nuclei maintain the same sub-node alignment for every point in the scan.
2. **Alignment-matched references:** Place the isolated-atom references at the exact same sub-node grid coordinates (and with the same Coulomb softening) as the molecular scan, so the numerical integration errors cancel in the binding energy.

The following script scans the $LiH$ potential energy curve using ADAPT-VQE with a Coupled-Exchange Operator (`"ceo"`) pool:

```python
import numpy as np
from ase import Atoms
from carcara.algorithms import QuantumCalculator

# Set grid resolution h (Angstrom)
h_val = 0.15

# Step size is exactly 2 * h to maintain node alignment
distances = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
energies = []

for r in distances:
    # Place Li and H along the z-axis
    atoms = Atoms("LiH", positions=[[4.0, 4.0, 4.0 - r/2], [4.0, 4.0, 4.0 + r/2]],
                  cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

    # Attach the calculator with the ADAPT-VQE method
    atoms.calc = QuantumCalculator(
        method="adapt-vqe",
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

Referencing these energies to the isolated Li + H atoms (computed as in the H$_2$ section, with a `GTO` basis and the matching grid) yields a bound minimum around the experimental equilibrium distance of $1.6$ Å. The full pool-and-mapping comparison over this curve — with the atomic-sum reference and its caveats about the curated distance set — is `examples/16_ADAPTVQE_LiH_dissociation.py`.
