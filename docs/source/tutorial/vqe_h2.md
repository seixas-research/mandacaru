# A smaller example: hydrogen with VQE

After completing [the LiH tutorial](vqe_lih.md), you can apply the same workflow
to molecular hydrogen, $\mathrm{H}_2$. Neutral hydrogen has two electrons and
requires no frozen-core approximation.

## Define and solve the molecule

This complete example uses one analytic 1s orbital per hydrogen atom. The two
spatial orbitals give four spin orbitals and four Jordan–Wigner qubits.

```python
from ase import Atoms
from carcara.algorithms import Carcara
from carcara.integrals import Grid
from carcara.optimizers import Optimizer

bond_length = 0.74  # Å
atoms = Atoms(
    "H2",
    positions=[[0.0, 0.0, -bond_length / 2], [0.0, 0.0, bond_length / 2]],
    pbc=False,
)
grid = Grid(center=[0.0, 0.0, 0.0], box_size=4.0, h=0.1)
atoms.calc = Carcara(
    method="vqe",
    basis="FAO",
    grid=grid,
    mapping="jordan_wigner",
    optimizer=Optimizer("L-BFGS-B", maxiter=500),
    execute_circuits=False)
energy_ev = atoms.get_potential_energy()
print(f"Molecular total energy: {energy_ev:.6f} eV")
print(f"Optimiser succeeded: {atoms.calc.result.success}")
```

The calculator evaluates integrals, constructs Hartree–Fock molecular orbitals,
maps the Hamiltonian and optimises the UCCSD circuit. As with LiH, the energy
includes nuclear repulsion. The integration box spans −4 to +4 Å on each axis.

## Compare with LiH

Hydrogen has no lithium 1s core, so its basis functions are easier to sample on
a modest grid. Nevertheless, the result depends on the grid spacing and box
size. Tight optimiser convergence does not establish numerical convergence of
the integrals.

The molecular Hamiltonian depends on the bond length $R$, while VQE varies the
circuit parameters at each fixed geometry:

```{math}
E(\boldsymbol{\theta};R)
= \langle\psi(\boldsymbol{\theta})|\hat H(R)|\psi(\boldsymbol{\theta})\rangle.
```

Use the [LiH scan](pes_scan.md) for the general procedure for varying $R$,
comparing pools and plotting the energies. The hydrogen examples
`examples/08_energy_levels_H2.py` and `examples/09_SubspaceVQE_H2.py` explore
excited-state calculations in this smaller system.
