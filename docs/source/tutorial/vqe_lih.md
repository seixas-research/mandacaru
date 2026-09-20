# Your first LiH calculation: VQE

Lithium hydride (LiH) contains one lithium nucleus, one hydrogen nucleus and
**four electrons** when neutral. In this tutorial you will estimate its total
energy at a fixed Li–H distance using the variational quantum eigensolver
(VQE). No quantum device credentials are required.

Complete the [installation](../installation.md) first. Save the following
blocks, in order, as `lih_vqe.py`, then run `python lih_vqe.py`.

## 1. Define the molecule

```python
from ase import Atoms

bond_length = 1.6  # Å; a starting geometry, not an optimized bond length
atoms = Atoms(
    "LiH",
    positions=[
        [0.0, 0.0, -bond_length / 2],
        [0.0, 0.0, bond_length / 2],
    ],
    pbc=False,
)
```

The nuclei lie on the $z$ axis, equally far from the origin. The distance
between them is `bond_length`. `pbc=False` describes an isolated molecule.

## 2. Choose the numerical model

A **basis set** supplies the spatial orbitals used to describe the electrons.
A **grid** supplies the points at which Mandacaru evaluates the integrals.
These are independent choices.

```python
from mandacaru.integrals import Grid

grid = Grid(center=[0.0, 0.0, 0.0], box_size=4.8, h=0.12)
```

`box_size` is a **half-width**: this box extends from −4.8 to +4.8 Å along
each axis. `h` is the requested spacing in ångströms. An explicit grid means
that this example does not need an ASE unit cell. These modest settings are
for learning; check finer grids and larger boxes before interpreting energies
quantitatively.

We use Mandacaru's `STO-3G` family and freeze the lowest doubly occupied
Hartree–Fock molecular orbital, associated mainly with lithium's 1s core.
The resulting active problem has two electrons. The core still contributes
to the Hamiltonian and total energy.

```{important}
Neutral LiH has four electrons. Setting `n_electrons=2` alone would describe a
different charged system. Use `frozen_core=True` to remove the core pair from
the variational search consistently.
```

Mandacaru's minimal `STO-3G` construction contains Li 1s and 2s functions and an
H 1s function: three spatial orbitals before freezing, two afterwards. Each
active spatial orbital has two spin orbitals, giving four qubits under the
Jordan–Wigner mapping. This small basis omits lithium p functions. It is a
teaching model, and its internally fitted Gaussian exponents differ from
published basis tables; see [basis sets](../guide/basis_sets.md).

## 3. Attach and run the calculator

```python
from mandacaru.algorithms import Mandacaru
from mandacaru.optimizers import Optimizer

atoms.calc = Mandacaru(
    method="vqe",
    basis="STO-3G",
    frozen_core=True,
    mapping="jordan_wigner",
    grid=grid,
    optimizer=Optimizer(method="L-BFGS-B", maxiter=500, tol=1e-12),
    execute_circuits=False)

print(atoms.calc.dry_run(atoms))
energy_ev = atoms.get_potential_energy()
result = atoms.calc.result

print(f"Total energy: {energy_ev:.6f} eV")
print(f"Reference energy: {result.reference_energy:.6f} eV")
print(f"Energy change from the reference: {result.correlation_energy:.6f} eV")
print(f"Optimizer succeeded: {result.success}")
print(f"Variational parameters: {result.num_parameters}")
```

The dry run estimates the qubit requirements without evaluating integrals.
`get_potential_energy()` then builds the molecular Hamiltonian and runs VQE.
For a fixed molecular geometry, the returned energy includes electronic
energy and nuclear repulsion; it excludes nuclear kinetic energy.

With `method="vqe"`, Mandacaru constructs a fixed UCCSD ansatz automatically.
The classical optimizer varies its circuit parameters to minimize

```{math}
E(\boldsymbol{\theta}; R)
= \langle\psi(\boldsymbol{\theta})|\hat H(R)
  |\psi(\boldsymbol{\theta})\rangle.
```

Here $R$ is the fixed Li–H distance and $\boldsymbol{\theta}$ contains the
circuit parameters. `execute_circuits=False` evaluates the state locally;
this example does not submit quantum hardware jobs.

## 4. Interpret the output

`reference_energy` is the starting reference determinant's energy.
`correlation_energy` is the difference between the optimized and reference
energies for this model. Neither quantity is a molecular binding energy.

An optimizer success flag reports its stopping criterion. It does not prove
that the ansatz reaches the exact ground state or that the grid and basis are
converged. If the optimization fails, examine the energy history and increase
`maxiter` before changing the physical model.

Continue with [ADAPT-VQE](adapt_vqe_lih.md) to let the calculation choose its
circuit generators, then [scan the bond length](pes_scan.md).
