# Removing qubits the Hamiltonian does not use

A molecular qubit Hamiltonian commutes with a handful of Z₂ operators, and every
one of them is a qubit carrying no information. `taper=True` finds them and
removes them.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("LiH", positions=[(0, 0, 0), (0, 0, 1.60)], cell=(9, 9, 10))
atoms.center()
atoms.calc = Mandacaru(method="adapt-vqe", basis="STO-3G", taper=True)
energy = atoms.get_potential_energy()

print(atoms.calc.solver._taper_info.summary())
# 2 qubit(s) removed by Z2 symmetry -> 4 on the register
```

On the command line: `mandacaru LiH --cell 10 --basis STO-3G --taper`.

## What it buys

| system | Jordan-Wigner | `mapping="parity_reduced"` | `taper=True` |
|---|---|---|---|
| H₂ / STO-3G | 4 | 2 | **1** |
| H₂ / 6-31G | 8 | 6 | **5** |
| LiH / STO-3G | 6 | 4 | 4 |

The energies are identical to the untapered run in every case — to all digits,
not to a tolerance. H₂ in a minimal basis reducing to a *single* qubit is the
known result for the full Z₂ group, and it is the check that the symmetry finder
is neither missing symmetries nor inventing them.

LiH matching `parity_reduced` rather than beating it is the expected behavior,
not a shortfall. The finder reads the symmetries **off the Hamiltonian**, so it
finds the ones that are there. H₂ has one more than LiH — the spatial parity of a
homonuclear diatomic — and that is the qubit it gains.

## Why it is not a `mapping` value

`mapping` names the fermion-to-qubit **encoding**. Tapering is a symmetry
reduction applied *after* one, so it is its own flag: it composes with an
encoding rather than being one, and a `mapping="tapered"` would have to silently
pick an encoding for you.

It requires `mapping="jordan_wigner"` and refuses the others by name. Two
reasons. It reads the symmetry sector off the reference determinant, whose bits
are the orbital occupations only under Jordan-Wigner. And `parity_reduced` is
*itself* a taper of two known symmetries — the alpha-sector parity and the total
parity — so combining them would try to remove the same qubits twice.

## How it works

Each Pauli string is a pair of bit vectors, its X-support and its Z-support, and
two strings commute exactly when their symplectic form vanishes. So the Z-type
Pauli operators commuting with **every** term of `H` are the kernel over GF(2) of
the X-support matrix. That kernel is the symmetry group; nothing about the point
group is supplied, which is what makes it robust — a symmetry the orbitals happen
to break is not found, and one you did not think of is.

The generators are then reduced to a set where each has an **anchor** qubit it
acts on and no other generator does, and each is rotated to that anchor's `X` by
the Clifford $(\tau_j + X_{q_j})/\sqrt2$. After the rotation the Hamiltonian
commutes with every $X_{q_j}$, so those qubits are constants of motion: replace
each by its ±1 eigenvalue and delete it.

The eigenvalues come from the reference determinant. That matters: the tapered
spectrum is a *subset* of the original, and the lowest eigenvalue of the wrong
subset is not the ground-state energy.

## Pool generators that change a symmetry are dropped

A generator that does not commute with a symmetry cannot be tapered — deleting a
qubit would replace it by its projection, and $\exp(PAP) \neq P\exp(A)P$. This
is the same trap the particle-number sector documents, with the same consequence:
an ansatz built from projected generators explores the wrong manifold and
converges above the ground state, plausibly.

So they are **dropped**, and the count is reported. They connect different
symmetry sectors, so within the sector the reference fixes they cannot contribute
at all: an exact-ground-state amplitude on them is zero and ADAPT would screen
them out at zero gradient anyway. The evidence that this is right rather than
merely convenient is that H₂/6-31G drops **eight** generators and its energy does
not move.

```{note}
Two consequences worth stating. A ground state of *different* symmetry than the
reference determinant is unreachable — that is inherent to tapering, not to this
choice. And if every generator leaked, the pool would be empty; that raises,
rather than running an ansatz with nothing in it.
```

## Observables and forces

Forces, densities, charges, cube files and natural orbitals use RDM observables
mapped from the original spin orbitals through the **same** Clifford and sector
signs as the Hamiltonian. Terms that change the selected symmetry sector have
zero expectation in that sector and are discarded before tapering the
observable. This projection is valid for an expectation value; pool generators
still have to commute with every symmetry before their exponentials are used.

The particle-number sector is also disabled with tapering: a tapered register has
no particle-number basis to enumerate. That costs nothing, because the taper has
already shrunk the register using the same symmetries the sector was exploiting.

## In a dry run

A dry run reports the **untapered** width and says so. How many Z₂ symmetries a
Hamiltonian has is a property of its Pauli terms, which a dry run does not build;
two is the floor for a molecule (the two `parity_reduced` already knows) and a
symmetric molecule usually has more. So the number printed is an upper bound, and
calling it one is better than printing a width the run will not use.

## Composing with an active space

`taper=True` and `active_orbitals=` are independent and multiply: the active
space decides which orbitals reach the register, tapering removes the qubits the
resulting Hamiltonian does not use. Force and density observables are mapped
from that reduced orbital space to the tapered register. See
{doc}`active_space`.
