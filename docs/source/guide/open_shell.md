# Open-shell systems: radicals, ions and high-spin states

Any system with an odd number of electrons — a radical, a cation such as
H₂⁺, a single hydrogen atom — and any even-electron system asked for a
high-spin state, runs through the same calculator as a closed-shell molecule:

```python
from ase.build import molecule
from mandacaru.algorithms import Mandacaru

oh = molecule("OH"); oh.center(vacuum=3.0)          # 9 electrons: a doublet
oh.calc = Mandacaru(method="adapt-vqe",
                    basis="HAO",
                    frozen_core=True)
oh.get_potential_energy()
oh.calc.num_particles                                 # (4, 3) after freezing the O 1s
```

## How the spin state is chosen

The reference occupation `(n_alpha, n_beta)` is read from the geometry's
**initial magnetic moments**, the ASE convention: their rounded total is the
number of unpaired electrons. Without magnetic moments an even electron count
is the closed-shell singlet and an odd count is the doublet. Anything else is
one line of ASE:

```python
o2 = molecule("O2"); o2.set_initial_magnetic_moments([1, 1])      # triplet
h3 = Atoms("H3", positions=..., magmoms=[1, 1, 1])                # quartet
```

A spin state whose parity does not match the electron count is rejected with
a clear message. The `spin` keyword of the drivers is kept for compatibility
only; it no longer decides anything.

## The orbital basis underneath

A closed shell has a natural molecular-orbital basis: the restricted
Hartree–Fock orbitals, in which the reference determinant is a stationary
point of the energy. An odd electron count has no such basis — RHF is not
defined for it — so Mandacaru solves the **unrestricted** (UHF) problem instead
and builds the Hamiltonian in the **natural orbitals of the UHF total
density**, $D_\alpha + D_\beta$, ordered by occupation. Filling the first
$n_\alpha$ of them with spin-up and the first $n_\beta$ with spin-down
electrons is the reference determinant.

Two things follow from that choice:

- The basis is **shared by both spins**, so the spin-orbital Hamiltonian keeps
  exactly the α-block / β-block form the closed-shell path produces. Every
  operator pool, ansatz, fermion-to-qubit mapping, the frozen-core
  approximation and the dry run work unchanged; the plane-wave basis included.
- The reference is **not a stationary point** of the energy (it is not the
  UHF determinant, which lives in two different spatial bases), so single
  excitations carry a non-zero gradient and ADAPT-VQE selects some of them.
  That is expected, and the solvers still reach the exact ground state of the
  $(n_\alpha, n_\beta)$ sector — the test suite checks this on H, H₃ and H₂⁺.

The same machinery is available directly:

```python
uhf = integrals.open_shell_hartree_fock(n_alpha=2, n_beta=1)   # a UHFResult
uhf.natural_occupations       # eigenvalues of the total density, descending
uhf.natural_orbitals          # the shared spatial basis, as columns
uhf.spin_contamination        # <S^2> - S(S+1) of the UHF determinant
H = integrals.molecular_hamiltonian(mo_basis=True, n_electrons=3,
                                    num_particles=(2, 1))
```

`open_shell=True` forces the UHF natural-orbital basis for an even count too,
which can be a better start for a triplet than the closed-shell RHF orbitals
used by default; `open_shell=False` insists on RHF and rejects odd counts.

## Frozen core with an odd count

Only doubly occupied orbitals can be frozen. With one unpaired electron that
is the first $n_\beta$ natural orbitals, so lithium with its 1s frozen becomes
a one-electron, one-orbital, two-qubit problem — and asking to freeze the
singly occupied orbital is refused.

See {mod}`mandacaru.algorithms.hartree_fock` and
{meth}`mandacaru.core.MolecularIntegrals.molecular_hamiltonian`.
