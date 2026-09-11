# Interaction energies on a shared grid

The binding energy of a complex is a difference of three total energies,

$$E_\text{int} = E(\text{AB}) - E(\text{A}) - E(\text{B}),$$

and on a real-space grid that difference is only meaningful if all three are
evaluated on **the same grid, with every atom at the same position relative
to the grid nodes**. `Carcara` normally re-centres its box on the geometry it
is given, so a fragment computed on its own is sampled differently than it is
inside the complex. For a soft hydrogen that costs a few meV; for an atom
with a sharp core it costs electronvolts — a sodium ion at a 0.3 Å spacing
moves by hundreds of eV — and the "interaction energy" then measures the
grid, not the chemistry.

{func}`~carcara.algorithms.interaction.interaction_energy` builds the grid
once, from the complex, and evaluates the complex and each fragment on it.
Each fragment is the complex with the other atoms deleted: same coordinates,
same box, same spacing, same Coulomb softening.

```python
from ase import Atoms
from ase.build import molecule
from carcara.algorithms import Carcara, interaction_energy

water = molecule("H2O")
complex_ = water + Atoms("Na", positions=[[0, 0, 2.3]])
complex_.center(vacuum=4.0)

result = interaction_energy(complex_, fragments=[[0, 1, 2], [3]],
                            charges=[0, 1], charge=1,
                            method="adapt-vqe", basis="FAO",
                            frozen_core=True, h=0.25)
result.energy              # Hartree
result.in_units("eV")
result.fragment_energies   # one per fragment, Hartree
result.results             # the per-fragment run results
```

- `fragments` are atom indices that must partition the complex; `charges`
  gives each fragment's charge and must add up to the complex `charge`.
- Spin states follow the geometry's magnetic moments, per fragment.
- `method="rhf"` returns the mean-field interaction energy with no circuit,
  a cheap way to check a setup before the variational runs.
- Every other keyword (`basis`, `frozen_core`, `pool`, `optimizer`, ...) is
  forwarded to every run, so the three energies are strictly comparable.

The same helper is available on a configured calculator, reusing its method,
basis and options:

```python
calc = Carcara(method="adapt-vqe", basis="FAO", frozen_core=True, h=0.25)
calc.interaction_energy(complex_, [[0, 1, 2], [3]], charges=[0, 1], charge=1)
```

What the helper cannot do is make an under-resolved core converge: the
sodium ion above is still a bare nucleus of charge 11 on a 0.25 Å grid, and
its interaction energy will not converge with `h`. Pair the helper with a
basis that keeps such atoms cheap (a per-element mapping, see
{doc}`basis_sets`) and check the mean-field number first.


## Numerics that keep the difference meaningful

Chasing the Na⁺·H₂O binding energy through the grid spacing exposed two
latent defects that affected every molecule with p or d functions (water,
O₂, anything beyond LiH), and they are fixed:

- the Löwdin orthogonalization transformed the two-electron tensor with the
  wrong conjugation pattern, which breaks the tensor's symmetries whenever
  the overlap matrix is complex (the spherical harmonics are complex, so any
  p shell off a symmetry plane does it); water's Hartree–Fock energy moved by
  1.2 Ha;
- the Fock build contracted the density with its indices transposed, the
  complex conjugate of the right density.

Both were invisible for the s-only systems the suite validated against exact
diagonalization. The decisive test now in place evaluates the SCF density's
energy directly with the raw AO matrices and requires the orthonormal-basis
energy to match it, and a random complex unitary must leave the energy
unchanged.

Three further numerical safeguards came out of the same investigation:

- the SCF is **DIIS-extrapolated with a level shift** and run from two
  starting guesses, keeping the lower converged solution;
- `kinetic="spectral"` evaluates the Laplacian by FFT instead of the
  3-point stencil, which under-estimates the kinetic energy of compact
  functions (by up to 15% at 0.2–0.3 Å); it is exact for resolved
  functions and never underestimates an unresolved one. It is opt-in because
  the force code differentiates the stencil;
- every Hamiltonian build compares each basis function's grid kinetic energy
  (and each pseudopotential projector's grid norm) with the exact radial
  value and **warns** about functions the grid cannot resolve, naming them.
  A warning means the energies that involve those functions are not to be
  trusted at that spacing.
