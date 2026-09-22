# Visualizing the wavefunction

A finished run can write its electron density, its natural orbitals and its
correlation density to a Gaussian `.cube` file, which VESTA, VMD, XCrySDen and
Avogadro open directly:

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[7, 7, 7])
atoms.center()
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       h=0.30)
atoms.get_potential_energy()

atoms.calc.write_cube("density.cube")
atoms.calc.write_cube("difference.cube", quantity="difference_density")
atoms.calc.write_cube("no1.cube", quantity="natural_orbital", index=1)
```

`examples/32_wavefunction_cube.py` is the worked version of this, with a
figure.

## What is in the file, and what is not

This is the point to be precise about. The state an ADAPT-VQE or VQE run
converges to is a **many-electron** wavefunction $|\Psi(\theta)\rangle$: a
function of $3N$ coordinates. No volumetric file holds such a thing, and
Mandacaru does not pretend it does.

What a file *can* hold, and what these are, are the state's **one-particle
reductions**. They follow from $|\Psi\rangle$ exactly -- no further
approximation -- through the one-particle reduced density matrix
$\gamma_{pq} = \langle\Psi| a^\dagger_p a_q |\Psi\rangle$, the molecular
orbitals $\phi_p = \sum_\mu A_{\mu p}\chi_\mu$ with $A = S^{-1/2}V$, and the
basis functions sampled on the grid:

$$ n(\mathbf r) = \sum_{pq} \gamma_{pq}\, \phi_p^*(\mathbf r)\,\phi_q(\mathbf r). $$

The same $\gamma$ the nuclear forces contract is the one drawn here, obtained
the same way, so the picture and the gradient describe the same state.

## The quantities

| `quantity` | What it is | Units | Integrates to |
|---|---|---|---|
| `density` (default) | $n_\alpha + n_\beta$ | e/Bohr³ | the electron count |
| `alpha_density` / `beta_density` | one spin channel | e/Bohr³ | $N_\alpha$ / $N_\beta$ |
| `spin_density` | $n_\alpha - n_\beta$ | e/Bohr³ | $N_\alpha - N_\beta$ |
| `difference_density` | $n - n_{\rm HF}$ | e/Bohr³ | zero |
| `natural_orbital` | eigenvector `index` of $\gamma$ | Bohr^(-3/2) | — |
| `molecular_orbital` | reference orbital `index` | Bohr^(-3/2) | — |

Natural orbitals are ordered by **descending occupation**, so `index=0` is the
most occupied one; the occupation is returned on the field and written into the
file's comment line. Reference (Hartree--Fock) orbitals are ordered by energy.

`spin_density` is the picture to draw for an open shell -- a doublet such as
H₃, or the O₂ triplet of `examples/06_ADAPTVQE_O2_triplet.py`. For a closed
shell it is zero everywhere, which is a useful thing to confirm.

`difference_density` is usually the most informative of the lot. Its reference
is the Hartree--Fock determinant the ansatz starts from (the frozen core plus
the lowest active orbital of each spin), evaluated in the *same* orbitals, so
what it shows is exactly what the variational optimization did. For H₂ at a
stretched bond, correlation takes density out of the bond and puts it back on
the two atoms:

```{image} ../../../examples/data/h2_cube_slice.png
:alt: Electron density and correlation density of stretched H2
:width: 100%
```

## Occupations as data

The same decomposition is available without writing a file:

```python
orbitals = atoms.calc.natural_orbitals()
print(orbitals.occupations)     # descending, in [0, 2], summing to N
print(orbitals.coefficients)    # column i: orbital i in the atomic-orbital basis
print(orbitals.mo_coefficients) # the same, in the molecular-orbital basis
```

Occupations that are not 2 or 0 *are* the correlation. H₂ in the minimal HAO
basis gives `(1.97, 0.03)` at 0.74 Å and `(1.76, 0.24)` at 1.60 Å: the
stretched bond is half broken, and a single determinant — which would report
`(2, 0)` at both — cannot say so.

## Grid, origin and units

Everything in a cube file is in **Bohr**, and a density is in electrons per
Bohr³. The data is written on the calculation's **own** grid -- the real-space
box the integrals used, centered on the molecule and with the node spacing `h`
asked for -- and the header carries that grid's origin and its three step
vectors, so the atoms and the data line up whatever the geometry. That matters
because Mandacaru's grid is not the cell divided by the node count (it is
centered on the molecule, the node count is `round(L / h) + 1`, and a
non-orthogonal cell gives step *vectors*, not a diagonal); a writer that
assumed otherwise would put every nucleus in the wrong place. This is why the
format is written by `mandacaru.utils.cube` rather than taken from ASE.

Pass `grid=` to sample on a different grid -- a finer one for a smoother
picture, a coarser one for a smaller file. The basis functions are
**re-evaluated** there; the data is never interpolated:

```python
from mandacaru.integrals import Grid

fine = Grid(center=atoms.get_positions().mean(axis=0),
            box_size=0.0, h=0.08, units="angstrom", cell=atoms.get_cell())
atoms.calc.write_cube("density_fine.cube", grid=fine)
```

On the calculation's own grid the electron count comes out exact to machine
precision, because that same quadrature is what defines the overlap the
orbitals are orthonormal under. On any other grid the integral instead shows
the difference between the two quadratures, which is a real measure of how well
`h` resolved the basis.

A cube file is one text number per node, so size grows as $1/h^3$: a 21³ grid
is about 120 kB, a 101³ grid about 14 MB.

## Excited states

`state=` selects the level. Subspace-search runs store their levels, so an
integer index is enough; a deflation run returns them from `energy_levels()`
and the state vector is passed directly:

```python
atoms.calc = Mandacaru(method="subspace-vqe",
                       basis="HAO",
                       h=0.30,
                       num_states=2)
atoms.get_potential_energy()
atoms.calc.write_cube("excited.cube", state=1)

# or, from a deflation run:
levels = atoms.calc.energy_levels(3)
atoms.calc.write_cube("second.cube", state=levels.states[2])
```

## Pseudopotentials: the smooth valence density

With a PAW, UPAW, ONCVPSP or NCPP basis the orbitals are the **smooth pseudo**
valence orbitals, so what is written is the *pseudo valence density*. The core
is absent by construction, and inside the augmentation spheres the smooth
density is not the physical one.

For PAW the overlap itself is augmented, $S = \tilde S + C q C^\dagger$, so the
smooth density does not integrate to the valence electron count:

$$ \int\tilde n\,d^3r \;+\; \underbrace{\textstyle\sum_{pq}\gamma_{pq}
   (A^\dagger C q C^\dagger A)_{pq}}_{\text{augmentation charge}}
   \;=\; N_{\rm valence}. $$

Both numbers come back on the field, and the file's comment line says which one
it holds:

```python
field = atoms.calc.write_cube("paw.cube")     # basis="PAW"
print(field.integral)              # what is on the grid
print(field.augmentation_charge)   # what is not
print(field.n_electrons)           # their sum: the valence electron count
```

For H₂ in a single-zeta PAW basis the augmentation holds a few per cent of the
two valence electrons; for oxygen, whose augmentation charge is an order of
magnitude larger, it is far more. The third number is exact whatever the
basis — it is the trace of the density matrix — so it is the one to check
against.

The all-electron density inside the augmentation spheres is **not**
reconstructed. Doing it properly means summing the one-center expansion on a
radial grid per atom and adding it back; the radial machinery exists
(`mandacaru.pseudopotentials.paw.reconstruct_ae`) but the three-dimensional
assembly does not, and an approximation there would be invisible in the
picture and wrong in the numbers. Use an all-electron basis when the core
matters.

The nuclei are written with their **real** atomic numbers, not the valence
charges the pseudopotential carries, so a viewer draws oxygen as oxygen; the
valence charge goes in the charge column, where it belongs.

## Frozen core, tapered registers, sectors

None of these change the picture, which is the point:

* a **frozen core** is refilled before the density is built, so an
  all-electron density is the all-electron density and integrates to the full
  electron count;
* a **parity-reduced** (two-qubit-tapered) register has no ladder operators of
  its own, so its RDM comes from the tapered qubit operators -- the same route
  the forces take -- and gives the same density to machine precision;
* a **particle-number sector** run keeps its state on the sector, and the
  ladder operators are applied there.

## Complex orbitals

A cube file holds real numbers. Densities are real by construction (the density
matrix is Hermitian) and Mandacaru verifies that rather than assuming it.

Orbitals are another matter. The basis carries complex spherical harmonics
$Y_{lm}$, and although the molecular path rotates the orbitals to
conjugation-real form, a natural orbital of a degenerate pair can still come out
genuinely complex. `component="auto"` (the default) removes the global phase,
checks that the residual imaginary part is numerical noise, and writes the real
part; if it is not noise it raises and asks you to choose
`component="modulus"`, `"real"` or `"imag"` deliberately. Note also that the
orbitals *within* a degenerate set are fixed only up to a rotation, so which
member of a π pair you get is arbitrary -- as it is in any code.

## Opening the files

### VESTA

`File ▸ Open` the `.cube` (or `.xsf`) file. VESTA reads the isosurface level
from `Properties ▸ Isosurfaces`; useful starting values, with the data in
e/Bohr³:

| Field | Isosurface level | Notes |
|---|---|---|
| `density` | `0.02` – `0.05` | `0.02` is the conventional "molecular shape" contour |
| `natural_orbital` / `molecular_orbital` | `±0.05` – `±0.10` | add both signs; VESTA colors them separately |
| `difference_density` | `±0.001` – `±0.005` | a correlation density is one to two orders of magnitude smaller than the density |
| `spin_density` | `±0.005` – `±0.02` | |

For a signed field, add two isosurfaces at `+v` and `-v` and give them
different colors -- red for the positive lobe, blue for the negative one is the
usual convention.

### VMD

```
vmd -cube density.cube
```

Then `Graphics ▸ Representations`, set **Drawing Method** to `Isosurface` and
enter the isovalue; the same values as above. For a signed field, create a
second representation with the negative isovalue and a different
**ColorID**. `Graphics ▸ Colors ▸ Display ▸ Background ▸ white` makes the
result printable. To load the geometry and the field separately, `mol new
density.cube type cube` works the same way.

### XCrySDen

`.xsf` is XCrySDen's native format; `write_cube("field.xsf")` writes one
(VESTA reads it too). ASE formats XSF data with six *decimal* places rather
than six significant digits, so values below about `1e-6` are written as zero;
`.cube` is the more precise of the two and is the format to prefer.

## The lower-level entry points

`Mandacaru.write_cube` is a thin wrapper. The two pieces underneath are usable
on their own:

* {func}`mandacaru.algorithms.volumetric.volumetric_field` takes an integral
  object and a spin-orbital RDM and returns a
  {class}`~mandacaru.algorithms.volumetric.VolumetricField` -- no solver
  involved, which is how it is tested;
* {func}`mandacaru.utils.cube.write_cube` takes a field, a grid origin and
  step, and the nuclei, and knows nothing about wavefunctions.

`Mandacaru.volumetric_field(...)` returns the field without writing anything,
which is what to use for a plot.

## What is deliberately not here

* **All-electron reconstruction inside PAW spheres** -- see above.
* **A command-line flag.** `mandacaru H2O --cube density.cube` would have to
  duplicate the quantity, index, component and grid options; the three-line
  Python form above is clearer and does not freeze an API into the CLI before
  the feature has been used in anger.
* **Spin-orbital-resolved orbitals.** The Hamiltonian uses one spatial basis
  for both spins (RHF, or the UHF natural orbitals for an open shell), so
  $\alpha$ and $\beta$ orbitals differ only in their occupations -- which
  `alpha_density` / `beta_density` already show.
