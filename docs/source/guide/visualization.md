(cube-files)=
# Visualizing densities and orbitals

A finished molecular run can write its electron density, a reference density, the
correlation density and its natural orbitals to a Gaussian `.cube` file, which
VESTA, VMD, XCrySDen and Avogadro open directly:

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[7, 7, 7])
atoms.center()
atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.30)
atoms.get_potential_energy()

atoms.calc.write_cube("density.cube")
atoms.calc.write_cube("difference.cube", quantity="difference_density")
atoms.calc.write_cube("no1.cube", quantity="natural_orbital", index=1)
```

`examples/visualization/02_h2_stretched_cube_files.py` is the worked version,
writing the cube files and a difference-density slice for stretched H₂.

## What is in the file, and what is not

The state an ADAPT-VQE or VQE run converges to is a **many-electron**
wavefunction $|\Psi(\theta)\rangle$, a function of $3N$ coordinates. No
volumetric file holds such a thing, and Mandacaru does not pretend it does. What
a file *can* hold are the state's **one-particle reductions**. They follow from
$|\Psi\rangle$ exactly through the one-particle reduced density matrix
$\gamma_{pq} = \langle\Psi| a^\dagger_p a_q |\Psi\rangle$, the molecular orbitals
and the basis functions sampled on the grid:

$$ n(\mathbf r) = \sum_{pq} \gamma_{pq}\, \phi_p^*(\mathbf r)\,\phi_q(\mathbf r). $$

The same $\gamma$ that the variational forces contract is the one drawn here.
For a mean-field run the density is the SCF density.

## The quantities

| `quantity` | What it is | Units | Integrates to |
|---|---|---|---|
| `density` (default) | $n_\alpha + n_\beta$ of the state | e/Bohr³ | the electron count |
| `reference_density` | the same for the mean-field reference | e/Bohr³ | the electron count |
| `difference_density` | `density` minus `reference_density` | e/Bohr³ | zero |
| `natural_orbital` | eigenvector `index` of the spin-summed density matrix | Bohr^(-3/2) | – |

Natural orbitals are ordered by **descending occupation**, so `index=0` is the
most occupied one; the occupation is returned on the field and written into the
file's comment line. `difference_density` is usually the most informative of the
lot: its reference is the Hartree–Fock determinant the ansatz starts from,
evaluated in the same orbitals, so it shows exactly what the variational
optimization did. For H₂ at a stretched bond, correlation takes density out of
the bond and puts it back on the two atoms:

```{image} ../../../examples/visualization/data/h2_cube_slice.png
:alt: Electron density and correlation density of stretched H2
:width: 100%
```

Spin-resolved densities, individual molecular orbitals, and fields of a crystal
are not yet available ([open items](#not-yet-available)). A frozen core is refilled
before the density is built, so an all-electron density integrates to the full
electron count whatever the active space.

## Occupations as data

```python
orbitals = atoms.calc.natural_orbitals()
print(orbitals.occupations)     # descending, in [0, 2], summing to N
print(orbitals.coefficients)    # column i: orbital i in the atomic-orbital basis
print(orbitals.mo_coefficients) # the same, in the molecular-orbital basis
```

Occupations that are not 2 or 0 *are* the correlation: a single determinant
would report $(2, 0)$ for stretched H₂, and the occupations of the converged
state say the bond is half broken.

## The field object

`write_cube` writes and returns the path. To keep the field, ask for it and write
it yourself, in either format:

```python
field = atoms.calc.volumetric_field("density")
print(field.integral)              # electrons on the grid
print(field.augmentation_charge)   # electrons not on the grid (PAW-LCAO)
print(field.n_electrons)           # the electron count of the state
field.write("density.xsf")         # the extension chooses cube or xsf
```

`write_cube` always writes the cube format; `field.write` follows the extension
(`.cube` or `.xsf`). A field carries `data`, `grid`, the nuclei (`numbers`,
`positions`, `charges`) and the `norm`.

## Grid, origin and units

Everything in a cube file is in **Bohr**, and a density is in electrons per
Bohr³. The data is written on the calculation's **own** grid and the header
carries that grid's origin and its three step vectors, so the atoms and the data
line up. A writer that assumed the grid was the cell divided by the node count
would put every nucleus in the wrong place; this is why the format is written by
Mandacaru and not by ASE.

`grid_spacing=` (Å) samples on a finer or coarser grid over the same box; the
basis functions are **re-evaluated** there and nothing is interpolated:

```python
atoms.calc.write_cube("density_fine.cube", grid_spacing=0.08)
```

On the calculation's own grid the electron count comes out exact to machine
precision, because that same quadrature defines the overlap under which the
orbitals are orthonormal; on any other grid the integral shows the difference
between the two quadratures, a real measure of how well the calculation's
`grid_spacing` resolved the basis. A cube file holds one number per node, so its
size grows as $1/h^3$. Molecular fields need an isolated grid.

## Excited states

`state=` selects the level of a subspace search, which stores its levels:

```python
atoms.calc = Mandacaru(method="subspace-vqe", basis="HAO",
                       grid_spacing=0.30, num_states=2)
atoms.get_potential_energy()
atoms.calc.write_cube("excited.cube", state=1)
```

A single-state result has one state; asking for another raises `IndexError`.

## Pseudopotentials: the smooth valence density

With a PAW-LCAO basis the orbitals are the **smooth pseudo** valence orbitals, so
a field is the *smooth valence density*: the core is absent by construction, and
inside the augmentation spheres the smooth density is not the physical one. The
overlap itself is augmented, so the smooth density does not integrate to the
valence electron count:

$$ \int\tilde n\,d^3r \;+\; \underbrace{\textstyle\sum_{pq}\gamma_{pq}
   (A^\dagger C q C^\dagger A)_{pq}}_{\text{augmentation charge}}
   \;=\; N_{\rm valence}. $$

`field.integral`, `field.augmentation_charge` and `field.n_electrons` are the
three terms, and the cube comment line says which one the file holds. The third
is exact whatever the basis; it is the one to check against. The nuclei are
written with their **real** atomic numbers so a viewer draws oxygen as oxygen,
and the valence charge goes in the charge column. A molecule's all-electron
density inside the spheres is not reconstructed; use an all-electron basis when
the core matters.

## Real and complex orbitals

A cube file holds real numbers. Densities are real by construction. A natural
orbital is determined up to a global phase; `component="auto"` (the default)
removes it, checks that the residual imaginary part is numerical noise and writes
the real part. If it is not noise the call raises and asks you to choose
`component="modulus"`, `"real"` or `"imag"` deliberately. The orbitals *within* a
degenerate set are fixed only up to a rotation, so which member of a π pair you
get is arbitrary, as in any code.

## Pictures and animations in Python: `Viewer3D`

`Viewer3D` draws the same quantities in three dimensions and writes a still
`.png` or an animated `.gif`. It takes the solved calculator, or a field already
in hand:

```python
from mandacaru import Camera, Viewer3D

# Antibonding natural orbital as a point cloud, sampled on a finer grid.
viewer = Viewer3D(atoms.calc, quantity="natural_orbital", index=1, grid_spacing=0.12)
viewer.save("no1.png")

# The same orbital as an isosurface, turning once while zooming in and out.
Viewer3D(atoms.calc, quantity="natural_orbital", index=1, grid_spacing=0.12,
         mode="isosurface").animate("no1.gif", frames=120, fps=12,
                                    zoom=(1.0, 1.8, 1.0))
```

| `mode` | What is drawn |
|---|---|
| `"scatter"` (default) | A Monte Carlo cloud of `points` points drawn with probability $\lvert\phi\rvert^2$ for an orbital and $\lvert n\rvert$ for a density, cyan where the field is positive and orange where it is negative |
| `"isosurface"` | The surfaces $f = \pm f_0$, extracted by marching tetrahedra and lit from the camera. Unless `isovalue=` is given, $f_0$ encloses a fraction `enclosed` (default 0.85) of $\int\lvert\phi\rvert^2$ or of $\int\lvert n\rvert$ |

Orbitals and `difference_density` are drawn with both signs; `density` is
non-negative. `grid_spacing=` (Å) re-evaluates the basis on a finer grid over the
same box; an isosurface takes longer to render as it shrinks, and 0.10–0.15 is a
good range for small molecules. `animate()` builds the camera path in three
layers: `trajectory=` (a list of `Camera(elev, azim, zoom, focus)` keyframes,
joined with eased interpolation; `focus` is `None`, an atom index, or a point in
Å), `rotate=` (`"azimuth"` by default, `"elevation"`, or `None`; `turns` times,
periodic so a looping GIF has no seam) and `zoom=` (a number or keyframes).
`camera_path()` returns the poses `animate()` would use, and `frames / fps` is the
animation length in seconds. The background is transparent unless `background=`
names a color.

## Opening the files

**VESTA.** `File ▸ Open` the `.cube` or `.xsf` file; the level is under
`Properties ▸ Isosurfaces`. Useful starting values, in e/Bohr³:

| Field | Isosurface level | Notes |
|---|---|---|
| `density` | `0.02` – `0.05` | `0.02` is the conventional "molecular shape" contour |
| `natural_orbital` | `±0.05` – `±0.10` | add both signs; VESTA colors them separately |
| `difference_density` | `±0.001` – `±0.005` | a correlation density is one to two orders of magnitude smaller than the density |

For a signed field add two isosurfaces at `+v` and `-v` with different colors.

**VMD.** `vmd -cube density.cube`, then `Graphics ▸ Representations`, set
**Drawing Method** to `Isosurface` and enter the isovalue; add a second
representation with the negative value for a signed field.

**XCrySDen.** `.xsf` is its native format; the cube format is the more precise
of the two and the one to prefer.
