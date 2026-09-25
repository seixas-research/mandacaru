# Periodic Systems

Molecules are open-boundary systems, but crystals are periodic. Mandacaru extends to
**1-, 2- and 3-dimensional crystals** through the same entry point as everything
else — `Mandacaru(method=..., kpts=...)` — with two periodic methods:

| `method=` | Supercell solver |
| :--- | :--- |
| `"bloch-adapt-vqe"` | adaptive ansatz (ADAPT-VQE) |
| `"bloch-vqe"` | fixed UCCSD ansatz (VQE) |

They do two things: solve the single-particle **Bloch Hamiltonian** at each k-point
for the band structure, and compute a **total energy that uses all k-points**.
`atoms.get_potential_energy()` returns that energy **per primitive cell**. The band
structure is single-particle, so it is **identical across the methods**; only the
correlated total energy differs.

The crystal is given as an ASE `Atoms` **primitive cell** — `atoms.cell` sets the
lattice vectors and `atoms.pbc` selects which directions are periodic. At least one
direction must be periodic; 1-D chains and 2-D slabs are fully supported.

```{admonition} The mesh must be Gamma-centered
:class: important
`kpts={"size": (2, 1, 1), "gamma": True}` is the spelling. A bare `(2, 1, 1)` triple
is **refused**: ASE's Monkhorst-Pack mesh for an even size sits at ±1/4 and does not
contain Gamma, and the Born-von Karman equivalence below is an identity only for a
Gamma-centered mesh. Odd sizes such as `(3, 1, 1)` already contain Gamma and are
accepted as bare triples.
```

```{admonition} The bands carry the two-body term
:class: note
They are the peaks of the interacting spectral function, built from the correlated
ground state by diagonalizing the Hamiltonian in the `N±1` particle-number sectors.
The exact check is the sum rule: removal and addition weights add to 1 for every
k, spin and orbital, since `{c, c†} = 1`.
```

```{admonition} Forces and stress are supported
:class: note
`atoms.get_forces()` and `calc.get_stress()` both work for the periodic methods —
see [Periodic forces and stress](#periodic-forces-and-stress) below for the Ewald
and Pulay terms behind them and for how a supercell gradient folds back onto the
primitive cell.
```

---

## A one-dimensional hydrogen chain

```python
import numpy as np
from ase import Atoms
from mandacaru.algorithms import Mandacaru

# One H per cell; periodic along x, vacuum in y and z.
atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
              pbc=[True, False, False])

atoms.calc = Mandacaru(method="bloch-vqe",
                       kpts={"size": (4, 1, 1), "gamma": True},
                       basis="HAO",
                       mapping="jordan_wigner",
                       h=0.20)
print(atoms.calc.dimension, "D crystal,", atoms.calc.n_bands, "band(s)")
```

`kpts` is the Brillouin-zone sampling and `h` the real-space grid spacing
(Ångström). `n_bands` is a property of the basis, so it answers before anything
runs; the bands themselves need the correlated state, so they come after
`get_potential_energy()`. The constructor is the same for every method — swap
`method="bloch-adapt-vqe"` for `method="bloch-vqe"`.

---

## Band structure

Because a band is the **spectral function** of the interacting state, not an
eigenvalue, `bands()` reports the **quasiparticle peak** — the pole of `A(E, k)`
carrying the most weight — together with how much weight that is:

```python
atoms.get_potential_energy()               # the bands need the correlated state

spectral = atoms.calc.get_spectral_function(eta=0.4)
bands = atoms.calc.bands(spectral=spectral)            # (nk, n_bands), eV
weights = atoms.calc.band_weights(spectral=spectral)   # (nk, n_bands), 0..1
mu = atoms.calc.get_fermi_level(spectral=spectral)     # interacting chemical potential
```

Energies are measured **from the N-electron ground state**: removal poles are
negative, addition poles positive, and `mu` falls between the branches. A weight
near 1 means the peak really is a one-particle excitation; a small one means the
weight is spread over satellites and there is no well-defined band there. That
spreading *is* the two-body term.

```{admonition} There is no continuous band path
:class: important
`band_structure(path)` raises `NotImplementedError`. The Bloch operators
`c_k = N^{-1/2} \sum_R e^{-ikR} c_R` exist only at the commensurate k-points the
Born-von Kármán supercell realizes, so an interacting band simply has no value
between them. A finer band means a larger `kpts`, hence a larger supercell and
proportionally more qubits — 4 k-points cost 8 qubits for this chain.
```

`examples/35_spectral_function_H_chain.py` writes and plots the full `A(E, k)`. In one
dimension the mesh already is a line, so this is enough. Two and three dimensions need
the band paths and symmetry tools described below.

---

## Total energy using all k-points

A correlated solver **cannot** be run independently per k-point and summed: the
two-electron interaction couples crystal momenta ($\mathbf{k}_1+\mathbf{k}_2 =
\mathbf{k}_3+\mathbf{k}_4+\mathbf{G}$). The exact way to fold **all** k-points into
a correlated total energy is the **Born–von Kármán equivalence**: an
$(n_1, n_2, n_3)$ Monkhorst-Pack mesh is a $\Gamma$-point calculation on the
$(n_1, n_2, n_3)$ supercell, so

```{math}
E_\text{cell} = \frac{E_\Gamma\big(\text{supercell}\big)}{n_\text{cells}} .
```

The driver builds that supercell with `atoms.repeat`, solves it with the selected
method (the box is the supercell's own cell), and divides by the number of cells.
`atoms.get_potential_energy()` is the energy per cell; the solver's own result object
— an `ADAPTVQEResult` or a `VQEResult` — is on `atoms.calc.result`:

```python
# Fixed-ansatz VQE.
atoms.calc = Mandacaru(method="bloch-vqe",
                       kpts={"size": (4, 1, 1), "gamma": True},
                       basis="HAO", h=0.20,
                       optimizer={"method": "L-BFGS", "maxiter": 2000,
                                  "tol": 1e-12})
e_cell = atoms.get_potential_energy()

# Adaptive ADAPT-VQE (extra adaptive controls).
atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                       kpts={"size": (4, 1, 1), "gamma": True},
                       basis="HAO", h=0.20,
                       max_iterations=10, gradient_tolerance=1e-3)
e_cell = atoms.get_potential_energy()
print(atoms.calc.result.num_operators, "operators grown")
```

The remaining options are the selected solver's own, so which are valid depends on
the method (`optimizer`/`h` for all; `pool`/`max_iterations`/`gradient_tolerance` for
the adaptive ones). This is a finite-supercell estimate
that converges to the bulk total energy as the mesh is refined (exact in the
infinite-mesh limit), and how tightly it settles depends on the gap. An
insulating H₂ chain (a = 3.0 Å, d = 0.74 Å, 10 Å vacuum, HAO, h = 0.30) gives
−29.315, −28.142, −28.156 and −28.108 eV per cell at 1 to 4 k-points: the first
refinement moves it by 1.2 eV and the rest stay inside a 0.05 eV band, though
the steps change sign, so read it as a band rather than a one-sided approach. A
**metal** swings an order of magnitude more, because meshes differ in whether
they are degenerate at the Fermi level: the half-filled hydrogen chain
(a = 1.0 Å, h = 0.35) gives −10.577, −12.940 and −11.592 eV per cell at 4, 6 and
8 k-points, since 4 and 8 are degenerate at `E_F` while 6 is closed-shell. Read
convergence off a gapped system, or off meshes of a single shell type. Note the supercell size — and therefore the qubit count —
grows with the mesh, so a dense mesh such as `(10, 1, 1)` (a 20-qubit supercell) is
heavy for exact state-vector simulation.

---

## Finite-size corrections

The Brillouin-zone mesh above is not the only finite-size error a crystal
calculation carries. Once the Madelung constant folded into the energy has
removed each electron's interaction with its own periodic images, a second,
smaller effect is left: the *pair* of electrons in the exchange-correlation
hole also interacts with `n_cells - 1` copies of itself, and that decays only
as `1/Omega`. `calc.finite_size_correction(scheme=...)` measures it directly
from the converged state — a **diagnostic**, so it never changes what
`atoms.get_potential_energy()` reports:

```python
mpc = atoms.calc.finite_size_correction(scheme="mpc")
print(mpc.summary())
```

Three schemes are available, and none of them is "the" answer to add on top of
the reported energy:

`"mpc"` (model periodic Coulomb)
: Contracts the exchange-correlation hole against the difference between the
  periodic Coulomb kernel the run used and the Wigner-Seitz minimum-image bare
  `1/r`. **Measured to overlap with the Madelung term the code already
  subtracts** — 86% of it on a 1x1x1 simple-cubic hydrogen cell and 92% on
  2x2x2, scaling as `Omega^(-1/3)` rather than `1/Omega`. Read it as a
  diagnostic of that overlap, not as a correction to add on top of the
  reported energy.

`"ccmh"` (Chiesa-Ceperley-Martin-Holzmann)
: The long-wavelength term the discrete k-sum omits at `k = 0`, taken from the
  computed structure factor `S(k)`. Additive and orthogonal to the Madelung
  constant — but it assumes the small-`k` quadratic regime of `S(k)` is
  resolved, and it warns (`RuntimeWarning`) rather than reporting a trustworthy
  number when it is not, which every cell size an exact state-vector
  calculation can reach fails to resolve.

`"kzk"` (Kwee-Zhang-Krakauer, *Phys. Rev. Lett.* 100, 126404, 2008)
: A size-dependent LDA exchange-correlation correction: a functional of the
  density alone, so it leaves every two-electron integral, and every Pulay
  force term, untouched. Its jellium fit assumes a cubic cell and a density
  range the fit has seen, so the result also carries `cubic_deviation` (how
  far the cell is from cubic) and `rs_range` (the Wigner-Seitz radii sampled),
  to say when that assumption is being stretched.

```{admonition} A diagnostic, not a correction to the reported energy
:class: important
Every scheme above evaluates a *post hoc* estimate of what the finite
supercell cost the exchange-correlation hole; nothing that was pinned during
the VQE or ADAPT-VQE optimization moves, and `atoms.get_potential_energy()`
is unchanged by calling any of them.
```

---

(periodic-forces-and-stress)=
## Periodic forces and stress

Both periodic methods support `atoms.get_forces()` and `calc.get_stress()`:

```python
atoms.get_potential_energy()          # converge the state first
forces = atoms.get_forces()           # eV/Angstrom, one row per primitive atom
stress = atoms.calc.get_stress()      # ASE's Voigt 6-vector, eV/Angstrom^3
```

The energy a crystal reports is

```{math}
E = \sum_{pq} D_{pq} h^{\text{MO}}_{pq}
  + \tfrac12 \sum_{pqrs} \Gamma_{pqrs} g^{\text{MO}}_{pqrs}
  + E_{\text{Ewald}} + \tfrac12 N_e v_{\text{M}} ,
```

and the force and the stress are its derivatives with respect to the atomic
positions and to a strain of the cell. The contraction against `(S, h, g)`, the
displaced-sampling derivatives and the orbital-response residual are the same
machinery the molecular gradient uses; only two terms are different objects in
a crystal: the electron feels the **Ewald potential of the whole ion lattice**
rather than a sum of `-Z/r`, and the ion-ion energy is the **Ewald energy**
rather than a finite pair sum.

**Hellmann-Feynman** rebuilds that ion-lattice Ewald potential with one ion
displaced and adds the analytic Ewald ion-ion force. **Pulay** re-samples the
displaced atom's basis functions *including their periodic images* — moving
one atom moves every image of it — and rebuilds `S`, `T`, `V` and the
two-electron tensor from the displaced stack.

```{admonition} The Madelung constant carries no force, and reappears in the stress
:class: note
$\tfrac12 N_e v_M$ depends on the cell alone, so it drops out of a fixed-cell
derivative. `get_stress` is where it shows up, since the cell is what is
varied there.
```

Validated against a central difference of the same fixed-state energy: the
analytic Ewald forces and stress agree with a numerical derivative of the
Ewald energy to round-off (forces to about `1e-10`, stress to `2.6e-10`
relative, and the forces sum to zero to `1e-14`, which is exact translational
invariance of a lattice sum); the full periodic gradient — Hellmann-Feynman
plus Pulay together — agrees with a central difference of the same
fixed-state total energy to about `1e-4` eV/Angstrom on forces of order
10 eV/Angstrom (`test/algorithms/test_periodic_forces.py`).

`get_stress(atoms=None, voigt=True, strain=None)` strains the cell, the atoms
and the grid *together* — twelve full Hamiltonian rebuilds, six independent
strain components either side of zero — and holds the grid's node *counts*
fixed so the strained grid stays commensurate with the strained cell. Straining
the grid along with the cell is deliberate: freezing the grid over a varying
cell would fold a change in discretization into the answer, and the two are
not separable.

```{admonition} The shear is independent of the grid's parity
:class: note
A single hydrogen atom in a cubic cell has an isotropic, shear-free stress by
symmetry, and every grid gives a shear at round-off, on an orthogonal or a
skewed cell alike, whatever its node count. `mandacaru.integrals.poisson.fft_g_squared`
symmetrizes the Nyquist plane of the FFT mesh, which is what removes an
otherwise even/odd-dependent spurious shear (a few times `1e-4` eV/Angstrom^3
on an even node count against round-off on an odd one): the Nyquist mode is
shared by the spectral kinetic operator and the periodic Coulomb kernel, so
the artifact traces to that mode, not to the real-space sampling. The
symmetrization is a no-op for an orthogonal cell, so it changes no energy
computed on one.
```

### A supercell gradient, folded to the primitive cell

The gradient is computed on the Born-von Kármán supercell — one row per
*supercell* atom — while `atoms` is the primitive cell and ASE expects
`len(atoms)` rows. The driver reconciles the two by the logic it already
applies to the energy: it reports `E_cell = E_super / n_cells`, and moving one
primitive atom moves *all* `n_cells` of its images together, so the
primitive-cell force is the **mean**, over a primitive atom's images, of the
supercell force on each of them. The raw, unfolded array survives on
`atoms.calc.force_result.details["supercell_forces"]`, and how far the images
disagree — ideally zero, for a supercell that respects its own translation
symmetry — is `details["image_spread"]`.

No example script accompanies this section: the force needs six displaced
Hamiltonian rebuilds per atom and the stress twelve more, on top of the state
already converged for `examples/11_Bloch_crystals.py`, which would multiply
that script's cost for what is, computationally, a classical contraction with
no new circuit evaluation. `test/algorithms/test_periodic_forces.py` is the
runnable, validated reference.

---

## Higher dimensions

The same interface handles 2-D and 3-D crystals, and the cell need not be
orthogonal — hexagonal, monoclinic, FCC, BCC and triclinic lattices all reach
an energy the same way a cubic one does, only `atoms.pbc`, the cell and the
k-points change. The grid-vs-cell commensurability check compares the two
consistently, so a skewed cell is accepted on the same footing as an
orthogonal one rather than raising `ValueError: not commensurate`. A square
lattice of hydrogen:

```python
square = Atoms("H", positions=[[0.0, 0.0, 0.0]],
               cell=[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 10.0]],
               pbc=[True, True, False])
square.calc = Mandacaru(method="bloch-vqe",
                        kpts={"size": (2, 2, 1), "gamma": True},
                        basis="HAO",
                        h=0.35)
square.get_potential_energy()          # the bands need the correlated state

# The four commensurate k-points of the (2, 2, 1) mesh -- and no others.
print(square.calc.kpoints)
print(square.calc.bands()[:, 0])
```

A path such as Gamma -> X -> M is not interpolated — it is **selected** from this
same four-point mesh, which is what the rest of this page is about.

---

## Band paths and crystal symmetry

In one dimension the k-point mesh already is the band path. In two and three
dimensions the mesh fills the zone while a band plot is a walk along segments
joining high-symmetry points, so selecting that walk out of the mesh is its own
step — `calc.band_path()` — and so is knowing the mesh's own symmetry —
`calc.symmetry()` and `calc.irreducible_zone()`. All three are properties of the
lattice and the mesh, so they answer **before anything is run**:

```python
print(square.calc.symmetry().summary())          # space group, from spglib
print(square.calc.irreducible_zone().summary())  # orbits of the mesh
kpath = square.calc.band_path("GXMG")             # Gamma -> X -> M -> Gamma
print(kpath.summary())
```

```{admonition} A path selects k-points; it never invents them
:class: important
The Bloch operators exist only at the commensurate k-points of the Born-von Kármán
supercell, so `band_path` returns the mesh points that **lie on** the requested
segments and nothing else — no interpolation. A high-symmetry point the mesh does
not carry is listed in `kpath.missing` and a `RuntimeWarning` names it, rather than
drawing an x-tick with no data under it. `X = (0, 1/2, 0)` is on a `2x2x1` or
`4x4x1` square-lattice mesh and is **not** on a `3x3x1` mesh, since that mesh only
has thirds. Resolving the missing point means enlarging `kpts`, hence the
supercell and the qubit count, not choosing a different path.
```

Pass `path=` straight to `get_spectral_function` (and to `bands`/`band_weights`,
which forward it) to get the rows in path order, with the matching x-axis on the
result:

```python
spectral = square.calc.get_spectral_function(path="GXMG")
bands = square.calc.bands(spectral=spectral)          # rows in path order
x = spectral.kpath.distances                          # the band-plot x-axis
```

`spectral.distances` (and `spectral.kpath.distances` when a path was used) follows
ASE's own convention — `2 pi` times the reciprocal cell, in inverse Ångström — so
it lines up with `ase.dft.kpoints.BandPath.get_linear_kpoint_axis()`. Falling back
to the plain k-index when there is no path is what makes `distances` the one axis
that is always correct, on a chain or on a lattice.

```{admonition} Symmetry reduces evaluations, not qubits
:class: note
`irreducible=True` evaluates `get_spectral_function` only at the symmetry-inequivalent
k-points and spreads the result over their orbits — a saving in **Lehmann
evaluations**. The Born-von Kármán supercell is fixed by the full `kpts` mesh
regardless, so the qubit count never changes.
```

That reduction is **audited**, not assumed: `A(E, k)` carries the symmetry of the
correlated *state*, and a state need not have the lattice's — a partially filled
degenerate manifold (a metallic mesh) makes the reference pick one member and
breaks the point group. One extra k-point from the largest orbit is evaluated and
compared with the representative it stands for; the result is
`IrreducibleZone.symmetry_residual`, and a value above `1e-6` warns rather than
returning a symmetrized fiction:

```python
reduced = square.calc.get_spectral_function(irreducible=True)   # may warn
print(reduced.irreducible.symmetry_residual)
```

Measured on the two-dimensional square lattice of `examples/36_spectral_function_square_lattice.py`:
a half-filled `2x2x1` mesh (a metal, C4 broken by the reference) gives a residual
of **1.2** and warns; a closed-shell 6-k-point H chain gives **1.3e-15** — exact
to machine precision, for a **1.50x** reduction in evaluations and no warning
(`test/algorithms/test_bloch.py::TestTheIrreducibleZoneIsAudited`; the chain of
`examples/35_spectral_function_H_chain.py` samples 8 k-points, which is a
half-filled metal and does *not* reduce cleanly).
`symmetry_residual` starts as `None`: nobody having checked is not the same as
having checked and found it clean.

The calculator also answers ASE's informal band-structure interface directly:

```python
square.calc.get_bz_k_points()      # every mesh k-point, fractional
square.calc.get_ibz_k_points()     # one representative per orbit
square.calc.get_k_point_weights()  # normalized orbit sizes, summing to 1
```

```{admonition} Non-cubic and 3-D cells are supported
:class: note
Square, hexagonal, simple cubic, FCC, BCC and triclinic cells all resolve their
ASE default path (`path=None`) — `"MGXM"` for the square lattice above, `"GMKG"`
for hexagonal, `"GXWKGLUWLK,UX"` for FCC, and so on. `atoms.pbc` restricts the
default to the periodic directions, so a slab in a tall box gets the 2-D path of
its plane rather than the 3-D path of its formal Bravais lattice.
```

`examples/36_spectral_function_square_lattice.py` runs all of this end to end on
the square lattice above: the space group and irreducible zone before any run, a
path with a missing point on purpose (`3x3x1`), the audited symmetry reduction,
and the updated `SpectralFunction.write()` CSV — now
`k_index, k1, k2, k3, k_distance_inv_ang, label, energy_eV, spectral_weight_per_eV`,
carrying every fractional component rather than only the first (which silently
dropped `k_y` for any lattice sampled in more than one direction).

Runnable end-to-end scripts: `examples/07_ADAPTVQE_H_chain_bands.py` (band structure
+ total energy to CSV, with a separate plotting script), `examples/11_Bloch_crystals.py`
(the three methods compared on the H chain), `examples/35_spectral_function_H_chain.py`
(the 1-D spectral function) and `examples/36_spectral_function_square_lattice.py`
(band paths and symmetry in 2-D).
