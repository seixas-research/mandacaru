# Crystals with the quantum methods

`method="vqe"`, `"adapt-vqe"`, `"vasqa"` and `"valsqa"` solve a crystal whenever the structure
has a periodic axis (`atoms.pbc`): there is no separate method name for a crystal. This guide
says what they solve, which options they take and refuse, and how to read bands, densities of
states and spectral functions from them.

## What is solved

A Gamma-centered `n1 x n2 x n3` Monkhorst-Pack mesh of a primitive cell is the Gamma point of the
`n1 x n2 x n3` supercell (the Born-von Karman equivalence). The correlated solvers act on that
supercell, and `get_potential_energy()` reports the energy **per primitive cell**:

```python
from ase import Atoms
from mandacaru import Mandacaru

chain = Atoms("H", positions=[[0, 0, 0]], cell=[2.0, 10.0, 10.0], pbc=[True, False, False])
chain.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space",
                       basis="HAO", grid_spacing=0.35,
                       kpts={"size": (2, 1, 1), "gamma": True},
                       optimizer={"method": "SLSQP", "maxiter": 1000, "tol": 1e-12})
energy = chain.get_potential_energy()      # -13.3828 eV per cell
```

The supercell Hamiltonian is a crystal's, built in Rust from the primitive cell:

- the grid holds one period of the supercell exactly, with a whole number of nodes per primitive
  vector (`round(|a_i| / grid_spacing)` per cell, times `n_i`), so a primitive translation is a
  whole number of grid steps and the Hamiltonian keeps the translation symmetry;
- every basis function is replaced by its Fourier series over the supercell, truncated at
  `|G| < pi / (2 max h)` (half the grid's Nyquist radius, so that every product of two orbitals
  stays on the grid): the series sums all periodic images and samples without aliasing;
- the electron-electron kernel is `4 pi / G^2` with `G = 0` dropped, the potential of the ions is
  the grid's own Fourier series of the same kernel (no softening), the ion-ion energy is an Ewald
  sum against the same uniform background, and the constant
  `1/2 N_e v_M` (the Madelung potential `v_M` of the supercell) restores each electron's
  interaction with its own periodic images;
- the kinetic energy is the spectral operator, periodic on the cell.

Because the orbitals and the ion potential are band-limited Fourier series, moving the atoms
changes only phases: the energy is exactly invariant under any rigid translation of the crystal,
not only under whole grid steps. The first native supercell point-sampled the hydrogen 1s cusp
and a softened `-Z/r` on the fixed grid, and a rigid translation of an H2 chain (a = 3.0 A, 2
k-points) changed its energy per cell by:

| `grid_spacing` | 0.001 A | 0.05 A | 0.1 A |
| --- | --- | --- | --- |
| 0.4 A, before | 6 meV | 0.16 eV | 0.09 eV |
| 0.2 A, before | 4 meV | 0.03 eV | -0.65 eV |
| 0.4, 0.3, 0.2 A, now | below 2e-12 eV | below 3e-12 eV | below 3e-12 eV |

The price of the band limit is that an all-electron cusp converges slowly with `grid_spacing`; the
energy now decreases monotonically as the grid is refined. The same H2 chain gives, in eV per
cell:

| `grid_spacing` (A) | 0.4 | 0.3 | 0.25 | 0.2 | 0.15 | 0.125 | 0.1 | 0.075 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| energy | -25.205 | -28.058 | -28.755 | -29.339 | -29.805 | -29.984 | -30.094 | -30.168 |

The kernel is periodic along all three lattice vectors. A chain or a slab is therefore a crystal of
chains or slabs separated by the cell's vacuum, which the cell vectors along the non-periodic
directions set (`cell=[a, 10, 10]` above). Neutral images interact through multipoles only; a
slab with a dipole would need a two-dimensionally truncated kernel, which is not implemented.

Without `kpts` the mesh is the Gamma point alone, as for `dft`: the supercell is the cell itself.
`vasqa` and `valsqa` build the same supercell Hamiltonian and run their architecture search on it
(VALSQA reads its trained proposal model as on a molecule: `proposal_model=`, or
`proposal_model.npz` in the `MANDACARU_PROPOSAL_DATA` store; without a ready model it uses
gradient proposals and says so in `result.proposal_reason`).

**`crystal_orbitals`.** The orbitals the quantum methods work in on a crystal: `"real-space"`,
the real orbitals of the supercell, or `"k-space"`, complex Bloch orbitals with
momentum-conserving excitations (the per-k register). `"k-space"` is the default but is not
implemented yet: it raises `NotImplementedError` naming `crystal_orbitals="real-space"`, which
every crystal run passes today. On a molecule the option is refused.

## Options

| Option | Rule |
| --- | --- |
| `kpts` | Gamma-centered: `{"size": (n1, n2, n3), "gamma": True}`, or an all-odd bare triple; the Gamma point when left out. A shifted mesh is not a supercell and raises. A size above 1 along a non-periodic lattice vector raises. |
| `crystal_orbitals` | `"real-space"`; the default `"k-space"` raises `NotImplementedError` until the per-k register exists. Any other value raises `ValueError`. |
| `basis` | `"HAO"` or a Gaussian recipe. A PAW-LCAO or UPAW-LCAO basis raises `NotImplementedError`: the periodic PAW supercell is not built yet. |
| `grid_spacing` | The target spacing; the supercell grid is built from it. An explicit `grid=` raises. |
| `kinetic` | `"spectral"` only. |
| `precision` | `"float64"` only. |
| `charge` | `0` only: a charged cell would need a background correction. |
| `transfer` | As on a molecule: `adapt-vqe`, `vasqa` and `valsqa` accept it, `vqe` refuses it. |

`bloch-vqe` and `bloch-adapt-vqe` are not method names: passing one raises `ValueError` naming the
plain method. `subspace-vqe` and `subspace-adapt-vqe` raise on a periodic geometry.

**Forces** of a quantum method on a crystal are central finite differences of the energy per
cell (a 1e-3 Bohr step; each displaced energy is a full calculation with the same options).
Displacing a primitive atom moves all its images, so the derivative of the energy per cell is the
force on that atom. The energy is translation invariant, so the forces sum to zero (1e-10 eV/A on
the H2 chain), and smooth in the positions (the bond derivative is -5.03257, -5.03272 and
-5.03333 eV/A at steps of 1, 2 and 4 mA). Their error is the grid convergence of the band-limited
basis: the bond derivative of the H2 chain is -5.155, -5.033, -5.321, -5.153, -5.036 and
-5.056 eV/A at `grid_spacing` 0.25, 0.2, 0.15, 0.125, 0.1 and 0.075 A, so a force is good to
about 0.3 eV/A at 0.25 A and to about 0.02 eV/A below 0.1 A. `vasqa` and `valsqa` need `seed=`
(each displaced energy is a new search), and a circuit run (shots, a device) raises.

**Stress** raises `NotImplementedError`: a strain moves the grid's wavevectors against the band
limit, so the set of plane waves kept changes and the energy jumps (6 to 30 meV per cell in the
second difference under a 1 percent uniaxial strain of the H2 chain). Crystal stress comes from
`method="dft"` with a PAW-LCAO basis.

## Validation

| Check | Result |
| --- | --- |
| Rigid sub-grid translation of the crystal (Rust, H2 chain, `grid_spacing` 0.25 and 0.15 A) | energy unchanged to 1e-9 Ha (Python: 3e-12 eV per cell) |
| Grid refinement 0.3, 0.25, 0.2, 0.15 A (Rust) | energy decreases monotonically |
| H chain, a = 2.0 A, (2,1,1) | -13.382786723 eV per cell, the exact ground state; the removed Python engine and the first native supercell gave -12.4222 on the point-sampled Hamiltonian (`test/migration/test_native_bloch.py` lists every moved pin) |
| Insulating H2 chain, a = 3.0 A, d = 0.74 A, (1,1,1) and (2,1,1) | -29.101976 and -27.979588 eV per cell (point-sampled: -29.315 and -28.142) |
| VQE against exact diagonalization, H chain (2,1,1), 4 qubits | equal to 1e-8 eV |
| VASQA and VALSQA, seeds 0-4, H chain (2,1,1) and (3,1,1) | every run at the exact ground state (1e-8 eV); (4,1,1), fermionic pool: -12.382599787 eV per cell, spread 1e-15 eV |
| Finite-difference forces, H2 chain, all four methods | sum 1e-10 eV/A; F_x = -6.353022 eV/A on atom 0 for each (to 1e-5) |
| Translation symmetry of the supercell Hamiltonian | sampled orbitals 1e-13, one-body matrix 1e-12, Coulomb integrals 1e-10 |
| Mesh (2,1,1) against the doubled cell at (1,1,1) | RHF energies equal to 1e-9 Ha |
| Madelung constant of the simple cubic lattice | -2.837297479 / L to 1e-8 |

The energy per cell converges slowly with the mesh, and a metal more slowly than an insulator:
the point-sampled H2 chain gave -29.315, -28.142 and -28.176 eV per cell at 1, 2 and 3 k-points. The
finite-size corrections of [the finite-size guide](finite_size.md) estimate the remainder.

## Bands, densities of states and the spectral function

After a run, the calculator resolves the mesh points the supercell realizes:

```python
chain.calc.kpoints                       # (nk, 3) fractional coordinates
```

**Mean-field band structure.** `band_structure(path, npoints)` returns an ASE `BandStructure` of
the supercell's Hartree-Fock reference unfolded onto the primitive cell. The Fock matrix of the
supercell orbitals is Fourier-interpolated (each matrix element assigned to the nearest supercell
image of its pair): at the mesh points this is the supercell spectrum exactly, between them an
interpolation whose error falls as the supercell grows. It is a mean-field band; the correlated
energies are in the spectral function, at the mesh points only. The exactness at the mesh points holds
for a reference that keeps the primitive translation (an insulator, or a metal whose degenerate
Fermi-level shell is fully occupied or empty). A restricted reference of a metal can mix a
degenerate pair of k-points to lower its energy, which breaks the translation: the band at
those k-points is then the average of the split pair.

**Density of states.** `dos()` and `pdos()` take `integration="gaussian"` (the default, a Gaussian
of standard deviation `width`) or `integration="tetrahedron"`: the linear tetrahedron method on
the full Gamma-centered mesh `kpts` (the calculation's own mesh by default), with no broadening.
For a Bloch run the levels are the unfolded mean-field bands, so any mesh can be asked for:

```python
energies, dos = chain.calc.dos(kpts=(48, 1, 1), integration="tetrahedron")
energies, pdos = chain.calc.pdos(kpts=(48, 1, 1), integration="tetrahedron")
```

`pdos` returns `{(atom, l): values}`. The projections are the Lowdin weights of each band on the
orbitals, so the channels add up to `dos`. The projected density of a Bloch run is computed with
the tetrahedron method only. The same keyword works for a DFT crystal, whose bands are diagonalized
at every mesh point.

**Berry phase and polarization.** The Berry phase, Berry curvature, Chern number, polarization
and anomalous Hall conductivity of a crystal are [a guide of their own](berry.md). They are
computed from the Bloch states of a `method="dft"` crystal: a quantum method on a crystal has
no Bloch bands there and `get_berry_phase` and the other Berry methods raise `NotImplementedError`.

**Spectral function.** `get_spectral_function(energies=None, eta=0.1, kpoints=None, krylov=200)`
returns a `SpectralFunction` (`mandacaru.analysis.bloch`). For every mesh point `k` and primitive
Lowdin orbital `nu` it holds the Lehmann representation of the one-particle Green's function of the
**variational state** of the run:

$$
A_{\nu\nu}(k,\omega) = \sum_n |\langle n,N{+}1|c^\dagger_{k\nu}|\Psi\rangle|^2\,\delta(\omega-(E_n-E_0))
 + \sum_n |\langle n,N{-}1|c_{k\nu}|\Psi\rangle|^2\,\delta(\omega-(E_0-E_n)),
$$

with $c^\dagger_{k\nu} = N^{-1/2}\sum_R e^{2\pi i k\cdot R} c^\dagger_{R\nu}$, $E_0$ the energy of the
VQE or ADAPT-VQE state and $E_n$ the eigenvalues of the same Hamiltonian in the $N\pm1$ sectors.
Each part is resolved by a Lanczos recursion of at most `krylov` vectors, which is the exact sum once
it spans the sector and otherwise reproduces the first `2 krylov` moments; the total weight of every
channel is 1 whatever `krylov` is (the sum rule `{c, c+} = 1`). The approximations are the state
(variational, not exact, whenever the ansatz falls short) and the mesh (the supercell resolves its
commensurate k-points only; another `k` raises).

| Attribute | Meaning |
| --- | --- |
| `values` | `(nk, n_energies)`: the trace over orbitals and spins broadened by a Lorentzian of half-width `eta`, states per eV per cell |
| `energies` | the axis, eV, excitation energies relative to $E_0$ |
| `sum_rule` | `(nk, n_orbitals, 2)`: the total weight of each channel and spin |
| `bands()`, `band_weights()` | `(nk, n_orbitals)`: the strongest pole of each channel and its weight (also `calc.bands()`, `calc.band_weights()`) |
| `removal_edge`, `addition_edge`, `gap`, `chemical_potential` | the highest removal pole, the lowest addition pole, their difference and their midpoint |
| `poles` | every pole: energy (eV), weight, k, orbital, spin, addition or removal |

The spectral function holds sector-sized vectors: it is an analysis of a state-vector run and
raises on a shot-based run, a tapered register or an active space (the Bloch operators expand over
every supercell orbital).

The Lehmann sum is checked against a brute-force diagonalization of a Hubbard ring with the
library's own Jordan-Wigner ladder operators (`crates/mandacaru-core/tests/spectral.rs`); at
`U = 0` every `A(k, omega)` is one pole of weight 1 at the tight-binding band energy. The
tetrahedron density holds one state per band, is half filled at the center of a simple cubic
tight-binding band, converges to the free-electron count at second order in the mesh step and its
projections add up to the total (`crates/mandacaru-core/tests/tetrahedron.rs`).

## Symmetry, the irreducible zone and band paths

All three are computed in Rust (`periodic::space_group`, `periodic::kpath`) and read the structure
and the `kpts` mesh off the calculator; they work before a run and for any method.

```python
calc.symmetry()          # CrystalSymmetry: space group, point group, operations
calc.irreducible_zone()  # IrreducibleZone: irreducible k-points, weights, star of every point
calc.band_path("GXMG")   # KPath: the mesh points on the path
```

**Symmetry.** `symmetry(symprec=1e-5)` is the space group of the structure (number, Hermann-Mauguin
symbol, point group, and the operations `x -> W x + t` in fractional coordinates). A cell with a
non-periodic axis is read as the crystal that repeats along it: one H atom in a 2.6 x 2.6 x 9.0 A cell
is P4/mmm (No. 123) with 16 operations.

**Irreducible zone.** The mesh is reduced by the rotation parts of the space group that map the
mesh onto itself, and by time reversal (the Hamiltonians of a Bloch run are real). A mesh the
lattice does not fit (a 2 x 1 mesh of a square lattice) keeps the subgroup that does. The weights
add up to 1. The 2 x 2 x 1 Gamma-centered mesh of the square lattice reduces to Gamma, X and M
with weights 1/4, 1/2 and 1/4; the 4 x 4 x 4 cubic mesh to the 10 points of the standard
reduction. Every star equals the orbit of its representative under the group, which the Rust
tests verify by brute force.

**Band paths.** A path is a string of high-symmetry labels (a comma jumps: `"GXMGRX,MR"`), a list
of fractional points (each optionally a `(label, point)` pair) or `None` for the lattice's default
path. The lattice class is read from the Gram matrix of the periodic cell vectors alone, so the
vacuum axes of a slab or a chain play no part. The labelled points follow Setyawan and Curtarolo
(Comput. Mater. Sci. 49, 299 (2010)) for the cubic, fcc, bcc, tetragonal, orthorhombic and
hexagonal lattices, in the cell orientations of that paper (a tetragonal cell lists its two equal
lengths first and an orthorhombic cell its lengths in increasing order, a < b < c; any other order
raises, so a label always means the paper's point), and the plain extensions below for one and two
dimensions (fractional coordinates of the periodic axes):

| Lattice | Points | Default path |
| --- | --- | --- |
| chain | G 0, X 1/2 | `GX` |
| square | G, X (1/2, 0), M (1/2, 1/2) | `GXMG` |
| rectangular | G, X (1/2, 0), Y (0, 1/2), S (1/2, 1/2) | `GXSYG` |
| hexagonal plane (gamma 120 or 60) | G, M (1/2, 0), K (1/3, +-1/3) | `GMKG` |
| cubic | G, X (0, 1/2, 0), M (1/2, 1/2, 0), R (1/2, 1/2, 1/2) | `GXMGRX,MR` |
| fcc, bcc, tetragonal, orthorhombic, hexagonal | the tables of the paper | `periodic/kpath/convention.rs` |

Any other lattice (oblique, centered rectangular, rhombohedral, monoclinic, triclinic, a cell in
another orientation) raises `ValueError` naming its lengths and cosines: give the path as
explicit points. Every named point is checked to lie on the boundary of the first Brillouin zone,
and those of the fcc and bcc lattices against their known Cartesian positions
(`crates/mandacaru-core/tests/kpath.rs`).

A Bloch run resolves its commensurate mesh points only, so `band_path(...)` and
`get_spectral_function(path=...)` select the mesh points (and their reciprocal-lattice images,
which carry the same Bloch operator) that lie on a segment, measured in the Cartesian reciprocal
metric. `KPath.distances` are the cumulative distances along the path in 1/A (`2 pi` included),
`labels` and `label_distances` the vertices (a jump is written `"X|M"`). A vertex that is no
mesh point is listed in `missing` and the path is not `complete`: `get_spectral_function` warns
and keeps the points it has (a 3 x 3 x 1 mesh has no X or M; a 2 x 2 x 1 or 4 x 4 x 1 Gamma-centered
mesh has both). A path through no mesh point at all raises.

**Spectral function along a path.** `get_spectral_function(path="GXMG")` returns the same
`SpectralFunction` over the path's points, one row each, with `.kpath`:

```python
spectral = calc.get_spectral_function(energies=np.linspace(-20, 20, 700), eta=0.3, path="GXMG")
plt.pcolormesh(spectral.kpath.distances, spectral.energies, spectral.values.T, shading="nearest")
```

`irreducible=True` resolves the Lehmann sum at the irreducible representative of each star only and
repeats it over the star. That is exact for a state with the symmetry of the crystal and wrong for
one without it, so it is audited: the representative of the largest star and one other member of
it are both resolved, and the relative L1 difference of their spectral functions is
`spectral.audit["residual"]`. Above 1e-6 the state is not symmetric and `get_spectral_function`
warns. A variational state of a degenerate frontier shell can break the point group (a
restricted reference fills one of two degenerate levels); the state of the 2 x 2 x 1 hydrogen
square lattice of `examples/spectral_function/02` does not: its X and Y spectral functions differ
by 3e-9 (relative L1), well below the tolerance. Without `irreducible` every row is an exact Lehmann
spectrum of the state at its own mesh point.

The spectral function is checked against the tight-binding square lattice (a 3 x 3 torus, where
`A(k, omega)` is one pole of weight 1 at `-2t(cos kx + cos ky)`: -4t at Gamma, -t at (1/3, 0), 2t
at (1/3, 1/3)), and for an interacting state against its invariance under the point group and
time reversal, `A(Rk) = A(k)` and `A(-k) = A(k)`
(`crates/mandacaru-core/tests/bloch_path.rs`).

## Not available

- band paths between mesh points (the correlated spectral function is resolved at the mesh points only);
- a PAW-LCAO supercell;
- stress (`NotImplementedError`) and analytic forces (the forces are finite differences);
- a correlated band between mesh points;
- a register per k-point (`crystal_orbitals="k-space"`).

See [the not-yet-available table](#not-yet-available).
