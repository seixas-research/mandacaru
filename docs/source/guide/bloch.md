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
energy = chain.get_potential_energy()      # -12.4222 eV per cell
```

The supercell Hamiltonian is a crystal's, built in Rust from the primitive cell:

- the grid holds one period of the supercell exactly, with a whole number of nodes per primitive
  vector (`round(|a_i| / grid_spacing)` per cell, times `n_i`), so a primitive translation is a
  whole number of grid steps and the Hamiltonian keeps the translation symmetry;
- every basis function is summed over the lattice translations that reach the grid;
- the electron-electron kernel is `4 pi / G^2` with `G = 0` dropped, the external potential and
  the ion-ion energy are Ewald sums against the same uniform background, and the constant
  `1/2 N_e v_M` (the Madelung potential `v_M` of the supercell) restores each electron's
  interaction with its own periodic images;
- the kinetic energy is the spectral operator, periodic on the cell.

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

**Forces and stress** of a quantum method on a crystal raise `NotImplementedError`. Only a finite
difference of the supercell energy could give them, and that energy depends on where the atoms
sit relative to the fixed periodic grid. On an H2 chain (a = 3.0 A, HAO, 2 k-points), translating
the whole crystal rigidly changes the energy per cell by:

| `grid_spacing` | 0.001 A | 0.05 A | 0.1 A |
| --- | --- | --- | --- |
| 0.4 A | 6 meV | 0.16 eV | 0.09 eV |
| 0.2 A | 4 meV | 0.03 eV | -0.65 eV |

The Hartree-Fock reference moves alike (measured 2026-11-10). A central difference would return
artifacts of several eV/A, and the finite-difference forces summed to -15.5 eV/A per cell instead
of zero. Energies compare only at a fixed placement of the atoms on the grid. Crystal forces and
stress come from `method="dft"` with a PAW-LCAO basis.

## Validation

| Check | Result |
| --- | --- |
| H chain, a = 2.0 A, (2,1,1), against the removed Python periodic engine | -12.4222428812 eV per cell against -12.4222428989 (1.8e-8 eV) |
| Insulating H2 chain, a = 3.0 A, d = 0.74 A, (1,1,1) and (2,1,1) | -29.31494 and -28.14154 eV per cell against the old -29.315 and -28.142 |
| VQE against exact diagonalization, H chain (2,1,1), 4 qubits | equal to 1e-8 eV |
| VASQA and VALSQA, seeds 0-4, H chain (2,1,1) and (3,1,1) | every run at the exact ground state (1e-8 eV); (4,1,1), fermionic pool: -11.26294664 eV per cell, spread 1.8e-15 eV |
| Translation symmetry of the supercell Hamiltonian | sampled orbitals 1e-13, one-body matrix 1e-12, Coulomb integrals 1e-10 |
| Mesh (2,1,1) against the doubled cell at (1,1,1) | RHF energies equal to 1e-9 Ha |
| Madelung constant of the simple cubic lattice | -2.837297479 / L to 1e-8 |

The energy per cell converges slowly with the mesh, and a metal more slowly than an insulator:
the H2 chain above gives -29.315, -28.142 and -28.176 eV per cell at 1, 2 and 3 k-points. The
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

## Not available

- the irreducible zone and its symmetry audit, and band paths that select mesh points;
- a PAW-LCAO supercell;
- forces and stress (`NotImplementedError`, see "Options");
- a correlated band between mesh points;
- a register per k-point (`crystal_orbitals="k-space"`).

See [the not-yet-available table](#not-yet-available).
