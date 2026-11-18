# Wannier downfolding and constrained RPA

`calc.wannier(...)` localizes the Kohn-Sham states of a periodic calculation into Wannier
functions. `wannier.downfold(...)` takes a **fragment** of those functions (the correlated
subspace: a Ni `e_g` set, the two most strongly coupled neighbor functions of silicon) and returns the many-body problem on it:
a one-body part, the Coulomb interaction between the functions, bare or screened by everything
outside the fragment, and the double counting that keeps the Kohn-Sham density from being
counted twice. The result goes straight to a quantum solver through `Mandacaru`.

```python
import numpy as np
from ase.build import bulk
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25, kpts={"size": (4, 4, 4), "gamma": True},
                       smearing={"method": "fermi-dirac", "width": 0.01})
atoms.get_potential_energy()
fermi = atoms.calc.get_fermi_level()
wannier = atoms.calc.wannier(8, guess="sp3", windows={"outer": (fermi - 20, fermi + 20),
                                                      "frozen": (fermi - 20, fermi + 1)})

# The fragment: the pair of functions on different atoms with the largest hopping |t|. Each
# function belongs to the atom of its largest trial weight, which does not depend on the gauge.
atom = [wannier.labels[int(np.argmax(w))].split(":")[0] for w in wannier.trial_weights]
h, vectors = wannier.hamiltonian_R, wannier.lattice_vectors     # <w_i,0|H|w_j,R>, eV
t, i, j, R = max((abs(h[r, i, j]), i, j, tuple(int(x) for x in vectors[r]))
                 for r in range(len(vectors)) for i in range(wannier.n_functions)
                 for j in range(wannier.n_functions) if atom[i] != atom[j])

model = wannier.downfold([(i, (0, 0, 0)), (j, R)], screening="crpa")
print(model.two_body[0, 0, 0, 0].real * HARTREE_TO_EV)   # on-site U of one hybrid, eV

result = Mandacaru(method="adapt-vqe", **model.as_quantum_problem()).run()
```

The minimum of the spread for this window is not the set of bond-pointing `sp3` hybrids that the
guess starts from: it is, on each atom, four hybrids that point into the interstitial, away from
the neighbors (the bond-pointing set is a saddle of the spread, which the saddle check below
leaves). No pair of functions sits on a bond, so the example does not pick one by position. Its
two-site fragment is the neighbor pair with the largest `|t|`, and `U` is the on-site repulsion of
one such atom-centered hybrid, not the repulsion of a bond orbital. `as_quantum_problem()` rotates
the pair to its bonding and antibonding combinations.

Everything numerical runs in Rust, from the mesh states and the PAW geometry of the overlaps to the
localization. Python only converts the arguments and presents the arrays: the request it sends
holds the choices of the user (mesh, trial functions, bands, windows), and the atom centers and
projector channels are read from the crystal's own native setup.

## Localization

`calc.wannier(n_functions=None, *, guess="bonds", windows=None, bands=None, kpts=None,
max_iterations=2000, tolerance=1e-10, step=0.5)` localizes the Bloch states of a converged `dft`
crystal on the Gamma-anchored mesh `kpts` (default: the calculation's own). `windows` is the outer
energy pair and an optional frozen pair in eV; without it the group is the lowest `n_functions`
bands (or the `bands` indices), one band per trial function, and it must be isolated.

- **Objective.** The functions minimize the total spread `Omega = Omega_I + Omega_OD + Omega_D`
  (maximal localization; the periodic counterpart of the Foster-Boys criterion, Marzari and
  Vanderbilt). With `windows`, the subspace is first chosen by minimizing the gauge-invariant
  part `Omega_I` (Souza, Marzari and Vanderbilt), keeping the frozen window exactly, and the
  localization then fixes the gauge inside it. The spread is the only objective; a
  population-based criterion is not available.
- **Overlaps.** The links between neighboring mesh points are the overlaps of the periodic parts,
  with the PAW one-center term of the full plane wave inside the spheres: the exact operator, not
  its first-order expansion in the displacement `b`.
- **Neighbor shells.** The finite-difference weights `w_b` satisfy the completeness relation
  `sum_b w_b b_a b_b = delta_ab` exactly, on any mesh, orthogonal or not. Shells of equal
  displacement are taken in order of length; one that is parallel to an accepted shell, or that
  adds no independent equation, is skipped, so no weight falls on a longer step of a direction a
  shorter one already covers. If the exact weights are not all positive, further shells are added
  and the weights found by nonnegative least squares. A mesh with no nonnegative solution raises
  `ValueError` and asks for a mesh closer to isotropic along the cell.
- **Minimizer.** A Polak-Ribiere conjugate gradient on the product of the unitary gauge groups,
  with restarts. Its line search follows the exact derivative of the spread along the search
  direction rather than the spread itself: it lengthens the first trial step (`step` scaled by the
  mesh and the shell weights) until the derivative changes sign, narrows the bracket, and never
  accepts a point that raises the spread. So it keeps descending where the spread's own decrease
  is below its round-off. The descent stops when the gradient norm is below 1e-6 Bohr^2 and the
  last step changed the spread by less than `tolerance` (Bohr^2).
- **Saddle check.** A vanishing gradient does not tell a minimum from a saddle, and the spread
  has both. A stopped point is turned by a small fixed rotation of the gauge (size 1e-2 at every
  k-point) and minimized again; a lower spread replaces it. The check repeats on the new point,
  up to three successive escapes. A direction of negative curvature weaker than about 6e-3
  Bohr^2/rad^2 lowers the spread by less than `tolerance` over such a rotation and goes
  undetected. `result.converged` is true only at a point a check confirmed (and when the
  disentanglement converged). It is false when the iterations ran out, including during a
  check.

What raises:

- **An unconverged SCF** raises `ValueError`: its states belong to an unconverged Hamiltonian.
- **A windowless group that is not isolated** raises `ValueError` naming the band, the spin
  channel and the k-point where it touches a band outside the group, and suggests `windows=`: a
  subspace that touches another band is not smooth, and would localize without an error into a
  gauge that is not.
- **A magnetic spinor subspace with a nonzero Chern number** raises `ValueError` naming the
  number and the k plane: such a subspace has no exponentially localized Wannier functions. Use
  `get_chern_number` or the Wannier charge-center flow of the [Berry guide](berry.md) instead. The
  check runs only for a spinor crystal without time-reversal symmetry, the only Hamiltonian that
  can give a subspace a Chern number, on every layer of every k plane the mesh resolves.
- **A crystal loaded from a snapshot** is rebuilt automatically from its `setup`/`sites`
  block (including dataset paths and content digests), so Wannier localization and cRPA
  downfolding work both in-memory and after `load_crystal`.

## What `downfold` returns

`DownfoldedProblem`, in Hartree:

| Attribute | Meaning |
| --- | --- |
| `kohn_sham` | the Kohn-Sham Hamiltonian between the fragment's functions |
| `two_body` | the Coulomb integrals `<pq|rs>` in physicists' order; `two_body[0, 0, 0, 0]` is the on-site repulsion `U` and `two_body[0, 1, 1, 0]` the exchange `J` |
| `density_matrix` | the Kohn-Sham density matrix `<c_p^+ c_q>` |
| `double_counting` | the Hartree and exchange potential of the fragment's own electrons at that density |
| `one_body` | `kohn_sham - double_counting`: with it, the Fock matrix of `density_matrix` is `kohn_sham` again |
| `dielectric_head` | the long-wavelength inverse dielectric `eps^-1_00(q -> 0)` of the screening, `None` when bare |
| `num_particles`, `n_spatial_orbitals`, `functions` | the electrons `(up, down)` (default: the Kohn-Sham count, which must be within 0.1 of an integer), the size and the modes `(n, R)` |

A restricted crystal's arrays are spatial, `two_body` is `(n, n, n, n)` and `density_matrix` holds
both spins. A spin-polarized crystal has a channel axis: `kohn_sham` is `(2, n, n)` and
`two_body[s, t]` is `<p^s q^t|r^s s^t>`, with each spin channel's own functions.
`as_quantum_problem()` writes the model in the natural orbitals of the density matrix, most
occupied first (a determinant of localized functions is a poor start for ADAPT-VQE) and returns the
options `Mandacaru(method="adapt-vqe", **options)` takes.

## Options

`wannier.downfold(functions=None, num_particles=None, *, screening=None, screening_cutoff=None,
screening_bands=None, coulomb="supercell")`

- `functions`: indices `n` of the home cell, or `(n, R)` with `R` an integer lattice vector; the
  default is every function of the home cell.
- `screening`: `None` (bare), `"crpa"` (every transition screens except those inside the Bloch
  subspace of the fragment's functions, in the projector-weighted form that is valid for entangled
  bands) or `"rpa"` (every transition screens). Any other value raises `ValueError`. The screening
  is static, from the Kohn-Sham states, not self-consistent.
- `screening_cutoff`: the reciprocal vectors `|q + G|` kept, in Bohr^-1 (default 3).
- `screening_bands`: how many of the Wannier calculation's lowest states screen (default: all it
  kept; more than it kept raises `NotImplementedError` naming `windows=`/`bands=`).
- `coulomb`: `"supercell"` (the default) evaluates the interaction on the Born-von Karman
  supercell of the k mesh, truncated beyond half its shortest lattice vector; `"isolated"` puts
  the fragment alone in a supercell twice as large, truncated at the whole shortest vector, at
  eight times the grid. A fragment whose charge reaches past the truncation radius meets its own
  periodic copies and a warning says so.

Both collinear (`spins=1` or `2`) and spinor/noncollinear (`spinor=True`, yielding spin-orbital
tensors) crystals are supported with a `"raw"` projector basis (`"dual"` projector basis is
refused cleanly). Screening is static ($\omega = 0$) and drops dielectric wings
($\mathbf{G} \neq \mathbf{0}, \mathbf{G}' = \mathbf{0}$).

## How it is built

The pair densities `w_p^* w_q` of the fragment's functions are formed on the supercell grid and
completed with the PAW compensation charges and the one-center remainder of every atom copy, so
that the interaction is all-electron beyond the multipoles. With a screening, the static
polarizability

```
chi_GG'(q) = 1/(N Omega) sum_{k,a,b} (f_a - f_b)/(e_a - e_b) (1 - p_a p_b) rho_ab(q+G) rho_ab(q+G')^*
```

is summed over the mesh pairs `(k, k+q)` of every spin channel (`p` is the weight of a state in the
correlated subspace; `p = 0` is the full RPA), `W = [1 - v chi]^-1 v` is solved per `q` with the
periodic Coulomb interaction, and `W - v` is contracted with the fragment's pair densities. The
divergent head is the `q -> 0` limit of the same polarizability at `q = 0.02` Bohr^-1 along each
axis, averaged; it scales the bare integrals' own `G = 0` term, and the wings are dropped.

## Reading the numbers

- The screened interaction depends on the mesh (`W` is a sum over `q` of the supercell), on the
  empty states the basis supplies and on the smearing. A smearing comparable to the mesh's gap puts
  thermal carriers at the band edges, and their intraband response makes `eps^-1_00` collapse
  (`0.01` for silicon at 0.1 eV): for a gapped crystal use a small smearing for the Kohn-Sham run.
- With `coulomb="isolated"` the screening and the dielectric head are the supercell's, not the
  doubled box's: `W - v` (including the head's `G = 0` term, which scales the supercell kernel's
  `2 pi R_c^2`) is added to the bare integrals of the fragment in the doubled box. The doubled
  box's own long-range tail (between the two truncation radii) is therefore unscreened; for a
  fragment that fits inside the plain radius the two boxes agree within meV.
- The functions of a fragment are the ones the localization returned: a different guess or window
  is a different model.
- Absolute numbers of this implementation, the gates they pass and the literature ranges they are
  compared with only as a sanity check are recorded in `dev_plan/PLAN5.md`.

## References

The truncated interaction is that of Spencer and Alavi (Phys. Rev. B 77, 193110, 2008); the
constrained RPA that of Aryasetiawan et al. (Phys. Rev. B 70, 195104, 2004), with the
projector weights of Sasioglu, Friedrich and Bluegel (Phys. Rev. B 83, 121101, 2011).
`citation_keys(analysis="downfold-screening")` lists them.
