# Wannier downfolding and constrained RPA

`calc.wannier(...)` localizes the Kohn-Sham states of a periodic calculation into Wannier
functions. `wannier.downfold(...)` takes a **fragment** of those functions (the correlated
subspace: a Ni `e_g` set, the two functions of a Si bond) and returns the many-body problem on it:
a one-body part, the Coulomb interaction between the functions, bare or screened by everything
outside the fragment, and the double counting that keeps the Kohn-Sham density from being
counted twice. The result goes straight to a quantum solver through `Mandacaru`.

```python
from ase.build import bulk
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25, kpts={"size": (4, 4, 4), "gamma": True})
atoms.get_potential_energy()
fermi = atoms.calc.get_fermi_level()
wannier = atoms.calc.wannier(8, guess="sp3", windows={"outer": (fermi - 20, fermi + 20),
                                                      "frozen": (fermi - 20, fermi + 1)})
model = wannier.downfold([2, 5], screening="crpa")       # two sp3 hybrids of one bond
print(model.two_body[0, 0, 0, 0].real * HARTREE_TO_EV)   # U in eV

result = Mandacaru(method="adapt-vqe", **model.as_quantum_problem()).run()
```

Everything numerical runs in Rust. Python only converts the arguments and presents the arrays.

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

A spinor crystal raises `NotImplementedError`. The crystal must come from a native setup (it
carries the dataset description the transition densities are rebuilt from).

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
  compared with only as a sanity check are recorded in `dev_plan/CRPA_PLAN.md`.

## References

The truncated interaction is that of Spencer and Alavi (Phys. Rev. B 77, 193110, 2008); the
constrained RPA that of Aryasetiawan et al. (Phys. Rev. B 70, 195104, 2004), with the
projector weights of Sasioglu, Friedrich and Bluegel (Phys. Rev. B 83, 121101, 2011).
`citation_keys(analysis="downfold-screening")` lists them.
