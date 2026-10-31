(dft-u)=
# DFT+U: the on-site Hubbard correction

A semilocal functional spreads the electrons of a partly localized `d` or `f` shell
over too many orbitals and underestimates the gap of an oxide such as NiO (0.6 eV in the example below). DFT+U adds an
energy that penalizes fractional occupation of that shell. In Mandacaru it is an
option of a periodic `method="dft"` calculation, `hubbard_u`; the occupation
matrices, the energy and the operator are computed in Rust, and Python only states
the request and lists which projectors belong to which atom. The script is
`examples/dft/08_dft_nio_afm_hubbard_u.py`.

## The option

```python
from mandacaru import Mandacaru

atoms.calc = Mandacaru(method="dft", xc="lda",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       kpts={"size": (2, 2, 2), "gamma": True},
                       spin=True,
                       hubbard_u={"Ni": {"l": "d", "U": 6.0, "J": 0.0}})
```

`hubbard_u` maps an element symbol to its correlated shell: `"l"` is `"d"` or
`"f"`, `"U"` and `"J"` are in eV, and `"J"` defaults to 0. Every atom of a listed
element gets the correction on that one shell; an element may carry one shell. The
energies must be finite, `U >= 0`, `J >= 0` and `J <= U`. Giving no `hubbard_u`
is a plain calculation, and `U = 0` is the same calculation to the precision of the
SCF (`result.hubbard_energy` is then exactly 0.0).

`hubbard_u` belongs to `method="dft"`: another method raises `ValueError`
([the methods reference](#methods-table)).

## The functional

The form is the simplified rotationally invariant one of Dudarev and colleagues
with the fully localized double counting (the only form there is; the snapshot
names it `dudarev-fll`). It sees one effective interaction, `U_eff = U - J`, and the
occupation matrix `n^sigma` (Hermitian, `(2l+1) x (2l+1)`) of each spin channel of
the shell:

$$
E_U=\frac{U_\mathrm{eff}}{2}\sum_\sigma \mathrm{Tr}\left[n^\sigma\left(1-n^\sigma\right)\right],
\qquad
V^\sigma=\frac{\partial E_U}{\partial n^\sigma}=U_\mathrm{eff}\left(\tfrac12-n^\sigma\right).
$$

`E_U` is zero for an idempotent `n` (every eigenvalue 0 or 1) and positive for a
fractional one, so the operator `V` moves a nearly filled orbital (`n` near 1) down by about
`U_eff/2` and a nearly empty one (`n` near 0) up by the same amount, which opens
the gap. `J` enters only through `U - J`: with `J != 0` the calculation is the one
with `U_eff` and nothing else
([what is not implemented](#not-yet-available)).

The operator `V` acts on the Kohn-Sham matrix through the projectors of the shell
and is part of the SCF: the energy, the eigenvalues, the gap and the occupations
are self-consistent with it. `E_U` is already inside `get_potential_energy()`.

## What `n` is: a filled shell is below 1

The occupation matrix is the projection of the Kohn-Sham density onto the PAW
partial waves of the shell: `n = W^T R W`, where `R` is the density matrix over
the shell's projectors summed over the k-mesh and `W` holds one weight per
projector radial index, from the dataset's reference all-electron wave and the
Vanderbilt matrix of the channel. The weights come from Rust (they are computed
once for each element and projector basis). A restricted run has the spin-summed
density, so each spin channel takes half of it.

This projection is not orthonormalized. It measures the weight of the shell inside
the augmentation sphere, so a completely filled shell has an occupation below 1
(Ni 3d in the NiO example below: the largest eigenvalue is 0.959 at U = 6 eV), and the `U` that reproduces a measured gap
is not interchangeable with a value from a code that orthonormalizes its
projectors. Quote the projection with the number.

The SCF starts from a diagonal seed built from the reference electrons of the
shell in the dataset and the atom's initial magnetic moment, signed per atom, so
an antiferromagnetic start gives opposite signs on the two Ni sites. A restricted
run has no spin to seed: a nonzero initial magnetic moment on it raises. Use
`spin=True` with the moments, as in [Open-shell systems](open_shell.md).

## What is exposed

On the result of a run (`calc.result`, or the value `run` returns):

| Property | Meaning |
| --- | --- |
| `hubbard_energy` | `E_U` in the result's energy unit, already inside the total energy; `None` without `hubbard_u` |
| `hubbard` | One dictionary per correlated atom, or `None` |

Each dictionary has `atom` (index in the geometry), `symbol`, `l` (`"d"` or
`"f"`), `U`, `J` and `energy` (this shell's part of `E_U`), `trace` (the trace of
`n` for each spin) and `n`, an array `spins x (2l+1) x (2l+1)` of complex numbers,
Hermitian, in the basis of complex harmonics ordered `m = -l ... l`. Read the
eigenvalues with `numpy.linalg.eigvalsh(shell["n"][spin])`.

```python
result = atoms.calc.result
print(result.hubbard_energy)                  # eV
for shell in result.hubbard:
    print(shell["symbol"], shell["atom"], shell["trace"])
```

The gap and the Fermi level come from the usual ASE readers on the calculator
(`get_eigenvalues`, `get_fermi_level`, `get_ibz_k_points`). Off-mesh bands
(`band_structure()`) and the Wannier frame diagonalize with the converged
operator, so they carry the correction; the example computes the gap from the
mesh eigenvalues.

## A saved crystal owns its correction

`save_crystal=` writes the DFT+U table with the rest of the setup, and
`load_crystal=` reads it back and applies it as written. Give no `hubbard_u` with
`load_crystal`, or give exactly the file's values: a different mapping raises
`ValueError` ("disagrees with the crystal file"), and so does a `hubbard_u` for a
file that has none.

## The run log

A run with `txt=` or `trace=True` states the correction once, in `[SCF SUMMARY]`,
and nowhere else: `hubbard_energy_eV` and one `hubbard_<element>_<shell>` line
with `U`, `J` and `U_eff`. The block is described in
[the run-log reference](../reference/run_log.md).

```text
    hubbard_energy_eV: 4.2560147802
    hubbard_Ni_d: U 6.000000 eV, J 0.000000 eV, U_eff 6.000000 eV
```

## What raises

Every unsupported request raises; none selects another calculation.

| Request | Raises |
| --- | --- |
| A molecule (no periodic direction) | `ValueError`: DFT+U is for crystals. The molecular route has no `+U` term. |
| A screened hybrid (`xc="hse06"`) | `NotImplementedError`: the correction has no exchange operator to share with it. |
| A spin-orbit (fully relativistic) dataset | `NotImplementedError`: scalar-relativistic sets only. |
| `precision="float32"` | `ValueError`: single precision is unvalidated. |
| A symmetry-reduced k-mesh with `symmetry=True` | The Rust loader refuses: the shell occupations are not symmetrized with the density yet. Use `symmetry=False`. A calculation with nonzero initial magnetic moments already keeps the full mesh. |
| A restricted calculation with a nonzero initial moment | The loader names `restricted` and `moment`. |
| A malformed mapping, an unknown element, a shell other than `d` or `f`, a negative or non-finite energy, `J > U` | `ValueError` naming the entry. |
| An element that is not in the system, or a dataset without the requested channel | `ValueError`. |
| A channel with no occupied reference wave in the dataset (for example Ba 5d, Au 5f) | The loader refuses: there is no wave to project on. |
| Crystal forces and stress | `NotImplementedError` and `PropertyNotImplementedError`, as for every crystal ([open items](#not-yet-available)). |

What is not implemented, and what that means for results, is listed in
[the not-yet-available table](#not-yet-available): forces and stress, the
symmetrizer that would allow a symmetry-reduced mesh, the Liechtenstein form with
separate `U` and `J`, and a linear-response `U`. The cRPA workflow of the Wannier
route *computes* an interaction; `hubbard_u` takes one as input.

## Example: NiO

`examples/dft/08_dft_nio_afm_hubbard_u.py` runs the type-II antiferromagnetic cell
of rocksalt NiO (a = 4.17 angstrom, LDA, PAW-LCAO `SZ`, `grid_spacing=0.25`, a
2x2x2 mesh, Fermi-Dirac smearing of 0.05 eV, initial moments +2 and -2 on the two
Ni atoms) at `U = 0` and `U = 6` eV. It prints the energy, the Kohn-Sham gap
(lowest empty minus highest occupied level over the mesh and both spins), `E_U` and
the occupation matrix of each Ni site, and writes the two run logs to
`examples/dft/data/`. On the last check it printed:

| | `U = 0` | `U = 6` eV |
| --- | --- | --- |
| Total energy | -10112.3922 eV | -10107.4683 eV |
| Kohn-Sham gap | 0.597 eV | 2.229 eV |
| `E_U` | none | 4.2560 eV (2.1280 eV on each Ni) |
| Ni 3d trace, spin up / down (site 0) | not printed | 4.789 / 3.382 |
| Eigenvalues of `n` | not printed | 0.256 to 0.959 |

Both SCFs converge (20 and 27 iterations) in a few seconds each on two threads.
The two Ni sites are spin mirrors: site 2 has the traces of site 0 swapped. These
numbers are those of this mesh and this `SZ` basis; converge both before reading a
gap against an experiment, and remember that the `U` is an input.

## Scientific references

- Dudarev, Botton, Savrasov, Humphreys and Sutton, *Physical Review B* **57**, 1505
  (1998), [doi:10.1103/PhysRevB.57.1505](https://doi.org/10.1103/PhysRevB.57.1505).
