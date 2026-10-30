(paw-lcao-basis)=
# PAW-LCAO bases

A projector augmented-wave (PAW) dataset replaces an atom's frozen core and the
singular $-Z/r$ potential by a smooth, valence-only problem, and a PAW-LCAO
basis is the dataset's own smooth pseudo-atomic orbitals. In Mandacaru the
family is a basis name, selected exactly like `"HAO"`:

```python
from mandacaru import Mandacaru

Mandacaru(method="rhf", basis="PAW-LCAO", grid_spacing=0.25)
Mandacaru(method="dft", xc="pbe",
          basis={"name": "PAW-LCAO", "size": "DZP"}, grid_spacing=0.2)
```

Two spellings are accepted, `"PAW-LCAO"` and `"UPAW-LCAO"`, written exactly so,
as a string or as the `name` of a dictionary. A per-element mapping of PAW-LCAO
bases is not available; mix families per element only among the all-electron
bases ([Basis sets](basis_sets.md)). PAW-LCAO is the recommended family and the
one to use for [forces and geometry optimization](forces_relaxation.md), because
the smooth valence orbitals are resolved on the real-space grid and an
all-electron core is not. `"UPAW-LCAO"` is the unitary variant, with no overlap
correction; it is optional and reads its own library.

The dataset gives the basis, the local potential, the projectors and the
compensation charges. Oxygen keeps its six valence electrons. The engine samples
the orbitals, builds the integrals, the augmentation and the SCF in Rust; Python
reads the dataset tables and prepares the radial basis tables.

## What is supported

| | |
| --- | --- |
| Methods | `rhf`, `uhf`, semilocal `dft` (`lda`, `pbe`, `r2scan`), and the variational methods through an RHF or UHF reference |
| Frozen-core exchange (Hartree–Fock, variational) | an empty core or a single closed 1s shell |
| Forces | RHF, UHF and semilocal Kohn–Sham ([Forces](forces_relaxation.md)) |
| Geometry | molecules; a crystal reads its own datasets through the periodic setup |

The following raise, or are not yet available ([open items](#not-yet-available)):
screened exact exchange (`hse06`) and nonlocal correlation (`pbe+rvv10`,
`r2scan+rvv10`) with PAW; multi-shell frozen-core exchange; spin-orbit
projectors, so a molecule on a `-fr` dataset raises `NotImplementedError`;
`precision="float32"` forces; the GHF-to-quantum handoff.

The two families of one-center terms differ by method. Kohn–Sham keeps the
dataset's frozen one-center Hartree and exchange–correlation terms. Hartree–Fock
and the variational methods carry the exact one-center Coulomb tensor, the frozen
core's exchange and signed compensation factors inside each augmentation sphere,
so their electron–electron interaction is exact there.

(paw-lcao-options)=
## Options

The basis dictionary accepts these keys; any other key, or an all-electron
option such as `tier`, is refused before an integral is computed.

| Key | Default | Meaning |
| --- | --- | --- |
| `size` | `"SZ"` | Multiple-zeta and polarization hierarchy, same names as [the NAO sizes](#basis-sizes) |
| `zeta_split` | `"first_zeta"` | Split-valence scheme ([owner](#basis-zeta-split)) |
| `energy_shift` | `0.1` (eV) | Confinement of the first zeta; `None`, `False` or `0` keeps the free-atom orbital |
| `confinement_potential` | `(12.0, 0.6)` | Amplitude (Hartree) and inner radius (fraction of the cutoff) of the smooth confining potential |
| `polarization_method` | `"gaussian"` when confined, else `"orbital"` | Polarization shell: a quasi-Gaussian or the raised outermost orbital |
| `diffuse` | off | [Diffuse shells](#paw-lcao-diffuse) |
| `filter_cutoff` | `"auto"` | Fourier filter of the radial functions |
| `projector_basis` | `"raw"` | `"raw"` or `"dual"` projector set; the energies are identical, only the resolution report differs |
| `paw_dataset` | the functional's set | Dataset folder inside the library ([below](#paw-lcao-libraries)) |

`energy_shift` is in eV, like the NAO option of the same name, but its default
differs: a plain `basis="PAW-LCAO"` is a confined basis (0.1 eV), while the NAO
default is 0.03 eV. The confined orbital is the lowest solution of the dataset's
own radial problem with a smooth confining potential added; its cutoff radius is
the one at which the eigenvalue lies `energy_shift` above the free one. The
projectors, couplings and local potential still come from the dataset. The extra
zetas and the polarization shell are split from the confined orbital, so a whole
atom shares its range.

`filter_cutoff` removes from each radial function the wave-vectors the grid
cannot represent, which would otherwise alias and make the energy depend on where
a nucleus sits between nodes. `"auto"` (or `True`) ties the cutoff to the grid
spacing, a positive number is a kinetic-energy cutoff in eV, and `False` keeps
the raw basis. Filtering changes the basis, so it changes the numbers; compare
against `False` when you need the unfiltered result.

(paw-lcao-diffuse)=
## Diffuse functions

No `energy_shift` makes a PAW-LCAO basis diffuse: the first zeta is at most the
free atom's orbital. An anion, a weak complex or a response property needs the
density further out. `diffuse` adds quasi-Gaussian shells and leaves the ordinary
basis unchanged:

```python
basis = {"name": "PAW-LCAO", "size": "DZP", "diffuse": True}
```

`True` adds one shell for every angular momentum of the atom's ordinary basis,
up to d; an integer is that many shells per channel; a dictionary sets any of

| Key | Meaning | Default |
| --- | --- | --- |
| `shells` | `"present"`, a string of letters such as `"sp"`, or a list | `"present"` |
| `count` | shells per channel | 1 |
| `rms_ratio` | rms radius of the first shell over the free-atom valence rms radius | 2.5 |
| `ratio` | factor between successive rms radii when `count > 1` | $\sqrt{3.5}$ |
| `rms_radius` | rms radius of the first shell in Å, instead of `rms_ratio` | none |
| `cutoff_norm` | squared norm of the Gaussian tail beyond the support | $10^{-4}$ |

A `{symbol: spec}` mapping, with an optional `"*"` default, gives elements
different recipes; an element that does not write `diffuse` gets none. Each shell
is Fourier-filtered and Gram–Schmidt orthogonalized on its atom against that
atom's functions of the same angular momentum, and dropped if almost nothing
remains.

Each diffuse function is one more spatial orbital, so two more qubits for a
quantum method; the cost is paid before any active space and is not reduced by
it. The cell must hold the shells: a shell cut by the box is a different
function, and the energy is wrong rather than merely unconverged. Allow several
ångström of vacuum more than for the ordinary basis, and the most for
lithium. Diffuse functions that become the lowest virtual orbitals defeat a
truncation ranked by orbital energy; rank by MP2 instead ([active
spaces](active_space.md)). A periodic geometry with `diffuse` is refused.

(paw-lcao-libraries)=
## The dataset libraries

The datasets live in the `mandacaru-paw` repository, one folder per set, and
`MANDACARU_PAW_PATH` names the checkout (see [Installation](../installation.md)).
An optional `mandacaru-upaw` checkout serves `"UPAW-LCAO"` through
`MANDACARU_UPAW_PATH`, set with `mandacaru-paw --set-upaw DIR`.

| Folder | Set |
| --- | --- |
| `lda-sr/` | scalar-relativistic LDA, the default |
| `pbe-sr/` | scalar-relativistic PBE, read by default for `method="dft"` with PBE, HSE06 or r²SCAN |
| `lda-fr/`, `pbe-fr/` | LDA and PBE with the spin-orbit term |

A calculation reads the set of its functional unless the calculator names another
folder with `paw_dataset=`, the name of one folder inside the checkout, not a
path:

```python
Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "DZP"},
          paw_dataset="lda-sr")
```

Running a Kohn–Sham functional on the other functional's set works but warns
about the mismatch. `paw_dataset=` with a basis that reads no
library is refused. A missing folder or an unset variable raises
`LibraryPathError`, naming the `mandacaru-paw --set` command that fixes it. The
spin-orbit sets serve periodic two-component calculations; a molecule on them
raises (above).

### Dataset files

Each element is one Parquet file (`<symbol>.parquet`; a `.json` file of the same
content also loads). The radial tables are float64 columns and the scalars are one
JSON entry in the file's key/value metadata. The native engine reads and writes
the file and no Python Parquet package is involved, so `save_pseudopotential` and
`load_pseudopotential` take no `engine=` keyword (passing one raises `TypeError`)
and the compression is always zstd. The read side is the one described for
[Hamiltonian caches](hamiltonian_cache.md): a null raises `ValueError`, where the
previous Python reader read it as NaN, and a GZIP, Brotli or LZ4 file raises with
the advice to re-save it with zstd. A NaN in a table is now written as a real
NaN, not as a null. A Parquet save is atomic: it is staged beside the destination
and renamed, so a failed save leaves an existing file as it was. A JSON save is a
plain write and is not.

A library dataset is read with a context of its own, with the default thread
count and memory budget, so the `threads` and `memory_limit_bytes` of a
calculator's `resources` do not limit the decoding of its dataset files.

The files already in a library were written by the previous writer and were not
regenerated; they load unchanged. Files saved by this build use the native
writer, whose zstd works at its fastest level, so a regenerated file can be larger
than the one it replaces.

## Why the family is the basis

A dataset fixes its own first zeta: the construction pseudizes each valence
orbital inside its cutoff and builds the projectors from those pseudo-orbitals.
Pairing the potential with an unrelated all-electron radial function would be
inconsistent, which is why the family is the basis name and cannot be combined
with an all-electron family. What you can vary is the size hierarchy, which
refines the pseudo-orbital instead of replacing it.

A calculation only reads finished datasets; how a dataset is generated is a
development topic and is not covered in this manual.
