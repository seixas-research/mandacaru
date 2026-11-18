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
correction; it is optional and reads its own library, which currently has no
format-3 data ([the dataset libraries](#paw-lcao-libraries)).

The dataset gives the basis, the local potential, the projectors and the
compensation charges. Oxygen keeps its six valence electrons. The engine builds
the basis shells from the dataset (the confined, split-valence, polarization and
diffuse shells of the options below), samples the orbitals, and builds the
integrals, the augmentation and the SCF, all in Rust; Python validates the
options and hands the dataset and the basis recipe to it.

## What is supported

| | |
| --- | --- |
| Methods | `rhf`, `uhf`, `ghf`, `dft` (`lda`, `pbe`, `r2scan` and the screened hybrid `hse06`), and the variational methods (`vqe`, `adapt-vqe`, `vasqa`, `valsqa`) |
| Frozen-core exchange (Hartree–Fock, screened hybrids, variational) | an empty core or any core of closed subshells whose `core_states` the dataset stores (`1s 2s 2p` for Si and Cu, closed `1s` for Li/Na) |
| Forces | RHF, UHF and Kohn–Sham, `hse06` included ([Forces](forces_relaxation.md)) |
| Spin-orbit datasets | a molecule on `lda-fr` or `pbe-fr` runs spinor Kohn–Sham with `method="dft"` (supporting `lda`, `pbe`, `pbe+rvv10`, `r2scan`, `hse06`, and analytic forces) or spinor variational methods (`ghf`, `vqe`, `adapt-vqe`, `vasqa`, `valsqa`) via a spinor `ghf` reference |
| Geometry | molecules; a crystal reads its own datasets through the periodic setup |

The following raise, or are not yet available ([open items](#not-yet-available)):
a Hartree–Fock or screened-hybrid (`hse06`) frozen core with a partially filled
subshell or missing/inconsistent `core_states` (`ValueError`, naming the element);
`rhf`, `uhf`, or DFT+U on a molecule with a spin-orbit dataset (`NotImplementedError`);
molecular stress; `precision="float32"` forces of the quantum methods (the
Hartree–Fock and Kohn–Sham forces take `float32`, see [Forces](forces_relaxation.md)).

The two families of one-center terms differ by method (see
[the PAW-LCAO formulation note](../development/paw_formulation.md) for equations
and audits). **Kohn–Sham** uses the Blöchl frozen-augmentation scheme linearized
around the reference atom's density matrix $D^0$ (`ks_difference` plus
full nonlinear compensation multipoles), while **Hartree–Fock and variational
quantum methods** evaluate the exact one-center Coulomb tensor $\Delta W$ and
exact core–valence exchange from `core_states` inside each augmentation sphere
(keeping the one-center XC offset frozen at $D^0$). When `nlcc` is active, the
grid XC terms evaluate on $\tilde{n} + \tilde{n}_c$ (`smooth_core`) to match
the one-center subtraction.

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
lithium.

The calculator checks it. After the integrals are built it samples every diffuse
function on the real-space grid, and a function that keeps less than
$1 - 10^{-4}$ of its radial norm on the grid counts as cut. Any cut function
issues a `RuntimeWarning` that begins "the cell cuts", gives how many were cut
and names the worst (the fraction of its norm that the grid keeps, its function
index and its atom); the calculation then goes on, so enlarge the vacuum until
the warning is gone. The check is also in the run log: the `[BASIS]` block of a
molecule with diffuse shells (`"PAW-LCAO"` or `"UPAW-LCAO"`) lists every shell and, for each atom, the largest
accepted cutoff radius against the distance from the atom to the nearest face of
the cell ([the diffuse tables of `[BASIS]`](#run-log-basis-diffuse)). A basis
without diffuse shells is not checked, and the check covers only the shells the
`diffuse` option adds.

Diffuse functions are among the lowest virtual orbitals, so a truncation ranked
by orbital energy would fill the register with them. An `active_space` that
ranks by `"energy"` (the default) and keeps fewer virtual orbitals than the basis
has raises `ValueError`, naming the number of diffuse functions and the virtual
orbitals kept. Rank by MP2 instead (`"method": "mp2"`, [active
spaces](active_space.md)), or keep every virtual orbital. A periodic geometry
with `diffuse` is refused.

(paw-lcao-libraries)=
## The dataset libraries

The datasets live in the `mandacaru-paw` repository, one folder per set, and
`MANDACARU_PAW_PATH` names the checkout (see [Installation](../installation.md)).
An optional `mandacaru-upaw` checkout serves `"UPAW-LCAO"` through
`MANDACARU_UPAW_PATH`, set with `mandacaru-paw --set-upaw DIR`. That library
currently has no format-3 data, the only format the reader takes ([below](#paw-lcao-format-versions)).
Where it has no file for an element, whether the variable is unset or its folder lacks
the file, `"UPAW-LCAO"` generates that element's dataset on demand: a `RuntimeWarning`
says so, it takes a few seconds per element, and the dataset is cached for the
process. A `paw_dataset=` folder that lacks the element raises `FileNotFoundError`
instead.

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
`LibraryPathError`, naming the `mandacaru-paw --set` command that fixes it. The spin-orbit sets serve two-component (spinor) calculations: Kohn–Sham on a
crystal or molecule (`method="dft"`), as well as molecular `ghf` and variational
quantum methods (`vqe`, `adapt-vqe`, `vasqa`, `valsqa`); `rhf` and `uhf` on a
molecule raise (above).

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
count and memory budget, so the `parallelization` and the `memory_limit_bytes` of a
calculator do not limit the decoding of its dataset files.

The installed library is regenerated. Every file is format 3 ([below](#paw-lcao-format-versions)),
written by the native generator with the core states of its element and, in `lda-fr/` and
`pbe-fr/`, with the spin-orbit coupling calibrated during generation: the generator
scales each shell's stored coupling until the engine's isolated atom reproduces the Dirac
splitting of the record, a factor derived from the atom and not tabulated. A file saved by
this build uses the native writer, whose zstd works at its fastest level, so a file can be
larger than one compressed at a higher level.

All 368 files of the library (92 elements in each of the four sets) pass the native
validator, including its gate G5 row "format version 3 with the core states". The reader
itself refuses a file of an older format, or without its core states, by name.

(paw-lcao-format-versions)=
#### Format 3 and core states

Format 3 is the only dataset format. A file carries, besides the valence data, the
**core states**: the occupied core subshells of the reference atom, each as `n`, `l`, its
occupation, its eigenvalue and its radial orbital `u = r R` in a column
`core_wave_n{n}_l{l}`, one state per `(n, l)` (at most 28; a hydrogen file has an empty
core). They are the orbitals that the stored core density was summed from. A file of an
older format is refused by name, and so are a file without its states, a malformed or
duplicated state, and more core electrons than the nuclear charge.

The development tool `mandacaru-build` (`mandacaru.pseudopotentials.build_cli`, not an
installed console command; generating a dataset is otherwise outside this manual)
generates, writes, reads back and checks a dataset inside the native engine
using the format-3 **Variational** radial XC scheme (`models::spherical`).
Unitary builds fix zero norm deficit (`norm_deficit = 0 (unitary)`, which is
also the `UPAW-LCAO` convention), and in `mode="refuse"` or `"enforce"` any
search that cannot clear ghost, node, or phase gates raises `GhostStateError`
rather than emitting a defective record. Records never cross into Python on
their way to the file.

Hartree–Fock and screened hybrids (`hse06`) read the core states: the exchange
of the frozen core with the valence orbitals comes from the stored states, for
an empty core or any core of closed subshells (`1s 2s 2p` for Si and Cu). A
partially filled core subshell, or states that do not reproduce the stored core
density to `1e-7`, leave Hartree–Fock and screened hybrids without a valid core
operator and raise `ValueError` naming the element. Semilocal Kohn–Sham never
reads `core_states` ([native cutover](../development/native_cutover.md)).

## Why the family is the basis

A dataset fixes its own first zeta: the construction pseudizes each valence
orbital inside its cutoff and builds the projectors from those pseudo-orbitals.
Pairing the potential with an unrelated all-electron radial function would be
inconsistent, which is why the family is the basis name and cannot be combined
with an all-electron family. What you can vary is the size hierarchy, which
refines the pseudo-orbital instead of replacing it.

A calculation only reads finished datasets; how a dataset is generated is a
development topic and is not covered in this manual.
