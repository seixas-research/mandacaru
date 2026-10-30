(native-calculations)=
# Native calculations

Rust is Mandacaru's only calculation implementation. Python supplies immutable
geometry values, option validation, result views and the ASE interface. There is
no execution selector: a calculation runs in the engine or it raises.

## A calculation without ASE

```python
import numpy as np
from mandacaru import Geometry, Mandacaru

geometry = Geometry(
    numbers=(1, 1),
    positions=np.array([[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]]),
    cell=np.eye(3) * 5,
)
with Mandacaru(
    method="dft", xc="pbe", basis="HAO", grid_spacing=.35,
    resources={"threads": 2, "memory_limit_bytes": 512 * 1024**2},
) as calculation:
    result = calculation.run(geometry)
    print(result.optimal_energy)  # eV, including nuclear repulsion
    print(result.forces())       # eV / angstrom
```

Positions and cells default to ångströms; `Geometry(..., units="bohr")` accepts
atomic units. Geometry copies its input arrays. `atomic_units=True` on the
calculator selects Hartree for reported energies. Native SCF and optimizer
options use atomic units. Result forces and the ASE protocol always use eV/Å.
For geometry work use a PAW-LCAO basis ([Forces](forces_relaxation.md)).

## With ASE

`Mandacaru` is an ASE calculator, and ASE is a required dependency:

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[2.5,2.5,2.13], [2.5,2.5,2.87]], cell=[5]*3)
atoms.calc = Mandacaru(method="dft", xc="pbe", grid_spacing=.35)
energy = atoms.get_potential_energy()
atoms.calc.close()
```

A standalone `Geometry` can also be passed to `run`. A saved crystal
(`load_crystal=`) already contains the data its Rust solver needs; building a
crystal from atoms uses a Python table adapter for the PAW setup.

## Hamiltonians are explicit

Ordinary SCF, energy, gradient and snapshot operations do not create a dense
four-index Hamiltonian. Semilocal DFT uses density fields directly. HF exchange
uses pivoted Coulomb Cholesky factors built from selected Poisson columns; HSE06
uses screened factors. `cholesky_tolerance` defaults to 1e-10 and controls the
maximum residual Coulomb diagonal in atomic units. For molecular PAW, signed
one-center and compensation factors remain separate from the smooth Cholesky
factors, and their negative contributions are preserved.

`result.hamiltonian`, `result.save_hamiltonian(path)` and the quantum solvers
request the transformed Hamiltonian explicitly. It is cached in its originating
native state. Low-level snapshots export four-index terms only with
`include_hamiltonian=True`. Dense export can still require O(M⁴) memory.

## Ownership and resource limits

Each calculator owns a resource object, a context, a state and result
references. Its thread setting never changes a global worker pool or BLAS
setting. Scratch is created lazily in a unique temporary directory, optionally
beneath `resources={"scratch_dir": ...}`, and closing the calculator deletes only
that directory. There is no automatic `.mandacaru/` workspace.

A result can retain its native state after its calculator is closed; native
memory stays allocated while any owner retains it. The memory limit covers native
buffers and reserved workspaces, not the whole Python process. Exceeding it
raises `MemoryError` and releases temporary allocations.

Scientific configuration is fixed when the calculator is constructed. Create a
new calculator to change the method, basis or functional. A supplied native model
retains its original context, so configure that context at its origin.

(native-coverage)=
## Coverage

| Calculation | Native support | Limits |
| --- | --- | --- |
| Molecular mean field | RHF, UHF, GHF | Gradients need real collinear orbitals and spectral kinetic energy; none for GHF |
| Molecular PAW | HF, semilocal Kohn–Sham and variational methods, exact valence one-center terms, RHF/UHF/KS forces, smooth fields and augmentation-charge accounting | Frozen-core exchange for an empty or single closed-1s core only; no PAW screened exchange or spin-orbit projectors; no PAW forces for GHF, screened exchange or nonlocal correlation |
| Molecular DFT | `lda`, `pbe`, `r2scan`, `hse06`, `pbe+rvv10`, `r2scan+rvv10` | Forces: analytic for all-electron bases, a fixed-density finite difference for PAW-LCAO ([Forces](forces_relaxation.md)) |
| Periodic DFT | LDA, PBE, r²SCAN and rVV10 pairings, occupations, spectra | Python PAW geometry assembly; no force or stress interface; `hse06` raises |
| Quantum methods | UCCSD and HVA VQE, fermionic/qubit/QEB/CEO/spin-orbit ADAPT, shared subspace searches, MCAS-VQE and VALQA | Other ansätze, the GHF quantum handoff and learned proposals remain unported |
| Correlation | Closed-shell MP2/FNO active spaces, thresholds and frozen cores | Open-shell and correlating-pair constraints remain unported |
| Persistence | Native JSON and Parquet Hamiltonians, native Parquet pseudopotential datasets, crystals and completed checkpoints; the Parquet container, Snappy and ZSTD decoding and ZSTD encoding are in Rust, with no Python Parquet package ([Parquet files](#native-parquet)) | Parquet files with nulls, nested columns or GZIP, Brotli and LZ4 pages raise; provider job lifecycle integration remains incomplete |

Unsupported combinations raise errors. What is planned but not yet available is
collected in [not yet available](#not-yet-available), and the
[audit](../development/native_cutover.md) records the remaining scientific Python
modules. D4 dispersion and a complete exchange–correlation catalog are not
enabled. A checkpoint supplies a compatible warm start; it does not restore an
optimizer Hessian or a partially completed iteration.

(native-parquet)=
## Parquet files

Hamiltonian caches and pseudopotential datasets are Parquet files that
`mandacaru-core` reads and writes itself; Mandacaru neither declares nor imports
`fastparquet`, `pandas` or `pyarrow`. The `engine=` and `codec=` keywords of
earlier releases are gone with no alias (they raise `TypeError`); `compression=`
accepts `zstd`, `uncompressed` and `none`.

- **Reads.** Data pages of version 1 and 2, dictionary pages, and UNCOMPRESSED,
  Snappy and ZSTD pages. Anything else raises `ValueError` naming the feature: a
  null (the previous reader returned NaN), a nested column, a GZIP, Brotli or LZ4
  page (re-save with zstd), an encrypted file, sizes that do not match.
- **Writes.** REQUIRED columns, PLAIN encoding, zstd at its fastest level (the only
  one the engine implements), a CRC32 per page. A file can therefore be larger
  than one written with libzstd at level 3. Parquet saves are atomic.
  A NaN in a pseudopotential table is stored as a real NaN, not as a null.
- **Hamiltonian entries.** The footer's key/value text is compact JSON. A missing
  format, version or qubit count raises, as does format version 0; a missing
  mapping is `jordan_wigner`.
- **Library.** The pseudopotential files already shipped were not regenerated and
  still load; only files saved by this build use the native writer.

[Hamiltonian caching](hamiltonian_cache.md) and
[PAW-LCAO bases](pseudopotentials.md) describe the two file kinds.

## Developer entry points

`mandacaru-core` has no Python dependency. Its `math` module owns grids, FFTs,
reciprocal operators and eigensolvers, and `models` owns the canonical functional
definitions. Molecular, periodic and quantum solvers use those shared layers.
`mandacaru-python` validates arrays and transfers ownership at the PyO3 boundary.

Periodic setup sends small JSON metadata and typed NumPy buffers. Arrays are
copied once into checked Rust allocations before detached work, and returned
snapshots own independent arrays. See [architectural
checks](../development/architecture.md).
