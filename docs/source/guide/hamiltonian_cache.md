# Caching the Hamiltonian

Building the molecular Hamiltonian is the expensive stage of a run: the
real-space integrals, the SCF and the fermion-to-qubit mapping. None of it
depends on the algorithm that runs afterwards, so the qubit Hamiltonian is worth
caching. Build it once, then sweep operator pools, optimizers, ansätze or
search settings against the same operator.

```python
import os
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)
path = os.path.join("data", "lih_hamiltonian.parquet")
os.makedirs("data", exist_ok=True)

# 1. Build once and save.
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25,
                       save_hamiltonian=path)
print(f"built   E = {atoms.get_potential_energy():.6f} eV")

# 2. Reload: no geometry, no integrals, no mapping.
for pool in ("fermionic", "qubit", "qeb", "ceo"):
    result = Mandacaru(method="adapt-vqe", pool=pool,
                       load_hamiltonian=path).run()
    print(f"{pool:10s} E = {result.optimal_energy:.6f} eV  "
          f"({result.num_operators} operators)")
```

The second calculator needs **no `Atoms` object**. The file records the mapping,
the particle counts and the spatial orbital count with the operator, which is
what the pool and the ansatz are built from, so a loaded calculator is a
complete problem. `examples/hamiltonian_cache/01_save_and_load_lih.py` runs the
same sweep and also saves the JSON form.

Loading skips the integral engine, the SCF and the mapping of the
*Hamiltonian*. The pool's own excitation generators are still mapped to qubits;
they are a different operator and cheap to build.

## Formats

`hamiltonian_format` is `"parquet"`, `"json"` or `None`.

* `"parquet"`, or `None` with a path ending in `.parquet` or `.pq`: a compressed
  columnar table with one row per Pauli term (`pauli`, `real`, `imag`) and the
  problem metadata in the file's key/value metadata.
* `"json"`, or `None` with any other path: the native text snapshot, the same
  content as plain JSON.

`load_hamiltonian` detects the format from the file itself: a Parquet file starts
with the `PAR1` magic number, anything else is read as the native snapshot. A
cache written either way loads through the same call, whatever the file is
called. `mandacaru.core.detect_format(path)` reports which one a file is.

## The Parquet file

The native engine reads and writes the file. Mandacaru has no Python Parquet
dependency: it declares and imports none of `fastparquet`, `pandas` or
`pyarrow`, and there is no engine to choose. The `engine=` and `codec=` keywords
of earlier releases are gone, with no alias, so passing either to
`save_hamiltonian` or `load_hamiltonian` raises `TypeError`; the compression is
the `compression=` keyword below. The rows are sorted by Pauli label, so saving
the same Hamiltonian twice writes the same bytes. The file is a standard Parquet
file: any Parquet reader opens it, with `pauli` as a string column and `real` and
`imag` as float64 columns.

* **What a write produces.** One layout: DATA_PAGE v1, REQUIRED columns (no
  definition levels), PLAIN encoding, no dictionary, a CRC32 on every page, and
  no `pandas` key in the metadata. Rows go in row groups of at most 1,048,576
  rows and pages of at most 65,536 rows. The key/value entries that hold JSON
  (the particle counts, the orbital count and the free-form metadata) are
  compact text, so a particle count reads `[1,1]`, not `[1, 1]`; a reader that
  parses the JSON is unaffected.
* **Codec.** `zstd` (the default) or `uncompressed`; `none` is accepted as a
  synonym for `uncompressed`. Any other name raises `ValueError`; Snappy files
  can be read but not written. The zstd encoder works at its fastest level, the
  only one the engine implements, so a file can be larger than one another tool
  wrote with a higher zstd level.
* **Atomic writes.** The file is staged next to its destination and renamed, so a
  failed save leaves an existing cache untouched.
* **What a read accepts.** Flat tables of float64 and UTF-8 string columns, with
  data pages of version 1 or 2, PLAIN or dictionary encoding (dictionary pages
  included), UNCOMPRESSED, Snappy or ZSTD pages and any number of row groups or
  pages, so a cache written by another Parquet tool still loads. A ZSTD page
  must be one frame that records its content size and has neither a dictionary
  nor a checksum. Anything else raises `ValueError` naming the unsupported
  feature: null values (a Hamiltonian has none, so a null is corruption, not a
  missing term, and the Python reader of earlier releases turned nulls into NaN
  instead of raising), nested columns, text columns annotated as something
  other than text (for example DECIMAL or BSON), more than 65,536 columns, other
  compression codecs such as GZIP, Brotli or LZ4 (the message says to re-save the
  file with zstd), encrypted files and files whose declared sizes do not match
  their contents. A table with zero rows loads whichever way the writing tool
  laid out its empty chunks.
* **Validation.** A loaded cache passes the same checks as a JSON snapshot: at
  most 50 qubits, finite coefficients, Pauli labels of the register width and
  the format version. The format, version and qubit-count entries of the
  key/value metadata are required: a file without the format entry is not a
  Mandacaru Hamiltonian, and a missing version or qubit count raises
  `ValueError`, as does format version 0 or a version newer than 1. A missing
  mapping entry defaults to `jordan_wigner`; a missing particle count, orbital
  count or metadata entry loads as none, none and an empty dictionary. A cache
  with no terms still keeps its register width.
* **Memory.** A read charges the context's memory budget before it allocates, so a
  file that claims more data than the budget allows raises `MemoryError` instead
  of exhausting the machine. The Python objects a read returns (the `str` of a
  text column, the metadata dictionary) count against the budget too; float64
  columns move into their arrays without a copy. Columns are decoded in parallel,
  on at most as many threads as the context has and as many as its budget can
  hold buffers for, and the compression codec runs on a thread of its own, so a
  caller on a small thread stack cannot be killed by it.

## What a cached run does not have

A calculator built from a cache never saw a molecule. It has the Hamiltonian
and nothing else, so everything that needs the molecular reference is
unavailable: forces, densities, cube files, natural orbitals and
`result.hamiltonian`. Rebuild from the geometry for those.

* If `mapping=` is given and disagrees with the cache, the run raises
  `ValueError`; without it the cache's mapping is used.
* `load_hamiltonian` cannot be combined with `hamiltonian=`, `native_model=` or
  `load_crystal=`: supply one source of the problem.
* Rebuild the cache after any change of geometry, basis, grid, charge, spin or
  active space. A cache belongs to one electronic problem.

## Building from a Hamiltonian you supply

`hamiltonian=` accepts a `Fermion` or a `PauliSum` (both in `mandacaru.core`) or
a native Pauli operator, and needs `n_spatial_orbitals` and `num_particles` with
it.
It is how `result.as_quantum_problem()` hands a mean-field Hamiltonian to a
quantum method; see [mean field and HVA](mean_field_hva.md).
