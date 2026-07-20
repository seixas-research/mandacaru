# Caching the Hamiltonian (Parquet / JSON)

Building a molecular Hamiltonian is by far the most expensive stage of a Carcará
run: the real-space one- and two-body integrals scale as $O(N \log N)$ and
$O(M^4)$ in the grid and basis size, and the fermion-to-qubit mapping then
composes a Pauli sum per fermionic term.

None of that depends on the *algorithm* you run afterwards. So the qubit
Hamiltonian is worth caching: build it once, then sweep operator pools,
optimizers, ansätze, mappings or temperature schedules essentially for free.

```python
from ase import Atoms
from carcara.algorithms import ADAPTVQE

atoms = Atoms("LiH", positions=[[7.5, 7.5, 6.7], [7.5, 7.5, 8.3]],
              cell=[[15, 0, 0], [0, 15, 0], [0, 0, 15]], pbc=True)

# 1. Build once and save.
atoms.calc = ADAPTVQE(pool="qeb", basis="FAO", h=0.25,
                      save_hamiltonian="lih.parquet")
atoms.get_total_energy()

# 2. Reload -- no geometry, no integrals, no mapping.
driver = ADAPTVQE(pool="ceo", load_hamiltonian="lih.parquet")
result = driver.run()
```

The second run needs **no `Atoms` object at all**. The file records
`num_particles` and `n_spatial_orbitals` alongside the operator, which is exactly
what the ADAPT pool and the UCCSD ansatz are built from, so a loaded driver is in
*direct mode* immediately.

:::{admonition} What is bypassed
:class: note

Loading a cached Hamiltonian skips **the integral engine and the
fermion-to-qubit transformation of the Hamiltonian**. On a small LiH problem this
turns a ~0.8 s build into a ~0.05 s load.

The *pool's own* excitation generators are still mapped to qubits — those are a
different operator, unrelated to the Hamiltonian, and are cheap to build.
:::

---

## Choosing a format

Two interchangeable formats are selected with `hamiltonian_format`:

| | `"parquet"` (default) | `"json"` |
|---|---|---|
| Layout | Compressed columnar table | Plain-text object |
| Size (LiH, 118 terms) | 1.9 KiB | 8.0 KiB |
| Dependencies | a Parquet engine | none (standard library) |
| Best for | large active spaces; analysis in pandas/Arrow/Spark | inspection, diffing, dependency-free environments |

```python
ADAPTVQE(basis="FAO", save_hamiltonian="lih", hamiltonian_format="json")
# -> writes lih.json
```

The extension is appended automatically from the format when the path has none.
`save_hamiltonian=True` writes `hamiltonian.parquet` (or `hamiltonian.json`).

### Format detection on load

`load_hamiltonian` **detects the format automatically** — first from the file
extension, and failing that from the file's own leading bytes (a Parquet file
starts with the `PAR1` magic number; a JSON document with `{`). A cache written
either way loads through the same call, even if the file is named `.cache`:

```python
from carcara.core import detect_format, load_hamiltonian

detect_format("lih.parquet")     # 'parquet'
detect_format("opaque.bin")      # 'parquet' or 'json', read from the bytes
record = load_hamiltonian("opaque.bin")
```

---

## File layout

### Parquet

One row per Pauli term, three columns:

| column | type | meaning |
|---|---|---|
| `pauli` | string | the Pauli string; character `k` acts on qubit `k` |
| `real` | double | real part of the coefficient |
| `imag` | double | imaginary part of the coefficient |

Rows are sorted by Pauli string, so an identical Hamiltonian produces an
identical file. The problem metadata (`mapping`, `num_particles`,
`n_spatial_orbitals`, `num_qubits`) travels in the Parquet key/value metadata, so
one file fully specifies the problem. Being ordinary Parquet, it is directly
queryable:

```python
import pandas as pd

terms = pd.read_parquet("lih.parquet")
terms["weight"] = terms["real"].abs() + terms["imag"].abs()
print(terms.nlargest(5, "weight"))
```

### JSON

The same content as one object: `format`, `version`, `num_qubits`, `mapping`,
`num_particles`, `n_spatial_orbitals`, `metadata`, and `terms` as a list of
`[pauli_string, real, imag]` triples.

---

## Working with the record directly

```python
from carcara.core import load_hamiltonian, save_hamiltonian

record = load_hamiltonian("lih.parquet")
record.hamiltonian          # a PauliSum
record.mapping              # 'jordan_wigner'
record.num_particles        # (2, 2)
record.n_spatial_orbitals   # 3
record.num_qubits           # 6
record.metadata             # {'driver': 'ADAPTVQE', 'basis': {...}, ...}

save_hamiltonian("copy.json", record.hamiltonian, mapping=record.mapping,
                 num_particles=record.num_particles,
                 n_spatial_orbitals=record.n_spatial_orbitals, format="json")
```

---

## Parquet engines

Parquet is read and written through either **`fastparquet`** (the default) or
**`pyarrow`**, selected with `engine=`. Both produce standard Parquet, so files
are portable between them and to any other Parquet reader.

:::{warning}
`fastparquet` leads **on purpose**. On some platforms — reproduced on CPython
3.14 with `qiskit` 2.5 and `pyarrow` 25 — calling
`pyarrow.parquet.write_table` in a process that has *also* run Qiskit's
`transpile` crashes the interpreter, since both ship their own native runtimes.
Carcará transpiles circuits for gate-count profiling in the same process that
saves the Hamiltonian, so the default engine avoids that combination, and
`resolve_engine("auto")` stops at the first engine that imports rather than
importing them all.

If your environment is unaffected, pass `engine="pyarrow"` explicitly. Or
side-step Parquet entirely with `hamiltonian_format="json"`, which has no native
dependency and cannot be affected.
:::

---

## A worked sweep

Caching turns a pool comparison into a few seconds of work:

```python
from carcara.algorithms import ADAPTVQE

# Build once ...
atoms.calc = ADAPTVQE(basis="FAO", h=0.25, save_hamiltonian="lih.parquet",
                      max_iterations=1, verbose=False)
atoms.get_total_energy()

# ... compare every pool against the *same* operator.
for pool in ("fermionic", "qubit", "qeb", "ceo"):
    result = ADAPTVQE(pool=pool, load_hamiltonian="lih.parquet",
                      verbose=False).run()
    print(f"{pool:<10} E = {result.optimal_energy:.8f} Ha  "
          f"{result.num_operators} ops  {result.metrics.cnot_count} CNOTs")
```

See `examples/17_hamiltonian_cache.py` for a runnable version that checks the
round trip in both formats, and `examples/12_ADAPTVQE_LiH_backends.py` for the
cache driving a multi-backend comparison.
