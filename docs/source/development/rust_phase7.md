# Rust migration: Phase 7 persistence and execution boundary

This milestone adds native bounded Hamiltonian/checkpoint JSON handling,
checkpoint reconstruction and warm restarts, streaming volumetric output,
measurement grouping/count reductions, provider-independent circuit compilation
and a checked submission ledger. These components run in the Python-independent
core. The broad Phase 7 capability groups remain partially migrated; remaining
codecs and execution work are listed below. Python remains the default engine.

## Native checkpoints and restarts

```python
from mandacaru import Mandacaru

options = dict(method="adapt-vqe", engine="rust",
               load_hamiltonian="h2_hamiltonian.json", trace=False)
result = Mandacaru(**options, checkpoint="state.json").run()
continued = Mandacaru(**options, resume="state.json",
                     max_iterations=50).run()
```

Both native VQE and fermionic ADAPT expose `calc.solver.checkpoint` after a
solve and accept `checkpoint=<path>` and `resume=<path>`. Checkpoints use the
existing schema v2, Hartree energies, MSB qubit ordering, embedded Hamiltonian,
ordered generators/parameters and explicit `product` or `sum` preparation.
Readers also accept schema v1's implicit product preparation. Existing Python
consumers can read native files, and native solvers can restart from compatible
Python files. Unknown metadata and status fields survive document round trips.

These are **final-solve snapshots and warm restarts**. They do not checkpoint
inside an optimizer iteration or restore a BFGS inverse Hessian. Native
`checkpoint_every` is not implemented. A restart reuses the ansatz and angles,
recomputes energy/convergence, and starts a fresh optimizer; histories and
evaluation counts cover the new solve. ADAPT's growth limit is the total
operator count, including restored operators. Subspace checkpoint/restart is
still explicitly refused.

Before a restart, Rust checks mapping, register, particles/orbitals, reference
determinant, Hamiltonian coefficients (absolute tolerance `1e-10`), method,
preparation and generator sequence. Fixed UCCSD needs its full ordered pool;
ADAPT restores its selected sequence. A different Hamiltonian is refused rather
than interpreted as a continuation at a new geometry.

Standalone checkpoint consumers also select the engine explicitly:

```python
from mandacaru.core.checkpoint import WavefunctionCheckpoint

state = WavefunctionCheckpoint.load("state.json", engine="rust")
psi = state.state_vector(engine="rust")
energy_ha = state.expectation(engine="rust")
state.save("copy.json", engine="rust")
```

`prepare_state(..., engine="rust")` supports ordered products and exact summed
exponentials, including noncommuting generators. Native QPE and time propagation
now reconstruct checkpoint inputs in Rust; QPE preparation tuples also use the
native state preparer. This supersedes Phase 6's tuple-preparation restriction.
Generic circuit compilation still refuses a generator whose Pauli terms
anticommute: its exact exponential cannot be represented by that rotation
product.

## Snapshots and the Parquet adapter

Native JSON parsing checks schema tags/versions, finite values, Pauli labels,
operator widths, supported mapping, available occupation metadata, Hartree
units, references and anti-Hermitian generators. Inputs are limited to 64 MiB
and 50 file qubits, consistent with the public file-width contract. Context
reservations conservatively charge 32 times the JSON input size for parsing
and retained nodes, in addition to operators and numerical work. The context
budget may therefore refuse an input below the file-size ceiling. Unknown
metadata is retained without treating it as executable instructions.

Writes create a unique staging file beside the destination, serialize and
flush/sync it, and rename it into place. Errors remove staging files and retain
the old destination. The native molecular model's Hamiltonian exporter now
uses this same atomic path. This promises atomic replacement, not a new
cross-process locking protocol.

`save_hamiltonian(..., engine="rust")` and
`load_hamiltonian(..., engine="rust")` support the public record view. JSON
parsing/validation/writing runs in Rust. Parquet uses an **explicit Python
columnar-I/O adapter** with the existing fastparquet/pyarrow selection; the
decoded schema is checked in Rust. Native calculators now accept Parquet caches
and record `timings["cache_engine"] == "python-parquet-adapter"`; JSON caches
record `"native-json"`. The solve never calls a Python numerical engine.
This does not implement a Rust Parquet codec or change the default Parquet
engine/import policy.

For Python-independent consumers, `persistence::Document` reads, validates,
writes and reconstructs snapshots directly. `Document::problem` builds a cached
quantum problem, and `Document::resume` restores a compatible solver state.
The Python `Context.read_snapshot`/`parse_snapshot` methods expose a frozen
`Snapshot` with independent output arrays.

## Provider-independent execution work

```python
from mandacaru.backends.providers import build_provider

provider = build_provider("qiskit", engine="rust",
                          engine_options={"threads": 1})
# The same engine option works with "braket" and "cirq".
```

Rust emits the shared X/H/S/S-dagger/CNOT/RZ gate stream; SDK adapters build
their own circuit objects from it. Terms retain the existing lexical ordering,
angles/signs, parity ladders and commuting-generator checks. Identity phases
are omitted in circuits, matching the existing SDK contract; checkpoint state
reconstruction retains them. `with_options` preserves the selected native
context. SDK simulation, transpilation, service credentials and job submission
remain in the existing Python integrations. No remote submission is required
by the migration tests.

`qwc_partition`, `qubit_wise_commuting_groups`, `chunk_labels_by_basis`,
`planned_jobs`, `pauli_expectation_from_counts` and `energy_from_group_counts`
accept `engine="rust"`. Grouping uses the same stable greedy first-fit policy:
descending weight, then lexical label order. It supports widths beyond one
machine word without allocating a state vector. The selected provider routes
its shared grouping/count work through these paths. Native count validation
requires matching binary strings and finite nonnegative frequencies.

`Context.run_ledger(jobs=..., total_shots=...)` supplies an independent native
quota primitive. `reserve(jobs, shots)` returns tickets and charges the whole
batch atomically before submission; rejected reservations change no counters.
`finish(ticket, success)` records completion or failure exactly once. Failed
jobs stay charged. History storage is context-budgeted, and the Python wrapper
serializes concurrent reservations/completions. This ledger is available to
orchestrators; it does not replace the existing SDK adapters' run ledger or
infer terminal job status from remote services.

## Streaming scientific output

`write_cube`, `write_xsf`, `write_volumetric` and `write_bxsf` accept
`engine="rust"` and `engine_options`. All write through the atomic snapshot
path, with finite real inputs and checked dimensions.

- CUBE preserves Bohr coordinates, column-oriented step vectors, z-fast data,
  six values per line and the existing five-decimal scientific format.
- XSF converts coordinates to Angstrom, emits x-fast general-grid data and
  adds periodic closing planes only while streaming. Cell/grid spans must
  agree. Values retain the existing six-decimal precision.
- BXSF writes eV energies, reciprocal vectors as rows, a C-ordered closed
  periodic grid and one block per band. Closing planes repeat the first plane
  without allocating a padded tensor.

Input arrays are copied into owned native buffers before releasing Python.
Writing uses a bounded stream buffer in addition to that snapshot, rather than
Python line lists or a second padded grid. ASE independently reads native CUBE
and XSF output; the existing BXSF reader checks band-grid round trips.

## Evidence and remaining work

`test/migration/test_native_persistence.py` checks cross-engine formats/states,
public restarts and rejection paths, native callback exclusion, all three SDK
gate streams, wide QWC groups, count reductions, concurrent ticket accounting,
atomic errors, memory cleanup and independent volumetric readers. Standalone
Rust tests in `crates/core/tests/persistence.rs` exercise these components
without Python or quantum SDKs. Package smoke tests cover checkpoint restart,
Parquet cache loading, grouping, circuit compilation and scientific output with
runtime compilers disabled. Outcomes and hashes are recorded in
`test/migration/phase7/validation.json`.

Remaining Phase 7 work includes Rust Parquet/compression codecs, family-specific
pseudopotential dataset codecs, proposal/training/model persistence, volumetric
readers, in-flight solver checkpointing, full measurement-gradient and
factorization planning, and native integration of SDK job lifecycle/accounting.
Structured report/citation formatting remains in the Python presentation layer.
Earlier phases' outstanding scientific algorithms remain outstanding; this
milestone does not assert full migration of either broad Phase 7 capability
group.
