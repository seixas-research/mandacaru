# Reading a run

A variational run reports itself in three places, each with a different job:
**standard output** while it runs, the structured **`output.txt`** log, and two
optional **JSON dumps** for the objects that are too large to print.

## What goes where

`output=<path>` decides the split, on the same principle as GPAW's `txt=`: when
the detail has a destination, standard output is left to the **evolution of
energies and forces**, which is what an ASE optimizer prints there.

| | standard output | `output.txt` |
| :--- | :--- | :--- |
| with `output=<path>` | ASE's `Step Time Energy fmax` table | banner, metadata, iterations, summary, forces, performance |
| without it | the full trace (header, iteration table, timings) | — |

```text
      Step     Time          Energy          fmax
BFGS:    0 15:43:58     -477.459029        2.549943
BFGS:    1 15:46:19     -477.616855        1.552129
BFGS:    2 15:48:12     -477.790440        0.534798
```

That is the whole terminal output of a relaxation that writes a log — the
energies and the largest force per step, in ASE's own format, with the 85 lines
per step of configuration, iterations and timings going to the file instead.

`trace=True` prints the trace anyway (a log file *and* a running commentary),
and `trace=False` suppresses it even without a log file. A single-point run with
the trace off prints nothing: the energy is the return value and the file has
the rest. There is no `verbose` argument on `Carcara` — it is refused with a
message pointing at `trace=`.

## The standard-output trace

Printed when no log file is given (or with `trace=True`): a header, one row per
ADAPT iteration, and a closing summary with the timings.

```text
======================================================================
ADAPT-VQE | mapping: jordan_wigner | 6 qubits | device: AER_simulator
pool: qeb (QEBPool) | 8 operators | optimizer: COBYLA | gradient: analytic
k-points: Gamma (1x1x1 Monkhorst-Pack) | spin: False | reference: hartree-fock
provider: qiskit | circuits: False | quenching: True
======================================================================
Qubit Hamiltonian: 118 Pauli terms
Hartree-Fock reference energy = -154.59070457 eV
======================================================================
iter   |grad|      E (eV)       dE   expr  cnot    1q depth   type operator
-----------------------------------------------------------------------------
   1 2.72e-01 -154.730899 -1.4e-01   6.21    48    40    65 double QD(0,3->2,5)
   2 8.31e-02 -154.765242 -3.4e-02   2.91    96    74   129 double QD(1,4->2,5)
   3 5.54e-02 -154.776105 -1.1e-02   2.05   144   108   193 double QD(1,3->2,5)
```

The **pool's type and size** and the Hamiltonian's **term count** appear in the
header, before the iterations. Neither the pool's operators nor the
Hamiltonian's Pauli strings are printed: a realistic pool grows with the fourth
power of the number of orbitals and a realistic Hamiltonian runs to thousands of
terms, so printing either buries the run.

Each iteration is one row, one column per property computed at that step:

| Column | Meaning |
| :--- | :--- |
| `iter` | Growth step (1-based). |
| `\|grad\|` | Largest pool gradient; the operator with this gradient is the one selected. Convergence is when it falls below `gradient_tolerance`. |
| `E (eV)` | Energy after the inner re-optimization (Hartree with `atomic_units=True`). |
| `dE` | Change from the previous step. |
| `expr` | Expressivity of the grown ansatz: KL divergence from the Haar distribution over the number-conserving sector. It falls as the ansatz specializes. **Off by default** and the column is then absent rather than blank; `run(log_expressivity=True)` adds it. It is a diagnostic, not a result, and not cheap: `2 x 400` state preparations per iteration, each applying every operator in the ansatz, so the cost is linear in the ansatz and quadratic over a run (0.010 / 0.031 / 0.059 / 0.125 s at 1 / 4 / 8 / 16 operators, 6 qubits). |
| `npar` | Variational parameters in the ansatz. Equal to `iter`, so it is the first column dropped on a narrow terminal. |
| `cnot` | CNOT gates after compiling to the native gate set. |
| `1q` | Single-qubit gates in the same compilation. |
| `depth` | Circuit depth in the same compilation. |
| `type` | Kind of the selected operator (`double`, `single`, `ceo`, ...), with the pool's name stripped — the header already carries it. |
| `operator` | Its label, e.g. `QD(0,3->2,5)`. |

**One iteration is always one line.** The row is sized to the terminal
(`shutil.get_terminal_size()`, falling back to 80 columns when the output is
piped): if the full set will not fit, columns are dropped in the order `npar`,
`dE`, `1q`, `depth`, `expr`, `cnot` — the derivable ones first — rather than
letting rows wrap. `iter`, the gradient, the energy, the operator type and its
label are never dropped. Set the `COLUMNS` environment variable to override the
detected width.

The circuit columns need `profile=True` (the default); with `profile=False`
they read `-`.

## The `output.txt` log

`output=` writes the machine-readable protocol of
{mod}`carcara.utils.logging`: the start-up banner, metadata, optimizer setup,
one row per iteration, a summary, and -- when forces were computed -- the
forces. Each iteration row names the **selected operator** and the setup block
the **pool's size** -- not the pool's contents, for the same reason the trace
does not. {func}`carcara.utils.logging.parse_output` reads it back.

```python
calc = Carcara(method="adapt-vqe",
               basis="FAO",
               h=0.25,
               output="output.txt")
```

### A geometry optimization writes one log

A relaxation runs one complete ADAPT-VQE per geometry, so its log is a sequence
of blocks -- energies then forces, one pair per step -- in one file:

```text
<banner: version, host, interpreter, dependencies>       (once, at the top)
========================================================================
    ADAPT-VQE (CEOPool, 12 qubits)
========================================================================
[METADATA]                 step: 1, this geometry, the cell
[OPTIMIZATION SETUP]
[ITERATIONS]               one row per grown operator
[SUMMARY]
[FORCES]                   step: 1, the vectors and their breakdown
========================================================================

========================================================================
    ADAPT-VQE (CEOPool, 12 qubits) -- geometry step 2
========================================================================
[METADATA]                 step: 2, the geometry BFGS moved to
...
```

The section markers and the rules sit at column 0 and everything a block
*contains* is indented one level of
{data}`carcara.utils.logging.INDENT` (4 spaces), with a nested table or the
atoms under `geometry:` one level further -- so the structure of the file can be
read off the left margin:

```text
[METADATA]
    step: 1
    units: Angstrom
    n_atoms: 3
    geometry:
        O       5.0000000000     5.0000000000     5.0000000000
        H       5.0000000000     6.0000000000     5.0000000000
    cell_present: True
    cell_vectors:
        a1 = [ 10.0000000000   0.0000000000   0.0000000000]
    cell_lengths: a=10.0000000000 b=10.0000000000 c=10.0000000000

[ITERATIONS]
    iter        energy (eV)              type        |grad|      1q    cnot   depth operator
    ----------------------------------------------------------------------------------------
       1     -27.6211823512  fermionic-double  2.714649e-01      38      48      65 D(0,2->1,3)
========================================================================
[SUMMARY]
    converged: True
    optimal_energy_eV: -27.6211823512
    num_operators: 1
    cnot_count: 48
    circuit_depth: 65
```

The log's table keeps full precision and the pool's own operator *kind*
(`fermionic-double`, not the trace's stripped `double`), and its circuit columns
run cheapest gate first -- `1q`, `cnot`, `depth`. The summary does **not** repeat
the operator sequence: the table above already names the operator every step
selected, in order, and `result.operators` has it as data.

Indentation is cosmetic to the reader and invisible to the parser --
`parse_output` strips every line -- so a log written before this convention
still reads back.

The **banner belongs to the file**, so it is written once, before the first
block; every step's own block is headed with its `step:` number. The first
logger of a path in a process truncates the file, and every later one appends --
so nothing a relaxation computed is erased, and a run picking up a path from an
earlier run in the same process (a notebook cell) can start a fresh file with
{func}`carcara.utils.logging.reset_log`.

`parse_output` returns the blocks as `result["steps"]`, in order, while the
top-level `metadata` / `setup` / `iterations` / `summary` / `forces` keys
describe the **last** step -- so reading a single-point log is unchanged:

```python
from carcara.utils import parse_output

log = parse_output("output.txt")
for step in log["steps"]:
    print(step["metadata"]["step"],
          step["summary"]["optimal_energy_eV"],
          step["forces"]["max_force"])
```

### The forces block

Whenever `Carcara` computes forces it appends them to the same log, under the
iteration table of the step they belong to (the solver's own log is already
closed by then -- the gradient is taken after the variational run):

```text
[FORCES]
    step: 3
    units: eV/Angstrom
    convention: forces = -dE/dR (ASE sign); hellmann_feynman and pulay are +dE/dR
    force_method: rdm
    include_pulay: True
    pulay_fraction: 37.943206
    n_electrons: 8.000000
    orbital_gradient: 8.122069e-03
    translational_residual: 5.098009e-02
    forces:
         atom symbol                x                y                z            |F|
        ------------------------------------------------------------------------------
            1 O            0.00000001       0.90285479       0.90326428     1.27711908
            2 H           -0.00000000       0.00017172      -0.85216336     0.85216338
            3 H            0.00000000      -0.85204642      -0.00017513     0.85204644
    hellmann_feynman:
        ...
    pulay:
        ...
    max_force: 1.27711908
    rms_force: 1.01377162
    net_force: 0.05098009
```

`forces` is what ASE consumes ($-dE/dR$); `hellmann_feynman` and `pulay` are the
gradient components ($+dE/dR$) in the convention of
{class}`~carcara.algorithms.forces.ForceResult`, so the split between "the
operators moved" and "the basis functions moved" can be read off per atom.
`max_force` is the number an ASE optimizer converges on, and the two residuals
are the honest measures of whether the gradient is usable for geometry at all:
`orbital_gradient` is the orbital response the RDM gradient neglects
({data}`~carcara.algorithms.calculator.ORBITAL_RESPONSE_TOLERANCE` is where it
starts warning) and `net_force` is the grid's egg-box
-- a free molecule feels no net force, so whatever appears there is
discretization artifact.

### The performance block

Every evaluation closes with where its time and memory went:

```text
[PERFORMANCE]
    step: 2
    stages (wall-clock seconds):
        gradient screening                          0.0393
        parameter optimization                      6.6403
        circuit profiling                           0.2014
        integration: one-body integrals             0.0246
        integration: two-body integrals (fft)       2.6896
        nuclear gradient (forces)                  32.2787
        --------------------------------------------------
        sum of timed stages                        41.8739
    untimed_s: 7.0650
    wall_time_s: 48.9390
    integration_backend: C (OpenMP)
    openmp_threads: 8
    cpu_count: 8
    mpi: not used (single process, shared-memory OpenMP)
    peak_memory_MiB: 4971.7
    resident_memory_MiB: 1497.5
    solver_wall_time_s: 16.6582
```

The **calculator** writes this block, not the solver, and that is why it comes
after `[FORCES]`: on a real relaxation the nuclear gradient is the largest single
stage. A six-step water relaxation in PAW-SZ at `h = 0.10` spends **62 %** of its
325 s on forces and **19 %** on the variational optimization, so a block closed
when the solver finished would account for the smaller part of the step.

`untimed_s` is the honest remainder: `wall_time_s` minus the stages, i.e. the
work no stage wraps (building the Hamiltonian, the fermion-to-qubit mapping,
materializing the operator). `peak_memory_MiB` is the process high-water mark and
therefore monotonic across a relaxation; `resident_memory_MiB` is what is
resident at the end of *this* step, so the two together show whether a stage
allocated and released. `openmp_threads` is what the C integral backend used
against the `cpu_count` the machine offers, and `mpi` states plainly that there
is no distributed parallelism -- Carcará is a single process.

When a run reaches a quantum processor the block also carries the QPU
accounting:

```text
    qpu_device: ibm_fez
    qpu_shots: 4096
    qpu_wall_time_s: 51.2470
    qpu_jobs: 1
    qpu_job_ids: daicus1hvn6c73ct2je0
    qpu_seconds: 27.0
    qpu_billed_seconds: 31.5
```

`qpu_wall_time_s` is measured locally around the submission and is always
present; `qpu_seconds` is the **metered** quantum time the provider reports
(Qiskit Runtime does, for a real device) and is what counts against an open
plan's monthly budget. A local simulator or a fake backend therefore reports the
wall clock and the device name, and nothing is invented. `parse_output` reads the
whole block back as `result["performance"]`, with the stages as
`performance["stages_s"]`.

## `verbose_operators` and `verbose_hamiltonian`

Two options write what the trace leaves out, as JSON, once per run:

```python
calc = Carcara(method="adapt-vqe",
               basis="FAO",
               h=0.25,
               verbose_operators=True,      # -> pool.json
               verbose_hamiltonian=True)    # -> hamiltonian.json
```

`pool.json` carries the pool's name and size and, for every operator, its
label, kind, support and the Pauli expansion of its anti-Hermitian generator.
`hamiltonian.json` carries the qubit Hamiltonian's Pauli terms with their
complex coefficients **in Hartree** -- the operator's own unit, whatever units
the driver reports energies in.

Either option also takes a path, which is how a scan gives each geometry its
own file:

```python
calc = Carcara(method="adapt-vqe",
               basis="FAO",
               h=0.25,
               verbose_operators=f"data/pool_{distance:.2f}.json",
               verbose_hamiltonian=f"data/hamiltonian_{distance:.2f}.json")
```

Both are capped at
{data}`carcara.core.serialization.MAX_FILE_QUBITS` qubits: above that the file
is skipped with a `RuntimeWarning` and the run continues.

The same flags exist on the command line:

```bash
carcara LiH --cell 10 --verbose-operators --verbose-hamiltonian
carcara LiH --cell 10 --verbose-operators data/pool.json
```

```{note}
`verbose_hamiltonian` writes a file for *reading*. To write one Carcará can
read *back* -- skipping the integrals and the fermion-to-qubit mapping on the
next run -- use `save_hamiltonian=` and `load_hamiltonian=` instead; see
[the Hamiltonian cache](hamiltonian_cache.md).
```
