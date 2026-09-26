# Reading a run

A variational run reports itself in three ways: **one structured report**, two
optional **JSON dumps** for the objects that are too large to print, and a
**`references.bib`** of the methods it used.

## What goes where

There is one report -- the blocks described below -- and `txt=` decides *where*
it goes, on the same principle as GPAW's `txt=`:

| | standard output | the `txt=` file |
| :--- | :--- | :--- |
| with `txt=<path>` | ASE's `Step Time Energy fmax` table | banner, system, basis, electrons, setup, iterations, summary, forces, performance |
| without it | **the same blocks**, printed | — |
| `txt=<path>`, `trace=True` | the same blocks, printed | the same blocks, written |

It is rendered **once**, by {mod}`mandacaru.utils.logging`, and written to every
destination the driver's `log_targets` names. So the terminal and the file never
disagree: what you read while a run goes by is what a collaborator reads in the
log afterwards, key for key, and
{func}`mandacaru.utils.logging.parse_output` reads either one back.

```text
      Step     Time          Energy          fmax
BFGS:    0 15:43:58     -477.459029        2.549943
BFGS:    1 15:46:19     -477.616855        1.552129
BFGS:    2 15:48:12     -477.790440        0.534798
```

That is the whole terminal output of a relaxation that writes a log — the
energies and the largest force per step, in ASE's own format, with the 85 lines
per step of configuration, iterations and timings going to the file instead.

`trace=True` prints the blocks anyway (a log file *and* a running commentary),
and `trace=False` suppresses them even without a log file. A single-point run
with the trace off reports nothing: the energy is the return value and the file,
if there is one, has the rest. There is no `verbose` argument on `Mandacaru` —
it is refused with a message pointing at `trace=`.

```{note}
`txt=` is accepted only by a method whose `run()` goes through this protocol —
`"adapt-vqe"` today — and refused with a message by the others rather than
leaving an empty file. `"rhf"` and `"uhf"` print an SCF summary;
`"hva"`, `"vqe"`, `"subspace-vqe"` and `"subspace-adapt-vqe"` print their own
run header, which is not this one.

There is one standard output per process, so consecutive printed runs number
their blocks `step: 1`, `step: 2`, … just as consecutive runs sharing one
`txt=` file do. {func}`~mandacaru.utils.logging.reset_log` with
{data}`~mandacaru.utils.logging.STDOUT` restarts the count.
```

## The iteration table

The block every run is read through, whichever destination it went to:

```text
[ITERATIONS]
    iter     time        energy (eV)                 dE        |grad|   steps      1q    cnot   depth operator
    ----------------------------------------------------------------------------------------------------------
       1 13:45:10     -33.1720283176      -0.1939118279      0.225802       3      38     104     121 D(0,4->3,7)
       2 13:45:10     -33.2886620234      -0.1166337058      0.158793       4      73     176     209 D(0,4->1,7)
       3 13:45:10     -33.3953428910      -0.1066808676      0.151665       5     108     256     304 D(0,4->3,5)
       4 13:45:10     -33.4961593720      -0.1008164810      0.134065       7     143     304     368 D(0,4->1,5)
```

The **pool's type and size** and the Hamiltonian's **term count** are in the
`[OPTIMIZATION SETUP]` and `[ELECTRONS]` blocks above it. Neither the pool's
operators nor the Hamiltonian's Pauli strings are printed: a realistic pool
grows with the fourth power of the number of orbitals and a realistic
Hamiltonian runs to thousands of terms, so printing either buries the run.

Each iteration is one row, one column per property computed at that step:

| Column | Meaning |
| :--- | :--- |
| `iter` | Growth step (1-based). |
| `time` | Wall-clock time the step finished, `HH:MM:SS`. |
| `energy (eV)` | Energy after the inner re-optimization (Hartree with `atomic_units=True`). |
| `expr` | Expressivity of the grown ansatz: KL divergence from the Haar distribution over the number-conserving sector. It falls as the ansatz specializes. **Off by default** and the column is then absent rather than blank; `run(log_expressivity=True)` adds it. It is a diagnostic, not a result, and not cheap: `2 x 400` state preparations per iteration, each applying every operator in the ansatz, so the cost is linear in the ansatz and quadratic over a run (0.010 / 0.031 / 0.059 / 0.125 s at 1 / 4 / 8 / 16 operators, 6 qubits). |
| `dE` | Energy change from the previous row, signed, in the energy column's unit; the first row's is from the reference state (`reference_energy_<unit>` in `[OPTIMIZATION SETUP]`), and a resumed run's from the energy its restored ansatz had. |
| `\|grad\|` | Largest pool gradient, to six decimals; the operator with this gradient is the one selected. Convergence is when it falls below `gradient_tolerance`. A gradient below 5e-7 reads `0.000000`. |
| `steps` | **Steps the classical optimizer took** to re-optimize the grown ansatz — parameter updates, not cost evaluations. The two differ by the method: L-BFGS spends several evaluations per step on a finite-difference gradient and a line search, SPSA two or three, while COBYLA evaluates once per trial point. `-` when a method reports neither a count nor a per-iteration callback. |
| `cnot` | CNOT gates after compiling to the native gate set. |
| `1q` | Single-qubit gates in the same compilation. |
| `depth` | Circuit depth in the same compilation. |
| `operator` | The selected operator's label, e.g. `D(0,4->3,7)`; the label names its kind, and the pool is named in `[OPTIMIZATION SETUP]`. |

**The table does not depend on the terminal.** It is the same width whether it
is printed or written, because it is the same table: no column is dropped and no
operator label is abbreviated, so a narrow terminal wraps a row rather than
quietly losing a number. `expr` is the one optional column — absent unless
`run(log_expressivity=True)` asked for it, rather than present and blank.

The circuit columns need `profile=True` (the default); with `profile=False`
they read `-`.

## The `output.txt` log

`txt=` writes the machine-readable protocol of
{mod}`mandacaru.utils.logging`: the start-up banner, metadata, optimizer setup,
one row per iteration, a summary, and -- when forces were computed -- the
forces. Each iteration row names the **selected operator** and the setup block
the **pool's size** -- not the pool's contents, for the same reason the table
does not. {func}`mandacaru.utils.logging.parse_output` reads it back.

```python
calc = Mandacaru(method="adapt-vqe",
                 basis="HAO",
                 h=0.25,
                 txt="output.txt")
```

### A geometry optimization writes one log

A relaxation runs one complete ADAPT-VQE per geometry, so its log is a sequence
of blocks -- energies then forces, one pair per step -- in one file:

```text
<banner: version, host, interpreter, dependencies>       (once, at the top)
========================================================================
    ADAPT-VQE (CEOPool, 12 qubits)
========================================================================
[SYSTEM]                        step: 1, this geometry, its cell and spins
[BASIS]                         the basis that ran: options, radii, datasets
[ELECTRONS]                     grid, reference, mapping, register, Hamiltonian
[OPTIMIZATION SETUP]            the optimizer, the gradient, the operator pool
[ITERATIONS]                    one row per grown operator
[VARIATIONAL QUANTUM SUMMARY]   the converged state of *this* geometry
[FORCES]                        step: 1, the vectors and their breakdown
[PERFORMANCE]                   step: 1, where the time and memory went
========================================================================

========================================================================
    ADAPT-VQE (CEOPool, 12 qubits) -- geometry step 2
========================================================================
[SYSTEM]                        step: 2, the geometry BFGS moved to
...
========================================================================

[GEOMETRY OPTIMIZATION SUMMARY] the trajectory as one thing
[RELAXATION COMPLETE]           the footer: the run finished
```

The per-step block is the **variational** summary -- the converged state of one
geometry -- and the relaxation's own summary is the single block at the end.

The section markers and the rules sit at column 0 and everything a block
*contains* is indented one level of
{data}`mandacaru.utils.logging.INDENT` (4 spaces), with a nested table or the
atoms under `geometry:` one level further -- so the structure of the file can be
read off the left margin:

```text
[SYSTEM]
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
    pbc: a=False b=False c=False (non-periodic)
    initial_magnetic_moments: [0.0, 0.0, 0.0]

[BASIS]
    name: PAW-LCAO
    family: PAW-LCAO (projector augmented wave (Bloechl 1994), frozen core, ...)
    size: SZ
    zeta_split: tail_norm 0.16, 0.3, 0.6 (norm of the tail, every zeta split from the first)
    projector_basis: raw
    energy_shift: 0.1 eV
    confinement_potential: A exp(-(r_c - r_i)/(r - r_i)) / (r_c - r), A = 12 Ha, r_i = 0.6 r_c
    polarization: gaussian (quasi-Gaussian: r^l [exp(-r^2/r_char^2) - (a - b r^2)], further shells split from it)
    filter: filtered (auto: 1 x Nyquist)
    filter_cutoff: 16.6244 Bohr^-1 (3760.30 eV)
    local_potential: range-separated: long range on the grid, short range on atom-centered quadrature (sigma = 0.2646 Bohr)
    dataset_xc: LDA (atomic reference only; the valence interaction is the bare Coulomb operator)
    directory: lda (/data/mandacaru-paw/lda)
    basis_functions: 6
    datasets:
        symbol  Z  Z_ion  l_max  r_cut_Bohr  source
        ---------------------------------------------------
        O       8  6      1      1.4499      O.parquet
        H       1  1      0      1.2998      H.parquet
    orbitals:
        symbol  l  zetas  polarization  r_c_Bohr  r_c_Angstrom  eps_free_eV  eps_basis_eV  shift_eV
        -------------------------------------------------------------------------------------------
        O       0  1      0             4.3652    2.3100        -23.706920   -23.606915    0.100005
        O       1  1      0             5.3500    2.8311        -9.205641    -9.105680     0.099961
        H       0  1      0             6.6822    3.5361        -6.358289    -6.258289     0.100000
    functions:
        symbol  atoms  functions_per_atom
        ---------------------------------
        O       1      4
        H       2      1

[ELECTRONS]
    grid spacing: 0.1 Angstrom (requested)
    grid points: 101 x 101 x 101 (spacing 0.1000 x 0.1000 x 0.1000 Angstrom)
    kinetic operator: finite difference
    k-points: Gamma (1x1x1 Monkhorst-Pack)
    charge: 0
    spin-polarized: False (multiplicity 1)
    reference state: hartree-fock
    frozen core: none
    mapping: Jordan-Wigner
    Hamiltonian: 1079 Pauli terms
    spatial orbitals: 6
    electrons (alpha, beta): (4, 4)
    qubits: 12

[OPTIMIZATION SETUP]
    classical_optimizer: SLSQP
    max_iterations: 50
    gradient_method: analytic
    gradient_formula: exact derivative, g = 2 Re<H psi|A psi>
    gradient_tol: 0.001
    gradient_units: Hartree
    pool: ceo
    pool_class: CEOPool
    pool_size: 92
    reoptimize_all_parameters: True
    state_vector_backend: sparse matrices, 2^12 amplitudes
    device: AER_simulator
    backend_provider: qiskit
    circuit_execution: False
    shots: 0 (exact expectation values)
    circuit_profiling: True
    energy_unit: eV
    reference_energy_eV: -476.8628634829

[ITERATIONS]
    iter     time        energy (eV)                 dE        |grad|   steps      1q    cnot   depth operator
    ----------------------------------------------------------------------------------------------------------
       1 13:45:10     -33.1720283176      -0.1939118279      0.225802       3      38     104     121 D(0,4->3,7)
========================================================================
[VARIATIONAL QUANTUM SUMMARY]
    converged: True
    optimal_energy_eV: -27.6211823512
    num_operators: 1
    cost_evaluations: 31
    optimizer_steps: 4
    cnot_count: 48
    circuit_depth: 65
```

The table keeps full precision and the pool's own operator *kind*
(`fermionic-double`), and its circuit columns run cheapest gate first --
`1q`, `cnot`, `depth`. The summary closes the two
classical-effort counters the `steps` column opens: `optimizer_steps` is that
column summed over the run and `cost_evaluations` the energy evaluations those
steps spent -- the number a QPU would be billed for. The summary does **not**
repeat the operator sequence: the table above already names the operator every step
selected, in order, and `result.operators` has it as data.

Indentation is cosmetic to the reader and invisible to the parser --
`parse_output` strips every line -- so a log written before this convention
still reads back.

The **banner belongs to the file**, so it is written once, before the first
block; every step's own block is headed with its `step:` number. The first
logger of a path in a process truncates the file, and every later one appends --
so nothing a relaxation computed is erased, and a run picking up a path from an
earlier run in the same process (a notebook cell) can start a fresh file with
{func}`mandacaru.utils.logging.reset_log`.

`[BASIS]` records the single-particle basis **that ran**, which is more than
the options that were typed: a basis here is built at run time, so the family
defaults left alone (a PAW-LCAO basis is Fourier-filtered and its local potential is
range-separated unless told otherwise), the cutoff radius an `energy_shift`
gave each orbital together with the eigenvalue shift actually achieved, the
folder the datasets were read from (`directory`: the folder's name, such as
`lda` or `pbe`, and its path), the file of every dataset, the functional the
datasets record (`dataset_xc`) and the function count per element are all
decided below the calculator. It is the block a comparison against another code
is made from. Its three tables -- `datasets`, `orbitals`, `functions` -- read
back from `parse_output(path)["basis"]` as lists of rows keyed by the column
names. An all-electron basis gets the block too (name, options, functions); a
direct-mode run, where no basis was built, has none.

Each fact has **one owner**. The basis is in `[BASIS]` and not repeated in
`[ELECTRONS]`; the classical optimizer is in `[OPTIMIZATION SETUP]` and not
repeated in the variational summary. A value written in two blocks can disagree
with itself, which is worth more than the convenience of not scrolling.

`[SYSTEM]` says *where the atoms are*, `[ELECTRONS]` *what was solved* (the
reference state, the frozen core, the charge, the mapping, the register width)
and `[OPTIMIZATION SETUP]` *how* it was solved (the optimizer, the gradient, the
pool, the growth rule, the backend, the device and the shots).

Two of `[SYSTEM]`'s lines are easy to miss and worth naming:

| Line | Meaning |
| :--- | :--- |
| `pbc` | Whether each lattice direction is periodic, plus what the flags add up to: `a=True b=False c=False (1-D, periodic along a)`. **Not the same fact as `cell_present`** -- Mandacaru always needs a cell, because it is the real-space box the grid is cut from, so a molecule has one and is still `(non-periodic)`. |
| `initial_magnetic_moments` | The geometry's per-atom moments, in atom order, as the same list a caller passes to `Atoms(magmoms=...)`: `[1.0, -1.0]`. It lines up index for index with the `geometry:` rows and reads back with `ast.literal_eval`. They are what selects the spin state (a triplet comes from `Atoms(..., magmoms=[1, 1])`, not from a flag), so a run given them says so. A closed-shell geometry is `[0.0, 0.0]` rather than a word -- an all-zero list cannot be confused with nobody having looked, which is `(not provided)`. |

### `[OPTIMIZATION SETUP]`: how the run was configured

The block is written in five groups, each contiguous, so a fact sits next to the
fact it qualifies and a diff between two runs reads as a diff between two
settings:

| Group | Lines |
| :--- | :--- |
| classical optimizer | `classical_optimizer`, `max_iterations` |
| screening gradient | `gradient_method`, `gradient_formula`, `gradient_tol`, `gradient_units` |
| operator pool | `pool`, `pool_class`, `pool_size` |
| growth and execution | `reoptimize_all_parameters`, `state_vector_backend`, `device`, `backend_provider`, `circuit_execution`, `shots`, `circuit_profiling` |
| the loop's starting point | `energy_unit`, `reference_energy_<unit>` |

A fresh run starts from the reference state that `[ELECTRONS]` names, with an
empty ansatz; the block does not repeat it. A **resumed** run closes the block
with its lineage -- `resumed_from`, `restored_operators`, `restored_parameters`,
`restored_energy_<unit>`, `resume_same_hamiltonian` -- which is the one record
of the grown ansatz it started from.

#### The screening gradient

`gradient_method` is the `gradient=` option of the driver: how the pool
screening gradients in the `|grad|` column were computed. It sits directly above
`gradient_tol`, which is compared against those numbers, and `gradient_formula`
says in one line what was evaluated. `gradient_units` is `Hartree` whatever unit
the energy columns are in -- the gradient is an expectation value of the
Hamiltonian's commutator, not an energy in the reported unit.

| `gradient_method` | What it computes | What it costs |
| :--- | :--- | :--- |
| `analytic` (default) | The exact derivative `g_i = 2 Re<H psi\|A_i psi>` -- what the other two estimate. | One matrix-vector product per pool operator; no step-size truncation error. |
| `finite_difference` | A central difference of the energy at shifted parameters, step `1e-4` (`FINITE_DIFFERENCE_STEP`, reported in `gradient_formula`). | `2 x \|pool\|` energy evaluations, plus the `\|pool\|` dense diagonalizations those evaluations are read from. |
| `parameter_shift` | The quantum parameter-shift rule, reconstructed over the generator's frequency set (exact, not an approximation). | The same shifted-energy evaluations and diagonalizations, several shifts per frequency. |

The shift-based estimators are opt-in, for studying the estimator itself: they
converge to the number `analytic` computes directly. The eigendecompositions they
need are built lazily, so a default run never pays for them.

Those three names are the **canonical spelling** and the only one written to the
log, but the input is spelling-insensitive the way `method=` is --
`gradient="parameter-shift"`, `"parameter_shift"` and `"Parameter Shift"` all
select the same estimator and all log `parameter_shift`:

```python
calc = Mandacaru(method="adapt-vqe",
                 basis="HAO",
                 h=0.25,
                 gradient="parameter_shift",
                 txt="output.txt")
```

The block reports **what ran**, not what was asked for. A sparse or
sector-restricted run never forms the eigendecompositions the shift estimators
need and screens analytically whatever was requested, so it logs
`gradient_method: analytic` with the override named in the formula line:

```text
    gradient_method: analytic
    gradient_formula: exact derivative, g = 2 Re<H psi|A psi> (overrides parameter_shift: sparse pool)
```

`parse_output` returns the blocks as `result["steps"]`, in order, while the
top-level `system` / `electrons` / `setup` / `iterations` / `summary` / `forces` /
`performance` keys describe the **last** step -- so reading a single-point log is
unchanged:

```python
from mandacaru.utils import parse_output

log = parse_output("output.txt")
for step in log["steps"]:
    print(step["system"]["step"],
          step["summary"]["optimal_energy_eV"],
          step["forces"]["max_force"])
```

### Closing a relaxation

A relaxation ends with two blocks that belong to the file rather than to any one
geometry:

```text
[GEOMETRY OPTIMIZATION SUMMARY]
    geometry_steps: 6
    units: Angstrom
    initial_energy_eV: -477.4590289021
    final_energy_eV: -477.7982429880
    energy_change_eV: -0.3392140859
    initial_max_force: 2.54994331
    final_max_force: 0.01307071
    final_net_force: 0.02219607
    center_of_mass_drift: 0.07050000
    total_wall_time_s: 325.1913
    convergence:
         step          energy (eV)        max force        net force
        ------------------------------------------------------------
            1      -477.4590289021       2.54994331       0.01918785
            ...
            6      -477.7982429880       0.01307071       0.02219607
    relaxed_geometry:
        O       5.0000000001     5.0076240379     5.0076249036
        H       5.0000000000     6.0218198454     4.9708302184
        H       5.0000000001     4.9708300569     6.0218188182
========================================================================

[RELAXATION COMPLETE]
    status: converged after 6 geometry steps (max force 0.013071 <= 0.020000 eV/Angstrom)
========================================================================
```

`center_of_mass_drift` is worth reading: a free molecule cannot translate under
its own forces, so whatever appears there is the grid's egg-box pushing it (see
`project_translation`). `net_force` in the table is always the **unprojected**
residual, so the column shows the artifact even when the reported forces had it
removed.

**Who writes it.** ASE never tells a calculator that a relaxation is over -- the
optimizer simply stops calling it -- so there is no in-band moment to close the
log. A plain script therefore gets the blocks from an **interpreter-exit hook**,
which needs no change to the script at all. Handing the optimizer over closes the
log immediately *and* lets the footer state the verdict, since `fmax` is the
optimizer's own:

```python
opt = BFGS(atoms, trajectory="relax.traj")
opt.run(fmax=0.02)
atoms.calc.write_optimization_summary(optimizer=opt)   # or fmax=0.02
```

Without it the footer reports the final force and says only that the run
finished -- the threshold is the optimizer's, and the log does not guess it.
Nothing is written for a single geometry (there is no trajectory to summarize),
and calling the method *and* letting the hook run still writes one summary.
`parse_output` returns the two blocks as `result["optimization"]` and
`result["completion"]`, with the table as a list of
`{step, energy, max_force, net_force}`.

### The forces block

Whenever `Mandacaru` computes forces it appends them to the same log, under the
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
{class}`~mandacaru.algorithms.forces.ForceResult`, so the split between "the
operators moved" and "the basis functions moved" can be read off per atom.
`max_force` is the number an ASE optimizer converges on, and the two residuals
are the honest measures of whether the gradient is usable for geometry at all:
`orbital_gradient` is the orbital response the RDM gradient neglects
({data}`~mandacaru.algorithms.calculator.ORBITAL_RESPONSE_TOLERANCE` is where it
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
stage. A six-step water relaxation in PAW-LCAO-SZ at `h = 0.10` spends **62 %** of its
325 s on forces and **19 %** on the variational optimization, so a block closed
when the solver finished would account for the smaller part of the step.

`untimed_s` is the honest remainder: `wall_time_s` minus the stages, i.e. the
work no stage wraps (building the Hamiltonian, the fermion-to-qubit mapping,
materializing the operator). `peak_memory_MiB` is the process high-water mark and
therefore monotonic across a relaxation; `resident_memory_MiB` is what is
resident at the end of *this* step, so the two together show whether a stage
allocated and released. `openmp_threads` is what the C integral backend used
against the `cpu_count` the machine offers, and `mpi` states plainly that there
is no distributed parallelism -- Mandacaru is a single process.

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

Two options write what the blocks leave out, as JSON, once per run:

```python
calc = Mandacaru(method="adapt-vqe",
                 basis="HAO",
                 h=0.25,
                 verbose_operators=True,      # -> pool.json
                 verbose_hamiltonian=True)    # -> hamiltonian.inspect.json
```

`pool.json` carries the pool's name and size and, for every operator, its
label, kind, support and the Pauli expansion of its anti-Hermitian generator.
`hamiltonian.inspect.json` carries the qubit Hamiltonian's Pauli terms with their
complex coefficients **in Hartree** -- the operator's own unit, whatever units
the driver reports energies in.

Either option also takes a path, which is how a scan gives each geometry its
own file:

```python
calc = Mandacaru(method="adapt-vqe",
                 basis="HAO",
                 h=0.25,
                 verbose_operators=f"data/pool_{distance:.2f}.json",
                 verbose_hamiltonian=f"data/hamiltonian_{distance:.2f}.json")
```

Both are capped at
{data}`mandacaru.core.serialization.MAX_FILE_QUBITS` qubits: above that the file
is skipped with a `RuntimeWarning` and the run continues.

The same flags exist on the command line:

```bash
mandacaru LiH --cell 10 --verbose-operators --verbose-hamiltonian
mandacaru LiH --cell 10 --verbose-operators data/pool.json
```

```{note}
`verbose_hamiltonian` writes a file for *reading*. To write one Mandacaru can
read *back* -- skipping the integrals and the fermion-to-qubit mapping on the
next run -- use `save_hamiltonian=` and `load_hamiltonian=` instead; see
[the Hamiltonian cache](hamiltonian_cache.md).
```

(references-bib)=
## `references.bib`: what the run should cite

A run also writes the bibliography of the methods it used, in BibTeX, ready to
`\bibliography{references}` from a manuscript:

```
% ADAPT-VQE, and the fermionic operator pool
@article{Grimsley2019,
  author  = {Grimsley, Harper R. and Economou, Sophia E. and Barnes, Edwin and
             Mayhall, Nicholas J.},
  title   = {An adaptive variational algorithm for exact molecular simulations
             on a quantum computer},
  journal = {Nat. Commun.},
  ...
}
```

Journal names are abbreviated, and each entry carries a comment saying what in
the run pulled it in.

### It cites what ran, not what was typed

The selection comes from the run's actual configuration, which is not the same
as the options a user wrote. A PAW-LCAO basis is Fourier-filtered and confined by
default, so

```python
calc = Mandacaru(method="adapt-vqe",
                 basis={"name": "PAW-LCAO", "size": "SZ"},
                 h=0.25,
                 pool="ceo")
```

cites Blöchl for the datasets, Anglada and Soler for the filter, *and*
Sankey--Niklewski and Junquera *et al.* for the confining potential behind the
default `energy_shift` -- none of which appear in the call. Equally, `"ceo"`
cites both the coupled-exchange paper and the qubit-excitation paper whose
operators it couples, `tetris=True` adds TETRIS-ADAPT-VQE and the default does
not, and Qiskit is cited only when circuits actually went through it.

A few citations cannot be known from the configuration at all, because they
depend on what is *called*: `energy_levels()` adds the variational-deflation
paper, and an expressibility trace adds Sim *et al.* Those are added when they
happen, and the file already on disk is rewritten so it does not go stale.

### Where it goes

`references=` follows the same convention as the other files a run writes:

| value | effect |
| --- | --- |
| `"auto"` (default) | `references.bib` beside the `txt=` log, and nothing when there is no log |
| `True` | `references.bib` in the working directory |
| `"papers.bib"` | that file |
| `False` / `None` | nothing |

```python
calc = Mandacaru(method="adapt-vqe",
                 basis="PAW-LCAO",
                 h=0.20,
                 pool="qubit",
                 txt="run/output.txt")      # -> run/references.bib
```

On the command line:

```bash
mandacaru H2O --cell 10 --txt run/output.txt        # run/references.bib
mandacaru H2O --cell 10 --references papers.bib
```

The calculator also exposes the two pieces directly, which is what a script
that assembles one bibliography for a whole study wants:

```python
atoms.calc.citation_keys()            # ['Grimsley2019', 'Jordan1928', ...]
atoms.calc.write_references("all.bib")
```

```{warning}
The table lives in {mod}`mandacaru.utils.bibliography`. Entries taken from a
paper in the author's own library are marked verified; the rest are standard
citations written from the usual bibliographic data and are rendered with an
`[unverified record]` comment. **Check those against the journal before they go
into a manuscript** --
{data}`mandacaru.utils.bibliography.UNVERIFIED` lists them.
```
