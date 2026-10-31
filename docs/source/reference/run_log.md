(run-log-reference)=
# The run log

A calculation can describe itself in a plain-text log: a fixed set of blocks of
`key: value` lines, readable by eye and by a line scanner. This page lists the
blocks that the calculator writes, with a real excerpt of each, and says which
block owns which fact. Everything below was produced by running the calculator on
H2, LiH and a PAW-LCAO H2, with `txt=` and `trace=True`. [What is still not
written](#run-log-not-written) is listed at the end.

## Turning the log on

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6, 6, 6])
atoms.center()
atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35,
                       txt="adapt.log")
atoms.get_potential_energy()
```

- `txt="path"` writes the log to a file; `trace=True` writes the same blocks to
  standard output; both can be given at once. Without either, nothing is written.
  With `txt=` alone the calculator prints nothing to standard output, which is left to
  whatever an ASE optimizer prints there.
- **The log is written when the run ends, not while it runs.** The `time` column of a
  table is the clock when the row was written, so the rows of one run carry almost
  the same time. Follow a long run through the optimizer, not through this file.
- Within one Python process, the first write to a path truncates the file and writes
  the start-up banner; later runs append their blocks, each headed with the next
  `step:` number. That is how a geometry relaxation becomes one continuous log with
  one set of blocks per geometry. A file already on disk from an earlier process
  is truncated by the first run of the new process.
- Standard output receives no banner.
- Energies are in eV, or Ha with `atomic_units=True`; the unit is in the key name
  (`reference_energy_eV`, `reference_energy_Ha`).
- Read a log back with `parse_output`, which returns a dictionary of the blocks
  (`system`, `basis`, `electrons`, `setup`, `scf_iterations`, `active_space`,
  `gradient_measurement`, `iterations`, `summary`, `forces`, `performance` and a
  `steps` list for a multi-step file; a key is present only when its block was
  written). The `scf_`-prefixed keys of `[SCF SETUP]` land in `setup`:

```python
from mandacaru.utils import parse_output

blocks = parse_output("adapt.log")
print(blocks["summary"]["optimal_energy_eV"])
```

A periodic calculation (a crystal) writes `[SYSTEM]` (from a geometry), `[BASIS]`,
`[ELECTRONS]` and a framed `[SCF SUMMARY]`; it has no `[SCF SETUP]` or
`[SCF ITERATIONS]` yet.

## Which blocks a run writes

| Block | Written by | When |
| --- | --- | --- |
| Banner | every run, once per file | Top of a freshly opened file only |
| `[SYSTEM]` | runs from a geometry | Always for them; a run from a Hamiltonian has no geometry and writes none |
| `[BASIS]` | every run | Always (a run from a Hamiltonian states only the precision) |
| `[ELECTRONS]` | every run | Always |
| `[SCF SETUP]`, `[SCF ITERATIONS]` | molecular runs from a geometry | After `[ELECTRONS]`: the mean-field reference of every method; a crystal writes neither |
| `[ACTIVE SPACE]` | quantum methods | Only when `active_space=` reduced the register |
| `[GRADIENT MEASUREMENT]` | quantum methods | Only for a measured-gradient run (`shots > 0` with `adapt-vqe`) |
| `[OPTIMIZATION SETUP]` | quantum methods | Always for them |
| `[ITERATIONS]` | `adapt-vqe` | One row per accepted operator |
| `[MARKOV CHAIN]` | `mcas-vqe`, `valqa` | One row per proposal |
| `[VARIATIONAL QUANTUM SUMMARY]` | quantum methods | Always for them |
| `[SCF SUMMARY]` | `rhf`, `uhf`, `ghf`, `dft` | Always for them |
| `[FORCES]` | any method | Each time forces are computed |
| `[PERFORMANCE]` | any method | At the end of every ASE calculation (`get_potential_energy`, `get_forces`), after `[FORCES]`; never from a direct `run()` |
| `[GEOMETRY OPTIMIZATION SUMMARY]`, `[RELAXATION COMPLETE]` | a relaxation | When you call `write_optimization_summary` |

`vqe`, `subspace-vqe` and `subspace-adapt-vqe` write no progress table: a fixed
ansatz is one minimization, and the subspace methods report levels through the
result object.

## The banner

The first block of a file is the banner: the wordmark, then one table each for the
program, the machine, Python and the packages the run imported.

```text
-----------------------------------------------------------------
    version:       26.10.43
    developed by:  Leandro Seixas Rocha
    website:       https://mandacaru.seixas.dev
    repository:    https://github.com/seixas-research/mandacaru
    documentation: https://mandacaru.readthedocs.io/
-----------------------------------------------------------------

System:
├── architecture: arm64
├── platform:     Darwin
├── user:         leseixas
├── hostname:     safira
├── cwd:          /path/to/the/run
└── PID:          66983

Python:
├── version:    3.14.7 | packaged by conda-forge | ...
└── executable: /path/to/envs/mandacaru/bin/python

Dependencies:
├── ase version:         3.29.0     [/path/to/site-packages/ase]
├── numpy version:       2.5.3      [/path/to/site-packages/numpy]
...
```

The banner owns the program version, the machine and the package versions. No
other block repeats them.

## `[SYSTEM]`

The geometry of this step, the cell and the periodicity.

```text
[SYSTEM]
    step: 1
    units: angstrom
    n_atoms: 2
    geometry:
        H       3.0000000000     3.0000000000     2.6300000000
        H       3.0000000000     3.0000000000     3.3700000000
    cell_present: True
    cell_vectors:
        a1 = [  6.0000000000   0.0000000000   0.0000000000]
        ...
    cell_lengths: a=6.0000000000 b=6.0000000000 c=6.0000000000
    cell_angles: alpha=90.000000 beta=90.000000 gamma=90.000000
    pbc: a=False b=False c=False (non-periodic)
    initial_magnetic_moments: (not provided)
```

A cell and periodicity are different facts: the cell is the box the grid is cut
from, `pbc` says which axes repeat. `initial_magnetic_moments` lists the moments of
the `Atoms` object, one per atom (`[1.0, -1.0]`; all zeros when none were set). It
records the input: moments that sum to zero do not select a spin state, and the spin
populations that result are in `[ELECTRONS]`.

## `[BASIS]`

The basis that ran, not the options that were typed. For an all-electron basis:

```text
[BASIS]
    name: HAO
    family: all-electron
    basis_functions: 2
    precision: float64
    functions:
        symbol  atoms  functions_per_atom
        ---------------------------------
        H       2      1
```

`options` follows `family` when the basis has options (`{virtual_orbitals: 1}`), as the
native engine received them, defaults resolved. `basis_functions` is the number of
basis functions of the whole system, and `functions` counts them per element.
`precision` is the one place the storage precision is stated.

A PAW-LCAO basis resolves what the options left alone: the defaults (confined by
`energy_shift`, Fourier-filtered), the radii an `energy_shift` gives each orbital, and
the dataset files with the digest of the bytes that were read.

```text
[BASIS]
    name: PAW-LCAO
    family: PAW-LCAO (projector augmented wave (Bloechl 1994), frozen core, multipole compensation)
    size: DZ
    zeta_split: first_zeta: tail norms 0.15, 0.31, 0.54 (norm of the tail, every zeta split from the first)
    projector_basis: raw
    polarization_method: gaussian (quasi-Gaussian: ...)
    energy_shift: 0.1 eV
    confinement_potential: A exp(-(r_c - r_i)/(r - r_i)) / (r_c - r), A = 12 Ha, r_i = 0.6 r_c
    diffuse: off
    filter_cutoff: 5.5415 Bohr^-1 (417.81 eV), filtered (auto: 1 x Nyquist)
    local_potential: range-separated: long range on the grid, short range on atom-centered quadrature
    dataset_xc: LDA (atomic reference; the exact one-center Coulomb tensor and the frozen core's exchange)
    paw_dataset: lda-sr (/path/to/mandacaru-paw/lda-sr)
    basis_functions: 4
    precision: float64
    datasets:
        symbol  Z  Z_ion  l_max  r_cut_Bohr  source     sha256
        ...
    orbitals:
        symbol  n  l  zetas  polarization  r_c_Bohr    r_c_Angstrom  eps_free_eV  eps_basis_eV  shift_eV
        ...
    functions:
        symbol  atoms  functions_per_atom
        ---------------------------------
        H       2      2
```

`datasets`, `orbitals` and `functions` come in that order, `functions` last; `orbitals`
has one row per basis state and one per dataset channel without one; an
unconfined orbital reads `unconfined` for its radius and has a shift of 0. With an
unconfined basis (`energy_shift=None`) `energy_shift` reads `unconfined` and there is
no `confinement_potential` line. `local_potential` and `dataset_xc` describe how the
native engine integrates the local potential and the one-center terms.

## `[ELECTRONS]`

How the electronic problem was posed: the discretization first, then the encoding and
the size of what came out. For a quantum method from a geometry, thirteen fields in
this order (`basis` is owned by `[BASIS]`):

```text
[ELECTRONS]
    grid spacing: 0.35 Angstrom (requested)
    grid points: 18 x 18 x 18 (spacing 0.3500 x 0.3500 x 0.3500 Angstrom)
    kinetic operator: spectral
    k-points: Gamma (1x1x1 Monkhorst-Pack)
    charge: 0
    spin-polarized: False (multiplicity 1)
    reference state: hartree-fock
    Z2 tapering: none
    mapping: Jordan-Wigner
    Hamiltonian: 15 Pauli terms
    spatial orbitals: 2
    electrons (alpha, beta): (1, 1)
    qubits: 4
```

`grid points` is what the cell turned the requested spacing into. `Hamiltonian` is the
number of Pauli terms the native problem holds. A line
`two-body integrals: Cholesky factors, tolerance ..., rank ..., largest residual
diagonal ...` (it also states the rank against the number of basis functions) follows
`kinetic operator` when you passed `two_body=`; a run that did
not ask for a factorization says nothing about it. A classical run (`rhf`, `uhf`,
`ghf`, `dft`) writes no `reference state`, `Z2 tapering`, `mapping`, `Hamiltonian` or
`qubits`: a line `Jordan-Wigner` would describe a calculation that did not happen. A
calculation from a Hamiltonian has no grid, kinetic operator, k-points or charge.
Above 20 qubits a line `operator_details: omitted (...)` follows and the operator
column of the iteration table reads `(omitted)`.

## `[SCF SETUP]` and `[SCF ITERATIONS]`

The mean-field calculation every method starts from, in the unit of the log:

```text
[SCF SETUP]
    scf_method: restricted Hartree-Fock (closed shell)
    scf_max_iterations: 300
    scf_convergence_Hartree: 1e-09 (energy change and residual)
    scf_acceleration: DIIS
    scf_history: iterations of the lowest-energy start

[SCF ITERATIONS]
     iter    electronic_energy (eV)               dE (eV)        residual
    ---------------------------------------------------------------------
        1            -45.0904756509                     -       4.441e-16
        2            -45.0904756509         +0.0000000000       1.665e-16
```

The setup values are the ones the native method declares, overridden by
`scf_options`. `electronic_energy` leaves out the nuclear repulsion. The table is the
saved history of the start that won, and it has no `time` column: the native SCF
records no clock, and the log does not invent one. The `scf_` prefix keeps these keys
apart from the keys of `[OPTIMIZATION SETUP]` when a reader parses both.

## `[ACTIVE SPACE]`

Written between `[ELECTRONS]` and `[OPTIMIZATION SETUP]` when `active_space=` is set.
For LiH with `active_space={"frozen": "auto", "orbitals": 2}`:

```text
[ACTIVE SPACE]
    method: energy
    frozen_orbitals: 1
    orbitals:
        index  role
        -------------
        0      frozen
        1      active
        2      active
```

`method` is the selector (`energy` when none is named, `mp2` for natural orbitals),
and the table gives each source orbital's fate: `frozen`, `active` or `deleted`. The
orbital counts are not repeated here: `[BASIS]` owns the number of basis functions and
`[ELECTRONS]` the number of orbitals of the register. The selector's occupation
numbers and the reference occupancy are not in the table.

## `[GRADIENT MEASUREMENT]`

For `adapt-vqe` with `shots > 0`, the pool gradients are measured on the device as
commutators `[H, A_k]`. The block states what each growth step costs.

```text
========================================================================
[GRADIENT MEASUREMENT]
    scheme: native commutators [H, A_k], merged QWC measurement bases
    first_step: classical (exact at the reference determinant)
    measured_operators: 3
    strings_per_step: 28
    bases_per_step: 13
    jobs_per_step: 1
    shots_per_step: 6500
========================================================================
```

Here `shots=500` and 13 measurement bases give 6,500 shots per growth step.

## `[OPTIMIZATION SETUP]`

```text
[OPTIMIZATION SETUP]
    classical_optimizer: SLSQP
    max_iterations: 50
    convergence: gradient and energy (both required)
    convergence_marker: * after a |grad| or dE that meets its threshold
    gradient_method: analytic
    convergence_gradient: 0.001
    gradient_units: Ha/rad
    pool: fermionic
    pool_class: fermionic excitations
    pool_size: 3
    reoptimize_all_parameters: True
    state_vector_backend: particle-number sector, 4 states of 2^4
    device: AER_simulator
    backend_provider: none
    circuit_execution: False
    shots: 0 (exact expectation values)
    energy_unit: eV
    convergence_energy_eV: 0.001
    reference_energy_eV: -25.6314952746
```

- `classical_optimizer` is the `optimizer["method"]` actually used (SPSA by default
  when `shots > 0`).
- `max_iterations` is the growth budget (`max_iterations=`) for the ADAPT methods and
  `max_steps` for `mcas-vqe` and `valqa`. A fixed-ansatz `vqe` run still prints the
  growth budget, which it does not use.
- `gradient_method` reads `analytic` in every run, including a shot-based one; the
  measured gradient is described by `[GRADIENT MEASUREMENT]`. `gradient_units` is the
  unit of the screening gradient, Ha/rad.
- `convergence` names the criteria that stop the growth (`adapt-vqe` only), with the
  thresholds beside it: `convergence_gradient` in Ha/rad and `convergence_energy_<unit>`
  in the unit of the log. The table below marks a value that meets its threshold with
  `*`, and so does `final_max_gradient` in the summary.
- `pool` and `pool_size` name the operator pool and its size; `pool_class` is what the
  native engine says that pool generates.
- `reoptimize_all_parameters` is always `True`: the native solver re-optimizes every
  parameter after each growth step. `state_vector_backend` says how the state is held.
- `device`, `backend_provider`, `circuit_execution` and `shots` say where the energies
  came from (`backend_provider` is `none` without a provider). `shots` reads
  `0 (exact expectation values)` or `N (sampled expectation values)`.
- `reference_energy_*` is the energy of the starting state (the Hartree-Fock
  determinant), in `energy_unit`; with `shots > 0` it is a sampled value. It is
  written here and nowhere else.
- A run restarted with `resume=` closes the block with its lineage:
  `resumed_from`, `restored_operators`, `restored_parameters`, `restored_energy_<unit>`
  and `resume_same_hamiltonian: True`. The native resume raises when the Hamiltonian
  differs, so `True` is a checked fact. The first `dE` of the table is measured from
  the restored energy.

## `[ITERATIONS]`

One row per accepted ADAPT-VQE operator (LiH, active space of two orbitals):

```text
[ITERATIONS]
    iter    time (HH:MM:SS)           energy (eV)            dE (eV)      |grad| (eV)      steps         1q       cnot      depth    operator
    -----------------------------------------------------------------------------------------------------------------------------------------
       1           19:00:13       -102.7806034553      -0.0081417878         1.987758          3         38         48         65    D(0,2->1,3)
       2           19:00:13       -102.7806993799      -0.0000959246         0.114538          6         43         52         71    S(0->1)
       3           19:00:13       -102.7807889875      -0.0000896075         0.110659          8         49         56         73    S(2->3)
```

| Column | Meaning |
| --- | --- |
| `iter` | Growth step. |
| `time` | The clock when the row was written (see above). |
| `energy` | Energy after the whole ansatz was re-optimized. |
| `expr` | Expressivity of the ansatz (KL divergence from the Haar distribution). Absent unless you call `run(log_expressivity=True)` (`adapt-vqe` without shots): it samples state fidelities at every row. |
| `dE` | Change from the previous row (the first row: from the reference energy, or from the restored energy of a resumed run). |
| `\|grad\|` | Screening gradient of the operator that was chosen, converted to eV. The setup block states the same quantity in Ha/rad. |
| `steps` | Parameter updates the optimizer took at this step. |
| `1q`, `cnot`, `depth` | Gate counts and depth of the ansatz compiled after the step. |
| `operator` | The label of the chosen operator, such as `D(0,2->1,3)` (a double) or `S(0->1)` (a single). A pool whose labels are too long to read prints `pool[index]` instead. |

The screening gradient of operator $A_k$ is $2\,\mathrm{Re}\langle\psi|\hat H A_k|\psi\rangle$,
the quantity the [ADAPT-VQE theory page](../theory/adapt_vqe.md) derives; the table
prints its magnitude for the largest.

## `[MARKOV CHAIN]`

For `mcas-vqe` and `valqa`, one row per proposal, accepted or not:

```text
[MARKOV CHAIN]
     step    time (HH:MM:SS)       move      L       energy (eV)       dE (eV)      current (eV)        T (eV)        ln(a)     acc      steps    action
    ----------------------------------------------------------------------------------------------------------------------------------------------------
        1           18:59:58     insert      1        -26.489012     -0.433007        -26.489012     1.000e-01       +3.245     yes          -    insert@0
        2           18:59:58    replace      1        -26.056005     +0.433007        -26.489012     2.154e-02      -19.412      no          -    replace@0
```

`move` is the edit proposed (`insert`, `delete`, `replace`, `swap`), `L` the length of
the proposed ansatz, `energy` its relaxed energy, `dE` the change from the chain's
current state, `current` the chain's energy after the decision, `T` the temperature,
`ln(a)` the log acceptance ratio, and `acc` the decision. `steps` is always `-`:
the per-proposal optimizer steps are not written.

## `[VARIATIONAL QUANTUM SUMMARY]`

The converged state of a quantum run.

```text
========================================================================
[VARIATIONAL QUANTUM SUMMARY]
    converged: True
    optimal_energy_eV: -26.0180254756
    correlation_energy_eV: -0.3865302010
    num_operators: 1
    num_parameters: 1
    final_max_gradient: 5.967449e-15 *
    cost_evaluations: 6
    optimizer_steps: 4
    termination: pool gradients converged
========================================================================
```

`converged` is the outer verdict. A finished Markov chain does not certify an
eigenstate, so for `mcas-vqe` and `valqa` it can read `False` after a complete run
(`result.completed` is the separate fact that the chain ran to its end). `correlation_energy` is
`optimal_energy` minus the `reference_energy` of `[OPTIMIZATION SETUP]`.
`final_max_gradient` is the largest pool gradient after the last step, in Ha/rad, so
it differs in unit from the table's `|grad|` column; `*` marks it when it meets the
threshold. `termination` is the reason the optimizer or the growth loop stopped
(`pool gradients converged`, `growth budget exhausted`, ...), as the native solver
reports it. `cost_evaluations` counts optimizer objective calls, and
`optimizer_steps` the parameter updates summed over the run.

## `[SCF SUMMARY]`

The block that closes a classical run, after `[SCF SETUP]` and `[SCF ITERATIONS]`:

```text
========================================================================
[SCF SUMMARY]
    converged: True
    energy_eV: -25.87060596308835
========================================================================
```

Whether the SCF converged is the `converged` line; the iterations are in
`[SCF ITERATIONS]` above it.

A periodic run with `hubbard_u=` adds the on-site DFT+U correction to this block, and
states it nowhere else: `hubbard_energy_eV` (the correction `E_U`, already inside
`energy_eV`) and one `hubbard_<element>_<shell>` line per element with its `U`, `J` and
`U_eff`:

```text
    hubbard_energy_eV: 2.4217827594
    hubbard_Ni_d: U 3.000000 eV, J 0.000000 eV, U_eff 3.000000 eV
```

A run without the correction writes neither line. With `atomic_units=True` the unit is `Ha`.

## `[PERFORMANCE]`

Written at the end of every ASE calculation, after `[FORCES]` when forces were asked
for, so the nuclear gradient (usually the largest stage of a relaxation step) is
inside it. A direct `run()` writes none. Forces requested on a result that already
exists write a second block that holds only the gradient stage.

```text
[PERFORMANCE]
    step: 1
    stages (wall-clock seconds):
        integration: molecular integrals                     0.0009
        mean-field SCF (builds the two-body integrals)       0.0047
        parameter optimization                               0.0000
        nuclear gradient (forces)                            0.0540
        -----------------------------------------------------------
        sum of timed stages                                  0.0596
    untimed_s: 0.4153
    wall_time_s: 0.4749
    solver_wall_time_s: 0.0000
    integration_backend: Rust (2 threads)
    threads: 2
    cpu_count: 8
    mpi: not used (single process, shared-memory scoped threads)
    peak_memory_MiB: 194.7
========================================================================
```

The stages are clocks around the native calls. The native integrals are lazy, so the
two-body integrals are built by the first call that needs them and are part of the SCF
stage; `parameter optimization` is the whole solve, pool-gradient screening included.
`untimed_s` is what no stage accounts for (building the Hamiltonian, the mapping, the
circuit metrics, writing the log). `solver_wall_time_s` is the native solver's own
clock, next to `wall_time_s`. `peak_memory_MiB` is the process's peak resident size. A classical run has no
`solver_wall_time_s`.
On a processor, `qpu_*` lines account for the jobs that ran there. The block is
timing data: the stdout and the file carry the same report up to it.

(run-log-forces)=
## `[FORCES]`

Written each time forces are computed, headed with the geometry `step`. The `scheme`
line names the method that produced them. For an all-electron DFT force:

```text
[FORCES]
    step: 1
    units: eV/Angstrom
    convention: forces = -dE/dR with its mean translation removed (ASE sign)
    scheme: Analytic-Hellmann-Feynman-Pulay
    forces:
         atom symbol                x                y                z            |F|
        ------------------------------------------------------------------------------
            1 H           -0.00000000       0.00000000      -8.38811595     8.38811595
            2 H            0.00000000      -0.00000000       8.38811595     8.38811595
    max_force: 8.38811595
    rms_force: 8.38811595
    net_force: 0.00000000
```

Other schemes add their own lines after `scheme`:

| `scheme` | Method | Extra lines |
| --- | --- | --- |
| `Analytic-Hellmann-Feynman-Pulay` | all-electron HF and Kohn-Sham | none |
| `Richardson-fixed-density-PAW` | PAW-LCAO RHF, UHF and semilocal Kohn-Sham | `step_bohr`, `step_error_estimate` |
| `Central-reduced-Hamiltonian` | quantum methods | `energy_ha`, `directions`, `step_bohr`, `step_error_estimate` (`None` for two-point differences), `orbital_response` |

`net_force` is the sum of the force vectors, which a free molecule should not feel:
it measures the discretization error. For one atom it is left out. The `convention`
line states the sign and the translation projection that were applied.

## `[GEOMETRY OPTIMIZATION SUMMARY]` and `[RELAXATION COMPLETE]`

A relaxation driven by an ASE optimizer writes one set of blocks per geometry. Close
the log by calling the calculator after the optimizer has finished:

```python
from ase.optimize import BFGS

optimizer = BFGS(atoms, logfile=None)
optimizer.run(fmax=0.05, steps=50)
atoms.calc.write_optimization_summary(atoms, fmax=0.05, optimizer=optimizer)
```

It appends the trajectory as one block, and a footer that says whether the optimizer
converged. A log without the footer was cut off, or never closed.

```text
[GEOMETRY OPTIMIZATION SUMMARY]
    geometry_steps: 2
    units: angstrom
    initial_energy_eV: -26.0560050388
    final_energy_eV: -26.8002051204
    energy_change_eV: -0.7442000816
    initial_max_force: 6.43515587
    final_max_force: 0.80311623
    final_net_force: 0.00000000
    convergence:
         step          energy (eV)        max force        net force
        ------------------------------------------------------------
            1       -26.0560050388       6.43515587       0.00000000
            2       -26.8002051204       0.80311623       0.00000000
    relaxed_geometry:
        H       3.0000000000     3.0000000000     2.5080692019
        H       3.0000000000     3.0000000000     3.4919307981
========================================================================

[RELAXATION COMPLETE]
    status: converged after 2 geometry steps (max force 0.803116, threshold 2.000000 eV/Angstrom)
========================================================================
```

The method returns `False` and writes nothing when there is no log, fewer than two
steps (unless you pass `force=True`), or the summary was already written.

A script that never calls it still gets a footer, written at interpreter exit, but one
that does not claim a verdict: reaching exit says the process ended, not that the
optimization finished or converged.

```text
[RELAXATION COMPLETE]
    status: completion not signaled: the process exited after 3 geometry step(s), which does not establish that the optimization finished or converged (call write_optimization_summary(optimizer=...) to record the verdict)
```

## One owner per fact

Each fact has one block that owns it. A reader that wants a number takes it from the
owner.

| Fact | Owner |
| --- | --- |
| Program version, machine, Python and package versions | Banner |
| Geometry, cell, periodicity | `[SYSTEM]` |
| The basis that ran, its number of functions, the storage precision | `[BASIS]` |
| Grid, mapping, particle numbers, orbitals and qubits of the register | `[ELECTRONS]` |
| What the mean-field reference stops on, and its iterations | `[SCF SETUP]`, `[SCF ITERATIONS]` |
| Which orbitals were frozen, active or deleted | `[ACTIVE SPACE]` |
| Optimizer, thresholds, growth budget, pool, device, shots, reference energy, lineage of a resumed run | `[OPTIMIZATION SETUP]` |
| Cost of measuring the pool gradients | `[GRADIENT MEASUREMENT]` |
| Per-step energy, gradient and circuit cost | `[ITERATIONS]` |
| Per-proposal energy and decision | `[MARKOV CHAIN]` |
| Final variational energy, operator and parameter counts, termination reason | `[VARIATIONAL QUANTUM SUMMARY]` |
| Final mean-field energy and convergence | `[SCF SUMMARY]` |
| The DFT+U correction (`hubbard_u=`): its energy, and `U`, `J` and `U_eff` per element | `[SCF SUMMARY]` |
| Force vectors and their convention | `[FORCES]` |
| Stage timings, wall times, peak memory | `[PERFORMANCE]` |
| Trajectory, relaxed geometry, verdict | `[GEOMETRY OPTIMIZATION SUMMARY]`, `[RELAXATION COMPLETE]` |

`reference_energy_*` is written only in `[OPTIMIZATION SETUP]`, and the orbital count
is said once per meaning: `basis_functions` in `[BASIS]`, the orbitals of the register
in `[ELECTRONS]`, and the fate of each orbital in `[ACTIVE SPACE]`.

(run-log-not-written)=
## Still not written

The older manual described blocks and lines that the calculator does not write. Do
not look for them in a log, and do not parse for them:

- The tetris, pruning and screening lines of `[OPTIMIZATION SETUP]`: the native solver
  has no such options.
- The `time` column of `[SCF ITERATIONS]`: the native SCF records no clock. Likewise
  `[PERFORMANCE]` times the native solve as one stage (`parameter optimization`), and
  the log is written when the run ends, not while the SCF iterates.
- `[MEASUREMENT]` and the quantum-echo and nested-OTOC helper blocks. The helper
  functions `append_block`, `append_quantum_echoes` and `append_nested_otoc` are still
  exported from `mandacaru.utils`, but no calculation calls them.
- `[SCF SETUP]` and `[SCF ITERATIONS]` of a periodic (crystal) calculation: its log holds
  `[SYSTEM]`, `[BASIS]`, `[ELECTRONS]` and `[SCF SUMMARY]` only.

(references-bib)=
### `references.bib`

An earlier version wrote a BibTeX file of the references a run used, beside the log.
That file is no longer written: the calculator has no `references=` option. The
reference table and its BibTeX renderer are still in the package, so a script can
assemble the entries it needs:

```python
from mandacaru.utils import REFERENCES, bibtex

print(len(REFERENCES))
```
