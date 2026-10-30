(run-log-reference)=
# The run log

A calculation can describe itself in a plain-text log: a fixed set of blocks of
`key: value` lines, readable by eye and by a line scanner. This page lists the
blocks that the calculator writes today, with a real excerpt of each, and says which
block owns which fact. Everything below was produced by running the calculator on
H2 and LiH, with `txt=` and `trace=True`. [Blocks documented earlier but not written
now](#run-log-not-written) are listed at the end.

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
  (`system`, `basis`, `electrons`, `setup`, `iterations`, `summary`, `forces` and a
  `steps` list for a multi-step file):

```python
from mandacaru.utils import parse_output

blocks = parse_output("adapt.log")
print(blocks["summary"]["optimal_energy_eV"])
```

A periodic calculation (a crystal) cannot be logged yet: with `txt=` or
`trace=True` the energy is computed and then the run raises `AttributeError` while
the log is assembled. Leave both off for a crystal.

## Which blocks a run writes

| Block | Written by | When |
| --- | --- | --- |
| Banner | every run, once per file | Top of a freshly opened file only |
| `[SYSTEM]` | every run | Always |
| `[BASIS]` | every run | Always |
| `[ELECTRONS]` | every run | Always |
| `[ACTIVE SPACE]` | quantum methods | Only when `active_space=` reduced the register |
| `[GRADIENT MEASUREMENT]` | quantum methods | Only for a measured-gradient run (`shots > 0` with `adapt-vqe`) |
| `[OPTIMIZATION SETUP]` | quantum methods | Always for them |
| `[ITERATIONS]` | `adapt-vqe` | One row per accepted operator |
| `[MARKOV CHAIN]` | `mcas-vqe`, `valqa` | One row per proposal |
| `[VARIATIONAL QUANTUM SUMMARY]` | quantum methods | Always for them |
| `[SCF SUMMARY]` | `rhf`, `uhf`, `ghf`, `dft` | Always for them |
| `[FORCES]` | any method | Each time forces are computed |
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
from, `pbc` says which axes repeat. `initial_magnetic_moments` always reads
`(not provided)` today, even when you set moments on the `Atoms` object; the spin
populations that result are in `[ELECTRONS]`.

## `[BASIS]`

```text
[BASIS]
    name: HAO
    spatial_orbitals: 2
```

`name` is the `basis` option as you gave it; a dictionary (`{"name": "PAW-LCAO",
"size": "SZ"}`) is printed as that dictionary, not resolved into its cutoffs or
dataset files. `spatial_orbitals` is the orbital count of the register the quantum
or mean-field problem uses. With an active space it is the active count, not the
size of the basis (`source_orbitals` in `[ACTIVE SPACE]` is that).

## `[ELECTRONS]`

```text
[ELECTRONS]
    mapping: jordan_wigner
    n_qubits: 4
    num_particles: (1, 1)
```

The fermion-to-qubit mapping, the register width, and the particle numbers
`(alpha, beta)`. Classical runs write the same three lines; `mapping` and `n_qubits`
then describe the register a quantum method would use. Above 20 qubits a line
`operator_details: omitted (...)` follows and the operator column of the iteration
table reads `(omitted)`.

## `[ACTIVE SPACE]`

Written between `[ELECTRONS]` and `[OPTIMIZATION SETUP]` when `active_space=` is set.
For LiH with `active_space={"frozen": "auto", "orbitals": 2}`:

```text
[ACTIVE SPACE]
    method: energy
    spatial_orbitals: 2
    frozen_orbitals: 1
    source_orbitals: 3
    orbitals:
        index  role
        -------------
        0      frozen
        1      active
        2      active
```

`method` is the selector (`energy` when none is named, `mp2` for natural orbitals),
and the table gives each source orbital's fate: `frozen`, `active` or `deleted`. The
selector's occupation numbers and the reference occupancy are not in the table.

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
    gradient_method: analytic
    gradient_units: Ha/rad
    pool: fermionic
    pool_size: 3
    device: AER_simulator
    circuit_execution: False
    shots: 0 (exact expectation values)
    energy_unit: eV
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
- `pool` and `pool_size` name the operator pool and its size.
- `device`, `circuit_execution` and `shots` say where the energies came from. With a
  circuit provider a line `backend_provider: qiskit` (or the provider's name) follows
  `device`. `shots` reads `0 (exact expectation values)` or `N (sampled expectation
  values)`.
- `reference_energy_*` is the energy of the starting state (the Hartree-Fock
  determinant), in `energy_unit`; with `shots > 0` it is a sampled value.

The convergence thresholds (`convergence=`) are not written, so the table below
carries no threshold marker.

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
| `dE` | Change from the previous row (the first row: from the reference energy). |
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
    reference_energy_eV: -25.6314952746
    correlation_energy_eV: -0.3865302010
    num_operators: 1
    num_parameters: 1
    final_max_gradient: 5.967449e-15
    cost_evaluations: 6
    optimizer_steps: 4
========================================================================
```

`converged` is the outer verdict. A finished Markov chain does not certify an
eigenstate, so for `mcas-vqe` and `valqa` it can read `False` after a complete run
(`result.completed` is the separate fact that the chain ran to its end). `correlation_energy` is
`optimal_energy` minus `reference_energy`. `final_max_gradient` is the largest pool
gradient after the last step, in Ha/rad, so it differs in unit from the table's
`|grad|` column. `cost_evaluations` counts optimizer objective calls, and
`optimizer_steps` the parameter updates summed over the run.

## `[SCF SUMMARY]`

The only block a classical run writes after `[ELECTRONS]`:

```text
========================================================================
[SCF SUMMARY]
    converged: True
    energy_eV: -25.87060596308835
========================================================================
```

No iteration table is written. Whether the SCF converged is the `converged` line.

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

## One owner per fact

Each fact has one block that owns it. A reader that wants a number takes it from the
owner.

| Fact | Owner |
| --- | --- |
| Program version, machine, Python and package versions | Banner |
| Geometry, cell, periodicity | `[SYSTEM]` |
| Basis as requested; orbital count of the register | `[BASIS]` |
| Mapping, register width, particle numbers | `[ELECTRONS]` |
| Which orbitals were frozen, active or deleted | `[ACTIVE SPACE]` |
| Optimizer, growth budget, pool, device, shots, reference energy | `[OPTIMIZATION SETUP]` |
| Cost of measuring the pool gradients | `[GRADIENT MEASUREMENT]` |
| Per-step energy, gradient and circuit cost | `[ITERATIONS]` |
| Per-proposal energy and decision | `[MARKOV CHAIN]` |
| Final variational energy, operator and parameter counts | `[VARIATIONAL QUANTUM SUMMARY]` |
| Final mean-field energy and convergence | `[SCF SUMMARY]` |
| Force vectors and their convention | `[FORCES]` |
| Trajectory, relaxed geometry, verdict | `[GEOMETRY OPTIMIZATION SUMMARY]`, `[RELAXATION COMPLETE]` |

Two repeats exist today that this rule forbids: `reference_energy_*` is also written
in `[VARIATIONAL QUANTUM SUMMARY]`, and the orbital count appears in both `[BASIS]`
and `[ACTIVE SPACE]`. Treat `[OPTIMIZATION SETUP]` and `[ACTIVE SPACE]` as the
owners.

(run-log-not-written)=
## Documented earlier, not written now

The older manual described blocks and lines that the calculator does not write. Do
not look for them in a log, and do not parse for them:

- `[PERFORMANCE]` (stage timings, memory).
- The full `[ELECTRONS]` block (grid, kinetic operator, k-points, spin, Hamiltonian
  size and thirteen fields in all); only mapping, register width and particle
  numbers are written.
- `[SCF SETUP]` and `[SCF ITERATIONS]`: no mean-field iteration is logged.
- The lineage of a resumed run (`resume=` writes nothing about where it restarted).
- The termination reason of the optimizer or the growth loop; read it from the
  result object.
- `pool_class` in the setup block, the convergence thresholds and the `*` marker
  on a met threshold, and the `expr` (expressivity) column.
- The initial magnetic moments, although the `[SYSTEM]` line exists.
- `[MEASUREMENT]` and the quantum-echo, nested-OTOC and performance helper blocks.
  The helper functions `append_block`, `append_performance`, `append_quantum_echoes`
  and `append_nested_otoc` are still exported from `mandacaru.utils`, but no
  calculation calls them.
- Logs of periodic (crystal) calculations; see above.

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
