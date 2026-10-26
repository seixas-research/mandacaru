# Reading a run

A variational run reports itself in three ways: **one structured report**, two
optional **JSON dumps** for the objects that are too large to print, and a
**`references.bib`** of the methods it used.

## What goes where

There is one report -- the blocks described below -- and `txt=` decides *where*
it goes:

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
`"rhf"`, `"uhf"`, `"ghf"`, `"dft"`, `"vqe"` (including `ansatz="hva"`), `"adapt-vqe"`,
`"mcas-vqe"`, `"valqa"` and `"subspace-adapt-vqe"` — and refused with a message
by the others rather than leaving an empty file. `"subspace-vqe"` retains its
console report.

There is one standard output per process, so consecutive printed runs number
their blocks `step: 1`, `step: 2`, … just as consecutive runs sharing one
`txt=` file do. {func}`~mandacaru.utils.logging.reset_log` with
{data}`~mandacaru.utils.logging.STDOUT` restarts the count.
```

## The iteration table

The block every run is read through, whichever destination it went to:

```text
[ITERATIONS]
    iter    time (HH:MM:SS)           energy (eV)            dE (eV)      |grad| (eV)      steps         1q       cnot      depth    operator
    -----------------------------------------------------------------------------------------------------------------------------------------
       1           13:45:10        -33.1720283176      -0.1939118279         6.144385          3         38        104        121    D(0,4->3,7)
       2           13:45:10        -33.2886620234      -0.1166337058         4.320978          4         73        176        209    D(0,4->1,7)
       3           13:45:10        -33.3953428910      -0.1066808676         4.127015          5        108        256        304    D(0,4->3,5)
       4           13:45:10        -33.4961593720      -0.1008164810         3.648094          7        143        304        368    D(0,4->1,5)
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
| `dE` | Energy change from the previous row, signed, in the energy column's unit; the first row's is from the reference state (`reference_energy_<unit>` in `[OPTIMIZATION SETUP]`), and a resumed run's from the energy its restored ansatz had. A `*` follows a value whose magnitude is below `convergence_energy_<unit>`. |
| `\|grad\|` | Largest pool gradient, to six decimals; the operator with this gradient is the one selected. It is compared against `convergence_gradient`, and a `*` follows a value below it. A gradient below 5e-7 reads `0.000000`. |
| `steps` | **Steps the classical optimizer took** to re-optimize the grown ansatz — parameter updates, not cost evaluations. The two differ by the method: L-BFGS spends several evaluations per step on a finite-difference gradient and a line search, SPSA two or three, while COBYLA evaluates once per trial point. `-` when a method reports neither a count nor a per-iteration callback. |
| `cnot` | CNOT gates after compiling to the native gate set. |
| `1q` | Single-qubit gates in the same compilation. |
| `depth` | Circuit depth in the same compilation. |
| `operator` | The selected operator's label, e.g. `D(0,4->3,7)`; the label names its kind, and the pool is named in `[OPTIMIZATION SETUP]`. A CEO operator is shown by a short name: a single excitation as itself (`S(0->4)`), a coupled pair as its spin-orbital set and the sign of the combination (`CEO(0,1,6,7)+`). A same-spin set carries three excitations, so the pair is named by letters in the order its full label lists them (`CEO(0,1,2,3)a-c`). The full label, `CEO[o0,o1,o6,o7]{D(0,6->1,7)+D(0,7->1,6)}`, stays on the result, in checkpoints and in `pool.json`. |

**The table does not depend on the terminal.** It is the same width whether it
is printed or written, because it is the same table: no column is dropped and no
operator label is cut at the terminal's width (only the CEO pool's labels have a
fixed short form, above), so a narrow terminal wraps a row rather than
quietly losing a number. `expr` is the one optional column — absent unless
`run(log_expressivity=True)` asked for it, rather than present and blank.

The circuit columns need `profile=True` (the default); with `profile=False`
they read `-`.

## The Markov-chain table

`method="mcas-vqe"` and `method="valqa"` write the same blocks, with one
difference: their
`[OPTIMIZATION SETUP]` holds the chain's settings (moves, temperature, length
penalty, seed, `replace_start`), and `start`, which says where the chain
began: the empty ansatz, or the previous geometry's ansatz with `transfer=True`,
and the reason when a transfer fell back to empty. With `transfer=True`,
`max_steps` also shows the length of a chain that starts from a transferred
ansatz. For `method="valqa"` the `proposal` line also says which model drew
the operators and, with `update_between_geometries=True`, what the update
before this chain did (the same text as the result's `model_update`). The
summary adds `edit_distance_from_start`, the number of insert,
delete and replace moves between the starting architecture and the reported
one. In place of `[ITERATIONS]` it writes `[MARKOV CHAIN]`, one
row per proposal, accepted or not:

```text
[MARKOV CHAIN]
     step    time (HH:MM:SS)       move      L       energy (eV)       dE (eV)      current (eV)        T (eV)        ln(a)     acc      steps    action
    ----------------------------------------------------------------------------------------------------------------------------------------------------
        4           05:35:17     insert      2        -27.958716     -0.501456        -27.958716     2.848e-02      +18.383     yes          4    insert(QD(0,2->1,3)@1)
        5           05:35:17     delete      1        -27.457260     +0.501456        -27.958716     1.874e-02      -27.519      no          1    delete(1)
```

| column | meaning |
| :--- | :--- |
| `move`, `action` | the proposed move, and its position and operator |
| `L` | operators in the proposed ansatz |
| `energy` | relaxed energy of the **proposed** ansatz |
| `dE` | proposed minus **current** energy before the step, without the length penalty |
| `current` | the chain's energy after the decision: `energy` if accepted, unchanged if not |
| `T` | architecture temperature |
| `ln(a)` | log Metropolis-Hastings ratio, before $\min(0,\cdot)$ |
| `acc` | whether the proposal was accepted |
| `steps` | optimizer parameter updates of the relaxation (`-` for the empty ansatz) |

The summary reports the lowest-cost ansatz evaluated, its operator sequence
(`operators:`), the lowest energy and the chain's final state, and the
acceptance of each move. It has no `converged:` line, because a chain stops on
its step budget, not on a criterion.
{func}`~mandacaru.utils.logging.parse_output` returns the rows under
`"markov_chain"`.

The clock column is `time (HH:MM:SS)`. Energies, including `dE` and `T`,
carry parenthesized units; the architecture temperature is an energy,
not a temperature in Kelvin. All columns have at least four spaces between
them. Actions exceeding 40 characters or containing control characters use
`step[index]`; the result retains each step's full move and architecture.

`method="subspace-adapt-vqe"` uses the same `[ITERATIONS]` table as ADAPT-VQE,
including eV screening gradients, compact operator names and `txt=` support.
Its setup records `num_states` and `state_weights`. The table's energy is
the lowest subspace energy; the screening gradient belongs to the weighted
subspace objective. The summary reports the ground energy as `optimal_energy`
and each excited level as `excited_state_<index>_energy_<unit>`.

## A fixed ansatz

`method="vqe"` writes the same blocks with no progress table: its ansatz is
fixed before the run, so there is no growth step or proposal to give a row to,
only one classical minimization. Its `[OPTIMIZATION SETUP]` names the ansatz,
its own configuration (the layers and grouping of `ansatz="hva"`) and where the
parameters start; the summary gives the optimized energy, `num_parameters`,
`cost_evaluations` and `optimizer_steps`, and has no `num_operators` line.

```text
[OPTIMIZATION SETUP]
    classical_optimizer: SLSQP
    ansatz: UCCSD
    initial_parameters: all zero (the reference state)
    reoptimize_all_parameters: True
    ...
    energy_unit: eV
    reference_energy_eV: -187.7936353791

========================================================================
[VARIATIONAL QUANTUM SUMMARY]
    converged: True
    optimal_energy_eV: -188.3280359894
    reference_energy_eV: -187.7936353791
    correlation_energy_eV: -0.5344006102
    num_parameters: 8
    cost_evaluations: 76
    optimizer_steps: 8
========================================================================
```

Every method renders these blocks with the one
{class}`~mandacaru.utils.logging.Logger`; a method's own table
({meth}`~mandacaru.utils.logging.Logger.write_iteration`,
{meth}`~mandacaru.utils.logging.Logger.write_chain_step`) is the only part that
differs.

## A classical mean field

`method="rhf"`, `"uhf"`, `"ghf"` and `"dft"` write the same `[SYSTEM]` and `[BASIS]`
blocks, a classical `[ELECTRONS]` -- grid, charge, spin, orbitals and electrons,
with no reference state, mapping, register or qubit Hamiltonian, since none is
built -- and then `[SCF SETUP]`, `[SCF ITERATIONS]` and `[SCF SUMMARY]`.
RHF, UHF and GHF retain the history of the selected lowest-energy start,
with each iteration's original clock time and density residual. A two-stage
direct RHF run retains both its coarse and exact stages in that history:

```text
[SCF SETUP]
    scf_method: restricted Hartree-Fock (closed shell)
    max_iterations: 200
    convergence_Hartree: 1e-09 (energy change and largest density change)
    acceleration: DIIS; level shift 0.5 Hartree after the first energy rise
    scf_history: selected lowest-energy start; original completion times
    energy_unit: eV

[SCF ITERATIONS]
     iter    time (HH:MM:SS)    electronic_energy (eV)               dE (eV)        residual
    ----------------------------------------------------------------------------------------
        1           23:57:55            -44.2307330029                     -       1.332e-15
        2           23:57:55            -44.2307330029         +0.0000000000       0.000e+00

========================================================================
[SCF SUMMARY]
    converged: True
    optimal_energy_eV: -24.7717526266
    electronic_energy_eV: -44.2307330029
    ion_ion_energy_eV: 19.4589803763
    one_center_energy_eV: 0.0000000000
    homo_energy_eV: -14.466156
    lumo_energy_eV: 14.271960
    homo_lumo_gap_eV: 28.738116
    scf_iterations: 2
========================================================================
```

The summary's method-specific lines are the RHF frontier orbitals, the UHF
natural-orbital reference energy and spin contamination, and the GHF Kramers
pairing (`spin_orbit_coupling` is in the GHF setup).
{func}`~mandacaru.utils.logging.parse_output` reads `[SCF SETUP]` under
`"setup"` and `[SCF SUMMARY]` under `"summary"`, the keys a variational run's
blocks are read under.

`method="dft"` writes the same blocks with Kohn-Sham content
({doc}`dft`). `[SCF SETUP]` adds `xc_functional`, `vdw` (`NONE` when
there is none) and, for a molecule, `hartree` (a Poisson solve per iteration,
or the two-body tensor with `hartree="tensor"`), and states the convergence criteria and the level shift of the
Kohn-Sham iteration. `[SCF SUMMARY]` adds the `reference_energy_<unit>` of the
exported Hamiltonian, `hartree_energy_<unit>` and `xc_energy_<unit>`, then
`core_correction_energy_<unit>` and `dispersion_energy_<unit>`, including
zero contributions. Molecular summaries also report `electronic_energy`,
`ion_ion_energy` and `one_center_energy` (each suffixed by its unit), plus
`other_constant_energy` when present, so the total reads as their sum.
Hartree and XC are already included in the electronic contribution.
A spin-unrestricted molecule also reports `stability_check` in `[SCF SETUP]`
and the verdict `stability` (plus `stability_energy_lowering_<unit>` when a
lower state was found) in `[SCF SUMMARY]`.
A hybrid (`xc="hse06"`) adds `exact_exchange` to `[SCF SETUP]` (the
fraction and the range separation) and `exact_exchange_energy_<unit>` to
`[SCF SUMMARY]`: the part of `xc_energy_<unit>` the exact exchange carries,
not a further term of the sum. A crystal's summary holds no energy terms, so
a hybrid crystal adds only the `[SCF SETUP]` entry. A molecule on a
spin-orbit set (two-component Kohn-Sham) says so in `scf_method` and adds
`spin_orbit_coupling: True` to `[SCF SETUP]`; its `[SCF SUMMARY]` carries the
GHF-style `kramers_pairing_Hartree`, `spin_orbit_energy_<unit>` (the energy
minus that of the collinear solution the SCF started from: a change, not a
term of the sum) and `spin_moment_bohr_magneton`, the spin moment vector
`(Mx, My, Mz)`, $2\langle\mathbf S\rangle$ (the spin part only: the
orbital moment is not computed). A noncollinear start (`(N, 3)` initial
magnetic moments, written as one triple per atom in `[SYSTEM]`) runs the
same solver; on a scalar set it has `spin_orbit_coupling: False`, and
`noncollinear_energy_<unit>` takes the place of `spin_orbit_energy_<unit>`
as the change against the collinear start. A two-component crystal (a
spin-orbit set, or `(N, 3)` moments) does the same: `scf_method` says it,
`spin_orbit_coupling` joins `[SCF SETUP]`, whose `spin` reads
`two-component, non-magnetic` or `two-component, magnetic (initial moments
...)` and whose `k_points_irreducible` reads `(the full mesh, unreduced)` for
a magnetic one; its `[SCF SUMMARY]` adds `spin_orbit_energy_<unit>` (or
`noncollinear_energy_<unit>`), against the collinear crystal it started from
on the same mesh, and the per-cell `spin_moment_bohr_magneton` in place of
`magnetic_moment_bohr_magneton`.

Between setup and summary, every mean-field run writes `[SCF ITERATIONS]`, one row per
iteration, so the convergence can be followed and diagnosed:

```text
[SCF ITERATIONS]
     iter    time (HH:MM:SS)        free_energy (eV)               dE (eV)        residual    moment (muB)
    ------------------------------------------------------------------------------------------------------
        1           13:45:10         -366.8600438496                     -       6.872e+00        0.000001
        2           13:45:11         -366.8962799616         -0.0362361119       5.212e+00        0.000000
       19           13:45:12         -366.9776701675         +0.0000000009       7.906e-06        0.000000
```

`time (HH:MM:SS)` is the local completion time, also preserved when a history
is replayed. Elapsed seconds remain in each solver's history.
`electronic_energy (<unit>)` is the molecular electronic energy (the summary
adds its constants); `free_energy (<unit>)` is the free energy of a crystal.
`dE (<unit>)` is its change (`-` on the first row);
`residual` what the convergence test measures -- the largest density-matrix
change of a molecule, the density residual (electrons) of a crystal; and a
spin-polarized crystal adds `moment (muB)`, the moment per cell (for a magnetic
two-component crystal the length of its spin moment vector).
`parse_output` returns the rows under `"scf_iterations"`, one dict per row
keyed by the column headings.

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
                 grid_spacing=0.25,
                 txt="output.txt")
```

### A geometry optimization writes one log

A relaxation runs one complete ADAPT-VQE per geometry, so its log is a sequence
of blocks -- energies then forces, one pair per step -- in one file:

```text
<banner: version, host, interpreter, dependencies>       (once, at the top)
========================================================================
    ADAPT-VQE (CEOPool)
========================================================================
[SYSTEM]                        step: 1, this geometry, its cell and spins
[BASIS]                         the basis that ran: options, radii, datasets
[ELECTRONS]                     grid, reference, mapping, register, Hamiltonian
[ACTIVE SPACE]                  only if reduced: selector, occupations, orbitals
[OPTIMIZATION SETUP]            the optimizer, the gradient, the operator pool
[ITERATIONS]                    one row per grown operator
[VARIATIONAL QUANTUM SUMMARY]   the converged state of *this* geometry
[FORCES]                        step: 1, the vectors and their breakdown
[PERFORMANCE]                   step: 1, where the time and memory went
========================================================================

========================================================================
    ADAPT-VQE (CEOPool) -- geometry step 2
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
    zeta_split: first_zeta: tail norms 0.16, 0.3, 0.6 (norm of the tail, every zeta split from the first)
    projector_basis: raw
    energy_shift: 0.1 eV
    confinement_potential: A exp(-(r_c - r_i)/(r - r_i)) / (r_c - r), A = 12 Ha, r_i = 0.6 r_c
    polarization_method: gaussian (quasi-Gaussian: r^l [exp(-r^2/r_char^2) - (a - b r^2)], further shells split from it)
    filter_cutoff: 16.6244 Bohr^-1 (3760.30 eV), filtered (auto: 1 x Nyquist)
    local_potential: range-separated: long range on the grid, short range on atom-centered quadrature (sigma = 0.2646 Bohr)
    dataset_xc: LDA (atomic reference only; the valence interaction is the bare Coulomb operator)
    one_center: exact: one-center Coulomb and exchange in the two-body operator, frozen-core exchange
    paw_dataset: lda-sr (/data/mandacaru-paw/lda-sr)
    diffuse: off
    basis_functions: 6
    datasets:
        symbol  Z  Z_ion  l_max  r_cut_Bohr  source     sha256
        -----------------------------------------------------------
        O       8  6      1      1.4499      O.parquet  3f0c...e91a
        H       1  1      0      1.2998      H.parquet  a74b...07d2
    orbitals:
        symbol  n  l  zetas  polarization  r_c_Bohr  r_c_Angstrom  eps_free_eV  eps_basis_eV  shift_eV
        ----------------------------------------------------------------------------------------------
        O       2  0  1      0             4.3652    2.3100        -23.706920   -23.606915    0.100005
        O       2  1  1      0             5.3500    2.8311        -9.205641    -9.105680     0.099961
        H       1  0  1      0             6.6822    3.5361        -6.358289    -6.258289     0.100000
    functions:
        symbol  atoms  functions_per_atom
        -------------------------------------
        O       1      4
        H       2      1

[ELECTRONS]
    grid spacing: 0.1 Angstrom (requested)
    grid points: 101 x 101 x 101 (spacing 0.1000 x 0.1000 x 0.1000 Angstrom)
    kinetic operator: spectral
    k-points: Gamma (1x1x1 Monkhorst-Pack)
    charge: 0
    spin-polarized: False (multiplicity 1)
    reference state: hartree-fock
    Z2 tapering: none
    mapping: Jordan-Wigner
    Hamiltonian: 1079 Pauli terms
    spatial orbitals: 6
    electrons (alpha, beta): (4, 4)
    qubits: 12

[OPTIMIZATION SETUP]
    classical_optimizer: SLSQP
    max_iterations: 50
    convergence: gradient and energy (both required)
    gradient_method: analytic
    gradient_formula: exact derivative, g = 2 Re<H psi|A psi>
    convergence_gradient: 0.001
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
    convergence_energy_eV: 0.001
    reference_energy_eV: -476.8628634829

[ITERATIONS]
    iter    time (HH:MM:SS)           energy (eV)            dE (eV)      |grad| (eV)      steps         1q       cnot      depth    operator
    -----------------------------------------------------------------------------------------------------------------------------------------
       1           13:45:10        -33.1720283176      -0.1939118279         6.144385          3         38        104        121    D(0,4->3,7)
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
folder the datasets were read from (`paw_dataset`: the folder's name, such as
`lda` or `pbe`, and its path), the file of every dataset and the SHA-256 of
its bytes (`sha256`, read when the file was loaded: a library is a checkout
that can be rebuilt under the same file names, and the digest says which
bytes the run used, 64 hexadecimal digits, shortened in the example above;
`none` for a dataset generated in the session), the functional the
datasets record (`dataset_xc`), what the augmentation spheres carry
(`one_center`: `exact` one-center Coulomb and exchange for a wave-function
method, the dataset's functional linearized at the reference atom
(`frozen`) for Kohn-Sham DFT) and the function count per element are all
decided below the calculator. It is the block a comparison against another code
is made from. Its three tables -- `datasets`, `orbitals`, `functions` -- read
back from `parse_output(path)["basis"]` as lists of rows keyed by the column
names. An all-electron basis gets the block too (name, options, functions); a
direct-mode run, where no basis was built, has none.

A PAW-LCAO or UPAW-LCAO block always has a `diffuse` line: `off`, or the
recipe that ran, such as `shells present, count 1, rms_ratio 2.5, cutoff_norm
0.0001` (a `{symbol: recipe}` mapping when elements differ). With diffuse
shells it gains two tables. `diffuse_shells` has one row per shell and element:
`l`, its `index` within the channel, the rms radius in Bohr and Å, the
`reference_l` and `reference_rms_Bohr` it was sized from, `alpha_Bohr^-2`,
`r_cut` in Bohr and Å, the squared norm left after the one-center Gram-Schmidt
(`residual`), and `accepted` or `rejected` (a rejected shell is not in the
basis). `diffuse_box` has one row per atom: its largest `r_cut` against its
distance to the nearest face of the grid (`face_distance`), both in Bohr and
Å, and the smallest grid norm over radial norm among its diffuse functions
(`grid_norm`; below `0.9999` the run also warns that the cell cuts them):

```text
    diffuse: shells present, count 1, rms_ratio 2.5, cutoff_norm 0.0001
    diffuse_shells:
        symbol  l  index  rms_Bohr  rms_Angstrom  reference_l  reference_rms_Bohr  alpha_Bohr^-2  r_cut_Bohr  r_cut_Angstrom  residual  status
        ----------------------------------------------------------------------------------------------------------------------------------------
        H       0  0      4.9356    2.6118        0            1.9742              0.028376       13.6367     7.2163          0.437179  accepted
        H       1  0      4.9356    2.6118        0            1.9742              0.048674       11.4992     6.0851          0.940034  accepted
    diffuse_box:
        atom  symbol  r_cut_Bohr  r_cut_Angstrom  face_distance_Bohr  face_distance_Angstrom  grid_norm
        -----------------------------------------------------------------------------------------------
        0     H       13.6367     7.2163          13.7005             7.2500                  0.999999
```

(a hydrogen atom in PAW-LCAO-DZP, `examples/diffuse/01_diffuse_h_anion.py`).

Each fact has **one owner**. The basis is in `[BASIS]` and not repeated in
`[ELECTRONS]`; the classical optimizer is in `[OPTIMIZATION SETUP]` and not
repeated in the variational summary. A value written in two blocks can disagree
with itself, which is worth more than the convenience of not scrolling.

`[ACTIVE SPACE]` records how the register was reduced, and is written only
when it was: when orbitals were frozen or virtual orbitals deleted
(`active_space=...`, see {doc}`active_space`). A run that puts every orbital on
the register has no such block. It names the selector (`method`: `mp2`,
`natural` or `energy`; `none (frozen core only)` when nothing was ranked), what
was asked for (`orbitals_requested`, `threshold`, `frozen_requested`,
`correlating_pairs`, `symmetry`), the
partition that was actually built -- a count can freeze more than `frozen`
asked for -- how the two-body integrals were reached (`two_body_integrals`:
the four-index tensor transformed to the active orbitals, or integral-direct,
the active orbitals' integrals alone; see {doc}`active_space`) and, for MP2,
the correlation energy in the full virtual space.
Its `orbitals` table lists every spatial orbital, in the selector's ranked
basis, with its occupancy in the reference (`occupied`, `singly` or
`virtual`), where it went (`frozen`, `active` or `deleted`) and its occupation
number when the selector computed one; the `occupation` line says what that
number is. With `correlating_pairs` or `symmetry` on, those lines say what the
constraint changed (`brought in`, `displaced`), and the table gains a
`partner` column or an `irrep` column with a `point_group` line. Water in
PAW-LCAO-DZP on eight orbitals, both constraints on:

```text
[ACTIVE SPACE]
    method: mp2
    orbitals_requested: 8
    frozen_requested: none
    correlating_pairs: on (the ranked selection already satisfied it)
    symmetry: on (the ranked selection already satisfied it)
    spatial_orbitals: 23 (0 frozen, 8 active, 15 deleted)
    two_body_integrals: four-index tensor, transformed to the active orbitals
    point_group: C2v
    mp2_correlation_energy_Ha: -0.13424985
    occupation: eigenvalues of the MP2 one-particle density (natural orbitals); frozen rows are its diagonal, since frozen orbitals are not rotated
    orbitals:
        index  occupancy  role     irrep  partner  occupation
        -----------------------------------------------------
        0      occupied   active   1a1    7        1.99039670
        1      occupied   active   1b1    6        1.98225191
        2      occupied   active   2a1    5        1.97772555
        3      occupied   active   1b2    4        1.97436675
        4      virtual    active   2b2    3        0.01919443
        5      virtual    active   3a1    2        0.01608963
        6      virtual    active   2b1    1        0.00992414
        7      virtual    active   4a1    0        0.00722908
        8      virtual    deleted  1a2    -        0.00429837
        ...
        22     virtual    deleted  7b2    -        0.00001275
```

It reads back from `parse_output(path)["active_space"]`, the table as a list of
rows keyed by the column names.

A run whose two-body integrals are Cholesky factors
(`two_body={"method": "cholesky", ...}`, see {doc}`scalability`) has one more
`[ELECTRONS]` line, right after `kinetic operator`, and it is the only place
the factors are described:

```text
    two-body integrals: Cholesky factors, tolerance 1.000e-07 Ha, rank 143 (8.41 per basis function), largest residual diagonal 6.885e-17 Ha
```

The tolerance is the truncation that was asked for, the rank the number of
factors it took, and the largest residual diagonal the bound the factors
actually reached: every integral is within it. Factors rebuilt on a
reference's pivots (the displaced geometries of an active-space force) do
not compute the diagonal and say so. `[PERFORMANCE]` keeps only the time of
the `two-body integrals (cholesky)` stage. The exact dense integrals have no
such line.

`[SYSTEM]` says *where the atoms are*, `[ELECTRONS]` *what was solved* (the
reference state, the charge, the mapping, the register width), `[ACTIVE SPACE]`
*which orbitals* the register carries
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
| classical optimizer | `classical_optimizer`, `max_iterations`, `convergence`, `convergence_marker` |
| screening gradient | `gradient_method`, `gradient_formula`, `convergence_gradient`, `gradient_units` |
| operator pool | `pool`, `pool_class`, `pool_size` |
| growth and execution | `growth`, `pruning`, `screening`, `reoptimize_all_parameters`, `parameter_gradient`, `state_vector_backend`, `device`, `backend_provider`, `circuit_execution`, `shots`, `circuit_profiling` |
| the loop's starting point | `energy_unit`, `convergence_energy_<unit>`, `reference_energy_<unit>` |

A fixed-ansatz `method="vqe"` writes the `ansatz` lines, `initial_parameters`,
`parameter_gradient` (`analytic (adjoint ...)`, `finite difference ...` or
`none (<method> uses no gradient)`) and the execution lines above.

A fresh run starts from the reference state that `[ELECTRONS]` names, with an
empty ansatz; the block does not repeat it. A **resumed** run closes the block
with its lineage -- `resumed_from`, `restored_operators`, `restored_parameters`,
`restored_energy_<unit>`, `resume_same_hamiltonian` -- which is the one record
of the grown ansatz it started from.

#### The screening gradient

`gradient_method` is the `gradient=` option of the driver: how the pool
screening gradients in the `|grad|` column were computed. It sits directly above
`convergence_gradient`, which is compared against the internal gradients, and
`gradient_formula` says in one line what was evaluated. `gradient_units` is
`Hartree` whatever unit the energy columns are in: it describes the internal
expectation value of the Hamiltonian's commutator and its stopping threshold.
The iteration table displays `|grad| (eV)`, multiplying that value by
27.211386245988. `parse_output` returns `max_gradient` in Hartree, consistent
with the threshold and the final screening gradient in the summary.

The `operator` column uses a pool's compact name when available. Names longer
than 40 characters, or containing control characters, appear as `pool[index]`
with the zero-based pool index. The full names remain in the result,
checkpoints and, with `verbose_operators=True`, `pool.json`. This keeps each
iteration on one line without an unbounded operator label.

#### The convergence criteria

`convergence` names the criteria the `convergence=` option set: `gradient`,
`energy`, or `gradient and energy (both required)`. Each threshold is written
once, beside the unit it is read in: `convergence_gradient` in Hartree, next to
the gradient group, and `convergence_energy_<unit>` next to `energy_unit`, so
it reads directly against the `dE` column. The option's energy threshold is
given in eV, so an eV log writes it as given; a Hartree log
(`atomic_units=True`) writes it converted. A criterion the run does not use has no line.

`convergence_marker` says how the `[ITERATIONS]` table shows them: a `*`, one
space after each `|grad|` or `dE` value that meets its own threshold (strictly below
it, the test the loop applies). A column whose criterion is not used carries no
marker. Each row's `|grad|` is the screening that *selected* that row's
operator, so the gradient that finally stops the run is the screening after the
last row: it is `final_max_gradient` in `[VARIATIONAL QUANTUM SUMMARY]`, marked
the same way. With both criteria required, the run stops at the first
screening whose gradient is marked while the last row's `dE` is.

| `gradient_method` | What it computes | What it costs |
| :--- | :--- | :--- |
| `analytic` (default) | The exact derivative `g_i = 2 Re<H psi\|A_i psi>` -- what the other two estimate. | One matrix-vector product per pool operator; no step-size truncation error. |
| `finite_difference` | A central difference of the energy at shifted parameters, step `1e-4` (`FINITE_DIFFERENCE_STEP`, reported in `gradient_formula`). | `2 x \|pool\|` energy evaluations, plus the `\|pool\|` dense diagonalizations those evaluations are read from. |
| `parameter_shift` | The quantum parameter-shift rule, reconstructed over the generator's frequency set (exact, not an approximation). | The same shifted-energy evaluations and diagonalizations, several shifts per frequency. |
| `measured` (any run with `shots > 0`) | `g_i = <psi\|[H, A_i]\|psi>` from Pauli expectation values measured on the device; exact and classical at the reference determinant, before the first operator. | One measured gradient step per growth step, sized in `[GRADIENT MEASUREMENT]`. No state vector. |

With `shots > 0`, `gradient_method` is `measured` whatever `gradient=` asked
for (the formula says when it overrides a request), `state_vector_backend` is
`none (...)`, and `screening` is the `screening=` option: which pool operators
a growth step screens ({doc}`measurement_cost`).

### `[GRADIENT MEASUREMENT]`: what a shot-based ADAPT-VQE measures

A shot-based ADAPT-VQE writes this block once, after `[OPTIMIZATION SETUP]`.
It is the only block that states the cost of the screening, and every number
in it is computed before anything is submitted: the commutator observables
`[H, A_k]` do not depend on the state, so one plan serves every growth step.
LiH (HAO, `grid_spacing=0.4`, `pool="qeb"`, `shots=4096`, local estimator):

```text
[GRADIENT MEASUREMENT]
    scheme: commutator observables [H, A_k], strings merged per batch, qubit-wise-commuting bases
    first_step: classical (exact at the reference determinant)
    measured_operators: 8
    strings_per_step: 507
    bases_per_step: 162
    bases_per_step_in_energies: 6.0 (an energy is 27 bases)
    jobs_per_step: 1
    largest_job: 162 bases, 507 strings
    circuit_instances_per_step: 162
    shots_per_step: 663552
    gradient_noise_bound_Ha: 2.059e-02
    batches_per_step: 1
    batch_memory_bytes: 60150
```

| Line | Meaning |
| :--- | :--- |
| `measured_operators` | How many pool operators each step measures (the pool size is `pool_size`, the device `device` and the shots per basis `shots` in `[OPTIMIZATION SETUP]`). Fewer than the pool only under `screening={"operators": K}` or the automatic subset, and then `screened_subset` says which and why. |
| `strings_per_step`, `bases_per_step` | Distinct Pauli strings of the step's commutators and the qubit-wise-commuting bases they are measured in. A string shared by several operators is measured once. |
| `bases_per_step_in_energies` | The same bases in units of one energy measurement of the Hamiltonian. |
| `jobs_per_step`, `largest_job` | The jobs the bases are packed into, each within `measurement_budget=` (bases and observables). |
| `shots_per_step`, `circuit_instances_per_step` | With the provider's resilience multipliers, as in `[MEASUREMENT PLAN]`. |
| `gradient_noise_bound_Ha` | `max_k (sum_R c_kR^2)^(1/2) / sqrt(shots)`: a bound on one gradient's standard error. The `\|grad\|` column's largest of many noisy gradients sits above it, so a gradient criterion below it cannot be met. |
| `batches_per_step`, `batch_memory_bytes` | The classical working set: commutators are formed in batches, and the largest one (plus the step's string store) is what the step holds in memory. |

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
                 grid_spacing=0.25,
                 gradient="parameter_shift",
                 txt="output.txt")
```

The block reports **what ran**, not what was asked for.
`state_vector_backend` says how the run held its states and operators:
`dense matrices` or `sparse matrices` on the full register, `particle-number
sector` when the states were restricted to one (`method="vqe"` too, from
10 qubits), and `matrix-free Pauli
products` when no operator matrix was stored ({doc}`scalability`). A sparse, matrix-free or
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
    orbital_response: z-vector (3 CPHF iterations, residual 4.1e-14)
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
`orbital_gradient` is the largest orbital-rotation derivative of the state's
energy, in Hartree (zero for a Hartree-Fock determinant and for the exact
eigenstate of the orbital space). `orbital_response` says how the gradient
accounts for it: `z-vector` (one coupled-perturbed Hartree-Fock solve, folded
into the density matrices; the residual is the solve's), or `central difference
of the reduced Hamiltonian` for a selector or reference the z-vector does not
cover (see the active-space guide). Only when neither applies does the gradient
hold the orbitals fixed, and then
{data}`~mandacaru.algorithms.calculator.ORBITAL_RESPONSE_TOLERANCE` is where it
starts warning. `net_force` is the grid's egg-box
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
    integration_backend: Rust (8 threads)
    threads: 8
    cpu_count: 8
    mpi: not used (single process, shared-memory scoped threads)
    peak_memory_MiB: 4971.7
    resident_memory_MiB: 1497.5
    solver_wall_time_s: 16.6582
```

The **calculator** writes this block, not the solver, and that is why it comes
after `[FORCES]`: on a real relaxation the nuclear gradient is the largest single
stage. A six-step water relaxation in PAW-LCAO-SZ at `grid_spacing = 0.10` spends **62 %** of its
325 s on forces and **19 %** on the variational optimization, so a block closed
when the solver finished would account for the smaller part of the step.

`untimed_s` is the honest remainder: `wall_time_s` minus the stages, i.e. the
work no stage wraps (building the Hamiltonian, the fermion-to-qubit mapping,
materializing the operator). `peak_memory_MiB` is the process high-water mark and
therefore monotonic across a relaxation; `resident_memory_MiB` is what is
resident at the end of *this* step, so the two together show whether a stage
allocated and released. `integration_backend` reports the active native engine
(`Rust` with `std::thread::scope` or `C (OpenMP)`), `threads` is the number of
worker threads used against the `cpu_count` the machine offers, and `mpi` states plainly that there
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

The block above is one measurement -- `qpu_jobs` / `qpu_job_ids` list the jobs
of that single submission, e.g. a `measurement_provider=` energy plus RDMs. A
solver whose *whole optimization* ran on a processor (`device="ibm_*"` with
`shots > 0`) instead reports `qpu_submissions`, the cumulative job count of
the run so far -- one per energy evaluation, summed across every growth step
of an ADAPT-VQE run.

## `verbose_operators` and `verbose_hamiltonian`

Two options write what the blocks leave out, as JSON, once per run:

```python
calc = Mandacaru(method="adapt-vqe",
                 basis="HAO",
                 grid_spacing=0.25,
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
                 grid_spacing=0.25,
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
                 grid_spacing=0.25,
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
                 grid_spacing=0.20,
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
