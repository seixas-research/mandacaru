# VALQA development plan

Living plan for `Mandacaru(method="valqa")`, the Variational Adaptive Learnable
Quantum Algorithm: VASQA's Markov Chain Ansatz Search with a learned proposal
`q(C'|C,H)`. Dated decisions, measured numbers and post-mortems go to
`HISTORY.md` (the VASQA section of 2026-09-29, the VALQA section of
2026-09-30); this file says **what exists, what is left, in what order, and
how each step is accepted**. The theory is in `../MCAS.md` (sections 10-11)
and `codex/LP-MCAS.md` (sections 11-19); the stage names below are theirs.

**Scope (2026-10-01, second spec).** VALQA's purpose is geometry
optimization and molecular dynamics. The ansatz `(C_n, theta_n*)` found at
one geometry is transferred to the next and edited locally, instead of being
rebuilt, and the learned proposal is conditioned on `(C_n, H_n, H_{n+1})`.
The single-geometry chain below is step 0 of such a trajectory, and the
items V8-V11 extend it (section 6 maps the spec onto the repository).

Status as of 2026-10-01: Level I (surrogate-assisted LP-MCAS, operator head
only) is implemented and tested. No model has passed the readiness gate, so
every VALQA run draws exactly what VASQA's gradient proposal draws.

---

## 1. What exists

| Piece | Where | Notes |
| --- | --- | --- |
| Chain (moves, positions, MH test) | `algorithms/vasqa.py::MarkovChainSearch`, `algorithms/mcas.py` | `MOVES = (insert, delete, replace, swap)`, manual `move_weights` (`mcas.py:68`); `insert`/`delete` must stay > 0 (`mcas.py:184`) |
| Learned operator proposal | `algorithms/valqa.py`, `algorithms/proposal_model.py` | `P = (1-eps) p_ML + eps p_grad`; `eps = 1` until a model is ready, then 0.3 |
| Model | `proposal_model.py` | Pauli factor graph, 2 message-passing layers, deep-kernel GP on `insert` rows, score `-mu + kappa sigma`, softmax |
| Model checkpoint | `ProposalModel.save/load` (`proposal_model.py:439/459`) | one `.npz`, `MODEL_SCHEMA = 1`, content-hash `version`, readiness `report` inside |
| Data (replay buffer) | `algorithms/proposal_data.py::EditRecorder` | append-only `edits.jsonl`, one flushed row per proposal (rejections included), `problems/<hash>.npz` once per problem |
| Shared store | `MANDACARU_PROPOSAL_DATA`, `mandacaru --set-proposal-data` | default `record=` target; `train_proposal_model()` writes `proposal_model.npz` there; VALQA loads it at construction |
| Readiness gate | `assess_data_volume` | leave-one-molecule-out Spearman vs. `-|g|` and vs. the descriptor-only GP, margin 0.05, >= 3 groups |
| Tests | `test/algorithms/test_{valqa,proposal_model,proposal_data}.py` | VALQA without a model reproduces VASQA's chain step for step |

The "simplest baseline" that the original request asked for already exists in
two forms. The **gradient softmax** (`mcas.gradient_softmax`) is the physics
baseline and the fallback. The **descriptor-only GP** (`use_graph=False`) is
the no-representation surrogate that the gate compares against. A separate
feed-forward or logistic module would duplicate them, so this plan does not
add one. Item V2 below adds the one baseline that is missing: a within-state
ranker.

## 2. Decisions this plan rests on

1. **The model is frozen during a run.** The forward and reverse proposal
   probabilities come from the same snapshot, so the Hastings ratio is exact
   and VALQA keeps VASQA's stationary distribution
   (`../MCAS.md` sections 10.2 and 11.1).
2. **"Online updating" means one of three things. Only A exists, and only A
   keeps the sampling guarantee:**
   - **A. Between runs (offline LP-MCAS; implemented).** Each run records into
     the store, `train_proposal_model()` refits and re-gates, and the next run
     loads the new version, which the log and `VALQAResult.proposal_model`
     name.
   - **B. Snapshot refits within a run (optimization mode; proposed, V4).**
     The model is refit every `K` evaluated proposals, and each acceptance
     decision uses one snapshot for both directions. The chain then adapts.
     The result must say so (`sampling_exact=False`), and "best architecture
     found" is the only claim it supports.
   - **A'. Between geometry steps (the trajectory spec's "online GP";
     chosen and implemented 2026-10-01, V10).** Each geometry is its own chain, so refitting the
     GP between geometry `n` and `n+1`, with the GNN frozen, keeps every
     chain exact. It is mode A at the scale of a trajectory, and it is what
     the spec's predict -> evaluate -> record -> update cycle means. V4 below
     (refits *inside* one chain) is therefore demoted to research.
   - **C. Diminishing adaptation (adaptive MCMC; research).** Refits happen on
     a schedule whose weight on the new model goes to zero (Roberts and
     Rosenthal containment plus diminishing adaptation). It could restore
     ergodicity in principle, but proving it for a GP refit is open work.
3. **Proposal probabilities must be evaluable at any state.** Every
   acquisition is a deterministic function of `(C, H)` (or of a seed derived
   from the state hash), because MH needs `q(C|C')` as well as `q(C'|C)`.
   Plain Thompson sampling is ruled out by this.
4. **The gradient stays in the mixture** (`0 < eps`). It keeps every reverse
   move supported and is the physics prior that any learned model has to
   beat.

## 3. Roadmap

Status: `[ ]` open, `[~]` in progress, `[x]` done.

### V0. Housekeeping (P1) -- done 2026-10-01, uncommitted

- [x] JAX stays a core dependency (`pyproject.toml`). The stale
  `[learned-proposals]` and `[legacy-forces]` install hints are gone from
  `proposal_model.py`, `_jax_energy.py` and `docs/source/tutorial/valqa.md`.
  The `ImportError` guards became plain lazy imports (`_jax()`, which also
  turns on x64). `jax_available()` is removed, along with the
  `needs_jax` / `importorskip("jax")` skips in `test_forces.py`,
  `test_scf_convergence.py` and `test_proposal_model.py`.
- [x] Store writes are safe against concurrent runs:
  - `save_npz_atomically` (`proposal_data.py`) writes a temporary file, sets
    mode 0644 and calls `os.replace`; both `ProposalModel.save` and
    `ProblemRecord.save` use it.
  - `EditRecorder` writes each row with one `os.write` on an `O_APPEND`
    descriptor, and continues on a new line after a cut row.
  - `load_edits` skips unparseable rows with a `RuntimeWarning` naming their
    lines.
  - `ProposalModel.load` names a file that is not a model (garbage,
    truncated zip, no metadata) with a `ValueError`.
- [x] The recorder is closed whatever fails after it opens, the banner and
  the first evaluation included (`vasqa.py`, inside the `try`).
- [ ] *Not done, by decision:* bounding `_cache`, `seen` and `steps`. With
  `warm_start=False` the cache is an exact memo, and bounding it would
  re-relax revisited states, change `num_screenings` and record duplicate
  rows. `steps` is the result's trajectory. All three are
  `O(max_steps x pool)`; revisit only if V4 makes chains open-ended.
- [x] Ted's tests are in the three existing files: `TestTheModelFile`
  (`test_proposal_model.py`), `TestAppending` and two refusals
  (`test_proposal_data.py`), `TestTheModelIsFrozenForTheRun`
  (`test_valqa.py`). A model is read when the solver is built, so the
  frozen-model test builds `calc.solver` before overwriting the file.
- [x] TODO.md 4.8 points here.

### V1. Make the gate measure the decision the proposal makes (P1)

The gate ranks held-out edits *across* states, and that comparison favors the
gradient almost by construction (`dE ~ -g^2/2k`). The proposal only chooses
*within* a state.

- [ ] Record several insertions per state. Add an optional screening mode that
  evaluates the top-`k` candidates of a state and writes each one as a row
  flagged `screened=True`. Those rows are not chain moves, so they stay out of
  the MH accounting.
- [ ] Add a within-state metric to `assess_data_volume`: the mean per-state
  Spearman correlation, and top-1 regret in Hartree against the best screened
  candidate. Report it next to the current across-state metric.
- **Accept:** on the 2026-09-30 data (`~/mandacaru-valqa-edits/`) plus new
  screened rows, the gate reports both metrics, and HISTORY records them.

### V2. Simplest within-state baseline: a pairwise logistic ranker (P1)

This is the baseline the original request had in mind (logistic regression on
operator features), applied to the decision the proposal actually makes.

- [ ] Fit a Bradley-Terry / pairwise logistic model on descriptor differences
  `x_mu - x_nu` within a state, with the label "mu gained more than nu". Its
  logits feed the same softmax, so `p_ML` is evaluable and the MH ratio stays
  exact.
- [ ] Save it through the same `ProposalModel` file, either with a `kind`
  field or as a sibling class sharing `save/load`. Bump `MODEL_SCHEMA` and
  refuse old files without migrating them.
- **Accept:** it enters the gate as a third baseline. If it beats `-|g|`
  within states, it becomes the default model kind and the GNN-GP has to beat
  it.

### V3. Gradient-residual GP (P1)

- [ ] Replace the descriptor-linear prior mean with the second-order estimate
  `m(mu) = -g_mu^2 / (2 k)`, where `k` is a curvature learned per problem or a
  global scale, so the GP models only the **residual** over the gradient. Far
  from the data it returns exactly the gradient ranking, which is VASQA.
- [ ] Use inducing points (sparse GP / SVGP) to lift
  `MAX_TRAINING_ROWS = 400`.
- **Accept:** held-out within-state Spearman is at least that of `-|g|` on
  every fold, and better on most.

### V4. Online refits within a run (decision 2B; P3 research -- superseded by 2A' for trajectories)

- [ ] Add `refit_every=K` (default off). It refits only on rows already in
  the store and installs the new snapshot between MH decisions, never inside
  one.
- [ ] When a snapshot is installed, re-bind `_bound`, including the
  unready -> ready case, and recompute the cached `operator_probabilities`
  of `current` and of every cached state from their stored gradients
  (`vasqa.py:401-415`, `:582-586`).
- [ ] Record the model version in each row, not only in the run header.
  Pad training shapes to fixed sizes, or clear the JAX caches.
- [ ] Add `sampling_exact` to the result, set `False` whenever a refit
  happened, and give the run log one owner block for the model-version
  history.
- [ ] Memory: the buffer is the append-only file, not process memory. A refit
  reads the last `N` rows (ring window), and each `BoundProposal` is replaced,
  not accumulated.
- **Accept:** a 500-step H4 chain keeps peak RSS flat across refits (the
  conftest budget is < 8 GB), and the same seed with `refit_every=None`
  reproduces the frozen chain.

### V5. Learned move-type and position heads (LP-MCAS-2/3; P2)

- [ ] Make `P(m|C,H)` over `insert/delete/replace/swap` learnable. The
  simplest version is a Dirichlet-multinomial whose counts are weighted by
  reward per length bucket. It must stay a function of the state only, keep
  `insert` and `delete` > 0 at every non-boundary length, and mask invalid
  moves before normalizing (`mcas.py:201`). Counts updated from the chain's
  own history are adaptation: freeze them offline, or put them under V4's
  `sampling_exact`. Cache the per-state move distribution the way operator
  probabilities are cached, because `ProposalKernel` takes static
  `move_weights`.
- [ ] Add a position head: an insertion-gap embedding built from the
  neighboring operators' embeddings, so the score depends on the slot.
- [ ] Learn from `replace` rows, which needs the removed operator as a
  feature.
- **Accept:** the reverse-probability tests in `test_vasqa.py` and
  `test_valqa.py` pass with the learned move weights, and a detailed-balance
  check on a 3-operator H2 pool (exact enumeration) agrees with the target.

### V6. Equivariant / gauge-aware encoders (P2; see analysis section 4.2)

- [ ] Make the term features **MO-sign-gauge invariant**. Today
  `term_features[:, 0]`, the signed coefficient, is the only gauge-dependent
  column (`proposal_model.py:168-176`). Either drop the sign on off-diagonal
  terms, or build the invariants (`h_pq h_qr h_rp` loops) on the pre-taper
  orbital graph, because after tapering the sign rule is no longer local to
  one qubit (section 4.2).
- [ ] Add an orbital-graph encoder on `h_pq`, `(pq|rs)` with irrep labels
  (abelian-subgroup labels where degenerate shells are not symmetry-adapted).
  Symmetry-forbidden excitations get zero gradient at symmetric states, which
  makes them a prior for down-weighting, not an exact energy mask
  (section 4.2, item 3).
- [ ] (Research) Add an atom-level E(3)-equivariant encoder (`e3nn`) for
  geometry transfer along a PES scan.

### V7. Level II and III (P3)

- [ ] Add a direct local policy, trained by reward-weighted likelihood or
  distilled from the GP proposal (`../MCAS.md` 11, `LP-MCAS.md` 16).
- [ ] Add conditional whole-ansatz generation `p(C|H)` (`LP-MCAS.md` 17).
  This waits for a ready Level I model.

### V8. Trajectory transfer (spec stages 1-3; vertical slice done 2026-10-01, uncommitted)

- [x] Moves and parameter transfer: `Action.apply` / `Action.transfer`
  (`mcas.py`). The new piece is `replace_start="zero" | "inherit"`. Every
  move has an inverse, tested for both architecture and angles.
- [x] `edit_distance(a, b)` (`mcas.py`): the Levenshtein distance over
  insert/delete/replace. It is an upper bound on the spec's `d_edit` with
  swaps, because a swap counts 2 here.
- [x] `transfer=True` (VASQA and VALQA; CLI `--transfer`). The ASE
  calculator hands each new geometry's solver the previous solver
  (`calculator.py`, `inherit_ansatz`), and the chain starts from the
  previous reported `(C_n, theta_n*)`. With `warm_start=False` only the
  architecture is carried, which gives the spec's "architecture transfer
  only" ablation.
- [x] Transfer vs. reconstruction rule, logged once in
  `[OPTIMIZATION SETUP]` as `start`. The chain transfers when every
  operator label is in the new pool, the ansatz is no longer than
  `max_length`, and the transferred ansatz relaxes below the reference
  energy. Otherwise it starts empty.
- [x] `transfer_steps`: a short local search after a transferred start,
  which is what saves evaluations. On H2 (HAO, QEB pool, 0.74 -> 0.76 ->
  0.78 Angstrom) a 6-step transferred chain reached the same energy as a
  30-step rebuild with 22 evaluations instead of 286 (one smooth-region
  run, not a benchmark).
- [x] Result fields: `start`, `start_operators`, `edit_distance_from_start`.
  The summary block also logs `edit_distance_from_start`.
- [x] The transfer threshold comes from V9's overlaps
  (`transfer_threshold`). `||dX||` is not used: the matched overlap
  measures the electronic change directly.
- [ ] A trajectory record: one row per geometry with `X_n`, the
  `(C, theta)` start and end, the edit distance, evaluations and the
  transfer decision, kept on the calculator next to `calc.trajectory`.
  Checkpoint and restart of a trajectory: `supports_checkpoints = False`
  today.

### V9. Orbital tracking between geometries (spec stage 9; done 2026-10-01, uncommitted)

- [x] `algorithms/orbital_tracking.py`:
  - `OrbitalSnapshot` holds a geometry's active MOs as AO coefficients
    `A = S^-1/2 V` plus its basis functions.
  - `orbital_overlap` computes `O = A_n^+ <chi_n|chi_{n+1}> A_{n+1}`, with
    the previous basis sampled on the new grid, normalized so `|O| <= 1`.
  - `match_orbitals` runs a Hungarian match on `|O|` inside each occupation
    block, plus sign alignment.
  - `transfer_ansatz` renames the excitations and multiplies each angle by
    the product of the orbital signs (one more sign per fermionic index
    transposition; qubit excitations commute).
- [x] The chain uses it in `_transferred_start`; `inherit_ansatz` stores the
  previous snapshot. The confidence is the smallest matched overlap, and
  `transfer_threshold` (default 0.9, CLI `--transfer-threshold`) decides
  between transferring and rebuilding. The `start` line reports
  "N reordered, N sign(s) aligned, smallest overlap x".
- [x] Verified on H4 (HAO, fermionic pool) by relabeling the second
  geometry's orbitals on purpose: virtuals 2 and 3 exchanged, orbitals 1 and
  2 sign-flipped. The tracked transfer reproduces the previous energy before
  any re-optimization (-56.773771 eV, 9 evaluations to confirm). Untracked,
  the start is 1.23 eV high and needs 95 evaluations to recover. This is
  pinned by `test_orbital_tracking.py`.
- [x] Measured on smooth scans: on H2 0.74 -> 0.78 and H4 0.90 -> 1.05
  Angstrom the eigensolver flipped no orbital, and the smallest overlaps
  were 0.999-1.000 at 0.02-0.05 Angstrom steps.
- Limits, all stated in the module and in the tutorial:
  - Fermionic excitations are renamed exactly. Qubit excitations get a close
    start after a reordering.
  - `iP[...]` and CEO operators are carried only when the match is the
    identity.
  - Spin-orbit spinors and periodic orbitals are not tracked; the label
    transfer proceeds untracked and says so.
  - PAW-LCAO's cross overlap leaves out the augmentation, so it is a
    pseudo-orbital measure.
  - Near-degenerate rotations are not aligned: they show as a low overlap,
    and the chain rebuilds.
- [ ] Unitary (Procrustes) alignment inside near-degenerate blocks. This
  needs the Hamiltonian rebuilt in the rotated orbitals, so it is left for
  when a degenerate system (CH4, CO2, O2) shows it matters.
- [ ] State fidelity `|<psi_n|psi_{n+1}>|^2` as a further confidence
  indicator. It needs both states in one orbital basis, which the overlap
  above provides.
- [ ] Use the matched gauge for the training data as well (section 4.2,
  item 1).

### V10. Proposal conditioned on two Hamiltonians (spec stages 4-6; done 2026-10-01, uncommitted)

- [x] Rows carry `trajectory`, `geometry_step` and `previous_problem` (run
  header; no row-schema bump, and a row without them is the single-geometry
  case). The previous problem is saved in the same store. Chains link to the
  previous geometry's trajectory whether or not `transfer=True`, and a
  different molecule or pool starts a new trajectory.
- [x] Features `[z_{n+1}, z_n, dz, z_C, z_A, descriptors]`, mean+sum pooling
  (a `|c|`-weighted term sum, a stated deviation from the spec's plain
  sum), hidden width 32. The signed coefficient was dropped (section 4.2,
  item 1). `MODEL_SCHEMA = 2`.
- [x] `group_by="trajectory"` in `assess_data_volume` /
  `train_proposal_model`.
- [x] Decision 2A': `update_between_geometries=True` (VALQA, CLI
  `--update-between-geometries`) conditions the GP on the previous
  geometry's insertions (`ProposalModel.condition`: network and kernel
  frozen, residuals recovered as `L L^T alpha`, stored standardization,
  newest 400 rows). The insertions are scored first (RMSE, Spearman)
  test-then-train, and the result is reported on `result.model_update` and
  in the setup block's proposal line. The store's model file is not
  changed.
- [x] Regression gate on the 2026-09-30 data: gradient 0.779 and
  descriptors unchanged. The graph model rose from 0.667 / 0.753 / 0.749 to
  0.797 / 0.782 / 0.766 (seeds 0-2). Ablation: sum pooling is the cause.
  Still not ready. Table in HISTORY.
- [ ] Record trajectory data (relaxations and scans of the six molecules
  plus the spec's set: H2O, O2, H2O2, CH4, CO, CO2), then measure what `dz`
  adds: same-trajectory later region, an unseen trajectory of a known
  molecule, a held-out molecule. Report the prequential score along each
  trajectory.
- [ ] Move-specific `z_dC` for delete, replace and swap (overlaps V5).
- [ ] The mixture partner stays the gradient proposal, not uniform. MH
  needs the baseline to supply every reverse move, and the gradient does
  that while also being the physics prior; uniform remains an ablation.

### V11. Baselines, metrics and ablations (spec sections 20, 21, 26; P2)

- Baselines that already exist: rebuild at every geometry (`transfer=False`),
  ADAPT-VQE rebuilt (`method="adapt-vqe"`), uniform and gradient MCAS
  transfer (VASQA with `transfer=True`), and VALQA with or without a model.
  Still missing: a fixed ansatz with only a parameter warm start
  (`transfer_steps=0`), and VALQA with between-step GP updates (V10).
- Metrics on the result today: energy, evaluations, optimizer steps,
  architectures, screenings, acceptance, edit distance, and gate counts from
  `metrics`. Missing: state fidelity between steps, and GP uncertainty per
  step.
- Benchmarks go in `examples/` with outputs in `examples/data/`. A relaxation
  benchmark needs an energy scan as well (CLAUDE.md: a converged relaxation
  is not evidence of binding).

---

## 4. Architecture analysis

### 4.1 Gaussian processes as surrogates of the VQE landscape

**What the GP models.** The GP models the map `(H, C, mu) -> dE`, the relaxed
energy change of inserting `mu` into `C`. It does not model the energy
landscape over angles: the angles are integrated out by the inner VQE. The
target is compressed (`sign(dE) log(1 + |dE|/1e-5 Ha)`) because the changes
span five decades.

**How uncertainty steers `q`.** The posterior `(mu, sigma)` becomes a score.
The current score is the UCB score `b = -mu + kappa sigma`, and `q` is
`softmax(b/tau)` mixed with the gradient proposal. `kappa` buys exploration
where the model is ignorant, which is exactly where a new molecule's edits
fall. Alternatives that keep `q` evaluable (decision 3):

- **Expected improvement** over the state's best known insertion. It is
  closed form in `(mu, sigma)` and has no extra hyperparameter.
- **Probability of improvement** `Phi(-mu/sigma)`. It is bounded, which keeps
  the softmax well conditioned.
- **State-seeded Thompson sampling.** A posterior sample drawn with a seed
  equal to the hash of `C` is a deterministic function of the state, so
  `q(C|C')` stays computable. It costs one joint pool-sized sample per
  evaluated state.

**What has to change for it to beat the gradient** (V1, V3):

1. **Residual modeling.** The perturbative estimate `dE ~ -g^2/(2k)` already
   explains most of the variance. A GP with that prior mean spends its
   capacity on what the gradient misses: operator redundancy with `C`,
   higher-order coupling, and saturation near the sector minimum.
2. **Preference likelihood.** The decision is a within-state ranking, so a
   pairwise-preference GP (probit likelihood on "mu gains more than nu")
   matches the proposal's objective better than regression on `dE`.
3. **Noise.** Inner optimizations that hit `maxiter` produce labels with
   larger error. A heteroscedastic noise term on the recorded convergence
   flag keeps them from anchoring the posterior.
4. **Scale.** Exact GP cost is `O(N^3)` and is capped at 400 rows. Inducing
   points (`M ~ 100-200`) or random Fourier features of the deep-kernel space
   remove the cap. Multi-task kernels (a molecule-level latent) are the
   principled way to share strength across Hamiltonians.
5. **Cost-aware acquisition.** The reward in `../MCAS.md` 11 can divide by
   the evaluations a proposal costs. The GP can predict that cost as a second
   output, which favors insertions that relax cheaply.

**What it cannot do.** A GP surrogate does not change the chain's target. It
changes only how fast the chain finds low-`F` architectures. Any claim has to
be made at matched total cost, counting gradients, failed optimizations, and
training and inference separately (`LP-MCAS.md` 19).

### 4.2 Graph networks and E(3)/O(3) symmetry

**Graphs in use and proposed.**

- *Hamiltonian factor graph (implemented).* Qubit nodes and Pauli-term nodes,
  with `X`/`Y`/`Z`-labeled edges, size-independent and invariant to term order
  (`test_the_embedding_does_not_depend_on_the_term_order`). A pool generator
  is embedded through the qubits it touches.
- *Circuit graph (proposed, V5).* A sequence of operator nodes, with edges to
  the next operator and to operators sharing qubit support. Insertion gaps are
  nodes too, so the position head scores slots directly.
- *Orbital graph (proposed, V6).* MO nodes carrying `h_pp`, the occupation and
  the irrep label. Edges carry `h_pq` and hyperedges `(pq|rs)`. This is the
  fermionic view, before mapping and tapering, and it transfers across
  mappings.

**Where E(3)/O(3) enters, and where it does not.** The qubit Hamiltonian
VALQA reads is built from integrals in the RHF molecular-orbital basis,
obtained from the Loewdin-orthonormal AO basis, with UHF natural orbitals for
open shells (`core/hamiltonian.py:1100-1125`). It is recorded after any Z2
taper (`proposal_data.py:114-142`). A rigid
rotation, translation or inversion of the molecule rotates the atomic
orbitals, but the MO-basis integrals `h_pq` and `(pq|rs)` stay the same up to
the MO gauge (below), for a fixed active space and up to the grid's egg-box
error (meV). **The factor graph is therefore already invariant under the full Euclidean
group (translations, rotations and inversion), so the relevant point-group
part is O(3), not merely SO(3).
Putting `e3nn` layers on it would enforce a symmetry the input cannot break.**
The proposal's output is a probability, a scalar, so the requirement is
**invariance** (`0e` irreps under O(3): parity-even scalars, since the energy
of a molecule and of its mirror image are equal), not equivariance.

The symmetries that actually constrain the learned map are these:

1. **MO sign gauge (Z2 per spatial orbital).** The sign of a real MO from
   the eigensolver is arbitrary. `conjugation_real_orbitals`
   (`core/hamiltonian.py:1135-1175`) fixes complex phases to +-1 but leaves
   real orbitals alone. Under `phi_p -> -phi_p`, both spin qubits `p_up` and
   `p_down` flip together, which is conjugation by `Z_{p_up} Z_{p_down}`.
   Before tapering, a Pauli term flips sign when it has an **odd total number
   of `X`/`Y` letters on the two qubits of orbital `p`**. That Pauli commutes
   with the Z-type tapering generators, so after tapering each surviving
   string still only changes by +-1. Which strings flip is then set by
   anticommutation with the tapered image of that Pauli, not by the letters
   on one qubit. The pool gradients flip too, which is harmless because
   `|g|` is used. **The current model reads the signed coefficient**
   (`term_features[:, 0]`, the only gauge-dependent column), so two gauges
   of the same molecule give different embeddings. The fix is V6. Easiest is
   to build invariants on the pre-taper orbital graph (`h_pq h_qr h_rp`
   loops). A generic closed loop of the bipartite factor graph is *not*
   invariant, because it can pass a qubit through one `X`/`Y` edge and one
   `Z` edge. The alternative is a documented MO sign convention applied
   before recording.
2. **Degenerate-shell rotations (U(k) inside a degenerate MO shell).** Linear
   molecules (pi shells), square H4 and any symmetric-top molecule have them.
   The integrals mix under these rotations, so an invariant model must use
   shell-level invariants. This is the one place where the equivariant
   machinery is genuinely useful, applied to irreps of the molecular point
   group rather than of E(3).
3. **Point-group selection rules.** Take an excitation whose irrep product
   does not contain the totally symmetric irrep, inserted at any slot of a
   circuit whose operators are all totally symmetric. Its gradient
   `<[H, G]>` is zero, so gradient screening never selects it. Its *relaxed*
   gain is second order (`<[[H, G], G]>`) and is not zero in general, unless
   the state is stable against symmetry breaking. So the irrep labels are an
   exact **gradient** mask and a learnable prior on the gain, not an exact
   energy mask. In degenerate shells that are not symmetry-adapted, a
   single excitation is not an irrep operator, so use abelian-subgroup
   labels.
4. **Particle number and `S_z`.** Fermionic excitation pools conserve them
   (not `S^2`, and not arbitrary Pauli-string pools), and the tapered sector
   fixes them. For those pools there is nothing to learn here.
5. **Qubit relabeling.** This is deliberately *not* a symmetry of the input:
   the Jordan-Wigner order sets string lengths, and the normalized qubit
   index is a feature for that reason. Z2 tapering also rewrites terms by a
   Clifford that depends on the sector, so the recorded problem keeps the
   tapered Hamiltonian it actually ran.

**When an E(3)/O(3)-equivariant network does pay off.** It pays off when the
model reads geometry: atom positions and atom-centered orbitals with their
angular momentum `l` (natural in Mandacaru's HAO and multiple-zeta bases).
The intended design is:

- Atoms are nodes with element embeddings. Atom-centered basis functions are
  irreps `l = 0, 1, 2` (`1o` for p, `2e` for d). Edges carry spherical
  harmonics of the bond vector and a radial basis. Messages are tensor
  products (`e3nn.o3.FullyConnectedTensorProduct`, NequIP/MACE style).
- MO features follow by contracting the atom-level irreps with the MO
  coefficients, which turns them into invariants. A pool operator's score is
  a scalar readout over the MOs it excites.
- Use: **transfer along a potential energy surface** (scans, the relaxation
  path), where geometry changes smoothly but the SCF orbitals may reorder.
  The atom-level model sees a continuous input where the MO-level one sees a
  discontinuity at an orbital crossing.

The output is invariant by construction (`0e` readout). The model should
be O(3)-invariant (with parity) rather than only SO(3)-invariant, because the
Coulomb Hamiltonian is inversion invariant and a chiral molecule and its
mirror image have the same energy. An SO(3) model would only waste capacity.

**Scope: the analysis above applies to `lda-sr` (scalar-relativistic).**
With the `lda-dirac` sets, spin-orbit coupling (`core/spin_orbit.py:37-55`)
changes three things.

- Space and spin must rotate together, which needs half-integer `j` irreps
  (the SU(2) double cover, which `e3nn` lacks).
- The MO gauge becomes U(1) per Kramers pair, not Z2, so the sign argument
  of item 1 does not carry over.
- `S_z` is not conserved, only `J_z` (`core/sector.py:163, 188`), so the
  sector and pool arguments of item 4 fail.

The energy is still parity-even, so O(3) invariance of the output survives.
Spin-orbit runs need their own treatment. The cost is a new dependency (`e3nn` is
PyTorch-based, while Mandacaru trains in JAX; `e3nn-jax` exists and matches
the stack). So this enters only behind a measured gain from V1-V3, as a
research item.

---

## 5. Reviews of this plan

Outcomes of the agent reviews are recorded here when they finish:

- **Rita (2026-10-01): V4 can proceed once amended.** No hard rule is
  violated. Items to add before implementation:
  - **V4 cached probabilities.** `_Evaluated.operator_probabilities` is cached
    (`vasqa.py:401-415`), and the Hastings ratio reads it for both directions
    (`vasqa.py:582-586`). A refit must re-bind `_bound` (including the
    unready -> ready case; today `_prepare_proposal` binds only at the start)
    and recompute the distribution of `current` and of every cached state
    from the stored gradients.
  - **V4 run log.** The model version is written once by the setup block
    (`valqa.py:121`) and once on the result. The version history needs one
    owner (the summary block), and each recorded row has to carry its own
    model version, because the run header (`vasqa.py:464`) is written once.
    Name the option, which takes a path, and the result field, which holds a
    version string, unambiguously. Validate `refit_every` in the dry run.
  - **V4 resources.** JAX jit caches grow with every new training shape, so
    pad training sets to fixed shapes or clear the caches. Refits read a
    store that other runs write to, so they are not reproducible; say so in
    the result.
  - **V5.** Dirichlet counts updated from the chain's history are adaptation
    too: freeze them offline, or put them under `sampling_exact`.
    `ProposalKernel` takes static `move_weights`, so the per-state move
    distribution must be cached the way operator probabilities are.
  - **New V0 items (existing code, needed before V4):**
    - `ProposalModel.save` truncates in place (`proposal_model.py:454`), and
      `ProblemRecord.save` checks for the file and then writes it
      (`proposal_data.py:155`). Both can be torn by a concurrent reader. Use
      a temporary file plus `os.replace`.
    - `EditRecorder.write` is a buffered text write, so a pool-sized row
      larger than 8 KB can interleave with another run's row. Use one
      `os.write` on an `O_APPEND` descriptor per row, or one file per run.
    - `load_edits` must tolerate a truncated last line; today it raises
      `JSONDecodeError`.
    - The recorder opens before the `try` (`vasqa.py:546-564`), so a failure
      in the banner or the first evaluation leaks the handle.
    - `_cache`, `seen` and `steps` grow without bound when
      `warm_start=False`. This, not the recorder, is the real growth over a
      long chain.
  - **V0 widened.** `_require_jax` is dead code now that JAX is a core
    dependency (`proposal_model.py:519`, `_jax_energy.py:62`). The stale
    `[legacy-forces]` extra in `_jax_energy.py:70` goes in the same sweep.
- **Vera (2026-10-01), code read only, nothing computed.** Claims 1 and 4
  hold with caveats, claim 2 holds once reworded, and claim 3 was partly
  wrong. Section 4.2 now carries her corrections:
  - a sign flip acts on both spin qubits of the orbital
  - the sign rule after tapering
  - generic factor-graph loops are not gauge invariant
  - point-group symmetry gives an exact *gradient* mask but not an energy
    mask (the relaxed gain is second order)
  - `N`/`S_z` conservation depends on the pool
  - O(3) is preferred, not required
  - `lda-dirac` is out of scope
  The finding that still holds: the signed coefficient column makes the
  current model depend on an arbitrary MO sign that no step of the
  pipeline fixes.
- **Rita, on the transfer (2026-10-01), all applied:**
  - the `start` line is now right when the pool is empty;
  - the transfer is kept on *cost*, not energy, matching the chain;
  - `_start_description` is initialized in `__init__`;
  - a system-identity guard (same atoms, same pool labels) stops a
    different molecule's ansatz from being renamed onto this one;
  - the `min_length` wording is fixed.
- **Rita, on V10 (2026-10-01):** no violations; MH exactness, memory and
  `condition()` were checked. Applied: the tutorial no longer claims the
  update needs `transfer=True`, and `_insertions` is reset per run. Open:
  repeated `(state, operator)` insertions can crowd older rows out of the
  400-row cap (deduplicate before conditioning); the recorded `model`
  version of a conditioned model is not in the store, so those rows name a
  model that only lived for its trajectory.
- **Ted (2026-10-01): 44 of 44 tests pass** in the three VALQA files (5.7 s,
  0.62 GB). Coverage is 95% for `proposal_data.py` and 91% for
  `proposal_model.py`.
  - The CLAUDE.md coverage command (`pytest --cov`) crashes at start-up on
    Python 3.14 (`ImportError: cannot load module more than once per process`
    from numpy, raised at `conftest.py:163`). `coverage run -m pytest` works.
  - **Missing tests, checkpoint** (`test_proposal_model.py::TestTheMixture`,
    `test_valqa.py::TestAReadyModel`):
    - a file with another schema is refused (`proposal_model.py:466`)
    - an unready model with an empty `cholesky` round-trips
    - a `use_graph=False` model round-trips with an equal `predict`
    - every metadata field survives the round trip, not just `version`,
      `ready` and `reason`
    - pickled object arrays are refused
    - a corrupt file fails with a clear error
    - the model is read at construction, so overwriting the file before
      `run` changes nothing
    - a schema mismatch fails in `__init__`, not in `run`
  - **Missing tests, appending** (`test_proposal_data.py`):
    - a second run keeps the first run's rows byte for byte
    - a truncated last line gets a named error, or is skipped (pin whichever
      is chosen)
    - a new recorder appends after a partial line on its own line; today it
      would be glued to the partial line
    - `close()` is idempotent, and `write` after `close` raises
    - the recorder is closed when a run raises, and the rows written before
      the failure stay on disk
    - rows of another schema are refused (`proposal_data.py:287`)
    - a directory with problems but no edits is refused (`:290`)
    - two problems share one directory
    - an existing problem file is not rewritten

---

## 6. The trajectory spec mapped onto the repository (2026-10-01)

The spec's software architecture (a standalone `valqa/` package, PyTorch
Geometric, GPyTorch, Python 3.11) does not apply. Algorithms are reached
only through `Mandacaru(method=...)`, JAX stays a core dependency, and the
project targets Python 3.14. Each spec section maps onto existing modules:

| Spec | Status | Where |
| --- | --- | --- |
| 1 geometry -> H -> state -> forces -> next X | exists | `calculator.py` (ASE), `forces` |
| 3 architecture, moves | exists | `mcas.py::Action`, `ProposalKernel` |
| 3 edit distance | done (V8) | `mcas.edit_distance` |
| 4 parameter transfer, REPLACE options | done (V8) | `Action.transfer(replace_start=)` |
| 5 uniform / gradient `q_0` | exists | `proposal="uniform" \| "gradient"`, `gradient_softmax` |
| 6 `q(C'\|C_n,H_n,H_{n+1})` | new (V10) | |
| 7 Fermion -> JW -> Pauli, variable size | exists | `core/`, `ProblemRecord` |
| 8-10 factor graph, relational MP, pooling | exists, smaller (V10) | `proposal_model.py::HamiltonianGraph`, `embed` |
| 11 operator embedding from qubit embeddings | exists | `pool_incidence` |
| 11-12 circuit and move embeddings | partial (V5, V10) | insert only, position-free |
| 13-14 GP, UCB softmax, exploration mixture | exists (gradient mixture) | `ProposalModel`, `BoundProposal` |
| 15 offline multi-molecule data | exists (single geometry rows) | `EditRecorder`, shared store |
| 16 online GP during a trajectory | decided (2A'), new (V10) | |
| 17 trajectory-aware splits, prequential | partial | `assess_data_volume(group_by=)` |
| 18 orbital tracking | new (V9) | |
| 19 transfer / reconstruction | done, basic rule (V8) | `transfer=`, `start` |
| 20-21, 26 baselines, metrics, ablations | partial (V11) | |
| 22 data classes | mostly exist under other names | `Action`, `_Evaluated`, `MCASStep`, `ProblemRecord`, `VASQAResult` |
| 23 reproducibility, restart | seeds yes; restart no (V8) | |
