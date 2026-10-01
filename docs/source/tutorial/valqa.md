# VALQA: A Learned Proposal for the Ansatz Search

MCAS-VQE ({doc}`mcas_vqe`) draws each new operator from a softmax of the pool
gradients at the current state. `method="valqa"` (Variational Adaptive
Learnable Quantum Algorithm) runs the same Markov chain and draws the new
operator from a **trained model** instead. The model reads the qubit
Hamiltonian as a graph, predicts how much inserting each pool operator would
lower the energy, and turns the predictions into probabilities. Moves,
positions, VQE relaxation and the Metropolis-Hastings test are MCAS-VQE's.

The model starts where MCAS-VQE is. Until a model has been trained on enough
recorded edits, and has passed a readiness check, VALQA draws exactly what
MCAS-VQE's gradient proposal draws: the same chain for the same seed.

## The workflow

0. **Name the shared store once.** Every chain records into it, training
   reads it, and the trained model is saved in it:

   ```console
   $ mandacaru --set-proposal-data ~/mandacaru-proposals
   ```

   This writes `MANDACARU_PROPOSAL_DATA` to `~/.zshrc` or `~/.bashrc` (and
   creates the directory). Data then accumulates across all your runs.

1. **Record.** With the store set, every MCAS-VQE and VALQA run records by
   default. Each proposal, rejected ones included, is appended to
   `edits.jsonl`, and the Hamiltonian, reference and pool it searched are
   stored once in `problems/`. Along an ASE trajectory a row also carries
   its `trajectory`, its `geometry_step` and the `previous_problem`, whose
   file is saved into the same store:

   ```python
   atoms.calc = Mandacaru(method="mcas-vqe", basis="HAO", h=0.30,
                          active_space={"frozen": "auto"}, pool="qeb",
                          max_steps=150, warm_start=False, seed=1)
   atoms.get_potential_energy()
   ```

   `record="DIR"` sends one run elsewhere and `record=False` (`--no-record`)
   records nothing. `warm_start=False` is the better setting for collecting
   data: each architecture is then relaxed from zero angles, so its recorded
   energy is a property of the architecture rather than of the path.
   `screen_insertions=K` (`--screen-insertions K`) also relaxes and records
   K insertions at every state the chain visits, flagged `screened`. They are
   drawn from that state's operator distribution, beside the chain and
   without moving it. They are the data that judges a proposal's choice
   *within* a state, which is the decision it actually makes.

2. **Train and assess.** `train_proposal_model()` measures whether the data
   in the store is enough (see below), trains on all of it and saves the
   model, with its readiness report, as `proposal_model.npz` in the store.

   ```python
   from mandacaru.algorithms import train_proposal_model

   model = train_proposal_model()
   print(model.ready, model.report["reason"])
   ```

3. **Search.** VALQA takes the store's model without being told:

   ```python
   atoms.calc = Mandacaru(method="valqa", basis="HAO", h=0.30, pool="qeb",
                          max_steps=150)
   ```

   `proposal_model="model.npz"` names another model, and
   `proposal_model=False` uses none. A model that is not ready leaves the
   gradient proposal in effect; the `[OPTIMIZATION SETUP]` block's
   `proposal` line says why. Its version is on the result
   (`result.proposal_model`, `result.model_ready`) and in every recorded
   edit. Retraining replaces the file, and the next run picks it up. The
   model file has schema 2: a model trained before must be retrained with
   `train_proposal_model()`, and loading an old file raises a `ValueError`
   that says so.

On the command line: `--method valqa`, `--proposal-model PATH`,
`--update-between-geometries`, and `--record DIR` or `--no-record` for either
chain method.

## Along a trajectory

`transfer=True` works for VALQA exactly as for MCAS-VQE ({doc}`mcas_vqe`, "Along a
trajectory"): in a relaxation or scan, each geometry's chain starts from the
ansatz the previous geometry reported, and `transfer_steps` sets the shorter
chain that start allows.

```python
from ase.optimize import BFGS

atoms.calc = Mandacaru(method="valqa",
                       basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.20,
                       pool="qeb", max_steps=30, transfer=True,
                       transfer_steps=6, seed=1234)
BFGS(atoms).run(fmax=0.05)
print(atoms.calc.result.start)
```

The molecular orbitals are followed from one geometry to the next: operators
are renamed where orbitals exchanged places, and angles change sign where an
orbital did. The transferred ansatz is kept only if it relaxes below the
reference's cost at the new geometry; otherwise the chain starts empty, and
the `start` line of `[OPTIMIZATION SETUP]` says why. The details are in the
MCAS-VQE tutorial. The start does not change what the chain samples. The
command-line flags are `--transfer`, `--transfer-steps N` and
`--transfer-threshold S`.

### Two geometries

Along any ASE trajectory, with or without `transfer`, the model is
conditioned on two Hamiltonians, the previous geometry's and the current one,
$q(C'\mid C_n, H_n, H_{n+1})$. Its features are $z_{n+1}$, $z_n$ and
$\Delta z = z_{n+1} - z_n$, where $z$ is a Hamiltonian's graph embedding. A
single geometry is the case $\Delta z = 0$, and so is every recorded row
without a previous problem.

### Updating between geometries

By default the offline model, trained from recorded MCAS-VQE chains, is the
same for every geometry. With `update_between_geometries=True`
(`--update-between-geometries`), an online model takes its place: before
each chain, the model is updated with the insertions the previous geometry
evaluated. The online model replaces the offline one in the mixture; the
gradient proposal keeps its share, so the exploration it provides is not
traded away:

```python
atoms.calc = Mandacaru(method="valqa",
                       basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.20,
                       pool="qeb", max_steps=30, transfer=True,
                       transfer_steps=6, update_between_geometries=True,
                       seed=1234)
BFGS(atoms).run(fmax=0.05)
print(atoms.calc.result.model_update)
```

Those insertions are first scored by the model that has not seen them
(test-then-train), and the line reports their count, the error and rank
correlation of that prediction, and the model version before and after. A
ranker model takes a sequential Bayesian step: its weights are refitted on
the new within-state pairs with the previous weights as the prior, so the
offline training is refined, not forgotten. A graph model conditions its
Gaussian process; its network and kernel stay as trained. Each chain still
runs with one frozen model, so its
Metropolis-Hastings ratio stays exact, and the updated model lives for the
trajectory: the store's model file is not changed. The update needs a ready
model and a previous geometry of the same molecule and pool (`transfer` is not
required); otherwise `model_update` says `none` and why.
The `proposal` line of `[OPTIMIZATION SETUP]` carries the same text.

## The proposal

The operator an `insert` or `replace` places is drawn from

$$
P(\mu\mid C,H)=(1-\varepsilon)\,p_{\mathrm{ML}}(\mu\mid C,H)
+\varepsilon\,p_\nabla(\mu\mid C),
$$

where $p_\nabla$ is MCAS-VQE's gradient softmax and $\varepsilon = 1$ until the
model is ready (0.3 afterwards). Because $\varepsilon > 0$, every operator
keeps a positive probability, so every insertion has its reverse deletion.
The model is frozen for the run and the distribution at a state depends only
on that state, so the Metropolis-Hastings ratio stays exact: the forward
probability is taken at the current state and the reverse at the proposed one.

$p_{\mathrm{ML}}$ comes from three pieces:

- **A factor graph of the Hamiltonian.** One node per qubit, one per
  non-identity Pauli term of $H = c_0 + \sum_\alpha c_\alpha P_\alpha$, and an
  edge labeled X, Y or Z wherever a term acts on a qubit. A term acting on
  three qubits is one node with three edges, not three pairwise couplings.
  A term's features use the magnitude of its coefficient, not its sign: the
  sign of a term with X or Y letters follows the arbitrary sign of a molecular
  orbital, so two gauges of one molecule must look the same.
- **A message-passing network.** Two layers of width 32 let terms gather from
  their qubits and qubits from their terms, with one weight matrix per Pauli
  letter. The Hamiltonian's embedding pools qubits and terms by mean and
  sum. A pool generator (a sum of Pauli strings) is embedded from the
  qubit embeddings on each string's support. The same operator therefore gets
  a different representation in a different Hamiltonian.
- **A Gaussian process.** It predicts the energy change of inserting
  operator $\mu$ into circuit $C$, with an uncertainty. Its inputs are the
  graph embeddings (of both geometries' Hamiltonians along a trajectory) plus
  what is known *before* the insertion is evaluated:
  the operator's pool gradient, the circuit's operators and length, and the
  current energy above the reference.

The score $b_\mu = -\mu_\mu + \kappa\sigma_\mu$ favors predicted improvement
and, through the uncertainty, operators the model has not seen enough of;
$p_{\mathrm{ML}} = \mathrm{softmax}(b/\tau)$. The score does not depend on the
insertion slot: positions stay uniform, as in MCAS-VQE.

## Is there enough data?

`train_proposal_model` runs `assess_data_volume`, which answers this on the
recorded data. It holds out one **molecule** at a time, trains on the others
at a growing fraction of their edits, and ranks the held-out insertions with
three predictors:

- the graph model;
- the same Gaussian process on hand-made descriptors only, with no graph;
- the gradient heuristic $-|g_\mu|$, which is what MCAS-VQE draws from.
- a pairwise logistic ranker on the same descriptors, fitted only on the
  states with several recorded insertions: the simplest model of the choice
  within a state.

A model is **ready** when, over at least three held-out molecules, it
orders the insertions recorded at one chain state better than the gradient
does: its mean within-state Spearman correlation must beat the gradient's by
0.05, and beat it for most held-out molecules. That is the decision the
proposal makes, and a better ordering moves the whole softmax toward the
better insertions even when the first choice is the same. The pairwise
ranker is the first candidate; the graph model is chosen only when it also
beats the ranker. `train_proposal_model` saves the chosen kind with the model
(`model.kind`: `"ranker"` or `"graph"`), and the run log names it. A ranker
model and a graph model can both be updated between geometries (below). The
judgment needs states with several recorded
insertions, so record chains with `screen_insertions`. `group_by="hamiltonian"` holds out single geometries instead,
which tests transfer along a potential-energy curve, and
`group_by="trajectory"` holds out whole trajectories. Those are weaker claims
than the molecule, and the report records which grouping was used. Both
`assess_data_volume` and `train_proposal_model` take `group_by`.

The correlation above compares insertions across all held-out states, and
there a large gradient means a large gain almost by construction. The
proposal, however, only chooses among the insertions of one state. Each curve
point therefore also has `within_state`: over the held-out states with
several recorded insertions, each predictor's mean Spearman correlation
inside a state and its regret, which is the energy its first choice gave up
against the best insertion recorded there, in Hartree. The within-state
correlation is what the readiness rule above uses; the regret and the
across-state correlations are reported beside it.

`model.report["curve"]` is the learning curve: for each fraction, the
training labels used and the three correlations. A curve still rising at the
full fraction says more data would help. A graph model that never separates
from the descriptor baseline says the representation, not the data volume,
is the limit.

### What the first data showed

The first measurement (2026-09-30) recorded 942 insertions from MCAS-VQE chains
on H2, LiH, BeH2, H4, HF and H2O (HAO, frozen core, QEB pool). Holding out
each molecule in turn, the gradient heuristic ranked the held-out insertions
at a Spearman correlation of 0.78. The graph model reached 0.67, and the
descriptor-only model 0.70. The curve was flat between 78 and 400 training
labels, and stayed flat with every row (785 labels: 0.65). So the model is
not ready, and more chains on the same molecules
would not make it ready. The pool gradient is already close to a sufficient
predictor of an insertion's energy change. A model has to find what the
gradient misses, and six molecules, three of them with 3- to 8-operator
pools, are too few transfer tests to show that. Molecules with larger pools
are what the next recordings should add.

### Trajectory data with screened insertions

The second measurement (2026-10-01) recorded bond scans in the PAW-LCAO basis
(SZ, h = 0.20 Angstrom), the basis whose forces and energy surfaces are smooth
enough to move atoms on. Each scan has five geometries, two seeds and
`transfer=True` with `screen_insertions=4`, for H2, LiH, BeH2, H4, HF and
H2O. That gave 4939 insertions, in 1170 states with several measured
insertions. Holding out each molecule, over three training seeds:

| predictor | across states | within a state | regret (mHa) |
| --- | --- | --- | --- |
| gradient | 0.64 | 0.38 | 0.014 |
| pairwise ranker | 0.59 | 0.46 | 0.013 |
| graph model | 0.66 to 0.68 | 0.46 to 0.49 | 0.014 to 0.030 |
| descriptors only | 0.40 to 0.49 | 0.25 to 0.33 | 0.037 to 0.072 |

Within a state, both the graph model and a plain pairwise logistic ranker on
the descriptors order the insertions better than the gradient. Their first
choice, however, is no better: the regret is the same. Under the
within-state rule the pairwise ranker is ready (0.459 against 0.383, better
for H2, BeH2, HF and LiH, narrowly worse for H2O and H4), and it is the first
learned proposal VALQA uses. Conditioning on the previous geometry made no
consistent difference.
