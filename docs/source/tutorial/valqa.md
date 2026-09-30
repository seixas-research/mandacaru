# VALQA: A Learned Proposal for the Ansatz Search

VASQA ({doc}`vasqa`) draws each new operator from a softmax of the pool
gradients at the current state. `method="valqa"` (Variational Adaptive
Learnable Quantum Algorithm) runs the same Markov chain and draws the new
operator from a **trained model** instead. The model reads the qubit
Hamiltonian as a graph, predicts how much inserting each pool operator would
lower the energy, and turns the predictions into probabilities. Moves,
positions, VQE relaxation and the Metropolis-Hastings test are VASQA's.

The model starts where VASQA is. Until a model has been trained on enough
recorded edits, and has passed a readiness check, VALQA draws exactly what
VASQA's gradient proposal draws: the same chain for the same seed.

## The workflow

0. **Name the shared store once.** Every chain records into it, training
   reads it, and the trained model is saved in it:

   ```console
   $ mandacaru --set-proposal-data ~/mandacaru-proposals
   ```

   This writes `MANDACARU_PROPOSAL_DATA` to `~/.zshrc` or `~/.bashrc` (and
   creates the directory). Data then accumulates across all your runs.

1. **Record.** With the store set, every VASQA and VALQA run records by
   default. Each proposal, rejected ones included, is appended to
   `edits.jsonl`, and the Hamiltonian, reference and pool it searched are
   stored once in `problems/`:

   ```python
   atoms.calc = Mandacaru(method="vasqa", basis="HAO", h=0.30,
                          active_space={"frozen": "auto"}, pool="qeb",
                          max_steps=150, warm_start=False, seed=1)
   atoms.get_potential_energy()
   ```

   `record="DIR"` sends one run elsewhere and `record=False` (`--no-record`)
   records nothing. `warm_start=False` is the better setting for collecting
   data: each architecture is then relaxed from zero angles, so its recorded
   energy is a property of the architecture rather than of the path.

2. **Train and assess.** `train_proposal_model()` measures whether the data
   in the store is enough (see below), trains on all of it and saves the
   model, with its readiness report, as `proposal_model.npz` in the store.
   It needs JAX (`pip install 'mandacaru[learned-proposals]'`).

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
   edit. Retraining replaces the file, and the next run picks it up.

On the command line: `--method valqa`, `--proposal-model PATH`, and
`--record DIR` or `--no-record` for either chain method.

## The proposal

The operator an `insert` or `replace` places is drawn from

$$
P(\mu\mid C,H)=(1-\varepsilon)\,p_{\mathrm{ML}}(\mu\mid C,H)
+\varepsilon\,p_\nabla(\mu\mid C),
$$

where $p_\nabla$ is VASQA's gradient softmax and $\varepsilon = 1$ until the
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
- **A message-passing network.** Two layers let terms gather from their
  qubits and qubits from their terms, with one weight matrix per Pauli
  letter. A pool generator (a sum of Pauli strings) is embedded from the
  qubit embeddings on each string's support. The same operator therefore gets
  a different representation in a different Hamiltonian.
- **A Gaussian process.** It predicts the energy change of inserting
  operator $\mu$ into circuit $C$, with an uncertainty. Its inputs are the
  graph embeddings plus what is known *before* the insertion is evaluated:
  the operator's pool gradient, the circuit's operators and length, and the
  current energy above the reference.

The score $b_\mu = -\mu_\mu + \kappa\sigma_\mu$ favors predicted improvement
and, through the uncertainty, operators the model has not seen enough of;
$p_{\mathrm{ML}} = \mathrm{softmax}(b/\tau)$. The score does not depend on the
insertion slot: positions stay uniform, as in VASQA.

## Is there enough data?

`train_proposal_model` runs `assess_data_volume`, which answers this on the
recorded data. It holds out one **molecule** at a time, trains on the others
at a growing fraction of their edits, and ranks the held-out insertions with
three predictors:

- the graph model;
- the same Gaussian process on hand-made descriptors only, with no graph;
- the gradient heuristic $-|g_\mu|$, which is what VASQA draws from.

The model is **ready** when at least three molecules can be held out and,
with all the data, its mean held-out Spearman correlation beats both
baselines by 0.05 and beats the better one for most held-out molecules. The
gain has to come from the Hamiltonian representation, not merely from having
a surrogate. `group_by="hamiltonian"` holds out single geometries instead,
which tests transfer along a potential-energy curve. That is a weaker claim,
and the report records which grouping was used.

`model.report["curve"]` is the learning curve: for each fraction, the
training labels used and the three correlations. A curve still rising at the
full fraction says more data would help. A graph model that never separates
from the descriptor baseline says the representation, not the data volume,
is the limit.

### What the first data showed

The first measurement (2026-09-30) recorded 942 insertions from VASQA chains
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
