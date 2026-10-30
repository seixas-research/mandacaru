# VALQA: A Learned Proposal for the Ansatz Search

MCAS-VQE ([the MCAS-VQE tutorial](mcas_vqe.md)) draws each new operator from a
softmax of the pool gradients at the current state. `method="valqa"` (Variational
Adaptive Learnable Quantum Algorithm) runs the same Markov chain and draws the
new operator from a **trained model** instead. The model reads the qubit
Hamiltonian as a graph, scores each pool operator, and turns the scores into
probabilities. Moves, positions, VQE relaxation and the Metropolis-Hastings test
are MCAS-VQE's. The graph model is trained and evaluated in Rust.

The model starts where MCAS-VQE is. Until a model has been trained on enough
recorded edits and has passed a readiness check, VALQA draws exactly what
MCAS-VQE's gradient proposal draws: the same chain for the same seed.

## The workflow

0. **Name the shared store once.** Every chain records into it, training reads it,
   and the trained model is saved in it:

   ```console
   $ mandacaru --set-proposal-data ~/mandacaru-proposals
   ```

   This writes `MANDACARU_PROPOSAL_DATA` to `~/.zshrc` or `~/.bashrc`, and creates
   the directory. A run can also name a store directly with `record="DIR"`.

1. **Record.** With a store set, every MCAS-VQE and VALQA run records by default.
   Each proposal, rejected ones included, is appended under `chains/`, and the
   Hamiltonian, reference and pool it searched are stored once under `problems/`.
   `record=False` records nothing. `warm_start=False` is the better setting for
   collecting data: each architecture is then relaxed from zero angles, so its
   recorded energy is a property of the architecture rather than of the path.
   `screen_insertions=K` also relaxes and records `K` insertions at every state the
   chain visits, drawn from that state's operator distribution beside the chain
   and without moving it. They are the data that judges a proposal's choice
   *within* a state, which is the decision it actually makes, and they require a
   record directory.

2. **Train and assess.** `fit()` measures whether the data in the store is enough,
   trains the graph neural network on all of it and saves the model, with its
   readiness report, as `proposal_model.npz` in the store. `kinds=` accepts only
   `("graph",)`.

3. **Search.** VALQA takes the store's model without being told.
   `proposal_model="model.npz"` names another model and `proposal_model=False`
   uses none. A model that is not ready leaves the gradient proposal in effect;
   `result.proposal_ready` and `result.proposal_reason` say which.

```python
import os
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.analysis.proposals import fit

store = os.path.abspath("proposals")
atoms = molecule("H2")
atoms.center(vacuum=3.0)

# 1. Record a few chains.
atoms.calc = Mandacaru(method="mcas-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25, pool="qeb", max_steps=20,
                       max_length=6, warm_start=False, screen_insertions=2,
                       record=store, seed=1)
atoms.get_potential_energy()

# 2. Train and assess. One molecule is not enough data, and the model says so.
model = fit(store)
print(model.ready, model.kind, model.report["reason"])

# 3. Search with the store's model.
atoms.calc = Mandacaru(method="valqa",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25, pool="qeb", max_steps=20,
                       max_length=6, record=store, seed=1)
atoms.get_potential_energy()
result = atoms.calc.result
print(result.proposal_ready, result.proposal_reason)
```

`fit(directory=None, path=None, *, group_by="molecule", fractions=(.25, .5, 1.),
seed=0, folds=None, gnn_epochs=100, mixing=.3, temperature=1., kinds=("graph",))`
reads the recorded observations and writes the native model archive and the
held-out assessment. Missing or insufficient observations produce a saved model
that is explicitly **not ready**, with the reason in `model.report["reason"]`;
they are never replaced by generated observations.

## When a model is ready

`fit` holds out one group at a time (`group_by="molecule"`, the default, or
`"hamiltonian"` for single geometries, or `"trajectory"` for whole trajectories),
trains on the others at a growing fraction of their edits, and ranks the held-out
insertions of each chain state. A model is **ready** when

* at least three held-out groups carry informative states (states with several
  recorded insertions, so record chains with `screen_insertions`),
* its mean within-state Spearman correlation beats the gradient heuristic's by
  at least 0.05, and
* it wins in a strict majority of those groups.

`folds=5` holds out a fifth of the groups at a time, which bounds the cost of
training the network once per fold. `model.report` holds the assessment. Holding
out single geometries or whole trajectories tests transfer along a
potential-energy curve and is a weaker claim than holding out a molecule.

## The proposal

The operator an `insert` or `replace` places is drawn from

$$
P(\mu\mid C,H)=(1-\varepsilon)\,p_{\mathrm{ML}}(\mu\mid C,H)
+\varepsilon\,p_\nabla(\mu\mid C),
$$

where $p_\nabla$ is MCAS-VQE's gradient softmax and $\varepsilon=1$ until the
model is ready, then a positive gradient share. Because $\varepsilon > 0$, every
operator keeps a positive probability, so every insertion has its reverse
deletion. The model is fixed throughout a chain and the distribution at a state
depends only on that state, so the Metropolis-Hastings ratio stays exact.

$p_{\mathrm{ML}}$ comes from a graph neural network in three pieces:

* **A factor graph of the Hamiltonian.** One node per qubit, one per non-identity
  Pauli term, and an edge labeled X, Y or Z wherever a term acts on a qubit. A
  term's features use the magnitude of its coefficient, not its sign, since the
  sign of a term with X or Y letters follows the arbitrary sign of a molecular
  orbital.
* **A message-passing network** of two layers, with one weight matrix per Pauli
  letter. A pool generator is embedded from the qubit embeddings on each string's
  support, so the same operator has a different representation in a different
  Hamiltonian.
* **A scoring head** that gives each pool operator a score from the Hamiltonian's
  embedding, the operator's embedding and what is known before the insertion is
  evaluated: its gradient, its multiplicity in the circuit, the circuit's length
  and the energy above the reference. It is trained with a first-choice loss: at
  every recorded state with two or more measured insertions, the softmax of the
  scores is fitted to a target concentrated on the insertions that lowered the
  energy most. It learns which operator to choose, not the size of the energy
  change.

## Along a trajectory

`transfer=True` works for VALQA exactly as for MCAS-VQE ([along a
trajectory](mcas_vqe.md)): in a relaxation or scan, each geometry's chain starts
from the ansatz the previous geometry reported, and `transfer_steps` sets the
shorter chain that allows. `update_between_geometries=True` is accepted, but the
graph model is carried over unchanged: its training is offline.

```python
from ase.optimize import BFGS

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="valqa",
                       basis={"name": "PAW-LCAO", "size": "SZ"}, grid_spacing=0.25,
                       pool="qeb", max_steps=30, transfer=True,
                       transfer_steps=6, seed=1234)
BFGS(atoms).run(fmax=0.05)
print(atoms.calc.result.transfer)
```

A learned proposal must earn its place on held-out data: until it does, VALQA
draws MCAS-VQE's gradient proposal. Optional Gaussian-process and pairwise-ranker
proposal models are listed as open work in the
[native coverage](../development/native_cutover.md).
