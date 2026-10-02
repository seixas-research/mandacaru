# -*- coding: utf-8 -*-
# file: examples/new/07_VALQA_fit_and_inference.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# Train VALQA's proposal model with `fit`, then use it on a new molecule.
#
# 1. Training.  `fit` reads an edit store of recorded MCAS-VQE chains
#    (example 06 writes one; any store recorded with `record=` and
#    `screen_insertions` works) and trains the graph neural network on it
#    (kind "graph": message passing over the qubit Hamiltonian's factor
#    graph and a scoring head, trained with a first-choice loss).  It holds
#    molecules out to check the network against the gradient rule on
#    molecules it has not seen, and saves the model and its readiness
#    report to `proposal_model.npz` in the store.
# 2. Inference.  `Mandacaru(method="valqa", proposal_model=...)` runs the
#    Markov-chain search on a molecule outside the training data.  The
#    operator of every insertion or replacement is sampled from
#    (1 - eps) * softmax(scores / tau) + eps * gradient softmax, and each
#    proposal is accepted or rejected by the Metropolis-Hastings test.  The
#    same search without a model is MCAS-VQE, shown for comparison.
#
# If the network does not pass the readiness check, VALQA uses the gradient proposal
# (it is then MCAS-VQE) and says so.  Example 06's store holds 4 molecules,
# which is usually too few: point VALQA_TRAINING_DATA at a larger store.

import os

from ase.build import molecule

from mandacaru import Mandacaru
from mandacaru.algorithms import fit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
STORE = os.environ.get("VALQA_TRAINING_DATA",
                       os.path.join(OUT, "offline_store"))     # example 06
MODEL = os.path.join(STORE, "proposal_model.npz")

# -- 1. Training -------------------------------------------------------------
model = fit(STORE,                      # the recorded MCAS-VQE chains
            fractions=(1.0,),           # judge on all the data (no curve)
            folds=5)                    # hold out a fifth of the molecules at a time
report = model.report
within = report["curve"][-1]["within_state"]
print(f"training data: {report['insertions']} insertions, "
      f"{report['groups']} molecules")
print("held-out within-state Spearman (higher orders a state's candidates "
      "better):")
for kind in ("gradient", "graph"):
    print(f"  {kind:9s} {within[kind]['spearman']:.3f}   first-pick regret "
          f"{1e3 * within[kind]['regret']:.3f} mHa")
print(f"ready: {model.ready}, kind: {model.kind if model.ready else 'none'} "
      f"-- {report['reason']}")
print(f"saved: {MODEL}")

# -- 2. Inference on a new molecule ------------------------------------------
# Formaldehyde is not among example 06's molecules.  The problem setup is the
# one the data were recorded with (PAW-LCAO SZ, 3 + 3 active orbitals).
setup = dict(basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.20, pool="qeb",
             active_space={"orbitals": {"occupied": 3, "virtual": 3},
                           "method": "mp2"},
             max_steps=40, max_length=30, length_penalty=1e-3, seed=1,
             record=False, trace=False)

for method, options in (("valqa", {"proposal_model": MODEL}),
                        ("mcas-vqe", {})):
    atoms = molecule("H2CO")
    atoms.center(vacuum=3.0)
    atoms.calc = Mandacaru(method=method, **setup, **options,
                           txt=os.path.join(OUT, f"output_07_{method}.txt"),
                           references=os.path.join(OUT, "references_07.bib"))
    energy = atoms.get_potential_energy()
    result = atoms.calc.result
    print(f"{method:9s} E = {energy:.6f} eV   operators = "
          f"{result.num_operators}   evaluations = {result.num_evaluations}")
    if method == "valqa":
        print(f"          model {result.proposal_model} in use: "
              f"{result.model_ready}")
