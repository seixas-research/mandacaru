# -*- coding: utf-8 -*-
# file: examples/new/06_MCAS-VQE_offline_training.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# Offline training of VALQA's proposal model from MCAS-VQE simulations.
#
# MCAS-VQE (Markov Chain Ansatz Search with VQE) searches ansatz structures
# with a Markov chain and the gradient proposal; it learns nothing.  Every
# proposal it evaluates can be recorded, and those records train the model
# that turns MCAS-VQE into VALQA -- the variational, Markov-chain adaptive and
# learnable algorithm.  This script is that offline step:
#
# 1. record MCAS-VQE chains on several molecules, each at several bond
#    lengths, into one edit store.  The PAW-LCAO basis is used throughout --
#    it is the basis whose forces are trusted for moving atoms, and the model
#    is meant for geometry optimization and molecular dynamics.
#    warm_start=False makes each recorded energy a property of the
#    architecture; screen_insertions records several measured insertions per
#    chain state, which is what judges a proposal's choice within a state.
# 2. train and assess the model with fit(): each molecule
#    is held out in turn, and a model is ready when it orders the insertions
#    of a state better than the gradient does.
#
# Point MANDACARU_PROPOSAL_DATA at the store (`mandacaru --set-proposal-data
# DIR`) and every later VALQA run picks the model up.  More molecules and
# more geometries make a better model: each held-out molecule is one test of
# transfer to a system the model has not seen.

import os

import numpy as np
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms import fit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
STORE = os.path.join(OUT, "offline_store")
os.makedirs(STORE, exist_ok=True)


def linear(symbols, spacings):
    """A linear molecule along z, centered in a 10 Angstrom box."""
    z = np.concatenate([[0.0], np.cumsum(spacings)])
    z -= z.mean()
    return Atoms(symbols, positions=[[5.0, 5.0, 5.0 + zi] for zi in z],
                 cell=[10.0, 10.0, 10.0])


# Molecules and the bond lengths (Angstrom) each is recorded at.
MOLECULES = {
    "H2": (lambda r: linear("H2", [r]), (0.65, 0.75, 0.85)),
    "LiH": (lambda r: linear("LiH", [r]), (1.45, 1.55, 1.65)),
    "BeH2": (lambda r: linear("HBeH", [r, r]), (1.25, 1.35, 1.45)),
    "H4": (lambda r: linear("H4", [r, r, r]), (0.85, 0.95, 1.05)),
}

for name, (build, distances) in MOLECULES.items():
    for r in distances:
        atoms = build(r)
        atoms.calc = Mandacaru(method="mcas-vqe",            # no learning
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               h=0.20,                       # grid spacing (Angstrom)
                               pool="qeb",
                               max_steps=40,
                               max_length=12,
                               warm_start=False,             # energy of the architecture
                               screen_insertions=4,          # insertions per state
                               length_penalty=1e-3,          # eV per operator
                               record=STORE,                 # the training data
                               seed=1,
                               trace=False)                  # one line per chain below
        energy = atoms.get_potential_energy()
        result = atoms.calc.result
        print(f"{name:5s} r = {r:.2f} A  E = {energy:.6f} eV  "
              f"operators = {result.num_operators}  "
              f"screened insertions = {result.num_screened_insertions}")

model = fit(STORE, fractions=(1.0,))
point = model.report["curve"][-1]["within_state"]
print()
print(f"recorded insertions : {model.report['insertions']} in "
      f"{model.report['groups']} molecules")
print(f"within-state Spearman, held-out molecules: "
      f"gradient {point['gradient']['spearman']:.3f}, "
      f"ranker {point['ranker']['spearman']:.3f}, "
      f"graph {point['graph']['spearman']:.3f}")
print(f"ready: {model.ready} ({model.kind if model.ready else 'none'}) -- "
      f"{model.report['reason']}")
print(f"model: {os.path.join(STORE, 'proposal_model.npz')}")
