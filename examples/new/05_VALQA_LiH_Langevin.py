# -*- coding: utf-8 -*-
# file: examples/new/05_VALQA_LiH_Langevin.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# Langevin molecular dynamics of LiH with VALQA, learning along the trajectory.
#
# Every MD step is a new geometry, and VALQA searches an ansatz for it with a
# Markov chain.  Three options make the steps work together instead of
# starting over:
#
# - transfer=True: each chain starts from the ansatz (operators and angles)
#   the previous geometry reported, after following the molecular orbitals
#   from one geometry to the next, and runs only transfer_steps proposals.
#   The `start` line says what was carried over, or why the chain started
#   from the empty ansatz instead.
# - record=...: every evaluated proposal, with the screened insertions, is
#   appended to an edit store, linked along the trajectory.  This is the data
#   train_proposal_model() learns the proposal from.
# - update_between_geometries=True: once an offline proposal model that has
#   passed its readiness check is in use, an online model takes its place:
#   before each chain the model is updated with the previous geometry's
#   insertions -- scored first, test-then-train -- and the `model_update`
#   line reports how well it predicted them.  The gradient proposal keeps its
#   share of the mixture throughout.  The offline model comes from the shared
#   store (MANDACARU_PROPOSAL_DATA): train_proposal_model() trains it from
#   recorded MCAS-VQE chains and picks the kind that ranks the insertions of
#   a state best.  Without a ready model the gradient proposal is in effect,
#   and the line says why.
#
# Each chain is run with one frozen proposal, so its Metropolis-Hastings test
# stays exact; learning happens only between geometries.
#
# The basis is PAW-LCAO: it is the basis whose forces are trusted for moving
# atoms.  Langevin dynamics is a thermostat, not energy conserving, so the
# total energy is expected to drift toward the bath temperature.

import os

import numpy as np
from ase import Atoms, units
from ase.constraints import FixCom
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta

from mandacaru import Mandacaru

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

TEMPERATURE = 300.0                                       # bath temperature (K)
TIMESTEP = 0.5 * units.fs                                 # light hydrogen: 0.5 fs
FRICTION = 0.01 / units.fs                                # Langevin friction
MD_STEPS = 20
SEED = 7

# Molecule: LiH near its PAW-LCAO-SZ bond length, in a 10 Angstrom box.
atoms = Atoms("LiH",
              positions=[[5.0, 5.0, 4.2],    # Li
                         [5.0, 5.0, 5.8]],   # H
              cell=[10.0, 10.0, 10.0])

atoms.calc = Mandacaru(method="valqa",                     # learned-proposal Markov chain
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.20,                             # grid spacing (Angstrom)
                       pool="qeb",                         # qubit-excitation operators
                       max_steps=40,                       # the first geometry, from empty
                       transfer=True,                      # carry the ansatz between steps
                       transfer_steps=10,                  # every later geometry
                       length_penalty=1e-3,                # eV per operator
                       screen_insertions=2,                # measured insertions per state
                       update_between_geometries=True,     # learn between geometries
                       record=os.path.join(OUT, "valqa_md_edits"),
                       seed=SEED,
                       txt=os.path.join(OUT, "output_05.txt"),
                       references=os.path.join(OUT, "references_05.bib"))

rng = np.random.default_rng(SEED)
thermalize_momenta(atoms, temperature_K=TEMPERATURE, rng=rng)
# The center of mass stays put through a constraint: Langevin's own
# fixcm=True does not sample the NVT distribution exactly for a molecule this
# small.
atoms.set_constraint(FixCom())
dynamics = Langevin(atoms, TIMESTEP, temperature_K=TEMPERATURE,
                    friction=FRICTION, fixcm=False, rng=rng)
trajectory = Trajectory(os.path.join(OUT, "05_lih_langevin.traj"), "w", atoms)
dynamics.attach(trajectory.write, interval=1)

evaluations = []


def report():
    """One line per MD step: the dynamics, then what the chain did."""
    result = atoms.calc.result
    evaluations.append(result.num_evaluations)
    bond = atoms.get_distance(0, 1)
    potential = atoms.get_potential_energy()
    kinetic = atoms.get_kinetic_energy()
    print(f"step {len(evaluations) - 1:3d}  r = {bond:.4f} A  "
          f"E_pot = {potential:.6f} eV  E_tot = {potential + kinetic:.6f} eV  "
          f"T = {atoms.get_temperature():6.1f} K  "
          f"operators = {result.num_operators}  "
          f"evaluations = {result.num_evaluations}")
    print(f"           start: {result.start}")
    print(f"           model update: {result.model_update}")


dynamics.attach(report, interval=1)
dynamics.run(MD_STEPS)
trajectory.close()

print()
print(f"VQE energy evaluations: first geometry {evaluations[0]}, later "
      f"geometries {np.mean(evaluations[1:]):.0f} on average")
print(f"trajectory: {os.path.join(OUT, '05_lih_langevin.traj')}")
print(f"edit store: {os.path.join(OUT, 'valqa_md_edits')}")
