# -*- coding: utf-8 -*-
# file: examples/39_mcas_vqe.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Water (H2O) with MCAS-VQE: searching the ansatz with a Markov chain.

MCAS-VQE -- the Variational, Adaptive and Stochastic Quantum Algorithm,
``Mandacaru(method="mcas-vqe")`` -- treats the *structure* of the ansatz as the
unknown.  An ansatz is an ordered sequence of operators taken from a pool,

.. math::

    |\psi(C, \theta)\rangle = e^{\theta_L A_{\mu_L}} \cdots
                              e^{\theta_1 A_{\mu_1}} |\mathrm{HF}\rangle,
    \qquad C = (\mu_1, \ldots, \mu_L),

and MCAS-VQE explores the space of such sequences with a Markov chain
(Markov Chain Ansatz Search, MCAS).  Each step of the chain

1. **proposes a move** on the current sequence ``C``:

   * ``insert`` -- put a pool operator into one of the ``L + 1`` slots;
   * ``delete`` -- take one operator out;
   * ``replace`` -- swap one operator for a different pool operator;
   * ``swap`` -- exchange the positions of two different operators;

   the new operator of an ``insert`` or ``replace`` is drawn from a
   **softmax of the pool gradients** at the current state,

   .. math::

       P(\mu) = \frac{\exp\!\left(\dfrac{|g_\mu|}{\tau\, g_{\max}}\right)}
                     {\displaystyle\sum_{\nu=1}^{M}
                      \exp\!\left(\dfrac{|g_\nu|}{\tau\, g_{\max}}\right)},
       \qquad g_\mu = 2\,\mathrm{Re}\langle H\psi|A_\mu\psi\rangle,
       \qquad g_{\max} = \max_\nu |g_\nu|,

   with the same softmax temperature :math:`\tau` in the numerator and in
   every term of the normalizing sum over the ``M`` pool operators, so the
   operators ADAPT-VQE would pick are proposed most often while every
   other operator keeps a positive probability;

2. **relaxes the proposal with VQE**: the angles are carried over from the
   current state (a new operator starts at zero, so an insertion begins at
   exactly the current state) and the classical optimizer minimizes the energy,
   giving :math:`\widehat E(C')`;

3. **accepts or rejects it** by the Metropolis-Hastings rule on the cost
   :math:`F(C) = \widehat E(C) + \lambda_L L`,

   .. math::

       \alpha = \min\left\{1,\; e^{-[F(C') - F(C)]/T}\,
                \frac{q(C \mid C')}{q(C' \mid C)}\right\},

   where ``T`` is the *architecture temperature* and ``q`` the probability of
   proposing one sequence from the other (an insertion is undone by a
   deletion, and the two are not equally likely, which the ratio corrects).

Where ADAPT-VQE only ever *appends* the operator with the largest energy
gradient and never takes one back, the chain can delete, replace and reorder
operators, so an early choice can be revised.  The price is that every
proposal is a full re-optimization.

The gradient softmax is what makes the search work on a pool this size.  Most
of the 92 operators have zero gradient at any given state -- at the
Hartree-Fock reference every single excitation does (Brillouin) -- and an
operator inserted at zero angle with zero gradient sits at a stationary point:
the optimizer leaves it at zero and it fills the ansatz without lowering the
energy.  Drawn uniformly (``proposal="uniform"``) such operators are most of
the proposals, and on this problem 200 uniform steps recover only about 20 %
of the correlation energy; the softmax recovers about 99 %.

The problem is the frozen-core water of ``03_ADAPTVQE_H2O.py``: HAO basis,
oxygen ``1s`` frozen, 6 active spatial orbitals on **12 qubits** with a
``(4, 4)`` closed shell, and the QEB (qubit-excitation) pool.  The script

* runs MCAS-VQE as an ASE calculator and writes the full run log, with its
  ``[MARKOV CHAIN]`` table, to ``examples/data/output_H2O_mcas_vqe.txt``;
* explains the chain from the result object: the move statistics, the first
  steps of the trajectory, the lowest-cost ansatz and where the chain ended;
* compares with ADAPT-VQE on the same qubit Hamiltonian and with the exact
  ground state of the ``(4, 4)`` sector;
* writes the trajectory to ``examples/data/h2o_mcas_vqe_trajectory.csv`` and,
  when Matplotlib is available, plots it to ``h2o_mcas_vqe_trajectory.png``.

.. note::

   On this coarse uniform grid the absolute energies are **qualitative** (see
   ``03_ADAPTVQE_H2O.py``); the comparison between the two algorithms on one
   and the same Hamiltonian is what the example is about.  The chain is
   stochastic: another ``seed`` gives another trajectory and, within a short
   run, possibly another final energy.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.units import from_hartree

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

#: Seed of the chain's random stream: the same seed replays the same chain.
SEED = 2026


def sector_fci(pauli_hamiltonian, n_qubits, na, nb):
    """Lowest eigenvalue (Hartree) in the ``(na, nb)`` particle/spin sector.

    Every pool operator conserves the number of alpha electrons (qubits
    ``0..M-1``) and beta electrons (qubits ``M..2M-1``), so the chain never
    leaves the Hartree-Fock sector and that sector's exact ground state is the
    right target.  Qubit 0 is the most significant bit.
    """
    M = n_qubits // 2
    keep = np.array([
        i for i in range(2 ** n_qubits)
        if sum((i >> (n_qubits - 1 - q)) & 1 for q in range(M)) == na
        and sum((i >> (n_qubits - 1 - q)) & 1 for q in range(M, n_qubits)) == nb])
    block = pauli_hamiltonian.to_sparse_matrix()[np.ix_(keep, keep)].toarray()
    return float(np.linalg.eigvalsh(0.5 * (block + block.conj().T)).min())


# --------------------------------------------------------------------------- #
# 1. The molecule: O at the cell center, O-H = 0.958 Angstrom, 104.5 degrees.
# --------------------------------------------------------------------------- #
theta = np.deg2rad(104.5) / 2.0
r = 0.958
atoms = Atoms("OH2",
              positions=[[5.0, 5.0, 5.0],
                         [5.0, 5.0 + r * np.sin(theta), 5.0 + r * np.cos(theta)],
                         [5.0, 5.0 - r * np.sin(theta), 5.0 + r * np.cos(theta)]],
              cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
              pbc=True)

# --------------------------------------------------------------------------- #
# 2. The MCAS-VQE calculator.  Every option below is a MCAS-VQE option except the
#    problem setup (basis, grid, active space, pool), which is ADAPT-VQE's.
# --------------------------------------------------------------------------- #
LOG = os.path.join(DATA, "output_H2O_mcas_vqe.txt")
atoms.calc = Mandacaru(
    method="mcas-vqe",
    # The problem: HAO orbitals on a 0.30 Angstrom grid, oxygen 1s frozen.
    basis={"name": "HAO"},
    h=0.30,
    active_space={"frozen": "auto"},
    # The pool the sequences are drawn from.  QEB operators are the
    # fermionic excitations without their parity strings: cheap circuits,
    # and they conserve the particle number, so the chain stays in (4, 4).
    pool="qeb",
    # Chain length: 200 proposals, i.e. 200 VQE relaxations.
    max_steps=200,
    # The ansatz may hold at most 30 operators; the chain starts from the
    # empty ansatz (the Hartree-Fock state itself).
    max_length=50,
    # New operators come from a softmax of |gradient| at the current state.
    # tau = 0.1 of the largest gradient makes the steepest operator e^10 times
    # as likely as a zero-gradient one: close to ADAPT's greedy pick, but
    # still stochastic.  (The default is 0.2; "uniform" ignores gradients.)
    proposal="gradient",
    proposal_temperature=0.2,
    # How often each move is proposed (relative weights among the moves that
    # are possible at the current length).  Equal weights are the default;
    # they are spelled out here to show the knob.
    move_weights={"insert": 0.5, "delete": 0.3, "replace": 0.1, "swap": 0.1},
    # Architecture temperature in eV, annealed geometrically from 50 meV --
    # uphill proposals of a few tens of meV are often accepted early on, so
    # the chain can leave a poor region -- to 0.1 meV, where only downhill
    # moves survive.  It weights ansatz *structures* and has nothing to do
    # with the temperature of the molecule.
    temperature={"initial": 0.20, "final": 1e-4},
    # Each operator costs 1 meV: an operator is kept only if it lowers the
    # energy by more than that.  Without a penalty an insertion can never
    # raise the optimal energy, and the chain would drift to max_length.
    length_penalty=1e-4,
    # Relax each proposal from the angles the move carried over (the default).
    # warm_start=False would start every proposal from zero angles instead,
    # which makes each sequence's cost a fixed number (memoized) at the price
    # of more optimizer work.
    warm_start=True,
    seed=SEED,
    # The classical optimizer of every relaxation.
    optimizer={"method": "SLSQP", "maxiter": 200, "tol": 1e-8},
    # The run log: [SYSTEM], [BASIS], [ELECTRONS], [OPTIMIZATION SETUP],
    # [MARKOV CHAIN] (one row per proposal), the summary and [PERFORMANCE].
    txt=LOG,
)

# Asking ASE for the energy builds the Hamiltonian and runs the whole chain.
t0 = time.perf_counter()
energy_ev = atoms.get_total_energy()            # eV (ASE convention)
mcas_vqe_seconds = time.perf_counter() - t0
result = atoms.calc.result                      # a MCASVQEResult, energies in eV
n_qubits = atoms.calc.n_qubits
na, nb = atoms.calc.num_particles

assert n_qubits == 12 and (na, nb) == (4, 4), "unexpected active space"
assert result.optimal_energy < result.reference_energy, "no correlation found"

# --------------------------------------------------------------------------- #
# 3. What the chain did.
# --------------------------------------------------------------------------- #
rule = "-" * 72
print(rule)
print(f"H2O, frozen-core HAO: {n_qubits // 2} active orbitals, {n_qubits} "
      f"qubits, (n_alpha, n_beta) = ({na}, {nb})")
print(f"pool: {atoms.calc.pool.name} ({len(atoms.calc.solver._pool_ops)} "
      f"operators), seed {result.seed}, {len(result.steps)} steps in "
      f"{mcas_vqe_seconds:.1f} s")
print(rule)

print("Moves proposed and accepted:")
for move, (taken, tried) in result.acceptance_by_move.items():
    rate = taken / tried if tried else 0.0
    print(f"  {move:<8} {taken:4d} of {tried:4d}  ({rate:5.1%})")
print(f"  overall acceptance {result.acceptance_rate:.1%}; "
      f"{result.num_architectures} distinct sequences evaluated with "
      f"{result.num_evaluations} energy evaluations and "
      f"{result.num_screenings} pool-gradient screenings")

# The first steps, one line each: the move, the proposed ansatz length, its
# relaxed energy, dE against the current state, and the decision.
print("\nFirst ten steps (energies in eV):")
print(f"  {'step':>4} {'move':<8} {'L':>2} {'E proposed':>15} "
      f"{'dE':>10} {'T':>9} {'acc':>4}  action")
previous = result.reference_energy
for step in result.steps[:10]:
    print(f"  {step.step:>4} {step.move:<8} {len(step.proposed):>2} "
          f"{step.proposed_energy:>15.6f} "
          f"{step.proposed_energy - previous:>+10.6f} "
          f"{step.temperature:>9.2e} {'yes' if step.accepted else 'no':>4}  "
          f"{step.action}")
    previous = step.current_energy
print("  (every step is in the [MARKOV CHAIN] table of the log)")

# Three different "answers", which agree only when the chain has settled.
print("\nStates reported:")
print(f"  lowest cost   E = {result.optimal_energy:.6f} eV with "
      f"{result.num_operators} operators  <- the calculator's energy")
print(f"  lowest energy E = {result.best_energy:.6f} eV with "
      f"{len(result.best_energy_operators)} operators")
print(f"  chain's end   E = {result.final_energy:.6f} eV with "
      f"{len(result.final_operators)} operators")
# With a length penalty the reported ansatz is the lowest *cost*, E + 1 meV
# per operator: a longer ansatz is kept only if each extra operator buys more
# than 1 meV.
penalty = atoms.calc.length_penalty                 # eV per operator
best_energy_cost = (result.best_energy
                    + penalty * len(result.best_energy_operators))
print(f"  cost F = E + {penalty * 1e3:g} meV x operators: "
      f"{result.optimal_cost:.6f} eV (lowest cost) vs "
      f"{best_energy_cost:.6f} eV (lowest energy)")
print("  lowest-cost operator sequence (applied first to last):")
for k, label in enumerate(result.operators, 1):
    print(f"    {k:2d}. {label}")

# --------------------------------------------------------------------------- #
# 4. The same Hamiltonian with ADAPT-VQE, and the exact sector ground state.
#    The qubit Hamiltonian MCAS-VQE used is passed directly, so nothing is
#    rebuilt: both algorithms see exactly the same operator.
# --------------------------------------------------------------------------- #
t0 = time.perf_counter()
adapt = Mandacaru(method="adapt-vqe",
                  hamiltonian=atoms.calc.hamiltonian,
                  num_particles=(na, nb),
                  n_spatial_orbitals=n_qubits // 2,
                  pool="qeb",
                  max_iterations=50,
                  convergence={"gradient": 1e-3},
                  optimizer={"method": "SLSQP", "maxiter": 200, "tol": 1e-8},
                  trace=False).run()
adapt_seconds = time.perf_counter() - t0

exact_ev = float(from_hartree(
    sector_fci(atoms.calc.hamiltonian, n_qubits, na, nb), "eV"))
correlation = exact_ev - result.reference_energy

print(rule)
print(f"{'':<12} {'E (eV)':>15} {'E - exact (eV)':>15} {'operators':>10} "
      f"{'evaluations':>12} {'time (s)':>9}")
for name, energy, operators, evaluations, seconds in (
        ("Hartree-Fock", result.reference_energy, 0, 0, 0.0),
        ("MCAS-VQE", result.optimal_energy, result.num_operators,
         result.num_evaluations, mcas_vqe_seconds),
        ("ADAPT-VQE", adapt.optimal_energy, adapt.num_operators,
         adapt.num_evaluations, adapt_seconds)):
    print(f"{name:<12} {energy:>15.6f} {energy - exact_ev:>+15.6f} "
          f"{operators:>10d} {evaluations:>12d} {seconds:>9.1f}")
print(f"{'exact (4,4)':<12} {exact_ev:>15.6f}")
recovered = (result.optimal_energy - result.reference_energy) / correlation
print(f"MCAS-VQE recovered {recovered:.1%} of the sector correlation energy "
      f"({correlation:.4f} eV).")
print(rule)

# Every energy the chain evaluated is variational: none can fall below the
# exact ground state of the sector.
assert min(s.proposed_energy for s in result.steps) >= exact_ev - 1e-6

# --------------------------------------------------------------------------- #
# 5. The trajectory, as data and (optionally) as a picture.
# --------------------------------------------------------------------------- #
CSV = os.path.join(DATA, "h2o_mcas_vqe_trajectory.csv")
with open(CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["step", "move", "action", "proposed_length",
                     "proposed_energy_eV", "proposed_cost_eV",
                     "current_energy_eV", "current_cost_eV", "temperature_eV",
                     "log_acceptance", "accepted", "evaluations"])
    for s in result.steps:
        writer.writerow([s.step, s.move, s.action, len(s.proposed),
                         f"{s.proposed_energy:.10f}", f"{s.proposed_cost:.10f}",
                         f"{s.current_energy:.10f}", f"{s.current_cost:.10f}",
                         f"{s.temperature:.6e}", f"{s.log_acceptance:.6f}",
                         int(s.accepted), s.num_evaluations])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:                                 # optional dependency
    plt = None

if plt is not None:
    steps = np.array([s.step for s in result.steps])
    proposed = np.array([s.proposed_energy for s in result.steps]) - exact_ev
    current = np.array([s.current_energy for s in result.steps]) - exact_ev
    length = np.array([len(s.current) for s in result.steps])
    accepted = np.array([s.accepted for s in result.steps])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    top.semilogy(steps[~accepted], proposed[~accepted] + 1e-9, "x",
                 color="0.6", ms=4, label="rejected proposal")
    top.semilogy(steps[accepted], proposed[accepted] + 1e-9, "o",
                 color="C0", ms=3, label="accepted proposal")
    top.semilogy(steps, current + 1e-9, "-", color="C1", lw=1.5,
                 label="current state")
    top.axhline(adapt.optimal_energy - exact_ev + 1e-9, color="C2", ls="--",
                label="ADAPT-VQE")
    top.set_ylabel("E - E(exact, 4,4)  (eV)")
    top.legend(fontsize=8)
    bottom.step(steps, length, where="post", color="C1")
    bottom.set_ylabel("operators in current ansatz")
    bottom.set_xlabel("Markov-chain step")
    fig.suptitle("H2O (12 qubits, QEB pool): MCAS-VQE trajectory")
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "h2o_mcas_vqe_trajectory.png"), dpi=150)

print(f"run log:    {LOG}")
print(f"trajectory: {CSV}")
if plt is not None:
    print(f"plot:       {os.path.join(DATA, 'h2o_mcas_vqe_trajectory.png')}")
