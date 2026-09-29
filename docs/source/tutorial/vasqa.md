# VASQA: Searching the Ansatz with a Markov Chain

ADAPT-VQE builds its ansatz greedily: each step appends the pool operator with
the largest energy gradient, and an operator once chosen stays. `method="vasqa"`
(Variational, Adaptive and Stochastic Quantum Algorithm) treats the operator
*sequence* itself as the thing to search. A Markov chain proposes changes to
the sequence, VQE relaxes the angles of each proposal, and a
Metropolis-Hastings test decides whether the chain moves there. An early
choice can therefore be undone.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
atoms.center(vacuum=2.5)
atoms.calc = Mandacaru(method="vasqa",
                       basis="HAO",
                       h=0.25,
                       pool="qeb",
                       max_steps=100,
                       max_length=10,
                       temperature={"initial": 0.1, "final": 1e-3},
                       length_penalty=1e-3,
                       seed=1234)
energy = atoms.get_potential_energy()

result = atoms.calc.result
print(result.operators)            # the lowest-cost operator sequence
print(result.acceptance_by_move)   # {"insert": (accepted, proposed), ...}
```

From the command line, the same run is

```console
$ mandacaru H2 --cell 5.5 --h 0.25 --method vasqa --pool qeb \
      --max-steps 100 --max-length 10 --temperature 0.1 0.001 \
      --length-penalty 1e-3 --seed 1234 --txt output.txt
```

`--temperature` takes one value (fixed) or two (annealed from the first to
the second), `--move-weight swap=0` changes one move's weight and leaves the
others at 1, and `--no-warm-start` relaxes every proposal from zero angles.
Energies on the command line are in eV.

The pool is any of those ADAPT-VQE uses: `"fermionic"`, `"qubit"`, `"qeb"`,
`"ceo"`, `"ceo-ovp"` or `"spin-orbit"`. The problem setup (basis, grid,
mapping, active space, tapering, sector, `sparse`) is shared with ADAPT-VQE
too. `"ceo-ovp"` differs from `"ceo"` only in how ADAPT-VQE grows it, so the
chain sees the same set of operators for both.

---

## One step of the chain

An **architecture** is an ordered tuple of pool operators
$C=(\mu_1,\dots,\mu_L)$, with the state
$|\psi(C,\boldsymbol\theta)\rangle=\prod_k e^{\theta_k A_{\mu_k}}|\mathrm{HF}\rangle$.
The first operator acts first. Each operator in the sequence has its own
angle, and the same operator may appear more than once. The chain starts from
the empty architecture, which is the reference itself.

Each step draws one of four moves:

| Move | What changes | Angles of the new sequence |
|---|---|---|
| `insert` | an operator enters at one of the $L+1$ slots | old angles; the new one is $0$ |
| `delete` | one occurrence leaves | old angles minus the deleted one |
| `replace` | one occurrence becomes a different operator | old angles; the replaced one is $0$ |
| `swap` | two occurrences of different operators exchange places | each angle moves with its operator |

A zero-angle insertion leaves the state exactly unchanged, since
$e^{0\cdot A}=I$. The move is picked with probability proportional to
`move_weights` among the moves that are possible at the current length: no
`delete` below `min_length` and no `insert` at `max_length`. Its position and
operator are then drawn uniformly.

The proposed sequence is relaxed with the configured `optimizer`, starting
from those angles, which gives $\widehat E(C')$. Its cost is

```{math}
F(C) = \widehat E(C) + \lambda_L L,
```

with $\lambda_L$ = `length_penalty`. The proposal is accepted with probability

```{math}
\alpha = \min\left\{1,\;
  e^{-\beta\,[F(C')-F(C)]}\,\frac{q(C\mid C')}{q(C'\mid C)}\right\},
\qquad \beta = 1/T .
```

Here $q(C'\mid C)$ is the probability of proposing $C'$ from $C$, summed over
every move that produces it. Inserting `A` into `(A)` gives `(A, A)` at
either slot, and both count. The ratio of reverse to forward proposals is what
makes the rule unbiased when the moves are not symmetric. An insertion is
reversed by a deletion, and the two have different probabilities.

---

## The temperature is not physical

`temperature` is an **architecture** temperature, in eV (Hartree with
`atomic_units=True`). It controls how often the chain accepts a sequence whose
relaxed cost is higher than the current one. It is not the temperature of the
molecule, and nothing thermal is being sampled.

- A number gives a fixed temperature. `0` is the greedy limit: decreases are
  accepted, increases rejected, and ties are decided by the proposal ratio.
- `{"initial": T0, "final": T1}` anneals geometrically from `T0` to `T1` over
  `max_steps`. The default is 0.1 eV → 1 meV, in eV even with
  `atomic_units=True`.

Without a length penalty the chain drifts toward `max_length`. With
independent angles, an insertion can never raise the optimal energy, so a
longer sequence is never worse. `length_penalty` is the energy one operator
must buy to be kept.

---

## Optimization, not sampling

With `warm_start=True` (the default), a proposal is relaxed from the angles
the chain arrived with. Its energy then depends on the path, and the chain is
a **stochastic optimizer** (simulated annealing over ansatz structures). It
does not sample a fixed distribution.

With `warm_start=False`, every architecture is relaxed from zero angles and
its energy is memoized, so the cost is a fixed function of the sequence. At a
fixed temperature the chain then leaves
$\pi(C)\propto e^{-\beta \widehat F(C)}$ invariant. This distribution is over
the optimizer's $\widehat E(C)$. A local optimizer does not certify the global
minimum over the angles.

---

## What the result holds

`result.optimal_energy`, `result.operators` and `result.optimal_parameters`
describe the **lowest-cost** architecture evaluated during the run, including
proposals that were rejected. This is the state the calculator uses for forces,
densities and cube files. The result also has:

- `best_energy`, `best_energy_operators`: the lowest energy seen. It differs
  from the lowest cost when `length_penalty > 0`.
- `final_energy`, `final_operators`: where the chain stopped.
- `steps`: one `MCASStep` per proposal, with the move, the proposed and
  current sequences, energies and costs, both log proposal probabilities, the
  log acceptance ratio and the decision. A rejected step repeats the current
  state.
- `acceptance_by_move`, `acceptance_rate`, `num_architectures` (distinct
  sequences evaluated) and `num_evaluations` (energy evaluations over all
  relaxations).

Each step is a full VQE optimization, so `max_steps` proposals cost roughly
`max_steps` ADAPT-VQE growth steps.

`txt="output.txt"` writes the same log as ADAPT-VQE, with a `[MARKOV CHAIN]`
table in place of `[ITERATIONS]`: one row per proposal, with the proposed
energy, its `dE` from the current state, the temperature, the log acceptance
ratio and the decision (see {doc}`../guide/run_output`). Without `txt=` the
same blocks are printed. Checkpoints are refused: a chain's state is more than
one ansatz.

The proposal is uniform. Learned proposal distributions are not implemented.
