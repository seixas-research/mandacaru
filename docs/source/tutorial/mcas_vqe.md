# MCAS-VQE: Searching the Ansatz with a Markov Chain

ADAPT-VQE builds its ansatz greedily: each step appends the pool operator with
the largest energy gradient, and an operator once chosen stays.
`method="mcas-vqe"` (Markov Chain Ansatz Search with the Variational Quantum
Eigensolver) treats the operator *sequence* itself as the thing to search. A
Markov chain proposes changes to the sequence, VQE relaxes the angles of each
proposal, and a Metropolis-Hastings test decides whether the chain moves there.
An early choice can therefore be undone.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
atoms.center(vacuum=2.5)
atoms.calc = Mandacaru(method="mcas-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.20,
                       pool="qeb",
                       max_steps=40,
                       max_length=10,
                       temperature={"initial": 0.1, "final": 1e-3},
                       length_penalty=1e-3,
                       seed=1234)
energy = atoms.get_potential_energy()

result = atoms.calc.result
print(result.operators)            # the lowest-cost operator sequence
print(result.acceptance_by_move)   # {"insert": (accepted, proposed), ...}
```

The pool is any of those ADAPT-VQE uses: `"fermionic"`, `"qubit"`, `"qeb"`,
`"ceo"`, `"ceo-ovp"` or `"spin-orbit"`. The problem setup (basis, grid, mapping,
active space, sector, `sparse`) is shared with ADAPT-VQE. The search uses local
state vectors and starts from the empty sequence: `min_length` must be `0`, and
circuit execution (`shots`, a non-default `device`) and `resume=` raise
`NotImplementedError`.

## One step of the chain

An **architecture** is an ordered tuple of pool operators $C=(\mu_1,\dots,\mu_L)$,
with the state
$|\psi(C,\boldsymbol\theta)\rangle=\prod_k e^{\theta_k A_{\mu_k}}|\mathrm{HF}\rangle$.
The first operator acts first. Each operator in the sequence has its own angle,
and the same operator may appear more than once. The chain starts from the empty
architecture, which is the reference itself.

Each step draws one of four moves:

| Move | What changes | Angles of the new sequence |
|---|---|---|
| `insert` | an operator enters at one of the $L+1$ slots | old angles; the new one is $0$ |
| `delete` | one occurrence leaves | old angles minus the deleted one |
| `replace` | one occurrence becomes a different operator | old angles; the replaced one is $0$ (or the replaced angle, with `replace_start="inherit"`) |
| `swap` | two occurrences of different operators exchange places | each angle moves with its operator |

A zero-angle insertion leaves the state exactly unchanged, since $e^{0\cdot A}=I$.
`move_weights` is a dictionary over these four names; **a move you do not name
gets weight 0**, and the default (`None`) weights all four equally. The move is
picked among those possible at the current length: no `delete` at length zero and
no `insert` at `max_length`. Its position is drawn uniformly.

### Which operator: a softmax of the gradient

The new operator of an `insert` or a `replace` is drawn from a softmax of the
pool gradients at the current state (`proposal="gradient"`, the default),

```{math}
P(\mu) = \frac{\exp\!\left(\dfrac{|g_\mu|}{\tau\, g_{\max}}\right)}
               {\displaystyle\sum_{\nu=1}^{M}
                \exp\!\left(\dfrac{|g_\nu|}{\tau\, g_{\max}}\right)},
\qquad g_\mu = 2\,\mathrm{Re}\langle H\psi|A_\mu\psi\rangle,
\quad g_{\max} = \max_\nu |g_\nu| ,
```

where $\tau$ is `proposal_temperature` (default 0.2), relative to the largest
gradient. A small $\tau$ approaches ADAPT's greedy choice, a large one the
uniform draw, and every operator keeps a positive probability. `proposal="uniform"`
draws every operator with probability $1/M$. Most pool operators have zero
gradient at a given state (at the Hartree-Fock reference every single excitation
does, by Brillouin's theorem), and an operator inserted at zero angle there starts
at a stationary point and occupies the ansatz without lowering the energy; the
uniform draw proposes those most of the time.

The proposed sequence is relaxed with the configured `optimizer`, starting from
those angles, which gives $\widehat E(C')$. Its cost is $F(C) = \widehat E(C) +
\lambda_L L$, with $\lambda_L$ = `length_penalty`. The proposal is accepted with
probability

```{math}
\alpha = \min\left\{1,\;
  e^{-\beta\,[F(C')-F(C)]}\,\frac{q(C\mid C')}{q(C'\mid C)}\right\},
\qquad \beta = 1/T ,
```

where $q(C'\mid C)$ is the probability of proposing $C'$ from $C$, summed over
every move that produces it. Inserting `A` into `(A)` gives `(A, A)` at either
slot, and both count. The ratio of reverse to forward proposals keeps the rule
unbiased when the moves are not symmetric.

## The temperature is not physical

`temperature` is an **architecture** temperature, in eV (Hartree with
`atomic_units=True`). It controls how often the chain accepts a sequence whose
relaxed cost is higher than the current one. It is not the temperature of the
molecule, and nothing thermal is being sampled.

* A number gives a fixed temperature. `0` is the greedy limit.
* `{"initial": T0, "final": T1}` anneals geometrically from `T0` to `T1` over
  `max_steps`. Both keys are required. The default is 0.1 eV to 1 meV.

Without a length penalty the chain drifts toward `max_length`: with independent
angles an insertion can never raise the optimal energy. `length_penalty` is the
energy one operator must buy to be kept (in eV, Hartree with `atomic_units=True`).

## Optimization, not sampling

With `warm_start=True` (the default), a proposal is relaxed from the angles the
chain arrived with. Its energy then depends on the path, and the chain is a
**stochastic optimizer** (simulated annealing over ansatz structures). It does
not sample a fixed distribution. With `warm_start=False`, every architecture is
relaxed from zero angles and its energy is memoized, so the cost is a fixed
function of the sequence and, at a fixed temperature, the chain leaves
$\pi(C)\propto e^{-\beta \widehat F(C)}$ invariant. A local optimizer does not
certify the global minimum over the angles.

## Along a trajectory

In an ASE relaxation, scan or dynamics run, every geometry is its own chain, and
by default each starts from the empty ansatz. With `transfer=True` a geometry's
chain starts from the ansatz the previous geometry reported, matched through the
molecular orbitals; `transfer_steps` sets a shorter chain for it, which is where
the saving comes from, and `transfer_threshold` (default 0.9) is the smallest
orbital overlap the matching accepts. The transferred ansatz is relaxed at the new
geometry and kept only if it improves on the reference; otherwise the chain starts
empty. `result.transfer` describes what happened.

Run relaxations and dynamics in the PAW-LCAO basis: its forces are the ones to
trust. All-electron bases such as HAO leave the compact core functions unresolved
at practical grid spacings.

```python
from ase import Atoms
from ase.optimize import BFGS
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]])
atoms.center(vacuum=2.5)
atoms.calc = Mandacaru(method="mcas-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25,
                       pool="qeb",
                       max_steps=30,          # the first geometry, from empty
                       transfer=True,
                       transfer_steps=6,      # every later geometry
                       length_penalty=1e-3,
                       seed=1234)
BFGS(atoms).run(fmax=0.05)
print(atoms.calc.result.transfer)
```

Transfer changes where the chain starts, not what it samples. Keep the steps
between geometries small, since a rotation inside a nearly degenerate shell is
not aligned and shows as a low overlap.

## What the result holds

`result.optimal_energy`, `result.operators` and `result.optimal_parameters`
describe the **lowest-cost** architecture evaluated during the run, rejected
proposals included. This is the state the calculator uses for forces, densities
and cube files. The result also has:

* `best_energy`, `best_energy_operators`: the lowest energy seen. It differs from
  the lowest cost when `length_penalty > 0`; `optimal_cost` is the cost.
* `final_energy`, `final_operators`: where the chain stopped.
* `steps`: one record per proposal, with the proposed and current sequences, the
  proposed and source energies and costs, the temperature and the decision
  (`accepted`). A rejected step repeats the current state.
* `acceptance_by_move`, `num_architectures` (distinct sequences evaluated),
  `num_screenings` (pool screenings, one per proposal) and `num_evaluations`.

A completed chain does not certify eigenstate convergence: `result.completed` and
`result.converged` report those distinct facts. Each step is a full VQE
optimization, so `max_steps` proposals cost roughly `max_steps` ADAPT-VQE growth
steps.

A chain whose operator probabilities come from a trained model is VALQA,
`method="valqa"` ([the VALQA tutorial](valqa.md)); `record="DIR"` makes an
MCAS-VQE run collect the edits that model is trained on. `seed` is a nonnegative
64-bit integer; without it the chain draws a fresh seed.
