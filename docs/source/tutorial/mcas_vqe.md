# MCAS-VQE: Searching the Ansatz with a Markov Chain

ADAPT-VQE builds its ansatz greedily: each step appends the pool operator with
the largest energy gradient, and an operator once chosen stays.
`method="mcas-vqe"` (Markov Chain Ansatz Search with the Variational Quantum
Eigensolver) treats the operator
*sequence* itself as the thing to search. A Markov chain proposes changes to
the sequence, VQE relaxes the angles of each proposal, and a
Metropolis-Hastings test decides whether the chain moves there. An early
choice can therefore be undone.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
atoms.center(vacuum=2.5)
atoms.calc = Mandacaru(method="mcas-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.20,
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
$ mandacaru H2 --cell 5.5 --h 0.25 --method mcas-vqe --pool qeb \
      --max-steps 100 --max-length 10 --temperature 0.1 0.001 \
      --length-penalty 1e-3 --seed 1234 --txt output.txt
```

`--temperature` takes one value (fixed) or two (annealed from the first to
the second), `--proposal uniform` and `--proposal-temperature` choose how new
operators are drawn, `--move-weight swap=0` changes one move's weight and
leaves the others at 1, and `--no-warm-start` relaxes every proposal from zero
angles. `--replace-start inherit` lets a `replace` start the new operator from
the angle of the one it replaces (default `zero`), and `--transfer` and
`--transfer-steps N` are described under "Along a trajectory" below.
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
`delete` below `min_length` and no `insert` at `max_length`. Its position is
drawn uniformly.

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

with the same $\tau$ in the numerator and in every term of the sum, so
$\sum_\mu P(\mu) = 1$ for any $\tau > 0$.

the gradient ADAPT-VQE screens with. $\tau$ is `proposal_temperature`
(default 0.2), relative to the largest gradient: the steepest operator is
$e^{1/\tau}$ times as likely as one with zero gradient ($e^5 \approx 150$ at
the default). A small $\tau$ approaches ADAPT's greedy choice, a large one the
uniform draw, and every operator keeps a positive probability. A `replace`
draws from the same distribution with the replaced operator taken out.

Why it matters: most pool operators have zero gradient at a given state (at
the Hartree-Fock reference every single excitation does, by Brillouin's
theorem). Inserted at zero angle, such an operator starts at a stationary
point, the optimizer leaves it there, and it occupies the ansatz without
lowering the energy. The uniform draw (`proposal="uniform"`, each operator
$1/M$) proposes those operators most of the time.

The gradient is the one for appending at the end of the circuit; it is used
for every slot, as a guide to *which* operator rather than a derivative at
each position. The acceptance rule stays exact because the reverse
probability is computed with the distribution at the proposed state. The cost
is one pool screening per proposal, `result.num_screenings`.

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

## Along a trajectory

In an ASE relaxation, scan or dynamics run, every geometry is its own chain,
and by default each one starts from the empty ansatz. With `transfer=True`, a
geometry's chain starts from the ansatz the previous geometry reported
instead: its operators and, with `warm_start=True`, its angles (with
`warm_start=False` only the operators are carried and the angles start from
zero). Because the new start is usually close to the answer, `transfer_steps`
sets a shorter chain for it, and that is where the saving comes from.

Run relaxations and dynamics in the PAW-LCAO basis, as below: its forces are
the ones to trust. All-electron bases such as HAO leave the compact core
functions unresolved at practical grid spacings, so their energy surfaces are
not smooth enough to move atoms on.

```python
from ase import Atoms
from ase.optimize import BFGS
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]])
atoms.center(vacuum=2.5)
atoms.calc = Mandacaru(method="mcas-vqe",
                       basis="HAO",
                       h=0.25,
                       pool="qeb",
                       max_steps=30,          # the first geometry, from empty
                       transfer=True,
                       transfer_steps=6,      # every later geometry
                       length_penalty=1e-3,
                       seed=1234)
BFGS(atoms).run(fmax=0.05)

result = atoms.calc.result                    # the last geometry's chain
print(result.start)                           # where this chain began
print(result.start_operators)                 # the architecture it began with
print(result.edit_distance_from_start)        # moves to the reported ansatz
```

The same is `--transfer --transfer-steps 6` on the command line
(`--transfer-threshold S` sets the overlap threshold).

Each geometry's molecular orbitals come from their own mean-field
calculation, which may flip an orbital's sign or, where two orbital energies
cross, exchange two orbitals. Before the transfer, the chain therefore
follows the orbitals from the previous geometry to this one
(`mandacaru.algorithms.orbital_tracking`). It overlaps the two sets on the
grid, matches each orbital to its largest overlap inside its occupation block
(doubly occupied, singly occupied, empty), renames the operators of
exchanged orbitals, and flips the angle of every excitation that is odd in a
flipped orbital. The `start` line reports how many orbitals were reordered
and aligned, and the smallest matched overlap.

The transferred ansatz is relaxed at the new geometry first, and the chain
starts from it only if its cost lies below the reference's. Otherwise the
chain falls back to the empty ansatz, with the full `max_steps`, and says why
in the `start` line of `[OPTIMIZATION SETUP]` (and in `result.start`). The
possible reasons are:

- there was no previous geometry;
- the previous geometry was another molecule or pool;
- an orbital's best match is below `transfer_threshold` (default 0.9),
  which is the sign of an electronic reorganization rather than a
  relabeling;
- an operator cannot be followed through the matching;
- the ansatz is longer than `max_length`;
- it relaxed to no better than the reference. `result.edit_distance_from_start` is the
number of insert, delete and replace moves between the architecture the chain
started from and the one it reports (a swap counts as two, so it is an upper
bound on the distance when swaps are allowed); the function
`mandacaru.algorithms.mcas.edit_distance(a, b)` compares any two.

Carrying the ansatz along exploits what earlier geometries found; it can also
keep the search in the basin it found first. `rebuild_every=N`
(`--rebuild-every N`) balances that with exploration: at every N-th geometry,
after the transferred chain's `transfer_steps`, a fresh chain from the empty
ansatz searches for the full `max_steps`, and the lowest-cost state of both is
reported. The transferred chain runs first and unchanged, so a rebuild can
only keep or improve the reported state. It costs one full search every N
geometries. `result.rebuild` and the run log's summary say whether the
transferred or the rebuilt ansatz won. On the PAW-LCAO H2O scan (2026-10-01,
`rebuild_every=2`), the transferred ansatz was kept at all four rebuild
geometries, at 40 % more evaluations: there the carried basin was the better
one.

Transfer changes where the chain starts, not what it samples: the moves, the
proposals and the acceptance rule are the same, so only how soon the chain
reaches the low-cost architectures changes. The renaming is exact for
fermionic excitations. For qubit excitations after a reordering, it gives a
close start rather than the same state. Pauli-string and coupled-exchange
operators are carried only when no orbital moved. A rotation inside a nearly
degenerate shell is not aligned: it shows as a low matched overlap, and the
chain rebuilds. Keep the steps between geometries small. The calculator hands
the ansatz on only between chains of the same method.

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

A chain whose operator probabilities come from a trained model is VALQA,
`method="valqa"` ({doc}`valqa`); `record="DIR"` makes a MCAS-VQE run collect
the edits that model is trained on.
