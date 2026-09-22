# Choosing a classical optimizer

Every variational run ends its inner loop in a classical minimization over the
ansatz parameters — once for `method="vqe"`, once per growth step for
ADAPT-VQE. The `optimizer=` argument picks it, in any of three equivalent
spellings.

A **method name** takes the library's budget and tolerance:

```python
from mandacaru import Mandacaru

calc = Mandacaru(method="adapt-vqe",
                 basis="HAO",
                 h=0.30,
                 pool="qubit",
                 optimizer="L-BFGS")
```

A **dict** sets them, with nothing extra to import:

```python
calc = Mandacaru(method="adapt-vqe",
                 basis="HAO",
                 h=0.30,
                 pool="qubit",
                 optimizer={"method": "L-BFGS",
                            "maxiter": 2000,
                            "tol": 1e-12})
```

Keys left out keep their defaults, so `{"maxiter": 500}` is the default method
on a shorter budget. `options` and `seed` are accepted too
({data}`~mandacaru.optimizers.optim.OPTIMIZER_KEYS`), and an unknown key is
refused as the typo it is.

An {class}`~mandacaru.optimizers.Optimizer` is what the dict builds, and is
worth importing when the same configuration is reused across runs:

```python
from mandacaru.optimizers import Optimizer

spsa = Optimizer(method="SPSA", maxiter=500, tol=1e-6, options={"a": 0.1})
calc = Mandacaru(method="adapt-vqe",
                 basis="HAO",
                 h=0.30,
                 pool="qubit",
                 optimizer=spsa)
```

The names are in {data}`~mandacaru.optimizers.NAMED_OPTIMIZERS`, in three
families:

| family | methods |
| :--- | :--- |
| derivative-free | `"COBYLA"`, `"Nelder-Mead"` |
| quasi-Newton / gradient | `"SLSQP"` (the default, {data}`~mandacaru.optimizers.DEFAULT_OPTIMIZER`), `"BFGS"`, `"L-BFGS"`, `"NLCG-PR"` |
| stochastic | `"SPSA"` |

Everything but SPSA is dispatched to `scipy.optimize.minimize`; SPSA is
implemented natively, since SciPy has no equivalent.

Two of the names need a word about what they map onto. **`"L-BFGS"`** is
limited-memory BFGS without bounds, which is exactly what SciPy's `"L-BFGS-B"`
reduces to when no bounds are given — a variational ansatz's parameters are
unbounded angles, so the driver never passes any. SciPy's bounded spelling is
not offered under its own name: it was, and it ran the same code for the same
cost, which is a second name for one thing. **`"NLCG-PR"`** is the nonlinear
conjugate gradient in its Polak–Ribière variant, which is what SciPy implements
under the bare name `"CG"`.

```{note}
For `"BFGS"` and `"NLCG-PR"`, SciPy reads `tol` as a **gradient norm**, not as a
function-value change. Handing those two the default `1e-12` asks for something
a finite-difference gradient cannot deliver — its own accuracy is about `1e-8`
— so every line search ends in "precision loss" and the run reports
non-convergence at every growth step while sitting exactly on the minimum.
Mandacaru therefore passes them `sqrt(tol)`, the gradient criterion of equal
strength (near a minimum, `f − f* ~ |g|²/2λ`). An explicit
`options={"gtol": ...}` is left alone.
```

A bare name is shorthand for the defaults
({data}`~mandacaru.optimizers.DEFAULT_MAXITER` = 1000 and
{data}`~mandacaru.optimizers.DEFAULT_TOL` = `1e-12`); a run that pins an energy
should say what it was optimized with, so the examples write both out through
the dict and the test suite through module-level `Optimizer` constants.

```{note}
The default tolerance is **explicit and tight on purpose**. Several methods'
SciPy defaults are far too loose for a cost measured in Hartree — Nelder-Mead's
`xatol = fatol = 1e-4` stops the simplex four orders of magnitude above
chemical accuracy — and every recommendation below was measured with the tight
one. It also gives the native SPSA something to certify convergence against,
which `tol=None` does not. Pass `tol=None` to get each method's own
default back.
```

## Steps are not evaluations

Two numbers measure the classical cost, and they are not interchangeable:

* **steps** — parameter updates, the optimizer's own moves;
* **cost evaluations** — energy evaluations, which is what a QPU is billed for.

A gradient-based method takes few steps and spends many evaluations on each one
(a finite-difference gradient costs `2N`, plus a line search); a direct search
takes many cheap steps. Reporting only one of them flatters one family and
slanders the other, so a run reports both: the `steps` column of the
`[ITERATIONS]` table is the per-growth-step count, and the summary block's
`optimizer_steps` / `cost_evaluations` are the run's totals.

```python
result = atoms.calc.result
result.optimizer_steps                    # parameter updates, whole run
result.num_evaluations                    # energy evaluations, whole run
result.iterations[0].optimizer_steps      # ... and per growth step
```

The same count is on every {class}`~mandacaru.optimizers.OptimizeResult` as
`nit`, for a bare optimizer used outside a driver.

## Measured: a small system

LiH at 1.6 Å, HAO, `pool="qubit"`, Jordan-Wigner (6 qubits), ADAPT-VQE to
`max_iterations=12`, every optimizer on the shipped defaults `maxiter=1000`
and `tol=1e-12` (`examples/33_optimizer_comparison.py`; FCI = −162.953987 eV):

| optimizer | steps | evaluations | operators | E − E(FCI) (eV) | CNOTs | s | converged |
|---|---|---|---|---|---|---|---|
| SPSA | 11377 | 34143 | 12 | 5.4e-05 | 66 | 2.4 | no |
| COBYLA | 4247 | 5470 | 12 | 4.8e-06 | 66 | 2.6 | yes |
| Nelder-Mead | 3270 | 5935 | 10 | 7.2e-07 | 60 | 0.5 | yes |
| **SLSQP** | **76** | **632** | **10** | **7.2e-07** | **60** | **0.1** | **yes** |
| BFGS | 80 | 1066 | 10 | 7.2e-07 | 60 | 0.2 | yes |
| L-BFGS | 85 | 844 | 10 | 7.2e-07 | 60 | 0.1 | yes |
| NLCG-PR | 137 | 2339 | 10 | 7.2e-07 | 60 | 0.3 | yes |

**SLSQP wins on every axis**: the exact ground state of the qubit Hamiltonian,
the shortest circuit anyone found, and it gets there in 76 parameter updates and
632 energy evaluations — an order of magnitude below the direct searches.

**Read the rest this way.** The whole quasi-Newton family lands on the same
answer and the same 60-CNOT circuit within a factor of four of each other:
**L-BFGS** costs a third more evaluations than SLSQP, **BFGS** a few hundred
more for its dense Hessian approximation, and **NLCG-PR** about three times
SLSQP's — a conjugate gradient stores no curvature, so it needs more
directions. **Nelder-Mead** reaches the same energy and the same circuit on 9×
the evaluations. **COBYLA** stops two operators short of the best circuit.
**SPSA** never
certifies convergence on a noiseless state-vector cost (it is built for a noisy
one) and burns its whole iteration budget at every growth step; it belongs on
hardware, where two evaluations per gradient is the entire point.

```{warning}
This table was different at the previous `tol=1e-8`: SLSQP and L-BFGS stopped
early, left ADAPT two extra operators and 78 CNOTs instead of 60, and never
reported convergence. **That was the tolerance, not the method.** For SLSQP and
COBYLA `tol` is a *function-value* criterion, so it must be tighter than a
gradient method's to leave an equally small gradient — and ADAPT's own
convergence test reads exactly that residual gradient. At `tol=1e-8` on
H₂/HAO, SLSQP reaches the ground state to 1e-10 eV but leaves `max|g| = 4.6e-06`
against a `gradient_tolerance` of `1e-6`, so the growth loop never stops and
piles up 50 redundant operators; at `1e-12` the same run converges after one
operator in 4 steps and 9 evaluations. If you override `tol`, keep it tight.
```

## Measured: a realistic system

The same comparison on water — H₂O, `basis={"name": "PAW", "size": "SZ"}`,
`h=0.25`, `pool="qubit"`, Jordan-Wigner (12 qubits, a 640-operator pool, sector
dimension 225), ADAPT-VQE to `max_iterations=40`; sector FCI = −492.521982 eV.
No run reaches `gradient_tolerance` here, so all seven stop on ADAPT's
`max_iterations=40` and the comparison is at equal circuit length:

| optimizer | steps | evaluations | E − E(FCI) (eV) | CNOTs | s | uncertified steps |
|---|---|---|---|---|---|---|
| SPSA | 40000 | 120040 | 1.79e-02 | 538 | 89.3 | 40 |
| COBYLA | 17067 | 26521 | 9.68e-03 | 488 | 46.9 | 14 |
| Nelder-Mead | 33297 | 44701 | 9.73e-03 | 508 | 44.9 | 28 |
| **SLSQP** | **471** | **13424** | **9.64e-03** | 488 | **16.7** | **0** |
| BFGS | 819 | 45672 | 9.64e-03 | **486** | 59.4 | 19 |
| **L-BFGS** | **444** | 15342 | **9.64e-03** | 488 | 22.6 | **0** |
| NLCG-PR | 963 | 75309 | 9.64e-03 | 490 | 94.0 | 13 |

Every gradient method reaches the same energy on effectively the same circuit —
486 to 490 CNOTs — so again the choice is cost, and **SLSQP is cheapest on both
currencies**: half COBYLA's evaluations, a third of Nelder-Mead's, and the
fastest wall-clock of the seven.

```{note}
**BFGS and NLCG-PR stop certifying here** (19 and 13 uncertified growth steps),
where they certified every step on LiH. With 40 parameters the
finite-difference gradient's own accuracy no longer reaches the `sqrt(tol)`
gradient criterion, and the line search ends in "precision loss". The energy is
right; the convergence flag is not. SLSQP and L-BFGS, whose `tol` is a
function-value criterion, are unaffected — which is the practical reason to
prefer them.
```

The same molecule with `pool="qeb"`, which *does* converge on
`gradient_tolerance`, as operators / evaluations / CNOTs at the same energy
(−492.508772 eV):

| optimizer | operators | evaluations | CNOTs | s | uncertified steps |
|---|---|---|---|---|---|
| SPSA | 40 (cap) | 120040 | 1208 | 85.9 | 40 |
| COBYLA | 30 | 16307 | 1168 | 21.5 | 3 |
| Nelder-Mead | 31 | 34309 | 1216 | 21.3 | 19 |
| **SLSQP** | **30** | **4308** | **1168** | 5.6 | **0** |
| BFGS | 30 | 12478 | 1168 | 9.7 | 6 |
| **L-BFGS** | **30** | 4871 | **1168** | **5.5** | **0** |
| NLCG-PR | 30 | 13131 | 1168 | 10.3 | 4 |

Every method but SPSA and Nelder-Mead builds the identical 30-operator,
1168-CNOT circuit; SLSQP gets there on a quarter of COBYLA's evaluations and an
eighth of Nelder-Mead's, four times faster in wall time.

**SLSQP is the shipped default** because all three tables say the same thing:
the best energy anyone reached, on the shortest circuit anyone built, for a
fraction of the cost, with every inner optimization certified. What it needs in
return is the tight `tol` — see the warning above. `L-BFGS` is the alternative,
within 15 % of it everywhere.

(line-search)=
## The line search, and why there is no `linesearch=` option

A line search is the inner 1-D minimization a gradient method runs along each
search direction. Mandacaru does **not** expose an option for it, and this
section is the measurement behind that decision.

**Which methods have one at all:**

| method | line search | knobs |
| :--- | :--- | :--- |
| `"SLSQP"` | internal, on the merit function | none exposed by SciPy |
| `"BFGS"` | Wolfe | `c1`, `c2`, `xrtol` |
| `"NLCG-PR"` | Wolfe | `c1`, `c2` |
| `"L-BFGS"` | bounded Moré–Thuente | `maxls` (default 20) |
| `"COBYLA"` | none — trust region on a linear model | — |
| `"Nelder-Mead"` | none — simplex reflection | — |
| `"SPSA"` | none — a decaying gain sequence | — |

The knobs that exist already reach SciPy, through the `options` key of the
dict form:

```python
optimizer={"method": "BFGS", "maxiter": 1000, "tol": 1e-12,
           "options": {"c1": 1e-4, "c2": 0.9}}
```

**Tuning them does not pay.** On the LiH problem of the table above, in
evaluations to the same energy: `L-BFGS` with `maxls` at 5, 20 and 50 is
**identical** (580 each — the line search never exhausts its budget on a cost
this smooth); `BFGS` tightened from the default `(c1, c2) = (1e-4, 0.9)` to
`(1e-2, 0.1)` costs **21 % more** (725 → 877); only `NLCG-PR` improves, 1283 →
1083 (−16 %), and it is still twice SLSQP's cost afterwards. There is no
setting here that beats picking a better method.

**A Mandacaru-specific exact line search is possible, and still does not pay.**
Every pool generator satisfies `A³ = −A`, so `exp(θA)` has eigenvalues
`1, e^{iθ}, e^{-iθ}` and the energy along a *single* parameter is exactly

$$E(\theta) = a_0 + a_1\cos\theta + b_1\sin\theta
             + a_2\cos 2\theta + b_2\sin 2\theta,$$

five coefficients, so **five evaluations determine the whole curve** and its
minimum follows in closed form. Verified on H₂ and LiH across the `qeb`,
`qubit` and `fermionic` pools: a 5-point fit predicts 40 other angles to
**3.6e-15 eV**.

```{note}
The first harmonic does **not** vanish in general — measured `|a₁, b₁|` is
2.1e-02 for LiH/`qeb` and 9.7e-03 for LiH/`fermionic`, against zero for
LiH/`qubit`. The familiar three-point Rotosolve formula assumes only the `2θ`
harmonic and would be wrong for those pools. Five points, not three.
```

Wiring that into the growth step was measured both ways:

* with `quenching=True` (the default), placing the new parameter at its exact
  1-D optimum before the joint re-optimization is a **wash** — SLSQP pays 1–7 %
  *more* evaluations, COBYLA saves 3–6 %, and the energy is unchanged to
  1e-12 eV. The warm start from θ = 0 is already good, and the joint problem is
  no easier for having one coordinate pre-placed.
* with `quenching=False`, where a 1-D search *is* the optimization, it is
  **worse**: same evaluation count, but −162.9526 eV against −162.9538
  (`qubit`) and −162.9503 against −162.9540 (`qeb`). The exact solve returns the
  *global* minimum over `[0, 2π)`, which is a greedier step than the growth loop
  wants; the iterative search stays on the branch near zero and the following
  operators are selected from a better state.

So: the knobs that exist are reachable and not worth turning, and the exact
line search this problem admits is neutral at best. `tol` and the choice of
method are the levers that matter.

## A rule of thumb

* **State-vector simulation** — the `"SLSQP"` default, or any of the
  quasi-Newton family (`"L-BFGS"`, `"BFGS"`, `"NLCG-PR"`): one to
  two orders of magnitude fewer steps and evaluations than anything else, and
  the only ones that certified convergence at every growth step of the water
  run. Keep `tol` tight; that is what turns them from the worst circuit in the
  LiH table into the best.
* **A starting point that is already stationary** — a gradient method cannot
  leave one, and correctly reports convergence there. ADAPT never hits this
  (its screening only ever selects an operator with a non-zero gradient), but a
  hand-built fixed ansatz can. `"Nelder-Mead"` or `"COBYLA"` escape it.
* **Shot-based execution** (`shots > 0`) — `"SPSA"` or `"COBYLA"`: neither needs
  a derivative, and a finite-difference gradient through shot noise is the worst
  of both worlds. SPSA's two evaluations per gradient is the whole point on
  hardware, and the noiseless tables above cannot show it.
* **Cost-limited runs** — read `cost_evaluations`, not `optimizer_steps`; they
  rank the methods differently. SPSA is the extreme case: the fewest evaluations
  per step of any method, and by far the most steps.
* **Do not reach for the line search.** Its knobs are tunable through
  `options=` and measure as noise; see [above](#line-search).
* **Whatever you choose, write it out.** `optimizer=Optimizer(method=...,
  maxiter=..., tol=...)` is what the examples and the test suite do, so a run
  that pins an energy records what produced it.

An inner optimization that does not certify its own convergence is not silently
accepted: it is recorded in `result.optimizer_failures` and reported once as a
`RuntimeWarning` at the end of the run, naming the growth steps involved.
