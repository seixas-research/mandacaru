# Choosing a classical optimizer

Every variational run ends its inner loop in a classical minimization over the
ansatz parameters — once for `method="vqe"`, once per growth step for
ADAPT-VQE. The `optimizer=` argument picks it, in any of three equivalent
spellings.

A **method name** takes the library's budget and tolerance:

```python
from mandacaru import Mandacaru

calc = Mandacaru(method="adapt-vqe",
                 basis="FAO",
                 h=0.30,
                 pool="qubit",
                 optimizer="L-BFGS-B")
```

A **dict** sets them, with nothing extra to import:

```python
calc = Mandacaru(method="adapt-vqe",
                 basis="FAO",
                 h=0.30,
                 pool="qubit",
                 optimizer={"method": "L-BFGS-B",
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
                 basis="FAO",
                 h=0.30,
                 pool="qubit",
                 optimizer=spsa)
```

The names are in {data}`~mandacaru.optimizers.NAMED_OPTIMIZERS`, in three
families:

| family | methods |
| :--- | :--- |
| derivative-free | `"COBYLA"`, `"Nelder-Mead"` |
| quasi-Newton / gradient | `"SLSQP"` (the default, {data}`~mandacaru.optimizers.DEFAULT_OPTIMIZER`), `"BFGS"`, `"L-BFGS"`, `"L-BFGS-B"`, `"NLCG-PR"` |
| stochastic | `"SPSA"`, `"Adam"` |

Everything but SPSA and Adam is dispatched to `scipy.optimize.minimize`; those
two are implemented natively, since SciPy has no equivalent.

Two of the names need a word about what they map onto. **`"L-BFGS"`** is
limited-memory BFGS without bounds, which is exactly what SciPy's `"L-BFGS-B"`
reduces to when no bounds are given — a variational ansatz's parameters are
unbounded angles, so the driver never passes any and the two names run the same
code. **`"NLCG-PR"`** is the nonlinear conjugate gradient in its Polak–Ribière
variant, which is what SciPy implements under the bare name `"CG"`.

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
one. It also gives the native SPSA and Adam something to certify convergence
against, which `tol=None` does not. Pass `tol=None` to get each method's own
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

LiH at 1.6 Å, FAO, `pool="qubit"`, Jordan-Wigner (6 qubits), ADAPT-VQE to
`max_iterations=12`, every optimizer on the shipped defaults `maxiter=1000`
and `tol=1e-12` (`examples/33_optimizer_comparison.py`; FCI = −162.953987 eV):

| optimizer | steps | evaluations | operators | E − E(FCI) (eV) | CNOTs | s | converged |
|---|---|---|---|---|---|---|---|
| SPSA | 11377 | 34143 | 12 | 5.4e-05 | 66 | 2.5 | no |
| COBYLA | 4247 | 5470 | 12 | 4.8e-06 | 66 | 2.8 | yes |
| Nelder-Mead | 3270 | 5935 | 10 | 7.2e-07 | 60 | 0.5 | yes |
| **SLSQP** | **76** | **632** | **10** | **7.2e-07** | **60** | **0.1** | **yes** |
| Adam | 9427 | 118291 | 10 | 7.2e-07 | 60 | 6.6 | yes |
| L-BFGS-B | 85 | 844 | 10 | 7.2e-07 | 60 | 0.1 | yes |
| BFGS | 80 | 1066 | 10 | 7.2e-07 | 60 | 0.2 | yes |
| L-BFGS | 85 | 844 | 10 | 7.2e-07 | 60 | 0.1 | yes |
| NLCG-PR | 137 | 2339 | 10 | 7.2e-07 | 60 | 0.3 | yes |

**SLSQP wins on every axis**: the exact ground state of the qubit Hamiltonian,
the shortest circuit anyone found, and it gets there in 76 parameter updates and
632 energy evaluations — an order of magnitude below the direct searches and two
below Adam. L-BFGS-B is the same answer at slightly more of everything.

**Read the rest this way.** The whole quasi-Newton family lands on the same
answer and the same 60-CNOT circuit within a factor of four of each other:
`L-BFGS` is `L-BFGS-B` to the evaluation (they are the same code), **BFGS**
costs a few hundred more evaluations for its dense Hessian approximation, and
**NLCG-PR** about three times SLSQP's — a conjugate gradient stores no curvature,
so it needs more directions. **Nelder-Mead** and **Adam** reach the same energy
and the same circuit, Nelder-Mead on 9× and Adam on 190× the evaluations —
Adam's gradient is a finite difference, so it spends `2N + 1` evaluations per
step. **COBYLA** stops two operators short of the best circuit. **SPSA** never
certifies convergence on a noiseless state-vector cost (it is built for a noisy
one) and burns its whole iteration budget at every growth step; it belongs on
hardware, where two evaluations per gradient is the entire point.

```{warning}
This table was different at the previous `tol=1e-8`: SLSQP and L-BFGS-B stopped
early, left ADAPT two extra operators and 78 CNOTs instead of 60, and never
reported convergence. **That was the tolerance, not the method.** For SLSQP and
COBYLA `tol` is a *function-value* criterion, so it must be tighter than a
gradient method's to leave an equally small gradient — and ADAPT's own
convergence test reads exactly that residual gradient. At `tol=1e-8` on
H₂/FAO, SLSQP reaches the ground state to 1e-10 eV but leaves `max|g| = 4.6e-06`
against a `gradient_tolerance` of `1e-6`, so the growth loop never stops and
piles up 50 redundant operators; at `1e-12` the same run converges after one
operator in 4 steps and 9 evaluations. If you override `tol`, keep it tight.
```

## Measured: a realistic system

The same comparison on water — H₂O, `basis={"name": "PAW", "size": "SZ"}`,
`h=0.25`, `pool="qubit"`, Jordan-Wigner (12 qubits, a 640-operator pool, sector
dimension 225), ADAPT-VQE to `max_iterations=40`; sector FCI = −492.521982 eV.
Here the tolerance makes no difference (every inner optimization is limited by
`maxiter`, not by `tol`), so these numbers hold at both `1e-8` and `1e-12`:

| optimizer | steps | evaluations | E − E(FCI) (eV) | CNOTs | depth | s | uncertified steps |
|---|---|---|---|---|---|---|---|
| SPSA | 77454 | 232402 | 1.38e-02 | 538 | 576 | 166.1 | 38 |
| COBYLA | 23273 | 34184 | 9.64e-03 | 486 | 527 | 54.5 | 10 |
| Nelder-Mead | 59751 | 77245 | 9.67e-03 | 488 | 526 | 66.1 | 24 |
| **SLSQP** | **236** | **6244** | **9.64e-03** | **486** | 527 | **8.3** | **0** |
| Adam | 15703 | 746747 | 9.64e-03 | 490 | 528 | 784.6 | 2 |
| **L-BFGS-B** | **166** | **5912** | 9.72e-03 | **486** | **524** | 10.8 | **0** |

Every method except SPSA reaches the same energy on the same circuit — the
CNOT counts span 486 to 490 — so again the choice is entirely about cost, and
**SLSQP spends a twentieth of the evaluations of the direct searches and a
hundredth of Adam's**, at 8× the wall-clock speed of Nelder-Mead.

The same molecule with `pool="qeb"`, which does reach
`gradient_tolerance=1e-4`, as operators / CNOTs / seconds at the same energy
(−492.508772 eV):

| optimizer | operators | evaluations | CNOTs | s | uncertified steps |
|---|---|---|---|---|---|
| SPSA | 60 (cap) | 351117 | 1288 | 299.5 | 57 |
| COBYLA | 30 | 13921 | 1168 | 19.0 | 0 |
| Nelder-Mead | 31 | 54540 | 1216 | 33.8 | 15 |
| **SLSQP** | **30** | **2447** | **1168** | 5.6 | **0** |
| Adam | 30 | 366943 | 1168 | 215.5 | 1 |
| **L-BFGS-B** | 31 | 3012 | 1172 | **5.1** | **0** |

Nelder-Mead pays an extra operator and 48 extra CNOTs here, while SLSQP
matches COBYLA's circuit exactly on a twentieth of the evaluations and is 6×
faster than Nelder-Mead in wall time.

**SLSQP is the shipped default** because both tables say the same thing: the
best energy anyone reached, on the shortest circuit anyone built, for a
fraction of the cost, with every inner optimization certified. What it needs in
return is the tight `tol` — see the warning above.

## A rule of thumb

* **State-vector simulation** — the `"SLSQP"` default, or any of the
  quasi-Newton family (`"L-BFGS"` / `"L-BFGS-B"`, `"BFGS"`, `"NLCG-PR"`): one to
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
  rank the methods differently. Adam is the extreme case: the best energy in the
  LiH table at ten times the evaluations of the next method.
* **Whatever you choose, write it out.** `optimizer=Optimizer(method=...,
  maxiter=..., tol=...)` is what the examples and the test suite do, so a run
  that pins an energy records what produced it.

An inner optimization that does not certify its own convergence is not silently
accepted: it is recorded in `result.optimizer_failures` and reported once as a
`RuntimeWarning` at the end of the run, naming the growth steps involved.
