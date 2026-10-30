# Choosing a classical optimizer

Every variational run ends its inner loop in a classical minimization over the
ansatz parameters: once for `method="vqe"`, once per growth step for
`method="adapt-vqe"`. The `optimizer=` option picks the method and its budget.
This page says which methods exist, what the default depends on, and what
changes when the energy is measured with `shots`. The full list of options and
their defaults is the [options reference](#options-reference).

## Spelling

`optimizer=` is a method name or a dictionary. A name takes the defaults of
the objective; a dictionary overrides only the keys it names.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for optimizer in ("SLSQP",
                  {"method": "L-BFGS", "maxiter": 500, "tol": 1e-10},
                  {"method": "COBYLA", "maxiter": 500, "tol": 1e-8}):
    atoms.calc = Mandacaru(method="vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.25,
                           optimizer=optimizer)
    energy = atoms.get_potential_energy()
    result = atoms.calc.result
    print(f"{energy:.6f} eV  {result.optimizer_steps} steps  "
          f"{result.num_evaluations} evaluations")
```

The dictionary takes exactly five keys: `method`, `maxiter`, `maxfev`, `tol` and
`track_best`. Any other key (`options`, `seed`, a SciPy keyword) is refused
when the calculator is constructed. On an exact objective the defaults are
SLSQP, `maxiter=2000` and `tol=1e-12`, so `{"maxiter": 500}` is SLSQP on a
shorter budget. Write the method, `maxiter` and `tol` out in any run that pins
an energy, so the record says what produced it.

All seven methods run in Rust; none is dispatched to SciPy.

| family | methods |
| :--- | :--- |
| derivative-free | `"COBYLA"`, `"Nelder-Mead"` |
| gradient | `"SLSQP"` (the default on an exact objective), `"BFGS"`, `"L-BFGS"`, `"CG"` |
| stochastic | `"SPSA"` (the default when `shots > 0`) |

The gradient methods receive the analytic energy gradient of the ansatz
parameters. `gradient="analytic"` is the only value the constructor accepts;
the finite-difference and parameter-shift spellings are refused. COBYLA,
Nelder-Mead and SPSA use no gradient.

(shots-optimizer)=
## With `shots`, the default follows the objective

With `shots > 0` the energy is a measured quantity and the default changes:

* **No `method` key:** strict SPSA with `maxiter=100`, `tol=None` and
  `track_best=False`. Strict SPSA takes two energy evaluations per step plus
  one at the end, so a default run makes `2 * 100 + 1 = 201` evaluations, and
  on a processor each one is a job. A dictionary without `"method"` inherits
  this too, so `optimizer={"maxiter": 50}` is SPSA on a shorter budget.
* **`"SLSQP"`, `"BFGS"`, `"L-BFGS"` and `"CG"` are refused,** however they are
  spelled. They difference the energy at tiny steps, which turns shot noise into
  gradient noise.
* **`"COBYLA"` keeps the exact-objective defaults** (`maxiter=2000`,
  `tol=1e-12`), which is far more jobs than a device allows. Set `maxiter`
  explicitly.

```python
from mandacaru import Mandacaru

try:
    Mandacaru(method="vqe", shots=1000, optimizer="SLSQP")
except ValueError as error:
    print(error)
```

The classical methods (`rhf`, `uhf`, `ghf`, `dft`) refuse `shots`, `device`,
`backend_provider`, `backend_options` and `measurement_budget`: they have no
quantum state to measure.

### What the run budget does to an optimizer

On a processor the run-wide job and shot budget is on by default (see
[the run budget](#run-budget)). For `method="vqe"` it is checked before the
first job is queued:

* SPSA and COBYLA have a bound on their energy evaluations, from `maxiter`.
  If that bound times the jobs one energy needs cannot fit the budget, the run
  is refused with `RunBudgetExceeded` before anything is submitted. COBYLA is
  bounded by `maxiter` unless `maxfev` is given, in which case `maxfev` is the
  bound.
* Nelder-Mead is not pre-flighted: it has no bound from its iteration count,
  so the budget ledger stops it where the budget ends.

A run that the budget stops part-way keeps its best point. The calculator
issues a `RuntimeWarning` and records the reason in `calc.run_budget_exhausted`
(`None` when the budget did not stop the run).

## Steps are not evaluations

Two numbers measure the classical cost, and they are not interchangeable:

* **steps**, the optimizer's own parameter updates;
* **evaluations**, the energy evaluations, which is what a processor bills.

A gradient method takes few steps and spends several evaluations on each; SPSA
takes many steps at two evaluations each. Report both:

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25, max_iterations=3)
atoms.get_potential_energy()
result = atoms.calc.result
result.optimizer_steps                    # parameter updates, whole run
result.num_evaluations                    # energy evaluations, whole run
result.iterations[0].optimizer_steps      # per ADAPT growth step
result.iterations[0].num_evaluations
```

An inner optimization that does not certify its own convergence is not
silently accepted. ADAPT-VQE records it in `result.optimizer_failures` and
issues one `RuntimeWarning` at the end of the run. Whether the run as a whole
converged is `result.success`, and the reason it stopped is
`result.timings["termination"]`.

## A rule of thumb

* **State-vector simulation:** the SLSQP default, or any other gradient method.
  Keep `tol` tight. ADAPT-VQE's own convergence test reads the largest pool
  gradient the inner optimizer leaves behind, so a loose `tol` can keep the
  growth loop from stopping.
* **A starting point that is already stationary:** a gradient method cannot
  leave one and correctly reports convergence there. ADAPT-VQE never meets this
  (it only selects an operator with a nonzero gradient), but a fixed ansatz can.
  Nelder-Mead and COBYLA escape it.
* **Shot-based execution:** SPSA (the default) or COBYLA, neither of which needs
  a derivative.
* **Cost-limited runs:** compare `num_evaluations`, not `optimizer_steps`. They
  rank the methods differently.

`examples/classical_optimizers/01_optimizer_comparison.py` runs all seven methods
on the same H2 problem and prints each one's energy and evaluation count. The
numbers depend on the basis, the grid and the tolerance, so measure on your own
problem before choosing.
