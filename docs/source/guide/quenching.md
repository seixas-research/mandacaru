# Dynamic Parametrisation (`quenching`)

Every method accepts a `quenching` flag that controls **how many
parameters the classical optimiser varies at each step**.

```python
Mandacaru(method="adapt-vqe",
          basis="FAO",
          quenching=True)    # default: re-optimise everything
Mandacaru(method="adapt-vqe",
          basis="FAO",
          quenching=False)   # freeze the past, tune only the newest
```

| | `quenching=True` (default) | `quenching=False` |
|---|---|---|
| Adaptive methods | re-optimise **all** parameters each growth step | optimise **only the newest**; earlier angles frozen at their previous optimum |
| Fixed ansatz (`method="vqe"`) | one joint minimisation | sweep parameters one at a time, in index order |
| Cost per step | $k$-dimensional optimisation | 1-dimensional line search |
| Variational freedom | all current parameters may vary | earlier parameters remain fixed |

---

## Adaptive methods

`quenching=True` is textbook ADAPT-VQE: after appending $e^{\theta_k A_k}$ with
$\theta_k = 0$, the optimiser is handed the **whole** parameter vector,
warm-started from the previous optimum. Joint optimisation allows earlier parameters to respond to the new generator.
It does not by itself guarantee the full configuration-interaction energy.

`quenching=False` *quenches* each angle into place: parameters
$\theta_1 \dots \theta_{k-1}$ are held fixed and only $\theta_k$ is varied. Each
step is then a one-dimensional line search — far cheaper per operator, at the
cost of variational freedom.

```python
seen = []
calc = Mandacaru(method="adapt-vqe",
                 pool="qeb",
                 load_hamiltonian="lih.parquet",
                 quenching=False)
calc.run(callback=lambda info: seen.append(info["parameters"].copy()))

# Every step appends exactly one parameter and leaves the earlier ones untouched.
for earlier, later in zip(seen, seen[1:]):
    assert later.size == earlier.size + 1
    assert (later[:earlier.size] == earlier).all()
```

The two policies agree on the very first operator — with a single parameter there
is nothing to freeze.

---

## Fixed ansätze

`method="vqe"` has no growth loop, so `quenching=False` takes
the natural analogue: a **sequential sweep**. Parameter $k$ is optimised alone,
with $0 \dots k-1$ already at their optimised values and $k+1 \dots$ at their
starting values.

This is the same trade: cheaper individual optimisations, a weaker variational
result. The exact joint minimum cannot be higher than the minimum with parameters
fixed, but practical local optimisers need not find either exact minimum.

---

## Which to use

Keep the default `quenching=True` for production energies — it is the standard
algorithm and the one validated against FCI throughout the test suite.

`quenching=False` is useful when the per-step optimisation cost dominates: deep
ansätze with many parameters, expensive cost functions (shot-based hardware
evaluation, for instance), or when you want to study how much of ADAPT-VQE's
accuracy comes from re-optimisation rather than from operator selection.

---

## Where it applies

`quenching` is implemented once on
{class}`~mandacaru.algorithms.base.VariationalDriver` and is honoured by every
driver that inherits from it:

- `_optimize_grown` — the growth loop of `ADAPTVQE`, the deflation
  excited-state growth, and `SubspaceADAPTVQE` (and any subclass that only
  overrides operator selection);
- `_optimize_all` — `VQE.run`, its deflated excited states, and `SubspaceVQE`.
