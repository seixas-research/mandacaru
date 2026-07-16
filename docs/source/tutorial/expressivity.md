# Expressibility of parameterized circuits

The **expressibility** of a parameterized quantum circuit measures how uniformly
its output states cover the accessible Hilbert space as the parameters are
sampled at random (Sim, Johnson & Aspuru-Guzik, 2019). It compares the
distribution of fidelities between random parameter pairs,

```{math}
F = \bigl|\langle\psi(\vec\theta_a)|\psi(\vec\theta_b)\rangle\bigr|^2,
```

to the fidelity distribution of Haar-random states. For a $d$-dimensional space
the Haar density is $P_{\text{Haar}}(F) = (d-1)(1-F)^{d-2}$, and the
expressibility score is the Kullback-Leibler divergence

```{math}
E = D_{\mathrm{KL}}\!\bigl(P_{\text{PQC}} \,\|\, P_{\text{Haar}}\bigr).
```

A **smaller** $E$ means a **more** expressive circuit ($E = 0$ is the maximally
expressive limit).

## The effective dimension is not $2^N$

This is the key physical subtlety. Carcará's fermionic ansätze (UCCSD and the
fermionic / QEB / CEO ADAPT ansätze) **conserve particle number and spin**, so
their states never leave the symmetry sector of the Hartree-Fock reference. That
sector has dimension

```{math}
d = \binom{M}{n_\alpha}\binom{M}{n_\beta}, \qquad M = N/2,
```

not $2^N$ — for H₂, $d = \binom{2}{1}\binom{2}{1} = 4$, not $16$. Comparing a
number-conserving ansatz against a full $2^N$ Haar distribution would label
*every* such circuit inexpressive. Carcará provides both the analytic sector
dimension and an empirical estimate (the rank of the span of sampled states):

```python
from carcara.algorithms import active_space_dimension, estimate_effective_dimension
from carcara.circuits import UCCSD

active_space_dimension(4, (1, 1))          # -> 4
estimate_effective_dimension(UCCSD(2, (1, 1)))   # -> 4 (matches the sector)
```

## Measuring expressibility

`compute_expressibility` samples the fidelities and returns the score. Pass the
particle number so the correct sector dimension is used automatically:

```python
from carcara.algorithms import compute_expressibility

ansatz = UCCSD(n_spatial_orbitals=2, num_particles=(1, 1))
result = compute_expressibility(ansatz, num_samples=2000, num_particles=(1, 1))
print(result)     # ExpressibilityResult(E=..., d=4, n_samples=2000)
```

## Tracking expressibility during ADAPT-VQE

The `ADAPTExpressivityTracker` hooks into `AdaptVQE.run(callback=...)` and records
the score after every operator the ansatz gains, against a **fixed** Haar
reference so the values are comparable across steps. On linear H₄ the score falls
steadily and then saturates as ADAPT adds operators:

```python
import numpy as np

from carcara.algorithms import AdaptVQE, track_adapt_expressivity
from carcara.core import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.integrals import Grid

# Linear H4 chain (8 qubits, 4 electrons; number-conserving sector d = 36).
zs = [(-1.5 + i) * 1.0 for i in range(4)]
nuclei = [(1.0, np.array([0.0, 0.0, z])) for z in zs]
grid = Grid(center=[0.0, 0.0, 0.0], box_size=9.0, h=0.22)
H = HydrogenicIntegrals(nuclei, minimal_hydrogenic_basis(nuclei),
                        grid).molecular_hamiltonian(mo_basis=True, n_electrons=4)

adapt = AdaptVQE(H, "fermionic", num_particles=(2, 2), n_spatial_orbitals=4,
                 profile=False)
result, history = track_adapt_expressivity(adapt, num_samples=600,
                                           max_iterations=8, gradient_tol=1e-3)

for step in history:
    print(f"#ops={step.num_operators}  E={step.kl_divergence:.3f}")
```

The example [`examples/adapt_expressivity.py`](https://github.com/seixas-research/carcara/blob/main/examples/adapt_expressivity.py)
runs this for the fermionic and QEB pools and plots both the expressibility-growth
curve and the final fidelity distribution against the Haar curve
(`plot_expressivity_growth`, `plot_fidelity_distribution`).
