# Adaptive ansatz construction: ADAPT-VQE

ADAPT-VQE grows an ansatz from a pool of anti-Hermitian generators
$\{A_i\}$, with $A_i^\dagger=-A_i$. It alternates between selecting a generator
and optimizing circuit parameters. The [LiH tutorial](../tutorial/adapt_vqe_lih.md)
shows the corresponding Python interface.

## Screen candidate generators

Suppose the current normalized state is $|\psi^{(n)}\rangle$. Appending a
candidate unitary gives $e^{\theta A_i}|\psi^{(n)}\rangle$. Its energy derivative
at zero parameter is

```{math}
\begin{aligned}
g_i &={\left.\frac{\partial}{\partial\theta}
\langle\psi^{(n)}|e^{-\theta A_i}\hat H e^{\theta A_i}
|\psi^{(n)}\rangle\right|}_{\theta=0}\\
&=\langle\psi^{(n)}|[\hat H,A_i]|\psi^{(n)}\rangle.
\end{aligned}
```

The commutator is Hermitian, so this derivative is real. Mandacaru's local
`gradient="analytic"` option evaluates it directly from the state vector.

## Grow and optimize

Choose the index with largest gradient magnitude and add that generator:

```{math}
i_* = \operatorname*{arg\,max}_i |g_i|,
\qquad
|\psi^{(n+1)}\rangle
= e^{\theta_{n+1}A_{i_*}}
  e^{\theta_n A_{i_n}}\cdots e^{\theta_1 A_{i_1}}|\mathrm{HF}\rangle.
```

Initially, the new parameter is zero and the old parameters keep their previous
values. With the default `quenching=True`, all parameters are then reoptimized.
The [quenching guide](../guide/quenching.md) describes the alternative sequential
policy.

The outer loop stops when

```{math}
\max_i |g_i| < \varepsilon,
```

or when the growth budget is exhausted. `gradient_tolerance` specifies
$\varepsilon$ in the internal Hartree convention. `max_iterations` limits the
number of growth steps; it is distinct from the inner optimizer's `maxiter`.

A small gradient means the state is locally stationary along the available pool
directions. It does not prove global optimality, sufficient pool expressivity
or convergence of the basis and grid. Inspect unsuccessful inner optimizations
as well as the outer convergence flag.

## Why use Hartree–Fock molecular orbitals?

A converged Hartree–Fock determinant is stationary with respect to the allowed
occupied–virtual orbital rotations. Under the usual closed-shell assumptions,
Brillouin's theorem gives

```{math}
\langle\Phi_i^a|\hat H|\mathrm{HF}\rangle=0,
```

where $|\Phi_i^a\rangle$ is a singly excited determinant. The corresponding
single-excitation screening gradients therefore vanish at the initial state,
up to numerical error. Double excitations can then introduce correlation.

Hartree–Fock is a useful reference, not a requirement that every pool gradient
vanish before ADAPT-VQE starts. Nor is its determinant the exact electronic
ground state. After the ansatz grows, single-excitation gradients can become
non-zero.

## Growth strategies

Two options change *how* the ansatz grows, independently of the pool and of
each other. Both are off by default.

`tetris=True` — TETRIS-ADAPT-VQE (Anastasiou *et al.*, *Phys. Rev. Research*
**6**, 013254, 2024). Instead of appending only the largest-gradient operator,
append every operator whose support is disjoint from the ones already taken
this step, in descending gradient order, until the register is covered. The
extra operators cost **no extra measurement** — their gradients were screened
anyway — and because they act on different qubits they compile into the same
circuit layer. The ansatz becomes denser and shallower.

`prune=True` — Pruned-ADAPT-VQE (*J. Chem. Theory Comput.* **21**, 8720, 2025).
After each growth, remove the one operator that has become irrelevant, ranked by

```{math}
f_i = \frac{e^{-\alpha x_i}}{\theta_i^{2}}, \qquad x_i = i/N,
```

which favors small amplitudes ($1/\theta^2$) early in the ansatz
($e^{-\alpha x}$, $\alpha = 10$) — the ones a later operator has made
redundant. The top-ranked operator is removed only if its amplitude is below
one tenth of the mean amplitude of the four most recent operators; recently
added operators legitimately have small parameters, so the threshold is
measured against them rather than being absolute. The parameters are not
re-optimized, since the removed operator was by construction doing almost
nothing.

Measured on BeH₂ (FAO, *h* = 0.35, `qeb`, L-BFGS-B, gradient tolerance 10⁻⁵),
all four variants reaching the same energy:

| | operators | growth steps | CNOTs | depth | cost evaluations |
| :--- | ---: | ---: | ---: | ---: | ---: |
| default | 21 | 21 | 656 | 871 | 1731 |
| `tetris=True` | 17 | 13 | 464 | 609 | 982 |
| `prune=True` | 16 | 20 | 504 | 665 | 1647 |
| both | 16 | 13 | 460 | 609 | 982 |

On a strongly correlated linear H₄ that does not converge within the operator
budget, pruning is the stronger of the two: 15 operators against 24 for the same
energy, 44 % fewer CNOTs and 53 % less depth — at 2.5× the classical cost,
because near convergence it can remove an operator the next step re-selects.
That cycling is why the growth steps carry their own budget when pruning is on.

`tetris=True` is close to free and never lengthens the circuit; `prune=True`
trades classical evaluations for gate count. The `[OPTIMIZATION SETUP]` block
records which ran, and `result.pruned_operators` lists what pruning removed.

## Pool choice and conserved quantities

Mandacaru provides `fermionic`, `qubit`, `qeb` and `ceo` pools. Their constructions
and implementation limits are summarized in the
[LiH pool comparison](../tutorial/adapt_vqe_lih.md).

Fermionic excitation generators preserve the specified electron counts. A
single Pauli term extracted from an excitation need not do so; a qubit-pool
ansatz can therefore leave the intended particle-number sector. Compare the
final state's conserved quantities when using such a pool. A low energy in a
different sector is not a better solution of the original molecular problem.

The `ceo` pool follows Ramôa *et al.*, *npj Quantum Information* **11**, 86
(2025). A set of four spin-orbitals admits more than one qubit excitation — two
when it holds two α and two β orbitals, three when all four share a spin — and
every one of them is built from the *same* eight Pauli strings, differing only in
signs. Coupling them therefore costs no extra entangling structure, and their sum
or difference needs only four of those strings, which roughly halves the compiled
CNOT count against `qeb` (LiH: 112 against 208).

Two consequences are worth knowing. First, `ceo` is the one pool built from
**generalized** excitations: with the source and target orbitals restricted to be
occupied and unoccupied in the reference, each set carries a single excitation and
there is nothing to couple. The pool is correspondingly larger — 660 operators
against QEB's 92 for a frozen-core water — although the gradients are linear
combinations of the same Pauli expectation values, so the measurement cost is
unchanged. Second, a growth step may add more than one parameter: when several of
the coupled excitations have a non-zero gradient they are appended with
independent parameters (the paper's MVP-CEO), so the parameter count can exceed
the iteration count.

The paper's specialized 9- and 13-CNOT circuit syntheses are not implemented;
Mandacaru compiles every generator through the generic `{cx, u}` path. That has
one consequence worth knowing: an MVP step compiles as two or three separate
eight-string excitations and therefore *costs* gates rather than saving them.
`ceo-ovp` is the paper's one-parameter variant (its Supplementary Sec. I) — the
same pool, never expanded — and it is the one that delivers the reduction here.
On the LiH curve of `examples/24_ADAPTVQE_LiH_IBM.py` (STO-3G, 12 qubits) it
matches `qeb` in energy with **104 CNOTs against 208**, while the adaptive `ceo`
needs 248. Use `ceo-ovp` when the gate count is what matters.
