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

## Pool choice and conserved quantities

Mandacaru provides `fermionic`, `qubit`, `qeb` and `ceo` pools. Their constructions
and implementation limits are summarized in the
[LiH pool comparison](../tutorial/adapt_vqe_lih.md).

Fermionic excitation generators preserve the specified electron counts. A
single Pauli term extracted from an excitation need not do so; a qubit-pool
ansatz can therefore leave the intended particle-number sector. Compare the
final state's conserved quantities when using such a pool. A low energy in a
different sector is not a better solution of the original molecular problem.

In Mandacaru's Jordan–Wigner construction, `ceo` reduces to `qeb`. Other mappings
can produce larger groups, but the published CEO gate savings also depend on
specialized circuit synthesis that is not implemented here.
