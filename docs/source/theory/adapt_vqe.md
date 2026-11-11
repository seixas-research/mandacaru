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

The commutator is Hermitian, so this derivative is real. Because $A_i$ is
anti-Hermitian, $\langle\psi|A_i\hat H|\psi\rangle$ is minus the complex conjugate
of $\langle\psi|\hat H A_i|\psi\rangle$, and the derivative can be computed as
$g_i = 2\,\mathrm{Re}\,\langle\psi^{(n)}|\hat H A_i|\psi^{(n)}\rangle$, one
Hamiltonian application shared by the whole pool. This is the form the state-vector
solver evaluates, in Hartree per radian. The screening gradient it reports is the
magnitude $|g_i|$.

On a device (`shots > 0`) there is no state vector: $[\hat H, A_i]$ is formed in the
Pauli algebra and **measured** as an observable, the strings of every screened
generator merged into qubit-wise-commuting bases. At the Hartree-Fock determinant,
before the first operator, the derivative is a diagonal element and is computed
exactly instead (see [ADAPT-VQE on a device](#adapt-on-a-device)).

## Grow and optimize

Choose the index with largest gradient magnitude and add that generator:

```{math}
i_* = \operatorname*{arg\,max}_i |g_i|,
\qquad
|\psi^{(n+1)}\rangle
= e^{\theta_{n+1}A_{i_*}}
  e^{\theta_n A_{i_n}}\cdots e^{\theta_1 A_{i_1}}|\mathrm{HF}\rangle.
```

The new parameter starts at zero and the old parameters keep their values; then
**all** parameters are re-optimized together, starting from those values. There is
no option to freeze the earlier angles.

With the default optimizer (SLSQP), and with BFGS, the re-optimization also starts
from the curvature the previous step learned. SLSQP keeps a quasi-Newton
approximation of the Hessian of the energy in the angles and BFGS one of its
inverse; each growth step carries the previous step's approximation, extended by
the identity for the appended angles, instead of relearning it from the identity.
BFGS goes from 3310 to 633 evaluations on the same 20-qubit H<sub>2</sub>O growth
(30 steps), with per-step energies equal to 1e-14 Ha. On the 20-qubit H<sub>2</sub>O ansatz of 95
operators this takes 909 evaluations instead of 8050 and reaches the same energy
within 1 nHa; operators of equal gradient may then be picked in the other order.
The approximation is stored in a checkpoint and handed to the next geometry of a
relaxation (below), and `subspace-adapt-vqe` does the same. L-BFGS keeps only ten
correction pairs, which measured no gain on a growing ansatz, and the other
optimizers keep no curvature: they start every step from scratch.

The outer loop stops when

```{math}
\max_i |g_i| < \varepsilon,
```

or when the growth budget is exhausted. The growth budget is `max_iterations`, the
total number of operators in the ansatz (including any restored from a checkpoint);
it is distinct from the inner optimizer's `maxiter`. The thresholds are set with
`convergence={"gradient": ..., "energy": ...}`:

- `"gradient"` is $\varepsilon$ in Hartree per radian, compared with the largest pool
  gradient.
- `"energy"` is $\delta$ in eV, compared with the energy change of the last growth
  step, $|E_k - E_{k-1}| < \delta$, where $E_0$ is the reference energy.
- The default is `{"gradient": 1e-3, "energy": 1e-3}`. A criterion that is left out
  or set to `None` is not used, and a dictionary you supply enables only the keys it
  names. At least one criterion is required. When both are in use, both must hold: a
  small energy change alone can be a plateau the next operator leaves, and a small
  gradient alone a flat saddle. With an energy criterion in use the loop cannot stop
  before the first growth step, because there is no energy change to compare yet.
- Whatever the criteria, a largest gradient below $10^{-12}$ stops the growth: the
  pool has nothing left to offer.

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

The `pool` option selects one of six generator sets: `fermionic`, `qubit`, `qeb`,
`ceo`, `ceo-ovp` and `spin-orbit`. [The LiH tutorial](../tutorial/adapt_vqe_lih.md)
compares them on one molecule.

Left out, the pool is `ceo` for the growing methods (`adapt-vqe`, `subspace-adapt-vqe`)
and `fermionic` for the others (`vqe`, whose fixed ansatz and `ansatz="hva"` are built on
the fermionic excitations, `subspace-vqe`, `vasqa` and `valsqa`), on a molecule and on a
crystal alike. The fermionic pool can stop above the exact energy: on
the 4-cell metallic hydrogen chain it stops 0.0196 eV above it, where `qubit` and `ceo`
reach it. The price of `ceo` is a larger pool, with the same measurement cost per gradient.
An explicit `pool=` always wins.

Fermionic excitation generators preserve the specified electron counts. A
single Pauli term extracted from an excitation need not do so; a qubit-pool
ansatz can therefore leave the intended particle-number sector. Compare the
final state's conserved quantities when using such a pool. A low energy in a
different sector is not a better solution of the original molecular problem. By
default Mandacaru restricts the state to the particle-number sector for every pool
except `qubit` and `spin-orbit`.

The `spin-orbit` pool contains both $T - T^\dagger$ and $i(T + T^\dagger)$ for
spin-changing singles and doubles, so the ansatz can mix $\alpha$ and $\beta$
spin. It conserves the total particle number but not the spin populations, and it
needs the untapered register: it refuses `mapping="parity_reduced"`.

The `ceo` pool follows Ramôa *et al.*, *npj Quantum Information* **11**, 86
(2025). A set of four spin-orbitals admits more than one qubit excitation, and every
one of them is built from the same Pauli strings, differing only in signs. Coupling
them (their sum and difference) therefore needs no extra entangling structure. Two
consequences are worth knowing. First, `ceo` is built from **generalized**
excitations, with source and target orbitals not restricted to be occupied and
unoccupied in the reference, so its pool is larger than the others; the
gradients are linear combinations of the same Pauli expectation values, so the
measurement cost does not grow with it. Second, a growth step may add more than one
parameter: when several of the coupled excitations have a non-zero gradient they are
appended with independent parameters (the paper's MVP-CEO), so the parameter count
can exceed the iteration count.

The `ceo-ovp` pool is the paper's one-parameter variant: the same coupled operators,
but a growth step appends the single chosen operator and never expands it. The
paper's specialized 9- and 13-CNOT circuit syntheses are not implemented; every
generator is compiled through the generic path, so compare the compiled gate counts
of the pools on your molecule (the iteration table of
[the run log](../reference/run_log.md) reports them) rather than assuming the
paper's savings.
