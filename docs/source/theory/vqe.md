# Variational quantum eigensolver

VQE estimates a ground-state energy by minimising the expectation value of a
Hamiltonian in a parameterised trial state. A quantum circuit prepares the state;
a classical optimiser updates the parameters. A local simulator can perform the
same workflow, as in [the LiH tutorial](../tutorial/vqe_lih.md).

## The variational principle

For a Hermitian Hamiltonian $\hat H$ with lowest eigenvalue $E_0$, any non-zero
trial state satisfies

```{math}
E(\boldsymbol{\theta}) =
\frac{\langle\psi(\boldsymbol{\theta})|\hat H
|\psi(\boldsymbol{\theta})\rangle}
{\langle\psi(\boldsymbol{\theta})|\psi(\boldsymbol{\theta})\rangle}
\geq E_0.
```

A unitary circuit $U(\boldsymbol{\theta})$ acting on a normalised reference state
preserves its norm. The denominator is then one:

```{math}
|\psi(\boldsymbol{\theta})\rangle
= U(\boldsymbol{\theta})|\Phi_{\mathrm{ref}}\rangle,
\qquad
E(\boldsymbol{\theta})
= \langle\Phi_{\mathrm{ref}}|U^\dagger(\boldsymbol{\theta})
\hat H U(\boldsymbol{\theta})|\Phi_{\mathrm{ref}}\rangle.
```

The exact minimum within the ansatz is an upper bound to the ground-state energy
of that Hamiltonian. A practical optimiser may find a local minimum. Measurement
noise can also obscure the bound. Neither optimisation nor a flexible ansatz
corrects an inaccurate basis or numerical integration grid.

## Electronic Hamiltonian

In the Born–Oppenheimer approximation, the nuclei are fixed while we solve the
electronic problem. With orthonormal spin orbitals $\chi_p(x)$, where $x$ includes
position and spin, the molecular Hamiltonian in atomic units is

```{math}
\hat H = \sum_{pq} h_{pq} a_p^\dagger a_q
+ \frac{1}{2}\sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_s a_r
+ E_{\mathrm{nuc}}\hat I.
```

The fermionic operators obey

```{math}
\{a_p,a_q^\dagger\}=\delta_{pq},
\qquad \{a_p,a_q\}=\{a_p^\dagger,a_q^\dagger\}=0.
```

Using the physicists' convention consistently with the operator order above,

```{math}
\begin{aligned}
h_{pq} &= \int \chi_p^*(x)
\left[-\frac{1}{2}\nabla^2
-\sum_A\frac{Z_A}{|\mathbf r-\mathbf R_A|}\right]\chi_q(x)\,\mathrm dx,\\
g_{pqrs} &= \iint
\frac{\chi_p^*(x_1)\chi_q^*(x_2)\chi_r(x_1)\chi_s(x_2)}
{|\mathbf r_1-\mathbf r_2|}\,\mathrm dx_1\,\mathrm dx_2,\\
E_{\mathrm{nuc}} &= \sum_{A<B}\frac{Z_AZ_B}{|\mathbf R_A-\mathbf R_B|}.
\end{aligned}
```

The ideal Coulomb expressions explain the model. Carcará evaluates integrals on
a finite grid and uses numerical treatments of nuclear singularities, so a grid
convergence study remains necessary. Its low-level integral arrays are in atomic
units; the tutorial's result objects report eV.

For neutral LiH, the all-electron problem contains four electrons. A frozen-core
calculation keeps the lowest doubly occupied molecular orbital fixed and builds
an effective Hamiltonian for the remaining two electrons. Its constant and
one-electron terms retain the core contribution; simply reducing the electron
count is not equivalent.

## Map fermions to qubits

Carcará supports Jordan–Wigner, parity and Bravyi–Kitaev mappings. For zero-based
orbital indices, the Jordan–Wigner operators can be written

```{math}
a_j^\dagger = \left(\prod_{k<j}Z_k\right)\frac{X_j-iY_j}{2},
\qquad
a_j = \left(\prod_{k<j}Z_k\right)\frac{X_j+iY_j}{2}.
```

An omitted factor is an identity on the other qubits. The Z string enforces the
fermionic anticommutation signs. Its weight can grow linearly with the orbital
index; Bravyi–Kitaev distributes the parity information to obtain logarithmic
operator weight. Parity mapping also allows an optional two-qubit reduction
when the required spin-parity sectors are fixed.

A mapping changes the representation, not the underlying spectrum. The physical
particle-number sector must still be identified when comparing eigenvalues.

## The UCCSD ansatz

Unitary coupled cluster with single and double excitations starts from a
Hartree–Fock determinant:

```{math}
|\psi(\boldsymbol{\theta})\rangle
= e^{T(\boldsymbol{\theta})-T^\dagger(\boldsymbol{\theta})}
|\mathrm{HF}\rangle,
\qquad T=T_1+T_2,
```

with occupied indices $i,j$ and unoccupied indices $a,b$:

```{math}
T_1=\sum_{ia}\theta_i^a a_a^\dagger a_i,
\qquad
T_2=\sum_{i<j,\,a<b}\theta_{ij}^{ab}a_a^\dagger a_b^\dagger a_j a_i.
```

A circuit usually implements an ordered product of individual excitation
unitaries. This generally approximates the exponential of their sum because
different excitations need not commute. Within an individual generator whose
Pauli terms commute, the factorisation into Pauli rotations is exact. These are
two distinct statements; exact compilation of each factor does not remove the
ansatz approximation.
