# Variational Quantum Eigensolver (VQE)

The Variational Quantum Eigensolver (VQE) is a hybrid classical-quantum algorithm designed to find the ground state and ground-state energy of a given Hamiltonian. It is particularly suited for Noisy Intermediate-Scale Quantum (NISQ) devices because it delegates the parameter optimization to a classical computer, keeping the quantum coherence time requirements relatively low.

---

## The Variational Principle

VQE is mathematically anchored on the Ritz variational principle from quantum mechanics. Let $H$ be a Hermitian operator representing the Hamiltonian of a system. Its ground-state energy $E_0$ is the lowest eigenvalue of $H$:

$$H |\psi_0\rangle = E_0 |\psi_0\rangle$$

For any parameterized trial state (or ansatz) $|\psi(\boldsymbol{\theta})\rangle$, where $\boldsymbol{\theta} = (\theta_1, \theta_2, \dots, \theta_M)^T$ is a vector of classical parameters, the expectation value of the energy (Rayleigh quotient) satisfies:

$$E(\boldsymbol{\theta}) = \frac{\langle\psi(\boldsymbol{\theta})| H |\psi(\boldsymbol{\theta})\rangle}{\langle\psi(\boldsymbol{\theta})|\psi(\boldsymbol{\theta})\rangle} \ge E_0$$

If the ansatz state is prepared via a unitary quantum circuit acting on a reference state $|\Phi_{\text{ref}}\rangle$ (such as the Hartree-Fock state), the state is normalized ($\langle\psi(\boldsymbol{\theta})|\psi(\boldsymbol{\theta})\rangle = 1$), and the energy expectation simplifies to:

$$E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})| H |\psi(\boldsymbol{\theta})\rangle = \langle\Phi_{\text{ref}}| U^\dagger(\boldsymbol{\theta}) H U(\boldsymbol{\theta}) |\Phi_{\text{ref}}\rangle$$

The goal of VQE is to minimize $E(\boldsymbol{\theta})$ by classically updating the parameter vector $\boldsymbol{\theta}$ using optimization algorithms (e.g., COBYLA, Nelder-Mead, L-BFGS-B):

$$E_{\text{VQE}} = \min_{\boldsymbol{\theta}} \langle\psi(\boldsymbol{\theta})| H |\psi(\boldsymbol{\theta})\rangle \approx E_0$$

---

## Second-Quantized Electronic Hamiltonian

In quantum chemistry, the electronic Hamiltonian in the Born-Oppenheimer approximation is expressed in second quantization using a set of orthonormal spin-orbitals:

$$H = \sum_{pq} h_{pq} a^\dagger_p a_q + \frac{1}{2} \sum_{pqrs} g_{pqrs} a^\dagger_p a^\dagger_q a_s a_r + E_{\text{nuc}}$$

where:
* $a^\dagger_p$ and $a_q$ are fermionic creation and annihilation operators satisfying the canonical anticommutation relations:
  $$\{a_p, a^\dagger_q\} = \delta_{pq}, \quad \{a_p, a_q\} = \{a^\dagger_p, a^\dagger_q\} = 0$$
* $h_{pq}$ represent the one-body integrals (kinetic energy of electrons and nuclear attraction):
  $$h_{pq} = \int \phi^*_p(\mathbf{r}) \left( -\frac{1}{2} \nabla^2 - \sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|} \right) \phi_q(\mathbf{r}) \, d\mathbf{r}$$
* $g_{pqrs}$ are the two-body electron-repulsion integrals (ERI) in physicists' convention:
  $$g_{pqrs} = \iint \frac{\phi^*_p(\mathbf{r}_1) \phi^*_q(\mathbf{r}_2) \phi_s(\mathbf{r}_1) \phi_r(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} \, d\mathbf{r}_1 d\mathbf{r}_2$$
* $E_{\text{nuc}}$ is the classical nuclear repulsion energy.

---

## Fermion-to-Qubit Mappings

Since quantum computers operate on qubits (represented by spin-1/2 algebra with Pauli matrices $I, X, Y, Z$) rather than fermions, fermionic creation and annihilation operators must be mapped to spin operators. Carcará supports three mapping methods:

### 1. Jordan-Wigner (JW) Mapping
The Jordan-Wigner mapping maps fermionic occupations directly to qubit states, representing the nonlocal fermionic anticommutation phases using chains of $Z$ gates:

$$a^\dagger_j = I^{\otimes j-1} \otimes \left( \frac{X - iY}{2} \right) \otimes Z^{\otimes N-j}$$
$$a_j = I^{\otimes j-1} \otimes \left( \frac{X + iY}{2} \right) \otimes Z^{\otimes N-j}$$

### 2. Parity Mapping
The Parity mapping stores the parity of occupations of preceding spin-orbitals on each qubit. This configuration allows a **two-qubit reduction** by exploiting the conservation of total electron number and spin parity, which reduces the active qubit count by 2.

### 3. Bravyi-Kitaev (BK) Mapping
The Bravyi-Kitaev mapping uses a binary tree structure to store partial parities, balancing the locality of update operations and parity calculations. Both JW and BK scale the operator weight logarithmically or linearly.

---

## The UCCSD Ansatz

A common physically motivated trial state is the **Unitary Coupled Cluster** (UCC) ansatz. The Unitary Coupled Cluster with Singles and Doubles (UCCSD) defines the parameterized state as:

$$|\psi(\boldsymbol{\theta})\rangle = e^{T(\boldsymbol{\theta}) - T^\dagger(\boldsymbol{\theta})} |\text{HF}\rangle$$

where $|\text{HF}\rangle$ is the Hartree-Fock reference state, and $T = T_1 + T_2$ represents the single and double excitation operators:

$$T_1 = \sum_{i \in \text{occ}, a \in \text{virt}} \theta_i^a a^\dagger_a a_i$$
$$T_2 = \sum_{i < j \in \text{occ}, a < b \in \text{virt}} \theta_{ij}^{ab} a^\dagger_a a^\dagger_b a_j a_i$$

Because exponentiating a large sum of non-commuting operators is difficult to implement on a quantum circuit, the unitary is approximated using the first-order **Trotter-Suzuki decomposition**:

$$e^{T(\boldsymbol{\theta}) - T^\dagger(\boldsymbol{\theta})} \approx \prod_k e^{\theta_k (G_k - G_k^\dagger)}$$

where $G_k$ represents individual single or double excitations.
