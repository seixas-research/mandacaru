# Adaptive Ansatz Generation (ADAPT-VQE)

While the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz is physically motivated and chemically accurate, its circuit depth and parameter count scale rapidly with system size. On NISQ devices, this fixed template often leads to deep circuits that exceed coherence times and introduce significant gate error.

**ADAPT-VQE** (Adaptive Derivative-Assembled Pseudo-Trotter VQE), introduced by Grimsley *et al.* in 2019, addresses this constraint by growing the ansatz **one operator at a time**. It dynamically constructs a compact, system-specific ansatz tailored to the electronic structure of the target system.

---

## The ADAPT-VQE Algorithm

ADAPT-VQE constructs the parameterized trial state iteratively. Let $\{A_i\}$ be a pre-defined **operator pool** consisting of anti-Hermitian generators ($A_i^\dagger = -A_i$). 

Starting from a reference state $|\psi^{(0)}\rangle = |\text{HF}\rangle$ and an empty ansatz:

1. **Gradient Screening:** At iteration $n$, evaluate the energy gradient of the current state $|\psi^{(n)}\rangle$ with respect to appending each operator $A_i$ from the pool:
   $$g_i = \frac{\partial}{\partial \theta_i} \langle\psi^{(n)}| e^{-\theta_i A_i} H e^{\theta_i A_i} |\psi^{(n)}\rangle \Big|_{\theta_i=0}$$
   Using the derivative of the matrix exponential, this gradient simplifies to the expectation value of the commutator between the Hamiltonian and the pool operator:
   $$g_i = \langle\psi^{(n)}| [H, A_i] |\psi^{(n)}\rangle$$
2. **Convergence Check:** Compute the norm or the maximum absolute value of the gradients. If:
   $$\max_i |g_i| < \varepsilon$$
   where $\varepsilon$ is a user-defined threshold (e.g., $10^{-6}$ Ha), terminate the algorithm.
3. **Ansatz Growth:** Identify the operator $A_{\text{opt}}$ that yields the largest gradient magnitude:
   $$A_{\text{opt}} = \text{argmax}_{A_i} |g_i|$$
   Append this operator to the ansatz:
   $$|\psi^{(n+1)}(\boldsymbol{\theta}, \theta_{n+1})\rangle = e^{\theta_{n+1} A_{\text{opt}}} e^{\theta_n A_n} \dots e^{\theta_1 A_1} |\text{HF}\rangle$$
   where $\boldsymbol{\theta} = (\theta_1, \dots, \theta_n)^T$.
4. **VQE Parameter Optimization:** Re-optimize the entire parameter vector $(\boldsymbol{\theta}, \theta_{n+1})$ by minimizing the energy:
   $$E^{(n+1)} = \min_{\boldsymbol{\theta}, \theta_{n+1}} \langle\psi^{(n+1)}(\boldsymbol{\theta}, \theta_{n+1})| H |\psi^{(n+1)}(\boldsymbol{\theta}, \theta_{n+1})\rangle$$
   This step is **warm-started** by using the optimal parameters from the previous iteration $n$ and setting the initial value of the new parameter $\theta_{n+1}$ to zero.
5. Set $n \leftarrow n + 1$ and repeat from Step 1.

---

## Importance of the Molecular-Orbital Basis

ADAPT-VQE must start from a **stationary reference state** with respect to the pool. 

If the Hamiltonian is represented in an arbitrary orthogonalized atomic-orbital (AO) basis, the Hartree-Fock state $|\text{HF}\rangle$ is not a stationary state of the mean-field potential. Under these conditions, the gradients of single-excitation operators are very large, causing the algorithm to select single excitations first and potentially get trapped.

To prevent this, the Hamiltonian is transformed into the **restricted Hartree-Fock molecular-orbital (MO) basis**. In this basis:
* By **Brillouin's theorem**, the electronic ground state is stationary with respect to all single-electron excitations:
  $$\langle\text{HF}| [H, a^\dagger_a a_i] |\text{HF}\rangle = 0$$
* The single-excitation gradients are zero at the first step, forcing the algorithm to select physical electron correlation operators (double excitations) first. This avoids artificial optimization traps and speeds up convergence.

---

## Operator Pools

The efficiency of ADAPT-VQE depends heavily on the chosen operator pool. Carcará implements four distinct pools:

### 1. Fermionic Pool (`"fermionic"`)
Consists of spin-adapted single and double fermionic excitation operators, mapped to qubits using Jordan-Wigner. While chemically intuitive, these operators contain long Jordan-Wigner $Z$-string chains, which compile to deep quantum circuits.

### 2. Qubit Pool (`"qubit"`)
Translates each fermionic excitation into individual Pauli string terms (after Jordan-Wigner mapping) and treats each string as an independent pool operator. This pool yields shallow quantum circuits per step but requires a larger number of optimization parameters.

### 3. Qubit Excitation Basis Pool (`"qeb"`)
Constructed by dropping the non-local Jordan-Wigner $Z$-strings from the fermionic excitation operators. The resulting operators retain the excitation character but compile to shallow, distance-independent CNOT networks.

### 4. Coupled-Exchange Operator Pool (`"ceo"`)
Uses generators that share an entangling structure, allowing multiple excitations to be implemented on hardware using a single, unified CNOT block. This pool achieves the highest energy accuracy per CNOT gate.
