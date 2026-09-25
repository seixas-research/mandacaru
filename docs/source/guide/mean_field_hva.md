# Classical mean field and Hamiltonian variational ansatz

## RHF and UHF baselines

`method="rhf"` solves a closed-shell restricted Hartree–Fock problem.
`method="uhf"` solves an unrestricted problem, including open shells. Both
methods use the same basis and real-space integrals as the quantum solvers.
They perform classical SCF only: there is no ansatz, operator pool, circuit,
qubit Hamiltonian materialization, or quantum energy evaluation.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6, 6, 6])
atoms.calc = Mandacaru(method="rhf", basis="HAO", h=0.35)
energy_ev = atoms.get_potential_energy()
mean_field = atoms.calc.result

# Reuse the same molecular-orbital Hamiltonian and HF reference in ADAPT-VQE.
post_hf = Mandacaru(method="adapt-vqe", **mean_field.as_quantum_problem())
correlated = post_hf.run()
```

`mean_field.scf` holds the detailed `RHFResult` or `UHFResult`; the total energy
includes nuclear repulsion and any constant term of the integral model.
`as_quantum_problem()` exports the fermionic Hamiltonian, active particle
counts, spatial orbital count, and the occupation-ordered reference. A later
calculation can also use `mean_field.qubit_hamiltonian(mapping)` and
`mean_field.reference_state(mapping)` as a matched operator/state pair for
state-vector workflows such as Quantum Echoes. Mapping is performed only when
these methods are called.

For RHF, the exported determinant is the converged SCF state. For UHF, the
classical energy is that of the unrestricted determinant, whose alpha and beta
orbitals differ. The post-HF Hamiltonian uses the shared natural-orbital basis;
its exported occupation determinant is the natural-orbital reference. Its
energy is `mean_field.reference_energy` and may exceed the UHF energy.
`mean_field.scf_state(mapping)` instead expands the **actual** UHF determinant
into that same mapped register, using Slater-determinant overlap minors. It is
the state to pass to Quantum Echoes when the unrestricted SCF state is wanted;
its expectation of `mean_field.qubit_hamiltonian(mapping)` is
`mean_field.optimal_energy` (converted to Hartree). The full state vector is
allocated only when requested.

RHF requires equal alpha and beta counts. Classical SCF nuclear forces are not
yet exposed through these methods because they require self-consistent orbital
response; the variational-state force expression is not their derivative.

## Why HVA is a method

The Hamiltonian variational ansatz has an **ordered, fixed sequence** of
Hamiltonian groups in each layer. A standalone `method="hva"` can share VQE's
optimizer, mapping, and result format while keeping that sequence intact. An
ADAPT `pool="hva"` would instead screen and append individual generators by
gradient. That is useful for an adaptive Hamiltonian-inspired ansatz, but it
does not implement the fixed-layer HVA and has a different cost and parameter
meaning. For real Hamiltonian groups and a real HF reference, every pool
gradient can also vanish at the start, causing ADAPT to stop before adding any
operator. The standalone method is therefore the framework's HVA interface.

The default decomposition groups the full one-body and two-body parts of the
fermionic Hamiltonian, $H = H_1 + H_2 + E_0$. For `layers=L`,

$$
|\Psi(\boldsymbol\theta)\rangle
= \prod_{\ell=1}^{L}
  e^{-i\theta_{\ell,2}H_2}e^{-i\theta_{\ell,1}H_1}
  |\Phi_{\mathrm{HF}}\rangle .
$$

Each complete group conserves particle number. `hva_groups=` can supply
another physical decomposition whose sum is the nonconstant Hamiltonian.
The local implementation applies each exponential exactly to a sparse state
vector. Circuit execution and checkpoints are refused until a gate-level
decomposition for noncommuting terms within each group is defined; silently
turning a group into individual Pauli rotations would change the ansatz.

```python
atoms.calc = Mandacaru(method="hva", basis="HAO", h=0.35,
                       layers=2, mapping="parity_reduced")
energy_ev = atoms.get_potential_energy()
```

The default is two layers. On a real Hamiltonian with a real HF reference,
every first derivative at the all-zero angles can vanish; HVA starts from
small nonzero deterministic angles. Some symmetric systems need a second
layer before the ansatz can lower the HF energy.
