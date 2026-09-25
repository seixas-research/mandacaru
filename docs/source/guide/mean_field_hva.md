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

HVA can start from the actual unrestricted determinant without constructing
that full vector in advance: `Mandacaru(method="hva", reference=mean_field)`
reuses the mean-field Hamiltonian and prepares the UHF state by spin-resolved
Givens rotations. The default `as_quantum_problem()` handoff still starts from
the occupation-ordered natural-orbital determinant. The two starts can have
different reference energies.

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

Each complete group conserves particle number. `grouping="body_order"` is the
default. `grouping="spin_resolved"` splits one-body alpha/beta and two-body
alpha-alpha/alpha-beta/beta-beta blocks; it requires each term to conserve
both spin populations. `hva_groups=` supplies an explicit ordered
decomposition whose sum is the nonconstant Hamiltonian. Every group is
validated for finite coefficients, Hermiticity, and the symmetries required
by the selected register.

`evolution="exact"` (the default) applies each full group exponential to a
local state vector with sparse `expm_multiply`. `evolution="trotter"` compiles
each group to first- or second-order Pauli-rotation steps. The same ordered
rotations and **tied group angles** are used by the local objective and the
Qiskit, Cirq, or Braket circuit providers. A finite-step product formula is a
different ansatz from the exact group exponential; increasing `steps=` makes
it converge toward the exact one. The Pauli term order is fixed before the
fermion-to-qubit encoding, so a fixed-step ansatz has the same physical order
under each mapping. Circuit export and `execute_circuits=True`
therefore require `evolution="trotter"`. `shots=` is not an HVA optimization
option: optimize locally and use `measurement_provider=` or `remeasure()` for
one budgeted final measurement.

```python
atoms.calc = Mandacaru(method="hva", basis="HAO", h=0.35,
                       layers=2, mapping="parity_reduced")
energy_ev = atoms.get_potential_energy()
```

```python
# A finite-depth circuit-compatible HVA on the same geometry.
atoms.calc = Mandacaru(method="hva", basis="HAO", h=0.35,
                       evolution="trotter", order=2, steps=2,
                       execute_circuits=True, taper=True,
                       checkpoint="hva.json")
energy_ev = atoms.get_potential_energy()
prepared_state = atoms.calc.solver.checkpoint.state_vector()
```

`taper=True` is available with Jordan–Wigner mapping. It reduces the
Hamiltonian, **every** HVA group, and the reference with the same Z₂ sector.
If a custom group changes that sector, the run fails: dropping a fixed group
would change the ansatz. `parity_reduced` remains a separate mapping and is
not combined with `taper=True`. An actual UHF reference is tapered only when
it belongs to one selected symmetry sector and its Givens preparation
preserves that sector.

Both evolution modes can write and resume wavefunction checkpoints. Exact
checkpoints reconstruct their state vector and can initialize QPE or Quantum
Echoes; they do not claim a generic gate circuit for a noncommuting group.
Product-formula checkpoints contain the compiled rotation sequence and the
logical group angles, so their state vectors and circuits represent the same
finite-step state. Resuming checks the Hamiltonian, group order, evolution
policy, mapping, taper sector, and reference preparation.

The default is two layers. On a real Hamiltonian with a real HF reference,
every first derivative at the all-zero angles can vanish; HVA starts from
small nonzero deterministic angles (`seed_angle=0.1` by default). Some
symmetric systems need a second layer before the ansatz can lower the HF
energy. The Givens circuit path currently requires real UHF orbitals;
complex unrestricted orbitals are rejected explicitly.

The runnable [H₂ HVA example](../../../examples/37_HVA_H2.py) compares exact
and circuit-compatible layers and writes its checkpoint under `examples/data/`.
