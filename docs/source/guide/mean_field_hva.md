# Classical mean field and Hamiltonian variational ansatz

## RHF and UHF baselines

`method="rhf"` solves a closed-shell restricted Hartree-Fock problem and
`method="uhf"` an unrestricted one, including open shells. Both use the same
basis and real-space integrals as the quantum solvers. They are classical SCF
only: no ansatz, pool, circuit or quantum energy evaluation, and so no `shots`,
`device`, `backend_provider`, `backend_options` or `measurement_budget` (the
constructor refuses them). RHF requires equal alpha and beta counts.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="rhf", basis="HAO", grid_spacing=0.35)
energy_ev = atoms.get_potential_energy()

# Reuse the same molecular-orbital Hamiltonian and reference in ADAPT-VQE.
problem = atoms.calc.result.as_quantum_problem()
correlated = Mandacaru(method="adapt-vqe", **problem).run()
print(f"RHF {energy_ev:.6f} eV, ADAPT-VQE {correlated.optimal_energy:.6f} eV")
```

`as_quantum_problem()` returns the fermionic Hamiltonian in the mapping the
calculator uses, the active spatial orbital count and the particle counts, as
the keyword arguments `hamiltonian`, `n_spatial_orbitals`, `num_particles` and
`mapping`. Passing them back to `Mandacaru` makes a complete problem without a
geometry. `result.qubit_hamiltonian(mapping)` and `result.reference_state(mapping)`
give a matched operator and state for state-vector workflows. The total energy
includes nuclear repulsion and any constant term of the integral model. The
mean-field result (`result.scf`) reports convergence and iteration count.

For open shells see [the open-shell guide](open_shell.md). `method="dft"` is the Kohn-Sham counterpart, with the same hand-off; see
[Kohn-Sham DFT](dft.md).

## GHF: the mean field with spin-orbit coupling

`method="ghf"` solves generalized Hartree-Fock: one determinant of
two-component spinors with no assumption that an orbital has a definite
$S_z$. It is the Hartree-Fock reference for a spin-orbit dataset
(`paw_dataset="lda-fr"`). Without spin-orbit coupling a closed shell gives the
RHF determinant back; an open shell ends at or below UHF, since GHF may break
collinearity. A spatial `active_space` does not apply to it.

The hand-off from a GHF result to the quantum solvers is listed as unfinished in
the [native coverage](../development/native_cutover.md), so this page does not
describe it as usable.

## HVA as a VQE ansatz

`ansatz=` is an option of `method="vqe"`: a name (`"uccsd"`, the default, or
`"hva"`) or a dictionary with the name and the template's options. Nothing else
is accepted as an ansatz.

The Hamiltonian variational ansatz applies a **fixed, ordered** sequence of
Hamiltonian groups in each layer. The default decomposition groups the one-body
and two-body parts of the fermionic Hamiltonian, $H = H_1 + H_2 + E_0$, and for
`layers` $= L$

$$
|\Psi(\boldsymbol\theta)\rangle
= \prod_{\ell=1}^{L}
  e^{-i\theta_{\ell,2}H_2}e^{-i\theta_{\ell,1}H_1}
  |\Phi_{\mathrm{HF}}\rangle .
$$

The dictionary keys are exactly `layers` (default 2), `evolution`
(`"exact"`, the default, or `"trotter"`), `order`, `steps` and `seed_angle`
(default `0.1`); `name` is required. Any other key is refused, and the HVA
requires `method="vqe"` and the default `pool`.

* `"evolution": "exact"` applies each full group exponential to the state.
* `"evolution": "trotter"` compiles each group to first- or second-order
  Pauli-rotation steps with `"order"` and `"steps"`. A finite-step product
  formula is a different ansatz from the exact group exponential; raising
  `"steps"` converges toward the exact one. The HVA forces product form, which
  is also what circuit execution needs.
* `seed_angle` matters because, for a real Hamiltonian and a real reference,
  every first derivative at all-zero angles can vanish. The ansatz therefore
  starts from small nonzero angles. Some symmetric systems need a second layer
  before the ansatz can lower the Hartree-Fock energy.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="rhf", basis="HAO", grid_spacing=0.35)
atoms.get_potential_energy()
problem = atoms.calc.result.as_quantum_problem()

for label, ansatz in (("exact groups", {"name": "hva", "layers": 2}),
                      ("Trotter", {"name": "hva", "layers": 2,
                                   "evolution": "trotter", "order": 2,
                                   "steps": 2})):
    result = Mandacaru(method="vqe", ansatz=ansatz, **problem).run()
    print(f"{label:12s} {result.optimal_energy:.8f} eV")
```

`examples/vqe/03_hva_h2.py` is this comparison as a script.

What the HVA does not do:

* It needs `method="vqe"` and the default `pool`; `method="subspace-vqe"` with
  `ansatz="hva"` is refused by the constructor.
* An `ansatz` other than `"uccsd"` and `"hva"`, including a Python ansatz
  object, is refused. Z2 tapering of the Hamiltonian and groups is not
  available either. See [what is not yet available](#not-yet-available).

The checkpoint of an HVA run records its evolution policy. An exact-group
checkpoint reproduces the state through its `state_vector()` and can initialize
QPE or quantum echoes, but it does not export a generic gate circuit, since a
noncommuting group exponential is not a list of Pauli rotations; a
`"trotter"` checkpoint can be exported. See
[checkpoints and QPE](checkpoints_qpe.md).
