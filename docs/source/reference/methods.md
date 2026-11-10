# Methods and capabilities

(methods-table)=
## The methods table

`Mandacaru(method=...)` builds exactly the ten methods below. Any other name raises
`ValueError` and lists them. The default is `adapt-vqe`. The "quantum", "forces"
and "periodic" columns are the three capability flags in
`mandacaru.specifications.METHODS`; the notes give the run-time conditions the flags
do not carry.

| Method | What it is | Quantum | Forces | Periodic | Notes |
| --- | --- | --- | --- | --- | --- |
| `rhf` | Restricted Hartree-Fock on the real-space grid | no | yes | no | Needs equal alpha and beta populations; an open-shell system raises `ValueError`. |
| `uhf` | Unrestricted Hartree-Fock | no | yes | no | Spin populations come from the electron count and the initial magnetic moments. |
| `ghf` | Generalized (complex spinor) Hartree-Fock | no | no | no | `get_forces()` raises `PropertyNotImplementedError`. |
| `dft` | Kohn-Sham density functional theory | no | yes | yes | Functionals are `lda`, `pbe`, `r2scan`, `hse06`, `pbe+rvv10` and `r2scan+rvv10`. A geometry with a periodic axis runs the crystal solver (see below); `spin=True` or `unrestricted=True` separates the spin channels at equal populations. `hubbard_u=` adds the on-site DFT+U correction to a crystal ([DFT+U](../guide/dft_u.md)). |
| `vqe` | Variational quantum eigensolver with a fixed ansatz | yes | yes | yes | UCCSD by default, or `ansatz={"name": "hva", ...}`. The only method with `energy_levels()`. On a crystal, see "Periodic" below. |
| `adapt-vqe` | ADAPT-VQE: the ansatz grows from an operator pool | yes | yes | yes | The default method. Pools are `fermionic`, `qubit`, `qeb`, `ceo` (the default), `ceo-ovp` and `spin-orbit`. |
| `subspace-vqe` | Shared-unitary VQE for the lowest `num_states` states | yes | no | no | Levels are bare expectations, not Ritz values of the final subspace. |
| `subspace-adapt-vqe` | ADAPT-VQE over the same shared subspace | yes | no | no | The pool gradient is the weighted sum over all references. |
| `vasqa` | VASQA (Variational Adaptive Stochastic Quantum Algorithm): Markov-chain search over ansatz architectures | yes | yes | yes | The search controls are `max_steps` (default 200), `max_length`, `seed`, `temperature` and `move_weights`. Local state vector and fresh chains only: circuit execution and `resume=` raise `NotImplementedError`. |
| `valsqa` | VALSQA (Variational Adaptive Learnable Stochastic Quantum Algorithm): the same chain with learned proposals (a pairwise ranker or a graph model) | yes | yes | yes | Uses a trained `proposal_model` when one is supplied and ready; otherwise it falls back to gradient proposals and says so (`result.proposal_reason`). |

## What the flags do not say

**Forces.** A `yes` in the forces column is necessary, not sufficient.
`get_forces()` is available only for a molecular (non-periodic) geometry, with
`kinetic="spectral"` and no circuit execution (`shots > 0`, a non-default `device`
or `execute_circuits=True`). Otherwise it raises `PropertyNotImplementedError` or
`NotImplementedError`. The remaining conditions per family:

- All-electron HF and Kohn-Sham: analytic stationary forces on the frozen grid.
- PAW-LCAO RHF, UHF and semilocal Kohn-Sham (`lda`, `pbe`, `r2scan`): the force is
  a Richardson difference of the total energy at fixed orbital-frame densities.
  Hartree-Fock GHF, rVV10 functionals, `precision="float32"` and an unconverged
  state raise.
- PAW-LCAO `dft` on a spin-orbit dataset (`paw_dataset="lda-fr"` or `"pbe-fr"`):
  spinor Kohn-Sham with analytic forces in Rust for `lda`, `pbe`, `pbe+rvv10`,
  `r2scan` and `hse06`. A molecule on such a dataset runs only `dft`: every other
  method raises `NotImplementedError` ("a molecule with a spin-orbit dataset
  needs method='dft' (spinor Kohn-Sham)"). Its stress raises
  `PropertyNotImplementedError`, and `hubbard_u` raises.
- The quantum methods (`vqe`, `adapt-vqe`, `vasqa`, `valsqa`): spin-summed
  reduced density matrices of the variational state contract displaced reduced
  Hamiltonians. They need the local state vector; on a circuit device
  (`execute_circuits=True`, `shots > 0`) `get_forces()` raises
  `PropertyNotImplementedError`.

**Periodic.** A geometry with a periodic axis (or `load_crystal=`) is solved by
`dft` (`lda`, `pbe`, `r2scan`, `pbe+rvv10`, `r2scan+rvv10` or `hse06`) and by `vqe`,
`adapt-vqe`, `vasqa` and `valsqa`. A quantum method solves the Born-von Karman supercell
of the Gamma-centered `kpts` mesh (the Gamma point alone when `kpts` is left out) with
lattice-summed electrostatics and image-summed HAO or Gaussian orbitals, and reports the
energy per primitive cell; it needs `crystal_orbitals="real-space"` (the default
`"k-space"` raises `NotImplementedError` until the per-k register exists), and its
`get_forces()` and `get_stress()` raise `NotImplementedError` (see
[the crystal guide](../guide/bloch.md)). `subspace-vqe`, `subspace-adapt-vqe`, `rhf`,
`uhf` and `ghf` raise on a crystal. `dft` forces and stress on a crystal are supported
through `atoms.get_forces()` and `atoms.get_stress()`. A hybrid (`xc="hse06"`) crystal needs the whole k
mesh: on a mesh reduced by symmetry or halved by time reversal it raises
`NotImplementedError` (see `symmetry` in the `Mandacaru` docstring).
`hubbard_u` is accepted for a crystal only: a molecule raises `ValueError`, and a
screened hybrid or a spin-orbit dataset raises `NotImplementedError`
(the full list is in [DFT+U](../guide/dft_u.md)).

**Quantum options on classical methods.** `rhf`, `uhf`, `ghf` and `dft` refuse
`shots`, `execute_circuits`, `backend_provider`, `device`, `backend_options` and
`measurement_budget`, because there is no quantum state to measure.

**Method-specific options.** An option that only some methods use raises
`ValueError` ("method=... does not use: ...") when you give it a non-default value
on another method, so nothing is silently ignored:

| Options | Used by |
| --- | --- |
| `xc`, `smearing`, `periodic_scf_options`, `hubbard_u`, `exxdiv_treatment`, `gygi_baldereschi` | `dft` |
| `kpts` | `dft`, `vqe`, `adapt-vqe`, `vasqa` and `valsqa` |
| `crystal_orbitals` | `vqe`, `adapt-vqe`, `vasqa` and `valsqa` (a value other than the default on a molecule also raises, at `run()`) |
| `spin`, `unrestricted` | `dft` and `uhf` |
| `pool` | the quantum methods |
| `taper` | `vqe`, `adapt-vqe`, `subspace-vqe` and `subspace-adapt-vqe` |
| `seed`, `min_length`, `max_length`, `move_weights`, `temperature`, `length_penalty`, `warm_start`, `replace_start`, `proposal_temperature`, `proposal` | `vasqa` and `valsqa` |

A few more conditions are checked against the method:

- `ansatz="hva"` is accepted only with `method="vqe"` and `pool="fermionic"`.
- `pool=` left out resolves by method: `"ceo"` for `adapt-vqe` and `subspace-adapt-vqe`,
  `"fermionic"` for `vqe`, `subspace-vqe`, `vasqa` and `valsqa`
  (the fixed ansatz and the search methods are built on the fermionic excitations). An explicit
  `pool=` always wins; the log's `pool:` line reports the one used.
- `energy_levels()` raises `ValueError` unless the calculator is `vqe`.
- Architecture-search controls (`max_steps`, `record`, `screen_insertions`,
  `proposal_model`) are accepted only by `vasqa` and `valsqa`; `proposal_model`
  only by `valsqa`.
- The `subspace-*` methods refuse circuit execution and `checkpoint=`/`resume=`
  (`NotImplementedError`).

**Z2 tapering.** `taper=True` removes one qubit per independent Z2 symmetry of the
qubit Hamiltonian, found in Rust on the Hamiltonian as the mapping wrote it. The
particle-number parities that `mapping="parity_reduced"` removes are among them; the
spatial symmetries of the molecule come on top (water in a minimal basis goes from 12
qubits to 8). The sector is the one that holds the reference determinant, so the
energy equals the untapered one to the optimizer's tolerance. What changes:

- The simulation is on the whole tapered register, so `sector=True` raises
  `ValueError` (the default `sector=None` resolves to `False`). `ansatz="hva"`,
  `transfer=True` and `save_hamiltonian` raise `ValueError` too.
- A pool generator that does not commute with the symmetries connects different sectors
  and cannot act in the reference's; it is dropped and counted on the run log's
  `Z2 tapering` line. The `spin-orbit` and `qubit` pools need the untapered register and
  raise at `run()`.
- The `subspace-*` methods taper their reference determinants too: the excited states
  are the ones in the reference's symmetry sector.
- Reduced density matrices (natural orbitals, volumetric fields) are read from the
  tapered state lifted back to the fermionic register, which has `2**n` amplitudes:
  they are state-vector only and raise on a shot-based run. Forces are not available on
  a tapered register (`get_forces()` raises).
- `result.state()` is the state of the tapered register, and `calc.n_qubits` its
  width. A Hamiltonian with no symmetry beyond the identity runs untapered and the run
  log says `none found`.

Every option and its default is in [the options reference](#options-reference).
