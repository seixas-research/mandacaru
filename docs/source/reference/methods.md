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
| `vqe` | Variational quantum eigensolver with a fixed ansatz | yes | yes | no | UCCSD by default, or `ansatz={"name": "hva", ...}`. The only method with `energy_levels()`. |
| `adapt-vqe` | ADAPT-VQE: the ansatz grows from an operator pool | yes | yes | no | The default method. Pools are `fermionic`, `qubit`, `qeb`, `ceo`, `ceo-ovp` and `spin-orbit`. |
| `subspace-vqe` | Shared-unitary VQE for the lowest `num_states` states | yes | no | no | Levels are bare expectations, not Ritz values of the final subspace. |
| `subspace-adapt-vqe` | ADAPT-VQE over the same shared subspace | yes | no | no | The pool gradient is the weighted sum over all references. |
| `mcas-vqe` | Markov-chain search over ansatz architectures | yes | yes | no | The search controls are `max_steps` (default 200), `max_length`, `seed`, `temperature` and `move_weights`. Local state vector and fresh chains only: circuit execution and `resume=` raise `NotImplementedError`. |
| `valqa` | The same chain with graph-model proposals | yes | yes | no | Uses a trained `proposal_model` when one is supplied and ready; otherwise it falls back to gradient proposals and says so (`result.proposal_reason`). |

## What the flags do not say

**Forces.** A `yes` in the forces column is necessary, not sufficient.
`get_forces()` is available only for a molecular (non-periodic) geometry, with
`kinetic="spectral"` and no circuit execution (`shots > 0`, a non-default `device`
or `execute_circuits=True`). Otherwise it raises `PropertyNotImplementedError` or
`NotImplementedError`. The remaining conditions per family:

- All-electron HF and Kohn-Sham: analytic stationary forces on the frozen grid.
- PAW-LCAO RHF, UHF and semilocal Kohn-Sham (`lda`, `pbe`, `r2scan`): the force is
  a Richardson difference of the total energy at fixed orbital-frame densities.
  GHF, `hse06`, rVV10 functionals, `precision="float32"` and an unconverged state
  raise.
- The quantum methods (`vqe`, `adapt-vqe`, `mcas-vqe`, `valqa`): spin-summed
  reduced density matrices of the variational state contract displaced reduced
  Hamiltonians. They need the local state vector; on a circuit device
  (`execute_circuits=True`, `shots > 0`) `get_forces()` raises
  `PropertyNotImplementedError`.

**Periodic.** A geometry with a periodic axis (or `load_crystal=`) is solved only by
`dft`, with `lda`, `pbe`, `r2scan`, `pbe+rvv10` or `r2scan+rvv10`. Every other
method raises `NotImplementedError` ("periodic geometry calculations currently
provide native DFT"), and so does `hse06`. Periodic forces and stress are not
implemented, with or without `hubbard_u`; see [what is not available yet](#not-yet-available).
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
| `xc`, `kpts`, `smearing`, `periodic_scf_options`, `hubbard_u` | `dft` |
| `spin`, `unrestricted` | `dft` and `uhf` |
| `pool` | the quantum methods |
| `seed`, `min_length`, `max_length`, `move_weights`, `temperature`, `length_penalty`, `warm_start`, `replace_start`, `proposal_temperature`, `proposal` | `mcas-vqe` and `valqa` |

A few more conditions are checked against the method:

- `ansatz="hva"` is accepted only with `method="vqe"` and `pool="fermionic"`.
- `energy_levels()` raises `ValueError` unless the calculator is `vqe`.
- Architecture-search controls (`max_steps`, `record`, `screen_insertions`,
  `proposal_model`) are accepted only by `mcas-vqe` and `valqa`; `proposal_model`
  only by `valqa`.
- The `subspace-*` methods refuse circuit execution and `checkpoint=`/`resume=`
  (`NotImplementedError`).

Every option and its default is in [the options reference](#options-reference).
