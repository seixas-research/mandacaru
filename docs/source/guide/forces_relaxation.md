(forces-relaxation)=
# Forces, geometry optimization and molecular dynamics

`atoms.get_forces()` returns the negative gradient of the total energy in
eV/Å (ASE's sign and units; inside the engine derivatives are per Bohr). Any
ASE optimizer or dynamics object can drive a `Mandacaru` calculator through
that call. This page owns every statement about forces; the other guides link
here.

(forces-model)=
## Which model to use for geometry work

The grid is uniform and does not follow the atoms. An all-electron basis
(`HAO`, `NAO-AE`, the named Gaussian families) must resolve the nuclear 1s cusp
on that grid, and it cannot: the energy is not converged with respect to the
grid spacing near a core, and neither are its forces. Do not use them for
geometry optimization or dynamics.

**PAW-LCAO is the model for forces.** Its smooth valence orbitals and
compensation charges are resolved on the grid, so the forces are the exact
derivative of the discretized energy ([PAW-LCAO bases](pseudopotentials.md)).
Check that the energy actually binds with a scan of the bond length before you
relax: a converged optimizer is not evidence that the model has a minimum.

(forces-available)=
## What returns forces

| Calculation | Forces |
| --- | --- |
| `rhf`, `uhf`, `dft` with an all-electron basis | analytic stationary gradient (converged, real, collinear state; `kinetic="spectral"`) |
| `rhf`, `uhf`, `dft` with a PAW-LCAO basis | [analytic stationary gradient](#forces-paw-lcao) (real, collinear state; fractional occupations and `hubbard_u` included) |
| `vqe`, `adapt-vqe`, `vasqa`, `valsqa` on the local state-vector simulator, PAW-LCAO basis | [analytic, with the orbital response and the Z-vector correction](#forces-variational) |
| `dft` with a spin-orbit PAW-LCAO dataset (`lda-fr`, `pbe-fr`) on a molecule | analytic spinor gradient in Rust for `lda`, `pbe`, `pbe+rvv10`, `r2scan` and `hse06`; stress raises `PropertyNotImplementedError` |
| `ghf`, `subspace-vqe`, `subspace-adapt-vqe` | not available |
| `vqe`, `adapt-vqe` with `shots > 0` or `execute_circuits=True` | [analytic, on the density matrices measured on the device](#forces-variational), with a propagated statistical error |
| `vasqa`, `valsqa` on a circuit device; a tapered register on a device | not available |
| `dft` on periodic structures (crystals) | analytic Rust gradient for LDA, PBE, r2SCAN, rVV10 and HSE06, unpolarized, spin-polarized and with spin-orbit coupling; stress tensor via `get_stress()` for the same functionals; a DFT+U crystal with a hybrid or a spin-orbit dataset raises `NotImplementedError` |

Forces require the default `kinetic="spectral"`. Asking for forces where they are not available raises ASE's
`PropertyNotImplementedError` or `NotImplementedError`. An SCF that does not
converge raises `RuntimeError` before any force is requested. The methods are
listed in the [methods table](#methods-table).

(forces-paw-lcao)=
## PAW-LCAO forces for Hartree–Fock and Kohn–Sham states

For `method="rhf"`, `"uhf"` or `"dft"` (LDA, PBE, r²SCAN, HSE06 and the rVV10
pairings) on a PAW-LCAO basis the force is the analytic gradient of the
converged state, in Rust, with no displaced SCF:

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]], cell=[6, 6, 6])
atoms.center()
atoms.calc = Mandacaru(method="rhf",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25)
forces = atoms.get_forces()                  # eV / angstrom
details = atoms.calc.result.force_details    # scheme, stationarity
```

At a converged state the gradient is the partial derivative of the energy at
fixed atomic-orbital densities minus the Pulay term `-tr[W dS]`, with `W` the
energy-weighted density of the occupied orbitals. It is a sum of named terms,
each the derivative of one energy term of the model:

- the terms that act on the grid orbitals (kinetic, long-range ionic potential,
  Hartree, exchange–correlation with its gradient and kinetic-density parts,
  compensation potential, exact exchange, overlap) are contracted with the
  derivative of the basis functions;
- the terms on the atoms' spheres (projections, short-range local potential,
  compensation charges, ionic potentials) and the potentials centered on an
  atom (long-range ionic, partial core) are differentiated with the setup's
  own quadrature;
- the ion–ion repulsion is added analytically.

rVV10 needs no term of its own: its kernel depends on the positions only
through the density. A state with Fermi-Dirac occupations
([Occupations](#dft-degenerate-shell)) is differentiated at fixed occupations,
which is the derivative of the free energy `E - width S` that the SCF
minimizes. A molecule with `hubbard_u` adds the derivative of the +U energy
through the shell projections of the moving atom ([DFT+U](dft_u.md)).

Before contracting, the force checks the state: its density must be built from
its own occupations, and the largest element of `(1 - D) F D` (the
`stationarity` in `force_details`) must stay below `1e-6` (`1e-5` with
`precision="float32"`). `force_details` holds `scheme` (`"analytic"`) and
`stationarity`; they are also written to the run log ([run-log
reference](#run-log-reference)).

(forces-variational)=
## Forces of a VQE or ADAPT-VQE state

For `vqe`, `adapt-vqe`, `vasqa` and `valsqa` the force is analytic, on a
PAW-LCAO basis only (an all-electron basis raises `NotImplementedError`). With
the state's reduced density matrices held fixed, the energy depends on the
geometry through the integrals and through the orbitals the Hamiltonian was
built in. Their response is solved, not displaced: the natural-orbital rotation
of an MP2 active space by its eigenvector adjoint (zero inside an exactly
degenerate pair), the MP2 amplitudes by their Lagrangian, the canonical
occupied-occupied and virtual-virtual rotations from the canonical condition,
and the occupied-virtual ones by one CPHF solve. On an unrestricted reference the
model orbitals are the natural orbitals of the total density; their eigenvector
response and one CPHF solve over both spins take that place. An unrestricted MP2
active space adds the response of its open-shell MP2 natural orbitals (the
virtual ones and the active doubly occupied ones, each by its eigenvector
adjoint): the open-shell MP2 is a function of the integrals in those total-density
natural orbitals alone, so its amplitude Lagrangian (singles, same-spin and
opposite-spin pairs, the per-spin Fock blocks) becomes a pair of effective
densities there. Everything folds into relaxed
one- and two-body densities and an energy-weighted density, contracted once with
the derivative integrals of the PAW-LCAO model; the two-body density enters as
Coulomb forms (the eigenpairs of its pair matrix). The ansatz is not
re-optimized and no SCF is repeated. `force_details` reports `scheme`
(`"analytic"`), `orbital_response`, `cphf_residual`,
`reference_stationarity` (the largest occupied-virtual Fock element of the
reference) and `coulomb_forms`:

```python
atoms.get_potential_energy()
result = atoms.calc.result
force = result.forces()            # eV / angstrom
result.force_details["cphf_residual"]
```

The analytic force reproduces the finite-difference route it replaced (displaced
SCF, selection and Hamiltonian, Richardson steps), which is kept as a test
oracle, to below `1e-7` eV/angstrom on every component, with the one exception
below. The orbitals the Hamiltonian is built in sit in a deterministic frame
inside every exactly degenerate cluster ([Degenerate
orbitals](#degenerate-orbitals)), so the production energy is a function of the
geometry that the force differentiates.

**Known limitation (unrestricted references).** The analytic force of a truncated
`uhf` ansatz whose active space cuts the exact zero-occupation cluster of
natural orbitals differs from the derivative of the energy: on linear H3 it is
20.1 meV/angstrom off a force of 15.4 eV/angstrom (0.13 percent), while a central
difference of the production energy and the finite-difference oracle agree with
each other to `1e-3` meV/angstrom. A pair of orbitals degenerate by symmetry is
exact. The cause is open; it is not the deterministic frame. Read from the
maintainers' measurement, not requested on this build.

On a circuit device (`shots > 0` or `execute_circuits=True`, `vqe` and
`adapt-vqe`) the force is the same analytic one on the density matrices measured
on the device (`measured_rdms()`). The parameter gradient and Hessian are
measured with the shift rule of the circuit's Pauli rotations (`+-pi/4`, exact):
`1 + 2 R^2` energy evaluations for `R` rotations, in one batch. `H z = g` is
solved by the same truncated solve, a negative curvature beyond its cutoff
raises by name, and the Z-vector correction is applied in full through the
measured RDMs of the `2 R` shifted circuits (one more batch, with the state's
own). Every circuit is counted before the first job and checked against
`measurement_budget["circuit_instances"]` and the run budget; a refusal names the
count. The statistical error is propagated exactly from the measured RDM errors
(`Var F = J Sigma J^T`, one force per unique measured element, at most 256)
and, on a shot run, from the standard errors of the Hessian job's energies:
every element of `g` and `H` is linear in those independent energies, so
`Cov(z)` follows to first order, and the force adds `G Cov(z) G^T` with
`G = dF/dz` from the shifted circuits' densities (the two jobs are
independent). When the Hessian energies have standard errors, a Hessian
eigenvalue within five of its own standard errors of zero raises by name with
both numbers: the correction along it would be set by shot noise (at five sigma
the propagation's neglected bias is under 4 percent). Left out, and stated in
`error_excludes`: the covariance between different RDM elements of one circuit,
and the second-order terms. When a measurement basis needs more covariance
products than the classical limit (see `energy_error`), the energies have no
standard error: the force is still measured and returned,
`gradient_error_ha_per_bohr` is `None` with `gradient_error_reason`, and the
run log states once (`hessian_curvature_check`) that the five-sigma check was
skipped (written only when a Z vector was solved). `force_details` reports
`gradient_error_ha_per_bohr`, `hessian_rule`, `hessian_rotations`,
`hessian_energy_evaluations`, `hessian_circuits`, `rdm_circuits` and
`device_circuits`, and, when the errors are available, `z_vector_error_ha_per_bohr`
(the Z-vector part alone), `error_elements` and `error_excludes`, with
`hessian_lowest_curvature_error` and `hessian_curvature_significance` (the
smallest `|lambda| / sigma(lambda)`) when a Z vector was solved on a shot run.
No state vector is formed on a shot run.
`examples/relaxation/02_lih_relaxation.py` relaxes LiH with ADAPT-VQE forces.

(forces-lagrangian)=
### The residual-gradient (Z-vector) correction

Contracting the state's density matrices with the displaced Hamiltonian is the
force of the optimized energy surface only when the parameter gradient
`g = dE/dtheta` of the state is zero. A loosely converged optimization leaves a
residual `g`, and the bare contraction then misses the force by a term first
order in `|g|`. The force is therefore the geometry derivative of the
Lagrangian `E - z . g`, where `z` solves `H z = g` with `H` the parameter
Hessian at fixed geometry. This needs no displaced re-optimization. For a
product-form ansatz (every ADAPT-VQE, VASQA, VALSQA and Trotterized state) `H`
is exact: one forward sweep keeps every partial state, one reverse sweep every
adjoint, each derivative state is carried once to the end collecting the second
derivatives, and one Hamiltonian action per generator gives the rest. The
summed UCC exponential of `vqe` builds `H` column by column from central
differences of its analytic gradients (`2 n` gradient evaluations for `n`
parameters). More than 1500 parameters raise. The error of `H` is measured, not
assumed: for the exact Hessian `H - H^T = 0`, so the norm of the antisymmetric
part is the size of the truncation and roundoff error (roundoff alone when `H`
is exact). The exact tangent `psi' = sum_k z_k d psi / d theta_k` then gives
`z . g = 2 Re <psi'| H |psi>` exactly (the states `psi +- s psi'`, a quadratic
form), so the correction is a change of the density matrices,
`gamma - (gamma_+ - gamma_-) / (2 s)`, relaxed and contracted with the state's
own in one analytic contraction. The
corrected force of a loose run differs from a tightly converged run's by a term
of order `|g|^2`; the bare one by `|g|`.

(forces-lagrangian-rule)=
**The rule for flat directions.** The `ceo` and fermionic pools give Hessians
whose smallest eigenvalues are many orders below the largest (a spread of
`1e5` to `1e7` in water with 6 to 10 active orbitals), and a redundant parameter gives an exact zero. The
solve is the pseudo-inverse of the symmetrized `H`: every eigen-direction with
`|lambda| <= cutoff` is dropped, where `cutoff` is the larger of ten times the
measured noise and `1e-10` of the largest eigenvalue. A dropped direction
carries no correction; this is the minimum-norm `z`. An eigenvalue below
`-cutoff` is a real negative curvature (the state is not near a minimum of its
parameters) and raises, as does a Hessian with every direction under the cutoff.
What the truncation left out is reported: the number of dropped directions, the
norm of `g` along them and `z_vector_neglected_scale` (that norm over the
cutoff, a floor on the `|z|` they would add).

The correction is always applied to `vqe`, `adapt-vqe`, `vasqa` and `valsqa`
and is skipped when `|g|` is below `1e-10`. `result.force_details` reports it:

| key | meaning |
| --- | --- |
| `stationarity` | `"lagrangian"` (corrected), `"stationary"` (the norm of `g` under the floor, nothing added) or `"assumed"` (a state with no ansatz, an exact eigenvector) |
| `parameter_gradient_norm` | the norm of `g`, Ha per parameter unit |
| `z_vector_norm` | the norm of `z`, the Newton step to the optimum |
| `hessian_parameters`, `z_vector_gradient_evaluations` | the size of `H` and the gradient evaluations that built it (none when exact) |
| `hessian_exact`, `hessian_hamiltonian_actions` | whether `H` is exact (a product-form ansatz) or differenced (the summed exponential), and the Hamiltonian actions it cost |
| `hessian_noise`, `hessian_cutoff` | the measured error of `H` and the curvature below which a direction is dropped |
| `hessian_lowest_curvature`, `hessian_highest_curvature`, `hessian_smallest_kept_curvature` | the spectrum of `H` |
| `z_vector_dropped`, `z_vector_dropped_gradient_norm`, `z_vector_neglected_scale` | what the truncation dropped and left out |

A negative curvature beyond the cutoff raises, and so does a Hessian with no
direction above it ([the rule](#forces-lagrangian-rule)); a redundant or flat
direction does not. The states of the subspace
methods (`"subspace-vqe"`, `"subspace-adapt-vqe"`) are an ensemble: they have no
single parameter gradient and their forces are refused, as before.

The correction is the z-vector of Handy and Schaefer, in the Lagrangian form of
Helgaker and Jørgensen, applied to the parameters of a variational quantum
state as in the derivative theory of Mitarai, Nakagawa and Mizukami.
Mandacaru's own choices are to apply it to an optimization that stopped short of
stationarity, to form `z . g` from the exact tangent of the state, and to build
the Hessian of a product-form state exactly from its adjoint sweeps:

- Handy, N. C. and Schaefer III, H. F.,
  *On the evaluation of analytic energy derivatives for correlated wave functions*,
  *J. Chem. Phys.* **81**, 5031–5033 (1984),
  [doi:10.1063/1.447489](https://doi.org/10.1063/1.447489).
- Helgaker, T. and Jørgensen, P.,
  *Configuration-interaction energy derivatives in a fully variational formulation*,
  *Theor. Chim. Acta* **75**, 111–127 (1989).
- Mitarai, K., Nakagawa, Y. O., and Mizukami, W.,
  *Theory of analytical energy derivatives for the variational quantum eigensolver*,
  *Phys. Rev. Research* **2**, 013129 (2020),
  [doi:10.1103/PhysRevResearch.2.013129](https://doi.org/10.1103/PhysRevResearch.2.013129).

(forces-translation)=
## `project_translation`

`project_translation` accepts `"auto"` (the default), `True` or `False`. For a
molecule with more than one atom, `"auto"` and `True` subtract the mean force,
so a free molecule cannot drift; `False` returns the raw derivative. For a
variational method on a PAW-LCAO basis the projection is onto internal
molecular directions instead, and the convention in the run log says so. A free
molecule's exact forces sum to zero, so the projection enforces a symmetry; it
removes only the translational part of the grid error, and it does not replace a
grid fine enough to trust.

(forces-grid)=
## The grid

The first force request freezes the integration grid; later geometries of the
same atoms in the same cell reuse it, which is what makes a relaxation smooth.
`calc.reset()` forgets it. Start from a cell that holds the molecule through the
whole optimization. Do not compare forces from calculations on different grids.

Because the grid does not move with the atoms, the force is the exact
derivative of the discretized energy, egg-box term included. The net force on a
free molecule is therefore a direct measure of the grid error: reduce
`grid_spacing` until it is small next to the residual forces you want to
resolve.

(forces-unsupported)=
## What raises

Molecular forces raise for:

- generalized Hartree–Fock (`ghf`) states, and anything that is not real and
  collinear (the spinor Kohn–Sham state of a spin-orbit dataset has its own
  gradient, above);
- a state that does not reproduce its own energy on the rebuilt integrals;
- the force of a quantum method with `precision="float32"` (`ValueError`, before
  any SCF): the analytic variational force is implemented in `float64` only,
  while the mean-field PAW-LCAO gradient takes `float32`;
- a variational state whose parameters and operator sequence do not prepare the
  state they accompany, a parameter Hessian with a negative curvature beyond its
  noise (or with none above it), and more than 1500 parameters
  ([the Z-vector correction](#forces-lagrangian)).

The force contraction accepts a frozen core of several shells. The limit of
Hartree–Fock frozen-core exchange to an empty core or a single closed 1s shell
belongs to the energy and is listed with the other
[open items](#not-yet-available). Nothing falls back to a
different calculation.

(forces-relaxation-run)=
## Relaxation and dynamics

```python
from ase.optimize import BFGS

atoms.calc = Mandacaru(method="rhf",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25, txt="relax.log")
optimizer = BFGS(atoms, logfile=None)
optimizer.run(fmax=0.05, steps=30)
atoms.calc.write_optimization_summary(atoms, fmax=0.05, optimizer=optimizer)
```

With `txt=` (or `trace=True`) every force evaluation appends its forces, the
force convention and the scheme details to the run log, and
`write_optimization_summary` closes it with the history of energy, maximum force
and net force and ASE's verdict. Scripts for relaxation and dynamics with the
supported methods are in `examples/relaxation/` and
`examples/molecular_dynamics/`.
