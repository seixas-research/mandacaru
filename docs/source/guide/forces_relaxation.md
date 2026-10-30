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
| `rhf`, `uhf`, `dft` with a PAW-LCAO basis | [finite difference of the energy at fixed densities](#forces-paw-lcao) |
| `vqe`, `adapt-vqe`, `mcas-vqe`, `valqa` on the local state-vector simulator | [displaced energies of the converged state](#forces-variational) |
| `ghf`, `subspace-vqe`, `subspace-adapt-vqe` | not available |
| any run with `shots > 0` or a device other than `AER_simulator` | not available |
| periodic structures | not available (`NotImplementedError`) |

Forces require the default `kinetic="spectral"`. Asking for forces where they are not available raises ASE's
`PropertyNotImplementedError` or `NotImplementedError`. An SCF that does not
converge raises `RuntimeError` before any force is requested. The methods are
listed in the [methods table](#methods-table).

(forces-paw-lcao)=
## PAW-LCAO forces for Hartree–Fock and Kohn–Sham states

For `method="rhf"`, `"uhf"` or `"dft"` (LDA, PBE, r²SCAN) on a PAW-LCAO basis
the force comes from the converged state without a displaced SCF:

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]], cell=[6, 6, 6])
atoms.center()
atoms.calc = Mandacaru(method="rhf",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25)
forces = atoms.get_forces()                  # eV / angstrom
details = atoms.calc.result.force_details    # scheme, step_bohr, step_error_estimate
```

The orbitals are orthonormal in a Löwdin frame that is rebuilt at every
geometry, so at a converged SCF the force is the partial derivative of the total
energy (electronic plus ion–ion) at fixed densities in that frame. The overlap
(Pulay) term is carried by the geometry dependence of the Löwdin matrix inside
the rebuilt integrals, and no separate term is needed for GGA, meta-GGA or open
shells.

Mandacaru takes the derivative as a Richardson difference of that energy on the
same fixed grid, with the retained PAW setup:

- the step is `1e-3` Bohr, combined with the half step; the truncation error is
  fourth order in the step, and `step_error_estimate` is `|fine - coarse| / 3`
  (the largest over all coordinates);
- each Cartesian coordinate needs four energies, so one force evaluation
  rebuilds the integrals `12 N` times for `N` atoms. No SCF is repeated;
- Hartree–Fock keeps the pivots of the reference Coulomb factorization, so the
  truncation of the factors is a smooth function of the positions;
- the basis and projector centers, the overlap and compensation augmentation,
  the compensation charges and core densities on the grid, the one-center
  Coulomb and exchange–correlation terms, the frozen-core exchange, the local
  potential, the kinetic energy and the ion–ion repulsion are differentiated
  together.

`atoms.calc.result.force_details` holds `scheme`, `step_bohr` and
`step_error_estimate`; they are also written to the run log ([run-log
reference](#run-log-reference)).

(forces-variational)=
## Forces of a VQE or ADAPT-VQE state

For `vqe`, `adapt-vqe`, `mcas-vqe` and `valqa` the force contracts the reduced
density matrices of the converged state with the displaced Hamiltonian. The
mean-field reference is continued to each displaced geometry and the active-space
selection followed; the ansatz is not re-optimized. The ASE default is a
two-point difference with a `1e-4` Bohr step; its error estimate is reported
as unmeasured. For a convergence check call the result directly:

```python
atoms.get_potential_energy()
result = atoms.calc.result
force = result.forces(step=1e-3, extrapolate=True)   # eV / angstrom, Richardson
result.force_details["step_error_estimate"]
```

The variational forces require a local state (no shots, no device). Run the
optimization to tight convergence first; the force is the derivative of the
energy of the state you have, not of the exact ground state of the sector.
`examples/relaxation/02_lih_relaxation.py` relaxes LiH with ADAPT-VQE forces.

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

PAW-LCAO mean-field forces raise for:

- generalized (`ghf`) states, and anything that is not real and collinear;
- screened exact exchange (`xc="hse06"`) and nonlocal correlation
  (`"pbe+rvv10"`, `"r2scan+rvv10"`) on a PAW-LCAO basis;
- `precision="float32"`, because the difference needs `float64` samples;
- a state that does not reproduce its own energy on the rebuilt integrals.

Frozen-core exchange in Hartree–Fock is limited to an empty core or a single
closed 1s shell, and spin-orbit datasets are refused; both are listed with the
other [open items](#not-yet-available). Nothing falls back to a
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
