# Native calculations

Rust is Mandacaru's only calculation implementation. Python supplies immutable
geometry values, option validation, result views and optional integrations.
There is no `engine` or `engine_options` argument.

## A calculation without ASE

```python
import numpy as np
from mandacaru import Geometry, Mandacaru

geometry = Geometry(
    numbers=(1, 1),
    positions=np.array([[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]]),
    cell=np.eye(3) * 5,
)
with Mandacaru(
    method="dft", xc="pbe", basis="HAO", grid_spacing=.35,
    resources={"threads": 2, "memory_limit_bytes": 512 * 1024**2},
) as calculation:
    result = calculation.run(geometry)
    print(result.optimal_energy)  # eV, including nuclear repulsion
    print(result.forces())       # eV / angstrom
```

Positions and cells default to ångströms; `Geometry(..., units="bohr")` accepts
atomic units. Geometry copies its input arrays. `atomic_units=True` on the
calculator selects Hartree for reported energies. Native SCF and optimizer
options use atomic units. Result forces and the ASE protocol always use eV/Å.

## Optional ASE adapter

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[2.5,2.5,2.13], [2.5,2.5,2.87]], cell=[5]*3)
atoms.calc = Mandacaru(method="dft", xc="pbe", grid_spacing=.35)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
atoms.calc.close()
```

`Mandacaru` is an ASE calculator, and ASE is a required dependency. A standalone
`Geometry` can also be passed to `run`. Periodic PAW setup uses a Python table
adapter; a saved crystal already contains the data needed by its Rust solver.

## Hamiltonians are explicit

Ordinary SCF, energy, gradient and snapshot operations do not create a dense
four-index Hamiltonian. Semilocal DFT uses density fields directly. HF exchange
uses pivoted Coulomb Cholesky factors built from selected Poisson columns;
HSE06 uses screened factors. `cholesky_tolerance` defaults to 1e-10 and controls
the maximum residual Coulomb diagonal in atomic units.
For molecular PAW, signed one-center and compensation factors remain separate
from the smooth Cholesky factors. Their negative contributions are preserved.

`result.hamiltonian`, `result.save_hamiltonian(path)` and quantum solvers request
the transformed Hamiltonian explicitly. It is cached in its originating native
state. Low-level snapshots export four-index terms only with
`include_hamiltonian=True`. Dense export can still require O(M⁴) memory.

## Ownership and resource limits

Each calculator owns a resource object, context, state and result references.
Its thread setting never changes a global worker pool or BLAS setting. Scratch
is created lazily in a unique temporary directory, optionally beneath
`resources={"scratch_dir": ...}`. Closing the calculator deletes only that
directory. There is no automatic `.mandacaru/` workspace.

A result can retain its native state after its calculator is closed. Native
memory remains allocated while any owner retains it. The memory limit covers
native buffers and reserved workspaces, rather than the whole Python process.
Exceeding the limit raises `MemoryError` and releases temporary allocations.

Scientific configuration is fixed when the calculator is constructed. Create
a new calculator to change the method, basis or functional. A supplied native
model retains its original context, so configure that context at its origin.

## Coverage

| Calculation | Native support | Limits |
| --- | --- | --- |
| Molecular mean field | RHF, UHF, GHF | Gradients require real collinear orbitals and spectral kinetic energy |
| Molecular PAW | HF and variational methods, exact valence one-center terms, smooth fields and augmentation-charge accounting | Empty or single closed-1s cores; no PAW DFT, spin-orbit projectors or forces |
| Molecular DFT | `lda`, `pbe`, `r2scan`, `hse06`, `pbe+rvv10`, `r2scan+rvv10` | All-electron geometry interface |
| Periodic DFT | LDA/PBE SCF, occupations, spectra | Python PAW geometry assembly; no high-level periodic force/stress interface |
| Quantum methods | UCCSD VQE, fermionic/qubit/QEB/CEO/spin-orbit ADAPT, shared subspace searches | Other ansätze, GHF quantum handoff and learned proposals remain unported |
| Correlation | Closed-shell MP2/FNO active spaces, thresholds and frozen cores | Open-shell/local variants and symmetry constraints remain unported |
| Persistence | Native JSON Hamiltonians, crystals, completed checkpoints; Python columnar codecs | Provider job lifecycle integration remains incomplete |

Unsupported combinations raise errors. The [audit](../development/native_cutover.md)
records remaining scientific Python modules. D4 and a complete XC catalog are
not enabled. A checkpoint supplies a compatible warm start; it does not restore
an optimizer Hessian or a partially completed iteration.

## Developer entry points

`mandacaru-core` has no Python dependency. Its `math` module owns grids, FFTs,
reciprocal operators and eigensolvers. `models` owns canonical functional
definitions. Molecular, periodic and quantum solvers use those shared layers.
`mandacaru-python` validates arrays and transfers ownership at the PyO3 boundary.

Periodic setup sends small JSON metadata and typed NumPy buffers. Arrays are
copied once into checked Rust allocations before detached work; no Python
numerical lists are retained. Returned snapshots own independent arrays.
See [architectural checks](../development/architecture.md).
