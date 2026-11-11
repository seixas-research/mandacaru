# Native computational architecture

The target architecture has one Rust computational engine. Python supplies the
public interface, array conversion, ASE integration, visualization and external
service adapters. Numerical routines have one implementation, with no Python
fallback and no alternative-backend aliases.

Scientific behavior must be preserved through independent reference cases,
analytic limits, derivative checks and the existing ASE examples. The earlier
cutover removed public functionality before replacement paths were complete;
recovery is still in progress. Removing an old implementation requires
identifying every caller and moving its numerical work first; an unsupported
feature must never silently select an unrelated calculation.

See [the implemented ownership and dependency boundaries](architecture.md) for
the current calculator, lazy Hamiltonian and array-transfer contracts, and
[the history](history.md) for how each phase of the migration closed. This page is
the single live status page; features that were never ported are collected in
[the not-yet-available table](#not-yet-available).

## Coverage audit

`scripts/numerics_budget.py` counts, for every module of `src/mandacaru`, the
Python that still does numerical work: NumPy and SciPy calls outside array
hand-off, `@`, `math` calls, builtin reductions, loops and comprehensions over
data, dynamic access to numerical modules and imports outside the allowlist. It
reads the syntax tree, so docstrings, comments and strings score zero and an
alias, a from-import or a function-local import is still counted.
`test/architecture/python_numerics_budget.toml` is the allowlist: one row per
module with its measured counts, and a wave and a `dies` date for every row
that still holds numerics. A row can only be lowered; a module without a row in
the zero tier (`pseudopotentials`, `basis`, `integrals`, `models`, the PAW
integrations, `core/symmetry.py`, `core/spin_orbit.py`) is budgeted at zero.
`python scripts/check_architecture.py` runs the comparison and warns 30 days
before a `dies` date, and `python scripts/numerics_budget.py --ratchet` records
a port by lowering the row. The coverage table below says which native routine
replaces which numerical Python; it does not infer parity from a module name.

| Area | Current native coverage | Work before removing numerical Python |
| --- | --- | --- |
| Molecular mean field | RHF/UHF/GHF, all six all-electron molecular XC choices, PAW Kohn–Sham (including HSE06 screened exchange, one-center terms and, for open shells, the spin-scaling remainder of the spheres' short-range semilocal exchange), stationary all-electron forces, stationary PAW-LCAO RHF/UHF/KS forces (including HSE06), variational molecular PAW forces, Fermi–Dirac occupations of a degenerate frontier shell and real/radial bases | Multi-shell core exchange for Hartree–Fock |
| Periodic calculations | LDA/PBE/meta-GGA/HSE06 iterations (a hybrid crystal carries the compensation charges and the one-center screened terms of its PAW spheres, read from the dataset files recorded with their SHA-256 in the snapshot's `sites` block; the exchange operator is part of the mixed state), two-component LDA/PBE/r²SCAN/rVV10/HSE06 spinor SCF and bands, spectra, Ewald primitives, the crystal build itself (the setup reduces the k mesh by the space group, builds the basis and the PAW spheres, and writes the `setup` and `sites` blocks of a saved crystal, all in Rust), analytic forces and stress, DFT+U (Dudarev and Liechtenstein forms, linear-response U, a symmetry-reduced k mesh whose occupation matrices are completed over the kept operations, and a collinear magnetic crystal that keeps the operations preserving its moments; [DFT+U](../guide/dft_u.md)) and native finite-size corrections (KZK, MPC, CCMH) | Electronic response and spinor DFT+U. A hybrid (HSE06) runs on the whole k mesh: its exact exchange sums over the listed points, so the setup never reduces a hybrid crystal's mesh by symmetry or time reversal. Periodic forces and stress are analytic (Rust) for LDA, PBE, r²SCAN, rVV10 and HSE06, unpolarized, spin-polarized and with spin-orbit coupling ([Forces](../guide/forces_relaxation.md)). The analytic gradient and stress have no compensation or one-center hybrid term. The forces and stress of a hybrid or spinor DFT+U crystal, the stress of a Tkatchenko–Scheffler crystal and `float32` forces, stress and DFT+U raise. Crystal HSE06 energies and gaps are not validated against an all-electron reference; a magnetic spinor hybrid and a hybrid snapshot without its `sites` block raise; `resources` threads do not yet bound the Bloch kernels |
| Variational quantum methods | UCCSD/HVA VQE, fermionic/qubit/QEB/CEO/spin-orbit ADAPT, shared subspaces, sequential deflation, VASQA/VALSQA, ranker and graph proposals, Z2 tapering, the dense or double-factorized two-body representation (`two_body=`, with its truncation in eV and a run log that states the bound; [Scalability](../guide/scalability.md)), analytic variational forces and SDK execution | GHF-to-quantum handoff; on a double-factorized model the analytic forces, MP2 natural orbitals and the factorized measurement raise |
| Molecular PAW | Native quadratures, overlap/compensation augmentation, signed one-center factors, HF/semilocal KS/variational methods and their forces (multi-shell frozen cores included in the force contraction); molecular DFT+U (Dudarev and Liechtenstein, restricted and spin-polarized) with its energy and analytic forces; empty/closed-1s cores, HSE06 screened exchange; spinor Kohn–Sham with the PAW spin-orbit projector term on a spin-orbit dataset (LDA, PBE, PBE+rVV10, r²SCAN, HSE06) and its analytic forces | Multi-shell core exchange for Hartree–Fock and spinor DFT+U |
| Radial PAW spectrum | Generalized separable eigenpairs, overlap normalization, confined and free channel spectra, and the basis recipe that molecules and crystals share | Remaining dataset construction ([not yet available](../reference/not_yet_available.md)) and the regeneration of the shipped library, which needs the owner's approval |
| Correlation | Closed-shell MP2/FNO, local pair domains and PNOs, coupled local MP2 amplitudes, symmetry-preserving active spaces, open-shell MP2 ranking on a UHF reference and correlating-pair constraints | Open-shell local (`dlpno-mp2`) correlation; correlating pairs together with symmetry-preserving selection |
| Molecular result fields | Native natural orbitals, charge/reference/difference densities, orbital sampling and cube/XSF output; PAW augmentation charge accounting | Periodic field views |
| Dynamics | Propagation, checkpoint preparation, QPE, the echo correlation loop, the nested OTOC and the spectrum of an echo series (Python keeps the validation, the result views and the circuit builders) | Response orchestration |
| Wannier functions and cRPA | `calc.wannier(...).downfold(functions, screening=None/"rpa"/"crpa", coulomb="supercell"/"isolated")`: the bare or statically screened Coulomb integrals of a fragment with the PAW compensation and one-center terms, the long-wavelength head, the double counting, and a natural-orbital hand-off to `adapt-vqe` ([Wannier downfolding](../guide/wannier_downfolding.md)) | The transition loop of the screening is not threaded; a spinor crystal raises |
| Periodic quantum methods | `vqe`, `adapt-vqe`, `vasqa` and `valsqa` solve a crystal's Born–von Karman supercell, with the Lehmann spectral function at the mesh points and along a band path, the unfolded mean-field bands, the tetrahedron density of states, the symmetry and irreducible zone of the mesh and band paths in the conventions of the guide ([crystals with the quantum methods](../guide/bloch.md)) | The per-k register, a PAW-LCAO supercell, analytic forces and stress; see the remaining work below |
| Persistence and providers | JSON and Parquet snapshots (Parquet container, v1 and v2 data pages, dictionary pages, UNCOMPRESSED, Snappy and ZSTD decoding, ZSTD encoding in `mandacaru-core`; no Python Parquet package), pseudopotential Parquet datasets, circuit streams, grouping and field writers | Dataset codecs other than Parquet and job lifecycle integration; the shipped pseudopotential library was not regenerated with the native writer ([Parquet in Rust](#cutover-parquet)) |

## Functional implementation contract

The currently exposed XC choices are `lda`, `pbe`, `r2scan`, `hse06`,
`pbe+rvv10`, and `r2scan+rvv10`. Point evaluations must return the energy density
and derivatives with respect to spin densities, three gradient contractions
and two kinetic-energy densities. A meta-GGA cannot be enabled in an SCF by
substituting a fixed kinetic-energy density. A molecular meta-GGA also takes
its density gradient from the orbitals (`2 sum D phi grad phi`, the partial
core by the grid), the same gradients that build tau, so that a one-orbital
density is exactly iso-orbital; its Fock matrix and analytic force carry the
matching flux term and the divergence of the flux is not subtracted from the
potential. A screened hybrid also needs its
short-range exact-exchange operator.

Nonlocal correlation must use the same discrete kernel in its energy and
potential. Approximate replacements must not be labeled as the requested
functional.

Numerical implementations are written independently from scientific equations.
Scientific references and any notices required for reused material remain
part of the provenance. Existing implementations are removed after their native
replacements pass the relevant scientific and integration checks.

## Completed native XC work

The native spin point API covers LDA, PBE, r²SCAN and the HSE06 semilocal
contribution. Rust computes forward analytic partials of its own expressions.
HSE06 screening uses exact erfc quadrature. Molecular SCF includes the
short-range exact-exchange matrix and matching gradient action. r²SCAN SCF
includes kinetic densities, their matrix terms and their gradient action.

The rVV10 implementation owns its cardinal interpolation, analytic reciprocal
kernel, symmetry-averaged aliases, convolution energy and variational partials.
Both molecular pairings are integrated in SCF and stationary gradients.

The Python r²SCAN formula module has been deleted. Spin-XC, exchange-hole and
nonlocal-correlation modules are array/result adapters to Rust; their former
JAX, NumPy and SciPy numerical implementations have been removed. Canonical
XC names are required in both native and Python dispatch.

Validation includes frozen spin-point references, independent exact quadrature,
finite differences of seven partials, spin interchange, periodic/isolated
nonlocal field response, memory-budget failures and stationary molecular
forces against total-energy displacements. The original migration references
remain unchanged.

Product-ansatz energy gradients use a native adjoint sweep through the unitary
operators. Tests compare these gradients with forward derivatives and finite
differences for noncommuting generators. ADAPT retains the best parameters when
an inner optimizer fails and continues growth within the requested budget;
only its outer convergence criteria determine success. Inner failures remain
available through `result.optimizer_failures`. Checkpoint serialization
preserves small Pauli coefficients instead of applying a simplification cutoff.

Closed-shell MP2 active-space ranking now executes in Rust using rectangular
Coulomb-factor contractions. Energy selection, occupation thresholds and
explicit frozen cores construct the reduced model without a full source-MO
tensor. Rotations remain native-owned and are applied to density fields for
single-state and shared-subspace results. Local MP2 owns Boys localization,
PAO domains, PNO truncation and coupled amplitude equations. Its full-domain,
zero-cutoff limit reproduces canonical MP2 energies and densities. Molecular
symmetry selection retains whole orbital multiplets. Open-shell MP2 selection
and correlating-pair constraints remain unported.

## Molecular geometry response

Molecular variational forces execute in Rust and are analytic (PAW-LCAO only):
the state's spin-summed reduced density matrices, relaxed for the response of the
canonical Hartree-Fock orbitals (CPHF), the MP2 natural orbitals and the MP2
amplitudes, are contracted once with the derivative integrals. The
finite-difference route they replaced (displaced SCF, selection and fixed-pivot
Hamiltonian) survives only as a test oracle, which the analytic force matches on
every component. The references covered are closed-shell and unrestricted
Hartree–Fock, the latter with its open-shell MP2 natural orbitals; a Kohn–Sham,
spinor or fractional-occupation reference, a `dlpno-mp2` active space and a tapered
register raise. On a circuit device the force contracts the density matrices measured
on the device, measures the parameter Hessian for the residual-gradient (Z-vector)
correction, and reports a first-order statistical error that includes the noise of
the Hessian's energies; a curvature within five of its standard errors of zero raises
by name, and the measurement budget refuses the extra circuits before any job
([the forces guide](../guide/forces_relaxation.md)). A retained PAW setup rebuilds its
geometry without reloading datasets.

Stationary PAW-LCAO forces (RHF, UHF and semilocal Kohn–Sham, LDA/PBE/r²SCAN)
use that rebuild without a displaced SCF. Every matrix of the engine lives in
the Löwdin-orthonormal frame of its own geometry, so the orthonormality
constraint does not depend on the positions and, at a converged SCF, the force
is the partial derivative of the energy at fixed orthonormal-frame densities:
the overlap (Pulay) term is carried by the position dependence of the Löwdin
matrix inside the rebuilt integrals. The derivative is a Richardson difference
of the total energy (electronic plus ion–ion) on the same fixed grid, at the
documented step `paw_gradient::STEP = 1e-3` Bohr, with the error estimate
`|fine - coarse| / 3` reported as `step_error_estimate`; Hartree–Fock keeps the
reference Coulomb pivots. The same finite difference differentiates the basis
and projector centers, the overlap and compensation augmentation, the
compensation charges and core densities on the fixed grid, the one-center
Coulomb and exchange–correlation terms, closed-1s core exchange, the local
potential and the kinetic energy at once, so GGA and open-shell states need no
separate term. It includes the screened exact exchange of HSE06 (the energy is differenced with the
hybrid operator rebuilt at each displaced geometry). It raises for GHF,
`float32` samples, unconverged states and a state that does not
reproduce its energy on these integrals.

The force of a molecule with `hubbard_u` carries the DFT+U term, the derivative of the
+U energy through the shell projections of the moving atom (`hubbard/projection`), and
matches a central difference of the +U energy for the Dudarev and Liechtenstein forms,
restricted and spin-polarized; a Hartree-Fock gradient of such a molecule raises.

The ASE adapter supports free-molecule internal-direction projection and
records its force convention in the text output. Unprojected Cartesian
response remains available from the result. Geometry transfer matches occupied
and virtual orbital domains, carries generator signs and parameters, and
reoptimizes the seed against the new Hamiltonian before deciding whether to
retain it. Seed relaxation is reported separately from ansatz growth. A small
immutable handover owns only the source basis, active orbital
coefficients, operator labels and parameters. The previous integral cache and
quantum workspace are released before constructing the next calculation unless
the caller retains its previous result. A budget regression distinguishes these
lifetimes explicitly.

The public optimizer default is SLSQP with 2000 iterations and tolerance
`1e-12`, restoring the original calculator preset. With `shots > 0` the default is
instead the strict two-evaluation form of SPSA (`track_best=False`, maxiter 100, so
`2 * maxiter + 1` energy evaluations, each one job on a processor), and SLSQP, BFGS,
L-BFGS and CG are refused there because they difference a sampled energy at tiny
steps. Default ADAPT stopping
criteria are a maximum gradient of `1e-3` Ha and an energy change of `1e-3` eV;
an explicitly supplied criterion dictionary enables only its specified keys.
User-supplied budgets and tolerances are preserved.

Molecular `precision="float32"` retains sampled orbitals in native `float32`
buffers. One-body matrices, integral accumulation and FFT workspaces remain in
`float64`. Temporary accumulation views share the calculation budget and
release their reservations on exit; displaced molecular setups retain the
selected storage precision. The PBE/DZP water comparison differs by less than
one micro-eV between the two storage choices. Periodic complex orbital samples
also support compact `complex64` storage;
Bloch matrices and SCF accumulation stay in `float64`. Input snapshots
retain the selected storage precision.

The periodic r²SCAN code takes its density gradient as the spectral derivative of
the sampled density, while the molecular code builds it from the orbitals by the
product rule (the 2026-10 repair). It was checked on 2026-11-10. The spectral
gradient keeps a one-orbital density iso-orbital to 1e-5 of τ (H₂ in a box, 8e-6
Ha of r²SCAN energy at a 0.4 Bohr grid), so the molecular defect is absent. On
water the two gradients differ by 3e-4 Ha at the default grid, a discretization
difference that neither side wins against a finer grid
([the DFT guide](../guide/dft.md)).

Periodic meta-GGA SCF computes kinetic densities from the occupied complex
Bloch orbitals and includes the matching kinetic-potential matrix. Both density
and kinetic density participate in mixing and convergence. Reciprocal symbols
average lattice-equivalent aliases, preserving the distinction between the
mean vector and mean squared wave number at grid faces. Native complex
projections preserve imaginary coherence and use bounded real matrix products.
The rVV10 energy and potential use one discrete periodic kernel. Derivative,
operator-contraction, converged r²SCAN and input-roundtrip tests cover these
paths; the original MoS₂ r²SCAN+rVV10 example completes through its band path and DOS.

Periodic LDA/PBE spinors retain charge and three magnetization components.
Rust diagonalizes the two-component generalized Bloch Hamiltonian with the
separable spin-orbit operator and uses unit band degeneracy. Exchange and
correlation rotate the local density to its eigenframe and rotate the
functional derivative back, retaining transverse spin coherence. Unrestricted spinors use the full
k-point mesh; scalar symmetry averaging is disabled for axial-vector fields.
Nonmagnetic spinors can impose time reversal: charge and scalar PAW moments
use the time-reversal-reduced mesh, while magnetization remains zero. Scalar
point-group averaging is disabled for both spinor modes. Spinor bands still
have unit degeneracy and retain the complete SOC matrix. Off-mesh bands, DOS and projection weights retain both spinor blocks.
The L.S angular matrices and radial-projector contractions also execute in Rust.
Tests check exact angular spectra, all four XC derivatives, the density/operator
contraction, the scalar zero-SOC limit, and the free-energy derivative with
respect to SOC strength. The original Bi₂Te₃ example completes both its
scalar-relativistic and spinor band paths and saves their comparison plot.

Large periodic sample caches, coefficient matrices, density matrices and mixing
histories can use owned Rust scratch arrays. A read charges its accumulation
buffer to the same calculation budget and releases it after use. Scratch files
use exclusive creation, private permissions and owner-based cleanup; damaged
records surface an error. Small arrays remain resident. Tests compare streamed
and resident SCF energies and densities, verify projected-DOS state counts,
and check allocation failures and file cleanup. Ordinary ASE energy requests
transfer scalar results, spectra and fields; dense coefficient and density
matrices are copied only when explicitly requested from the native snapshot.
The crystal is built entirely by the Rust setup: Python sends one request (the
lattice, the species and the options), and the setup reduces the k mesh by the space
group, builds the basis and the PAW spheres, and writes the `setup` block that a
saved crystal carries. An option it cannot build raises by name; no other route
builds the crystal. Compensation form factors use the scratch storage and are read by
channel, keeping their cache out of the Bloch projection workspace. The dense
compensation and ion coupling sum runs over bounded reciprocal-vector blocks and
never materializes the full reciprocal lattice or channel-by-vector intermediates.
The request once sent ASE's cell rows where Rust reads columns, so the lattice
arrived transposed and every non-symmetric cell was wrong (LiH by
10.8 meV, with forces off their own energy derivative); it now sends the
transpose.

Molecular PAW spherical contractions use bounded matrix batches with the same
quadrature nodes and weights. Shared angular mathematics evaluates Cartesian
harmonics without recovering polar angles. Tests compare the Cartesian and
polar forms, exact shell normalization, direct signed quadratures and force
responses. Large real-grid projections partition their tiles among the owning
context's workers. Workspace size also limits that partition; each worker's
BLAS call remains single-threaded and partial matrices are reduced in a fixed
order. Thread, memory and independent direct-contraction checks cover this path.

(cutover-parquet)=
## Parquet in Rust

`fastparquet`, `pandas` and `pyarrow` are no longer declared or imported. The
container, the Thrift footer, PLAIN and dictionary decoding, Snappy, the CRC32
and the writer are hand-written in `mandacaru-core::parquet`; the one third-party
crate, `ruzstd` (pinned, declared once in the workspace `Cargo.toml`), is used only
in `parquet/codec/zstd.rs`. Python makes one native call per file. User-visible
behavior changes of the cutover:

- `engine=` and `codec=` are removed with no alias (`TypeError`); `compression=`
  accepts `zstd`, `uncompressed` and `none`.
- A null in a Parquet file raises `ValueError` (the old reader returned NaN). A
  NaN in a PAW table is written as a real NaN, not as a null.
- Format version 0 and a missing format, version or qubit count raise; a missing
  mapping defaults to `jordan_wigner`. The key/value JSON text is compact.
- Parquet pseudopotential saves are atomic. GZIP, Brotli and LZ4 files raise with
  a hint to re-save with zstd.
- The reader takes v1 and v2 data pages, dictionary pages and UNCOMPRESSED,
  Snappy and ZSTD. New files use REQUIRED columns, PLAIN encoding and zstd at its
  fastest level (the only one ruzstd 0.8.1 implements), so they can be larger
  than files written with libzstd at level 3.
- The shipped pseudopotential library was not regenerated; regenerating it needs
  the owner's approval.

The user-facing description is in [Hamiltonian caching](../guide/hamiltonian_cache.md)
and [Parquet files](#native-parquet).

## Numerical safeguards and run control

**Eigensolvers.** The tridiagonal QR iteration in `math/linalg.rs` can return an
orthonormal but inaccurate decomposition of a degenerate or clustered spectrum; one
such result had made a PAW energy wrong by 2.2 Ha for H₂ displaced by 1e-3 Bohr. Both
the real symmetric `real_eigen` and the complex Hermitian `eigen` (every SCF Fock
matrix, periodic and spinor Hamiltonians) now check the result and fall back to cyclic
Jacobi rotations. Acceptance is relative: the residual divided by the matrix norm, and
the departure from orthogonality, must both lie below `1e-13 max(n, 10)`. The
regression is `crates/mandacaru-core/tests/linalg.rs`.

**Kohn–Sham level shift, as a retry only.** A molecular Kohn–Sham start that does
not converge with plain DIIS is rerun with a 0.5 Ha level shift, engaged at the first
rise in energy. This damps the flip of a degenerate open-shell electron (a diatomic
radical's π electron). The shift is not applied to a start that converges without
it, because an always-on shift can settle into a different stationary point than the
plain iteration.

**Run budget and shot-based runs.** SPSA has a strict form and is the default under
`shots > 0` (see the optimizer paragraph above). A pre-flight refuses an optimization
whose evaluation bound cannot fit the run budget (Nelder–Mead is left to the ledger;
COBYLA is bounded by `maxiter` unless `maxfev` is given). When the budget ends a
solve, the executor's stop message ends it at the best point reached: the result
carries `budget_stopped`, the calculator sets `run_budget_exhausted` and issues a
`RuntimeWarning`. A stop before any energy has been measured raises
`RunBudgetExceeded`. `execute_circuits` defaults to true when a non-Qiskit provider
is named.

**Validation of options.** The classical methods (`rhf`, `uhf`, `ghf`, `dft`) refuse
`shots`, `execute_circuits`, `backend_provider`, `device`, `backend_options` and
`measurement_budget`, since there is no quantum state to measure. The molecular PAW
path warns when the dataset was generated for a different functional. After
`results` is cleared, `get_forces()` reruns the calculation instead of serving a
cached result.

## Open findings

Recorded in the maintainers' session logs.

- **Degenerate frontier shells: fixed on 2026-11-10.** Restricted closed-shell O₂
  on its exact axis used to converge to an integer state far above the triplet. A
  molecular Kohn–Sham run now fills each spin channel with Fermi–Dirac occupations,
  one Fermi level per spin, at 1e-4 Ha (`smearing=` sets the width; Fermi–Dirac
  only), annealed from 3e-3 Ha when the direct solve stalls, and levels within 1e-6
  Ha share one occupation. Restricted O₂ is 0.778 eV above the triplet, the number
  the crystal path gives. `energy` is the energy, `free_energy` the Mermin free
  energy, and the PAW forces are the derivative of the latter. Hartree–Fock keeps
  its integer occupations. The all-electron analytic force and an active space raise
  on a fractional state, so the quantum handover never receives one.
- **`frozen=[...]` that splits a degenerate block.** Which of the equal-occupation
  orbitals are frozen is decided by round-off, so the active-space Hamiltonian is
  not reproducible. Proposed: order orbitals within equal-occupation blocks by Fock
  energy, or reject a list that splits a block. (Not rerun for this page.)
- **Run log.** The calculator writes `[PERFORMANCE]`, the thirteen-field
  `[ELECTRONS]` block, `[SCF SETUP]` and `[SCF ITERATIONS]`, the termination reason,
  the lineage of a resumed run, the resolved `[BASIS]` and the initial magnetic
  moments again, each value read from a native result and formatted in Python. What
  the code still does not write is listed in
  [the run log reference](../reference/run_log.md): the `[MEASUREMENT]` and echo
  helper blocks, the screening lines, a `time` column for the SCF
  iterations and per-stage native clocks (the native solve is one lumped stage,
  `parameter optimization`), and a crystal's `[BASIS]`, which states only the
  precision (a crystal does write `[ELECTRONS]`, `[SCF SETUP]` and `[SCF ITERATIONS]`).
  The log is written when the run ends, not while it runs.
  `ansatz="hva"` on a cached Hamiltonian is rejected only at `.run()`. (The last item
  was not rerun for this page.)
- **Tests.** The collection errors of the older suite were reduced from 109 to 2 by
  deleting test files whose subject modules no longer exist; two remaining modules
  have dangling imports and no native counterpart. A full run on 2026-10-29 recorded
  3,174 passed, 291 failed, 62 errors, with the failures concentrated in
  untriaged files that probably still test the pre-cutover interface. These counts
  are the session log's and were not remeasured here.

## Remaining cutover work

The following public paths from the Python implementation are still unavailable
or disconnected from their native replacements:

| Public functionality | Remaining gap |
| --- | --- |
| Molecular PAW | Nonlinear one-center XC, multi-shell frozen-core exchange for Hartree–Fock, spinor DFT+U; the r²SCAN spin-orbit force gate on the Au-Be dimer is open because the local-frame meta-GGA is sharp near magnetization nodes |
| Advanced periodic DFT | Spinor DFT+U in the periodic SCF, DFT+U with a hybrid, and a hybrid or spin-orbit SCF on a symmetry-reduced k mesh (both run on the whole mesh) |
| Periodic quantum calculations | The supercell Hamiltonian, the Lehmann spectral function at the mesh points and along a band path, the crystal symmetry and irreducible zone of the mesh, the unfolded mean-field bands and the tetrahedron DOS are native ([the crystal guide](../guide/bloch.md)); missing are the per-k register (`crystal_orbitals="k-space"`), a PAW-LCAO supercell, analytic crystal forces for the quantum methods (they are finite differences) and their stress (refused), hence quantum unit-cell relaxation |
| Circuit execution | Multi-reference SDK execution and continuation of device runs |
| Geometry workflows | Remaining ethanol relaxation verification |
| Monitoring and analysis | Live SCF logging while the run proceeds (the iteration history is logged when the run ends, without a clock) and migration of older standalone analysis interfaces |

Some helper modules and native primitives for these features remain in the
tree. Their presence does not make the ASE calculation path operational.
The example runner reports script completion; solver convergence must also be
checked through the result and its warnings. This inventory does not describe
the already restored paths as missing: all-electron HF/DFT, molecular PAW HF
within its supported core partitions, closed-shell MP2/FNO selection, standard
VQE/ADAPT and subspace searches, basic periodic LDA/PBE, molecular fields,
Hamiltonian/checkpoint persistence, QPE and quantum echoes have recovery checks.

The spin-orbit ADAPT pool now includes both `T - T†` and `i(T + T†)` for
spin-changing singles and doubles. It conserves total particle number and
uses an untapered register. Tests cover complex spin mixing, three fermion
encodings and checkpoint continuation. The GHF model-to-quantum handoff remains
a separate unfinished path.

Molecular PAW HF and variational paths cover empty cores and single closed
1s cores. Molecular semilocal PAW Kohn–Sham uses the frozen atomic one-center
model, compensation Hartree terms, smooth-core XC and the atomic energy offset.
Its signed compensation Fock contribution and LDA/PBE/r²SCAN potentials pass
energy-derivative checks. Multi-shell core exchange for Hartree–Fock remains
incomplete. All
examples are retained; `scripts/check_examples.py` reports failures instead of
filtering them out. The full scientific test suite does not yet pass in full (see
the open findings above). Passing focused recovery tests does not establish
repository-wide parity.

Fixed VQE supports Hamiltonian-group evolution and shared-angle first/second
order product formulas. Sequential deflation includes overlap penalties and
reports bare physical energies. SLSQP (unconstrained), BFGS, L-BFGS, CG, COBYLA
(unconstrained), Nelder–Mead and seeded SPSA execute in Rust. The classical
optimizer comparison, HVA and deflation examples exercise these paths.

VASQA performs bounded insertion, deletion, replacement and swap edits in
Rust. Proposal masses include every edit leading to the same architecture.
Native parameter relaxation, gradient softmax, geometric temperature schedules,
length penalties, reproducible streams and recorded insertion screens are
integrated. Tests check detailed balance, the independent ground-state limit,
the separation of best cost from best energy, and checkpoint/state consistency.
A completed chain does not certify eigenstate convergence: `result.completed`
and `result.converged` report those distinct facts.

VALSQA graph proposals train and score in Rust. The term/qubit factor graph has
two message-passing layers, a state-conditioned scoring head and a within-state
first-choice loss. A shared reverse-derivative tape reuses graph embeddings
within each minibatch; Adam and normalization statistics also run natively.
`mandacaru.analysis.proposals.fit` reads the recorded observations and writes
the native model archive and held-out assessment. No Python numerical trainer
or scorer is used. Readiness requires at least three informative held-out
groups, a mean within-state Spearman improvement of 0.05 over the gradient
baseline, and wins in a strict majority of those groups. Insufficient or
uncompetitive observations produce an explicitly unready saved model. The
search uses learned scores only when ready, mixed with a positive gradient
share to preserve reverse support. A model stays fixed throughout a chain.
`result.proposal_ready` and `result.proposal_reason` expose that decision.

Variational energy derivatives use the normalized Rayleigh residual with the
identity contribution separated before Hamiltonian action. This avoids losing
gradient accuracy when the total energy has a large constant offset. Near
stationarity, BFGS and L-BFGS use the approximate-Wolfe step test described by
[Hager and Zhang](https://www.math.lsu.edu/~hozhang/papers/cg_compare.pdf).
Step acceptance remains distinct from successful termination, which still
requires the requested maximum-gradient tolerance. Identity-shift tests cover
matrix-free, dense and sparse storage. The original ethanol TZP example now
finishes with all inner optimizations successful and outer ADAPT convergence.

Growth observers, expressibility sampling and structured iteration logs are
connected to the native solver. Circuit execution sends the native Pauli
rotation schedule to Qiskit, Braket or Cirq. Raw states or Pauli measurements
return to Rust for energy contraction, parameter-shift derivatives, ADAPT pool
screening and optimization. Shot ADAPT measures merged commutator observables;
its reference-determinant screening is analytic. Aer is required for IBM fake
devices and supplies their noise models. The state retained from a shot run
is identified as the ideal circuit state, rather than a measured device state.

This change does not complete the repository-wide Rust-only cutover. In
particular, the Python modules that still hold numerical work are the rows of
the numerics budget described in the coverage audit above. Their numerical
work must move before deletion. The public calculator has one Rust execution
route and no execution selector. JAX is no longer a package dependency.

No external implementation source or parameter table has been copied into
the new XC kernels. A complete independent catalog
of nonlocal density functionals also needs its own kernel definitions and
scientific validation; rVV10 coverage must not be described as that catalog.
