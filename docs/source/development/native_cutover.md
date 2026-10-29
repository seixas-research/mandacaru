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
the current calculator, lazy Hamiltonian and array-transfer contracts.

## Coverage audit

`test/migration/audit_native.py` inventories Python numerical call sites.
`test/migration/native_cutover/audit.json` is the generated review artifact.
Array conversions are excluded from its candidate list. The audit does not
infer parity from a module name or assume that every NumPy call is an algorithm.

| Area | Current native coverage | Work before removing numerical Python |
| --- | --- | --- |
| Molecular mean field | RHF/UHF/GHF, all six all-electron molecular XC choices, semilocal molecular PAW Kohn–Sham, stationary all-electron forces and real/radial bases | PAW screened exchange, multi-shell core exchange and PAW response |
| Periodic calculations | LDA/PBE iterations, spectra and Ewald primitives | Geometry assembly, advanced XC and electronic response |
| Variational quantum methods | UCCSD/HVA VQE, fermionic/qubit/QEB/CEO/spin-orbit ADAPT, shared subspaces, sequential deflation, MCAS-VQE, SDK execution | GHF-to-quantum handoff and learned proposals |
| Molecular PAW | Native quadratures, overlap/compensation augmentation, signed one-center factors, HF/semilocal KS/variational methods; empty/closed-1s cores | Basis recipe preparation, multi-shell cores, screened exchange, spin-orbit terms and forces |
| Radial PAW spectrum | Generalized separable eigenpairs, overlap normalization, confined and free channel spectra | Remaining dataset construction and basis recipes |
| Correlation | Closed-shell MP2/FNO, local pair domains and PNOs, coupled local MP2 amplitudes, symmetry-preserving active spaces | Open-shell correlation and correlating-pair constraints |
| Molecular result fields | Native natural orbitals, charge/reference/difference densities, orbital sampling and cube/XSF output; PAW augmentation charge accounting | Periodic field views |
| Dynamics | Propagation, checkpoint preparation and QPE | Echo and response orchestration |
| Persistence and providers | JSON snapshots, circuit streams, grouping and field writers | Columnar codecs, dataset codecs and job lifecycle integration |

## Functional implementation contract

The currently exposed XC choices are `lda`, `pbe`, `r2scan`, `hse06`,
`pbe+rvv10`, and `r2scan+rvv10`. Point evaluations must return the energy density
and derivatives with respect to spin densities, three gradient contractions
and two kinetic-energy densities. A meta-GGA cannot be enabled in an SCF by
substituting a fixed kinetic-energy density. A screened hybrid also needs its
short-range exact-exchange operator.

Nonlocal correlation must use the same discrete kernel in its energy and
potential. Dispersion needs charge conservation, the complete coordination and
charge response, consistent pair/three-body damping and converged periodic
image handling. Approximate replacements must not be labeled as the requested
functional or dispersion model.

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

## Remaining cutover work

The following public paths from the Python implementation are still unavailable
or disconnected from their native replacements:

| Public functionality | Remaining gap |
| --- | --- |
| Active spaces | Correlating-pair constraints and open-shell MP2 |
| Variational methods | Optional Gaussian-process and pairwise-ranker proposal models; subspace checkpoint continuation |
| Molecular PAW | Nonlinear one-center XC, augmented screened exchange, multi-shell frozen-core exchange, spin-orbit projectors and force response |
| Dispersion and advanced periodic DFT | Complete D4, periodic meta-GGA/nonlocal XC, mixed-precision controls and spin-orbit band calculations |
| Periodic quantum calculations | Bloch VQE/ADAPT, quasiparticle weights, interacting spectral functions and quantum unit-cell relaxation |
| Electronic response | Wannier construction, constrained RPA and downfolding; magnetic-moment accessors also need reconnection |
| Circuit execution | Multi-reference SDK execution and continuation of device runs |
| Geometry workflows | Variational forces and wavefunction transfer between nuclear steps, blocking the relaxation and MD examples |
| Monitoring and analysis | Complete live SCF logging and migration of older standalone analysis interfaces |

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
encodings and checkpoint continuation. PAW spin-orbit terms and the GHF
model-to-quantum handoff remain separate unfinished paths.

Molecular PAW HF and variational paths cover empty cores and single closed
1s cores. Molecular semilocal PAW Kohn–Sham uses the frozen atomic one-center
model, compensation Hartree terms, smooth-core XC and the atomic energy offset.
Its signed compensation Fock contribution and LDA/PBE/r²SCAN potentials pass
energy-derivative checks. Multi-shell core exchange, spin-orbit projectors,
PAW screened exchange and forces remain incomplete. Variational forces and
advanced periodic methods also remain incomplete. All
examples are retained; `scripts/check_examples.py` reports failures instead of
filtering them out. The full scientific test suite still has collection errors
from removed modules. Passing focused recovery tests does not resolve those
errors or establish repository-wide parity.

Fixed VQE supports Hamiltonian-group evolution and shared-angle first/second
order product formulas. Sequential deflation includes overlap penalties and
reports bare physical energies. SLSQP (unconstrained), BFGS, L-BFGS, CG, COBYLA
(unconstrained), Nelder–Mead and seeded SPSA execute in Rust. The classical
optimizer comparison, HVA and deflation examples exercise these paths.

MCAS-VQE performs bounded insertion, deletion, replacement and swap edits in
Rust. Proposal masses include every edit leading to the same architecture.
Native parameter relaxation, gradient softmax, geometric temperature schedules,
length penalties, reproducible streams and recorded insertion screens are
integrated. Tests check detailed balance, the independent ground-state limit,
the separation of best cost from best energy, and checkpoint/state consistency.
A completed chain does not certify eigenstate convergence: `result.completed`
and `result.converged` report those distinct facts.

VALQA graph proposals train and score in Rust. The term/qubit factor graph has
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
particular, radial LDA/PBE helpers, PAW/noncollinear grid composition,
PAW construction, advanced ansatz families,
response and legacy standalone numerical modules still exist. Their numerical
work must move before deletion. The public calculator has one Rust execution
route and no execution selector. JAX is no longer a package dependency.

Native D4 needs a complete, provenance-checked reference dataset and all model
responses; it is not enabled. No external implementation source or parameter
table has been copied into the new XC kernels. A complete independent catalog
of nonlocal density functionals also needs its own kernel definitions and
scientific validation; rVV10 coverage must not be described as that catalog.

## Dispersion foundations

The independent Rust dispersion module now provides damped pair energies with
coefficient and radial response, neutral-reference three-body energies with
all nine partials, and constrained Gaussian charge equilibration with analytic
coordinate response. An optional electronegativity gradient carries the
coordination/environment response through the constrained solve. The solver
requires positive curvature on the charge-conserving subspace.

These primitives are tested against coefficient/coordinate finite differences,
charge conservation, translation invariance and long-range/sign limits. They
are not a complete parameterized D4 calculator and are not substituted for one.
Reference data, coordination interpolation, dynamic-polarizability integration,
model-specific charge scaling and periodic sums remain to implement and verify.
