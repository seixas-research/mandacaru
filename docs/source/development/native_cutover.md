# Native computational architecture

The target architecture has one Rust computational engine. Python supplies the
public interface, array conversion, ASE integration, visualization and external
service adapters. Numerical routines have one implementation, with no Python
fallback and no alternative-backend aliases.

The cutover preserves scientific behavior through independent reference cases,
analytic limits and derivative checks. Removing an old implementation requires
identifying every caller and moving its numerical work first; an unsupported
feature must never silently select an unrelated calculation.

## Coverage audit

`test/migration/audit_native.py` inventories Python numerical call sites.
`test/migration/native_cutover/audit.json` is the generated review artifact.
Array conversions are excluded from its candidate list. The audit does not
infer parity from a module name or assume that every NumPy call is an algorithm.

| Area | Current native coverage | Work before removing numerical Python |
| --- | --- | --- |
| Molecular mean field | RHF/UHF/GHF, all six molecular XC choices, stationary forces and real/radial bases | PAW assembly and response |
| Periodic calculations | LDA/PBE iterations, spectra and Ewald primitives | Geometry assembly, advanced XC and electronic response |
| Variational quantum methods | UCCSD VQE, fermionic ADAPT and shared-subspace searches | Other ansatz/pool families, deflation and learned proposals |
| Correlation | Real closed-shell MP2/FNO | Open-shell/local correlation and active-space selection |
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

## Remaining cutover work

This change does not complete the repository-wide Rust-only cutover. In
particular, radial LDA/PBE helpers, PAW/noncollinear grid composition,
PAW construction, advanced ansatz families,
response and the Python execution route still exist. Their numerical work must
move before deletion. JAX also remains a dependency of other unported features.

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
