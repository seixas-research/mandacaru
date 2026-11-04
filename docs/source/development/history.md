# History of the migration to Rust

This page folds the migration diary into one record: what each phase decided, what it
delivered, and the validation result that closed it. It is a record, not a status
page. The coverage of the native engine today, and the work that remains, are on
[the cutover page](native_cutover.md); where this page and that one disagree, that
one is current.

The phase pages were first committed on 2026-10-27 and 2026-10-28, after the
repository history was rewritten, so commit dates do not separate the phases. They
are ordered by phase number. The frozen references, the per-phase validation reports
and the parity tests are in `test/migration/`.

## Decisions that held throughout

- **One implementation, no selector.** Numerical work moves to Rust before its Python
  is deleted. There is no Python fallback, no alternative-backend alias and no
  execution selector: during the migration an `engine` option chose between the
  routes, and it was removed once Rust was the only one. An unsupported request raises.
- **Evidence before replacement.** A ported routine must reproduce the number it
  replaces, against frozen references (`test/migration/reference`) or an independent
  check. Coarse reference grids measure parity of the program, not converged physics.
- **One entry point.** Algorithms are reached through `Mandacaru(method=...)`.
- **Hartree and Bohr inside, eV and angstrom at the ASE edge.** Unit conversion
  happens once, at the binding boundary.
- **Owned buffers.** Bindings copy validated inputs into budgeted Rust buffers and
  return independent arrays; no borrowed or writable Rust view is exposed.
- **No external tables.** Basis sets and pseudopotentials are generated natively.

## Phase 0: evidence before code

Phase 0 changed no algorithm. It built the tooling that later phases were judged
against: a static inventory of source hashes, imports, signatures and result fields;
a capability index tying features to tests; an explicit corpus of geometries,
options, seeds and tolerances; an isolated runner that records energies, integral
arrays, spectra and resource use; and a frozen cached-H2 Hamiltonian.

It also fixed the rules for reading the evidence: the inventory is static, test
counts are not coverage, a workload names a representative case rather than a whole
group, and a comparison rejects failed, skipped or partial evidence. Its one finding
about the code was a thread audit: some shell and two-center kernels ignored the
configured thread limit, so the first timing comparison was not at equal thread
counts. Unifying thread control became a Phase 1 requirement.

## Phase 1: workspace and binding foundation

Delivered the Rust workspace (`mandacaru-core`, `mandacaru-kernels`,
`mandacaru-python`), checked engine types and native wheel packaging.

- `Geometry`, `Grid`, `Context` and `Workspace` types with checked shapes, a thread
  limit and a shared byte budget per context; independent contexts run concurrently.
- Checked interfaces for one-body integrals, Laplacians, projector contractions,
  Bessel transforms, radial eigenpairs and shell evaluation.
- A single thread policy for every parallel kernel: context limit, then
  `MANDACARU_NUM_THREADS`, then `OMP_NUM_THREADS`, then the hardware.
- A wheel check that installs into an isolated environment with compiler invocation
  blocked.

Closing result: 142 binding, API and documentation tests, 8 Rust tests, and all 11
numerical-reference workloads passed. Four failures elsewhere in a broader run
reproduced on the Phase 0 source and were left visible.

## Phase 2: the cached quantum workflow

Delivered the first complete native workflow: load a cached Hamiltonian and solve it
with VQE or ADAPT-VQE entirely in Rust.

- Pauli algebra, the four fermion-to-qubit mappings (Jordan-Wigner, parity,
  Bravyi-Kitaev, reduced parity), full and sector bases, dense, compressed and
  matrix-free Hamiltonian application, UCCSD and the fermionic pool.
- Scaled Taylor exponential actions with exact forward derivatives of both the summed
  and the ordered-product ansatz; BFGS and L-BFGS with a strong-Wolfe line search.
- A 62-mode register limit, and a 64 MiB limit on JSON input.

Closing result: both standalone solvers took the cached H2 Hamiltonian from
-0.9371240086056045 Ha to -0.9507746213870719 Ha, ADAPT selecting `D(0,2->1,3)`; 210
migration, API and documentation tests and 14 Rust tests passed, with two failures
that reproduced on the Phase 0 source.

## Phase 3: the molecular workflow

Delivered ASE geometry to quantum result in Rust: grid construction, analytic basis
sampling, one- and two-electron integrals, orthogonalization, RHF, UHF and complex
spinor GHF, MO transformation, frozen cores and the Phase 2 solvers.

- HAO, STO-nG and native Pople, Dunning and Karlsruhe Gaussian recipes (generated
  recipes, not published exponent tables), driven by 80 stored Slater fits.
- Isolated and periodic Coulomb solvers, pair-packed integrals and pivoted Cholesky
  factors.
- Energy-ordered or explicit active spaces, and schema-v1 JSON Hamiltonian caches.

Closing result: the original H2, H, GHF H2 and STO-3G water energies agreed with the
frozen Phase 0 corpus.

## Phase 4: atoms, density functional theory and forces

Delivered numerical atomic orbitals (NAO and NAO-AE), a nonrelativistic spherical LDA
atom solver, molecular LDA and PBE Kohn-Sham with restricted and collinear
unrestricted states, and stationary analytic forces for RHF, UHF and Kohn-Sham.

- A forward-derivative type replaced the JAX differentiation of the exchange and
  correlation expressions.
- Forces differentiate the stationary Lagrangian (Hellmann-Feynman plus Pulay) and
  perform no displaced SCF solves.

Closing result: the frozen H2 LDA energy and force were reproduced, and forces matched
fixed-grid finite differences for HAO, Gaussian and NAO bases in restricted and
unrestricted states.

## Phase 5: periodic SCF and analysis

Delivered the crystal SCF in Rust: restricted and collinear-spin LDA and PBE, Fermi-Dirac,
Gaussian and Methfessel-Paxton occupations, Kerker-damped mixing with DIIS, reciprocal
mesh reduction, Ewald energy, potential, forces and stress as a primitive, projections
onto trial subspaces and Wilson loops. A portable crystal snapshot made a solve
independent of the setup code.

Closing result: the frozen silicon energy and bands agreed with the native solve to
about 1e-12 eV.

## Phase 6: advanced quantum methods and correlation

Delivered shared-subspace VQE and ADAPT-VQE, ordered Pauli time propagation, finite-register
quantum phase estimation, and closed-shell MP2 natural orbitals.

Closing result: propagation and its adjoint matched dense exponentials and exported
circuits; dense and matrix-free QPE agreed; MP2 density and energy matched an
independent spin-orbital formulation.

## Phase 7: persistence and the execution boundary

Delivered native bounded Hamiltonian and checkpoint handling with warm restarts,
atomic file replacement, provider-independent circuit compilation, qubit-wise
commuting measurement grouping and count reductions, a checked job and shot ledger,
and streaming cube, XSF and BXSF writers.

Closing result: restarts, all three SDK gate streams, wide measurement groups,
concurrent ledger accounting and independent volumetric readers passed their tests.

## After Phase 7

Later work is recorded on [the cutover page](native_cutover.md), which carries the live
coverage table. In outline: the six exchange-correlation choices (including HSE06,
r2SCAN and the rVV10 pairings), molecular PAW-LCAO with its forces, native MCAS-VQE
and VALQA, native measured-gradient ADAPT-VQE, the run budget, and the removal of the
`engine` option and of JAX as a dependency. Several features that existed in the
Python implementation are not yet ported; they are in the
[not-yet-available table](#not-yet-available).
