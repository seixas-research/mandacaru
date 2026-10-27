# Phase 0 baseline findings

Source revision: `b37d2af3f29a963c684e3b90a5522e194d5608a4`. Scientific source SHA-256: `75d9c23424aef63dd31c02fead07cc14becaff132e6dea3bdb9276c39d4a8320`.

The production implementation and existing tests were unchanged during capture. The source manifest includes Python, Rust, C, headers and build inputs; generated libraries are identified separately in workload reports.

## Measured corpus

All 11 workloads passed three independent interpreter samples under each of the NumPy and existing Rust integral backends: 66 samples total. The scientific cross-backend comparison passed with no differences beyond declared tolerances. This measures the current Python program with different integral backends, not a completed Rust engine.

Environment: Python 3.14.6, macOS-26.6.2-arm64-arm-64bit-Mach-O, 8 logical CPUs, 16.0 GiB physical memory. Thread controls requested one thread; the Rust thread-control limitation below applies. NumPy 2.5.4 and SciPy 1.18.1; all dependency pins are in `requirements.txt`.

Compute time includes calculation, snapshot extraction and invariant checks after common imports and backend setup. Peak RSS includes imports and is measured in a fresh child process. The table uses median compute time and maximum RSS across the three samples; MB is decimal. Small-case timing differences should not be treated as reliable speedup estimates.

| Workload | NumPy compute (s) | Rust compute (s) | NumPy peak RSS (MB) | Rust peak RSS (MB) |
| --- | ---: | ---: | ---: | ---: |
| `operator_algebra` | 0.0027 | 0.0027 | 151.2 | 151.2 |
| `h2_rhf` | 0.0320 | 0.0300 | 154.3 | 154.4 |
| `h_uhf` | 0.0289 | 0.0303 | 154.0 | 154.6 |
| `h2_ghf` | 0.0402 | 0.0409 | 154.3 | 154.6 |
| `h2_cached_vqe` | 0.0158 | 0.0161 | 153.7 | 153.4 |
| `h2_cached_adapt` | 0.0144 | 0.0143 | 153.8 | 153.4 |
| `h2_dft_lda_forces` | 0.1313 | 0.1293 | 155.4 | 155.4 |
| `h2_rhf_atomic_units` | 0.0299 | 0.0301 | 154.3 | 154.0 |
| `h2o_rhf_sto3g` | 0.1950 | 0.1954 | 165.7 | 165.7 |
| `h2_paw_pbe` | 1.3464 | 1.4739 | 287.5 | 287.7 |
| `si_periodic_lda` | 5.4582 | 2.8897 | 485.0 | 432.9 |

The fixed-grid H₂ force differs from its central energy finite difference by 0.000121 eV/Å (check limit: 0.005 eV/Å). The same case checks an A→B→A geometry round trip. Cached VQE and ADAPT energies are independently checked against exact diagonalization in the particle sector.

## Existing test status

`full-suite.json` and `full-suite-progress.txt` preserve the broad non-slow run: it reached the 1,200-second budget, with the last printed progress marker at 75%. It is incomplete and produced no finalized JUnit report. The progress stream contains 74 failure and 27 error indicators, whose identities cannot be reconstructed from this quiet log. These are progress indicators, not a finalized count of unique failing testcases. The remainder of the suite and the slow tests are unmeasured in this capture.

A completed focused run of calculator, force, API, documentation-link and repository-hygiene tests recorded 232 passes and 6 failures in 38.40 seconds (`focused-suite.json`). Its six failures are:

- `test.algorithms.test_calculator.TestCholeskyTwoBody::test_active_space_forces_match_the_dense_tensor` — AttributeError: 'NoneType' object has no attribute 'hellmann_feynman'
- `test.algorithms.test_forces.TestByPartsMode::test_calculator_exposes_the_option` — AttributeError: 'NoneType' object has no attribute 'details'
- `test.algorithms.test_forces.TestMandacaru::test_get_forces_matches_the_driver` — AttributeError: 'NoneType' object has no attribute 'unprojected'
- `test.algorithms.test_forces.TestMandacaru::test_force_breakdown_is_exposed` — RuntimeError: no forces have been computed yet
- `test.algorithms.test_forces.TestProjectedForcesAreQualified::test_both_arrays_are_written` — AttributeError: 'NoneType' object has no attribute 'forces'
- `test.test_repo_hygiene::test_no_file_names_another_code` — AssertionError: files naming another electronic-structure code: ['src/mandacaru/integrals/buffers.py']. Name a scheme by what it does (CLAUDE.md hard rule).

Five focused failures concern unavailable force results or breakdowns after calculation cleanup. The measured force case independently records `result_available_after_forces=false` and `force_breakdown_available_after_forces=false`; these lifecycle diagnostics are not golden behavior that Rust must preserve. The hygiene failure originates in an existing production-file docstring and was left unchanged.

The new `suite.py` runner records explicit scope, exit status, missing/invalid JUnit evidence and timeout status. On timeout it first requests graceful interruption and gives pytest 15 seconds to finalize evidence before terminating the process group. Future broad captures should use this runner rather than the initial timeout wrapper.

## Inventory and scope

The static inventory contains 172 Python modules, 104,276 Python source lines, 8 Rust files (3,656 lines), and 171 existing test files with 3,784 declared test functions. Parameterization expands the executed test count. `capabilities.json` indexes 22 capability groups with implementation and test paths.

The corpus is intentionally small and coarse. It does not measure metallic or SOC crystals, periodic stress, NAO/NAO-AE generation, mixed precision, memory pressure, disk buffers, hardware providers, stochastic algorithms, learned-model training, or all optimizer/basis/functional combinations. Those capabilities remain in the inventory and migration scope. PAW input hashes cover the two dataset files used by the numerical corpus; the broad suite may use other external datasets, which were not fully fingerprinted.

Phase 1 can use the checked H₂ JSON fixture and existing native kernels to establish Rust types, safe kernel interfaces and bindings. It must investigate relevant existing failures and preserve the successful scientific checks. JAX differentiation/training and Python quantum-SDK execution/transpilation remain explicit migration decisions.

## Tooling verification

The final tooling, explicit public-API and documentation-link check passed all 117 tests, including 25 migration-tooling cases. `validation.json` records its scope and result. These checks cover missing cases, failed/skipped samples, nonfinite values, changed units/shapes/datasets, inconsistent repeats, timeouts, JUnit evidence, environment isolation and checked-reference integrity. They do not replace the incomplete broad scientific suite.

## Thread-control finding

Rust shell and two-center kernels call `std::thread::available_parallelism()` independently of `MANDACARU_NUM_THREADS`. The configured request and backend-reported count therefore do not guarantee one native thread throughout a workload; actual kernel concurrency was not measured. The raw reports carry this audit finding. The NumPy/Rust timings must not be interpreted as comparisons at equal thread counts. Numerical equality remains valid. Phase 1 must unify kernel thread control before performance acceptance thresholds are established.
