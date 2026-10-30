# Precision baseline (WP0), frozen 2026-10-29

These files freeze what the extension produced BEFORE the precision rename and before any
float32 work: `src/mandacaru/_native.cpython-314-darwin.so`, built 2026-10-29 22:36:22, sha256 in
`provenance.extension_sha256` of every file, git HEAD `8e50d99`, dirty working tree. Every run
(2 threads unless noted, 1.8 GB Context limit) loaded that one binary; 12 `.rs` files under `crates/` had been
edited by other teams after it was built, so the numbers belong to the binary, not to the tree.
The two modes are keyed `float64` (default) and `float32` (what the pre-rename extension called its
reduced mode: float32 orbital samples and complex64 Bloch samples, all else float64; the quantum
state is complex128 in both). Same keys will be written by re-running the producing scripts
(`baseline/*.py`, driven by `drive.py` with `MANDACARU_PREC_NAMES=float64,float32`) against the
renamed build.

- `molecular_energies.json`: total energies at full precision, H2/H2O/O2 (triplet)/C2H6/C6H6 x
  rhf, uhf, dft lda/pbe/r2scan x SZ/DZ/DZP x h in {0.35, 0.25, 0.20} A, families NAO (all-electron)
  and PAW-LCAO, SCF tolerance 1e-12. `not_run_case_ids` lists the planned cases that were NOT
  run (benzene DZ/DZP and a few 0.20 A cases: cost under the 2 GB / shared-machine limit).
  Cases whose SCF raised are stored with the error text (NAO O2 SZ: spin occupation does not fit the
  minimal basis; NAO O2 DZ lda and three C6H6 SZ dft runs: native SCF did not converge).
- Benzene NAO DZ rhf 0.25 A (a memory-gate case) has no entry in `molecular_energies.json`; its only
  frozen float64 energy is `electronic_energy` of the `scf` stage in `memory_table.json`, taken at the
  native default SCF tolerance of `mem_staged.py` (`kohn_sham`/`mean_field` defaults), not 1e-12.
- `repeat_and_tol13.json`: bit-for-bit repeats, 1- vs 4-thread runs, tolerance 1e-13 on H2O SZ
  0.25, and C6H6 SZ lda/pbe at tolerance 1e-10 (a frozen float64 energy for the memory-gate molecule).
- `forces_all_electron.json`: analytic forces (NAO and HAO), `project_translation=False`, eV/A.
  They are derivatives of the discretized energy on a 0.25 A grid, NOT physical forces (1e3 eV/A
  on H2O, net force of the same size). Independent check: central difference of the same energy on
  the same frozen grid, in `fd_check_H2.txt` and `fd_check_H2O.txt` (H2: Richardson-extrapolated difference 7e-7 eV/A; H2O SZ lda 1.6e-3 on a 296 eV/A force).
- `periodic_energies.json`: Al/Si lda, SZ/DZP, k 2^3/4^3 (PAW-LCAO) and the extra xc/spin-orbit cases
  (PBE, r2SCAN, PBE+rvv10, lda-fr; hse06 raises NotImplementedError for periodic systems).
- `quantum_energies.json`: vqe / adapt-vqe energies on 8 to 14 qubits (H2, LiH, HF, H2O, NH3) next
  to the lowest eigenvalue of the Hamiltonian each run was given.
- `memory_table.json`: process RSS (ru_maxrss in a fresh subprocess, post-import baseline
  subtracted) and Context peak (`memory_peak_bytes`) per stage; molecular (gate cases and a
  17-pair ladder), periodic (Si 3x3x3 k h=0.15, Al 8 atoms Gamma), and the state vector at 16, 20,
  22, 24 qubits (full register and sector, 4-parameter evaluate with and without Trotter, plus
  solve runs).
