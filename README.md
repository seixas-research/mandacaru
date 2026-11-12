<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<img src="https://raw.githubusercontent.com/seixas-research/mandacaru/refs/heads/main/logo/logo_light.png#gh-light-mode-only" alt="Mandacaru logo" height="200">
<img src="https://raw.githubusercontent.com/seixas-research/mandacaru/refs/heads/main/logo/logo_dark.png#gh-dark-mode-only" alt="Mandacaru logo" height="200">
</h1>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-red?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-fcbc2c.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Rust 1.85+](https://img.shields.io/badge/Rust-1.85+-b7410e.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyPI version](https://img.shields.io/pypi/v/mandacaru.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/mandacaru/)
[![Documentation Status](https://img.shields.io/readthedocs/mandacaru?style=for-the-badge&logo=readthedocs&logoColor=white&label=Docs)](https://mandacaru.readthedocs.io/en/latest/)
[![GitHub](https://img.shields.io/badge/GitHub-mandacaru-181717?style=for-the-badge&logo=github)](https://github.com/seixas-research/mandacaru)

</div>

# Mandacaru

Mandacaru is a framework for electronic structure and fermionic quantum
simulation with variational quantum algorithms, aimed at NISQ-era hardware
(IBM Quantum through Qiskit, Amazon Braket). One ASE calculator, `Mandacaru`,
runs the whole pipeline: geometry, basis, real-space integrals, a
second-quantized Hamiltonian, the fermion-to-qubit mapping, the ansatz and the
variational solver, on an exact state vector, a simulator or a quantum
processor. The same calculator also runs Hartree-Fock and Kohn-Sham density
functional theory for molecules and crystals.

The numerical engine is written in Rust, and Python is the interface: input
validation, the ASE integration, result views, plotting and the quantum SDK
adapters. The move from Python to Rust is not finished yet; see
[Migration to Rust](#migration-to-rust).

## Getting Started

### Requirements

- Python 3.14 or later.
- A Rust toolchain, Rust 1.85 or later, with `cargo` on the `PATH`. The install
  compiles the native extension.
- The PAW-LCAO datasets, for calculations with a PAW-LCAO basis (below).

### Install from source

```bash
conda create -n mandacaru python=3.14
conda activate mandacaru
git clone https://github.com/seixas-research/mandacaru.git
cd mandacaru
pip install -e '.[dev]'
```

The editable install runs `cargo build` and copies the extension into
`src/mandacaru/`. A change under `crates/` is not seen by Python until the
extension is rebuilt, so rerun `pip install -e '.[dev]'` after editing Rust
code. A release is also published on PyPI (`pip install mandacaru`); where no
wheel matches your platform, pip builds from the source distribution and needs
the same Rust toolchain.

### PAW-LCAO datasets

No pseudopotential data ships with the package. The datasets live in the
`mandacaru-paw` repository: clone it, then record where it is.

```bash
mandacaru-paw --set /path/to/mandacaru-paw   # writes MANDACARU_PAW_PATH to ~/.zshrc or ~/.bashrc
mandacaru-paw --status                       # draws a table of every dataset set it reaches
```

Open a new terminal for the variable to take effect. A shell that does not read
your profile (a script runner, for instance) needs `MANDACARU_PAW_PATH` on its
command line. A calculation that needs an unset library raises
`LibraryPathError` naming the `mandacaru-paw --set` command.

| Folder | Set | Read by default for |
| --- | --- | --- |
| `lda-sr/` | scalar-relativistic LDA | everything else (the default set) |
| `pbe-sr/` | scalar-relativistic PBE | `method="dft"` with `xc="pbe"`, `"hse06"` or `"r2scan"` |
| `lda-fr/` | LDA with the spin-orbit term | only when named with `paw_dataset="lda-fr"` |
| `pbe-fr/` | PBE with the spin-orbit term | only when named with `paw_dataset="pbe-fr"` |

The `HAO`, `NAO` and Gaussian bases read no data and need none of this.

### Check the install

```bash
mandacaru --version
python -c "from mandacaru import Mandacaru; print(Mandacaru(method='rhf').method)"
```

## Density functional theory

A Kohn-Sham calculation of water on a PAW-LCAO basis, with its energy and
analytic forces. The comments list the other values each option takes.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)                         # the cell is the box the real-space grid is cut from

atoms.calc = Mandacaru(
    method="dft",                                # alternatives: "rhf", "uhf" (without xc, which only dft uses)
    xc="pbe",                                    # alternatives: "lda" (default), "r2scan", "hse06"
    basis={"name": "PAW-LCAO", "size": "DZP"},   # sizes: "SZ" (default), "SZP", "DZ", "TZP", ...; other bases: "HAO", "NAO"
    paw_dataset=None,                            # None: the set of xc (pbe-sr here); "pbe-fr" adds spin-orbit coupling
    grid_spacing=0.25,                           # angstrom; converge it before comparing energies
    charge=0,                                    # total charge
    spin=False,                                  # True: unrestricted (spin-polarized) Kohn-Sham
    scf_options={"max_iterations": 300, "tolerance": 1e-9, "diis": True},
    precision="float64",                         # or "float32" for the stored orbital samples
    parallelization={"threads": 4},              # processors; a crystal also takes {"kpts": 4, "threads": 2}
    resources={"memory_limit_bytes": 2 * 1024**3},
    txt="h2o_dft.log",                           # run log file; trace=True also prints it
)

energy = atoms.get_potential_energy()            # eV
forces = atoms.get_forces()                      # eV/angstrom, analytic gradient computed in Rust
result = atoms.calc.result

print(f"converged: {result.success}")
print(f"energy: {energy:.6f} eV")
print(f"force scheme: {result.force_details['scheme']}")
print(forces)

atoms.write("h2o.xyz")                           # extended XYZ keeps the cell, for the command line below
atoms.calc.close()
```

A geometry with a periodic axis (`pbc=True`) runs the crystal solver with
`kpts`, smearing, band structures and densities of states, analytic forces and
stress, and DFT+U through `hubbard_u`. Use a PAW-LCAO basis for forces and
geometry optimization: an all-electron basis cannot resolve the nuclear cusp on
the uniform grid.

The `mandacaru` command runs one calculation and prints the result as JSON.
The geometry needs a cell, so use a JSON geometry or an extended XYZ file such
as the one written above:

```bash
mandacaru h2o.xyz --method dft --xc pbe --basis PAW-LCAO --grid-spacing 0.25 --parallelization threads=4 --forces
```

## Variational quantum algorithms

The quantum methods start from a Hartree-Fock reference on the same geometry
and basis, build the qubit Hamiltonian and run the variational solver in Rust.
ADAPT-VQE, the default method, grows the ansatz one growth step at a time from
an operator pool, appending the operator with the largest energy gradient.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)

atoms.calc = Mandacaru(
    method="adapt-vqe",                                     # alternatives: "vqe" (fixed UCCSD ansatz, with pool="fermionic"), "subspace-adapt-vqe" (excited states)
    basis={"name": "PAW-LCAO", "size": "DZP"},
    grid_spacing=0.25,
    active_space={"method": "mp2", "orbitals": 6},          # 6 MP2 natural orbitals reach the register; "energy" ranks by orbital energy
    pool="ceo",                                             # the default; alternatives: "fermionic", "qubit", "qeb", "ceo-ovp", "spin-orbit"
    mapping="jordan_wigner",                                # alternatives: "parity", "bravyi_kitaev", "parity_reduced" (two fewer qubits)
    optimizer={"method": "SLSQP", "maxiter": 2000, "tol": 1e-12},  # also "BFGS", "L-BFGS", "CG", "COBYLA", "Nelder-Mead", "SPSA"
    convergence={"gradient": 1e-3, "energy": 1e-3},         # stop at a largest pool gradient (Ha) or energy change (eV) below these
    max_iterations=30,                                      # growth limit: at most 30 growth steps
    device="AER_simulator",                                 # the default: exact local state vector
    shots=0,                                                # 0: exact energies; > 0: measured energies
    txt="h2o_adapt.log",                                    # run log file
)

energy = atoms.get_potential_energy()                       # eV
result = atoms.calc.result

print(f"qubits: {atoms.calc.n_qubits}")
print(f"Hartree-Fock: {result.reference_energy:.6f} eV")
print(f"ADAPT-VQE: {energy:.6f} eV")
print(f"correlation: {result.correlation_energy:.6f} eV")
print(f"operators: {result.num_operators}, evaluations: {result.num_evaluations}")
print(f"stopped: {result.timings['termination']}")
for step in result.iterations:                              # one row per growth step
    print(f"{step.operator_label:40s} {step.energy:.6f} eV  max gradient {step.max_gradient:.1e}")
```

`atoms.get_forces()` on the same calculator returns the analytic ADAPT-VQE
forces (PAW-LCAO basis, state vector), so ASE optimizers can relax a geometry on
the variational state. `subspace-vqe` and `subspace-adapt-vqe` target the
lowest `num_states` states together.

**On a quantum processor.** A fake or real processor requires `shots > 0`,
which measures every energy, and every ADAPT pool gradient, on the device. Use
SPSA or COBYLA there; the gradient optimizers
are refused because they turn shot noise into gradient noise. Rehearse the
identical call on a local fake backend (`device="fake_kingston"`, `"fake_fez"`
or `"fake_marrakesh"`) before any real `ibm_*` submission. On an `ibm_*` device
the whole optimization runs on the hardware, one job per energy evaluation.
Keep `measurement_budget=` on, as it refuses an oversized job before it queues,
and keep the run-wide job budget on, as it holds the run inside the IBM Runtime
quota (ten minutes per month on the open plan). Check `service.usage()` first and
pass `backend_options={"estimator_options": {"max_execution_time": ...}}`. Amazon
Braket devices and Cirq's exact simulator are selected the same way through
`device` and `backend_provider`.

## Migration to Rust

The numerical engine is moving from Python to Rust, and the move is not
complete. The rule is that numerical work moves to Rust first, every caller is
identified, and only then is the Python deleted: there is no Python fallback
and no execution selector. An option the native engine cannot honor raises an
error. It never selects a different calculation.

**Already in Rust:**

- Molecular RHF, UHF, GHF and Kohn-Sham with the six functionals (`lda`, `pbe`,
  `r2scan`, `hse06`, `pbe+rvv10`, `r2scan+rvv10`), the integrals, and the
  analytic forces of RHF, UHF and Kohn-Sham.
- The PAW machinery of a calculation: dataset reading, basis construction,
  projectors, one-center terms and the crystal setup.
- Crystals: the self-consistent iterations, bands and densities of states,
  spin-orbit spinors, DFT+U, analytic forces and stress, Wannier downfolding
  and finite-size corrections.
- The quantum layer: operators, mappings, operator pools, the ansatz, VQE,
  ADAPT-VQE and the subspace methods, all seven classical optimizers, Z2
  tapering, the double-factorized measurement, the analytic variational forces,
  phase estimation and time evolution.
- Parquet input and output for Hamiltonian caches and datasets, with no Python
  Parquet package.

**Next steps, ordered by dependency:**

1. Validate the HSE06 energies and gaps against an all-electron reference.
2. Spinor DFT+U for molecules and crystals, DFT+U with a hybrid functional,
   and hybrid and spin-orbit calculations on a symmetry-reduced k mesh.
3. Finish the PAW physics: scalar-relativistic dataset generation, the
   nonlinear one-center exchange-correlation term and multi-shell frozen-core
   exchange for Hartree-Fock.
4. Complete the quantum methods on crystals: the per-k register
   (`crystal_orbitals="k-space"`), a PAW-LCAO supercell, and analytic forces
   and stress.
5. Extend circuit execution to the multi-reference methods, continue device
   runs, and hand a GHF reference to the quantum methods.
6. Parallelize the rest of the crystal engine over the new `parallelization`
   layout (the hybrid exchange, forces and stress, post-processing). The basis,
   dataset and PAW adapter modules no longer hold numerical Python: they hand
   over to the Rust engine.
7. Write the SCF log while the run proceeds, not only when it ends.

The [native engine status](https://mandacaru.readthedocs.io/en/latest/development/native_cutover.html)
page of the manual tracks what is native and what is still missing.

## Source layout

| Location | Responsibility |
| --- | --- |
| `Cargo.toml`, `Cargo.lock` | Native workspace and locked dependencies |
| `crates/mandacaru-core` | Basis, models, molecular and periodic solvers, PAW, quantum algorithms, Parquet |
| `crates/mandacaru-kernels` | Low-level numerical kernels |
| `crates/mandacaru-python` | PyO3 bindings and checked NumPy transfers |
| `src/mandacaru` | Python interface, workflows, result views and integrations |
| `build_hooks/native.py` | Native extension build hook |
| `examples` | Runnable scripts, one folder per category |
| `scripts` | Local architecture, example and installation checks |
| `docs` | Sphinx manual |

## Local checks

The test suite runs locally; the repository has no hosted workflows.

```bash
python -m pytest                              # Python tests (the slow tier is skipped)
cargo test --profile release-test             # Rust tests (the slow tier is skipped)
python scripts/check_architecture.py          # crate and package boundaries
make -C docs html SPHINXOPTS="-W"             # the manual; warnings are errors
```

## Documentation and license

The [manual](https://mandacaru.readthedocs.io/en/latest/) covers installation,
the guides for each method and option, and the native engine status.
Mandacaru is released under the [MIT License](LICENSE).
