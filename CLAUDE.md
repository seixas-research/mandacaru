# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Carcará** is a framework for fermionic quantum simulation based on variational quantum algorithms (VQAs), targeting real NISQ-era quantum hardware (IBM Quantum via Qiskit). The intended end-to-end pipeline is: fermionic system → second-quantized Hamiltonian → qubit (Pauli) Hamiltonian → parameterized ansatz → VQA optimization → QPU/simulator execution → error mitigation. See `plan/roadmap.md` for the full phased design and its flagship algorithm, **ADAPT-VQE** with pluggable operator pools (fermionic / qubit / QEB / CEO).

**Important:** the project is mid-build. The fermionic front end is implemented — localized bases, the real-space integral engine, the second-quantized `Fermion` Hamiltonian, and the three fermion-to-qubit mappings (`core/`) — but the quantum-algorithm back end (circuits, VQE/ADAPT, optimizers, hardware backends, mitigation) is still empty stubs; the roadmap describes that target, not the current state. Before building on a module, check whether it actually has code (`wc -l`), because `plan/roadmap.md` mostly describes intended design: its Phase 2+ modules (circuits, VQE/ADAPT, optimizers, backends, mitigation) are not yet written. `pyproject.toml` is the source of truth for the build (currently Python ≥ 3.11).

## Current state of the code

Implemented:
- `src/carcara/basis/` — localized single-particle basis functions (`BasisFunction` ABC) and three concrete families: `HydrogenicOrbital` (analytic, with Slater charges), `NumericalAtomicOrbital` (confined Sankey/SIESTA-type, `nao.py`), `GaussianOrbital` (contracted GTOs, `gaussian.py`). **All families are generated natively from scratch — no tabulated/external basis-set data.** The `BasisSet` factory (`factory.py`) builds NAO or GTO bases; the GTO family is a minimal **STO-nG** basis generated in `sto_ng.py` by least-squares fitting `n_gaussians` primitives to the Slater-type orbital of each occupied subshell (Slater's-rules ζ = (Z−S)/n\*), with the reference fit done once at ζ=1 (cached, `_fit_reference`) and exponents rescaled by ζ². `BasisSet.build(method="GTO", n_gaussians=3)` → STO-3G-like; the fit reproduces published STO-nG exponents closely (H 1s → 2.22766/0.40577/0.10982). Shared aufbau config in `_config.py`, angular helpers in `_angular.py`.
- `src/carcara/integrals/` — real-space one- and two-body integral engine, grid, FFT Poisson solver, C backend binding
- `src/carcara/core/` — the second-quantized layer. `mapping.py`: `Fermion` (fermionic ladder operators with full algebra and a `from_integrals` builder for `H = Σ h_pq a†_p a_q + ½ Σ ⟨pq|rs⟩ a†_p a†_q a_s a_r`), `PauliSum` (the qubit-operator output type, with `to_sparse_pauli_op`), and the three fermion-to-qubit mappings — **Jordan-Wigner** (default), **parity** (with optional two-qubit reduction) and **Bravyi-Kitaev** — via a shared encoding-matrix construction (`map_to_qubits(method=...)`). `hamiltonian.py`: `HydrogenicIntegrals` drives the integral engine to build the spin-orbital molecular Hamiltonian as a `Fermion`.
- `src/carcara/wavefunction.py` — atomic-system facade over basis + grid + engine (ASE XYZ I/O)

Empty stubs (0 LOC — the roadmap's target modules, not yet written): `circuits/ansatz.py`, `circuits/gates.py`, `algorithms/vqe.py`, `optimizers/optim.py`, `backends/hardware.py`, `backends/mitigation.py`, `utils/logging.py`.

## Commands

The package is **not** pip-installed in this dev environment. Run everything against the `src/` layout via `PYTHONPATH`:

```bash
# Run all tests
PYTHONPATH=src python -m pytest test/ -q

# Run one test file / class / test
PYTHONPATH=src python -m pytest test/test_integrals.py -v
PYTHONPATH=src python -m pytest test/test_integrals.py::TestPoissonFFT -v
PYTHONPATH=src python -m pytest test/test_integrals.py::TestPoissonFFT::test_1s_self_repulsion_matches_exact

# Run an example
PYTHONPATH=src python examples/H2_integrals.py

# Editable install (alternative to PYTHONPATH)
pip install -e .
```

Build the docs (Sphinx + furo theme, config in `docs/source/conf.py`):

```bash
cd docs && make html   # output in docs/build/html/
```

## The C integral backend

The heavy integral kernels live in C (`src/carcara/integrals/csrc/carcara_integrals.c`), an OpenMP-parallel shared library — **not** a CPython extension, so one compiled `.dylib`/`.so` serves every Python version. `_backend.py` loads it via `ctypes` with zero-copy pointer passing (NumPy `complex128` ⇄ C99 `double _Complex`).

The build is **optional**: if the library is absent, `HAS_C_BACKEND` is `False` and vectorized NumPy reference implementations (which mirror the C kernels exactly) run instead, so the package is always importable and testable without compiling anything.

Build it with CMake; the output must land in `csrc/build/` where `_backend.py` looks (it also honors the `CARCARA_INTEGRALS_LIB` env var):

```bash
cd src/carcara/integrals/csrc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# macOS / Apple Clang needs Homebrew libomp:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenMP_ROOT=$(brew --prefix libomp)
```

When editing the numerics, keep the C kernel and its NumPy fallback in `_backend.py` in lockstep — tests may exercise either depending on whether the library is built.

## Architecture: the integral engine

The one design principle that spans multiple files is that the integral machinery is **basis-agnostic**. The C backend and `IntegralEngine` never see analytic orbital forms — only **sampled values** `psi[i, :]` on a uniform cubic grid. This is what lets any basis drop in with zero changes to the integral core: a new family only implements `BasisFunction.evaluate(x, y, z)` (returning `complex128` in Bohr). `HydrogenicOrbital`, `NumericalAtomicOrbital` (spline of a confined radial solve) and `GaussianOrbital` (contracted Gaussians) all follow the same `R(r)·Y_lm` pattern; NAO/GTO share `_angular.py` for the Cartesian→spherical + spherical-harmonic step. Orbitals are normalized in 3D with orthonormal `Y_lm` (i.e. `∫|R|²r²dr = 1`) — preserve that convention for any new radial family.

The single contract is `BasisFunction.evaluate(x, y, z)` (in `basis/base.py`). The data flow:

1. `Grid` (`integrals/grid.py`) — a uniform cubic grid specified by a physical spacing `h` (not a node count); `points`/`dx` are derived. Owns sampling points, spacing `dx`, volume element `dV`.
2. Each `BasisFunction.sample(grid)` produces a contiguous `complex128` vector.
3. `IntegralEngine` (`integrals/engine.py`) stacks them into `(M, ngrid)`, evaluates the external potential callable `V(x,y,z)`, and dispatches to the backend:
   - `one_body(potential)` → kinetic `T` (finite-difference Laplacian) and potential `V` matrices → core Hamiltonian `h = T + V`.
   - `two_body(method=...)` → electron-repulsion tensor `<ab|cd>` in **physicists' notation** (electron 1 carries indices `a,c`; electron 2 `b,d`). `method="fft"` (default) uses the O(N log N) `PoissonFFTSolver` (`integrals/poisson.py`); `method="direct"` uses the O(N²) real-space double sum in C.
4. `Wavefunction` (`wavefunction.py`) is a thin facade: it reads geometry (ASE), builds the grid + hydrogenic basis, and delegates all physics to the engine.

**Units convention** (`units.py` is the single source of truth): the numerical core — `evaluate`, the grid coordinate arrays, the C backend, the Poisson solver — always works in **atomic units (Bohr, Hartree)**. The user-facing classes (`Grid`, `HydrogenicOrbital`, `Potentials`, `IntegralEngine`) default to the **frontend units (Ångström, eV)** and convert at their boundary via a `units=`/`energy_units=` argument. The legacy `Wavefunction` facade opts back into atomic units (`units="bohr"`, `energy_units="Ha"`), which is why its public API and tests are unchanged. When adding a user-facing length/energy argument, route it through `units.py` and default to Ångström/eV.

`HydrogenicOrbital` also provides Slater's-rules effective charges: `slater_effective_charge(atomic_number, n, l)` (static) and the `from_slater(n, l, m, atomic_number, ...)` constructor.

Validation anchor used throughout the tests: the hydrogen 1s on-site repulsion `<00|00>` must recover the exact `5/8 Ha` (`= 0.625` Ha `= 17.007` eV, depending on `energy_units`).

## Versioning

CalVer, single source of truth in `src/carcara/version.py` (`__version__`), consumed dynamically by hatchling. The scheme is `YY.M.patch` (e.g. `26.7.3` = 2026, month 7, patch 3). Releases are git-tagged `v<version>` and published to PyPI.
