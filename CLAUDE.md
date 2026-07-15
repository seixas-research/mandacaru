# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Carcará** is a framework for fermionic quantum simulation based on variational quantum algorithms (VQAs), targeting real NISQ-era quantum hardware (IBM Quantum via Qiskit). The intended end-to-end pipeline is: fermionic system → second-quantized Hamiltonian → qubit (Pauli) Hamiltonian → parameterized ansatz → VQA optimization → QPU/simulator execution → error mitigation. See `plan/roadmap.md` for the full phased design and its flagship algorithm, **ADAPT-VQE** with pluggable operator pools (fermionic / qubit / QEB / CEO).

**Important:** the project is early. Most of the pipeline is still empty stubs — the roadmap describes the target, not the current state. Only the real-space integral computation is implemented. Before building on a module, check whether it actually has code (`wc -l`), because `plan/roadmap.md` describes intent and is partly out of date (e.g. it says Python ≥ 3.10, but `pyproject.toml` requires ≥ 3.14).

## Current state of the code

Implemented:
- `src/carcara/basis/` — localized single-particle basis functions (`BasisFunction` ABC, `HydrogenicOrbital`)
- `src/carcara/integrals/` — real-space one- and two-body integral engine, grid, FFT Poisson solver, C backend binding
- `src/carcara/wavefunction.py` — atomic-system facade over basis + grid + engine (ASE XYZ I/O)

Empty stubs (0 LOC — the roadmap's target modules, not yet written): `core/hamiltonian.py`, `core/mappings.py`, `circuits/ansatz.py`, `circuits/gates.py`, `algorithms/vqe.py`, `optimizers/optim.py`, `backends/hardware.py`, `backends/mitigation.py`, `utils/logging.py`.

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
PYTHONPATH=src python examples/H2.py

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

The one design principle that spans multiple files is that the integral machinery is **basis-agnostic**. The C backend and `IntegralEngine` never see analytic orbital forms — only **sampled values** `psi[i, :]` on a uniform cubic grid. This is what lets new bases (Wannier, numerical atomic orbitals) drop in with zero changes to the integral core.

The single contract is `BasisFunction.evaluate(x, y, z)` (in `basis/base.py`). The data flow:

1. `Grid` (`integrals/grid.py`) — a uniform cubic grid; owns sampling points, spacing `dx`, volume element `dV`. All coordinates are in **Bohr** (atomic units); `Wavefunction` converts Ångström input via `ANGSTROM_TO_BOHR`.
2. Each `BasisFunction.sample(grid)` produces a contiguous `complex128` vector.
3. `IntegralEngine` (`integrals/engine.py`) stacks them into `(M, ngrid)`, evaluates the external potential callable `V(x,y,z)`, and dispatches to the backend:
   - `one_body(potential)` → kinetic `T` (finite-difference Laplacian) and potential `V` matrices → core Hamiltonian `h = T + V`.
   - `two_body(method=...)` → electron-repulsion tensor `(ab|cd)` in **chemists' notation**. `method="fft"` (default) uses the O(N log N) `PoissonFFTSolver` (`integrals/poisson.py`); `method="direct"` uses the O(N²) real-space double sum in C.
4. `Wavefunction` (`wavefunction.py`) is a thin facade: it reads geometry (ASE), builds the grid + hydrogenic basis, and delegates all physics to the engine.

Validation anchor used throughout the tests: the hydrogen 1s on-site repulsion `(00|00)` must recover the exact `5/8 Ha = 0.625`.

## Versioning

CalVer, single source of truth in `src/carcara/version.py` (`__version__`), consumed dynamically by hatchling. The scheme is `YY.M.patch` (e.g. `26.7.3` = 2026, month 7, patch 3). Releases are git-tagged `v<version>` and published to PyPI.
