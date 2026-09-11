# Installation Guide

This guide details how to set up Carcará for development or direct simulation use.

---

## Prerequisites

To run and build Carcará, you need:

1. **Python:** Version 3.11 or higher.
2. **C++ Compiler:** Supporting OpenMP (e.g., `gcc` on Linux/macOS or Apple Clang with `libomp` on macOS).
3. **CMake:** Version 3.15 or higher (required to compile the accelerated C integral backend).

---

## Installation Methods

### Method A: Install via pip (Stable Release)

You can install the stable version of Carcará directly from PyPI:

```bash
pip install carcara
```

This installs all core Python dependencies:

**Numerical core**
* `numpy` (>= 2.0.0)
* `scipy`
* `matplotlib` — plotting helpers (expressibility, band structures)
* `ase` (Atomic Simulation Environment) — geometries and the calculator interface

**Hamiltonian cache**
* `fastparquet` — the default Apache Parquet engine
* `pandas` — its table interface

**Quantum SDKs** (backend providers and the device registry)
* `qiskit` — default circuit provider and gate-count profiling
* `qiskit-nature`
* `qiskit-ibm-runtime` — the reserved `ibm-quantum` device
* `amazon-braket-sdk` — local simulator plus the AWS Braket devices and QPUs
* `cirq`

**Testing**
* `pytest`

### Method B: Install from Source (Developer Setup)

To make changes to the source code, run the examples, or work on the C backend, clone the repository and install it in editable mode:

```bash
# Clone the repository
git clone https://github.com/seixas-research/carcara.git
cd carcara

# Install in editable mode
pip install -e .
```

---

## Compiling the C Backend (Recommended)

Carcará features an OpenMP-parallelized C backend (`libcarcara_integrals`) for real-space integrals, and it prefers that backend for every integration. **You normally do not have to build it by hand:** the first integral engine created in a process checks for the library and, when it is missing or stale, compiles it on the spot for the current machine (CMake when installed, otherwise the system C compiler; OpenMP through Homebrew's `libomp` on macOS). A full log lands in `src/carcara/integrals/csrc/build/build.log`. Only when that compile fails does the framework fall back to the vectorized NumPy reference kernels, and it says so once with a `RuntimeWarning`.

```python
from carcara.integrals import check_backend, ensure_backend, build_backend

print(check_backend())     # what is loaded now, no build
print(ensure_backend())    # the check the engine runs: build if needed, then load
build_backend(verbose=True)  # force a (re)compile by hand
```

The policy is the environment variable `CARCARA_BACKEND`: `auto` (default) prefers C and builds when needed; `c` does the same but raises instead of falling back; `numpy` forces the reference kernels (for benchmarks and debugging).

To compile the C shared library by hand:

### On Linux
```bash
cd src/carcara/integrals/csrc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### On macOS (Apple Silicon or Intel)
macOS's default Apple Clang does not ship with OpenMP. You must install `libomp` via Homebrew:

```bash
# Install OpenMP library
brew install libomp

# Configure with brew prefix path
cd src/carcara/integrals/csrc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenMP_ROOT=$(brew --prefix libomp)
cmake --build build
```

The compiled dynamic library (`.so` or `.dylib`) will be built under `src/carcara/integrals/csrc/build/`. Carcará's loading system detects it automatically. You can verify that it is loaded in Python:

```python
from carcara.integrals import check_backend
print(check_backend().label)   # e.g. "C (OpenMP, 8 threads)"
```

### Memory

The C backend speeds up the one-body, projector and direct two-body kernels, but it does not change the memory peak of a calculation: that peak is the FFT two-body step, which forms the pair densities and their Coulomb potentials on the grid. That step now runs in memory-bounded blocks over the Hermitian pairs (`rho_ji = conj(rho_ij)`, so only `M(M+1)/2` pairs are ever built or solved), with a working-set budget of 256 MB by default. Lower or raise it with `CARCARA_ERI_MEMORY_MB` or per call with `IntegralEngine.two_body(max_memory_mb=...)`; the result is the same tensor to round-off, and a smaller budget only costs a little wall time.

---

## Optional Dependency Groups

| Extra | Installs | Purpose |
|---|---|---|
| `pyarrow` | `pyarrow` | An alternative Apache Parquet engine for the Hamiltonian cache. |
| `docs` | `sphinx`, `sphinx-rtd-theme`, `myst-parser`, `furo` | Building this documentation. |
| `dev` | everything above | Full development environment. |

```bash
pip install "carcara[dev]"
```

### A note on the `pyarrow` extra

The Hamiltonian cache reads and writes Parquet through either `fastparquet` (the
default, installed with the package) or `pyarrow`. Both produce standard Parquet
files, so they are interchangeable.

```{warning}
`fastparquet` is the default **on purpose**. On some platforms — reproduced on
CPython 3.14 with `qiskit` 2.5 and `pyarrow` 25 — calling
`pyarrow.parquet.write_table` in a process that has *also* run Qiskit's
`transpile` crashes the interpreter, since both ship their own native runtimes.
Carcará transpiles circuits for gate-count profiling in the same process that
saves the Hamiltonian, so the default engine avoids that combination.

If your environment is unaffected, pass `engine="pyarrow"` explicitly. If you
would rather not depend on a Parquet engine at all, use
`hamiltonian_format="json"`, which needs only the standard library.
```

---

## AWS Braket Credentials (Optional)

Running on the Amazon Braket **service** — the managed simulators or a real QPU —
needs AWS credentials configured on your machine:

```bash
aws configure
```

Local Braket simulation (`device="braket-local"`) needs no credentials and costs
nothing. See {doc}`guide/aws_braket` for the details, including how to estimate
the cost of a hardware run before submitting it.
