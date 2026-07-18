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
* `numpy` (>= 2.0.0)
* `scipy`
* `matplotlib`
* `ase` (Atomic Simulation Environment)
* `qiskit`
* `qiskit-nature`
* `qiskit-ibm-runtime`
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

Carcará features an OpenMP-parallelized C backend (`libcarcara_integrals`) for real-space integrals. If the compiled library is missing, the framework automatically falls back to a vectorized NumPy implementation, which is correct but substantially slower.

To compile the C shared library:

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
from carcara.integrals import HAS_C_BACKEND
print(f"C backend active: {HAS_C_BACKEND}")
```

---

## Optional Acceleration Packages

To enable accelerated numerical operations on CPU, you can install the optional performance bundle:

```bash
pip install "carcara[accel]"
```

This installs `numba` (>= 0.60), which accelerates intermediate python loops.
