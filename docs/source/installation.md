# Installation

Use Python 3.11 or later. A virtual environment keeps Carcará's scientific and
quantum SDK dependencies separate from other projects.

## Install a release

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install carcara
python -c "import carcara; print(carcara.__version__)"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

The package installs NumPy, SciPy, ASE, Matplotlib, Hamiltonian-cache dependencies,
quantum SDKs and the supporting numerical packages listed in `pyproject.toml`.
Installing a provider's SDK does not require an account on its service. The
introductory LiH examples run locally without credentials.

## Install a source checkout

Use this route to run the repository's examples or edit the documentation:

```bash
git clone https://github.com/seixas-research/carcara.git
cd carcara
python -m pip install -e ".[docs]"
```

The `/en/latest/` manual follows the development sources. If an option in that
manual is unavailable in an older installed release, use a matching source
checkout or select the documentation version for your release.

## Numerical backend

Carcará prefers its compiled C integral backend. It checks for the library when
an integral engine is created and attempts to build it if necessary. A C compiler
is required for that build; CMake is supported, with a compiler-based fallback.
OpenMP enables parallel execution. If compilation fails under the default
policy, Carcará reports a warning and uses the NumPy implementation.

```python
from carcara.integrals import check_backend, ensure_backend

print(check_backend())   # Inspect the backend without compiling it.
print(ensure_backend())  # Build if necessary, then load the library.
```

`CARCARA_BACKEND=auto` selects the default policy. Use `CARCARA_BACKEND=c` to
require the compiled backend, or `CARCARA_BACKEND=numpy` to select NumPy.

For an explicit CMake build on Linux:

```bash
cmake -S src/carcara/integrals/csrc -B src/carcara/integrals/csrc/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/carcara/integrals/csrc/build
```

On macOS, install Homebrew's OpenMP library before configuring:

```bash
brew install libomp
cmake -S src/carcara/integrals/csrc -B src/carcara/integrals/csrc/build -DCMAKE_BUILD_TYPE=Release -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build src/carcara/integrals/csrc/build
```

The shared library is written into the build directory and detected
automatically. Inspect `src/carcara/integrals/csrc/build/build.log` when an
automatic build fails.

The FFT integral stage processes orbital pairs in blocks. Set
`CARCARA_ERI_MEMORY_MB` to change its working memory budget (256 MB by default).
This budget controls integral workspace, not the total memory used by grids,
Hamiltonians or state vectors.

## Optional dependencies

| Extra | Purpose |
| :--- | :--- |
| `docs` | Sphinx, MyST and Furo for building this manual. |
| `pyarrow` | An alternative Parquet engine for Hamiltonian caches. |
| `coverage` | Test coverage reporting with pytest-cov. |
| `dev` | Documentation, the alternative Parquet engine and coverage tools. |

For example:

```bash
python -m pip install "carcara[dev]"
```

The default Parquet engine is `fastparquet`. The project has observed native
runtime conflicts when combining some Qiskit and PyArrow versions in the same
process. Keep the default engine unless you have checked your environment, or
use `hamiltonian_format="json"` for a cache without a Parquet dependency.

Continue with [your first LiH calculation](tutorial/vqe_lih.md). To edit this
manual, follow [the documentation build guide](contributing_docs.md).
