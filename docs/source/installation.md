# Installation

Use Python 3.14 or later. A virtual environment keeps Mandacaru's scientific and
quantum SDK dependencies separate from other projects.

## Install a release

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mandacaru
python -c "import mandacaru; print(mandacaru.__version__)"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

The package installs NumPy, SciPy, ASE, Matplotlib, spglib (crystal symmetry for
the periodic methods), Hamiltonian-cache dependencies, quantum SDKs and the
supporting numerical packages listed in `pyproject.toml`.
Installing a provider's SDK does not require an account on its service. The
introductory LiH examples run locally without credentials.

## Install a source checkout

Use this route to run the repository's examples or edit the documentation:

```bash
git clone https://github.com/seixas-research/mandacaru.git
cd mandacaru
python -m pip install -e ".[docs]"
```

The `/en/latest/` manual follows the development sources. If an option in that
manual is unavailable in an older installed release, use a matching source
checkout or select the documentation version for your release.

## Pseudopotential datasets

None of the generated families ships with the package: **NCPP**, **ONCVPSP**,
**PAW-LCAO** and **UPAW-LCAO** each live in their own repository — about 11 MB,
110 MB and 190 MB for all 92 elements of the first three — and an environment
variable names the checkout Mandacaru reads from:

| basis | variable | repository |
| :--- | :--- | :--- |
| `NCPP` | `MANDACARU_NCPP_PATH` | `mandacaru-ncpp` |
| `ONCVPSP` | `MANDACARU_ONCVPSP_PATH` | `mandacaru-oncvpsp` |
| `PAW-LCAO` | `MANDACARU_PAW_PATH` | `mandacaru-paw` |
| `UPAW-LCAO` | `MANDACARU_UPAW_PATH` | `mandacaru-upaw` (optional) |

```bash
git clone https://github.com/seixas-research/mandacaru-ncpp.git
mandacaru --set-ncpp mandacaru-ncpp

git clone https://github.com/seixas-research/mandacaru-paw.git
mandacaru --set-paw mandacaru-paw

git clone https://github.com/seixas-research/mandacaru-oncvpsp.git
mandacaru --set-oncvpsp mandacaru-oncvpsp

git clone https://github.com/seixas-research/mandacaru-upaw.git
mandacaru --set-upaw mandacaru-upaw

mandacaru --pseudo-status        # each variable, where it points, and how many datasets it serves
```

Each `--set-*` command writes `export MANDACARU_..._PATH=DIR` into `~/.zshrc`
or `~/.bashrc` (whichever `$SHELL` reads), asking `[Y/n]` before replacing a
different value; open a new terminal, or `source` the file, for the variable
to take effect in your shell. Inside a checkout the datasets sit one folder
per exchange-correlation functional (`<checkout>/lda/<Symbol>.parquet`; the
PAW-LCAO library also has `pbe/`). A calculation reads `lda/` unless the
calculator names another folder:

```python
atoms.calc = Mandacaru(method="adapt-vqe", basis="PAW-LCAO",
                       directory="pbe")      # $MANDACARU_PAW_PATH/pbe/
```

Without this, `basis="NCPP"`, `basis="PAW-LCAO"` and `basis="ONCVPSP"` raise a
`LibraryPathError` that repeats the matching `--set-*` command; the all-electron
bases (`HAO`, `NAO`, `NAO-AE`, the Gaussian families) need nothing extra. A
basis option `directory=...` (a full path) overrides the variable for one run.

`basis="UPAW-LCAO"` needs nothing either: `MANDACARU_UPAW_PATH` is the one
optional variable, and without it a missing dataset is generated on demand
(0.4-2.2 s per element, measured) and cached for the session.
`build_upaw_library()` writes them out into `$MANDACARU_UPAW_PATH/<xc>/` if
you would rather not pay that again.

## Numerical backend

Mandacaru's integral kernels are written in C. Nothing has to be built by hand:
the first integral engine of a session looks for the shared library, compiles it
when it is missing or stale, and loads it. Only if that compile fails does
Mandacaru warn and fall back to the NumPy reference kernels, which give the same
numbers more slowly.

To build it ahead of time — in a container image, in CI, or to read the error
when the automatic build did not work — run the command:

```bash
mandacaru --build-backend
```

It prints the library path and the OpenMP thread count on success, and on
failure prints the build log and exits with status 1. The same call from Python:

```python
from mandacaru.integrals import check_backend, ensure_backend

print(check_backend())   # Inspect the backend without compiling it.
print(ensure_backend())  # Build if necessary, then load the library.
```

The build needs a C compiler; CMake is used when it is installed, with a direct
compiler invocation as the fallback. OpenMP gives the parallel kernels — it is
found automatically on Linux, and on macOS comes from `brew install libomp`.

`MANDACARU_BACKEND=auto` selects the default policy. Use `MANDACARU_BACKEND=c` to
require the compiled backend (it raises rather than falling back), or
`MANDACARU_BACKEND=numpy` to select NumPy.

The shared library is written into `src/mandacaru/integrals/csrc/build` and
detected automatically; set `MANDACARU_INTEGRALS_LIB` to point at one built
elsewhere. Inspect `src/mandacaru/integrals/csrc/build/build.log` when a build
fails. The equivalent manual invocation is:

```bash
cmake -S src/mandacaru/integrals/csrc -B src/mandacaru/integrals/csrc/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/mandacaru/integrals/csrc/build
```

The FFT integral stage processes orbital pairs in blocks. Set
`MANDACARU_ERI_MEMORY_MB` to change its working memory budget (256 MB by default).
This budget controls integral workspace, not the total memory used by grids,
Hamiltonians or state vectors.

## Optional dependencies

| Extra | Purpose |
| :--- | :--- |
| `docs` | Sphinx, MyST and Furo for building this manual. |
| `pyarrow` | An alternative Parquet engine for Hamiltonian caches. |
| `memory` | `psutil`, for the resident-memory line of the `[PERFORMANCE]` block and the QPE memory check (both fall back to the standard library). |
| `legacy-forces` | JAX, needed only by `force_method="scf-response"`; the default `force_method="rdm"` does not use it. |
| `coverage` | Test coverage reporting with pytest-cov. |
| `dev` | Everything above. |

For example:

```bash
python -m pip install "mandacaru[dev]"
```

The default Parquet engine is `fastparquet`. The project has observed native
runtime conflicts when combining some Qiskit and PyArrow versions in the same
process. Keep the default engine unless you have checked your environment, or
use `hamiltonian_format="json"` for a cache without a Parquet dependency.

Continue with [your first LiH calculation](tutorial/vqe_lih.md). To edit this
manual, follow [the documentation build guide](contributing_docs.md).
