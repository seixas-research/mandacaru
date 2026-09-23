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

The NCPP datasets ship with the package. The **ONCVPSP** and **PAW** ones do
not — about 110 MB and 190 MB for all 92 elements — so they live in their own
repositories, and the library holds a symbolic link to a checkout:

```bash
git clone https://github.com/seixas-research/mandacaru-paw.git
mandacaru --link-paw mandacaru-paw

git clone https://github.com/seixas-research/mandacaru-oncvpsp.git
mandacaru --link-oncvpsp mandacaru-oncvpsp

mandacaru --pseudo-status        # what is linked, and how many datasets each serves
```

Each command links the directory and then loads one dataset through the normal
loader to prove the link works, exiting non-zero if it does not. Re-running with
a new path moves the link, so the data repository can be relocated freely; a
real, non-empty `library/paw/` directory is refused rather than deleted.

Without this, `basis="PAW"` and `basis="ONCVPSP"` raise a `FileNotFoundError`
that repeats these commands. `basis="NCPP"` and the all-electron bases (`HAO`,
`NAO`, `NAO-AE`, the Gaussian families) need nothing extra. Set
`MANDACARU_PSEUDO_PATH` to serve the library from somewhere else entirely.

`basis="UPAW"` needs nothing either: no library is shipped for it, so a missing
dataset is generated on demand (a fraction of a second per element) and cached
for the session. `build_upaw_library()` writes them out if you would rather not
pay that again.

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
