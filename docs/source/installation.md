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

The package installs NumPy, SciPy, JAX (the r²SCAN functional of
`method="dft"` is written in it), ASE, Matplotlib, spglib (crystal symmetry for
the periodic methods), threadpoolctl (keeps BLAS single-threaded beside the
OpenMP kernels), Hamiltonian-cache dependencies, quantum SDKs and the
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

To also run the test suite from that checkout, install the `dev` extra
instead, and keep the `-e`: `python -m pip install -e ".[dev]"`. Without `-e`,
`import mandacaru` resolves to whatever was last installed into
`site-packages`, not the checkout — so a subsequent edit to `src/` is silently
not what the tests, or a calculation, actually run.

## Pseudopotential datasets

None of the generated families ships with the package: **PAW-LCAO** and
**UPAW-LCAO** each live in their own repository — about 190 MB for all 92
elements of PAW-LCAO — and an environment variable names the checkout
Mandacaru reads from:

| basis | variable | repository |
| :--- | :--- | :--- |
| `PAW-LCAO` | `MANDACARU_PAW_PATH` | `mandacaru-paw` |
| `UPAW-LCAO` | `MANDACARU_UPAW_PATH` | `mandacaru-upaw` (optional) |

```bash
git clone https://github.com/seixas-research/mandacaru-paw.git
mandacaru --set-paw mandacaru-paw

git clone https://github.com/seixas-research/mandacaru-upaw.git
mandacaru --set-upaw mandacaru-upaw

mandacaru --pseudo-status        # each variable, where it points, and how many datasets it serves
```

Each `--set-*` command writes `export MANDACARU_..._PATH=DIR` into `~/.zshrc`
or `~/.bashrc` (whichever `$SHELL` reads), asking `[Y/n]` before replacing a
different value; open a new terminal, or `source` the file, for the variable
to take effect in your shell. Inside a checkout the datasets sit one folder
per set (`<checkout>/lda-sr/<Symbol>.parquet` — the scalar-relativistic
LDA set); PAW-LCAO also has
`lda-fr/` (the spin-orbit set), `pbe-sr/` (the scalar-relativistic
PBE set) and `pbe-fr/` (PBE with the spin-orbit term). A calculation reads the set of its functional — `pbe-sr/` for
`method="dft"` with PBE, HSE06 or r2SCAN, `lda-sr/` for LDA and every other
method — unless the calculator names another folder (`--paw-dataset` on the
command line):

```python
atoms.calc = Mandacaru(method="adapt-vqe", basis="PAW-LCAO",
                       paw_dataset="lda-fr")  # $MANDACARU_PAW_PATH/lda-fr/
```

Without this, `basis="PAW-LCAO"` raises a
`LibraryPathError` that repeats the matching `--set-*` command; the all-electron
bases (`HAO`, `NAO`, `NAO-AE`, the Gaussian families) need nothing extra. A
basis option `paw_dataset=...` (a full path) overrides the variable for one run.
A PAW-LCAO dataset whose own recorded functional disagrees with
the folder it was loaded from is refused, so a file cannot be
read as the wrong functional by being copied into the wrong folder.

`basis="UPAW-LCAO"` needs nothing either: `MANDACARU_UPAW_PATH` is the one
optional variable, and without it a missing dataset is generated on demand
(0.4-2.2 s per element, measured) and cached for the session.
`build_upaw_library()` writes them out into `$MANDACARU_UPAW_PATH/lda-sr/` if
you would rather not pay that again.

## Numerical backend

Mandacaru's high-performance integral kernels are implemented in **Rust** (with a
C fallback). Nothing has to be built by hand: the first integral engine of a
session looks for the compiled shared library, compiles it on demand when it is
missing or stale, and loads it. Only if the native compilation fails does
Mandacaru warn and fall back to the NumPy reference kernels, which give the same
numbers more slowly.

The **Rust backend** (`src/mandacaru/integrals/rust/`) offers several key advantages:
- **Zero external runtime dependencies**: Uses Rust's standard library scoped
  multithreading (`std::thread::scope`), completely eliminating the need for
  external OpenMP runtimes (such as `libomp` on macOS).
- **Native mixed precision**: Provides hardware-accelerated single-precision
  kernels (`mandacaru_real_shell_f32`, `mandacaru_laplacian_real_f32`,
  `mandacaru_kb_project_f32`) powering the `precision="single"` calculator option,
  which reduces real-space grid RAM by ~50% with zero loss of secular matrix accuracy.
- **Memory safety and portability**: Fast C ABI exports linkable into standard
  Python processes via `ctypes`.

To build it ahead of time — in a container image, in CI, or to verify the toolchain —
run the command:

```bash
mandacaru --build-backend
```

It prints the library path and the thread count on success, and on failure prints
the build log and exits with status 1. The equivalent call from Python:

```python
from mandacaru.integrals import check_backend, ensure_backend

print(check_backend())   # Inspect the backend without compiling it.
print(ensure_backend())  # Build if necessary, then load the library.
```

The Rust backend is built using `cargo`. If Cargo is not installed on the system,
Mandacaru automatically falls back to compiling the C backend (`src/mandacaru/integrals/csrc/`)
using CMake or the bare C compiler with OpenMP.

`MANDACARU_BACKEND=auto` selects the default policy (prefers Rust, then C, then NumPy).
Use `MANDACARU_BACKEND=rust` to require the Rust backend, `MANDACARU_BACKEND=c` to
require the compiled C backend, or `MANDACARU_BACKEND=numpy` to force the pure Python/NumPy
reference kernels. Any other value raises a `ValueError`.

The same variable also selects the backend for a second, separate compiled
library: the radial kernels (tridiagonal eigenpairs and Numerov recursions)
behind atom and pseudopotential generation
(`mandacaru.basis.radial_backend`). `mandacaru-build --build-backend` builds
and reports on that one; `mandacaru-build --backend {auto,c,python}` selects
it for one generation run (`python` there maps to `MANDACARU_BACKEND=numpy`).
The [pseudopotentials guide](guide/pseudopotentials.md) shows `mandacaru-build`
in use.

The shared library is located in `src/mandacaru/integrals/` (or `rust/target/release/`
and `csrc/build/`) and detected automatically; set `MANDACARU_INTEGRALS_LIB` to point
at one built elsewhere. The equivalent manual invocation to build the Rust backend is:

```bash
cargo build --release --manifest-path src/mandacaru/integrals/rust/Cargo.toml
```

Or for the C backend:

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
| `memory` | `psutil`, for the resident-memory line of the `[PERFORMANCE]` block (it falls back to the standard library). |
| `dispersion` | `dftd4`, for the D4 correction of `Mandacaru(method="dft", vdw="d4")`. |
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
