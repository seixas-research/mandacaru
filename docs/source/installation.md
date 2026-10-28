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

The Python package provides ASE integration, arrays, visualization and service
adapters. Native calculations use the packaged Rust extension. A compiler is
needed to build the extension from source, not to execute an installed native
calculation. The introductory examples run locally without credentials.

## Install a source checkout

Install Rust with Cargo before building a source checkout. The crates declare
a minimum Rust version of 1.85. Use this route to edit the code or documentation:

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

## Native engine

Use `engine="rust"` on the ASE calculator. `engine_options` configures its owned
thread pool and memory limit; see [native execution](guide/native_engine.md).
The extension is built by the package build hook. It must be rebuilt after a
change to Rust source, even with an editable Python installation:

```bash
python -m pip install --no-deps -e .
python -c "from mandacaru import _native; print(_native.Context(threads=1))"
```

Standalone core checks do not import Python:

```bash
cargo test --offline --manifest-path src/mandacaru/integrals/rust/Cargo.toml \
  --package mandacaru-core --package mandacaru_integrals
```

Remove `--offline` if the Cargo dependencies have not yet been downloaded.
The native extension reports missing components and invalid input explicitly;
there is no runtime compiler or numerical fallback on this path.

## Optional dependencies

| Extra | Purpose |
| :--- | :--- |
| `docs` | Sphinx, MyST and Furo for building this manual. |
| `coverage` | Test coverage reporting with pytest-cov. |
| `dev` | Documentation and additional validation dependencies. |

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
