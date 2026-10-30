(installation)=
# Installation

Mandacaru requires Python 3.14 or later. The numerical engine is a Rust
extension that the install compiles, so a source install also needs a Rust
toolchain (Rust 1.85 or later, with `cargo` on the `PATH`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

The `dev` extra adds the documentation tools (Sphinx, MyST, Furo) and
`pytest-cov`; `.[docs]` adds only the documentation tools. Once the
extension is built, calculations compile nothing.

A change under `crates/` is not seen by Python until the extension is rebuilt;
Python edits alone do not rebuild it. Rerun the install after changing Rust
code, or force the rebuild when the package version has not changed:

```bash
python -m pip install --force-reinstall --no-deps -e .
# With uv:
uv pip install --reinstall-package mandacaru --no-deps -e .
```

Check the install:

```bash
mandacaru --version
python -c "from mandacaru import Mandacaru; print(Mandacaru(method='rhf').method)"
```

## Python dependencies

The install pulls in NumPy, SciPy, threadpoolctl, ASE, spglib, Matplotlib,
Qiskit, Qiskit Aer, Qiskit IBM Runtime, the Amazon Braket SDK, Cirq, pytest and
psutil. The scientific algorithms run in Rust; the Python packages serve the ASE
interface, plotting and the quantum SDKs, and the core imports them only when they
are used. Aer is required for the IBM fake devices, which take their noise models
from it.

Parquet files need no Python package. The native engine reads and writes the
Hamiltonian caches and the pseudopotential datasets itself, so `pandas`,
`pyarrow` and `fastparquet` are neither declared nor imported, and the `engine=`
and `codec=` keywords that once chose between them no longer exist. Any of them
may still be installed for your own work, for example to inspect a cache; Cirq,
for one, pulls in `pandas`. The file rules (what raises, which codecs, what a
new file looks like) are in [Parquet files](#native-parquet).

(installation-paw)=
## PAW-LCAO datasets

No pseudopotential data ships with the package. The PAW-LCAO datasets live in
the `mandacaru-paw` repository; clone it and tell Mandacaru where it is:

```bash
mandacaru-paw --set /path/to/mandacaru-paw   # writes MANDACARU_PAW_PATH to your shell profile
mandacaru-paw --status                       # one row per dataset set that is reachable
```

Open a new terminal (or `source` the profile) for the variable to take effect;
in a non-interactive shell, export `MANDACARU_PAW_PATH` yourself. A calculation
that needs an unset library raises `LibraryPathError` naming the command that
fixes it. The `HAO`, `NAO`, `NAO-AE` and named Gaussian bases read no data and
need none of this. The library layout and the optional UPAW-LCAO library are in
[PAW-LCAO bases](#paw-lcao-libraries).

## Verification

```bash
python -m pytest test/test_docs_links.py     # the manual against the code
python scripts/check_architecture.py         # crate and package boundaries
cargo test --release                         # Rust tests (default workspace members)
make -C docs html SPHINXOPTS="-W"            # the manual; warnings are errors
```

The suite runs locally; the repository has no hosted workflows. Contributors
should read [the architecture page](development/architecture.md) for the
contracts between the Python interface, the PyO3 boundary and the Rust crates.
