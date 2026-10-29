# Installation

Mandacaru requires Python 3.14 or later. The package installs its scientific Python dependencies and quantum SDKs,
and includes its Rust extension in a platform wheel.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install mandacaru
```

On Windows use `.venv\Scripts\activate`. Installed wheels need no compiler
at calculation time. Building a source checkout requires Rust/Cargo:

```bash
python -m pip install -e ".[dev]"
```

Reinstall the editable package after changing Rust code; Python edits alone do
not rebuild the extension. Package construction is the only compilation step.
Force a rebuild when the package version has not changed:

```bash
python -m pip install --force-reinstall --no-deps -e .
# With uv:
uv pip install --reinstall-package mandacaru --no-deps -e .
```

## Required Python dependencies

A standard installation includes NumPy, SciPy, threadpoolctl, ASE, spglib,
Matplotlib, pandas, fastparquet, Qiskit, Qiskit IBM Runtime, Amazon Braket SDK,
Cirq, pytest and psutil. Documentation and development tools are extras.

The scientific algorithms execute in Rust. Python integration packages remain
required for the distributed user interface, examples, plotting, persistence
and external SDK workflows. The core imports these integrations only when used.

Attach the calculator directly with `atoms.calc = Mandacaru(...)`. Standalone
`Geometry` values are also supported. No execution-engine option is provided.

## PAW datasets

Periodic PAW geometry assembly currently uses a Python setup adapter to the
Rust crystal solver.
Set `MANDACARU_PAW_PATH` to a `mandacaru-paw` checkout, or
`MANDACARU_UPAW_PATH` to a `mandacaru-upaw` checkout. Dataset selection belongs
to the basis specification; datasets do not ship in the base wheel.
Molecular PAW HF, semilocal Kohn–Sham and variational calculations, and the
stationary RHF/UHF/Kohn–Sham forces, use Rust for sampling, integral
assembly, overlap augmentation, compensation multipoles and exact valence
one-center interactions. Frozen-core exchange currently supports an empty core
or a single closed 1s shell. Python still prepares the PAW basis radial tables.
Molecular PAW screened exchange, spin-orbit projectors, GHF forces and
multi-shell core exchange remain unavailable.

## Verification

```bash
python scripts/check_architecture.py
python -m pytest test/architecture
cargo test --release --manifest-path Cargo.toml --workspace
python -m sphinx -W --keep-going -b html docs/source /tmp/mandacaru-docs
```

Local scripts validate the installed wheel and the ASE adapter. A separate
smoke check blocks integration imports to verify the native calculation boundary. No GitHub workflows are used.
The [development guide](development/architecture.md) explains the contracts.
