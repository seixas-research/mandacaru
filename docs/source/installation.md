# Installation

Mandacaru requires Python 3.14 or later. The base package depends only on NumPy
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

## Optional integrations

| Extra | Purpose |
| --- | --- |
| `ase` | ASE Calculator adapter and geometry file readers |
| `periodic` | ASE, PAW setup, symmetry and columnar datasets |
| `visualization` | Matplotlib plots |
| `columnar` | Parquet persistence adapters |
| `qiskit`, `braket`, `cirq` | External SDK integrations |
| `docs` | Sphinx documentation tools |
| `dev` | Scientific tests and documentation tools |

```bash
python -m pip install "mandacaru[ase]"
```

ASE is optional for molecular calculations and cached crystal calculations.
See [native execution](guide/native_engine.md) for a standalone `Geometry`
example. No execution-engine option or runtime numerical fallback is provided.

## PAW datasets

Periodic PAW geometry assembly remains an optional Python integration.
Set `MANDACARU_PAW_PATH` to a `mandacaru-paw` checkout, or
`MANDACARU_UPAW_PATH` to a `mandacaru-upaw` checkout. Dataset selection belongs
to the basis specification; datasets do not ship in the base wheel.
The native molecular geometry interface currently supports all-electron bases;
molecular PAW setup has not been migrated to that interface.

## Verification

```bash
python scripts/check_architecture.py
python -m pytest test/architecture
cargo test --release --manifest-path Cargo.toml --workspace
python -m sphinx -W --keep-going -b html docs/source /tmp/mandacaru-docs
```

Local scripts build and validate the wheel with only NumPy and pytest, then
verify the optional ASE adapter separately. No GitHub workflows are used.
The [development guide](development/architecture.md) explains the contracts.
