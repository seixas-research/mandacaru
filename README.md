<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<img src="https://raw.githubusercontent.com/seixas-research/mandacaru/refs/heads/main/logo/logo_light.png#gh-light-mode-only" alt="Mandacaru logo" height="200">
<img src="https://raw.githubusercontent.com/seixas-research/mandacaru/refs/heads/main/logo/logo_dark.png#gh-dark-mode-only" alt="Mandacaru logo" height="200">
</h1>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-red?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/Python-3.14+-fcbc2c.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/mandacaru.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/mandacaru/)
[![Documentation Status](https://img.shields.io/readthedocs/mandacaru?style=for-the-badge&logo=readthedocs&logoColor=white&label=Docs)](https://mandacaru.readthedocs.io/en/latest/)
[![GitHub](https://img.shields.io/badge/GitHub-mandacaru-181717?style=for-the-badge&logo=github)](https://github.com/seixas-research/mandacaru)

</div>

# Mandacaru

Mandacaru provides a Rust scientific core with a Python interface for electronic
structure and fermionic quantum simulation. Rust owns molecular HF/DFT,
variational optimization, scientific buffers and numerical kernels. Python
provides input values, result views, ASE integration and external service adapters.
There is no Python calculation engine or execution-engine selector.

## Install and run

The base package requires Python 3.14 and NumPy. Platform wheels include the
native extension. Building from source also requires Rust/Cargo.

```bash
python -m pip install mandacaru
```

```python
import numpy as np
from mandacaru import Geometry, Mandacaru

geometry = Geometry((1, 1), np.array([[2.5,2.5,2.13], [2.5,2.5,2.87]]), np.eye(3)*5)
with Mandacaru(method="dft", xc="pbe", grid_spacing=.35,
               resources={"threads": 2}) as calculation:
    result = calculation.run(geometry)
    print(result.optimal_energy)  # eV
    print(result.forces())       # eV/angstrom
```

Install `mandacaru[ase]` for the optional
`mandacaru.integrations.ase.ASECalculator` adapter. Molecular calculations
accept standalone geometry values and do not require ASE.

Four-index Hamiltonians are built only when explicitly requested through
`result.hamiltonian`, Hamiltonian export or a quantum calculation. HF and
screened hybrid exchange use native Cholesky factors.

## Source layout

| Location | Responsibility |
| --- | --- |
| `Cargo.toml`, `Cargo.lock` | Native workspace and locked dependencies |
| `crates/mandacaru-core` | Basis, models, molecular/periodic solvers, PAW and quantum algorithms |
| `crates/mandacaru-kernels` | Low-level numerical kernels |
| `crates/mandacaru-python` | PyO3 bindings and checked NumPy transfers |
| `src/mandacaru` | Python interface, workflows, result views and integrations |
| `build_hooks/native.py` | Native extension build hook |
| `scripts` | Local architectural and installation checks |
| `docs` | Sphinx manual |

There are no Rust sources inside the Python package. No GitHub workflows are used.

## Local checks

```bash
python scripts/check_architecture.py
cargo test --release --workspace
python -m pytest test/architecture
python -m sphinx -W --keep-going -b html docs/source /tmp/mandacaru-docs
```

Rebuild the editable extension after changing Rust:

```bash
python -m pip install --no-deps -e .
```

The [manual](https://mandacaru.readthedocs.io/en/latest/) describes native
coverage, scientific conventions and remaining migration work. Unsupported
models raise errors and do not select a Python numerical fallback.
