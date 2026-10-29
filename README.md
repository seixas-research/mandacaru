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

Mandacaru requires Python 3.14 and installs its scientific Python stack and
quantum SDK integrations, including ASE, SciPy, Matplotlib, Parquet support,
Qiskit, Braket and Cirq. Platform wheels include the native extension. Building
from source also requires Rust/Cargo.

```bash
python -m pip install mandacaru
```

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5])
atoms.center()
with Mandacaru(method="dft", xc="pbe", basis="HAO", grid_spacing=.35,
               resources={"threads": 2}) as calculation:
    atoms.calc = calculation
    print(atoms.get_potential_energy())  # eV
    print(atoms.get_forces())            # eV/angstrom
```


Attach the calculator directly with `atoms.calc = Mandacaru(...)` for ASE
calculations. Standalone geometry values are also accepted. Required integration
packages are imported only by the Python features that use them.

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
