<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/mandacaru" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/mandacaru/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/mandacaru/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/mandacaru/refs/heads/main/logo/logo_light.png" alt="Mandacaru logo" style="height: auto; width: auto; max-height: 100px;">
  </picture>
</a>
</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-red?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/Python-3.14+-fcbc2c.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/mandacaru.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/mandacaru/)
[![Documentation Status](https://readthedocs.org/projects/mandacaru/badge/?version=latest&style=for-the-badge&logo=readthedocs&logoColor=white)](https://mandacaru.readthedocs.io/en/latest/?badge=latest)

# Mandacaru

**Mandacaru** is a Python framework for fermionic quantum simulation with variational quantum algorithms. From an ASE geometry it builds a real-space Hamiltonian, maps it to qubits, and solves it with VQE or ADAPT-VQE on a state-vector simulator or on quantum hardware (IBM Quantum, Amazon Braket).

## Installation

```bash
pip install mandacaru

# PAW datasets (kept in a separate repository because of their size)
git clone https://github.com/seixas-research/mandacaru-paw.git
mandacaru --link-paw mandacaru-paw
```

## LiH with ASE

```python
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.optimizers import Optimizer

atoms = Atoms("LiH",
              positions=[[0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.6]],
              cell=[10.0, 10.0, 10.0])
atoms.center()                                      # the cell is the real-space box

atoms.calc = Mandacaru(method="adapt-vqe",                   # "vqe" | "adapt-vqe" | "subspace-vqe" | "subspace-adapt-vqe"
                       basis={"name": "PAW",
                              "size": "DZP",
                              "energy_shift": 0.1},          # pseudopotential family + valence basis
                       h=0.10,                               # grid spacing (Å)
                       pool="fermionic",                     # "fermionic" | "qubit" | "qeb" | "ceo" | "ceo-ovp"
                       mapping="jordan_wigner",              # "jordan_wigner" | "parity" | "parity_reduced" | "bravyi_kitaev"
                       optimizer=Optimizer(method="SLSQP",   # or SPSA | COBYLA | Nelder-Mead | Adam | L-BFGS-B
                                           maxiter=2000,
                                           tol=1e-12),
                       max_iterations=300,                   # at most 300 operators
                       gradient_tolerance=1e-3,              # stop when every pool gradient is smaller
                       device="AER_simulator",               # or an IBM Quantum / Amazon Braket device
                       shots=0,                              # 0 = exact expectation values
                       verbose_operators=False,              # True -> the pool to pool.json
                       verbose_hamiltonian=False)            # True -> hamiltonian.inspect.json

forces = atoms.get_forces()                         # eV/Å, runs the simulation
energy = atoms.get_potential_energy()               # eV, from the same run
result = atoms.calc.result

print(f"E = {energy:.4f} eV with {result.num_operators} operators")
print(f"F(Li) = {forces[0, 2]:+.3f} eV/Å along the bond")
```

## Potential energy surface

```python
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.optimizers import Optimizer

distances = np.linspace(1.2, 3.0, 10)
energies = []
for d in distances:
    atoms = Atoms("LiH",
                  positions=[[0.0, 0.0, 0.0],
                             [0.0, 0.0, d]],
                  cell=[10.0, 10.0, 10.0])
    atoms.center()

    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW",
                                  "size": "DZP",
                                  "energy_shift": 0.1},
                           h=0.10,
                           pool="fermionic",
                           optimizer=Optimizer(method="L-BFGS-B",
                                               maxiter=2000,
                                               tol=1e-12))
    energies.append(atoms.get_potential_energy())

plt.plot(distances, energies, "o-")
plt.xlabel("Li–H distance (Å)")
plt.ylabel("Energy (eV)")
plt.savefig("lih_pes.png", dpi=150)
```

## Theory

**VQE.** The variational quantum eigensolver prepares a parameterized state |ψ(θ)⟩ = U(θ)|Φ<sub>HF</sub>⟩ on a quantum processor, measures the energy ⟨ψ(θ)|H|ψ(θ)⟩, and lets a classical optimizer update θ to minimize it. By the variational principle the minimum is an upper bound to the ground-state energy, reached exactly when the ansatz can represent the ground state. Mandacaru starts from the Hartree–Fock determinant in the molecular-orbital basis; the fixed ansatz of `method="vqe"` is UCCSD.

**ADAPT-VQE.** ADAPT-VQE builds the ansatz during the calculation instead of fixing it in advance. At each iteration it evaluates the energy gradient ⟨ψ|[H, A<sub>k</sub>]|ψ⟩ of every generator A<sub>k</sub> in an operator pool, appends exp(θ<sub>k</sub>A<sub>k</sub>) for the largest one, and re-optimizes all parameters. It stops when every gradient falls below `gradient_tolerance`, producing compact circuits tailored to the molecule.

**Operator pools.** The pool is the set of anti-Hermitian generators ADAPT-VQE chooses from, and it sets the trade-off between circuit depth and the number of iterations. `fermionic` holds spin-adapted single and double excitations; `qubit` splits them into individual Pauli strings (the shallowest gates, more iterations); `qeb` uses qubit excitations — the same occupation moves without the fermionic sign; `ceo` couples the qubit excitations that act on the same spin-orbitals, and `ceo-ovp` keeps that coupling to one parameter per step, roughly halving the two-qubit gate count of `qeb`. Every pool is built in the encoding you ask for (Jordan–Wigner, parity, reduced parity or Bravyi–Kitaev) and reaches the same ground state. The fermionic and qubit-excitation pools conserve the particle number; the individual Pauli strings of `qubit` do not, by design.

**Classical optimization.** The parameters are updated by the optimizer in `optimizer=` — a method name, or an `Optimizer` with an explicit iteration budget and tolerance. SLSQP (the default) and L-BFGS-B use gradients and stop in one to two orders of magnitude fewer steps on exact simulators; Nelder–Mead and COBYLA are gradient-free and more robust on a small, nearly-converged problem; SPSA (two energy evaluations per step, whatever the number of parameters) and Adam tolerate the statistical noise of shot-based hardware. Both costs of a run — optimizer steps and energy evaluations — are reported per growth step and in total; see [the guide](https://mandacaru.readthedocs.io/en/latest/guide/optimizers.html).

## License

Mandacaru is released under the [MIT License](LICENSE).

Developer: **Leandro Seixas Rocha** (<leandro.rocha@ilum.cnpem.br>).

Documentation: [mandacaru.readthedocs.io](https://mandacaru.readthedocs.io/).

We thank financial support from [INCT Materials Informatics](https://inct-mi.com.br/) (Grant No. 406447/2022-5).
