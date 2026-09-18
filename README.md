<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/carcara" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_light.png" alt="Carcará logo" style="height: auto; width: auto; max-height: 100px;">
  </picture>
</a>
</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-red?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/carcara.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/carcara/)
[![Documentation Status](https://readthedocs.org/projects/carcara/badge/?version=latest&style=for-the-badge&logo=readthedocs&logoColor=white)](https://carcara.readthedocs.io/en/latest/?badge=latest)

# Carcará

**Carcará** is a Python framework for fermionic quantum simulation with variational quantum algorithms. From an ASE geometry it builds a real-space Hamiltonian — with all-electron basis sets or NCPP / ONCVPSP / PAW pseudopotentials — maps it to qubits, and solves it with VQE or ADAPT-VQE on a state-vector simulator or on quantum hardware (IBM Quantum, Amazon Braket). Energies are reported in eV and distances in Å.

**Latest updates**

- **Forces with PAW + DZP.** Hellmann–Feynman and Pulay forces for `basis={"name": "PAW", "size": "DZP"}` — the augmented overlap, the projectors and the compensation charges are all differentiated — through `atoms.get_forces()`.
- **Particle-number sectors.** 20-qubit problems such as LiH in PAW-DZP are solved exactly in their (n<sub>α</sub>, n<sub>β</sub>) sector: 100 states instead of 2<sup>20</sup>.
- **Real molecular orbitals.** Orbitals with l > 0 are rotated to real form, so the operator pools reach the exact ground state.
- **Wavefunction checkpoints.** `Carcara(..., checkpoint="state.json")` writes the reference, the generators, the angles and the Hamiltonian after every accepted operator; `resume="state.json"` continues an interrupted or unconverged run where it stopped.
- **Virtual orbitals for the FAO basis.** `basis={"name": "FAO", "virtual_orbitals": 1}` appends the lowest unoccupied atomic levels (H gains 2s, C gains 3s), giving a correlated method room above the occupied orbitals; the default `0` is the minimal basis as before.
- **Quantum phase estimation.** `QuantumPhaseEstimation(n_evaluation_qubits=10).run("state.json")` reads the exact eigenvalue off the checkpointed state, with a memory estimate checked before the 2<sup>n+t</sup> state vector is allocated.

## Installation

```bash
pip install carcara

# PAW datasets (kept in a separate repository because of their size)
git clone https://github.com/seixas-research/carcara-paw.git
python -m carcara.pseudopotentials.link_library --paw carcara-paw
```

## LiH with ASE

```python
from ase import Atoms
from carcara import Carcara

atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], cell=[10.0, 10.0, 10.0])
atoms.center()                                      # the cell is the real-space box

atoms.calc = Carcara(method="adapt-vqe",                   # "vqe", "subspace-vqe", "subspace-adapt-vqe"
                     basis={"name": "PAW", "size": "DZP"}, # pseudopotential family + valence basis
                     h=0.25,                               # grid spacing (Å)
                     pool="fermionic",                     # "qubit", "qeb", "ceo"
                     mapping="jordan_wigner",              # "parity", "bravyi_kitaev"
                     optimizer="L-BFGS-B",                 # "COBYLA", "SLSQP", "Nelder-Mead", "SPSA", "Adam"
                     max_iterations=80,                    # at most 80 operators
                     gradient_tolerance=1e-5,              # stop when every pool gradient is smaller
                     device="AER_simulator",               # or an IBM Quantum / Amazon Braket device
                     shots=0,                              # 0 = exact expectation values
                     verbose=True,                         # iteration table and timings
                     verbose_operators=False,              # True -> the pool to pool.json
                     verbose_hamiltonian=False)            # True -> the Hamiltonian to hamiltonian.json

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
from carcara import Carcara

distances = np.linspace(1.2, 3.0, 10)               # Å
energies = []
for d in distances:
    atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, d]], cell=[10.0, 10.0, 10.0])
    atoms.center()
    atoms.calc = Carcara(method="adapt-vqe",
                         basis={"name": "PAW", "size": "DZP"},
                         h=0.25,
                         pool="fermionic",
                         optimizer="L-BFGS-B",
                         verbose=False)
    energies.append(atoms.get_potential_energy())

plt.plot(distances, energies, "o-")
plt.xlabel("Li–H distance (Å)")
plt.ylabel("Energy (eV)")
plt.savefig("lih_pes.png", dpi=150)
```

## Theory

**VQE.** The variational quantum eigensolver prepares a parameterized state |ψ(θ)⟩ = U(θ)|Φ<sub>HF</sub>⟩ on a quantum processor, measures the energy ⟨ψ(θ)|H|ψ(θ)⟩, and lets a classical optimizer update θ to minimize it. By the variational principle the minimum is an upper bound to the ground-state energy, reached exactly when the ansatz can represent the ground state. Carcará starts from the Hartree–Fock determinant in the molecular-orbital basis; the fixed ansatz of `method="vqe"` is UCCSD.

**ADAPT-VQE.** ADAPT-VQE builds the ansatz during the calculation instead of fixing it in advance. At each iteration it evaluates the energy gradient ⟨ψ|[H, A<sub>k</sub>]|ψ⟩ of every generator A<sub>k</sub> in an operator pool, appends exp(θ<sub>k</sub>A<sub>k</sub>) for the largest one, and re-optimizes all parameters. It stops when every gradient falls below `gradient_tolerance`, producing compact circuits tailored to the molecule.

**Operator pools.** The pool is the set of anti-Hermitian generators ADAPT-VQE chooses from, and it sets the trade-off between circuit depth and the number of iterations. `fermionic` holds spin-adapted single and double excitations; `qubit` splits them into individual Pauli strings (the shallowest gates, more iterations); `qeb` uses qubit excitations — the same occupation moves without the fermionic sign; `ceo` groups QEB generators that share a qubit support. Every pool is built in the encoding you ask for (Jordan–Wigner, parity or Bravyi–Kitaev) and reaches the same ground state. The fermionic and qubit-excitation pools conserve the particle number; the individual Pauli strings of `qubit` do not, by design.

**Classical optimization.** The parameters are updated by the optimizer named in `optimizer=`. COBYLA (the default) and Nelder–Mead are gradient-free and robust; L-BFGS-B and SLSQP use gradients and converge quickly on exact simulators; SPSA (two energy evaluations per step, whatever the number of parameters) and Adam tolerate the statistical noise of shot-based hardware.

## License

Carcará is released under the [MIT License](LICENSE). Developer: **Leandro Seixas Rocha** (<leandro.rocha@ilum.cnpem.br>). Documentation: [carcara.readthedocs.io](https://carcara.readthedocs.io/).

We thank financial support from [INCT Materials Informatics](https://inct-mi.pesquisa.ufabc.edu.br/) (Grant No. 406447/2022-5).
