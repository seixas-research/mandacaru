"""Triplet O2 with spin-polarized ADAPT-VQE: the spin comes from the magnetic moments."""
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.208]],
              magmoms=[1.0, 1.0])
atoms.center(vacuum=3.4)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       h=0.25,
                       pool="fermionic",
                       active_space={"frozen": [0, 1, 2, 3, 4]},
                       max_iterations=12,
                       convergence={"gradient": 1e-3})

energy = atoms.get_potential_energy()
n_alpha, n_beta = atoms.calc.num_particles
print(f"ADAPT-VQE energy: {energy:.6f} eV on {atoms.calc.n_qubits} qubits")
print(f"Electrons: {n_alpha} alpha, {n_beta} beta (2Sz = {n_alpha - n_beta})")
