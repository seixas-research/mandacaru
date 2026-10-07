"""How the expressibility of the LiH ADAPT-VQE ansatz grows with each operator."""
import numpy as np
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.algorithms.expressivity import (active_space_dimension,
                                               calculate_kl_divergence,
                                               sample_pqc_fidelities)

atoms = molecule("LiH")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.25)
atoms.get_potential_energy()

calc = Mandacaru(method="adapt-vqe", pool="qeb", max_iterations=10,
                 **atoms.calc.result.as_quantum_problem())
dimension = active_space_dimension(calc.n_qubits, calc.num_particles)
rng = np.random.default_rng(11)


def score(info):
    """Kullback-Leibler distance to Haar of the ansatz grown so far (lower is more expressive)."""
    fidelities = sample_pqc_fidelities(info["ansatz"], 3000, rng)
    kl = calculate_kl_divergence(fidelities, calc.n_qubits, 60, dimension)
    print(f"{info['num_operators']:3d} operators  E = {info['energy']:.6f} eV  "
          f"expressibility = {kl:.4f}")


calc.run(callback=score)
