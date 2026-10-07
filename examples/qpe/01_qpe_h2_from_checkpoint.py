"""H2: a checkpointed ADAPT-VQE state as the input of quantum phase estimation."""
import os
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.algorithms import QuantumPhaseEstimation
from mandacaru.core import load_checkpoint

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
CHECKPOINT = os.path.join(DATA, "h2_wavefunction.json")

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       h=0.25,
                       pool="fermionic",
                       checkpoint=CHECKPOINT)
print(f"ADAPT-VQE: {atoms.get_potential_energy():.6f} eV")

checkpoint = load_checkpoint(CHECKPOINT)
window = (checkpoint.energy - 0.02, checkpoint.energy + 0.02)       # Hartree
result = QuantumPhaseEstimation(n_evaluation_qubits=10, energy_window=window).run(checkpoint)
print(f"QPE:       {result.energy:.6f} eV  (resolution {result.resolution:.1e} eV, "
      f"probability {result.probability:.3f})")
