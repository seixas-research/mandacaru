"""Quantum echoes of the H2 ADAPT-VQE ground state and the spectrum of their correlation."""
import numpy as np
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.algorithms import QuantumEchoes
from mandacaru.core import PauliSum

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25)
atoms.get_potential_energy()

prepared = atoms.calc.solver.checkpoint
echoes = QuantumEchoes(prepared.hamiltonian, PauliSum({"ZIII": 1.0}))
for time in (0.0, 1.0, 2.0, 4.0):
    result = echoes.run(prepared, time, tau_p=0.01, steps=40)
    print(f"t = {time:.1f}  fidelity = {result.fidelity:.10f}  "
          f"response = {result.response:+.3e} Ha")

spectrum = echoes.spectrum(prepared, time_step=0.5, num_samples=1024)
print("Peaks (eV):", np.round(spectrum.energies_ev[spectrum.peaks()], 4))
