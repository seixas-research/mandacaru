"""The absorption spectrum of H2 from a real-time propagation after a delta kick, against the
linear-response (Casida) excitations of the same state."""
from pathlib import Path

import numpy as np
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="dft", xc="lda",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25)
atoms.get_potential_energy()

# A weak kick along the bond, then 6 fs in steps of 0.01 fs (ETRS).
trace = atoms.calc.get_real_time(duration=6.0, time_step=0.01, direction="z")
print(f"{trace.steps} steps, {trace.builds} Fock builds, energy drift "
      f"{trace.energy_drift:.2e} kick energies, dt W = {trace.step_width:.2f}")

energies = np.linspace(5.0, 40.0, 701)
spectrum = trace.strength_function(energies, damping=0.5)
peak = energies[int(np.argmax(spectrum))]

casida = atoms.calc.get_excitations(5, approximation="casida")
bright = int(np.argmax(casida.energies * casida.transition_dipoles[:, 2] ** 2))
print(f"Strongest line of the real-time spectrum: {peak:.2f} eV")
print(f"Brightest Casida root along the bond:     {casida.energies[bright]:.2f} eV")

out = Path(__file__).parent / "data" / "04_real_time_h2.csv"
np.savetxt(out, np.column_stack([energies, spectrum]), delimiter=",",
           header="energy_eV,strength_per_eV", comments="")
print(f"Wrote {out}")
