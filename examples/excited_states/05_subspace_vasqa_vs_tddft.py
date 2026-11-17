"""Compare excited states of H2 between Subspace-VASQA and linear-response TD-DFT.

Matches spin symmetry (<S^2>), vertical excitation energies, and dipole transition
observables within the same PAW-LCAO SZ basis and geometry.
"""

from ase.build import molecule
import numpy as np
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)

# 1. Subspace-VASQA (lowest 2 correlated states via weighted ensemble Markov search + Ritz)
vasqa_calc = Mandacaru(
    method="subspace-vasqa",
    basis={"name": "PAW-LCAO", "size": "SZ"},
    grid_spacing=0.25,
    pool="fermionic",
    num_states=2,
    weights=[0.7, 0.3],
    max_steps=40,
    max_length=4,
    seed=7,
    trace=False,
)
vasqa_res = vasqa_calc.run(atoms)

# 2. Adiabatic LR-TD-DFT (spin-conserving singlet excitations)
dft_calc = Mandacaru(
    method="dft",
    xc="pbe",
    basis={"name": "PAW-LCAO", "size": "SZ"},
    grid_spacing=0.25,
    trace=False,
)
dft_calc.run(atoms)
# Minimal-basis H2 (SZ) has 1 occupied -> 1 virtual 1p-1h transition in LR-TDDFT,
# whereas correlated Subspace-VASQA spans both singly and doubly excited roots.
exc = dft_calc.get_excitations(1, spin="singlet")

print("=== Subspace-VASQA vs TD-DFT (H2, PAW-LCAO SZ) ===")
sector_states = vasqa_res._raw["states"]
for i, energy in enumerate(vasqa_res.energies):
    s2 = vasqa_calc.problem.spin_squared(sector_states[i])
    gap = vasqa_res.excitation_energies[i]
    print(f"VASQA root {i}: E = {energy:10.6f} eV, dE = {gap:8.4f} eV, <S^2> = {s2:.4f}")

one_tdm, _ = vasqa_calc.problem.transition_rdms(
    sector_states[0], sector_states[1]
)
print(f"VASQA ||1-TDM(0->1)||_F = {np.linalg.norm(one_tdm):.6f}")

for i, gap in enumerate(exc.energies):
    f_osc = exc.oscillator_strengths[i]
    print(f"TD-DFT root {i + 1}: dE = {gap:8.4f} eV, f_osc = {f_osc:.6f}")
