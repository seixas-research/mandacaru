"""Nested OTOCs of the three-dimensional H2 ADAPT-VQE ground state.

Run with: conda run -n mandacaru python examples/nested_otoc.py
The standard report is output.txt; the signed OTOC FFT is otoc_spectrum.csv.
Its frequencies are generally sums of gaps, not optical excitation energies.
"""

from pathlib import Path

import numpy as np

from mandacaru.algorithms import NestedOTOC
from mandacaru.core import PauliSum
from mandacaru.utils import append_nested_otoc
from quantum_echoes import build_h2_problem


def main(txt: str = "output.txt") -> None:
    """Reuse the molecular basis and prepared state of the dipole example."""
    atoms, _, _, _ = build_h2_problem(txt=txt)
    prepared = atoms.calc.solver.checkpoint
    n = prepared.n_qubits
    # Orbital occupation parities: number-conserving Pauli insertions in the
    # same Jordan-Wigner basis. These probe electronic correlations, not a
    # calibrated electric dipole response.
    b_label, m_label = "I"*(n-1) + "Z", "Z" + "I"*(n-1)
    otoc = NestedOTOC(prepared.hamiltonian, PauliSum({b_label: 1}),
                      PauliSum({m_label: 1}), trotter_order=2)
    results = [otoc.run(prepared, float(t), otoc_order=k, steps=40)
               for t in np.linspace(0, 4, 9) for k in (1, 2)]
    spectrum = otoc.spectrum(prepared, time_step=0.2, num_samples=128,
                             steps_per_sample=2, otoc_order=2)
    destination = Path(txt).with_name("otoc_spectrum.csv")
    for target in atoms.calc.solver.log_targets:
        append_nested_otoc(target, results, trotter_order=2, steps=40,
                           butterfly=b_label, measurement=m_label,
                           spectrum=spectrum, spectrum_steps_per_sample=2,
                           spectrum_path=destination)
    print(f"H2 ADAPT-VQE nested OTOCs: {txt}")
    print(f"C^(2)(4) = {results[-2].correlator:.6g}")
    print(f"C^(4)(4) = {results[-1].correlator:.6g}")
    print(f"Signed OTOC Fourier spectrum: {destination}")
    print("OTOC frequencies are combinations of energy gaps, not excitation energies.")


if __name__ == "__main__":
    main()
