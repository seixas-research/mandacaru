# -*- coding: utf-8 -*-
# file: examples/37_HVA_H2.py

# This code is part of Mandacaru.
# MIT License

"""Compare exact-group and circuit-compatible HVA on the same H2 model.

Both calculations reuse the RHF molecular-orbital Hamiltonian.  The finite
product formula is optimized as its own ansatz, then exported as a checkpoint
whose state vector and circuit describe that same finite-step state.
"""

from pathlib import Path

from ase import Atoms

from mandacaru import Mandacaru


def main() -> None:
    """Run RHF and two HVA evolution policies without a hardware submission."""
    data = Path(__file__).resolve().parent / "data"
    data.mkdir(exist_ok=True)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6] * 3)
    atoms.calc = Mandacaru(method="rhf", basis="HAO", h=0.35, trace=False)
    rhf_energy = atoms.get_potential_energy()
    problem = atoms.calc.result.as_quantum_problem()

    exact = Mandacaru(method="hva", layers=2, trace=False, **problem)
    exact_energy = exact.run().optimal_energy

    circuit = Mandacaru(
        method="hva", layers=2, evolution="trotter", order=2, steps=2,
        checkpoint=str(data / "h2_hva.json"), trace=False, **problem)
    circuit_energy = circuit.run().optimal_energy

    print(f"RHF:              {rhf_energy:+.8f} eV")
    print(f"HVA exact groups: {exact_energy:+.8f} eV")
    print(f"HVA circuit:      {circuit_energy:+.8f} eV")
    print(f"Circuit checkpoint: {data / 'h2_hva.json'}")


if __name__ == "__main__":
    main()
