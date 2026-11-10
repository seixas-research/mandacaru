"""Freeze the Python Ewald sums before the module is deleted (MG-6).

Run once, on the Python route, from the repository root: ``python test/migration/freeze_ewald.py``.
It writes ``test/migration/reference/ewald.json``, which ``test_native_periodic.py`` reads.

Two point charges in an orthogonal and in a skewed cell, with the automatic splitting parameter
and two fixed ones: the energy, the forces, the stress and the potential at two points with a
softened short-range kernel (Hartree atomic units).
"""
import json
from pathlib import Path

import numpy as np

from mandacaru.core.ewald import ewald_energy, ewald_forces, ewald_potential, ewald_stress

OUT = Path(__file__).parent / "reference" / "ewald.json"
SOFTENING = 0.2


def case(skew, eta):
    cell = np.diag([6., 7., 8.])
    if skew:
        cell += np.array([[0., .3, .1], [.2, 0., .4], [.1, .2, 0.]])
    positions = np.array([[.7, 1., 1.2], [2.1, 3., 3.1]])
    charges = np.array([1., -2.])
    points = np.array([[.3, 2., 1.1], [3.3, 1., .9]])
    return {
        "skew": skew, "eta": eta, "cell": cell.tolist(), "positions": positions.tolist(),
        "charges": charges.tolist(), "points": points.tolist(), "softening": SOFTENING,
        "energy": float(ewald_energy(positions, charges, cell, eta=eta)),
        "forces": np.asarray(ewald_forces(positions, charges, cell, eta=eta)).tolist(),
        "stress": np.asarray(ewald_stress(positions, charges, cell, eta=eta)).tolist(),
        "potential": np.asarray(ewald_potential(positions, charges, cell, points, eta=eta,
                                                softening=SOFTENING)).tolist(),
    }


def main():
    cases = [case(skew, eta) for skew in (False, True) for eta in (None, .18, .45)]
    OUT.write_text(json.dumps({"cases": cases}))
    print(len(cases), "cases", OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
