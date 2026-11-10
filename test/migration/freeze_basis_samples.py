"""Freeze the all-electron basis families' sampled values before their Python is deleted (MG-6).

Run on the Python route, from the repository root, in two passes. It writes
``test/migration/reference/basis_samples.json``, which ``test_native_basis_samples.py`` reads.

1. ``python test/migration/freeze_basis_samples.py`` freezes everything but the contracted Gaussian
   families (the hydrogenic, numerical and all-electron numerical families).
2. ``PYTHONPATH=<tree>/src python test/migration/freeze_basis_samples.py --gaussian`` adds the
   Gaussian families, on the Python that fitted its contractions with SciPy: ``<tree>`` is
   ``git archive d48efe7^ src/mandacaru`` (v26.11.8) with the current extension copied beside it.
   The Rust engine embeds the fits that Python produced (``data/slater_fits.json``); from v26.11.9
   the Python refits at run time in the native optimizer and lands on other contractions (s and p
   shells differ by 5e-4 for STO-6G and 0.08 for cc-pVTZ at the nucleus), so the current Python
   is not the number Rust reproduces.

Every case is one atom: the points (Bohr), the shells' angular momenta, ``R(0)`` of the tabulated
families (the value the arbitrary direction at a nucleus scales), and the values of each basis
function at the points, as the Python ``BasisSet.build(...).atom(...)`` sampled them.

The cases are those of the tests that used the Python families as the oracle: the generated
hydrogenic and Gaussian families at four to nine elements, a per-element mixture, the numerical
atomic orbitals at three elements and five sizes, and the all-electron numerical atomic orbitals at
three tiers, with the custom-function case.
"""

import json
from pathlib import Path

import numpy as np
from ase import Atoms

from mandacaru.basis import BasisSet
from mandacaru.units import ANGSTROM_TO_BOHR

OUT = Path(__file__).parent / "reference" / "basis_samples.json"



def pack(name, points, functions, **extra):
    values = np.stack([f.evaluate(*points.T) for f in functions])
    case = {"name": name, "points": points.tolist(), "l": [int(f.l) for f in functions],
            "values": values.tolist()}
    case.update(extra)
    return case


GAUSSIAN = [(8, "STO-6G"), (8, "6-31G(d)"), (8, "cc-pVTZ"), (11, "aug-cc-pVQZ"), (8, "def2-TZVP")]
HYDROGENIC = [(1, "HAO"), (8, "HAO"), (26, "HAO"), (118, "HAO")]


def molecular(cases_of):
    cases = []
    for z, basis in cases_of:
        atoms = Atoms(numbers=[z], positions=[[.1, -.2, .3]])
        points = np.ascontiguousarray(np.random.default_rng(90).normal(size=(91, 3)) * 2)
        points[0] = atoms.positions[0] * ANGSTROM_TO_BOHR
        points = np.ascontiguousarray(np.vstack([points, points[0] + 1e-9 * np.eye(3)]))
        functions = BasisSet.build(basis).atom(z, center=atoms.positions[0])
        cases.append(pack(f"Z{z} {basis}", points, functions, numbers=[z],
                          positions=atoms.positions.tolist(), basis=basis))
    return cases


def per_element():
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]])
    spec = {"O": "STO-3G", "*": "HAO"}
    points = np.ascontiguousarray(np.random.default_rng(9).normal(size=(41, 3)))
    functions = [f for atom in atoms for f in BasisSet.build(spec).atom(atom.symbol, atom.position)]
    return pack("OH per element", points, functions, symbols=list(atoms.symbols),
                positions=atoms.positions.tolist(), spec=spec)


def numerical():
    cases = []
    for z in (1, 8, 26):
        for size, scheme in [("SZ", None), ("DZ", None), ("DZP", None), ("TZ2P", "last_zeta"),
                             ("QZ", [.2, .1, .05])]:
            options = dict(size=size, r_c=7., n_grid=300, zeta_split=scheme)
            points = np.ascontiguousarray(np.random.default_rng(55).normal(size=(90, 3)) * 1.5)
            points[:4] = [[0, 0, 0], [0, 0, 1], [0, 0, -2], [0, 0, 8]]
            points = np.ascontiguousarray(np.vstack([points, 1e-4 * np.eye(3)]))
            functions = BasisSet.build("NAO", **options).atom(z)
            r0 = [abs(float(f.radial(np.zeros(1))[0])) for f in functions]
            cases.append(pack(f"Z{z} NAO {size}", points, functions, number=z, options=options, r0=r0))
    return cases


def all_electron():
    cases = []
    for tier in (0, 1, 2):
        for z in (1, 8):
            options = dict(tier=tier, onset=1.5, width=.8, points=500, r_max=12.)
            wall = (options["onset"] + options["width"]) * ANGSTROM_TO_BOHR
            points = np.ascontiguousarray(np.random.default_rng(27).normal(size=(91, 3)))
            points[:5] = [[0, 0, 0], [0, 0, .7], [0, 0, -1.3], [0, 0, wall - .003],
                          [0, 0, wall + .001]]
            functions = BasisSet.build("NAO-AE", **options).atom(z)
            cases.append(pack(f"Z{z} NAO-AE tier {tier}", points, functions, number=z,
                              options=options, wall=wall))
    options = dict(tier=0, onset=1.5, width=.8, points=400, r_max=12.,
                   extra=[(2, 1, 1.2), (2, 0, .6)], tail_norm=None)
    points = np.ascontiguousarray(np.random.default_rng(28).normal(size=(41, 3)))
    functions = BasisSet.build("NAO-AE", **options).atom(2)
    cases.append(pack("He NAO-AE custom functions", points, functions, number=2, options=options))
    return cases


def main():
    import sys

    if "--gaussian" in sys.argv:
        record = json.loads(OUT.read_text())
        record["molecular"] = [c for c in record["molecular"] if c["basis"] == "HAO"]
        record["molecular"] += molecular(GAUSSIAN)
    else:
        record = {"molecular": molecular(HYDROGENIC), "per_element": per_element(),
                  "numerical": numerical(), "all_electron": all_electron()}
    OUT.write_text(json.dumps(record, separators=(",", ":")))
    print(OUT.stat().st_size, "bytes")
    for group, cases in record.items():
        for case in ([cases] if isinstance(cases, dict) else cases):
            print(group, case["name"], len(case["l"]), "functions")


if __name__ == "__main__":
    main()
