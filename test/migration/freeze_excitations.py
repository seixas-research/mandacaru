"""Freeze the first linear-response excitations of water and of the NH2 radical (TDDFT_PLAN
stages T1, T2 and T4).

The numbers are those of the native route as it landed (there is no earlier implementation to
reproduce); `test_native_excitations.py` holds every later change of the code to them. Only
quantities that do not depend on the frame inside a degenerate multiplet are kept: the energies
and the oscillator strengths summed over each multiplet (a multiplet is a run of roots closer
than the solver's degeneracy tolerance).

Run from the repository root with the `mandacaru` interpreter and `MANDACARU_PAW_PATH` set:
``python test/migration/freeze_excitations.py``. It writes ``reference/excitations.json``.
"""

import json
import platform
import time
from pathlib import Path

from ase.build import molecule

from mandacaru import Mandacaru, __version__

OUT = Path(__file__).parent / "reference" / "excitations.json"
SETTINGS = dict(method="dft", basis={"name": "PAW-LCAO", "size": "DZP"}, grid_spacing=0.25,
                parallelization={"threads": 2})
NUM_STATES = 5
XCS = ("lda", "pbe", "hse06")


def water():
    atoms = molecule("H2O")
    atoms.center(vacuum=4.0)
    return atoms


def amidogen():
    """NH2, a doublet with a nondegenerate singly occupied orbital: an unrestricted reference."""
    atoms = molecule("NH2")
    atoms.center(vacuum=4.0)
    return atoms


def system(key):
    """The structure and the ``spin`` argument of a case key."""
    if key.startswith("NH2/"):
        return amidogen(), None
    return water(), key.split("/")[2]


def multiplets(energies, strengths, tolerance):
    """``(first, count, summed strength)`` of each run of roots within ``tolerance``."""
    groups, first = [], 0
    for k in range(1, len(energies) + 1):
        if k == len(energies) or energies[k] - energies[k - 1] >= tolerance:
            groups.append([first, k - first, float(sum(strengths[first:k]))])
            first = k
    return groups


def case(key):
    atoms, spin = system(key)
    xc, approximation = key.removeprefix("NH2/").split("/")[:2]
    atoms.calc = Mandacaru(xc=xc, **SETTINGS)
    started = time.perf_counter()
    found = atoms.calc.get_excitations(NUM_STATES, approximation=approximation, spin=spin,
                                       tolerance=1e-8)
    seconds = time.perf_counter() - started
    record = {
        "energies_eV": [float(e) for e in found.energies],
        "multiplets": multiplets(found.energies, found.oscillator_strengths,
                                 found.degenerate_tolerance),
        "velocity_multiplets": multiplets(found.energies, found.velocity_oscillator_strengths,
                                          found.degenerate_tolerance),
        "one_center_hartree_eV": None if found.one_center_hartree is None
        else [float(e) for e in found.one_center_hartree],
        "spin_squared": None if found.spin_squared is None
        else [float(v) for v in found.spin_squared],
        "seconds": seconds,
    }
    atoms.calc.close()
    return record


def main():
    out = {
        "description": "H2O (closed shell, singlets and triplets) and NH2 (doublet, unrestricted), "
                       "PAW-LCAO DZP, grid_spacing 0.25 Angstrom, 4 Angstrom vacuum, the lowest "
                       "five roots (completed to multiplets), tolerance 1e-8",
        "settings": {k: v for k, v in SETTINGS.items()},
        "version": __version__,
        "platform": platform.platform(),
        "cases": {},
    }
    keys = [f"{xc}/{approximation}/{spin}" for xc in XCS for approximation in ("tda", "casida")
            for spin in ("singlet", "triplet")]
    keys += [f"NH2/{xc}/{approximation}" for xc in XCS for approximation in ("tda", "casida")]
    for key in keys:
        out["cases"][key] = case(key)
        print(key, out["cases"][key]["energies_eV"], flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
