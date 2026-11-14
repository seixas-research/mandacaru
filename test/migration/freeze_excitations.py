"""Freeze the first linear-response excitations of water, of the NH2 radical and of CuH with +U
(TDDFT_PLAN stages T1 to T4).

The numbers are those of the native route as it landed (there is no earlier implementation to
reproduce); `test_native_excitations.py` holds every later change of the code to them. Only
quantities that do not depend on the frame inside a degenerate multiplet are kept: the energies
and the oscillator strengths summed over each multiplet (a multiplet is a run of roots closer
than the solver's degeneracy tolerance).

Run from the repository root with the `mandacaru` interpreter and `MANDACARU_PAW_PATH` set:
``python test/migration/freeze_excitations.py``. It writes ``reference/excitations.json``, but
only after recomputing every key already frozen and finding each bit-identical (the timing apart):
a frozen number that moved is reported and nothing is written. The records already frozen are kept
verbatim; only the new keys are added.
"""

import json
import platform
import time
from pathlib import Path

from ase import Atoms
from ase.build import molecule

from mandacaru import Mandacaru, __version__

OUT = Path(__file__).parent / "reference" / "excitations.json"
SETTINGS = dict(method="dft", basis={"name": "PAW-LCAO", "size": "DZP"}, grid_spacing=0.25,
                parallelization={"threads": 2})
NUM_STATES = 5
#: r2SCAN water and NH2 are not frozen yet: the gauge-invariant kinetic density adds a
#: current-current term to the r2SCAN A - B that would move those roots, and the owner decides
#: that first (TDDFT_PLAN Q19, section 13). Add "r2scan" here when it is decided.
XCS = ("lda", "pbe", "hse06")
#: The +U case: CuH (closed shell) with the Dudarev correction on the copper d shell (eV).
HUBBARD = {"Cu": {"l": "d", "U": 4.0}}


def water():
    atoms = molecule("H2O")
    atoms.center(vacuum=4.0)
    return atoms


def amidogen():
    """NH2, a doublet with a nondegenerate singly occupied orbital: an unrestricted reference."""
    atoms = molecule("NH2")
    atoms.center(vacuum=4.0)
    return atoms


def copper_hydride():
    atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[9.0] * 3)
    atoms.center()
    return atoms


def system(key):
    """The structure, the ``spin`` argument and the extra calculator options of a case key."""
    if key.startswith("NH2/"):
        return amidogen(), None, {}
    if key.startswith("CuH+U/"):
        return copper_hydride(), key.split("/")[3], {"hubbard_u": HUBBARD}
    return water(), key.split("/")[2], {}


def functional_and_approximation(key):
    parts = key.split("/")
    return (parts[1], parts[2]) if parts[0] in ("NH2", "CuH+U") else (parts[0], parts[1])


def multiplets(energies, strengths, tolerance):
    """``(first, count, summed strength)`` of each run of roots within ``tolerance``."""
    groups, first = [], 0
    for k in range(1, len(energies) + 1):
        if k == len(energies) or energies[k] - energies[k - 1] >= tolerance:
            groups.append([first, k - first, float(sum(strengths[first:k]))])
            first = k
    return groups


def case(key):
    atoms, spin, extra = system(key)
    xc, approximation = functional_and_approximation(key)
    atoms.calc = Mandacaru(xc=xc, **SETTINGS, **extra)
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


def moved(frozen, found):
    """The fields of ``found`` that differ bitwise from ``frozen`` (the timing excluded)."""
    return [name for name in frozen if name != "seconds" and frozen[name] != found[name]]


def main():
    previous = json.loads(OUT.read_text()) if OUT.exists() else {"cases": {}}
    out = {
        "description": "H2O (closed shell, singlets and triplets), NH2 (doublet, unrestricted) "
                       "and CuH with hubbard_u Cu d U = 4 eV (closed shell), PAW-LCAO DZP, "
                       "grid_spacing 0.25 Angstrom, 4 Angstrom vacuum (CuH: a 9 Angstrom cell), "
                       "the lowest five roots (completed to multiplets), tolerance 1e-8",
        "settings": {k: v for k, v in SETTINGS.items()},
        "version": __version__,
        "platform": platform.platform(),
        "cases": {},
    }
    keys = [f"{xc}/{approximation}/{spin}" for xc in XCS for approximation in ("tda", "casida")
            for spin in ("singlet", "triplet")]
    keys += [f"NH2/{xc}/{approximation}" for xc in XCS for approximation in ("tda", "casida")]
    keys += [f"CuH+U/lda/{approximation}/{spin}" for approximation in ("tda", "casida")
             for spin in ("singlet", "triplet")]
    changed = {}
    for key in keys:
        record = case(key)
        frozen = previous["cases"].get(key)
        if frozen is None:
            out["cases"][key] = record
            print("new  ", key, record["energies_eV"], flush=True)
        else:
            out["cases"][key] = frozen
            changed_fields = moved(frozen, record)
            if changed_fields:
                changed[key] = changed_fields
            print("same " if not changed_fields else "MOVED", key, record["energies_eV"],
                  flush=True)
    missing = sorted(set(previous["cases"]) - set(keys))
    if changed or missing:
        raise SystemExit(f"nothing written: moved {changed}, dropped {missing}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
