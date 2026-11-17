"""Freeze short real-time traces of water, of the NH2 radical and of CuH with +U after a density
kick (TDDFT_PLAN stages T6.0 and T6a), and of silicon after a vector-potential kick (T6b).

The numbers are those of the native route as it landed (there is no earlier implementation to
reproduce); `test_native_real_time.py` holds every later change of the code to them. Each case
keeps the induced dipole along the kick every `STRIDE` steps, the energy the kick put in, the
energy drift and the Fock builds. A crystal keeps instead the corrected current along the field
(e / (Angstrom^2 fs)), the valence count, the band sum ``N_dia`` and ``epsilon`` at a few
energies.

Run from the repository root with the `mandacaru` interpreter and `MANDACARU_PAW_PATH` set:
``python test/migration/freeze_real_time.py``. It writes ``reference/real_time.json``, but only
after recomputing every key already frozen and finding each bit-identical (the timing apart): a
frozen number that moved is reported and nothing is written. The records already frozen are kept
verbatim; only the new keys are added.
"""

import json
import platform
import time
from pathlib import Path

from ase.build import bulk

from mandacaru import Mandacaru, __version__

try:  # imported by test_native_real_time.py as part of the package
    from .freeze_excitations import HUBBARD, SETTINGS, amidogen, copper_hydride, water
except ImportError:  # run as a script: its directory is on the path
    from freeze_excitations import HUBBARD, SETTINGS, amidogen, copper_hydride, water

OUT = Path(__file__).parent / "reference" / "real_time.json"
#: 1 fs in steps of 0.005 fs (0.2067 atomic units), ETRS with the default corrector.
DURATION_FS = 1.0
TIME_STEP_FS = 0.005
STRIDE = 20
#: The kick, 1/Angstrom: the default, and 1e-4 / Bohr for HSE06 (Q17: the hole model's kink).
STRENGTH = 0.002
HYBRID_STRENGTH = 1e-4 / 0.529177210903
CASES = ("water/lda", "water/pbe", "water/hse06", "NH2/lda", "CuH+U/lda", "Si/lda")
#: Silicon, the crystal of the Q20 decision, small: SZ on a 2^3 Gamma-centered mesh (rebuilt whole
#: by the propagation), its kick 0.002 / Angstrom, and epsilon at these energies (eV, damping 1 eV).
CRYSTAL = dict(method="dft", basis={"name": "PAW-LCAO", "size": "SZ"}, grid_spacing=0.25,
               kpts={"size": (2, 2, 2), "gamma": True},
               smearing={"method": "fermi-dirac", "width": 0.01}, parallelization={"threads": 2})
CRYSTAL_ENERGIES = (1.0, 3.0, 5.0, 8.0)
#: Dated re-freezes: a key here is recomputed and replaced once, the old and new summaries kept in
#: the file's ``refrozen`` history; after that it is compared bitwise again like every other key.
LIBRARY_V3 = ("2026-11-16: the PAW-LCAO library was regenerated (file format 3, mandacaru-paw 8a6e977); "
              "a self-pin of the production route on the installed library (the earlier 2026-11-12 "
              "water/hse06 re-freeze stays in the history)")
REFREEZE = {key: LIBRARY_V3 for key in CASES}


def system(key):
    name, xc = key.split("/")
    if name == "Si":
        return bulk("Si", "diamond", a=5.43), xc, {}
    if name == "NH2":
        return amidogen(), xc, {}
    if name == "CuH+U":
        return copper_hydride(), xc, {"hubbard_u": HUBBARD}
    return water(), xc, {}


def run(key):
    atoms, xc, extra = system(key)
    atoms.calc = Mandacaru(xc=xc, **(CRYSTAL if atoms.pbc.all() else SETTINGS), **extra)
    atoms.get_potential_energy()
    strength = HYBRID_STRENGTH if xc == "hse06" else STRENGTH
    started = time.perf_counter()
    rt = atoms.calc.get_real_time(duration=DURATION_FS, time_step=TIME_STEP_FS, direction="x",
                                  strength=strength)
    seconds = time.perf_counter() - started
    return rt, seconds, atoms.calc


def case(key):
    rt, seconds, calc = run(key)
    if rt.periodic:
        epsilon = rt.dielectric_function(CRYSTAL_ENERGIES, damping=1.0)
        record = {
            "corrected": [float(v) for v in rt.corrected[::STRIDE]],
            "electrons": float(rt.electrons),
            "diamagnetic": float(rt.diamagnetic[0]),
            "epsilon_re": [float(v.real) for v in epsilon],
            "epsilon_im": [float(v.imag) for v in epsilon],
        }
    else:
        record = {"along_eA": [float(v) for v in rt.along[::STRIDE]]}
    record.update({
        "kick_energy_eV": float(rt.kick_energy),
        "energy_drift": float(rt.energy_drift),
        "builds": int(rt.builds),
        "width_eV": float(rt.width),
        "seconds": seconds,
    })
    calc.close()
    return record


def moved(frozen, found):
    """The fields of ``found`` that differ bitwise from ``frozen`` (the timing excluded)."""
    return [name for name in frozen if name != "seconds" and frozen[name] != found[name]]


def main():
    previous = json.loads(OUT.read_text()) if OUT.exists() else {"cases": {}}
    out = {
        "description": "induced dipole along x (e Angstrom, every 20 steps) after a density kick "
                       "of 0.002 / Angstrom (HSE06: 1e-4 / Bohr), ETRS with corrector tolerance "
                       "1e-6, 1 fs in steps of 0.005 fs; H2O, NH2 (unrestricted) and CuH with "
                       "hubbard_u Cu d U = 4 eV, PAW-LCAO DZP, grid_spacing 0.25 Angstrom; Si "
                       "(diamond, 5.43 Angstrom, SZ, 2^3 mesh): the corrected current along x "
                       "(e / (Angstrom^2 fs)) after a vector-potential kick of 0.002 / Angstrom, "
                       "N, N_dia and epsilon(E + 1 eV i) at 1, 3, 5 and 8 eV",
        "settings": {k: v for k, v in SETTINGS.items()},
        "crystal_settings": {k: v for k, v in CRYSTAL.items()},
        "version": __version__,
        "platform": platform.platform(),
        "cases": {},
        "refrozen": previous.get("refrozen", {}),
    }
    changed = {}
    for key in CASES:
        record = case(key)
        frozen = previous["cases"].get(key)
        if frozen is None:
            out["cases"][key] = record
            print("new  ", key, record["kick_energy_eV"], record["energy_drift"], flush=True)
        elif key in REFREEZE and REFREEZE[key] not in [
                entry["reason"] for entry in out["refrozen"].get(key, [])]:
            out["cases"][key] = record
            summary = lambda r: {name: r[name] for name in
                                 ("kick_energy_eV", "energy_drift", "builds", "along_eA",
                                              "corrected") if name in r}
            out["refrozen"].setdefault(key, []).append(
                {"reason": REFREEZE[key], "old": summary(frozen), "new": summary(record)})
            print("REFROZEN", key, frozen["builds"], "->", record["builds"], "builds;",
                  frozen["energy_drift"], "->", record["energy_drift"], "drift", flush=True)
        else:
            out["cases"][key] = frozen
            changed_fields = moved(frozen, record)
            if changed_fields:
                changed[key] = changed_fields
            print("same " if not changed_fields else "MOVED", key, flush=True)
    missing = sorted(set(previous["cases"]) - set(CASES))
    if changed or missing:
        raise SystemExit(f"nothing written: moved {changed}, dropped {missing}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
