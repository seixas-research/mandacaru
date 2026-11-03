"""Freeze the analytic crystal forces and stress as the user gets them (CF-7 / AS-11): ``Mandacaru(...)`` + ASE.

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set::

    python test/migration/freeze_periodic_response.py list
    python test/migration/freeze_periodic_response.py case lih [--output FILE]

For each case this script builds the crystal of the matching force freeze (``freeze_periodic_gradients.py``) and strain freeze
(``freeze_periodic_strain.py``), asks ASE for ``get_forces()`` and ``get_stress()`` (the native Rust engines; no differencing),
and records next to them the reconverged-finite-difference ORACLES those two freezes measured for the same crystal (the
Richardson protocol of ``PAW_COMPLETION_PLAN.md`` 14.4) and the gate of each component. The Python test
(``test_native_periodic_response.py``) rebuilds the atoms and the options from this file alone and compares the live numbers
with the engine's frozen ones (tight) and with the oracles (at the gates).

Units: eV, Angstrom, ASE Voigt order ``[xx, yy, zz, yz, xz, xy]`` for the stress.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freeze_periodic_gradients as gradients  # noqa: E402
import freeze_periodic_strain as strains  # noqa: E402
from freeze_stamp import stamp  # noqa: E402

#: The plan's crystal force gate (1e-5 eV/Angstrom) and the stress gates (``ANALYTIC_STRESS_PLAN.md`` section 9 rung 4).
FORCE_GATE = 1e-5
STRESS_GATE_DIAGONAL, STRESS_GATE_SHEAR = 3e-6, 5e-8
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

#: ``name -> (force case of freeze_periodic_gradients or None, stress case of freeze_periodic_strain or None)``. A case whose two
#: freezes share the crystal and the options (the same atoms, mesh, functional and datasets) is one run.
CASES = {
    "lih": ("lih", "lih"),
    "si": ("si", "si"),
    "al2": ("al2", None),
    "al_small": (None, "al_small"),
    "lih_hse06": ("lih_hse06", "lih_hse06"),
    "beh_lda": ("beh_lda", None),
    "si_ldafr": ("si_ldafr", "si_ldafr"),
}


def read(folder: str, name: str) -> dict:
    return json.loads((HERE / "reference" / folder / f"{name}.json").read_text())


def options_of(spec: dict) -> dict:
    """The keyword arguments of ``Mandacaru`` the freezes used, as plain JSON."""
    scf = dict(gradients.SCF, **spec.get("scf", {}), width_ha=spec.get("width_ev", gradients.WIDTH_EV) / 27.211386245988)
    out = dict(xc=spec.get("xc", "lda"), basis=spec.get("basis", gradients.BASIS), grid_spacing=spec["grid_spacing"],
               kpts=spec["kpts"], smearing={"method": "fermi-dirac", "width": spec.get("width_ev", gradients.WIDTH_EV)},
               periodic_scf_options=scf, symmetry=False)
    if "exxdiv" in spec:
        out["exxdiv_treatment"] = spec["exxdiv"]
    if "paw_dataset" in spec:
        out["paw_dataset"] = spec["paw_dataset"]
    if spec.get("spin"):
        out["spin"] = True
    return out


def atoms_record(atoms) -> dict:
    return dict(symbols=atoms.get_chemical_symbols(), cell_angstrom=atoms.cell.array.tolist(),
                positions_angstrom=atoms.positions.tolist(), pbc=[True, True, True],
                magmoms=atoms.get_initial_magnetic_moments().tolist())


def force_oracle(case: str) -> list:
    from mandacaru._native import BOHR_TO_ANGSTROM, HARTREE_TO_EV
    to_ev_a = HARTREE_TO_EV / BOHR_TO_ANGSTROM
    manifest = read("periodic_gradients", case)
    plan = FORCE_GATE / to_ev_a
    out = []
    for entry in manifest["coordinates"]:
        gate = min(max(100 * abs(entry["floor_richardson"]), 1e-9), plan)
        out.append(dict(atom=entry["atom"], axis=entry["axis"], force_ev_angstrom=-entry["rr"] * to_ev_a,
                        gate_ev_angstrom=gate * to_ev_a))
    return out


def stress_oracle(case: str) -> list:
    """Per Voigt component: the symmetrized oracle ``(g_ab + g_ba) / (2 Omega)`` and the larger gate of the two entries."""
    from mandacaru._native import BOHR_TO_ANGSTROM, HARTREE_TO_EV
    to_ev_a3 = HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3
    manifest = read("periodic_strain", case)
    volume, step = manifest["volume_bohr3"], manifest["steps"][0]
    entries = {(e["a"], e["b"]): e for e in manifest["entries"]}

    def gate(a, b):
        plan = (STRESS_GATE_DIAGONAL if a == b else STRESS_GATE_SHEAR) / to_ev_a3 * volume
        return min(max(100 * abs(entries[(a, b)]["floor_richardson"]), 2e-11 / step), plan)

    out = []
    for a, b in VOIGT:
        g = 0.5 * (entries[(a, b)]["rr"] + entries[(b, a)]["rr"])
        out.append(dict(a=a, b=b, stress_ev_angstrom3=g / volume * to_ev_a3,
                        gate_ev_angstrom3=max(gate(a, b), gate(b, a)) / volume * to_ev_a3))
    return out


def freeze(name: str, output: Path) -> None:
    force_case, stress_case = CASES[name]
    spec = (gradients.CASES[force_case] if force_case else strains.CASES[stress_case])
    start = time.time()
    atoms = spec["atoms"]()
    record = dict(atoms=atoms_record(atoms), options=options_of(spec), force_case=force_case, stress_case=stress_case)
    atoms.calc = gradients.calculator(spec)
    if force_case:
        record["forces_ev_angstrom"] = atoms.get_forces().tolist()
        record["force_oracle"] = force_oracle(force_case)
    if stress_case:
        record["stress_ev_angstrom3"] = atoms.get_stress().tolist()
        record["stress_oracle"] = stress_oracle(stress_case)
    else:
        # A spin-polarized crystal's stress is refused by the engine; the test replays the refusal.
        record["stress_refused"] = bool(spec.get("spin"))
    record["seconds"] = time.time() - start
    cases = json.loads(output.read_text())["cases"] if output.exists() else {}
    cases[name] = record
    datasets = []
    for root in {spec.get("paw_dataset", gradients.DATASET_SETS[spec.get("xc", "lda")])}:
        for symbol in sorted(set(atoms.get_chemical_symbols())):
            datasets.append(str(Path(os.environ["MANDACARU_PAW_PATH"]) / root / f"{symbol}.parquet"))
    payload = dict(stamp=stamp(datasets=datasets, extra={"generator": "test/migration/freeze_periodic_response.py"}),
                   cases=cases)
    output.write_text(json.dumps(payload))
    print(f"{name}: written to {output} ({record['seconds']:.0f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("list", "case"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--output", type=Path, default=HERE / "reference" / "periodic_response.json")
    ns = parser.parse_args()
    if ns.action == "list":
        print("\n".join(CASES))
    else:
        freeze(ns.name, ns.output)


if __name__ == "__main__":
    main()
