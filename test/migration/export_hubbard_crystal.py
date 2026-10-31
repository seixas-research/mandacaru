#!/usr/bin/env python
"""HU-3: write the crystal snapshots that the Rust DFT+U tests read.

Each case is the density-independent PAW setup of ``test/migration/freeze_hubbard.py`` (same
atoms, grid, mesh and basis), captured from the current setup adapter
(``integrations/paw.crystal_input``) before the native problem is built, so no SCF runs here.
It writes, into ``test/migration/reference/hubbard/crystals/``:

* ``<case>.json``: the snapshot metadata (small, tracked) plus a ``hubbard`` object with
  U = 0 and the shell tables (projector positions ordered m then radial, the PAW partial-wave
  weights of ``frozen.json`` for the snapshot's projector basis, and the seed);
* ``<case>.npz``: the arrays the metadata refers to by key (git-ignored, about 15 MB for NiO).

The Rust tests (``crates/mandacaru-core/tests/periodic_hubbard.rs``) load the pair through
``Crystal::from_arrays``, set U on the shells, and call ``solve``.  U = 0 in the file keeps the
fixture equal to the plain DFT problem; the tests choose U.

    python test/migration/export_hubbard_crystal.py [--cases nio_afm2,ni_atom_7,cuh_7,cuh_7_spin]

Run it in the ``mandacaru`` environment with ``MANDACARU_PAW_PATH`` set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "reference" / "hubbard" / "crystals"
FROZEN = HERE / "reference" / "hubbard" / "frozen.json"
sys.path.insert(0, str(HERE))

#: case -> (freeze system, element carrying +U, spin channels explicitly requested, width in eV, cell override)
CASES = {
    "nio_afm2": dict(system="nio", element="Ni", spin=True, width=0.05, cell=None),
    "ni_atom_7": dict(system="ni_atom", element="Ni", spin=True, width=0.02, cell=None),
    "cuh_7": dict(system="cuh_crystal", element="Cu", spin=False, width=0.01, cell=7.0),
    "cuh_7_spin": dict(system="cuh_crystal_spin", element="Cu", spin=True, width=0.01, cell=7.0),
}
SHELL = {"Ni": ("d", 2), "Cu": ("d", 2)}


class _Captured(BaseException):
    """Raised by the capture hook so that no native problem is built."""


def capture(case: str):
    from freeze_hubbard import _atoms
    from mandacaru import Mandacaru
    from mandacaru.integrations import paw

    spec = CASES[case]
    atoms, cfg = _atoms(spec["system"], "A")
    if spec["cell"]:
        atoms.set_cell([spec["cell"]] * 3)
        atoms.center()
    taken: dict = {}
    original = paw.crystal_input

    def hook(crystal, *args, **kwargs):
        metadata, arrays = original(crystal, *args, **kwargs)
        taken.update(metadata=metadata, arrays=arrays, projectors=list(crystal.projectors),
                     moments=np.asarray(kwargs.get("magnetic_moments"), dtype=float))
        raise _Captured

    paw.crystal_input = hook
    try:
        common = dict(method="dft", xc="lda", grid_spacing=cfg["h"], trace=False,
                      basis={"name": "PAW-LCAO", "size": cfg["size"]}, symmetry=False,
                      kpts={"size": cfg["kpts"], "gamma": True}, spin=spec["spin"],
                      smearing={"method": "fermi-dirac", "width": cfg["width"]})
        atoms.calc = Mandacaru(**common)
        try:
            atoms.get_potential_energy()
        except _Captured:
            pass
    finally:
        paw.crystal_input = original
    if not taken:
        raise RuntimeError(f"{case}: the setup was not captured")
    return atoms, cfg, taken


def shell_tables(case: str, taken: dict, frozen: dict) -> list:
    spec = CASES[case]
    symbol = spec["element"]
    letter, l = SHELL[symbol]
    row = frozen["cases"]["weights"]["rows"][symbol]
    assert row["l"] == l
    reference = float(sum(row["occupations"]))
    groups: dict = {}
    for position, p in enumerate(taken["projectors"]):
        if p.symbol == symbol and int(p.l) == l:
            groups.setdefault(int(p.atom_index), []).append((int(p.m), int(p.index), position, p))
    shells = []
    for atom, entries in sorted(groups.items()):
        entries.sort(key=lambda e: (e[0], e[1]))
        basis = str(getattr(entries[0][3], "projector_basis", "raw"))
        weights = row["w_raw"] if basis == "raw" else row["w_dual"]
        assert len(entries) == (2 * l + 1) * len(weights), (case, atom, len(entries))
        shells.append(dict(
            atom=atom, symbol=symbol, l=l, u=0.0, j=0.0,
            projectors=[e[2] for e in entries], weights=[float(w) for w in weights],
            projector_basis=basis,
            seed=dict(n_electrons=reference, moment=float(taken["moments"][atom]))))
    return shells


def write(case: str, frozen: dict) -> None:
    atoms, cfg, taken = capture(case)
    spec = CASES[case]
    metadata = json.loads(taken["metadata"])
    metadata["hubbard"] = dict(form="dudarev-fll", shells=shell_tables(case, taken, frozen))
    metadata["fixture"] = dict(
        generator="test/migration/export_hubbard_crystal.py", case=case, width_ev=spec["width"],
        symmetry=False, spin=spec["spin"], h_angstrom=cfg["h"], size=cfg["size"],
        symbols=atoms.get_chemical_symbols(),
        initial_moments=[float(x) for x in taken["moments"]],
        note="U = 0 in the file; the tests set U. Energies are Hartree, lengths Bohr.")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{case}.npz", **taken["arrays"])
    (OUT / f"{case}.json").write_text(json.dumps(metadata, indent=1, allow_nan=False) + "\n")
    size = (OUT / f"{case}.npz").stat().st_size / 1e6
    print(f"{case}: m={metadata['m']} p={metadata['p']} k={len(metadata['kpoints'])} "
          f"spins={metadata['spins']} grid={metadata['grid']['shape']} permutations={metadata['permutation_count']} "
          f"shells={len(metadata['hubbard']['shells'])} npz={size:.1f} MB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=",".join(CASES))
    args = parser.parse_args()
    if not os.environ.get("MANDACARU_PAW_PATH"):
        raise SystemExit("set MANDACARU_PAW_PATH")
    frozen = json.loads(FROZEN.read_text())
    for case in args.cases.split(","):
        write(case, frozen)


if __name__ == "__main__":
    main()
