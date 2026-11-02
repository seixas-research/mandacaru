"""Freeze the spinor-hybrid snapshots (SH-0 and SH-1a of SPINOR_HYBRID_PLAN) from the live Python setup.

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set::

    python test/migration/freeze_hybrid_spinor.py snapshot si_so_tr3      # one case, committed fixture
    python test/migration/freeze_hybrid_spinor.py snapshot au_sz_222 --output SCRATCH   # a measurement case
    python test/migration/freeze_hybrid_spinor.py list

A case builds the crystal through ``Mandacaru(..., save_crystal=...)``, stops after the snapshot is written (no
SCF runs here) and stamps it.  The Python build halves the mesh of a time-reversal-symmetric spinor and a hybrid
refuses a halved mesh (``input.rs::hybrid_needs_the_full_mesh``), so the script forces the whole mesh for the build only:
``build_crystal`` is called with ``full_mesh=True, symmetry=False`` (the one-expression change SH-1b proposes for
``integrations/paw.py::configure``).  The committed fixtures (``test/migration/reference/hybrid_spinor/``): ``si_so_tr3`` and ``si_so_tr4`` keep ``functional: lda`` like the other
phase-5 snapshots, ``si4_y2`` and ``si8_gamma`` carry ``functional: hse06`` (they are built with ``xc="hse06"``). All four
list their datasets by library-relative file name (``<library>/<symbol>.parquet``) with the SHA-256 of the build machine,
so none loads raw into a hybrid crystal: a test turns the functional into ``hse06`` and rebuilds the ``sites`` block from
the shipped files (``periodic_hybrid_spinor_support::sites``) itself.  Every snapshot records the ``(atom, l, m)`` label of each basis function and the dataset library it was
built from under ``provenance`` (a basis function's label is not in the snapshot proper, and the time-reversal
operator of the complex harmonics needs it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASIS = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}
SCF = {"tolerance": 1e-12, "density_tolerance": 1e-10}


class Stop(Exception):
    """Raised after the snapshot is on disk: no SCF is run by the freeze."""


def definition(case: str):
    """``(atoms, calculator options, library, meta)`` of a case."""
    import numpy as np
    from ase.build import bulk
    base = dict(method="dft", xc="lda", basis=dict(BASIS), trace=False, periodic_scf_options=dict(SCF),
                smearing={"method": "fermi-dirac", "width": 0.01})

    def silicon(name, library, size, xc="lda", basis=None, spacing=None):
        atoms = bulk("Si", "diamond", a=5.43)
        opts = dict(base, xc=xc, paw_dataset=library, kpts={"gamma": True, "size": size},
                    grid_spacing=float(np.linalg.norm(atoms.cell[0])) / 10 if spacing is None else spacing)
        if basis:
            opts["basis"] = dict(BASIS, **basis)
        return atoms, opts, library, dict(case=name, mesh=list(size))

    if case == "si_so_tr3":      # committed: the complex-k fixture (mesh 0, +1/3, -1/3)
        return silicon(case, "lda-fr", (3, 1, 1))
    if case == "si_so_tr4":      # committed: four points, two of them complex (0, +-1/4, 1/2)
        return silicon(case, "lda-fr", (4, 1, 1))
    if case == "si_pbefr_sz_222":
        return silicon(case, "pbe-fr", (2, 2, 2), xc="hse06")
    if case == "si_pbefr_dzp_222":
        return silicon(case, "pbe-fr", (2, 2, 2), xc="hse06", basis={"size": "DZP"})
    if case == "si_ldafr_sz_222":
        return silicon(case, "lda-fr", (2, 2, 2))
    if case == "si_pbefr_sz_333":
        return silicon(case, "pbe-fr", (3, 3, 3), xc="hse06")
    if case in ("au_sz_222", "pt_sz_222"):
        symbol = "Au" if case.startswith("au") else "Pt"
        atoms = bulk(symbol, "fcc", a=4.08 if symbol == "Au" else 3.92)
        opts = dict(base, xc="hse06", paw_dataset="pbe-fr", kpts={"gamma": True, "size": (2, 2, 2)},
                    grid_spacing=float(np.linalg.norm(atoms.cell[0])) / 16,
                    smearing={"method": "fermi-dirac", "width": 0.1})
        return atoms, opts, "pbe-fr", dict(case=case, mesh=[2, 2, 2])
    if case in ("si4_y2", "si8_gamma"):   # the supercell fold: 4 atoms on 1x2x1, 8 atoms at Gamma (doubled along y)
        from ase import Atoms
        a = 5.43      # the orthorhombic cell (a/sqrt2, a/sqrt2, a) of diamond: 4 atoms, two fcc points
        frac = [(0, 0, 0), (.5, .5, .5), (.5, 0, .25), (0, .5, .75)]
        cell = [a / 2 ** .5, a / 2 ** .5, a]
        atoms = Atoms("Si4", scaled_positions=frac, cell=cell, pbc=True)
        size = (1, 2, 1)
        if case == "si8_gamma":
            atoms = atoms.repeat((1, 2, 1))
            size = (1, 1, 1)
        opts = dict(base, xc="hse06", paw_dataset="pbe-fr", grid_spacing=0.8, symmetry=False,
                    kpts={"gamma": True, "size": size})
        return atoms, opts, "pbe-fr", dict(case=case, mesh=list(size))
    raise SystemExit(f"unknown case {case!r}; known: {CASES}")


CASES = ("si_so_tr3", "si_so_tr4", "si_ldafr_sz_222", "si_pbefr_sz_222", "si_pbefr_dzp_222", "si_pbefr_sz_333",
         "au_sz_222", "pt_sz_222", "si4_y2", "si8_gamma")


def snapshot(case: str, output: Path, threads: int, name: str | None = None, precision: str = "float64"):
    for var in ("MANDACARU_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    if not os.environ.get("MANDACARU_PAW_PATH"):
        raise SystemExit("set MANDACARU_PAW_PATH to the mandacaru-paw checkout")
    from mandacaru import Mandacaru
    from mandacaru.integrations import paw as adapter
    from mandacaru.pseudopotentials import periodic_paw
    from freeze_stamp import stamp
    atoms, opts, library, meta = definition(case)
    opts["precision"] = precision
    target = Path(output) / f"{name or case}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    built = {}
    real_build, real_configure = periodic_paw.build_crystal, adapter.configure

    def whole_mesh(*args, **kwargs):
        kwargs.update(full_mesh=True, symmetry=False)
        crystal, context = real_build(*args, **kwargs)
        built["crystal"] = crystal
        return crystal, context

    def configure_then_stop(driver, geometry):
        real_configure(driver, geometry)
        raise Stop

    periodic_paw.build_crystal = whole_mesh
    adapter.configure = configure_then_stop
    atoms.calc = Mandacaru(**opts, resources={"threads": threads}, save_crystal=str(target))
    try:
        atoms.get_potential_energy()
    except Stop:
        pass
    finally:
        periodic_paw.build_crystal, adapter.configure = real_build, real_configure
    payload = json.loads(target.read_text())
    crystal = built["crystal"]
    labels = [[int(a), int(f.l), int(f.m)] for a, f in zip(crystal.atom_of_orbital, crystal.basis)]
    weights = [k["weight"] for k in payload["kpoints"]]
    assert payload["spinor"] is True and payload["time_reversal"] is True
    assert len(payload["kpoints"]) == meta_points(opts["kpts"]["size"]), "the mesh was reduced"
    assert max(weights) - min(weights) < 1e-12 and abs(sum(weights) - 1) < 1e-12, weights
    assert payload["permutations"] == [] and len(labels) == payload["m"]
    datasets = [os.path.join(os.environ["MANDACARU_PAW_PATH"], library, f"{s}.parquet")
                for s in sorted(set(atoms.get_chemical_symbols()))]
    # The datasets are named by library, not by this machine's path: a test rebuilds the block.
    for entry in payload["sites"]["datasets"]:
        entry["file"] = f"{library}/{entry['symbol']}.parquet"
    payload["provenance"].update(
        reference_case=case, generator="test/migration/freeze_hybrid_spinor.py", library=library, orbital_labels=labels,
        stamp=stamp(datasets=datasets, extra=dict(case=case, threads=threads, mesh=list(opts["kpts"]["size"]),
                                                  full_mesh_forced=True, n_kpoints=len(payload["kpoints"]),
                                                  grid_shape=payload["grid"]["shape"], m=payload["m"], p=payload["p"])))
    target.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(case=case, path=str(target), bytes=target.stat().st_size, nk=len(payload["kpoints"]),
                          m=payload["m"], p=payload["p"], grid=payload["grid"]["shape"],
                          fractional=[k["fractional"] for k in payload["kpoints"]])))


def meta_points(size) -> int:
    return int(size[0]) * int(size[1]) * int(size[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("snapshot", "list"))
    parser.add_argument("case", nargs="?")
    parser.add_argument("--output", type=Path, default=HERE / "reference" / "hybrid_spinor")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--name", default=None, help="file stem (default: the case)")
    parser.add_argument("--precision", default="float64", choices=("float32", "float64"))
    args = parser.parse_args()
    if args.action == "list":
        print("\n".join(CASES))
        return
    if args.case is None:
        raise SystemExit("name a case")
    snapshot(args.case, args.output, args.threads, args.name, args.precision)


if __name__ == "__main__":
    main()
