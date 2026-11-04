"""Profile one crystal in a fresh process; report RSS high-water marks in JSON.

Run with MANDACARU_PAW_PATH set, for example:
    python scripts/profile_lcao_memory.py si
Use separate processes for each material and storage policy; ru_maxrss never decreases.
"""
import argparse
import functools
import json
import resource
import sys
import time

from ase.build import bulk, graphene
from mandacaru import Mandacaru
from mandacaru.integrations import paw
from mandacaru.pseudopotentials import periodic_paw


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("si", "al", "graphene"))
    parser.add_argument("--offload-bytes", type=int, default=None)
    parser.add_argument("--repeat", nargs=3, type=int, default=(1, 1, 1), metavar=("X", "Y", "Z"))
    parser.add_argument("--kmesh", nargs=3, type=int, default=None, metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    phases = []

    def wrap(module, name):
        original = getattr(module, name)
        @functools.wraps(original)
        def measured(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                phases.append({"phase": name, "wall_s": time.perf_counter() - start,
                               "peak_rss_bytes": rss_bytes()})
        setattr(module, name, measured)

    wrap(periodic_paw, "build_crystal")
    wrap(paw, "crystal_input")
    wrap(paw, "configure")
    if args.case == "si":
        atoms, kpts = bulk("Si", "diamond", a=5.43), (4, 4, 4)
    elif args.case == "al":
        atoms, kpts = bulk("Al", "fcc", a=4.05), (6, 6, 6)
    else:
        atoms, kpts = graphene(a=2.461, vacuum=6.0), (6, 6, 1)
        atoms.pbc = True
    if any(value < 1 for value in args.repeat) or (args.kmesh and any(value < 1 for value in args.kmesh)):
        parser.error("repeat and kmesh dimensions must be positive")
    atoms = atoms.repeat(args.repeat)
    kpts = tuple(args.kmesh) if args.kmesh else kpts
    options = {"threads": 8}
    if args.offload_bytes is not None:
        options["disk_offload_bytes"] = args.offload_bytes
    atoms.calc = Mandacaru(method="dft", xc="pbe", basis={"name": "PAW-LCAO", "size": "DZP"},
                           grid_spacing=0.20, kpts={"size": kpts, "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.05},
                           precision="float64", resources=options, txt=None)
    baseline = rss_bytes()
    start = time.perf_counter()
    energy = float(atoms.get_potential_energy())
    elapsed = time.perf_counter() - start
    context = atoms.calc.context
    result = {"case": args.case, "atoms": len(atoms), "kmesh": kpts, "energy_ev": energy, "wall_s": elapsed,
              "baseline_peak_rss_bytes": baseline, "peak_rss_bytes": rss_bytes(),
              "native_peak_bytes": context.memory_peak_bytes,
              "native_live_bytes": context.memory_used_bytes,
              "archive_bytes_written": context.disk_archived_bytes, "phases": phases}
    atoms.calc.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
