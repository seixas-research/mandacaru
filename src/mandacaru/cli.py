"""Small command-line adapter for the native calculation API."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    from .version import __version__
    from .specifications import METHODS, FUNCTIONALS

    parser = argparse.ArgumentParser(
        description="Mandacaru native electronic-structure calculations"
    )
    parser.add_argument(
        "geometry",
        nargs="?",
        help="Geometry JSON, or an ASE-supported file with the ase extra",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--method", choices=METHODS, default="dft")
    parser.add_argument("--xc", choices=FUNCTIONALS, default="lda")
    parser.add_argument("--basis", default="HAO")
    parser.add_argument("--grid-spacing", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--load-hamiltonian")
    parser.add_argument("--save-hamiltonian")
    parser.add_argument("--load-crystal")
    parser.add_argument("--output")
    options = parser.parse_args(argv)
    from . import Mandacaru
    from .structure import Geometry

    geometry = None
    if options.geometry:
        path = Path(options.geometry)
        if path.suffix == ".json":
            geometry = Geometry(**json.loads(path.read_text()))
        else:
            try:
                from ase.io import read
            except ImportError:
                parser.error(
                    "this geometry format needs the ase extra; JSON input needs no ASE"
                )
            geometry = read(path)
    if geometry is None and not (options.load_hamiltonian or options.load_crystal):
        parser.error("provide geometry or a cached problem")
    with Mandacaru(
        method=options.method,
        xc=options.xc,
        basis=options.basis,
        grid_spacing=options.grid_spacing,
        resources={
            "threads": options.threads,
            "memory_limit_bytes": options.memory_mb * 1024**2,
        },
        load_hamiltonian=options.load_hamiltonian,
        save_hamiltonian=options.save_hamiltonian,
        load_crystal=options.load_crystal,
    ) as calculator:
        result = (
            calculator.calculate(geometry, properties=("energy", "forces"))
            if options.forces
            else calculator.run(geometry)
        )
        report = {
            "energy_ev": result.in_units("eV"),
            "converged": result.success,
            "method": options.method,
        }
        if options.forces:
            report["forces_ev_per_angstrom"] = calculator.results["forces"].tolist()
        text = json.dumps(report, indent=2)
        if options.output:
            Path(options.output).write_text(text + "\n")
        print(text)
    return 0
