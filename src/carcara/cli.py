# -*- coding: utf-8 -*-
# file: cli.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The ``carcara`` command line.

A thin front end over :class:`~carcara.algorithms.QuantumCalculator`: a geometry
in, an energy out, with every solver option exposed as a flag.  The one feature
that is *only* natural on the command line is the **dry run**:

.. code-block:: console

    $ carcara water.xyz --frozen-core --dry-run
    $ carcara H2O --basis NAO --basis-option size=DZP --device ibm-quantum --dry-run
    $ carcara --load-hamiltonian lih.parquet --dry-run --json

``--dry-run`` stops before any integral, mapping or circuit and prints the
qubit budget of the calculation (see :mod:`carcara.algorithms.dry_run`), which
is what you want to know before submitting to a QPU or a large simulator.
Without it the full variational run is performed:

.. code-block:: console

    $ carcara water.xyz --method adapt-vqe --basis FAO --h 0.25 --frozen-core

The geometry is any file :func:`ase.io.read` understands (``.xyz``, ``.cif``,
``POSCAR``, ...) or the name of a molecule in ASE's ``g2`` collection
(``H2O``, ``LiH``, ``NH3``, ...).  A geometry without a unit cell is centered in
a box with ``--vacuum`` padding, as the real-space grid needs one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .version import __version__


# --------------------------------------------------------------------------- #
# Argument parsing.
# --------------------------------------------------------------------------- #

def _key_value(text: str):
    """Parse ``key=value`` with a JSON-then-string value (``size=DZP``, ``n=3``)."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"expected key=value, got {text!r}")
    key, raw = text.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def _frozen_core(text: str):
    """``--frozen-core`` value: ``auto`` / ``true`` / an integer / ``false``."""
    key = str(text).strip().lower()
    if key in ("auto", "true", "yes", "on"):
        return True
    if key in ("false", "no", "off", "0", "none"):
        return False
    try:
        return int(key)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--frozen-core takes 'auto', an integer or 'false', not {text!r}")


def build_parser() -> argparse.ArgumentParser:
    """The ``carcara`` argument parser."""
    from .algorithms.calculator import DEFAULT_METHOD, METHODS
    from .backends.hardware import available_devices
    from .optimizers.optim import NAMED_OPTIMIZERS

    parser = argparse.ArgumentParser(
        prog="carcara",
        description="Carcará -- variational quantum simulation of molecules "
                    "and crystals (ADAPT-VQE by default).",
        epilog="Examples:\n"
               "  carcara water.xyz --frozen-core --dry-run\n"
               "  carcara H2O --basis NAO --basis-option size=DZP --dry-run\n"
               "  carcara --load-hamiltonian lih.parquet --dry-run --json\n"
               "  carcara LiH --method adapt-vqe --pool qeb --h 0.3\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version",
                        version=f"carcara {__version__}")

    parser.add_argument("geometry", nargs="?", default=None,
                        help="geometry file readable by ASE (xyz, cif, ...) or "
                             "an ASE g2 molecule name (H2O, LiH, ...).  Not "
                             "needed with --load-hamiltonian.")

    run = parser.add_argument_group("run")
    run.add_argument("--dry-run", action="store_true",
                     help="estimate the qubit requirements and stop: no "
                          "integrals, no Hamiltonian, no circuits are executed")
    run.add_argument("--json", action="store_true",
                     help="print the dry-run estimate as JSON (implies "
                          "--quiet for the banner)")
    run.add_argument("--quiet", "-q", action="store_true",
                     help="suppress the banner and the solver trace")
    run.add_argument("--method", default=DEFAULT_METHOD, choices=METHODS,
                     help=f"variational method (default {DEFAULT_METHOD})")
    run.add_argument("--device", default="AER_simulator",
                     help="execution device: " + ", ".join(available_devices())
                          + ", or an Amazon Braket ARN (default AER_simulator)")
    run.add_argument("--device-qubits", type=int, default=None,
                     help="qubit capacity to compare against in a dry run, "
                          "for devices whose name does not fix one (e.g. a "
                          "specific IBM Quantum processor)")
    run.add_argument("--shots", type=int, default=0,
                     help="measurement shots (required for a real QPU)")

    system = parser.add_argument_group("system")
    system.add_argument("--charge", type=int, default=0)
    system.add_argument("--spin", action="store_true",
                        help="spin-polarized (high-spin) reference")
    system.add_argument("--magmoms", type=float, nargs="+", default=None,
                        help="initial magnetic moment per atom (their sum is "
                             "the number of unpaired electrons)")
    system.add_argument("--vacuum", type=float, default=3.0,
                        help="padding (Angstrom) used to box a geometry that "
                             "has no unit cell (default 3.0)")

    basis = parser.add_argument_group("basis and Hamiltonian")
    basis.add_argument("--basis", default="FAO",
                       help="basis family: FAO (default), NAO, NAO-AE, GTO, "
                            "PW, or a named Gaussian set -- STO-nG, Pople "
                            "(6-31+G*, 6-311+G(2df,2p), ...), Dunning "
                            "(cc-pVDZ, aug-cc-pVTZ, cc-pCVDZ) or Karlsruhe "
                            "(def2-SVP, def2-TZVP, ...)")
    basis.add_argument("--basis-option", action="append", type=_key_value,
                       default=[], metavar="KEY=VALUE",
                       help="basis option, repeatable (size=DZP, "
                            "energy_shift=0.03, n_gaussians=3, "
                            "energy_cutoff=300, ...)")
    basis.add_argument("--h", type=float, default=0.20,
                       help="real-space grid spacing in Angstrom (default 0.20)")
    basis.add_argument("--frozen-core", nargs="?", const=True, default=False,
                       type=_frozen_core, metavar="N|auto",
                       help="frozen-core approximation: 'auto' (the noble-gas "
                            "core; also the bare flag) or an integer number "
                            "of lowest MOs")
    basis.add_argument("--frozen-orbitals", type=int, nargs="+", default=None,
                       help="explicit spatial-MO indices to freeze")
    basis.add_argument("--pseudopotentials", action="store_true",
                       help="(experimental) valence-only run with the bundled "
                            "Troullier-Martins pseudopotentials; not part of "
                            "the stable API")
    basis.add_argument("--mapping", default="jordan_wigner",
                       choices=("jordan_wigner", "parity", "bravyi_kitaev"),
                       help="fermion-to-qubit mapping (default jordan_wigner)")
    basis.add_argument("--load-hamiltonian", metavar="PATH", default=None,
                       help="reuse a cached qubit Hamiltonian (Parquet/JSON); "
                            "no geometry needed")
    basis.add_argument("--save-hamiltonian", metavar="PATH", default=None,
                       help="write the qubit Hamiltonian after building it")

    solver = parser.add_argument_group("solver")
    solver.add_argument("--pool", default="fermionic",
                        choices=("fermionic", "qubit", "qeb", "ceo"),
                        help="ADAPT operator pool (default fermionic)")
    solver.add_argument("--optimizer", default="COBYLA",
                        choices=tuple(NAMED_OPTIMIZERS),
                        help="classical optimizer (default COBYLA)")
    solver.add_argument("--max-iterations", type=int, default=None,
                        help="ADAPT growth steps (adaptive methods)")
    solver.add_argument("--gradient-tolerance", type=float, default=None,
                        help="ADAPT convergence threshold on max|grad|")
    solver.add_argument("--num-states", type=int, default=None,
                        help="number of states for the subspace methods")
    solver.add_argument("--output", metavar="PATH", default=None,
                        help="ADAPT output.txt log path")
    return parser


# --------------------------------------------------------------------------- #
# Geometry and calculator construction.
# --------------------------------------------------------------------------- #

def load_geometry(spec: str, vacuum: float = 3.0, magmoms=None):
    """An ASE ``Atoms`` from a file path or a g2 molecule name, boxed if needed."""
    import numpy as np

    if os.path.exists(spec):
        from ase.io import read
        atoms = read(spec)
    else:
        try:
            from ase.build import molecule
            atoms = molecule(spec)
        except KeyError:
            raise SystemExit(
                f"geometry {spec!r} is neither a readable file nor a molecule "
                "name from ASE's g2 collection")
    if not np.any(np.asarray(atoms.get_cell(), dtype=float)):
        atoms.center(vacuum=float(vacuum))
    if magmoms is not None:
        if len(magmoms) != len(atoms):
            raise SystemExit(
                f"--magmoms needs one value per atom ({len(atoms)}), got "
                f"{len(magmoms)}")
        atoms.set_initial_magnetic_moments(magmoms)
    return atoms


def solver_options(args) -> dict:
    """Keyword arguments for :class:`~carcara.algorithms.QuantumCalculator`."""
    from .algorithms.calculator import resolve_method

    basis = args.basis if not args.basis_option else \
        {"name": args.basis, **dict(args.basis_option)}
    options = dict(method=args.method, basis=basis, h=args.h,
                   vacuum=args.vacuum, verbose=not (args.quiet or args.json),
                   charge=args.charge, spin=args.spin,
                   frozen_core=args.frozen_core,
                   frozen_orbitals=args.frozen_orbitals,
                   pseudopotentials=args.pseudopotentials,
                   mapping=args.mapping, optimizer=args.optimizer,
                   device=args.device, shots=args.shots,
                   load_hamiltonian=args.load_hamiltonian,
                   save_hamiltonian=args.save_hamiltonian or False,
                   dry_run=args.dry_run)
    _name, cls = resolve_method(args.method)
    adaptive = hasattr(cls, "_select_operator")
    if adaptive:
        options["pool"] = args.pool
        if args.max_iterations is not None:
            options["max_iterations"] = args.max_iterations
        if args.gradient_tolerance is not None:
            options["gradient_tolerance"] = args.gradient_tolerance
        if args.output is not None:
            options["output"] = args.output
    if args.num_states is not None:
        options["num_states"] = args.num_states
    return options


# --------------------------------------------------------------------------- #
# Entry points.
# --------------------------------------------------------------------------- #

def run_dry(calc, atoms, args) -> int:
    """Estimate the qubit budget, print it, and stop."""
    estimate = calc.dry_run(atoms)
    if args.device_qubits is not None:
        estimate.device_qubits = int(args.device_qubits)
        estimate.fits_device = estimate.n_qubits <= estimate.device_qubits
    if args.json:
        print(estimate.to_json())
    else:
        if not args.quiet:
            from .utils import banner
            banner.show()
        print(estimate.summary())
    return 0


def run_full(calc, atoms, args) -> int:
    """Run the variational calculation and print the energy."""
    from .units import from_hartree

    if atoms is None:
        result = calc.run()
    else:
        atoms.calc = calc
        atoms.get_potential_energy()
        result = calc.result
    energy = float(result.optimal_energy)
    print(f"\nFinal energy: {energy:.8f} Ha  ({from_hartree(energy, 'eV'):.6f} eV)")
    ops = getattr(result, "operators", None)
    if ops is not None:
        print(f"Operators grown: {len(ops)}")
    return 0


def main(argv=None) -> int:
    """``carcara`` entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.geometry is None and args.load_hamiltonian is None:
        parser.error("a geometry (file or molecule name) is required unless "
                     "--load-hamiltonian is given")

    from .algorithms import QuantumCalculator

    atoms = None
    if args.geometry is not None:
        atoms = load_geometry(args.geometry, args.vacuum, args.magmoms)
    calc = QuantumCalculator(**solver_options(args))
    if args.dry_run:
        return run_dry(calc, atoms, args)
    return run_full(calc, atoms, args)


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
