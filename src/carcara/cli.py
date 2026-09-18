# -*- coding: utf-8 -*-
# file: cli.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The ``carcara`` command line.

A thin front end over :class:`~carcara.algorithms.Carcara`: a geometry
in, an energy out, with every solver option exposed as a flag.  The one feature
that is *only* natural on the command line is the **dry run**:

.. code-block:: console

    $ carcara water.xyz --frozen-core --dry-run
    $ carcara H2O --cell 8 --basis NAO --basis-option size=DZP --device ibm-quantum --dry-run
    $ carcara --load-hamiltonian lih.parquet --dry-run --json

``--dry-run`` stops before any integral, mapping or circuit and prints the
qubit budget of the calculation (see :mod:`carcara.algorithms.dry_run`), which
is what you want to know before submitting to a QPU or a large simulator.
Without it the full variational run is performed:

.. code-block:: console

    $ carcara water.xyz --method adapt-vqe --basis FAO --h 0.25 --frozen-core
    $ carcara LiH --cell 10 --basis PAW --h 0.25

The geometry is any file :func:`ase.io.read` understands (``.xyz``, ``.cif``,
``POSCAR``, ...) or the name of a molecule in ASE's ``g2`` collection
(``H2O``, ``LiH``, ``NH3``, ...).  The real-space box **is the geometry's unit
cell**: a file must carry one (an extended-XYZ ``Lattice``, a CIF / POSCAR
cell), and a bare molecule name needs an explicit ``--cell`` (``--cell 10``
for a 10 Angstrom cube, ``--cell 12 10 10`` for an orthorhombic box, or the
nine components of the three lattice vectors), which centers the molecule in
that box.  There is no padding option.
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
    from .algorithms.calculator import DEFAULT_METHOD, STABLE_METHODS
    from .backends.hardware import available_devices
    from .optimizers.optim import NAMED_OPTIMIZERS

    parser = argparse.ArgumentParser(
        prog="carcara",
        description="Carcará -- variational quantum simulation of molecules "
                    "and crystals (ADAPT-VQE by default).",
        epilog="Examples:\n"
               "  carcara water.xyz --frozen-core --dry-run\n"
               "  carcara H2O --cell 8 --basis NAO --basis-option size=DZP --dry-run\n"
               "  carcara H2O --cell 8 --basis PAW --basis-option size=DZP --dry-run\n"
               "  carcara --load-hamiltonian lih.parquet --dry-run --json\n"
               "  carcara LiH --cell 10 --method adapt-vqe --pool qeb --h 0.3\n"
               "  carcara --build-backend\n"
               "  carcara --link-paw ~/Repositories/carcara-paw\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version",
                        version=f"carcara {__version__}")
    parser.add_argument("--link-paw", metavar="DIR", default=None,
                        help="link a checkout of the PAW dataset repository "
                             "(carcara-paw) into Carcará's library, then "
                             "exit.  The datasets are too large to ship, so "
                             "they live in their own repository and the "
                             "library holds a symlink to it.")
    parser.add_argument("--link-oncvpsp", metavar="DIR", default=None,
                        help="the same for the ONCVPSP datasets "
                             "(carcara-oncvpsp).")
    parser.add_argument("--pseudo-status", action="store_true",
                        help="report which pseudopotential libraries are "
                             "linked and how many datasets each serves, then "
                             "exit.")
    parser.add_argument("--build-backend", action="store_true",
                        help="compile the C integral backend (or report why it "
                             "cannot be), then exit.  Carcará does this by "
                             "itself on first use; run it here to pre-build, "
                             "e.g. in a container or CI, or to see the error.")

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
    run.add_argument("--method", default=DEFAULT_METHOD, choices=STABLE_METHODS,
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
                        help="kept for compatibility: the spin state comes from "
                             "--magmoms; an odd electron count is a doublet "
                             "by default (open-shell UHF natural orbitals)")
    system.add_argument("--magmoms", type=float, nargs="+", default=None,
                        help="initial magnetic moment per atom (their sum is "
                             "the number of unpaired electrons)")
    system.add_argument("--cell", type=float, nargs="+", default=None,
                        metavar="L",
                        help="unit cell in Angstrom for a geometry that has "
                             "none (a bare molecule name): one length for a "
                             "cube, three for an orthorhombic box, or nine "
                             "lattice-vector components; the molecule is "
                             "centered in it.  A cell given here overrides "
                             "the file's.")

    basis = parser.add_argument_group("basis and Hamiltonian")
    basis.add_argument("--basis", default="FAO",
                       help="basis family: FAO (default), NAO, NAO-AE, GTO, "
                            "PW, a named Gaussian set -- STO-nG, Pople "
                            "(6-31+G*, 6-311+G(2df,2p), ...), Dunning "
                            "(cc-pVDZ, aug-cc-pVTZ, cc-pCVDZ) or Karlsruhe "
                            "(def2-SVP, def2-TZVP, ...) -- or a "
                            "pseudopotential family, NCPP (Troullier-"
                            "Martins), ONCVPSP or PAW, for a valence-only "
                            "run (size=DZP etc. through --basis-option)")
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
    basis.add_argument("--mapping", default="jordan_wigner",
                       choices=("jordan_wigner", "parity", "bravyi_kitaev"),
                       help="fermion-to-qubit mapping (default jordan_wigner)")
    basis.add_argument("--load-hamiltonian", metavar="PATH", default=None,
                       help="reuse a cached qubit Hamiltonian (Parquet/JSON); "
                            "no geometry needed")
    basis.add_argument("--save-hamiltonian", metavar="PATH", default=None,
                       help="write the qubit Hamiltonian after building it")
    basis.add_argument("--verbose-hamiltonian", metavar="PATH", nargs="?",
                       const=True, default=False,
                       help="write the qubit Hamiltonian as readable JSON "
                            "(default hamiltonian.json).  The run trace only "
                            "reports its term count")
    basis.add_argument("--verbose-operators", metavar="PATH", nargs="?",
                       const=True, default=False,
                       help="write the ADAPT operator pool as JSON (default "
                            "pool.json).  The run trace only reports the "
                            "pool's name and size")

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

def parse_cell(values):
    """The ``--cell`` values as a 3x3 cell (Angstrom): 1, 3 or 9 numbers."""
    import numpy as np

    values = [float(v) for v in values]
    if len(values) == 1:
        cell = np.diag(values * 3)
    elif len(values) == 3:
        cell = np.diag(values)
    elif len(values) == 9:
        cell = np.asarray(values).reshape(3, 3)
    else:
        raise SystemExit(
            f"--cell takes 1, 3 or 9 numbers (Angstrom), got {len(values)}")
    if np.any(np.linalg.norm(cell, axis=1) <= 0.0):
        raise SystemExit("--cell needs three lattice vectors of non-zero length")
    return cell


def load_geometry(spec: str, cell=None, magmoms=None):
    """An ASE ``Atoms`` from a file path or a g2 molecule name.

    The real-space box is the geometry's unit cell, so one is required: from
    the file, or from ``cell`` (the ``--cell`` values, see :func:`parse_cell`),
    which centers the atoms in it.  A geometry with neither is refused.
    """
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
    if cell is not None:
        atoms.set_cell(parse_cell(cell))
        atoms.center()
    if not np.any(np.asarray(atoms.get_cell(), dtype=float)):
        raise SystemExit(
            f"geometry {spec!r} has no unit cell, and the real-space box is "
            "the cell.  Give one with --cell (e.g. --cell 10 for a 10 Angstrom "
            "cube, or --cell 12 10 10), or use a geometry file that carries "
            "its cell (an extended-XYZ Lattice, a CIF, a POSCAR).")
    if magmoms is not None:
        if len(magmoms) != len(atoms):
            raise SystemExit(
                f"--magmoms needs one value per atom ({len(atoms)}), got "
                f"{len(magmoms)}")
        atoms.set_initial_magnetic_moments(magmoms)
    return atoms


def solver_options(args) -> dict:
    """Keyword arguments for :class:`~carcara.algorithms.Carcara`."""
    from .algorithms.calculator import resolve_method

    basis = args.basis if not args.basis_option else \
        {"name": args.basis, **dict(args.basis_option)}
    options = dict(method=args.method, basis=basis, h=args.h,
                   verbose=not (args.quiet or args.json),
                   charge=args.charge, spin=args.spin,
                   frozen_core=args.frozen_core,
                   frozen_orbitals=args.frozen_orbitals,
                   mapping=args.mapping, optimizer=args.optimizer,
                   device=args.device, shots=args.shots,
                   load_hamiltonian=args.load_hamiltonian,
                   save_hamiltonian=args.save_hamiltonian or False,
                   verbose_hamiltonian=args.verbose_hamiltonian,
                   verbose_operators=args.verbose_operators,
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
    """Run the variational calculation and print the energy (eV)."""
    if atoms is None:
        result = calc.run()
    else:
        atoms.calc = calc
        atoms.get_potential_energy()
        result = calc.result
    energy = float(result.optimal_energy)
    unit = getattr(result, "energy_unit", "eV")
    print(f"\nFinal energy: {energy:.8f} {unit}")
    ops = getattr(result, "operators", None)
    if ops is not None:
        print(f"Operators grown: {len(ops)}")
    return 0


def build_backend_command() -> int:
    """``carcara --build-backend``: compile the C integral backend.

    The same call the engine makes on its own before the first integration, run
    explicitly and reported in full.  Returns 0 when the C backend is usable
    afterwards and 1 when the NumPy kernels are all that is available -- so it
    doubles as a build check in a container or CI.
    """
    from .integrals import _backend

    # An explicit request always really tries: the one-attempt-per-process
    # guard exists so a routine calculation does not re-run a doomed compile
    # for every engine it creates, not to refuse the user who asked for it.
    # The CARCARA_BACKEND policy is about which kernels a *calculation* runs
    # on, so it is set aside here -- asking to build is asking to build, and
    # `c` would otherwise raise instead of reporting the failure.
    _backend._build_attempted = False
    policy = os.environ.get("CARCARA_BACKEND", "auto").strip().lower() or "auto"
    os.environ["CARCARA_BACKEND"] = "auto"
    try:
        status = _backend.ensure_backend(build=True, verbose=True)
    finally:
        os.environ["CARCARA_BACKEND"] = policy
    if status.available:
        where = status.path or "(already loaded)"
        built = "compiled" if status.compiled else "already built"
        threads = (f", {status.n_threads} OpenMP thread(s)"
                   if status.n_threads else "")
        print(f"C integral backend: {built}{threads}\n  {where}")
        if policy == "numpy":
            print("\nCARCARA_BACKEND=numpy is set, so calculations in this "
                  "environment will still use the NumPy reference kernels.")
        return 0

    log = _backend._BUILD_DIR / "build.log"
    print(f"C integral backend unavailable: {status.message}")
    if log.is_file():
        print(f"\n--- {log} ---")
        print(log.read_text().strip()[-2000:])
    print("\nCarcará will run on the NumPy reference kernels, which give the "
          "same numbers more slowly.  A C compiler (and ideally CMake) on PATH "
          "is all that is needed; on macOS `brew install libomp` adds OpenMP.")
    return 1


def link_library_command(*, paw=None, oncvpsp=None) -> int:
    """``carcara --link-paw DIR`` / ``--link-oncvpsp DIR`` / ``--pseudo-status``.

    The ONCVPSP and PAW datasets are ~100 MB and ~200 MB for Z <= 92, too large
    to ship, so they live in their own repositories and the library holds a
    symlink to a checkout (see
    :mod:`carcara.pseudopotentials.link_library`).  This links them and then
    *proves* the link works by loading one dataset through the normal loader --
    a symlink that points at the wrong directory layout would otherwise only
    fail later, in the middle of a calculation.

    Returns 0 when every requested family is linked and loadable.
    """
    from .pseudopotentials.link_library import link_library, status

    requested = [("paw", paw), ("oncvpsp", oncvpsp)]
    failed = False
    for family, source in requested:
        if source is None:
            continue
        try:
            # From the command line, naming a path *is* the request to use it,
            # so an existing link is replaced; a real populated directory is
            # still refused, with a message saying so.
            link_library(family, source, force=True)
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            print(f"{family}: {exc}")
            failed = True

    for family, (target, source, count) in status().items():
        if not count:
            print(f"{family:8s} {target}: MISSING")
            continue
        where = f" -> {source}" if source and source != target else ""
        print(f"{family:8s} {target}{where}: {count} datasets")

    # Load one dataset per newly linked family: the real check.
    for family, source in requested:
        if source is None or failed:
            continue
        try:
            element = _probe_element(family)
            print(f"{family}: loaded {element} successfully")
        except Exception as exc:                      # noqa: BLE001 - reported
            print(f"{family}: linked, but loading a dataset failed: "
                  f"{type(exc).__name__}: {exc}")
            failed = True
    return 1 if failed else 0


def _probe_element(family: str) -> str:
    """Load the first available dataset of ``family`` through its own loader."""
    from .pseudopotentials.io import available_elements
    from .pseudopotentials.link_library import FAMILY_SUBDIRS, _family
    from .pseudopotentials.families import resolve_family
    from .pseudopotentials.io import library_root
    import os

    key = _family(family)
    folder = os.path.join(library_root(), FAMILY_SUBDIRS[key])
    elements = available_elements(folder)
    if not elements:
        raise FileNotFoundError(f"no datasets under {folder}")
    element = "H" if "H" in elements else elements[0]
    resolve_family(key).get(element)
    return element


def main(argv=None) -> int:
    """``carcara`` entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.build_backend:
        return build_backend_command()
    if args.link_paw or args.link_oncvpsp or args.pseudo_status:
        return link_library_command(paw=args.link_paw,
                                    oncvpsp=args.link_oncvpsp)
    if args.geometry is None and args.load_hamiltonian is None:
        parser.error("a geometry (file or molecule name) is required unless "
                     "--load-hamiltonian is given")

    from .algorithms import Carcara

    atoms = None
    if args.geometry is not None:
        atoms = load_geometry(args.geometry, args.cell, args.magmoms)
    calc = Carcara(**solver_options(args))
    if args.dry_run:
        return run_dry(calc, atoms, args)
    return run_full(calc, atoms, args)


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
