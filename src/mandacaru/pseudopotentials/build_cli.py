r"""``mandacaru-build``: generate pseudopotential datasets from the command line.

::

    mandacaru-build --pp PAW --relativistic --xc LDA --element Fe
    mandacaru-build --pp ONCV --element Fe Cu Ga --workers 3
    mandacaru-build --pp PAW --all --workers 7 --output staging/
    mandacaru-build --build-backend

Every dataset is generated natively (the reference atom, the partial waves,
the projectors), through the same functions a calculation uses:
:func:`~mandacaru.pseudopotentials.paw.generate_paw` (PAW-LCAO),
:func:`~mandacaru.pseudopotentials.paw.generate_upaw` (UPAW-LCAO),
:func:`~mandacaru.pseudopotentials.oncv.generate_oncv` (ONCVPSP) and
:func:`~mandacaru.pseudopotentials.generation.generate_pseudopotential`
(NCPP).  The radial kernels of that generation -- the tridiagonal eigenpair of
the uniform-grid radial equation and the Numerov recursions -- run in C
(:mod:`mandacaru.basis.radial_backend`, compiled on first use); ``--backend
python`` selects the reference kernels instead, which give the same numbers
more slowly.

A dataset is written to ``--output`` (default: the current directory, one
subdirectory per family) at the library stride.  ``--install`` writes into
Mandacaru's own library for the family instead -- the linked
``mandacaru-paw`` / ``mandacaru-oncvpsp`` checkout -- and so replaces the
dataset calculations load; it is never the default.

Each channel is checked after generation: its two lowest levels against the
reference (a ghost state is refused or repaired, per ``--ghosts``), and with
``--check`` the scattering phase against the all-electron atom as well.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

#: ``--pp`` spellings -> family name.
FAMILIES = {"paw": "paw-lcao", "paw-lcao": "paw-lcao",
            "upaw": "upaw-lcao", "upaw-lcao": "upaw-lcao",
            "oncv": "oncvpsp", "oncvpsp": "oncvpsp",
            "ncpp": "ncpp", "tm": "ncpp"}
#: Families whose generator takes ``xc``, ``relativity`` and ``ghosts``.
MODERN = ("paw-lcao", "upaw-lcao", "oncvpsp")


def build_parser() -> argparse.ArgumentParser:
    """The ``mandacaru-build`` argument parser."""
    from .io import PSEUDO_FORMATS
    from .oncv import GHOST_MODES

    parser = argparse.ArgumentParser(
        prog="mandacaru-build",
        description="Generate PAW-LCAO, UPAW-LCAO, ONCVPSP or NCPP datasets "
                    "natively, with the radial kernels in C.",
        epilog="examples:\n"
               "  mandacaru-build --pp PAW --relativistic --xc LDA --element Fe\n"
               "  mandacaru-build --pp ONCV --element Fe Cu --workers 2\n"
               "  mandacaru-build --pp PAW --all --workers 7 --output staging/\n"
               "  mandacaru-build --build-backend\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pp", default="PAW", metavar="FAMILY",
                        help="PAW (PAW-LCAO, default), UPAW, ONCV or NCPP")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--element", "-e", nargs="+", metavar="SYMBOL",
                       help="element(s) to generate")
    which.add_argument("--all", action="store_true",
                       help="every element of the library (Z <= --z-max)")
    parser.add_argument("--z-max", type=int, default=None,
                        help="last atomic number with --all (default 92)")
    parser.add_argument("--xc", default="LDA", type=str.lower,
                        choices=["lda", "pbe"],
                        help="exchange-correlation functional (default LDA)")
    rel = parser.add_mutually_exclusive_group()
    rel.add_argument("--relativistic", "--scalar", dest="relativity",
                     action="store_const", const="scalar",
                     help="scalar-relativistic reference atom (the default)")
    rel.add_argument("--dirac", dest="relativity", action="store_const",
                     const="dirac",
                     help="scalar-relativistic plus a spin-orbit term")
    rel.add_argument("--nonrelativistic", dest="relativity",
                     action="store_const", const="none",
                     help="non-relativistic reference atom")
    parser.add_argument("--output", "-o", default=None, metavar="DIR",
                        help="directory to write into (default: ./<family>/)")
    parser.add_argument("--install", action="store_true",
                        help="write into Mandacaru's library for the family, "
                             "replacing the dataset calculations load")
    parser.add_argument("--format", default="parquet", choices=PSEUDO_FORMATS)
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="elements generated in parallel (default 1)")
    parser.add_argument("--ghosts", default="repair", choices=GHOST_MODES,
                        help="repair (default), refuse or keep a ghost state")
    parser.add_argument("--check", action="store_true",
                        help="also compare every channel's scattering phase "
                             "with the all-electron atom")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "c", "python"],
                        help="radial kernels: C when available (auto), C or "
                             "fail (c), or the Python reference (python)")
    parser.add_argument("--build-backend", action="store_true",
                        help="compile the C radial backend, report, and exit")
    return parser


def _environment(backend: str) -> None:
    """Set the kernel policy before anything is generated (inherited by the
    worker processes)."""
    os.environ["MANDACARU_BACKEND"] = {"auto": "auto", "c": "c",
                                       "python": "numpy"}[backend]


def build_backend_command() -> int:
    """``mandacaru-build --build-backend``: compile and load the radial kernels."""
    from ..basis import radial_backend

    os.environ["MANDACARU_BACKEND"] = "auto"
    path = radial_backend.build_radial_backend(verbose=True)
    radial_backend._LIB = None
    radial_backend._attempted = path is not None
    uses_c, message = radial_backend.radial_backend_status(build=False)
    if uses_c:
        print(f"C radial backend: {message}")
        return 0
    print(f"C radial backend unavailable: {message}\n"
          "Generation will use the Python reference kernels, which give the "
          "same numbers more slowly.")
    return 1


def _directory(family: str, output, install: bool) -> str:
    if install:
        if family == "paw-lcao":
            from .paw import paw_library_path
            return paw_library_path()
        if family == "upaw-lcao":
            from .paw import upaw_library_path
            return upaw_library_path()
        if family == "oncvpsp":
            from .oncv import oncv_library_path
            return oncv_library_path()
        from .io import default_library_path
        return default_library_path()
    return os.path.join(output if output is not None else os.getcwd(), family)


def _generate(family: str, symbol: str, options: dict):
    if family == "paw-lcao":
        from .paw import generate_paw
        return generate_paw(symbol, **options)
    if family == "upaw-lcao":
        from .paw import generate_upaw
        return generate_upaw(symbol, **options)
    if family == "oncvpsp":
        from .oncv import generate_oncv
        return generate_oncv(symbol, **options)
    from ase.data import atomic_numbers

    from .generation import generate_pseudopotential
    from .io import generation_points
    return generate_pseudopotential(
        symbol, points=generation_points(atomic_numbers[symbol]))


def _levels_and_phase(family: str):
    if family in ("paw-lcao", "upaw-lcao"):
        from .paw import _paw_levels, log_derivative_paw
        return _paw_levels, log_derivative_paw
    from .oncv import _oncv_levels, log_derivative_ps
    return _oncv_levels, log_derivative_ps


def build_one(family: str, symbol: str, options: dict, directory: str,
              fmt: str, check: bool) -> tuple[bool, str]:
    """Generate, write and check one dataset; ``(ok, report)``."""
    from .io import STRIDE, library_file, save_pseudopotential
    from ..basis.radial_backend import radial_backend_status

    warnings.simplefilter("ignore", RuntimeWarning)
    started = time.perf_counter()
    try:
        pp = _generate(family, symbol, options)
        os.makedirs(directory, exist_ok=True)
        path = save_pseudopotential(pp, library_file(symbol, directory, fmt),
                                    format=fmt, stride=STRIDE)
    except Exception as error:                          # noqa: BLE001
        return False, (f"{symbol:>2}  FAILED after "
                       f"{time.perf_counter() - started:.1f} s  "
                       f"{type(error).__name__}: {error}")
    elapsed = time.perf_counter() - started
    uses_c, _message = radial_backend_status(build=False)
    lines = [f"{symbol:>2}  {pp!r}",
             f"    {elapsed:.1f} s, {'C' if uses_c else 'Python'} radial "
             f"kernels -> {path}"]
    if family in MODERN:
        from .oncv import ghost_errors, scattering_errors
        levels, log_derivative = _levels_and_phase(family)
        ghosts = ghost_errors(pp, levels)
        phases = scattering_errors(pp, log_derivative) if check else {}
        for l, channel in sorted(pp.channels.items()):
            first, second = (float(e) for e in levels(pp, l)[:2])
            reference = float(channel.reference_energies[0])
            line = (f"    l={l}  r_c={channel.r_cut:.3f}  eps_ref="
                    f"{reference:+.6f}  lowest-eps_ref={first - reference:+.1e}"
                    f"  {'GHOST' if l in ghosts else 'no ghost'}")
            if l in phases:
                near, far = phases[l]
                line += f"  phase {near:.3f}/{far:.3f} rad"
            lines.append(line)
        lines.append(f"    local potential: r_cl={pp.r_cut_local:.3f} Bohr, "
                     f"shift {pp.local_shift:+.1f} Ha")
    return True, "\n".join(lines)


def main(argv=None) -> int:
    """``mandacaru-build`` entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _environment(args.backend)
    if args.build_backend:
        return build_backend_command()

    family = FAMILIES.get(args.pp.strip().lower())
    if family is None:
        parser.error(f"--pp must be one of PAW, UPAW, ONCV, NCPP, not {args.pp!r}")
    if args.install and args.output is not None:
        parser.error("--install and --output are exclusive")
    relativity = args.relativity or "scalar"
    if family == "ncpp" and (args.relativity not in (None, "none")
                             or args.xc != "lda"):
        parser.error("the NCPP generator is non-relativistic LDA only; drop "
                     "--relativistic/--dirac/--xc, or use --pp PAW or ONCV")
    if args.all:
        from .io import LIBRARY_Z_MAX, library_elements
        symbols = list(library_elements(args.z_max or LIBRARY_Z_MAX))
    elif args.element:
        from ase.data import atomic_numbers
        symbols = [s.strip().capitalize() for s in args.element]
        unknown = [s for s in symbols if s not in atomic_numbers]
        if unknown:
            parser.error(f"unknown element(s): {', '.join(unknown)}")
    else:
        parser.error("give --element SYMBOL ... or --all")

    options = ({"xc": args.xc, "relativity": relativity,
                "ghosts": args.ghosts} if family in MODERN else {})
    directory = _directory(family, args.output, args.install)

    from ..basis.radial_backend import radial_backend_status
    uses_c, message = radial_backend_status()
    if args.backend == "c" and not uses_c:
        print(f"mandacaru-build: the C radial backend is unavailable: {message}",
              file=sys.stderr)
        return 2
    print(f"mandacaru-build: {family}, "
          + (f"{args.xc.upper()}, relativity={relativity}, ghosts={args.ghosts}"
             if family in MODERN else "LDA, non-relativistic")
          + f", {len(symbols)} element(s), {args.workers} worker(s)")
    print(f"radial kernels: {message}")
    print(f"writing to {directory}\n", flush=True)

    failures = 0
    jobs = [(family, s, options, directory, args.format, args.check)
            for s in symbols]
    if args.workers <= 1 or len(jobs) == 1:
        for job in jobs:
            ok, report = build_one(*job)
            failures += not ok
            print(report, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_one, *job) for job in jobs]
            for future in as_completed(futures):
                ok, report = future.result()
                failures += not ok
                print(report, flush=True)
    print(f"\n{len(jobs) - failures} of {len(jobs)} dataset(s) written"
          + (f", {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
