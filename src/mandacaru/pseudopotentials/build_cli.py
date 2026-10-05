r"""``mandacaru-build``: generate pseudopotential datasets from the command line.

::

    mandacaru-build --pp PAW --relativistic --xc LDA --element Fe
    mandacaru-build --pp PAW --dirac --element Fe Cu Ga --workers 3
    mandacaru-build --pp UPAW --element O
    mandacaru-build --pp PAW --all --workers 7 --output staging/
    mandacaru-build --build-backend

Every dataset is generated natively (the reference atom, the partial waves,
the projectors), through the same functions a calculation uses:
:func:`~mandacaru.pseudopotentials.paw.generate_paw` (PAW-LCAO) and
:func:`~mandacaru.pseudopotentials.paw.generate_upaw` (UPAW-LCAO).  The radial
kernels of that generation -- the tridiagonal eigenpair of
the uniform-grid radial equation and the Numerov recursions -- run in C
(:mod:`mandacaru.basis.radial_backend`, compiled on first use); ``--backend
python`` selects the reference kernels instead, which give the same numbers
more slowly.

A dataset is written to ``--output`` (default: the current directory, one
subdirectory per family) at the library stride.  ``--install`` writes into
Mandacaru's own library for the family instead -- the set's folder
(``lda-sr/``, ``lda-dirac/``, ...) of the checkout its environment variable
names (``MANDACARU_PAW_PATH``, ``MANDACARU_UPAW_PATH``) -- and so replaces the dataset
calculations load; it is never the default.

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
            "upaw": "upaw-lcao", "upaw-lcao": "upaw-lcao"}


def build_parser() -> argparse.ArgumentParser:
    """The ``mandacaru-build`` argument parser."""
    from .io import PSEUDO_FORMATS
    from .partial_waves import GHOST_MODES

    parser = argparse.ArgumentParser(
        prog="mandacaru-build",
        description="Generate PAW-LCAO or UPAW-LCAO datasets "
                    "natively, with the radial kernels in C.",
        epilog="examples:\n"
               "  mandacaru-build --pp PAW --relativistic --xc LDA --element Fe\n"
               "  mandacaru-build --pp PAW --dirac --element Fe Cu --workers 2\n"
               "  mandacaru-build --pp UPAW --element O\n"
               "  mandacaru-build --pp PAW --all --workers 7 --output staging/\n"
               "  mandacaru-build --build-backend\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pp", default="PAW", metavar="FAMILY",
                        help="PAW (PAW-LCAO, default) or UPAW")
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
                        help="write into the family's library, <checkout>/<xc>/ "
                             "of the checkout its MANDACARU_*_PATH variable "
                             "names, replacing the dataset calculations load")
    parser.add_argument("--format", default="parquet", choices=PSEUDO_FORMATS)
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="elements generated in parallel (default 1)")
    parser.add_argument("--ghosts", default=None, choices=GHOST_MODES,
                        help="repair, refuse or keep a ghost state, or flag: "
                             "write an element no repair "
                             "cleans with its defect recorded, so loading it "
                             "warns (default: repair)")
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
    radial_backend.build_radial_backend(verbose=True)
    radial_backend.reload_radial_backend()
    uses_c, message = radial_backend.radial_backend_status(build=False)
    if uses_c:
        print(f"C radial backend: {message}")
        return 0
    print(f"C radial backend unavailable: {message}\n"
          "Generation will use the Python reference kernels, which give the "
          "same numbers more slowly.")
    return 1


def _directory(family: str, output, install: bool, xc: str = "lda",
               relativity: str = "scalar") -> str:
    """Where the datasets go; ``--install`` resolves (and validates) the
    family's library variable and the set's folder (:mod:`.environment`:
    ``lda-sr/`` or ``lda-dirac/`` for PAW-LCAO)."""
    if install:
        from .environment import library_directory
        return library_directory(family, xc, must_exist=False,
                                 relativity=relativity)
    return os.path.join(output if output is not None else os.getcwd(), family)


def _generate(family: str, symbol: str, options: dict):
    if family == "paw-lcao":
        from .paw import generate_paw
        return generate_paw(symbol, **options)
    from .paw import generate_upaw
    return generate_upaw(symbol, **options)


def _levels_and_phase(family: str):
    from .paw import _paw_levels, log_derivative_paw
    return _paw_levels, log_derivative_paw


def build_one(family: str, symbol: str, options: dict, directory: str,
              fmt: str, check: bool) -> tuple[bool, str]:
    """Generate, write and check one dataset; ``(ok, report)``.

    Numerical ``RuntimeWarning`` noise is silenced for this call only; any
    other warning the generator raises (a cutoff past where a channel is
    trustworthy, say) is kept and listed in the report.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("always", UserWarning)
        ok, report = _build_one(family, symbol, options, directory, fmt, check)
    notes = sorted({str(w.message) for w in caught
                    if not issubclass(w.category, RuntimeWarning)})
    if notes:
        report += "".join(f"\n    warning: {note}" for note in notes)
    return ok, report


def worker_threads(workers: int) -> int:
    """BLAS/OpenMP threads per worker: the physical cores shared out.

    Each worker otherwise opens a pool as wide as the machine: seven workers
    ran 48 threads each on 12 cores (load average 150) and built a library
    set at a quarter of its speed.
    """
    from ..integrals._backend import grid_blas_threads

    return max(1, grid_blas_threads() // max(1, int(workers)))


def _limit_threads(threads: int) -> None:
    """Pool initializer: cap a worker's BLAS and OpenMP pools at ``threads``
    -- the environment for pools not loaded yet, threadpoolctl for those
    already loaded."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    from threadpoolctl import threadpool_limits

    threadpool_limits(threads)


def _build_one(family: str, symbol: str, options: dict, directory: str,
               fmt: str, check: bool) -> tuple[bool, str]:
    from .io import STRIDE, library_file, save_pseudopotential
    from ..basis.radial_backend import radial_backend_status

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
    lines = [f"{symbol:>2}  {pp!r}"]
    if getattr(pp, "defects", None):
        from .partial_waves import defect_message
        lines = [f"{symbol:>2}  FLAGGED  "
                 + defect_message(symbol, family, pp.defects), f"    {pp!r}"]
    lines.append(f"    {elapsed:.1f} s, {'C' if uses_c else 'Python'} radial "
                 f"kernels -> {path}")
    from .partial_waves import ghost_errors, scattering_errors
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
    for l in sorted(set(ghosts) - set(pp.channels)):
        lines.append(f"    l={l}  no projectors  local potential binds "
                     f"{ghosts[l]:+.1e} Ha below the atom  GHOST")
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
        parser.error(f"--pp must be one of PAW, UPAW, not {args.pp!r}")
    if args.install and args.output is not None:
        parser.error("--install and --output are exclusive")
    relativity = args.relativity or "scalar"
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
    ghosts = args.ghosts or "repair"
    options = {"xc": args.xc, "relativity": relativity, "ghosts": ghosts}
    try:
        directory = _directory(family, args.output, args.install, args.xc,
                               args.relativity or "scalar")
    except FileNotFoundError as error:
        print(f"mandacaru-build: {error}", file=sys.stderr)
        return 2

    from ..basis.radial_backend import radial_backend_status
    uses_c, message = radial_backend_status()
    if args.backend == "c" and not uses_c:
        print(f"mandacaru-build: the C radial backend is unavailable: {message}",
              file=sys.stderr)
        return 2
    print(f"mandacaru-build: {family}, {args.xc.upper()}, "
          f"relativity={relativity}, ghosts={ghosts}, "
          f"{len(symbols)} element(s), {args.workers} worker(s)")
    print(f"radial kernels: {message}")
    print(f"writing to {directory}\n", flush=True)

    failures = flagged = 0
    jobs = [(family, s, options, directory, args.format, args.check)
            for s in symbols]
    if args.workers <= 1 or len(jobs) == 1:
        for job in jobs:
            ok, report = build_one(*job)
            failures += not ok
            flagged += "  FLAGGED  " in report
            print(report, flush=True)
    else:
        from ase.data import atomic_numbers

        # Generation time grows steeply with Z (O 5 s, Fe 30 s, Th 670 s), so
        # the heaviest start first and the light ones fill in around them.
        jobs.sort(key=lambda job: -atomic_numbers[job[1]])
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_limit_threads,
                                 initargs=(worker_threads(args.workers),)) as pool:
            futures = {pool.submit(build_one, *job): job[1] for job in jobs}
            for future in as_completed(futures):
                try:
                    ok, report = future.result()
                except Exception as error:              # noqa: BLE001
                    # A worker that died (a crash in native code, say) takes
                    # only its own element down; the rest still report.
                    ok, report = False, (f"{futures[future]:>2}  FAILED  "
                                         f"{type(error).__name__}: {error}")
                failures += not ok
                flagged += "  FLAGGED  " in report
                print(report, flush=True)
    print(f"\n{len(jobs) - failures} of {len(jobs)} dataset(s) written"
          + (f", {flagged} flagged" if flagged else "")
          + (f", {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
