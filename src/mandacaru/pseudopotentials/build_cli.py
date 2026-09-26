r"""``mandacaru-build``: generate pseudopotential datasets from the command line.

::

    mandacaru-build --pp PAW --relativistic --xc LDA --element Fe
    mandacaru-build --pp ONCV --element Fe Cu Ga --workers 3
    mandacaru-build --pp ONCV --element Ce --occupations 4f=1 5d=1 6s=2
    mandacaru-build --pp ONCV --element Bi --freeze-subshell 4f
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
Mandacaru's own library for the family instead -- ``<checkout>/<xc>/`` of the
checkout its environment variable names (``MANDACARU_PAW_PATH``,
``MANDACARU_ONCVPSP_PATH``, ``MANDACARU_NCPP_PATH``,
``MANDACARU_UPAW_PATH``) -- and so replaces the dataset
calculations load; it is never the default.

Each channel is checked after generation: its two lowest levels against the
reference (a ghost state is refused or repaired, per ``--ghosts``), and with
``--check`` the scattering phase against the all-electron atom as well.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

#: ``--pp`` spellings -> family name.
FAMILIES = {"paw": "paw-lcao", "paw-lcao": "paw-lcao",
            "upaw": "upaw-lcao", "upaw-lcao": "upaw-lcao",
            "oncv": "oncvpsp", "oncvpsp": "oncvpsp",
            "ncpp": "ncpp", "tm": "ncpp"}
#: Families whose generator takes ``xc``, ``relativity`` and ``ghosts`` --
#: all of them.
MODERN = ("paw-lcao", "upaw-lcao", "oncvpsp", "ncpp")


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
               "  mandacaru-build --pp ONCV --element Ce "
               "--occupations 4f=1 5d=1 6s=2\n"
               "  mandacaru-build --pp ONCV --element Bi "
               "--freeze-subshell 4f\n"
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
                             "warns (default: repair; flag for NCPP)")
    parser.add_argument("--check", action="store_true",
                        help="also compare every channel's scattering phase "
                             "with the all-electron atom")
    parser.add_argument("--occupations", nargs="+", metavar="NL=COUNT",
                        help="ONCV, one element only: override neutral-atom "
                             "subshell occupations, for example "
                             "4f=1 5d=1 6s=2 for Ce")
    parser.add_argument("--freeze-subshell", nargs="+", metavar="NL",
                        help="ONCV: place occupied subshells in each "
                             "selected element's pseudopotential core, "
                             "for example 4f for Bi")
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


def _directory(family: str, output, install: bool, xc: str = "lda") -> str:
    """Where the datasets go; ``--install`` resolves (and validates) the
    family's library variable (:mod:`.environment`)."""
    if install:
        from .environment import library_directory
        return library_directory(family, xc, must_exist=False)
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
    from .generation import generate_pseudopotential
    return generate_pseudopotential(symbol, **options)


def _reference_occupations(symbol: str,
                           entries: list[str]) -> dict[tuple[int, int], int]:
    """Apply explicit subshell occupations to the neutral Aufbau reference.

    The total electron count and orbital capacities are checked by the ONCV
    generator. This parser only handles the concise command-line notation.
    """
    from ase.data import atomic_numbers
    from ..basis._config import ground_state_config

    result = ground_state_config(atomic_numbers[symbol])
    labels = {letter: l for l, letter in enumerate("spdf")}
    for entry in entries:
        match = re.fullmatch(r"([1-9][0-9]*)([spdf])=([0-9]+)", entry)
        if match is None:
            raise ValueError(f"invalid occupation {entry!r}; use 4f=1")
        n, letter, count = match.groups()
        key = (int(n), labels[letter])
        if key[0] <= key[1]:
            raise ValueError(f"invalid subshell {n}{letter}")
        result[key] = int(count)
    return {key: value for key, value in result.items() if value}


def _frozen_subshells(entries: list[str]) -> tuple[tuple[int, int], ...]:
    """Parse occupied core subshell labels from ``--freeze-subshell``."""
    labels = {letter: l for l, letter in enumerate("spdf")}
    result: set[tuple[int, int]] = set()
    for entry in entries:
        match = re.fullmatch(r"([1-9][0-9]*)([spdf])", entry)
        if match is None:
            raise ValueError(f"invalid frozen subshell {entry!r}; use 4f")
        n, letter = match.groups()
        key = (int(n), labels[letter])
        if key[0] <= key[1] or key in result:
            raise ValueError(f"invalid or repeated frozen subshell {entry!r}")
        result.add(key)
    return tuple(sorted(result))


def _levels_and_phase(family: str):
    if family == "ncpp":
        from .generation import kb_levels
        from .oncv import log_derivative_ps
        return kb_levels, log_derivative_ps
    if family in ("paw-lcao", "upaw-lcao"):
        from .paw import _paw_levels, log_derivative_paw
        return _paw_levels, log_derivative_paw
    from .oncv import _oncv_levels, log_derivative_ps
    return _oncv_levels, log_derivative_ps


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
        from .oncv import defect_message
        lines = [f"{symbol:>2}  FLAGGED  "
                 + defect_message(symbol, family, pp.defects), f"    {pp!r}"]
    lines.append(f"    {elapsed:.1f} s, {'C' if uses_c else 'Python'} radial "
                 f"kernels -> {path}")
    if family in MODERN:
        from .oncv import ghost_errors, scattering_errors
        levels, log_derivative = _levels_and_phase(family)
        # A Kleinman-Bylander dataset is read through the view the shared
        # tests understand (one projector, E_KB, the local channel's V_scr).
        subject = pp
        if family == "ncpp":
            from .generation import KBView
            subject = KBView(pp)
        ghosts = ghost_errors(subject, levels)
        phases = scattering_errors(subject, log_derivative) if check else {}
        for l, channel in sorted(subject.channels.items()):
            first, second = (float(e) for e in levels(subject, l)[:2])
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
        if family == "ncpp":
            lines.append(f"    local potential: the l={pp.local_l} channel"
                         + (", NLCC" if (pp.nlcc or {}).get("applied") else "")
                         + (f"; phase near = eps +- {subject.phase_window:g} Ha"
                            if check else ""))
        else:
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
    if family == "ncpp" and relativity == "dirac":
        parser.error("the NCPP family has one projector per l and no "
                     "spin-orbit term; use --relativistic, or --pp PAW or ONCV "
                     "for --dirac")
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
    if args.occupations:
        if family != "oncvpsp" or args.all or len(symbols) != 1:
            parser.error("--occupations requires --pp ONCV and one --element")
        try:
            from ase.data import atomic_numbers
            from .oncv import _validate_reference_configuration
            configuration = _reference_occupations(symbols[0],
                                                    args.occupations)
            _validate_reference_configuration(
                int(atomic_numbers[symbols[0]]), configuration)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    if args.freeze_subshell:
        if family != "oncvpsp" or args.all:
            parser.error("--freeze-subshell requires --pp ONCV and --element")
        try:
            frozen_subshells = _frozen_subshells(args.freeze_subshell)
        except ValueError as error:
            parser.error(str(error))

    # Without --ghosts each family keeps its own default: repair for
    # PAW/UPAW/ONCV, flag for NCPP (a single-channel atom has nothing to try).
    ghosts = args.ghosts or ("flag" if family == "ncpp" else "repair")
    options = ({"xc": args.xc, "relativity": relativity,
                "ghosts": ghosts} if family in MODERN else {})
    if args.occupations:
        options["reference_configuration"] = configuration
    if args.freeze_subshell:
        options["frozen_subshells"] = frozen_subshells
    try:
        directory = _directory(family, args.output, args.install, args.xc)
    except FileNotFoundError as error:
        print(f"mandacaru-build: {error}", file=sys.stderr)
        return 2

    from ..basis.radial_backend import radial_backend_status
    uses_c, message = radial_backend_status()
    if args.backend == "c" and not uses_c:
        print(f"mandacaru-build: the C radial backend is unavailable: {message}",
              file=sys.stderr)
        return 2
    print(f"mandacaru-build: {family}, "
          + (f"{args.xc.upper()}, relativity={relativity}, ghosts={ghosts}"
             if family in MODERN else "LDA, non-relativistic")
          + f", {len(symbols)} element(s), {args.workers} worker(s)")
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
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
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
