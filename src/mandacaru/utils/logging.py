# -*- coding: utf-8 -*-
# file: utils/logging.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Structured runtime logging for ADAPT-VQE (the ``output.txt`` protocol).

:class:`AdaptOutputLogger` writes a human-readable, machine-parseable trace of an
ADAPT-VQE run to a plain text file (``output.txt`` by default) *as the loop
runs* -- every block is flushed immediately, so a reader can follow the
optimization live.  The file has these kinds of section:

* the start-up **banner** (:func:`mandacaru.utils.banner.lines`), written once at
  the very top of the file so the log carries its own provenance;
* a **system** block with the geometry of this step and the explicit unit-cell
  parameters;
* an **electrons** block with how the electronic problem was posed -- basis,
  grid, kinetic operator, k-points, spin, the fermion-to-qubit mapping and the
  size of the register and Hamiltonian it produced;
* an **optimization setup** block naming the classical optimizer, how the pool
  screening gradient is computed and against which tolerance, the operator pool,
  and the reference (Hartree-Fock) energy the VQE starts from;
* one **iteration** block per accepted ADAPT operator, giving the pool's
  *size*, the operator selected to join the ansatz (with its screening
  gradient and Pauli expansion), the energy, the circuit metrics and the
  parameterized circuit's expressivity score :math:`E` (KL divergence from the
  Haar distribution);
* a **summary** block with the final parameterization;
* for a geometry step, a **forces** block with the atomic force vectors and
  their Hellmann-Feynman / Pulay breakdown (:func:`append_forces`).

Geometry steps concatenate
--------------------------
A geometry optimization runs one complete ADAPT-VQE per step, and every step
builds its own logger.  Truncating the file each time would leave only the last
step's iterations, so the *first* logger of a given path in a process truncates
and writes the banner and every later one **appends**.  The per-path step
counter is read with :func:`log_steps` and forgotten with :func:`reset_log`,
which is what a long-lived process -- a notebook cell re-run -- needs to start a
fresh file.  Each block is headed with its ``step:`` number, so a
relaxation reads as one continuous log and :func:`parse_output` can separate the
steps again (``parse_output(...)["steps"]``).

The pool's *contents* are not listed: a realistic pool grows with the fourth
power of the number of orbitals, and repeating it every iteration is what used
to make this file unreadable.  ``AdaptOutputLogger(..., log_pool=True)``
restores the listing, but the supported route is the driver option
``verbose_operators=True``, which writes the pool once as structured JSON (see
:mod:`mandacaru.utils.dumps`).

Above :data:`DETAILED_LOG_MAX_QUBITS` qubits the log stays smaller still: the
selected operator's expansion and the summary's operator sequence are omitted
too (they are Pauli-string expansions that grow with the register); the
energies, gradients and circuit metrics are always written.

The format uses ``KEY: value`` lines and fixed section banners so it can be
parsed by simple line scanning (see :func:`parse_output` for a reference reader
used by the tests).
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Iterable, Sequence

import numpy as np

_BANNER = "=" * 72
_RULE = "-" * 72
#: One level of indentation inside a block.  The section markers, the rules and
#: the banner sit at column 0; everything a block *contains* is indented one
#: level, and what a keyed line contains (the geometry under ``geometry:``, the
#: rows of a table) one level further.
INDENT = "    "
#: Widest register whose log lists the operator pool and the selected operator.
DETAILED_LOG_MAX_QUBITS = 20

#: The destination meaning **standard output** wherever this module takes a
#: path.  A run without a ``txt=`` file still has a report to make, and it is
#: the same report: writing it here routes every block through the one renderer
#: instead of a second, drifting one (see
#: :meth:`~mandacaru.algorithms.base.VariationalDriver.log_targets`).
STDOUT = "<stdout>"


def _is_stdout(target) -> bool:
    """Whether ``target`` names standard output rather than a file."""
    return str(target) == STDOUT


def _targets(path) -> tuple[str, ...]:
    """Normalize one destination, several, or none into a tuple.

    A destination is a file path or :data:`STDOUT`; ``None`` is "nowhere", and
    a sequence writes the same block to each -- which is how ``txt=<path>``
    together with ``trace=True`` puts one report in two places.
    """
    if path is None:
        return ()
    if isinstance(path, (str, os.PathLike)):
        return (os.fspath(path),)
    return tuple(os.fspath(p) for p in path)


def _key(target) -> str:
    """Registry key of a destination: its real path, or the stdout sentinel."""
    return target if _is_stdout(target) else os.path.abspath(str(target))


@contextlib.contextmanager
def _open_target(target, mode: str = "a"):
    """Yield a writable handle for ``target``; only a file is closed again."""
    if _is_stdout(target):
        yield sys.stdout
        sys.stdout.flush()
    else:
        with open(target, mode, encoding="utf-8") as fh:
            yield fh


def _write_lines(path, lines: Iterable[str]) -> None:
    """Append ``lines`` to every destination in ``path``."""
    lines = list(lines)
    for target in _targets(path):
        with _open_target(target) as fh:
            for line in lines:
                fh.write(line + "\n")


#: Geometry steps already written to each log destination in this process,
#: keyed by absolute path (or by :data:`STDOUT`).  The first logger of a
#: destination truncates the file and writes the banner; every later one
#: appends a further step (see :func:`log_steps`).
_LOG_STEPS: dict[str, int] = {}


def log_steps(path) -> int:
    """How many blocks this process has written to ``path`` (0 = none yet).

    ``path`` is one destination or several; several report the largest count,
    which is the step they were last written together at.
    """
    return max((int(_LOG_STEPS.get(_key(t), 0)) for t in _targets(path)),
               default=0)


def reset_log(path: str | None = None) -> None:
    """Forget the step counter of ``path`` (or of every path when ``None``).

    The next :class:`AdaptOutputLogger` for that path then truncates the file
    and writes the banner again, as though the process had just started.  A
    script does not need this -- a fresh process starts with an empty registry
    -- but a notebook or a driver loop that wants each run in its own file does.
    """
    if path is None:
        _LOG_STEPS.clear()
    else:
        for target in _targets(path):
            _LOG_STEPS.pop(_key(target), None)


def _indent(lines: Iterable[str], level: int = 1) -> list[str]:
    """``lines`` prefixed with ``level`` indentation levels.

    Empty lines stay empty -- a blank separator must not become trailing
    whitespace.
    """
    pad = INDENT * level
    return [pad + line if line else line for line in lines]


class AdaptOutputLogger:
    """Append-only writer for the ADAPT-VQE ``output.txt`` protocol.

    Parameters
    ----------
    path : str or sequence of str
        Where the blocks go: a file (``"output.txt"`` by convention),
        :data:`STDOUT`, or several destinations, which each receive the same
        blocks.  The **first** logger of a file in a process truncates it and
        writes the banner; every later one appends its block, so the steps of a
        geometry optimization concatenate instead of erasing each other.
        Standard output is only ever appended to.
    n_qubits : int, optional
        Register width.  Above :data:`DETAILED_LOG_MAX_QUBITS` the selected
        operator's expansion is left out of the log.
    log_pool : bool
        List the whole operator pool in every iteration block (default
        ``False``).  Off, the log records only the pool's size and the
        operator that was selected; the driver option ``verbose_operators``
        writes the pool once to ``pool.json``, which is easier to read and does
        not grow with the iteration count.
    append : bool, optional
        Override the automatic choice: ``True`` always appends (never truncates
        an existing file), ``False`` always truncates.  ``None`` (the default)
        truncates only the first time this process writes to ``path``.
    banner : bool
        Write the start-up banner (:func:`mandacaru.utils.banner.lines`) at the
        top of a **freshly opened** file (default ``True``).  A file that
        already has content is appended to, so the banner is never repeated.

    Attributes
    ----------
    step : int
        1-based index of this block within the file -- the geometry step of a
        relaxation.  Written as the ``step:`` line of every block.
    """

    def __init__(self, path: str = "output.txt", n_qubits: int | None = None,
                 log_pool: bool = False, append: bool | None = None,
                 banner: bool = True):
        self.n_qubits = None if n_qubits is None else int(n_qubits)
        #: Whether iteration blocks expand the selected operator in Pauli strings.
        self.detailed = (self.n_qubits is None
                         or self.n_qubits <= DETAILED_LOG_MAX_QUBITS)
        #: Whether iteration rows are followed by the whole pool listing.
        self.log_pool = bool(log_pool)
        #: Whether the iteration table's heading has been written yet.
        self._table_open = False
        #: Columns of this run's table, fixed when the first row is written.
        self._columns = None

        targets = _targets(path)
        if not targets:
            raise ValueError("AdaptOutputLogger needs a destination: a file "
                             "path, logging.STDOUT, or several of them")
        # Every destination this logger writes to, in order, and the first of
        # them -- which is what `path` meant when there could only be one.
        self.targets = targets
        self.path = targets[0]

        # One geometry step is one block; the first block of a process opens the
        # file for writing (truncating a previous run's log), every later one
        # appends so a relaxation's steps accumulate.  With several
        # destinations they advance together, so the file and the terminal
        # never disagree about which geometry step this is.
        keys = [_key(target) for target in targets]
        step = max(_LOG_STEPS.get(key, 0) for key in keys) + 1
        if append is None:
            append = step > 1
        for key in keys:
            parent = "" if _is_stdout(key) else os.path.dirname(key)
            if parent:
                # Other writers (checkpoints, dumps) create their parents; a log
                # named under a missing directory used to raise *after* the step
                # counter had advanced, so a retry started at step 2.
                os.makedirs(parent, exist_ok=True)

        #: ``(handle, owned)`` per destination; only an owned handle is closed.
        self._sinks: list[tuple[Any, bool]] = []
        fresh = []
        for target in targets:
            if _is_stdout(target):
                # Standard output is never truncated and never closed, and the
                # banner has already gone there once per process
                # (`VariationalDriver._show_banner`), so it is never "fresh".
                self._sinks.append((sys.stdout, False))
                continue
            fh = open(target, "a" if append else "w", encoding="utf-8")
            self._sinks.append((fh, True))
            if fh.tell() == 0:
                fresh.append(fh)
        # Only now: the counter records blocks written, so a failed open must
        # not consume a step number.
        self.step = step
        for key in keys:
            _LOG_STEPS[key] = step
        wrote_banner = bool(fresh) and banner
        if wrote_banner:
            # The banner belongs to the file, not to the block: it is written
            # only when the file starts empty, so appending never repeats it
            # (and an explicit append= to a file that does not exist yet still
            # gets it).
            from . import banner as _banner
            self._write(fresh, _banner.lines())
        elif step > 1:
            # Appending under an earlier block, which closed with a rule: one
            # blank line separates the two instead of stacking the rules.
            self._emit("")

    # -- low-level helpers ------------------------------------------------- #

    @staticmethod
    def _write(handles, lines: Iterable[str]) -> None:
        """Write ``lines`` to each of ``handles`` and flush them."""
        lines = list(lines)
        for fh in handles:
            for line in lines:
                fh.write(line + "\n")
            fh.flush()

    def _emit(self, *lines: str) -> None:
        """Write lines verbatim: the banner, the rules and the section markers."""
        self._write([fh for fh, _owned in self._sinks], lines)

    def _emit_body(self, *lines: str, level: int = 1) -> None:
        """Write lines as a block's *contents*, indented (see :data:`INDENT`)."""
        self._emit(*_indent(lines, level))

    # -- metadata / initialization block ----------------------------------- #

    def write_system(self, symbols: Sequence[str] | None = None,
                     positions=None, cell=None, pbc=None, magmoms=None,
                     units: str = "Angstrom", title: str = "ADAPT-VQE run",
                     extra: dict | None = None) -> None:
        """Write the ``[SYSTEM]`` block: this step's geometry, cell and spins.

        Parameters
        ----------
        symbols : sequence of str, optional
            Chemical symbols, one per atom.
        positions : (N, 3) array_like, optional
            Cartesian coordinates in ``units``.
        cell : (3, 3) array_like, optional
            Lattice/cell tensor (rows are lattice vectors) in ``units``.  ``None``
            (or an all-zero matrix) means a non-periodic molecule.
        pbc : (3,) array_like of bool, optional
            Whether each lattice direction is periodic.  A cell and periodicity
            are **different facts** -- Mandacaru always needs a cell, because it
            is the real-space box the grid is cut from, while ``pbc`` says
            whether the system repeats along each of its vectors -- so the block
            reports both.  ``None`` when the geometry did not carry it.
        magmoms : (N,) array_like, optional
            Initial magnetic moments, one per atom.  They select the spin state
            (`resolve_num_unpaired`), so a run that was given them should say
            so; ``None`` when the geometry did not carry them.
        units : str
            Length unit label for the geometry and cell.
        title : str
            Header title.
        extra : dict, optional
            Additional ``KEY: value`` metadata lines.
        """
        # A relaxation writes one titled block per geometry, so the title names
        # the step it belongs to; a single-point run's block is step 1 and says
        # nothing extra.
        heading = (title if self.step <= 1 else
                   f"{title} -- geometry step {self.step}")
        self._emit(_BANNER, INDENT + heading, _BANNER, "", "[SYSTEM]")
        self._emit_body(f"step: {self.step}")
        if extra:
            for key, value in extra.items():
                self._emit_body(f"{key}: {value}")

        # Geometry of this step.
        self._emit_body(f"units: {units}")
        if not self.detailed:
            self._emit_body(f"operator_details: omitted ({self.n_qubits} qubits "
                            f"> {DETAILED_LOG_MAX_QUBITS})")
        if symbols is not None and positions is not None:
            positions = np.asarray(positions, dtype=float)
            self._emit_body(f"n_atoms: {len(symbols)}", "geometry:")
            self._emit(*_indent(_geometry_rows(symbols, positions), 2))
        else:
            self._emit_body("geometry: (not provided)")

        # Explicit unit-cell parameters.
        cell = None if cell is None else np.asarray(cell, dtype=float)
        if cell is not None and np.any(cell):
            self._emit_body("cell_present: True", "cell_vectors:")
            for i, vec in enumerate(cell):
                self._emit_body(f"a{i + 1} = [{vec[0]:14.10f} {vec[1]:14.10f} "
                                f"{vec[2]:14.10f}]", level=2)
            a, b, c, alpha, beta, gamma = _cell_parameters(cell)
            self._emit_body(f"cell_lengths: a={a:.10f} b={b:.10f} c={c:.10f}",
                            f"cell_angles: alpha={alpha:.6f} beta={beta:.6f} "
                            f"gamma={gamma:.6f}")
        else:
            self._emit_body("cell_present: False",
                            "cell_vectors: (non-periodic)")

        # Periodicity per direction.  Not derivable from the cell: Mandacaru
        # always has one (it is the grid's box), periodic or not.
        self._emit_body(f"pbc: {_pbc_text(pbc)}")

        # Initial magnetic moments, when the geometry carried any.  Reported
        # per atom by 1-based index, matching the geometry rows above.
        self._emit_body(f"initial_magnetic_moments: {_magmom_text(magmoms)}")
        self._emit("")

    # -- basis block ------------------------------------------------------- #

    def write_basis(self, fields: dict, tables: dict | None = None) -> None:
        """Write the ``[BASIS]`` block: the single-particle basis that ran.

        A Mandacaru basis is built at run time from options, several of which
        have family defaults (a PAW-LCAO basis is Fourier-filtered unless told
        otherwise) or resolve to numbers only the builder knows (the cutoff
        radius an ``energy_shift`` gives each orbital, the file each dataset
        was read from).  The options the user typed are therefore not a record
        of the basis; this block is.  ``fields`` are keyed lines in the order
        given; each entry of ``tables`` is a list of rows, the first being the
        column names, written as an aligned table one level deeper.  Cells must
        not contain spaces: a reader splits the rows on whitespace.
        """
        self._emit("[BASIS]")
        for key, value in fields.items():
            if value is not None:
                self._emit_body(f"{key}: {value}")
        for name, rows in (tables or {}).items():
            rows = [[str(cell) for cell in row] for row in rows]
            if len(rows) < 2:
                continue
            widths = [max(len(row[i]) for row in rows)
                      for i in range(len(rows[0]))]
            self._emit_body(f"{name}:")
            for index, row in enumerate(rows):
                # Left-aligned, so a long last cell (a dataset's path) cannot
                # push the columns before it around.
                self._emit_body("  ".join(
                    cell.ljust(width) for cell, width in zip(row, widths)
                ).rstrip(), level=2)
                if index == 0:
                    self._emit_body("-" * (sum(widths) + 2 * (len(widths) - 1)),
                                    level=2)
        self._emit("")

    # -- electronic-structure block ---------------------------------------- #

    def write_electrons(self, fields: dict) -> None:
        """Write the ``[ELECTRONS]`` block: how the electronic problem was posed.

        The basis, the real-space grid, the fermion-to-qubit encoding and the
        size of what came out of it -- everything needed to know *which*
        Hamiltonian the iterations below belong to.  ``fields`` is written in the
        order given, so the caller fixes the reading order (the driver groups it
        from the discretization to the register).

        This is the configuration the standard-output trace prints in its header;
        with the trace routed to a log file
        (:class:`~mandacaru.algorithms.calculator.Mandacaru`'s ``trace``), the file is
        where it has to be.
        """
        self._emit("[ELECTRONS]")
        for key, value in fields.items():
            if value is not None:
                self._emit_body(f"{key}: {value}")
        self._emit("")

    # -- optimization setup block ------------------------------------------ #

    def write_optimizer_setup(self, optimizer_method: str,
                              reference_energy: float, energy_unit: str = "eV",
                              gradient_tol: float | None = None,
                              max_iterations: int | None = None,
                              initial_ansatz: str = "|HF> (0 parameters)",
                              extra: dict | None = None,
                              gradient_method: str | None = None,
                              gradient_formula: str | None = None,
                              gradient_units: str | None = None,
                              lineage: dict | None = None) -> None:
        """Write the classical optimizer and the pre-loop (reference) results.

        The *final classical ansatz optimization result leading into the VQE
        runtime* is, before the first operator is added, the Hartree-Fock
        reference energy of the empty ansatz -- logged here as the loop's
        starting point.

        The block is written in four groups, each contiguous so that a reader
        (and a diff between two runs) finds a fact where the fact it qualifies
        is: the **classical optimizer** and its budget, the **screening
        gradient** -- how it is computed, against which tolerance and in which
        unit -- then the driver's own settings (``extra``; for ADAPT-VQE the
        operator pool), the **loop's starting point**, and last the run's
        **lineage** (``lineage``; what a resumed run was restored from).
        """
        self._emit("[OPTIMIZATION SETUP]")
        self._emit_body(f"classical_optimizer: {optimizer_method}")
        if max_iterations is not None:
            self._emit_body(f"max_iterations: {max_iterations}")

        # The gradient group.  `gradient_method` says how the number in the
        # `|grad|` column was obtained, and it sits next to the tolerance it is
        # compared against -- the two are one fact read together.
        if gradient_method is not None:
            self._emit_body(f"gradient_method: {gradient_method}")
        if gradient_formula is not None:
            self._emit_body(f"gradient_formula: {gradient_formula}")
        if gradient_tol is not None:
            self._emit_body(f"gradient_tol: {gradient_tol:g}")
        if gradient_units is not None:
            self._emit_body(f"gradient_units: {gradient_units}")

        if extra:
            for key, value in extra.items():
                self._emit_body(f"{key}: {value}")

        # Where the loop starts: the unit the energies below are in, the
        # reference energy in it, and the ansatz that energy belongs to.
        self._emit_body(
            f"energy_unit: {energy_unit}",
            f"reference_energy_{energy_unit}: {reference_energy:.10f}",
            f"initial_ansatz: {initial_ansatz}")

        if lineage:
            for key, value in lineage.items():
                self._emit_body(f"{key}: {value}")
        self._emit("")

    # -- per-iteration table ------------------------------------------------ #

    #: Columns of the iteration table: ``(key, heading, width, format)``, in the
    #: order the properties were asked for -- energy, expressivity, the selected
    #: operator's type, the screening gradient, CNOTs, circuit depth and
    #: single-qubit gates -- with the growth index first and the operator's
    #: label last (it is the only variable-width field).
    #:
    #: This is a **file**, so nothing is dropped to fit a terminal and the
    #: numbers keep full precision.
    ITERATION_COLUMNS = (("iter", "iter", 4, "d"),
                         ("energy", "energy", 18, ".10f"),
                         ("expr", "expr", 10, ".6f"),
                         # "fermionic-double" is 16 characters: the log keeps
                         # the pool's own kind verbatim (the stdout table
                         # strips the pool prefix, which it can because the
                         # pool is named in its header), so the column has to
                         # be wide enough or every later column shifts.
                         ("type", "type", 17, "s"),
                         ("grad", "|grad|", 13, ".6e"),
                         # Classical effort of this growth step: how many
                         # parameter updates the optimizer made to re-optimize
                         # the grown ansatz (not cost evaluations -- the two
                         # differ by an order of magnitude for a
                         # gradient-based method).  The run's total is in the
                         # summary block.
                         ("steps", "steps", 7, "s"),
                         # Circuit cost, cheapest gate first: single-qubit
                         # gates, then CNOTs, then the depth they compile to.
                         ("1q", "1q", 7, "s"),
                         ("cnot", "cnot", 7, "s"),
                         ("depth", "depth", 7, "s"),
                         ("operator", "operator", 0, "s"))

    def _iteration_heading(self, energy_unit: str) -> str:
        cells = []
        for key, heading, width, _fmt in (self._columns
                                          or self.ITERATION_COLUMNS):
            label = f"energy ({energy_unit})" if key == "energy" else heading
            cells.append(label if key == "operator"
                         else f"{label:>{max(width, len(label))}}")
        return " ".join(cells).rstrip()

    def write_iteration(self, iteration: int, pool_operators: Sequence,
                        gradients: Iterable[float], selected_index: int,
                        expressivity: float | None, energy: float,
                        num_parameters: int, energy_unit: str = "eV",
                        metrics: Any = None,
                        optimizer_steps: int | None = None) -> None:
        """Append **one row** describing this ADAPT iteration.

        Each iteration is a single line and each tracked property is a column
        (:data:`ITERATION_COLUMNS`); the heading is written once, before the
        first row.  This replaced a twelve-line block per iteration whose bulk
        was the selected operator's Pauli expansion -- the operators belong in
        ``pool.json`` (driver option ``verbose_operators=True``), not repeated
        through the log.  The pool's size and type are in the setup block
        above, since neither changes between iterations.

        Parameters
        ----------
        iteration : int
            1-based macro-iteration index.
        pool_operators : sequence of PoolOperator
            The operator pool at this step (only the selected entry is read).
        gradients : iterable of float
            Screening gradient of each pool operator, in the same order.
        selected_index : int
            Index of the operator chosen for the ansatz.
        expressivity : float or None
            Expressivity score of the ansatz (``None`` if not computed).
        energy : float
            Energy after the inner re-optimization, in ``energy_unit``.
        num_parameters : int
            Parameters in the ansatz after this step.
        energy_unit : str
            Unit label for ``energy`` (default ``"eV"``).
        metrics : optional
            Object exposing ``cnot_count`` / ``depth`` / ``num_1q_gates``.
        optimizer_steps : int, optional
            Parameter updates the classical optimizer took at this step
            (``None`` when the method reports none, printed as ``-``).
        """
        grads = [float(g) for g in gradients]
        selected = pool_operators[selected_index]

        if not self._table_open:
            # Whether the expressivity is computed is a per-run setting, so the
            # first row decides for the whole table: not computed there means
            # not computed anywhere, and an all-"-" column is worse than none.
            self._columns = tuple(
                c for c in self.ITERATION_COLUMNS
                if c[0] != "expr" or expressivity is not None)
            heading = self._iteration_heading(energy_unit)
            self._emit("[ITERATIONS]")
            self._emit_body(heading, "-" * len(heading))
            self._table_open = True

        def count(value):
            return "-" if value is None else str(value)

        values = {
            "iter": int(iteration),
            "energy": float(energy),
            "expr": None if expressivity is None else float(expressivity),
            "type": str(selected.kind),
            "grad": abs(grads[selected_index]),
            "steps": count(optimizer_steps),
            "cnot": count(getattr(metrics, "cnot_count", None)),
            "depth": count(getattr(metrics, "depth", None)),
            "1q": count(getattr(metrics, "num_1q_gates", None)),
            "operator": selected.label if self.detailed else "(omitted)",
        }
        cells = []
        for key, heading, width, fmt in (self._columns
                                         or self.ITERATION_COLUMNS):
            label = f"energy ({energy_unit})" if key == "energy" else heading
            width = max(width, len(label))
            value = values[key]
            if key == "operator":
                cells.append(str(value))
            elif value is None:
                cells.append(f"{'-':>{width}}")
            elif fmt == "s":
                cells.append(f"{value:>{width}}")
            else:
                cells.append(f"{value:>{width}{fmt}}")
        self._emit_body(" ".join(cells).rstrip())

        if self.log_pool:
            # Opt-in only: the per-iteration pool listing this table replaced.
            self._emit_body("operator_pool:")
            for i, op in enumerate(pool_operators):
                marker = " (selected)" if i == selected_index else ""
                self._emit_body(f"[{i:3d}] {op.label}  "
                                f"|grad|={abs(grads[i]):.6e}{marker}", level=2)

    # -- summary block ----------------------------------------------------- #

    def write_summary(self, converged: bool, optimal_energy: float,
                      num_operators: int, energy_unit: str = "eV",
                      reference_energy: float | None = None,
                      correlation_energy: float | None = None,
                      num_parameters: int | None = None,
                      final_max_gradient: float | None = None,
                      expressivity: float | None = None,
                      num_evaluations: int | None = None,
                      optimizer_steps: int | None = None,
                      metrics: Any = None, optimizer: str | None = None,
                      extra: dict | None = None) -> None:
        """Write this step's ``[VARIATIONAL QUANTUM SUMMARY]``: the converged state.

        One per geometry -- it closes the variational run, not the relaxation
        (:func:`append_optimization_summary` closes that).

        Records the converged energy, the final ansatz size (operators /
        parameters), its expressivity, and the compiled-circuit cost (CNOTs,
        single-qubit gates, total gates, depth), plus the classical-optimizer
        effort -- everything describing the final variational state.

        The grown ansatz's *operator sequence* is not repeated here: the
        ``[ITERATIONS]`` table above already names the operator each step
        selected, in order, and it is on ``result.operators`` for a caller that
        wants it as data.  On a wide register that one line was longer than the
        table it duplicated.
        """
        self._emit(_BANNER, "[VARIATIONAL QUANTUM SUMMARY]")
        self._emit_body(f"converged: {converged}")
        # `optimizer` is accepted for the callers that pass it and is not
        # written: [OPTIMIZATION SETUP] owns the optimizer, and a fact stated in
        # two blocks is a fact that can disagree with itself.
        self._emit_body(f"optimal_energy_{energy_unit}: {optimal_energy:.10f}")
        if reference_energy is not None:
            self._emit_body(f"reference_energy_{energy_unit}: "
                            f"{reference_energy:.10f}")
        if correlation_energy is not None:
            self._emit_body(f"correlation_energy_{energy_unit}: "
                            f"{correlation_energy:.10f}")
        self._emit_body(f"num_operators: {num_operators}")
        if num_parameters is not None:
            self._emit_body(f"num_parameters: {num_parameters}")
        if expressivity is not None:
            self._emit_body(f"final_expressivity_E: {expressivity:.6f}")
        if final_max_gradient is not None:
            self._emit_body(f"final_max_gradient: {final_max_gradient:.6e}")
        if num_evaluations is not None:
            self._emit_body(f"cost_evaluations: {num_evaluations}")
        # The classical effort, in the optimizer's own currency: the `steps`
        # column of the table above, summed over the run.
        if optimizer_steps is not None:
            self._emit_body(f"optimizer_steps: {optimizer_steps}")

        # Final compiled-circuit cost.
        if metrics is not None and getattr(metrics, "cnot_count", None) is not None:
            self._emit_body(f"cnot_count: {metrics.cnot_count}",
                            f"circuit_depth: {metrics.depth}")
            if getattr(metrics, "total_gates", None) is not None:
                self._emit_body(f"one_qubit_gates: {metrics.num_1q_gates}",
                                f"total_gates: {metrics.total_gates}")
        if extra:
            for key, value in extra.items():
                self._emit_body(f"{key}: {value}")
        self._emit(_BANNER)

    # -- footer / teardown ------------------------------------------------- #

    def close(self) -> None:
        """Close the files this logger opened; standard output is left alone."""
        for fh, owned in self._sinks:
            if owned and not fh.closed:
                fh.close()

    def __enter__(self) -> "AdaptOutputLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _pbc_text(pbc) -> str:
    """The ``pbc:`` value: one flag per lattice direction, plus what they mean.

    ``a=True b=False c=False (1-D, periodic along a)`` reads the same way to a
    person and to the parser, which splits on the first colon only.
    """
    if pbc is None:
        return "(not provided)"
    flags = [bool(x) for x in np.atleast_1d(pbc)]
    if len(flags) == 1:                       # ASE accepts a scalar
        flags = flags * 3
    names = "abc"[:len(flags)]
    cells = " ".join(f"{name}={flag}" for name, flag in zip(names, flags))
    periodic = [name for name, flag in zip(names, flags) if flag]
    if not periodic:
        return f"{cells} (non-periodic)"
    return (f"{cells} ({len(periodic)}-D, periodic along "
            f"{', '.join(periodic)})")


def _magmom_text(magmoms) -> str:
    """The ``initial_magnetic_moments:`` value: one list, in atom order.

    ``[1.0, -1.0]`` -- the same literal a caller would pass to
    ``Atoms(magmoms=...)``, so it reads back with ``ast.literal_eval`` and
    lines up index for index with the ``geometry:`` rows above.  A
    closed-shell geometry is ``[0.0, 0.0]`` rather than a word: the list says
    what was asked for either way, and an all-zero one cannot be confused with
    nobody having looked, which is ``(not provided)``.
    """
    if magmoms is None:
        return "(not provided)"
    values = np.atleast_1d(np.asarray(magmoms, dtype=float))
    # repr() of a float keeps the decimal point and round-trips exactly.
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def _geometry_rows(symbols: Sequence[str], positions) -> list[str]:
    """One ``symbol x y z`` row per atom, unindented.

    Shared by the ``[SYSTEM]`` block and the relaxed geometry of
    :func:`append_optimization_summary`, so a geometry reads the same wherever
    it appears in the file.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    return [f"{str(sym):<3s} {x:16.10f} {y:16.10f} {z:16.10f}"
            for sym, (x, y, z) in zip(symbols, positions)]


def _force_table(symbols: Sequence[str], vectors,
                 magnitude: str = "|F|") -> list[str]:
    """The heading, rule and one row per atom of a force/gradient table.

    Components keep full precision -- this is a file, and the forces are what a
    relaxation is judged on.  ``magnitude`` names the last column, which is the
    row's norm (a force for the reported vectors, a gradient for the
    Hellmann-Feynman and Pulay breakdown).
    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    heading = (f"{'atom':>5s} {'symbol':<6s} {'x':>16s} {'y':>16s} "
               f"{'z':>16s} {magnitude:>14s}")
    rows = [heading, "-" * len(heading)]
    for index, (symbol, vector) in enumerate(zip(symbols, vectors), start=1):
        norm = float(np.linalg.norm(vector))
        rows.append(f"{index:5d} {str(symbol):<6s} "
                    f"{vector[0]:16.8f} {vector[1]:16.8f} {vector[2]:16.8f} "
                    f"{norm:14.8f}")
    return rows


def _force_block_lines(symbols: Sequence[str], forces, hellmann_feynman=None,
                       pulay=None, units: str = "eV/Angstrom",
                       extra: dict | None = None, unprojected=None,
                       step: int | None = None) -> list[str]:
    """Lines of a ``[FORCES]`` block: the vectors, their breakdown and norms.

    ``forces`` is the ASE-facing force :math:`-dE/dR`; ``hellmann_feynman`` and
    ``pulay`` are the *gradient* components :math:`+dE/dR` (the convention
    :class:`~mandacaru.algorithms.forces.ForceResult` uses), so the block states
    which is which rather than leaving the sign to be guessed.
    """
    forces = np.atleast_2d(np.asarray(forces, dtype=float))
    keys = [] if step is None else [f"step: {int(step)}"]
    keys.append(f"units: {units}")
    if unprojected is None:
        keys.append("convention: forces = -dE/dR (ASE sign); hellmann_feynman "
                    "and pulay are +dE/dR")
    else:
        # With a projection in place the reported force is *not* the negative
        # derivative of the discretized energy, and does not equal
        # -(hellmann_feynman + pulay): the block states the relation instead of
        # claiming a convention it no longer obeys.
        keys.append("convention: forces = forces_unprojected - "
                    "mean(forces_unprojected); forces_unprojected = -dE/dR "
                    "(ASE sign); hellmann_feynman and pulay are +dE/dR")
    if extra:
        keys += [f"{key}: {value}" for key, value in extra.items()]

    # The energy blocks above already closed with a rule, so this opens with a
    # blank line, like the other sections, and closes the step with the rule.
    # The block's keys take one indentation level and each table, being the
    # contents of the key that opens it, one more.
    lines = ["", "[FORCES]"] + _indent(keys)
    lines += _indent(["forces:"]) + _indent(_force_table(symbols, forces), 2)
    if unprojected is not None:
        lines += _indent(["forces_unprojected:"]) + _indent(
            _force_table(symbols, unprojected), 2)
    if hellmann_feynman is not None:
        lines += _indent(["hellmann_feynman:"]) + _indent(
            _force_table(symbols, hellmann_feynman, "|dE/dR|"), 2)
    if pulay is not None:
        lines += _indent(["pulay:"]) + _indent(
            _force_table(symbols, pulay, "|dE/dR|"), 2)

    norms = np.linalg.norm(forces, axis=1)
    norm_keys = [f"max_force: {float(norms.max()):.8f}",
                 f"rms_force: {float(np.sqrt((norms ** 2).mean())):.8f}"]
    if len(forces) > 1:
        # A free molecule feels no net force, so this is the pure discretization
        # artifact -- the number that says whether the force is usable for
        # geometry at all.  For a single atom the grid re-centers on it, so the
        # sum is the atom's own force and says nothing: it is left out.
        norm_keys.append(f"net_force: "
                         f"{float(np.abs(forces.sum(axis=0)).max()):.8f}")
    return lines + _indent(norm_keys) + [_BANNER]


def _performance_block_lines(stages=None, wall_time_s=None, resources=None,
                             step: int | None = None,
                             extra: dict | None = None) -> list[str]:
    """Lines of a ``[PERFORMANCE]`` block: where the step's time and memory went.

    ``stages`` is an ordered mapping of stage name to seconds (as
    :class:`~mandacaru.utils.profiling.Timings` accumulates them), ``wall_time_s``
    the step's end-to-end time and ``resources`` the machine context
    (:meth:`~mandacaru.utils.profiling.Timings.resources`).  A stage costing more
    than the timed total is the sign of work nobody timed, so the block states
    the sum next to the wall time rather than leaving it to be added up.
    """
    keys = [] if step is None else [f"step: {int(step)}"]
    lines = ["", "[PERFORMANCE]"] + _indent(keys)

    if stages:
        rows = []
        width = max(len(str(name)) for name in stages)
        for name, seconds in stages.items():
            rows.append(f"{str(name):<{width}s} {float(seconds):12.4f}")
        total = float(sum(float(v) for v in stages.values()))
        rows += ["-" * (width + 13),
                 f"{'sum of timed stages':<{width}s} {total:12.4f}"]
        lines += _indent(["stages (wall-clock seconds):"]) + _indent(rows, 2)
        if wall_time_s is not None:
            # What the timed stages do not account for: Hamiltonian
            # construction, the fermion-to-qubit mapping, materialization.
            lines += _indent([f"untimed_s: {float(wall_time_s) - total:.4f}"])
    if wall_time_s is not None:
        lines += _indent([f"wall_time_s: {float(wall_time_s):.4f}"])
    # The solver's own share of the step sits with the step's wall time, not
    # after the memory lines where `extra` would otherwise put it.
    extra = dict(extra or {})
    if "solver_wall_time_s" in extra:
        value = extra.pop("solver_wall_time_s")
        lines += _indent(["solver_wall_time_s: "
                          f"{_performance_value('solver_wall_time_s', value)}"])

    for table in (resources, extra):
        if table:
            lines += _indent([f"{key}: {_performance_value(key, value)}"
                              for key, value in table.items()])
    return lines + [_BANNER]


def _performance_value(key: str, value) -> str:
    """Render a performance value at a precision that suits what it measures.

    A time needs four decimals (a stage can be milliseconds), a memory figure
    one (a tenth of a MiB is already noise), and anything else is written as it
    is.  ``nan`` -- what an unavailable reading gives -- reads ``n/a`` rather
    than being printed as a number.
    """
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:                                   # nan
            return "n/a"
        if key.endswith("_MiB"):
            return f"{value:.1f}"
        return f"{value:.4f}" if key.endswith("_s") else f"{value:g}"
    return str(value)


def append_optimization_summary(path: str, history, symbols=None,
                                positions=None, units: str = "Angstrom",
                                energy_unit: str = "eV", fmax: float | None = None,
                                converged: bool | None = None,
                                status: str | None = None,
                                extra: dict | None = None) -> None:
    """Close the log with the relaxation's own two blocks.

    ``[GEOMETRY OPTIMIZATION SUMMARY]`` is the trajectory seen as one thing --
    how far the energy fell, how the largest force came down, the relaxed
    geometry -- and ``[RELAXATION COMPLETE]`` is the one-line footer that says the
    run finished, so a reader (or a script) can tell a complete log from one that
    was cut off mid-step.

    Parameters
    ----------
    path : str
        The log the steps were written to.
    history : sequence of dict
        One entry per geometry step, with ``energy`` (in ``energy_unit``),
        ``max_force`` and ``net_force`` (eV/Angstrom); optional ``wall_time_s``
        and ``com`` (center of mass) are used when present.
    symbols, positions : optional
        The relaxed geometry, written as the ``[SYSTEM]`` block writes one.
    fmax : float, optional
        The optimizer's force threshold.  Given, the footer states whether the
        run **converged** against it; without it the footer reports the final
        force and says only that the run finished -- the threshold is the
        optimizer's, and this is not the place to guess it.
    converged : bool, optional
        The optimizer's *own* verdict, which is what a run with constraints or a
        cell filter has: a bare force-threshold comparison is not that verdict.
        Given, it decides the footer and ``fmax`` only adds the number.
    status : str, optional
        The footer verbatim, for a caller that knows something this function
        cannot -- that the run was interrupted, or that completion was never
        signaled at all.
    extra : dict, optional
        Further ``KEY: value`` lines for the summary block.
    """
    history = [dict(entry) for entry in history]
    if not history:
        return
    first, last = history[0], history[-1]
    steps = len(history)

    keys = {"geometry_steps": steps, "units": units,
            f"initial_energy_{energy_unit}": f"{first['energy']:.10f}",
            f"final_energy_{energy_unit}": f"{last['energy']:.10f}",
            f"energy_change_{energy_unit}":
                f"{last['energy'] - first['energy']:+.10f}",
            "initial_max_force": f"{first['max_force']:.8f}",
            "final_max_force": f"{last['max_force']:.8f}",
            "final_net_force": f"{last['net_force']:.8f}"}
    if first.get("com") is not None and last.get("com") is not None:
        drift = float(np.linalg.norm(np.asarray(last["com"], float)
                                     - np.asarray(first["com"], float)))
        # A free molecule cannot translate under its own forces: whatever this
        # is, the grid put it there (see `project_translation`).
        keys["center_of_mass_drift"] = f"{drift:.8f}"
    total = sum(float(entry.get("wall_time_s") or 0.0) for entry in history)
    if total:
        keys["total_wall_time_s"] = f"{total:.4f}"
    for key, value in (extra or {}).items():
        keys[key] = value

    lines = ["", "[GEOMETRY OPTIMIZATION SUMMARY]"] + _indent(
        [f"{key}: {value}" for key, value in keys.items()])

    heading = (f"{'step':>5} {'energy (' + energy_unit + ')':>20} "
               f"{'max force':>16} {'net force':>16}")
    rows = [heading, "-" * len(heading)]
    for index, entry in enumerate(history, start=1):
        rows.append(f"{index:5d} {entry['energy']:20.10f} "
                    f"{entry['max_force']:16.8f} {entry['net_force']:16.8f}")
    lines += _indent(["convergence:"]) + _indent(rows, 2)

    if symbols is not None and positions is not None:
        lines += _indent(["relaxed_geometry:"])
        lines += _indent(_geometry_rows(symbols, positions), 2)
    lines.append(_BANNER)

    threshold = "" if fmax is None else f", threshold {float(fmax):.6f}"
    if status is None:
        if converged is not None:
            verdict = "converged" if converged else "NOT converged"
            status = (f"{verdict} after {steps} geometry steps (max force "
                      f"{last['max_force']:.6f}{threshold} eV/Angstrom)")
        elif fmax is not None:
            verdict = ("converged" if last["max_force"] <= float(fmax)
                       else "NOT converged")
            status = (f"{verdict} after {steps} geometry steps (max force "
                      f"{last['max_force']:.6f} "
                      f"{'<=' if verdict == 'converged' else '>'} "
                      f"{float(fmax):.6f} eV/Angstrom)")
        else:
            status = (f"finished after {steps} geometry steps "
                      f"(final max force {last['max_force']:.6f} eV/Angstrom)")
    lines += ["", "[RELAXATION COMPLETE]"] + _indent([f"status: {status}"])
    lines.append(_BANNER)

    _write_lines(path, lines)


def append_block(path: str, section: str, fields: dict,
                 step: int | None = None) -> None:
    """Append a simple ``[SECTION]`` block of ``KEY: value`` lines.

    For a block that is a flat record -- ``[MEASUREMENT]``, say -- where the
    caller already has the values in the order they should be read.  ``None``
    values are skipped, so an optional field costs nothing when it is absent.
    """
    lines = ["", f"[{section.strip('[]')}]"]
    if step is not None:
        lines += _indent([f"step: {int(step)}"])
    lines += _indent([f"{key}: {value}" for key, value in fields.items()
                      if value is not None])
    lines.append(_BANNER)
    _write_lines(path, lines)


def append_performance(path: str, stages=None, wall_time_s=None,
                       resources=None, step: int | None = None,
                       extra: dict | None = None) -> None:
    """Append a ``[PERFORMANCE]`` block to an existing ``output.txt``.

    Written once per evaluation, after the energies (and the forces, when they
    were computed -- their gradient is usually the largest single stage of a
    relaxation step, so a block that closed before them would account for the
    smaller half of the time).

    Parameters
    ----------
    path : str
        The log this step's other blocks were written to.
    stages : dict, optional
        Stage name -> seconds, in the order they should be listed.
    wall_time_s : float, optional
        End-to-end time of the step.
    resources : dict, optional
        Machine context (backend, threads, CPUs, memory); see
        :meth:`~mandacaru.utils.profiling.Timings.resources`.
    step : int, optional
        Geometry step; defaults to the number of blocks already written to
        ``path`` (:func:`log_steps`).
    extra : dict, optional
        Further ``KEY: value`` lines -- the QPU accounting, for instance.
    """
    if step is None:
        step = log_steps(path) or None
    lines = _performance_block_lines(stages=stages, wall_time_s=wall_time_s,
                                     resources=resources, step=step,
                                     extra=extra)
    _write_lines(path, lines)


def append_forces(path: str, symbols: Sequence[str], forces,
                  hellmann_feynman=None, pulay=None,
                  units: str = "eV/Angstrom", extra: dict | None = None,
                  unprojected=None, step: int | None = None) -> None:
    """Append a ``[FORCES]`` block to an existing ``output.txt``.

    The energies of a geometry step are written by the driver's own logger,
    which is closed by the time the nuclear gradient is known; this appends the
    forces to the same file, under the iteration table they belong to, without
    disturbing the per-path step counter.

    Parameters
    ----------
    path : str
        The log the step's energies were written to.
    symbols : sequence of str
        Chemical symbols, one per atom.
    forces : (N, 3) array_like
        The reported forces, :math:`-dE/dR` in ``units`` (ASE's convention).
    hellmann_feynman, pulay : (N, 3) array_like, optional
        The gradient breakdown, :math:`+dE/dR`; omitted from the block when not
        given.
    units : str
        Unit label of every vector in the block.
    extra : dict, optional
        Additional ``KEY: value`` lines (the force method, the residuals, ...).
    unprojected : (N, 3) array_like, optional
        The raw gradient, when ``forces`` has had its translational component
        projected out.  Given, it is written as a second table and the block's
        ``convention`` line states the relation between the two.
    step : int, optional
        Geometry step the block belongs to; defaults to the number of blocks
        already written to ``path`` (:func:`log_steps`), i.e. the step whose
        energies were logged last.
    """
    if step is None:
        step = log_steps(path) or None
    lines = _force_block_lines(symbols, forces,
                               hellmann_feynman=hellmann_feynman, pulay=pulay,
                               units=units, extra=extra,
                               unprojected=unprojected, step=step)
    _write_lines(path, lines)


def _cell_parameters(cell: np.ndarray):
    """Return ``(a, b, c, alpha, beta, gamma)`` for a (3, 3) cell tensor.

    Lengths in the cell's own units; angles in degrees (alpha between b and c,
    beta between a and c, gamma between a and b -- crystallographic convention).
    """
    cell = np.asarray(cell, dtype=float)
    a_vec, b_vec, c_vec = cell[0], cell[1], cell[2]
    a, b, c = (np.linalg.norm(a_vec), np.linalg.norm(b_vec),
               np.linalg.norm(c_vec))

    def angle(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    return (float(a), float(b), float(c),
            angle(b_vec, c_vec), angle(a_vec, c_vec), angle(a_vec, b_vec))


#: ``[GEOMETRY OPTIMIZATION SUMMARY]`` keys that are counts, not measurements.
_RELAXATION_COUNTS = ("geometry_steps",)

#: ``[PERFORMANCE]`` keys that are counts rather than measurements, so they read
#: back as ``int``.
_PERFORMANCE_COUNTS = ("step", "openmp_threads", "cpu_count", "qpu_jobs")

#: Section markers of the protocol, mapped to the key they fill (the step
#: markers are handled separately: they open a new geometry step).
_SECTIONS = {"[BASIS]": "basis",
             "[ELECTRONS]": "electrons", "[MEASUREMENT]": "measurement",
             "[OPTIMIZATION SETUP]": "setup", "[ITERATIONS]": "iterations",
             "[FORCES]": "forces", "[PERFORMANCE]": "performance",
             "[VARIATIONAL QUANTUM SUMMARY]": "summary",
             "[GEOMETRY OPTIMIZATION SUMMARY]": "optimization",
             "[RELAXATION COMPLETE]": "completion"}

#: Retired ``[OPTIMIZATION SETUP]`` keys, mapped to the key that replaced them.
#: The writer emits only the canonical name -- a replacement is carried through,
#: not aliased -- but the reader stays tolerant of files written before it, so a
#: log kept from an earlier version answers the same question through the same
#: key.  ``screening_gradient`` named the *quantity*; ``gradient_method`` names
#: what the block actually reports, which is how that quantity was computed.
_SETUP_ALIASES = {"screening_gradient": "gradient_method"}


def parse_output(path: str) -> dict:
    """Reference parser for an ADAPT ``output.txt`` (used by the tests).

    Reads the system, electrons and setup blocks as ``KEY: value`` lines, the
    ``[ITERATIONS]`` table as one record per row keyed by its column heading, and
    the ``[FORCES]`` block as per-atom vectors plus its scalar keys --
    demonstrating that the protocol is machine-parseable as written.

    A file written by a geometry optimization holds one block per step
    (see the module docstring).  ``result["steps"]`` is the list of those blocks,
    in order; the top-level ``system`` / ``electrons`` / ``setup`` /
    ``iterations`` / ``summary`` / ``forces`` / ``performance`` keys describe the
    **last** step, so reading a single-point log is unchanged.

    Setup keys that have been renamed are read back under their current name
    (:data:`_SETUP_ALIASES`), so a log written by an earlier version answers the
    same question through the same key as a log written today.
    """
    steps: list[dict[str, Any]] = []
    step: dict[str, Any] | None = None
    relaxation: dict[str, Any] = {}
    section = None
    columns: list[str] = []
    energy_unit = "eV"
    table = None                      # which force table rows are landing in
    basis_columns = None              # heading of the [BASIS] table being read

    def number(text):
        try:
            return float(text)
        except ValueError:
            return None

    def new_step() -> dict[str, Any]:
        """Start a block: one geometry step of the log."""
        nonlocal columns
        columns = []
        fresh: dict[str, Any] = {"system": {}, "setup": {}, "iterations": []}
        steps.append(fresh)
        return fresh

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            indent = len(raw) - len(raw.lstrip())
            if stripped == "[SYSTEM]":
                # The system block opens a geometry step; a second one starts
                # the next step of a relaxation.
                step = new_step()
                section = "system"
                continue
            if stripped in _SECTIONS:
                # A log need not start with a system block (a forces block appended on
                # its own is still a step), so the first section opens one.
                if step is None:
                    step = new_step()
                section = _SECTIONS[stripped]
                if section == "iterations":
                    columns = []
                elif section == "summary":
                    step["summary"] = {}
                elif section == "forces":
                    table = None
                    step["forces"] = {"symbols": []}
                elif section == "performance":
                    table = None
                    step["performance"] = {"stages_s": {}}
                elif section in ("electrons", "measurement"):
                    step[section] = {}
                elif section == "basis":
                    table = None
                    basis_columns = None
                    step["basis"] = {}
                elif section in ("optimization", "completion"):
                    # The relaxation's own blocks: they close the file, not a
                    # geometry step, so they are kept at the top level.
                    table = None
                    relaxation[section] = {}
                continue
            if step is None:
                # The banner, above the first block: provenance, not data.
                continue
            if not stripped or set(stripped) == {"-"} or set(stripped) == {"="}:
                continue

            if section in ("optimization", "completion"):
                block = relaxation[section]
                fields = stripped.split()
                if stripped.endswith(":"):
                    table = stripped[:-1]           # "convergence", "relaxed_geometry"
                    block[table] = []
                elif table is not None and indent >= len(INDENT) * 2:
                    # A table row, told from a key by its deeper indentation.
                    if fields[0] == "step":
                        continue                            # the table heading
                    if table == "convergence" and len(fields) >= 4:
                        block[table].append(
                            {"step": int(fields[0]), "energy": number(fields[1]),
                             "max_force": number(fields[2]),
                             "net_force": number(fields[3])})
                    else:                     # a geometry row: symbol x y z
                        block[table].append(
                            [fields[0]] + [number(v) for v in fields[1:4]])
                elif ":" in stripped:
                    key, _, value = stripped.partition(":")
                    key, value = key.strip(), value.strip()
                    numeric = number(value)
                    if numeric is None:
                        block[key] = value
                    elif key in _RELAXATION_COUNTS:
                        block[key] = int(numeric)
                    else:
                        block[key] = numeric
            elif section == "basis":
                block = step["basis"]
                if stripped.endswith(":") and indent < len(INDENT) * 2:
                    table = stripped[:-1]          # "datasets", "orbitals", ...
                    basis_columns = None
                    block[table] = []
                elif table is not None and indent >= len(INDENT) * 2:
                    cells = stripped.split()
                    if basis_columns is None:
                        basis_columns = cells               # the heading row
                    else:
                        def typed(cell):
                            # Counts and quantum numbers stay integers.
                            if cell.lstrip("+-").isdigit():
                                return int(cell)
                            value = number(cell)
                            return cell if value is None else value
                        block[table].append(
                            {name: typed(cell)
                             for name, cell in zip(basis_columns, cells)})
                elif ":" in stripped:
                    table = None
                    key, _, value = stripped.partition(":")
                    block[key.strip()] = value.strip()
            elif section == "system" and stripped.startswith(("a1 =", "a2 =",
                                                              "a3 =")):
                # "a1 = [ x y z ]": the lattice vector itself, which the cell
                # lengths and angles cannot recover (they lose the orientation).
                name, _, vector = stripped.partition("=")
                step["system"].setdefault("cell_vectors", []).append(
                    [number(v) for v in vector.strip(" []").split()])
            elif section == "system" and indent >= len(INDENT) * 2 \
                    and ":" not in stripped:
                # A geometry row, one level deeper than the `geometry:` key.
                fields = stripped.split()
                if len(fields) >= 4:
                    step["system"].setdefault("geometry", []).append(
                        [fields[0]] + [number(v) for v in fields[1:4]])
            elif section in ("system", "setup", "electrons", "measurement") \
                    and ":" in stripped:
                key, _, value = stripped.partition(":")
                key, value = key.strip(), value.strip()
                if section == "setup":
                    key = _SETUP_ALIASES.get(key, key)
                # `geometry:` and `cell_vectors:` open their own tables above;
                # keeping the empty string would overwrite the parsed rows.
                if not (key in ("geometry", "cell_vectors") and not value):
                    step[section][key] = value
            elif section == "summary" and ":" in stripped:
                key, _, value = stripped.partition(":")
                step["summary"][key.strip()] = value.strip()
            elif section == "forces":
                forces = step["forces"]
                fields = stripped.split()
                if stripped.endswith(":"):
                    # "forces:", "hellmann_feynman:", "pulay:" open a table.
                    table = stripped[:-1]
                    forces[table] = []
                elif fields[0] == "atom":
                    continue                            # the table's heading
                elif table is not None and fields[0].isdigit():
                    index, symbol = int(fields[0]), fields[1]
                    forces[table].append([float(x) for x in fields[2:5]])
                    if len(forces["symbols"]) < index:
                        forces["symbols"].append(symbol)
                elif ":" in stripped:
                    key, _, value = stripped.partition(":")
                    key, value = key.strip(), value.strip()
                    numeric = number(value)
                    if numeric is None:
                        forces[key] = value
                    elif key == "step":
                        forces[key] = int(numeric)     # an index, not a measure
                    else:
                        forces[key] = numeric
            elif section == "performance":
                performance = step["performance"]
                if stripped.endswith(":"):
                    # "stages (wall-clock seconds):" opens the stage table.
                    table = "stages_s"
                elif table is not None and indent >= len(INDENT) * 2:
                    # A stage row, told from a key by its deeper indentation --
                    # a stage name can itself contain a colon ("integration:
                    # two-body integrals (fft)"), so the ``key: value`` shape
                    # does not separate them.  The name may contain spaces, so
                    # the seconds are the last field.
                    name, _, seconds = stripped.rpartition(" ")
                    value = number(seconds)
                    name = name.strip()
                    if value is None or not name:
                        continue
                    if name == "sum of timed stages":
                        performance["total_s"] = value
                    else:
                        performance[table][name] = value
                elif ":" in stripped:
                    key, _, value = stripped.partition(":")
                    key, value = key.strip(), value.strip()
                    numeric = number(value)
                    if numeric is None:
                        performance[key] = value
                    elif key in _PERFORMANCE_COUNTS:
                        performance[key] = int(numeric)     # a count, not a time
                    else:
                        performance[key] = numeric
            elif section == "iterations":
                if stripped.startswith("operator_pool:") or stripped.startswith("["):
                    continue
                fields = stripped.split()
                if not columns:
                    # The heading: "energy (eV)" is one column, two tokens.
                    head = stripped
                    for unit in ("eV", "Ha"):
                        if f"energy ({unit})" in head:
                            energy_unit = unit
                            head = head.replace(f"energy ({unit})", "energy")
                    columns = head.split()
                    continue
                if not fields[0].isdigit():
                    continue                       # an opt-in pool listing line
                # `operator` is the last column and may contain spaces, so the
                # row is split into exactly as many fields as there are columns
                # -- an unbounded split truncated a label at its first space.
                fields = stripped.split(None, len(columns) - 1)
                record = dict(zip(columns, fields))
                entry: dict[str, Any] = {
                    "index": int(record["iter"]),
                    "selected_operator": record.get("operator", ""),
                    "operator_kind": record.get("type", ""),
                    "energy": number(record.get("energy", "")),
                    "energy_unit": energy_unit,
                    "expressivity_E": record.get("expr", "-"),
                    "max_gradient": number(record.get("|grad|", "")),
                    # Not serialized by the writer: reported as unknown rather
                    # than inferred from the growth index, which they coincide
                    # with for ADAPT but not by construction.
                    "num_parameters": record.get("npar"),
                    "cnot_count": record.get("cnot", "-"),
                    "circuit_depth": record.get("depth", "-"),
                    "one_qubit_gates": record.get("1q", "-"),
                    # Classical optimizer steps for this growth step; "-" both
                    # when the method reports none and when an older log has
                    # no such column at all.
                    "optimizer_steps": record.get("steps", "-"),
                    "columns": list(columns),
                }
                step["iterations"].append(entry)

    # The top level is the last step, so a single-point log parses exactly as it
    # did before there were steps; every step is kept under "steps".
    result: dict[str, Any] = {"system": {}, "setup": {}, "iterations": []}
    if steps:
        result.update(steps[-1])
    result["steps"] = steps
    result.update(relaxation)
    return result
