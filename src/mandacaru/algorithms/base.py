# -*- coding: utf-8 -*-
# file: algorithms/base.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Shared base for the variational state-vector drivers.

:class:`VariationalDriver` factors out everything the concrete algorithms
(:class:`~mandacaru.algorithms.vqe.VQE`,
:class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQE`, and their subspace variants)
have in common, so a new method only writes its own optimization loop:

* **problem setup** -- the ASE-calculator surface (``basis`` / ``grid`` / ``h`` /
  ``kpts`` / ``spin`` / ``active_space`` / ...) and the geometry-to-Hamiltonian
  builder;
* **the state-vector backend** -- materializing the qubit Hamiltonian as a dense
  or sparse matrix, the canonical expectation value ``energy(psi)``, and the
  Gamma-point guard;
* **run scaffolding** -- the ASE ``calculate`` hook, wall-clock / resource
  timings, and the start-up banner.

Subclasses implement two hooks: :meth:`_configure` (build the ansatz / pool for a
given Hamiltonian and materialize it) and :meth:`run` (the optimization).  The
``energy`` contract is a **state vector in** -- ``energy(psi)`` -- uniformly
across every driver.
"""

from __future__ import annotations

import os
import warnings
from collections import namedtuple
from time import perf_counter as _perf

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from ..backends.hardware import (device_arn, device_provider, is_aws_device,
                                 normalize_device, require_runnable,
                                 requires_shots)
from ..backends.providers import build_provider, normalize_provider
from ..core.mapping import Fermion, PauliSum
from ..core.matrix_free import (MATRIX_FREE, STORED_MEMORY_FRACTION,
                                PauliOperator, physical_memory_bytes,
                                refuse_unaffordable_state,
                                stored_operator_bytes)
from ..core.serialization import (DEFAULT_FORMAT, EXTENSION_FORMATS,
                                  load_hamiltonian, resolve_format,
                                  resolve_save_path, save_hamiltonian)
from ..core.spin_orbit import conserves_spin_projection
from ..backends.measurement import RunBudgetExceeded
from ..optimizers.optim import (NAMED_OPTIMIZERS,
                                OptimizeResult, resolve_optimizer)
from ..utils.dumps import (HAMILTONIAN_FILE, POOL_FILE, dump_hamiltonian,
                           dump_pool, resolve_dump_path)
from ..units import (ANGSTROM_TO_BOHR, DEFAULT_GRID_SPACING, convert_energy,
                     energy_unit_label, from_hartree)
from ._hamiltonian_from_atoms import monkhorst_pack_kpts, resolve_initial_state


def format_pauli_sum(pauli: PauliSum, indent: str = "    ",
                     max_terms: int | None = None) -> str:
    """Render a :class:`~mandacaru.core.mapping.PauliSum` as ``coeff * PauliString``.

    Real coefficients (Hermitian operators, e.g. the Hamiltonian) print as plain
    reals; purely imaginary ones (anti-Hermitian generators) print with a ``j``.
    ``max_terms`` truncates long sums with a trailing ``... (k more terms)`` line.
    """
    items = sorted(pauli.simplify().terms.items())
    if not items:
        return f"{indent}0"
    shown = items if max_terms is None else items[:max_terms]
    lines = []
    for label, coeff in shown:
        c = complex(coeff)
        if abs(c.imag) < 1e-12:
            coeff_str = f"{c.real:+.6f}"
        elif abs(c.real) < 1e-12:
            coeff_str = f"{c.imag:+.6f}j"
        else:
            coeff_str = f"({c.real:+.6f}{c.imag:+.6f}j)"
        lines.append(f"{indent}{coeff_str} * {label}")
    if max_terms is not None and len(items) > max_terms:
        lines.append(f"{indent}... ({len(items) - max_terms} more terms)")
    return "\n".join(lines)


#: Largest imaginary Pauli coefficient accepted before Hermitizing (relative
#: to the operator's coefficient scale).
HERMITICITY_TOLERANCE = 1e-9


#: Whether the start-up banner has already been written in this process.
_BANNER_SHOWN = False


#: What the ``[SYSTEM]`` block needs from a geometry.  ``pbc`` and ``magmoms``
#: are ``None`` for an input that carries neither -- a bare
#: ``(symbols, positions)`` pair -- which the block then reports as unknown
#: rather than inventing a default.
LoggedGeometry = namedtuple("LoggedGeometry",
                            "symbols positions cell pbc magmoms")


def _resolve_geometry(geometry) -> LoggedGeometry:
    """Normalize ``geometry`` to a :class:`LoggedGeometry` for logging.

    Accepts an ASE ``Atoms`` object (symbols, positions, cell, periodic
    directions and initial magnetic moments read directly), a
    ``(symbols, positions)`` pair, or ``None``.  ``cell`` is ``None`` for a
    non-periodic input.
    """
    if geometry is None:
        return LoggedGeometry(None, None, None, None, None)
    # ASE Atoms: duck-typed to avoid a hard dependency here.
    if hasattr(geometry, "get_chemical_symbols") and \
            hasattr(geometry, "get_positions"):
        cell = np.asarray(geometry.get_cell(), dtype=float)
        return LoggedGeometry(
            symbols=list(geometry.get_chemical_symbols()),
            positions=np.asarray(geometry.get_positions(), dtype=float),
            cell=cell if np.any(cell) else None,
            pbc=np.asarray(geometry.get_pbc(), dtype=bool),
            magmoms=np.asarray(geometry.get_initial_magnetic_moments(),
                               dtype=float))
    # (symbols, positions) pair.
    symbols, positions = geometry
    return LoggedGeometry(list(symbols), np.asarray(positions, dtype=float),
                          None, None, None)


#: How a fermion-to-qubit mapping is spelled in the ``[ELECTRONS]`` log block.
#: The names the code uses are identifiers; the log is read by people too, and
#: these are the transformations' own names.  An unlisted mapping is written
#: verbatim.
MAPPING_LABELS = {"jordan_wigner": "Jordan-Wigner", "parity": "parity",
                  "parity_reduced": "reduced parity",
                  "bravyi_kitaev": "Bravyi-Kitaev"}



def _checked_field(field):
    """A uniform electric field as a float tuple, or ``None``."""
    if field is None:
        return None
    try:
        values = np.asarray(field, dtype=float)
    except (TypeError, ValueError):
        values = None
    if values is None or values.shape != (3,) or not np.all(
            np.isfinite(values)):
        raise ValueError("electric_field must be three finite numbers "
                         f"(Hartree per e Bohr); got {field!r}")
    return tuple(float(x) for x in values)

class VariationalDriver(Calculator):
    """Base ASE calculator for the variational state-vector eigensolvers.

    Two class attributes declare what a driver's ``run()`` actually does with the
    reporting options, so :class:`~mandacaru.algorithms.calculator.Mandacaru` can
    refuse an option that would otherwise be accepted and silently ignored:
    :attr:`writes_output_log` and :attr:`supports_checkpoints`.

    Not used directly -- concrete drivers subclass it and implement
    :meth:`_configure` and :meth:`run`.  The constructor accepts the problem-setup
    surface shared by every driver; algorithm-specific arguments (ansatz, pool,
    stopping controls, ...) are added by the subclass and passed through
    ``**shared`` to :meth:`__init__`.

    Parameters
    ----------
    save_hamiltonian : bool or str
        Serialize the qubit Hamiltonian (as Pauli strings) to disk once it has
        been built (default ``False``).  ``True`` writes ``"hamiltonian"`` with
        the extension of ``hamiltonian_format``; a path writes there.  See
        :mod:`mandacaru.core.serialization`.
    verbose_operators : bool or str
        Write the **operator pool** -- every generator's label, kind, support
        and Pauli expansion -- to ``pool.json`` (default ``False``).  A path
        writes there instead.  The run trace only reports the pool's name and
        size, because a realistic pool is hundreds of operators long; this is
        where to read it.  See :mod:`mandacaru.utils.dumps`.
    verbose_hamiltonian : bool or str
        Write the **qubit Hamiltonian** as readable JSON (Pauli strings with
        complex coefficients, in Hartree) to ``hamiltonian.inspect.json`` (default
        ``False``); a path writes there instead.  This is for inspection --
        ``save_hamiltonian=`` writes the round-trippable form that
        ``load_hamiltonian=`` reads back.
    checkpoint : str, optional
        Path of a **wavefunction checkpoint** written during the run (see
        :mod:`mandacaru.core.checkpoint`): the reference determinant, the
        generators, the parameters, the qubit Hamiltonian and the solver's
        progress.  ADAPT-VQE writes it after every ``checkpoint_every``
        accepted operators, VQE after every ``checkpoint_every`` cost
        evaluations (best point so far); both write it once more when the run
        ends.  Written atomically, so an interruption never leaves a torn file.
    checkpoint_every : int
        Checkpoint cadence (default ``1``).
    resume : str, optional
        Path of a checkpoint to **continue from**: ADAPT-VQE rebuilds the grown
        ansatz and its history and keeps growing (``max_iterations`` counts the
        total number of operators, so raise it to go further); VQE starts its
        optimization from the stored parameters.  The file must describe the
        same register (width, mapping, tapering, reference determinant).
        With ``checkpoint`` set to the same path, every geometry of a
        relaxation warm-starts from the previous one.
    references : {"auto", True, False} or str
        Where to write the run's **BibTeX bibliography** -- the papers behind
        the method, the pool and its growth strategy, the fermion-to-qubit
        mapping, the basis family and the options that actually built it, the
        optimizer and the codes (see :mod:`mandacaru.utils.citations`).
        ``"auto"`` (default) writes ``references.bib`` beside the ``txt=``
        log and nothing when there is no log, ``True`` writes it in the working
        directory, a string names the file, ``False`` switches it off.
    load_hamiltonian : str, optional
        Path of a Hamiltonian file written by ``save_hamiltonian``.  When given,
        the driver loads the qubit Hamiltonian from disk and **skips the molecular
        integrals (one- and two-body) and the fermion-to-qubit transformation
        entirely** -- the file also carries ``num_particles`` /
        ``n_spatial_orbitals``, so the pool / ansatz are rebuilt without a
        geometry.  The format is detected from the file, so this accepts either.
    hamiltonian_format : {"parquet", "json"}
        Format ``save_hamiltonian`` writes (default ``"parquet"``: compact,
        compressed and columnar).  ``"json"`` is plain text and needs no Parquet
        engine, which sidesteps the Qiskit/pyarrow interaction documented in
        :mod:`mandacaru.core.serialization`.  Loading ignores this and
        auto-detects instead.
    backend_provider : str
        Quantum SDK used to construct (and, when executing, run) the ansatz
        circuits -- ``"qiskit"`` (default), ``"braket"`` (amazon-braket-sdk) or
        ``"cirq"``.  See :mod:`mandacaru.backends.providers`.
    execute_circuits : bool, optional
        Evaluate the ansatz by *executing* the compiled circuit on the provider's
        local state-vector simulator instead of the internal NumPy state-vector
        backend.  Defaults to ``True`` for ``"braket"`` / ``"cirq"`` (naming them
        is a request to use them) and ``False`` for ``"qiskit"``, which keeps the
        fast default numerics; the results agree to machine precision either way.
    quenching : bool
        Dynamic parametrization (default ``True``).  ``True`` re-optimizes **all**
        variational parameters at every iteration.  ``False`` optimizes only the
        most recently added parameter, freezing all previous ones at their
        already-optimized values.
    kinetic : {"fd", "spectral"}, optional
        Laplacian discretization of the real-space integrals: the 3-point
        finite-difference stencil (``"fd"``) or the FFT Laplacian
        (``"spectral"``), which never underestimates a compact function's
        kinetic energy.  ``None`` (default) means ``"fd"``, the operator the
        force code differentiates; use ``"spectral"`` for single-point
        energies on coarse grids.  A crystal (``method="dft"`` on a periodic
        geometry) is always spectral and refuses ``"fd"``, so a molecule
        compared with it in a box needs ``"spectral"``.
    atomic_units : bool
        Output-unit convention (default ``False``): every energy the driver
        *returns or prints* -- the result object, the energy levels, the verbose
        trace, ``output.txt``, :meth:`measured_energy` -- is in **eV** and every
        length in **Angstrom**.  ``True`` switches those outputs to Hartree and
        Bohr.  The internal layers (the qubit Hamiltonian, ``energy(psi)``, the
        optimizer cost) always work in Hartree, whatever this flag says.
    dry_run : bool
        Estimate the qubit requirements and **stop** (default ``False``).  In a
        dry run no integral is computed, no Hamiltonian is mapped and no circuit
        is executed: the ASE hook stores a
        :class:`~mandacaru.algorithms.dry_run.QubitEstimate` on
        :attr:`dry_run_result`, reports ``NaN`` as the energy, and :meth:`run`
        returns the estimate instead of a result.  See
        :mod:`mandacaru.algorithms.dry_run` and :meth:`estimate_qubits`.
    """

    implemented_properties = ["energy", "free_energy"]

    #: Whether ``run()`` writes the structured ``txt=`` log.
    writes_output_log = False
    #: Whether the geometry is a periodic cell whose Hamiltonian needs
    #: lattice-summed electrostatics rather than an isolated one.  Set by the
    #: periodic drivers; every molecular method leaves it False.
    periodic_hamiltonian = False

    def _grid_commensurate(self):
        """Node-count divisors the periodic grid must respect, or ``None``.

        Molecular paths have none.  The periodic drivers return their k-point
        mesh, because the supercell is that many primitive cells long and a
        primitive translation has to be a whole number of grid steps for
        Bloch's theorem to hold on the grid.
        """
        return None

    #: Whether ``run()`` honors ``checkpoint=`` / ``resume=``.
    supports_checkpoints = True
    _OPTIMIZERS = NAMED_OPTIMIZERS

    #: Default ``sparse`` policy (``False`` dense; ``"auto"`` for adaptive drivers).
    _default_sparse = False

    def __init__(self, *, optimizer=None,
                 mapping: str = "jordan_wigner",
                 basis="HAO", device: str = "AER_simulator", grid=None,
                 h: float = DEFAULT_GRID_SPACING, kpts=None, spin: bool = False,
                 initial_state: str | None = "hartree-fock", charge: int = 0,
                 n_electrons=None, active_space=None, taper: bool = False,
                 hamiltonian_builder=None,
                 run_options: dict | None = None,
                 verbose: bool = True, sparse=None,
                 save_hamiltonian: bool | str = False,
                 verbose_operators: bool | str = False,
                 verbose_hamiltonian: bool | str = False,
                 load_hamiltonian: str | None = None,
                 hamiltonian_format: str = DEFAULT_FORMAT,
                 backend_provider: str | None = None,
                 execute_circuits: bool | None = None,
                 backend_options: dict | None = None, shots: int = 0,
                 quenching: bool = True, dry_run: bool = False,
                 kinetic: str | None = None, ghosts=None,
                 electric_field=None,
                 atomic_units: bool = False,
                 checkpoint: str | None = None, checkpoint_every: int = 1,
                 resume: str | None = None, references="auto",
                 txt: str | None = None,
                 **calc_kwargs):
        if "two_qubit_reduction" in calc_kwargs:
            raise TypeError(
                "two_qubit_reduction is no longer a constructor option; "
                "use mapping='parity_reduced'")
        if "output" in calc_kwargs:
            raise TypeError(
                "output= is now txt=: "
                "the run log is written with `txt='output.txt'`, and with no "
                "`txt=` the same blocks go to standard output.")
        Calculator.__init__(self, **calc_kwargs)

        self.verbose = bool(verbose)
        if txt is not None and not self.writes_output_log:
            # Every driver has a `txt`, because `log_targets` is where *any* of
            # them reports; only the ones whose run() goes through the block
            # protocol can fill a file, and silently accepting the path would
            # leave an empty log.
            raise NotImplementedError(
                f"{type(self).__name__} does not write 'txt': its run() does "
                f"not go through the block protocol, so the file would stay "
                f"empty.  Use method='rhf', 'uhf', 'ghf', 'vqe', 'adapt-vqe', "
                f"'mcas-vqe' or 'valqa' for the structured log.")
        #: Path of the run log, or ``None`` for standard output only.
        self.txt = None if txt is None else os.fspath(txt)
        # Output-unit convention: eV / Angstrom unless atomic units are asked
        # for.  Internally everything stays in Hartree / Bohr; the conversion
        # happens once, where a result object or a printout is built.
        self.atomic_units = bool(atomic_units)
        self.energy_units = "Ha" if self.atomic_units else "eV"
        self.length_units = "bohr" if self.atomic_units else "angstrom"
        from ..core.mapping import resolve_mapping
        # The mapping name is the sole source of the register representation.
        self.mapping = resolve_mapping(mapping)
        if self.mapping == "parity_reduced":
            if not self._supports_parity_reduced:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support the two-qubit "
                    "reduction (its reference determinants are not tapered)")
        self.basis = basis
        self.device = normalize_device(device)      # raises on unknown device
        self.grid = grid
        self.h = float(h)
        self.spin = bool(spin)
        # Which spatial orbitals reach the register: the frozen core, the
        # virtual truncation and how the virtuals are ranked.  Resolved here
        # so a typo fails at construction rather than after the integrals --
        # the selector runs late, and a run that spent an hour on a grid
        # before rejecting 'natual' has told the user nothing it could not
        # have said at once.
        from .active_space import resolve_active_space_spec
        #: The resolved ``active_space`` option
        #: (:class:`~mandacaru.algorithms.active_space.ActiveSpaceSpec`), or
        #: ``None`` when every orbital reaches the register.
        self.active_space = resolve_active_space_spec(active_space)
        # Z2 symmetry tapering: one qubit removed per conserved parity, found
        # from the Hamiltonian rather than from a declared point group.  It sits
        # *after* the encoding rather than being one, which is why it is its own
        # option and not a `mapping` value: it composes with an encoding, and the
        # thing it needs from one is that the reference determinant's bits are
        # the orbital occupations, which is true for Jordan-Wigner alone.
        self.taper = bool(taper)
        if self.taper and self.mapping != "jordan_wigner":
            raise ValueError(
                f"taper=True needs mapping='jordan_wigner', not "
                f"{self.mapping!r}.  Tapering is a symmetry reduction applied to "
                f"an already-encoded Hamiltonian, and it reads the sector off "
                f"the reference determinant -- whose bits are the orbital "
                f"occupations only under Jordan-Wigner.  'parity_reduced' is "
                f"itself a taper of two known symmetries, so combining them "
                f"would remove the same qubits twice.")
        if self.taper and not self._supports_tapering:
            raise NotImplementedError(
                f"{type(self).__name__} does not support Z2 tapering")
        # A pseudopotential family is a basis name (basis="PAW-LCAO", ...); its
        # options are validated here so a typo fails at construction, not
        # after the grid.  It replaces the core + the -Z/r singularity with a
        # smooth valence-only problem and subsumes the frozen core.
        self._check_pseudo_basis(basis, self.active_space)
        # Laplacian discretization ("fd" / "spectral"; None = path default).
        if kinetic not in (None, "fd", "spectral"):
            raise ValueError(f"unknown kinetic operator {kinetic!r}; use "
                             "'fd', 'spectral' or None")
        self.kinetic = kinetic
        self.initial_state = resolve_initial_state(initial_state)
        # Monkhorst-Pack mesh (ASE); the engine is Gamma-point (molecular), so a
        # denser mesh is stored on ``kpoints`` but rejected at run time.
        self.kpts, self.kpts_gamma, self.kpoints = monkhorst_pack_kpts(kpts)
        self.charge = int(charge)
        # Atoms that keep their basis functions but carry no nucleus,
        # pseudopotential or electrons (the counterpoise correction);
        # checked against the geometry when the Hamiltonian is built.
        self.ghosts = (tuple(sorted({int(i) for i in ghosts}))
                       if ghosts is not None else ())
        # A uniform field on a molecule (Hartree per e Bohr), or None.
        self.electric_field = _checked_field(electric_field)
        self.n_electrons = n_electrons
        self.hamiltonian_builder = hamiltonian_builder
        self.run_options = dict(run_options or {})
        self.sparse = self._default_sparse if sparse is None else sparse

        # Hamiltonian disk cache: `load_hamiltonian` short-circuits the integral
        # engine and the fermion-to-qubit mapping; `save_hamiltonian` dumps the
        # qubit Hamiltonian (Pauli strings) once it has been built.
        self.load_hamiltonian = (None if load_hamiltonian is None
                                 else str(load_hamiltonian))
        if self.electric_field is not None and (
                self.load_hamiltonian is not None
                or hamiltonian_builder is not None):
            # Neither source goes through the integrals the field is put on,
            # and a cached file does not record one.
            raise ValueError(
                "electric_field acts on the integrals built from the "
                "geometry; a loaded or user-built Hamiltonian would ignore "
                "it")
        self.save_hamiltonian = save_hamiltonian
        # The format applies to *saving*; loading detects it from the file.
        self.hamiltonian_format = resolve_format(hamiltonian_format)
        self._save_path = resolve_save_path(save_hamiltonian,
                                            self.hamiltonian_format)
        # Readable dumps of the two objects the trace only summarizes.
        self.verbose_operators = verbose_operators
        self.verbose_hamiltonian = verbose_hamiltonian
        self._pool_dump_path = resolve_dump_path(verbose_operators, POOL_FILE)
        self._hamiltonian_dump_path = resolve_dump_path(verbose_hamiltonian,
                                                        HAMILTONIAN_FILE)

        # Circuit-construction / execution SDK.  Naming an Amazon Braket device
        # implies the braket provider, so `device="braket-sv1"` alone is enough.
        if backend_provider is None:
            backend_provider = device_provider(self.device) or "qiskit"
        self.backend_provider = normalize_provider(backend_provider)
        # `execute_circuits` defaults to True for the non-Qiskit providers
        # (naming them is a request to use them); the Qiskit default keeps the
        # fast NumPy state-vector numerics unless execution is asked for.
        self.execute_circuits = (self.backend_provider != "qiskit"
                                 if execute_circuits is None
                                 else bool(execute_circuits))
        self.backend_options = dict(backend_options or {})

        # Measurement shots.  Real QPUs never return a state vector, so they
        # require shots > 0 and the energy is estimated from measured
        # qubit-wise-commuting groups (see mandacaru.backends.measurement).
        self.shots = int(shots)
        # The optimizer follows the objective: ``None`` is SPSA on a
        # shot-based one, and a finite-difference method is refused there.
        self.optimizer = resolve_optimizer(optimizer, allowed=self._OPTIMIZERS,
                                           shots=self.shots)
        # A dry run only *estimates* the register, so hardware may be named
        # without shots there.
        if not dry_run:
            self._check_shots_for_hardware()
        if self.shots and self.backend_provider not in ("braket", "qiskit"):
            raise NotImplementedError(
                f"shot-based execution is implemented for the 'qiskit' and "
                f"'braket' providers, not {self.backend_provider!r}; use "
                "backend_provider='qiskit' (IBM Quantum) or 'braket' (AWS).")
        if self.shots:
            self.execute_circuits = True

        # Dynamic parametrization: True re-optimizes every parameter each
        # iteration; False freezes the previous ones (see `_optimize_grown`).
        self.quenching = bool(quenching)

        # Dry run: estimate the qubit budget and stop before any integral,
        # mapping or circuit (see `estimate_qubits` / `_dry_run_estimate`).
        self.dry_run = bool(dry_run)
        #: :class:`~mandacaru.algorithms.dry_run.QubitEstimate` of the last dry run.
        self.dry_run_result = None

        self._integration_profile = None
        self._gradient_context = None
        #: :class:`~mandacaru.core.sector.ParticleSector` the operators are
        #: restricted to, or ``None`` for the full register.
        self._sector = None
        self._configured = False
        #: Result of the most recent run (set by :meth:`run` / the ASE hook).
        self.result = None
        #: Set by :class:`~mandacaru.algorithms.calculator.Mandacaru` so the solver
        #: does *not* write the ``[PERFORMANCE]`` block itself: the caller
        #: writes it after the nuclear gradient, whose time is usually the
        #: largest stage of a relaxation step and would otherwise be left out.
        self.defer_performance = False
        # True when a Hamiltonian was supplied at construction (direct mode); the
        # ASE hook then never rebuilds it.  In calculator mode it stays False and
        # the Hamiltonian is (re)built from the geometry on each ``calculate``.
        self._built_from_hamiltonian = False
        #: ``(hamiltonian, num_particles, n_spatial_orbitals)`` held unmaterialized
        #: for a dry run given a Hamiltonian at construction.
        self._dry_run_problem = None

        # Wavefunction checkpoints (core.checkpoint): `checkpoint` is written
        # during the run -- every `checkpoint_every` accepted operators (ADAPT)
        # or cost evaluations (VQE) and always at the end; `resume` is read at
        # the start of `run()` and the optimization continues from it.
        self.checkpoint_path = None if checkpoint is None else str(checkpoint)
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.resume_path = None if resume is None else str(resume)
        #: The last :class:`~mandacaru.core.checkpoint.WavefunctionCheckpoint`
        #: written (or built) by this driver.
        self.checkpoint = None
        # BibTeX of the methods, bases, pools and codes the run used
        # (mandacaru.utils.citations).  `references_path` is a property, so a
        # subclass that sets `output` later -- ADAPT -- gets the "auto" path
        # beside its log without re-resolving anything here; resolving it once
        # now is what makes a bad value fail at construction.
        self.references = references
        self.references_path
        #: Path of the ``references.bib`` most recently written, if any.
        self.references_written = None
        # Keys a *part* of the run adds once it actually runs -- excited states,
        # an expressibility analysis.  Configuration alone cannot know them.
        self._citation_extras: set = set()

        # Every base-owned output path exists now, so this is the first point
        # the collision check can see them all (it used to run before
        # `checkpoint_path` was assigned, which let a VQE checkpoint overwrite
        # its own Hamiltonian cache).  A subclass owning another path -- ADAPT's
        # `output` -- checks again once it has set it.
        self._check_output_paths()

    def _check_shots_for_hardware(self) -> None:
        """Real hardware needs ``shots > 0`` -- the one constructor check a dry
        run waives (it may *ask about* a QPU without planning to run on it)."""
        if requires_shots(self.device) and self.shots <= 0:
            raise ValueError(
                f"device {self.device!r} is real quantum hardware, which cannot "
                "return a state vector: pass shots > 0 (e.g. shots=8192) so the "
                "energy is estimated from measurements.")

    # -- output units ----------------------------------------------------- #

    def _to_energy_units(self, energy_ha):
        """Convert an internal (Hartree) energy to the output units.

        Scalars come back as ``float``; sequences / arrays as a float array.
        The single conversion point between the internal layers and anything
        the user sees.
        """
        out = from_hartree(np.asarray(energy_ha, dtype=float), self.energy_units)
        return float(out) if out.ndim == 0 else out

    def _from_energy_units(self, energy, units: str = "eV"):
        """Convert an energy in the output units to ``units`` (``"eV"`` default)."""
        out = convert_energy(np.asarray(energy, dtype=float),
                             self.energy_units, units)
        return float(out) if out.ndim == 0 else out

    def _energy_unit_label(self) -> str:
        """``"eV"`` or ``"Ha"`` -- the label of the output energy unit."""
        return energy_unit_label(self.energy_units)

    def _length_unit_label(self) -> str:
        return "Bohr" if self.length_units.lower() in ("bohr", "au", "a0") \
            else "Angstrom"

    # -- k-points --------------------------------------------------------- #

    def _check_kpts(self) -> None:
        """Reject a non-Gamma Monkhorst-Pack mesh (the engine is Gamma-point)."""
        if len(self.kpoints) > 1:
            raise NotImplementedError(
                f"a {self.kpts[0]}x{self.kpts[1]}x{self.kpts[2]} Monkhorst-Pack "
                f"mesh ({len(self.kpoints)} k-points) is not yet supported: the "
                "real-space engine solves a Gamma-point (molecular) problem.  Use "
                "kpts=(1, 1, 1) or kpts=None.")

    def _kpts_label(self) -> str:
        n1, n2, n3 = self.kpts
        centered = ", Gamma-centered" if self.kpts_gamma else ""
        if len(self.kpoints) == 1:
            return f"Gamma ({n1}x{n2}x{n3} Monkhorst-Pack)"
        return (f"{len(self.kpoints)} k-points ({n1}x{n2}x{n3} "
                f"Monkhorst-Pack{centered})")

    # -- Hamiltonian materialization -------------------------------------- #

    @staticmethod
    def _resolve_sparse(sparse, n_qubits: int) -> bool:
        """Resolve the ``sparse`` spec to a bool: whether to avoid dense matrices.

        ``"auto"`` enables the sparse backend for ``n_qubits >= 10`` (where a dense
        pool would need tens of GB); ``True`` / ``False`` force it on / off, and
        ``"matrix-free"`` (:mod:`mandacaru.core.matrix_free`) is never dense.
        """
        if isinstance(sparse, str):
            spec = sparse.strip().lower()
            if spec == "auto":
                return int(n_qubits) >= 10
            if spec == MATRIX_FREE:
                return True
            raise ValueError(
                f"unknown sparse spec {sparse!r}; use True, False, 'auto' or "
                f"{MATRIX_FREE!r}")
        return bool(sparse)

    def _resolve_matrix_free(self, qubit_h: PauliSum, dim: int,
                             operators=()) -> bool:
        """Whether the operators are applied matrix-free rather than stored.

        ``sparse="matrix-free"`` forces it.  ``sparse="auto"`` chooses it only
        when the stored form might not fit: the CSR bound -- one entry per
        state per flip group (:func:`stored_operator_bytes`), which the real
        matrix never exceeds -- summed over the Hamiltonian and ``operators``
        (the pool) has to stay under :data:`STORED_MEMORY_FRACTION` of the
        physical memory.  Below that the stored matrices are kept, since they
        are several times faster to apply.
        """
        spec = self.sparse
        if isinstance(spec, str) and spec.strip().lower() == MATRIX_FREE:
            return True
        if not (isinstance(spec, str) and spec.strip().lower() == "auto"):
            return False
        bound = stored_operator_bytes(qubit_h, dim) + sum(
            stored_operator_bytes(getattr(op, "generator", op), dim)
            for op in operators)
        return bound > STORED_MEMORY_FRACTION * physical_memory_bytes()

    #: Whether this driver can run a Hamiltonian that breaks S_z (spin-orbit
    #: coupling).  Only ADAPT-VQE can, with the ``"spin-orbit"`` pool.
    _supports_spin_orbit = False
    #: Drivers whose reference states cannot be tapered override this.
    _supports_parity_reduced = True
    #: Whether this driver can run on a Z2-tapered register.  ``False`` for a
    #: driver whose reference states are built on the full register, the same
    #: reason ``_supports_sector`` exists.
    _supports_tapering = True

    def _refuse_spin_orbit(self):
        """Refuse a Hamiltonian that breaks S_z in a driver that cannot run one.

        Its ansatz and references conserve S_z, so it would settle in the
        ``(n_alpha, n_beta)`` sector the Hamiltonian does not keep and return
        a plausible-looking wrong energy.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot run a Hamiltonian with spin-orbit "
            f"coupling: it conserves N but not S_z.  Use method='adapt-vqe' "
            f"with pool='spin-orbit'")

    def _as_pauli_sum(self, hamiltonian, n_qubits: int,
                      num_particles=None) -> PauliSum:
        """Coerce a ``PauliSum`` / ``Fermion`` Hamiltonian to a ``PauliSum``.

        ``n_qubits`` is the register size; with the two-qubit reduction the
        fermionic operator has two more modes than that.
        """
        if isinstance(hamiltonian, PauliSum):
            return hamiltonian
        if isinstance(hamiltonian, Fermion):
            if (not self._supports_spin_orbit
                    and not conserves_spin_projection(hamiltonian)):
                self._refuse_spin_orbit()
            if self.mapping == "parity_reduced":
                particles = (num_particles if num_particles is not None
                             else getattr(self, "num_particles", None))
                return hamiltonian.map_to_qubits(
                    self.mapping, n_modes=n_qubits + 2,
                    num_particles=particles)
            return hamiltonian.map_to_qubits(self.mapping, n_modes=n_qubits)
        raise TypeError("hamiltonian must be a PauliSum or Fermion")

    def _materialize_hamiltonian(self, qubit_h: PauliSum, n_qubits: int,
                                 sector=None, operators=()) -> bool:
        """Store the (Hermitized) dense or sparse Hamiltonian matrix.

        Sets :attr:`hamiltonian`, :attr:`n_qubits`, :attr:`_sparse`,
        :attr:`_matrix_free` and :attr:`_h_matrix`; returns the resolved sparse
        flag.  ``operators`` (the pool of an adaptive driver) enter the memory
        estimate that decides between stored and matrix-free operators
        (:meth:`_resolve_matrix_free`).  A register whose state vectors alone
        cannot fit is refused here, before anything is allocated.  A sparse matrix is
        used when :meth:`_resolve_sparse` selects it (large active spaces).
        With a ``sector`` (:class:`~mandacaru.core.sector.ParticleSector`) the
        matrix is the Hamiltonian restricted to that particle-number sector --
        every state vector of the driver then has ``sector.dim`` amplitudes.
        """
        if qubit_h.num_qubits != n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {qubit_h.num_qubits} qubits but the ansatz "
                f"/ pool has {n_qubits}")
        residual = max((abs(coeff.imag) for coeff in qubit_h.terms.values()),
                       default=0.0)
        scale = max((abs(coeff) for coeff in qubit_h.terms.values()), default=1.0)
        if residual > HERMITICITY_TOLERANCE * max(scale, 1.0):
            # Symmetrizing below would quietly discard this: a Hamiltonian
            # with genuinely complex coefficients is not an observable.
            raise ValueError(
                f"the qubit Hamiltonian is not Hermitian: its largest "
                f"imaginary coefficient is {residual:.3e} against a coefficient "
                f"scale of {scale:.3e}.  Symmetrizing it would change the "
                f"operator; check the integrals or the supplied hamiltonian.")
        self.hamiltonian = qubit_h
        self.n_qubits = int(n_qubits)
        self._sector = sector
        dim = sector.dim if sector is not None else 1 << int(n_qubits)
        self._matrix_free = self._resolve_matrix_free(qubit_h, dim, operators)
        self._sparse = (True if self._matrix_free or sector is not None
                        else self._resolve_sparse(self.sparse, n_qubits))
        self.__dict__["_h_matrix_value"] = None
        self.__dict__["_h_matrix_builder"] = (
            lambda: self._build_h_matrix(qubit_h, int(n_qubits), sector, dim))
        if not self.shots:
            self._h_matrix                          # build it now, as before
        return self._sparse

    @property
    def _h_matrix(self):
        """The Hamiltonian matrix of the state-vector backend, built on first
        use.  A shot-based objective never touches it -- every energy is a
        measured circuit -- so a hardware run allocates no ``2^N`` operator
        and meets no state-memory guard unless something local asks for it
        (ADAPT's classical pool screening does)."""
        value = self.__dict__.get("_h_matrix_value")
        if value is None:
            builder = self.__dict__.get("_h_matrix_builder")
            if builder is None:
                raise AttributeError("no Hamiltonian has been materialized")
            value = self.__dict__["_h_matrix_value"] = builder()
        return value

    @_h_matrix.setter
    def _h_matrix(self, value):
        self.__dict__["_h_matrix_value"] = value

    def _build_h_matrix(self, qubit_h, n_qubits: int, sector, dim: int):
        """The stored (Hermitized) matrix, or the matrix-free operator."""
        refuse_unaffordable_state(int(n_qubits), dim, sector is not None)
        if self._matrix_free:
            # Only integer masks and coefficients are kept; the Hermitian part
            # is taken on the coefficients, as 0.5 * (H + H^dagger) would.
            return PauliOperator(qubit_h, sector=sector, hermitian=True)
        if sector is not None:
            hs = sector.restrict(qubit_h)
            return 0.5 * (hs + hs.conj().T)
        if self._sparse:
            hs = qubit_h.to_sparse_matrix()
            return 0.5 * (hs + hs.conj().T)
        h = qubit_h.to_matrix()
        return 0.5 * (h + h.conj().T)                   # Hermitize rounding noise

    def determinant_energy(self, occupied) -> float:
        """``<D|H|D>`` of the computational-basis state with ``occupied``
        qubits set, from the Hamiltonian's ``I``/``Z`` strings alone.

        Exact and ``O(terms)``: every string with an ``X`` or ``Y`` flips a
        bit and has no diagonal element.  What a shot-based run reports as
        its reference energy, with no ``2^N`` state.
        """
        occupied = {int(q) for q in occupied}
        total = 0.0
        for label, coeff in self.hamiltonian.terms.items():
            if any(ch in "XY" for ch in label):
                continue
            sign = 1.0
            for k, ch in enumerate(label):
                if ch == "Z" and k in occupied:
                    sign = -sign
            total += sign * float(np.real(coeff))
        return total

    def energy(self, psi: np.ndarray) -> float:
        r"""Expectation value ``<psi| H |psi>`` (real) for a state vector ``psi``."""
        return float(np.real(np.vdot(psi, self._h_matrix @ psi)))

    def ansatz_energy(self, ansatz, theta) -> float:
        r"""Energy of ``ansatz`` at parameters ``theta`` on the configured backend.

        This is the single place the *hardware* path diverges from the simulator
        path.  With ``shots = 0`` the state vector is prepared (internally or by
        executing a circuit) and contracted with the Hamiltonian.  With
        ``shots > 0`` -- mandatory on a real QPU, which never exposes amplitudes
        -- the provider measures ``<H>`` from qubit-wise-commuting groups instead
        (:mod:`mandacaru.backends.measurement`).
        """
        if not self.shots:
            return self.energy(ansatz.state(theta))
        # The same validated export the final measurement uses: a state the
        # circuit cannot prepare is refused, and logical angles are expanded
        # to one per generator, rather than zipped short.
        return self.circuit_provider().energy(
            *self.circuit_problem(ansatz, theta))

    def _preflight_run_budget(self, evaluations: int | None,
                              jobs_per_evaluation: int = 1,
                              shots_per_evaluation: int = 0) -> None:
        """Refuse an optimization that cannot fit its provider's run budget,
        before the first job is queued.

        ``evaluations`` is the optimizer's bound on its cost evaluations
        (:meth:`~mandacaru.optimizers.optim.Optimizer.max_evaluations`); an
        unbounded method is left to the provider's per-job check, which stops
        the run where the budget ends.
        """
        if not self.shots or evaluations is None:
            return
        provider = self.circuit_provider()
        if provider is None or not hasattr(provider, "run_usage"):
            return
        from ..backends.measurement import (RunBudgetExceeded,
                                            resolve_run_budget)
        budget = resolve_run_budget(getattr(provider, "run_budget", None),
                                    getattr(provider, "is_hardware", False))
        if budget is None:
            return
        usage = provider.run_usage()
        needed = {"jobs": usage["jobs"] + int(evaluations)
                  * max(int(jobs_per_evaluation), 1),
                  "total_shots": usage["shots"] + int(evaluations)
                  * max(int(shots_per_evaluation), 0)}
        over = [f"{key} {needed[key]:,} > {budget[key]:,}" for key in needed
                if budget.get(key) is not None and needed[key] > budget[key]]
        if over:
            raise RunBudgetExceeded(
                f"the optimizer may evaluate the energy {evaluations:,} "
                f"times on {provider!r}, over the run budget "
                f"({'; '.join(over)}).  Lower the optimizer's maxiter or the "
                f"shots, or raise run_budget= in backend_options if you "
                f"mean it.")

    # -- circuit provider ------------------------------------------------- #

    def circuit_provider(self):
        """The :class:`~mandacaru.backends.providers.CircuitProvider`, or ``None``.

        ``None`` means the driver evaluates the ansatz with the internal
        (NumPy / SciPy-sparse) state-vector backend; a provider means every state
        preparation is compiled to a circuit and executed on that SDK's
        simulator or, for an Amazon Braket device, on the AWS service.  Both
        produce the same unitary -- see :mod:`mandacaru.backends.providers`.

        The provider is configured from :attr:`device` (Braket devices carry an
        ARN; IBM and fake devices their Qiskit name), :attr:`shots` and
        :attr:`backend_options`.
        """
        if not self.execute_circuits:
            return None
        # Cached: a provider owns a backend handle (and, for IBM, an Estimator
        # and its transpilation target).  Rebuilding it per energy evaluation
        # throws those away and re-resolves the device on every call.
        options = self._provider_options()
        key = (self.backend_provider, repr(sorted(options.items(), key=str)))
        cached = getattr(self, "_provider_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        provider = build_provider(self.backend_provider, **options)
        self._provider_cache = (key, provider)
        return provider

    def ansatz_problem(self, theta=None):
        """``(n_qubits, occupied, generators, theta, hamiltonian)`` of the
        optimized ansatz -- what a provider's ``energy``/``energies`` takes."""
        if theta is None:
            theta = self.result.optimal_parameters
        return self.circuit_problem(self.ansatz, theta)

    def circuit_problem(self, ansatz, theta):
        """The provider problem of ``ansatz`` at ``theta``: the one circuit
        export, shared by every shot-based objective evaluation, the final
        measurement and a checkpoint.

        Refuses what a Pauli-rotation circuit cannot prepare -- exact HVA
        group evolution, an unserializable ansatz, the exact UCC exponential
        of a sum -- whatever built the ansatz (a named template or a prebuilt
        object), and expands logical angles to one angle per generator.  A
        count that still disagrees with the generators is an error, not a
        shorter circuit.
        """
        from ..circuits.base import is_circuit_serializable

        if getattr(ansatz, "evolution", None) == "exact":
            raise ValueError(
                "exact HVA group evolution has no generic Pauli-rotation "
                "circuit; use evolution='trotter' to export or measure it")
        if not is_circuit_serializable(ansatz):
            raise TypeError(
                f"{type(ansatz).__name__} needs SerializableAnsatz data and "
                "a provider-compatible generator sequence before its state "
                "can be exported or measured as a circuit")
        if getattr(ansatz, "preparation", "product") == "sum" \
                and ansatz.num_parameters > 1:
            raise ValueError(
                "this ansatz is the exact UCC exponential exp(sum theta_k A_k); "
                "a circuit prepares the ordered product, which is a different "
                "state.  Build the ansatz with trotter=True to export or "
                "measure it.")
        # An ansatz whose angles are not its generators' angles (the HVA:
        # one angle per group, a compiled rotation per Pauli string) expands
        # them to the generator stream itself.
        expand = getattr(ansatz, "circuit_parameters", None)
        angles = np.asarray(expand(theta) if expand is not None else theta,
                            dtype=float).ravel()
        generators = ansatz.pauli_generators
        if len(angles) != len(generators):
            raise ValueError(
                f"{type(ansatz).__name__} gives {len(angles)} circuit angles "
                f"for {len(generators)} generators; the circuit would not be "
                f"the state the ansatz declares")
        return (ansatz.n_qubits, ansatz.reference_qubits(), generators,
                angles, getattr(self, "hamiltonian", None))

    def measured_energy(self, provider, theta=None) -> float:
        """``<H>`` of the optimized ansatz evaluated on ``provider``.

        Measures an already optimized state on another provider -- e.g. a
        local optimization measured once on
        ``QiskitProvider(device="ibm_kingston", shots=4096)``; a driver whose
        own device is a processor optimizes there directly.  Returned in the
        driver's output units (eV; Hartree with ``atomic_units=True``) -- the
        provider itself measures the Hartree qubit Hamiltonian.
        """
        return self._to_energy_units(provider.energy(*self.ansatz_problem(theta)))

    def ansatz_provider(self):
        """The provider an *ansatz* should prepare its state vector with.

        With ``shots = 0`` this is :meth:`circuit_provider` (the SDK's exact
        simulator, or ``None`` for the internal backend).  With ``shots > 0``
        it is always ``None``: a shot-based provider cannot return amplitudes,
        so the state vector that ADAPT's classical pool-gradient screening
        needs comes from the internal backend, while every *energy* goes
        through :meth:`ansatz_energy` -- i.e. through the provider's
        measurement protocol.
        """
        if self.shots:
            return None
        return self.circuit_provider()

    def _provider_options(self) -> dict:
        """Constructor options for the configured circuit provider."""
        options = dict(self.backend_options)
        if self.backend_provider == "qiskit":
            # Only a shot-based or IBM/fake target needs a configured provider;
            # the exact local path keeps the shared default instance.
            if self.shots or device_provider(self.device) == "qiskit":
                options.setdefault("shots", self.shots)
                options.setdefault("device", self.device)
            return options
        if self.backend_provider != "braket":
            return options
        options.setdefault("shots", self.shots)
        if "device" not in options:
            arn = device_arn(self.device)
            if arn is not None:
                options["device"] = arn         # run through the AWS service
            elif is_aws_device(self.device) or self.device == "braket-local":
                options["device"] = "braket_sv"
        return options

    # -- Hamiltonian disk cache ------------------------------------------- #

    def _load_hamiltonian_record(self):
        """Read the cached qubit Hamiltonian named by ``load_hamiltonian``.

        Returns ``(PauliSum, num_particles, n_spatial_orbitals)``.  This is the
        whole point of the cache: neither the one-/two-body integrals nor the
        fermion-to-qubit mapping is touched, and the driver's ``mapping`` is
        adopted from the file so the ansatz / pool stay consistent with it.

        A record with ``mapping="parity_reduced"`` restores the reduced
        register the same way: the stored operator is ``2M - 2`` qubits wide,
        so the pool and reference state must use that mapping too.
        """
        record = load_hamiltonian(self.load_hamiltonian)
        self._adopt_cache_header(record)
        return (record.hamiltonian, record.num_particles,
                record.n_spatial_orbitals)

    def _check_cache_header(self) -> None:
        """A dry run's view of ``load_hamiltonian``: the header only.

        The file must exist, be a Mandacaru cache this build understands, and
        agree with the tapering the driver was asked for -- everything
        :meth:`_load_hamiltonian_record` checks, without decoding the Pauli
        table (:func:`~mandacaru.core.serialization.read_hamiltonian_header`).
        """
        from ..core.serialization import read_hamiltonian_header

        self._adopt_cache_header(read_hamiltonian_header(self.load_hamiltonian))

    def _adopt_cache_header(self, record) -> None:
        """Take the mapping from a cache record or header."""
        from ..core.mapping import resolve_mapping

        requested_reduced = self.mapping == "parity_reduced"
        record_mapping = resolve_mapping(record.mapping)
        record_reduced = record_mapping == "parity_reduced"
        if requested_reduced and not record_reduced:
            raise ValueError(
                f"{self.load_hamiltonian!r} holds an untapered "
                f"{record.num_qubits}-qubit Hamiltonian, but this driver was "
                "built with mapping='parity_reduced'; use mapping='parity' "
                "(the file decides) or point at a parity_reduced file")
        self.mapping = record_mapping
        if self.mapping == "parity_reduced" \
                and not self._supports_parity_reduced:
            raise NotImplementedError(
                f"{self.load_hamiltonian!r} uses mapping='parity_reduced', "
                f"which {type(self).__name__} does not support (its reference "
                "determinants are not tapered)")

    # -- where the run reports itself ------------------------------------- #

    @property
    def log_targets(self) -> tuple[str, ...]:
        """Every destination this run writes its blocks to.

        There is **one** report -- ``[SYSTEM]``, ``[BASIS]``, ``[ELECTRONS]``,
        ``[OPTIMIZATION SETUP]``, ``[ITERATIONS]``, the summary and what the
        calculator appends after it -- and this says where it goes:

        * ``txt=<path>`` adds that file;
        * a **verbose** run adds standard output
          (:data:`~mandacaru.utils.logging.STDOUT`).

        So a run with no ``txt=`` prints the report, a run with one writes it,
        and asking for both (``txt=<path>`` with ``trace=True``) gets the same
        blocks in both places.  An empty tuple means the run reports nothing.
        """
        from ..utils.logging import STDOUT

        targets = () if self.txt is None else (self.txt,)
        return targets + ((STDOUT,) if self.verbose else ())

    # -- bibliography ----------------------------------------------------- #

    @property
    def references_path(self) -> str | None:
        """Where this run writes its ``references.bib``, or ``None``.

        Resolved on demand rather than stored, so it follows ``txt`` even when
        a subclass sets that after the base constructor has run.
        """
        from ..utils.citations import resolve_references_path
        return resolve_references_path(self.references, self.txt)

    #: Method name this driver cites (``None`` = cite no specific algorithm).
    citation_method: str | None = None

    def _citation_config(self) -> dict:
        """The run configuration :func:`~mandacaru.utils.citations.citation_keys`
        reads.

        It reports what *ran*: the pool object's own name rather than the
        string that was typed, and the basis options the builder resolved
        (family defaults included) rather than the ones the user wrote.  In
        direct mode -- a Hamiltonian or a cache handed in, no geometry -- there
        is no basis to cite and none is claimed.
        """
        from ._hamiltonian_from_atoms import PER_ELEMENT, resolve_basis

        context = getattr(self, "_gradient_context", None) or {}
        built = bool(context)
        family = context.get("family")
        options = dict(context.get("options") or {})
        name = None
        if self.load_hamiltonian is None and not self._built_from_hamiltonian:
            name, spec = resolve_basis(self.basis)
            if name == PER_ELEMENT:
                # One entry per element; the families are cited through the
                # builder's context, the all-electron names through the specs.
                name = [resolve_basis(value)[0] for value in spec.values()]
            elif not options:
                options = dict(spec or {})
        pool = getattr(getattr(self, "pool", None), "name", None) \
            or getattr(self, "_pool_spec", None)
        # How the datasets were *generated* -- scalar-relativistic, GGA, with
        # a core correction -- is recorded on the datasets themselves, not in
        # the basis options, so a run that merely loads the shipped library
        # still cites what produced it.
        integrals = context.get("integrals")
        datasets = (getattr(integrals, "pseudopotentials", None)
                    or getattr(integrals, "datasets", None) or ())
        return {"method": self.citation_method,
                "pool": pool if isinstance(pool, str) else None,
                "mapping": self.mapping,
                "basis": name,
                "family": family,
                "basis_options": options,
                "optimizer": getattr(self.optimizer, "method", None),
                "backend_provider": self.backend_provider,
                "shots": self.shots,
                "execute_circuits": self.execute_circuits,
                "profile": bool(getattr(self, "profile", False)),
                "tetris": bool(getattr(self, "tetris", False)),
                "prune": bool(getattr(self, "prune", False)),
                "has_geometry": self.atoms is not None,
                "built_basis": built,
                "datasets": tuple(datasets),
                # Only cited when it was asked for: the energy ordering is the
                # canonical order the mean field already produced and borrows
                # nothing, so an untruncated run gains no reference here.
                "active_method": (
                    self.active_space.method
                    if self.active_space is not None
                    and self.active_space.truncates else None),
                "extras": tuple(sorted(self._citation_extras))}

    def citation_keys(self) -> list:
        """Bibliography keys for this run's configuration.

        Public because a script that assembles its own bibliography wants the
        keys, not the file.
        """
        from ..utils.citations import citation_keys

        config = self._citation_config()
        basis = config.pop("basis")
        if isinstance(basis, list):
            # A per-element basis cites every element's family.
            keys: list = []
            for one in basis:
                keys += citation_keys(basis=one, **config)
            config["extras"] = keys
            return citation_keys(basis=None, **config)
        return citation_keys(basis=basis, **config)

    def _cite(self, *keys) -> None:
        """Record that part of the run used ``keys``, and refresh the file.

        Called by the pieces whose citations the configuration cannot predict
        -- a deflation sweep, an expressibility analysis -- so the bibliography
        already on disk gains them instead of going stale.
        """
        new = set(keys) - self._citation_extras
        if not new:
            return
        self._citation_extras |= new
        if self.references_written:
            self._maybe_write_references()

    def write_references(self, path=None) -> str | None:
        """Write the run's ``references.bib``; returns the path, or ``None``.

        ``None`` when ``references=`` asks for no file and ``path`` names none.
        Called automatically once the problem is set up, so a run that writes a
        log writes its bibliography beside it.
        """
        from ..utils.citations import write_references

        target = path or self.references_path
        if target is None:
            return None
        header = ("References for a Mandacaru run: the methods, bases, pools "
                  "and codes\nit used, selected from "
                  "mandacaru.utils.bibliography by the run's own\n"
                  "configuration.  Entries marked [unverified record] were not "
                  "read from a\nlocal source -- check them before citing.")
        self.references_written = write_references(
            target, self.citation_keys(), header=header)
        return self.references_written

    def _maybe_write_references(self) -> None:
        """Write the bibliography once per configured problem (never raising).

        A missing citation must not end a calculation, so a failure here is a
        warning: the energy is the point of the run, the BibTeX is a courtesy.
        """
        if self.dry_run or self.references_path is None:
            return
        try:
            self.write_references()
        except Exception as error:                  # noqa: BLE001
            warnings.warn(f"could not write {self.references_path!r}: {error}",
                          RuntimeWarning, stacklevel=2)

    def _check_output_paths(self) -> None:
        """Refuse two outputs that resolve to the same file.

        Every one of these writes a *different* document -- a cache meant to be
        loaded back, a human-readable dump, a checkpoint, an appended log -- so
        two of them sharing a path means one silently destroys the other.  The
        collision that motivated this wrote an inspection dump over a
        Hamiltonian cache, after which loading the cache failed.  Checked here,
        before any work, rather than discovered afterwards.
        """
        named = {"save_hamiltonian": self._save_path,
                 "verbose_operators": self._pool_dump_path,
                 "verbose_hamiltonian": self._hamiltonian_dump_path,
                 "checkpoint": getattr(self, "checkpoint_path", None),
                 "txt": self.txt,
                 "references": self.references_path}
        seen: dict[str, str] = {}
        for option, path in named.items():
            if path is None:
                continue
            # realpath, not abspath: two spellings through a symlinked
            # directory are still one file.
            key = os.path.realpath(os.fspath(path))
            if key in seen:
                raise ValueError(
                    f"{option}= and {seen[key]}= both resolve to {key!r}, but "
                    f"they write different documents; give them separate paths.")
            seen[key] = option

    def _maybe_save_hamiltonian(self, num_particles=None,
                                n_spatial_orbitals=None) -> str | None:
        """Dump the materialized qubit Hamiltonian when ``save_hamiltonian`` is set.

        A no-op when saving is off, or when the Hamiltonian was just read from the
        very file it would be written to.  Returns the path written, if any.
        """
        if self._save_path is None:
            return None
        from ..core.serialization import file_qubits_allowed
        if not file_qubits_allowed(self.n_qubits,
                                   f"the Hamiltonian file {self._save_path!r}"):
            return None
        if (self.load_hamiltonian is not None
                and os.path.abspath(self.load_hamiltonian)
                == os.path.abspath(self._save_path)):
            return None
        # An explicit extension on the save path is the user's choice of format
        # and outranks the driver default, which would otherwise write Parquet
        # bytes into a file they named '.json'.
        extension = os.path.splitext(self._save_path)[1].lower()
        fmt = (None if extension in EXTENSION_FORMATS
               else self.hamiltonian_format)
        return save_hamiltonian(
            self._save_path, self.hamiltonian, mapping=self.mapping,
            num_particles=num_particles, n_spatial_orbitals=n_spatial_orbitals,
            format=fmt,
            metadata={"driver": type(self).__name__,
                      "basis": self.basis if isinstance(self.basis, str)
                      else dict(self.basis),
                      "active_space": (None if self.active_space is None
                                       else self.active_space.as_dict()),
                      "taper": self.taper,
                      "n_qubits": int(self.n_qubits)})

    def _maybe_dump_hamiltonian(self, num_particles=None,
                                n_spatial_orbitals=None) -> str | None:
        """Write ``hamiltonian.inspect.json`` when ``verbose_hamiltonian`` is set."""
        if self._hamiltonian_dump_path is None:
            return None
        return dump_hamiltonian(
            self._hamiltonian_dump_path, self.hamiltonian,
            n_qubits=self.n_qubits, mapping=self.mapping,
            num_particles=num_particles,
            n_spatial_orbitals=n_spatial_orbitals)

    def _maybe_dump_pool(self, pool, operators) -> str | None:
        """Write ``pool.json`` when ``verbose_operators`` is set."""
        if self._pool_dump_path is None:
            return None
        return dump_pool(
            self._pool_dump_path, pool, operators, n_qubits=self.n_qubits,
            mapping=self.mapping, num_particles=self.num_particles)

    # -- wavefunction checkpoints ---------------------------------------- #

    def _checkpoint_record(self, ansatz, parameters, energy_ha, status,
                           labels=None, kinds=None):
        """Describe the current state as a
        :class:`~mandacaru.core.checkpoint.WavefunctionCheckpoint`.

        Everything a consumer needs is on the ansatz -- the register, the
        reference determinant in register terms, the generators as Pauli sums
        -- so the record is the same for every ansatz class.
        """
        from ..core.checkpoint import WavefunctionCheckpoint

        generators = list(ansatz.pauli_generators)
        # exp(sum) and prod(exp) are different states; the record says which.
        preparation = getattr(ansatz, "preparation", "product")
        operators = getattr(ansatz, "operators", None)
        if labels is None:
            labels = ([op.label for op in operators] if operators is not None
                      else [f"T{k}" for k in range(len(generators))])
        if kinds is None:
            kinds = ([op.kind for op in operators] if operators is not None
                     else ["uccsd"] * len(generators))
        occupied = getattr(ansatz, "occupied", None)
        if occupied is None:
            occupied = getattr(ansatz, "_occupied", None)
        atoms = getattr(self, "atoms", None)
        metadata = {
            "driver": type(self).__name__,
            "mandacaru_version": __import__("mandacaru").__version__,
            "basis": (self.basis if isinstance(self.basis, str)
                      else {k: v for k, v in dict(self.basis).items()}),
            "optimizer": self.optimizer.method,
        }
        if atoms is not None:
            metadata["geometry"] = {
                "symbols": list(atoms.get_chemical_symbols()),
                "positions_angstrom": np.asarray(atoms.positions,
                                                 dtype=float).tolist(),
                "cell_angstrom": np.asarray(atoms.cell, dtype=float).tolist(),
            }
        particles = getattr(self, "num_particles", None)
        return WavefunctionCheckpoint(
            n_qubits=ansatz.n_qubits,
            reference_qubits=list(ansatz.reference_qubits()),
            generators=generators,
            parameters=np.asarray(parameters, dtype=float),
            labels=list(labels), kinds=list(kinds),
            mapping=str(self.mapping),
            num_particles=None if particles is None else tuple(particles),
            n_spatial_orbitals=getattr(self, "n_spatial_orbitals", None),
            occupied_orbitals=None if occupied is None else list(occupied),
            energy=None if energy_ha is None else float(energy_ha),
            hamiltonian=getattr(self, "hamiltonian", None),
            method=self._method_name(),
            status=dict(status), metadata=metadata, preparation=preparation)

    def _check_checkpointable(self, ansatz) -> None:
        """Refuse ``checkpoint=`` / ``resume=`` with an ansatz that cannot be
        described -- before the run, not after the optimization.

        The :class:`~mandacaru.circuits.base.Ansatz` protocol is enough to *run*;
        a checkpoint needs :class:`~mandacaru.circuits.base.SerializableAnsatz`.
        """
        from ..circuits.base import is_serializable

        asked = [name for name, path in (("checkpoint", self.checkpoint_path),
                                         ("resume", self.resume_path))
                 if path is not None]
        if asked and not is_serializable(ansatz):
            raise TypeError(
                f"{' / '.join(asked)}= needs an ansatz that exposes "
                "`pauli_generators` and `reference_qubits()` "
                f"(SerializableAnsatz); {type(ansatz).__name__} implements only "
                "the state-vector Ansatz protocol, so its state cannot be "
                "written to or read from a checkpoint")

    def _record_checkpoint(self, ansatz, parameters, energy_ha, status,
                           **labels):
        """Build and write the checkpoint of a serializable ansatz; an ansatz
        that only evaluates states (no generators) is left alone."""
        from ..circuits.base import is_serializable

        if not is_serializable(ansatz):
            return None
        return self._write_checkpoint(self._checkpoint_record(
            ansatz, parameters, energy_ha, status, **labels))

    def _write_checkpoint(self, record) -> str | None:
        """Save ``record`` to :attr:`checkpoint_path` (a no-op without one)."""
        self.checkpoint = record
        if self.checkpoint_path is None:
            return None
        return record.save(self.checkpoint_path)

    def _load_resume(self, ansatz):
        """The checkpoint named by :attr:`resume_path`, checked against ``ansatz``.

        A checkpoint is only resumable into the register it was written for:
        the width, the mapping, the tapering and the reference determinant must
        all agree, otherwise the generators would act on the wrong qubits.
        """
        from ..core.checkpoint import WavefunctionCheckpoint
        from ..core.mapping import _canonical_method

        if self.resume_path is None:
            return None
        record = WavefunctionCheckpoint.load(self.resume_path)
        problems = []
        if record.n_qubits != ansatz.n_qubits:
            problems.append(f"{record.n_qubits} qubits in the file, "
                            f"{ansatz.n_qubits} in this run")
        if _canonical_method(record.mapping) != _canonical_method(self.mapping):
            problems.append(f"mapping {record.mapping!r} in the file, "
                            f"{self.mapping!r} in this run")
        if sorted(record.reference_qubits) != sorted(ansatz.reference_qubits()):
            problems.append(f"reference determinant {record.reference_qubits} "
                            f"in the file, {ansatz.reference_qubits()} here")
        mine = getattr(ansatz, "preparation", "product")
        if not record.is_product and mine != "sum":
            problems.append(
                "the file holds the exact-UCC state exp(sum theta_k A_k)|ref>, "
                "but this run prepares an ordered product of exponentials -- "
                "the same angles would be a different state")
        elif record.preparation == "product" and mine == "sum" \
                and record.num_parameters > 1:
            problems.append(
                "the file holds a product of exponentials, but this run's "
                "UCCSD is the exact exponential of the sum (trotter=False)")
        if problems:
            raise ValueError(
                f"cannot resume from {self.resume_path!r}: "
                + "; ".join(problems))
        return record

    # -- optimization policy (quenching) ---------------------------------- #

    def _optimize_grown(self, cost, previous_parameters,
                        n_new: int = 1) -> OptimizeResult:
        """Optimize after appending ``n_new`` parameters, honoring :attr:`quenching`.

        ``quenching=True`` (default) hands **all** parameters to the classical
        optimizer, warm-started from the previous optimum with the new angles at
        zero -- standard ADAPT-VQE.  ``quenching=False`` freezes the previously
        optimized parameters and varies only the newly added ones, a cheaper
        search per growth step that trades variational freedom for cost-function
        evaluations.

        ``n_new`` is 1 for every pool but :class:`~mandacaru.circuits.pools.
        CEOPool`, whose growth step may append an MVP-CEO -- two or three
        coupled excitations, each with its own parameter.

        ``cost`` takes the **full** parameter vector in both cases; the returned
        :class:`~mandacaru.optimizers.optim.OptimizeResult` also carries the full
        vector, so callers need no branching.
        """
        cost, best = self._budget_tracked(cost)
        try:
            return self._optimize_grown_inner(cost, previous_parameters, n_new)
        except RunBudgetExceeded as exc:
            return self._budget_stop(exc, best)

    def _optimize_grown_inner(self, cost, previous_parameters, n_new):
        previous = np.asarray(previous_parameters, dtype=float).ravel()
        n_new = int(n_new)
        x0 = np.concatenate([previous, np.zeros(n_new)])
        if self.quenching:
            return self.optimizer.minimize(cost, x0)

        def last_only(tail):
            return cost(np.concatenate(
                [previous, np.asarray(tail, dtype=float).ravel()]))

        result = self.optimizer.minimize(last_only, np.zeros(n_new))
        full = np.concatenate([previous,
                               np.asarray(result.x, dtype=float).ravel()])
        return OptimizeResult(x=full, fun=result.fun, nfev=result.nfev,
                              history=result.history, success=result.success,
                              message=result.message, nit=result.nit)

    def _optimize_all(self, cost, x0, callback=None) -> OptimizeResult:
        """Optimize a fixed-size parameter vector, honoring :attr:`quenching`.

        ``quenching=True`` (default) is a single joint minimization over every
        parameter.  ``quenching=False`` sweeps the parameters one at a time in
        order -- parameter ``k`` is optimized alone with ``0..k-1`` frozen at their
        already-optimized values and ``k+1..`` held at their starting values --
        the fixed-ansatz analog of freezing previous growth steps.
        """
        cost, best = self._budget_tracked(cost)
        try:
            return self._optimize_all_inner(cost, x0, callback)
        except RunBudgetExceeded as exc:
            return self._budget_stop(exc, best)

    def _budget_tracked(self, cost):
        """``cost`` wrapped to remember its best evaluation, so a run the
        budget stops keeps the point it had reached."""
        best = {"x": None, "fun": np.inf, "history": []}

        def tracked(theta):
            value = float(cost(theta))
            best["history"].append(value)
            if value < best["fun"]:
                best["x"] = np.array(theta, dtype=float).ravel().copy()
                best["fun"] = value
            return value

        return tracked, best

    def _budget_stop(self, exc, best) -> OptimizeResult:
        """The best point before :class:`RunBudgetExceeded`, as an
        unconverged result -- the optimization is kept, as a failed
        measurement keeps it, and :attr:`run_budget_exhausted` says why it
        stopped.  With no evaluation at all there is nothing to keep."""
        if best["x"] is None:
            raise exc
        self.run_budget_exhausted = str(exc)
        warnings.warn(f"the run budget stopped the optimization at its best "
                      f"point so far: {exc}", RuntimeWarning, stacklevel=3)
        return OptimizeResult(x=best["x"], fun=best["fun"],
                              nfev=len(best["history"]),
                              history=list(best["history"]), success=False,
                              message=f"run budget exhausted: {exc}")

    def _optimize_all_inner(self, cost, x0, callback=None) -> OptimizeResult:
        x0 = np.asarray(x0, dtype=float).ravel()
        if self.quenching or x0.size <= 1:
            return self.optimizer.minimize(cost, x0, callback=callback)

        params = x0.copy()
        history: list[float] = []
        nfev = 0
        # One sweep is one pass over the parameters, but the effort is the sum
        # of the per-parameter searches -- that is what the step count reports.
        steps = 0
        value = float(cost(params))
        success = True
        for k in range(params.size):
            def single(t, _k=k, _p=params):
                trial = _p.copy()
                trial[_k] = float(np.asarray(t, dtype=float).ravel()[0])
                return cost(trial)

            def report(t, value, _n, _k=k, _p=params):
                if callback is not None:
                    trial = _p.copy()
                    trial[_k] = float(np.asarray(t, dtype=float).ravel()[0])
                    callback(trial, value, nfev + _n)

            step = self.optimizer.minimize(single, np.atleast_1d(params[k]),
                                           callback=report)
            params[k] = float(np.asarray(step.x, dtype=float).ravel()[0])
            history.extend(step.history)
            nfev += step.nfev
            steps += step.nit or 0
            value = float(step.fun)
            success = success and step.success
        return OptimizeResult(x=params, fun=value, nfev=nfev, history=history,
                              success=success, nit=steps,
                              message="sequential (quenching=False) sweep")

    def _taper_label(self) -> str:
        """One line describing the Z2 reduction, for the run log."""
        if not getattr(self, "taper", False):
            return "none"
        info = getattr(self, "_taper_info", None)
        if info is None:
            return "requested (no symmetry found, or not configured yet)"
        return info.summary()

    def _active_space_label(self) -> str:
        """One line describing the orbital truncation, for the run log.

        Reports the partition that was actually built when there is one --
        ``resolve_active_space`` can freeze more than was asked for -- and falls
        back to the request before a Hamiltonian exists.  ``"none"`` when the
        whole orbital set reaches the register, so the line is never blank.
        """
        space = (getattr(self, "_gradient_context", None) or {}).get(
            "active_space")
        if space is not None and (space.truncated or space.frozen):
            line = space.summary()
            if space.correlation_energy is not None:
                line += (f", MP2 E_corr {space.correlation_energy:+.6f} Ha "
                         f"in the full virtual space")
            return line
        spec = self.active_space
        if spec is None or not spec.truncates:
            return "none (every orbital on the register)"
        asked = []
        if spec.orbitals is not None:
            asked.append(str(spec.orbitals))
        if spec.threshold is not None:
            asked.append(f"occupation >= {spec.threshold:g}")
        return f"{' and '.join(asked)} by {spec.method}"

    @staticmethod
    def _check_pseudo_basis(basis, active_space):
        """Validate a pseudopotential basis spec up front (names, options,
        option *values*, no frozen core); all-electron specs pass untouched."""
        from ..basis.filtering import validate_filter
        from ..pseudopotentials.confinement import validate_energy_shift
        from ._hamiltonian_from_atoms import (PER_ELEMENT,
                                              pseudopotential_family,
                                              resolve_basis)
        name, options = resolve_basis(basis)
        if name == PER_ELEMENT:
            # Which element gets which family is only checkable against a
            # geometry (at build time), but an option *value* is not: a
            # mistyped filter is a mistyped filter whatever the atoms are, and
            # finding out after the integrals is finding out too late.
            for spec in options.values():
                if isinstance(spec, dict) and "filter" in spec:
                    validate_filter(spec["filter"])
                if isinstance(spec, dict) and "energy_shift" in spec:
                    validate_energy_shift(spec["energy_shift"])
                if isinstance(spec, dict) and "zeta_split" in spec:
                    from ..basis.multizeta import validate_zeta_split
                    validate_zeta_split(spec["zeta_split"])
            return
        family = pseudopotential_family(name)
        if family is None:
            return
        unknown = sorted(set(options) - set(family.options))
        if unknown:
            raise ValueError(
                f"unknown option(s) {unknown} for the {family.label} basis; "
                f"it accepts {list(family.options)}")
        if "filter" in options:
            validate_filter(options["filter"])
        if "energy_shift" in options:
            validate_energy_shift(options["energy_shift"])
        from ..basis.multizeta import resolve_split_scheme
        resolve_split_scheme(options.get("split_norm"),
                             options.get("tail_norm"),
                             options.get("zeta_split"))
        if "polarization" in family.options:
            from ..pseudopotentials.confinement import (
                validate_confinement, validate_polarization)
            validate_confinement(options.get("confinement"))
            # Only an *explicit* request can conflict: left unwritten, the
            # shell follows the confinement.  The family default counts as a
            # confinement, so `{"polarization": "gaussian"}` alone is fine.
            resolved = family.resolved_options(options)
            if (validate_polarization(options.get("polarization"))
                    == "gaussian"
                    and validate_energy_shift(
                        resolved.get("energy_shift")) is None):
                raise ValueError(
                    "polarization='gaussian' needs an energy_shift: the "
                    "Gaussian shell takes its cutoff from the confined "
                    "orbital, and an unconfined orbital has none")
        if active_space is not None and active_space.frozen is not None:
            raise ValueError(
                f"an active_space 'frozen' core is redundant with the "
                f"{family.label} basis -- the core is already absent from the "
                f"valence-only pseudo basis")

    # -- dry run ---------------------------------------------------------- #

    def _method_name(self) -> str:
        """The ``method=`` name of this driver, for reports."""
        from .calculator import available_methods, resolve_method
        for name in available_methods():
            try:
                if resolve_method(name)[1] is type(self):
                    return name
            except Exception:                        # pragma: no cover
                continue
        return type(self).__name__

    def estimate_qubits(self, atoms=None):
        """Qubit requirements of this driver's problem, **without running it**.

        Uses the driver's own settings (``basis`` / ``charge`` / ``spin`` /
        ``active_space`` / ``mapping`` / ``device`` /
        ``load_hamiltonian``).  ``atoms`` is needed in calculator mode; in
        direct mode (a Hamiltonian given or loaded at construction) it is
        ignored.  Nothing is integrated, mapped or executed.

        Returns a :class:`~mandacaru.algorithms.dry_run.QubitEstimate`.
        """
        from .dry_run import estimate_qubits

        common = dict(mapping=self.mapping, method=self._method_name(),
                      device=self.device)
        if self.load_hamiltonian is not None:
            # The cache header is the whole specification (and fixes the
            # mapping), whether or not the driver has already adopted it.
            return estimate_qubits(load_hamiltonian=self.load_hamiltonian,
                                   **common)
        operator = particles = n_orb = None
        if self._dry_run_problem is not None:
            # A dry run given an operator: measure it without materializing it.
            operator, particles, n_orb = self._dry_run_problem
        elif (self._built_from_hamiltonian
              and getattr(self, "hamiltonian", None) is not None):
            operator = self.hamiltonian
        if operator is not None:
            # Direct mode: the occupation lives on the driver (ADAPT) or on
            # its fixed ansatz (VQE).
            ansatz = getattr(self, "ansatz", None)
            if particles is None:
                particles = (getattr(self, "num_particles", None)
                             or getattr(ansatz, "num_particles", None))
            if n_orb is None:
                n_orb = (getattr(self, "n_spatial_orbitals", None)
                         or getattr(ansatz, "n_spatial_orbitals", None))
            return estimate_qubits(
                hamiltonian=operator, num_particles=particles,
                n_spatial_orbitals=n_orb,
                basis=self.basis if isinstance(self.basis, str)
                else str(self.basis), **common)
        if atoms is None:
            atoms = getattr(self, "atoms", None)
        if atoms is None:
            raise ValueError(
                "estimate_qubits needs a geometry: pass `atoms`, attach the "
                "calculator to an Atoms object, or construct the driver with a "
                "Hamiltonian / load_hamiltonian")
        return estimate_qubits(
            atoms, basis=self.basis, charge=self.charge,
            n_electrons=self.n_electrons, spin=self.spin,
            active_space=self.active_space, taper=self.taper,
            ghosts=self.ghosts, **common)

    def _dry_run_estimate(self, atoms=None):
        """Perform the dry run: store, optionally print, and return the estimate."""
        estimate = self.estimate_qubits(atoms)
        self.dry_run_result = estimate
        self.result = None
        if self.verbose:
            self._show_banner()
            print(estimate.summary())
        return estimate

    # -- geometry -> Hamiltonian (calculator mode) ------------------------ #

    def _system_extras(self) -> dict | None:
        """The ``[SYSTEM]`` lines beyond the geometry: ghosts and a field."""
        extra = {}
        if self.ghosts:
            extra["ghosts"] = ("atoms " + " ".join(str(i) for i in self.ghosts)
                               + " (basis functions only: no nucleus, no "
                               "electrons)")
        if self.electric_field is not None:
            extra["electric_field"] = (
                " ".join(f"{x:+.6g}" for x in self.electric_field)
                + " Hartree/(e Bohr)")
        return extra or None

    def _build_hamiltonian(self, atoms):
        """Build ``(hamiltonian, num_particles, n_spatial_orbitals)`` from ``atoms``.

        Reads the cached Pauli-string Hamiltonian when ``load_hamiltonian`` is set
        (skipping the integrals *and* the mapping entirely); otherwise uses an
        explicit ``hamiltonian_builder`` if given, or the built-in ``basis`` engine
        (stashing its integration profile for the run summary).
        """
        if self.load_hamiltonian is not None:
            return self._load_hamiltonian_record()
        if self.hamiltonian_builder is not None:
            hamiltonian, num_particles, n_orbitals = \
                self.hamiltonian_builder(atoms)
            return hamiltonian, num_particles, n_orbitals
        from ._hamiltonian_from_atoms import build_basis_hamiltonian
        (hamiltonian, num_particles, n_orbitals, profile,
         context) = build_basis_hamiltonian(
            atoms, self.basis, self.grid, self.h, self.charge, self.n_electrons,
            spin=self.spin, kinetic=self.kinetic,
            periodic=self.periodic_hamiltonian,
            commensurate=self._grid_commensurate(),
            active_space=self._active_space_request(), ghosts=self.ghosts,
            electric_field=self.electric_field)
        self._integration_profile = profile
        # Kept for the nuclear gradient: the integral engine that produced this
        # Hamiltonian, and which atom each basis function belongs to.
        self._gradient_context = context
        self._basis_symbols = list(atoms.get_chemical_symbols())
        return hamiltonian, num_particles, n_orbitals

    def _basis_report(self):
        """``(fields, tables)`` of the log's ``[BASIS]`` block, or ``None``.

        ``None`` in direct mode and for a loaded Hamiltonian: there the basis
        was never built here, and a block describing it would be a guess.
        """
        context = getattr(self, "_gradient_context", None)
        symbols = getattr(self, "_basis_symbols", None)
        if not context or not symbols or "atom_of_orbital" not in context:
            return None
        from .basis_report import basis_report
        return basis_report(self.basis, context, symbols)

    # -- ASE calculator hook ---------------------------------------------- #

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """ASE hook: build (if needed), run, and store the ground-state energy.

        In calculator mode the Hamiltonian is (re)built from the current geometry
        each call, so energies track the geometry; in direct mode the Hamiltonian
        supplied at construction is reused.  The run result is stored on
        :attr:`result` and the ground-state energy (eV) in ``results``.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms                # the Atoms copy stored by the base class

        if self.dry_run:
            # Estimate the qubit budget and stop: nothing is built or executed
            # (so even a reserved device is fine to *ask about*).
            self._dry_run_estimate(atoms)
            self.results["energy"] = float("nan")
            self.results["free_energy"] = float("nan")
            return

        require_runnable(self.device)     # refuses a reserved (non-runnable) entry
        self._wall_start = _perf()        # wall clock spans integration + run

        if not self._built_from_hamiltonian:
            # What is known before the build (the geometry) is reported
            # before it: building the basis and the integrals can take a
            # minute, and the log should not stay empty meanwhile.
            self._open_run_log(atoms)
            try:
                hamiltonian, num_particles, n_orbitals = \
                    self._build_hamiltonian(atoms)
                self._configure(hamiltonian, num_particles, n_orbitals)
            except BaseException:
                self._close_run_log()
                raise
            self._maybe_write_references()

        result = self.run(**self._run_kwargs(atoms))
        self.result = result

        # The result already carries the output units; ASE wants eV.
        energy_ev = self._from_energy_units(result.optimal_energy, "eV")
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev

    def _run_kwargs(self, atoms) -> dict:
        """Keyword arguments forwarded to :meth:`run` from the ASE hook."""
        return dict(self.run_options)

    # -- run scaffolding -------------------------------------------------- #

    def _make_timings(self):
        """Fresh :class:`~mandacaru.utils.profiling.Timings` and the run start time.

        Pops the ``_wall_start`` seeded by :meth:`calculate` (so the wall clock
        spans the integration too) or starts the clock now in direct mode.
        """
        from ..integrals import _backend
        from ..utils.profiling import Timings
        timings = Timings(n_cores=_backend.num_threads(),
                          backend="C (OpenMP)" if _backend.HAS_C_BACKEND
                          else "NumPy")
        # A new run starts with its budget verdict cleared.
        self.run_budget_exhausted = None
        run_t0 = self.__dict__.pop("_wall_start", None)
        if run_t0 is None:
            run_t0 = _perf()
        return timings, run_t0

    def _finalize_timings(self, timings, run_t0) -> None:
        """Fold the calculator-mode integration stage in and set the wall time."""
        if self._integration_profile is not None:
            for name, secs in self._integration_profile.get("stages_s", {}).items():
                timings.add(f"integration: {name}", secs)
        timings.wall_time = _perf() - run_t0

    # -- performance accounting -------------------------------------------- #

    def qpu_accounting(self, wall_time_s: float | None = None) -> dict:
        """QPU usage of this run's provider, or ``{}`` when none was used."""
        from ..backends.providers import qpu_usage

        cached = getattr(self, "_provider_cache", None)
        provider = None if cached is None else cached[1]
        if provider is None or not getattr(self, "shots", 0):
            # No provider, or an exact (shots = 0) evaluation: nothing ran on a
            # processor, so there is nothing to account for.
            return {}
        return qpu_usage(provider, wall_time_s, cumulative=True)

    def write_performance(self, timings, extra: dict | None = None) -> None:
        """Append this run's ``[PERFORMANCE]`` block to the run report.

        Skipped when there is nowhere to write it, and when
        :attr:`defer_performance` says the caller will write the block itself
        (:class:`~mandacaru.algorithms.calculator.Mandacaru` does, so the nuclear
        gradient's time lands in the same block as the solver's stages).
        """
        path = self.log_targets
        if not path or self.defer_performance:
            return
        from ..utils.logging import append_performance

        accounting = dict(self.qpu_accounting())
        accounting.update(extra or {})
        append_performance(path, stages=dict(timings.stages),
                           wall_time_s=timings.wall_time,
                           resources=timings.resources(),
                           extra=accounting or None)

    def _show_banner(self) -> None:
        """Write the start-up banner to stdout, **once per process**.

        It is provenance -- versions, host, interpreter -- so it is the same
        every time, while ``calculate()`` runs once per geometry: a twelve-step
        relaxation used to reprint the whole logo and dependency dump twelve
        times, burying the iteration tables it separates.  Reset
        ``mandacaru.algorithms.base._BANNER_SHOWN`` to print it again.
        """
        global _BANNER_SHOWN
        if self.verbose and not _BANNER_SHOWN:
            from ..utils import banner
            banner.show()
            _BANNER_SHOWN = True

    # -- the run log's opening blocks ------------------------------------ #

    #: The name in the ``[SYSTEM]`` block's title; each solver sets its own.
    log_title = ""

    def _log_title(self) -> str:
        """The ``[SYSTEM]`` block's title: the method, and what it runs on.

        Written before the Hamiltonian is built, so it can name only what the
        options already fix; a driver extends it with its pool or ansatz.
        """
        return self.log_title or type(self).__name__

    def _system_logger(self, targets, geometry, cell):
        """Open the run log and write its ``[SYSTEM]`` block.

        Everything in it -- the geometry, the cell, the periodicity, the
        initial moments, the pool or ansatz named in the title -- is known before the
        Hamiltonian is built, so a calculator run writes it first (see
        :meth:`_open_run_log`); the register width is not, and the title
        leaves it to ``[ELECTRONS]``.
        """
        from ..utils.logging import Logger

        resolved = _resolve_geometry(geometry)
        symbols, positions = resolved.symbols, resolved.positions
        cell = resolved.cell if cell is None else cell

        # Geometry from ASE is in Angstrom; convert to Bohr only if requested.
        if self.atomic_units:
            if positions is not None:
                positions = np.asarray(positions, float) * ANGSTROM_TO_BOHR
            if cell is not None:
                cell = np.asarray(cell, float) * ANGSTROM_TO_BOHR

        logger = Logger(targets)
        logger.write_system(
            symbols=symbols, positions=positions, cell=cell,
            pbc=resolved.pbc, magmoms=resolved.magmoms,
            units=self._length_unit_label(),
            title=self._log_title(),
            extra=self._system_extras())
        return logger

    def _open_run_log(self, atoms) -> None:
        """Write ``[SYSTEM]`` before the build; :meth:`_make_logger` continues."""
        targets = self.log_targets
        if not targets or not self.writes_output_log:
            return
        options = self._run_kwargs(atoms)
        # The terminal banner comes first, as it does for every run.
        self._show_banner()
        self._early_logger = self._system_logger(
            targets, options.get("geometry"), options.get("cell"))

    def _close_run_log(self) -> None:
        logger = self.__dict__.pop("_early_logger", None)
        if logger is not None:
            logger.close()

    def _open_header_logger(self, targets, geometry, cell):
        """The run log with ``[SYSTEM]``, ``[BASIS]`` and ``[ELECTRONS]`` written.

        Continues the logger :meth:`_open_run_log` opened before the build, or
        opens one; the solver writes its own ``[OPTIMIZATION SETUP]`` next.
        """
        logger = self.__dict__.pop("_early_logger", None)
        if logger is None:
            logger = self._system_logger(targets, geometry, cell)
        logger.set_register(self.n_qubits)
        # The basis that ran (defaults resolved, radii, dataset files) has a
        # block of its own; absent in direct mode, where no basis was built.
        report = self._basis_report()
        if report is not None:
            logger.write_basis(*report)
        logger.write_electrons(self._electron_fields())
        report = self._active_space_report()
        if report is not None:
            logger.write_active_space(*report)
        return logger

    def _spin_polarized_label(self) -> str:
        """Whether the reference is open-shell, with its multiplicity.

        Read from the particle numbers, not from the ``spin`` flag: that flag
        is only a fallback request, and an odd electron count or the geometry's
        magnetic moments make a doublet or a triplet with it left ``False`` --
        which is what this line used to print for them.
        """
        n_alpha, n_beta = (int(n) for n in self.num_particles)
        return (f"{n_alpha != n_beta} "
                f"(multiplicity {abs(n_alpha - n_beta) + 1})")

    def _electron_fields(self) -> dict:
        """The ``[ELECTRONS]`` block: how the electronic problem was posed.

        The values the standard-output header prints, in the same order, less
        the frozen core and the active space, which the ``[ACTIVE SPACE]``
        block owns --
        discretization first (what the integrals were computed on), then the
        encoding and the size of the register and Hamiltonian it produced.  The
        trace is off whenever this file is written, so this is the only place the
        configuration is recorded.
        """
        orbitals = (getattr(self, "n_spatial_orbitals", None)
                    or getattr(getattr(self, "pool", None),
                               "n_spatial_orbitals", None))
        # Adding a key here changes the [ELECTRONS] block, whose exact ordered
        # field list is pinned by `test/utils/test_logging.py`
        # (TestElectronsBlock.FIELDS).  That is deliberate -- the block is the
        # only record of the configuration when the trace goes to a file -- so a
        # new field means updating that tuple in the same change.
        return {
            **self._grid_fields(),
            "kinetic operator": self.kinetic or "finite difference",
            "k-points": self._kpts_label(),
            "charge": str(self.charge),
            "spin-polarized": self._spin_polarized_label(),
            "reference state": str(self.initial_state),
            "Z2 tapering": self._taper_label(),
            "mapping": MAPPING_LABELS.get(str(self.mapping), str(self.mapping)),
            "Hamiltonian": f"{len(self.hamiltonian.simplify().terms)} "
                           f"Pauli terms",
            "spatial orbitals": str(orbitals),
            "electrons (alpha, beta)": str(self.num_particles),
            "qubits": str(self.n_qubits),
        }

    def _grid_fields(self) -> dict:
        """The ``[ELECTRONS]`` lines of the real-space grid.

        The basis is not here: the ``[BASIS]`` block owns it.  The grid is --
        the requested spacing and what the cell turned it into, which is the
        number a comparison with another real-space code needs.
        """
        grid = getattr((getattr(self, "_gradient_context", None) or {})
                       .get("integrals"), "grid", None)
        realized = None
        if grid is not None:
            from ..units import BOHR_TO_ANGSTROM
            spacing = " x ".join(f"{d * BOHR_TO_ANGSTROM:.4f}"
                                 for d in (grid.dx, grid.dy, grid.dz))
            realized = (f"{grid.nx} x {grid.ny} x {grid.nz} "
                        f"(spacing {spacing} Angstrom)")
        return {"grid spacing": f"{self.h:g} Angstrom (requested)",
                "grid points": realized}

    #: What the ``[ACTIVE SPACE]`` occupation column is, per selector.  The
    #: MP2 selector diagonalizes the active blocks only, so a frozen row is a
    #: diagonal element and every other row an eigenvalue.
    _OCCUPATION_SOURCES = {
        "mp2": "eigenvalues of the MP2 one-particle density (natural "
               "orbitals); frozen rows are its diagonal, since frozen "
               "orbitals are not rotated",
        "dlpno-mp2": "eigenvalues of the local (DLPNO) MP2 one-particle "
                     "density, pair natural orbitals cut at occupation "
                     "1e-08, domains at differential overlap 0.01 and "
                     "distant pairs below an estimated 1e-06 Ha left to "
                     "their dipole estimate; frozen rows are its diagonal",
        "natural": "eigenvalues of the open-shell reference's one-particle "
                   "density (its natural occupations)",
    }

    def _active_space_request(self):
        """The ``active_space`` spec with the solver's target state count.

        ``symmetry`` keeps the orbitals the target states need, and how many
        states that is belongs to the solver (a subspace method's
        ``num_states``), not to the option.
        """
        from dataclasses import replace

        spec = self.active_space
        states = int(getattr(self, "num_states", 1) or 1)
        if spec is None or not spec.symmetry or states == spec.states:
            return spec
        return replace(spec, states=states)

    def _active_space_report(self):
        """``(fields, tables)`` of the ``[ACTIVE SPACE]`` block, or ``None``.

        ``None`` when every orbital reaches the register.  Otherwise the
        partition that was actually built -- a count or a threshold can freeze
        more than ``"frozen"`` asked for -- with every spatial orbital's
        reference occupancy, its role and, when the selector computed one, its
        occupation number.  Indices are those of the selector's ranked basis.
        """
        space = (getattr(self, "_gradient_context", None) or {}).get(
            "active_space")
        if space is None or not (space.truncated or space.frozen):
            return None
        spec = self.active_space
        # Only a truncation ranks anything: a frozen core alone keeps the
        # default method name without having run it.
        method = space.method if space.truncated else "none (frozen core only)"
        fields = {"method": method}
        if spec is not None:
            if spec.orbitals is not None:
                orbitals = spec.orbitals
                fields["orbitals_requested"] = (
                    list(orbitals) if isinstance(orbitals, tuple)
                    else orbitals)
            if spec.threshold is not None:
                fields["threshold"] = f"{spec.threshold:g}"
            fields["frozen_requested"] = self._frozen_label()
            notes = dict(space.notes)
            for key in ("correlating_pairs", "symmetry"):
                if getattr(spec, key):
                    fields[key] = notes.get(key, "on")
                else:
                    fields[key] = "off"
        fields["spatial_orbitals"] = (
            f"{space.n_orbitals} ({len(space.frozen)} frozen, "
            f"{space.n_active} active, {len(space.deleted)} deleted)")
        if space.point_group is not None:
            fields["point_group"] = space.point_group
        if space.correlation_energy is not None:
            fields["mp2_correlation_energy_Ha"] = (
                f"{space.correlation_energy:.8f}")
        occupations = (None if space.occupations is None or not space.truncated
                       else np.asarray(space.occupations, dtype=float))
        fields["occupation"] = (
            self._OCCUPATION_SOURCES.get(space.method, space.method)
            if occupations is not None else
            "none (the selector computed no occupation numbers)")
        heading = ["index", "occupancy", "role"]
        if space.irreps:
            heading.append("irrep")
        if space.partners:
            heading.append("partner")
        if occupations is not None:
            heading.append("occupation")
        rows = [heading]
        for p in range(space.n_orbitals):
            row = [str(p), space.occupancy(p) or "-", space.role(p)]
            if space.irreps:
                row.append(space.irreps[p])
            if space.partners:
                partner = space.partner(p)
                row.append("-" if partner is None else str(partner))
            if occupations is not None:
                row.append(f"{occupations[p]:.8f}")
            rows.append(row)
        return fields, {"orbitals": rows}

    def _frozen_label(self) -> str:
        """The requested frozen core, for the header and the run log."""
        spec = self.active_space
        frozen = None if spec is None else spec.frozen
        if frozen is None:
            return "none"
        if isinstance(frozen, tuple):
            return f"orbitals {list(frozen)}"
        return str(frozen)

    def _backend_description(self) -> str:
        """How the state vector and the pool are represented."""
        sector = getattr(self, "_sector", None)
        if getattr(self, "_matrix_free", False):
            space = (f"particle-number sector, {sector.dim} states of "
                     f"2^{self.n_qubits}" if sector is not None
                     else f"2^{self.n_qubits} amplitudes")
            return f"matrix-free Pauli products, {space}"
        if sector is not None:
            return (f"particle-number sector, {sector.dim} states "
                    f"of 2^{self.n_qubits}")
        if getattr(self, "_sparse", False):
            return f"sparse matrices, 2^{self.n_qubits} amplitudes"
        return f"dense matrices, 2^{self.n_qubits} amplitudes"

    # -- subclass hooks --------------------------------------------------- #

    def _configure(self, hamiltonian, num_particles, n_orbitals) -> None:
        """Build the ansatz / pool for ``hamiltonian`` and materialize it."""
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """Run the optimization and return the driver's result dataclass."""
        raise NotImplementedError


def measure_energies(drivers, provider) -> list[float]:
    """Energies of several optimized drivers, in **one** provider job.

    ``drivers`` are run calculators/solvers (each has ``.ansatz``, ``.result``
    and ``.hamiltonian``); ``provider`` is typically a
    :class:`~mandacaru.backends.providers.QiskitProvider` on IBM hardware, so a
    whole dissociation curve costs a single job.  Each energy comes back in its
    driver's output units (eV by default; Hartree for ``atomic_units=True``),
    while the provider measures the Hartree qubit Hamiltonian.
    """
    drivers = list(drivers)
    problems = [d.ansatz_problem() for d in drivers]
    if hasattr(provider, "energies"):
        energies = provider.energies(problems)
    else:
        energies = [provider.energy(*p) for p in problems]
    return [_driver_units(d)(e) for d, e in zip(drivers, energies)]


def _driver_units(driver):
    """The Hartree -> output-unit converter of a driver or ``Mandacaru`` wrapper."""
    convert = getattr(driver, "_to_energy_units", None)
    if convert is None:                       # a Mandacaru wrapper: ask the solver
        convert = getattr(getattr(driver, "solver", None), "_to_energy_units",
                          None)
    return convert if convert is not None else (lambda e: from_hartree(e, "eV"))
