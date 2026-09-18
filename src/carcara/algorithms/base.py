# -*- coding: utf-8 -*-
# file: algorithms/base.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Shared base for the variational state-vector drivers.

:class:`VariationalDriver` factors out everything the concrete algorithms
(:class:`~carcara.algorithms.vqe.VQE`,
:class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`, and their subspace variants)
have in common, so a new method only writes its own optimization loop:

* **problem setup** -- the ASE-calculator surface (``basis`` / ``grid`` / ``h`` /
  ``kpts`` / ``spin`` / ``frozen_core`` / ...) and the geometry-to-Hamiltonian
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
from time import perf_counter as _perf

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from ..backends.hardware import (device_arn, device_provider, is_aws_device,
                                 normalize_device, require_runnable,
                                 requires_shots)
from ..backends.providers import build_provider, normalize_provider
from ..core.mapping import Fermion, PauliSum
from ..core.serialization import (DEFAULT_FORMAT, EXTENSION_FORMATS,
                                  load_hamiltonian, resolve_format,
                                  resolve_save_path, save_hamiltonian)
from ..optimizers.optim import NAMED_OPTIMIZERS, OptimizeResult, resolve_optimizer
from ..utils.dumps import (HAMILTONIAN_FILE, POOL_FILE, dump_hamiltonian,
                           dump_pool, resolve_dump_path)
from ..units import convert_energy, energy_unit_label, from_hartree
from ._hamiltonian_from_atoms import monkhorst_pack_kpts, resolve_initial_state


def format_pauli_sum(pauli: PauliSum, indent: str = "    ",
                     max_terms: int | None = None) -> str:
    """Render a :class:`~carcara.core.mapping.PauliSum` as ``coeff * PauliString``.

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


class VariationalDriver(Calculator):
    """Base ASE calculator for the variational state-vector eigensolvers.

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
        :mod:`carcara.core.serialization`.
    verbose_operators : bool or str
        Write the **operator pool** -- every generator's label, kind, support
        and Pauli expansion -- to ``pool.json`` (default ``False``).  A path
        writes there instead.  The run trace only reports the pool's name and
        size, because a realistic pool is hundreds of operators long; this is
        where to read it.  See :mod:`carcara.utils.dumps`.
    verbose_hamiltonian : bool or str
        Write the **qubit Hamiltonian** as readable JSON (Pauli strings with
        complex coefficients, in Hartree) to ``hamiltonian.json`` (default
        ``False``); a path writes there instead.  This is for inspection --
        ``save_hamiltonian=`` writes the round-trippable form that
        ``load_hamiltonian=`` reads back.
    checkpoint : str, optional
        Path of a **wavefunction checkpoint** written during the run (see
        :mod:`carcara.core.checkpoint`): the reference determinant, the
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
        :mod:`carcara.core.serialization`.  Loading ignores this and
        auto-detects instead.
    backend_provider : str
        Quantum SDK used to construct (and, when executing, run) the ansatz
        circuits -- ``"qiskit"`` (default), ``"braket"`` (amazon-braket-sdk) or
        ``"cirq"``.  See :mod:`carcara.backends.providers`.
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
        energies on coarse grids.
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
        :class:`~carcara.algorithms.dry_run.QubitEstimate` on
        :attr:`dry_run_result`, reports ``NaN`` as the energy, and :meth:`run`
        returns the estimate instead of a result.  See
        :mod:`carcara.algorithms.dry_run` and :meth:`estimate_qubits`.
    """

    implemented_properties = ["energy", "free_energy"]
    _OPTIMIZERS = NAMED_OPTIMIZERS

    #: Default ``sparse`` policy (``False`` dense; ``"auto"`` for adaptive drivers).
    _default_sparse = False

    def __init__(self, *, optimizer="COBYLA", mapping: str = "jordan_wigner",
                 basis="FAO", device: str = "AER_simulator", grid=None,
                 h: float = 0.20, kpts=None, spin: bool = False,
                 initial_state: str | None = "hartree-fock", charge: int = 0,
                 n_electrons=None, frozen_core=False, frozen_orbitals=None,
                 hamiltonian_builder=None, run_options: dict | None = None,
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
                 kinetic: str | None = None,
                 two_qubit_reduction: bool = False,
                 atomic_units: bool = False,
                 checkpoint: str | None = None, checkpoint_every: int = 1,
                 resume: str | None = None, **calc_kwargs):
        Calculator.__init__(self, **calc_kwargs)

        self.verbose = bool(verbose)
        # Output-unit convention: eV / Angstrom unless atomic units are asked
        # for.  Internally everything stays in Hartree / Bohr; the conversion
        # happens once, where a result object or a printout is built.
        self.atomic_units = bool(atomic_units)
        self.energy_units = "Ha" if self.atomic_units else "eV"
        self.length_units = "bohr" if self.atomic_units else "angstrom"
        self.optimizer = resolve_optimizer(optimizer, allowed=self._OPTIMIZERS)
        self.mapping = mapping
        # Parity mapping's Z2 tapering: two qubits fewer, same physics.
        self.two_qubit_reduction = bool(two_qubit_reduction)
        if self.two_qubit_reduction:
            from ..core.mapping import _canonical_method
            if _canonical_method(mapping) != "parity":
                raise ValueError("two_qubit_reduction requires mapping='parity'")
            if not self._supports_two_qubit_reduction:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support the two-qubit "
                    "reduction (its reference determinants are not tapered)")
        self.basis = basis
        self.device = normalize_device(device)      # raises on unknown device
        self.grid = grid
        self.h = float(h)
        self.spin = bool(spin)
        self.frozen_core = frozen_core
        self.frozen_orbitals = frozen_orbitals
        # A pseudopotential family is a basis name (basis="PAW", ...); its
        # options are validated here so a typo fails at construction, not
        # after the grid.  It replaces the core + the -Z/r singularity with a
        # smooth valence-only problem and subsumes the frozen core.
        self._check_pseudo_basis(basis, frozen_core, frozen_orbitals)
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
        self.n_electrons = n_electrons
        self.hamiltonian_builder = hamiltonian_builder
        self.run_options = dict(run_options or {})
        self.sparse = self._default_sparse if sparse is None else sparse

        # Hamiltonian disk cache: `load_hamiltonian` short-circuits the integral
        # engine and the fermion-to-qubit mapping; `save_hamiltonian` dumps the
        # qubit Hamiltonian (Pauli strings) once it has been built.
        self.load_hamiltonian = (None if load_hamiltonian is None
                                 else str(load_hamiltonian))
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
        # qubit-wise-commuting groups (see carcara.backends.measurement).
        self.shots = int(shots)
        # A dry run only *estimates* the register, so hardware may be named
        # without shots there.
        if requires_shots(self.device) and self.shots <= 0 and not dry_run:
            raise ValueError(
                f"device {self.device!r} is real quantum hardware, which cannot "
                "return a state vector: pass shots > 0 (e.g. shots=8192) so the "
                "energy is estimated from measurements.")
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
        #: :class:`~carcara.algorithms.dry_run.QubitEstimate` of the last dry run.
        self.dry_run_result = None

        self._integration_profile = None
        self._gradient_context = None
        #: :class:`~carcara.core.sector.ParticleSector` the operators are
        #: restricted to, or ``None`` for the full register.
        self._sector = None
        self._configured = False
        #: Result of the most recent run (set by :meth:`run` / the ASE hook).
        self.result = None
        #: Set by :class:`~carcara.algorithms.calculator.Carcara` so the solver
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
        #: The last :class:`~carcara.core.checkpoint.WavefunctionCheckpoint`
        #: written (or built) by this driver.
        self.checkpoint = None

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
        """Resolve the ``sparse`` spec to a bool.

        ``"auto"`` enables the sparse backend for ``n_qubits >= 10`` (where a dense
        pool would need tens of GB); ``True`` / ``False`` force it on / off.
        """
        if isinstance(sparse, str):
            if sparse.strip().lower() == "auto":
                return int(n_qubits) >= 10
            raise ValueError(
                f"unknown sparse spec {sparse!r}; use True, False or 'auto'")
        return bool(sparse)

    #: Drivers whose reference states cannot be tapered override this.
    _supports_two_qubit_reduction = True

    def _as_pauli_sum(self, hamiltonian, n_qubits: int,
                      num_particles=None) -> PauliSum:
        """Coerce a ``PauliSum`` / ``Fermion`` Hamiltonian to a ``PauliSum``.

        ``n_qubits`` is the register size; with the two-qubit reduction the
        fermionic operator has two more modes than that.
        """
        if isinstance(hamiltonian, PauliSum):
            return hamiltonian
        if isinstance(hamiltonian, Fermion):
            if self.two_qubit_reduction:
                particles = (num_particles if num_particles is not None
                             else getattr(self, "num_particles", None))
                return hamiltonian.map_to_qubits(
                    self.mapping, n_modes=n_qubits + 2,
                    two_qubit_reduction=True, num_particles=particles)
            return hamiltonian.map_to_qubits(self.mapping, n_modes=n_qubits)
        raise TypeError("hamiltonian must be a PauliSum or Fermion")

    def _materialize_hamiltonian(self, qubit_h: PauliSum, n_qubits: int,
                                 sector=None) -> bool:
        """Store the (Hermitized) dense or sparse Hamiltonian matrix.

        Sets :attr:`hamiltonian`, :attr:`n_qubits`, :attr:`_sparse` and
        :attr:`_h_matrix`; returns the resolved sparse flag.  A sparse matrix is
        used when :meth:`_resolve_sparse` selects it (large active spaces).
        With a ``sector`` (:class:`~carcara.core.sector.ParticleSector`) the
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
        if sector is not None:
            self._sparse = True
            hs = sector.restrict(qubit_h)
            self._h_matrix = 0.5 * (hs + hs.conj().T)
            return self._sparse
        self._sparse = self._resolve_sparse(self.sparse, n_qubits)
        if self._sparse:
            hs = qubit_h.to_sparse_matrix()
            self._h_matrix = 0.5 * (hs + hs.conj().T)
        else:
            h = qubit_h.to_matrix()
            self._h_matrix = 0.5 * (h + h.conj().T)     # Hermitize rounding noise
        return self._sparse

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
        (:mod:`carcara.backends.measurement`).
        """
        if not self.shots:
            return self.energy(ansatz.state(theta))
        provider = self.circuit_provider()
        return provider.energy(ansatz.n_qubits, ansatz.reference_qubits(),
                               ansatz.pauli_generators, theta, self.hamiltonian)

    # -- circuit provider ------------------------------------------------- #

    def circuit_provider(self):
        """The :class:`~carcara.backends.providers.CircuitProvider`, or ``None``.

        ``None`` means the driver evaluates the ansatz with the internal
        (NumPy / SciPy-sparse) state-vector backend; a provider means every state
        preparation is compiled to a circuit and executed on that SDK's
        simulator or, for an Amazon Braket device, on the AWS service.  Both
        produce the same unitary -- see :mod:`carcara.backends.providers`.

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
        ansatz = self.ansatz
        if theta is None:
            theta = self.result.optimal_parameters
        return (ansatz.n_qubits, ansatz.reference_qubits(),
                ansatz.pauli_generators, np.asarray(theta, dtype=float),
                self.hamiltonian)

    def measured_energy(self, provider, theta=None) -> float:
        """``<H>`` of the optimized ansatz evaluated on ``provider``.

        The way to run on real hardware within a budget: optimize locally,
        then measure the final state once -- e.g. with
        ``QiskitProvider(device="ibm_kingston", shots=4096)``.  Returned in the
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

        A **tapered** record (``two_qubit_reduction``) restores that setting the
        same way: the stored operator is ``2M - 2`` qubits wide, so a driver that
        did not know would build a pool two qubits too wide and refuse to run.
        A driver that asked for tapering and is handed an untapered file is a
        real contradiction -- the file cannot be tapered after the fact without
        the mapping it was written in -- and says so.
        """
        record = load_hamiltonian(self.load_hamiltonian)
        self.mapping = record.mapping
        if record.two_qubit_reduction and not self.two_qubit_reduction:
            if not self._supports_two_qubit_reduction:
                raise NotImplementedError(
                    f"{self.load_hamiltonian!r} holds a tapered Hamiltonian, "
                    f"which {type(self).__name__} does not support (its "
                    "reference determinants are not tapered)")
            self.two_qubit_reduction = True
        elif self.two_qubit_reduction and not record.two_qubit_reduction:
            raise ValueError(
                f"{self.load_hamiltonian!r} holds an untapered "
                f"{record.num_qubits}-qubit Hamiltonian, but this driver was "
                "built with two_qubit_reduction=True; drop the flag (the file "
                "decides) or point at a file written with it")
        self._loaded_record = record
        return (record.hamiltonian, record.num_particles,
                record.n_spatial_orbitals)

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
            two_qubit_reduction=self.two_qubit_reduction, format=fmt,
            metadata={"driver": type(self).__name__,
                      "basis": self.basis if isinstance(self.basis, str)
                      else dict(self.basis),
                      "frozen_core": self.frozen_core,
                      "n_qubits": int(self.n_qubits)})

    def _maybe_dump_hamiltonian(self, num_particles=None,
                                n_spatial_orbitals=None) -> str | None:
        """Write ``hamiltonian.json`` when ``verbose_hamiltonian`` is set."""
        if self._hamiltonian_dump_path is None:
            return None
        return dump_hamiltonian(
            self._hamiltonian_dump_path, self.hamiltonian,
            n_qubits=self.n_qubits, mapping=self.mapping,
            num_particles=num_particles,
            n_spatial_orbitals=n_spatial_orbitals,
            two_qubit_reduction=self.two_qubit_reduction)

    def _maybe_dump_pool(self, pool, operators) -> str | None:
        """Write ``pool.json`` when ``verbose_operators`` is set."""
        if self._pool_dump_path is None:
            return None
        return dump_pool(
            self._pool_dump_path, pool, operators, n_qubits=self.n_qubits,
            mapping=self.mapping, num_particles=self.num_particles,
            two_qubit_reduction=self.two_qubit_reduction)

    # -- wavefunction checkpoints ---------------------------------------- #

    def _checkpoint_record(self, ansatz, parameters, energy_ha, status,
                           labels=None, kinds=None):
        """Describe the current state as a
        :class:`~carcara.core.checkpoint.WavefunctionCheckpoint`.

        Everything a consumer needs is on the ansatz -- the register, the
        reference determinant in register terms, the generators as Pauli sums
        -- so the record is the same for every ansatz class.
        """
        from ..core.checkpoint import WavefunctionCheckpoint

        generators = list(ansatz.pauli_generators)
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
            "carcara_version": __import__("carcara").__version__,
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
            two_qubit_reduction=bool(self.two_qubit_reduction),
            num_particles=None if particles is None else tuple(particles),
            n_spatial_orbitals=getattr(self, "n_spatial_orbitals", None),
            occupied_orbitals=None if occupied is None else list(occupied),
            energy=None if energy_ha is None else float(energy_ha),
            hamiltonian=getattr(self, "hamiltonian", None),
            method=self._method_name(),
            status=dict(status), metadata=metadata)

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
        if bool(record.two_qubit_reduction) != bool(self.two_qubit_reduction):
            problems.append("two-qubit reduction differs")
        if sorted(record.reference_qubits) != sorted(ansatz.reference_qubits()):
            problems.append(f"reference determinant {record.reference_qubits} "
                            f"in the file, {ansatz.reference_qubits()} here")
        if problems:
            raise ValueError(
                f"cannot resume from {self.resume_path!r}: "
                + "; ".join(problems))
        return record

    # -- optimization policy (quenching) ---------------------------------- #

    def _optimize_grown(self, cost, previous_parameters) -> OptimizeResult:
        """Optimize after appending one new parameter, honoring :attr:`quenching`.

        ``quenching=True`` (default) hands **all** parameters to the classical
        optimizer, warm-started from the previous optimum with the new angle at
        zero -- standard ADAPT-VQE.  ``quenching=False`` freezes the previously
        optimized parameters and varies only the newly added one, a cheaper
        one-dimensional line search per growth step that trades variational
        freedom for cost-function evaluations.

        ``cost`` takes the **full** parameter vector in both cases; the returned
        :class:`~carcara.optimizers.optim.OptimizeResult` also carries the full
        vector, so callers need no branching.
        """
        previous = np.asarray(previous_parameters, dtype=float).ravel()
        x0 = np.concatenate([previous, [0.0]])
        if self.quenching:
            return self.optimizer.minimize(cost, x0)

        def last_only(tail):
            return cost(np.concatenate(
                [previous, np.asarray(tail, dtype=float).ravel()]))

        result = self.optimizer.minimize(last_only, np.zeros(1))
        full = np.concatenate([previous,
                               np.asarray(result.x, dtype=float).ravel()])
        return OptimizeResult(x=full, fun=result.fun, nfev=result.nfev,
                              history=result.history, success=result.success,
                              message=result.message)

    def _optimize_all(self, cost, x0, callback=None) -> OptimizeResult:
        """Optimize a fixed-size parameter vector, honoring :attr:`quenching`.

        ``quenching=True`` (default) is a single joint minimization over every
        parameter.  ``quenching=False`` sweeps the parameters one at a time in
        order -- parameter ``k`` is optimized alone with ``0..k-1`` frozen at their
        already-optimized values and ``k+1..`` held at their starting values --
        the fixed-ansatz analogue of freezing previous growth steps.
        """
        x0 = np.asarray(x0, dtype=float).ravel()
        if self.quenching or x0.size <= 1:
            return self.optimizer.minimize(cost, x0, callback=callback)

        params = x0.copy()
        history: list[float] = []
        nfev = 0
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
            value = float(step.fun)
            success = success and step.success
        return OptimizeResult(x=params, fun=value, nfev=nfev, history=history,
                              success=success,
                              message="sequential (quenching=False) sweep")

    @staticmethod
    def _check_pseudo_basis(basis, frozen_core, frozen_orbitals):
        """Validate a pseudopotential basis spec up front (names, options,
        no frozen core); all-electron specs pass untouched."""
        from ._hamiltonian_from_atoms import (PER_ELEMENT,
                                              pseudopotential_family,
                                              resolve_basis)
        name, options = resolve_basis(basis)
        if name == PER_ELEMENT:
            return                      # validated per element at build time
        family = pseudopotential_family(name)
        if family is None:
            return
        unknown = sorted(set(options) - set(family.options))
        if unknown:
            raise ValueError(
                f"unknown option(s) {unknown} for the {family.label} basis; "
                f"it accepts {list(family.options)}")
        if frozen_core or frozen_orbitals:
            raise ValueError(
                f"frozen_core is redundant with the {family.label} basis -- "
                "the core is already absent from the valence-only pseudo basis")

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
        ``frozen_core`` / ``mapping`` / ``device`` /
        ``load_hamiltonian``).  ``atoms`` is needed in calculator mode; in
        direct mode (a Hamiltonian given or loaded at construction) it is
        ignored.  Nothing is integrated, mapped or executed.

        Returns a :class:`~carcara.algorithms.dry_run.QubitEstimate`.
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
            frozen_core=self.frozen_core, frozen_orbitals=self.frozen_orbitals,
            **common)

    def _dry_run_estimate(self, atoms=None):
        """Perform the dry run: store, optionally print, and return the estimate."""
        estimate = self.estimate_qubits(atoms)
        # A cached tapered Hamiltonian already reports the reduced width;
        # applying the reduction again would subtract two qubits twice.
        if self.two_qubit_reduction and not estimate.two_qubit_reduction:
            import dataclasses
            estimate = dataclasses.replace(
                estimate, n_qubits=estimate.n_qubits_reduced,
                two_qubit_reduction=True,
                notes=list(estimate.notes) + [
                    "parity two-qubit reduction applied: two qubits fewer"])
        self.dry_run_result = estimate
        self.result = None
        if self.verbose:
            self._show_banner()
            print(estimate.summary())
        return estimate

    # -- geometry -> Hamiltonian (calculator mode) ------------------------ #

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
            spin=self.spin, frozen_core=self.frozen_core,
            frozen_orbitals=self.frozen_orbitals, kinetic=self.kinetic)
        self._integration_profile = profile
        # Kept for the nuclear gradient: the integral engine that produced this
        # Hamiltonian, and which atom each basis function belongs to.
        self._gradient_context = context
        return hamiltonian, num_particles, n_orbitals

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

        require_runnable(self.device)     # e.g. 'ibm-quantum' is not runnable yet
        self._wall_start = _perf()        # wall clock spans integration + run

        if not self._built_from_hamiltonian:
            hamiltonian, num_particles, n_orbitals = self._build_hamiltonian(atoms)
            self._configure(hamiltonian, num_particles, n_orbitals)

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
        """Fresh :class:`~carcara.utils.profiling.Timings` and the run start time.

        Pops the ``_wall_start`` seeded by :meth:`calculate` (so the wall clock
        spans the integration too) or starts the clock now in direct mode.
        """
        from ..integrals import _backend
        from ..utils.profiling import Timings
        timings = Timings(n_cores=_backend.num_threads(),
                          backend="C (OpenMP)" if _backend.HAS_C_BACKEND
                          else "NumPy")
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
        return qpu_usage(provider, wall_time_s)

    def write_performance(self, timings, extra: dict | None = None) -> None:
        """Append this run's ``[PERFORMANCE]`` block to the ``output.txt`` log.

        Skipped when there is no log, and when
        :attr:`defer_performance` says the caller will write the block itself
        (:class:`~carcara.algorithms.calculator.Carcara` does, so the nuclear
        gradient's time lands in the same block as the solver's stages).
        """
        path = getattr(self, "output", None)
        if path is None or self.defer_performance:
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
        ``carcara.algorithms.base._BANNER_SHOWN`` to print it again.
        """
        global _BANNER_SHOWN
        if self.verbose and not _BANNER_SHOWN:
            from ..utils import banner
            banner.show()
            _BANNER_SHOWN = True

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
    :class:`~carcara.backends.providers.QiskitProvider` on IBM hardware, so a
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
    """The Hartree -> output-unit converter of a driver or ``Carcara`` wrapper."""
    convert = getattr(driver, "_to_energy_units", None)
    if convert is None:                       # a Carcara wrapper: ask the solver
        convert = getattr(getattr(driver, "solver", None), "_to_energy_units",
                          None)
    return convert if convert is not None else (lambda e: from_hartree(e, "eV"))
