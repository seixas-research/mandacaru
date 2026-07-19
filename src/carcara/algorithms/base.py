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

from time import perf_counter as _perf

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from ..backends.hardware import normalize_device, require_runnable
from ..core.mapping import Fermion, PauliSum
from ..optimizers.optim import NAMED_OPTIMIZERS, resolve_optimizer
from ..units import from_hartree
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


class VariationalDriver(Calculator):
    """Base ASE calculator for the variational state-vector eigensolvers.

    Not used directly -- concrete drivers subclass it and implement
    :meth:`_configure` and :meth:`run`.  The constructor accepts the problem-setup
    surface shared by every driver; algorithm-specific arguments (ansatz, pool,
    stopping controls, ...) are added by the subclass and passed through
    ``**shared`` to :meth:`__init__`.
    """

    implemented_properties = ["energy", "free_energy"]
    _OPTIMIZERS = NAMED_OPTIMIZERS

    #: Attribute name the ASE hook stores the run result under (``vqe_result`` ...).
    _result_attr = "result"
    #: Default ``sparse`` policy (``False`` dense; ``"auto"`` for adaptive drivers).
    _default_sparse = False

    def __init__(self, *, optimizer="COBYLA", mapping: str = "jordan_wigner",
                 basis="FAO", device: str = "AER_simulator", grid=None,
                 h: float = 0.20, kpts=None, spin: bool = False,
                 initial_state: str | None = "hartree-fock", charge: int = 0,
                 n_electrons=None, frozen_core=False, frozen_orbitals=None,
                 hamiltonian_builder=None, run_options: dict | None = None,
                 verbose: bool = True, sparse=None, **calc_kwargs):
        Calculator.__init__(self, **calc_kwargs)

        self.verbose = bool(verbose)
        self.optimizer = resolve_optimizer(optimizer, allowed=self._OPTIMIZERS)
        self.mapping = mapping
        self.basis = basis
        self.device = normalize_device(device)      # raises on unknown device
        self.grid = grid
        self.h = float(h)
        self.spin = bool(spin)
        self.frozen_core = frozen_core
        self.frozen_orbitals = frozen_orbitals
        self.initial_state = resolve_initial_state(initial_state)
        # Monkhorst-Pack mesh (ASE); the engine is Gamma-point (molecular), so a
        # denser mesh is stored on ``kpoints`` but rejected at run time.
        self.kpts, self.kpts_gamma, self.kpoints = monkhorst_pack_kpts(kpts)
        self.charge = int(charge)
        self.n_electrons = n_electrons
        self.hamiltonian_builder = hamiltonian_builder
        self.run_options = dict(run_options or {})
        self.sparse = self._default_sparse if sparse is None else sparse

        self._integration_profile = None
        self._configured = False
        # True when a Hamiltonian was supplied at construction (direct mode); the
        # ASE hook then never rebuilds it.  In calculator mode it stays False and
        # the Hamiltonian is (re)built from the geometry on each ``calculate``.
        self._built_from_hamiltonian = False

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
        centred = ", Gamma-centred" if self.kpts_gamma else ""
        if len(self.kpoints) == 1:
            return f"Gamma ({n1}x{n2}x{n3} Monkhorst-Pack)"
        return (f"{len(self.kpoints)} k-points ({n1}x{n2}x{n3} "
                f"Monkhorst-Pack{centred})")

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

    def _as_pauli_sum(self, hamiltonian, n_qubits: int) -> PauliSum:
        """Coerce a ``PauliSum`` / ``Fermion`` Hamiltonian to a ``PauliSum``."""
        if isinstance(hamiltonian, PauliSum):
            return hamiltonian
        if isinstance(hamiltonian, Fermion):
            return hamiltonian.map_to_qubits(self.mapping, n_modes=n_qubits)
        raise TypeError("hamiltonian must be a PauliSum or Fermion")

    def _materialize_hamiltonian(self, qubit_h: PauliSum, n_qubits: int) -> bool:
        """Store the (Hermitized) dense or sparse Hamiltonian matrix.

        Sets :attr:`hamiltonian`, :attr:`n_qubits`, :attr:`_sparse` and
        :attr:`_h_matrix`; returns the resolved sparse flag.  A sparse matrix is
        used when :meth:`_resolve_sparse` selects it (large active spaces).
        """
        if qubit_h.num_qubits != n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {qubit_h.num_qubits} qubits but the ansatz "
                f"/ pool has {n_qubits}")
        self.hamiltonian = qubit_h
        self.n_qubits = int(n_qubits)
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

    # -- geometry -> Hamiltonian (calculator mode) ------------------------ #

    def _build_hamiltonian(self, atoms):
        """Build ``(hamiltonian, num_particles, n_spatial_orbitals)`` from ``atoms``.

        Uses an explicit ``hamiltonian_builder`` if given, otherwise the built-in
        ``basis`` engine (stashing its integration profile for the run summary).
        """
        if self.hamiltonian_builder is not None:
            hamiltonian, num_particles, n_orbitals = \
                self.hamiltonian_builder(atoms)
            return hamiltonian, num_particles, n_orbitals
        from ._hamiltonian_from_atoms import build_basis_hamiltonian
        hamiltonian, num_particles, n_orbitals, profile = build_basis_hamiltonian(
            atoms, self.basis, self.grid, self.h, self.charge, self.n_electrons,
            spin=self.spin, frozen_core=self.frozen_core,
            frozen_orbitals=self.frozen_orbitals)
        self._integration_profile = profile
        return hamiltonian, num_particles, n_orbitals

    # -- ASE calculator hook ---------------------------------------------- #

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """ASE hook: build (if needed), run, and store the ground-state energy.

        In calculator mode the Hamiltonian is (re)built from the current geometry
        each call, so energies track the geometry; in direct mode the Hamiltonian
        supplied at construction is reused.  The run result is stored on
        :attr:`_result_attr` and the ground-state energy (eV) in ``results``.
        """
        require_runnable(self.device)     # e.g. 'ibm-quantum' is not runnable yet
        self._wall_start = _perf()        # wall clock spans integration + run
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms                # the Atoms copy stored by the base class

        if not self._built_from_hamiltonian:
            hamiltonian, num_particles, n_orbitals = self._build_hamiltonian(atoms)
            self._configure(hamiltonian, num_particles, n_orbitals)

        result = self.run(**self._run_kwargs(atoms))
        setattr(self, self._result_attr, result)

        energy_ev = float(from_hartree(result.optimal_energy, "eV"))
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

    def _show_banner(self) -> None:
        """Write the start-up banner to stdout (verbose runs only)."""
        if self.verbose:
            from ..utils import banner
            banner.show()

    # -- subclass hooks --------------------------------------------------- #

    def _configure(self, hamiltonian, num_particles, n_orbitals) -> None:
        """Build the ansatz / pool for ``hamiltonian`` and materialize it."""
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """Run the optimization and return the driver's result dataclass."""
        raise NotImplementedError
