# -*- coding: utf-8 -*-
# file: algorithms/pool_driver.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The problem setup shared by the solvers that build ansatze from a pool.

An operator pool (:mod:`mandacaru.circuits.pools`) is a list of generators;
turning it into something a solver can evaluate is the same work whichever
algorithm then decides which generators to use: build the pool by name,
reduce it with the Hamiltonian under a Z2 taper, pick the particle-number
sector every generator stays inside, materialize the generators in the run's
representation (dense, sparse, sector or matrix-free) and hand out empty
product-of-exponentials ansatze on that register.  :class:`PoolDriver` does
that once, for ADAPT-VQE (which grows the ansatz by gradient) and MCAS-VQE (which
searches it with a Markov chain) alike.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from ..circuits.adapt_ansatz import AdaptAnsatz
from ..circuits.pools import (PoolBase, PoolOperator, _support_of,
                             build_pool, pool_class)
from ..circuits.profiling import CircuitMetrics, profile_ansatz
from ..core.matrix_free import PauliOperator
from .base import VariationalDriver
from .sector_support import SectorMixin


#: Shortest the operator label is elided to before a row is allowed to wrap.


class PoolDriver(SectorMixin, VariationalDriver):
    """A variational driver whose ansatze are products of pool generators.

    Not used directly.  A subclass passes the pool options through
    ``super().__init__``, sets its own, and ends its constructor with
    :meth:`_adopt_problem`, which configures eagerly when a Hamiltonian was
    given (or loaded) and otherwise leaves it to the first ASE evaluation.

    Parameters
    ----------
    pool : PoolBase or str
        The operator pool, or a name for
        :func:`~mandacaru.circuits.pools.build_pool`.
    profile : bool
        Compile and profile ansatze for hardware cost (:meth:`_profile`).
    sector : bool or str
        Simulate only the ``(n_alpha, n_beta)`` particle-number sector:
        ``True``, ``False`` or ``"auto"`` (see :meth:`_resolve_sector`).
    """

    #: Whether this driver measures its pool gradients on the device when
    #: ``shots > 0`` (:mod:`~mandacaru.backends.gradient_measurement`).  A
    #: driver that does not screens on a local state vector, which a
    #: shot-based run may not build, so it refuses shots.
    _measures_pool_gradients = False

    def __init__(self, *, pool="fermionic", profile: bool = True,
                 sector: bool | str = "auto", **driver_kwargs):
        super().__init__(**driver_kwargs)
        if self.shots and not self._measures_pool_gradients:
            raise NotImplementedError(
                f"{type(self).__name__} screens its operator pool on a local "
                f"state vector, which a shot-based run (shots > 0) does not "
                f"build: its classical memory must stay polynomial in the "
                f"qubit count.  method='adapt-vqe' measures its pool "
                f"gradients on the device; use it, or shots=0 for a local "
                f"simulation.")
        self.profile = profile
        self._set_sector_option(sector)
        self._pool_spec = pool

    def _adopt_problem(self, hamiltonian, num_particles, n_spatial_orbitals
                       ) -> None:
        """Configure now for a given or cached Hamiltonian, else defer."""
        # A cached Hamiltonian is a complete problem specification (operator plus
        # num_particles / n_spatial_orbitals), so loading one puts the driver in
        # direct mode without a geometry -- no integrals, no mapping.
        # (A dry run does not even load it: the estimate reads the file's
        # header -- `read_hamiltonian_header` -- and never its Pauli table.)
        if hamiltonian is None and self.load_hamiltonian is not None \
                and self.dry_run:
            self._check_cache_header()
        elif hamiltonian is None and self.load_hamiltonian is not None:
            hamiltonian, loaded_particles, loaded_orbitals = \
                self._load_hamiltonian_record()
            num_particles = num_particles or loaded_particles
            n_spatial_orbitals = n_spatial_orbitals or loaded_orbitals

        # Configure eagerly when a Hamiltonian is given (direct mode); otherwise
        # defer to the first calculator evaluation (the ASE hook in the base).
        # A dry run never configures: materializing the Hamiltonian allocates
        # the 2^n matrix whose feasibility is the very thing being asked about.
        if hamiltonian is not None and not self.dry_run:
            self._configure(hamiltonian, num_particles, n_spatial_orbitals)
            self._built_from_hamiltonian = True
            self._maybe_write_references()
        elif hamiltonian is not None:
            self._dry_run_problem = (hamiltonian, num_particles,
                                     n_spatial_orbitals)

    def _configure(self, hamiltonian, num_particles, n_spatial_orbitals):
        """Resolve the pool, materialize the Hamiltonian and the pool matrices."""
        pool = self._pool_spec
        if isinstance(pool, PoolBase):
            self.pool = pool
        else:
            if n_spatial_orbitals is None or num_particles is None:
                raise ValueError(
                    "building a pool by name requires n_spatial_orbitals and "
                    "num_particles")
            self.pool = build_pool(pool, n_spatial_orbitals, num_particles,
                                   mapping=self.mapping)
        self.num_particles = (tuple(num_particles) if num_particles is not None
                              else self.pool.num_particles)
        self.spin_conserving = self._resolve_spin_conservation(hamiltonian)

        # Materialize the (dense or sparse) qubit Hamiltonian on the base; a dense
        # pool stores every operator's matrix *and* eigendecomposition (two
        # 2^n x 2^n arrays each), ~46 GB for a 12-qubit water active space, so
        # ``sparse="auto"`` keeps large active spaces as sparse matrices and
        # screens with the exact analytic gradient, densifying only selected
        # operators (in the growable ansatz).
        qubit_h = self._as_pauli_sum(hamiltonian, self.pool.n_qubits,
                                     self.num_particles)
        # The pool is built first: a sector may only be used when every
        # generator keeps the ansatz inside it (see _resolve_sector).
        self._pool_ops = self.pool.operators()
        # Z2 tapering replaces the register, so it has to happen before the
        # Hamiltonian is materialized and before a sector is resolved -- and it
        # has to reduce the Hamiltonian, every generator and the reference with
        # the *same* Clifford, which is why one call does all three.
        self._taper_info = self._resolve_taper(qubit_h)
        if self._taper_info is not None:
            qubit_h = self._taper_info.hamiltonian
            self._pool_ops = [
                self._tapered_pool_operator(self._pool_ops[index], generator)
                for index, generator in zip(self._taper_info.kept,
                                            self._taper_info.generators)]
        # The pool's width, not the Hamiltonian's: `_materialize_hamiltonian`
        # compares the two and raises when they disagree, so handing it the
        # Hamiltonian's own count makes that guard vacuous -- which is exactly
        # what a first version of this did, and what
        # `test_qubit_count_mismatch_raises` caught.  Only a taper legitimately
        # changes the register, and then the tapered width is the right answer
        # for both sides.
        n_register = (self._taper_info.n_qubits
                      if self._taper_info is not None else self.pool.n_qubits)
        self._materialize_hamiltonian(
            qubit_h, n_register,
            sector=(None if self._taper_info is not None
                    else self._resolve_sector(n_register, qubit_h,
                                              self._pool_ops)),
            operators=self._pool_ops)
        self._maybe_save_hamiltonian(self.num_particles,
                                     self.pool.n_spatial_orbitals)
        self._maybe_dump_hamiltonian(self.num_particles,
                                     self.pool.n_spatial_orbitals)
        self._maybe_dump_pool(self.pool, self._pool_ops)
        # The commutator observables of a shot-based run, built when first
        # measured (:attr:`pool_gradients`).
        self._pool_gradient_tables = None
        if self.shots:
            # Every gradient is measured on the device: no generator is
            # materialized in any representation.
            self._pool_matrices = None
        elif self._matrix_free:
            # Masks and coefficients only; a generator's product is one gather
            # (all its strings share a flip mask), so nothing is cached.
            self._pool_matrices = [
                PauliOperator(op.generator, sector=self._sector, cache_bytes=0)
                for op in self._pool_ops]
        elif self._sector is not None:
            self._pool_matrices = [self._sector.restrict(op.generator)
                                   for op in self._pool_ops]
        elif self._sparse:
            self._pool_matrices = [op.generator.to_sparse_matrix()
                                   for op in self._pool_ops]
        else:
            self._pool_matrices = [op.matrix() for op in self._pool_ops]
        self._configured = True

    @property
    def pool_gradients(self):
        """The pool's commutator observables ``[H, A_k]`` against this run's
        Hamiltonian (:class:`~mandacaru.backends.gradient_measurement.PoolGradients`),
        what a shot-based run measures its screening gradients from."""
        if self._pool_gradient_tables is None:
            from ..backends.gradient_measurement import PoolGradients
            self._pool_gradient_tables = PoolGradients(
                self.hamiltonian, [op.generator for op in self._pool_ops])
        return self._pool_gradient_tables

    def _analytic_gradients(self, psi: np.ndarray) -> np.ndarray:
        r"""Exact pool gradients ``g_i = 2 Re<H psi | A_i psi>`` (reference)."""
        h_psi = self._h_matrix @ psi
        grads = np.empty(len(self._pool_matrices))
        for i, a in enumerate(self._pool_matrices):
            grads[i] = 2.0 * np.real(np.vdot(h_psi, a @ psi))
        return grads

    def _operator_matrix(self, op: PoolOperator):
        """``op``'s generator in this run's representation.

        A pool operator caches its own dense matrix, but the sparse and sector
        paths never build one, so the representation has to follow the run
        rather than the operator.  Used for operators the growth step considers
        outside the pool (:meth:`CEOPool.grown_operators`).
        """
        if getattr(self, "_matrix_free", False):
            return PauliOperator(op.generator, sector=self._sector,
                                 cache_bytes=0)
        if self._sector is not None:
            return self._sector.restrict(op.generator)
        if getattr(self, "_sparse", False):
            return op.generator.to_sparse_matrix()
        return op.matrix()

    def _resolve_taper(self, qubit_h):
        """The :class:`TaperedRegister` this run uses, or ``None``.

        The particle-number sector is **not** available alongside it: a tapered
        register has no particle-number basis to enumerate, because the Clifford
        mixed the occupation bits into parities.  That costs nothing, since the
        taper already made the register smaller by the same symmetries the sector
        was exploiting.
        """
        if not getattr(self, "taper", False):
            return None
        from ..core.mapping import reference_qubit_bits
        from ..core.tapering import taper_problem

        bits = reference_qubit_bits("jordan_wigner", qubit_h.num_qubits,
                                    self.pool.occupied_orbitals)
        info = taper_problem(qubit_h,
                             [op.generator for op in self._pool_ops], bits)
        if info is None:
            warnings.warn(
                "taper=True found no Z2 symmetry in this Hamiltonian, so the "
                "register is unchanged; the run continues untapered",
                RuntimeWarning, stacklevel=3)
            return None
        # Only an error when the taper is what emptied the pool: a pool that was
        # already empty (one electron in one orbital, e.g. an isolated atom in
        # SZ) has nothing to correlate, and the untapered run accepts it too.
        if not info.generators and info.dropped:
            raise ValueError(
                f"tapering left the operator pool empty: all "
                f"{len(info.dropped)} generators change one of the "
                f"{len(info.symmetries)} Z2 symmetries, so none of them can act "
                f"within the sector the reference determinant fixes.  An ansatz "
                f"with no operators cannot correlate anything.  Use taper=False, "
                f"or a pool whose excitations conserve these symmetries.")
        return info

    def _tapered_pool_operator(self, op: PoolOperator,
                               generator) -> PoolOperator:
        """``op`` on the tapered register, with ``generator`` as its generator.

        The operators a coupled-exchange operator was built from travel with it
        in ``members``, and the growth step evaluates and appends *them* when it
        expands the selected operator.  They have to be reduced by the same
        Clifford as the operator itself: left on the full register they meet a
        state of the tapered width in :meth:`_operator_gradient`, and if that
        were patched they would put untapered generators into the ansatz.  A
        member cannot leak when its parent does not -- under Jordan-Wigner, which
        tapering requires, a qubit excitation puts an X or Y on every qubit it
        acts on, so it meets a Z-type symmetry the same way whichever
        excitation of the set it is -- so this never drops one.
        """
        members = tuple(
            self._tapered_pool_operator(
                member, self._taper_info.taper_operator(member.generator))
            for member in op.members)
        return replace(op, generator=generator, members=members, _matrix=None,
                       support=_support_of(generator))

    def _resolve_spin_conservation(self, hamiltonian) -> bool:
        """Whether the problem conserves :math:`S_z`, read off the Hamiltonian.

        A spin-orbit Hamiltonian mixes the two spin blocks, so only the total
        electron number is a good quantum number: the pool must then change
        :math:`S_z` too (``pool="spin-orbit"``), and the sector is the total-N
        one.  A qubit Hamiltonian carries no spin layout, so the pool decides.
        """
        from ..core.mapping import Fermion
        from ..core.spin_orbit import conserves_spin_projection
        if not isinstance(hamiltonian, Fermion):
            return bool(self.pool.conserves_spin_projection)
        conserving = conserves_spin_projection(hamiltonian)
        if not conserving and not self._supports_spin_orbit:
            self._refuse_spin_orbit()
        if not conserving and self.pool.conserves_spin_projection:
            raise ValueError(
                f"the Hamiltonian couples the two spin blocks (spin-orbit), "
                f"but the {self.pool.name!r} pool conserves S_z and cannot "
                f"reach its ground state; use pool='spin-orbit'")
        if not conserving and self.mapping == "parity_reduced":
            raise ValueError(
                "the parity_reduced mapping removes the two spin-parity "
                "qubits, which a spin-orbit Hamiltonian does not conserve; "
                "use 'jordan_wigner', 'parity' or 'bravyi_kitaev'")
        return conserving

    def _new_ansatz(self) -> AdaptAnsatz:
        """A fresh growable ansatz on the configured evaluation backend.

        Routes through :meth:`~mandacaru.algorithms.base.VariationalDriver.circuit_provider`,
        so the ansatz evaluates its states either with the internal state-vector
        backend or by executing circuits on Qiskit / Braket / Cirq.  With
        ``shots > 0`` it is a measured ansatz (``simulate=False``): the
        operators and the reference determinant a circuit is built from, and
        no state.
        """
        info = getattr(self, "_taper_info", None)
        if info is not None:
            # On a tapered register the reference is still a computational basis
            # state -- the Clifford maps |HF> to |rest> times an X eigenstate of
            # the anchor, so deleting the anchor bit is exactly the tapered
            # reference.  Its 1-positions are handed over as the "occupied" list
            # and the encoding is the identity, which is what Jordan-Wigner is.
            return AdaptAnsatz(self.n_qubits, info.occupied, "jordan_wigner",
                               sparse=getattr(self, "_sparse", False),
                               matrix_free=getattr(self, "_matrix_free", False),
                               provider=self.ansatz_provider(),
                               num_particles=None, sector=None,
                               simulate=not self.shots)
        return AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                           self.mapping, sparse=getattr(self, "_sparse", False),
                           matrix_free=getattr(self, "_matrix_free", False),
                           provider=self.ansatz_provider(),
                           num_particles=self.num_particles,
                           sector=self._sector, simulate=not self.shots)

    def _profile(self, ansatz) -> CircuitMetrics:
        """Compiled-circuit metrics for ``ansatz`` on the configured provider."""
        if not self.profile:
            return CircuitMetrics(None, None, ansatz.num_parameters)
        from ..backends.providers import build_provider
        provider = (None if self.backend_provider == "qiskit"
                    else build_provider(self.backend_provider))
        # The reference *qubits* (not the spin-orbital occupations): they
        # differ under the parity / Bravyi-Kitaev maps and the tapered register.
        return profile_ansatz(self.n_qubits, ansatz.reference_qubits(),
                              ansatz.operators, provider=provider)

    def reference_energy(self) -> float:
        return self.reference_energy_of(self._new_ansatz())

    # -- the run log ------------------------------------------------------ #

    def _log_title(self) -> str:
        """The method and the pool it screens, e.g. ``ADAPT-VQE (QEBPool)``."""
        spec = self._pool_spec
        pool = (type(spec) if isinstance(spec, PoolBase)
                else pool_class(spec)).__name__
        return f"{self.log_title} ({pool})"

    def _run_kwargs(self, atoms) -> dict:
        """Forward the geometry to :meth:`run` for the ``output.txt`` metadata."""
        return {"geometry": atoms, **self.run_options}
