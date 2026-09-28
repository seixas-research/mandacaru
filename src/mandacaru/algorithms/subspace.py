# -*- coding: utf-8 -*-
# file: algorithms/subspace.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Subspace-search VQE and ADAPT-VQE for simultaneous ground + excited states.

Where variational quantum deflation (:mod:`mandacaru.algorithms.deflation`)
finds excited states *one after another*, **subspace-search VQE** (SSVQE,
Nakanishi *et al.* 2019) finds the ground state and the first few excited states
**at once**, in a single optimization.

The idea: pick :math:`k` mutually orthogonal reference determinants
:math:`\{|\varphi_j\rangle\}`, send them all through the **same** parameterized
unitary :math:`U(\vec\theta)`, and minimize the *weighted* energy sum

.. math::

    L(\vec\theta) = \sum_{j=0}^{k-1} w_j\,
        \langle\varphi_j|U^\dagger(\vec\theta)\,H\,U(\vec\theta)|\varphi_j\rangle ,
    \qquad w_0 > w_1 > \dots > w_{k-1} > 0 .

Because :math:`U` is unitary the images :math:`U|\varphi_j\rangle` stay
orthonormal, and the descending weights force the largest weight onto the lowest
energy: at the optimum :math:`U|\varphi_0\rangle` is the ground state,
:math:`U|\varphi_1\rangle` the first excited state, and so on.  Each level's
reported energy is the *bare* expectation value
:math:`\langle\varphi_j|U^\dagger H U|\varphi_j\rangle`.

:class:`SubspaceVQE` uses a fixed ansatz; :class:`SubspaceADAPTVQE` grows one
shared adaptive ansatz whose pool-screening gradient is the weighted sum of the
per-reference gradients.  Both subclass the ground-state drivers and are ASE
calculators (the returned ASE energy is the ground state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


from ..circuits import CircuitMetrics
from ..core.mapping import Fermion, reference_qubit_bits
from .adapt_vqe import ADAPTVQE, _max_abs
from .ansatz_spec import resolve_ansatz
from .deflation import EnergyLevels
from .vqe import VQE


# --------------------------------------------------------------------------- #
# Reference determinants for the search subspace.
# --------------------------------------------------------------------------- #

def subspace_determinants(n_qubits: int, occupied, num_states: int) -> list[tuple]:
    """The ``num_states`` lowest determinants of the Hartree-Fock particle sector.

    Enumerates Slater determinants with the same ``(n_alpha, n_beta)`` occupation
    as the Hartree-Fock reference ``occupied`` (spin-blocked spin-orbital
    indices), ordered by excitation level from HF then lexicographically, and
    returns the first ``num_states`` as tuples of occupied spin-orbital indices.
    These are mutually orthogonal and share the electron count, so the
    number-conserving ansatz keeps them in the physical sector.
    """
    M = n_qubits // 2
    occ = sorted(int(o) for o in occupied)
    n_alpha = sum(1 for o in occ if o < M)
    n_beta = len(occ) - n_alpha
    alpha_orbitals = range(M)
    beta_orbitals = range(M, n_qubits)
    hf = frozenset(occ)

    dets: list[tuple[int, tuple]] = []
    for a_sel in combinations(alpha_orbitals, n_alpha):
        for b_sel in combinations(beta_orbitals, n_beta):
            det = frozenset(a_sel) | frozenset(b_sel)
            level = len(det - hf)                 # number of promoted electrons
            dets.append((level, tuple(sorted(det))))
    dets.sort(key=lambda t: (t[0], t[1]))
    if num_states > len(dets):
        raise ValueError(
            f"requested {num_states} states but the particle-number sector only "
            f"has {len(dets)} determinants")
    return [d[1] for d in dets[:num_states]]


def _determinant_vector(mapping: str, n_qubits: int, occupied) -> np.ndarray:
    """Computational-basis state vector of a determinant (qubit 0 = MSB)."""
    bits = reference_qubit_bits(mapping, n_qubits, occupied)
    index = 0
    for i, bit in enumerate(bits):
        if bit:
            index |= 1 << (n_qubits - 1 - i)
    vec = np.zeros(2 ** n_qubits, dtype=complex)
    vec[index] = 1.0
    return vec


def reference_matrix(mapping: str, n_qubits: int, occupied,
                     num_states: int) -> np.ndarray:
    """Stack the ``num_states`` lowest determinants as columns ``(2**n, k)``."""
    dets = subspace_determinants(n_qubits, occupied, num_states)
    cols = [_determinant_vector(mapping, n_qubits, det) for det in dets]
    return np.column_stack(cols)


def spin_squared(n_spatial_orbitals: int) -> Fermion:
    r""":math:`S^2 = S_- S_+ + S_z^2 + S_z` on a spin-blocked register.

    Spin-orbital ``p`` is alpha for ``p < M`` and ``p + M`` its beta partner,
    so :math:`S_+ = \sum_p a^\dagger_{p\alpha} a_{p\beta}`.
    """
    M = int(n_spatial_orbitals)
    n = 2 * M
    raise_spin = Fermion({((p, True), (p + M, False)): 1.0 for p in range(M)},
                         n_modes=n)
    lower_spin = Fermion({((p + M, True), (p, False)): 1.0 for p in range(M)},
                         n_modes=n)
    s_z = Fermion({((p, True), (p, False)): (0.5 if p < M else -0.5)
                   for p in range(n)}, n_modes=n)
    return lower_spin * raise_spin + s_z * s_z + s_z


def spin_adapted_references(mapping: str, n_qubits: int, num_particles,
                            num_states: int, multiplicity: int):
    r"""``num_states`` orthonormal references of spin ``multiplicity``.

    A determinant with open shells mixes spins: the singly excited
    :math:`|i_\alpha a_\beta\rangle` is half singlet, half triplet.  An ansatz
    that conserves :math:`S^2` -- the Hamiltonian variational ansatz, whose
    generators are the spin-free Hamiltonian's own parts -- keeps that mixture
    for good, and a subspace search from such references converges to the
    *average* of the two levels (H2: 133 mHa off both).  These references are
    :math:`S^2` eigenvectors instead.

    The determinants of the ``(n_alpha, n_beta)`` sector are grouped by
    orbital configuration -- the doubly, singly and unoccupied spatial
    orbitals -- and :math:`S^2` is diagonalized within each; every group is
    closed under :math:`S^2`, and its determinants share the configuration's
    spatial symmetry, which the references therefore keep.  Configurations
    are taken by excitation level from Hartree-Fock, then lowest orbitals
    first, and the eigenvectors with :math:`S(S+1)` of the requested
    multiplicity kept until ``num_states`` are found.

    Returns ``(references, configurations)``: a ``(2**n, num_states)`` matrix
    and each column's configuration (occupation of every spatial orbital).
    """
    from ..core.sector import ParticleSector

    n_alpha, n_beta = (int(v) for v in num_particles)
    multiplicity = int(multiplicity)
    spin = (multiplicity - 1) / 2
    projection = abs(n_alpha - n_beta) / 2
    if multiplicity < 1 or spin < projection \
            or (multiplicity - 1) % 2 != abs(n_alpha - n_beta) % 2:
        raise ValueError(
            f"multiplicity {multiplicity} is not reachable with "
            f"(n_alpha, n_beta) = ({n_alpha}, {n_beta}): it needs S >= "
            f"|S_z| = {projection:g} and the same parity of 2S")
    sector = ParticleSector(n_qubits, (n_alpha, n_beta), mapping)
    M = n_qubits // 2
    s2 = sector.restrict(spin_squared(M).map_to_qubits(mapping,
                                                       n_modes=n_qubits))
    spatial = sector.occupations[:, :M] + sector.occupations[:, M:]
    hartree_fock = np.array([(p < n_alpha) + (p < n_beta) for p in range(M)])
    groups: dict[tuple, list[int]] = {}
    for position, pattern in enumerate(map(tuple, spatial)):
        groups.setdefault(pattern, []).append(position)
    orbital = np.arange(M)

    def order(pattern):
        occupation = np.asarray(pattern)
        promoted = int(np.maximum(hartree_fock - occupation, 0).sum())
        return promoted, int(occupation @ orbital), pattern

    target = spin * (spin + 1)
    columns, configurations = [], []
    for pattern in sorted(groups, key=order):
        positions = groups[pattern]
        block = s2[positions][:, positions].toarray()
        values, vectors = np.linalg.eigh(0.5 * (block + block.conj().T))
        for value, vector in zip(values, vectors.T):
            if abs(value - target) > 1e-8:
                continue
            amplitudes = np.zeros(sector.dim, dtype=complex)
            amplitudes[positions] = vector
            columns.append(sector.embed(amplitudes))
            configurations.append(pattern)
            if len(columns) == num_states:
                return np.column_stack(columns), configurations
    raise ValueError(
        f"the ({n_alpha}, {n_beta}) sector holds only {len(columns)} states "
        f"of multiplicity {multiplicity}; ask for fewer")


def resolve_weights(weights, num_states: int) -> np.ndarray:
    """Validate / default the SSVQE weights (strictly decreasing, positive)."""
    if weights is None:
        w = np.arange(num_states, 0, -1, dtype=float)     # k, k-1, ..., 1
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != num_states:
            raise ValueError(
                f"expected {num_states} weights, got {w.size}")
        if np.any(w <= 0):
            raise ValueError("weights must be positive")
        if num_states > 1 and np.any(np.diff(w) >= 0):
            raise ValueError("weights must be strictly decreasing")
    return w


# --------------------------------------------------------------------------- #
# Result containers.
# --------------------------------------------------------------------------- #

class _SpectrumViews:
    """What both subspace results derive from ``energies`` / ``states``."""

    @property
    def optimal_energy(self) -> float:
        """Ground-state energy (lowest level) -- the ASE-facing energy."""
        return float(self.energies[0])

    @property
    def num_states(self) -> int:
        return int(len(self.energies))

    @property
    def excitation_energies(self) -> np.ndarray:
        return np.asarray(self.energies, float) - float(self.energies[0])

    @property
    def levels(self) -> EnergyLevels:
        """View as an :class:`~mandacaru.algorithms.EnergyLevels`."""
        return EnergyLevels(energies=np.asarray(self.energies, float),
                            states=list(self.states),
                            reference_energy=self.reference_energy,
                            num_evaluations=self.num_evaluations,
                            energy_unit=self.energy_unit)

    def in_units(self, units: str = "eV") -> np.ndarray:
        return self.levels.in_units(units)


@dataclass
class SubspaceVQEResult(_SpectrumViews):
    """Result of a :class:`SubspaceVQE` run (ground + excited states).

    Every energy is in :attr:`energy_unit` -- **eV** by default, Hartree when
    the driver was built with ``atomic_units=True``; :meth:`in_units` converts.
    """

    energies: np.ndarray                 # per-level energy, ascending
    optimal_parameters: np.ndarray       # shared ansatz parameters
    weights: np.ndarray                  # SSVQE weights used
    states: list = field(default_factory=list)      # optimal state vectors
    reference_energy: float | None = None
    num_evaluations: int = 0
    success: bool = True
    timings: dict | None = None
    integration_profile: dict | None = None
    energy_unit: str = "eV"              # unit of every energy above

    def __repr__(self) -> str:
        levels = ", ".join(f"{e:.6f}" for e in np.asarray(self.energies))
        return (f"SubspaceVQEResult([{levels}] {self.energy_unit}, "
                f"num_states={self.num_states}, success={self.success})")


@dataclass
class SubspaceADAPTVQEResult(_SpectrumViews):
    """Result of a :class:`SubspaceADAPTVQE` run (ground + excited states).

    Every energy is in :attr:`energy_unit` -- **eV** by default, Hartree when
    the driver was built with ``atomic_units=True``; :meth:`in_units` converts.
    """

    energies: np.ndarray                 # per-level energy, ascending
    optimal_parameters: np.ndarray
    weights: np.ndarray
    converged: bool
    final_max_gradient: float
    operators: list = field(default_factory=list)   # selected operator labels
    states: list = field(default_factory=list)
    reference_energy: float | None = None
    num_evaluations: int = 0
    metrics: CircuitMetrics | None = None
    timings: dict | None = None
    integration_profile: dict | None = None
    energy_unit: str = "eV"              # unit of every energy above

    @property
    def num_operators(self) -> int:
        return len(self.operators)

    def __repr__(self) -> str:
        levels = ", ".join(f"{e:.6f}" for e in np.asarray(self.energies))
        return (f"SubspaceADAPTVQEResult([{levels}] {self.energy_unit}, "
                f"num_states={self.num_states}, n_ops={self.num_operators}, "
                f"converged={self.converged})")


# --------------------------------------------------------------------------- #
# Shared subspace-search machinery.
# --------------------------------------------------------------------------- #

class SubspaceMixin:
    #: The extra reference determinants are built untapered.
    _supports_parity_reduced = False
    #: The reference determinants are spin-blocked; not validated with a
    #: Hamiltonian that breaks S_z.
    _supports_spin_orbit = False
    #: ... and on the full register, so no particle-number sector either.
    _supports_sector = False
    """Shared SSVQE scaffolding: one unitary over several orthogonal references.

    Owns the outer :meth:`run` (weights, reference determinants, timings, banner,
    sort, result assembly); as an ASE calculator the full spectrum is stored on
    :attr:`result`.  A concrete driver supplies the parts that differ:

    * :meth:`_reference_occupied` -- the reference determinant's occupied orbitals;
    * :meth:`_subspace_optimize` -- the actual optimization (fixed ansatz or grow);
    * :meth:`_make_subspace_result` -- the driver's result dataclass;
    * :meth:`_emit_run_header` / :meth:`_print_subspace_summary` -- verbose output.
    """

    def _init_subspace(self, num_states: int, weights) -> None:
        if int(num_states) < 1:
            raise ValueError("num_states must be >= 1")
        self.num_states = int(num_states)
        self._weights_spec = weights

    # -- references ------------------------------------------------------- #

    def _references(self) -> np.ndarray:
        return reference_matrix(self.mapping, self.n_qubits,
                                self._reference_occupied(), self.num_states)

    def _reference_determinants(self) -> list[tuple] | None:
        """Each reference's determinant, ``None`` when they are not single
        determinants (spin-adapted references)."""
        return subspace_determinants(
            self.n_qubits, self._reference_occupied(), self.num_states)

    # -- shared outer loop ----------------------------------------------- #

    # The subspace ``run()`` below replaces the common one and does not go
    # through `_make_logger`, `_write_checkpoint` or `_load_resume`, so those
    # options would be accepted and silently ignored.  `Mandacaru` refuses them
    # rather than letting a run report nothing at all.
    writes_output_log = False
    supports_checkpoints = False

    def run(self, initial_parameters=None, **_ignored):
        """Optimize the shared unitary and return the ``num_states`` levels."""
        if self.dry_run:
            return self._dry_run_estimate()
        if not self._configured:
            raise RuntimeError(
                f"{type(self).__name__} has no Hamiltonian; construct it with one, "
                "or use it as an ASE calculator with a `basis`")
        self._check_kpts()

        weights = resolve_weights(self._weights_spec, self.num_states)
        refs = self._references()
        timings, run_t0 = self._make_timings()
        ref_energy = self.reference_energy()

        if self.verbose:
            self._show_banner()
            self._emit_run_header(ref_energy)
            print(f"Subspace search: {self.num_states} states, weights = "
                  f"{np.array2string(weights, precision=3)}")

        energies, params, states, extra = self._subspace_optimize(
            refs, weights, initial_parameters, timings)

        order = np.argsort(energies)
        energies = np.asarray(energies, dtype=float)[order]
        states = [states[i] for i in order]
        # State j is U|phi_j>: sorting the levels must carry the references
        # along, or exporting "the ground state" would prepare U|HF> whichever
        # reference the lowest level actually grew from.
        determinants = self._reference_determinants()
        self.state_determinants = (None if determinants is None
                                   else [determinants[i] for i in order])

        self._finalize_timings(timings, run_t0)
        # The single Hartree -> output-unit boundary of the subspace search.
        result = self._make_subspace_result(
            self._to_energy_units(energies), params, weights, states,
            self._to_energy_units(ref_energy), timings, extra)
        if self.verbose:
            self._print_subspace_summary(result, timings)
        return result

    # -- export ------------------------------------------------------------ #

    def ansatz_problem(self, theta=None, state: int = 0):
        """``(n_qubits, occupied, generators, theta, hamiltonian)`` of level
        ``state`` (0 = the ground state, in the result's ascending order).

        The shared unitary is the same for every level; what distinguishes them
        is the reference determinant it acts on, which is the one the *sorted*
        level grew from -- not necessarily Hartree-Fock.
        """
        n_qubits, _hf, generators, theta, hamiltonian = super().ansatz_problem(
            theta)
        determinants = getattr(self, "state_determinants", None)
        if determinants is None:
            raise RuntimeError("run the solver before exporting a state")
        if not 0 <= int(state) < len(determinants):
            raise IndexError(f"state {state} of {len(determinants)} levels")
        bits = reference_qubit_bits(self.mapping, n_qubits,
                                    determinants[int(state)])
        occupied = [k for k, bit in enumerate(bits) if bit]
        return n_qubits, occupied, generators, theta, hamiltonian

    def measured_energy(self, provider, theta=None, state: int = 0) -> float:
        """``<H>`` of level ``state`` evaluated on ``provider`` (output units)."""
        return self._to_energy_units(
            provider.energy(*self.ansatz_problem(theta, state)))

    # -- driver hooks ----------------------------------------------------- #

    def _reference_occupied(self):
        raise NotImplementedError

    def _subspace_optimize(self, refs, weights, initial_parameters, timings):
        """Return ``(energies, params, states, extra)`` (energies/states unsorted)."""
        raise NotImplementedError

    def _make_subspace_result(self, energies, params, weights, states,
                              ref_energy, timings, extra):
        """Build the result; ``energies`` / ``ref_energy`` are already in output units."""
        raise NotImplementedError

    def _emit_run_header(self, ref_energy) -> None:
        raise NotImplementedError

    def _print_subspace_summary(self, result, timings) -> None:
        raise NotImplementedError


def _check_hva_subspace(spec, multiplicity, options) -> None:
    """What a subspace search with the HVA needs, checked at construction."""
    from .hartree_fock import UHFResult

    if multiplicity is None:
        raise ValueError(
            "the HVA conserves S^2, so a subspace search needs spin-adapted "
            "references: name their spin with multiplicity= (1 singlets, 3 "
            "triplets, ...).  From plain determinants an open-shell level "
            "would converge to an average of a singlet and a triplet")
    if options.get("taper"):
        raise ValueError("the subspace references are built on the full "
                         "register; the HVA cannot taper it here")
    reference = spec.options.get("reference")
    if reference is not None and isinstance(reference.scf, UHFResult):
        raise ValueError("a UHF determinant breaks S^2, so it cannot seed "
                         "spin-adapted references; use an RHF reference")


# --------------------------------------------------------------------------- #
# Subspace-search VQE (fixed ansatz).
# --------------------------------------------------------------------------- #

class SubspaceVQE(SubspaceMixin, VQE):
    """Subspace-search VQE: ground + first excited states in one optimization.

    Extends :class:`~mandacaru.algorithms.VQE`; every constructor argument of
    ``VQE`` is accepted (including the ASE-calculator ``basis`` / ``h`` mode),
    plus:

    Parameters
    ----------
    num_states : int
        Number of levels (ground + excited) to compute simultaneously
        (default ``2``).
    weights : sequence of float, optional
        Strictly decreasing positive SSVQE weights, one per level.  Defaults to
        ``(k, k-1, ..., 1)``.
    multiplicity : int, optional
        ``2S + 1`` of the levels to find, required with ``ansatz="hva"`` and
        refused otherwise: the references are then spin-adapted
        (:func:`spin_adapted_references`) and the result is the lowest
        ``num_states`` levels of that spin.

    Notes
    -----
    With UCCSD the references are the ``num_states`` lowest determinants of
    the Hartree-Fock particle-number sector.  The ansatz must supply
    ``evolve(theta, references)`` so one shared unitary acts on all of them.

    The HVA conserves :math:`S^2` and the spatial symmetry of the orbitals, so
    each spin-adapted reference reaches only the states of its own spin and
    spatial symmetry.  The levels are the lowest of the symmetry sectors the
    references span -- the lowest ``num_states`` of the spin whenever those
    sectors hold them, which the configuration order makes likely but a
    symmetry the references miss can break.  Every generator must commute
    with :math:`S^2` -- the default ``body_order`` grouping with
    ``evolution="exact"`` -- or the references would leak into other spins;
    that is checked before the search.  As an ASE calculator
    the reported energy is the ground state; the full spectrum is on
    :attr:`result`.
    """

    citation_method = "subspace-vqe"

    def __init__(self, hamiltonian=None, ansatz=None, *, num_states: int = 2,
                 weights=None, multiplicity: int | None = None, **kwargs):
        spec = resolve_ansatz(ansatz)
        if spec is not None and spec.name == "hva":
            _check_hva_subspace(spec, multiplicity, kwargs)
        elif multiplicity is not None:
            raise ValueError(
                "multiplicity= builds spin-adapted references for an ansatz "
                "that conserves S^2 (ansatz='hva'); UCCSD's generators mix "
                "spins, so its references are determinants")
        #: ``2S + 1`` of the spin-adapted references, ``None`` for
        #: determinants.
        self.multiplicity = None if multiplicity is None else int(multiplicity)
        self._init_subspace(num_states, weights)
        super().__init__(hamiltonian, ansatz, **kwargs)

    def _references(self) -> np.ndarray:
        if self.multiplicity is None:
            return super()._references()
        self._check_spin_conserving_ansatz()
        references, self.reference_configurations = spin_adapted_references(
            self.mapping, self.n_qubits, self.num_particles, self.num_states,
            self.multiplicity)
        return references

    def _reference_determinants(self) -> list[tuple] | None:
        if self.multiplicity is not None:
            return None
        return super()._reference_determinants()

    def _check_spin_conserving_ansatz(self) -> None:
        """Refuse an ansatz with a generator that does not commute with S^2.

        Spin-adapted references only help an ansatz that keeps them pure.
        The HVA's default groups -- the whole one-body and two-body parts --
        are spin-free, but the ``spin_resolved`` groups are not (only their
        sum is), and neither is a product formula's single Pauli rotation.
        With such a generator the states leak into other spins, and on H4 a
        "singlet" level fell 0.2 Ha below the true second singlet, onto the
        triplet.  Checked on the generators themselves, in the Pauli algebra.
        """
        s2 = spin_squared(self.n_qubits // 2).map_to_qubits(
            self.mapping, n_modes=self.n_qubits)
        for index, generator in enumerate(self.ansatz.pauli_generators):
            if generator.commutator(s2).simplify(1e-10).terms:
                raise ValueError(
                    f"generator {index + 1} of the ansatz does not commute "
                    f"with S^2, so it would mix the spin-adapted references "
                    f"with other spins; use the HVA's default body_order "
                    f"grouping with evolution='exact'")

    def _reference_occupied(self):
        occupied = getattr(self.ansatz, "occupied", None)
        if occupied is None:
            occupied = getattr(self.ansatz, "_occupied", None)
        if occupied is None:
            raise TypeError(
                "the ansatz does not expose its occupied orbitals; SubspaceVQE "
                "needs a determinant reference (e.g. a UCCSD ansatz)")
        return occupied

    def _evolve(self, theta, references) -> np.ndarray:
        evolve = getattr(self.ansatz, "evolve", None)
        if evolve is None:
            raise TypeError(
                f"ansatz {type(self.ansatz).__name__} has no evolve(theta, "
                "references); SubspaceVQE needs it to share one unitary across "
                "the reference states")
        return evolve(theta, references)

    def _emit_run_header(self, ref_energy) -> None:
        self._print_header(ref_energy)

    def _subspace_optimize(self, refs, weights, initial_parameters, timings):
        k = self.num_states
        n = self.ansatz.num_parameters
        x0 = (self._default_parameters() if initial_parameters is None
              else np.asarray(initial_parameters, dtype=float).ravel())
        if x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {x0.size}")

        def weighted_cost(theta):
            evolved = self._evolve(theta, refs)
            return sum(weights[j] * self.energy(evolved[:, j]) for j in range(k))

        with timings.time("parameter optimization"):
            result = self._optimize_all(weighted_cost, x0)

        evolved = self._evolve(result.x, refs)
        energies = [self.energy(evolved[:, j]) for j in range(k)]
        states = [evolved[:, j].copy() for j in range(k)]
        extra = {"num_evaluations": result.nfev, "success": result.success}
        return energies, np.asarray(result.x, float), states, extra

    def _make_subspace_result(self, energies, params, weights, states,
                              ref_energy, timings, extra) -> SubspaceVQEResult:
        return SubspaceVQEResult(
            energies=energies, optimal_parameters=params, weights=weights,
            states=states, reference_energy=ref_energy,
            num_evaluations=extra["num_evaluations"], success=extra["success"],
            timings=timings.as_dict(),
            integration_profile=self._integration_profile,
            energy_unit=self._energy_unit_label())

    def _print_subspace_summary(self, result: SubspaceVQEResult, timings) -> None:
        rule = "=" * 70
        print(rule)
        status = "converged" if result.success else "did not converge"
        print(f"Subspace-VQE finished ({status}): {result.num_states} levels")
        for i, e in enumerate(result.energies):
            tag = "ground" if i == 0 else f"excited {i}"
            print(f"  E[{i}] ({tag:>9s}) = {e:+.8f} {result.energy_unit}")
        if timings is not None:
            print(timings.format_report())
        print(rule)


# --------------------------------------------------------------------------- #
# Subspace-search ADAPT-VQE (one shared, adaptively grown ansatz).
# --------------------------------------------------------------------------- #

class SubspaceADAPTVQE(SubspaceMixin, ADAPTVQE):
    """Subspace-search ADAPT-VQE: grow one shared ansatz for several states.

    Extends :class:`~mandacaru.algorithms.ADAPTVQE`; accepts every ``ADAPTVQE``
    argument plus ``num_states`` / ``weights`` (as :class:`SubspaceVQE`).  A
    single adaptive ansatz :math:`U(\\vec\\theta)` is grown and applied to all
    reference determinants; the pool-screening gradient is the **weighted sum**
    of the per-reference gradients
    :math:`\\sum_j w_j\\,\\langle\\psi_j|[H, A_i]|\\psi_j\\rangle`, and the inner
    re-optimization minimizes the weighted energy.

    As an ASE calculator the reported energy is the ground state; the spectrum is
    on :attr:`result`.
    """

    citation_method = "subspace-adapt-vqe"

    def __init__(self, hamiltonian=None, pool="fermionic", *,
                 num_states: int = 2, weights=None, **kwargs):
        self._init_subspace(num_states, weights)
        super().__init__(hamiltonian, pool, **kwargs)

    def _reference_occupied(self):
        return self.pool.occupied_orbitals

    def _emit_run_header(self, ref_energy) -> None:
        self._print_header(ref_energy, self._energy_unit_label())

    def _weighted_gradients(self, evolved: np.ndarray,
                            weights: np.ndarray) -> np.ndarray:
        """Weighted-sum pool gradient over the reference states (columns)."""
        grads = np.zeros(len(self._pool_matrices))
        for j in range(evolved.shape[1]):
            grads += weights[j] * self._analytic_gradients(evolved[:, j])
        return grads

    def _subspace_optimize(self, refs, weights, initial_parameters, timings):
        k = self.num_states
        max_iterations = self.max_iterations
        convergence = self.convergence

        ansatz = self._new_ansatz()
        params = (np.asarray(initial_parameters, float).ravel()
                  if initial_parameters is not None else np.zeros(0))
        selected: list[str] = []
        total_evals = 0
        converged = False
        max_grad = np.inf
        energy = min(self.energy(ansatz.evolve(params, refs)[:, j])
                     for j in range(k))
        # The energy change of the last growth step (the trace's dE, on the
        # lowest level), for the energy criterion; None before the first.
        delta_energy: float | None = None
        if self.verbose:
            self._print_iteration_heading(self._energy_unit_label())

        for _ in range(max_iterations):
            with timings.time("gradient screening"):
                evolved = ansatz.evolve(params, refs)
                grads = self._weighted_gradients(evolved, weights)
            max_grad = _max_abs(grads)
            if convergence.reached(max_grad, delta_energy):
                converged = True
                break
            idx = self._select_operator(grads, len(selected))

            def weighted_gradient(op, _evolved=evolved, _weights=weights):
                """One operator's gradient of the weighted subspace energy."""
                return sum(_weights[j] * self._operator_gradient(op, _evolved[:, j])
                           for j in range(_evolved.shape[1]))

            # `op` is the selection the trace names; `n_new` is how many
            # operators it grew into (one, except for an MVP-CEO).
            op = self._pool_ops[idx]
            n_new = self._grow(ansatz, selected, op, weighted_gradient)

            def weighted_cost(theta, _ansatz=ansatz):
                ev = _ansatz.evolve(theta, refs)
                return sum(weights[j] * self.energy(ev[:, j]) for j in range(k))

            previous_energy = energy
            with timings.time("parameter optimization"):
                result = self._optimize_grown(weighted_cost, params,
                                              n_new=n_new)
            params = np.asarray(result.x, float)
            total_evals += result.nfev
            energy = min(self.energy(ansatz.evolve(params, refs)[:, j])
                         for j in range(k))
            delta_energy = energy - previous_energy

            if self.verbose:
                self._print_iteration(len(selected), op, max_grad, energy,
                                      energy - previous_energy,
                                      CircuitMetrics(None, None,
                                                     ansatz.num_parameters),
                                      self._energy_unit_label())

        # Final per-level energies (bare expectation values).
        evolved = ansatz.evolve(params, refs)
        energies = [self.energy(evolved[:, j]) for j in range(k)]
        states = [evolved[:, j].copy() for j in range(k)]

        metrics = self._profile(ansatz)

        if not converged:
            # The loop stopped on the operator budget (or ran none at all), so
            # the reported gradient -- and the verdict drawn from it -- must
            # describe the *final* state, exactly as ordinary ADAPT's run does.
            # Otherwise a stationary start with a zero growth budget reports a
            # gradient of 0.0 against a tolerance of 1e3 and `converged=False`.
            max_grad = _max_abs(self._weighted_gradients(evolved, weights))
            converged = convergence.reached(max_grad, delta_energy)

        extra = {"converged": converged, "final_max_gradient": max_grad,
                 "operators": selected, "metrics": metrics,
                 "num_evaluations": total_evals}
        self.ansatz = ansatz          # kept, like ADAPTVQE.run, for export
        return energies, params, states, extra

    def _make_subspace_result(self, energies, params, weights, states,
                              ref_energy, timings,
                              extra) -> SubspaceADAPTVQEResult:
        return SubspaceADAPTVQEResult(
            energies=energies, optimal_parameters=params, weights=weights,
            converged=extra["converged"],
            final_max_gradient=extra["final_max_gradient"],
            operators=extra["operators"], states=states,
            reference_energy=ref_energy,
            num_evaluations=extra["num_evaluations"], metrics=extra["metrics"],
            timings=timings.as_dict(),
            integration_profile=self._integration_profile,
            energy_unit=self._energy_unit_label())

    def _print_subspace_summary(self, result: SubspaceADAPTVQEResult,
                                timings) -> None:
        rule = "=" * 70
        print(rule)
        status = "converged" if result.converged else "not converged"
        print(f"Subspace-ADAPT-VQE finished ({status}): {result.num_states} "
              f"levels, {result.num_operators} operators, "
              f"final |grad| = {result.final_max_gradient:.6e}")
        for i, e in enumerate(result.energies):
            tag = "ground" if i == 0 else f"excited {i}"
            print(f"  E[{i}] ({tag:>9s}) = {e:+.8f} {result.energy_unit}")
        if timings is not None:
            print(timings.format_report())
        print(rule)
