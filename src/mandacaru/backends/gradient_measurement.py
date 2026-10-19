# -*- coding: utf-8 -*-
# file: backends/gradient_measurement.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""ADAPT-VQE's pool gradients, measured on the device.

ADAPT-VQE picks each operator by the energy gradient of appending
:math:`e^{\theta A_k}` to the current state.  For an anti-Hermitian generator
the derivative at :math:`\theta = 0` is an expectation value,

.. math::

    g_k = \langle\psi|[H, A_k]|\psi\rangle
        = \sum_R c_{kR}\,\langle\psi|R|\psi\rangle ,

of the Hermitian commutator observable :math:`G_k = [H, A_k]`, whose Pauli
strings :math:`R = P Q` and real coefficients come from the Pauli algebra
alone (:func:`~mandacaru.core.pauli_algebra.commutator_table`): with
:math:`H = \sum_P h_P P` and :math:`A_k = \sum_Q i a_Q Q`, every
anticommuting pair contributes :math:`2 i\,h_P a_Q P Q`.  The state is never
needed, only what a processor returns: Pauli expectation values.

**The scheme** (:class:`PoolGradients`).

* The commutator observables are formed **as bit tables, streamed in
  batches** of at most :data:`GRADIENT_BATCH_STRINGS` strings.  A full
  fermionic pool against a molecular Hamiltonian has :math:`10^7`--:math:`10^{10}`
  strings at 30--50 qubits; no batch holds more than the cap, so the classical
  memory is bounded by it, not by the pool.
* Within a batch the strings of every operator are **merged** (one string is
  measured once however many operators share it) and partitioned into
  qubit-wise-commuting bases by the same greedy rule as an energy
  (:func:`~mandacaru.backends.measurement.qwc_groups`).  Each basis is one
  circuit: the ansatz, single-qubit rotations, a readout.  The bases are packed
  into jobs of at most ``max_bases`` bases and ``max_strings`` strings
  (the ``measurement_budget`` ceilings), so no job is larger than the
  measurement budget allows.
* :math:`g_k` is the sparse contraction :math:`\sum_R c_{kR}\langle R\rangle`,
  and its standard error :math:`(\sum_R c_{kR}^2\sigma_R^2)^{1/2}` (the
  covariance of strings read from the same basis is neglected).

**At a computational basis state** -- the Hartree-Fock reference, before the
first operator -- the gradients are exact and classical
(:meth:`PoolGradients.reference_gradients`): :math:`\langle b|R|b\rangle` is
:math:`\pm 1` for an ``I``/``Z`` string and 0 otherwise, so only the products
:math:`PQ` with the same flip pattern contribute.  No measurement is made.

**The cost** (:class:`GradientMeasurementPlan`) depends on the Hamiltonian
and on the operators measured, not on the state, so it is computed once and
checked before the first job of every growth step.  Measured with the
fermionic pool: water DZ CAS(8, 7) (14 qubits, 204 operators) 15,911 bases,
79 energies' worth; methanol DZ CAS(10, 10) (20 qubits, 875 operators)
229,641 bases, 137 energies (:doc:`/guide/measurement_cost`); measuring each
operator's commutator separately would need 10 to 25 times more.  The
asymptotically cheaper grouping of Anastasiou *et al.* (arXiv:2306.03227),
which pivots on a Hamiltonian term and needs up to :math:`N - 3` extra CNOTs
per diagonalizing circuit, is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mapping import PauliSum
from ..core.pauli_algebra import (PauliTable, _I_POWERS, _popcount,
                                  commutator_table, label_letters,
                                  pack_labels)
from .measurement import qwc_groups, resilience_multipliers

#: Commutator strings (before merging) held in memory at once.  At 50 qubits
#: a row is 24 bytes and its label 50, so a batch peaks near 100 MB.
GRADIENT_BATCH_STRINGS = 1_000_000

#: Distinct strings a step remembers across its batches, so a string two
#: batches share is measured once (40 bytes each: 160 MB at the default).
GRADIENT_STORE_STRINGS = 4_000_000

#: Batches kept between growth steps when they all fit (so a step does not
#: rebuild the commutators): beyond this the observables are rebuilt each step.
GRADIENT_CACHE_BYTES = 256 * 2 ** 20


@dataclass(frozen=True)
class GradientMeasurementPlan:
    """What measuring one growth step's pool gradients asks a device to do.

    Computed before anything is submitted: the commutators and their grouping
    are classical, and the same for every step (they do not depend on the
    state).  ``strings`` and ``bases`` are summed over the batches; a string
    an earlier batch measured is read from the step's store, and measured
    again only past :data:`GRADIENT_STORE_STRINGS`.
    """

    n_qubits: int
    pool_size: int
    operators: int
    strings: int
    bases: int
    jobs: int
    batches: int
    shots_per_basis: int
    noise_factors: int
    twirls: int
    shots_per_twirl: int
    max_job_bases: int
    max_job_strings: int
    batch_bytes: int
    #: Largest :math:`(\sum_R c_{kR}^2)^{1/2}` over the measured operators.
    coefficient_norm: float = 0.0
    device: str = "?"

    @property
    def noise_bound_hartree(self) -> float:
        r"""Bound on one gradient's standard error, :math:`\max_k
        \|c_k\|_2/\sqrt{\text{shots}}` (each :math:`\langle R\rangle` has a
        variance of at most one per shot).  The largest of :math:`N` noisy
        gradients is biased up by about :math:`\sqrt{2\ln N}` of it, which is
        the floor a gradient criterion can reach."""
        if self.shots_per_basis <= 0:
            return 0.0
        return float(self.coefficient_norm / np.sqrt(self.shots_per_basis))

    @property
    def circuit_instances(self) -> int:
        return self.bases * max(self.noise_factors, 1) * max(self.twirls, 1)

    @property
    def total_shots(self) -> int:
        """Shots of one gradient pass, with the provider's resilience
        multipliers (:func:`~mandacaru.backends.measurement.resilience_multipliers`)."""
        per = (self.twirls * self.shots_per_twirl if self.twirls > 1
               else max(self.shots_per_basis, 0))
        return self.bases * max(self.noise_factors, 1) * per

    def fields(self, energy_bases: int | None = None) -> dict:
        """The ``[GRADIENT MEASUREMENT]`` lines (one owner: nothing else
        writes these numbers; the device and the shots per basis are the
        setup block's ``device`` and ``shots``)."""
        out = {
            "scheme": ("commutator observables [H, A_k], strings merged per "
                       "batch, qubit-wise-commuting bases"),
            "first_step": "classical (exact at the reference determinant)",
            "measured_operators": self.operators,
            "strings_per_step": self.strings,
            "bases_per_step": self.bases,
        }
        if energy_bases:
            out["bases_per_step_in_energies"] = (
                f"{self.bases / energy_bases:.1f} (an energy is "
                f"{energy_bases} bases)")
        out.update({
            "jobs_per_step": self.jobs,
            "largest_job": (f"{self.max_job_bases} bases, "
                            f"{self.max_job_strings} strings"),
            "circuit_instances_per_step": self.circuit_instances,
            "shots_per_step": self.total_shots,
            "gradient_noise_bound_Ha": f"{self.noise_bound_hartree:.3e}",
            "batches_per_step": self.batches,
            "batch_memory_bytes": self.batch_bytes,
        })
        return out


class _StringStore:
    """The strings already measured in a step, so a later batch reads them
    instead of measuring them again.

    Sorted keys (the ``[x | z]`` words of a string as one void scalar) and,
    for each, its slot in the step's value arrays.  At most ``capacity``
    strings are kept (:data:`GRADIENT_STORE_STRINGS`); past it a string is
    measured again by every batch that needs it, which costs shots, never
    memory.
    """

    def __init__(self, capacity: int, width: int):
        self.capacity = int(capacity)
        self.keys = np.zeros(0, dtype=f"V{width}")
        self.slots = np.zeros(0, dtype=np.int64)
        self.count = 0

    def lookup(self, keys) -> np.ndarray:
        """Slot of each key, ``-1`` where it is not stored."""
        if not self.count or not len(keys):
            return np.full(len(keys), -1, dtype=np.int64)
        position = np.searchsorted(self.keys, keys)
        clipped = np.minimum(position, self.count - 1)
        found = self.keys[clipped] == keys
        return np.where(found, self.slots[clipped], -1)

    def add(self, keys) -> np.ndarray:
        """Store ``keys`` (new ones) while there is room; their slots, ``-1``
        for those past the capacity."""
        room = max(min(len(keys), self.capacity - self.count), 0)
        slots = np.full(len(keys), -1, dtype=np.int64)
        if room:
            slots[:room] = np.arange(self.count, self.count + room)
            merged = np.concatenate([self.keys, keys[:room]])
            order = np.argsort(merged, kind="stable")
            self.keys = merged[order]
            self.slots = np.concatenate([self.slots, slots[:room]])[order]
            self.count += room
        return slots


class _Batch:
    """The merged commutator strings of a run of pool operators.

    ``operators`` are pool indices; ``matrix`` is the ``(operators, strings)``
    sparse coefficient matrix, so ``matrix @ <R>`` is the gradients; the
    strings are kept in lexical order of their labels (the order the greedy
    grouping visits them in) as bit tables and ASCII letters.  After
    :meth:`assign` only the strings no earlier batch measured (``new``) are
    grouped into bases and measured here; the others (``stored``) are read
    from the step's store.
    """

    def __init__(self, operators, x, z, coeffs, owners, n_qubits):
        from scipy.sparse import csr_matrix

        self.operators = np.asarray(operators, dtype=np.int64)
        keys = np.ascontiguousarray(np.concatenate([x, z], axis=1))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        words = x.shape[1]
        ux, uz = unique[:, :words], unique[:, words:]
        letters = label_letters(ux, uz, n_qubits)
        # Lexical order of the labels: the order the plan and the provider's
        # own grouping of a job both visit them in.
        order = np.argsort(np.ascontiguousarray(letters).view(
            f"S{n_qubits}").ravel(), kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        self.x = np.ascontiguousarray(ux[order])
        self.z = np.ascontiguousarray(uz[order])
        self.letters = np.ascontiguousarray(letters[order])
        self.n_qubits = int(n_qubits)
        self.matrix = csr_matrix(
            (coeffs, (owners, rank[inverse.ravel()])),
            shape=(len(self.operators), len(order)))
        self.new = np.arange(len(order))
        self.stored = np.full(len(order), -1, dtype=np.int64)
        self.slots = np.full(len(order), -1, dtype=np.int64)
        self.groups = None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def keys(self) -> np.ndarray:
        rows = np.ascontiguousarray(np.concatenate([self.x, self.z], axis=1))
        return rows.view(f"V{rows.shape[1] * 8}").ravel()

    def assign(self, store: _StringStore) -> None:
        """Split the strings into stored and new, and group the new ones."""
        keys = self.keys()
        self.stored = store.lookup(keys)
        self.new = np.flatnonzero(self.stored < 0)
        self.slots = np.full(len(self), -1, dtype=np.int64)
        self.slots[self.new] = store.add(keys[self.new])
        _bx, _bz, groups = qwc_groups(self.x[self.new], self.z[self.new],
                                      np.arange(len(self.new)))
        self.groups = [self.new[np.asarray(group)].tolist()
                       for group in groups]

    @property
    def nbytes(self) -> int:
        m = self.matrix
        return int(self.x.nbytes + self.z.nbytes + self.letters.nbytes
                   + m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
                   + 3 * 8 * len(self))

    def labels(self, rows) -> list[str]:
        n = self.n_qubits
        text = self.letters[np.asarray(rows, dtype=np.int64)].tobytes()
        text = text.decode("ascii")
        return [text[i * n:(i + 1) * n] for i in range(len(rows))]

    def jobs(self, max_bases: int | None, max_strings: int | None):
        """Row indices of each job: whole bases, at most ``max_bases`` of them
        and ``max_strings`` strings (a single basis larger than that is a job
        of its own)."""
        jobs, current, bases = [], [], 0
        for group in self.groups:
            over = ((max_bases and bases + 1 > max_bases)
                    or (max_strings and current
                        and len(current) + len(group) > max_strings))
            if current and over:
                jobs.append((current, bases))
                current, bases = [], 0
            current.extend(group)
            bases += 1
        if current:
            jobs.append((current, bases))
        return jobs


class PoolGradients:
    """The commutator observables of a pool against one Hamiltonian.

    Parameters
    ----------
    hamiltonian : PauliSum
        The qubit Hamiltonian (its Hermitian part is used).
    generators : sequence of PauliSum
        The pool's anti-Hermitian generators, in pool order.
    batch_strings : int
        Commutator strings held at once (:data:`GRADIENT_BATCH_STRINGS`).
    """

    def __init__(self, hamiltonian: PauliSum, generators, *,
                 batch_strings: int = GRADIENT_BATCH_STRINGS):
        h = hamiltonian.hermitian_part()
        self.num_qubits = int(h.num_qubits)
        self._h = PauliTable.from_pauli_sum(h)
        self._generators = [PauliTable.from_pauli_sum(g.simplify())
                            for g in generators]
        self.batch_strings = max(int(batch_strings), 1)
        self._plans: dict = {}
        self._flip_rows = None
        self._references: dict = {}

    def with_generators(self, generators) -> PoolGradients:
        """The same Hamiltonian against other generators (its tables shared):
        a coupled-exchange operator's members, say."""
        other = PoolGradients.__new__(PoolGradients)
        other.num_qubits = self.num_qubits
        other._h = self._h
        other._generators = [PauliTable.from_pauli_sum(g.simplify())
                             for g in generators]
        other.batch_strings = self.batch_strings
        other._plans = {}
        other._flip_rows = self._flip_rows
        other._references = {}
        return other

    def __len__(self) -> int:
        return len(self._generators)

    def energy_bases(self) -> int:
        """Qubit-wise-commuting bases of one energy of the Hamiltonian, as an
        energy is partitioned (:func:`~mandacaru.backends.measurement.qwc_partition`,
        weighted): the unit the gradient step's cost is quoted in."""
        if getattr(self, "_energy_bases", None) is None:
            from .measurement import qwc_partition

            h = self._h
            identity = ~(h.x.any(axis=1) | h.z.any(axis=1))
            keep = np.flatnonzero(~identity)
            labels = PauliTable(h.x[keep], h.z[keep], h.coeffs[keep],
                                self.num_qubits).labels()
            self._energy_bases = len(qwc_partition(
                labels, np.abs(h.coeffs[keep]).tolist())) if labels else 0
        return self._energy_bases

    # -- the observables -------------------------------------------------- #

    def observable(self, k: int) -> PauliTable:
        """``[H, A_k]`` as a merged table with real coefficients."""
        table = commutator_table(self._h, self._generators[int(k)])
        return PauliTable(table.x, table.z, table.coeffs.real.copy(),
                          self.num_qubits)

    def observable_strings(self, k: int) -> int:
        """Strings of ``[H, A_k]`` (what one operator alone would measure)."""
        return len(self.observable(k))

    def _batches(self, indices):
        """Yield :class:`_Batch` es covering ``indices`` in order."""
        xs, zs, cs, owners, members, count = [], [], [], [], [], 0
        for k in indices:
            table = self.observable(k)
            if len(table):
                xs.append(table.x)
                zs.append(table.z)
                cs.append(table.coeffs)
                owners.append(np.full(len(table), len(members),
                                      dtype=np.int64))
                count += len(table)
            members.append(int(k))
            if count >= self.batch_strings:
                yield self._batch(members, xs, zs, cs, owners)
                xs, zs, cs, owners, members, count = [], [], [], [], [], 0
        if members:
            yield self._batch(members, xs, zs, cs, owners)

    def _batch(self, members, xs, zs, cs, owners) -> _Batch:
        words = self._h.x.shape[1]
        if not xs:
            empty = np.zeros((0, words), dtype=np.uint64)
            return _Batch(members, empty, empty.copy(), np.zeros(0),
                          np.zeros(0, dtype=np.int64), self.num_qubits)
        return _Batch(members, np.concatenate(xs), np.concatenate(zs),
                      np.concatenate(cs), np.concatenate(owners),
                      self.num_qubits)

    # -- exact gradients at a computational basis state ------------------- #

    def reference_gradients(self, occupied_qubits) -> np.ndarray:
        r"""Exact :math:`g_k = \langle b|[H, A_k]|b\rangle` at the basis
        state :math:`|b\rangle` with ``occupied_qubits`` set.

        Only products :math:`PQ` with no ``X``/``Y`` survive the diagonal
        element, i.e. Hamiltonian strings with the generator string's own
        flip pattern; each contributes
        :math:`2 i h_P a_Q\, i^{e}(-1)^{|z \wedge b|}`.  Classical, exact,
        and :math:`O(\sum_Q |H_{x_Q}|)`: no commutator is formed.
        """
        n = self.num_qubits
        occupied = {int(q) for q in occupied_qubits}
        key = tuple(sorted(occupied))
        if key in self._references:
            return self._references[key].copy()
        # The occupation as a z mask: the Z string on the occupied qubits.
        b = pack_labels(["".join("Z" if q in occupied else "I"
                                 for q in range(n))], n)[1][0]
        if self._flip_rows is None:
            keys = self._h.x
            groups: dict[bytes, list[int]] = {}
            for row, key in enumerate(map(bytes, keys)):
                groups.setdefault(key, []).append(row)
            self._flip_rows = {key: np.asarray(rows, dtype=np.int64)
                               for key, rows in groups.items()}
        h = self._h
        phase_h = _popcount(h.x & h.z)
        out = np.zeros(len(self._generators))
        for k, gen in enumerate(self._generators):
            total = 0.0 + 0.0j
            for row in range(len(gen)):
                x2, z2 = gen.x[row], gen.z[row]
                rows = self._flip_rows.get(bytes(x2))
                if rows is None:
                    continue
                x1, z1 = h.x[rows], h.z[rows]
                odd = ((_popcount(x1 & z2) + _popcount(z1 & x2)) & 1
                       ).astype(bool)
                if not odd.any():
                    continue
                rows, x1, z1 = rows[odd], x1[odd], z1[odd]
                x3, z3 = x1 ^ x2, z1 ^ z2
                exponent = (phase_h[rows]
                            + int(_popcount((x2 & z2)[None, :])[0])
                            - _popcount(x3 & z3)
                            + 2 * _popcount(z1 & x2)) % 4
                # sigma(x3 = 0, z3) = Z^z3, diagonal: (-1)^{|z3 & b|}.
                sign = 1 - 2 * (_popcount(z3 & b[None, :]) & 1)
                total += np.sum(2.0 * h.coeffs[rows] * gen.coeffs[row]
                                * _I_POWERS[exponent] * sign)
            out[k] = float(total.real)
        self._references[key] = out
        return out.copy()

    # -- the plan ---------------------------------------------------------- #

    def plan(self, provider, indices, max_bases: int | None = None,
             max_strings: int | None = None,
             cache: bool = True) -> GradientMeasurementPlan:
        """The cost of measuring ``indices`` (cached: it is state-independent).

        One classical pass over the commutators; when every batch fits
        :data:`GRADIENT_CACHE_BYTES` the batches are kept for
        :meth:`measure`, which then rebuilds nothing.  ``cache=False`` keeps
        neither (a probe: the automatic screening sizes several subsets).
        """
        indices = tuple(int(k) for k in indices)
        key = self._plan_key(provider, indices, max_bases, max_strings)
        if key in self._plans:
            return self._plans[key][0]
        strings = bases = jobs = batches = 0
        job_bases = job_strings = 0
        peak = 0
        norm = 0.0
        kept, cached_bytes = ([] if cache else None), 0
        store = self._store()
        for batch in self._batches(indices):
            batch.assign(store)
            batches += 1
            if len(batch):
                rows = np.asarray(batch.matrix.multiply(batch.matrix)
                                  .sum(axis=1)).ravel()
                norm = max(norm, float(np.sqrt(rows.max(initial=0.0))))
            strings += len(batch.new)
            bases += len(batch.groups)
            for rows, n_bases in batch.jobs(max_bases, max_strings):
                jobs += 1
                job_bases = max(job_bases, n_bases)
                job_strings = max(job_strings, len(rows))
            peak = max(peak, batch.nbytes)
            if kept is not None:
                cached_bytes += batch.nbytes
                kept = (kept + [batch] if cached_bytes <= GRADIENT_CACHE_BYTES
                        else None)
        noise, twirls, per = resilience_multipliers(provider)
        plan = GradientMeasurementPlan(
            n_qubits=self.num_qubits, pool_size=len(self),
            operators=len(indices), strings=strings, bases=bases, jobs=jobs,
            batches=batches,
            shots_per_basis=int(getattr(provider, "shots", 0) or 0),
            noise_factors=noise, twirls=twirls, shots_per_twirl=per,
            max_job_bases=job_bases, max_job_strings=job_strings,
            batch_bytes=peak + 40 * store.count, coefficient_norm=norm,
            device=str(getattr(provider, "device_spec", provider)))
        if cache:
            self._plans[key] = (plan, kept)
        return plan

    @staticmethod
    def _plan_key(provider, indices, max_bases, max_strings):
        """A plan belongs to its operators, its job ceilings and the provider
        it was counted for (the shots and the multipliers are the provider's)."""
        return (indices, max_bases, max_strings,
                int(getattr(provider, "shots", 0) or 0),
                resilience_multipliers(provider),
                str(getattr(provider, "device_spec", provider)))

    def _store(self) -> _StringStore:
        return _StringStore(GRADIENT_STORE_STRINGS,
                            16 * self._h.x.shape[1])

    # -- the measurement --------------------------------------------------- #

    def measure(self, provider, problem, indices, max_bases=None,
                max_strings=None):
        """``(gradients, standard_errors)`` of ``indices`` at the state of
        ``problem`` -- ``(n_qubits, occupied, generators, thetas)`` -- from
        the provider's Pauli expectation values.

        Every job is one ``provider.expectation_values`` call on the labels
        of whole bases; the provider's run ledger counts and refuses each one
        before it is queued.  A string an earlier batch of the step measured
        is read from the step's store, not measured again.
        """
        indices = tuple(int(k) for k in indices)
        self.plan(provider, indices, max_bases, max_strings)
        kept = self._plans[self._plan_key(provider, indices, max_bases,
                                          max_strings)][1]
        batches = kept if kept is not None else self._batches(indices)
        position = {k: i for i, k in enumerate(indices)}
        grads = np.zeros(len(indices))
        variances = np.zeros(len(indices))
        n_qubits, occupied, generators, thetas = problem[:4]
        store = self._store()
        stored_values = np.zeros(0)
        stored_sigma2 = np.zeros(0)
        for batch in batches:
            if kept is None:
                batch.assign(store)
            values = np.zeros(len(batch))
            sigma2 = np.zeros(len(batch))
            old = np.flatnonzero(batch.stored >= 0)
            values[old] = stored_values[batch.stored[old]]
            sigma2[old] = stored_sigma2[batch.stored[old]]
            for rows, _n_bases in batch.jobs(max_bases, max_strings):
                labels = batch.labels(rows)
                measured, stds = _expectation_values(
                    provider, n_qubits, occupied, generators, thetas, labels)
                values[rows] = [measured[label] for label in labels]
                sigma2[rows] = [float(stds.get(label, 0.0)) ** 2
                                for label in labels]
            slots = batch.slots[batch.new]
            keep = slots >= 0
            if keep.any():
                size = int(slots[keep].max()) + 1
                if size > stored_values.size:
                    stored_values = np.resize(stored_values, size)
                    stored_sigma2 = np.resize(stored_sigma2, size)
                stored_values[slots[keep]] = values[batch.new[keep]]
                stored_sigma2[slots[keep]] = sigma2[batch.new[keep]]
            local = batch.matrix @ values
            spread = batch.matrix.multiply(batch.matrix) @ sigma2
            for i, k in enumerate(batch.operators.tolist()):
                grads[position[k]] = local[i]
                variances[position[k]] = spread[i]
        return grads, np.sqrt(variances)


def _expectation_values(provider, n_qubits, occupied, generators, thetas,
                        labels):
    """One job of Pauli expectation values: ``({label: <P>}, {label: std})``.

    ``max_bases_per_job=0`` because the labels are already one job's worth of
    whole bases; the provider must not split them again.
    """
    method = getattr(provider, "expectation_values", None)
    if method is None:
        raise NotImplementedError(
            f"{provider!r} cannot measure Pauli expectation values, which "
            f"the pool gradients of a shot-based ADAPT-VQE are made of; use "
            f"backend_provider='qiskit' or 'braket'")
    values, stds = method(n_qubits, occupied, generators, thetas, labels,
                          max_bases_per_job=0)
    shots = int(getattr(provider, "shots", 0) or 0)
    if shots > 0 and not any(stds.get(label, 0.0) for label in labels):
        # An estimator that reports no spread (the local one samples at a
        # fixed precision): each <P> has a variance of at most 1 per shot.
        stds = dict.fromkeys(labels, 1.0 / np.sqrt(shots))
    return values, stds


# --------------------------------------------------------------------------- #
# The dry run's estimate, before any Hamiltonian exists.
# --------------------------------------------------------------------------- #

#: Jordan-Wigner Pauli strings of a molecular Hamiltonian per ``M^4`` spatial
#: orbitals: 1.41 for dense random integrals (measured at M = 4-10, 361 to
#: 14,251 strings); a real molecule has fewer (water DZ CAS(8, 7): 902 at
#: M = 7, a quarter of it), so the estimate is an upper one.
PAULI_STRINGS_PER_M4 = 1.41
#: Pauli strings an energy's qubit-wise-commuting basis holds on average
#: (dense integrals: 4.6-4.8 at M = 4-10; water DZ: 4.5).
STRINGS_PER_ENERGY_BASIS = 4.6
#: A merged gradient step's bases per pool operator, in energies' worth of
#: bases, by pool -- the fitted constants of :func:`estimate_gradient_step`.
#: Measured (merged plans, :doc:`/guide/measurement_cost`): fermionic 0.10-0.39
#: (dense integrals at 8-20 qubits, water DZ 0.39); QEB about twice the
#: fermionic (its strings carry no Z chains, so fewer are shared: water 0.76);
#: OVP-CEO 0.38 per operator of its larger, generalized pool; the qubit pool
#: measures the fermionic pool's own strings, spread over 2 S + 8 D operators.
ENERGIES_PER_OPERATOR = {"fermionic": 0.25, "qeb": 0.5, "ceo": 0.4,
                         "ceo-ovp": 0.4, "qubit": 0.03, "spin-orbit": 0.25}
#: Strings per basis of a merged gradient step (3.6-5.9 measured).
STRINGS_PER_GRADIENT_BASIS = 4.5


def estimate_energy_bases(n_spatial_orbitals: int,
                          h_strings: int | None = None) -> int:
    """Qubit-wise-commuting bases of one energy, from the Hamiltonian's
    string count (``PAULI_STRINGS_PER_M4 * M^4`` when it is not known)."""
    strings_h = (int(h_strings) if h_strings is not None
                 else int(round(PAULI_STRINGS_PER_M4
                                * int(n_spatial_orbitals) ** 4)))
    return max(int(np.ceil(strings_h / STRINGS_PER_ENERGY_BASIS)), 1)


def estimate_gradient_step(pool_size: int, n_spatial_orbitals: int,
                           shots: int, h_strings: int | None = None,
                           budget=None, pool: str = "fermionic") -> dict:
    """Order-of-magnitude cost of one measured screening step, for the dry
    run (which builds no Hamiltonian).

    ``bases ~ ENERGIES_PER_OPERATOR[pool] * pool_size * energy_bases``: a
    merged fermionic gradient step costs about a quarter of an energy per
    pool operator (measured on merged plans at 8-20 qubits, within a factor
    of three; see :doc:`/guide/measurement_cost`), and an energy is
    ``h_strings / STRINGS_PER_ENERGY_BASIS`` bases with
    ``h_strings ~ PAULI_STRINGS_PER_M4 * M^4`` when the Hamiltonian is not
    known.  Jobs are what the measurement budget's per-job ceilings make of
    it.  The exact plan is computed by the run itself
    (:meth:`PoolGradients.plan`) and written to ``[GRADIENT MEASUREMENT]``.
    """
    from .measurement import resolve_measurement_budget

    M = int(n_spatial_orbitals)
    strings_h = (int(h_strings) if h_strings is not None
                 else int(round(PAULI_STRINGS_PER_M4 * M ** 4)))
    energy_bases = estimate_energy_bases(M, strings_h)
    factor = ENERGIES_PER_OPERATOR.get(str(pool), ENERGIES_PER_OPERATOR[
        "fermionic"])
    bases = int(np.ceil(factor * int(pool_size) * energy_bases))
    strings = int(np.ceil(STRINGS_PER_GRADIENT_BASIS * bases))
    limits = resolve_measurement_budget(budget)
    jobs = 1
    if limits["bases"]:
        jobs = max(jobs, -(-bases // int(limits["bases"])))
    if limits["observables"]:
        jobs = max(jobs, -(-strings // int(limits["observables"])))
    return {"pool_size": int(pool_size), "hamiltonian_strings": strings_h,
            "energy_bases": energy_bases, "strings": strings, "bases": bases,
            "jobs": jobs, "shots": bases * max(int(shots), 0),
            "energies": round(bases / energy_bases, 1)}


#: Bytes of one Python ``{term: coefficient}`` entry of a ``Fermion`` (the
#: key's four ``(mode, dagger)`` tuples, the complex value, the dict slot).
FERMION_TERM_BYTES = 420
#: Bytes of one ``PauliSum`` entry at ``n`` qubits beyond the label itself
#: (string header, complex value, dict slot), and of its bit-table row.
PAULI_TERM_BYTES = 170
TABLE_ROW_BYTES = 24


def shot_run_memory_bytes(n_qubits: int, n_spatial_orbitals: int,
                          pool_size: int | None = None,
                          h_strings: int | None = None,
                          strings_per_generator: int = 8,
                          pool: str = "fermionic") -> int:
    """Classical memory of a shot-based run, every term polynomial.

    The fermionic Hamiltonian (``4 M^4`` two-body terms at most), its qubit
    image as labels and as bit tables, the pool's generators, and the
    measured gradients' working set: one batch (:data:`GRADIENT_BATCH_STRINGS`
    rows), the step's string store (:data:`GRADIENT_STORE_STRINGS`) and the
    kept batches (:data:`GRADIENT_CACHE_BYTES`), each bounded by its constant
    or by the step's own size.  No ``2^n`` term: there is none to count.
    """
    n = int(n_qubits)
    M = int(n_spatial_orbitals)
    strings_h = (int(h_strings) if h_strings is not None
                 else int(round(PAULI_STRINGS_PER_M4 * M ** 4)))
    hamiltonian = (FERMION_TERM_BYTES * 4 * M ** 4
                   + (PAULI_TERM_BYTES + n + TABLE_ROW_BYTES) * strings_h)
    pool = ((PAULI_TERM_BYTES + n + TABLE_ROW_BYTES) * strings_per_generator
            * int(pool_size or 0))
    step = 0
    if pool_size:
        estimate = estimate_gradient_step(pool_size, M, 0, strings_h,
                                          pool=pool)
        rows = min(estimate["strings"], GRADIENT_BATCH_STRINGS)
        step = (rows * (48 + n) + 40 * min(estimate["strings"],
                                           GRADIENT_STORE_STRINGS)
                + min(estimate["strings"] * (48 + n), GRADIENT_CACHE_BYTES))
    return int(hamiltonian + pool + step)
