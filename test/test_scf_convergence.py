# -*- coding: utf-8 -*-
# file: test/test_scf_convergence.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The nuclear gradient must not differentiate an unconverged SCF.

``_jax_energy._scf`` unrolls a fixed number of RHF iterations so the whole
pipeline is a straight-line function of the integrals.  That is only a valid
*derivative* if the final iterations are no-ops.  When they are not -- when the
iteration count stops the SCF mid-transient -- the gradient is taken through a
transient and explodes, silently: the forces stay finite and only a symmetry
argument reveals they are wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from carcara.algorithms._jax_energy import (SCF_ITERATION_MARGIN,
                                            _resolve_scf_iterations,
                                            scf_iterations_required)

pytest.importorskip("jax")


def _two_level_system(n=6, seed=0):
    """A small symmetric h/eri pair with a well-defined RHF solution."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    h = 0.5 * (A + A.T) - 3.0 * np.eye(n)
    B = rng.normal(size=(n, n, n, n)) * 0.05
    eri = 0.125 * (B + B.transpose(1, 0, 3, 2) + B.transpose(2, 3, 0, 1)
                   + B.transpose(3, 2, 1, 0) + B.transpose(2, 1, 0, 3)
                   + B.transpose(0, 3, 2, 1) + B.transpose(1, 2, 3, 0)
                   + B.transpose(3, 0, 1, 2))
    return h, eri


class TestConvergenceProbe:
    """`scf_iterations_required` measures what the unroll actually needs."""

    def test_returns_a_positive_count(self):
        h, eri = _two_level_system()
        needed = scf_iterations_required(h, eri, n_occ=2)
        assert needed is not None and needed > 0

    def test_the_reported_count_really_converges(self):
        """Running exactly that many iterations must leave the density still."""
        h, eri = _two_level_system()
        n_occ = 2
        needed = scf_iterations_required(h, eri, n_occ)

        _w, C = np.linalg.eigh(h)
        for _ in range(needed):
            occ = C[:, :n_occ]
            D = 2.0 * (occ @ occ.T)
            J = np.einsum("rs,prqs->pq", D, eri, optimize=True)
            K = np.einsum("rs,prsq->pq", D, eri, optimize=True)
            F = h + J - 0.5 * K
            _e, C = np.linalg.eigh(0.5 * (F + F.T))
        D_final = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)

        occ = C[:, :n_occ]
        J = np.einsum("rs,prqs->pq", 2.0 * (occ @ occ.T), eri, optimize=True)
        K = np.einsum("rs,prsq->pq", 2.0 * (occ @ occ.T), eri, optimize=True)
        _e, C2 = np.linalg.eigh(0.5 * ((h + J - 0.5 * K)
                                       + (h + J - 0.5 * K).T))
        D_next = 2.0 * (C2[:, :n_occ] @ C2[:, :n_occ].T)
        assert np.abs(D_next - D_final).max() < 1e-8

    def test_non_convergence_reports_none(self):
        """A runaway iteration returns None rather than a plausible number."""
        h, eri = _two_level_system()
        assert scf_iterations_required(h, eri, n_occ=2, cap=1) is None


class TestUnrollSizing:
    """The caller's `scf_iterations` is a floor, never a ceiling."""

    def test_requested_count_is_raised_when_insufficient(self):
        h, eri = _two_level_system()
        needed = scf_iterations_required(h, eri, n_occ=2)
        resolved = _resolve_scf_iterations(
            np.eye(h.shape[0]), h, eri,
            {"n_electrons": 4, "scf_iterations": 1})
        assert resolved >= needed + SCF_ITERATION_MARGIN
        assert resolved > 1

    def test_a_generous_request_is_respected(self):
        h, eri = _two_level_system()
        resolved = _resolve_scf_iterations(
            np.eye(h.shape[0]), h, eri,
            {"n_electrons": 4, "scf_iterations": 5000})
        assert resolved == 5000

    def test_skipped_without_an_scf(self):
        h, eri = _two_level_system()
        resolved = _resolve_scf_iterations(
            np.eye(h.shape[0]), h, eri,
            {"n_electrons": 4, "scf_iterations": 7, "mo_basis": False})
        assert resolved == 7

    def test_hard_non_convergence_is_an_error_not_a_number(self):
        """Better to refuse than to return a gradient through a transient."""
        import carcara.algorithms._jax_energy as je

        original = je.scf_iterations_required
        je.scf_iterations_required = lambda *a, **k: None
        try:
            with pytest.raises(RuntimeError, match="did not converge"):
                _resolve_scf_iterations(np.eye(6), *_two_level_system(),
                                        {"n_electrons": 4})
        finally:
            je.scf_iterations_required = original


class TestGradientStability:
    """The end-to-end symptom: gradients must not depend on the unroll length."""

    def test_gradient_is_stable_once_converged(self):
        """Past convergence, more iterations must change nothing.

        This is the property the fixed-40 default silently violated.
        """
        from carcara.algorithms._jax_energy import integral_gradients

        h, eri = _two_level_system()
        n = h.shape[0]
        S = np.eye(n)
        rng = np.random.default_rng(1)
        gamma = np.diag(np.concatenate([np.ones(2), np.zeros(n - 2),
                                        np.ones(2), np.zeros(n - 2)]))
        gamma2 = np.zeros((2 * n,) * 4)

        results = []
        for requested in (10, 40, 200):
            ds, dh, dg = integral_gradients(
                S, h, eri, gamma, gamma2, n_electrons=4, frozen=(),
                nuclear_repulsion=0.0, scf_iterations=requested)
            results.append((ds, dh, dg))

        for later in results[1:]:
            for a, b in zip(results[0], later):
                np.testing.assert_allclose(a, b, atol=1e-6)
