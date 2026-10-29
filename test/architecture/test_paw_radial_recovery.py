"""The typed PAW boundary preserves the radial operator and its metric."""

import numpy as np
import pytest
from scipy.linalg import eigh

from mandacaru import _native


def channel():
    r = np.arange(1, 81, dtype=float) * 0.08
    v = -1.0 / np.sqrt(r * r + 0.4)
    projectors = np.array([r * np.exp(-r * r), r * np.exp(-0.3 * r * r)])
    d = np.array([[-0.3, 0.07], [0.07, 0.2]])
    q = np.array([[0.2, -0.03], [-0.03, -0.05]])
    return r, v, projectors, d, q


def test_native_radial_paw_states_match_dense_generalized_reference():
    r, v, p, d, q = channel()
    h = r[0]
    t = np.diag(1 / h**2 + v)
    t += np.diag(np.full(len(r) - 1, -0.5 / h**2), 1)
    t += np.diag(np.full(len(r) - 1, -0.5 / h**2), -1)
    hamiltonian = t + h * p.T @ d @ p
    overlap = np.eye(len(r)) + h * p.T @ q @ p
    expected = eigh(hamiltonian, overlap, eigvals_only=True)
    ctx = _native.Context(threads=1, memory_limit_bytes=4 * 1024**2)
    for state in [0, 1, 4, len(r) - 1]:
        energy, u = ctx.paw_eigenpair(r, v, p, d, q, state=state)
        assert energy == pytest.approx(expected[state], abs=1e-9)
        assert h * u @ overlap @ u == pytest.approx(1.0, abs=1e-12)
        assert np.linalg.norm(hamiltonian @ u - energy * overlap @ u) < 1e-7
        value, vector = ctx.paw_eigenpair(r, v, p, d, q, state=state, vectors=False)
        assert value == energy and vector is None
    assert ctx.memory_used_bytes == 0


def test_native_radial_paw_rejects_wrong_geometry_and_indefinite_overlap():
    r, v, p, d, q = channel()
    ctx = _native.Context(threads=1)
    irregular = r.copy()
    irregular[1] += 0.01
    with pytest.raises(ValueError, match="uniformly spaced"):
        ctx.paw_eigenpair(irregular, v, p, d, q)
    with pytest.raises(ValueError, match="positive definite"):
        ctx.paw_eigenpair(r, v, p, d, -100 * np.eye(2))
    assert ctx.memory_used_bytes == 0
