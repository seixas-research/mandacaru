"""Nonlocal energy, field derivatives, limiting cases and bounded ownership."""
import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native

REFERENCE = json.loads((Path(__file__).parent / "native_cutover/nonlocal_reference.json").read_text())


def grid(periodic):
    return _native.SpatialGrid(REFERENCE["shape"], [0., 0., 0.], np.asarray(REFERENCE["step"]), periodic=periodic)


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_nonlocal_matches_frozen_convolution(case):
    ctx = _native.Context(threads=1)
    n, sigma = (np.asarray(REFERENCE[key]) for key in ("density", "sigma"))
    original = n.copy()
    out = ctx.nonlocal_correlation(grid(case["periodic"]), n, sigma, b=case["b"])
    # An analytic reciprocal transform tabulated at uniform wave numbers
    # defines the same discrete energy and variational response.
    assert out["energy"] == pytest.approx(case["energy"], rel=2e-7, abs=2e-11)
    assert np.allclose(out["d_density"], case["d_density"], rtol=2e-6, atol=1e-10)
    assert np.allclose(out["d_sigma"], case["d_sigma"], rtol=2e-6, atol=1e-10)
    assert np.array_equal(n, original)
    assert ctx.memory_used_bytes == 0


@pytest.mark.parametrize("periodic", [False, True])
def test_variational_density_and_gradient_partials(periodic):
    ctx = _native.Context(threads=1)
    g = grid(periodic)
    n, sigma = (np.asarray(REFERENCE[key]) for key in ("density", "sigma"))
    rng = np.random.default_rng(271026)
    dn, ds = rng.uniform(-1., 1., (2, n.size)) * np.array([n, sigma])
    terms = ctx.nonlocal_correlation(g, n, sigma)
    delta = 1e-5
    plus = ctx.nonlocal_correlation(g, n+delta*dn, sigma+delta*ds)["energy"]
    minus = ctx.nonlocal_correlation(g, n-delta*dn, sigma-delta*ds)["energy"]
    expected = np.sum(terms["d_density"]*dn + terms["d_sigma"]*ds) * g.dv_bohr3
    assert (plus-minus)/(2*delta) == pytest.approx(expected, rel=2e-7, abs=2e-11)


def test_uniform_gas_and_invalid_inputs():
    ctx = _native.Context(threads=1)
    g = grid(True)
    n = np.full(np.prod(REFERENCE["shape"]), .1)
    sigma = np.zeros_like(n)
    result = ctx.nonlocal_correlation(g, n, sigma)
    beta = (3/11.95**2)**.75/32
    assert abs(result["energy"]) < 3e-4 * beta * np.sum(n) * g.dv_bohr3
    for kwargs in [dict(b=0.), dict(nodes=1), dict(nodes=65), dict(q_min=1.), dict(c=-1.)]:
        with pytest.raises(ValueError):
            ctx.nonlocal_correlation(g, n, sigma, **kwargs)
    with pytest.raises(ValueError):
        ctx.nonlocal_correlation(g, -n, sigma)
    with pytest.raises(MemoryError):
        _native.Context(memory_limit_bytes=1 << 16).nonlocal_correlation(g, n, sigma)
    assert ctx.memory_used_bytes == 0
