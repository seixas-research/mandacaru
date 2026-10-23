"""Native quadrature sampling preserves the reference basis and derivatives."""
import numpy as np
import pytest

from mandacaru.integrals import _backend
from mandacaru.integrals.sampling import sample_functions


def functions(real):
    # Cover multiple zetas, all magnetic components, and angular momenta
    # beyond the s/p shells of light atoms, including the radial origin.
    from mandacaru.pseudopotentials.orbitals import _RadialTabulated
    r = np.linspace(0.0, 7.0, 1201)
    out = []
    for l in range(4):
        for width in (0.8, 1.3):
            radial = r**l * np.exp(-r*r / width**2)
            for m in range(-l, l + 1):
                out.append(_RadialTabulated(r, radial, l, m,
                                            center=[0.0, 0.0, 0.0], units="bohr", real=real))
    return out


@pytest.mark.parametrize("real", [True, False])
@pytest.mark.parametrize("native", [True, False])
def test_values_and_gradients_match_individual_evaluation(real, native, monkeypatch):
    if native and not _backend.HAS_C_BACKEND:
        pytest.skip("native integral backend unavailable")
    if not native:
        monkeypatch.setattr(_backend, "HAS_C_BACKEND", False)
    basis = functions(real)
    random = np.random.default_rng(4).normal(size=(3, 131))
    points = tuple(np.r_[0.0, random[i], 20.0] for i in range(3))
    values, slopes = sample_functions(basis, points, gradients=True)
    reference = np.array([f.evaluate(*points) for f in basis])
    derivative = np.array([f.gradient(*points) for f in basis])
    assert np.max(np.abs(values - reference)) < 2e-13
    assert np.max(np.abs(slopes - derivative)) < 2e-11
    values_only, empty = sample_functions(basis, points)
    assert empty is None
    assert np.max(np.abs(values_only - reference)) < 2e-13


def test_custom_functions_keep_their_evaluation_contract():
    from mandacaru.basis.base import BasisFunction

    class Custom(BasisFunction):
        real = True
        def evaluate(self, x, y, z):
            return x + 2*y + 3*z

    points = tuple(np.arange(5.0) + i for i in range(3))
    values, slopes = sample_functions([Custom()], points, gradients=True)
    assert values[0] == pytest.approx(points[0] + 2*points[1] + 3*points[2])
    assert slopes[0] == pytest.approx(np.broadcast_to(np.array([1, 2, 3])[:, None], (3, 5)))
