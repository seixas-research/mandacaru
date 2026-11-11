"""VQ-8: the values the workflow result views derive (the magnitudes, bin spacing and peaks of a
spectrum, the QPE readout, the collapsed state, the ground-state overlap, the normalized input, the
real terms of a propagator, the pulse bound, the real-part check of the MP2 integrals) are computed
in Rust and reproduce, bit for bit, what the Python route produced
(``reference/workflow_views.json``, ``freeze_workflow_views.py``)."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.workflows.quantum_echoes import fourier_spectrum

from .freeze_workflow_views import views

# NumPy's complex magnitude and its sort of equal keys follow the CPU's vector extensions: the Rust
# formulas are those of the Arm build, so a live comparison with NumPy is made there only. The
# frozen references were taken on Apple Arm (the system libm); the bit-for-bit comparison of their
# digests is made on that platform only, since the C library's cos, exp, log and pow differ in the
# last bit from one libm to another.
ARM_NUMPY = pytest.mark.skipif(platform.machine() not in ("arm64", "aarch64"),
                               reason="NumPy's x86 build uses other magnitude and sort kernels")
FROZEN_PLATFORM = pytest.mark.skipif(not (sys.platform == "darwin" and platform.machine() == "arm64"),
                                     reason="the references were frozen on Apple Arm (system libm)")

REFERENCE = json.loads((Path(__file__).parent / "reference" / "workflow_views.json").read_text())


@pytest.fixture(scope="module")
def current():
    return views()


@FROZEN_PLATFORM
@pytest.mark.parametrize("group", list(REFERENCE))
def test_the_views_are_the_frozen_ones_bit_for_bit(group, current):
    # json.dumps writes every float with its shortest round-trip form, NaN included.
    assert json.dumps(current[group], sort_keys=True) == json.dumps(REFERENCE[group], sort_keys=True)


@ARM_NUMPY
def test_the_complex_magnitude_is_numpys_for_every_length_and_stride():
    rng = np.random.default_rng(4)
    for n in (1, 2, 3, 7, 8, 9, 31, 1000):
        z = rng.standard_normal(n) * 10.0 ** rng.integers(-5, 5, n) + 1j * rng.standard_normal(n)
        assert np.array_equal(_native.spectrum_magnitudes(z), np.abs(z))
    special = np.array([0j, 1 + 0j, 1j, -3 + 4j, complex(np.inf, np.nan), complex(np.nan, 1), 1e300 + 1e300j,
                        1e-300 + 1e-300j])
    with np.errstate(all="ignore"):
        assert np.array_equal(_native.spectrum_magnitudes(special), np.abs(special), equal_nan=True)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_a_state_over_its_norm_is_numpys_complex_quotient(dtype):
    rng = np.random.default_rng(5)
    for n in (1, 4, 64):
        state = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
        norm = _native.state_norm(state)
        out = _native.divide_state(state, norm)
        assert out.dtype == dtype
        assert np.array_equal(out, state / norm)


@ARM_NUMPY
def test_the_ordering_of_equal_probabilities_is_numpys():
    rng = np.random.default_rng(6)
    for t in (2, 4, 5, 8, 10):
        for levels in (2, 5, 1000):
            p = rng.integers(0, levels, 2 ** t).astype(float)
            readout = _native.qpe_readout(p, t, -1.0, 2.0)
            order = np.argsort(p)[::-1][: max(8, 2 ** min(t, 4))]
            expected = [(readout["energies"][m], p[m]) for m in order if p[m] > 0]
            assert readout["peaks"] == [(float(e), float(q)) for e, q in expected]


def test_the_sum_over_the_ground_levels_is_numpys_pairwise_sum():
    rng = np.random.default_rng(7)
    for n in (1, 3, 8, 9, 127, 128, 129, 1000):
        overlaps = rng.random(n) * 10.0 ** rng.integers(-8, 3, n)
        exact = np.full(n, -1.0)
        assert _native.qpe_ground_overlap(exact, overlaps, True) == float(np.sum(overlaps))


@ARM_NUMPY
def test_the_spectrum_axes_are_numpys():
    rng = np.random.default_rng(8)
    for n in (4, 5, 64, 1001):
        dt = 0.1 + rng.random()
        spectrum = fourier_spectrum(rng.standard_normal(n) + 1j * rng.standard_normal(n), dt)
        assert np.array_equal(spectrum.times, np.arange(n) * dt)
        assert spectrum.resolution == 2 * np.pi / (n * dt)
        assert spectrum.nyquist_energy == np.pi / dt
        assert np.array_equal(spectrum.intensities, np.abs(spectrum.amplitudes))
