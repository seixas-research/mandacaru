"""Fourier sign, energy units, resolution, and incremental echo correlations."""

import numpy as np
import pytest
from scipy.linalg import expm

from mandacaru.algorithms import QuantumEchoes, fourier_spectrum
from mandacaru.core import PauliSum
from mandacaru.units import HARTREE_TO_EV


def test_fourier_resolves_known_lines_and_elastic_without_shifting_sign():
    n, dt = 128, 0.4
    times = np.arange(n)*dt
    spacing = 2*np.pi/(n*dt)
    gap1, gap2 = 7*spacing, 19*spacing
    signal = 0.3 + 2*np.exp(1j*gap1*times) + 0.8*np.exp(1j*gap2*times)
    saved = signal.copy()
    spectrum = fourier_spectrum(signal, dt, elastic=0.3, window="none")
    peaks = spectrum.peaks()
    np.testing.assert_allclose(spectrum.energies[peaks], [gap1, gap2])
    np.testing.assert_allclose(spectrum.intensities[peaks], np.array([2, 0.8])*n*dt)
    np.testing.assert_allclose(spectrum.energies_ev[peaks], np.array([gap1, gap2])*HARTREE_TO_EV)
    assert spectrum.resolution == pytest.approx(spacing)
    assert spectrum.nyquist_energy == pytest.approx(np.pi/dt)
    assert abs(spectrum.amplitudes[n//2]) < 1e-12
    np.testing.assert_array_equal(signal, saved)


def test_hann_window_resolution_and_negative_frequencies():
    n, dt = 256, 0.5
    times = np.arange(n)*dt
    spectrum = fourier_spectrum(np.exp(1j*0.63*times), dt)
    peak = spectrum.peaks()[0]
    assert abs(spectrum.energies[peak] - 0.63) <= spectrum.resolution/2
    negative = fourier_spectrum(np.exp(-1j*0.63*times), dt)
    assert negative.energies[np.argmax(negative.intensities)] < 0
    assert fourier_spectrum(np.zeros(n), dt).peaks().size == 0


def test_incremental_correlation_matches_dense_evolution_for_generic_state():
    h = PauliSum({"X": 0.2, "Z": 0.4, "I": 0.6})
    v = PauliSum({"Y": 0.1, "Z": 0.3, "I": -0.2})
    psi = np.array([1, 2j])/np.sqrt(5)
    echoes = QuantumEchoes(h, v, order=4)
    spectrum = echoes.spectrum(psi, time_step=0.1, num_samples=24, steps_per_sample=4)
    hm, vm = h.to_matrix(), v.to_matrix()
    exact = []
    for t in spectrum.times:
        u = expm(-1j*t*hm)
        exact.append(np.vdot(psi, vm @ u.conj().T @ vm @ u @ psi))
    np.testing.assert_allclose(spectrum.correlation, exact, rtol=0, atol=1e-10)
    assert spectrum.elastic == pytest.approx(np.vdot(psi, vm @ psi).real**2)
    expected_fft = 0.1*np.fft.fftshift(np.fft.fft(
        np.hanning(24)*(np.array(exact) - spectrum.elastic)))
    np.testing.assert_allclose(spectrum.amplitudes, expected_fft, atol=1e-10)


def test_spectrum_uses_linear_response_without_dense_matrices(monkeypatch):
    gap, coupling = 0.7, 0.3
    echo = QuantumEchoes(PauliSum({"Z": -gap/2}), PauliSum({"X": coupling}))

    def forbidden(*args, **kwargs):
        raise AssertionError("dense/sparse matrix materialized")

    monkeypatch.setattr(PauliSum, "to_matrix", forbidden)
    monkeypatch.setattr(PauliSum, "to_sparse_matrix", forbidden)
    spectrum = echo.spectrum([1, 0], num_samples=256, steps_per_sample=1)
    np.testing.assert_allclose(spectrum.correlation,
                               coupling**2*np.exp(1j*gap*spectrum.times), atol=1e-12)
    assert abs(spectrum.energies[spectrum.peaks()[0]] - gap) < spectrum.resolution


@pytest.mark.parametrize("signal", [[1, 2, 3], [[1]*4], [1, 2, np.nan, 3]])
def test_invalid_correlations(signal):
    with pytest.raises(ValueError, match="correlation"):
        fourier_spectrum(signal, 0.5)


@pytest.mark.parametrize("dt", [0, -1, np.inf, 1j])
def test_invalid_time_steps(dt):
    with pytest.raises(ValueError, match="time_step"):
        fourier_spectrum([1]*4, dt)


@pytest.mark.parametrize("options", [{"num_samples": 3}, {"num_samples": 5.5},
                                     {"steps_per_sample": 0}, {"window": "invalid"}])
def test_invalid_sampling_options(options):
    echo = QuantumEchoes(PauliSum({"Z": 1}), PauliSum({"X": 0.1}))
    with pytest.raises(ValueError):
        echo.spectrum([1, 0], **options)
