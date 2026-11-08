"""The echo and nested-OTOC numbers after their loops moved to Rust (MG-Q4).

``reference/dynamics.json`` was written by ``freeze_dynamics.py`` on the Python
route (NumPy FFT, Python sampling loops) before the port; the native route must
reproduce it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru.core import PauliSum
from mandacaru.workflows.nested_otoc import NestedOTOC
from mandacaru.workflows.quantum_echoes import QuantumEchoes, fourier_spectrum

REFERENCE = json.loads((Path(__file__).parent / "reference" / "dynamics.json").read_text())
TOL = {"float64": 1e-12, "float32": 2e-6}


def values(pair):
    return np.asarray(pair[0]) + 1j * np.asarray(pair[1])


def state():
    return values(REFERENCE["state"])


@pytest.mark.parametrize("window", ["hann", "none"])
def test_the_fourier_spectrum_reproduces_the_numpy_transform(window):
    frozen = REFERENCE["fourier"]
    s = fourier_spectrum(values(frozen["signal"]), frozen["dt"], elastic=frozen["elastic"],
                         window=window)
    case = frozen["cases"][window]
    np.testing.assert_allclose(s.energies, case["energies"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(s.amplitudes, values(case["amplitudes"]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_the_echo_spectrum_reproduces_the_sampling_loop(precision):
    echo = QuantumEchoes(PauliSum(REFERENCE["hamiltonian"]), PauliSum(REFERENCE["perturbation"]),
                         precision=precision)
    s = echo.spectrum(state(), time_step=0.3, num_samples=8, steps_per_sample=3)
    frozen = REFERENCE["echo_spectrum"][precision]
    tol = TOL[precision]
    np.testing.assert_allclose(s.correlation, values(frozen["correlation"]), rtol=0, atol=tol)
    np.testing.assert_allclose(s.amplitudes, values(frozen["amplitudes"]), rtol=0, atol=10 * tol)
    assert s.elastic == pytest.approx(frozen["elastic"], abs=tol)


def test_the_finite_pulse_echo_is_unchanged():
    echo = QuantumEchoes(PauliSum(REFERENCE["hamiltonian"]), PauliSum(REFERENCE["perturbation"]))
    r = echo.run(state(), 1.3, tau_p=1e-3, steps=4, kick_steps=2)
    frozen = REFERENCE["echo_run"]
    assert r.amplitude == pytest.approx(values(frozen["amplitude"])[0], abs=1e-12)
    assert r.response == pytest.approx(frozen["response"], abs=1e-12)
    assert r.correlation == pytest.approx(values(frozen["correlation"])[0], abs=1e-12)


def otoc():
    return NestedOTOC(PauliSum(REFERENCE["hamiltonian"]), PauliSum({"IZ": 1}), PauliSum({"ZI": 1}))


@pytest.mark.parametrize("label", ["generic", "eigenstate"])
@pytest.mark.parametrize("k", [1, 2])
def test_the_nested_otoc_reproduces_both_branch_forms(label, k):
    psi = state() if label == "generic" else np.array([1, 0, 0, 0], dtype=complex)
    r = otoc().run(psi, 0.7, otoc_order=k, steps=3)
    frozen = REFERENCE["otoc_run"][f"{label}-{k}"]
    assert r.initial_measurement_eigenvalue == frozen["eigenvalue"]
    assert r.correlator == pytest.approx(values(frozen["correlator"])[0], abs=1e-12)
    assert r.measurement_expectation == pytest.approx(frozen["measured"], abs=1e-12)
    np.testing.assert_allclose(r.state, values(frozen["state"]), rtol=0, atol=1e-12)


def test_the_otoc_spectrum_reproduces_the_sampling_loop():
    s = otoc().spectrum(state(), time_step=0.4, num_samples=6, steps_per_sample=2, otoc_order=2)
    frozen = REFERENCE["otoc_spectrum"]
    np.testing.assert_allclose(s.correlation, values(frozen["correlation"]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(s.amplitudes, values(frozen["amplitudes"]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(s.frequencies, frozen["frequencies"], rtol=0, atol=1e-13)
