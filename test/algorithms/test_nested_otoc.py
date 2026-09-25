"""Paper Eq. (1), state/readout semantics, circuits, and multi-gap spectra."""

import numpy as np
import pytest
from scipy.linalg import expm

from mandacaru.algorithms import NestedOTOC, SuzukiTrotter
from mandacaru.core import PauliSum, WavefunctionCheckpoint


def model():
    return NestedOTOC(PauliSum({"XX": 0.5}), PauliSum({"IZ": 1}), PauliSum({"ZI": 1}))


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_equation_one_for_generic_complex_state(k, order):
    h = PauliSum({"YX": 0.7, "ZI": -0.4, "II": 0.8})
    b, m = PauliSum({"IZ": -1}), PauliSum({"XI": 1})
    psi = np.array([1, 2j, -1j, 3]) / np.sqrt(15)
    saved = psi.copy()
    echo = NestedOTOC(h, b, m, trotter_order=order)
    result = echo.run(psi, 0.7, otoc_order=k, steps=4)
    # Build the discrete U independently of nesting; includes the nonsymmetric
    # first-order case, where reversing time alone is not its exact adjoint.
    propagation = SuzukiTrotter(h, order=order)
    u = np.column_stack([propagation.evolve(v, 0.7, steps=4) for v in np.eye(4)])
    bt, mm = u.conj().T @ b.to_matrix() @ u, m.to_matrix()
    uk = bt @ np.linalg.matrix_power(mm @ bt, k - 1)
    expected = np.vdot(psi, np.linalg.matrix_power(bt @ mm, 2*k) @ psi)
    np.testing.assert_allclose(result.state, uk @ psi, atol=2e-13)
    assert result.correlator == pytest.approx(expected, abs=3e-13)
    assert result.measurement_expectation == pytest.approx(
        np.vdot(uk @ psi, mm @ uk @ psi).real, abs=2e-13)
    assert result.initial_measurement_eigenvalue is None
    assert abs(expected.imag) > 1e-3  # Do not accidentally discard its phase.
    np.testing.assert_array_equal(psi, saved)


def test_continuous_hamiltonian_reference_and_energy_offset():
    h = PauliSum({"XX": 0.7, "ZI": -0.4})
    shifted = PauliSum({"XX": 0.7, "ZI": -0.4, "II": 12.0})
    b, m = PauliSum({"IZ": 1}), PauliSum({"XI": 1})
    psi = np.array([1, 2j, -1j, 3]) / np.sqrt(15)
    u = expm(-0.7j*h.to_matrix())
    bt = u.conj().T @ b.to_matrix() @ u
    expected = np.vdot(psi, np.linalg.matrix_power(bt @ m.to_matrix(), 4) @ psi)
    for operator in (h, shifted):
        result = NestedOTOC(operator, b, m, trotter_order=4).run(
            psi, 0.7, otoc_order=2, steps=16)
        assert result.correlator == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("index,sign", [(0, 1), (2, -1)])
def test_direct_readout_and_signed_eigenstate(k, index, sign):
    psi = np.eye(4)[index]
    for time in (0, np.pi/8, np.pi/4, -0.2):
        result = model().run(psi, time, otoc_order=k)
        assert result.correlator == pytest.approx(np.cos(2*k*time), abs=1e-13)
        assert result.initial_measurement_eigenvalue == sign
        assert result.correlator == pytest.approx(sign * result.measurement_expectation)
        assert np.linalg.norm(result.state) == pytest.approx(1.0, abs=1e-13)


def test_initial_state_average_is_normalized_trace():
    echo = model()
    value = sum(echo.run(psi, 0.37, otoc_order=2).correlator for psi in np.eye(4))/4
    assert value == pytest.approx(np.cos(4*0.37), abs=1e-13)


def test_checkpoint_and_matrix_free_execution(tmp_path, monkeypatch):
    ck = WavefunctionCheckpoint(2, [0], [PauliSum({"XY": 0.4j})], np.array([0.3]))
    path = tmp_path / "state.json"
    ck.save(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dense/sparse operator materialized")

    echo = model()
    psi = ck.state_vector()  # Existing checkpoint preparation may use sparse matrices.
    expected = echo.run(psi, 0.4)
    for initial in (ck, path, str(path)):
        result = echo.run(initial, 0.4)
        np.testing.assert_allclose(result.state, expected.state, atol=1e-13)
        assert result.correlator == pytest.approx(expected.correlator)
    monkeypatch.setattr(PauliSum, "to_matrix", forbidden)
    monkeypatch.setattr(PauliSum, "to_sparse_matrix", forbidden)
    assert echo.run(psi, 0.4).correlator == pytest.approx(expected.correlator)
    assert echo.spectrum(psi, num_samples=8).correlation.shape == (8,)


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_qiskit_sequence_matches_native(order, k):
    from qiskit.quantum_info import Statevector

    echo = NestedOTOC(PauliSum({"YX": 0.7, "ZI": -0.4, "II": 0.8}),
                      PauliSum({"IZ": -1}), PauliSum({"XI": -1}), trotter_order=order)
    psi = np.array([1, 2j, -1j, 3])/np.sqrt(15)
    circuit = echo.circuit(0.7, otoc_order=k, steps=3).decompose(reps=2)
    actual = Statevector(psi).evolve(circuit).data
    np.testing.assert_allclose(actual, echo.run(psi, 0.7, otoc_order=k, steps=3).state,
                               atol=2e-13)


@pytest.mark.parametrize("k", [1, 2])
def test_otoc_fourier_peaks_are_harmonics_not_hamiltonian_gaps(k, tmp_path):
    # H=XX/2 has gap 1 Ha; C^(2k)=cos(2*k*t) has frequencies +/-2*k.
    spectrum = model().spectrum([1, 0, 0, 0], time_step=np.pi/16,
                                num_samples=64, otoc_order=k, window="none")
    np.testing.assert_allclose(spectrum.correlation, np.cos(2*k*spectrum.times), atol=1e-12)
    peaks = np.sort(np.argpartition(spectrum.intensities, -2)[-2:])
    np.testing.assert_allclose(spectrum.frequencies[peaks], [-2*k, 2*k], atol=1e-13)
    np.testing.assert_allclose(spectrum.intensities[peaks], 64*(np.pi/16)/2, atol=1e-11)
    assert spectrum.resolution == pytest.approx(0.5)
    assert spectrum.nyquist_frequency == pytest.approx(16)
    path = spectrum.to_csv(tmp_path / "otoc.csv")
    csv = np.genfromtxt(path, delimiter=",", names=True)
    np.testing.assert_allclose(csv["angular_frequency_ev"], spectrum.frequencies_ev)
    np.testing.assert_allclose(csv["fft_real_au_time"] + 1j*csv["fft_imag_au_time"],
                               spectrum.amplitudes, atol=1e-11)


def test_spectrum_noncommuting_hamiltonian_and_window():
    h = PauliSum({"XI": 0.3, "ZZ": 0.7})
    b, m = PauliSum({"IZ": 1}), PauliSum({"XI": 1})
    psi = np.array([1, 2j, -1j, 3])/np.sqrt(15)
    spectrum = NestedOTOC(h, b, m, trotter_order=4).spectrum(
        psi, time_step=0.2, num_samples=16, steps_per_sample=4)
    exact = []
    for time in spectrum.times:
        u = expm(-1j*time*h.to_matrix())
        bt = u.conj().T @ b.to_matrix() @ u
        exact.append(np.vdot(psi, np.linalg.matrix_power(bt @ m.to_matrix(), 4) @ psi))
    np.testing.assert_allclose(spectrum.correlation, exact, atol=2e-8)
    np.testing.assert_allclose(spectrum.amplitudes,
        0.2*np.fft.fftshift(np.fft.fft(np.hanning(16)*exact)), atol=3e-8)


@pytest.mark.parametrize("operator", [PauliSum({"IZ": 0.2}), PauliSum({"IZ": 1j}),
    PauliSum({"IZ": 0.5, "XI": 0.5}), PauliSum(num_qubits=2), PauliSum({"Z": 1})])
def test_invalid_insertions(operator):
    with pytest.raises(ValueError):
        NestedOTOC(PauliSum({"XX": 0.5}), operator, PauliSum({"ZI": 1}))
    with pytest.raises(ValueError):
        NestedOTOC(PauliSum({"XX": 0.5}), PauliSum({"ZI": 1}), operator)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_otoc_order(k):
    echo = model()
    with pytest.raises(ValueError, match="otoc_order"):
        echo.run([1, 0, 0, 0], 0.3, otoc_order=k)
    with pytest.raises(ValueError, match="otoc_order"):
        echo.spectrum([1, 0, 0, 0], otoc_order=k, num_samples=4)
    with pytest.raises(ValueError, match="otoc_order"):
        echo.circuit(0.3, otoc_order=k)


@pytest.mark.parametrize("options", [{"time_step": 0}, {"time_step": np.nan},
    {"num_samples": 3}, {"num_samples": 4.5}, {"steps_per_sample": 0}, {"window": "bad"}])
def test_invalid_sampling(options):
    with pytest.raises(ValueError):
        model().spectrum([1, 0, 0, 0], **options)
