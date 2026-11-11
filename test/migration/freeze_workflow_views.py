"""Freeze the values the workflow result views derive before they move to Rust (VQ-8).

Run once, on the Python route, from the repository root:
``python test/migration/freeze_workflow_views.py``. It writes
``test/migration/reference/workflow_views.json``, which ``test_native_workflow_views.py`` reads
(and recomputes with ``views()``, so the comparison is bit for bit).

The cases: the spectra of random complex signals (magnitudes, bin spacing, Nyquist energy, peaks
at several thresholds, the sample times), the QPE readout (energies, phases, peaks of a
distribution with exact ties, the ground-state overlap of a degenerate spectrum with eight or
more levels, the collapsed state of complex64 and complex128 rows), the state-vector input of QPE
(normalized by the complex division NumPy uses), the real terms and norm bound of a Suzuki-Trotter
Hamiltonian, the echo's pulse bound and the real-part check of the MP2 integrals.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from mandacaru.analysis.correlation import _real
from mandacaru.core import PauliSum
from mandacaru.workflows.nested_otoc import OTOCSpectrum
from mandacaru.workflows.qpe import QPEResult, QuantumPhaseEstimation
from mandacaru.workflows.quantum_echoes import QuantumEchoes, fourier_spectrum
from mandacaru.workflows.time_evolution import SuzukiTrotter

OUT = Path(__file__).parent / "reference" / "workflow_views.json"


def listed(value):
    """An array as it is: a list up to 40 numbers, else a digest of every bit and its dtype."""
    array = np.ascontiguousarray(value)
    if array.size <= 40:
        return {"dtype": str(array.dtype), "real": array.real.tolist(),
                "imag": array.imag.tolist() if np.iscomplexobj(array) else None}
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def signal(n, seed, kind="random"):
    rng = np.random.default_rng(seed)
    if kind == "zero":
        return np.zeros(n, dtype=complex)
    if kind == "line":
        t = np.arange(n) * 0.3
        return 1.5 * np.exp(1j * 0.9 * t) + 0.5 * np.exp(1j * 1.7 * t)
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


def spectra():
    out = {}
    for n, seed, kind, window in ((64, 1, "random", "hann"), (33, 2, "random", "none"),
                                  (5, 3, "random", "hann"), (4, 4, "random", "none"),
                                  (96, 5, "line", "hann"), (16, 6, "zero", "hann"),
                                  (1024, 7, "random", "hann")):
        dt = 0.35 + 0.01 * seed
        spectrum = fourier_spectrum(signal(n, seed, kind), dt, elastic=0.1 * seed, window=window)
        record = {"times": listed(spectrum.times), "intensities": listed(spectrum.intensities),
                  "resolution": spectrum.resolution, "nyquist": spectrum.nyquist_energy,
                  "peaks": {}}
        for threshold in (0.0, 0.05, 0.5, 1.0):
            record["peaks"][str(threshold)] = listed(spectrum.peaks(threshold).astype(float))
        otoc = OTOCSpectrum(spectrum.energies, spectrum.amplitudes, spectrum.correlation,
                            spectrum.times, dt, window, 2)
        record["otoc"] = {"intensities": listed(otoc.intensities), "resolution": otoc.resolution,
                          "nyquist": otoc.nyquist_frequency}
        out[f"{kind}-{n}-{window}"] = record
    return out


def distribution(n, levels, seed):
    """Probabilities quantized to ``levels`` values: many exact ties and zeros."""
    rng = np.random.default_rng(seed)
    p = rng.integers(0, levels, n).astype(float)
    return p / p.sum()


def qpe_readout():
    out = {}
    for t, levels, seed in ((3, 3, 1), (4, 4, 2), (6, 3, 3), (8, 4, 4), (8, 1000, 5), (9, 5, 6)):
        n = 2 ** t
        probabilities = distribution(n, levels, seed)
        rng = np.random.default_rng(seed + 10)
        exact = np.sort(np.concatenate([np.full(9, -1.25), -1.25 + 1e-11 * rng.random(3),
                                        rng.random(7) - 1.0, [-1.25 + 2e-10]]))
        exact = exact[:12 + seed]
        overlaps = rng.random(exact.size)
        amplitudes = (rng.standard_normal((n, 4)) + 1j * rng.standard_normal((n, 4)))
        for dtype in (np.complex128, np.complex64):
            for unit in ("Ha", "eV"):
                result = QPEResult(
                    energies=np.linspace(-2, 1, n), probabilities=probabilities,
                    phases=np.arange(n) / n, resolution=0.1, energy_window=(-2.0, 1.0),
                    n_system_qubits=2, n_evaluation_qubits=t, peaks=[(-1.0, 0.5)],
                    exact_energies=exact, overlaps=overlaps, energy_unit=unit,
                    _amplitudes=amplitudes.astype(dtype))
                index = int(np.argmax(probabilities))
                key = f"{t}-{levels}-{np.dtype(dtype).name}-{unit}"
                out[key] = {"overlap": result.ground_state_overlap,
                            "collapsed": listed(result.collapsed_state()),
                            "collapsed_at": listed(result.collapsed_state(index // 2 + 1)),
                            "summary": result.summary(), "exact": listed(result.exact_energies)}
    return out


def qpe_runs():
    out = {}
    cases = (({"ZI": 0.5, "IZ": 0.5, "XX": 0.1}, 3), ({"ZI": 0.5, "IZ": -0.5}, 4),
             ({"ZZ": 0.7, "XI": 0.2, "IX": 0.2}, 6), ({"ZI": 0.4, "IZ": 0.25, "YY": 0.1}, 8))
    rng = np.random.default_rng(21)
    for index, (terms, t) in enumerate(cases):
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        for precision in ("float64", "float32"):
            for atomic in (True, False):
                qpe = QuantumPhaseEstimation(PauliSum(terms), n_evaluation_qubits=t,
                                             atomic_units=atomic, precision=precision,
                                             verbose=False)
                result = qpe.run(psi)
                out[f"{index}-{precision}-{atomic}"] = {
                    "energies": listed(result.energies), "phases": listed(result.phases),
                    "probabilities": listed(result.probabilities),
                    "resolution": result.resolution, "window": list(result.energy_window),
                    "peaks": [list(p) for p in result.peaks],
                    "overlap": result.ground_state_overlap,
                    "collapsed": listed(result.collapsed_state()),
                    "summary": result.summary()}
        for size, dtype in ((4, np.complex128), (4, np.complex64), (8, np.complex128),
                            (16, np.complex64), (2, np.complex128), (1, np.complex128)):
            raw = rng.standard_normal(size) + 1j * rng.standard_normal(size)
            qpe = QuantumPhaseEstimation(PauliSum({"I" * max(size.bit_length() - 1, 1): 1.0}),
                                         precision="float32" if dtype == np.complex64 else "float64",
                                         verbose=False)
            psi = qpe._resolve_input(raw.astype(dtype))[0]
            out[f"input-{index}-{size}-{np.dtype(dtype).name}"] = listed(psi)
    return out


def trotter_terms():
    out = {}
    hamiltonians = ({"XX": 0.5, "ZI": 0.3, "IZ": -0.2, "YY": 0.1},
                    {"XX": 0.5, "ZI": 0.0, "IZ": -0.2 + 1e-13j, "YY": 0.1 - 0.3},
                    {"ZZ": 1e-300, "XI": 1e300, "IZ": 1e300},
                    {"XX": 3.0, "ZI": -4.5, "II": 0.25})
    for index, terms in enumerate(hamiltonians):
        trotter = SuzukiTrotter(PauliSum(terms))
        out[str(index)] = {"terms": [[label, c] for label, c in trotter._terms],
                           "norm_bound": trotter.norm_bound}
    for index, terms in enumerate(({"XX": 0.5, "ZI": 1e-10j}, {"XX": float("nan")},
                                   {"XX": float("inf")}, {"XX": 1.7e308, "YY": 1.7e308})):
        try:
            SuzukiTrotter(PauliSum(terms))
            out[f"error-{index}"] = None
        except Exception as error:
            out[f"error-{index}"] = [type(error).__name__, str(error)]
    return out


def echo_pulse():
    out = {}
    echoes = QuantumEchoes(PauliSum({"XX": 0.5, "ZI": 0.3, "IZ": -0.2, "YY": 0.1}),
                           PauliSum({"ZI": 0.4, "IZ": 0.25, "XX": 0.05}))
    psi = np.array([0.6, 0.3 + 0.2j, -0.1j, 0.5], dtype=complex)
    psi /= np.linalg.norm(psi)
    for tau_p in (1e-3, -2.5e-2, 0.0, 0.15):
        try:
            r = echoes.run(psi, 1.1, tau_p=tau_p, steps=3)
            out[str(tau_p)] = {"bound": r.perturbation_bound, "fidelity": r.fidelity,
                               "response": r.response, "amplitude": [r.amplitude.real, r.amplitude.imag]}
        except ValueError as error:
            out[str(tau_p)] = str(error)
    s = echoes.spectrum(psi, time_step=0.3, num_samples=9, steps_per_sample=2)
    out["elastic"] = s.elastic
    return out


def real_part():
    out = {}
    rng = np.random.default_rng(8)
    for name, array in (("real", rng.standard_normal((3, 4))),
                        ("tiny", rng.standard_normal((2, 3)) + 1e-9j * rng.standard_normal((2, 3))),
                        ("empty", np.zeros((0, 2), dtype=complex)),
                        ("big", rng.standard_normal(5) + 1e-7j * rng.standard_normal(5)),
                        ("nan", np.array([1.0 + np.nan * 1j, 2.0]))):
        try:
            out[name] = listed(_real(array, "block"))
        except ValueError as error:
            out[name] = str(error)
    return out


def views():
    return {"spectra": spectra(), "qpe_readout": qpe_readout(), "qpe_runs": qpe_runs(),
            "trotter_terms": trotter_terms(), "echo_pulse": echo_pulse(), "real_part": real_part()}


if __name__ == "__main__":
    OUT.write_text(json.dumps(views()))
    print(f"wrote {OUT}")
