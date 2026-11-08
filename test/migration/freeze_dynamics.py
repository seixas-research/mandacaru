"""Freeze the echo and nested-OTOC numbers before their loops move to Rust (MG-Q4).

Run once, on the Python route, from the repository root:
``python test/migration/freeze_dynamics.py``. It writes
``test/migration/reference/dynamics.json``, which ``test_native_dynamics.py`` reads.
"""

import json
from pathlib import Path

import numpy as np

from mandacaru.core import PauliSum
from mandacaru.workflows.nested_otoc import NestedOTOC
from mandacaru.workflows.quantum_echoes import QuantumEchoes, fourier_spectrum

OUT = Path(__file__).parent / "reference" / "dynamics.json"
H = {"XX": 0.5, "ZI": 0.3, "IZ": -0.2, "YY": 0.1}
V = {"ZI": 0.4, "IZ": 0.25}


def pair(z):
    z = np.asarray(z, dtype=complex)
    return [z.real.tolist(), z.imag.tolist()]


def state():
    psi = np.array([0.6, 0.3 + 0.2j, -0.1j, 0.5], dtype=complex)
    return psi / np.linalg.norm(psi)


def main():
    out = {"hamiltonian": H, "perturbation": V, "state": pair(state())}
    n, dt = 9, 0.4
    times = np.arange(n) * dt
    signal = 0.3 + 2 * np.exp(1j * 1.1 * times) + 0.8 * np.exp(-1j * 0.5 * times)
    out["fourier"] = {"signal": pair(signal), "dt": dt, "elastic": 0.3, "cases": {}}
    for window in ("hann", "none"):
        s = fourier_spectrum(signal, dt, elastic=0.3, window=window)
        out["fourier"]["cases"][window] = {"energies": s.energies.tolist(),
                                           "amplitudes": pair(s.amplitudes)}
    echoes = QuantumEchoes(PauliSum(H), PauliSum(V))
    out["echo_spectrum"] = {}
    for precision, echo in (("float64", echoes),
                            ("float32", QuantumEchoes(PauliSum(H), PauliSum(V), precision="float32"))):
        s = echo.spectrum(state(), time_step=0.3, num_samples=8, steps_per_sample=3)
        out["echo_spectrum"][precision] = {"correlation": pair(s.correlation),
                                           "amplitudes": pair(s.amplitudes),
                                           "elastic": float(s.elastic)}
    r = echoes.run(state(), 1.3, tau_p=1e-3, steps=4, kick_steps=2)
    out["echo_run"] = {"amplitude": pair([r.amplitude]), "response": float(r.response),
                       "correlation": pair([r.correlation])}
    otoc = NestedOTOC(PauliSum(H), PauliSum({"IZ": 1}), PauliSum({"ZI": 1}))
    out["otoc_run"] = {}
    for label, psi in (("generic", state()),
                       ("eigenstate", np.array([1, 0, 0, 0], dtype=complex))):
        for k in (1, 2):
            r = otoc.run(psi, 0.7, otoc_order=k, steps=3)
            out["otoc_run"][f"{label}-{k}"] = {"correlator": pair([r.correlator]),
                                               "measured": float(r.measurement_expectation),
                                               "eigenvalue": r.initial_measurement_eigenvalue,
                                               "state": pair(r.state)}
    s = otoc.spectrum(state(), time_step=0.4, num_samples=6, steps_per_sample=2, otoc_order=2)
    out["otoc_spectrum"] = {"correlation": pair(s.correlation), "amplitudes": pair(s.amplitudes),
                            "frequencies": s.frequencies.tolist()}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
