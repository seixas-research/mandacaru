"""The ``RealTime`` view: native unit conversions of a hand-built trace and the spectra computed
in Rust (``models::time_spectrum``), with no arithmetic of the view's own."""

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.analysis.real_time import RealTime


def trace(omega=0.5, weight=0.8, kappa=1e-3, dt=0.1, steps=20000):
    """A kicked two-level dipole: delta mu = kappa (f / omega) sin(omega t), atomic units."""
    times = np.arange(steps + 1) * dt
    along = kappa * weight / omega * np.sin(omega * times)
    zeros = np.zeros_like(times)
    return dict(
        times_au=times, dipole_au=np.stack([zeros, zeros, along], axis=1), along_au=along,
        momentum_au=np.zeros((steps + 1, 3)), energy_ha=zeros - 1.0, norm_error=zeros,
        ground_energy_ha=-1.0, kick_energy_ha=1e-7, energy_drift=0.0, width_ha=2.0,
        largest_width_ha=2.0, step_width=0.2, step_width_bound=4.0, aliased_weight=0.0,
        worst_norm_error=0.0, builds=steps, corrector_iterations=2, functional="lda",
        propagator="etrs", kick="density", gauge="length", strength_au=kappa,
        direction=[0.0, 0.0, 1.0], time_step_au=dt, steps=steps, corrector_tolerance=1e-6,
        effective_corrector_tolerance=1e-6,
        max_energy_drift=0.05)


def test_units_follow_the_calculator():
    context = _native.Context(threads=1)
    data = trace()
    ev = RealTime(data, atomic_units=False, context=context)
    au = RealTime(data, atomic_units=True, context=context)
    assert au.time_unit == "au" and ev.time_unit == "fs"
    assert ev.times[1] == pytest.approx(0.1 * 0.024188843265857)
    assert ev.width == pytest.approx(2.0 * 27.211386245988)
    assert ev.strength == pytest.approx(1e-3 / 0.529177210903)
    assert au.dipole.shape == (data["steps"] + 1, 3)


def test_the_strength_function_has_its_line_at_the_transition():
    context = _native.Context(threads=1)
    view = RealTime(trace(), atomic_units=True, context=context)
    energies = np.linspace(0.3, 0.7, 401)
    spectrum = view.strength_function(energies, damping=0.005)
    assert energies[int(np.argmax(spectrum))] == pytest.approx(0.5, abs=1e-3)
    alpha = view.polarizability(np.array([0.0]), damping=0.005)
    assert alpha[0].real == pytest.approx(0.8 / 0.25, rel=2e-3)
    with pytest.raises(ValueError, match="window"):
        view.strength_function(energies, window="hann")
