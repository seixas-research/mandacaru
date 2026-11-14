"""Unit-aware view over a native real-time propagation (``molecular::realtime``)."""

import numpy as np
from .. import _native


class RealTime:
    """The trace of a molecule propagated in real time after a delta kick.

    Returned by :meth:`Mandacaru.get_real_time`. Every number comes from Rust
    (``molecular::realtime`` and ``models::time_spectrum``); the view converts
    units through the native conversions and holds no arithmetic of its own.

    Attributes
    ----------
    times : ndarray
        ``(nt,)`` sample times from the kick, fs (atomic time units with
        ``atomic_units=True``).
    dipole : ndarray
        ``(nt, 3)`` induced electronic dipole, e Angstrom (e Bohr); zero at the
        kick. For ``kick="spin"`` the spin dipole ``-tr[(D_up - D_down) r]``.
    along : ndarray
        ``(nt,)`` the induced dipole along the kick direction: the signal of the spectra.
    momentum : ndarray
        ``(nt, 3)`` the electrons' kinetic momentum ``sum tr[D (p + A)]``,
        1/Angstrom (1/Bohr; ``hbar = 1``).
    energy : ndarray
        ``(nt,)`` total electronic energy from the propagated state's builder, eV (Ha).
    norm_error : ndarray
        ``(nt,)`` ``max |C^dagger C - 1|`` of the propagated orbitals.
    kick_energy, energy_drift : float
        The energy the kick put in (eV; Ha) and the largest deviation of the
        energy from its value after the kick, in units of the kick energy.
    width, step_width, step_width_bound : float
        The basis width ``W = e_max - e_min`` of the reference Fock matrix (eV;
        Ha), ``dt W`` and the propagator's bound on it.
    aliased_weight : float
        The bare Kohn-Sham oscillator weight along the kick above the Nyquist
        frequency ``pi / dt``, as a fraction of the bare total (a proxy:
        interacting roots can lie above it).
    ground_energy : float
        The energy of the unkicked reference from the same builder, eV (Ha).
    worst_norm_error : float
        The largest entry of ``norm_error``.
    builds, corrector_iterations : int
        The Fock builds of the run (the initial ones included) and the largest
        number of corrector iterations in one step.
    propagator, kick, gauge : str
        The request: ``"etrs"``, ``"aetrs"`` or ``"midpoint"``; ``"density"`` or
        ``"spin"``; ``"length"`` or ``"velocity"``.
    direction : tuple of float
        The unit kick direction.
    strength : float
        The kick strength, 1/Angstrom (1/Bohr).
    time_step : float
        The time step, fs (atomic time units).
    steps : int
        The number of steps; ``times`` has ``steps + 1`` entries.
    corrector_tolerance, effective_corrector_tolerance : float
        The requested ETRS corrector tolerance, which holds at the default kick of
        0.002 / Angstrom, and the one applied: the requested one times the kick
        strength over the default, because the induced density scales with the kick.
    max_energy_drift : float or None
        The largest energy drift the run accepted, in kick energies (``None``: not checked).
    """

    def __init__(self, data, *, atomic_units, context):
        self._data = data
        self._context = context
        self._atomic_units = atomic_units
        #: The reference's functional; ``None`` for Hartree-Fock (TDHF).
        self.functional = data["functional"]
        self.energy_unit = "Ha" if atomic_units else "eV"
        self.length_unit = "Bohr" if atomic_units else "Angstrom"
        self.time_unit = "au" if atomic_units else "fs"
        self.times = self._convert(data["times_au"], "time")
        self.dipole = self._convert(data["dipole_au"], "length")
        self.along = self._convert(data["along_au"], "length")
        self.momentum = self._convert(data["momentum_au"], "wavenumber")
        self.energy = self._convert(data["energy_ha"], "energy")
        self.norm_error = np.asarray(data["norm_error"])
        self.ground_energy = self._convert(data["ground_energy_ha"], "energy")
        self.kick_energy = self._convert(data["kick_energy_ha"], "energy")
        self.energy_drift = data["energy_drift"]
        self.width = self._convert(data["width_ha"], "energy")
        self.step_width = data["step_width"]
        self.step_width_bound = data["step_width_bound"]
        self.aliased_weight = data["aliased_weight"]
        self.worst_norm_error = data["worst_norm_error"]
        self.builds = data["builds"]
        self.corrector_iterations = data["corrector_iterations"]
        self.propagator = data["propagator"]
        self.kick = data["kick"]
        self.gauge = data["gauge"]
        self.direction = tuple(data["direction"])
        self.strength = self._convert(data["strength_au"], "wavenumber")
        self.time_step = self._convert(data["time_step_au"], "time")
        self.steps = data["steps"]
        self.corrector_tolerance = data["corrector_tolerance"]
        self.effective_corrector_tolerance = data["effective_corrector_tolerance"]
        self.max_energy_drift = data["max_energy_drift"]

    def _convert(self, values, quantity):
        if np.ndim(values) == 0:
            return _native.to_user_units(float(values), quantity,
                                         atomic_units=self._atomic_units)
        return _native.to_user_units(np.ascontiguousarray(values, dtype=float), quantity,
                                     atomic_units=self._atomic_units)

    def _spectrum(self, energies, damping, window, kind):
        values, _ = _native.time_spectrum(
            self._context, np.ascontiguousarray(self._data["along_au"], dtype=float),
            np.ascontiguousarray(energies, dtype=float),
            time_step=float(self._data["time_step_au"]),
            strength=float(self._data["strength_au"]), damping=float(damping),
            window=window, kind=kind, atomic_units=self._atomic_units)
        return np.asarray(values)

    def strength_function(self, energies, damping=0.1, window="exponential"):
        """The dipole strength function ``S(E) = (2 E / pi) Im alpha(E)`` along the kick on
        ``energies`` (the view's energy unit), in 1/eV (1/Ha): its lines have the areas of the
        oscillator strengths along the kick. ``window`` is ``"exponential"`` (Lorentzian lines of
        half width ``damping``) or ``"gaussian"`` (Gaussian lines of standard deviation
        ``damping``). Computed in Rust."""
        return self._spectrum(energies, damping, window, "strength")

    def polarizability(self, energies, damping=0.1, window="exponential"):
        """The complex dynamic polarizability along the kick on ``energies``, Angstrom^3
        (Bohr^3); with the exponential window it is ``alpha(E + i damping)``. Computed in
        Rust."""
        return self._spectrum(energies, damping, window, "polarizability")
