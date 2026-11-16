"""Unit-aware view over a native real-time propagation (``molecular::realtime`` for a molecule,
``periodic::optics::realtime`` for a crystal)."""

import numpy as np
from .. import _native


class RealTime:
    """The trace of a molecule or a crystal propagated in real time after a kick.

    Returned by :meth:`Mandacaru.get_real_time`. Every number comes from Rust
    (``molecular::realtime``, ``periodic::optics::realtime`` and
    ``models::time_spectrum``); the view converts units through the native
    conversions and holds no arithmetic of its own. A molecule fills ``dipole``,
    ``along`` and ``momentum`` and has ``strength_function`` and
    ``polarizability``; a crystal (``periodic`` true) fills ``current``,
    ``along`` (the current along the field), ``corrected`` and the diamagnetic
    record, and has ``dielectric_function``. The other attributes are ``None``.

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
        The unit kick direction; for a crystal run with ``direction="all"`` the flattened
        3 x 3 identity (one run along each axis).
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
    periodic : bool
        Whether the run propagated a crystal.
    current : ndarray or None
        Crystal: ``(nt, 3)`` the macroscopic current density
        ``j = -(1 / Omega) sum_k w_k tr[D (P + A)]``, e / (Angstrom^2 fs) (atomic units).
    corrected : ndarray or None
        Crystal: ``(nt,)`` the current along the field without the basis's static deficit,
        ``j + A (N - N_dia) / Omega``: the signal of ``dielectric_function``, which returns the
        complex ``epsilon(E + i damping)`` along the field. With ``direction="all"`` the
        ``(nt, 3, 3)`` tensor ``[t, a, b]`` (the current along ``a`` of the run along ``b``),
        ``current`` is ``(nt, 3, 3)`` too, ``along``, ``energy`` and ``norm_error`` are
        ``(nt, 3)``, and ``dielectric_function`` returns the ``(n, 3, 3)`` tensor.
    electrons : float or None
        Crystal: the valence electrons ``N`` of the cell.
    diamagnetic, deficit : list of float or None
        Crystal: per run (one, or three for ``direction="all"``) the static paramagnetic
        response ``N_dia`` of the finite basis along the field and ``N - N_dia``.
    diamagnetic_source : str or None
        Crystal: ``"band sum"`` (read from the ground-state bands, exact for a density-only
        kernel) or ``"static field"`` (DFT+U and meta-GGAs, whose kernels respond to the
        current: measured before the run by self-consistency at a static field of the kick's
        strength, both signs).
    momentum_residual : float or None
        Crystal: the largest anti-Hermitian part of the momentum matrix before it was
        Hermitized, relative to its largest element.
    kernel : str or None
        Crystal: ``"adiabatic"`` (the functional's own Hamiltonian) or ``"independent"`` (frozen
        at the ground state).
    mesh : tuple of int or None
        Crystal: the whole Gamma-anchored propagation mesh; ``kpoints`` its point count.
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
        self.periodic = "current_au" in data
        # The record as the binding returned it (counts, ratios, names; None where the other
        # kind of system has the field).
        for name in ("energy_drift", "step_width", "step_width_bound", "aliased_weight",
                     "worst_norm_error", "builds", "corrector_iterations", "propagator", "kick",
                     "gauge", "steps", "corrector_tolerance", "effective_corrector_tolerance",
                     "max_energy_drift", "electrons", "diamagnetic", "deficit",
                     "momentum_residual", "kernel", "mesh", "kpoints", "diamagnetic_source"):
            setattr(self, name, data.get(name))
        # The quantities with a unit, from the binding's atomic-unit keys.
        along = "current_density" if self.periodic else "length"
        for name, key, quantity in (
                ("times", "times_au", "time"), ("current", "current_au", "current_density"),
                ("corrected", "corrected_au", "current_density"), ("along", "along_au", along),
                ("dipole", "dipole_au", "length"), ("momentum", "momentum_au", "wavenumber"),
                ("energy", "energy_ha", "energy"), ("ground_energy", "ground_energy_ha", "energy"),
                ("kick_energy", "kick_energy_ha", "energy"), ("width", "width_ha", "energy"),
                ("strength", "strength_au", "wavenumber"), ("time_step", "time_step_au", "time")):
            setattr(self, name, self._convert(data[key], quantity) if key in data else None)
        self.norm_error = np.asarray(data["norm_error"])
        self.direction = tuple(data["direction"])

    def _convert(self, values, quantity):
        return _native.to_user_units(values, quantity, atomic_units=self._atomic_units)

    def _spectrum(self, energies, damping, window, kind):
        signal = self._data["corrected_au" if self.periodic else "along_au"]
        values, _ = _native.time_spectrum(
            self._context, np.ascontiguousarray(signal, dtype=float),
            np.ascontiguousarray(energies, dtype=float),
            time_step=float(self._data["time_step_au"]),
            strength=float(self._data["strength_au"]), damping=float(damping),
            window=window, kind=kind, atomic_units=self._atomic_units, current=self.periodic)
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

    def dielectric_function(self, energies, damping=0.1, window="exponential"):
        """Crystal: the complex macroscopic dielectric function along the field on ``energies``
        (the view's energy unit), ``epsilon(z) = 1 - 4 pi i j(z) / (z A_0)`` at
        ``z = E + i damping``, from ``corrected``: the total-field ``epsilon_M`` with the local
        fields of the functional's adiabatic kernel. Only ``window="exponential"`` gives the
        response at a complex frequency; another window raises ``ValueError``, and so does a
        molecule. Computed in Rust."""
        return self._spectrum(energies, damping, window, "dielectric")


class DielectricFunction:
    """The macroscopic dielectric function of a crystal at ``q -> 0`` in the frequency domain.

    Returned by :meth:`Mandacaru.get_dielectric_function`. Every number comes from Rust
    (``periodic::optics::dielectric``) in the user's units; the view holds no arithmetic.

    Attributes
    ----------
    energies : ndarray
        ``(n,)`` the real part ``E`` of the complex frequency ``E + i damping``, eV (Ha).
    tensor : ndarray
        ``(n, 3, 3)`` complex ``epsilon_ab(E + i damping)``, Cartesian.
    epsilon : ndarray
        ``(n,)`` complex ``e . epsilon . e`` along ``direction``, or the isotropic average (the
        trace over three) when ``direction`` is ``None``.
    loss : ndarray
        ``(n,)`` the loss function ``-Im 1 / epsilon``.
    absorption : ndarray
        ``(n,)`` the absorption coefficient ``2 E Im sqrt(epsilon) / (hbar c)``, 1/Angstrom
        (1/Bohr).
    static_tensor : ndarray
        ``(3, 3)`` ``epsilon_ab(0)`` without damping: the electronic static dielectric tensor.
    sum_rule : ndarray
        ``(3, 3)`` ``N_ab``, with ``epsilon -> 1 - 4 pi N / (Omega z^2)`` at large ``z``: the
        diamagnetic band sum of the basis (with the scissor's factor ``D / (e_c - e_v)`` per
        transition), which a frozen-core PAW basis puts above the valence count.
    damping, scissor, onset : float
        The imaginary part of the frequency, the scissor and the smallest transition energy of
        the mesh (the onset of the continuum), eV (Ha).
    cutoff : float or None
        ``kernel="hartree"``: the largest ``|G|`` of the local fields, 1/Angstrom (1/Bohr).
    kernel : str
        ``"hartree"`` (RPA with local fields) or ``"independent"``.
    direction : list of float or None
        The direction of ``epsilon``; ``None`` the isotropic average.
    mesh, kpoints : tuple of int, int
        The whole Gamma-anchored mesh and its point count.
    bands : list of tuple
        ``(valence, conduction)`` bands per spin channel.
    transitions, vectors : int
        The valence-to-conduction transitions and the local-field vectors.
    electrons, momentum_residual : float
        The valence electrons of the cell and the largest anti-Hermitian part of the momentum
        matrix before it was Hermitized (relative).
    functional, energy_unit : str
        The reference's functional and ``"eV"`` or ``"Ha"``.
    record : list of tuple
        The ``[DIELECTRIC FUNCTION]`` run-log record, ``(name, text)`` pairs.
    """

    def __init__(self, data):
        self.__dict__.update(data)
