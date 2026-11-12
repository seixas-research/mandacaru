"""Unit-aware views over native excited-state results: VQE sequential deflation and linear response."""

from collections import namedtuple
from dataclasses import dataclass
import numpy as np
from .. import _native


@dataclass
class EnergyLevels:
    energies: np.ndarray
    states: np.ndarray
    reference_energy: float
    num_evaluations: int
    success: tuple[bool, ...]
    messages: tuple[str, ...]
    energy_unit: str = "eV"

    @property
    def num_states(self):
        return len(self.energies)

    @property
    def ground_state_energy(self):
        return float(self.energies[0])

    @property
    def excitation_energies(self):
        return _native.relative_to_first([float(e) for e in self.energies])

    @property
    def gaps(self):
        return _native.gaps([float(e) for e in self.energies])

    def in_units(self, units="eV"):
        if units not in ("Ha", "eV"):
            raise ValueError("energy units must be Ha or eV")
        if units == self.energy_unit:
            return self.energies.copy()
        return _native.convert_energy(
            np.ascontiguousarray(self.energies, dtype=float), self.energy_unit, units)

    def excitation_energies_in_units(self, units="eV"):
        return _native.relative_to_first([float(e) for e in self.in_units(units)])


#: One dominant orbital transition of a root: ``spin`` is ``None`` for a closed shell (spin-adapted)
#: and ``"alpha"`` or ``"beta"`` for an unrestricted reference; orbital indices are the reference's
#: (0-based, ascending energy, in that spin); ``weight`` is ``X^2`` (TDA) or ``X^2 - Y^2`` (Casida).
Transition = namedtuple("Transition", "spin occupied empty weight")


class Excitations:
    """The linear-response excitations of a converged molecular state.

    Returned by :meth:`Mandacaru.get_excitations`. Every number comes from Rust
    (``molecular::excitations``); the view converts units through the native
    conversions and holds no arithmetic of its own.

    Attributes
    ----------
    energies : ndarray
        Excitation energies, ascending, in eV (Ha with ``atomic_units=True``).
    wavelengths : ndarray
        The vacuum wavelengths of the same photons, nm.
    oscillator_strengths : ndarray
        Dimensionless, ``2/3 omega |mu|^2``; zero for triplets.
    velocity_oscillator_strengths : ndarray
        The velocity form ``2/3 |<0|grad|n>|^2 / omega``, with the PAW one-center
        part of the gradient. It equals the length form for exact states of a local
        Hamiltonian; the difference measures the basis (and, with exact exchange,
        the nonlocal potential).
    transition_dipoles : ndarray
        ``(n, 3)`` length-gauge transition moments ``<0|r|n>`` in e Angstrom
        (e Bohr with ``atomic_units=True``). The sign of each root is a convention.
    spin_squared : ndarray or None
        On an unrestricted reference: ``<S^2>`` of each root, as the singles
        combination of the reference's determinants with the root's ``X``
        amplitudes (a contamination diagnostic; 0.75 for a clean doublet).
        ``None`` for a closed shell, whose roots are pure singlets or triplets.
    reference_spin_squared : float or None
        ``<S^2>`` of the unrestricted reference determinant.
    one_center_hartree : ndarray or None
        On a PAW-LCAO Kohn-Sham state: the first-order shift of each root from
        the one-center Hartree kernel that the linearized one-center energy
        omits (energy units). The one-center exchange-correlation part is not
        estimated. ``None`` on other references.
    """

    def __init__(self, native, *, atomic_units, context):
        values = native.data()
        self._native = native
        self._context = context
        self._atomic_units = atomic_units
        self.energy_unit = "Ha" if atomic_units else "eV"
        self.length_unit = "Bohr" if atomic_units else "Angstrom"
        self.approximation = values["approximation"]
        self.spin = values["spin"]
        self.kernel = values["kernel"]
        self.functional = values["functional"]
        self.energies = self._energy(values["energies_ha"])
        self.wavelengths = np.asarray(values["wavelengths_nm"])
        self.oscillator_strengths = np.asarray(values["oscillator_strengths"])
        self.velocity_oscillator_strengths = np.asarray(values["velocity_oscillator_strengths"])
        self.transition_dipoles = _native.to_user_units(
            np.ascontiguousarray(values["transition_dipoles_bohr"]), "length",
            atomic_units=atomic_units)
        shifts = values["one_center_hartree_ha"]
        self.one_center_hartree = None if shifts is None else self._energy(shifts)
        #: Per spin channel (one for a closed shell, alpha and beta otherwise).
        self.orbital_energies = tuple(self._energy(e) for e in values["orbital_energies_ha"])
        self.occupied = tuple(values["occupied"])
        self.virtual = tuple(values["virtual"])
        squared = values["spin_squared"]
        self.spin_squared = None if squared is None else np.asarray(squared)
        self.reference_spin_squared = values["reference_spin_squared"]
        self.requested = values["requested"]
        self.added = values["added"]
        self.degenerate_tolerance = self._energy(values["degenerate_tolerance_ha"])
        self.residuals = tuple(values["residuals"])
        self.iterations = values["iterations"]
        self.applications = values["applications"]
        self._transitions = values["transitions"]
        self._nto_weights = values["nto_weights"]
        self._plus = values["plus"]
        self._minus = values["minus"]

    def _energy(self, value_ha):
        if np.ndim(value_ha) == 0:
            return _native.to_user_units(float(value_ha), "energy",
                                         atomic_units=self._atomic_units)
        return _native.to_user_units(np.ascontiguousarray(value_ha, dtype=float), "energy",
                                     atomic_units=self._atomic_units)

    @property
    def num_states(self):
        return len(self.energies)

    def __len__(self):
        return self.num_states

    def transitions(self, root):
        """The dominant orbital transitions of ``root`` (:data:`Transition`
        records); those below a weight of 0.01 are left out."""
        unrestricted = len(self.occupied) == 2
        labels = ("alpha", "beta")
        return [Transition(labels[c] if unrestricted else None, i, a, w)
                for c, i, a, w in self._transitions[root]]

    def nto_weights(self, root):
        """The natural transition orbital weights of ``root``, descending (sum 1)."""
        return tuple(self._nto_weights[root])

    def amplitudes(self, root, channel=0):
        """``(X + Y, X - Y)`` of ``root`` in spin channel ``channel`` (0 for a
        closed shell; 0 alpha, 1 beta), each ``(occupied, virtual)``; summed
        over the channels, ``(X + Y) . (X - Y) = 1`` (TDA: both are ``X``)."""
        return np.asarray(self._plus[channel][root]), np.asarray(self._minus[channel][root])

    def natural_transition_orbitals(self, root, channel=0):
        """``(weights, occupied, empty)`` of ``root`` in spin channel
        ``channel``: the weights of the natural transition orbital pairs within
        the channel in descending order, and the pairs as columns over the
        atomic basis."""
        weights, occupied, empty = self._native.natural_transition_orbitals(root, channel)
        return np.asarray(weights), np.asarray(occupied), np.asarray(empty)

    def transition_density_matrix(self, root, channel=0):
        """The symmetrized transition density matrix of ``root`` in spin
        channel ``channel`` in the atomic basis: the spatial factor of that
        spin's first-order density change."""
        return np.asarray(self._native.transition_density(root, channel))

    def spectrum(self, energies, width=0.1, shape="gaussian"):
        """The broadened absorption spectrum ``sum_n f_n g(E - E_n)`` on
        ``energies`` (the view's energy unit), with unit-area ``"gaussian"``
        (standard deviation ``width``) or ``"lorentzian"`` (half width
        ``width``) lines, in 1/energy unit. Computed in Rust."""
        return np.asarray(_native.line_spectrum(
            self._context, np.ascontiguousarray(self.energies, dtype=float),
            np.ascontiguousarray(self.oscillator_strengths, dtype=float),
            np.ascontiguousarray(energies, dtype=float), width=float(width), shape=shape))
