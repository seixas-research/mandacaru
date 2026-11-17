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
        (e Bohr with ``atomic_units=True``). The sign of each root is a convention;
        on a spinor reference (``spin == "spinor"``) they are complex and the phase
        of each state is.
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
        omits (energy units). ``None`` on other references.
    one_center_xc : ndarray or None
        On a PAW-LCAO Kohn-Sham state with LDA or PBE: the first-order shift of
        each root from the one-center exchange-correlation kernel the same
        linearization omits, at the spherical reference atom (energy units).
        ``None`` on other references and functionals.
    """

    def __init__(self, native, *, atomic_units, context):
        values = native.data()
        self._native = native
        self._context = context
        self._atomic_units = atomic_units
        self.energy_unit, self.length_unit = ("Ha", "Bohr") if atomic_units else ("eV", "Angstrom")
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
        shifts = values["one_center_xc_ha"]
        self.one_center_xc = None if shifts is None else self._energy(shifts)
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
        self._transitions, self._nto_weights, self._plus, self._minus = (
            values[key] for key in ("transitions", "nto_weights", "plus", "minus"))

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

    def transition_density(self, root, channel=None, grid=None):
        """The transition density of ``root`` sampled in Rust, as a
        :class:`~mandacaru.analysis.fields.VolumetricField` (``write`` gives a
        cube or xsf file), on ``grid`` (a native ``SpatialGrid``; default the
        integration grid). ``channel=None`` gives the charge transition density,
        whose dipole with the PAW one-center part is ``transition_dipoles[root]``
        (a closed-shell triplet has none and raises ``ValueError``); a channel
        index gives the spatial factor of that spin's change. A one-particle
        quantity in e/Bohr^3; the PAW one-center charge it leaves out is
        ``augmentation_charge``."""
        from .fields import SamplingGrid, VolumetricField

        values = self._native.transition_field(root, channel, getattr(grid, "native", grid))
        return VolumetricField(
            quantity="transition_density", data=values["data"],
            grid=SamplingGrid(values["grid"]), numbers=np.asarray(values["numbers"]),
            positions=np.asarray(values["positions"]),
            charges=np.asarray(values["charges"], dtype=float), units="e/Bohr^3",
            integral=values["integral"], norm=values["norm"], n_electrons=0.0,
            context=self._context, index=root,
            augmentation_charge=values["augmentation_charge"])

    def spectrum(self, energies, width=0.1, shape="gaussian"):
        """The broadened absorption spectrum ``sum_n f_n g(E - E_n)`` on
        ``energies`` (the view's energy unit), with unit-area ``"gaussian"``
        (standard deviation ``width``) or ``"lorentzian"`` (half width
        ``width``) lines, in 1/energy unit. Computed in Rust."""
        return np.asarray(_native.line_spectrum(
            self._context, np.ascontiguousarray(self.energies, dtype=float),
            np.ascontiguousarray(self.oscillator_strengths, dtype=float),
            np.ascontiguousarray(energies, dtype=float), width=float(width), shape=shape))


class CrystalExcitations:
    """The Bethe-Salpeter excitons of a converged insulating crystal (Tamm-Dancoff, or the
    coupling form with ``approximation="casida"``).

    Returned by :meth:`Mandacaru.get_excitations` on a crystal. Every number comes from Rust
    (``periodic::optics::bse``), already in user units; the view holds no arithmetic.

    Attributes
    ----------
    energies : ndarray
        Exciton energies ``E_l``, ascending, eV (Ha with ``atomic_units=True``).
    binding : ndarray
        ``onset - E_l``: positive for an exciton bound below the interband continuum of the mesh.
    onset : float
        The smallest transition energy ``e_ck - e_vk + scissor`` of the mesh.
    weights : ndarray
        ``(n, 3)`` the area of each exciton's line in ``Im epsilon`` along x, y and z,
        ``4 pi^2 s |sum_t h_t A_t|^2`` (eV; Ha), with ``s = g / (N_k Omega)`` and ``h`` the
        k . p dipoles (in the coupling form ``sum_t (h_t X_t + h~_t Y_t)``, ``h~`` the dipoles
        of the antiresonant transitions); zero for triplets.
    participation : ndarray
        ``1 / sum_t |A_t|^4``: the number of transitions an exciton spreads over (of
        ``transitions``); ``A`` is the normalized resonant amplitude ``X``.
    kpoint_weights : ndarray
        ``(n, kpoints)`` ``sum_vc |A_vck|^2`` per mesh point (mesh order, C order of ``mesh``).
    dominant : list of tuple
        Per exciton ``(k, valence, conduction, weight)``: its largest transition (band indices
        of the crystal, 0-based).
    dominant_channels : list of int
        The spin channel of each ``dominant`` transition (0, or 1 for the second channel of a
        spin-polarized crystal).
    spin, kernel, approximation : str
        ``"singlet"`` or ``"triplet"``; ``"bse"``, ``"hartree"`` (exchange only) or
        ``"independent"``; ``"tda"`` or ``"casida"``.
    static_tensor : ndarray
        ``(3, 3)`` the static ``eps_M`` with local fields that screens the head of ``W``.
    screened_head : float
        The ``G = G' = 0`` element of ``W(q = 0)``, averaged over the mesh's mini-zone
        (Hartree Bohr^3).
    mesh, kpoints, bands, transitions : tuple, int, tuple, int
        The whole Gamma-anchored mesh, its point count, ``(valence, conduction)`` and the
        dimension of the operator.
    hermiticity, residuals, converged, solver :
        ``max |H - H^dagger| / max |H|`` before the Hermitization, each root's residual, whether
        every root converged, and ``"dense"``, ``"davidson"`` or ``"coupling"``.
    record : list of tuple
        The ``[EXCITED STATES]`` run-log record, ``(name, text)`` pairs.
    """

    def __init__(self, data):
        self.__dict__.update(data)

