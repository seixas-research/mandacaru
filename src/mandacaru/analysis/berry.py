"""Unit-aware views of the native Berry phases, Berry curvature and polarization of a crystal.

Every number is computed in Rust (``periodic::berry``); these views only convert units: lengths
to Angstrom, areas to Angstrom squared, the polarization to C/m^2 (atomic units when the
calculator was built with ``atomic_units=True``). Angles stay in radians and k-points in
fractional coordinates of the reciprocal lattice vectors ``G_j``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .. import _native


def _user(values, quantity, atomic):
    return _native.to_user_units(np.ascontiguousarray(values, dtype=float), quantity, atomic)


def _user_scalar(value, quantity, atomic):
    return float(_native.to_user_units(float(value), quantity, atomic))


def _optional(value, transform):
    if value is None:
        return None
    return transform(value)


@dataclass(frozen=True)
class BerryPhase:
    r"""Berry phases of closed k-point strings along ``G_direction``.

    For each string :math:`\mathbf k_s = \mathbf k_\perp + s\,\mathbf G_j/N`,
    :math:`\gamma = -\sum_s \arg\det M(\mathbf k_s, \mathbf k_{s+1})` over the chosen bands, with
    :math:`M_{mn} = \langle u_{m\mathbf k}|u_{n\mathbf k'}\rangle` the PAW overlap of the periodic
    parts; it equals :math:`\oint \mathbf A\cdot d\mathbf k`, :math:`\mathbf A = i\langle u|
    \nabla_{\mathbf k} u\rangle`, and is gauge invariant modulo :math:`2\pi`.

    ``phases`` is ``(spins, n1, n2)`` in radians on (-pi, pi], over the transverse mesh
    (``transverse_axes``, the two other reciprocal directions, ascending) starting from the
    fractional points ``starts``; ``phase`` the string average of each spin channel (on the branch
    of its circular mean) and ``phase_spread`` the largest distance of a string from it.
    ``wannier_charge_centers[s]`` ``(n1, n2, n_bands)`` are the eigenphases of the Wilson loop as
    hybrid Wannier centers along ``a_direction`` (fractional, in [0, 1)). The Berry connection is
    gauge dependent: ``connection[s]`` gives it in the parallel-transport gauge of each string,
    where it is constant along the string, ``2 pi x_n / |G_direction|`` with the center ``x_n`` on
    (-1/2, 1/2] (Angstrom), the projection of the hybrid Wannier center on ``G_direction``.
    ``bands`` is the half-open band range of each channel.
    """

    direction: int
    strings: tuple
    transverse_axes: tuple
    starts: np.ndarray
    phases: np.ndarray
    phase: tuple
    phase_spread: tuple
    wannier_charge_centers: list = field(repr=False)
    connection: list = field(repr=False)
    bands: tuple = ()
    minimum_singular_value: float = 1.0

    @classmethod
    def from_native(cls, raw, atomic=False):
        return cls(
            direction=int(raw["direction"]),
            strings=tuple(raw["size"]),
            transverse_axes=tuple(raw["transverse_axes"]),
            starts=np.asarray(raw["starts"]),
            phases=np.asarray(raw["phases"]),
            phase=tuple(raw["mean"]),
            phase_spread=tuple(raw["spread"]),
            wannier_charge_centers=[np.asarray(c) for c in raw["centers"]],
            connection=[_user(c, "length", atomic) for c in raw["connection"]],
            bands=tuple(tuple(b) for b in raw["bands"]),
            minimum_singular_value=float(raw["minimum_singular_value"]),
        )


@dataclass(frozen=True)
class BerryCurvature:
    r"""The Berry curvature of a k plane (Fukui, Hatsugai and Suzuki).

    The plane is normal to ``G_normal`` and spanned by ``G_axes[0]``, ``G_axes[1]`` (cyclic), with
    ``size`` points along them, at the fractional coordinate ``offset`` along the normal. Each
    plaquette carries the flux
    :math:`\phi = -\arg[\det U_a(\mathbf k)\det U_b(\mathbf k+\mathbf b_a)
    \det U_a(\mathbf k+\mathbf b_b)^*\det U_b(\mathbf k)^*]` of the band group (non-Abelian
    through the determinant); ``flux`` is ``(spins, n_a, n_b)`` in radians and ``curvature`` the
    flux over the plaquette area, the component along ``orientation`` (the unit vector of
    :math:`\mathbf G_a\times\mathbf G_b`), in Angstrom squared, at the fractional plaquette
    ``centers``. ``chern_numbers`` per spin channel and their sum ``chern_number`` are
    :math:`(1/2\pi)\sum\phi`, integers by construction, oriented along ``orientation``
    (:math:`\mathbf G_a\times\mathbf G_b`): relabeling the lattice vectors so the cell changes
    handedness (``a_1`` and ``a_2`` swapped, say) flips that vector and with it the sign of the
    curvature and of the Chern number of the same physical plane;
    :meth:`~mandacaru.Mandacaru.get_anomalous_hall_conductivity` corrects for it, so the
    conductivity does not depend on the labels. ``max_flux`` is the largest plaquette flux
    (a flux beyond 3 pi / 4 is refused: the lattice field would alias).
    """

    normal: int
    axes: tuple
    size: tuple
    offset: float
    orientation: np.ndarray
    flux: np.ndarray
    curvature: np.ndarray
    centers: np.ndarray
    chern_numbers: tuple
    chern_number: int
    bands: tuple = ()
    max_flux: float = 0.0
    minimum_singular_value: float = 1.0

    @classmethod
    def from_native(cls, raw, atomic=False):
        return cls(
            normal=int(raw["normal"]),
            axes=tuple(raw["axes"]),
            size=tuple(raw["size"]),
            offset=float(raw["offset"]),
            orientation=np.asarray(raw["orientation"]),
            flux=np.asarray(raw["flux"]),
            curvature=_user(raw["curvature"], "area", atomic),
            centers=np.asarray(raw["centers"]),
            chern_numbers=tuple(int(c) for c in raw["chern"]),
            chern_number=int(raw["chern_total"]),
            bands=tuple(tuple(b) for b in raw["bands"]),
            max_flux=float(raw["max_flux"]),
            minimum_singular_value=float(raw["minimum_singular_value"]),
        )


@dataclass(frozen=True)
class Polarization:
    r"""The Berry-phase polarization of an insulator (King-Smith and Vanderbilt; Resta).

    ``vector`` is the total on the branch nearest zero, in C/m^2; ``fractional`` its coordinates
    in units of the quanta (each in (-1/2, 1/2]), whose rows ``quanta`` are
    :math:`g\,e\,\mathbf a_j/\Omega` with :math:`g = \gcd(f, Z_1, Z_2, \dots)` (``quantum``) and
    :math:`f` the electrons per band (``degeneracy``: 2 unpolarized, 1 spin-polarized or
    spinor); ``raw`` the coordinates before the reduction (ionic plus electronic), which a
    difference along a path should compare. ``electronic`` and ``ionic`` are the two parts,
    Cartesian and unreduced; ``phases`` the string-averaged Berry phases ``(spins, 3)`` along the
    three ``G_j`` (radians) and ``phase_spread`` their largest string spread. Above pi / 2 the
    result comes with a warning: there the branch of the string average is no longer unique. The
    threshold does not bound the error of the transverse sampling (a spread below it, or a single
    transverse point, gives no warning): compare the result on two transverse meshes. ``strings`` is the
    mesh the strings were cut from. A polarization is defined modulo its quanta: only a change
    along a path is physical.
    """

    vector: np.ndarray
    fractional: np.ndarray
    raw: np.ndarray
    quanta: np.ndarray
    electronic: np.ndarray
    ionic: np.ndarray
    phases: np.ndarray
    phase_spread: np.ndarray
    strings: tuple
    quantum: int
    degeneracy: float
    minimum_singular_value: float = 1.0

    @classmethod
    def from_native(cls, raw, atomic=False, stacklevel=1):
        """The view of a native record; each of its ``warnings`` (string phases spread widely
        about their average) is issued as a :class:`UserWarning`."""
        for note in raw.get("warnings", ()):
            warnings.warn(note, stacklevel=stacklevel + 1)
        return cls(
            vector=_user(raw["vector"], "polarization", atomic),
            fractional=np.asarray(raw["fractional"]),
            raw=np.asarray(raw["raw"]),
            quanta=_user(raw["quanta"], "polarization", atomic),
            electronic=_user(raw["electronic_vector"], "polarization", atomic),
            ionic=_user(raw["ionic_vector"], "polarization", atomic),
            phases=np.asarray(raw["phases"]),
            phase_spread=np.asarray(raw["spread"]),
            strings=tuple(raw["size"]),
            quantum=int(raw["quantum"]),
            degeneracy=float(raw["degeneracy"]),
            minimum_singular_value=float(raw["minimum_singular_value"]),
        )


@dataclass(frozen=True)
class AnomalousHall:
    r"""The intrinsic anomalous Hall conductivity of a crystal.

    ``sigma`` is the antisymmetric ``(3, 3)`` tensor
    :math:`\sigma_{ab} = (e^2/h)(1/2\pi)\epsilon_{abd}K_d` in S/cm, with
    :math:`\mathbf K = \operatorname{sign}(\det G)\sum_c \nu_c\mathbf G_c` and ``numbers`` the
    plane numbers :math:`\nu_c` (electrons per band included). For an insulator (``metallic``
    false) :math:`\nu_c = f C_c` with the integer Chern numbers ``chern_numbers`` (per plane
    family and spin channel), the same on every layer. For a metal :math:`\nu_c` is the
    Fermi-sea term :math:`(1/2\pi)\sum_n\int f_n\Omega_n` of the converged smearing, from the
    occupation-weighted lattice field, averaged over the ``layers`` (``[c][channel][layer]``).
    A metal's value converges slowly with ``kpts``: ``sigma_coarse`` is the same formula on the
    stride-2 sub-mesh of the same states (every ``n_j`` even and at least 8, else ``None``), and a
    warning is issued when it is missing or differs by more than 10% of the largest component.
    ``merged`` counts the plaquettes (of ``plaquettes``) where a band crossing or a point node
    merged two band groups (their weight is shared there); above 25% a warning says the mesh does
    not resolve the bands near the Fermi level. A metal's occupations use ``fermi`` (eV), the Fermi
    level re-solved on the Berry mesh with the converged smearing so the mesh holds the cell's
    electrons; ``scf_electron_error`` is the electron excess the SCF level would have given there.
    ``cap`` is the number of bands linked per point.
    """

    sigma: np.ndarray
    sigma_coarse: np.ndarray | None
    metallic: bool
    numbers: np.ndarray
    layers: list = field(repr=False)
    chern_numbers: tuple = ()
    strings: tuple = ()
    max_flux: float = 0.0
    cap: int = 0
    merged: int = 0
    plaquettes: int = 0
    fermi: float = 0.0
    scf_electron_error: float = 0.0
    method: str = "direct"

    @classmethod
    def from_native(cls, raw, atomic=False, stacklevel=1):
        """The view of a native record; each of its ``warnings`` is issued as a
        :class:`UserWarning`."""
        for note in raw.get("warnings", ()):
            warnings.warn(note, stacklevel=stacklevel + 1)
        coarse = raw.get("sigma_coarse")
        return cls(
            sigma=_user(raw["sigma"], "conductivity", atomic),
            sigma_coarse=None if coarse is None else _user(coarse, "conductivity", atomic),
            metallic=bool(raw["metallic"]),
            numbers=np.asarray(raw["numbers"]),
            layers=[[np.asarray(layer) for layer in channels] for channels in raw["layers"]],
            chern_numbers=tuple(tuple(int(n) for n in row) for row in raw["chern"]),
            strings=tuple(raw["size"]),
            max_flux=float(raw["max_flux"]),
            cap=int(raw["cap"]),
            merged=int(raw["merged"]),
            plaquettes=int(raw.get("plaquettes", 0)),
            fermi=_user(raw.get("fermi", 0.0), "energy", atomic),
            scf_electron_error=float(raw.get("scf_electron_error", 0.0)),
            method=str(raw.get("method", "direct")),
        )


@dataclass(frozen=True)
class WannierChargeCenterFlow:
    r"""The flow of the hybrid Wannier charge centers of a k plane (Soluyanov and Vanderbilt).

    The plane is normal to ``G_normal`` at the fractional ``offset``; the strings run along
    ``G_direction`` (``length`` points) and ``G_transverse`` is the flow parameter ``k_perp``.
    ``wannier_charge_centers`` is ``(samples, n)``: the eigenphases of each string's Wilson loop,
    fractional along ``G_direction`` in [0, 1), ascending, at the transverse coordinates
    ``k_perp`` (chosen adaptively: an interval is bisected where a center moves by more than a
    quarter of the largest gap; ``strings`` counts the strings computed).

    A time-reversal-invariant plane (``half``: the offset is 0 or 1/2, the crystal is
    time-reversal invariant and the bands are the occupied ones) is flown over
    :math:`k_\perp\in[0,1/2]`, between its two time-reversal-invariant lines, where the centers
    are Kramers pairs. ``z2`` is the parity of the number of times they cross the middle of the
    largest gap between consecutive centers (``gap_midpoints`` and ``gaps`` per sample; the line
    follows the largest gap, and the centers it passes over count as crossings, ``crossings``).
    A spin-degenerate crystal (``spin_degenerate``: no spin-orbit coupling) holds every center
    twice, once per spin, so ``z2`` is the zero of that doubled manifold. Any other plane is flown
    over the whole period and ``winding``, the total displacement of the centers, is its Chern
    number (oriented as :meth:`~mandacaru.Mandacaru.get_chern_number`, along
    :math:`\mathbf G_a\times\mathbf G_b` for the cyclic in-plane directions).
    """

    normal: int
    direction: int
    transverse: int
    offset: float
    length: int
    k_perp: np.ndarray
    wannier_charge_centers: np.ndarray
    half: bool
    winding: int | None
    z2: int | None
    crossings: int | None
    gap_midpoints: np.ndarray | None = None
    gaps: np.ndarray | None = None
    spin_degenerate: bool = False
    bands: tuple = ()
    strings: int = 0
    minimum_singular_value: float = 1.0

    @classmethod
    def from_native(cls, raw):
        lines, gaps = raw["lines"], raw["gaps"]
        return cls(
            normal=int(raw["normal"]),
            direction=int(raw["direction"]),
            transverse=int(raw["transverse"]),
            offset=float(raw["offset"]),
            length=int(raw["length"]),
            k_perp=np.asarray(raw["k_perp"]),
            wannier_charge_centers=np.asarray(raw["centers"]),
            half=bool(raw["half"]),
            winding=None if raw["winding"] is None else int(raw["winding"]),
            z2=None if raw["z2"] is None else int(raw["z2"]),
            crossings=None if raw["crossings"] is None else int(raw["crossings"]),
            gap_midpoints=None if lines is None else np.asarray(lines),
            gaps=None if gaps is None else np.asarray(gaps),
            spin_degenerate=bool(raw["spin_degenerate"]),
            bands=tuple(raw["bands"]),
            strings=int(raw["strings"]),
            minimum_singular_value=float(raw["minimum_singular_value"]),
        )


@dataclass(frozen=True)
class Z2Invariant:
    r"""The Z2 topological invariant of a time-reversal-invariant insulator.

    ``indices`` is :math:`(\nu_0;\nu_1\nu_2\nu_3)` for a crystal (``strong`` and ``weak``), or the
    one index of a single plane (``strong`` and ``weak`` are then ``None``). :math:`\nu_j` is the
    index of the plane :math:`k_j=1/2` and :math:`\nu_0` that of :math:`k_j=0` plus it, mod 2,
    the same for the three ``j``; ``planes`` maps ``(j, half)`` to the index of the plane
    :math:`k_j = 0` (``half = 0``) or :math:`1/2` (``half = 1``).

    ``method`` is ``"flow"`` (the parity of the crossings of the Wannier-center flow of each
    plane, ``flows``, one :class:`WannierChargeCenterFlow` per plane in ``j``-major order) or
    ``"parity"`` (the Fu-Kane product of the inversion eigenvalues of the occupied Kramers pairs
    at the eight time-reversal-invariant points; ``parity`` then lists, per point, its fractional
    ``index``, the number of occupied states ``occupied`` and of those with parity ``-1``
    (``minus``), ``delta`` (the product over the pairs), ``energies`` (eV) and ``parities`` of the
    occupied and the lowest empty levels, and the ``inversion_center`` in fractional
    coordinates). A spin-degenerate crystal (no spin-orbit coupling) has index zero.
    """

    method: str
    indices: tuple
    strong: int | None
    weak: tuple | None
    planes: dict
    flows: tuple = ()
    parity: tuple | None = None
    inversion_center: tuple | None = None
    spin_degenerate: bool = False
    strings: tuple = ()

    @classmethod
    def from_native(cls, raw, atomic=False):
        detail = raw["parity"]
        parity = None
        if detail is not None:
            parity = tuple(
                {
                    "index": tuple(int(i) for i in t["index"]),
                    "occupied": int(t["occupied"]),
                    "minus": int(t["minus"]),
                    "delta": int(t["delta"]),
                    "residual": float(t["residual"]),
                    "energies": _user(t["energies"], "energy", atomic),
                    "parities": np.asarray(t["parities"], dtype=int),
                }
                for t in detail["trims"]
            )
        flows = tuple(WannierChargeCenterFlow.from_native(f) for f in raw["flows"])
        degenerate = (
            bool(detail["spin_degenerate"]) if detail is not None
            else bool(flows) and all(f.spin_degenerate for f in flows)
        )
        strong, weak = int(raw["strong"]), tuple(int(w) for w in raw["weak"])
        return cls(
            method=str(raw["method"]),
            indices=(strong, *weak),
            strong=strong,
            weak=weak,
            planes={tuple(k): int(v) for k, v in raw["planes"].items()},
            flows=flows,
            parity=parity,
            inversion_center=None if detail is None else tuple(detail["center"]),
            spin_degenerate=degenerate,
            strings=tuple(raw["size"]),
        )


@dataclass(frozen=True)
class BornCharges:
    r"""Born effective charges :math:`Z^*_{a,ij} = (\Omega / e)\,\partial P_i / \partial u_{a,j}`.

    Computed by central finite differences of the Berry-phase polarization on perturbed
    self-consistent crystals. ``charges`` has shape ``(N, 3, 3)`` in units of ``e``. The
    per-run acoustic sum rule residual is ``asr_residual``; when ``acoustic_sum_rule="impose"``
    was requested, ``corrected`` holds the sum-rule-projected tensor (else ``None``).
    ``channels`` holds the spin-resolved electronic contributions, ``displacement`` the step
    size in user length units (Angstrom, or Bohr when ``atomic_units=True``),
    ``polarization_steps`` the folded polarization step per run in quanta, and
    ``scf_iterations`` the SCF iteration count of every perturbed run.
    """

    charges: np.ndarray
    asr_residual: float
    corrected: np.ndarray | None
    channels: np.ndarray
    displacement: float
    polarization_steps: np.ndarray
    scf_iterations: tuple[int, ...]
    asr_mean: np.ndarray

    @classmethod
    def from_native(cls, raw, atomic=False, stacklevel=1):
        """Build from a native ``born_charges`` dictionary, issuing any ASR warning."""
        note = raw.get("warnings", ())
        if note:
            warnings.warn(note[0], stacklevel=stacklevel)
        return cls(
            charges=np.asarray(raw["charges"], dtype=float),
            asr_residual=float(raw["asr_residual"]),
            corrected=_optional(raw["corrected"], lambda c: np.asarray(c, dtype=float)),
            channels=np.asarray(raw["channels"], dtype=float),
            displacement=_user_scalar(raw["delta"], "length", atomic),
            polarization_steps=np.asarray(raw["steps"], dtype=float),
            scf_iterations=tuple(int(n) for n in raw["iterations"]),
            asr_mean=np.asarray(raw["asr_mean"], dtype=float),
        )


@dataclass(frozen=True)
class PiezoelectricTensor:
    r"""Proper piezoelectric tensor :math:`e_{i,v}` (Voigt indices ``xx, yy, zz, yz, xz, xy``).

    In C/m\ :sup:`2` (``e / Bohr^2`` when ``atomic_units=True``). ``tensor`` is the active
    tensor (``relaxed`` when ``ions="relaxed"``, otherwise ``clamped``). When
    ``ions="relaxed"`` was requested, ``born_charges``, ``internal_strain`` (``3N x 6`` in user
    length units), ``force_constants`` (``3N x 3N`` in ``eV / Angstrom^2`` or
    ``Hartree / Bohr^2``), and ``acoustic_frequencies_removed`` (the 3 translational eigenvalues
    removed before pseudo-inversion) are also populated.
    """

    tensor: np.ndarray
    clamped: np.ndarray
    strain: float
    polarization_steps: np.ndarray
    scf_iterations: tuple[int, ...]
    relaxed: np.ndarray | None = None
    born_charges: BornCharges | None = None
    internal_strain: np.ndarray | None = None
    force_constants: np.ndarray | None = None
    acoustic_frequencies_removed: np.ndarray | None = None

    @classmethod
    def from_native(cls, raw, atomic=False, stacklevel=1):
        """Build from a native ``piezoelectric`` dictionary, converting units via Rust."""
        return cls(
            tensor=_user(raw["tensor"], "polarization", atomic),
            clamped=_user(raw["clamped"], "polarization", atomic),
            strain=float(raw["delta"]),
            polarization_steps=np.asarray(raw["steps"], dtype=float),
            scf_iterations=tuple(int(n) for n in raw["iterations"]),
            relaxed=_optional(raw["relaxed"], lambda r: _user(r, "polarization", atomic)),
            born_charges=_optional(
                raw["born_charges"],
                lambda z: BornCharges.from_native(z, atomic=atomic, stacklevel=stacklevel),
            ),
            internal_strain=_optional(raw["internal_strain"], lambda g: _user(g, "length", atomic)),
            force_constants=_optional(
                raw["force_constants"], lambda k: _user(k, "force_constant", atomic)
            ),
            acoustic_frequencies_removed=_optional(
                raw["acoustic_eigenvalues"], lambda e: _user(e, "force_constant", atomic)
            ),
        )
