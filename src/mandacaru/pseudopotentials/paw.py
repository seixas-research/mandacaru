# -*- coding: utf-8 -*-
# file: pseudopotentials/paw.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""PAW-LCAO and UPAW-LCAO datasets: the view the basis and the integral engines read, the
libraries, and the family registry.

A dataset is Bloechl's projector-augmented-wave transformation (PRB 50, 17953) with a
frozen core, carried on a *localized* basis -- the bound smooth partial waves and their
multiple-zeta hierarchy -- rather than on plane waves, which is what the **-LCAO** says.
Everything numerical about one is native (``mandacaru-core``):

- generation -- the reference atom, the smooth waves, the projectors, the one-center
  terms, the Dirac spin-orbit chain and the search that repairs a ghost state
  (``paw::generate``), reached through :func:`generate_paw` / :func:`generate_upaw`
  (:mod:`.generation`);
- the file record (``paw::record``), reached through :mod:`.record` and
  :func:`~.io.load_pseudopotential`;
- the checks (``paw::diagnose``): :func:`paw_spectrum`, :func:`log_derivative_paw`,
  :func:`reconstruct_ae` and :func:`report_paw` (:mod:`.diagnostics`).

What stays here is the :class:`PAWDataset` / :class:`PAWChannel` view (arrays by channel, the
local potential through the engine's spline), the library readers and writers
(:func:`get_paw`, :func:`get_upaw`, :func:`build_paw_library`, :func:`build_upaw_library`),
the projector blocks the periodic setup reads (:func:`paw_projectors` and its block builders,
which die with the periodic setup) and the registration of the two families.

UPAW-LCAO (``basis="UPAW-LCAO"``) is the same construction at ``norm_deficit = 0``
(Ivanov *et al.*, arXiv:2408.03159): the transformation is unitary and the overlap operator
is the identity.  It has no shipped library; without ``MANDACARU_UPAW_PATH`` it is generated
on demand, in the engine, and cached for the process.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from .. import _native
from .confinement import CONFINEMENT_DEFAULT_OPTIONS, CONFINEMENT_OPTIONS
from .dataset import Channel, PseudoPotential
from .environment import DEFAULT_XC
from .generation import SearchReport, generate_paw, generate_upaw
from .native_context import context

#: Registry name of the family (no aliases).  The **-LCAO** is not decoration:
#: this is Bloechl's projector-augmented-wave transformation carried on a
#: *localized* basis -- the bound smooth partial waves and their multiple-zeta
#: hierarchy -- rather than on plane waves, which is what the name of a basis
#: set has to say.  The method it implements is still PAW-LCAO.
FAMILY = "paw-lcao"

#: Registry name of the **unitary** variant (``basis="UPAW-LCAO"``).
UPAW_FAMILY = "upaw-lcao"

#: Radial points of the compensation charge's atom-centered quadrature.
COMPENSATION_RADIAL_POINTS = 48
#: Projector basis sampled on the molecular grid (:meth:`PAWDataset.projector_set`).
DEFAULT_PROJECTOR_BASIS = "raw"
#: Spherical product quadrature of the atom-centered projections: Gauss-Legendre
#: in radius and cos(theta), uniform in phi.
PROJECTION_RADIAL_POINTS = 64
PROJECTION_POLAR_POINTS = 24
PROJECTION_AZIMUTHAL_POINTS = 48


# --------------------------------------------------------------------------- #
# One channel and one dataset: the view.
# --------------------------------------------------------------------------- #

@dataclass
class PAWChannel(Channel):
    """One PAW-LCAO channel: the shared :class:`Channel` plus the partial-wave sets.

    ``pseudo_radial``/``eigenvalue``/``coefficients`` describe the first
    (bound) smooth partial wave, which is also the first-zeta basis function.
    All matrices are ``(n, n)`` over the radial partial waves of the channel.
    ``occupations`` gives each partial wave's electrons in the reference
    atom: a scattering reference holds none, and a semicore channel has two
    bound, occupied references (sodium's 2s and 3s); ``occupation`` is
    their sum.
    """

    reference_energies: list = field(default_factory=list)   # Hartree
    occupations: list = field(default_factory=list)          # per wave: electrons
    wavevectors: list = field(default_factory=list)          # per wave: q_n
    wave_coefficients: list = field(default_factory=list)    # per wave: c_n
    ae_waves: list = field(default_factory=list)             # per wave: phi(r)
    pseudo_waves: list = field(default_factory=list)         # per wave: phi~(r)
    projectors: list = field(default_factory=list)           # per wave: p~(r), dual
    raw_projectors: list = field(default_factory=list)       # per wave: chi(r)
    overlap_correction: np.ndarray = None                    # q_ij (charge)
    norm_correction: np.ndarray = None                       # q^norm_ij (energy)
    kinetic_difference: np.ndarray = None                    # Delta T_ij
    potential_difference: np.ndarray = None                  # Delta V^scr_ij
    coupling_screened: np.ndarray = None                     # D^scr_ij
    coupling: np.ndarray = None                              # D^ion_ij
    vanderbilt: np.ndarray = None                            # B_ij
    duality_error: float = 0.0                               # max |<p|phi~> - 1|
    overlap_minimum: float = 1.0                             # min eig of S in channel
    asymmetry: float = 0.0                                   # max |D^scr - D^scr^T|
    consistency_error: float = 0.0                           # |D^scr - dT - dV|
    residual_kinetic: list = field(default_factory=list)     # Hartree per wave
    q_cut: float = None

    def __repr__(self) -> str:
        energies = ", ".join(f"{e:+.4f}" for e in self.reference_energies)
        q = (np.diag(self.overlap_correction).tolist()
             if self.overlap_correction is not None else [])
        return (f"PAWChannel(l={self.l}, n={self.n}, eps=[{energies}] Ha, "
                f"rc={self.r_cut:.3f} Bohr, {len(self.projectors)} projectors, "
                f"q_ii={np.round(q, 4).tolist()})")


def _deficit_label(norm_deficit) -> str:
    """``norm_deficit`` as printed: a zero deficit is a unitary dataset."""
    if norm_deficit is None:
        return "free"
    return f"{float(norm_deficit):g}" + (" (unitary)" if norm_deficit == 0 else "")



@dataclass
class PAWDataset(PseudoPotential):
    r"""A PAW-LCAO dataset: local potential, projectors, one-center matrices.

    Inherits the :class:`~.dataset.PseudoPotential` layout so the valence
    basis (:func:`~.orbitals.pseudo_basis`, first zeta = the bound smooth
    partial wave), the local-potential sampler (:meth:`local_potential`,
    the **ionic** :math:`\tilde v^{ion}`) and the multiple-zeta hierarchy work
    unchanged; every ``channels[l]`` is a :class:`PAWChannel`.  ``projectors``
    maps ``l -> [p~_1(r), p~_2(r)]``, ``coupling`` maps ``l -> D^ion``, and
    ``overlap_correction`` maps ``l -> q`` -- the two block matrices the
    molecular path hands to :class:`~mandacaru.core.hamiltonian.MolecularIntegrals`.
    """

    coupling: dict = field(default_factory=dict)             # l -> D^ion (n, n)
    coupling_screened: dict = field(default_factory=dict)    # l -> D^scr
    overlap_correction: dict = field(default_factory=dict)   # l -> q (charge)
    norm_correction: dict = field(default_factory=dict)      # l -> q^norm
    kinetic_difference: dict = field(default_factory=dict)   # l -> Delta T
    v_local_screened: np.ndarray = None
    core_density: np.ndarray = None                          # n_c(r)
    smooth_core_density: np.ndarray = None                   # n~_c(r)
    r_cut_local: float = 0.0
    local_shift: float = 0.0
    compensation_radius: float = 0.0                         # r_g (Bohr)
    compensation_charge: float = 0.0                         # Q of n^ (electrons)
    hartree_screening: float = 0.0                           # int v_H[n~+n^] g
    one_center_energy: float = 0.0                           # frozen constant
    energies: dict = field(default_factory=dict)             # bookkeeping (Ha)
    q_cut: float | None = None
    energy_offset: float | None = None
    norm_deficit: float | None = None
    #: How the reference atom was solved.  Defaults describe the *pre-
    #: relativistic* construction, so a dataset that does not say carries no
    #: false provenance; a generated or a stored record always has the real values.
    xc: str = "lda"
    relativity: str = "none"
    #: Record of the nonlinear core correction, and channels added above the
    #: highest valence l.
    nlcc: dict = field(default_factory=dict)
    extra_l: int = 0
    #: Occupied valence subshells generated into the frozen core instead
    #: (``(n, l)`` pairs); their channel scatters at positive energy.
    frozen_subshells: tuple = ()
    #: Filled core subshells generated into the valence instead (``(n, l)``
    #: pairs): their states are occupied references of their channels.
    semicore_subshells: tuple = ()
    #: ``l -> D_SO``, the one-center spin-orbit difference of the channel.
    #: Empty unless the dataset was generated with ``relativity="dirac"``.
    #: These multiply the dataset's **own** projectors -- there is one set
    #: per l, not one per j -- so the overlap
    #: operator is untouched and stays diagonal in spin.
    spin_orbit: dict = field(default_factory=dict)
    #: What its generator could not remove (``ghosts="flag"``):
    #: ``{"ghosts": {l: eps_0 - eps_ref}, "phases": {l: (near, far)}}``.
    #: Empty for a clean dataset.  Stored in the file; loading a dataset that
    #: has any raises a :class:`~.defects.GhostStateWarning`.
    defects: dict = field(default_factory=dict)
    #: What the search of :func:`generate_paw` noted (:class:`~.generation.SearchReport`);
    #: ``None`` for a dataset read from a file.
    search: SearchReport | None = None

    def projector_radius(self, l: int) -> float:
        """Radius beyond which channel ``l``'s projectors vanish and the
        smooth wave obeys the all-electron equation: its own ``r_cut``, or
        the local potential's ``r_cut_local`` when that is larger
        (the engine's channel assembly, ``paw::generate::assemble``)."""
        return max(float(self.channels[int(l)].r_cut), float(self.r_cut_local))

    @property
    def has_spin_orbit(self) -> bool:
        """Whether this dataset carries a spin-orbit term."""
        return bool(self.spin_orbit)

    def _local(self, radius):
        """``(v_local, dv_local/dr)`` at ``radius``: the engine's spline of the table
        (``paw::record::values``), flat below the first point, the ionic tail beyond
        the last."""
        radius = np.asarray(radius, dtype=float)
        values, slopes = _native.paw_local_value(
            context(), np.ascontiguousarray(self.r, dtype=np.float64),
            np.ascontiguousarray(self.v_local, dtype=np.float64),
            float(self.valence_charge),
            np.ascontiguousarray(radius.reshape(-1), dtype=np.float64))
        return values.reshape(radius.shape), slopes.reshape(radius.shape)

    def local_potential(self, radius) -> np.ndarray:
        r"""The ionic local potential at arbitrary radii (Bohr), **C\ :sup:`2`**.

        Interpolated with a cubic spline rather than linearly: the PAW-LCAO force is
        the derivative of the energy, and a piecewise-linear potential puts a
        kink at every table point, which left the analytic and finite-difference
        forces depending on their step at the 1e-2 eV/Angstrom level.  Beyond
        the table the potential is the ionic tail :math:`-Z_{ion}/r`; below its
        first point it is flat.
        """
        return self._local(radius)[0]

    def local_potential_derivative(self, radius) -> np.ndarray:
        """``d/dr`` of :meth:`local_potential`, analytic: the spline's
        derivative inside the table (zero below its first point, where the
        potential is flat), ``Z_ion/r^2`` beyond it."""
        return self._local(radius)[1]

    def reference_energies(self, l: int) -> list:
        return list(self.channels[int(l)].reference_energies)

    def projector_set(self, l: int, basis: str = "dual"):
        r"""``(functions, D, q)`` of channel ``l`` in one of the two
        equivalent projector bases.

        ``"dual"``: the PAW-LCAO projectors :math:`\tilde p_i` with the blocks
        :math:`D^{ion}` and :math:`q` as stored.  ``"raw"``: the smooth
        :math:`\chi_k = (\varepsilon_k - T - \tilde v^{scr})\tilde\varphi_k`
        they were built from, with the transformed blocks
        :math:`B^{-1} D B^{-T}` and :math:`B^{-1} q B^{-T}` -- the same
        operators :math:`\sum|\tilde p\rangle D\langle\tilde p|` and
        :math:`\sum|\tilde p\rangle q\langle\tilde p|` exactly, but the
        functions the grid samples are much softer (the dual projectors mix
        two nearly parallel :math:`\chi` with large coefficients, so their
        smooth parts cancel; on a 0.25-0.30 Angstrom grid their norm ratios
        fall to 0.76 while the raw ones stay within 10 %).
        """
        channel = self.channels[int(l)]
        D = np.asarray(channel.coupling, dtype=float)
        q = np.asarray(channel.overlap_correction, dtype=float)
        if basis == "dual":
            return list(channel.projectors), D, q
        if basis == "raw":
            B_inv = np.linalg.inv(np.asarray(channel.vanderbilt, dtype=float))
            return (list(channel.raw_projectors), B_inv @ D @ B_inv.T,
                    B_inv @ q @ B_inv.T)
        raise ValueError(f"unknown projector basis {basis!r}; use 'dual' or "
                         "'raw'")

    def multipole_moments(self, l1: int, l2: int, L: int,
                          basis: str = DEFAULT_PROJECTOR_BASIS) -> np.ndarray:
        r"""``Delta^{(L)}_{ij}`` between channels ``l1`` and ``l2``, in the
        projectors' basis.

        The radial half of the compensation multipole (see
        :mod:`mandacaru.pseudopotentials.multipoles`); ``l1 == l2`` and ``L == 0``
        gives back :attr:`PAWChannel.overlap_correction`.  ``"raw"`` applies the
        same :math:`B^{-1}\cdot B^{-T}` transformation the coupling and overlap
        blocks get, once per side.
        """
        from .multipoles import radial_moments

        key = (int(l1), int(l2), int(L), str(basis))
        cache = getattr(self, "_multipole_cache", None)
        if cache is None:
            cache = self._multipole_cache = {}
        if key not in cache:
            if int(l1) == int(l2) and int(L) == 0:
                # The monopole is stored exactly; reconstructing it from the
                # tabulated waves agrees only to ~1e-8, which the raw basis's
                # large B^{-1} amplifies to ~1e-4.  Use the stored value so the
                # L = 0 term stays bit-for-bit what the monopole-only code had.
                delta = np.asarray(self.channels[int(l1)].overlap_correction,
                                   dtype=float)
            else:
                delta = radial_moments(self, l1, l2, L)
            if basis == "raw":
                b1 = np.linalg.inv(np.asarray(
                    self.channels[int(l1)].vanderbilt, dtype=float))
                b2 = np.linalg.inv(np.asarray(
                    self.channels[int(l2)].vanderbilt, dtype=float))
                delta = b1 @ delta @ b2.T
            elif basis != "dual":
                raise ValueError(f"unknown projector basis {basis!r}")
            cache[key] = delta
        return cache[key]

    def __repr__(self) -> str:
        channels = ", ".join(f"l={l}x{len(self.projectors.get(l, []))}"
                             for l in sorted(self.channels))
        return (f"PAWDataset({self.symbol}, Z_ion={self.valence_charge:g}, "
                f"[{channels}], rg={self.compensation_radius:.2f}, "
                f"Q^={self.compensation_charge:+.4f}, "
                f"E1c={self.one_center_energy:+.4f} Ha, "
                f"deficit={_deficit_label(self.norm_deficit)}"
                f"{', GHOSTED' if (self.defects or {}).get('ghosts') else ''}"
                f"{', SCATTERING OFF' if (self.defects or {}).get('phases') else ''})")


# --------------------------------------------------------------------------- #
# The projector blocks of the periodic setup (they die with it).
# --------------------------------------------------------------------------- #

def paw_projectors(symbols, positions, datasets, units: str = "angstrom",
                   projector_basis: str = DEFAULT_PROJECTOR_BASIS, *,
                   real: bool = True):
    """The :class:`~.orbitals.KBProjector` set: every radial projector of
    every ``(atom, l, m)``, in the ``projector_basis`` of
    :meth:`PAWDataset.projector_set` (each projector records it as
    ``projector_basis`` so the block builders transform consistently), with
    real spherical harmonics (a molecule) or complex ones (``real=False``, a
    crystal).  The compensation channels follow the projectors'."""
    from .orbitals import KBProjector

    projectors = []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = datasets[symbol]
        for l in sorted(pp.projectors):
            functions, D, _q = pp.projector_set(l, projector_basis)
            for m in range(-l, l + 1):
                for i, p in enumerate(functions):
                    projector = KBProjector(
                        pp, l, m, center=position, units=units,
                        atom_index=index, index=i, radial=p,
                        kb_energy=float(D[i, i]), real=real)
                    projector.projector_basis = projector_basis
                    projectors.append(projector)
    return projectors


def _blocks(projectors, symbols, datasets, which) -> dict:
    blocks = {}
    for projector in projectors:
        key = projector.block_key
        if key not in blocks:
            pp = datasets[symbols[projector.atom_index]]
            basis = getattr(projector, "projector_basis",
                            DEFAULT_PROJECTOR_BASIS)
            _f, D, q = pp.projector_set(projector.l, basis)
            blocks[key] = np.asarray(D if which == "coupling" else q,
                                     dtype=float)
    return blocks


def paw_spin_orbit_projectors(symbols, positions, datasets,
                              units: str = "angstrom", *, real: bool = True):
    """The projectors the spin-orbit term acts through: for every atom with a
    Dirac dataset, every ``(l, m)`` of each channel that carries the term and
    every function of its union of the two j branches
    (:func:`j_resolved_spin_orbit`, j = l-1/2 first), in the dual basis they
    are stored in.  Empty for scalar datasets."""
    from .orbitals import KBProjector

    projectors = []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = datasets[symbol]
        for l, entry in sorted((getattr(pp, "spin_orbit", None) or {}).items()):
            for m in range(-l, l + 1):
                for i, p in enumerate(entry["projectors"]):
                    projectors.append(KBProjector(
                        pp, l, m, center=position, units=units,
                        atom_index=index, index=i, radial=p, kb_energy=0.0,
                        real=real))
    return projectors


def paw_spin_orbit_blocks(projectors, symbols, datasets) -> dict:
    r"""``{(atom, l): D_SO}`` over :func:`paw_spin_orbit_projectors`.

    The L.S part of the j-resolved term, unscreened like the scalar
    :math:`D^{ion}` (the molecule screens both).  Its (2j+1) average is not
    used: the scalar channel stands in for it (mandacaru-paw/FR.md, option (a)), which
    keeps the overlap, the compensation charges, the basis and the forces
    those of the scalar channel.  The price, measured on the atom's radial
    operator: the 6p of Pb and Bi 0.6-1.6 mHa off at the centroid, up to
    2.6 mHa for 6p1/2, and the splitting 2.0-2.4 % short (I 5p 0.4 %, d
    shells <= 0.15 %); with the stored average, under a spin-free metric,
    both branches are within 0.02 mHa (HISTORY.md, 2026-10-11, "The PAW
    audit report checked").  Using it needs the averaged channel's own
    partial waves (compensation multipoles, one-center terms), which the
    files do not store.
    """
    blocks: dict = {}
    for projector in projectors:
        key = (projector.atom_index, projector.l)
        if key not in blocks:
            table = datasets[symbols[projector.atom_index]].spin_orbit
            blocks[key] = np.asarray(table[projector.l]["coupling"],
                                     dtype=float)
    return blocks


def paw_coupling_blocks(projectors, symbols, datasets) -> dict:
    """``{(atom, l, m): D_l}`` for ``nonlocal_coupling`` (in the
    projectors' basis)."""
    return _blocks(projectors, symbols, datasets, "coupling")


def paw_multipole_blocks(projectors, datasets_by_atom) -> dict:
    r"""``{(atom, L, M): matrix}`` -- the compensation multipole moments.

    ``matrix[a, b]`` is the ``LM`` moment carried by the pair density of this
    atom's projectors ``a`` and ``b``, i.e. the radial moment
    :math:`\Delta^{(L)}` times the angular coupling
    :math:`\int Y^*_{LM}Y^*_{l_am_a}Y_{l_bm_b}d\Omega`.  Rows and columns are
    numbered within the atom, in the order the projectors appear;
    ``datasets_by_atom[atom]`` is that atom's dataset.

    ``L = 0`` reproduces ``paw_overlap_blocks`` scaled by
    :math:`1/\sqrt{4\pi}`, which is the normalization that makes the
    :math:`L = 0` term of the augmentation identical to the monopole-only code
    it replaces.  Channels that carry no moment are omitted.
    """
    from .multipoles import gaunt, multipole_range

    per_atom: dict = {}
    for position, projector in enumerate(projectors):
        per_atom.setdefault(projector.atom_index, []).append(
            (position, projector))

    blocks: dict = {}
    for atom, entries in per_atom.items():
        dataset = datasets_by_atom[atom]
        basis = getattr(entries[0][1], "projector_basis",
                        DEFAULT_PROJECTOR_BASIS)
        n = len(entries)
        levels = sorted({L for _p, a in entries for _q, b in entries
                         for L in multipole_range(a.l, b.l)})
        dtype = float if entries[0][1].real else complex
        for L in levels:
            for M in range(-L, L + 1):
                matrix = np.zeros((n, n), dtype=dtype)
                for a, (_pa, pa) in enumerate(entries):
                    for b, (_pb, pb) in enumerate(entries):
                        angular = gaunt(L, M, pa.l, pa.m, pb.l, pb.m,
                                        real=pa.real)
                        if angular == 0:
                            continue
                        delta = dataset.multipole_moments(pa.l, pb.l, L, basis)
                        matrix[a, b] = delta[pa.index, pb.index] * angular
                if np.any(np.abs(matrix) > 1e-14):
                    blocks[(atom, L, M)] = matrix
    return blocks


def paw_overlap_blocks(projectors, symbols, datasets) -> dict:
    """``{(atom, l, m): q_l}`` for ``nonlocal_overlap`` (in the
    projectors' basis)."""
    return _blocks(projectors, symbols, datasets, "overlap")


def _projector_sphere(projector, *, split_radial: bool = False):
    """Quadrature points (Bohr, a coordinate triple) and weights over the
    sphere of radius ``projector.r_cut`` around ``projector.center``."""
    x, wx = np.polynomial.legendre.leggauss(PROJECTION_RADIAL_POINTS)
    t, wt = np.polynomial.legendre.leggauss(PROJECTION_POLAR_POINTS)
    n_phi = PROJECTION_AZIMUTHAL_POINTS
    azimuth = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    sin_t = np.sqrt(1.0 - t * t)
    directions = np.stack([(sin_t[:, None] * np.cos(azimuth)[None, :]).ravel(),
                           (sin_t[:, None] * np.sin(azimuth)[None, :]).ravel(),
                           np.repeat(t, n_phi)])
    angular = np.repeat(wt, n_phi) * (2.0 * np.pi / n_phi)
    r_cut = float(projector.r_cut)
    channel_cut = float(getattr(projector, "channel_r_cut", r_cut))
    # A projector continued from its own cutoff to the local radius
    # changes formula at channel_cut.  Put that junction on a panel boundary
    # so the radial rule sees smooth functions on both sides.
    edges = ([0.0, channel_cut, r_cut]
             if split_radial and 0.0 < channel_cut < r_cut else [0.0, r_cut])
    radii, radial_weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        half = 0.5 * (hi - lo)
        nodes = lo + half * (x + 1.0)
        radii.append(nodes)
        radial_weights.append(half * wx * nodes * nodes)
    radii = np.concatenate(radii)
    weights = np.concatenate(radial_weights)[:, None] * angular[None, :]
    center = np.asarray(projector.center, dtype=float)
    points = tuple(center[i] + radii[:, None] * directions[i][None, :]
                   for i in range(3))
    return points, weights


# --------------------------------------------------------------------------- #
# Libraries.
# --------------------------------------------------------------------------- #

def paw_library_path(paw_dataset=None, xc: str = DEFAULT_XC, *,
                     must_exist: bool = True,
                     relativity: str = "scalar") -> str:
    """The PAW-LCAO library folder: ``paw_dataset`` when given, else
    ``$MANDACARU_PAW_PATH/lda-sr`` (``lda-fr`` for ``relativity="dirac"``;
    :func:`.environment.paw_dataset_folder`)."""
    from .environment import paw_dataset_folder
    return paw_dataset_folder(FAMILY, xc, paw_dataset, must_exist=must_exist,
                             relativity=relativity)


_CACHE: dict = {}


def get_paw(symbol: str, paw_dataset=None, xc: str = DEFAULT_XC) -> PAWDataset:
    """Load ``symbol`` from the PAW-LCAO library (cached): ``paw_dataset``, or
    ``$MANDACARU_PAW_PATH/lda-sr``."""
    from .io import load_library_dataset

    return load_library_dataset(
        symbol, paw_library_path(paw_dataset, xc), FAMILY, _CACHE,
        label="PAW-LCAO", noun="dataset", builder="build_paw_library")


def build_paw_library(elements=("H", "Li", "C", "N", "O", "F"),
                      paw_dataset=None, *, verbose: bool = True,
                      format: str | None = None, stride: int | None = None,
                      **generation_options):
    """Generate and save PAW-LCAO datasets for ``elements``; returns the paths."""
    from .io import DEFAULT_FORMAT, STRIDE, library_file, save_pseudopotential

    folder = paw_library_path(
        paw_dataset, generation_options.get("xc", DEFAULT_XC), must_exist=False,
        relativity=generation_options.get("relativity", "scalar"))
    os.makedirs(folder, exist_ok=True)
    format = DEFAULT_FORMAT if format is None else format
    stride = STRIDE if stride is None else int(stride)
    written = []
    for symbol in elements:
        pp = generate_paw(symbol, **generation_options)
        path = save_pseudopotential(pp, library_file(symbol, folder, format),
                                    format=format, stride=stride)
        written.append(path)
        if verbose:
            print(f"  {symbol:>2}  Z_ion={pp.valence_charge:>4.0f}  "
                  + "  ".join(f"l{l}: rc={c.r_cut:.2f} q11="
                              f"{c.overlap_correction[0, 0]:+.4f}"
                              for l, c in sorted(pp.channels.items()))
                  + f"  E1c={pp.one_center_energy:+.4f}  -> "
                  f"{os.path.basename(path)}")
    _CACHE.clear()
    return written


def upaw_library_path(paw_dataset=None, xc: str = DEFAULT_XC,
                      relativity: str = "scalar") -> str | None:
    """The UPAW-LCAO library folder: ``paw_dataset`` when given, else
    ``$MANDACARU_UPAW_PATH/lda-sr`` (``lda-fr`` for ``relativity="dirac"``), or
    ``None`` when that variable is unset (UPAW-LCAO is then generated on
    demand)."""
    from .environment import upaw_dataset_folder
    return upaw_dataset_folder(xc, paw_dataset, relativity)


def get_upaw(symbol: str, paw_dataset=None, xc: str = DEFAULT_XC) -> PAWDataset:
    """Load ``symbol`` from the UPAW-LCAO library, **or generate it** (cached).

    Unlike PAW-LCAO, UPAW-LCAO has no shipped library: there is no
    sibling data repository for it, and requiring a 92-element build before the
    option can be tried at all would make it unusable.  Generation is a few
    seconds per element (H 0.4 s, O 2.2 s, measured) and the result is cached
    for the process, so a small molecule pays that once.  Build a library with
    :func:`build_upaw_library` to skip it, and it is used whenever it exists.
    """
    from .io import library_file, load_pseudopotential

    folder = upaw_library_path(paw_dataset, xc)
    key = f"{symbol}@{folder}@{xc}@{UPAW_FAMILY}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    path = None if folder is None else library_file(symbol, folder)
    if path is not None and os.path.exists(path):
        pp = load_pseudopotential(path)
        family = str(getattr(pp, "family", "")).lower()
        if family != UPAW_FAMILY:
            raise ValueError(f"{path!r} belongs to family {family!r}, not "
                             f"{UPAW_FAMILY!r}")
    else:
        if paw_dataset is not None:
            raise FileNotFoundError(
                f"no UPAW-LCAO dataset for {symbol!r} at {path!r}; build one with "
                f"build_upaw_library([{symbol!r}], paw_dataset={paw_dataset!r})")
        where = ("MANDACARU_UPAW_PATH is not set" if folder is None
                 else f"no library at {folder!r}")
        warnings.warn(
            f"generating a UPAW-LCAO dataset for {symbol} ({where}); it is "
            f"cached for this process.  Build one once with "
            f"mandacaru.pseudopotentials.paw.build_upaw_library([...]) to skip "
            f"this.", RuntimeWarning, stacklevel=2)
        pp = generate_upaw(symbol, xc=xc)
    _CACHE[key] = pp
    return pp


def build_upaw_library(elements=("H", "Li", "C", "N", "O", "F"),
                       paw_dataset=None, *, verbose: bool = True,
                       format: str | None = None, stride: int | None = None,
                       **generation_options):
    """Generate and save UPAW-LCAO datasets for ``elements``; returns the paths."""
    from .io import DEFAULT_FORMAT, STRIDE, library_file, save_pseudopotential

    folder = upaw_library_path(
        paw_dataset, generation_options.get("xc", DEFAULT_XC),
        generation_options.get("relativity", "scalar"))
    if folder is None:
        from .environment import repository_path
        repository_path(UPAW_FAMILY)     # raises the "not set" explanation
    os.makedirs(folder, exist_ok=True)
    format = DEFAULT_FORMAT if format is None else format
    stride = STRIDE if stride is None else int(stride)
    written = []
    for symbol in elements:
        pp = generate_upaw(symbol, **generation_options)
        path = save_pseudopotential(pp, library_file(symbol, folder, format),
                                    format=format, stride=stride)
        written.append(path)
        if verbose:
            print(f"  {symbol:>2}  Z_ion={pp.valence_charge:>4.0f}  "
                  + "  ".join(f"l{l}: rc={c.r_cut:.2f} |q|="
                              f"{abs(c.overlap_correction[0, 0]):.1e}"
                              for l, c in sorted(pp.channels.items()))
                  + f"  E1c={pp.one_center_energy:+.4f}  -> "
                  f"{os.path.basename(path)}")
    _CACHE.clear()
    return written


# --------------------------------------------------------------------------- #
# Registration.
# --------------------------------------------------------------------------- #

#: Both PAW-LCAO families filter their basis by default (see
#: :mod:`mandacaru.basis.filtering`).  A PAW-LCAO smooth partial wave is already
#: *built* to be band-limited -- the generator's constrained minimization (``paw::pseudize``)
#: minimizes the kinetic energy beyond ``q_cut`` -- so removing what is left above the
#: grid's Nyquist wave-vector costs little and takes most of the egg-box with
#: it.  ``basis={"name": "PAW-LCAO", "filter_cutoff": False}`` restores
#: the unfiltered basis exactly.
#:
#: ``energy_shift = 0.1`` eV makes the default PAW-LCAO basis a
#: **confined** one, as for every pseudopotential family
#: (:data:`~.confinement.CONFINEMENT_DEFAULT_OPTIONS`); the polarization shell
#: then defaults to the quasi-Gaussian.
PAW_DEFAULT_OPTIONS = {"filter_cutoff": "auto", **CONFINEMENT_DEFAULT_OPTIONS}

#: What a PAW-LCAO / UPAW-LCAO basis dict may say beyond the family-independent
#: options: which projector set is sampled, the confinement options every
#: pseudopotential family shares (:data:`~.confinement.CONFINEMENT_OPTIONS`)
#: and the diffuse shells (:mod:`.diffuse`; off unless written).
PAW_EXTRA_OPTIONS = ("projector_basis",) + CONFINEMENT_OPTIONS + ("diffuse",)


def _register():
    from .families import (COMMON_OPTIONS, FamilySpec, PSEUDO_FAMILIES,
                           register_family)
    PAW_OPTIONS = COMMON_OPTIONS + PAW_EXTRA_OPTIONS
    if FAMILY in PSEUDO_FAMILIES:
        return PSEUDO_FAMILIES[FAMILY]
    return register_family(FamilySpec(
        name=FAMILY,
        description="projector augmented wave (Bloechl 1994), frozen core, "
                    "multipole compensation",
        generate=lambda symbol, **options: generate_paw(symbol, **options),
        get=get_paw,

        norm_conserving=False,
        aliases=(),
        options=PAW_OPTIONS,
        default_options=dict(PAW_DEFAULT_OPTIONS),
    ))


def _register_upaw():
    """Register the **unitary** PAW-LCAO family (``basis="UPAW-LCAO"``).

    Same machinery as PAW-LCAO with :math:`q = 0`, so the transformation is unitary
    and the overlap operator is the identity.  Kept as an option rather than the
    default: Mandacaru orthogonalizes the augmented overlap anyway, the constraint
    fixes only the monopole (the higher compensation multipoles survive), and the
    harder waves cost about five times the grid egg-box on oxygen (measured
    with the generator's smooth waves, ``paw::generate::waves``).
    """
    from .families import (COMMON_OPTIONS, PSEUDO_FAMILIES, FamilySpec,
                           register_family)
    PAW_OPTIONS = COMMON_OPTIONS + PAW_EXTRA_OPTIONS
    if UPAW_FAMILY in PSEUDO_FAMILIES:
        return PSEUDO_FAMILIES[UPAW_FAMILY]
    return register_family(FamilySpec(
        name=UPAW_FAMILY,
        description="unitary projector augmented wave (Ivanov 2024, "
                    "arXiv:2408.03159): PAW-LCAO with q = 0, so T is unitary and "
                    "the pseudo states are orthonormal",
        generate=lambda symbol, **options: generate_upaw(symbol, **options),
        get=get_upaw,

        norm_conserving=False,
        aliases=("unitary-paw-lcao",),
        options=PAW_OPTIONS,
        # UPAW-LCAO's partial waves are the harder ones (five times water's grid
        # egg-box at grid_spacing = 0.25), so if the filter earns its place for PAW-LCAO it
        # earns it for UPAW-LCAO a fortiori.
        default_options=dict(PAW_DEFAULT_OPTIONS),
    ))


PAW_FAMILY = _register()
UPAW_FAMILY_SPEC = _register_upaw()
