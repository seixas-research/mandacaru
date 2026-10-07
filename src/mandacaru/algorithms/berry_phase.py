# -*- coding: utf-8 -*-
# file: algorithms/berry_phase.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Electric polarization of a periodic insulator: the Berry phase.

The polarization of a crystal is not the dipole of its cell -- that depends
on where the cell is cut -- but a Berry phase of the occupied Bloch states
(King-Smith and Vanderbilt 1993; Resta 1994).  Along a string of
:math:`N` k-points :math:`\mathbf k_s = \mathbf k_\perp + s\,\mathbf b_j/N`
spanning the reciprocal vector :math:`\mathbf b_j`,

.. math::

    \phi_j = -\operatorname{Im}\ln\prod_{s=0}^{N-1}
             \det M(\mathbf k_s, \mathbf k_{s+1}),
    \qquad
    M_{mn}(\mathbf k, \mathbf k') = \langle u_{m\mathbf k}|u_{n\mathbf k'}\rangle
      = \langle\psi_{m\mathbf k}|e^{-i\mathbf b\cdot\mathbf r}
        |\psi_{n\mathbf k'}\rangle

over the occupied bands, averaged over the strings of the transverse mesh.
The electrons contribute :math:`\mathbf P_{el} = -(f e/2\pi\Omega)
\sum_j\phi_j\mathbf a_j` (:math:`f = 2` for a spin-restricted crystal; a
spin-polarized one sums its two channels with :math:`f = 1`), the ions
:math:`\mathbf P_{ion} = (e/\Omega)\sum_A Z_A\mathbf R_A` with the valence
charges.  The Berry phase is defined modulo :math:`2\pi`, so
:math:`\mathbf P_{el}` modulo :math:`f e\,\mathbf a_j/\Omega`, and moving
one ion by a lattice vector changes :math:`\mathbf P_{ion}` by
:math:`Z_A e\,\mathbf a_j/\Omega`: :math:`\mathbf P` is defined modulo the
quantum :math:`g\,e\,\mathbf a_j/\Omega` with :math:`g = \gcd(f, Z_1,
Z_2, \dots)`.  It is reported on the branch nearest zero, together with
the quanta; only a *change* of :math:`\mathbf P` along a path is physical.

**The overlaps in the PAW-LCAO basis.**  :math:`M = V_{\mathbf k}^\dagger
S(\mathbf k, \mathbf k') V_{\mathbf k'}`, with :math:`V` the occupied
eigenvectors and

.. math::

    S_{\mu\nu}(\mathbf k, \mathbf k') = \int_\Omega \chi^*_{\mu\mathbf k}
    e^{-i\mathbf b\cdot\mathbf r}\chi_{\nu\mathbf k'}\,d^3r
    + \sum_A e^{-i\mathbf b\cdot\mathbf R_A}\,
      C_A(\mathbf k)\,[q_A - i\,\mathbf b\cdot\mathbf d_A]\,
      C_A(\mathbf k')^\dagger,

the smooth part on the cell grid and the on-site part to first order in
:math:`\mathbf b` inside each sphere: the augmentation charge and dipole
(the compensation moments, :meth:`PeriodicPAW.moment_operator`).  The
neglected terms are :math:`O(|\mathbf b|^2 r_c^2)` and vanish as the strings
get denser.  The string closes for free: a Bloch sum at
:math:`\mathbf k + \mathbf G` is the one at :math:`\mathbf k`, so the last
link reuses the first point's states (the periodic gauge).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..integrals import reciprocal as rc
from ..units import BOHR_TO_ANGSTROM

#: An occupation within this of empty or full counts as that: anything in
#: between is a metal (or a smearing too wide for the gap) and is refused.
OCCUPATION_TOLERANCE = 1e-4

#: Most memory (bytes) the Bloch sums of the whole mesh may take: within it
#: every point is diagonalized once and shared by the three directions'
#: strings; beyond it each string is diagonalized on its own (three times
#: the work, one string's memory).
MESH_CACHE_BYTES = 2 * 1024 ** 3

#: e / Bohr^2 in C/m^2.
E_PER_BOHR2_IN_C_PER_M2 = 1.602176634e-19 / (BOHR_TO_ANGSTROM * 1e-10) ** 2


@dataclass
class Polarization:
    """The Berry-phase polarization of a crystal (C/m^2, Cartesian).

    ``vector`` is the total on the branch nearest zero; ``fractional`` its
    coordinates in units of the quanta (each in (-1/2, 1/2]), whose rows
    ``quanta`` are :math:`g e\\,\\mathbf a_j/\\Omega`; ``raw`` the
    coordinates before the reduction (ionic minus Berry phase), which a
    finite difference along a path should compare.  ``phases`` are the
    electrons' Berry phases per lattice direction (radians, summed over spin
    channels), ``electronic`` and ``ionic`` the two parts (unreduced).
    """

    vector: np.ndarray
    fractional: np.ndarray
    raw: np.ndarray
    quanta: np.ndarray
    phases: np.ndarray
    electronic: np.ndarray
    ionic: np.ndarray
    strings: tuple

    def summary(self) -> str:
        lines = ["Berry-phase polarization (C/m^2)",
                 f"  P        {np.round(self.vector, 6)}",
                 f"  in quanta {np.round(self.fractional, 6)} "
                 "(branch nearest zero)",
                 f"  strings  {self.strings}"]
        lines += [f"  quantum {j + 1} {np.round(q, 6)}"
                  for j, q in enumerate(self.quanta)]
        return "\n".join(lines)


def occupied_bands(result) -> list:
    """Occupied bands per spin channel of an insulating crystal.

    Every occupation must be empty or full to :data:`OCCUPATION_TOLERANCE`,
    with the same count at every k-point; otherwise the occupied manifold is
    not separated from the empty one and its Berry phase is not defined.
    """
    spin = getattr(result, "n_spins", 1) == 2
    full = 1.0 if spin else 2.0
    occupations = [np.asarray(f, dtype=float) for f in result.occupations]
    counts = []
    for channel in range(2 if spin else 1):
        per_k = set()
        for f in occupations:
            row = f[channel] if spin else f
            partial = (row > OCCUPATION_TOLERANCE) & (
                row < full - OCCUPATION_TOLERANCE)
            if np.any(partial):
                raise NotImplementedError(
                    "the Berry-phase polarization needs an insulator: a "
                    f"state is occupied {row[partial][0]:.4f} of {full:g} "
                    "(a metal, or a smearing wider than the gap -- narrow "
                    "it, e.g. smearing={'method': 'fermi-dirac', "
                    "'width': 0.001})")
            per_k.add(int(np.sum(row > 0.5 * full)))
        if len(per_k) != 1:
            raise NotImplementedError(
                "the number of occupied bands changes across the Brillouin "
                "zone: the bands overlap (a semimetal), so there is no "
                "separated occupied manifold")
        counts.append(per_k.pop())
    return counts


def onsite_operator(crystal, b) -> np.ndarray:
    r"""``(P, P)``: :math:`\sum_A e^{-i\mathbf b\cdot\mathbf R_A}[q_A -
    i\,\mathbf b\cdot\mathbf d_A]` on the projector blocks
    (:meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.moment_operator`):
    the augmentation's monopole and dipole, with the combinations of
    :func:`~mandacaru.algorithms.volumetric.dipole_moment`."""
    b = np.asarray(b, dtype=float)
    root4 = np.sqrt(4.0 * np.pi)
    root8 = np.sqrt(8.0 * np.pi / 3.0) / 2.0
    root3 = np.sqrt(4.0 * np.pi / 3.0)
    weights = {}
    for key in crystal.multipole_blocks:
        atom, L, m = key
        phase = np.exp(-1j * float(b @ crystal.centers[atom]))
        if L == 0:
            value = root4
        elif L == 1 and m == 0:
            value = root3 * (-1j * b[2])
        elif L == 1 and m == -1:
            value = root8 * (-1j * b[0] - b[1])
        elif L == 1 and m == 1:
            value = root8 * (1j * b[0] - b[1])
        else:
            value = 0.0
        weights[key] = phase * value
    return crystal.moment_operator(weights)


def link_overlap(crystal, first, second, b, onsite, grid_phase) -> np.ndarray:
    r"""``(M, M)``: :math:`S(\mathbf k, \mathbf k')` between the k-point
    matrices ``first`` and ``second`` (``k' = k + b``, or the image of the
    string's first point)."""
    smooth = ((first.psi.conj() * grid_phase) @ second.psi.T) * crystal.grid.dV
    return smooth + first.projections @ onsite @ second.projections.conj().T


def string_phase(crystal, states, n_occ, b) -> float:
    r"""Berry phase :math:`-\sum_s\arg\det M(\mathbf k_s, \mathbf k_{s+1})`
    of one closed string: ``states`` are its ``(data, vectors)`` in order."""
    grid = crystal.grid
    r = np.stack([np.ravel(grid.X), np.ravel(grid.Y), np.ravel(grid.Z)])
    grid_phase = np.exp(-1j * (np.asarray(b) @ r))
    onsite = onsite_operator(crystal, b)
    phase = 0.0
    for s in range(len(states)):
        data, V = states[s]
        data_next, V_next = states[(s + 1) % len(states)]
        S = link_overlap(crystal, data, data_next, b, onsite, grid_phase)
        M = V[:, :n_occ].conj().T @ S @ V_next[:, :n_occ]
        sign, logdet = np.linalg.slogdet(M)
        phase -= float(np.angle(sign))
    return phase


def polarization_quantum(f: float, charges) -> int:
    """``g = gcd(f, Z_1, Z_2, ...)``: the polarization is defined modulo
    ``g e a_j / Omega`` -- the electrons' Berry phase modulo ``f e a /
    Omega``, each ion's position modulo a lattice vector.  A fractional
    valence charge (none of the shipped datasets has one) leaves only the
    electrons' quantum."""
    values = [int(round(f))]
    for Z in np.asarray(charges, dtype=float):
        if abs(Z - round(Z)) > 1e-8:
            return int(round(f))
        values.append(int(round(Z)))
    return int(np.gcd.reduce(values))


def _average_phases(phases) -> float:
    """Mean of string phases on the branch of the first: each differs from
    it by less than pi once wrapped."""
    phases = np.asarray(phases, dtype=float)
    reference = phases[0]
    aligned = reference + np.angle(np.exp(1j * (phases - reference)))
    return float(np.mean(aligned))


def berry_polarization(solver, result, size) -> Polarization:
    """:class:`Polarization` of a converged periodic Kohn-Sham crystal.

    ``solver`` is the :class:`~mandacaru.algorithms.periodic_dft.
    PeriodicKohnSham` run and ``result`` its result (the occupations);
    ``size`` the ``(n1, n2, n3)`` mesh the strings are cut from: ``n_j``
    points along each string of direction ``j``, the other two the
    transverse mesh.  All points are diagonalized non-self-consistently at
    the converged potential.
    """
    crystal = solver.crystal
    counts = occupied_bands(result)
    spin = solver.n_spins == 2
    f = 1.0 if spin else 2.0
    size = tuple(int(n) for n in size)
    if len(size) != 3 or min(size) < 1:
        raise ValueError(f"size must be three positive integers; got {size!r}")
    lattice = np.asarray(crystal.lattice, dtype=float)       # columns a_j
    B = rc.reciprocal_vectors(lattice)                        # columns b_j
    for j in range(3):
        if size[j] < 2:
            raise ValueError(
                f"a string along direction {j + 1} needs at least 2 "
                f"k-points; size is {size}")
    channels = solver.channel_potentials()
    mesh = np.indices(size).reshape(3, -1).T            # C order
    cached = (len(mesh) * crystal.M * crystal.grid.size * 16
              <= MESH_CACHE_BYTES)
    phases = np.zeros(3)
    for V_args, n_occ in zip(channels, counts):
        cache = None
        if cached:
            cache = []
            for data, _eps, vectors in solver._diagonalized(
                    (mesh / np.asarray(size, float)) @ B.T, *V_args):
                cache.extend(zip(data, vectors))
        for j in range(3):
            others = [d for d in range(3) if d != j]
            b = B[:, j] / size[j]
            per_string = []
            for i in range(size[others[0]]):
                for i2 in range(size[others[1]]):
                    index = np.zeros((size[j], 3), dtype=int)
                    index[:, j] = np.arange(size[j])
                    index[:, others[0]] = i
                    index[:, others[1]] = i2
                    if cache is not None:
                        states = [cache[int(n)] for n in
                                  np.ravel_multi_index(index.T, size)]
                    else:
                        states = []
                        for data, _eps, vectors in solver._diagonalized(
                                (index / np.asarray(size, float)) @ B.T,
                                *V_args):
                            states.extend(zip(data, vectors))
                    per_string.append(string_phase(crystal, states, n_occ, b))
            phases[j] += _average_phases(per_string)
    return polarization_from_phases(crystal, phases, f, size)


def polarization_from_phases(crystal, phases, f: float, size) -> Polarization:
    """The :class:`Polarization` of the electrons' Berry ``phases`` (per
    lattice direction, summed over spin channels) plus the ions."""
    lattice = np.asarray(crystal.lattice, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    phases = np.asarray(phases, dtype=float)
    # Coordinates in units of the quanta g e a_j / Omega.
    charges = np.asarray(crystal.charges, dtype=float)
    centers = np.asarray(crystal.centers, dtype=float)
    g = polarization_quantum(f, charges)
    fractional_atoms = centers @ np.linalg.inv(lattice).T
    ionic_coordinates = (charges @ fractional_atoms) / g
    electronic_coordinates = -(f / g) * phases / (2.0 * np.pi)
    raw = ionic_coordinates + electronic_coordinates
    reduced = raw - np.round(raw)
    reduced = np.where(reduced <= -0.5 + 1e-9, reduced + 1.0, reduced)
    quanta = (g / volume) * lattice.T * E_PER_BOHR2_IN_C_PER_M2    # rows
    return Polarization(
        vector=reduced @ quanta, fractional=reduced, raw=raw, quanta=quanta,
        phases=phases, electronic=electronic_coordinates @ quanta,
        ionic=ionic_coordinates @ quanta, strings=tuple(size))


#: Displacement (Angstrom) of the Born effective charges' central differences.
DEFAULT_BORN_DELTA = 0.01


@dataclass
class BornCharges:
    """Born effective charges :math:`Z^*_{A,ij} = (\\Omega/e)\\,\\partial P_i
    /\\partial u_{Aj}` (units of e), ``tensors`` ``(n_atoms, 3, 3)``.

    ``sum_rule`` is :math:`\\sum_A Z^*_A`, which vanishes for an exact
    calculation (the acoustic sum rule); its size measures the k-point
    convergence."""

    tensors: np.ndarray
    delta: float
    symbols: tuple

    @property
    def sum_rule(self) -> np.ndarray:
        return self.tensors.sum(axis=0)

    @property
    def isotropic(self) -> np.ndarray:
        """``Tr Z*_A / 3`` per atom."""
        return np.trace(self.tensors, axis1=1, axis2=2) / 3.0


def born_effective_charges(atoms, *, delta: float = DEFAULT_BORN_DELTA,
                           kpts_polarization=None, **options) -> BornCharges:
    """Born effective charges of an insulating crystal by central
    differences of the Berry-phase polarization.

    Every atom is moved by ``+-delta`` (Angstrom) along each Cartesian axis
    and the crystal recomputed with ``options`` (those of
    ``Mandacaru(method="dft", ...)``, ``kpts`` and ``smearing`` included):
    :math:`6N` self-consistent runs.  The raw polarization coordinates are
    differenced, wrapped to the nearest quantum, so a branch jump cannot
    pass for a charge.  ``kpts_polarization`` is the string mesh (default:
    the SCF's).
    """
    from ..units import ANGSTROM_TO_BOHR
    from .calculator import Mandacaru

    if not bool(np.all(atoms.get_pbc())):
        raise NotImplementedError("Born effective charges are for a crystal "
                                  "periodic in three directions")
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be a positive length; got {delta!r}")

    def raw(geometry):
        geometry.calc = Mandacaru(**options)
        geometry.get_potential_energy()
        geometry.calc.get_polarization(kpts=kpts_polarization)
        result = geometry.calc.polarization_result
        crystal = geometry.calc.solver._periodic_solver.crystal
        g = polarization_quantum(
            1.0 if geometry.calc.solver.get_number_of_spins() == 2 else 2.0,
            crystal.charges)
        return result.raw, g * np.asarray(crystal.lattice, float).T

    n_atoms = len(atoms)
    tensors = np.zeros((n_atoms, 3, 3))
    for atom in range(n_atoms):
        for axis in range(3):
            moved = []
            for sign in (1.0, -1.0):
                geometry = atoms.copy()
                geometry.positions[atom, axis] += sign * delta
                moved.append(raw(geometry))
            (plus, quanta), (minus, _q) = moved
            change = plus - minus
            change -= np.round(change)
            tensors[atom, :, axis] = (change @ quanta) / (
                2.0 * delta * ANGSTROM_TO_BOHR)
    return BornCharges(tensors=tensors, delta=delta,
                       symbols=tuple(atoms.get_chemical_symbols()))


#: Strain amplitude of the piezoelectric central differences.
DEFAULT_PIEZO_STRAIN = 0.005

#: Voigt order of the strains: xx, yy, zz, yz, xz, xy (engineering shears).
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@dataclass
class PiezoelectricTensor:
    """The proper piezoelectric tensor :math:`e_{i\\mu} = \\partial P_i /
    \\partial\\eta_\\mu` (C/m^2), ``tensor`` ``(3, 6)`` in Voigt order.

    ``relaxed`` says whether the internal coordinates were relaxed at every
    strain (the relaxed-ion tensor) or kept in fractional coordinates (the
    clamped-ion one)."""

    tensor: np.ndarray
    strain: float
    relaxed: bool


class _FixedCentroid:
    """An ASE constraint that keeps the centroid of the atoms fixed and
    removes the mean force.

    A crystal's forces sum to zero exactly: moving every atom together
    changes nothing.  The real-space grid breaks that symmetry (the
    egg-box), and wurtzite AlN at h = 0.25 Angstrom carries a net force of
    0.12 eV/Angstrom, as large as the forces of the internal strain.  Left
    in, it drags the whole crystal along the grid during a relaxation (0.1
    Angstrom in seven extra BFGS steps) before the internal coordinates
    settle.  Removing the mean is an orthogonal projection onto forces that
    sum to zero, which the exact forces do."""

    def adjust_positions(self, atoms, new):
        new -= new.mean(axis=0) - atoms.positions.mean(axis=0)

    def adjust_forces(self, atoms, forces):
        forces -= forces.mean(axis=0)

    def adjust_momenta(self, atoms, momenta):
        self.adjust_forces(atoms, momenta)

    def get_removed_dof(self, atoms):
        return 3

    def index_shuffle(self, atoms, ind):
        pass

    def copy(self):
        return _FixedCentroid()

    def todict(self):
        return {"name": type(self).__name__, "kwargs": {}}


def _voigt_strain(index: int, eta: float) -> np.ndarray:
    a, b = VOIGT_PAIRS[index]
    strain = np.zeros((3, 3))
    if a == b:
        strain[a, a] = eta
    else:
        strain[a, b] = strain[b, a] = 0.5 * eta
    return strain


def piezoelectric_tensor(atoms, *, strain: float = DEFAULT_PIEZO_STRAIN,
                         relax_ions: bool = False, fmax: float = 0.01,
                         kpts_polarization=None, components=None,
                         **options) -> PiezoelectricTensor:
    r"""The **proper** piezoelectric tensor of an insulating crystal.

    Each of the six Voigt strains is applied at :math:`\pm\eta`
    (``strain``), the atoms carried along in fractional coordinates
    (clamped ion) or, with ``relax_ions``, then relaxed at the strained cell
    to ``fmax`` (eV/Angstrom) with the crystal forces (relaxed ion), the
    centroid held fixed and the grid's net force removed.  The
    proper tensor differentiates the raw polarization coordinates
    :math:`p_\alpha` -- the Berry phases and the ionic part in units of the
    quanta -- against the **unstrained** lattice (Vanderbilt 2000),

    .. math::

        e_{i\mu} = \frac{g\,e}{\Omega}\sum_\alpha a_{\alpha i}
                   \frac{\partial p_\alpha}{\partial\eta_\mu},

    which removes the improper part a strained quantum would add.  The
    differences are wrapped to the nearest quantum.  ``options`` are those
    of ``Mandacaru(method="dft", ...)``: 12 self-consistent runs, more with
    the relaxations.  ``components`` (Voigt indices 0-5) limits the strains
    computed -- the columns of the others are left zero, e.g. ``(0, 2)``
    for a wurtzite's :math:`e_{31}` and :math:`e_{33}`.
    """
    from .calculator import Mandacaru

    components = tuple(range(6)) if components is None else tuple(
        int(c) for c in components)
    if not components or any(c not in range(6) for c in components):
        raise ValueError(f"components must be Voigt indices 0-5; got "
                         f"{components!r}")
    if not bool(np.all(atoms.get_pbc())):
        raise NotImplementedError("a piezoelectric tensor is for a crystal "
                                  "periodic in three directions")
    eta = float(strain)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError(f"strain must be a positive number; got {eta!r}")

    def raw(geometry):
        geometry.calc = Mandacaru(**options)
        if relax_ions:
            from ase.optimize import BFGS
            geometry.set_constraint(_FixedCentroid())
            BFGS(geometry, logfile=None).run(fmax=float(fmax))
        geometry.get_potential_energy()
        geometry.calc.get_polarization(kpts=kpts_polarization)
        return geometry.calc.polarization_result

    reference = atoms.copy()
    reference.calc = Mandacaru(**options)
    reference.get_potential_energy()
    crystal = reference.calc.solver._periodic_solver.crystal
    g = polarization_quantum(
        1.0 if reference.calc.solver.get_number_of_spins() == 2 else 2.0,
        crystal.charges)
    lattice = np.asarray(crystal.lattice, dtype=float)       # columns, Bohr
    volume = abs(float(np.linalg.det(lattice)))
    steps = (g / volume) * lattice.T * E_PER_BOHR2_IN_C_PER_M2  # rows
    tensor = np.zeros((3, 6))
    cell = np.asarray(atoms.get_cell(), dtype=float)
    scaled = atoms.get_scaled_positions()
    for mu in components:
        coordinates = []
        for sign in (1.0, -1.0):
            geometry = atoms.copy()
            geometry.set_cell(cell @ (np.eye(3) + _voigt_strain(mu,
                                                                sign * eta)),
                              scale_atoms=False)
            geometry.set_scaled_positions(scaled)
            coordinates.append(raw(geometry).raw)
        change = coordinates[0] - coordinates[1]
        change -= np.round(change)
        tensor[:, mu] = (change @ steps) / (2.0 * eta)
    return PiezoelectricTensor(tensor=tensor, strain=eta,
                               relaxed=bool(relax_ions))


# --------------------------------------------------------------------------- #
# A finite electric field: the Berry-phase energy functional.
# --------------------------------------------------------------------------- #

#: Smallest magnitude of a link's determinant a field calculation tolerates:
#: below it the occupied manifolds of neighboring k-points have lost their
#: overlap -- the field is past the Zener limit for this mesh (e E N a ~ the
#: gap) or the mesh is too coarse -- and the functional has no minimum to
#: find.
FIELD_LINK_FLOOR = 1e-3


def mesh_neighbors(crystal, size):
    """``(forward, backward)``: per lattice direction ``j``, the index of
    each k-point's neighbor at :math:`\\pm\\mathbf b_j/N_j` on the crystal's
    full mesh (the periodic image at the zone's edge)."""
    size = tuple(int(n) for n in size)
    B = rc.reciprocal_vectors(crystal.lattice)
    fractional = np.asarray(crystal.kpoints, dtype=float) @ np.linalg.inv(B).T
    scaled = (fractional - fractional[0]) * np.asarray(size, dtype=float)
    index = np.rint(scaled).astype(int)
    if np.abs(scaled - index).max() > 1e-6 or len(fractional) != int(
            np.prod(size)):
        raise ValueError(f"the crystal's k-points are not the full {size} "
                         "mesh a finite field needs (no symmetry reduction)")
    index %= np.asarray(size)
    flat = np.ravel_multi_index(index.T, size)
    if len(set(flat.tolist())) != len(flat):
        raise ValueError("the crystal's k-points repeat a mesh point")
    position = np.empty(len(flat), dtype=int)
    position[flat] = np.arange(len(flat))
    forward, backward = [], []
    for j in range(3):
        step = np.zeros(3, dtype=int)
        step[j] = 1
        forward.append(position[np.ravel_multi_index(
            ((index + step) % size).T, size)])
        backward.append(position[np.ravel_multi_index(
            ((index - step) % size).T, size)])
    return index, forward, backward


def field_terms(crystal, vectors, n_occ: int, size, field):
    r"""The Berry-phase field term of a restricted insulating crystal.

    Returns ``(W, energy, phases)``: per k-point the Hermitian ``(M, M)``
    operator :math:`W_k` the Kohn-Sham matrix gains, the field's energy
    :math:`-\Omega\,\boldsymbol{\mathcal E}\cdot(\mathbf P_{el} + \mathbf
    P_{ion})` (Hartree per cell) and the electrons' Berry phase along each
    lattice direction (averaged over the strings, as
    :func:`berry_polarization` does).

    With :math:`E_{field} = \frac{f}{2\pi}\sum_j(\boldsymbol{\mathcal E}
    \cdot\mathbf a_j)\bar\phi_j` and :math:`\bar\phi_j` the string average of
    :math:`-\sum_s\operatorname{Im}\ln\det M(\mathbf k_s, \mathbf k_{s+1})`,
    the derivative by the occupied coefficients :math:`V_{\mathbf k}^*`,
    divided by the occupation :math:`w_{\mathbf k} f`, is

    .. math::

        g_{\mathbf k} = \frac{i}{4\pi}\sum_j N_j(\boldsymbol{\mathcal E}
        \cdot\mathbf a_j)\,\big[S(\mathbf k, \mathbf k+\mathbf b)V_{\mathbf
        k+\mathbf b}M_{\mathbf k}^{-1} - S(\mathbf k-\mathbf b, \mathbf
        k)^\dagger V_{\mathbf k-\mathbf b}M_{\mathbf k-\mathbf
        b}^{-\dagger}\big]

    and :math:`W_k = g(SV)^\dagger + (SV)g^\dagger`, which acts on the
    occupied states as :math:`g` up to a rotation among them (Souza,
    Iniguez and Vanderbilt 2002; Umari and Pasquarello 2002).
    """
    size = tuple(int(n) for n in size)
    field = np.asarray(field, dtype=float)
    lattice = np.asarray(crystal.lattice, dtype=float)
    B = rc.reciprocal_vectors(lattice)
    volume = abs(float(np.linalg.det(lattice)))
    index, forward, backward = mesh_neighbors(crystal, size)
    data = crystal.kpoint_data
    V = [np.asarray(v)[:, :n_occ] for v in vectors]
    SV = [d.overlap @ v for d, v in zip(data, V)]
    nk, M = len(data), V[0].shape[0]
    W = np.zeros((nk, M, M), dtype=complex)
    grid = crystal.grid
    r = np.stack([np.ravel(grid.X), np.ravel(grid.Y), np.ravel(grid.Z)])
    phases = np.zeros(3)
    f = 2.0
    for j in range(3):
        strength = float(field @ lattice[:, j])
        b = B[:, j] / size[j]
        grid_phase = np.exp(-1j * (b @ r))
        onsite = onsite_operator(crystal, b)
        S, inverse, link_phase = [], [], np.zeros(nk)
        for k in range(nk):
            S_k = link_overlap(crystal, data[k], data[forward[j][k]], b,
                               onsite, grid_phase)
            M_k = V[k].conj().T @ S_k @ V[forward[j][k]]
            sign, logdet = np.linalg.slogdet(M_k)
            if np.exp(logdet) < FIELD_LINK_FLOOR:
                raise RuntimeError(
                    f"the occupied states of neighboring k-points no longer "
                    f"overlap along direction {j + 1} (|det M| = "
                    f"{np.exp(logdet):.1e}): the field is past the Zener "
                    f"limit for this mesh, or the mesh is too coarse -- "
                    f"lower the field or refine kpts")
            S.append(S_k)
            inverse.append(np.linalg.inv(M_k))
            link_phase[k] = -float(np.angle(sign))
        # Each string: the links of the k-points sharing the other two indices.
        others = [d for d in range(3) if d != j]
        strings: dict = {}
        for k in range(nk):
            strings.setdefault(tuple(index[k, others]), []).append(k)
        phases[j] = _average_phases([float(np.sum(link_phase[members]))
                                     for members in strings.values()])
        if strength == 0.0:
            continue
        coefficient = 1j * strength * size[j] / (4.0 * np.pi)
        for k in range(nk):
            before = backward[j][k]
            g = coefficient * (
                S[k] @ V[forward[j][k]] @ inverse[k]
                - S[before].conj().T @ V[before] @ inverse[before].conj().T)
            W[k] += g @ SV[k].conj().T + SV[k] @ g.conj().T
    electronic = -(f / (2.0 * np.pi * volume)) * (lattice @ phases)
    ionic = (np.asarray(crystal.charges, dtype=float)
             @ np.asarray(crystal.centers, dtype=float)) / volume
    energy = -volume * float(field @ (electronic + ionic))
    return W, energy, phases


#: Field strength (Hartree per e Bohr) of the dielectric tensor's central
#: differences: small enough to stay below the Zener limit e E N a < E_gap
#: of a typical mesh, large enough for the polarization change to stand
#: well above the SCF convergence.
DEFAULT_DIELECTRIC_FIELD = 2e-4


@dataclass
class DielectricTensor:
    r"""The clamped-ion (electronic) dielectric tensor
    :math:`\varepsilon^\infty_{ij} = \delta_{ij} + 4\pi\chi_{ij}`, with
    :math:`\chi_{ij} = \partial P_i/\partial\mathcal E_j` (atomic units) from
    the polarization in finite fields; ``energy_diagonal`` is the same
    :math:`\chi_{jj}` from the energy's curvature,
    :math:`-\partial^2 E/\partial\mathcal E_j^2/\Omega`, the independent
    check of a variational functional."""

    tensor: np.ndarray
    susceptibility: np.ndarray
    energy_diagonal: np.ndarray
    field: float


def dielectric_tensor(atoms, *, field: float = DEFAULT_DIELECTRIC_FIELD,
                      **options) -> DielectricTensor:
    r""":math:`\varepsilon^\infty` of an insulating crystal by finite
    electric fields coupled through the Berry phase.

    Seven self-consistent runs on the full mesh (``Mandacaru(method="dft",
    electric_field=...)``, :func:`field_terms`): no field and
    :math:`\pm\mathcal E` along each Cartesian axis, the ions clamped.  The
    polarization changes are wrapped to the nearest quantum.  The field must
    stay below the Zener limit :math:`e\mathcal E N_j a_j < E_{gap}`;
    beyond it the occupied manifolds of neighboring k-points stop
    overlapping and the run is refused.
    """
    from ..units import ANGSTROM_TO_BOHR, HARTREE_TO_EV
    from .calculator import Mandacaru

    if not bool(np.all(atoms.get_pbc())):
        raise NotImplementedError("the finite-field dielectric tensor is for "
                                  "a crystal; a molecule's response is its "
                                  "polarizability (algorithms.polarizability)")
    if "electric_field" in options:
        raise ValueError("dielectric_tensor applies the fields itself; do "
                         "not pass electric_field")
    field = float(field)
    if not np.isfinite(field) or field <= 0.0:
        raise ValueError(f"field must be a positive number; got {field!r}")
    to_hartree = 1.0 if options.get("atomic_units") else 1.0 / HARTREE_TO_EV
    volume = atoms.get_volume() * ANGSTROM_TO_BOHR ** 3

    def run(vector):
        geometry = atoms.copy()
        geometry.calc = Mandacaru(electric_field=tuple(vector), **options)
        energy = geometry.get_potential_energy() * to_hartree
        geometry.calc.get_polarization()
        return energy, geometry.calc.polarization_result

    e0, p0 = run(np.zeros(3))
    chi = np.zeros((3, 3))
    curvature = np.zeros(3)

    def unbranched(energy, vector, result):
        # E carries -Omega E.P on the Berry phase's branch; a run whose phase
        # landed a whole quantum away (a polarization on a half quantum) is
        # moved back, so the curvature sees one smooth E(field).
        jump = np.round(result.raw - p0.raw)
        shift = (jump @ result.quanta) / E_PER_BOHR2_IN_C_PER_M2
        return energy + volume * float(vector @ shift)

    for axis in range(3):
        step = np.zeros(3)
        step[axis] = field
        e_plus, p_plus = run(step)
        e_minus, p_minus = run(-step)
        e_plus = unbranched(e_plus, step, p_plus)
        e_minus = unbranched(e_minus, -step, p_minus)
        change = p_plus.raw - p_minus.raw
        change -= np.round(change)
        chi[:, axis] = (change @ p_plus.quanta) / (
            E_PER_BOHR2_IN_C_PER_M2 * 2.0 * field)
        curvature[axis] = -(e_plus + e_minus - 2.0 * e0) / (
            field ** 2 * volume)
    return DielectricTensor(tensor=np.eye(3) + 4.0 * np.pi * chi,
                            susceptibility=chi, energy_diagonal=curvature,
                            field=field)
