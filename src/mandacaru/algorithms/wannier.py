# -*- coding: utf-8 -*-
# file: algorithms/wannier.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Maximally localized Wannier functions of a crystal.

A Wannier function is a unitary mixture of Bloch states, localized in one
cell,

.. math::

    w_{n\mathbf R}(\mathbf r) = \frac{1}{N}\sum_{\mathbf k}
        e^{-i\mathbf k\cdot\mathbf R}\sum_m U_{mn}(\mathbf k)\,
        \psi_{m\mathbf k}(\mathbf r),

and the gauge :math:`U(\mathbf k)` is chosen to minimize the total spread
:math:`\Omega = \sum_n(\langle r^2\rangle_n - \langle\mathbf r\rangle_n^2)`
(Marzari and Vanderbilt 1997).  On a k-mesh every expectation value comes
from the overlaps of neighboring k-points,

.. math::

    M_{mn}(\mathbf k, \mathbf b) = \langle u_{m\mathbf k}|u_{n,\mathbf
        k+\mathbf b}\rangle,
    \qquad
    \bar{\mathbf r}_n = -\frac1N\sum_{\mathbf k,\mathbf b} w_b\,\mathbf b\,
        \operatorname{Im}\ln M_{nn},
    \qquad
    \langle r^2\rangle_n = \frac1N\sum_{\mathbf k,\mathbf b} w_b
        \big[1 - |M_{nn}|^2 + (\operatorname{Im}\ln M_{nn})^2\big],

with shells of neighbors :math:`\mathbf b` and weights :math:`w_b` such that
:math:`\sum_b w_b b_i b_j = \delta_{ij}`.  The overlaps are those of the
Berry-phase polarization (:func:`~mandacaru.algorithms.berry_phase.
link_overlap`: the smooth part on the cell grid and the PAW on-site charge
and dipole).  :math:`\Omega = \Omega_I + \tilde\Omega`: the invariant part
:math:`\Omega_I` depends only on the subspace, the rest on the gauge, which
steepest descent on :math:`U(\mathbf k) \to U(\mathbf k)e^{\Delta W}`
minimizes.

**Entangled bands** (Souza, Marzari and Vanderbilt 2001).  When the
:math:`J` functions wanted are fewer than the bands in an outer energy
window, the :math:`J`-dimensional subspace at each k-point is chosen first,
to minimize :math:`\Omega_I` (:func:`disentangle`): the states of an
optional inner, *frozen* window are kept as they are, and the rest of the
subspace is the leading eigenvectors of
:math:`Z(\mathbf k) = \sum_b w_b M(\mathbf k,\mathbf b) P(\mathbf k+\mathbf
b) M(\mathbf k,\mathbf b)^\dagger` over the window's other states, mixed
between iterations.  The gauge within the subspace is then minimized as for
an isolated group.

**The Hamiltonian in the Wannier basis** is
:math:`H_{mn}(\mathbf R) = \langle w_{m\mathbf 0}|H|w_{n\mathbf R}\rangle`
on the Born-von Karman supercell of the mesh.  Interpolation assigns each
pair's hopping either to the replica :math:`\mathbf R + \mathbf T`
(:math:`\mathbf T` a supercell vector) whose centers are nearest, shared
equally among equidistant replicas (:func:`nearest_replicas`, the default),
or to the supercell's Wigner-Seitz vectors between cell origins; either way
the bands stay exact on the mesh.

**The functions on the grid** (:meth:`WannierResult.orbital`,
:meth:`WannierResult.write`) are one-particle orbitals, the smooth part of
:math:`w_{n\mathbf 0}` on the supercell, written as ``.cube`` or ``.xsf``.

**A downfolded many-body problem** (:meth:`WannierResult.downfold`): the
Kohn-Sham Hamiltonian and the Coulomb tensor in a fragment's functions,
minus the double counting (:class:`DownfoldedProblem`), handed to a quantum
method through :meth:`DownfoldedProblem.as_quantum_problem`.  The
interaction is bare, or screened by the static constrained RPA
(:func:`screened_interaction`); it is truncated within the Born-von Karman
supercell (:class:`SupercellCoulomb`), or evaluated on the fragment's
nearest images in a supercell twice as large (:func:`padded_fragment`).

**A spin-polarized crystal** gets one set of functions per spin channel,
each from its own Bloch states (:class:`SpinWannierResult`), and a
spin-resolved downfolded problem.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..integrals import reciprocal as rc
from ..units import BOHR_TO_ANGSTROM, EV_TO_HARTREE, HARTREE_TO_EV

#: Width (Bohr) of the Gaussian trial orbitals of the initial projection.
TRIAL_WIDTH = 1.0

#: Steepest-descent step, in units of the inverse total shell weight
#: (Marzari and Vanderbilt's ``alpha / 4 w``, stable for ``alpha < 1``).
DESCENT_STEP = 0.5

#: Convergence of the minimization: the change of the total spread (Bohr^2)
#: per iteration.
SPREAD_TOLERANCE = 1e-10

#: Fraction of the new :math:`Z(\mathbf k)` mixed into the previous one per
#: disentanglement iteration (Souza, Marzari and Vanderbilt's beta).
DISENTANGLE_MIXING = 0.5

#: Convergence of the disentanglement: the change of Omega_I (Bohr^2) per
#: iteration.
DISENTANGLE_TOLERANCE = 1e-10

#: Relative tolerance below which two replicas of a pair count as equally
#: near (:func:`nearest_replicas`).
REPLICA_TOLERANCE = 1e-5

#: Radius (Bohr^-1) of the vectors q + G on which the screened interaction
#: is solved (:func:`screened_interaction`); beyond it the interaction is
#: bare.
SCREENING_CUTOFF = 3.0

#: The small wave vector (Bohr^-1) at which the screening's
#: long-wavelength dielectric head is taken (:func:`screened_interaction`).
HEAD_WAVEVECTOR = 0.02

#: Reciprocal vectors per block of compensation transforms
#: (:func:`pair_densities`), which bounds their memory.
TRANSFORM_BLOCK = 65536

#: Natural occupations closer than this count as one degenerate set, in
#: which the Kohn-Sham Hamiltonian fixes the orbitals
#: (:meth:`DownfoldedProblem.natural_orbitals`).
NATURAL_DEGENERACY = 1e-6

#: How many root-mean-square radii past its outermost centers a fragment's
#: charge is taken to reach (:func:`fragment_extent`).  Calibrated on
#: silicon's sp3 fragments against the isolated interaction: below the
#: truncation radius with 1.5 the image error stayed under 10 meV (7 meV at
#: most), above it reached 31 meV.
FRAGMENT_REACH = 1.5


#: The real spherical harmonics a trial orbital names, ``(l, m)``
#: (:func:`real_harmonic`): ``m > 0`` the :math:`\cos m\phi` combination,
#: ``m < 0`` the :math:`\sin|m|\phi` one, in the trial's frame.
ORBITALS = {"s": (0, 0), "py": (1, -1), "pz": (1, 0), "px": (1, 1),
            "dxy": (2, -2), "dyz": (2, -1), "dz2": (2, 0), "dxz": (2, 1),
            "dx2-y2": (2, 2)}

#: Whole shells, every ``m`` from ``-l`` to ``l`` (the basis' order: p is
#: py, pz, px; d is dxy, dyz, dz2, dxz, dx2-y2).
SHELLS = {"s": 0, "p": 1, "d": 2, "f": 3}

#: The d orbitals of an octahedral (cubic) site, by symmetry: the
#: :math:`t_{2g}` lobes point between the frame's axes, the :math:`e_g`
#: lobes along them.
SUBSETS = {"t2g": ("dxy", "dyz", "dxz"), "eg": ("dz2", "dx2-y2")}

#: Hybrids, rows of coefficients on (s, px, py, pz) in the frame: ``sp``
#: along +z and -z; ``sp2`` in the xy plane, toward +x and 120 degrees
#: either side; ``sp3`` toward (1, 1, 1), (1, -1, -1), (-1, 1, -1) and
#: (-1, -1, 1).
HYBRIDS = {
    "sp": np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]])
    / np.sqrt(2.0),
    "sp2": np.array([[1.0 / np.sqrt(3.0), -1.0 / np.sqrt(6.0),
                      1.0 / np.sqrt(2.0), 0.0],
                     [1.0 / np.sqrt(3.0), -1.0 / np.sqrt(6.0),
                      -1.0 / np.sqrt(2.0), 0.0],
                     [1.0 / np.sqrt(3.0), 2.0 / np.sqrt(6.0), 0.0, 0.0]]),
    "sp3": 0.5 * np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0],
                           [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]]),
}

#: The (l, m) of the s and Cartesian p components of a hybrid.
_SP = ((0, 0), (1, 1), (1, -1), (1, 0))


@dataclass(frozen=True)
class Trial:
    r"""A trial orbital of the starting gauge, centered at ``center``
    (Bohr): :math:`g(\mathbf r) = \sum_{(l, m, c)} c\,R_l(r)\,S_{lm}(\hat
    r')` over ``angular``, the real spherical harmonics
    :math:`S_{lm}` (:func:`real_harmonic`) of the offset :math:`\mathbf r'`
    in the trial's ``frame`` -- rows the local x, y and z axes as Cartesian
    unit vectors, ``None`` the Cartesian axes.  The radial functions
    :math:`R_l` are ``radial``, pairs ``(l, function of r in Bohr)``, or
    with ``radial=None`` normalized Gaussians :math:`r^l e^{-r^2/2w^2}`
    of the projection's width (:func:`gaussian_radial`).  An
    :math:`sp^3` hybrid toward the unit vector :math:`\hat d` is
    :math:`\tfrac12 S_{00} + \tfrac{\sqrt3}{2}\hat d\cdot(p_x, p_y,
    p_z)`.  ``atom`` is the index of the atom it sits on, if any, and
    ``label`` names it."""

    center: tuple
    angular: tuple = ((0, 0, 1.0),)
    frame: tuple | None = None
    radial: tuple | None = None
    atom: int | None = None
    label: str = ""


@dataclass
class DownfoldedProblem:
    r"""The many-body problem of a fragment of Wannier functions (Hartree).

    ``kohn_sham`` is the Kohn-Sham Hamiltonian :math:`h^{KS}` between the
    fragment's functions, on the crystal's eigenvalue zero; ``two_body`` the
    Coulomb integrals :math:`\langle pq|rs\rangle` (physicists' notation),
    bare or screened (``screening``); ``density_matrix`` the Kohn-Sham
    one-particle density matrix :math:`\gamma_{pq} = \langle c^\dagger_p
    c_q\rangle`; and ``double_counting`` the Hartree and exchange potential
    of the fragment's own electrons at the Kohn-Sham density, which
    :math:`h^{KS}` already holds,

    .. math::

        V^{dc,\sigma}_{pq} = \sum_{\sigma'}\sum_{rs}\gamma^{\sigma'}_{rs}
            \langle p^\sigma r^{\sigma'}|q^\sigma s^{\sigma'}\rangle
            - \sum_{rs}\gamma^\sigma_{rs}
            \langle p^\sigma r^\sigma|s^\sigma q^\sigma\rangle .

    The model's one-body part is ``one_body`` = :math:`h^{KS} - V^{dc}`, so
    its Hartree-Fock Fock matrix at :math:`\gamma` is :math:`h^{KS}` again.
    For a restricted crystal the arrays are spatial: ``kohn_sham`` and
    ``double_counting`` ``(n, n)``, ``density_matrix`` summed over spin and
    ``two_body`` ``(n, n, n, n)``.  For a spin-polarized one each spin
    channel has its own functions: ``kohn_sham``, ``double_counting`` and
    ``density_matrix`` are ``(2, n, n)`` (up, down) and ``two_body[s, t]``
    is :math:`\langle p^s q^t|r^s s^t\rangle`, ``(2, 2, n, n, n, n)``.
    ``functions`` names each mode ``(n, R)``: function ``n`` of the cell at
    the integer lattice vector ``R``; ``num_particles`` the electrons
    ``(up, down)`` the model holds; ``screening`` and ``coulomb`` the
    interaction's options (:meth:`WannierResult.downfold`).
    """

    functions: tuple
    kohn_sham: np.ndarray
    two_body: np.ndarray
    density_matrix: np.ndarray
    double_counting: np.ndarray
    num_particles: tuple
    screening: str | None = None
    coulomb: str = "supercell"
    #: The inverse dielectric head the screening took for q -> 0
    #: (:class:`ScreenedInteraction`), or ``None``.
    dielectric_head: float | None = None

    @property
    def spin_polarized(self) -> bool:
        return np.ndim(self.kohn_sham) == 3

    @property
    def n_electrons(self) -> int:
        return int(sum(self.num_particles))

    @property
    def one_body(self) -> np.ndarray:
        return self.kohn_sham - self.double_counting

    @property
    def n_spatial_orbitals(self) -> int:
        return len(self.functions)

    def natural_orbitals(self):
        """``(occupations, C)``: the eigenvectors of :attr:`density_matrix`
        (columns, in the fragment's functions), most occupied first; within
        a set of equal occupations (a full or an empty channel, say) the
        eigenvectors of :attr:`kohn_sham`, lowest first.  For a
        spin-polarized problem a list of one pair per spin channel."""
        def natural(gamma, h):
            gamma = 0.5 * (gamma + gamma.conj().T)
            values, vectors = np.linalg.eigh(gamma)
            order = np.argsort(-values, kind="stable")
            values, vectors = values[order], vectors[:, order]
            vectors = _real_columns(vectors)
            start = 0
            while start < len(values):
                stop = start + 1
                while (stop < len(values) and abs(values[stop]
                                                  - values[start])
                       < NATURAL_DEGENERACY):
                    stop += 1
                if stop - start > 1:
                    block = vectors[:, start:stop]
                    _e, rotation = np.linalg.eigh(block.conj().T @ h @ block)
                    vectors[:, start:stop] = _real_columns(block @ rotation)
                start = stop
            return values, vectors

        if self.spin_polarized:
            return [natural(g, h) for g, h in zip(self.density_matrix,
                                                  self.kohn_sham)]
        return natural(self.density_matrix, self.kohn_sham)

    def integrals(self, basis: str = "wannier") -> tuple:
        """``(h, <pq|rs>)`` of the model (Hartree), shaped as the fields:
        in the fragment's Wannier functions, or with ``basis="natural"`` in
        :meth:`natural_orbitals` -- the same model, rotated."""
        if basis == "wannier":
            return self.one_body, self.two_body
        if basis != "natural":
            raise ValueError(f"basis must be 'wannier' or 'natural'; got "
                             f"{basis!r}")
        from .hartree_fock import transform_integrals

        if not self.spin_polarized:
            _occupations, C = self.natural_orbitals()
            return transform_integrals(self.one_body, self.two_body, C)
        C = [c for _occupations, c in self.natural_orbitals()]
        h = np.stack([c.conj().T @ hs @ c for c, hs in zip(C, self.one_body)])
        g = np.empty_like(self.two_body)
        for s, t in itertools.product(range(2), repeat=2):
            g[s, t] = np.einsum("ap,bq,cr,ds,abcd->pqrs", C[s].conj(),
                                C[t].conj(), C[s], C[t], self.two_body[s, t],
                                optimize=True)
        return h, g

    def spin_orbital_integrals(self, basis: str = "wannier") -> tuple:
        """``(h, g)`` over the ``2n`` spin-orbitals, up first: the one-body
        matrix (block diagonal in spin) and :math:`\\langle PQ|RS\\rangle`,
        nonzero when the spins of ``P`` and ``R``, and of ``Q`` and ``S``,
        agree."""
        from ..core.hamiltonian import spin_block_integrals

        h, g = self.integrals(basis)
        if not self.spin_polarized:
            return spin_block_integrals(h, g)
        n = self.n_spatial_orbitals
        h_so = np.zeros((2 * n, 2 * n), dtype=complex)
        g_so = np.zeros((2 * n,) * 4, dtype=complex)
        for s in range(2):
            h_so[s * n:(s + 1) * n, s * n:(s + 1) * n] = h[s]
            for t in range(2):
                a, b = slice(s * n, (s + 1) * n), slice(t * n, (t + 1) * n)
                g_so[a, b, a, b] = g[s, t]
        return h_so, g_so

    def fermion_hamiltonian(self, basis: str = "wannier"):
        """The spin-orbital :class:`~mandacaru.core.mapping.Fermion`
        Hamiltonian (Hartree; no constant) in ``basis``
        (:meth:`integrals`)."""
        from ..core.mapping import Fermion

        return Fermion.from_integrals(*self.spin_orbital_integrals(basis))

    def as_quantum_problem(self) -> dict[str, object]:
        """Options for ``Mandacaru(method='adapt-vqe', **options)``.

        The model is written in the natural orbitals of the Kohn-Sham
        density matrix (per spin channel for a spin-polarized crystal), most
        occupied first, so the reference determinant (the first orbitals of
        each spin occupied) is the one nearest the Kohn-Sham state; a
        determinant of localized functions is a poor start.
        """
        return {"hamiltonian": self.fermion_hamiltonian("natural"),
                "num_particles": tuple(int(x) for x in self.num_particles),
                "n_spatial_orbitals": self.n_spatial_orbitals,
                "initial_state": "hartree-fock"}


@dataclass
class WannierResult:
    """Maximally localized Wannier functions of a group of bands.

    ``centers`` (Angstrom, Cartesian) and ``spreads`` (Angstrom^2) per
    function; ``omega_invariant``, ``omega_offdiagonal`` and
    ``omega_diagonal`` the three parts of the total spread (Angstrom^2);
    ``unitary[k]`` the gauge per k-point of ``kpoints`` (fractional), a
    ``(bands, functions)`` matrix -- rectangular after disentanglement;
    ``hamiltonian_R`` the Kohn-Sham matrix in the Wannier basis per lattice
    vector ``lattice_vectors`` (integer), in eV, each pair's hopping on the
    replica its ``replicas`` rule chose (:func:`wannier_functions`), the
    degeneracy already divided out.  ``windows`` are the outer
    and frozen energy windows (eV) of a disentanglement, ``None`` for an
    isolated group; ``disentanglement`` the Omega_I (Angstrom^2) per
    iteration of the subspace selection.
    """

    centers: np.ndarray
    spreads: np.ndarray
    omega_invariant: float
    omega_offdiagonal: float
    omega_diagonal: float
    kpoints: np.ndarray
    unitary: np.ndarray
    lattice_vectors: np.ndarray
    hamiltonian_R: np.ndarray
    lattice: np.ndarray                      # rows a_j, Angstrom
    bands: tuple
    iterations: int
    history: list = field(default_factory=list)
    windows: dict | None = None
    disentanglement: list = field(default_factory=list)
    size: tuple = ()
    #: The trial orbital each function started from, by name
    #: (:attr:`Trial.label`): ``"Cu0:dxy"``, say.
    labels: tuple = ()
    # The Born-von Karman blocks H(t) and the density operator rho(t) for
    # t in the supercell (Hartree), the crystal and the Bloch coefficients
    # of the functions, from which the grid and the two-body tensor follow.
    _hamiltonian_bvk: np.ndarray | None = field(default=None, repr=False)
    _density_bvk: np.ndarray | None = field(default=None, repr=False)
    _crystal: object = field(default=None, repr=False)
    #: The spin channel (0 up, 1 down) of a spin-polarized crystal, else
    #: ``None``.
    spin: int | None = None
    _coefficients: np.ndarray | None = field(default=None, repr=False)
    _grid: tuple | None = field(default=None, repr=False)
    _solver: object = field(default=None, repr=False)
    _fermi_level: float | None = field(default=None, repr=False)
    _box: object = field(default=None, repr=False)
    # Records the references a later step (screening) uses, when the
    # functions came through a calculator.
    _cite: object = field(default=None, repr=False)

    @property
    def n_functions(self) -> int:
        return len(self.spreads)

    @property
    def total_spread(self) -> float:
        return float(np.sum(self.spreads))

    def hamiltonian(self, R) -> np.ndarray:
        """``(n, n)`` block :math:`\\langle w_{m\\mathbf 0}|H|w_{n\\mathbf
        R}\\rangle` (eV) of the interpolation for the integer lattice vector
        ``R``, as placed for interpolation (``hamiltonian_R``): zero for a
        pair whose hopping went to another replica."""
        R = np.asarray(R, dtype=int)
        for vector, block in zip(self.lattice_vectors, self.hamiltonian_R):
            if np.array_equal(vector, R):
                return block
        raise KeyError(f"no Wannier hopping for R = {tuple(R)}")

    def interpolate(self, kpoints) -> np.ndarray:
        """Band energies (eV) at fractional ``kpoints`` (rows) from the
        Wannier Hamiltonian -- the Wannier interpolation."""
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        phases = np.exp(2j * np.pi * kpoints @ self.lattice_vectors.T)
        H = np.einsum("kr,rmn->kmn", phases, self.hamiltonian_R)
        return np.linalg.eigvalsh(0.5 * (H + np.conj(np.swapaxes(H, 1, 2))))

    def summary(self) -> str:
        lines = ["Maximally localized Wannier functions",
                 f"  bands {self.bands}, {self.n_functions} functions, "
                 f"{self.iterations} iterations"]
        if self.windows is not None:
            from .band_selection import selection_lines

            lines += selection_lines(self.windows)
            outer, frozen = self.windows["outer"], self.windows["frozen"]
            span = "span" if "selection" in self.windows else "window"
            if outer is not None:
                lines.append(f"  outer {span}  {outer[0]:.3f} .. "
                             f"{outer[1]:.3f} eV")
            if frozen is not None:
                lines.append(f"  frozen window {frozen[0]:.3f} .. "
                             f"{frozen[1]:.3f} eV")
            lines.append(f"  disentanglement {len(self.disentanglement)} "
                         "iterations")
        lines += [f"  Omega_I  {self.omega_invariant:.6f} A^2",
                  f"  Omega_OD {self.omega_offdiagonal:.6f} A^2",
                  f"  Omega_D  {self.omega_diagonal:.6f} A^2",
                  "  n   center (A)                         spread (A^2)"
                  "   trial"]
        labels = self.labels or ("",) * self.n_functions
        for n, (c, s, label) in enumerate(zip(self.centers, self.spreads,
                                              labels)):
            lines.append(f"  {n:<3d} {c[0]:10.5f} {c[1]:10.5f} {c[2]:10.5f}"
                         f"   {s:10.5f}   {label}")
        return "\n".join(lines)

    # -- the functions on the grid ------------------------------------------ #

    def _supercell(self):
        """``(values, projections)``: the smooth parts of the home cell's
        functions on the supercell grid, ``(n, S1 n1, S2 n2, S3 n3)``, and
        :math:`\\langle\\tilde p_{p\\mathbf T}|w_{n\\mathbf 0}\\rangle` per
        supercell cell ``T`` (C order), ``(n, N, P)``; built once."""
        if self._grid is None:
            if self._crystal is None:
                raise RuntimeError("this result carries no crystal to build "
                                   "its functions from")
            self._grid = supercell_functions(self._crystal, self.kpoints,
                                             self._coefficients, self.size)
        return self._grid

    def orbital(self, index: int):
        """``(values, origin, step)``: function ``index`` on the supercell
        grid (Bohr^-3/2; the smooth part), its global phase chosen to make
        it as real as possible, the box centered on the function.
        ``origin`` (Bohr) is node ``(0, 0, 0)`` and ``step`` (Bohr) holds the
        step vectors as columns."""
        values, _projections = self._supercell()
        index = int(index)
        if not 0 <= index < self.n_functions:
            raise IndexError(f"function {index} of {self.n_functions}")
        w = values[index]
        # The phase that makes sum w^2 real and positive maximizes |Re w|.
        w = w * np.exp(-0.5j * np.angle(np.sum(w * w)))
        crystal = self._crystal
        step = np.asarray(crystal.grid.step, dtype=float)
        origin = rc.grid_origin(crystal.grid)
        shape = np.array(w.shape)
        # Roll the periodic box so the function sits in its middle.
        center = np.asarray(self.centers[index]) / BOHR_TO_ANGSTROM
        nodes = np.linalg.solve(step, center - origin)
        shift = np.floor(nodes - 0.5 * shape).astype(int)
        w = np.roll(w, tuple(-shift), axis=(0, 1, 2))
        return w, origin + step @ shift, step

    def imaginary_ratio(self, index: int) -> float:
        """``||Im w|| / ||Re w||`` of :meth:`orbital` ``index``: zero for a
        function real up to a phase (a centrosymmetric crystal's)."""
        w, _origin, _step = self.orbital(index)
        return float(np.linalg.norm(w.imag) / np.linalg.norm(w.real))

    def write(self, index: int, path, part: str = "real", format=None) -> str:
        """Write function ``index`` (:meth:`orbital`) as a ``.cube`` or
        ``.xsf`` file and return the path.  ``part`` is ``"real"``,
        ``"imaginary"`` or ``"modulus"``; the atoms in the box are those of
        the supercell."""
        from ase.data import atomic_numbers

        from ..utils.cube import write_volumetric

        fields = {"real": np.real, "imaginary": np.imag, "modulus": np.abs}
        if part not in fields:
            raise ValueError(f"part must be one of {sorted(fields)}; got "
                             f"{part!r}")
        w, origin, step = self.orbital(index)
        crystal = self._crystal
        lattice = np.asarray(crystal.lattice, dtype=float)       # columns
        box = step @ np.diag(w.shape)
        numbers, charges, positions = [], [], []
        for atom, center in enumerate(crystal.centers):
            dataset = crystal.datasets[atom]
            for t in itertools.product(*[range(-n - 1, n + 1)
                                         for n in self.size]):
                r = center + lattice @ np.asarray(t, dtype=float)
                f = np.linalg.solve(box, r - origin)
                if np.all(f >= -1e-9) and np.all(f < 1.0 - 1e-9):
                    numbers.append(atomic_numbers[dataset.symbol])
                    charges.append(float(dataset.valence_charge))
                    positions.append(r)
        return write_volumetric(
            path, fields[part](w), origin, step, numbers,
            np.reshape(positions, (-1, 3)), charges=charges, cell=box,
            comment=f"Mandacaru Wannier function {index} ({part} part)",
            format=format)

    # -- the downfolded many-body problem ----------------------------------- #

    def _coulomb_box(self) -> "SupercellCoulomb":
        if self._box is None:
            self._box = SupercellCoulomb(self._crystal, self.size)
        return self._box

    def downfold(self, functions=None, num_particles=None, *,
                 screening=None, screening_cutoff=None,
                 screening_bands=None,
                 coulomb: str = "supercell") -> DownfoldedProblem:
        """The many-body problem of a fragment of these functions
        (:class:`DownfoldedProblem`).

        ``functions`` lists the fragment's modes: an index ``n`` (the
        function of the home cell) or ``(n, R)`` with ``R`` an integer
        lattice vector; default every function of the home cell.
        ``num_particles`` (up, down) defaults to the Kohn-Sham count, the
        trace of the density matrix rounded (it must be within 0.1 of an
        integer) and shared equally between the spins.  The interaction is
        the Coulomb interaction on the Born-von Karman supercell, truncated
        beyond half its shortest lattice vector (:func:`coulomb_integrals`),
        or with ``coulomb="isolated"`` on the fragment's nearest images set
        in a supercell twice as large, truncated at the whole shortest
        vector (:func:`padded_fragment`; eight times the grid); a fragment
        whose charge reaches past the radius (:func:`fragment_extent`) meets
        its own periodic copies and is warned about.  It is bare unless
        ``screening`` is ``"crpa"`` (the
        static constrained RPA: every transition but those inside the Bloch
        subspace of the fragment's functions screens) or ``"rpa"`` (every
        transition screens), from the lowest ``screening_bands`` Kohn-Sham
        states on the vectors within ``screening_cutoff`` (Bohr^-1)
        (:func:`screened_interaction`).
        """
        return downfold_channels([self], functions, num_particles,
                                 screening, screening_cutoff,
                                 screening_bands, coulomb)

    def as_quantum_problem(self, functions=None, num_particles=None,
                           **screening) -> dict[str, object]:
        """Options for ``Mandacaru(method='adapt-vqe', **options)``: the
        downfolded problem of the fragment ``functions`` (:meth:`downfold`,
        :meth:`DownfoldedProblem.as_quantum_problem`)."""
        return self.downfold(functions, num_particles,
                             **screening).as_quantum_problem()


@dataclass
class SpinWannierResult:
    """Maximally localized Wannier functions of a spin-polarized crystal:
    one :class:`WannierResult` per spin channel (``channels``, up then
    down), each from its own Bloch states, energy windows and gauge.  The
    per-channel quantities are stacked with a leading spin axis."""

    channels: tuple

    @property
    def up(self) -> WannierResult:
        return self.channels[0]

    @property
    def down(self) -> WannierResult:
        return self.channels[1]

    def _stacked(self, name):
        values = [getattr(c, name) for c in self.channels]
        try:
            return np.stack(values)
        except ValueError:
            return values

    @property
    def labels(self) -> tuple:
        """The trial orbital of each function, shared by the channels."""
        return self.channels[0].labels

    @property
    def centers(self):
        """``(2, n, 3)`` (Angstrom), up and down."""
        return self._stacked("centers")

    @property
    def spreads(self):
        """``(2, n)`` (Angstrom^2)."""
        return self._stacked("spreads")

    @property
    def omega_invariant(self) -> np.ndarray:
        return np.array([c.omega_invariant for c in self.channels])

    @property
    def omega_offdiagonal(self) -> np.ndarray:
        return np.array([c.omega_offdiagonal for c in self.channels])

    @property
    def omega_diagonal(self) -> np.ndarray:
        return np.array([c.omega_diagonal for c in self.channels])

    @property
    def total_spread(self) -> np.ndarray:
        return np.array([c.total_spread for c in self.channels])

    def hamiltonian(self, R):
        """Each channel's :meth:`WannierResult.hamiltonian` (eV)."""
        values = [c.hamiltonian(R) for c in self.channels]
        try:
            return np.stack(values)
        except ValueError:
            return values

    def interpolate(self, kpoints):
        """Each channel's Wannier-interpolated bands (eV), ``(2, nk, n)``."""
        values = [c.interpolate(kpoints) for c in self.channels]
        try:
            return np.stack(values)
        except ValueError:
            return values

    def summary(self) -> str:
        return "\n".join(f"Spin {name}\n{c.summary()}" for name, c in
                         zip(("up", "down"), self.channels))

    def orbital(self, index: int, spin: int):
        """:meth:`WannierResult.orbital` of channel ``spin``."""
        return self.channels[spin].orbital(index)

    def imaginary_ratio(self, index: int, spin: int) -> float:
        return self.channels[spin].imaginary_ratio(index)

    def write(self, index: int, path, spin: int, part: str = "real",
              format=None) -> str:
        """:meth:`WannierResult.write` of function ``index`` of channel
        ``spin`` (0 up, 1 down)."""
        if spin not in (0, 1):
            raise ValueError("a spin-polarized crystal has functions per "
                             "channel: pass spin=0 or 1")
        return self.channels[spin].write(index, path, part=part,
                                         format=format)

    def downfold(self, functions=None, num_particles=None, *,
                 screening=None, screening_cutoff=None,
                 screening_bands=None,
                 coulomb: str = "supercell") -> DownfoldedProblem:
        """The spin-resolved many-body problem of a fragment
        (:meth:`WannierResult.downfold`): each spin channel's functions
        ``functions`` (the same indices in both), its own Kohn-Sham one-body
        part, density matrix and double counting, and the interaction
        between every pair of channels' functions; ``num_particles``
        defaults to each channel's Kohn-Sham count."""
        return downfold_channels(list(self.channels), functions,
                                 num_particles, screening, screening_cutoff,
                                 screening_bands, coulomb)

    def as_quantum_problem(self, functions=None, num_particles=None,
                           **screening) -> dict[str, object]:
        """Options for ``Mandacaru(method='adapt-vqe', **options)``
        (:meth:`downfold`, :meth:`DownfoldedProblem.as_quantum_problem`)."""
        return self.downfold(functions, num_particles,
                             **screening).as_quantum_problem()


# --------------------------------------------------------------------------- #
# The finite-difference shells.
# --------------------------------------------------------------------------- #

def neighbor_shells(reciprocal, size, max_shells: int = 6):
    """``(vectors, weights, steps)``: the shells of nearest mesh vectors
    :math:`\\mathbf b` (Cartesian, Bohr^-1, both signs), their weights
    :math:`w_b` with :math:`\\sum_b w_b b_i b_j = \\delta_{ij}`, and each
    vector's step in mesh indices.

    Shells are added in order of length until the completeness condition
    can be met (one for a cubic mesh, more for a low-symmetry one), the
    weights solved shell by shell in the least-squares sense and checked.
    """
    reciprocal = np.asarray(reciprocal, dtype=float)       # columns b_j
    size = np.asarray(size, dtype=int)
    mesh = reciprocal / size[None, :]
    candidates = []
    for steps in itertools.product(range(-2, 3), repeat=3):
        if steps == (0, 0, 0):
            continue
        vector = mesh @ np.asarray(steps, dtype=float)
        candidates.append((round(float(np.linalg.norm(vector)), 8), steps,
                           vector))
    candidates.sort(key=lambda c: c[0])
    lengths = sorted({c[0] for c in candidates})
    target = np.array([1, 1, 1, 0, 0, 0], dtype=float)   # xx yy zz xy xz yz

    def moments(vectors):
        v = np.asarray(vectors)
        return np.array([np.sum(v[:, 0] ** 2), np.sum(v[:, 1] ** 2),
                         np.sum(v[:, 2] ** 2), np.sum(v[:, 0] * v[:, 1]),
                         np.sum(v[:, 0] * v[:, 2]),
                         np.sum(v[:, 1] * v[:, 2])])

    chosen: list = []
    for length in lengths[:max_shells]:
        shell = [c for c in candidates if c[0] == length]
        trial = chosen + [shell]
        A = np.stack([moments([c[2] for c in s]) for s in trial], axis=1)
        weights, *_ = np.linalg.lstsq(A, target, rcond=None)
        if np.abs(A @ weights - target).max() < 1e-8:
            vectors, w, steps = [], [], []
            for weight, s in zip(weights, trial):
                if abs(weight) < 1e-12:
                    continue
                for _length, step, vector in s:
                    vectors.append(vector)
                    w.append(weight)
                    steps.append(step)
            return np.array(vectors), np.array(w), np.array(steps, dtype=int)
        chosen = trial
    raise ValueError("no set of neighbor shells satisfies the completeness "
                     f"condition within {max_shells} shells")


# --------------------------------------------------------------------------- #
# Overlaps and the initial gauge.
# --------------------------------------------------------------------------- #

def mesh_states(solver, size, bands, spin: int = 0):
    """``(fractional, data, vectors, energies)``: the full Gamma-centered
    ``size`` mesh (C order), its :class:`KPointMatrices`, the ``bands``
    columns of the eigenvectors and their energies (Hartree, on the
    crystal's eigenvalue zero, :meth:`PeriodicKohnSham.eigenvalue_reference`),
    diagonalized at the converged potential of spin channel ``spin``."""
    crystal = solver.crystal
    B = rc.reciprocal_vectors(crystal.lattice)
    index = np.indices(size).reshape(3, -1).T
    fractional = index / np.asarray(size, dtype=float)
    V, v_tau, w, channel = solver.channel_potentials()[spin]
    data, vectors, energies = [], [], []
    for block_data, eps, block_vectors in solver._diagonalized(
            fractional @ B.T, V, v_tau, w, channel):
        data.extend(block_data)
        vectors.extend(v[:, list(bands)] for v in block_vectors)
        energies.extend(e[list(bands)] for e in eps)
    return (fractional, data, vectors,
            np.array(energies) + solver.eigenvalue_reference())


def overlaps(crystal, data, vectors, size, steps, b_vectors) -> np.ndarray:
    r"""``M[k, b]`` (``(nk, nb, n, n)``) between each mesh point and its
    neighbor at :math:`\mathbf k + \mathbf b`: the periodic image's states
    serve across the zone's edge (a Bloch sum at :math:`\mathbf k + \mathbf
    G` is the one at :math:`\mathbf k`), with the true :math:`\mathbf b` in
    the phase."""
    from .berry_phase import link_overlap, onsite_operator

    size = np.asarray(size, dtype=int)
    index = np.indices(tuple(size)).reshape(3, -1).T
    grid = crystal.grid
    r = np.stack([np.ravel(grid.X), np.ravel(grid.Y), np.ravel(grid.Z)])
    nk, n = len(data), vectors[0].shape[1]
    M = np.zeros((nk, len(steps), n, n), dtype=complex)
    for ib, (step, b) in enumerate(zip(steps, b_vectors)):
        grid_phase = np.exp(-1j * (b @ r))
        onsite = onsite_operator(crystal, b)
        neighbors = np.ravel_multi_index(((index + step) % size).T,
                                         tuple(size))
        for k in range(nk):
            S = link_overlap(crystal, data[k], data[neighbors[k]], b, onsite,
                             grid_phase)
            M[k, ib] = vectors[k].conj().T @ S @ vectors[neighbors[k]]
    return M


def neighbor_index(size, steps) -> np.ndarray:
    """``(nk, nb)``: the mesh index of each point's neighbor."""
    size = np.asarray(size, dtype=int)
    index = np.indices(tuple(size)).reshape(3, -1).T
    return np.stack([np.ravel_multi_index(((index + s) % size).T, tuple(size))
                     for s in steps], axis=1)


def gaussian_trials(centers) -> list:
    """s Gaussians (:class:`Trial`) at ``centers`` (Bohr)."""
    return [Trial(tuple(float(x) for x in c), label=f"s@{i}")
            for i, c in enumerate(np.atleast_2d(np.asarray(centers,
                                                           dtype=float)))]


def real_harmonic(l: int, m: int, x, y, z) -> np.ndarray:
    r"""The real spherical harmonic :math:`S_{lm}` at the Cartesian offsets
    ``x, y, z``, built from the complex :math:`Y_l^m` of
    :func:`~mandacaru.basis._angular.spherical_harmonic` (Condon-Shortley
    phase):

    .. math::

        S_{l0} = Y_l^0,\qquad
        S_{lm} = \sqrt2\,(-1)^m\operatorname{Re}Y_l^m,\qquad
        S_{l,-m} = \sqrt2\,(-1)^m\operatorname{Im}Y_l^m\quad(m > 0),

    orthonormal on the sphere and positive along the Cartesian lobes:
    :math:`S_{11}\propto x`, :math:`S_{1,-1}\propto y`,
    :math:`S_{10}\propto z`, :math:`S_{2,-2}\propto xy`,
    :math:`S_{2,-1}\propto yz`, :math:`S_{20}\propto 3z^2 - r^2`,
    :math:`S_{21}\propto xz`, :math:`S_{22}\propto x^2 - y^2`."""
    from ..basis._angular import spherical_coords, spherical_harmonic

    l, m = int(l), int(m)
    if abs(m) > l:
        raise ValueError(f"require |m| <= l; got l={l}, m={m}")
    _r, theta, phi = spherical_coords(x, y, z, (0.0, 0.0, 0.0))
    Y = spherical_harmonic(l, abs(m), theta, phi)
    if m == 0:
        return Y.real
    sign = np.sqrt(2.0) * (-1.0) ** abs(m)
    return sign * (Y.real if m > 0 else Y.imag)


def gaussian_radial(l: int, r, width: float = TRIAL_WIDTH) -> np.ndarray:
    r""":math:`N_l r^l e^{-r^2/2w^2}`, normalized
    (:math:`\int_0^\infty R^2r^2\,dr = 1`), :math:`w` = ``width`` (Bohr)."""
    from scipy.special import gamma

    l = int(l)
    norm = np.sqrt(2.0 / (gamma(l + 1.5) * width ** (2 * l + 3)))
    r = np.asarray(r, dtype=float)
    return norm * r ** l * np.exp(-0.5 * r * r / width ** 2)


#: The radial functions of a trial orbital are taken to vanish past the
#: radius where they fall below this fraction of their largest value.
TRIAL_SUPPORT_TOLERANCE = 1e-8


def trial_reach(trial: Trial, width: float = TRIAL_WIDTH) -> float:
    """The radius (Bohr) past which the trial orbital is negligible: six
    widths for Gaussians, else where its radial functions fall below
    :data:`TRIAL_SUPPORT_TOLERANCE` of their peak."""
    if trial.radial is None:
        return 6.0 * width
    r = np.linspace(0.0, 40.0, 8001)
    reach = 0.0
    for _l, function in trial.radial:
        values = np.abs(np.asarray(function(r), dtype=float))
        keep = np.flatnonzero(values > TRIAL_SUPPORT_TOLERANCE
                              * values.max())
        reach = max(reach, float(r[min(keep[-1] + 1, r.size - 1)]))
    return reach


def trial_values(trial: Trial, offset, width: float = TRIAL_WIDTH):
    """The trial orbital at Cartesian ``offset`` from its center (``(3,
    n)``, Bohr)."""
    offset = np.asarray(offset, dtype=float)
    local = (offset if trial.frame is None
             else np.asarray(trial.frame, dtype=float) @ offset)
    r = np.sqrt(np.sum(local * local, axis=0))
    functions = None if trial.radial is None else dict(trial.radial)
    radials: dict = {}
    out = np.zeros(offset.shape[1])
    for l, m, c in trial.angular:
        if l not in radials:
            radials[l] = (gaussian_radial(l, r, width) if functions is None
                          else np.asarray(functions[l](r), dtype=float))
        out += c * radials[l] * real_harmonic(l, m, *local)
    return out


def trial_projections(crystal, data, vectors, fractional, trials,
                      width: float = TRIAL_WIDTH) -> np.ndarray:
    r"""``A[k]`` (``(nk, n_bands, n_trial)``): :math:`\langle\psi_{m\mathbf
    k}|g_{n\mathbf k}\rangle` with :math:`g_n` the :class:`Trial` orbitals
    (Gaussians of ``width``, Bohr, unless a trial carries its own radial
    functions), Bloch-summed over the images that reach the cell,
    :math:`g_{n\mathbf k} = \sum_{\mathbf R}e^{i\mathbf k\cdot\mathbf R}
    g_n(\mathbf r - \mathbf R)`; the smooth part only, which is all a
    starting gauge needs."""
    grid = crystal.grid
    lattice = np.asarray(crystal.lattice, dtype=float)     # columns
    B = rc.reciprocal_vectors(lattice)
    r = np.stack([np.ravel(grid.X), np.ravel(grid.Y), np.ravel(grid.Z)])
    out = np.zeros((len(data), vectors[0].shape[1], len(trials)),
                   dtype=complex)
    diagonal = float(np.linalg.norm(lattice.sum(axis=1)))
    kc = fractional @ B.T
    for n, trial in enumerate(trials):
        center = np.asarray(trial.center, dtype=float)
        reach = trial_reach(trial, width)
        # The trial's values on the points each image reaches, once.
        pieces = []
        for R in rc.lattice_translations(lattice, reach + diagonal):
            offset = r - (center + R)[:, None]
            near = np.flatnonzero(np.sum(offset ** 2, axis=0)
                                  < reach * reach)
            if near.size:
                pieces.append((R, near, trial_values(trial, offset[:, near],
                                                     width)))
        for k, (d, v) in enumerate(zip(data, vectors)):
            g = np.zeros(r.shape[1], dtype=complex)
            for R, near, values in pieces:
                g[near] += np.exp(1j * (kc[k] @ R)) * values
            out[k, :, n] = v.conj().T @ (d.psi.conj() @ g) * grid.dV
    return out


def loewdin_unitary(A) -> np.ndarray:
    """The (semi-)unitary closest to each ``A[k]``: :math:`A(A^\\dagger
    A)^{-1/2}` through its singular-value decomposition."""
    out = np.empty_like(A)
    for k, a in enumerate(A):
        u, _s, vh = np.linalg.svd(a, full_matrices=False)
        out[k] = u @ vh
    return out


# --------------------------------------------------------------------------- #
# Disentanglement.
# --------------------------------------------------------------------------- #

def resolve_windows(windows) -> tuple:
    """``(outer, frozen)`` in Hartree of ``{"outer": (lo, hi), "frozen":
    (lo, hi)}`` in eV; a missing outer window takes every band, a missing
    frozen window freezes none."""
    if windows is None:
        windows = {}
    if not isinstance(windows, dict):
        raise ValueError("windows must be a dict {'outer': (lo, hi), "
                         "'frozen': (lo, hi)} in eV")
    unknown = set(windows) - {"outer", "frozen"}
    if unknown:
        raise ValueError(f"unknown window(s) {sorted(unknown)}; use 'outer' "
                         "and 'frozen' (eV)")

    def pair(name):
        value = windows.get(name)
        if value is None:
            return None
        lo, hi = (float(x) for x in value)
        if not lo < hi:
            raise ValueError(f"the {name} window must be (lo, hi) with "
                             f"lo < hi; got {value!r}")
        return lo * EV_TO_HARTREE, hi * EV_TO_HARTREE

    outer = pair("outer") or (-np.inf, np.inf)
    frozen = pair("frozen")
    if frozen is not None and (frozen[0] < outer[0] or frozen[1] > outer[1]):
        raise ValueError("the frozen window must lie inside the outer one")
    return outer, frozen


def disentangle(M, neighbors, weights, energies, A, n_functions, outer,
                frozen=None, *, mixing: float = DISENTANGLE_MIXING,
                max_iter: int = 5000, tol: float = DISENTANGLE_TOLERANCE,
                masks=None):
    r"""The ``n_functions``-dimensional subspace per k-point that minimizes
    :math:`\Omega_I` (Souza, Marzari and Vanderbilt 2001).

    ``M`` are the overlaps between all the bands of ``energies``
    (``(nk, nb)``, Hartree), ``A`` the trial projections onto them.  At each
    k-point the subspace holds the states of the ``frozen`` window and the
    leading eigenvectors of the mixed :math:`Z(\mathbf k)` over the other
    states of the ``outer`` window, starting from the trials' projection.
    ``masks`` -- boolean ``(inside, fixed)``, ``(nk, nb)`` each -- give the
    two sets state by state instead of by energy (the projectability
    windows of :mod:`~mandacaru.algorithms.band_selection`).
    Returns ``(S, history)``: ``S[k]`` ``(nb, n_functions)`` orthonormal
    columns in the band basis (zero outside the window) and Omega_I (Bohr^2)
    per iteration.
    """
    nk, nb = energies.shape
    J = int(n_functions)
    if masks is not None:
        inside, fixed = (np.asarray(m, dtype=bool) for m in masks)
        fixed = fixed & inside
    else:
        inside = (energies >= outer[0]) & (energies <= outer[1])
        fixed = np.zeros_like(inside)
        if frozen is not None:
            fixed = inside & (energies >= frozen[0]) \
                & (energies <= frozen[1])
    counts, n_fixed = inside.sum(axis=1), fixed.sum(axis=1)
    if counts.min() < J:
        k = int(np.argmin(counts))
        raise ValueError(f"the outer window holds {counts[k]} bands at "
                         f"k-point {k}, fewer than the {J} functions")
    if n_fixed.max() > J:
        k = int(np.argmax(n_fixed))
        raise ValueError(f"the frozen window holds {n_fixed[k]} bands at "
                         f"k-point {k}, more than the {J} functions")
    free = [np.flatnonzero(inside[k] & ~fixed[k]) for k in range(nk)]
    S = np.zeros((nk, nb, J), dtype=complex)
    for k in range(nk):
        f = np.flatnonzero(fixed[k])
        S[k, f, np.arange(len(f))] = 1.0
        rest = J - len(f)
        if rest:
            u, _s, _vh = np.linalg.svd(A[k][free[k]], full_matrices=False)
            S[k, free[k], len(f):] = u[:, :rest]
    history: list = []
    Z_in = None
    for _it in range(max_iter):
        X = np.einsum("kbmn,kbnj->kbmj", M, S[neighbors])
        projected = np.einsum("kmi,kbmj->kbij", np.conj(S), X)
        omega_I = float(np.einsum("b,kb->", weights, J - np.sum(
            np.abs(projected) ** 2, axis=(2, 3)))) / nk
        history.append(omega_I)
        if len(history) > 1 and abs(history[-2] - omega_I) < tol:
            break
        Z = np.einsum("b,kbmj,kbnj->kmn", weights, X, np.conj(X),
                      optimize=True)
        Z_in = Z if Z_in is None else mixing * Z + (1.0 - mixing) * Z_in
        for k in range(nk):
            rest = J - int(n_fixed[k])
            if not rest:
                continue
            block = Z_in[k][np.ix_(free[k], free[k])]
            _values, vecs = np.linalg.eigh(0.5 * (block + block.conj().T))
            S[k, free[k], int(n_fixed[k]):] = vecs[:, -rest:]
    return S, history


# --------------------------------------------------------------------------- #
# The spread and its minimization.
# --------------------------------------------------------------------------- #

def _phases(diagonal, b_vectors, reference=None) -> np.ndarray:
    r""":math:`\operatorname{Im}\ln M_{nn}(\mathbf k, \mathbf b)` on the
    branch nearest :math:`-\mathbf b\cdot\mathbf r_n` for ``reference``
    centers :math:`\mathbf r_n` (Bohr), so a center on the branch cut of
    the plain logarithm -- halfway across the cell along a direction with
    a single k-point, say -- does not flip between iterations."""
    if reference is None:
        return np.angle(diagonal)
    shift = np.einsum("bi,ni->bn", b_vectors, reference)[None]
    return np.angle(diagonal * np.exp(1j * shift)) - shift


def spread(M, weights, b_vectors, reference=None):
    """``(centers, second, omega_I, omega_OD, omega_D)``: the centers
    (Bohr), :math:`\\langle r^2\\rangle_n` (Bohr^2) and the three parts of
    the total spread (Marzari and Vanderbilt 1997, eqs. 31-36), the phases
    taken near the ``reference`` centers (:func:`_phases`)."""
    nk = M.shape[0]
    diagonal = np.einsum("kbnn->kbn", M)
    phase = _phases(diagonal, b_vectors, reference)          # Im ln M_nn
    centers = -np.einsum("b,bi,kbn->ni", weights, b_vectors, phase) / nk
    second = np.einsum("b,kbn->n", weights,
                       1.0 - np.abs(diagonal) ** 2 + phase ** 2) / nk
    n = M.shape[2]
    omega_I = float(np.einsum("b,kb->", weights,
                              n - np.sum(np.abs(M) ** 2, axis=(2, 3)))) / nk
    off = np.abs(M) ** 2
    off = np.sum(off, axis=(2, 3)) - np.sum(np.abs(diagonal) ** 2, axis=2)
    omega_OD = float(np.einsum("b,kb->", weights, off)) / nk
    projected = phase + np.einsum("bi,ni->bn", b_vectors, centers)[None]
    omega_D = float(np.einsum("b,kbn->", weights, projected ** 2)) / nk
    return centers, second, omega_I, omega_OD, omega_D


def _gradient(M, weights, b_vectors, centers) -> np.ndarray:
    """``G[k]``: the steepest-descent direction of the gauge-dependent
    spread, anti-Hermitian (eq. 52 of Marzari and Vanderbilt 1997)."""
    diagonal = np.einsum("kbnn->kbn", M)
    phase = _phases(diagonal, b_vectors, centers)
    q = phase + np.einsum("bi,ni->bn", b_vectors, centers)[None]
    R = M * np.conj(diagonal)[:, :, None, :]
    T = (M / diagonal[:, :, None, :]) * q[:, :, None, :]

    def anti(X):
        return 0.5 * (X - np.conj(np.swapaxes(X, -1, -2)))

    def sym(X):
        return (X + np.conj(np.swapaxes(X, -1, -2))) / 2j

    return 4.0 * np.einsum("b,kbmn->kmn", weights, anti(R) - sym(T))


def _expm_antihermitian(W) -> np.ndarray:
    """``exp(W)`` of anti-Hermitian matrices through the Hermitian ``iW``."""
    out = np.empty_like(W)
    for k, w in enumerate(W):
        values, vectors = np.linalg.eigh(1j * w)
        out[k] = (vectors * np.exp(-1j * values)) @ vectors.conj().T
    return out


def minimize(M0, neighbors, weights, b_vectors, U0, max_iter: int = 2000,
             tol: float = SPREAD_TOLERANCE, reference=None):
    """Steepest descent of the gauge-dependent spread from the gauge
    ``U0``.  ``M0`` are the overlaps in the Bloch gauge; the phases follow
    the centers from ``reference`` (Bohr; the trial orbitals', say).
    Returns ``(U, M, history, centers)``."""
    U = np.array(U0, copy=True)
    step = DESCENT_STEP / (4.0 * float(np.sum(weights)))

    def rotated(U):
        # M(k, b) = U(k)^+ M0(k, b) U(k + b).
        return np.einsum("kam,kbac,kbcn->kbmn", np.conj(U), M0, U[neighbors],
                         optimize=True)

    M = rotated(U)
    history = []
    previous = np.inf
    centers = reference
    for _it in range(max_iter):
        centers, second, oI, oOD, oD = spread(M, weights, b_vectors,
                                              centers)
        total = oI + oOD + oD
        history.append((oI, oOD, oD))
        if abs(previous - total) < tol:
            break
        previous = total
        G = _gradient(M, weights, b_vectors, centers)
        U = np.einsum("kmc,kcn->kmn", U, _expm_antihermitian(step * G))
        M = rotated(U)
    return U, M, history, centers


# --------------------------------------------------------------------------- #
# The Hamiltonian in the Wannier basis.
# --------------------------------------------------------------------------- #

def real_phases(data, vectors, U, size) -> np.ndarray:
    r"""``(n,)``: :math:`e^{-i\theta_n}` with :math:`2\theta_n` the
    argument of :math:`\sum_{\mathbf r}w_n(\mathbf r)^2 \propto
    \sum_{\mathbf k}\int_\Omega\phi_{n\mathbf k}\,\phi_{n,-\mathbf
    k}` (the smooth parts, :math:`\phi_{n\mathbf k} = \sum_m U_{mn}
    \psi_{m\mathbf k}`): the global phase that makes each function as
    real as it can be -- real outright where a real gauge exists -- so the
    integrals of a downfolded problem are real too.  The spreads do not
    depend on it."""
    size = np.asarray(size, dtype=int)
    index = np.indices(tuple(size)).reshape(3, -1).T
    minus = np.ravel_multi_index(tuple(((-index) % size).T), tuple(size))
    phi = [(v @ u).T @ d.psi for d, v, u in zip(data, vectors, U)]
    total = sum(np.sum(phi[k] * phi[minus[k]], axis=1)
                for k in range(len(phi)))
    return np.exp(-0.5j * np.angle(total))


def wigner_seitz_vectors(lattice, size):
    """``(vectors, degeneracy)``: the integer lattice vectors of the
    supercell's Wigner-Seitz cell, each with the number of equivalent images
    it is shared among."""
    lattice = np.asarray(lattice, dtype=float)              # columns
    size = np.asarray(size, dtype=int)
    supercell = [np.array(t) * size for t in itertools.product((-1, 0, 1),
                                                              repeat=3)]
    vectors, degeneracy = [], []
    ranges = [range(-int(n), int(n) + 1) for n in size]
    for R in itertools.product(*ranges):
        R = np.array(R)
        distances = [np.linalg.norm(lattice @ (R - S)) for S in supercell]
        own = np.linalg.norm(lattice @ R)
        smallest = min(distances)
        if own <= smallest + 1e-7:
            vectors.append(R)
            degeneracy.append(sum(1 for d in distances
                                  if abs(d - smallest) < 1e-7))
    return np.array(vectors), np.array(degeneracy, dtype=float)


def bvk_blocks(fractional, Hk, size) -> np.ndarray:
    r"""``(N, n, n)``: :math:`X(\mathbf t) = \frac1N\sum_{\mathbf k}
    e^{-i\mathbf k\cdot\mathbf t}X(\mathbf k)` for every cell ``t`` of the
    Born-von Karman supercell (C order)."""
    cells = np.indices(tuple(size)).reshape(3, -1).T
    phases = np.exp(-2j * np.pi * fractional @ cells.T)        # (nk, N)
    return np.einsum("kt,kmn->tmn", phases, Hk) / len(fractional)


def nearest_replicas(lattice, size, centers, blocks):
    r"""``(vectors, H)``: the Born-von Karman ``blocks`` (``(N, n, n)``)
    unfolded for interpolation.  For each Wigner-Seitz vector
    :math:`\mathbf R` of the supercell and each pair :math:`(m, n)`, the
    hopping goes to the replicas :math:`\mathbf R + \mathbf T` that minimize
    :math:`|\mathbf R + \mathbf T + \boldsymbol\tau_n -
    \boldsymbol\tau_m|` (``centers`` :math:`\boldsymbol\tau`, Bohr), shared
    equally among equidistant ones and divided by the vector's own
    degeneracy.  On the mesh the sum is unchanged; between mesh points the
    hoppings sit at their physical distances."""
    lattice = np.asarray(lattice, dtype=float)
    size = np.asarray(size, dtype=int)
    centers = np.asarray(centers, dtype=float)
    R_ws, degeneracy = wigner_seitz_vectors(lattice, size)
    shifts = np.array([np.array(t) * size for t in
                       itertools.product(range(-2, 3), repeat=3)])
    pair = centers[None, :, :] - centers[:, None, :]          # tau_n - tau_m
    out: dict = {}
    for R, g in zip(R_ws, degeneracy):
        H = blocks[np.ravel_multi_index(tuple(R % size), tuple(size))]
        candidates = R[None, :] + shifts                       # (nT, 3)
        cartesian = candidates @ lattice.T
        distance = np.linalg.norm(cartesian[:, None, None, :]
                                  + pair[None], axis=-1)       # (nT, n, n)
        best = distance.min(axis=0)
        near = distance <= best * (1.0 + REPLICA_TOLERANCE) + 1e-8
        weight = near / (near.sum(axis=0)[None] * g)
        for T in np.flatnonzero(near.any(axis=(1, 2))):
            key = tuple(int(x) for x in candidates[T])
            out[key] = out.get(key, 0.0) + weight[T] * H
    keys = sorted(out)
    return np.array(keys, dtype=int), np.array([out[k] for k in keys])


# --------------------------------------------------------------------------- #
# The functions on the supercell and their Coulomb integrals.
# --------------------------------------------------------------------------- #

def supercell_functions(crystal, fractional, coefficients, size):
    r"""``(values, projections)`` of the home cell's Wannier functions on
    the Born-von Karman supercell: :math:`\tilde w_n(\mathbf r + \mathbf T)
    = \frac1N\sum_{\mathbf k}e^{i\mathbf k\cdot\mathbf T}\sum_\mu
    c_{\mu n}(\mathbf k)\chi_{\mu\mathbf k}(\mathbf r)` on the cell grid of
    every cell ``T`` (``(n, S1 n1, S2 n2, S3 n3)``), and
    :math:`\langle\tilde p_{p\mathbf T}|w_n\rangle` (``(n, N, P)``).
    ``coefficients[k]`` (``(M, n)``) are the functions' Bloch
    coefficients."""
    size = tuple(int(s) for s in size)
    grid = crystal.grid
    shape = tuple(grid.shape)
    B = rc.reciprocal_vectors(crystal.lattice)
    cells = np.indices(size).reshape(3, -1).T
    N = len(cells)
    phases = np.exp(2j * np.pi * fractional @ cells.T) / N     # (nk, N)
    J = coefficients.shape[2]
    values = np.zeros((J, N, grid.size), dtype=complex)
    projections = np.zeros((J, N, len(crystal.projectors)), dtype=complex)
    kpoints = fractional @ B.T
    block = crystal.kpoint_block()
    for start in range(0, len(kpoints), block):
        data = crystal.kpoint_matrices(kpoints[start:start + block])
        for i, d in enumerate(data):
            k = start + i
            c = coefficients[k]
            phi = c.T @ d.psi                                  # (J, ngrid)
            values += phases[k][None, :, None] * phi[:, None, :]
            a = (d.projections.conj().T @ c).T                 # (J, P)
            projections += phases[k][None, :, None] * a[:, None, :]
    values = values.reshape(J, *size, *shape).transpose(0, 1, 4, 2, 5, 3, 6)
    values = values.reshape(J, *(s * n for s, n in zip(size, shape)))
    return values, projections


def resolve_fragment(functions, n_functions) -> list:
    """``[(n, R), ...]`` of a fragment given as indices or ``(n, R)``."""
    if functions is None:
        functions = range(n_functions)
    modes = []
    for item in functions:
        if isinstance(item, (int, np.integer)):
            n, R = int(item), (0, 0, 0)
        else:
            n, R = item
            n, R = int(n), tuple(int(x) for x in R)
            if len(R) != 3:
                raise ValueError(f"a lattice vector has three integers; got "
                                 f"{R!r}")
        if not 0 <= n < n_functions:
            raise IndexError(f"function {n} of {n_functions}")
        modes.append((n, R))
    if len(set(modes)) != len(modes):
        raise ValueError("a fragment lists each function once")
    return modes


def truncated_coulomb(G2, radius: float) -> np.ndarray:
    r""":math:`\frac{4\pi}{G^2}\big(1 - \cos(G R_c)\big)`, the Coulomb
    interaction cut at :math:`R_c`, with its finite :math:`\mathbf G = 0`
    limit :math:`2\pi R_c^2` (Spencer and Alavi 2008)."""
    G2 = np.asarray(G2, dtype=float)
    safe = np.where(G2 > 0.0, G2, 1.0)
    return np.where(G2 > 0.0, 4.0 * np.pi / safe
                    * (1.0 - np.cos(np.sqrt(safe) * radius)),
                    2.0 * np.pi * radius ** 2)


class SupercellCoulomb:
    r"""The Born-von Karman supercell of a mesh as the box of the fragment
    interaction: its reciprocal grid (``G``, ``m`` the integer indices, C
    order of the supercell grid), the Coulomb ``kernel`` truncated at
    ``radius`` (default half the supercell's shortest lattice vector,
    :func:`truncated_coulomb`) and each compensation channel's on-site
    self-energy beyond the sphere inscribed in the grid's reciprocal box
    (``tails``), where the compensation charges stop interacting on the
    grid -- as in the crystal's own Hartree energy."""

    def __init__(self, crystal, size, radius=None):
        from ..pseudopotentials.periodic_paw import SHAPE_TABLE_MAX

        self.crystal = crystal
        self.size = np.asarray(size, dtype=int)
        grid = crystal.grid
        self.shape = np.asarray(grid.shape, dtype=int)
        lattice = np.asarray(crystal.lattice, dtype=float)
        self.lattice = lattice
        supercell = lattice * self.size[None, :]
        self.volume = abs(float(np.linalg.det(supercell)))
        self.total = tuple(int(n) for n in self.shape * self.size)
        B_super = rc.reciprocal_vectors(supercell)
        self.m = np.stack(np.meshgrid(
            *[np.fft.fftfreq(n, d=1.0 / n).astype(int) for n in self.total],
            indexing="ij"))
        self.G = np.einsum("ci,ixyz->cxyz", B_super, self.m)
        self.G2 = np.sum(self.G * self.G, axis=0)
        #: Half the supercell's shortest lattice vector (Bohr).
        self.half_shortest = 0.5 * min(
            float(np.linalg.norm(supercell @ np.asarray(t)))
            for t in itertools.product(range(-1, 2), repeat=3) if any(t))
        self.radius = (self.half_shortest if radius is None
                       else float(radius))
        self.kernel = truncated_coulomb(self.G2, self.radius)
        self.sphere = np.pi * np.min(self.shape
                                     / np.linalg.norm(lattice, axis=0))
        self.outside = self.G2 > self.sphere ** 2
        origin = rc.grid_origin(grid)
        self.origin_phase = np.exp(-1j * np.einsum("c,cxyz->xyz", origin,
                                                   self.G))
        self.tails = []
        for atom, L, _M in crystal.channels:
            table_q, table_F = crystal._shape_table(atom, L, SHAPE_TABLE_MAX)
            keep = table_q >= self.sphere
            self.tails.append(8.0 * np.trapezoid(table_F[keep] ** 2,
                                                 table_q[keep]))


@dataclass
class PairDensities:
    r"""Pair densities :math:`\rho_{pq} = w_p^*w_q` of a fragment's
    functions on the supercell's reciprocal grid: ``density`` (``(n, n,
    N_G)``, the smooth product plus the compensation charges), ``compact``
    (the compensation charges alone) and ``charges[c]`` (``(N, n, n)``,
    :math:`Q^{A\mathbf T}_{LM,pq}` of channel ``c`` per supercell cell)."""

    density: np.ndarray
    compact: np.ndarray
    charges: list

    @property
    def swapped(self) -> np.ndarray:
        """``density`` with the pair order reversed, ``rho_qp`` at
        ``(p, q)``, flattened to ``(n * n, N_G)``."""
        n = self.density.shape[0]
        return self.density.transpose(1, 0, 2).reshape(n * n, -1)


def pair_densities(box: SupercellCoulomb, values, projections,
                   modes) -> PairDensities:
    r""":class:`PairDensities` of the fragment ``modes`` ``(n, R)`` from the
    supercell functions (``values``, ``projections``,
    :func:`supercell_functions`): the smooth products on the supercell grid
    plus the PAW compensation charges of every atom of the supercell,
    :math:`\sum_{LM}Q^{A\mathbf T}_{LM,pq}\hat g_{LM}` with :math:`Q =
    a_p^\dagger\,\mathrm{blk}_{LM}\,a_q` from the projector
    coefficients."""
    crystal = box.crystal
    shape = box.shape
    cells = tuple(int(s) for s in box.size)
    dV = crystal.grid.dV

    def rolled(n, R):
        return np.roll(values[n], tuple(int(r) * int(s) for r, s in
                                        zip(R, shape)), axis=(0, 1, 2))

    def shifted_projections(n, R):
        # <p_T|w_{nR}> = <p_{T-R}|w_{n0}>.
        table = projections[n].reshape(*cells, -1)
        return np.roll(table, tuple(int(r) for r in R),
                       axis=(0, 1, 2)).reshape(projections.shape[1], -1)

    functions = [rolled(n, R) for n, R in modes]
    coefficients = [shifted_projections(n, R) for n, R in modes]
    n = len(modes)
    m_mod = [np.mod(mi, s) for mi, s in zip(box.m, cells)]
    density = np.empty((n, n, box.G2.size), dtype=complex)
    compact = np.zeros((n, n, box.G2.size), dtype=complex)
    charges, sums = [], []
    for p in range(n):
        for q in range(n):
            product = np.conj(functions[p]) * functions[q]
            density[p, q] = np.ravel(dV * box.origin_phase
                                     * np.fft.fftn(product))
    for atom, L, M in crystal.channels:
        own = crystal.projector_columns(atom)
        blk = crystal.multipole_blocks[(atom, L, M)]
        a = np.stack([coefficient[:, own] for coefficient in coefficients])
        # Q[t, p, q] = a_p(t)^+ blk a_q(t) per supercell cell t.
        Q = np.einsum("pti,ij,qtj->tpq", np.conj(a), blk, a)
        charges.append(Q)
        sums.append(np.fft.fftn(Q.reshape(*cells, n, n), axes=(0, 1, 2)))
    # The compensation transforms a block of G at a time.
    G = box.G.reshape(3, -1)
    cell_of = [np.ravel(mi) for mi in m_mod]
    for start in range(0, G.shape[1], TRANSFORM_BLOCK):
        part = slice(start, start + TRANSFORM_BLOCK)
        transforms = crystal.compensation_transforms(G[:, part])
        index = tuple(c[part] for c in cell_of)
        for c, lattice_sum in enumerate(sums):
            compact[:, :, part] += transforms[c][None, None] * np.moveaxis(
                lattice_sum[index], 0, -1)
    density += compact
    return PairDensities(density=density, compact=compact, charges=charges)


def coulomb_integrals(box: SupercellCoulomb, first: PairDensities,
                      second: PairDensities) -> np.ndarray:
    r"""``(n, n, n', n')``: :math:`(pq|rs) = \iint\rho_{pq}(\mathbf r)
    v(|\mathbf r-\mathbf r'|)\rho_{rs}(\mathbf r')` (chemists' notation,
    Hartree), ``pq`` from ``first`` and ``rs`` from ``second``.

    The interaction is the ``box``'s kernel, :func:`truncated_coulomb`, so
    the supercell's images do not interact with a fragment whose charge
    fits within the radius.  The compensation charges interact among
    themselves within the sphere inscribed in the grid's reciprocal box,
    plus each charge's own self-energy beyond it.
    """
    n1 = first.density.shape[0]
    n2 = second.density.shape[0]
    kernel = box.kernel.ravel()
    left = first.swapped
    right = second.density.reshape(n2 * n2, -1)
    out = (np.conj(left) * kernel) @ right.T
    cut = kernel * box.outside.ravel()
    left_compact = first.compact.transpose(1, 0, 2).reshape(n1 * n1, -1)
    out -= (np.conj(left_compact) * cut) @ second.compact.reshape(
        n2 * n2, -1).T
    out = out.reshape(n1, n1, n2, n2) / box.volume
    for tail, Qa, Qb in zip(box.tails, first.charges, second.charges):
        out += tail * np.einsum("tpq,trs->pqrs", Qa, Qb)
    return out


def screening_correction(box: SupercellCoulomb, first: PairDensities,
                         second: PairDensities,
                         screening: "ScreenedInteraction") -> np.ndarray:
    r"""``(n, n, n', n')``: what the screening adds to
    :func:`coulomb_integrals`,
    :math:`\frac{1}{N\Omega}\sum_{\mathbf q\mathbf G\mathbf G'}
    \rho_{qp}(\mathbf q+\mathbf G)^*\,[W - v]_{\mathbf G\mathbf G'}
    (\mathbf q)\,\rho_{rs}(\mathbf q+\mathbf G')`, from pair densities on
    the supercell ``box`` the ``screening`` (:func:`screened_interaction`)
    was solved on."""
    if tuple(box.size) != tuple(screening.size):
        raise ValueError("the screened interaction belongs to another "
                         "supercell")
    n1 = first.density.shape[0]
    n2 = second.density.shape[0]
    left = first.swapped
    right = second.density.reshape(n2 * n2, -1)
    out = np.zeros((n1 * n1, n2 * n2), dtype=complex)
    total = np.asarray(box.total)[:, None]
    for m, delta in screening.blocks:
        index = np.ravel_multi_index(tuple(np.mod(m, total)), box.total)
        out += np.conj(left[:, index]) @ delta @ right[:, index].T
    return out.reshape(n1, n1, n2, n2) / box.volume


def nearest_image_mask(points, center, supercell, period,
                       block: int = 262144):
    """Whether each of ``points`` (``(3, n)``, Bohr, on a box periodic with
    the lattice ``period``, columns) is its own nearest image to
    ``center`` among the translations by the ``supercell``'s lattice
    vectors (columns): the Wigner-Seitz cell of the supercell centered
    there.  Points equidistant from several images are shared, each
    weighted by one over their number."""
    shifts = supercell @ np.array(list(itertools.product((-1, 0, 1),
                                                         repeat=3))).T
    period = np.asarray(period, dtype=float)
    out = np.empty(points.shape[1])
    for start in range(0, points.shape[1], block):
        d = points[:, start:start + block] - np.asarray(center)[:, None]
        f = np.linalg.solve(period, d)
        d = period @ (f - np.round(f))
        own = np.sum(d * d, axis=0)
        other = np.sum((d[:, None, :] + shifts[:, :, None]) ** 2, axis=0)
        nearest = other.min(axis=0)
        tie = np.sum(other <= own * (1 + 1e-9) + 1e-12, axis=0)
        out[start:start + block] = np.where(own <= nearest * (1 + 1e-9)
                                            + 1e-12, 1.0 / tie, 0.0)
    return out


def padded_fragment(channel, modes, center=None, box=None):
    r"""``(box, values, projections, modes)`` of the fragment ``modes``
    isolated (minimum image): each function, Born-von Karman periodic on
    the supercell, kept on the supercell's Wigner-Seitz cell centered on
    the fragment and set in a supercell twice as large in every direction,
    zero elsewhere; each atom copy's projector coefficients kept the same
    way.  The truncation radius of that ``box`` is the original supercell's
    shortest vector, twice the plain one, so the fragment's charge stays
    clear of its copies up to that diameter.  Eight times the grid
    points.  ``center`` (Bohr) defaults to the mean of the fragment's
    centers; ``box`` reuses a doubled :class:`SupercellCoulomb`."""
    values, projections = channel._supercell()
    crystal = channel._crystal
    size = np.asarray(channel.size, dtype=int)
    shape = np.asarray(crystal.grid.shape, dtype=int)
    lattice = np.asarray(crystal.lattice, dtype=float)
    step = np.asarray(crystal.grid.step, dtype=float)
    origin = rc.grid_origin(crystal.grid)
    supercell = lattice * size[None, :]
    if center is None:
        center = fragment_center(channel, modes)
    big = 2 * size * shape
    index = np.indices(tuple(big)).reshape(3, -1)
    period = 2.0 * supercell
    mask = nearest_image_mask(origin[:, None] + step @ index, center,
                              supercell, period).reshape(tuple(big))
    tiles = (2, 2, 2)
    cells = np.indices(tuple(2 * size)).reshape(3, -1)
    atom_mask = np.stack([nearest_image_mask(
        np.asarray(position)[:, None] + lattice @ cells, center, supercell,
        period) for position in crystal.centers])          # (atoms, N')
    column_mask = np.zeros((cells.shape[1], projections.shape[2]))
    for atom in range(len(crystal.centers)):
        column_mask[:, crystal.projector_columns(atom)] = atom_mask[atom][
            :, None]
    out_values = np.empty((len(modes),) + tuple(big), dtype=complex)
    out_projections = np.empty((len(modes), cells.shape[1],
                                projections.shape[2]), dtype=complex)
    for i, (n, R) in enumerate(modes):
        shifted = np.roll(values[n], tuple(np.asarray(R) * shape),
                          axis=(0, 1, 2))
        out_values[i] = np.tile(shifted, tiles) * mask
        table = np.roll(projections[n].reshape(*size, -1), tuple(R),
                        axis=(0, 1, 2))
        out_projections[i] = np.tile(table, tiles + (1,)).reshape(
            -1, table.shape[-1]) * column_mask
    if box is None:
        box = SupercellCoulomb(crystal, 2 * size)
    return (box, out_values, out_projections,
            [(i, (0, 0, 0)) for i in range(len(modes))])


def _real_columns(vectors) -> np.ndarray:
    """Each column times the phase that makes its largest entry real and
    positive: an eigenvector's arbitrary phase fixed, so a real matrix's
    eigenvectors stay real."""
    rows = np.argmax(np.abs(vectors), axis=0)
    entries = vectors[rows, np.arange(vectors.shape[1])]
    return vectors * (np.abs(entries) / np.where(entries == 0, 1.0,
                                                 entries))[None, :]


def fragment_center(channel, modes) -> np.ndarray:
    """The mean of the fragment's centers (Bohr)."""
    lattice = np.asarray(channel.lattice, dtype=float).T          # columns
    return np.mean([channel.centers[n] + lattice @ np.asarray(R, float)
                    for n, R in modes], axis=0) / BOHR_TO_ANGSTROM


def fragment_extent(channel, modes) -> float:
    """The fragment's diameter (Bohr): the largest distance between two of
    its functions' centers plus :data:`FRAGMENT_REACH` root-mean-square
    radii of each -- how far apart two points of its charge lie."""
    lattice = np.asarray(channel.lattice, dtype=float).T          # columns
    centers = np.array([channel.centers[n] + lattice @ np.asarray(R)
                        for n, R in modes]) / BOHR_TO_ANGSTROM
    radii = np.sqrt(np.maximum(np.array([channel.spreads[n]
                                         for n, _R in modes]), 0.0)) \
        / BOHR_TO_ANGSTROM
    distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    return float(np.max(distance + FRAGMENT_REACH
                        * (radii[:, None] + radii[None])))


def downfold_channels(channels, functions=None, num_particles=None,
                      screening=None, screening_cutoff=None,
                      screening_bands=None,
                      coulomb: str = "supercell") -> DownfoldedProblem:
    """The :class:`DownfoldedProblem` of a fragment of the Wannier functions
    of one channel (restricted) or two (up, down); see
    :meth:`WannierResult.downfold`."""
    first = channels[0]
    for channel in channels:
        if channel._density_bvk is None or channel._crystal is None:
            raise RuntimeError("this result carries no Kohn-Sham "
                               "occupations or crystal to downfold with")
        if channel.n_functions != first.n_functions:
            raise ValueError(
                "the spin channels have different numbers of functions "
                f"({[c.n_functions for c in channels]}): a fragment names "
                "the same functions in both")
    if screening not in (None, "crpa", "rpa"):
        raise ValueError(f"screening must be None, 'crpa' or 'rpa'; got "
                         f"{screening!r}")
    if coulomb not in ("supercell", "isolated"):
        raise ValueError(f"coulomb must be 'supercell' or 'isolated'; got "
                         f"{coulomb!r}")
    modes = resolve_fragment(functions, first.n_functions)
    size = np.asarray(first.size, dtype=int)
    if first._cite is not None:
        first._cite("Spencer2008")
        if screening is not None:
            first._cite("Aryasetiawan2004", "Sasioglu2011")
    box = first._coulomb_box()
    density_box = box
    if coulomb == "isolated":
        density_box = SupercellCoulomb(first._crystal, 2 * size)
        center = fragment_center(first, modes)
    for channel in channels:
        extent = fragment_extent(channel, modes)
        if extent > density_box.radius:
            warnings.warn(
                f"the fragment spans about {extent * BOHR_TO_ANGSTROM:.2f} A "
                f"({FRAGMENT_REACH:g} rms radii past its outer centers), "
                "beyond the interaction's truncation radius "
                f"{density_box.radius * BOHR_TO_ANGSTROM:.2f} A: its "
                "functions meet their own periodic copies; use "
                "coulomb='isolated', a denser k-mesh or a smaller "
                "fragment", stacklevel=3)

    def block(table, a, b):
        (m, Ra), (n, Rb) = a, b
        t = tuple((np.asarray(Rb) - np.asarray(Ra)) % size)
        return table[np.ravel_multi_index(t, tuple(size))][m, n]

    kohn_sham, gamma, densities, plain = [], [], [], []
    for channel in channels:
        h = np.array([[block(channel._hamiltonian_bvk, a, b) for b in modes]
                      for a in modes])
        rho = np.array([[block(channel._density_bvk, a, b) for b in modes]
                        for a in modes])
        kohn_sham.append(0.5 * (h + h.conj().T))
        gamma.append(rho.T)                 # <c+_p c_q> = <w_q|rho|w_p>
        plain_needed = coulomb == "supercell" or screening is not None
        if plain_needed:
            values, projections = channel._supercell()
            plain.append(pair_densities(box, values, projections, modes))
        if coulomb == "isolated":
            _box, values, projections, own = padded_fragment(
                channel, modes, center, density_box)
            densities.append(pair_densities(density_box, values,
                                            projections, own))
        else:
            densities.append(plain[-1])
    blocks = None
    if screening is not None:
        targets = [[n for n, _R in modes]] * len(channels)
        blocks = screened_interaction(box, channels, targets,
                                      exclude=screening == "crpa",
                                      cutoff=screening_cutoff,
                                      n_bands=screening_bands)
    ns = len(channels)
    chemist = [[coulomb_integrals(density_box, densities[s], densities[t])
                + (0.0 if blocks is None else
                   screening_correction(box, plain[s], plain[t], blocks))
                for t in range(ns)] for s in range(ns)]
    if ns == 1:
        (C,), = chemist
        g = gamma[0]
        double = (np.einsum("rs,pqrs->pq", g, C)
                  - 0.5 * np.einsum("rs,psrq->pq", g, C))
        doubles = [double]
        two_body = C.transpose(0, 2, 1, 3)          # <pq|rs> = (pr|qs)
    else:
        doubles = []
        for s in range(2):
            double = sum(np.einsum("rs,pqrs->pq", gamma[t], chemist[s][t])
                         for t in range(2))
            double = double - np.einsum("rs,psrq->pq", gamma[s],
                                        chemist[s][s])
            doubles.append(double)
        two_body = np.array([[chemist[s][t].transpose(0, 2, 1, 3)
                              for t in range(2)] for s in range(2)])
    doubles = [0.5 * (d + d.conj().T) for d in doubles]
    traces = [float(np.real(np.trace(g))) for g in gamma]
    if num_particles is None:
        counts = [int(round(t)) for t in traces]
        if any(abs(t - c) > 0.1 for t, c in zip(traces, counts)):
            raise ValueError(
                f"the fragment holds {sum(traces):.3f} Kohn-Sham electrons "
                f"({', '.join(f'{t:.3f}' for t in traces)} per channel), "
                "not near an integer: give num_particles")
        if ns == 1:
            n = counts[0]
            num_particles = (n - n // 2, n // 2)
        else:
            num_particles = tuple(counts)
    num_particles = tuple(int(x) for x in num_particles)
    if len(num_particles) != 2:
        raise ValueError(f"num_particles is (up, down); got "
                         f"{num_particles!r}")
    squeeze = (lambda x: x[0]) if ns == 1 else np.array
    return DownfoldedProblem(
        functions=tuple(modes), kohn_sham=squeeze(kohn_sham),
        two_body=two_body, density_matrix=squeeze(gamma),
        double_counting=squeeze(doubles), num_particles=num_particles,
        screening=screening, coulomb=coulomb,
        dielectric_head=None if blocks is None else blocks.head)


# --------------------------------------------------------------------------- #
# The screened interaction: constrained RPA.
# --------------------------------------------------------------------------- #

def _occupation_slope(x, method: str, step: float = 1e-4) -> np.ndarray:
    """``d occupation / dx`` by central differences."""
    from .periodic_dft import occupation

    return (occupation(x + step, method) - occupation(x - step, method)) \
        / (2.0 * step)


def screening_states(channel, n_bands=None, targets=None, shift=None,
                     weights=None):
    """``(psi, projections, energies, F, weight)`` per mesh point of one
    spin ``channel`` (a :class:`WannierResult`): the lowest ``n_bands``
    Kohn-Sham states on the cell grid and on the projectors, their energies
    and electrons per state (Hartree; the SCF's Fermi level and smearing),
    and :math:`p_a(\\mathbf k) = \\sum_{n\\in T}|\\langle\\psi_{a\\mathbf
    k}|\\phi_{n\\mathbf k}\\rangle|^2`, each state's weight in the Bloch
    subspace of the ``targets`` functions (zero without targets).  With a
    Cartesian ``shift`` (Bohr^-1) the points are moved by it and each
    state takes the ``weights`` of its band at the unshifted point."""
    from .periodic_dft import occupation

    solver = channel._solver
    crystal = solver.crystal
    B = rc.reciprocal_vectors(crystal.lattice)
    n_bands = crystal.M if n_bands is None else int(n_bands)
    if not 0 < n_bands <= crystal.M:
        raise ValueError(f"screening_bands must be in 1..{crystal.M}; got "
                         f"{n_bands}")
    V, v_tau, w, spin = solver.channel_potentials()[channel.spin or 0]
    reference = solver.eigenvalue_reference()
    degeneracy = 2.0 / solver.n_spins
    targets = [] if targets is None else sorted(set(targets))
    kpoints = channel.kpoints @ B.T
    if shift is not None:
        kpoints = kpoints + np.asarray(shift, dtype=float)[None, :]
    out = []
    k = 0
    for data, eps, vectors in solver._diagonalized(kpoints, V, v_tau, w,
                                                   spin):
        for d, e, v in zip(data, eps, vectors):
            v = v[:, :n_bands]
            e = e[:n_bands] + reference
            F = degeneracy * occupation((e - channel._fermi_level)
                                        / solver.width, solver.method)
            if weights is not None:
                weight = weights[k]
            elif targets:
                c = channel._coefficients[k][:, targets]
                weight = np.sum(np.abs(v.conj().T @ d.overlap @ c) ** 2,
                                axis=1)
            else:
                weight = np.zeros(n_bands)
            out.append((v.T @ d.psi, d.projections.conj().T @ v, e, F,
                        weight))
            k += 1
    return out


def polarizability(crystal, channel_pairs, q, g, n_cells: int) -> tuple:
    r"""``(chi, qG)``: the Kohn-Sham polarizability
    :math:`\chi_{\mathbf G\mathbf G'}(\mathbf q)` (Bohr^-3 Hartree^-1) on
    :math:`\mathbf q + \mathbf G` (``q`` Cartesian, ``g`` integer
    reciprocal-lattice coordinates, ``(3, n)``), from ``channel_pairs``:
    per spin channel ``(channel, [(state, partner), ...])`` with the states
    of :func:`screening_states` at :math:`\mathbf k` and :math:`\mathbf k +
    \mathbf q`; each transition is weighted by :math:`(F_a - F_b) /
    (\varepsilon_a - \varepsilon_b)` (the smearing's slope for a degenerate
    pair) times :math:`1 - p_a p_b`."""
    grid = crystal.grid
    shape = tuple(int(n) for n in grid.shape)
    B = rc.reciprocal_vectors(crystal.lattice)
    volume = abs(float(np.linalg.det(crystal.lattice)))
    r = np.stack([np.ravel(grid.X), np.ravel(grid.Y), np.ravel(grid.Z)])
    q = np.asarray(q, dtype=float)
    qG = q[:, None] + B @ g
    fft_index = np.ravel_multi_index(tuple(np.mod(g, np.asarray(
        shape)[:, None])), shape)
    to_G = grid.dV * np.exp(-1j * (rc.grid_origin(grid) @ (B @ g)))
    transforms = crystal.compensation_transforms(qG)          # (n_ch, nG)
    q_phase = np.exp(-1j * (q @ r))
    chi = np.zeros((qG.shape[1],) * 2, dtype=complex)
    for channel, pairs in channel_pairs:
        solver = channel._solver
        for (psi1, a1, e1, F1, p1), (psi2, a2, e2, F2, p2) in pairs:
            dE = e1[:, None] - e2[None, :]
            dF = F1[:, None] - F2[None, :]
            close = np.abs(dE) < 1e-9
            weight = dF / np.where(close, 1.0, dE)
            if np.any(close):
                # A degenerate pair: the slope of the smearing.
                x = (0.5 * (e1[:, None] + e2[None, :])
                     - channel._fermi_level) / solver.width
                slope = (2.0 / solver.n_spins) * _occupation_slope(
                    x, solver.method) / solver.width
                weight = np.where(close, slope, weight)
            weight = weight * (1.0 - p1[:, None] * p2[None, :])
            a_index, b_index = np.nonzero(np.abs(weight) > 1e-12)
            if a_index.size == 0:
                continue
            product = np.conj(psi1[a_index]) * psi2[b_index] * q_phase
            transform = np.fft.fftn(product.reshape(-1, *shape),
                                    axes=(1, 2, 3)).reshape(len(a_index), -1)
            rho = transform[:, fft_index] * to_G[None, :]
            for c, (atom, L, M) in enumerate(crystal.channels):
                own = crystal.projector_columns(atom)
                blk = crystal.multipole_blocks[(atom, L, M)]
                Q = np.einsum("ip,ij,jp->p", np.conj(a1[own][:, a_index]),
                              blk, a2[own][:, b_index])
                rho += Q[:, None] * transforms[c][None, :]
            chi += (rho.T * weight[a_index, b_index][None, :]) @ np.conj(rho)
    return chi / (n_cells * volume), qG


def dyson(chi, qG) -> tuple:
    r"""``(inverse, W - v)``: :math:`\epsilon^{-1} = [1 - v\chi]^{-1}` in
    the symmetric form and the screening correction on the vectors ``qG``,
    with the periodic :math:`v = 4\pi/|\mathbf q+\mathbf G|^2`; a zero
    vector (the head at :math:`\mathbf q = 0`) is left out."""
    G2 = np.sum(qG * qG, axis=0)
    zero = G2 < 1e-14
    v = np.where(zero, 0.0, 4.0 * np.pi / np.where(zero, 1.0, G2))
    s = np.sqrt(v)
    inverse = np.linalg.inv(np.eye(len(v)) - s[:, None] * chi * s[None, :])
    return inverse, s[:, None] * (inverse - np.eye(len(v))) * s[None, :]


@dataclass
class ScreenedInteraction:
    r""":math:`W - v` of :func:`screened_interaction`: ``blocks`` per mesh
    point, ``(m, delta)`` with ``m`` (``(3, n)``) the vectors
    :math:`\mathbf q + \mathbf G` in units of the reciprocal vectors of
    the ``size`` supercell and ``delta`` (``(n, n)``, Hartree Bohr^3) the
    correction on them; ``head`` the :math:`\epsilon^{-1}_{00}(\mathbf q
    \to 0)` the long-wavelength limit took."""

    size: tuple
    blocks: list
    head: float | None = None


def screened_interaction(box: SupercellCoulomb, channels, targets, *,
                         exclude: bool = True, cutoff: float = None,
                         n_bands=None) -> ScreenedInteraction:
    r""":class:`ScreenedInteraction`: per mesh point :math:`\mathbf q` the
    static screened interaction of the random-phase approximation,
    :math:`W = [1 - v\chi_r]^{-1}v`, minus the bare :math:`v`, on the
    vectors :math:`\mathbf q + \mathbf G` within ``cutoff`` (Bohr^-1;
    default :data:`SCREENING_CUTOFF`).

    .. math::

        \chi_{\mathbf G\mathbf G'}(\mathbf q) = \frac{1}{N\Omega}
        \sum_{\mathbf k, a, b}\frac{F_{a\mathbf k} - F_{b\mathbf k+\mathbf q}}
        {\varepsilon_{a\mathbf k} - \varepsilon_{b\mathbf k+\mathbf q}}
        \big(1 - p_{a\mathbf k}\,p_{b\mathbf k+\mathbf q}\big)
        \rho_{ab}(\mathbf q+\mathbf G)\rho_{ab}(\mathbf q+\mathbf G')^*

    over the ``channels``' lowest ``n_bands`` Kohn-Sham states
    (:func:`polarizability`), with the transition densities' compensation
    charges.  With ``exclude`` the weights ``p`` are each state's share of
    the Bloch subspace of the channel's ``targets`` functions (constrained
    RPA, Aryasetiawan et al. 2004, in the projector-weighted form of
    Sasioglu, Friedrich and Bluegel 2011): a transition inside the target
    subspace does not screen, one partly inside screens in proportion.
    Without it nothing is excluded (RPA).  The Dyson equation takes the
    periodic Coulomb interaction :math:`4\pi/|\mathbf q+\mathbf G|^2`
    (:func:`dyson`).  At :math:`\mathbf q = 0` the divergent head is left
    out of it; the long-wavelength :math:`\epsilon^{-1}_{00}` comes from
    the same polarizability at a small :math:`\mathbf q`
    (:data:`HEAD_WAVEVECTOR`) along each Cartesian axis, averaged, each
    state keeping its band's weight, and scales the bare integrals' own
    :math:`\mathbf G = 0` term (the ``box``'s truncated kernel there); the
    wings are dropped.  Static (:math:`\omega = 0`), from the Kohn-Sham
    states, not self-consistent.
    """
    crystal = box.crystal
    shape = tuple(int(n) for n in crystal.grid.shape)
    B = rc.reciprocal_vectors(crystal.lattice)
    cutoff = SCREENING_CUTOFF if cutoff is None else float(cutoff)
    size = tuple(int(s) for s in box.size)
    N = int(np.prod(size))
    cells = np.indices(size).reshape(3, -1).T
    g_all = np.stack(np.meshgrid(*[np.fft.fftfreq(n, d=1.0 / n).astype(int)
                                   for n in shape], indexing="ij"))
    g_all = g_all.reshape(3, -1)

    def within(q):
        shifted = q[:, None] + B @ g_all
        return g_all[:, np.sum(shifted * shifted, axis=0) <= cutoff ** 2]

    states = [screening_states(c, n_bands, t if exclude else None)
              for c, t in zip(channels, targets)]
    blocks = []
    origin_block = None
    for q_cell in cells:
        q = B @ (q_cell / np.asarray(size, dtype=float))
        g = within(q)
        partner = np.ravel_multi_index(tuple(((cells + q_cell)
                                              % np.asarray(size)).T), size)
        pairs = [(c, [(s[k], s[k2]) for k, k2 in enumerate(partner)])
                 for c, s in zip(channels, states)]
        chi, qG = polarizability(crystal, pairs, q, g, N)
        _inverse, delta = dyson(chi, qG)
        if not np.any(q_cell):
            origin_block = len(blocks)
            origin_head = int(np.flatnonzero(np.all(g == 0, axis=0))[0])
        blocks.append((np.asarray(size)[:, None] * g + q_cell[:, None],
                       delta))
    # The long-wavelength head, from a small q along each axis.
    heads = []
    for axis in range(3):
        q = np.zeros(3)
        q[axis] = HEAD_WAVEVECTOR
        g = within(q)
        shifted = [screening_states(c, n_bands, shift=q,
                                    weights=[st[4] for st in s])
                   for c, s in zip(channels, states)]
        pairs = [(c, list(zip(s, t)))
                 for c, s, t in zip(channels, states, shifted)]
        chi, qG = polarizability(crystal, pairs, q, g, N)
        inverse, _delta = dyson(chi, qG)
        head = int(np.flatnonzero(np.all(g == 0, axis=0))[0])
        heads.append(float(np.real(inverse[head, head])))
    value = float(np.mean(heads))
    m, delta = blocks[origin_block]
    delta[origin_head, origin_head] = (value - 1.0) * truncated_coulomb(
        0.0, box.radius)
    return ScreenedInteraction(size=size, blocks=blocks, head=value)


# --------------------------------------------------------------------------- #
# The driver.
# --------------------------------------------------------------------------- #

def per_spin(value, n_spins: int, kind: str) -> list:
    """``value`` per spin channel: one shared by every channel, or for a
    spin-polarized crystal a pair ``(up, down)``.  ``kind`` names what a
    single value is -- ``"int"``, ``"bands"`` (a list of indices) or
    ``"windows"`` (a dict) -- which tells a pair from one value."""
    if n_spins == 2 and _is_pair(value, kind):
        return list(value)
    return [value] * n_spins


def _is_pair(value, kind: str) -> bool:
    if value is None or isinstance(value, (dict, str)):
        return False
    try:
        items = list(value)
    except TypeError:
        return False
    if len(items) != 2:
        return False
    if kind == "int":
        return all(np.ndim(x) == 0 and x is not None for x in items)
    if kind == "bands":
        return all(x is None or np.ndim(x) == 1 for x in items)
    return all(x is None or isinstance(x, dict) for x in items)


def wannier_functions(solver, size, bands, trials, *, n_functions=None,
                      windows=None, fermi_level=None,
                      replicas: str = "centers",
                      width: float = TRIAL_WIDTH, max_iter: int = 2000,
                      cite=None):
    """Maximally localized Wannier functions of the ``bands`` of a converged
    crystal: a :class:`WannierResult`, or for a spin-polarized crystal a
    :class:`SpinWannierResult` with one per spin channel, each from its own
    Bloch states.

    ``size`` is the full Gamma-centered mesh (diagonalized at the converged
    potential); ``trials`` are the :class:`Trial` orbitals of the starting
    gauge, one per function, shared by the spin channels.  With
    ``n_functions`` equal to the number of bands and no ``windows`` the
    bands are an isolated group; otherwise the ``n_functions``-dimensional
    subspace is disentangled from them within ``windows``
    (:func:`resolve_windows`, eV) first.  For a spin-polarized crystal
    ``bands``, ``n_functions`` and ``windows`` may each be a pair ``(up,
    down)``.  ``fermi_level`` (Hartree, the SCF's) gives the occupations of
    the density matrix the downfolded problem needs.  ``replicas`` places
    each hopping for the interpolation: ``"centers"`` on the replica nearest
    in the distance between the functions' centers
    (:func:`nearest_replicas`), ``"cells"`` on the supercell's Wigner-Seitz
    vectors between cell origins (:func:`wigner_seitz_vectors`, shared by
    degeneracy).  ``cite`` records the references a later
    :meth:`WannierResult.downfold` uses.
    """
    if replicas not in ("centers", "cells"):
        raise ValueError(f"replicas must be 'centers' or 'cells'; got "
                         f"{replicas!r}")
    n_spins = int(solver.n_spins)
    options = dict(fermi_level=fermi_level, replicas=replicas, width=width,
                   max_iter=max_iter, cite=cite)
    channels = [
        channel_wannier_functions(solver, spin, size, b, trials,
                                  n_functions=n, windows=w, **options)
        for spin, (b, n, w) in enumerate(zip(
            per_spin(bands, n_spins, "bands"),
            per_spin(n_functions, n_spins, "int"),
            per_spin(windows, n_spins, "windows")))]
    if n_spins == 1:
        return channels[0]
    return SpinWannierResult(channels=tuple(channels))


def channel_wannier_functions(solver, spin, size, bands, trials, *,
                              n_functions=None, windows=None,
                              fermi_level=None, replicas: str = "centers",
                              width: float = TRIAL_WIDTH,
                              max_iter: int = 2000,
                              cite=None) -> WannierResult:
    """:func:`wannier_functions` of one spin channel (``spin`` 0 for a
    restricted crystal)."""
    from .periodic_dft import occupation

    bands = tuple(int(b) for b in bands)
    size = tuple(int(n) for n in size)
    J = len(bands) if n_functions is None else int(n_functions)
    if len(trials) != J:
        raise ValueError(f"give one trial orbital per function ({J}); got "
                         f"{len(trials)}")
    if J > len(bands):
        raise ValueError(f"{J} functions from {len(bands)} bands")
    entangled = windows is not None or J != len(bands)
    crystal = solver.crystal
    lattice = np.asarray(crystal.lattice, dtype=float)       # columns, Bohr
    B = rc.reciprocal_vectors(lattice)
    b_vectors, weights, steps = neighbor_shells(B, size)
    fractional, data, vectors, energies = mesh_states(solver, size, bands,
                                                      spin)
    neighbors = neighbor_index(size, steps)
    M0 = overlaps(crystal, data, vectors, size, steps, b_vectors)
    A = trial_projections(crystal, data, vectors, fractional, trials,
                          width=width)
    record = None
    disentanglement: list = []
    if entangled:
        from .band_selection import resolve_selection, select_states

        selection = resolve_selection(windows)
        if selection is None:
            outer, frozen = resolve_windows(windows)
            S, disentanglement = disentangle(M0, neighbors, weights,
                                             energies, A, J, outer, frozen)
            record = {"outer": tuple(x * HARTREE_TO_EV for x in outer),
                      "frozen": None if frozen is None
                      else tuple(x * HARTREE_TO_EV for x in frozen)}
        else:
            kind, chosen, record = select_states(
                selection, crystal=crystal, data=data, vectors=vectors,
                energies=energies, trials=trials, n_functions=J,
                bands=bands, fermi_level=fermi_level)
            if kind == "masks":
                S, disentanglement = disentangle(M0, neighbors, weights,
                                                 energies, A, J, None,
                                                 masks=chosen)
            else:
                S = chosen
        # The Hamiltonian's eigenstates within the subspace.
        Hs = np.einsum("kam,ka,kan->kmn", np.conj(S), energies, S)
        _values, rotation = np.linalg.eigh(Hs)
        S = np.einsum("kam,kmn->kan", S, rotation)
        M_sub = np.einsum("kam,kbac,kbcn->kbmn", np.conj(S), M0,
                          S[neighbors], optimize=True)
        A_sub = np.einsum("kam,kan->kmn", np.conj(S), A)
    else:
        S = np.broadcast_to(np.eye(J, dtype=complex), (len(data), J, J))
        M_sub, A_sub = M0, A
    U_gauge, M, history, centers_b = minimize(
        M_sub, neighbors, weights, b_vectors, loewdin_unitary(A_sub),
        max_iter=max_iter, reference=np.array([t.center for t in trials]))
    U = np.einsum("kam,kmn->kan", S, U_gauge)
    U = U * real_phases(data, vectors, U, size)[None, None, :]
    centers_b, second, oI, oOD, oD = spread(M, weights, b_vectors,
                                            centers_b)
    spreads = second - np.sum(centers_b ** 2, axis=1)
    # H(R) = 1/N sum_k e^{-ik.R} U^+ diag(eps_k) U, Hartree.
    Hk = np.einsum("kam,ka,kan->kmn", np.conj(U), energies, U)
    H_bvk = bvk_blocks(fractional, Hk, size)
    if replicas == "centers":
        R, H_R = nearest_replicas(lattice, size, centers_b, H_bvk)
    else:
        R, degeneracy = wigner_seitz_vectors(lattice, size)
        H_R = np.stack([H_bvk[np.ravel_multi_index(tuple(v % np.asarray(
            size)), size)] for v in R]) / degeneracy[:, None, None]
    if fermi_level is None:
        rho_bvk = None
    else:
        # Electrons per state: two in a restricted crystal, one per channel.
        f = (2.0 / solver.n_spins) * occupation(
            (energies - fermi_level) / solver.width, solver.method)
        rho_bvk = bvk_blocks(fractional, np.einsum(
            "kam,ka,kan->kmn", np.conj(U), f, U), size)
    coefficients = np.stack([v @ u for v, u in zip(vectors, U)])
    a2 = BOHR_TO_ANGSTROM ** 2
    return WannierResult(
        centers=centers_b * BOHR_TO_ANGSTROM, spreads=spreads * a2,
        omega_invariant=oI * a2, omega_offdiagonal=oOD * a2,
        omega_diagonal=oD * a2, kpoints=fractional, unitary=U,
        lattice_vectors=R, hamiltonian_R=H_R * HARTREE_TO_EV,
        lattice=lattice.T * BOHR_TO_ANGSTROM, bands=bands,
        iterations=len(history), history=history, windows=record,
        disentanglement=[x * a2 for x in disentanglement], size=size,
        labels=tuple(t.label for t in trials),
        spin=spin if solver.n_spins == 2 else None,
        _hamiltonian_bvk=H_bvk, _density_bvk=rho_bvk, _crystal=crystal,
        _coefficients=coefficients, _solver=solver,
        _fermi_level=fermi_level, _cite=cite)


def bond_centers(atoms, tolerance: float = 0.1) -> np.ndarray:
    """Midpoints (Angstrom) of the nearest-neighbor bonds of a crystal, one
    per bond of the cell: pairs within ``1 + tolerance`` of each atom's
    nearest distance, folded into the cell and made unique modulo the
    lattice -- the trial centers of a covalent crystal's valence bands."""
    from ase.neighborlist import neighbor_list

    i, _j, d, D = neighbor_list("ijdD", atoms, 6.0)
    nearest = np.full(len(atoms), np.inf)
    np.minimum.at(nearest, i, d)
    keep = d <= (1.0 + tolerance) * nearest[i]
    midpoints = atoms.positions[i[keep]] + 0.5 * D[keep]
    cell = np.asarray(atoms.get_cell(), dtype=float)
    fractional = np.linalg.solve(cell.T, midpoints.T).T % 1.0
    unique: list = []
    for f in fractional:
        if not any(np.allclose((f - g + 0.5) % 1.0 - 0.5, 0.0, atol=1e-6)
                   for g in unique):
            unique.append(f)
    return np.asarray(unique) @ cell


def sp3_trials(atoms, tolerance: float = 0.1) -> list:
    r""":math:`sp^3` hybrids (:class:`Trial`, Bohr) on every atom, one
    toward each of its four nearest neighbors -- the trial orbitals of a
    tetrahedral crystal's bonding and antibonding bands together."""
    from ase.neighborlist import neighbor_list

    from ..units import ANGSTROM_TO_BOHR

    i, _j, d, D = neighbor_list("ijdD", atoms, 6.0)
    nearest = np.full(len(atoms), np.inf)
    np.minimum.at(nearest, i, d)
    trials = []
    for atom in range(len(atoms)):
        own = (i == atom) & (d <= (1.0 + tolerance) * nearest[atom])
        if np.count_nonzero(own) != 4:
            raise ValueError(f"atom {atom} has {np.count_nonzero(own)} "
                             "nearest neighbors, not the four of an sp3 "
                             "hybrid set")
        center = tuple(float(x) for x in
                       atoms.positions[atom] * ANGSTROM_TO_BOHR)
        for bond, direction in enumerate(D[own]):
            unit = direction / np.linalg.norm(direction)
            coefficients = np.concatenate([[0.5], 0.5 * np.sqrt(3.0) * unit])
            trials.append(Trial(
                center, angular=_hybrid(coefficients), atom=atom,
                label=f"{atoms[atom].symbol}{atom}:sp3-{bond + 1}"))
    return trials


def _hybrid(coefficients) -> tuple:
    """``angular`` of a :class:`Trial` from coefficients on (s, px, py,
    pz)."""
    return tuple((l, m, float(c)) for (l, m), c in zip(_SP, coefficients)
                 if c != 0.0)


def _orbital_name(l: int, m: int) -> str:
    for name, lm in ORBITALS.items():
        if lm == (l, m):
            return name
    return f"{'spdfghik'[l]}{m:+d}"


def angular_parts(name: str) -> list:
    """``[(label, angular), ...]``: the trial orbitals one orbital name
    stands for, ``angular`` as in :class:`Trial`.  A name is a shell
    (:data:`SHELLS`: ``"s"``, ``"p"``, ``"d"``, ``"f"``, every ``m``), one
    real harmonic (:data:`ORBITALS`), a subset of the d shell
    (:data:`SUBSETS`: ``"t2g"``, ``"eg"``) or a set of hybrids
    (:data:`HYBRIDS`: ``"sp"``, ``"sp2"``, ``"sp3"``)."""
    if not isinstance(name, str):
        raise ValueError(f"an orbital is named by a string; got {name!r}")
    if name in SHELLS:
        l = SHELLS[name]
        return [(_orbital_name(l, m), ((l, m, 1.0),))
                for m in range(-l, l + 1)]
    if name in ORBITALS:
        l, m = ORBITALS[name]
        return [(name, ((l, m, 1.0),))]
    if name in SUBSETS:
        return [part for orbital in SUBSETS[name]
                for part in angular_parts(orbital)]
    if name in HYBRIDS:
        return [(f"{name}-{i + 1}", _hybrid(row))
                for i, row in enumerate(HYBRIDS[name])]
    known = list(SHELLS) + list(ORBITALS) + list(SUBSETS) + list(HYBRIDS)
    raise ValueError(f"unknown orbital {name!r}; use one of "
                     f"{', '.join(dict.fromkeys(known))}")


def resolve_frame(frame):
    """``None`` or the frame as a tuple of three rows, the local x, y and
    z axes (Cartesian; normalized here), checked to be orthogonal."""
    if frame is None:
        return None
    axes = np.asarray(frame, dtype=float)
    if axes.shape != (3, 3):
        raise ValueError("a frame is three rows, the local x, y and z axes "
                         f"(Cartesian); got shape {axes.shape}")
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("a frame axis has zero length")
    axes = axes / lengths[:, None]
    if np.abs(axes @ axes.T - np.eye(3)).max() > 1e-6:
        raise ValueError("the frame's axes must be orthogonal: rows are the "
                         "local x, y and z axes")
    return tuple(tuple(float(x) for x in row) for row in axes)


def is_orbital_spec(guess) -> bool:
    """Whether ``guess`` names orbitals on atoms (:func:`orbital_trials`)
    rather than listing trial centers."""
    if isinstance(guess, dict):
        return True
    if isinstance(guess, (str, np.ndarray)):
        return False
    try:
        first = list(guess)[0]
    except (TypeError, IndexError):
        return False
    if isinstance(first, (str, np.ndarray)) or not isinstance(first,
                                                               (tuple, list)):
        return False
    return len(first) in (2, 3) and (
        isinstance(first[0], (str, int, np.integer))
        or _is_position(first[0])) \
        and (isinstance(first[1], str)
             or (isinstance(first[1], (list, tuple)) and len(first[1]) > 0
                 and all(isinstance(x, str) for x in first[1])))


def _spec_entries(spec) -> list:
    """``[(atom, names, frame), ...]`` of an orbital spec."""
    if isinstance(spec, dict):
        items = [(atom, orbitals, None) for atom, orbitals in spec.items()]
    else:
        items = []
        for entry in spec:
            if not isinstance(entry, (tuple, list)) or len(entry) not in (2,
                                                                          3):
                raise ValueError(
                    "each guess entry is (atom, orbitals) or (atom, "
                    f"orbitals, frame); got {entry!r}")
            items.append((entry[0], entry[1],
                          entry[2] if len(entry) == 3 else None))
    out = []
    for atom, orbitals, frame in items:
        names = [orbitals] if isinstance(orbitals, str) else list(orbitals)
        out.append((atom, names, resolve_frame(frame)))
    return out


def _is_position(atom) -> bool:
    if isinstance(atom, (str, bytes)):
        return False
    try:
        values = np.asarray(atom, dtype=float)
    except (TypeError, ValueError):
        return False
    return values.shape == (3,)


def _sites(atoms, atom) -> list:
    """``[(index or None, position in Angstrom, label prefix), ...]`` of a
    spec entry's site: every atom of a chemical symbol, one atom by
    index, or a Cartesian position (Angstrom)."""
    if isinstance(atom, (int, np.integer)) and not isinstance(atom, bool):
        index = int(atom)
        if not 0 <= index < len(atoms):
            raise ValueError(f"atom {index} of {len(atoms)}")
        return [(index, atoms.positions[index],
                 f"{atoms[index].symbol}{index}")]
    if isinstance(atom, str):
        indices = [i for i, s in enumerate(atoms.get_chemical_symbols())
                   if s == atom]
        if not indices:
            raise ValueError(f"no {atom!r} atom in the cell "
                             f"({atoms.get_chemical_formula()})")
        return [(i, atoms.positions[i], f"{atom}{i}") for i in indices]
    if _is_position(atom):
        position = np.asarray(atom, dtype=float)
        return [(None, position, "(" + ", ".join(f"{x:.3f}" for x in
                                                 position) + ")")]
    raise ValueError(f"a site is a chemical symbol, an atom index or a "
                     f"position (Angstrom); got {atom!r}")


def basis_radials(crystal, atom: int, ls) -> tuple:
    """``((l, R), ...)``: the radial function of the basis' own orbital of
    each angular momentum in ``ls`` on ``atom`` -- its first zeta, a
    polarization function only where there is no other."""
    out = []
    for l in sorted(set(int(x) for x in ls)):
        candidates = [f for f, a in zip(crystal.basis,
                                        crystal.atom_of_orbital)
                      if a == atom and int(f.l) == l]
        if not candidates:
            raise ValueError(
                f"the basis has no l = {l} orbital on atom {atom}: use "
                "Gaussian trial orbitals (trial_radial='gaussian')")
        candidates.sort(key=lambda f: (bool(getattr(f, "polarization",
                                                    False)),
                                       int(getattr(f, "zeta", 1))))
        out.append((l, candidates[0].radial))
    return tuple(out)


def orbital_trials(atoms, spec, crystal=None,
                   radial: str = "gaussian") -> list:
    r"""Trial orbitals (:class:`Trial`, Bohr) named by atom and orbital.

    ``spec`` is a dict ``{atom: orbitals}`` or a list of ``(atom,
    orbitals)`` or ``(atom, orbitals, frame)``: ``atom`` a chemical symbol
    (every atom of the element, in index order), an index, or in a list a
    Cartesian position (Angstrom) that need not hold an atom; ``orbitals``
    one name or a list of them (:func:`angular_parts`), e.g. ``"d"``,
    ``"t2g"``, ``["s", "d"]``, ``"sp3"``; ``frame`` the local axes,
    three rows x, y, z (Cartesian; default the Cartesian axes), in which
    the harmonics, the :math:`t_{2g}`/:math:`e_g` split and the hybrids'
    directions are defined -- an octahedron's axes when it is rotated.
    ``radial`` is ``"gaussian"`` (:func:`gaussian_radial`) or ``"basis"``:
    on an atom the radial functions of its own basis orbitals
    (:func:`basis_radials`, which needs the ``crystal``), at a position
    without one Gaussians still."""
    trials = _orbital_trials(atoms, spec, crystal, radial)
    if radial == "basis" and all(t.atom is None for t in trials):
        raise ValueError("trial_radial='basis' takes the radial functions "
                         "of the atoms' own basis orbitals: no trial "
                         "orbital sits on an atom")
    return trials


def _orbital_trials(atoms, spec, crystal, radial) -> list:
    from ..units import ANGSTROM_TO_BOHR

    if radial not in ("gaussian", "basis"):
        raise ValueError(f"trial_radial must be 'gaussian' or 'basis'; got "
                         f"{radial!r}")
    trials = []
    for atom, names, frame in _spec_entries(spec):
        parts = [part for name in names for part in angular_parts(name)]
        for index, position, prefix in _sites(atoms, atom):
            center = tuple(float(x) for x in position * ANGSTROM_TO_BOHR)
            for label, angular in parts:
                functions = None
                if radial == "basis" and index is not None:
                    functions = basis_radials(crystal, index,
                                              [l for l, _m, _c in angular])
                trials.append(Trial(center, angular=angular, frame=frame,
                                    radial=functions, atom=index,
                                    label=f"{prefix}:{label}"))
    if not trials:
        raise ValueError("the guess names no trial orbitals")
    return trials


def resolve_guess(guess, atoms, crystal=None,
                  radial: str = "gaussian") -> list:
    """The :class:`Trial` orbitals of a ``guess``: ``"bonds"`` (s
    Gaussians on the bond midpoints, :func:`bond_centers`), ``"sp3"``
    (hybrids toward each atom's four neighbors, :func:`sp3_trials`), orbitals
    on atoms (:func:`orbital_trials`) or trial centers (Angstrom) for s
    Gaussians.  ``radial="basis"`` takes the radial functions of the atoms'
    own basis orbitals for the trials on atoms (:func:`orbital_trials`);
    ``"bonds"`` and centers have none."""
    from ..units import ANGSTROM_TO_BOHR

    if radial not in ("gaussian", "basis"):
        raise ValueError(f"trial_radial must be 'gaussian' or 'basis'; got "
                         f"{radial!r}")
    if is_orbital_spec(guess):
        return orbital_trials(atoms, guess, crystal, radial)
    if isinstance(guess, str):
        if guess == "sp3":
            trials = sp3_trials(atoms)
        elif guess == "bonds":
            trials = gaussian_trials(bond_centers(atoms) * ANGSTROM_TO_BOHR)
        else:
            raise ValueError(
                f"unknown guess {guess!r}; use 'bonds', 'sp3', orbitals on "
                "atoms such as [('Cu', 'd')] or {'V': 't2g'}, or trial "
                "centers (Angstrom)")
    else:
        trials = gaussian_trials(np.asarray(guess, dtype=float)
                                 * ANGSTROM_TO_BOHR)
    if radial == "basis":
        if any(t.atom is None for t in trials):
            raise ValueError("trial_radial='basis' takes the radial "
                             "functions of the atoms' own basis orbitals: "
                             f"the {guess!r} guess puts trials off the atoms")
        trials = [Trial(t.center, angular=t.angular, frame=t.frame,
                        radial=basis_radials(crystal, t.atom,
                                             [l for l, _m, _c in t.angular]),
                        atom=t.atom, label=t.label) for t in trials]
    return trials
