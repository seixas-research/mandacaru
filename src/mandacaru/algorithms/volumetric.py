# -*- coding: utf-8 -*-
# file: algorithms/volumetric.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""One-particle pictures of a many-body variational state, on a real-space grid.

What can and cannot be written to a file
----------------------------------------
The state an ADAPT-VQE or VQE run converges to is a **many-electron**
wavefunction :math:`|\Psi(\theta)\rangle`: a function of :math:`3N` coordinates
(and :math:`N` spins).  No volumetric file holds such a thing, and calling one
"the wavefunction" would be wrong.  What *is* a function of three coordinates,
and follows from :math:`|\Psi\rangle` exactly -- no further approximation --
are its **one-particle reductions**, all built from the one-particle reduced
density matrix

.. math::

    \gamma_{pq} = \langle\Psi|\, a^\dagger_p a_q \,|\Psi\rangle

(:mod:`mandacaru.algorithms.rdm`, the same object the nuclear forces contract),
the molecular orbitals :math:`\phi_p = \sum_\mu A_{\mu p}\chi_\mu` with
:math:`A = S^{-1/2} V`, and the basis functions :math:`\chi_\mu` sampled on the
grid.  The electron density is

.. math::

    n(\mathbf r) = \sum_{pq}\gamma_{pq}\,
                   \phi_p^*(\mathbf r)\,\phi_q(\mathbf r) ,

which follows from the density operator
:math:`\hat n(\mathbf r) = \hat\psi^\dagger(\mathbf r)\hat\psi(\mathbf r)`.

Quantities
----------
``density``
    The spin-summed electron density :math:`n_\alpha + n_\beta`, in
    e/Bohr^3.  It integrates to the electron count (see the PAW-LCAO note below).
``alpha_density`` / ``beta_density``
    The two spin channels separately.  The spin-orbitals are spin-blocked
    (:math:`P = p + \sigma M`, alpha first), so these are the two diagonal
    blocks of :math:`\gamma`.
``spin_density``
    :math:`n_\alpha - n_\beta`.  It integrates to :math:`N_\alpha - N_\beta`
    and is zero everywhere for a closed shell -- the picture to draw for a
    doublet or a triplet.
``difference_density``
    :math:`n - n_{\rm HF}`, the density the *correlation* moved.  The
    reference is the Hartree-Fock determinant the ansatz starts from (the
    frozen core plus the lowest active orbitals of each spin), evaluated in
    the same orbitals, so the difference is exactly what the variational
    optimization did.  It integrates to zero.
``natural_orbital``
    Eigenvector :math:`i` of the spin-summed :math:`\gamma`, ordered by
    **descending occupation**.  For a single determinant the occupations are
    exactly 2 and 0; the departure from those values *is* the correlation, and
    it is the most direct picture of it there is.
``molecular_orbital``
    Orbital :math:`i` of the reference (RHF, or UHF natural-orbital) basis the
    Hamiltonian was built in, ordered by energy.

Frozen core, tapered registers, sectors
---------------------------------------
The RDM comes from whatever representation the run used -- a plain Jordan-Wigner
register, a parity-tapered one, or a particle-number sector -- through
:meth:`~mandacaru.algorithms.calculator.Mandacaru._state_rdms`, the same route
the forces take.  A **frozen core** is refilled here (its spatial orbitals are
doubly occupied and inert), so an all-electron density is the *all-electron*
density and integrates to the full electron count.

The pseudopotential caveat
--------------------------
With a PAW-LCAO / UPAW-LCAO / ONCVPSP / NCPP basis the orbitals are the **smooth pseudo**
valence orbitals, so what is written is the *pseudo valence density*: the core
is absent by construction, and inside the augmentation spheres the smooth
density is not the physical one.  For PAW-LCAO the overlap itself is augmented,
:math:`S = \tilde S + C q C^\dagger`, so the smooth density does **not**
integrate to the valence electron count:

.. math::

    \int \tilde n\,d^3r \;+\; \underbrace{\sum_{pq}
        \gamma_{pq}\,(A^\dagger C q C^\dagger A)_{pq}}_{\text{augmentation}}
    \;=\; N_{\rm valence}.

Both numbers are reported (:attr:`VolumetricField.integral` and
:attr:`VolumetricField.augmentation_charge`) and the file's comment line says
which one it holds.  Reconstructing the all-electron density inside the spheres
would need the one-center expansion summed on a radial grid per atom; the
radial machinery exists (:func:`mandacaru.pseudopotentials.paw.reconstruct_ae`)
but the three-dimensional assembly does not, so it is deliberately left out
rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Largest ``max|Im| / max|Re|`` an orbital may have and still be written as a
#: real function.  The molecular path rotates the orbitals to conjugation-real
#: form (:func:`~mandacaru.core.hamiltonian.conjugation_real_orbitals`), so a
#: residual this size is arithmetic noise; anything larger is a genuinely
#: complex orbital and the user has to say what to write.
REAL_ORBITAL_TOLERANCE = 1e-6

#: Largest relative imaginary part a *density* may carry.  A density is real by
#: construction (``gamma`` is Hermitian), so this is an assertion, not a choice.
REAL_DENSITY_TOLERANCE = 1e-8

#: Largest relative ``max|gamma - gamma^H|`` a state's RDM may show.  What makes
#: a density real is the Hermiticity of the density matrix, so *that* is the
#: property worth checking -- once, on the way in, before anything symmetrizes
#: it and hides the problem.
RDM_HERMITICITY_TOLERANCE = 1e-8

#: The quantities :func:`volumetric_field` knows how to build.
QUANTITIES = ("density", "alpha_density", "beta_density", "spin_density",
              "difference_density", "natural_orbital", "molecular_orbital")

#: Quantities that take an orbital ``index``.
ORBITAL_QUANTITIES = ("natural_orbital", "molecular_orbital")

#: How the complex amplitude of an orbital becomes a real number in the file.
COMPONENTS = ("auto", "real", "imag", "modulus")

#: Unit strings written into the file comment.
DENSITY_UNITS = "e/Bohr^3"
ORBITAL_UNITS = "Bohr^-3/2"


# --------------------------------------------------------------------------- #
# Reduced density matrices, in the shape a one-particle picture needs.
# --------------------------------------------------------------------------- #

def spin_resolved_rdm(gamma, n_spatial_orbitals: int, frozen=(), active=None):
    r"""``(D_alpha, D_beta)`` spatial one-RDMs, with the frozen core refilled.

    ``gamma`` is the **active-space** spin-orbital RDM the solver's register
    produced, spin-blocked (:math:`P = p + \sigma M_{\rm act}`).  The frozen
    spatial orbitals are doubly occupied and inert, so each spin channel gains
    exactly one electron in each of them -- the one-body half of
    :func:`~mandacaru.algorithms.rdm.expand_frozen_core`, which the forces use
    for the same reason.

    ``active`` names the spatial orbitals the register carried.  ``None`` means
    the complement of ``frozen``, which is what a frozen core alone leaves.  It
    has to be given explicitly once the **virtual** space is truncated as well
    (:mod:`mandacaru.algorithms.active_space`): a deleted virtual orbital is
    neither frozen nor active, so the complement is no longer the active set and
    the RDM's rows would be laid out against the wrong orbitals.  Deleted
    orbitals are empty, so they simply stay zero here.

    Returns two ``(M, M)`` Hermitian matrices over the **full** set of spatial
    molecular orbitals.
    """
    M = int(n_spatial_orbitals)
    gamma = np.asarray(gamma, dtype=complex)
    core = sorted({int(f) for f in frozen})
    active = ([p for p in range(M) if p not in set(core)] if active is None
              else sorted({int(p) for p in active}))
    _check_partition(M, core, active)
    n_act = len(active)
    if gamma.shape != (2 * n_act, 2 * n_act):
        raise ValueError(
            f"expected a {2 * n_act}x{2 * n_act} spin-orbital RDM for "
            f"{n_act} active orbitals ({len(core)} frozen, "
            f"{M - n_act - len(core)} deleted of {M}), got {gamma.shape}")
    scale = float(np.abs(gamma).max()) or 1.0
    asymmetry = float(np.abs(gamma - gamma.conj().T).max()) / scale
    if asymmetry > RDM_HERMITICITY_TOLERANCE:
        raise ValueError(
            f"the one-particle density matrix is not Hermitian (relative "
            f"|gamma - gamma^H| = {asymmetry:.2e}); the density it produces "
            f"would not be real, so the state it came from is wrong rather "
            f"than the picture")
    blocks = []
    for spin in (0, 1):
        full = np.zeros((M, M), dtype=complex)
        rows = slice(spin * n_act, (spin + 1) * n_act)
        full[np.ix_(active, active)] = gamma[rows, rows]
        for c in core:
            full[c, c] += 1.0
        blocks.append(0.5 * (full + full.conj().T))
    return blocks[0], blocks[1]


def _check_partition(M: int, frozen, active) -> None:
    """Refuse a frozen / active pair that does not describe one orbital set."""
    for label, indices in (("frozen", frozen), ("active", active)):
        for p in indices:
            if not 0 <= p < M:
                raise ValueError(
                    f"{label} spatial orbital {p} is out of range [0, {M})")
    overlap = sorted(set(frozen) & set(active))
    if overlap:
        raise ValueError(
            f"spatial orbital(s) {overlap} are listed as both frozen and "
            f"active; the density would count their electrons twice")


def reference_occupations(n_spatial_orbitals: int, num_particles, frozen=(),
                          active=None):
    r"""``(occ_alpha, occ_beta)`` of the Hartree-Fock reference determinant.

    The ansatz starts from the determinant that fills the frozen core plus the
    lowest ``num_particles`` *active* orbitals of each spin -- the same
    occupation :class:`~mandacaru.circuits.adapt_ansatz.AdaptAnsatz` prepares.
    ``num_particles`` is the **active** ``(n_alpha, n_beta)`` the solver
    reports.  Each array holds 0.0 or 1.0 per spatial orbital.
    """
    M = int(n_spatial_orbitals)
    core = sorted({int(f) for f in frozen})
    active = ([p for p in range(M) if p not in set(core)] if active is None
              else sorted({int(p) for p in active}))
    _check_partition(M, core, active)
    out = []
    for count in (int(num_particles[0]), int(num_particles[1])):
        if not 0 <= count <= len(active):
            raise ValueError(f"{count} electrons of one spin do not fit in "
                             f"{len(active)} active orbitals")
        occupation = np.zeros(M)
        occupation[core] = 1.0
        occupation[active[:count]] = 1.0
        out.append(occupation)
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Molecular orbitals on the grid.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NaturalOrbitals:
    """Eigen-decomposition of the spin-summed one-particle density matrix.

    Attributes
    ----------
    occupations : (M,) ndarray
        Eigenvalues in **descending** order, each in ``[0, 2]`` and summing to
        the electron count.  A single determinant gives exactly 2 and 0.
    coefficients : (M, M) ndarray
        Column ``i`` expands natural orbital ``i`` in the **atomic-orbital**
        basis, :math:`\\psi_i = \\sum_\\mu c_{\\mu i}\\chi_\\mu`.  Orthonormal
        under the (augmented, for PAW-LCAO) overlap.
    mo_coefficients : (M, M) ndarray
        The same orbitals in the **molecular-orbital** basis the Hamiltonian
        was built in -- which is where the departure from a single determinant
        is easiest to read.
    """

    occupations: np.ndarray
    coefficients: np.ndarray
    mo_coefficients: np.ndarray

    @property
    def n_electrons(self) -> float:
        """``sum(occupations)`` -- the electron count the density integrates to."""
        return float(np.sum(self.occupations))


class OrbitalExpansion:
    r"""Molecular orbitals of one calculation, evaluated on a real-space grid.

    Holds the two things every quantity in this module needs: the
    atomic-orbital coefficients of the molecular orbitals,
    :math:`A = S^{-1/2}V` (Loewdin orthogonalization followed by the SCF /
    UHF-natural-orbital rotation, exactly as
    :meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.molecular_hamiltonian`
    built them), and the sampled basis functions :math:`\chi_\mu(\mathbf r)`.

    On the calculation's own grid the samples are **reused** from the integral
    engine, which already holds them: nothing is re-evaluated and no
    ``G x M x M`` intermediate is ever formed.  Pass ``grid=`` to re-evaluate
    the basis on a different (coarser or finer) grid instead -- honest
    resampling of the functions, never interpolation of the data.
    """

    def __init__(self, integrals, grid=None):
        if getattr(integrals, "mo_coefficients", None) is None:
            raise RuntimeError(
                "the integrals carry no molecular orbitals; a one-particle "
                "picture needs the basis the Hamiltonian was built in "
                "(molecular_hamiltonian(mo_basis=True))")
        self.integrals = integrals
        self.grid = integrals.grid if grid is None else grid
        #: ``(M, M)`` atomic-orbital coefficients of the molecular orbitals.
        self.mo = integrals._lowdin_x() @ np.asarray(
            integrals.mo_coefficients, dtype=complex)
        if grid is None or grid is integrals.grid:
            # The engine sampled every function once when it was built; that is
            # the same (M, G) array the integrals were computed from.
            self.samples = integrals._engine._psi
        else:
            self.samples = np.ascontiguousarray(
                np.stack([fn.sample(self.grid) for fn in integrals.basis]),
                dtype=np.complex128)

    def orbital(self, coefficients) -> np.ndarray:
        r""":math:`\sum_\mu c_\mu \chi_\mu(\mathbf r)` for AO coefficients ``c``."""
        return np.asarray(coefficients, dtype=complex) @ self.samples

    def natural_orbitals(self, D) -> NaturalOrbitals:
        r"""Diagonalize a spatial one-RDM into occupations and orbitals.

        With :math:`D = U\,{\rm diag}(w)\,U^\dagger` the density is
        :math:`n(\mathbf r) = \sum_i w_i |\psi_i(\mathbf r)|^2` for
        :math:`\psi_i = \sum_q \overline{U_{qi}}\,\phi_q` -- note the
        conjugate, which comes from :math:`n = \sum_{pq}\gamma_{pq}
        \phi_p^*\phi_q` having the star on the *first* index.  The
        :math:`\psi_i` are orthonormal, so the occupations sum to the electron
        count.
        """
        D = np.asarray(D, dtype=complex)
        w, U = np.linalg.eigh(0.5 * (D + D.conj().T))
        order = np.argsort(w)[::-1]
        mo_coefficients = np.conj(U[:, order])
        return NaturalOrbitals(occupations=np.real(w[order]),
                               coefficients=self.mo @ mo_coefficients,
                               mo_coefficients=mo_coefficients)

    def density(self, D):
        r"""``(values, augmentation)`` of the density of a spatial one-RDM.

        The density is accumulated one natural orbital at a time,
        :math:`\sum_i w_i|\psi_i|^2`, so the working set is a single grid-sized
        array whatever the basis size.

        ``augmentation`` is the charge the *smooth* density on the grid does
        not carry: :math:`\sum_i w_i \langle\psi_i|(S - \tilde S)|\psi_i\rangle`
        with :math:`S - \tilde S = C q C^\dagger` the PAW augmentation.  It is
        exactly zero for every norm-conserving and all-electron basis.
        """
        orbitals = self.natural_orbitals(D)
        values = np.zeros(self.samples.shape[1])
        for w, c in zip(orbitals.occupations, orbitals.coefficients.T):
            if w <= 0.0:
                # A one-particle density matrix is positive semi-definite, so a
                # non-positive occupation is arithmetic noise (order 1e-17);
                # adding it would make the density negative somewhere.
                continue
            values += w * np.abs(self.orbital(c)) ** 2
        return values, self._augmentation(orbitals)

    def _augmentation(self, orbitals: NaturalOrbitals) -> float:
        """Charge held by the PAW augmentation rather than by the grid."""
        integrals = self.integrals
        if getattr(integrals, "nonlocal_overlap", None) is None:
            return 0.0
        correction = integrals.overlap() - integrals.bare_overlap()
        c = orbitals.coefficients
        per_orbital = np.real(np.einsum("mi,mn,ni->i", np.conj(c), correction, c))
        return float(np.sum(orbitals.occupations * per_orbital))


# --------------------------------------------------------------------------- #
# The field, and the file it is written to.
# --------------------------------------------------------------------------- #

@dataclass
class VolumetricField:
    """A real scalar field on the calculation's grid, ready to be written.

    ``integral`` is :math:`\\int f\\,d^3r` over the grid and ``norm`` is
    :math:`\\int |f|^2 d^3r`; for a density the first is the electron count it
    holds and for an orbital the second is its norm on this grid.
    """

    quantity: str
    data: np.ndarray                     # (nx, ny, nz), real
    grid: object                         # the Grid it was sampled on
    numbers: np.ndarray                  # atomic numbers (the element drawn)
    positions: np.ndarray                # nuclear positions, Bohr
    charges: np.ndarray                  # nuclear charges of the Hamiltonian
    units: str
    integral: float
    norm: float
    index: int | None = None
    occupation: float | None = None
    n_electrons: float | None = None
    augmentation_charge: float = 0.0
    component: str = "real"
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: Where :meth:`write` last put this field (``None`` until it is
    #: written).  Named ``output_path`` rather than ``path`` so the
    #: documentation's cross-references stay unambiguous.
    output_path: str | None = None

    @property
    def origin(self) -> np.ndarray:
        """Position of node ``(0, 0, 0)`` in Bohr."""
        return np.array([self.grid.X.flat[0], self.grid.Y.flat[0],
                         self.grid.Z.flat[0]], dtype=float)

    def label(self) -> str:
        """Human-readable name of the quantity, with its orbital index."""
        name = self.quantity.replace("_", " ")
        if self.index is None:
            return name
        return f"{name} #{self.index}"

    def comment(self) -> str:
        """The cube file's first line: what this is, and what it integrates to."""
        from ..version import __version__

        parts = [f"Mandacaru {__version__}",
                 f"{self.label()} [{self.units}]"]
        if self.index is None:
            parts.append(f"integral = {self.integral:.6f}")
        else:
            # For an orbital the interesting scalar is its norm on this grid;
            # the integral of a wavefunction amplitude means nothing.
            parts.append(f"norm = {self.norm:.6f}")
        if self.occupation is not None:
            parts.append(f"occupation = {self.occupation:.6f}")
        if self.augmentation_charge:
            parts.append(f"augmentation = {self.augmentation_charge:+.6f}")
        parts.extend(self.notes)
        return " | ".join(parts)

    def summary(self) -> str:
        """Multi-line report of the field, for a script to print."""
        lines = [f"{self.label()} [{self.units}]",
                 f"  grid            : {self.grid.shape[0]}x{self.grid.shape[1]}"
                 f"x{self.grid.shape[2]} nodes, dV = {self.grid.dV:.6f} Bohr^3",
                 f"  integral        : {self.integral:+.6f}",
                 f"  norm            : {self.norm:.6f}",
                 f"  range           : {float(self.data.min()):+.6e} .. "
                 f"{float(self.data.max()):+.6e}"]
        if self.occupation is not None:
            lines.append(f"  occupation      : {self.occupation:.6f}")
        if self.n_electrons is not None:
            lines.append(f"  electrons       : {self.n_electrons:.6f}")
        if self.augmentation_charge:
            lines.append(f"  augmentation    : {self.augmentation_charge:+.6f} "
                         f"(not on the grid)")
        return "\n".join(lines)

    def write(self, path, format=None, comment=None) -> str:
        """Write the field to ``path`` and return the path.

        The format follows the extension (``.cube``, ``.xsf``) unless
        ``format=`` names one.  The XSF file is written as a *molecule*
        (an ``ATOMS`` block): the grid is a real-space integration box
        centered on the molecule, not a periodic lattice, so declaring it a
        crystal would be a claim the calculation does not make.
        """
        from ..utils.cube import write_volumetric

        self.output_path = write_volumetric(
            path, self.data, self.origin, self.grid.step, self.numbers,
            self.positions, comment=self.comment() if comment is None
            else str(comment), charges=self.charges, format=format)
        return self.output_path


def _real_part(values, component: str, what: str):
    """Turn a complex field into the real one that goes in the file.

    A global phase is removed first (an orbital is defined only up to one), so
    the ``"auto"`` test asks the right question: whether the orbital *can* be
    made real, not whether it happens to have been.
    """
    values = np.asarray(values)
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, "
                         f"got {component!r}")
    if not np.iscomplexobj(values):
        return np.asarray(values, dtype=float), "real"
    largest = values.flat[int(np.argmax(np.abs(values)))]
    if abs(largest) > 0:
        values = values * np.exp(-1j * np.angle(largest))
    if component == "modulus":
        return np.abs(values), "modulus"
    if component == "real":
        return np.real(values), "real"
    if component == "imag":
        return np.imag(values), "imag"
    scale = float(np.abs(np.real(values)).max()) or 1.0
    residual = float(np.abs(np.imag(values)).max()) / scale
    if residual > REAL_ORBITAL_TOLERANCE:
        raise ValueError(
            f"the {what} is genuinely complex (max |Im| / max |Re| = "
            f"{residual:.2e} after removing the global phase), so writing its "
            f"real part would discard part of it.  Pass component='modulus' "
            f"(the amplitude), 'real' or 'imag' to say which real function to "
            f"write.  Degenerate orbitals of a shell with l > 0 are the usual "
            f"cause: the complex spherical harmonics mix them into a pair that "
            f"no global phase makes real.")
    return np.real(values), "real"


def _checked_density(values, what: str) -> np.ndarray:
    """A density is real by construction -- verify that, do not assume it."""
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        return np.asarray(values, dtype=float)
    scale = float(np.abs(np.real(values)).max()) or 1.0
    residual = float(np.abs(np.imag(values)).max()) / scale
    if residual > REAL_DENSITY_TOLERANCE:
        raise RuntimeError(
            f"the {what} came out complex (max |Im| / max |Re| = "
            f"{residual:.2e}); it is real by construction for a Hermitian "
            f"density matrix, so this is a bug rather than a choice")
    return np.real(values)


# --------------------------------------------------------------------------- #
# The public builder.
# --------------------------------------------------------------------------- #

def volumetric_field(integrals, gamma, *, quantity: str = "density",
                     index: int = 0, frozen=(), num_particles=None,
                     n_spatial_orbitals=None, component: str = "auto",
                     grid=None, numbers=None, active=None) -> VolumetricField:
    """Build one real-space quantity of a converged variational state.

    This is the solver-free entry point: give it the integral object the
    Hamiltonian was built from and the spin-orbital RDM of the state, and it
    returns a :class:`VolumetricField`.
    :meth:`mandacaru.algorithms.calculator.Mandacaru.write_cube` is the
    user-facing wrapper that supplies both from a finished run.

    Parameters
    ----------
    integrals : MolecularIntegrals
        The live integral object of the run -- its basis functions, its grid
        and its molecular orbitals.  The *actual* basis of the calculation is
        used (a confined, Fourier-filtered PAW-LCAO orbital included); nothing is
        rebuilt from options.
    gamma : (2*M_act, 2*M_act) array
        The active-space spin-orbital one-RDM, spin-blocked with the alpha
        block first.
    quantity : str
        One of :data:`QUANTITIES`.
    index : int
        Which orbital, for ``natural_orbital`` (0 = most occupied) and
        ``molecular_orbital`` (0 = lowest).
    frozen : sequence of int
        Frozen spatial-orbital indices; refilled into the density.
    active : sequence of int, optional
        Spatial orbitals the register carried.  Needed only when the virtual
        space was truncated as well -- see :func:`spin_resolved_rdm`.
    num_particles : (int, int)
        Active ``(n_alpha, n_beta)``.  Required for ``difference_density``,
        which needs the reference determinant.
    n_spatial_orbitals : int, optional
        Total spatial orbitals; defaults to the size of the basis.
    component : {"auto", "real", "imag", "modulus"}
        How a complex orbital becomes the real number a file holds.
    grid : Grid, optional
        Sample on this grid instead of the calculation's own.  The basis
        functions are **re-evaluated** on it (never interpolated), which is the
        honest way to get a smoother picture.  On the calculation's own grid
        the electron count comes out exact to machine precision, because that
        same quadrature is what defines the overlap the orbitals are
        orthonormal under; on any other grid the integral instead shows the
        difference between the two quadratures, which is a real measure of how
        well ``h`` resolved the basis.
    numbers : sequence of int, optional
        Atomic numbers to write for the nuclei.  Defaults to the charges the
        Hamiltonian carries, which for a pseudopotential run are the *valence*
        charges -- so pass the real ones (``atoms.get_atomic_numbers()``) to
        have a viewer draw the right element.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}; use one of {QUANTITIES}")
    expansion = OrbitalExpansion(integrals, grid=grid)
    M = int(len(integrals.basis) if n_spatial_orbitals is None
            else n_spatial_orbitals)
    D_alpha, D_beta = spin_resolved_rdm(gamma, M, frozen, active)

    notes: list[str] = []
    occupation = None
    orbital_index = None
    n_electrons = None
    augmentation = 0.0

    if quantity in ORBITAL_QUANTITIES:
        index = int(index)
        if not 0 <= index < M:
            raise ValueError(f"orbital index {index} is out of range: the "
                             f"calculation has {M} spatial orbitals")
        orbital_index = index
        if quantity == "natural_orbital":
            orbitals = expansion.natural_orbitals(D_alpha + D_beta)
            occupation = float(orbitals.occupations[index])
            coefficients = orbitals.coefficients[:, index]
            notes.append("natural orbitals are ordered by descending occupation")
        else:
            coefficients = expansion.mo[:, index]
            notes.append("reference (Hartree-Fock) orbitals, ordered by energy")
        values, component = _real_part(expansion.orbital(coefficients),
                                       component, f"{quantity} {index}")
        units = ORBITAL_UNITS
        if component == "modulus":
            notes.append("modulus of a complex orbital")
        elif component == "imag":
            notes.append("imaginary part of a complex orbital")
    else:
        if quantity == "alpha_density":
            values, augmentation = expansion.density(D_alpha)
            n_electrons = float(np.real(np.trace(D_alpha)))
        elif quantity == "beta_density":
            values, augmentation = expansion.density(D_beta)
            n_electrons = float(np.real(np.trace(D_beta)))
        elif quantity == "spin_density":
            alpha, aug_a = expansion.density(D_alpha)
            beta, aug_b = expansion.density(D_beta)
            values, augmentation = alpha - beta, aug_a - aug_b
            n_electrons = float(np.real(np.trace(D_alpha) - np.trace(D_beta)))
        elif quantity == "difference_density":
            if num_particles is None:
                raise ValueError(
                    "difference_density needs num_particles to build the "
                    "Hartree-Fock reference determinant it subtracts")
            occ_a, occ_b = reference_occupations(M, num_particles, frozen,
                                                active)
            reference = np.diag(occ_a + occ_b).astype(complex)
            total, aug_total = expansion.density(D_alpha + D_beta)
            hartree_fock, aug_ref = expansion.density(reference)
            values, augmentation = total - hartree_fock, aug_total - aug_ref
            n_electrons = float(np.real(np.trace(D_alpha + D_beta))
                                - np.sum(occ_a + occ_b))
            notes.append("correlated density minus the Hartree-Fock reference")
        else:                                                   # "density"
            values, augmentation = expansion.density(D_alpha + D_beta)
            n_electrons = float(np.real(np.trace(D_alpha + D_beta)))
        values = _checked_density(values, quantity)
        units = DENSITY_UNITS
        component = "real"

    if augmentation:
        notes.append("smooth pseudo valence density: the augmentation charge "
                     "above is NOT in this file")

    grid_used = expansion.grid
    data = np.ascontiguousarray(np.real(values).reshape(grid_used.shape))
    charges = np.array([float(Z) for Z, _c in integrals.nuclei])
    positions = np.array([np.asarray(c, dtype=float)
                          for _z, c in integrals._potentials.nuclei])
    if numbers is None:
        # Falling back on the Hamiltonian's charges draws the *valence* element
        # for a pseudopotential run (oxygen as carbon); the caller should pass
        # the real atomic numbers whenever it has them.
        numbers = np.rint(charges).astype(int)
    else:
        numbers = np.asarray(numbers, dtype=int).reshape(-1)
        if len(numbers) != len(charges):
            raise ValueError(f"{len(numbers)} atomic numbers were given for "
                             f"{len(charges)} nuclei")
    return VolumetricField(
        quantity=quantity, data=data, grid=grid_used, numbers=numbers,
        positions=positions, charges=charges, units=units,
        integral=float(data.sum() * grid_used.dV),
        norm=float(np.sum(data * data) * grid_used.dV),
        index=orbital_index, occupation=occupation, n_electrons=n_electrons,
        augmentation_charge=float(augmentation), component=component,
        notes=tuple(notes))


def state_natural_orbitals(integrals, gamma, *, frozen=(),
                           n_spatial_orbitals=None, grid=None,
                           active=None) -> NaturalOrbitals:
    """Natural orbitals and occupations of a state's one-particle RDM.

    The same decomposition :func:`volumetric_field` draws, returned as data:
    occupations in ``[0, 2]`` summing to the electron count, with the orbitals
    in both the atomic- and the molecular-orbital basis.
    """
    M = int(len(integrals.basis) if n_spatial_orbitals is None
            else n_spatial_orbitals)
    D_alpha, D_beta = spin_resolved_rdm(gamma, M, frozen, active)
    return OrbitalExpansion(integrals, grid=grid).natural_orbitals(
        D_alpha + D_beta)
