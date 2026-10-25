# -*- coding: utf-8 -*-
# file: algorithms/forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Nuclear gradients: Hellmann-Feynman and Pulay forces.

The force on nucleus :math:`A` is :math:`\mathbf F_A = -\,dE/d\mathbf R_A`.  At a
converged variational minimum the *parameter* response vanishes
(:math:`\partial E/\partial\vec\theta = 0`), so the reduced density matrices can
be held fixed and only the **integrals** have to be differentiated:

.. math::

    \frac{dE}{d\mathbf R_A}
      = \underbrace{\sum_{\mu\nu} D_{\mu\nu}
          \Big\langle \mu \Big| \frac{\partial \hat V_{ne}}{\partial \mathbf R_A}
          \Big| \nu \Big\rangle
        + \frac{\partial V_{nn}}{\partial \mathbf R_A}}_{\text{Hellmann-Feynman}}
      \; + \; \underbrace{\Big\langle \frac{\partial E}{\partial S},
                                      \frac{\partial S}{\partial \mathbf R_A}\Big\rangle
        + \Big\langle \frac{\partial E}{\partial h},
                      \frac{\partial h}{\partial \mathbf R_A}\Big\rangle_{\text{basis}}
        + \Big\langle \frac{\partial E}{\partial g},
                      \frac{\partial g}{\partial \mathbf R_A}\Big\rangle}_{\text{Pulay}} .

Why Pulay terms are **not** optional here
-----------------------------------------
Mandacaru's orbitals are atom-centered: when a nucleus moves, its basis functions
move with it, so the basis itself depends on :math:`\mathbf R`.  The
Hellmann-Feynman theorem assumes a fixed basis and is therefore *incomplete*.
The size of the omission is not academic -- for H\ :sub:`2` in the HAO basis the
Hellmann-Feynman force is only ~40 % of the true gradient and never crosses zero,
so a relaxation driven by it alone would not find an equilibrium at all.  Both
contributions are computed here, and :func:`nuclear_gradient` returns them
separately so the split can be inspected.

How each piece is obtained
--------------------------
* **Orbital derivatives.** For an atom-centered function
  :math:`\phi_\mu(\mathbf r) = f(\mathbf r - \mathbf R_A)`, moving the center is
  the same as shifting the sampling point:
  :math:`\partial\phi_\mu/\partial R_{A,k}` is evaluated by central differences of
  the analytic orbital on a displaced grid.  No integral is recomputed.
* **Integral derivatives.** :math:`\partial S`, :math:`\partial T` and
  :math:`\partial V` follow from the energy's own Laplacian
  (:func:`one_body_cross`): the FFT Laplacian of the unmoved functions for
  ``kinetic="spectral"`` (the default), the *same* finite-difference kernel on
  the stacked ``[phi; dphi]`` basis for ``kinetic="fd"``.  The two-body derivative never builds a
  :math:`\partial g` tensor: it is contracted on the fly against the Coulomb
  potentials of the pair densities, streamed a block at a time (see
  :func:`_two_body_pulay`).
* **Algebraic response.** :math:`\partial E/\partial S`, :math:`\partial E/\partial h`
  and :math:`\partial E/\partial g` come from automatic differentiation of the
  Löwdin / SCF / transform stack -- see :mod:`mandacaru.algorithms._jax_energy` for
  why AD is the right tool for that layer and not for the grid kernels.

Accuracy caveat
---------------
Holding the RDMs fixed is exact when the wavefunction is variationally complete
in its orbital space (full CI).  With a **frozen core** or a truncated ansatz an
orbital-relaxation (coupled-perturbed) term is neglected.  That approximation is
measured rather than assumed: every force carries
``ForceResult.details["orbital_gradient"]``, the norm of the orbital-rotation
gradient that the neglected term is proportional to, and
:class:`~mandacaru.algorithms.calculator.Mandacaru` warns when it exceeds
:data:`~mandacaru.algorithms.calculator.ORBITAL_RESPONSE_TOLERANCE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..integrals.poisson import PoissonFFTSolver
from ..units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, HARTREE_TO_EV

#: Central-difference step for the orbital derivatives, in Bohr.
DEFAULT_ORBITAL_DELTA = 1e-3

#: Ways of evaluating the electron-nucleus term of the Hellmann-Feynman force.
HELLMANN_FEYNMAN_FORMS = ("analytic", "by-parts")

#: Convert a gradient in Hartree/Bohr to ASE's eV/Angstrom.
HA_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


@dataclass
class ForceResult:
    """Nuclear gradient of a converged variational calculation.

    All arrays are ``(n_atoms, 3)`` in **eV/Angstrom**.  :attr:`forces` is what
    ASE consumes (already negated: it is :math:`-dE/d\\mathbf R`); the component
    breakdown keeps the *gradient* convention (:math:`+dE/d\\mathbf R`) so the
    physics can be inspected term by term; it is ``None`` when the gradient
    was computed without one (an integral-direct active space).  (The gradient mathematics itself
    runs in Hartree/Bohr; the conversion happens once, here.)
    """

    forces: np.ndarray                                  # -dE/dR, eV/Angstrom
    hellmann_feynman: np.ndarray | None                 # dE/dR, eV/Angstrom
    pulay: np.ndarray | None                            # dE/dR, eV/Angstrom
    gradient: np.ndarray                                # total dE/dR, eV/Angstrom
    n_electrons: float = 0.0                            # tr(gamma), a sanity check
    details: dict = field(default_factory=dict)

    @property
    def max_force(self) -> float:
        """Largest force magnitude on any atom (eV/Angstrom)."""
        return float(np.max(np.linalg.norm(self.forces, axis=1)))

    @property
    def unprojected(self) -> np.ndarray:
        """:attr:`forces` before the translational projection, if one was made.

        The projection (:meth:`~mandacaru.algorithms.calculator.Mandacaru._project_translation`)
        enforces the free-molecule identity ``sum_A F_A = 0``, which the
        discretized energy does not obey -- so it is *this* array, not
        :attr:`forces`, that is the gradient of the energy the solver reported
        and that a finite-difference check has to be compared against.  Equal to
        :attr:`forces` when nothing was projected.
        """
        return self.details.get("forces_unprojected", self.forces)

    @property
    def pulay_fraction(self) -> float:
        """‖Pulay‖ / ‖total gradient‖ -- how badly Hellmann-Feynman alone fails."""
        if self.pulay is None:
            return float("nan")                # no breakdown was computed
        total = float(np.linalg.norm(self.gradient))
        return float(np.linalg.norm(self.pulay) / total) if total > 0 else 0.0

    def __repr__(self) -> str:
        return (f"ForceResult(max|F| = {self.max_force:.4f} eV/A, "
                f"Pulay fraction = {self.pulay_fraction:.2f})")


# --------------------------------------------------------------------------- #
# Orbital derivatives on the grid.
# --------------------------------------------------------------------------- #

def orbital_derivatives(basis, grid, atom_of_orbital, atom, delta=None):
    r"""``d phi_mu / d R_{A,k}`` sampled on the grid, for one atom.

    Returns a ``(3, M, ngrid)`` array; row ``k`` is the derivative with respect to
    Cartesian direction ``k``.  Rows for orbitals *not* centered on ``atom`` are
    zero, since only that atom's functions move.  ``atom=None`` moves every
    function with its own center: each row is the one its atom's array
    carries (:func:`atom_rows` takes an atom's array back out of it).

    Because :math:`\phi_\mu(\mathbf r) = f(\mathbf r - \mathbf R_A)`, displacing
    the center is minus the gradient of the unchanged function
    (:meth:`~mandacaru.basis.base.BasisFunction.sample_gradient`): analytic
    for the tabulated radial functions, a central difference of step
    ``delta`` for the others.  Only the points inside a function's support
    are evaluated, and the array is ``float64`` when every function is real.
    """
    from ..integrals.sampling import sample_functions

    delta = DEFAULT_ORBITAL_DELTA if delta is None else float(delta)
    M = len(basis)
    real = all(getattr(fn, "real", False) for fn in basis)
    dtype = np.float64 if real else complex
    if atom is None:
        if M == 0:
            return np.zeros((3, 0, grid.size), dtype=dtype)
        out = np.empty((3, M, grid.size), dtype=dtype)
        sample_functions(basis, (grid.X, grid.Y, grid.Z),
                         gradients=True, delta=delta, values=False,
                         out_slopes=out)
        out *= -1.0
        return out
    out = np.zeros((3, M, grid.size), dtype=dtype)
    own = [mu for mu in range(M) if atom_of_orbital[mu] == atom]
    if own:
        own_out = np.empty((3, len(own), grid.size), dtype=dtype)
        sample_functions([basis[mu] for mu in own],
                         (grid.X, grid.Y, grid.Z),
                         gradients=True, delta=delta, values=False,
                         out_slopes=own_out)
        own_out *= -1.0
        out[:, own] = own_out
    return out


def atom_rows(derivatives, atom_of_orbital, atom):
    """``orbital_derivatives(..., atom)`` from the every-function array
    (``atom=None``): the rows of ``atom``'s functions, zeros elsewhere."""
    own = np.asarray(atom_of_orbital) == atom
    if np.all(derivatives[:, ~own] == 0.0):
        return derivatives
    out = np.zeros_like(derivatives)
    out[:, own] = derivatives[:, own]
    return out


# --------------------------------------------------------------------------- #
# Hellmann-Feynman term.
# --------------------------------------------------------------------------- #

def _nuclear_repulsion_gradient(nuclei_bohr, charges):
    r"""``dV_nn/dR_A = -sum_{B!=A} Z_A Z_B (R_A-R_B)/R_AB^3`` -- purely analytic."""
    gradient = np.zeros((len(charges), 3))
    for a, (Za, Ra) in enumerate(zip(charges, nuclei_bohr)):
        for b, (Zb, Rb) in enumerate(zip(charges, nuclei_bohr)):
            if b == a:
                continue
            sep = Ra - Rb
            gradient[a] -= Za * Zb * sep / np.linalg.norm(sep) ** 3
    return gradient


def orbital_gradients(basis, grid, delta=None):
    r"""``grad phi_mu`` sampled on the grid, for **every** orbital.

    Returns ``(3, M, ngrid)``.  Related to :func:`orbital_derivatives` by
    :math:`\nabla\phi_\mu = -\partial\phi_\mu/\partial\mathbf R_A` -- moving the
    center one way is the same as moving the sampling point the other -- but
    computed for all orbitals at once, since the density gradient needs them all
    regardless of which atom they sit on.
    """
    from ..integrals.sampling import sample_functions

    delta = DEFAULT_ORBITAL_DELTA if delta is None else float(delta)
    real = all(getattr(fn, "real", False) for fn in basis)
    if not basis:
        return np.zeros((3, 0, grid.size), dtype=np.float64 if real else complex)
    _, slopes = sample_functions(basis, (grid.X, grid.Y, grid.Z),
                                 gradients=True, delta=delta, values=False)
    return np.ascontiguousarray(slopes.transpose(1, 0, 2))


def density_gradient(density_ao, psi_grid, grad_psi):
    r"""``grad rho`` from the analytic orbital gradients.

    :math:`\rho = \sum_{\mu\nu} D_{\mu\nu}\phi^*_\mu\phi_\nu`, so

    .. math::

        \nabla\rho = \sum_{\mu\nu} D_{\mu\nu}
            \big[(\nabla\phi^*_\mu)\phi_\nu + \phi^*_\mu(\nabla\phi_\nu)\big].

    Using this instead of a finite-difference gradient of the sampled density
    matters: the density has a **cusp** at every nucleus, exactly where the
    potential it is integrated against is largest, and a finite-difference
    stencil across that cusp is badly wrong.  The orbital gradients are analytic
    (evaluated from the closed-form basis functions), so no stencil ever
    straddles the singularity.
    """
    out = np.zeros((3, psi_grid.shape[1]))
    for k in range(3):
        term = np.einsum("mn,mg,ng->g", density_ao, grad_psi[k].conj(),
                         psi_grid, optimize=True)
        out[k] = 2.0 * np.real(term)          # D Hermitian, so the two terms pair
    return out


def _electronic_by_parts(rho, grid, nuclei_bohr, charges, softening,
                         grad_rho=None):
    r"""Electron-nucleus gradient with the derivative moved onto the density.

    Integrating by parts turns

    .. math::

        \frac{\partial E_{ne}}{\partial \mathbf R_A}
          = -\int \rho\, \nabla V_A
          = +\int (\nabla \rho)\, V_A ,

    which is mathematically identical in the continuum but *numerically much
    better behaved on a grid*.  The standard form integrates
    :math:`\rho\,(\mathbf r - \mathbf R_A)/r^3`: a near-singular integrand whose
    physical value is a tiny anisotropy left over from an almost perfect
    cancellation of enormous core contributions.  The by-parts form instead
    integrates :math:`(\nabla\rho)\,V_A` against the *same* softened
    :math:`-Z/\max(r,s)` the energy already uses, so the worst singularity the
    quadrature ever sees is :math:`1/r` rather than :math:`1/r^2`.

    .. warning::

       **This does not fix the heavy-atom problem.**  Measured on isolated atoms,
       where the exact force is zero, the by-parts form with a *faithful*
       (analytic) density gradient is no better than the standard one, and often
       worse:

       ===================  ==========  =========================  =============================
       isolated Be          analytic    by-parts (finite-diff rho) by-parts (analytic grad rho)
       ===================  ==========  =========================  =============================
       grid_spacing = 0.20  1167        157                        4427
       grid_spacing = 0.10  1593        50                         1600
       ===================  ==========  =========================  =============================

       The apparently excellent finite-difference column is a **smoothing
       artifact**, not accuracy: differencing the sampled density across the
       nuclear cusp acts as a low-pass filter that happens to damp the spurious
       force.  It is not a method -- it also gets H\ :sub:`2`, where the standard
       form is verified correct to 0.04 %, wrong by a factor of two.  This
       implementation therefore uses the analytic orbital gradients by default and
       makes no accuracy claim for the result.

       The two forms are also equal only in the continuum: the standard form is
       exactly the derivative of the energy Mandacaru computes, whereas this one is
       not, so an optimizer using it will not reach a stationary point of the
       reported energy.

       The mode is kept as an independent formulation for diagnosis -- the
       agreement or disagreement between the two is itself a useful measure of
       how badly the grid resolves the core.  The real fix is to remove the core
       cusp (pseudopotentials) or the grid (analytic Gaussian integrals).
    """
    if grad_rho is None:
        # Fallback: finite-difference the sampled density.  Less accurate near
        # the nuclear cusp, so callers should pass the analytic `grad_rho`.
        if not grid.is_orthorhombic:
            raise NotImplementedError(
                "the by-parts Hellmann-Feynman form needs a density gradient; "
                "without analytic orbital gradients it falls back to a "
                "finite-difference stencil, which assumes an orthorhombic grid")
        dlist = np.gradient(rho.reshape(grid.shape), grid.dx, grid.dy, grid.dz)
        grad_rho = np.stack([d.ravel() for d in dlist])

    coords = np.stack(grid.flat_coords())
    gradient = np.zeros((len(charges), 3))

    for a, (Za, Ra) in enumerate(zip(charges, nuclei_bohr)):
        diff = coords - Ra[:, None]
        r = np.sqrt(np.sum(diff * diff, axis=0))
        potential = -Za / np.maximum(r, softening)          # V_A on the grid
        for k in range(3):
            gradient[a, k] = np.sum(grad_rho[k] * potential) * grid.dV
    return gradient


def pseudopotential_local_gradient(density_ao, psi_grid, grid, nuclei_bohr,
                                   potentials, grad_psi=None):
    r"""``dE/dR_A`` from the **local** channel of a pseudopotential.

    The same Hellmann-Feynman idea, but with a potential that is bounded rather
    than singular:

    .. math::

        rac{\partial E}{\partial \mathbf R_A}
          = \int 
ho(\mathbf r)\,
            rac{\partial V^A_{	ext{loc}}(|\mathbf r-\mathbf R_A|)}
                 {\partial \mathbf R_A}\, d^3r
          = -\int 
ho(\mathbf r)\, V^{A\prime}_{	ext{loc}}(r)\,
            \hat{\mathbf r}_A \, d^3r .

    :math:`V'_{	ext{loc}}` is obtained by differentiating the radial table, and
    is finite everywhere -- which is exactly why the pseudopotential force does
    not suffer the core pathology of the all-electron one.
    """
    rho = np.real(np.einsum("mn,mg,ng->g", density_ao,
                            psi_grid.conj(), psi_grid, optimize=True))
    coords = np.stack(grid.flat_coords())
    gradient = np.zeros((len(potentials), 3))

    for a, (pp, Ra) in enumerate(zip(potentials, nuclei_bohr)):
        diff = coords - Ra[:, None]
        r = np.sqrt(np.sum(diff * diff, axis=0))

        # The quantity actually needed is (dV/dr)/r, because
        #   dV/dR_A = -(dV/dr) * (r - R_A)/r ,
        # and it is *finite* at the origin: a smooth local potential is flat
        # there, so dV/dr ~ c r.  Interpolating dV/dr and then dividing by r
        # instead would manufacture a 1/r spike at any grid node that lands near
        # the nucleus -- which is precisely the pathology pseudopotentials exist
        # to remove.  So the ratio is formed on the radial table first.
        dv_dr = np.gradient(pp.v_local, pp.r)
        ratio = dv_dr / pp.r
        ratio[0] = ratio[1]                     # constant limit as r -> 0
        # Beyond the table V = -Z_ion/r, so (dV/dr)/r = Z_ion/r^3.
        outside = pp.valence_charge / pp.r[-1] ** 3
        value = np.interp(r, pp.r, ratio, left=ratio[0], right=outside)
        far = r > pp.r[-1]
        if np.any(far):
            value[far] = pp.valence_charge / r[far] ** 3

        weight = -rho * value * grid.dV
        gradient[a] = np.sum(weight * diff, axis=1)
    return gradient


def kb_nonlocal_gradient(density_ao, basis, grid, projectors, atom_of_orbital,
                         n_atoms, delta=None):
    r"""``dE/dR_A`` from the Kleinman-Bylander **nonlocal** term.

    The nonlocal energy is
    :math:`E_{NL} = \sum_p E^{KB}_p \sum_{\mu
u} D_{\mu
u}
    \langle\phi_\mu|\chi_p
angle\langle\chi_p|\phi_
u
angle`, and *two* things
    move when atom :math:`A` shifts: the basis functions centered on it (a Pulay
    term) and the projectors belonging to it.  Both derivatives are taken by the
    same displaced-sampling trick used for the orbitals, so no integral is
    recomputed from scratch.
    """
    from ..integrals import _backend

    delta = DEFAULT_ORBITAL_DELTA if delta is None else float(delta)
    X, Y, Z = grid.X, grid.Y, grid.Z
    psi = np.stack([f.evaluate(X, Y, Z).ravel() for f in basis])
    chi = np.stack([p.evaluate(X, Y, Z).ravel() for p in projectors])         if projectors else np.zeros((0, grid.size), dtype=complex)
    gradient = np.zeros((n_atoms, 3))
    if not projectors:
        return gradient

    energies = np.array([p.kb_energy for p in projectors])
    overlaps = _backend.kb_projections(psi, chi, grid.dV)        # (M, P)

    for atom in range(n_atoms):
        for k in range(3):
            shift = [np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)]
            shift[k] = np.full_like(X, delta)

            # d(basis)/dR_A: only functions centered on this atom move.
            dpsi = np.zeros_like(psi)
            for mu, fn in enumerate(basis):
                if atom_of_orbital[mu] != atom:
                    continue
                plus = fn.evaluate(X - shift[0], Y - shift[1], Z - shift[2])
                minus = fn.evaluate(X + shift[0], Y + shift[1], Z + shift[2])
                dpsi[mu] = ((plus - minus) / (2.0 * delta)).ravel()

            # d(projector)/dR_A: only projectors belonging to this atom move.
            dchi = np.zeros_like(chi)
            for index, projector in enumerate(projectors):
                if projector.atom_index != atom:
                    continue
                plus = projector.evaluate(X - shift[0], Y - shift[1],
                                          Z - shift[2])
                minus = projector.evaluate(X + shift[0], Y + shift[1],
                                           Z + shift[2])
                dchi[index] = ((plus - minus) / (2.0 * delta)).ravel()

            d_overlap = (_backend.kb_projections(dpsi, chi, grid.dV)
                         + _backend.kb_projections(psi, dchi, grid.dV))
            # dE = 2 Re sum_p E_p sum_mn D_mn d<mu|chi_p> <chi_p|nu>
            term = np.einsum("mn,mp,np->", density_ao, d_overlap,
                             (overlaps * energies).conj(), optimize=True)
            gradient[atom, k] = 2.0 * float(np.real(term))
    return gradient


def hellmann_feynman_gradient(density_ao, psi_grid, grid, nuclei_bohr, charges,
                              softening, form: str = "analytic",
                              grad_psi=None):
    r"""``dE/dR_A`` from the explicit nuclear dependence of the Hamiltonian.

    Two contributions, both analytic:

    * electron-nucleus attraction,
      :math:`\partial\langle \hat V_{ne}\rangle/\partial R_{A,k}
      = -Z_A \int \rho(\mathbf r)\,(r-R_A)_k/|\mathbf r - \mathbf R_A|^3`;
    * nuclear repulsion,
      :math:`\partial V_{nn}/\partial R_{A,k}
      = -\sum_{B\neq A} Z_A Z_B (R_A-R_B)_k/R_{AB}^3`.

    ``density_ao`` is the AO-basis one-particle density matrix
    (:math:`\partial E/\partial h`), which is what makes this the *correlated*
    Hellmann-Feynman force rather than a mean-field one.

    Parameters
    ----------
    form : {"analytic", "by-parts"}
        How the electron-nucleus term is evaluated.

        ``"analytic"`` (default) differentiates the potential directly.  It is
        exactly the derivative of the energy Mandacaru computes -- the right choice
        when force and energy must be consistent -- but its integrand is
        near-singular and, for heavy atoms on a practical grid, dominated by
        core-region noise.

        ``"by-parts"`` moves the derivative onto the density instead (see
        :func:`_electronic_by_parts`), which softens the integrand from
        :math:`1/r^2` to :math:`1/r` at the cost of no longer being exactly
        ``dE/dR`` of the discrete energy.  It does **not** cure the heavy-atom
        core noise: measured on isolated atoms, where the exact force is zero, it
        is no better than ``"analytic"`` and often worse.  See the table in
        :func:`_electronic_by_parts`, and prefer a pseudopotential basis, whose
        dedicated path (:mod:`mandacaru.algorithms.pseudo_forces`) removes the
        core rather than fighting it.

    .. note::

       For the analytic form the engine regularizes the Coulomb singularity as
       :math:`-Z/\max(r, s)`, so **inside** the softening radius the potential is
       *constant* and its nuclear derivative is exactly zero.  Differentiating the
       clamped expression instead would give a spurious :math:`1/s^3` spike: a
       single grid node closer than :math:`s` to a nucleus is enough to corrupt
       the force by ~20 %, which is why that region is masked out explicitly.
    """
    if form not in HELLMANN_FEYNMAN_FORMS:
        raise ValueError(
            f"unknown Hellmann-Feynman form {form!r}; use one of "
            f"{HELLMANN_FEYNMAN_FORMS}")

    rho = np.real(np.einsum("mn,mg,ng->g", density_ao,
                            psi_grid.conj(), psi_grid, optimize=True))

    if form == "by-parts":
        grad_rho = (density_gradient(density_ao, psi_grid, grad_psi)
                    if grad_psi is not None else None)
        electronic = _electronic_by_parts(rho, grid, nuclei_bohr, charges,
                                          softening, grad_rho=grad_rho)
    else:
        coords = np.stack(grid.flat_coords())                   # (3, ngrid)
        electronic = np.zeros((len(charges), 3))
        for a, (Za, Ra) in enumerate(zip(charges, nuclei_bohr)):
            diff = coords - Ra[:, None]                         # (3, ngrid)
            r = np.sqrt(np.sum(diff * diff, axis=0))
            # V = -Z/max(r, s) is flat inside the softening radius, so its
            # nuclear derivative vanishes there.
            weight = np.where(r >= softening,
                              -Za * rho / np.maximum(r, softening) ** 3,
                              0.0) * grid.dV
            electronic[a] = np.sum(weight * diff, axis=1)

    return electronic + _nuclear_repulsion_gradient(nuclei_bohr, charges)


# --------------------------------------------------------------------------- #
# Pulay terms.
# --------------------------------------------------------------------------- #

def one_body_cross(psi_grid, dpsi, vext, grid, kinetic, laplacian=None):
    r"""``<dphi_a| T + V |phi_b>``: the cross block of the one-body matrix.

    The Pulay derivative of a one-body matrix is this block plus its
    Hermitian conjugate, and it has to use the *same* Laplacian as the
    energy, so ``kinetic`` has no default: the caller passes the
    integrals' own operator (``integrals.kinetic``).

    * ``kinetic="fd"`` runs the energy's own finite-difference kernel on the
      stacked basis ``[phi; dphi]`` and reads the cross block;
    * ``kinetic="spectral"`` contracts ``dphi`` against ``laplacian``, the
      FFT Laplacian of the unmoved functions
      (:meth:`~mandacaru.integrals.engine.IntegralEngine.spectral_laplacian`),
      computed once per gradient.  The FFT Laplacian is real symmetric on
      the grid, so :math:`\langle\partial\phi_a|\nabla^2\phi_b\rangle` and
      :math:`\langle\phi_a|\nabla^2\partial\phi_b\rangle^*` are the same
      numbers and the moved functions are never transformed;
    * ``kinetic="analytic"`` (``two_center="analytic"``) leaves the kinetic
      energy to the two-center tables
      (:class:`~mandacaru.integrals.two_center.TwoCenter`): the block is the
      local potential's alone.
    """
    from ..integrals import _backend

    M = psi_grid.shape[0]
    if kinetic == "analytic":
        return (dpsi.conj() * vext) @ psi_grid.T * grid.dV
    if kinetic == "fd":
        stacked = np.ascontiguousarray(np.vstack([psi_grid, dpsi]))
        T_full, V_full = _backend.one_body_matrices(stacked, vext, grid)
        return (T_full + V_full)[M:, :M]
    if kinetic != "spectral":
        raise ValueError(f"unknown kinetic operator {kinetic!r}; use 'fd' or "
                         "'spectral'")
    if laplacian is None:
        raise ValueError("the spectral cross block needs the Laplacian of the "
                         "unmoved functions (`laplacian=`)")
    bra = dpsi.conj()
    return ((-0.5 * (bra @ laplacian.T) + (bra * vext) @ psi_grid.T)
            * grid.dV)


def _one_body_pulay(engine, psi_grid, dpsi, vext, grid, de_dh, de_ds,
                    kinetic, laplacian=None):
    r"""Basis-motion part of ``<dE/dh, dh/dR> + <dE/dS, dS/dR>``.

    ``dh/dR`` and ``dS/dR`` are evaluated from the cross blocks
    :func:`one_body_cross` returns, so the Laplacian is identical to the one
    the energy used.
    """
    # <dphi_mu|h|phi_nu> and its transpose partner.
    cross_h = one_body_cross(psi_grid, dpsi, vext, grid, kinetic, laplacian)
    dh = np.real(cross_h + cross_h.conj().T)
    # Overlap derivative from the same stacked sampling.
    cross_s = (dpsi.conj() @ psi_grid.T) * grid.dV
    ds = np.real(cross_s + cross_s.conj().T)

    return (float(np.sum(de_dh * dh)), float(np.sum(de_ds * ds)))


def _pair_block_size(n_orbitals: int, ngrid: int, n_pairs: int,
                     max_memory_mb: float | None = None,
                     itemsize: int = 16) -> int:
    """Pairs per block so the block's working set stays in the two-body
    budget (:func:`~mandacaru.integrals.engine.eri_memory_budget_mb`): per
    pair its density and potential plus the two ``(M, G)`` products
    :func:`pair_derivative_integrals` forms, ``itemsize`` bytes a number (16
    complex, 8 real).  At most ``M`` pairs, so a small system's block
    (``2 M^2 G`` products) is never larger than the pair-potential stack it
    replaces."""
    from ..integrals.engine import eri_memory_budget_mb

    budget = (eri_memory_budget_mb() if max_memory_mb is None
              else float(max_memory_mb)) * 2 ** 20
    per_pair = float(itemsize) * ngrid * (2 + 2 * n_orbitals)
    return int(max(1, min(n_pairs, n_orbitals, budget // per_pair)))


def pair_potential_blocks(psi_grid, grid, solver=None,
                          max_memory_mb: float | None = None):
    r"""Coulomb potentials :math:`\Phi_{cd}` of the ordered pair densities
    :math:`\phi_c^*\phi_d`, a block of pairs at a time.

    Yields ``(start, stop, phi)``: ``phi[j - start]`` is the potential of
    pair ``j = c M + d`` (the C order of an ``(M, M)`` array).  The density
    and potential buffers are allocated once and reused -- ``phi`` is
    overwritten by the next block, so a consumer contracts it first -- and
    the block size follows the two-body memory budget, so the ``M^2 G``
    stack of all pair potentials never exists.  ``solver`` defaults to the
    isolated :class:`~mandacaru.integrals.poisson.PoissonFFTSolver` of
    ``grid``; it solves on its shared FFT workspace into ``phi``
    (``solve_stack(rho, out=...)``).  Real orbitals on an even kernel (a
    molecule) keep the densities and potentials ``float64``.
    """
    M, ngrid = psi_grid.shape
    if solver is None:
        solver = PoissonFFTSolver(grid.shape, step=grid.step)
    n_pairs = M * M
    real = not np.iscomplexobj(psi_grid)
    potential_real = real and bool(getattr(solver, "even", False))
    size = _pair_block_size(M, ngrid, n_pairs, max_memory_mb,
                            8 if potential_real else 16)
    rho = np.empty((size, ngrid), dtype=np.float64 if real else complex)
    phi = np.empty((size, ngrid),
                   dtype=np.float64 if potential_real else complex)
    conj = np.conj(psi_grid)
    for start in range(0, n_pairs, size):
        stop = min(start + size, n_pairs)
        n = stop - start
        c, d = np.divmod(np.arange(start, stop), M)
        np.multiply(conj[c], psi_grid[d], out=rho[:n])
        solver.solve_stack(rho[:n], out=phi[:n])
        yield start, stop, phi[:n]


def pair_derivative_integrals(psi_grid, derivatives, phi):
    r"""``(T1, T2)``: one block of pair potentials against the orbital
    derivatives, ``(3, M, M, n)`` each.

    :math:`T_1[k, a, b, j] = \sum_g \partial_k\phi_a^*\,\phi_b\,\Phi_j`
    and :math:`T_2[k, a, b, j] = \sum_g \phi_a^*\,\partial_k\phi_b\,
    \Phi_j` (no volume element), with ``derivatives`` the every-function
    array of :func:`orbital_derivatives`.  The derivative of a pair density
    :math:`\rho_{ab}` contracted with :math:`\Phi_j` is ``T1`` on the rows
    of the moving atom plus ``T2`` on its columns.  Two ``(n M, G)``
    products are the only temporaries.
    """
    M, ngrid = psi_grid.shape
    n = phi.shape[0]
    right = (phi[:, None, :] * psi_grid[None, :, :]).reshape(n * M, ngrid)
    left = (phi[:, None, :] * np.conj(psi_grid)[None, :, :]).reshape(n * M,
                                                                     ngrid)
    dtype = np.result_type(right, derivatives)
    T1 = np.empty((3, M, M, n), dtype=dtype)
    T2 = np.empty((3, M, M, n), dtype=dtype)
    for k in range(3):
        T1[k] = (np.conj(derivatives[k]) @ right.T).reshape(M, n, M
                                                            ).transpose(0, 2, 1)
        T2[k] = (left @ derivatives[k].T).reshape(n, M, M).transpose(1, 2, 0)
    return T1, T2


def _two_body_pulay(psi_grid, derivatives, grid, de_dg, atom_of_orbital,
                    n_atoms, solver=None, pair_layout: bool = False,
                    basis=None, delta=None):
    r"""``<dE/dg, dg/dR>`` of every atom and direction, without ``dg``.

    With ``g[m,n,l,s] = <mn|ls> = int rho_ml(1) Phi_ns(1)``, the derivative has
    four terms; two act on electron 1 and two on electron 2.  Using the
    symmetry of the Coulomb kernel, both reduce to contracting a *derivative
    pair density* against a potential that is already available:
    :math:`\sum_{ab} \int \partial\rho_{ab} \sum_{cd} K_{ab,cd}\Phi_{cd}`
    with :math:`K_{ab,cd} = \partial E/\partial g_{acbd} + \partial
    E/\partial g_{cadb}`.

    The pair potentials are streamed (:func:`pair_potential_blocks`) and
    accumulated into the factored one-particle intermediate fields
    :math:`W_a(\mathbf{r}) = \sum_{b,j} K_{ab,j} \phi_b(\mathbf{r}) \Phi_j(\mathbf{r})`
    and :math:`Z_b(\mathbf{r}) = \sum_{a,j} K_{ab,j} \phi_a^*(\mathbf{r}) \Phi_j(\mathbf{r})`.
    Derivatives are contracted either against ``derivatives`` or streamed
    atom-by-atom from ``basis``, eliminating large 4D intermediate tensors
    and global derivative allocations.  Returns ``(n_atoms, 3)``.

    ``pair_layout`` takes ``de_dg`` as the ``(M^2, M^2)`` matrix
    :math:`P_{(ac),(bd)} = \partial E/\partial g_{abcd}`, whose
    :math:`K` is :math:`P + P^T` (no ``M^4`` transpose besides).
    """
    M, ngrid = psi_grid.shape
    if pair_layout:
        K = (de_dg + de_dg.T).reshape(M, M, M * M)
    else:
        K = (np.einsum("acbd->abcd", de_dg)
             + np.einsum("cadb->abcd", de_dg)).reshape(M, M, M * M)

    if derivatives is not None:
        rows = np.zeros((3, M), dtype=complex)
        columns = np.zeros((3, M), dtype=complex)
        for start, stop, phi in pair_potential_blocks(psi_grid, grid, solver):
            T1, T2 = pair_derivative_integrals(psi_grid, derivatives, phi)
            block = K[:, :, start:stop]
            rows += np.einsum("kabj,abj->ka", T1, block)
            columns += np.einsum("kabj,abj->kb", T2, block)
        out = np.zeros((n_atoms, 3))
        owner = np.asarray(atom_of_orbital)
        for atom in range(n_atoms):
            own = owner == atom
            out[atom] = np.real(rows[:, own].sum(axis=1)
                                + columns[:, own].sum(axis=1)) * grid.dV
        return out

    if basis is None:
        raise ValueError(
            "either derivatives or basis must be provided to _two_body_pulay")

    dtype = np.result_type(psi_grid, K)
    W = np.zeros((M, ngrid), dtype=dtype)
    Z = np.zeros((M, ngrid), dtype=dtype)
    psi_conj = np.conj(psi_grid)

    for start, stop, phi in pair_potential_blocks(psi_grid, grid, solver):
        block = K[:, :, start:stop]
        for a in range(M):
            Va = block[a] @ phi
            W[a] += np.sum(psi_grid * Va, axis=0)
            Ub = block[:, a, :] @ phi
            Z[a] += np.sum(psi_conj * Ub, axis=0)

    out = np.zeros((n_atoms, 3))
    owner = np.asarray(atom_of_orbital)
    for atom in range(n_atoms):
        own = np.where(owner == atom)[0]
        if len(own) == 0:
            continue
        dpsi_atom = orbital_derivatives(basis, grid, atom_of_orbital, atom,
                                        delta=delta)
        dpsi_own = dpsi_atom[:, own, :]
        W_own = W[own, :]
        Z_own = Z[own, :]
        for k in range(3):
            term_row = np.sum(dpsi_own[k].conj() * W_own)
            term_col = np.sum(dpsi_own[k] * Z_own)
            out[atom, k] = np.real(term_row + term_col) * grid.dV

    return out


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def nuclear_gradient(integrals, gamma, gamma2, *, n_electrons, atom_of_orbital,
                     frozen=(), orbital_delta=None, scf_iterations=40,
                     include_pulay=True,
                     hellmann_feynman: str = "analytic") -> ForceResult:
    r"""Total nuclear gradient of a converged variational calculation.

    Parameters
    ----------
    integrals : MolecularIntegrals
        The engine that built the Hamiltonian -- it carries the basis, the grid,
        the nuclei and the Coulomb softening, so the gradient is evaluated with
        exactly the quantities the energy used.
    gamma, gamma2 : ndarray
        One- and two-particle RDMs of the converged state, in the *active
        spin-orbital* basis (see :mod:`mandacaru.algorithms.rdm`).
    n_electrons : int
        Total electron count (including any frozen core).
    atom_of_orbital : sequence of int
        Which atom each basis function is centered on.
    frozen : sequence of int
        Frozen spatial molecular orbitals, if the frozen-core approximation was
        used.
    include_pulay : bool
        Compute the Pulay terms (default ``True``).  ``False`` returns the bare
        Hellmann-Feynman force -- much cheaper, but for an atom-centered basis it
        is *not* the physical gradient (see the module docstring).
    hellmann_feynman : {"analytic", "by-parts"}
        How the electron-nucleus term is evaluated (default ``"analytic"``).
        ``"by-parts"`` integrates a less singular integrand but is no longer
        exactly the derivative of the reported energy, and does not cure the
        heavy-atom core noise -- see :func:`hellmann_feynman_gradient` and the
        measured table in :func:`_electronic_by_parts`.

    Returns
    -------
    ForceResult
        Forces in eV/Angstrom plus the Hellmann-Feynman / Pulay breakdown.
    """
    from ._jax_energy import integral_gradients

    grid = integrals.grid
    basis = integrals.basis
    M = len(basis)

    psi_grid = np.stack([b.evaluate(grid.X, grid.Y, grid.Z).ravel()
                         for b in basis])

    # The *same* external potential the energy was built from.  Going straight
    # to `_potentials.nuclear_potential` would silently use the bare -Z/r even
    # on the pseudopotential path -- which is both the wrong operator and, since
    # that path sets softening = 0, divergent wherever a grid node lands near a
    # nucleus.  Route through the engine's own accessor instead.
    external = integrals.external_potential()
    vext = np.ascontiguousarray(
        np.real(external(grid.X, grid.Y, grid.Z)).reshape(-1),
        dtype=np.float64)

    # Raw AO integrals -- the inputs the algebraic layer differentiates.  This
    # must reproduce MolecularIntegrals._compute() exactly, Kleinman-Bylander
    # term included: `de_dh` and friends are derivatives of *this* h_ao, so any
    # term missing here makes them derivatives of an energy nobody evaluated.
    S = np.real(integrals.overlap())
    kinetic = integrals.kinetic
    T, V = integrals._engine.one_body(external, energy_units="Ha",
                                      kinetic=kinetic)
    one_body = T + V + integrals.kb_nonlocal()
    h_ao = np.real(0.5 * (one_body + one_body.conj().T))
    eri_ao = np.real(integrals._engine.two_body(energy_units="Ha"))

    de_ds, de_dh, de_dg = integral_gradients(
        S, h_ao, eri_ao, gamma, gamma2, n_electrons=n_electrons,
        frozen=tuple(frozen), nuclear_repulsion=0.0,
        scf_iterations=scf_iterations)

    charges = [Z for Z, _R in integrals.nuclei]
    nuclei_bohr = [np.asarray(R, dtype=float) * ANGSTROM_TO_BOHR
                   for _Z, R in integrals.nuclei]
    softening = integrals._potentials.softening

    if getattr(integrals, "uses_pseudopotentials", False):
        # Pseudopotential path: a smooth local channel plus the Kleinman-Bylander
        # nonlocal term, and ion-ion rather than nucleus-nucleus repulsion.
        hf = pseudopotential_local_gradient(de_dh, psi_grid, grid, nuclei_bohr,
                                            integrals.pseudopotentials)
        hf = hf + _nuclear_repulsion_gradient(nuclei_bohr, charges)
        hf = hf + kb_nonlocal_gradient(
            de_dh, basis, grid, integrals.kb_projectors, atom_of_orbital,
            len(charges), delta=orbital_delta)
    else:
        grad_psi = (orbital_gradients(basis, grid, orbital_delta)
                    if hellmann_feynman == "by-parts" else None)
        hf = hellmann_feynman_gradient(de_dh, psi_grid, grid, nuclei_bohr,
                                       charges, softening,
                                       form=hellmann_feynman,
                                       grad_psi=grad_psi)

    pulay = np.zeros_like(hf)
    if include_pulay:
        laplacian = (integrals._engine.spectral_laplacian(psi_grid)
                     if kinetic == "spectral" else None)
        # Every function moving with its own center: one sampling, read by
        # atom below, and the pair potentials streamed once for all atoms.
        derivatives = orbital_derivatives(basis, grid, atom_of_orbital, None,
                                          orbital_delta)
        two = _two_body_pulay(psi_grid, derivatives, grid, de_dg,
                              atom_of_orbital, len(charges))
        for atom in range(len(charges)):
            dstack = atom_rows(derivatives, atom_of_orbital, atom)
            if not np.any(dstack):
                continue                        # no basis function on this atom
            for k in range(3):
                dpsi = dstack[k]
                one_h, one_s = _one_body_pulay(integrals._engine, psi_grid,
                                               dpsi, vext, grid, de_dh, de_ds,
                                               kinetic, laplacian)
                pulay[atom, k] = one_h + one_s + two[atom, k]

    # The single Hartree/Bohr -> eV/Angstrom boundary of the force path.
    hf = hf * HA_BOHR_TO_EV_ANGSTROM
    pulay = pulay * HA_BOHR_TO_EV_ANGSTROM
    gradient = hf + pulay
    forces = -gradient
    return ForceResult(
        forces=forces, hellmann_feynman=hf, pulay=pulay, gradient=gradient,
        n_electrons=float(np.real(np.trace(gamma))),
        details={"n_orbitals": M, "frozen": tuple(frozen),
                 "include_pulay": bool(include_pulay),
                 "hellmann_feynman": hellmann_feynman})
