# -*- coding: utf-8 -*-
# file: pseudopotentials/onecenter.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The PAW-LCAO one-center electron-electron terms.

Blöchl's total energy splits the Hartree energy into a smooth part evaluated
everywhere and a correction confined to each augmentation sphere,

.. math::

    E_H[n] = E_H[\tilde n + \hat n]
           + \sum_A \Big( E_H[n^1_A] - E_H[\tilde n^1_A + \hat n_A] \Big),

where :math:`n^1_A = \sum_{ij} D_{ij}\,\phi_i^*\phi_j` is the all-electron
one-center density and :math:`\tilde n^1_A` its smooth counterpart.  The first
term is what the grid and the compensation charges already compute
(:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.two_body_augmentation`);
this module supplies the second.

It is **quadratic** in the one-center density matrix :math:`D`, and that is the
point.  Mandacaru linearizes it about the isolated atom -- a fixed
:math:`D^{ion}` plus a per-species constant -- which is exact only while
:math:`D` stays at its atomic reference :math:`D^0`.  Forming a bond is
precisely what moves it: charge transfers and the s and p channels rehybridize.
Writing :math:`E(D) = \tfrac12 D D \Delta W` and expanding about :math:`D^0`,

.. math::

    E(D) = \underbrace{E(D^0) + (D - D^0)\,D^0 \Delta W}_{\text{the linearization}}
         + \underbrace{\tfrac12 (D-D^0)(D-D^0)\,\Delta W}_{\text{what was missing}},

so the omission is exactly the second-order term.  A Kohn-Sham calculation
keeps that linearization (0.08-0.33 eV on oxygen for the Hartree part).  A
wave-function Hamiltonian does not linearize at all: :class:`OneCenterExact`
puts :math:`\Delta W` itself into its two-body operator, which makes the
spheres' Hartree *and* exchange exact at any density, and takes the
dataset's frozen functional out (see the last section of this module).

:math:`\Delta W` itself is
:math:`(\phi_a^*\phi_b|\phi_c^*\phi_d) - (\tilde\phi_a^*\tilde\phi_b + \hat n_{ab}
|\tilde\phi_c^*\tilde\phi_d + \hat n_{cd})` over one sphere, assembled from the
usual multipole expansion of :math:`1/r_{12}`: a radial double integral per
:math:`L` (the native radial Slater integral) times the angular couplings of
:mod:`mandacaru.pseudopotentials.multipoles`.

The one-center exact exchange of a screened hybrid
------------------------------------------------------
HSE06 (:mod:`mandacaru.basis.hse`) replaces the fraction :math:`a` of the
short-range semilocal exchange of the valence density by as much short-range
exact exchange.  On PAW-LCAO both halves of that swap have a smooth part and a
one-center part, and **both** one-center parts are needed: either alone moves
the energy by ~0.1 Ha for H2O.

*Exact exchange.*  Exactly as for the Hartree energy, the all-electron exchange
is the smooth exchange of the **augmented** pair densities plus a one-center
correction per sphere,

.. math::

    E_x^{\rm HF,SR} = \tilde E_x^{\rm SR}\big[\tilde\rho_{pr} + \hat\rho_{pr}\big]
      - \frac14\sum_A \sum_{ijkl} D^A_{li} D^A_{jk}\,
        \Delta W^{\rm SR}_A[i,j,k,l]

(closed shell, both spins in :math:`D`; a spin channel alone carries
:math:`-\tfrac12`).  The first term is the grid tensor with the compensation
charges under the erfc kernel
(:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.short_range_two_body`,
:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.long_range_augmentation`);
:math:`\Delta W^{\rm SR}` is :math:`\Delta W` with
:math:`\operatorname{erfc}(\omega r_{12})/r_{12}` (:func:`one_center_exchange_tensor`).
:math:`D^A = C_A^\dagger D C_A` is the projected density matrix,
:math:`C_{pi} = \langle\chi_p|\tilde p_i\rangle`, and the exchange matrix it
adds to the operator is :math:`C_A\,\Delta K\,C_A^\dagger` with
:math:`\Delta K_{il} = \sum_{jk} D^A_{jk}\,\Delta W^{\rm SR}[i,j,k,l]` -- the
one-center image of the molecular :math:`K_{pq} = \sum_{rs} D_{sr}
\langle pr|sq\rangle`.  It is kept **quadratic**, not linearized: an exchange
operator frozen at the spherical reference atom would shift occupied and empty
orbitals of a channel alike, which is the opposite of what exact exchange does.

*Semilocal exchange.*  The functional subtracts
:math:`a E_x^{\omega{\rm PBE,SR}}` of the valence density, whose PAW form is
:math:`E[\tilde n_v] + \sum_A (E[n^1_A] - E[\tilde n^1_A])` -- valence only,
no compensation charge, as for any semilocal term.  The grid takes the first
part.  The one-center part is exactly what the datasets **freeze**: their
one-center semilocal energies are linearized at the reference atom (the
constant ``one_center_energy`` and its derivative inside :math:`D^{ion}`).
The hybrid's share of them is frozen the same way,

.. math::

    \Delta X_A(D) = X^0_A + \sum_{ij} V^A_{ij}\,(D^A_{ij} - D^0_{ij}),
    \qquad
    X^0_A = E_x^{\rm SR}[n^1_0] - E_x^{\rm SR}[\tilde n^1_0],

    V^A_{ij} = \int v_x^{\rm SR}[n^1_0]\,\phi_i^*\phi_j
             - \int v_x^{\rm SR}[\tilde n^1_0]\,\tilde\phi_i^*\tilde\phi_j ,

with :math:`n^1_0` and :math:`\tilde n^1_0` the reference atom's all-electron
and smooth valence densities (spherical, so :math:`V` is diagonal in
:math:`lm`; the GGA term enters through :math:`2\,\partial f/\partial\sigma\,
n_0'\,(R_iR_j)'`), and :math:`D^0` its one-center density matrix (the
channel occupation on the bound wave).  The semilocal exchange is the hole
model of :mod:`mandacaru.basis.hse` (the piece HSE06 removes from PBE),
whatever functional the dataset was generated with, so that the change
HSE06 - PBE is the hybrid's own.

*Per atom, all together*, the hybrid adds to the frozen one-center energy

.. math::

    E^A_{\rm hyb}(D) = -a\big[X^0_A - V^A\!\cdot D^0\big]
      - a\,V^A\!\cdot D^A
      - \frac a4 \sum D^A_{li} D^A_{jk}\,\Delta W^{\rm SR}_A[i,j,k,l],

a per-species constant, a one-body term :math:`-a\,C_A V^A C_A^\dagger`
(both spins alike: the derivative of the spin-scaled functional at the
unpolarized reference) and the quadratic exchange.  :class:`OneCenterHybrid`
evaluates the three on the ``(P, P)`` projected density matrix, so a crystal
(:math:`D^A = \sum_k w_k C_A(k)^\dagger P^k C_A(k)`) uses it unchanged.
:math:`a \to 0` removes everything; :math:`\omega \to 0` turns the kernel into
:math:`1/r_{12}` and :math:`X^0` into the full hole-model exchange (a PBE0-like
hybrid); :math:`\omega \to \infty` removes everything.  The unitary datasets
(UPAW-LCAO, :math:`q = 0`) take the same path: their spheres carry no monopole
but still the higher multipoles and the full one-center difference.

The kernel.  :math:`\operatorname{erfc}(\omega r)/r = 1/r - \operatorname{erf}
(\omega r)/r`; the bare part keeps the closed-form radial kernel
:math:`r_<^L/r_>^{L+1}`, and the long-range part's multipole components
:math:`f_L(r, r') = \tfrac{2L+1}2\int_{-1}^1
\operatorname{erf}(\omega R)/R\,P_L(\mu)\,d\mu`,
:math:`R^2 = r^2 + r'^2 - 2rr'\mu`, are integrated by Gauss-Legendre in
:math:`\mu` (the native screened kernel).  :math:`\operatorname{erf}(x)/x`
is an entire function of :math:`x^2`, hence of :math:`\mu`, so the rule
converges exponentially (24 nodes: round-off, inside a sphere).  Inside an
augmentation sphere :math:`\omega r \lesssim 0.3`, where the series
:math:`2\omega/\sqrt\pi - (2\omega^3/3\sqrt\pi)R^2 + \dots` would also do:
its constant cancels between the two densities (they share every moment) and
the :math:`R^2` term is the whole correction to the bare :math:`\Delta W`
at the 1e-6 level (HISTORY.md, 2026-10-03).  The exact kernel costs nothing
measurable, so it is what is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _native
from .multipoles import partial_waves
from .native_context import context

#: Core orbitals of the re-solved reference atoms, keyed by the dataset file
#: (``source``, ``source_sha256``): one solve per file and process.
_CORE_ORBITALS: dict = {}


# --------------------------------------------------------------------------- #
# Dataset adapters: what the native routines are handed.
# --------------------------------------------------------------------------- #

def _occupations(dataset, l: int) -> list[float]:
    """The reference atom's electrons in each partial wave of channel ``l``."""
    channel = dataset.channels[int(l)]
    return [float(o) for o in (getattr(channel, "occupations", None)
                               or [channel.occupation])]


def _channel_arrays(dataset, *, occupied: bool = False) -> dict:
    """The keyword arguments describing a dataset's channels to the native
    routines: momenta, all-electron and smooth waves, Vanderbilt matrices and
    cutoffs, and (``occupied``) the occupations and projector radii."""
    levels = sorted(dataset.channels)
    waves = {l: partial_waves(dataset, l) for l in levels}
    arrays = {
        "ls": levels,
        "ae": [np.ascontiguousarray(waves[l][0], dtype=float) for l in levels],
        "ps": [np.ascontiguousarray(waves[l][1], dtype=float) for l in levels],
        "vanderbilt": [np.ascontiguousarray(dataset.channels[l].vanderbilt,
                                            dtype=float) for l in levels],
        "r_cuts": [float(dataset.channels[l].r_cut) for l in levels]}
    if occupied:
        arrays["occupations"] = [np.asarray(_occupations(dataset, l),
                                            dtype=float) for l in levels]
        arrays["projector_radii"] = [float(dataset.projector_radius(l))
                                     for l in levels]
    return arrays


def _layout(projectors) -> np.ndarray:
    """``(n, 4)`` rows ``l, m, index, real`` of a sphere's projectors."""
    return np.array([[p.l, p.m, p.index, int(bool(p.real))]
                     for p in projectors], dtype=np.int64).reshape(-1, 4)


def _members_layout(projectors) -> np.ndarray:
    """``(n, 3)`` rows ``l, m, index`` of a sphere's projectors."""
    return np.ascontiguousarray(_layout(projectors)[:, :3])


def _projector_basis(projectors, basis):
    if basis is not None:
        return str(basis)
    from .paw import DEFAULT_PROJECTOR_BASIS
    return str(getattr(projectors[0], "projector_basis",
                       DEFAULT_PROJECTOR_BASIS))


def _tensor(dataset, projectors, basis: str, omega: float,
            symmetrize: bool) -> np.ndarray:
    return _native.paw_one_center_tensor(
        context(), np.ascontiguousarray(dataset.r, dtype=float),
        **_channel_arrays(dataset),
        compensation_radius=float(dataset.compensation_radius),
        basis_name=str(basis), layout=_layout(projectors),
        omega=float(omega), symmetrize=bool(symmetrize))


def one_center_coulomb(dataset, projectors, basis: str = "raw") -> np.ndarray:
    r"""``DeltaW[a, b, c, d]`` for one atom, over its own ``projectors``.

    :math:`\Delta W = (\phi^*_a\phi_b|\phi^*_c\phi_d) -
    (\tilde\phi^*_a\tilde\phi_b + \hat n_{ab}|\tilde\phi^*_c\tilde\phi_d
    + \hat n_{cd})`, the all-electron minus smooth-plus-compensation
    electron-electron energy inside the augmentation sphere.  Both densities
    carry the same multipole moments by construction, so the difference is
    confined to the sphere and the result is finite and short-ranged.

    ``projectors`` are this atom's :class:`~mandacaru.pseudopotentials.orbitals.KBProjector`
    objects in the order the integrals index them; ``basis`` must match the one
    they were built in.
    """
    return _tensor(dataset, projectors, basis, 0.0, False)


def one_center_exchange_tensor(dataset, projectors, omega: float,
                               basis: str | None = None) -> np.ndarray:
    r"""``DeltaW^SR[a, b, c, d]`` -- :func:`one_center_coulomb` under the
    short-range kernel :math:`\operatorname{erfc}(\omega r_{12})/r_{12}`.

    The one-center correction of a screened hybrid's exact exchange (see the
    module docstring).  ``basis`` defaults to the projectors' own
    ``projector_basis``.  Cached on the dataset per ``(omega, basis)`` and
    projector layout; ``omega = 0`` is :func:`one_center_coulomb`,
    symmetrized.

    Symmetrized under the exchange of the two electrons,
    :math:`[ab|cd] = [cd|ab]`: the cumulative trapezoid inside
    the native radial Slater integral breaks it at ~1e-5 relative (raw projector
    basis), and the exchange operator is the derivative of the energy only
    for a symmetric tensor.
    """
    omega = float(omega)
    if omega < 0.0:
        raise ValueError(f"omega must be >= 0, got {omega!r}")
    basis = _projector_basis(projectors, basis)
    layout = tuple((p.l, p.m, p.index, bool(p.real)) for p in projectors)
    cache = dataset.__dict__.setdefault("_one_center_exchange_cache", {})
    key = (omega, basis, layout)
    if key not in cache:
        cache[key] = _tensor(dataset, projectors, basis, omega, True)
    return cache[key]


def frozen_exchange_terms(dataset, omega: float, basis: str = "raw"):
    r"""``(X0 - V.D0, {l: V_l})`` -- the reference atom's one-center
    short-range semilocal exchange, linearized.

    :math:`X^0 = E_x^{\rm SR}[n^1_0] - E_x^{\rm SR}[\tilde n^1_0]` and
    :math:`V_{ij} = \int v_x^{\rm SR}[n^1_0]\phi_i\phi_j -
    \int v_x^{\rm SR}[\tilde n^1_0]\tilde\phi_i\tilde\phi_j` per channel, in
    the hole model of :func:`~mandacaru.basis.hse.short_range_partials`
    (unpolarized, valence densities of the reference atom).  The first entry
    is the constant of the linearization, :math:`X^0 - \sum V D^0`; the blocks
    are in the projector ``basis`` (the same for every ``m`` of a channel).
    A hybrid with fraction :math:`a` adds :math:`-a` times both
    (:class:`OneCenterHybrid`).  Cached on the dataset per ``(omega, basis)``.
    """
    omega = float(omega)
    cache = dataset.__dict__.setdefault("_frozen_exchange_cache", {})
    key = (omega, str(basis))
    if key not in cache:
        constant, blocks = _native.paw_frozen_exchange(
            context(), np.ascontiguousarray(dataset.r, dtype=float),
            **_channel_arrays(dataset, occupied=True), omega=omega,
            basis_name=str(basis))
        cache[key] = (constant, dict(blocks))
    return cache[key]


@dataclass(frozen=True)
class OneCenterTerms:
    """The one-center hybrid terms at one projected density matrix.

    ``energy`` is everything the spheres add (Hartree): ``exact_exchange``,
    the quadratic one-center exact exchange, plus the linearized semilocal
    term (``semilocal``, constant included).  ``operators`` holds one
    ``(P, P)`` matrix per spin channel evaluated (one for a closed shell),
    :math:`\\partial E/\\partial D^T`, which enters a Hamiltonian as
    :math:`C\\,O\\,C^\\dagger`.
    """

    energy: float
    exact_exchange: float
    semilocal: float
    operators: tuple


def _by_atom(projectors) -> dict:
    positions: dict = {}
    for p, projector in enumerate(projectors):
        positions.setdefault(int(projector.atom_index), []).append(p)
    return positions


class OneCenterHybrid:
    r"""The PAW-LCAO one-center terms of a screened hybrid, in projector space.

    Parameters
    ----------
    projectors : list
        Every :class:`~mandacaru.pseudopotentials.orbitals.KBProjector` of
        the system, in the order the projections ``C[:, p]`` index them (the
        integrals' ``kb_projectors``; the crystal's ``PeriodicPAW.projectors``).
    datasets : sequence
        One dataset per atom, indexed by ``projector.atom_index``.
    omega, fraction : float
        The hybrid's range separation (1/Bohr) and exact-exchange fraction.

    :meth:`evaluate` takes the ``(P, P)`` projected density matrix
    :math:`D = C^\dagger R\,C` -- for a crystal
    :math:`\sum_k w_k C(k)^\dagger P^k C(k)` -- and returns the energy and
    the ``(P, P)`` operator :math:`O`, block diagonal over atoms, that the
    Hamiltonian gains as :math:`C\,O\,C^\dagger` (at every k for a crystal).
    The bookkeeping is in the module docstring.  :attr:`constant` is the
    per-species frozen constant :math:`-a\sum_A (X^0_A - V^A\cdot D^0_A)`,
    already inside every :meth:`evaluate` energy.  The arithmetic runs in
    Rust; the tensors stay there between evaluations.
    """

    def __init__(self, projectors, datasets, omega: float, fraction: float):
        self.omega = float(omega)
        self.fraction = float(fraction)
        self.n_projectors = len(projectors)
        members, tensors, layouts, block_ls, blocks, frozen = ([] for _ in
                                                               range(6))
        real_tensors = True
        for atom, columns in sorted(_by_atom(projectors).items()):
            dataset = datasets[atom]
            own = [projectors[p] for p in columns]
            basis = _projector_basis(own, None)
            tensor = one_center_exchange_tensor(dataset, own, self.omega,
                                                basis)
            constant, per_l = frozen_exchange_terms(dataset, self.omega,
                                                    basis)
            real_tensors = real_tensors and np.isrealobj(tensor)
            members.append(columns)
            tensors.append(np.ascontiguousarray(tensor, dtype=complex))
            layouts.append(_members_layout(own))
            block_ls.append(sorted(per_l))
            blocks.append([per_l[l] for l in sorted(per_l)])
            frozen.append(constant)
        self._native = _native.OneCenterHybrid(
            context(), members, tensors, layouts, block_ls, blocks,
            self.n_projectors, self.fraction, frozen, real_tensors)
        #: The per-species frozen constant (Hartree), fraction included.
        self.constant = self._native.constant

    def evaluate(self, density, density_down=None) -> OneCenterTerms:
        r"""The terms at the projected density matrix ``density``.

        One argument: a closed shell, ``density`` holding both spins; the
        exact exchange is :math:`-\tfrac a4 \sum D_{li}D_{jk}\Delta W` and
        the single operator :math:`-\tfrac a2\Delta K[D] - aV`.  Two: the
        spin channels, :math:`-\tfrac a2` per channel, operators
        :math:`-a\Delta K[D_\sigma] - aV` each.
        """
        spins = [np.asarray(density)] + (
            [] if density_down is None else [np.asarray(density_down)])
        energy, exact, semilocal, operators = self._native.evaluate(
            [np.ascontiguousarray(s, dtype=complex) for s in spins],
            all(np.isrealobj(s) for s in spins))
        return OneCenterTerms(energy=energy, exact_exchange=exact,
                              semilocal=semilocal, operators=tuple(operators))

    def evaluate_spinor(self, blocks) -> OneCenterTerms:
        r"""The terms of a two-component (spinor) state: ``blocks`` are the
        four spin blocks :math:`D^{\sigma\tau}` of the projected density
        matrix (``[[uu, ud], [du, dd]]``).

        The exchange operator is linear in the density matrix and the
        interaction spin-free, so the block :math:`(\sigma, \tau)` of the
        operator is :math:`-a\,\Delta K[D^{\sigma\tau}]` (plus
        :math:`-aV` on the diagonal blocks), and the exact exchange
        :math:`-\tfrac a2\sum_{\sigma\tau}\operatorname{tr}(D^{\tau\sigma}
        \Delta K[D^{\sigma\tau}])` -- the spin-channel form's when the
        off-diagonal blocks vanish.  ``operators`` holds the one ``(2P,
        2P)`` operator, alpha block first."""
        flat = [np.ascontiguousarray(x, dtype=complex)
                for row in blocks for x in row]
        energy, exact, semilocal, operators = self._native.evaluate_spinor(
            flat)
        return OneCenterTerms(energy=energy, exact_exchange=exact,
                              semilocal=semilocal, operators=tuple(operators))


# --------------------------------------------------------------------------- #
# The exact one-center terms of a Hartree-Fock (and correlated) Hamiltonian.
# --------------------------------------------------------------------------- #

def _dataset_file_key(dataset):
    source = getattr(dataset, "source", None)
    digest = getattr(dataset, "source_sha256", None)
    if source is None or digest is None:
        return None
    return (str(source), str(digest))


def core_orbitals(dataset) -> list:
    r"""``[(n, l, occupation, R_nl)]`` of the dataset's frozen core, ``R`` on
    ``dataset.r``.

    The files store the core *density*, not its orbitals, so the reference
    atom is solved again from the dataset's record
    (:func:`~.partial_waves.reference_atom`) unless the dataset still carries
    the one it was generated from.  The core subshells are the generator's:
    the aufbau split of that atom with the dataset's ``frozen_subshells``
    moved into the core and its ``semicore_subshells`` out of it.  Cached on
    the dataset and, for a dataset read from a file, per file
    (``source``, ``source_sha256``).  The orbitals reproduce the stored
    ``core_density`` (``dataset._core_density_error`` records the
    difference, in electrons).
    """
    cached = dataset.__dict__.get("_core_orbitals")
    if cached is not None:
        return cached
    key = _dataset_file_key(dataset)
    if key is not None and key in _CORE_ORBITALS:
        result, error = _CORE_ORBITALS[key]
    else:
        from .dataset import _valence_configuration

        atom = getattr(dataset, "atom", None)
        if atom is None or not getattr(atom, "orbitals", None):
            from .partial_waves import reference_atom
            atom = reference_atom(dataset)
        valence, core = _valence_configuration(
            int(dataset.atomic_number), configuration=atom.occupations)
        for orbital in dataset.frozen_subshells or ():
            orbital = (int(orbital[0]), int(orbital[1]))
            core[orbital] = valence.pop(orbital)
        for orbital in dataset.semicore_subshells or ():
            orbital = (int(orbital[0]), int(orbital[1]))
            valence[orbital] = core.pop(orbital)
        subshells = sorted(core.items())
        radial, error = _native.paw_core_orbitals(
            context(), np.ascontiguousarray(dataset.r, dtype=float),
            np.ascontiguousarray(atom.r, dtype=float),
            [int(l) for (_n, l), _f in subshells],
            [float(f) for _orbital, f in subshells],
            np.ascontiguousarray(
                [atom.orbitals[orbital] for orbital, _f in subshells],
                dtype=float).reshape(len(subshells), np.size(atom.r)),
            np.ascontiguousarray(dataset.core_density, dtype=float))
        result = [(int(n), int(l), float(f), radial[i])
                  for i, ((n, l), f) in enumerate(subshells)]
        if key is not None:
            _CORE_ORBITALS[key] = (result, error)
    dataset.__dict__["_core_orbitals"] = result
    dataset.__dict__["_core_density_error"] = error
    return result


def _smooth_xc_core(dataset):
    """The core density the dataset unscreened its smooth xc with."""
    r = np.asarray(dataset.r, dtype=float)
    record = dataset.nlcc or {}
    if not record.get("applied"):
        return np.zeros_like(r)
    if record.get("source") == "smooth_core" or record.get("r_nlcc") is None:
        return np.asarray(dataset.smooth_core_density, dtype=float)
    from .core_correction import partial_core_density
    density, _details = partial_core_density(
        r, np.asarray(dataset.core_density, dtype=float),
        np.asarray(dataset.valence_density, dtype=float),
        r_nlcc=float(record["r_nlcc"]))
    return density


def exact_one_center_terms(dataset, basis: str = "raw") -> dict:
    r"""``{l: (V, K, D0)}`` -- the one-body pieces that turn the dataset's
    one-center terms into Hartree-Fock ones, per channel, in the projector
    ``basis`` (the same block for every ``m``).

    ``V``: the dataset's one-center exchange-correlation, linearized at the
    reference atom and frozen into :math:`D^{ion}`,

    .. math::

        V_{ij} = \int v_{xc}[n_c + n^1_0]\,\phi_i\phi_j
               - \int v_{xc}[\tilde n^1_0 + \tilde n_c]\,\tilde\phi_i\tilde\phi_j ,

    in the functional, relativistic exchange and core correction the dataset
    was unscreened with (all-electron core plus valence density, smooth
    valence plus the smooth core of the nonlinear core correction).  The
    potentials are those of the atom solver's radial scheme (derivatives on
    subset nodes, splined back), the scheme the datasets were generated with.

    ``K``: the frozen core's exchange, one spin's worth,
    :math:`K_{ij} = -\sum_c f_c\,(2l_c+1)\sum_L
    \begin{pmatrix} l_c & L & l\\ 0&0&0\end{pmatrix}^2
    R^L(\phi_i R_c, R_c\phi_j)`, with :math:`f_c` the per-spin occupation of
    each of the :math:`2l_c+1` orbitals of core subshell ``c``
    (:func:`core_orbitals`) and :math:`R^L` the radial Slater integral
    (the native radial Slater integral).

    ``D0``: the reference atom's projected density matrix per ``m`` (both
    spins), the point the dataset's one-center Hartree term is linearized
    about.  Cached on the dataset per ``basis``.
    """
    cache = dataset.__dict__.setdefault("_exact_one_center_cache", {})
    if basis not in cache:
        core = core_orbitals(dataset)
        rows = [(int(l), float(f), R) for _n, l, f, R in core]
        terms = _native.paw_exact_terms(
            context(), np.ascontiguousarray(dataset.r, dtype=float),
            **_channel_arrays(dataset, occupied=True),
            core_density=np.ascontiguousarray(dataset.core_density,
                                              dtype=float),
            valence_density=np.ascontiguousarray(dataset.valence_density,
                                                 dtype=float),
            smooth_core=np.ascontiguousarray(_smooth_xc_core(dataset),
                                             dtype=float),
            functional=str(dataset.xc),
            relativistic=bool(getattr(dataset, "relativistic_exchange",
                                      False)),
            core_ls=[l for l, _f, _R in rows],
            core_occupations=[f for _l, f, _R in rows],
            core_radial=np.ascontiguousarray(
                [R for _l, _f, R in rows], dtype=float).reshape(
                    len(rows), np.size(dataset.r)),
            basis_name=str(basis))
        cache[basis] = {l: (V, K, D0) for l, V, K, D0 in terms}
    return cache[basis]


class OneCenterExact:
    r"""The exact one-center electron-electron terms of a molecular PAW-LCAO
    Hamiltonian, in projector space.

    Blöchl's electron-electron energy inside sphere ``A`` is
    :math:`\tfrac12\sum D_{ab}D_{cd}\,\Delta W_{abcd}` (Hartree) minus
    :math:`\tfrac12\sum_\sigma\sum D^\sigma_{ad}D^\sigma_{cb}\,\Delta
    W_{abcd}` (exchange), :math:`\Delta W` being :func:`one_center_coulomb`.
    Both are the two-body operator
    :math:`\Delta\langle pq|rs\rangle = \sum_{abcd} C_{pa}C_{rb}\,
    \Delta W_{abcd}\,C_{qc}C_{sd}`, so they go into the two-body tensor and
    every consumer of it -- a Hartree-Fock SCF, a frozen core, an active
    space, a qubit Hamiltonian -- gets the sphere's Coulomb and exchange
    exactly, at any density.  Written as the pair-pair matrix
    :math:`\Delta W_{(ab),(cd)} = \sum_k \lambda_k U^k_{ab}U^k_{cd}`, it is
    the low-rank sum :math:`\sum_k\lambda_k\,G^k\otimes G^k` with
    :math:`G^k = C_A U^k C_A^T` (:meth:`channels`): rank 33 for oxygen's
    eight projectors, 3 for hydrogen.

    What the dataset froze into :math:`D^{ion}` for the same sphere is then
    removed, and the frozen core's exchange added, as one block per
    ``(atom, l, m)`` of the nonlocal coupling (:meth:`coupling_blocks`):
    :math:`-V - \Delta W\cdot D^0 + K` with ``V``, ``K`` and ``D0`` of
    :func:`exact_one_center_terms` (``Delta W . D0`` is the linearized
    one-center Hartree term the quadratic one replaces).

    The constant that goes with it is **zero**.  The dataset's
    ``one_center_energy`` is :math:`-(E_H[n^1_0] - E_H[\tilde n^1_0 + \hat
    n_0]) + X^0`, the double counting of the linearized Hartree term and the
    functional's :math:`X^0 = (E_{xc} - \int n v_{xc})[n^1_0] - (\ldots)
    [\tilde n^1_0]`; with the Hartree term quadratic and no functional left,
    nothing remains: the one-center energy is exactly
    :math:`\tfrac12 DD\Delta W - \tfrac12\sum_\sigma D^\sigma D^\sigma\Delta
    W + \mathrm{tr}(K D)` (``D`` both spins), in the valence-only convention
    of the datasets (the frozen core's own energy is left out, as before).

    Molecules only: real spherical harmonics (crystals use complex ones and
    have no such path).
    """

    def __init__(self, projectors, datasets):
        if any(not bool(p.real) for p in projectors):
            raise NotImplementedError(
                "the exact one-center terms are built for real spherical "
                "harmonics (molecules); complex projectors belong to a "
                "crystal, which has no Hartree-Fock path")
        self.n_projectors = len(projectors)
        self.atoms = []
        for atom, members in sorted(_by_atom(projectors).items()):
            dataset = datasets[atom]
            own = [projectors[p] for p in members]
            basis = _projector_basis(own, None)
            W = one_center_exchange_tensor(dataset, own, 0.0, basis)
            terms = exact_one_center_terms(dataset, basis)
            ls = sorted(terms)
            V, K, D0, coupling, values, factors = _native.paw_exact_atom(
                context(), _members_layout(own), W, ls,
                [terms[l][0] for l in ls], [terms[l][1] for l in ls],
                [terms[l][2] for l in ls])
            self.atoms.append({"atom": atom, "members": np.asarray(members),
                               "projectors": own, "W": W, "V": V, "K": K,
                               "D0": D0, "coupling": coupling,
                               "values": values, "factors": factors})

    @property
    def rank(self) -> int:
        """Number of two-body factors over all spheres."""
        return int(sum(len(a["values"]) for a in self.atoms))

    def coupling_blocks(self) -> dict:
        """``{(atom, l, m): block}`` to add to the nonlocal coupling."""
        out = {}
        for entry in self.atoms:
            own = entry["projectors"]
            groups: dict = {}
            for i, projector in enumerate(own):
                groups.setdefault((projector.l, projector.m), []).append(
                    (projector.index, i))
            for (l, m), members in groups.items():
                members = sorted(members)
                rows = [i for _index, i in members]
                out[(entry["atom"], l, m)] = entry["coupling"][
                    np.ix_(rows, rows)]
        return out

    def _spheres(self):
        return ([[int(m) for m in entry["members"]] for entry in self.atoms],
                [entry["factors"] for entry in self.atoms])

    def channels(self, C) -> tuple:
        r"""``(G, lambda)``: the ``(k, M, M)`` factors :math:`C_A U^k C_A^T`
        of the projections ``C`` and their weights."""
        C = np.asarray(C)
        members, factors = self._spheres()
        G = _native.paw_exact_channels(
            context(), np.ascontiguousarray(C, dtype=complex), members,
            factors, not np.iscomplexobj(C))
        values = [entry["values"] for entry in self.atoms]
        return G, (np.concatenate(values) if values else np.zeros(0))

    def channel_derivatives(self, C, dC) -> np.ndarray:
        r"""``(k, M, M)``: :math:`dC_A U^k C_A^T + C_A U^k dC_A^T` for every
        factor of :meth:`channels` (zeros where ``dC`` does not touch the
        sphere)."""
        C, dC = np.asarray(C), np.asarray(dC)
        members, factors = self._spheres()
        return _native.paw_exact_channel_derivatives(
            context(), np.ascontiguousarray(C, dtype=complex),
            np.ascontiguousarray(dC, dtype=complex), members, factors,
            not (np.iscomplexobj(C) or np.iscomplexobj(dC)))
