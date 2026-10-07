# -*- coding: utf-8 -*-
# file: algorithms/band_selection.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Choosing the Bloch states a set of Wannier functions is built from.

Hand-set energy windows (:func:`~mandacaru.algorithms.wannier.
resolve_windows`) say *where* the target bands lie; this module chooses the
states by *what* they are.

**Projectability** (the atomic-valence-active-space idea of Sayfutyarova et
al. 2017 carried to Bloch states; Qiao, Pizzi and Marzari 2023).  For target
atomic orbitals :math:`\{g_j\}` -- the first-zeta basis orbitals the
``guess`` names, e.g. the d shell of every Cu -- each Bloch state's weight in
their span,

.. math::

    p_{n\mathbf k} = \sum_j |\langle\tilde g_{j\mathbf k}|\psi_{n\mathbf k}
        \rangle|^2
    = c_{n\mathbf k}^\dagger S T\,(T^\dagger S T)^{-1}\,T^\dagger S\,
        c_{n\mathbf k},

with :math:`\tilde g` the targets Loewdin-orthogonalized among themselves,
:math:`T` their coefficients in the basis, :math:`S(\mathbf k)` the PAW
overlap and :math:`c_{n\mathbf k}` the state.  It lies in :math:`[0, 1]`
and, over every band of the basis, sums to the number of targets at each
k-point -- the targets lie in the basis, so the overlaps are exact,
augmentation included.  The frozen window is the widest energy window
whose every state has :math:`p \ge p_\text{frozen}` (:func:`frozen_window`);
states with :math:`p < p_\text{outer}` are left out (Qiao, Pizzi and
Marzari's lower threshold), and the subspace of the rest is disentangled as
for energy windows (:func:`~mandacaru.algorithms.wannier.disentangle`, which
takes the two sets as masks unchanged); the number of functions is the
number of targets.  A window, not every state above the threshold: freezing
target-like states far up in energy -- silicon's antibonding states 10 eV
above the gap, copper's s-d states 8 eV above the Fermi level -- tied the
subspace to them and broke the interpolation of the bands that matter
(HISTORY, "Choosing the target space").

**Selected columns of the density matrix** (Damle, Lin and Ying 2018):
no windows at all.  The projections onto the targets are weighted by
:math:`f(\varepsilon) = \tfrac12\operatorname{erfc}((\varepsilon -
\mu)/\sigma)` and Loewdin-orthonormalized, which gives the subspace and the
starting gauge in one step, :math:`U(\mathbf k) = f A (A^\dagger f^2
A)^{-1/2}`.  Without a given :math:`(\mu, \sigma)` they come from an erfc
fit of the projectability against energy, :math:`\mu = \mu_\text{fit} -
3\sigma_\text{fit}`, :math:`\sigma = \sigma_\text{fit}` (Vitale et al. 2020).

A third target space -- the most correlated RPA natural orbitals at each
k-point -- comes from :mod:`~mandacaru.algorithms.rpa_density` and enters
through the same projectability masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import HARTREE_TO_EV

#: Projectability every state of the frozen window must reach
#: (:func:`frozen_window`) with ``windows="auto"``.  Measured on Si sp3
#: (HISTORY, "Choosing the target space"): with 0.95 the window ends 1.1-1.4
#: eV above the Fermi level, where a hand-set one ends at 1, and the
#: interpolation below E_F + 1 eV matches the hand-set windows' (6^3: 15.6
#: against 14.6 meV); 0.9 reaches 3.3-3.9 eV and doubles the error.
DEFAULT_FROZEN_PROJECTABILITY = 0.95

#: Projectability below which a state is left out of the disentanglement.
#: Measured with the frozen threshold above, Si sp3 on 6^3: 0.01, 0.02 and
#: 0.05 give Omega_I 8.41, 8.85 and 10.51 A^2 (hand-set 9.35) and 25, 15.6
#: and 13 meV of interpolation error; 0.02 is the balance.
DEFAULT_OUTER_PROJECTABILITY = 0.02

#: Points of the sphere quadrature that expresses a trial orbital's angular
#: part in the basis' complex harmonics (exact up to degree 2 n - 1).
_SPHERE_ORDER = 12


# --------------------------------------------------------------------------- #
# The targets in the basis.
# --------------------------------------------------------------------------- #

def _sphere_rule(n: int = _SPHERE_ORDER):
    """Unit vectors ``(3, npts)`` and weights of a Gauss-Legendre by uniform
    azimuth product rule on the sphere."""
    t, wt = np.polynomial.legendre.leggauss(n)
    phi = np.linspace(0.0, 2.0 * np.pi, 2 * n, endpoint=False)
    T, P = np.meshgrid(t, phi, indexing="ij")
    s = np.sqrt(1.0 - T ** 2)
    points = np.stack([np.ravel(s * np.cos(P)), np.ravel(s * np.sin(P)),
                       np.ravel(T)])
    return points, np.ravel(np.outer(wt, np.full(2 * n, np.pi / n)))


def shell_functions(crystal, atom: int, l: int) -> dict:
    """``{m: index}``: the basis functions of the first zeta of angular
    momentum ``l`` on ``atom`` -- the same radial function
    :func:`~mandacaru.algorithms.wannier.basis_radials` picks (a
    polarization function only where there is no other)."""
    candidates = [mu for mu, (f, a) in enumerate(zip(crystal.basis,
                                                     crystal.atom_of_orbital))
                  if a == atom and int(f.l) == l]
    if not candidates:
        raise ValueError(f"the basis has no l = {l} orbital on atom {atom}: "
                         "the projectability takes the targets from the "
                         "atoms' own basis orbitals")

    def key(mu):
        f = crystal.basis[mu]
        return (bool(getattr(f, "polarization", False)),
                int(getattr(f, "zeta", 1)), int(getattr(f, "n", 0)))

    first = min(key(mu) for mu in candidates)
    out = {}
    for mu in candidates:
        if key(mu) == first:
            out.setdefault(int(crystal.basis[mu].m), mu)
    return out


def target_vectors(crystal, trials) -> np.ndarray:
    r"""``(M, J)``: the trial orbitals as basis vectors, one column each.

    Each trial must sit on an atom (:attr:`~mandacaru.algorithms.wannier.
    Trial.atom`); its angular part -- real harmonics or hybrids in its
    frame -- is expanded in the complex harmonics of that atom's first-zeta
    basis orbitals of each ``l`` (by quadrature on the sphere), and the
    radial part is the basis orbital's own.  So the targets are atomic
    orbitals of the basis itself, whatever radial the trials of the
    starting gauge use."""
    from ..basis._angular import spherical_coords, spherical_harmonic
    from .wannier import real_harmonic

    points, weights = _sphere_rule()
    _r, theta, phi = spherical_coords(*points, (0.0, 0.0, 0.0))
    T = np.zeros((crystal.M, len(trials)), dtype=complex)
    for j, trial in enumerate(trials):
        if trial.atom is None:
            raise ValueError(
                f"the target {trial.label or j!r} does not sit on an atom: "
                "projectabilities need atomic orbitals as targets -- name "
                "them by atom, e.g. guess=[('Si', 'sp3')] or {'Cu': 'd'}")
        local = (points if trial.frame is None
                 else np.asarray(trial.frame, dtype=float) @ points)
        for l, m, c in trial.angular:
            shell = shell_functions(crystal, int(trial.atom), int(l))
            values = real_harmonic(l, m, *local)
            for m_basis, mu in shell.items():
                Y = spherical_harmonic(l, m_basis, theta, phi)
                T[mu, j] += c * np.sum(weights * np.conj(Y) * values)
    norms = np.linalg.norm(T, axis=0)
    if np.any(norms < 1e-8):
        raise ValueError("a target orbital has no component in the basis")
    return T


def projectability(data, vectors, targets) -> np.ndarray:
    r"""``(nk, nb)``: each state's weight in the span of the ``targets``
    (``(M, J)`` basis vectors), :math:`c^\dagger S T (T^\dagger S T)^{-1}
    T^\dagger S c`, from the k-point matrices ``data`` (their PAW overlap)
    and the ``S``-orthonormal ``vectors`` (``(M, nb)`` per k-point)."""
    out = []
    for d, v in zip(data, vectors):
        ST = d.overlap @ targets
        gram = targets.conj().T @ ST
        values, U = np.linalg.eigh(0.5 * (gram + gram.conj().T))
        if values.min() <= 1e-10 * values.max():
            raise ValueError("the target orbitals are linearly dependent")
        inverse_root = (U / np.sqrt(values)) @ U.conj().T
        amplitudes = v.conj().T @ ST @ inverse_root        # (nb, J)
        out.append(np.sum(np.abs(amplitudes) ** 2, axis=1))
    return np.clip(np.array(out), 0.0, 1.0)


def target_projections(data, vectors, targets) -> np.ndarray:
    r"""``(nk, nb, J)``: :math:`\langle\psi_{n\mathbf k}|g_{j\mathbf
    k}\rangle` with the PAW overlap, the targets as given (not
    orthogonalized)."""
    return np.array([v.conj().T @ d.overlap @ targets
                     for d, v in zip(data, vectors)])


# --------------------------------------------------------------------------- #
# The projectabilities a user inspects.
# --------------------------------------------------------------------------- #

@dataclass
class Projectabilities:
    r"""Each Bloch state's projectability onto target atomic orbitals.

    ``values[k, n]`` (``[s, k, n]`` for a spin-polarized crystal) is
    :math:`p_{n\mathbf k}` of band ``bands[n]`` at the fractional
    ``kpoints[k]``, ``energies`` the band energies (eV, on the scale of
    :meth:`~mandacaru.algorithms.dft.DFTDriver.get_fermi_level`),
    ``labels`` the targets (``"Cu0:dxy"``), ``fermi_level`` in eV.
    """

    kpoints: np.ndarray
    energies: np.ndarray
    values: np.ndarray
    bands: tuple
    labels: tuple
    fermi_level: float
    size: tuple | None = None

    @property
    def n_targets(self) -> int:
        return len(self.labels)

    def frozen(self, threshold: float = DEFAULT_FROZEN_PROJECTABILITY):
        """Boolean mask of the states ``windows="auto"`` would freeze."""
        return self.values >= float(threshold)

    def outer(self, threshold: float = DEFAULT_OUTER_PROJECTABILITY):
        """Boolean mask of the states it would disentangle from."""
        return self.values >= float(threshold)

    def erfc_fit(self) -> tuple:
        r"""``(mu, sigma)`` (eV) of the least-squares fit
        :math:`p \approx \tfrac12\operatorname{erfc}((\varepsilon -
        \mu)/\sigma)` over every state (Vitale et al. 2020)."""
        return erfc_fit(self.energies, self.values)

    def summary(self) -> str:
        """Per band: the energy range (eV, from the Fermi level) and the
        projectability range over the k-points."""
        energies = np.reshape(self.energies, (-1,) + self.energies.shape[-2:])
        values = np.reshape(self.values, energies.shape)
        lines = [f"Projectability onto {self.n_targets} targets "
                 f"({', '.join(self.labels[:6])}"
                 f"{', ...' if self.n_targets > 6 else ''})",
                 "  spin band   E - E_F (eV)          p min    p mean   "
                 "p max"]
        for s, (e, p) in enumerate(zip(energies, values)):
            for n, band in enumerate(self.bands):
                lo, hi = e[:, n].min(), e[:, n].max()
                lines.append(
                    f"  {s:<4d} {band:<4d} {lo - self.fermi_level:8.3f} .. "
                    f"{hi - self.fermi_level:8.3f}   {p[:, n].min():7.4f}  "
                    f"{p[:, n].mean():7.4f}  {p[:, n].max():7.4f}")
        return "\n".join(lines)


def erfc_fit(energies, values) -> tuple:
    r"""``(mu, sigma)`` of :math:`\tfrac12\operatorname{erfc}((\varepsilon
    - \mu)/\sigma)` fitted to the projectabilities ``values`` at
    ``energies`` (same units out as in)."""
    from scipy.optimize import least_squares
    from scipy.special import erfc

    e = np.ravel(np.asarray(energies, dtype=float))
    p = np.ravel(np.asarray(values, dtype=float))
    span = float(e.max() - e.min()) or 1.0

    def residual(x):
        return 0.5 * erfc((e - x[0]) / abs(x[1])) - p

    # The least squares has local minima (a wide erfc through a scattered
    # tail): start from a grid of centers and widths, keep the best.
    best = None
    for mu in np.quantile(e, np.linspace(0.05, 0.95, 10)):
        for sigma in span * np.array([0.003, 0.01, 0.03, 0.1]):
            fit = least_squares(residual, (float(mu), float(sigma)))
            if best is None or fit.cost < best.cost:
                best = fit
    return float(best.x[0]), float(abs(best.x[1]))


# --------------------------------------------------------------------------- #
# The windows that select states.
# --------------------------------------------------------------------------- #

@dataclass
class Selection:
    """A resolved state selection: ``kind`` ``"projectability"``,
    ``"scdm"`` or ``"natural"``; the projectability thresholds ``outer``
    and ``frozen``; SCDM's ``mu`` and ``sigma`` (eV, ``None`` to fit them);
    the natural-orbital result ``natural`` (``None`` to compute it)."""

    kind: str
    outer: float = DEFAULT_OUTER_PROJECTABILITY
    frozen: float = DEFAULT_FROZEN_PROJECTABILITY
    mu: float | None = None
    sigma: float | None = None
    natural: object = field(default=None, repr=False)


def _thresholds(value) -> tuple:
    try:
        low, high = (float(x) for x in value)
    except (TypeError, ValueError) as error:
        raise ValueError("projectability thresholds are a pair (outer, "
                         f"frozen); got {value!r}") from error
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("projectability thresholds must satisfy 0 <= outer "
                         f"< frozen <= 1; got ({low}, {high})")
    return low, high


def resolve_selection(windows) -> Selection | None:
    """The :class:`Selection` a ``windows`` value names, or ``None`` for
    energy windows (a dict of ``"outer"``/``"frozen"`` in eV) and no
    windows.

    - ``"auto"``: projectability onto the guess' atomic orbitals, the
      default thresholds; ``{"projectability": (outer, frozen)}`` sets them;
    - ``"scdm"``: the erfc-weighted projection, ``mu`` and ``sigma`` fitted;
      ``{"scdm": (mu, sigma)}`` (eV) sets them;
    - ``"rpa"``: projectability onto the most correlated RPA natural
      orbitals (computed with their defaults), or the
      :class:`~mandacaru.algorithms.rpa_density.CrystalNaturalOrbitals`
      to use; ``{"rpa": result, "projectability": (outer, frozen)}``.
    """
    from .rpa_density import CrystalNaturalOrbitals

    if windows is None or isinstance(windows, Selection):
        return windows
    if isinstance(windows, CrystalNaturalOrbitals):
        return Selection("natural", natural=windows)
    if isinstance(windows, str):
        names = {"auto": "projectability", "scdm": "scdm", "rpa": "natural"}
        if windows not in names:
            raise ValueError(f"unknown windows {windows!r}; use 'auto', "
                             "'scdm', 'rpa' or {'outer': (lo, hi), 'frozen': "
                             "(lo, hi)} in eV")
        return Selection(names[windows])
    if not isinstance(windows, dict):
        return None
    keys = set(windows)
    if keys <= {"outer", "frozen"}:
        return None
    if "scdm" in keys:
        if keys != {"scdm"}:
            raise ValueError("'scdm' takes no other window: {'scdm': (mu, "
                             "sigma)} in eV, or 'scdm' to fit them")
        value = windows["scdm"]
        if value is None or value == "auto":
            return Selection("scdm")
        try:
            mu, sigma = (float(x) for x in value)
        except (TypeError, ValueError) as error:
            raise ValueError("windows={'scdm': (mu, sigma)} in eV; got "
                             f"{value!r}") from error
        if sigma <= 0.0:
            raise ValueError(f"the SCDM width must be positive; got {sigma}")
        return Selection("scdm", mu=mu, sigma=sigma)
    unknown = keys - {"projectability", "rpa"}
    if unknown:
        raise ValueError(f"unknown window(s) {sorted(unknown)}; energy "
                         "windows are 'outer' and 'frozen' (eV), a selection "
                         "'projectability', 'scdm' or 'rpa'")
    low, high = _thresholds(windows.get("projectability", (
        DEFAULT_OUTER_PROJECTABILITY, DEFAULT_FROZEN_PROJECTABILITY)))
    if "rpa" in keys:
        natural = windows["rpa"]
        if natural is not None and natural is not True and not isinstance(
                natural, CrystalNaturalOrbitals):
            raise ValueError("windows={'rpa': ...} takes the result of "
                             "calc.natural_orbitals(method='rpa') or True")
        return Selection("natural", outer=low, frozen=high,
                         natural=natural if isinstance(
                             natural, CrystalNaturalOrbitals) else None)
    return Selection("projectability", outer=low, frozen=high)


def selection_citations(selection) -> tuple:
    """The reference keys of a :class:`Selection` (``None``: energy
    windows, the disentanglement alone)."""
    if selection is None:
        return ("Souza2001",)
    if selection.kind == "scdm":
        keys = ("Damle2017", "Damle2018")
        return keys + (("Vitale2020",) if selection.mu is None else ())
    keys = ("Souza2001", "Qiao2023")
    if selection.kind == "projectability":
        return keys + ("Sayfutyarova2017",)
    return keys + ("Scuseria2008",)


def frozen_window(energies, p, threshold: float, n_functions: int):
    """``(lo, hi)``: the widest energy window -- the longest run of states
    in energy order, over the whole mesh -- whose every state projects at
    least ``threshold`` onto the targets, narrowed from the top until no
    k-point holds more than ``n_functions`` states in it; ``None`` when no
    state reaches the threshold."""
    e = np.asarray(energies, dtype=float)
    p = np.asarray(p, dtype=float)
    flat_e, flat_p = e.ravel(), p.ravel()
    order = np.argsort(flat_e, kind="stable")
    good = flat_p[order] >= threshold
    best, start = (0, 0), None
    for i, ok in enumerate(np.append(good, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best[1] == best[0]:
        return None
    lo = float(flat_e[order[best[0]]])
    hi = float(flat_e[order[best[1] - 1]])
    J = int(n_functions)
    for k in range(e.shape[0]):
        inside = np.sort(e[k][(e[k] >= lo) & (e[k] <= hi)])
        if inside.size > J:
            hi = min(hi, float(0.5 * (inside[J - 1] + inside[J])))
    return lo, hi


def projectability_masks(p, n_functions: int, outer: float, frozen: float,
                         energies=None):
    """``(inside, fixed)`` boolean ``(nk, nb)`` masks: the states with
    projectability at least ``outer``, and the frozen ones -- with
    ``energies`` the states of the :func:`frozen_window` of ``frozen``,
    without them every state with projectability at least ``frozen`` (at
    most ``n_functions`` per k-point, the most projectable).  Where fewer
    than ``n_functions`` are inside, the most projectable of the rest
    join."""
    p = np.asarray(p, dtype=float)
    J = int(n_functions)
    if energies is None:
        fixed = p >= frozen
    else:
        e = np.asarray(energies, dtype=float)
        window = frozen_window(e, p, frozen, J)
        fixed = (np.zeros(p.shape, dtype=bool) if window is None
                 else (e >= window[0]) & (e <= window[1]))
    inside = (p >= outer) | fixed
    for k in range(p.shape[0]):
        order = np.argsort(-p[k], kind="stable")
        if np.count_nonzero(fixed[k]) > J:
            fixed[k] = False
            fixed[k, order[:J]] = True
        if np.count_nonzero(inside[k]) < J:
            inside[k, order[:J]] = True
    return inside, fixed


def scdm_subspace(energies, A, mu: float, sigma: float) -> np.ndarray:
    r"""``S[k]`` (``(nb, J)``, orthonormal columns): the Loewdin
    orthonormalization of the erfc-weighted projections
    :math:`f(\varepsilon_{n\mathbf k}) A_{nj}(\mathbf k)`, :math:`f =
    \tfrac12\operatorname{erfc}((\varepsilon - \mu)/\sigma)` (``energies``,
    ``mu`` and ``sigma`` in one unit)."""
    from scipy.special import erfc

    weights = 0.5 * erfc((np.asarray(energies) - mu) / sigma)
    out = np.empty(A.shape, dtype=complex)
    for k, a in enumerate(A):
        u, s, vh = np.linalg.svd(weights[k][:, None] * a,
                                 full_matrices=False)
        if s.min() < 1e-10 * max(s.max(), 1e-300):
            raise ValueError(
                f"at k-point {k} the erfc-weighted projections do not span "
                f"{A.shape[2]} states: raise mu or sigma")
        out[k] = u @ vh
    return out


def select_states(selection: Selection, *, crystal, data, vectors, energies,
                  trials, n_functions: int, bands, fermi_level=None):
    """The states of one channel a :class:`Selection` picks, for
    :func:`~mandacaru.algorithms.wannier.channel_wannier_functions`:
    ``("masks", (inside, fixed), record)`` for the projectability kinds,
    ``("subspace", S, record)`` for SCDM.  ``energies`` (Hartree) and
    ``vectors`` are the mesh states of ``bands``; ``record`` is what the
    result's ``windows`` reports (eV)."""
    J = int(n_functions)
    if selection.kind == "natural":
        natural = selection.natural
        if natural is None:
            raise ValueError("windows='rpa' needs the natural orbitals: "
                             "compute them through the calculator, "
                             "calc.wannier(..., windows='rpa')")
        p = natural.band_projectabilities(J, bands, len(data))
        source = f"RPA natural orbitals ({J} per k-point)"
    else:
        targets = target_vectors(crystal, trials)
        if selection.kind == "scdm":
            A = target_projections(data, vectors, targets)
            e_ev = np.asarray(energies) * HARTREE_TO_EV
            mu, sigma = selection.mu, selection.sigma
            fitted = mu is None
            if fitted:
                p = projectability(data, vectors, targets)
                mu_fit, sigma_fit = erfc_fit(e_ev, p)
                mu, sigma = mu_fit - 3.0 * sigma_fit, sigma_fit
            S = scdm_subspace(e_ev, A, mu, sigma)
            record = {"outer": None, "frozen": None,
                      "selection": "scdm", "mu": mu, "sigma": sigma,
                      "fitted": fitted}
            return "subspace", S, record
        p = projectability(data, vectors, targets)
        source = "projectability"
    e_ev = np.asarray(energies) * HARTREE_TO_EV
    inside, fixed = projectability_masks(p, J, selection.outer,
                                         selection.frozen, energies=e_ev)

    def span(mask):
        if not np.any(mask):
            return None
        return (float(e_ev[mask].min()), float(e_ev[mask].max()))

    record = {"outer": span(inside), "frozen": span(fixed),
              "selection": source,
              "thresholds": (selection.outer, selection.frozen),
              "n_frozen": (int(fixed.sum(axis=1).min()),
                           int(fixed.sum(axis=1).max()))}
    return "masks", (inside, fixed), record


def selection_lines(record) -> list:
    """The lines a Wannier summary prints for a selection ``record``
    (:func:`select_states`); none for energy windows."""
    kind = record.get("selection")
    if kind is None:
        return []
    if kind == "scdm":
        how = "fitted" if record.get("fitted") else "given"
        return [f"  states by SCDM, erfc mu {record['mu']:.3f} eV, sigma "
                f"{record['sigma']:.3f} eV ({how})"]
    low, high = record["thresholds"]
    lo, hi = record["n_frozen"]
    return [f"  states by {kind}: outer p >= {low:g}, frozen window where "
            f"p >= {high:g} ({lo}..{hi} frozen per k-point)"]


def mesh_projectabilities(solver, size, bands, trials, spin: int = 0):
    """``(fractional, energies eV, p)`` of one channel on the full
    Gamma-centered ``size`` mesh (:func:`~mandacaru.algorithms.wannier.
    mesh_states`)."""
    from .wannier import mesh_states

    fractional, data, vectors, energies = mesh_states(solver, size, bands,
                                                      spin)
    targets = target_vectors(solver.crystal, trials)
    return (fractional, energies * HARTREE_TO_EV,
            projectability(data, vectors, targets))


def point_projectabilities(solver, kpoints, bands, trials, spin: int = 0):
    """``(energies eV, p)`` at Cartesian ``kpoints`` (Bohr^-1), diagonalized
    at the converged potential of channel ``spin``."""
    targets = target_vectors(solver.crystal, trials)
    V, v_tau, w, channel = solver.channel_potentials()[spin]
    reference = solver.eigenvalue_reference()
    bands = list(bands)
    energies, values = [], []
    for data, eps, vecs in solver._diagonalized(kpoints, V, v_tau, w,
                                                channel):
        energies.extend(e[bands] + reference for e in eps)
        values.extend(projectability(data, [v[:, bands] for v in vecs],
                                     targets))
    return np.array(energies) * HARTREE_TO_EV, np.array(values)


__all__ = ["DEFAULT_FROZEN_PROJECTABILITY", "DEFAULT_OUTER_PROJECTABILITY",
           "Projectabilities", "Selection", "erfc_fit", "frozen_window",
           "projectability",
           "projectability_masks", "resolve_selection", "scdm_subspace",
           "select_states", "selection_citations", "selection_lines", "shell_functions", "target_projections",
           "target_vectors"]
