# -*- coding: utf-8 -*-
# file: algorithms/crystal_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Forces and stress of a converged periodic Kohn-Sham crystal.

Forces
------
The free energy :math:`F = E - \sigma S` of the converged state is stationary
in the density matrices under the orthonormality constraint, so its total
derivative needs no response of the states:

.. math::

    \frac{dF}{d\mathbf R_A} = \sum_{\mathbf k} w_{\mathbf k}\,
    \mathrm{Re}\,\mathrm{tr}\Bigl[P_{\mathbf k}
      \frac{\partial H_{\mathbf k}}{\partial\mathbf R_A}
    - W_{\mathbf k}\frac{\partial S_{\mathbf k}}{\partial\mathbf R_A}\Bigr]
    + \frac{\partial E_{\rm explicit}}{\partial\mathbf R_A},

with :math:`H` the Kohn-Sham matrix at the frozen potential (the
electrostatic and exchange-correlation potentials of the converged density
and the derivatives ``w`` of the energy with respect to the compensation
moments), :math:`P = \sum_n f_n c_nc_n^\dagger` and the energy-weighted
:math:`W = \sum_n f_n\varepsilon_n c_nc_n^\dagger` -- with the raw
eigenvalues, the ones :math:`H c = \varepsilon S c` gives; Mermin's free
energy makes the fractional occupations no exception.

The split follows the molecular gradient
(:func:`~mandacaru.algorithms.pseudo_forces.pseudo_nuclear_gradient`):

**Pulay** -- atom :math:`A`'s basis functions move: every matrix built from
the basis (kinetic, overlap, grid potential, the short-range local spheres,
the projections and the compensation moments they make) changes.  The
Cartesian derivatives of the functions are in the basis of one extended
build (analytic Bloch sums of :math:`\partial(R\,Y_{lm})`, sampled on the
same grid and quadratures, so the force is the derivative of the
*discretized* energy), and every matrix's derivative is read off it.

**Hellmann-Feynman** -- atom :math:`A`'s operators move: its projectors (the
nonlocal term, the augmented overlap and its compensation moments), its
short-range local sphere, its compensation shapes and Gaussian ion in the
reciprocal-space electrostatics (analytic: a translation multiplies a
transform by :math:`-i\mathbf G`), the short-range ion-compensation and
erfc ion-pair sums, and its partial core in the exchange-correlation
energy.

The state is the SCF's own irreducible k-points, diagonalized at the frozen
potential; the gradient summed over them with their weights is the full
mesh's after symmetrization (:func:`symmetrize_vectors`).

Stress
------
:math:`\sigma_{\alpha\beta} = \Omega^{-1}\,\partial F/\partial
\varepsilon_{\alpha\beta}` by straining the cell, the atoms and the grid
together (the same node counts), at fixed occupations and with the states
carried along in the symmetrically orthogonalized frame,
:math:`C(\varepsilon) = S(\varepsilon)^{-1/2}S(0)^{1/2}C(0)` -- which keeps
them orthonormal, so the difference is the full derivative with no overlap
term, and the entropy does not change.  Only the strains the space group
leaves unchanged are applied (:func:`invariant_strains`: one for a cubic
crystal, six without symmetry), each keeping the irreducible k-points: two
rebuilds of a cubic crystal.

Spin polarization
-----------------
A spin-polarized crystal is re-diagonalized per channel, with one Fermi
level over both channels' levels and one electron per state.  Its matrix
terms are sums over the channels, :math:`\sum_s\mathrm{tr}(P_s\,\partial
H_s) - \mathrm{tr}(W_s\,\partial S)`, each channel's :math:`H_s` at its
own potential (and, for a hybrid, its own :math:`-aK_s`); the electrostatic
terms see the total density and moments; each channel's functional took
half the partial core, so the core moves under the mean of the two
channels' potentials.  The stress carries each channel's states and
evaluates the spin energy.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

from ..integrals import reciprocal as rc
from ..integrals._backend import single_threaded_blas
from .periodic_device import make_device
from .periodic_dft import entropy, fermi_level, occupation

#: Displacement (Bohr) of the compensation charges, ion sums and partial cores
#: whose derivatives are central differences (the basis and the operators'
#: matrices are differentiated analytically), Richardson-extrapolated with
#: half the step (:func:`_richardson`): a cramped cell (O2 in a 4 A box) put
#: 6e-5 eV/A into the plain central difference.
DEFAULT_DELTA = 1.0e-3

#: Half-amplitude of the symmetric strains of the stress.
DEFAULT_STRAIN = 1.0e-4


# --------------------------------------------------------------------------- #
# The state on the full mesh.
# --------------------------------------------------------------------------- #

def _operations(c):
    """``[(R, atom_map)]``: the Cartesian rotation and atom map of every
    operation the crystal was reduced by -- the identity alone without
    symmetry.  ``atom_map[A]`` is the atom at the image of atom ``A``."""
    if c.symmetry is None:
        return [(np.eye(3), np.arange(len(c.centers)))]
    L = np.asarray(c.lattice, dtype=float)
    L_inv = np.linalg.inv(L)
    out = []
    for W, atom_map in zip(c.symmetry.info.rotations, c.symmetry.atom_maps):
        R = L @ np.asarray(W, dtype=float) @ L_inv
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-8):
            raise RuntimeError("a symmetry operation is not a Cartesian "
                               "rotation: the lattice convention is off")
        out.append((R, np.asarray(atom_map, dtype=int)))
    return out


def symmetrize_vectors(c, values) -> np.ndarray:
    r"""The symmetric part of a per-atom vector field (``(n_atoms, 3)``):
    :math:`\bar F_A = \frac1N\sum_g R_g^{-1} F_{g(A)}`.

    A sum over the irreducible k-points with their weights is the full mesh's
    up to this projection: each point stands for its star, whose members'
    contributions are the symmetry images of its own.
    """
    values = np.asarray(values, dtype=float)
    operations = _operations(c)
    out = np.zeros_like(values)
    for R, atom_map in operations:
        out += values[atom_map] @ R          # rows: R^T v = R^{-1} v
    return out / len(operations)


def invariant_strains(c) -> list:
    """An orthonormal (Frobenius) basis of the symmetric strains every
    operation leaves unchanged, :math:`R E R^T = E`: one for a cubic
    crystal, six without symmetry.  The stress lies in their span, and a
    strain in it keeps the space group -- and so the irreducible k-points."""
    basis = []
    for a in range(3):
        for b in range(a, 3):
            E = np.zeros((3, 3))
            E[a, b] = E[b, a] = 1.0
            basis.append(E / np.linalg.norm(E))
    operations = _operations(c)
    P = np.zeros((6, 6))
    for col, E in enumerate(basis):
        image = sum(R @ E @ R.T for R, _m in operations) / len(operations)
        P[:, col] = [np.sum(image * F) for F in basis]
    values, vectors = np.linalg.eigh(0.5 * (P + P.T))
    keep = values > 0.5                       # a projector: eigenvalues 0, 1
    return [sum(v * F for v, F in zip(vector, basis))
            for vector in vectors[:, keep].T]


# --------------------------------------------------------------------------- #
# The hybrid's terms: every piece of the Kohn-Sham matrix that depends on how
# the solver stores and builds its exact exchange is behind this helper.
# --------------------------------------------------------------------------- #

def _hybrid_hamiltonian_terms(solver, channel: int, *, device=None,
                              kpoint_data=None) -> np.ndarray:
    r"""Spin channel ``channel``'s hybrid correction to its Kohn-Sham matrix,
    ``(nk, M, M)``: :math:`-s\,K + C\,O\,C^\dagger` with :math:`K` the
    short-range exchange of the channel's occupied states of the output
    density matrices, held fixed, :math:`O` the spheres' one-center operator
    of those states, and ``s``
    :meth:`~.periodic_dft.PeriodicKohnSham._exchange_scale` (:math:`a/2`
    when a state holds both spins, :math:`a` in a channel).

    Over the SCF basis through ``device`` (the re-diagonalization of
    :func:`kpoint_state`), or over any basis -- the extended one of the
    Pulay terms -- from ``kpoint_data`` (its own projections).
    """
    states, operators = solver._hybrid_output
    if device is not None:
        kpoint_data = device.kpoint_data
        K = device.exchange_matrices(solver.exchange, states[channel])
    else:
        K = solver.exchange.matrices(kpoint_data, states[channel])
    O = operators[channel]
    scale = solver._exchange_scale()
    return np.stack([-scale * Kk + d.projections @ O @ d.projections.conj().T
                     for Kk, d in zip(K, kpoint_data)])


def kpoint_state(solver):
    """A dict of the converged crystal's state (``kpoints``, ``weights``,
    ``P``, ``W``, ``potentials``, per-channel entries for a spin crystal) on
    its own irreducible k-points (the SCF's: by the space group and time
    reversal, or time reversal alone without symmetry).

    The Kohn-Sham matrix is built from the potentials of the SCF's output
    density -- the density the reported energy is a functional of -- and
    diagonalized at every point; the Fermi level is found again on these
    eigenvalues.  A quantity summed over these points with their weights is
    the full mesh's after :func:`symmetrize_vectors` (a force) or the
    projection on :func:`invariant_strains` (a stress).

    A spin-polarized crystal is diagonalized per channel, each at its own
    potential (and, for a hybrid, its own exchange), with one Fermi level
    over both channels' levels and one electron per state, as its SCF.
    ``channels`` holds one entry per channel -- its density and
    energy-weighted matrices, states, occupations and potentials -- and
    ``P`` and ``W`` are their sums over the channels (the
    single channel of a restricted crystal, which holds both spins).
    """
    c = solver.crystal
    spin = solver.n_spins == 2
    rho, q, tau = solver.output_density
    if spin:
        up, dn = rho
        V, v_tau, w, terms = solver._potentials_spin(up, dn, q, *tau)
        potentials = list(zip(V, v_tau))
        density = (up + dn, q, tau)
    else:
        V, v_tau, w, terms = solver._potentials(rho, q, tau)
        potentials = [(V, v_tau)]
        density = (rho, q, tau)
    n_channels = len(potentials)
    kpoints = np.asarray(c.kpoints, dtype=float)
    weights = np.asarray(c.weights, dtype=float)
    B = rc.reciprocal_vectors(c.lattice)
    points = kpoints @ np.linalg.inv(B).T
    eigenvalues = [[] for _s in range(n_channels)]
    vectors = [[] for _s in range(n_channels)]
    overlaps = []
    block = c.kpoint_block()
    for start in range(0, len(kpoints), block):
        data = c.kpoint_matrices(kpoints[start:start + block])
        gradients = ([c.bloch_gradients(d.psi, d.k) for d in data]
                     if solver.meta else None)
        device = make_device(c, data, gradients)
        for s, (V_s, v_tau_s) in enumerate(potentials):
            H = device.hamiltonians(V_s, v_tau_s, w)
            if solver.screening is not None:
                H = H + _hybrid_hamiltonian_terms(solver, s, device=device)
            eps, C = device.eigensolve(H)
            eigenvalues[s].extend(eps)
            vectors[s].extend(C)
        overlaps.extend(d.overlap for d in data)
    if spin:
        # One electron per state, one Fermi level over both channels.
        mu = fermi_level(eigenvalues[0] + eigenvalues[1], list(weights) * 2,
                         solver.n_electrons, solver.method, solver.width,
                         degeneracy=1.0)
        degeneracy = 1.0
    else:
        mu = fermi_level(eigenvalues[0], weights, solver.n_electrons,
                         solver.method, solver.width)
        degeneracy = 2.0
    channels = []
    for s in range(n_channels):
        P, W, occupations = [], [], []
        for eps, C in zip(eigenvalues[s], vectors[s]):
            x = (eps - mu) / solver.width
            f = degeneracy * occupation(x, solver.method)
            occupations.append(f)
            P.append((C * f) @ C.conj().T)
            W.append((C * (f * eps)) @ C.conj().T)
        channels.append({
            "P": P, "W": W, "vectors": vectors[s],
            "occupations": occupations, "eigenvalues": eigenvalues[s],
            "V": potentials[s][0], "v_tau": potentials[s][1]})
    entropy_term = -solver.width * sum(
        wk * degeneracy * float(np.sum(entropy((e - mu) / solver.width,
                                               solver.method)))
        for channel in channels
        for wk, e in zip(weights, channel["eigenvalues"]))
    if spin:
        P_total = [a + b for a, b in zip(channels[0]["P"], channels[1]["P"])]
        W_total = [a + b for a, b in zip(channels[0]["W"], channels[1]["W"])]
    else:
        P_total, W_total = channels[0]["P"], channels[0]["W"]
    return {"kpoints": kpoints, "weights": np.asarray(weights, float),
            "fractional": points, "P": P_total, "W": W_total,
            "channels": channels, "spin": spin,
            "overlaps": overlaps, "fermi_level": mu,
            "entropy_term": entropy_term,
            "potentials": (V, v_tau, w, terms), "density": density,
            "channel_densities": rho if spin else None}


def _trace(A, B) -> float:
    """``Re tr(A B)``."""
    return float(np.real(np.sum(A * B.T)))


# --------------------------------------------------------------------------- #
# Forces.
# --------------------------------------------------------------------------- #

def _matrix_terms(solver, state):
    r"""``(pulay, hellmann_feynman)``, ``(n_atoms, 3)`` each: every term of
    the form ``tr(P dH) - tr(W dS)``.

    One extended build serves them all: the basis plus the Cartesian
    derivative of every function (:class:`~mandacaru.pseudopotentials.
    periodic_paw.BasisGradient`, analytic).  Moving atom :math:`A` by
    :math:`\delta` changes its functions by :math:`-\delta\,\partial
    \chi`, so a basis matrix changes by :math:`-(O_{\partial\mu,\nu} +
    O_{\mu,\partial\nu})` over the pairs that involve :math:`A`'s
    functions -- the Pulay term.  Moving an operator is the opposite of
    moving *all* the functions under it -- a rigid translation of basis and
    operator together changes nothing, at fixed k -- so a projector's column
    of the projections and a short-range sphere's matrix are differentiated
    by all the derivatives, with the sign reversed.

    A spin-polarized crystal's Pulay term is a sum over its channels,
    :math:`\sum_s \mathrm{tr}(P_s\,dH_s) - \mathrm{tr}(W\,dS)`, each
    channel's matrix at its own potential (and exchange); the operators
    that move -- spheres, projectors, the compensation moments' ``w`` -- are
    the same in both channels and see the summed ``P`` and ``W``.
    """
    from ..pseudopotentials.periodic_paw import BasisGradient, KPointMatrices

    c = solver.crystal
    _V, _v_tau, w, _terms = state["potentials"]
    channels = state["channels"]
    M, n_atoms = c.M, len(c.centers)
    owner = np.asarray(c.atom_of_orbital)
    extended = list(c.basis) + [BasisGradient(f, axis) for axis in range(3)
                                for f in c.basis]
    grad = [np.arange(M * (1 + axis), M * (2 + axis)) for axis in range(3)]
    ext = copy.copy(c)
    ext.basis = extended
    ext.atom_of_orbital = list(c.atom_of_orbital) * 4
    ext.M = len(extended)

    pulay = np.zeros((n_atoms, 3))
    hf = np.zeros((n_atoms, 3))
    kpoints, weights = state["kpoints"], state["weights"]
    block = ext.kpoint_block()
    base = np.arange(M)
    for start in range(0, len(kpoints), block):
        chunk = kpoints[start:start + block]
        psi_all = ext.bloch_sums(chunk)
        C_all = ext.projections(chunk)
        spheres = [ext.short_range_atom(atom, chunk)
                   for atom in range(n_atoms)]
        for i, k in enumerate(chunk):
            index = start + i
            psi, C = psi_all[i], C_all[i]
            S = (psi.conj() @ psi.T) * c.grid.dV + C @ c.q_overlap @ C.conj().T
            fixed = (ext._kinetic(psi, k) + sum(v[i] for v in spheres)
                     + C @ c.D_ion @ C.conj().T)
            data = KPointMatrices(k=np.asarray(k, float), weight=0.0, psi=psi,
                                  overlap=S, fixed=fixed, projections=C)
            gradients = (ext.bloch_gradients(psi, k) if solver.meta
                         else None)
            # The exchange is symmetric between its k side and its q side,
            # so moving every function at fixed coefficients is twice
            # moving the k side with the occupied states fixed: each
            # channel's -s K over the extended basis (s = a/2 restricted, a
            # per spin channel), through the same Pulay contraction as the
            # local terms.  Not through a device: the extended overlap need
            # not have a Cholesky factor.
            H_channels = []
            for s, channel in enumerate(channels):
                H = solver._hamiltonian(data, channel["V"], channel["v_tau"],
                                        w, gradients)
                if solver.screening is not None:
                    H = H + _hybrid_hamiltonian_terms(
                        solver, s, kpoint_data=[data])[0]
                H_channels.append(H)
            P, W, wk = state["P"][index], state["W"][index], weights[index]
            P_channels = [channel["P"][index] for channel in channels]
            C0 = C[base]
            if solver.screening is not None:
                # The compensation charges of each channel's exchange pairs:
                # their projectors and shapes move.
                states = solver._hybrid_output[0]
                for s, P_s in enumerate(P_channels):
                    hf += wk * solver.exchange.compensation_gradient(
                        SimpleNamespace(k=data.k, psi=psi[base],
                                        projections=C0),
                        states[s], P_s, [C[grad[d]] for d in range(3)],
                        solver._exchange_scale())
            for d in range(3):
                g = grad[d]
                G_Hs = [H[np.ix_(g, base)] for H in H_channels]
                G_S = S[np.ix_(g, base)]
                # Pulay: one atom's functions move.
                for atom in range(n_atoms):
                    own = (owner == atom).astype(float)
                    band = sum(
                        _trace(P_s, -(own[:, None] * G_H
                                      + G_H.conj().T * own[None, :]))
                        for P_s, G_H in zip(P_channels, G_Hs))
                    dS = -(own[:, None] * G_S + G_S.conj().T * own[None, :])
                    pulay[atom, d] += wk * (band - _trace(W, dS))
                # A short-range sphere moves: minus all functions moving.
                for atom in range(n_atoms):
                    G_V = spheres[atom][i][np.ix_(g, base)]
                    hf[atom, d] += wk * _trace(P, G_V + G_V.conj().T)
                # A projector moves: minus all functions moving, its column.
                dC_all = C[g]
                for atom in range(n_atoms):
                    columns = c.projector_columns(atom)
                    if not columns:
                        continue
                    dC = np.zeros_like(C0)
                    dC[:, columns] = dC_all[:, columns]
                    dH = (dC @ c.D_ion @ C0.conj().T
                          + C0 @ c.D_ion @ dC.conj().T)
                    dS = (dC @ c.q_overlap @ C0.conj().T
                          + C0 @ c.q_overlap @ dC.conj().T)
                    Ca, dCa = C0[:, columns], dC[:, columns]
                    for ch, blk in c.multipole_blocks.items():
                        if ch[0] == atom:
                            dH = dH + w[ch] * (dCa @ blk @ Ca.conj().T
                                               + Ca @ blk @ dCa.conj().T)
                    hf[atom, d] += wk * (_trace(P, dH) - _trace(W, dS))
                    if solver.screening is not None:
                        # The spheres' one-center operator, per channel,
                        # moves as the nonlocal projector term does.
                        for P_s, O in zip(P_channels,
                                          solver._hybrid_output[1]):
                            dO = (dC @ O @ C0.conj().T
                                  + C0 @ O @ dC.conj().T)
                            hf[atom, d] += wk * _trace(P_s, dO)
    return pulay, hf


def _ion_transform_atom(c, atom, G):
    """Atom ``atom``'s Gaussian ion at the reciprocal vectors ``G``."""
    G2 = np.sum(G * G, axis=0)
    return (-c.charges[atom] * np.exp(-0.5 * c.sigma ** 2 * G2)
            * rc.structure_factor(G, c.centers[atom]))


def _electrostatic(solver, state):
    r"""``(n_atoms, 3)``: compensation shapes and Gaussian ions move at fixed
    charges -- analytic, :math:`\partial_{\mathbf R}e^{-i\mathbf G\cdot
    \mathbf R} = -i\mathbf G\,e^{-i\mathbf G\cdot\mathbf R}`.

    The smooth density meets the compact charges on the grid's reciprocal
    set, :math:`\Omega^{-1}\sum K\,\bar{\tilde n}(\hat n + n_{\rm ion})`;
    the compact charges meet each other on the dense set, as
    :math:`(2\Omega)^{-1}\sum K\,|n_c|^2`.
    """
    from ..pseudopotentials.periodic_paw import (COMPENSATION_CUTOFF,
                                                 DENSE_BLOCK)

    c = solver.crystal
    rho, q, _tau = state["density"]
    n_atoms = len(c.centers)
    out = np.zeros((n_atoms, 3))
    rho_G = rc.to_reciprocal(c.grid, rho)
    for atom in range(n_atoms):
        source = _ion_transform_atom(c, atom, c.G)
        for channel in c.channels:
            if channel[0] == atom:
                source = source + q[channel] * solver.g_hat[channel]
        for d in range(3):
            out[atom, d] += float(np.real(np.sum(
                c.kernel * np.conj(rho_G) * (-1j * c.G[d]) * source))) \
                / c.volume

    B = rc.reciprocal_vectors(c.lattice)
    Gs = rc.lattice_translations(B, COMPENSATION_CUTOFF).T
    Gs = Gs[:, np.sum(Gs * Gs, axis=0) > 0.0]
    for start in range(0, Gs.shape[1], DENSE_BLOCK):
        G = Gs[:, start:start + DENSE_BLOCK]
        norm = rc.spherical(G)[0]
        kernel = 4.0 * np.pi / norm ** 2
        per_atom = [_ion_transform_atom(c, atom, G) for atom in range(n_atoms)]
        for atom, L, M in c.channels:
            per_atom[atom] = per_atom[atom] + q[(atom, L, M)] * \
                rc.multipole_transform(G, c.centers[atom],
                                       c._shape_transform(atom, L, norm), L, M)
        total = sum(per_atom)
        for atom in range(n_atoms):
            for d in range(3):
                out[atom, d] += float(np.real(np.sum(
                    kernel * np.conj(total) * (-1j * G[d]) * per_atom[atom]))) \
                    / c.volume
    return out


def _richardson(central, delta):
    """``(4 D(delta/2) - D(delta)) / 3`` of the central difference
    ``central(h)``: the step's h^2 error cancels."""
    return (4.0 * central(0.5 * delta) - central(delta)) / 3.0


def _displaced(function, centers, atom, d, delta):
    """Central difference of ``function(centers)`` as ``atom`` moves along
    ``d``."""
    values = []
    for sign in (+1.0, -1.0):
        moved = [np.array(x, dtype=float) for x in centers]
        moved[atom][d] += sign * delta
        values.append(function(moved))
    return values


def _sum_terms(solver, state, atom, delta):
    """``(3,)``: the short-range ion-compensation and erfc ion-pair sums."""
    c = solver.crystal
    _rho, q, _tau = state["density"]
    out = np.zeros(3)
    for d in range(3):
        def central(h, d=d):
            plus, minus = _displaced(
                lambda centers: c.short_range_ion_compensation(centers,
                                                               atom=atom),
                c.centers, atom, d, h)
            value = float(np.real(sum(q[ch] * (plus[ch] - minus[ch])
                                      for ch in c.channels))) / (2.0 * h)
            plus, minus = _displaced(c.ion_constants, c.centers, atom, d, h)
            return value + (plus - minus) / (2.0 * h)
        out[d] = _richardson(central, delta)
    return out


def _core_potentials(solver, state):
    """``(potential, tau_potential)`` the partial core moves under.

    Restricted: the exchange-correlation potential of the valence plus the
    core.  Spin-polarized: each channel's functional took half the core, so
    the core moves under the mean of the two channels' potentials.  A hybrid
    replaces the short-range semilocal exchange of the valence (channels)
    only; the core sees the potential without it -- ``screening=(omega,
    0)``.
    """
    from ..integrals.exchange_correlation import evaluate, evaluate_spin

    c = solver.crystal
    _V, _v_tau, _w, terms = state["potentials"]
    if not state["spin"]:
        potential = terms.potential
        if solver.screening is not None:
            rho, _q, _tau = state["density"]
            core = 0.0 if solver.core is None else solver.core
            potential = evaluate(c.grid, rho + core, solver.functional,
                                 relativistic=solver.relativistic,
                                 screening=(solver.screening[0],
                                            0.0)).potential
        return potential, terms.tau_potential
    up, dn = state["channel_densities"]
    half = 0.0 if solver.core is None else 0.5 * solver.core
    if solver.screening is not None:
        tau_up, tau_dn = state["density"][2]
        half_tau = (0.0 if solver._core_tau is None
                    else 0.5 * solver._core_tau)
        terms = evaluate_spin(
            c.grid, up + half, dn + half, solver.functional,
            relativistic=solver.relativistic,
            tau_up=tau_up + half_tau if solver.meta else None,
            tau_dn=tau_dn + half_tau if solver.meta else None,
            screening=(solver.screening[0], 0.0), exchange_densities=(up, dn))
    tau_potential = (None if terms.tau_potential_up is None
                     else 0.5 * (terms.tau_potential_up
                                 + terms.tau_potential_dn))
    # Each channel took half the core; where a channel's density is
    # negative the functional clipped it and reports no potential there
    # (`exchange_correlation._clipped`).
    potential = 0.5 * (terms.potential_up + terms.potential_dn)
    return potential, tau_potential


def _core_terms(solver, state, atom, delta, potentials=None):
    """``(3,)``: ``atom``'s partial core (and its kinetic-energy density)
    moves under the exchange-correlation potential
    (:func:`_core_potentials`, computed here unless given)."""
    from ..integrals.exchange_correlation import (core_tau_function,
                                                  xc_core_density)

    c = solver.crystal
    potential, tau_potential = (_core_potentials(solver, state)
                                if potentials is None else potentials)
    out = np.zeros(3)
    tables = [(xc_core_density, potential)]
    if solver.meta and tau_potential is not None:
        def tau_table(dataset):
            function = core_tau_function(dataset)
            return (None if function is None
                    else function(np.asarray(dataset.r, dtype=float)))
        tables.append((tau_table, tau_potential))
    for table, potential in tables:
        if c.place_core(atom, table) is None:
            continue
        for d in range(3):
            def central(h, d=d, table=table, potential=potential):
                shift = np.zeros(3)
                shift[d] = h
                plus = c.place_core(atom, table, c.centers[atom] + shift)
                minus = c.place_core(atom, table, c.centers[atom] - shift)
                return float(np.sum(potential * (plus - minus))) \
                    * c.grid.dV / (2.0 * h)
            out[d] += _richardson(central, delta)
    return out


@single_threaded_blas
def crystal_gradient(solver, *, delta: float = DEFAULT_DELTA,
                     include_pulay: bool = True):
    r"""Analytic :math:`dF/d\mathbf R` of a converged
    :class:`~mandacaru.algorithms.periodic_dft.PeriodicKohnSham`.

    Summed over the run's irreducible k-points and symmetrized
    (:func:`symmetrize_vectors`): the full mesh's gradient at a fraction of
    its cost (8 of 64 points for diamond on 4x4x4).  Returns a
    :class:`~mandacaru.algorithms.forces.ForceResult` in eV/Angstrom whose
    ``hellmann_feynman`` and ``pulay`` split the gradient as the module
    docstring describes; ``details["net_force_vector"]`` is their sum over
    atoms, zero but for the grid's egg-box.

    The matrix terms use the states of the output density's potential and the
    explicit terms that density itself; the two differ by the SCF residual,
    to first order -- converge tightly (``density_tol``) for a precise force.

    A spin-polarized crystal's matrix terms are summed over its channels
    (:func:`kpoint_state`, :func:`_matrix_terms`); the electrostatic terms
    see the total density and moments, and the partial core the mean of the
    channels' potentials (each channel's functional took half of it).
    """
    from .forces import HA_BOHR_TO_EV_ANGSTROM, ForceResult

    if solver.output_density is None:
        raise RuntimeError("the crystal gradient needs a converged run()")
    c = solver.crystal
    state = kpoint_state(solver)
    n_atoms = len(c.centers)
    pulay, hf = _matrix_terms(solver, state)
    if not include_pulay:
        pulay = np.zeros_like(pulay)
    hf = hf + _electrostatic(solver, state)
    core = _core_potentials(solver, state)
    for atom in range(n_atoms):
        hf[atom] += (_sum_terms(solver, state, atom, delta)
                     + _core_terms(solver, state, atom, delta, core))
    hf = symmetrize_vectors(c, hf) * HA_BOHR_TO_EV_ANGSTROM
    pulay = symmetrize_vectors(c, pulay) * HA_BOHR_TO_EV_ANGSTROM
    gradient = hf + pulay
    details = {"method": "kohn-sham (crystal)", "n_spins": solver.n_spins,
               "include_pulay":
               bool(include_pulay), "n_kpoints": len(state["kpoints"]),
               "n_symmetry_operations": len(_operations(c)),
               "net_force_vector": (-gradient).sum(axis=0)}
    return ForceResult(forces=-gradient, hellmann_feynman=hf, pulay=pulay,
                       gradient=gradient,
                       n_electrons=float(solver.n_electrons), details=details)


# --------------------------------------------------------------------------- #
# Stress.
# --------------------------------------------------------------------------- #

def _matrix_power(S, power: float) -> np.ndarray:
    values, U = np.linalg.eigh(S)
    return (U * values ** power) @ U.conj().T


def carried_energy(solver, state, atoms, options, family, kpts,
                   strain_matrix, symmetry: bool = False) -> float:
    r"""Energy (Hartree) of the converged state carried to a strained crystal
    -- the free energy up to the entropy, which the carried state keeps.

    Cell, atoms and grid are strained together by ``strain_matrix``
    (:math:`1 + \varepsilon`) with the grid's node counts kept; the basis
    filter is pinned at the run's own cutoff so the basis does not change
    with the spacing.  Each k-point's states move as
    :math:`C(\varepsilon) = S(\varepsilon)^{-1/2}S(0)^{1/2}C(0)` at fixed
    occupations, so the entropy is unchanged and only :math:`E` is returned.
    A spin-polarized state carries each channel's states and evaluates the
    spin energy (each channel's exchange-correlation and, for a hybrid, each
    channel's exact exchange).  ``symmetry`` keeps the space group (a strain from
    :func:`invariant_strains` does), so the strained crystal is reduced to
    the same irreducible k-points as the state.
    """
    from ..integrals import Grid
    from ..pseudopotentials.periodic_paw import build_crystal
    from ..units import HARTREE_TO_EV
    from .periodic_dft import PeriodicKohnSham

    c = solver.crystal
    strained = atoms.copy()
    cell = np.asarray(atoms.get_cell(), dtype=float) @ np.asarray(
        strain_matrix, dtype=float).T
    strained.set_cell(cell, scale_atoms=True)
    shape = tuple(c.grid.shape)
    spacing = np.linalg.norm(cell, axis=1) / np.asarray(shape, dtype=float)
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0, h=spacing,
                units="angstrom", cell=cell, periodic=True)
    if tuple(grid.shape) != shape:
        raise RuntimeError(f"the strained grid has {grid.shape} nodes, not "
                           f"{shape}: the stress needs the same grid")
    pinned = dict(options)
    if c.filter_cutoff is not None:
        pinned["filter"] = 0.5 * c.filter_cutoff ** 2 * HARTREE_TO_EV
    crystal, context = build_crystal(strained, 0.0, pinned, kpts=kpts,
                                     family=family, grid=grid,
                                     symmetry=symmetry)
    if not np.allclose(context["kpoints_fractional"], state["fractional"]):
        raise RuntimeError("the strained crystal's k-points are not the "
                           "state's: the carried states would be paired with "
                           "the wrong Bloch sums")
    strained_solver = PeriodicKohnSham(
        crystal, solver.n_electrons, solver.functional,
        smearing=None, relativistic=solver.relativistic,
        constant=solver.constant, screening=solver.screening,
        magnetic_moments=solver.initial_moments if state["spin"] else None)
    carry = [_matrix_power(data.overlap, -0.5) @ _matrix_power(S0, 0.5)
             for data, S0 in zip(crystal.kpoint_data, state["overlaps"])]
    carried = []                  # per channel: (vectors, matrices)
    for channel in state["channels"]:
        vectors = [T @ C0 for T, C0 in zip(carry, channel["vectors"])]
        carried.append((vectors, [(C * f) @ C.conj().T for C, f in
                                  zip(vectors, channel["occupations"])]))
    if state["spin"]:
        (up, q_up), (dn, q_dn) = (crystal.density(carried[0][1]),
                                  crystal.density(carried[1][1]))
        q = {ch: q_up[ch] + q_dn[ch] for ch in crystal.channels}
        tau = ((strained_solver._tau(carried[0][1]),
                strained_solver._tau(carried[1][1]))
               if strained_solver.meta else None)
        total = [a + b for a, b in zip(carried[0][1], carried[1][1])]
        energy = float(sum(strained_solver.energy_terms(total, (up, dn), q,
                                                        tau).values()))
    else:
        matrices = carried[0][1]
        rho, q = crystal.density(matrices)
        tau = strained_solver._tau(matrices) if strained_solver.meta else None
        energy = float(sum(strained_solver.energy_terms(matrices, rho, q,
                                                        tau).values()))
    if strained_solver.screening is not None:
        # One call over the channels: the spheres' frozen terms enter once.
        energy += strained_solver.hybrid_terms(
            strained_solver.device, [vectors for vectors, _m in carried],
            [channel["occupations"] for channel in state["channels"]],
            [matrices for _v, matrices in carried])[0]
    return energy


@single_threaded_blas
def crystal_stress(solver, atoms, options, family, kpts, *,
                   strain: float = DEFAULT_STRAIN):
    r"""``(3, 3)`` stress (Hartree/Bohr^3), ASE's sign: :math:`\Omega^{-1}
    \partial F/\partial\varepsilon`.  ``atoms``, ``options``, ``family``
    and ``kpts`` are the run's (the strained crystals are rebuilt from
    them).

    The stress is invariant under the space group, so it is fixed by its
    components along :func:`invariant_strains`: a central difference along
    each of them (one for a cubic crystal, six without symmetry), every
    strained crystal keeping the group and the irreducible k-points.
    """
    from ..units import to_bohr

    c = solver.crystal
    state = kpoint_state(solver)
    cell = to_bohr(np.asarray(atoms.get_cell(), dtype=float), "angstrom")
    volume = abs(float(np.linalg.det(cell)))
    stress = np.zeros((3, 3))
    for E in invariant_strains(c):
        # The largest element of the strain is `strain`, as for a single
        # component.
        t = float(strain) / float(np.max(np.abs(E)))
        plus, minus = (carried_energy(solver, state, atoms, options, family,
                                      kpts, np.eye(3) + sign * t * E,
                                      symmetry=c.symmetry is not None)
                       for sign in (+1.0, -1.0))
        stress += (plus - minus) / (2.0 * t) / volume * E
    return stress
