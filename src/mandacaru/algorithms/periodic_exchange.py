# -*- coding: utf-8 -*-
# file: algorithms/periodic_exchange.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Short-range exact exchange in a crystal: the nonlocal part of a screened hybrid.

At a k-point of the mesh the exchange matrix between the Bloch sums is

.. math::

    K^{\mathbf k}_{\mu\nu} = \sum_{\mathbf q} w_{\mathbf q} \sum_m f_{m\mathbf q}
    \big(\rho^{\mathbf k\mathbf q}_{m\mu} + \hat n^{\mathbf k\mathbf q}_{m\mu}
    \big|\,v\,\big|\,
    \rho^{\mathbf k\mathbf q}_{m\nu} + \hat n^{\mathbf k\mathbf q}_{m\nu}\big),
    \qquad \rho^{\mathbf k\mathbf q}_{m\nu} = \phi^*_{m\mathbf q}\chi_{\nu\mathbf k},

the sum over **every** point of the mesh, not only the irreducible ones the
SCF solves at, and over the occupied states :math:`\phi_{m\mathbf q}` with
their occupations :math:`f` (up to 2 in a restricted crystal, up to 1 per
spin channel).  The pair densities are **augmented**, as the Hartree term's
density is: :math:`\hat n` carries the PAW-LCAO compensation charges of each
pair, whose moments come from the pair's projections,
:math:`\langle\phi_{m\mathbf q}|\tilde p_i\rangle\,\mathrm{blk}_{ij}\,
\langle\tilde p_j|\chi_{\nu\mathbf k}\rangle`.  The one-center correction of
each augmentation sphere
(:class:`~mandacaru.pseudopotentials.onecenter.OneCenterHybrid`) is taken
over the same full-mesh projected density matrix (:meth:`MeshStates.projected_density`).

A pair is a Bloch function of wave vector :math:`\mathbf p = \mathbf k -
\mathbf q`, so the exchange is a sum over :math:`\mathbf G + \mathbf p` with
the short-range kernel :math:`\tfrac{4\pi}{G^2}(1 - e^{-G^2/4\omega^2})`
(:func:`~mandacaru.integrals.poisson.short_range_coulomb_kernel`), finite at
:math:`\mathbf G + \mathbf p = 0` (:math:`\pi/\omega^2`): a screened hybrid
needs no treatment of the Coulomb singularity.  The smooth pair and its
coupling to the compensation charges are summed over the grid's reciprocal
set; the compensation charges between themselves over the dense set the
Hartree term's are (:meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.dense_coulomb`),
with the same on-site tail.

The states at the points outside the irreducible wedge are images of the
wedge's: an operation :math:`x \to Wx + t` (fractional coordinates) that maps
the grid onto itself sends :math:`\phi_{\mathbf k}` to
:math:`\phi_{\mathbf k}(Wx + t)`, of Bloch vector :math:`W^T\mathbf k`, and
time reversal sends it to :math:`\phi^*_{\mathbf k}` at :math:`-\mathbf k`.
Every image of a basis function is a combination of the Bloch sums at the
image point, so the image is carried by an ``(M, M)`` matrix :math:`A` with
:math:`\chi_{\mathbf k}(Wx + t) = A\,\chi_{W^T\mathbf k}(x)`, found once per
crystal by least squares on the grid (and checked: the residual is
round-off); the image states' coefficients are then :math:`A^Tc`, their
projections those of the Bloch sums at the image point.  The exchange sees
the occupied states only through
:math:`\sum_m f_m\phi_m(\mathbf r)\phi_m^*(\mathbf r')`, which no unitary
mixing of degenerate states changes, so these images serve as well as states
solved at those points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import fft

from ..integrals import reciprocal as rc
from ..integrals._backend import grid_blas_threads
from ..integrals.poisson import short_range_coulomb_kernel

#: Occupations (electrons per state) below this carry no exchange.
OCCUPATION_FLOOR = 1e-10

#: Bytes one batch of pair densities may hold (complex, ``n x M x ngrid``).
PAIR_BLOCK_BYTES = 256 * 1024 ** 2

#: Bytes the cached compensation transforms of every pair wave vector may
#: hold; beyond it they are rebuilt for each pair of k-points.
TRANSFORM_CACHE_BYTES = 512 * 1024 ** 2

#: Largest residual (relative) of an image's least-squares carrier.
IMAGE_RESIDUAL = 1e-8


@dataclass
class MeshStates:
    """The occupied states of the whole k-mesh.

    Parallel lists, one entry per mesh point: the Cartesian Bloch vector
    (Bohr^-1), its weight (the weights sum to one), the occupations, the
    states on the grid ``(n, ngrid)`` and their projections
    :math:`\\langle\\phi_m|\\tilde p_i\\rangle`, ``(n, P)``.
    """

    kpoints: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    occupations: list = field(default_factory=list)
    orbitals: list = field(default_factory=list)
    projections: list = field(default_factory=list)

    def projected_density(self) -> np.ndarray:
        r"""``(P, P)``: :math:`R_{ij} = \sum_{\mathbf q} w_{\mathbf q}
        \sum_m f_m\langle\tilde p_i|\phi_m\rangle\langle\phi_m|\tilde p_j
        \rangle` over the full mesh -- the one-center density matrix the
        augmentation spheres see, symmetric by construction."""
        R = 0.0
        for w, f, proj in zip(self.weights, self.occupations,
                              self.projections):
            R = R + w * ((proj.conj().T * f) @ proj)
        return np.asarray(R)


@dataclass
class _Image:
    """One mesh point of a wedge point's star."""

    kpoint: np.ndarray          # Cartesian Bloch vector of the image
    weight: float
    carrier: np.ndarray | None  # A: chi_k(Wx + t) = A chi_{k'}(x); None = id
    permutation: np.ndarray | None
    reverse: bool               # time reversed (conjugated)
    projections: np.ndarray     # <chi_{k'} | p>, (M, P)


class PeriodicExchange:
    """The short-range exact exchange of one crystal.

    Holds what does not change during an SCF: the images of the wedge's
    Bloch sums (built on first use), the compensation transforms at each
    pair wave vector and the dense compensation-compensation matrices.
    """

    def __init__(self, crystal, omega: float):
        self.crystal = crystal
        self.omega = float(omega)
        self._images = None
        self._transforms: dict = {}
        self._dense: dict = {}
        self._dense_shape: dict = {}
        grid = crystal.grid
        self._nodes = np.stack([grid.X.ravel(), grid.Y.ravel(),
                                grid.Z.ravel()])
        self._B = rc.reciprocal_vectors(crystal.lattice)
        self._to_fractional = np.linalg.inv(self._B)

    # -- the full mesh ------------------------------------------------------ #

    def _operations(self) -> list:
        """``[(W, permutation)]`` the wedge was reduced with; the identity
        alone without symmetry (time reversal is added by :meth:`_stars`)."""
        symmetry = self.crystal.symmetry
        if symmetry is None:
            return [(np.eye(3), None)]
        return [(np.asarray(W, dtype=float), np.asarray(perm))
                for W, perm in zip(symmetry.info.rotations,
                                   symmetry.permutations)]

    def _stars(self, kpoint_data) -> list:
        """Per wedge point, ``[(fractional image, permutation, reversed)]``
        -- distinct modulo the reciprocal lattice."""
        stars = []
        for data in kpoint_data:
            k_frac = self._to_fractional @ np.asarray(data.k, dtype=float)
            star = []
            for W, perm in self._operations():
                image = W.T @ k_frac
                for reverse in (False, True):
                    point = -image if reverse else image
                    if any(np.allclose(point - other,
                                       np.round(point - other), atol=1e-8)
                           for other, _p, _r in star):
                        continue
                    star.append((point, perm, reverse))
            stars.append(star)
        weights = [float(d.weight) / len(s) for d, s in zip(kpoint_data,
                                                           stars)]
        if not np.allclose(weights, weights[0], rtol=1e-8):
            raise ValueError(
                "the irreducible k-points' stars do not share one weight per "
                f"mesh point ({sorted(set(np.round(weights, 12)))}): the "
                "wedge was not reduced by the crystal's operations")
        return stars

    def _image_function(self, values, k, image_k, perm, reverse):
        """The image of grid functions ``values`` (rows) at ``k``: the
        periodic part permuted, the Bloch phase of ``image_k`` restored."""
        if perm is None:
            out = values
        else:
            nodes = self._nodes
            periodic = values * np.exp(-1j * (k @ nodes))
            # A reversed image is the conjugate of the point's at -image_k.
            phase_k = -image_k if reverse else image_k
            out = periodic[:, perm] * np.exp(1j * (phase_k @ nodes))
        return out.conj() if reverse else out

    def images(self, kpoint_data) -> list:
        """Per wedge point, its star's :class:`_Image` list (built once)."""
        if self._images is not None:
            return self._images
        c = self.crystal
        out = []
        for data, star in zip(kpoint_data, self._stars(kpoint_data)):
            weight = float(data.weight) / len(star)
            k = np.asarray(data.k, dtype=float)
            images = []
            for point, perm, reverse in star:
                image_k = self._B @ point
                if perm is None and not reverse:
                    images.append(_Image(k, weight, None, None, False,
                                         data.projections))
                    continue
                psi = c.bloch_sums(image_k)[0]
                projections = c.projections(image_k)[0]
                target = self._image_function(data.psi, k, image_k, perm,
                                              reverse)
                # Normal equations of A psi = target: A (psi psi^H) =
                # target psi^H.
                gram = psi @ psi.conj().T
                carrier = np.linalg.solve(gram.T, (target @ psi.conj().T).T).T
                residual = (np.linalg.norm(carrier @ psi - target)
                            / np.linalg.norm(target))
                if residual > IMAGE_RESIDUAL:
                    raise RuntimeError(
                        f"an image of the Bloch sums at k = {k} is not a "
                        f"combination of those at {image_k} (residual "
                        f"{residual:.1e}): the basis does not carry the "
                        "operation")
                images.append(_Image(image_k, weight, carrier, perm,
                                     reverse, projections))
            out.append(images)
        self._images = out
        return out

    def mesh_states(self, kpoint_data, vectors, occupations) -> MeshStates:
        r"""The occupied states of the full mesh, from the wedge's
        eigenvectors (``(M, M)`` per k-point, columns the states) and
        occupations, in the order of ``kpoint_data``."""
        states = MeshStates()
        for data, C, f, images in zip(kpoint_data, vectors, occupations,
                                      self.images(kpoint_data)):
            f = np.asarray(f, dtype=float)
            occupied = f > OCCUPATION_FLOOR
            c = np.asarray(C)[:, occupied]
            k = np.asarray(data.k, dtype=float)
            phi = c.T @ data.psi
            for image in images:
                if image.carrier is None:
                    coefficients, values = c, phi
                else:
                    # phi'(x) = phi(Wx + t) = c^T A chi_{k'}(x): coefficients
                    # A^T c (A^T c* when time reversed, the image conjugated).
                    coefficients = image.carrier.T @ (c.conj() if
                                                      image.reverse else c)
                    values = self._image_function(
                        phi, k, image.kpoint, image.permutation, image.reverse)
                states.kpoints.append(image.kpoint)
                states.weights.append(image.weight)
                states.occupations.append(f[occupied])
                states.orbitals.append(values)
                states.projections.append(coefficients.conj().T
                                          @ image.projections)
        return states

    # -- the pair wave vectors ---------------------------------------------- #

    def _pair_vector(self, k, q):
        """``(p, key)``: k - q folded into the first cell of the reciprocal
        lattice (a pair is Bloch with any representative), and its cache
        key."""
        fractional = self._to_fractional @ (np.asarray(k) - np.asarray(q))
        fractional = fractional - np.floor(fractional + 1e-9)
        key = tuple(np.round(fractional, 9))
        return self._B @ fractional, key

    def _compensation_on_grid(self, p, key) -> np.ndarray:
        """``(n_ch, ngrid)``: every compensation shape's transform at the
        grid's G + p."""
        cached = self._transforms.get(key)
        if cached is not None:
            return cached
        c = self.crystal
        Gp = c.G + p[:, None, None, None]
        out = (c.compensation_transforms(Gp).reshape(len(c.channels), -1)
               if c.channels else np.zeros((0, c.grid.size), dtype=complex))
        if (len(self._transforms) + 1) * out.nbytes <= TRANSFORM_CACHE_BYTES:
            self._transforms[key] = out
        return out

    def _dense_correction(self, p, key, kernel, transforms) -> np.ndarray:
        """``(n_ch, n_ch)``: the compensation charges between themselves on
        the dense set (with the on-site tail) minus what the grid set already
        counts of them."""
        cached = self._dense.get(key)
        if cached is not None:
            return cached
        c = self.crystal
        grid = (transforms.conj() * kernel) @ transforms.T / c.volume
        dense = c.dense_compensation(p, lambda g2: short_range_coulomb_kernel(
            g2, self.omega))
        self._dense[key] = dense - grid
        return self._dense[key]

    # -- the matrices -------------------------------------------------------- #

    def matrices(self, kpoint_data, states: MeshStates) -> np.ndarray:
        r"""``(nk, M, M)``: :math:`K^{\mathbf k}` at each of
        ``kpoint_data`` -- the SCF's wedge, a band path, or an extended basis
        (the forces').  Hermitian to round-off; not symmetrized here."""
        c = self.crystal
        grid = c.grid
        shape, dV = grid.shape, grid.dV
        origin = np.exp(-1j * np.einsum("c,cxyz->xyz", rc.grid_origin(grid),
                                        c.G)).ravel()
        G = c.G
        nodes = self._nodes
        workers = grid_blas_threads()
        blocks = [(np.asarray(c.projector_columns(ch[0])), blk)
                  for ch, blk in c.multipole_blocks.items()]
        out = []
        for data in kpoint_data:
            psi, C_k = data.psi, data.projections
            M = len(psi)
            K = np.zeros((M, M), dtype=complex)
            k = np.asarray(data.k, dtype=float)
            for q, w, f, phi, proj in zip(states.kpoints, states.weights,
                                          states.occupations, states.orbitals,
                                          states.projections):
                p, key = self._pair_vector(k, q)
                Gp = G + p[:, None, None, None]
                kernel = short_range_coulomb_kernel(
                    np.sum(Gp * Gp, axis=0), self.omega).ravel()
                transforms = self._compensation_on_grid(p, key)
                correction = self._dense_correction(p, key, kernel,
                                                    transforms)
                phase = np.exp(-1j * (p @ nodes))
                block = max(1, int(PAIR_BLOCK_BYTES // (16 * M * psi.shape[1])))
                for start in range(0, len(phi), block):
                    part = phi[start:start + block]
                    n = len(part)
                    weights = w * f[start:start + block]
                    # T(G) = int e^{-i(G+p)r} (phi* chi + n^) dr: the smooth
                    # pair through the FFT, the compensation charges by
                    # their moments times the shapes' transforms.
                    pairs = (part.conj()[:, None, :] * psi[None, :, :]
                             * phase).reshape(n, M, *shape)
                    T = (fft.fftn(pairs, axes=(2, 3, 4), workers=workers)
                         .reshape(n, M, -1) * (origin * dV))
                    moments = np.stack([
                        proj[start:start + block][:, own] @ blk
                        @ C_k[:, own].conj().T for own, blk in blocks]) \
                        if blocks else np.zeros((0, n, M), dtype=complex)
                    if blocks:
                        T = T + np.einsum("cam,cg->amg", moments, transforms,
                                          optimize=True)
                    left = T.conj() * (weights[:, None, None] * kernel)
                    K += np.einsum("amg,ang->mn", left, T,
                                   optimize=True) / c.volume
                    if blocks:
                        K += np.einsum("cam,cd,dan,a->mn", moments.conj(),
                                       correction, moments, weights,
                                       optimize=True)
            out.append(K)
        return np.stack(out)

    # -- forces --------------------------------------------------------------- #

    def _dense_shape_derivative(self, p, key, kernel, transforms):
        r"""``(3, n_ch, n_ch)``: the dense-minus-grid compensation matrix
        with each element weighted by :math:`i(\mathbf G + \mathbf p)` --
        its derivative is this times :math:`\delta_{cA} - \delta_{dA}` when
        atom A moves."""
        cached = self._dense_shape.get(key)
        if cached is not None:
            return cached
        c = self.crystal
        Gp = (c.G + p[:, None, None, None]).reshape(3, -1)
        grid = np.stack([(transforms.conj() * (kernel * 1j * Gp[d]))
                         @ transforms.T for d in range(3)]) / c.volume
        dense = c.dense_compensation_gradient(
            p, lambda g2: short_range_coulomb_kernel(g2, self.omega))
        self._dense_shape[key] = dense - grid
        return self._dense_shape[key]

    def compensation_gradient(self, data, states: MeshStates, P, dC,
                              scale: float) -> np.ndarray:
        r"""``(n_atoms, 3)``: the derivative of the exchange energy
        :math:`-\tfrac{s}{2}\operatorname{tr}(PK)` at one k-point (weight
        not included) through the compensation charges of its pairs, at
        fixed coefficients -- the part the extended-basis Pulay term does not
        see.  ``scale`` is the factor of K in the Kohn-Sham matrix
        (:math:`a/2` restricted, :math:`a` per spin channel).

        * Each atom's **projectors** move: the k side's moments change
          through ``dC[d]`` (``(M, P)``, the derivative of the projections
          for a move along ``d``, nonzero in the moving atom's columns). The
          exchange is symmetric between its two k-points, so the k side
          alone, doubled -- the ``scale`` of the Kohn-Sham matrix.
        * Each atom's **compensation shapes** move, their transforms by
          :math:`-i(\mathbf G + \mathbf p)`: a property of the pair, not of
          one side, taken once -- half the ``scale``.
        """
        c = self.crystal
        grid = c.grid
        shape, dV = grid.shape, grid.dV
        origin = np.exp(-1j * np.einsum("c,cxyz->xyz", rc.grid_origin(grid),
                                        c.G)).ravel()
        workers = grid_blas_threads()
        channel_atom = np.array([ch[0] for ch in c.channels])
        blocks = [(np.asarray(c.projector_columns(ch[0])), blk)
                  for ch, blk in c.multipole_blocks.items()]
        n_atoms = len(c.centers)
        out = np.zeros((n_atoms, 3))
        if not blocks:
            return out
        psi, C_k = data.psi, data.projections
        M = len(psi)
        k = np.asarray(data.k, dtype=float)
        for q, w, f, phi, proj in zip(states.kpoints, states.weights,
                                      states.occupations, states.orbitals,
                                      states.projections):
            p, key = self._pair_vector(k, q)
            Gp = (c.G + p[:, None, None, None]).reshape(3, -1)
            kernel = short_range_coulomb_kernel(np.sum(Gp * Gp, axis=0),
                                                self.omega)
            transforms = self._compensation_on_grid(p, key)
            correction = self._dense_correction(p, key, kernel, transforms)
            shape_dense = self._dense_shape_derivative(p, key, kernel,
                                                       transforms)
            phase = np.exp(-1j * (p @ self._nodes))
            block = max(1, int(PAIR_BLOCK_BYTES // (16 * M * psi.shape[1])))
            for start in range(0, len(phi), block):
                part = phi[start:start + block]
                n = len(part)
                wf = w * f[start:start + block]
                pairs = (part.conj()[:, None, :] * psi[None, :, :]
                         * phase).reshape(n, M, *shape)
                T = (fft.fftn(pairs, axes=(2, 3, 4), workers=workers)
                     .reshape(n, M, -1) * (origin * dV))
                left = proj[start:start + block]
                moments = np.stack([left[:, own] @ blk @ C_k[:, own].conj().T
                                    for own, blk in blocks])
                T = T + np.einsum("cam,cg->amg", moments, transforms,
                                  optimize=True)
                # Y_a,mu(G) = sum_nu P_nu,mu T_a,nu(G), weighted by w f_a.
                Y = np.einsum("nm,ang->amg", P, T, optimize=True) \
                    * wf[:, None, None]
                # Coupling of a moment change to the grid sum and the dense
                # correction: Z_c,a,mu.
                Z = np.einsum("cg,amg->cam", transforms.conj() * kernel, Y,
                              optimize=True) / c.volume
                MP = np.einsum("dan,nm->dam", moments, P, optimize=True) \
                    * wf[None, :, None]
                Z = Z + np.einsum("cd,dam->cam", correction, MP,
                                  optimize=True)
                for atom in range(n_atoms):
                    mine = np.nonzero(channel_atom == atom)[0]
                    if not len(mine):
                        continue
                    for d in range(3):
                        # Projectors: dmom_c = left blk dC^dagger (k side).
                        dmom = np.stack([
                            left[:, blocks[ch][0]] @ blocks[ch][1]
                            @ dC[d][:, blocks[ch][0]].conj().T
                            for ch in mine])
                        projector = 2.0 * float(np.real(np.sum(
                            dmom.conj() * Z[mine])))
                        # Shapes: dT = sum_c mom_c dG_c, dG = -i(G+p) G_c.
                        shifted = transforms[mine].conj() * (
                            kernel * 1j * Gp[d])
                        Zs = np.einsum("cg,amg->cam", shifted, Y,
                                       optimize=True) / c.volume
                        shape_grid = 2.0 * float(np.real(np.sum(
                            moments[mine].conj() * Zs)))
                        sign = (np.isin(np.arange(len(c.channels)), mine)
                                [:, None].astype(float)
                                - np.isin(np.arange(len(c.channels)), mine)
                                [None, :].astype(float))
                        Mpair = np.einsum("cam,dam->cd", moments.conj(), MP,
                                          optimize=True)
                        dense_term = float(np.real(np.sum(
                            Mpair * shape_dense[d] * sign)))
                        out[atom, d] += (-scale * projector
                                         - 0.5 * scale * (shape_grid
                                                          + dense_term))
        return out


def exchange_energy(kpoint_data, matrices, K, factor: float) -> float:
    r""":math:`-f\sum_{\mathbf k} w_{\mathbf k}\operatorname{tr}(P^k K^k)`
    (Hartree): ``factor`` ``f`` is half the factor of ``K`` in the
    Kohn-Sham matrix -- :math:`a/4` for a restricted crystal (``P`` holds
    both spins), :math:`a/2` for one spin channel of a polarized one."""
    total = sum(float(d.weight) * float(np.real(np.sum(P * Kk.T)))
                for d, P, Kk in zip(kpoint_data, matrices, K))
    return -factor * total
