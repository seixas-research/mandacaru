# -*- coding: utf-8 -*-
"""Localized Function Collection for atom-centered basis functions.

Evaluates and stores atom-centered basis functions, real-space densities,
and effective potential matrix elements on localized bounding boxes / subgrids,
avoiding global grid allocations of shape (M, N_grid) and (3, M, N_grid).
"""
from __future__ import annotations

import numpy as np


class LocalizedBasisCollection:
    r"""Atom-centered localized representation of a basis on a real-space grid.

    Each basis function :math:`\phi_\mu(\mathbf r)` has compact radial support
    :math:`r_{c, \mu}` around its atomic center :math:`\mathbf R_A`. On a 3D
    simulation grid with large vacuum regions, a basis function is non-zero
    only within a small bounding box :math:`\Omega_A \subset \Omega`.

    This collection manages:

    1. Per-atom bounding boxes :math:`\Omega_A` and local subgrid slices.
    2. Overlapping atom pairs :math:`(A, B)` and their intersection boxes :math:`\Omega_{AB}`.
    3. Memory-efficient evaluation of the real-space density without
       forming intermediate :math:`(M, N_{\rm grid})` tensors.
    4. Localized matrix elements of the effective potential :math:`V_{\rm eff}(\mathbf r)`
       directly on subbox intersections.
    """

    def __init__(self, basis, grid, atom_of_orbital=None, psi_full=None):
        self.basis = list(basis)
        self.grid = grid
        self.shape = grid.shape
        self.M = len(self.basis)
        if atom_of_orbital is None:
            atom_of_orbital = [int(getattr(fn, "atom_index", 0)) for fn in self.basis]
        self.atom_of_orbital = np.asarray(atom_of_orbital, dtype=int)
        self.n_atoms = int(np.max(self.atom_of_orbital)) + 1 if self.M > 0 else 0

        X3d = grid.X.reshape(self.shape)
        Y3d = grid.Y.reshape(self.shape)
        Z3d = grid.Z.reshape(self.shape)

        self.atom_indices: list[np.ndarray] = []
        self.atom_boxes: list[tuple[slice, slice, slice]] = []
        self.atom_psis: list[np.ndarray] = []

        psi_3d = None
        if psi_full is not None:
            psi_3d = np.asarray(psi_full).reshape(self.M, *self.shape)

        for a in range(self.n_atoms):
            indices = np.flatnonzero(self.atom_of_orbital == a)
            self.atom_indices.append(indices)
            if indices.size == 0:
                self.atom_boxes.append((slice(0, 0), slice(0, 0), slice(0, 0)))
                self.atom_psis.append(np.zeros((0, 0, 0, 0), dtype=float))
                continue

            fn_first = self.basis[indices[0]]
            center = np.asarray(fn_first.center, dtype=float)
            rc_list = []
            for i in indices:
                r_val = getattr(self.basis[i], "support_radius", None)
                if r_val is None:
                    r_val = getattr(self.basis[i], "rc", None)
                if r_val is None:
                    r_val = 10.0
                rc_list.append(float(r_val))
            rc = max(rc_list) if rc_list else 10.0

            if not getattr(grid, "skewed", False):
                X1d = X3d[:, 0, 0]
                Y1d = Y3d[0, :, 0]
                Z1d = Z3d[0, 0, :]
                x_hits = np.flatnonzero(np.abs(X1d - center[0]) <= rc)
                y_hits = np.flatnonzero(np.abs(Y1d - center[1]) <= rc)
                z_hits = np.flatnonzero(np.abs(Z1d - center[2]) <= rc)
            else:
                dx = X3d - center[0]
                dy = Y3d - center[1]
                dz = Z3d - center[2]
                mask = (dx * dx + dy * dy + dz * dz) <= rc * rc
                x_hits = np.flatnonzero(mask.any(axis=(1, 2)))
                y_hits = np.flatnonzero(mask.any(axis=(0, 2)))
                z_hits = np.flatnonzero(mask.any(axis=(0, 1)))

            if x_hits.size == 0 or y_hits.size == 0 or z_hits.size == 0:
                self.atom_boxes.append((slice(0, 0), slice(0, 0), slice(0, 0)))
                self.atom_psis.append(np.zeros((len(indices), 0, 0, 0), dtype=float))
                continue

            sx = slice(int(x_hits[0]), int(x_hits[-1]) + 1)
            sy = slice(int(y_hits[0]), int(y_hits[-1]) + 1)
            sz = slice(int(z_hits[0]), int(z_hits[-1]) + 1)
            self.atom_boxes.append((sx, sy, sz))

            if psi_3d is not None:
                self.atom_psis.append(np.ascontiguousarray(psi_3d[indices, sx, sy, sz]))
            else:
                x_sub = X3d[sx, sy, sz]
                y_sub = Y3d[sx, sy, sz]
                z_sub = Z3d[sx, sy, sz]
                sub_psi = np.stack([self.basis[i].evaluate(x_sub, y_sub, z_sub) for i in indices])
                self.atom_psis.append(np.ascontiguousarray(sub_psi, dtype=float))

    def density(self, P_ao: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        r"""Evaluate the real-space density :math:`\rho(\mathbf r)` on the grid from the AO 1-RDM.

        Parameters
        ----------
        P_ao : ndarray of shape (M, M)
            One-particle density matrix in the atomic-orbital basis.
        out : ndarray, optional
            Optional preallocated output array of shape ``grid.shape`` or ``(grid.size,)``.

        Returns
        -------
        ndarray of shape ``(grid.size,)``
            Real-space valence density.
        """
        if out is None:
            rho = np.zeros(self.shape, dtype=float)
        else:
            rho = out.reshape(self.shape)
            rho.fill(0.0)

        for a in range(self.n_atoms):
            sx_a, sy_a, sz_a = self.atom_boxes[a]
            if sx_a.start >= sx_a.stop:
                continue
            idx_a = self.atom_indices[a]
            psi_a = self.atom_psis[a]

            for b in range(self.n_atoms):
                sx_b, sy_b, sz_b = self.atom_boxes[b]
                if sx_b.start >= sx_b.stop:
                    continue
                idx_b = self.atom_indices[b]
                psi_b = self.atom_psis[b]

                # Box intersection
                sx = slice(max(sx_a.start, sx_b.start), min(sx_a.stop, sx_b.stop))
                sy = slice(max(sy_a.start, sy_b.start), min(sy_a.stop, sy_b.stop))
                sz = slice(max(sz_a.start, sz_b.start), min(sz_a.stop, sz_b.stop))
                if sx.start >= sx.stop or sy.start >= sy.stop or sz.start >= sz.stop:
                    continue

                P_ab = P_ao[np.ix_(idx_a, idx_b)]
                if np.all(np.abs(P_ab) < 1e-15):
                    continue

                sub_a = (slice(None),
                         slice(sx.start - sx_a.start, sx.stop - sx_a.start),
                         slice(sy.start - sy_a.start, sy.stop - sy_a.start),
                         slice(sz.start - sz_a.start, sz.stop - sz_a.start))
                sub_b = (slice(None),
                         slice(sx.start - sx_b.start, sx.stop - sx_b.start),
                         slice(sy.start - sy_b.start, sy.stop - sy_b.start),
                         slice(sz.start - sz_b.start, sz.stop - sz_b.start))

                p_a = psi_a[sub_a]
                p_b = psi_b[sub_b]

                term = np.sum(p_a * np.tensordot(P_ab, p_b, axes=(1, 0)), axis=0)
                rho[sx, sy, sz] += np.real(term)

        return rho.ravel()

    def potential_matrix(self, V_grid: np.ndarray, dV: float | None = None) -> np.ndarray:
        r"""Compute the AO potential matrix elements :math:`\langle \phi_\mu | V_{\rm eff} | \phi_\nu \rangle`.

        Parameters
        ----------
        V_grid : ndarray
            Effective potential sampled on the grid.
        dV : float, optional
            Volume element of the grid. Defaults to ``self.grid.dV``.

        Returns
        -------
        ndarray of shape (M, M)
            Potential matrix in the AO basis.
        """
        dV = self.grid.dV if dV is None else float(dV)
        V_ao = np.zeros((self.M, self.M), dtype=float)
        V3d = V_grid.reshape(self.shape) * dV

        for a in range(self.n_atoms):
            sx_a, sy_a, sz_a = self.atom_boxes[a]
            if sx_a.start >= sx_a.stop:
                continue
            idx_a = self.atom_indices[a]
            psi_a = self.atom_psis[a]

            for b in range(a, self.n_atoms):
                sx_b, sy_b, sz_b = self.atom_boxes[b]
                if sx_b.start >= sx_b.stop:
                    continue
                idx_b = self.atom_indices[b]
                psi_b = self.atom_psis[b]

                sx = slice(max(sx_a.start, sx_b.start), min(sx_a.stop, sx_b.stop))
                sy = slice(max(sy_a.start, sy_b.start), min(sy_a.stop, sy_b.stop))
                sz = slice(max(sz_a.start, sz_b.start), min(sz_a.stop, sz_b.stop))
                if sx.start >= sx.stop or sy.start >= sy.stop or sz.start >= sz.stop:
                    continue

                sub_a = (slice(None),
                         slice(sx.start - sx_a.start, sx.stop - sx_a.start),
                         slice(sy.start - sy_a.start, sy.stop - sy_a.start),
                         slice(sz.start - sz_a.start, sz.stop - sz_a.start))
                sub_b = (slice(None),
                         slice(sx.start - sx_b.start, sx.stop - sx_b.start),
                         slice(sy.start - sy_b.start, sy.stop - sy_b.start),
                         slice(sz.start - sz_b.start, sz.stop - sz_b.start))

                p_a = psi_a[sub_a]
                p_b = psi_b[sub_b]
                v_sub = V3d[sx, sy, sz]

                p_a_w = p_a * v_sub
                block = p_a_w.reshape(len(idx_a), -1) @ p_b.reshape(len(idx_b), -1).T
                V_ao[np.ix_(idx_a, idx_b)] += block
                if a != b:
                    V_ao[np.ix_(idx_b, idx_a)] += block.T

        return V_ao
