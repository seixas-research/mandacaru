# -*- coding: utf-8 -*-
# file: integrals/poisson.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""FFT Poisson solver for the two-body Coulomb potential.

The electron-repulsion tensor needs, for every density pair
:math:`\rho(\mathbf r) = \phi_i^*(\mathbf r)\phi_j(\mathbf r)`, its Coulomb
potential on the grid

.. math::

    \Phi(\mathbf r_1) = \sum_{\mathbf r_2} \frac{\rho(\mathbf r_2)}
                                                 {|\mathbf r_1-\mathbf r_2|}\, dV .

That is a discrete convolution of :math:`\rho` with the Green's function
:math:`G(\mathbf d)=1/|\mathbf d|`.  Evaluated directly it costs
:math:`O(N_\text{grid}^2)`; as an FFT convolution it costs
:math:`O(N_\text{grid}\log N_\text{grid})`:

.. math::

    \Phi = \mathrm{IFFT}\big(\mathrm{FFT}(\rho)\cdot \mathrm{FFT}(G)\big) .

Two numerical points make this correct rather than merely fast:

* **Zero-padding.** The FFT computes a *circular* convolution.  Without padding
  each axis to at least :math:`2N-1`, the long Coulomb tail wraps around the box
  and contaminates the potential.  We pad to ``scipy.fft.next_fast_len(2N-1)``.
* **Self term.** The :math:`\mathbf d=0` node is the singular
  self-interaction.  Instead of an ad-hoc softening we integrate :math:`1/r`
  over the voxel itself, :math:`G(0)=\frac{1}{dV}\int_\text{cell}d^3r/|r|`
  (:func:`cell_self_potential`), the physically correct cell self-energy.  For
  a cube that is :math:`C_\text{cube}/dx` with
  :math:`C_\text{cube}=\int_{[-1/2,1/2]^3} d^3u/|u| \approx 2.3800774`.

* **Voxel geometry.** The displacement between two nodes of *any* Bravais
  sampling is ``step @ (di, dj, dk)``: it depends only on the index
  difference, so the convolution structure holds for anisotropic **and**
  skewed grids alike.  Distances come from the grid's step matrix and the
  source volume is ``|det(step)|``; only the ``d = 0`` voxel needs its own
  treatment (:func:`voxel_self_potential`).

``scipy.fft`` (pocketfft) runs the transforms in threaded C; ``workers=-1``
uses all cores, so this path is parallel without any custom kernel.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft

#: Average of 1/r over a unit cube centered at the origin (see module docstring).
#: Kept as a reference value; :func:`cell_self_potential` is the closed form.
CUBE_SELF_CONSTANT = 2.3800756


def _triangle_potential(a, b, c) -> float:
    r"""``int_T dS/|r|`` over the triangle ``(a, b, c)``, field point at the origin.

    The closed form for a uniformly charged planar polygon: one logarithm along
    each edge plus an arctangent pair for the out-of-plane offset.
    ``perpendicular`` is the in-plane distance from the projected field point to
    the edge line, ``l_*`` the endpoint coordinates along the edge and ``r_*``
    their distances to the field point.
    """
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return 0.0                                    # degenerate triangle
    normal = normal / norm
    offset = float(normal @ a)                        # signed plane distance
    projected = offset * normal
    total = 0.0
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length == 0.0:
            continue
        along = edge / length
        outward = np.cross(along, normal)
        perpendicular = float(outward @ (start - projected))
        l_minus = float((start - projected) @ along)
        l_plus = float((end - projected) @ along)
        r_minus = float(np.linalg.norm(start))
        r_plus = float(np.linalg.norm(end))
        r0_sq = perpendicular * perpendicular + offset * offset
        if perpendicular != 0.0 and (r_plus + l_plus) > 0.0 \
                and (r_minus + l_minus) > 0.0:
            total += perpendicular * np.log((r_plus + l_plus)
                                            / (r_minus + l_minus))
        total -= abs(offset) * (
            np.arctan2(perpendicular * l_plus, r0_sq + abs(offset) * r_plus)
            - np.arctan2(perpendicular * l_minus,
                         r0_sq + abs(offset) * r_minus))
    return float(total)


def voxel_self_potential(step) -> float:
    r"""``int_cell d^3r/|r|`` over the voxel of a general ``step`` matrix.

    ``step`` holds the three step vectors as columns, so the voxel is the
    parallelepiped :math:`\{\sum_m u_m s_m : u \in [-1/2, 1/2]^3\}` centered on
    the node.  Because :math:`\nabla^2 r = 2/r`, the divergence theorem turns
    the volume integral into a sum over the six faces,

    .. math::

        \int_V \frac{d^3r}{|r|}
            = \tfrac12 \oint_{\partial V} \hat r\cdot\hat n \, dS
            = \tfrac12 \sum_f h_f \int_f \frac{dS}{|r|} ,

    with :math:`h_f` the (constant) distance from the node to face :math:`f`;
    each parallelogram face splits into two triangles integrated in closed form
    (:func:`_triangle_potential`).  An orthogonal voxel takes the simpler
    rectangular formula :func:`cell_self_potential` instead -- the two agree to
    machine precision.
    """
    step = np.asarray(step, dtype=float)
    if step.shape != (3, 3):
        raise ValueError(f"step must be a 3x3 matrix, got {step.shape}")
    volume = abs(float(np.linalg.det(step)))
    if volume <= 0.0:
        raise ValueError("the voxel has zero volume (degenerate step matrix)")
    diagonal = np.abs(np.diag(step))
    off_diagonal = step - np.diag(np.diag(step))
    if np.all(np.abs(off_diagonal) <= 1e-12 * max(float(diagonal.max()), 1.0)):
        return cell_self_potential(*diagonal)

    corners = {signs: step @ (np.array(signs, dtype=float) - 0.5)
               for signs in np.ndindex(2, 2, 2)}
    total = 0.0
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for side in (0, 1):
            face = [s for s in np.ndindex(2, 2, 2) if s[axis] == side]
            # Order the four corners around the face (a cycle, not a zig-zag).
            face.sort(key=lambda s: (s[others[0]], s[others[1]] ^ s[others[0]]))
            v = [corners[s] for s in face]
            normal = np.cross(v[1] - v[0], v[2] - v[0])
            normal = normal / np.linalg.norm(normal)
            height = abs(float(normal @ v[0]))
            total += height * (_triangle_potential(v[0], v[1], v[2])
                               + _triangle_potential(v[0], v[2], v[3]))
    return float(0.5 * total)


def cell_self_potential(dx: float, dy: float, dz: float) -> float:
    r"""``int_cell d^3r/|r|`` over a rectangular voxel centered at the origin.

    The closed form of the box integral (Bohr\ :sup:`2`), eight times its
    positive octant.  Divided by the voxel volume it is the Green's function at
    ``d = 0``: the average of ``1/r`` over the cell one source node stands for.
    For a cube it gives ``dx**2 * 2.3800774``.
    """
    a, b, c = 0.5 * float(dx), 0.5 * float(dy), 0.5 * float(dz)
    if min(a, b, c) <= 0.0:
        raise ValueError(f"grid spacings must be positive, got {(dx, dy, dz)}")
    s = np.sqrt(a * a + b * b + c * c)
    octant = (a * b * np.log((c + s) / np.hypot(a, b))
              + b * c * np.log((a + s) / np.hypot(b, c))
              + c * a * np.log((b + s) / np.hypot(c, a))
              - 0.5 * a * a * np.arctan(b * c / (a * s))
              - 0.5 * b * b * np.arctan(c * a / (b * s))
              - 0.5 * c * c * np.arctan(a * b / (c * s)))
    return float(8.0 * octant)


class PoissonFFTSolver:
    """Solve the grid Coulomb convolution by zero-padded FFT.

    Parameters
    ----------
    shape : int or (int, int, int)
        Nodes per Cartesian axis of the source grid.  A scalar means a cubic
        grid ``(N, N, N)``; a triple ``(nx, ny, nz)`` a non-cubic one -- the FFT
        convolution is agnostic to the box shape, only the per-axis lengths and
        padding differ.
    spacing : float or (float, float, float), optional
        Grid spacing in Bohr: one value for every axis, or the per-axis
        ``(dx, dy, dz)`` of an anisotropic grid.  Ignored when ``step`` is given.
    step : (3, 3) array_like, optional
        The full voxel basis (step vectors as columns, i.e. ``Grid.step``), for
        a **skewed** grid: distances then use the lattice displacement
        ``step @ (di, dj, dk)`` and the volume is ``|det(step)|``.
    self_const : float, optional
        Overrides ``G(0)`` with the legacy cubic rule ``self_const / dx``.  By
        default ``G(0)`` is the exact cell average
        (:func:`cell_self_potential`) -- the same thing for a cube.
    workers : int, optional
        Threads for the FFTs (``-1`` uses all cores).
    """

    def __init__(self, shape, spacing=None, self_const: float | None = None,
                 workers: int = -1, step=None):
        if np.isscalar(shape):
            self.shape = (int(shape),) * 3
        else:
            self.shape = tuple(int(s) for s in shape)
        if step is not None:
            self.step = np.asarray(step, dtype=float)
            if self.step.shape != (3, 3):
                raise ValueError(f"step must be a 3x3 matrix, got "
                                 f"{self.step.shape}")
            self.spacing = tuple(float(np.linalg.norm(self.step[:, m]))
                                 for m in range(3))
        else:
            if spacing is None:
                raise ValueError("give either a spacing or a step matrix")
            self.spacing = ((float(spacing),) * 3 if np.isscalar(spacing)
                            else tuple(float(v) for v in spacing))
            if len(self.spacing) != 3 or min(self.spacing) <= 0.0:
                raise ValueError(f"spacing must be three positive lengths, got "
                                 f"{spacing!r}")
            self.step = np.diag(np.asarray(self.spacing, dtype=float))
        self.dx, self.dy, self.dz = self.spacing
        self.dV = abs(float(np.linalg.det(self.step)))
        if self.dV <= 0.0:
            raise ValueError("the voxel has zero volume (degenerate step)")
        self.workers = workers
        # Pad each axis to >= 2N-1 (FFT-friendly length) to avoid wraparound.
        self.L = tuple(sfft.next_fast_len(2 * n - 1) for n in self.shape)
        self._Gk = self._build_kernel_transform(self_const)

    def _build_kernel_transform(self, self_const: float | None) -> np.ndarray:
        """Precompute FFT of the 1/r Green's function on the padded grid."""
        offs = []
        for n, L in zip(self.shape, self.L):
            # Signed integer offsets: 0..n-1 positive, top of the array negative.
            idx = np.arange(L)
            offs.append(np.where(idx < n, idx, idx - L).astype(float))
        # Node (i, j, k) sits at step @ (i, j, k) from the origin node, so the
        # displacement depends only on the index difference -- skew included.
        SX = offs[0][:, None, None]
        SY = offs[1][None, :, None]
        SZ = offs[2][None, None, :]
        dist_sq = np.zeros(self.L, dtype=float)
        for row in range(3):
            component = (self.step[row, 0] * SX + self.step[row, 1] * SY
                         + self.step[row, 2] * SZ)
            dist_sq += component * component
        dist = np.sqrt(dist_sq, out=dist_sq)
        with np.errstate(divide="ignore"):
            G = np.where(dist > 0, 1.0 / dist, 0.0)
        # The d = 0 node carries the voxel's own average of 1/r.
        G[0, 0, 0] = (self_const / self.dx if self_const is not None
                      else voxel_self_potential(self.step) / self.dV)
        return sfft.fftn(G, workers=self.workers)

    def solve(self, rho_flat: np.ndarray) -> np.ndarray:
        """Coulomb potential of a single density on the grid (flattened)."""
        return self.solve_stack(rho_flat[None, :])[0]

    def solve_stack(self, rho_stack: np.ndarray) -> np.ndarray:
        """Coulomb potentials of a stack of ``P`` densities.

        Parameters
        ----------
        rho_stack : (P, nx*ny*nz) complex
            Densities sampled on the flattened grid.

        Returns
        -------
        (P, nx*ny*nz) complex
            The corresponding Coulomb potentials ``Phi``.
        """
        (nx, ny, nz), dV = self.shape, self.dV
        ngrid = nx * ny * nz
        rho_stack = np.ascontiguousarray(rho_stack, dtype=np.complex128)
        P = rho_stack.shape[0]
        out = np.empty((P, ngrid), dtype=np.complex128)
        pad = np.zeros(self.L, dtype=np.complex128)
        for p in range(P):
            pad[:] = 0.0
            pad[:nx, :ny, :nz] = rho_stack[p].reshape(nx, ny, nz)
            spec = sfft.fftn(pad, workers=self.workers)
            phi = sfft.ifftn(spec * self._Gk, workers=self.workers)
            out[p] = phi[:nx, :ny, :nz].reshape(-1) * dV
        return out


def fft_g_squared(shape, cell):
    r"""``|G|^2`` on the FFT mesh, with the Nyquist plane symmetrized.

    ``cell`` holds the lattice vectors as **columns** (``step @ diag(shape)``),
    so the matrix whose columns are the ``b_j`` is ``2 pi inv(cell).T`` and
    ``G = reciprocal @ m``.

    Why the symmetrization
    ----------------------

    ``fftfreq`` enumerates ``m`` over ``{0, ..., n/2 - 1, -n/2, ..., -1}``.  For
    **even** ``n`` that set contains ``-n/2`` but not ``+n/2``, which are the
    *same* discrete mode -- they differ by ``n``.  Writing
    :math:`|G|^2 = \sum_{ab} Q_{ab} m_a m_b` with
    :math:`Q = B^{\mathsf T} B`, the two aliases differ by the terms **linear**
    in the Nyquist index, :math:`\pm 2 (n/2) \sum_{b \neq a} Q_{ab} m_b`.  Those
    vanish for an orthogonal cell, where :math:`Q_{ab} = 0` off the diagonal --
    and do **not** vanish for a sheared one.

    The consequence is that the multiset of :math:`|G|^2` is not invariant when
    a shear changes sign: measured at ``0.22`` (``n = 10``) and ``0.32``
    (``n = 12``) for a 1e-3 shear, and exactly ``0`` for odd ``n``.  A quantity
    that symmetry forces to be even in the strain -- the shear stress of a
    cubic crystal -- then comes out nonzero, at 6.4e-4 eV/Angstrom^3 on a
    ``10x10x10`` grid.

    Averaging the aliases is the fix: because :math:`|G|^2` is quadratic, the
    mean over the independent sign choices of the Nyquist components cancels
    exactly the cross terms that involve them and keeps the diagonal
    :math:`(n/2)^2 Q_{aa}`.  **For an orthogonal cell this changes nothing**,
    which is why no existing energy moves.
    """
    shape = tuple(int(n) for n in shape)
    cell = np.asarray(cell, dtype=float)
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    Q = reciprocal.T @ reciprocal

    axes = [sfft.fftfreq(n) * n for n in shape]
    m = np.meshgrid(*axes, indexing="ij")
    # A node is "Nyquist along axis a" when n_a is even and m_a = -n_a/2; odd
    # axes have no such mode and are never masked.
    nyquist = [np.isclose(m[a], -(n // 2)) if n % 2 == 0
               else np.zeros(shape, dtype=bool)
               for a, n in enumerate(shape)]

    g_squared = np.zeros(shape, dtype=float)
    for a in range(3):
        g_squared += Q[a, a] * m[a] * m[a]
        for b in range(3):
            if b == a:
                continue
            # Dropped wherever either index sits on its Nyquist plane: that is
            # precisely the average over the two aliases.
            keep = ~(nyquist[a] | nyquist[b])
            g_squared += Q[a, b] * np.where(keep, m[a] * m[b], 0.0)
    return g_squared


#: Recognized Coulomb truncations for :class:`PeriodicPoissonSolver`.
#: ``"none"`` is the fully periodic ``4 pi / G^2``; ``"slab"`` is the
#: two-dimensionally truncated kernel of :func:`slab_truncated_kernel`.
KERNEL_TRUNCATIONS = ("none", "slab")

#: Largest ``|a_c . a_a| / (|a_c||a_a|)`` accepted between the non-periodic
#: lattice vector and an in-plane one.  The slab derivation splits ``G`` into an
#: in-plane part and an axial one, which only exists when the axis is
#: perpendicular to the plane.
SLAB_ORTHOGONALITY_TOLERANCE = 1e-10

#: Fraction of the cell length along the non-periodic axis that the density may
#: occupy before :func:`slab_occupancy_warning` complains.  The truncation cuts
#: the interaction at ``L/2``, so two points further apart than that in ``z``
#: stop interacting altogether: the density has to be confined to half the cell.
SLAB_MAX_OCCUPANCY = 0.5


def slab_truncated_kernel(shape, cell, axis: int = 2) -> np.ndarray:
    r"""The **two-dimensionally truncated** Coulomb kernel on the FFT mesh.

    A slab is periodic in two directions and finite in the third, but a
    three-dimensional kernel makes it interact with an infinite stack of copies
    of itself along the vacuum direction.  For a *neutral, non-polar* slab that
    error decays with the vacuum thickness and can be converged away.  For a
    slab carrying a **dipole** it cannot: two dipole sheets a distance ``L``
    apart interact with an energy per area that does not vanish as ``L`` grows,
    so the total energy converges to the wrong number no matter how much vacuum
    is added.  This kernel removes the images instead of out-running them.

    The construction (Rozzi *et al.*, Phys. Rev. B **73**, 205119, 2006;
    ``Rozzi2006`` in the bibliography) is to
    Fourier transform the Coulomb interaction *truncated* beyond half a cell
    along the non-periodic axis, :math:`v(\mathbf r) = \theta(L/2 - |z|)/|\mathbf
    r|`.  In-plane, :math:`\int d^2\rho\, e^{-i\mathbf G_\parallel \cdot
    \boldsymbol\rho}/\sqrt{\rho^2+z^2} = 2\pi e^{-G_\parallel |z|}/G_\parallel`,
    and the remaining integral over :math:`|z| \le L/2` is elementary:

    .. math::

        \tilde v(\mathbf G) = \frac{4\pi}{G^2}\left[1 + e^{-G_\parallel L/2}
            \left(\frac{G_z}{G_\parallel}\sin\frac{G_z L}{2}
                  - \cos\frac{G_z L}{2}\right)\right] ,
        \qquad G_\parallel \neq 0 .

    At :math:`G_\parallel = 0` that expression has a :math:`1/G_\parallel` pole
    -- the divergent self-interaction of a charged *sheet*, the two-dimensional
    counterpart of the :math:`\mathbf G = 0` divergence of the periodic kernel.
    Removing it leaves the interaction between neutral sheets, which is what a
    uniform plane at :math:`z'` produces, :math:`-2\pi|z-z'|`, truncated and
    transformed the same way:

    .. math::

        \tilde v(0, G_z) = \frac{2\pi}{G_z^2}
            \left[2 - 2\cos\frac{G_z L}{2}
                  - G_z L \sin\frac{G_z L}{2}\right] .

    **The two are one function.** Subtracting the pole
    :math:`4\pi \sin(G_z L/2)/(G_z G_\parallel)` from the first branch and
    letting :math:`G_\parallel \to 0` gives the second identically, which is the
    check that the algebra is right rather than merely plausible
    (``test/integrals/test_slab_kernel.py`` pins it numerically).

    ``G = 0`` is set to zero, the same neutrality convention the untruncated
    kernel uses: it is a choice of reference that cancels against the
    electron-ion and ion-ion :math:`\mathbf G = 0` terms for a neutral cell.
    Note that :math:`\mathbf G_\parallel = 0` with :math:`G_z \neq 0` is **not**
    dropped -- those are exactly the components a dipole lives in, and dropping
    them would discard the effect this kernel exists to produce.

    Parameters
    ----------
    shape : (int, int, int)
        Nodes per axis; the grid is the cell.
    cell : (3, 3) array_like
        Lattice vectors as **columns**, in Bohr.
    axis : int
        Which lattice vector is the non-periodic (vacuum) direction.

    Raises
    ------
    ValueError
        If ``axis``'s lattice vector is not perpendicular to the other two: the
        split of :math:`\mathbf G` into in-plane and axial parts, and with it
        every formula above, exists only then.

    What it does and does not give
    ------------------------------
    The convolution is still *circular*, and the kernel's real-space profile
    over one period is the true interaction restricted to :math:`|\Delta z| \le
    L/2`.  For a density confined to less than half the cell that makes the
    circular convolution equal the isolated one **wherever the density is**, so
    the Hartree **energy** is exact -- measured exact to every digit the grid
    carries, and independent of the vacuum thickness, where the untruncated
    kernel drifts as :math:`1/L` and is still moving at four times the slab
    width.

    It does **not** give the isolated potential out in the vacuum.  A field point
    far from the slab is more than :math:`L/2` from part of the density, so its
    circular displacement is not the true one and the kernel value is not the
    true one.  That does not touch the energy (which only samples
    :math:`\Phi` where :math:`\rho` is) nor the potential inside the slab, but
    it means **the vacuum level is not available from this kernel** -- a work
    function or a band alignment needs the mixed-space solver
    (:math:`\Phi(\mathbf G_\parallel, z)` convolved along :math:`z`), which is
    not implemented here.  :func:`slab_occupancy` measures the confinement the
    energy relies on.

    Notes
    -----
    Even in :math:`G_z`, which is what lets the Nyquist plane along the
    non-periodic axis be treated with :math:`|G_z|`: the two aliases
    :math:`\pm n_c/2` are the same mode and give the same kernel.  The in-plane
    Nyquist aliasing that :func:`fft_g_squared` documents for a sheared cell is
    handled the same way here -- by symmetrizing :math:`G_\parallel^2` over the
    sign choices -- so a hexagonal slab is treated as consistently as an
    orthogonal one.
    """
    shape = tuple(int(n) for n in shape)
    cell = np.asarray(cell, dtype=float)
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    plane = [a for a in (0, 1, 2) if a != axis]

    # The derivation needs a genuine in-plane / axial split.
    a_c = cell[:, axis]
    length = float(np.linalg.norm(a_c))
    for a in plane:
        a_i = cell[:, a]
        cosine = abs(float(a_c @ a_i)) / (length * float(np.linalg.norm(a_i)))
        if cosine > SLAB_ORTHOGONALITY_TOLERANCE:
            raise ValueError(
                f"the slab-truncated kernel needs lattice vector {axis} (the "
                f"non-periodic one) perpendicular to the periodic plane, and it "
                f"makes an angle with vector {a} whose cosine is {cosine:.3e}.  "
                f"The kernel splits G into an in-plane part and an axial one, "
                f"which is only defined for a perpendicular axis; tilt the cell "
                f"so the vacuum direction is normal to the surface.")

    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    Q = reciprocal.T @ reciprocal
    axes = [sfft.fftfreq(n) * n for n in shape]
    m = np.meshgrid(*axes, indexing="ij")
    nyquist = [np.isclose(m[a], -(n // 2)) if n % 2 == 0
               else np.zeros(shape, dtype=bool)
               for a, n in enumerate(shape)]

    # In-plane |G_par|^2, with the same Nyquist symmetrization fft_g_squared
    # applies to |G|^2 (identical to the plain form for an orthogonal plane).
    g_par2 = np.zeros(shape, dtype=float)
    for a in plane:
        g_par2 += Q[a, a] * m[a] * m[a]
        for b in plane:
            if b == a:
                continue
            keep = ~(nyquist[a] | nyquist[b])
            g_par2 += Q[a, b] * np.where(keep, m[a] * m[b], 0.0)
    g_par2 = np.maximum(g_par2, 0.0)
    # Axial part: no cross terms survive the orthogonality guard above.
    g_z2 = Q[axis, axis] * m[axis] * m[axis]

    g_par = np.sqrt(g_par2)
    g_z = np.sqrt(np.maximum(g_z2, 0.0))          # the kernel is even in G_z
    g2 = g_par2 + g_z2
    half = 0.5 * length
    phase = g_z * half

    kernel = np.zeros(shape, dtype=float)
    inplane = g_par > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        bracket = 1.0 + np.exp(-g_par * half) * (
            np.where(inplane, g_z / np.where(inplane, g_par, 1.0), 0.0)
            * np.sin(phase) - np.cos(phase))
        kernel = np.where(inplane & (g2 > 0.0),
                          4.0 * np.pi * bracket / np.where(g2 > 0.0, g2, 1.0),
                          0.0)
        # G_par = 0, G_z != 0: the neutral-sheet branch.
        sheet = 2.0 * np.pi * (2.0 - 2.0 * np.cos(phase)
                               - g_z * length * np.sin(phase))
        kernel = np.where(~inplane & (g_z2 > 0.0),
                          sheet / np.where(g_z2 > 0.0, g_z2, 1.0),
                          kernel)
    kernel[~np.isfinite(kernel)] = 0.0
    return kernel


def slab_occupancy(rho, shape, axis: int) -> float:
    """Fraction of the non-periodic axis the density ``rho`` actually occupies.

    The truncation stops two points interacting once they are more than ``L/2``
    apart along ``axis``, so a density spread over more than half the cell is
    having real interactions cut.  Measured as the extent of the nodes carrying
    at least ``1e-6`` of the peak plane-averaged density, divided by ``L``.
    """
    rho = np.abs(np.asarray(rho, dtype=complex)).reshape(shape)
    profile = rho.sum(axis=tuple(a for a in (0, 1, 2) if a != int(axis)))
    peak = float(profile.max())
    if peak <= 0.0:
        return 0.0
    occupied = np.flatnonzero(profile >= 1e-6 * peak)
    if occupied.size == 0:
        return 0.0
    # The axis wraps, so the extent is the smallest arc covering every occupied
    # node: the complement's largest gap is what is *not* occupied.
    n = shape[int(axis)]
    gaps = np.diff(np.concatenate([occupied, occupied[:1] + n]))
    return float(n - (gaps.max() - 1)) / float(n)


class PeriodicPoissonSolver:
    r"""Solve the Coulomb convolution under **periodic** boundary conditions.

    :class:`PoissonFFTSolver` zero-pads precisely so the Coulomb tail cannot
    wrap the box: that is what makes it the potential of an *isolated* density,
    which is right for a molecule.  A crystal needs the opposite.  Here the
    wrap is the physics -- the density really is repeated on every lattice
    translation -- so the convolution is circular and is done in reciprocal
    space, where the periodic kernel is diagonal:

    .. math::

        \Phi(\mathbf G) = rac{4\pi}{|\mathbf G|^2}\,
ho(\mathbf G), \qquad
        \Phi(\mathbf G = 0) \equiv 0 .

    Dropping :math:`\mathbf G = 0` is not an approximation but a choice of
    reference: the term diverges for a charged cell, and for a neutral one it
    cancels exactly against the electron-ion and ion-ion :math:`\mathbf G = 0`
    terms.  Setting all three to zero -- a uniform neutralizing background, the
    jellium convention -- is consistent as long as **every** electrostatic term
    of the total energy uses it.  ``PlaneWaveIntegrals`` makes the same choice.

    There is no self-term to regularize: the real-space kernel's :math:`1/0`
    never appears, because the sum runs over reciprocal-lattice vectors.

    Parameters
    ----------
    shape : int or (int, int, int)
        Nodes per axis of the grid.  The grid **is** the periodic cell here:
        node ``n`` and node ``n + shape`` are the same point.
    step : (3, 3) array_like
        Voxel basis (step vectors as columns, i.e. ``Grid.step``), in Bohr.
        The cell is ``step @ diag(shape)``.
    spacing : float or (float, float, float), optional
        Used only when ``step`` is not given: an orthogonal voxel of these
        lengths.
    workers : int, optional
        Threads for the FFTs (``-1`` uses all cores).
    truncation : {"none", "slab"}
        ``"none"`` (default) is the fully periodic kernel described above.
        ``"slab"`` truncates the Coulomb interaction beyond half a cell along
        ``axis``, giving a system periodic in **two** directions and finite in
        the third -- the kernel a slab carrying a dipole needs, because its
        image interaction does not decay with vacuum thickness.  See
        :func:`slab_truncated_kernel`.
    axis : int, optional
        The non-periodic (vacuum) direction, required by ``truncation="slab"``.
    """

    def __init__(self, shape, step=None, spacing=None, workers: int = -1,
                 truncation: str = "none", axis: int | None = None):
        if np.isscalar(shape):
            self.shape = (int(shape),) * 3
        else:
            self.shape = tuple(int(s) for s in shape)
        if step is not None:
            self.step = np.asarray(step, dtype=float)
            if self.step.shape != (3, 3):
                raise ValueError(f"step must be a 3x3 matrix, got "
                                 f"{self.step.shape}")
        elif spacing is not None:
            values = ((float(spacing),) * 3 if np.isscalar(spacing)
                      else tuple(float(v) for v in spacing))
            if len(values) != 3 or min(values) <= 0.0:
                raise ValueError(f"spacing must be three positive lengths, got "
                                 f"{spacing!r}")
            self.step = np.diag(np.asarray(values, dtype=float))
        else:
            raise ValueError("give either a spacing or a step matrix")
        self.dV = abs(float(np.linalg.det(self.step)))
        if self.dV <= 0.0:
            raise ValueError("the voxel has zero volume (degenerate step)")
        self.cell = self.step @ np.diag(self.shape)
        self.volume = abs(float(np.linalg.det(self.cell)))
        #: FFT transform shape, the name the engine sizes its blocks from.  No
        #: padding here -- the wraparound the isolated solver pads against is
        #: the periodicity -- so the transform is the grid itself.
        self.L = self.shape
        self.workers = workers
        name = str(truncation).strip().lower()
        if name not in KERNEL_TRUNCATIONS:
            raise ValueError(f"unknown truncation {truncation!r}; use one of "
                             f"{KERNEL_TRUNCATIONS}")
        #: Which Coulomb kernel this solver carries (:data:`KERNEL_TRUNCATIONS`).
        self.truncation = name
        if name == "slab":
            if axis is None:
                raise ValueError(
                    "truncation='slab' needs axis=, the non-periodic (vacuum) "
                    "direction: the kernel is built by cutting the Coulomb "
                    "interaction along it, so there is no sensible default")
            self.axis = int(axis)
        elif axis is not None:
            raise ValueError(
                f"axis={axis!r} means nothing with truncation={name!r}: the "
                f"fully periodic kernel has no distinguished direction")
        else:
            self.axis = None
        self._kernel = self._build_kernel()

    def _build_kernel(self) -> np.ndarray:
        """The reciprocal-space kernel this solver convolves with.

        ``4 pi / G^2`` on the FFT grid with ``G = 0`` set to zero, or the
        two-dimensionally truncated kernel when ``truncation="slab"``.

        ``|G|^2`` comes from :func:`fft_g_squared`, whose Nyquist plane is
        symmetrized; for an orthogonal cell that is identical to the plain
        quadratic form, so no existing energy changes.
        """
        if self.truncation == "slab":
            return slab_truncated_kernel(self.shape, self.cell, self.axis)
        g_squared = fft_g_squared(self.shape, self.cell)
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = np.where(g_squared > 0.0, 4.0 * np.pi / g_squared, 0.0)
        return kernel

    def solve(self, rho_flat: np.ndarray) -> np.ndarray:
        """Periodic Coulomb potential of one density on the grid (flattened)."""
        return self.solve_stack(rho_flat[None, :])[0]

    def solve_stack(self, rho_stack: np.ndarray) -> np.ndarray:
        """Periodic Coulomb potentials of a stack of ``P`` densities.

        The FFT normalization works out so that no volume factor is needed: with
        ``rho(G) = fftn(rho)/N`` and ``Phi(r) = sum_G Phi(G) e^{iGr}``, the
        forward and inverse ``1/N`` cancel.
        """
        nx, ny, nz = self.shape
        rho_stack = np.ascontiguousarray(rho_stack, dtype=np.complex128)
        out = np.empty((rho_stack.shape[0], nx * ny * nz), dtype=np.complex128)
        for index in range(rho_stack.shape[0]):
            spectrum = sfft.fftn(rho_stack[index].reshape(nx, ny, nz),
                                 workers=self.workers)
            phi = sfft.ifftn(spectrum * self._kernel, workers=self.workers)
            out[index] = phi.reshape(-1)
        return out
