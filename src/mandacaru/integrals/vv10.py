# -*- coding: utf-8 -*-
# file: integrals/vv10.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""rVV10 nonlocal correlation on the real-space grid.

The VV10 nonlocal correlation (Vydrov and Van Voorhis, J. Chem. Phys. **133**,
244103 (2010)) is

.. math::

    E_c^{nl} = \int n(\mathbf r)\Big[\beta + \tfrac12\int n(\mathbf r')
               \Phi(\mathbf r, \mathbf r')\,d\mathbf r'\Big]d\mathbf r,
    \qquad \beta = \tfrac1{32}\Big(\frac{3}{b^2}\Big)^{3/4},

with :math:`\omega_0 = \sqrt{\omega_g^2 + \omega_p^2/3}`, :math:`\omega_g^2 =
C|\nabla n/n|^4`, :math:`\omega_p^2 = 4\pi n` and :math:`\kappa = b\,v_F^2/
\omega_p = \tfrac{3\pi}{2} b\,(n/9\pi)^{1/6}` (atomic units); :math:`\beta`
makes :math:`E_c^{nl}` vanish for the uniform gas.  The rVV10 kernel (Sabatini,
Gorni and de Gironcoli, Phys. Rev. B **87**, 041108(R) (2013)) replaces VV10's
by one that separates in :math:`q = \omega_0/\kappa`,

.. math::

    \Phi = -\frac32\,\frac{1}{(\kappa\kappa')^{3/2}}\,
           \phi(q, q', R), \qquad
    \phi = \frac{1}{(qR^2+1)(q'R^2+1)(qR^2+q'R^2+2)},

so the double integral is evaluated by the interpolation of Roman-Perez and
Soler (Phys. Rev. Lett. **103**, 096102 (2009)): on a mesh :math:`q_a` with
cardinal splines :math:`p_a(q)`, :math:`\phi(q, q', R) \approx \sum_{ab}
p_a(q)\,p_b(q')\,\phi(q_a, q_b, R)`, and with :math:`\theta_a = n\,
\kappa^{-3/2} p_a(q)`

.. math::

    E_c^{nl} = -\frac34 \sum_{ab} \int \theta_a\,(\phi_{ab} * \theta_b)
               + \beta N,

one FFT per :math:`\theta_a` and a product per pair in reciprocal space.

**The kernel's transform is analytic.**  In :math:`x = R^2` the kernel is a
sum of partial fractions, :math:`\phi = \sum_i c_i/(\alpha_i R^2 + 1)` with
:math:`(\alpha_i) = (a, b, (a+b)/2)` and :math:`(c_i) = (a^2, b^2,
-(a+b)^2/2)/(a-b)^2`, and :math:`1/(\alpha R^2 + 1)` transforms to
:math:`2\pi^2 e^{-k/\sqrt\alpha}/(\alpha k)` (:func:`kernel_transform`), so the
table needs no radial quadrature.

**Geometry.**  A crystal convolves on its cell; a molecule on the zero-padded
grid (:data:`ISOLATED_PADDING`), where the kernel's images are as far away as
the padding puts them.

The functional's derivatives come out as the partials of a GGA --
:math:`\partial E/\partial n(\mathbf r)` and :math:`\partial E/\partial
\sigma(\mathbf r)`, :math:`\sigma = |\nabla n|^2` -- so the caller adds them to
the semilocal ones and the Kohn-Sham potential, the forces and the spin
channels (which see the total density) follow the GGA path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import fft as sfft
from scipy.interpolate import CubicSpline

#: C of VV10 (fitted to C6 coefficients; every published pairing keeps it).
C_VV10 = 0.0093
#: b of r2SCAN+rVV10 (Ning et al., 2022: refit to the CCSD(T) Ar2 curve).
B_R2SCAN = 11.95

#: The interpolation mesh: logarithmic in q from :data:`Q_MIN` to
#: :data:`Q_CUT`, :data:`N_Q` points.  q above :data:`Q_CUT` is saturated
#: smoothly (:func:`saturate`); the regions it belongs to are density tails,
#: where theta ~ n^(3/4) carries almost nothing.
Q_MIN = 1e-4
Q_CUT = 0.5
N_Q = 16
#: Points of the kernel's radial table per unit of |G| (1/Bohr): linear
#: interpolation of the smooth exponentials is then good to ~1e-7 relative.
KERNEL_TABLE_DENSITY = 800
#: Order of the saturation polynomial (Roman-Perez and Soler use 12).
SATURATION_ORDER = 12
#: Densities below this carry no theta (and no potential).
DENSITY_CUTOFF = 1e-12
#: Each axis of a molecule's grid is zero-padded to this many times its
#: length (rounded up to an FFT-friendly size) before the convolution.
ISOLATED_PADDING = 2


def beta(b: float) -> float:
    r""":math:`\beta = (3/b^2)^{3/4}/32` (Hartree): zero nonlocal energy for
    the uniform gas."""
    return (3.0 / (b * b)) ** 0.75 / 32.0


def kappa_prefactor(b: float) -> float:
    r""":math:`\kappa / n^{1/6} = \tfrac{3\pi}{2} b\,(9\pi)^{-1/6}`
    (:math:`b\,v_F^2/\omega_p`)."""
    return b * 1.5 * np.pi * (9.0 * np.pi) ** (-1.0 / 6.0)


def q_mesh(q_min: float = Q_MIN, q_cut: float = Q_CUT,
           n_q: int = N_Q) -> np.ndarray:
    """The interpolation mesh, logarithmic from ``q_min`` to ``q_cut``."""
    return np.geomspace(q_min, q_cut, n_q)


def saturate(q, q_cut: float = Q_CUT, order: int = SATURATION_ORDER):
    r"""``(h(q), h'(q))``: :math:`h = q_c(1 - e^{-\sum_{m=1}^{M}(q/q_c)^m/m})`,
    equal to q for :math:`q \ll q_c` and to :math:`q_c` beyond it."""
    x = np.asarray(q, dtype=float) / q_cut
    total = np.zeros_like(x)
    derivative = np.zeros_like(x)
    power = np.ones_like(x)
    for m in range(1, order + 1):
        derivative += power                       # x^(m-1)
        power = power * x
        total += power / m
    damping = np.exp(-total)
    return q_cut * (1.0 - damping), damping * derivative


def kernel_real(a, b, R):
    r""":math:`\phi(a, b, R) = 1/[(aR^2+1)(bR^2+1)((a+b)R^2+2)]`."""
    x = np.asarray(R, dtype=float) ** 2
    return 1.0 / ((a * x + 1.0) * (b * x + 1.0) * ((a + b) * x + 2.0))


def kernel_transform(a: float, b: float, k) -> np.ndarray:
    r"""The 3D Fourier transform of :func:`kernel_real` at wave numbers ``k``.

    Partial fractions in :math:`R^2`: :math:`\phi = \sum_i c_i/(\alpha_i R^2
    + 1)`, each transforming to :math:`2\pi^2 e^{-k/\sqrt{\alpha_i}}/(\alpha_i
    k)`.  :math:`\sum_i c_i/\alpha_i = 0` (the kernel decays as
    :math:`R^{-6}`), so :math:`e^{-k/\sqrt\alpha}` enters as ``expm1`` and the
    transform is finite and stable at :math:`k \to 0`, where it is
    :math:`-2\pi^2\sum_i c_i\alpha_i^{-3/2}`.  At :math:`a = b` the kernel is
    :math:`1/(2(aR^2+1)^3)`, with transform :math:`\pi^2
    e^{-k/\sqrt a}(1 + k/\sqrt a)/(8a^{3/2})`.
    """
    k = np.asarray(k, dtype=float)
    a, b = float(a), float(b)
    if abs(a - b) <= 1e-12 * max(a, b):
        mu = 1.0 / np.sqrt(a)
        return np.pi ** 2 * np.exp(-k * mu) * (1.0 + k * mu) * mu ** 3 / 8.0
    d2 = (a - b) ** 2
    terms = ((a * a / d2, a), (b * b / d2, b),
             (-0.5 * (a + b) ** 2 / d2, 0.5 * (a + b)))
    out = np.zeros_like(k)
    small = k < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        for c, alpha in terms:
            s = 1.0 / np.sqrt(alpha)
            out += np.where(small, -c * s ** 3,
                            c * np.expm1(-k * s) / (alpha * np.where(small, 1.0, k)))
    return 2.0 * np.pi ** 2 * out


@lru_cache(maxsize=8)
def _kernel_table(q_min: float, q_cut: float, n_q: int, k_max: float):
    """``(dk, table)``: :func:`kernel_transform` of every mesh pair on a
    uniform grid in k from 0 past ``k_max`` -- built once per mesh and
    range, then interpolated (evaluating the exponentials on every G of
    every call was 87 % of the time)."""
    mesh = q_mesh(q_min, q_cut, n_q)
    count = int(np.ceil(k_max * KERNEL_TABLE_DENSITY)) + 2
    k = np.arange(count) / KERNEL_TABLE_DENSITY
    table = np.empty((n_q, n_q, count))
    for a in range(n_q):
        for b in range(a, n_q):
            table[a, b] = table[b, a] = kernel_transform(mesh[a], mesh[b], k)
    return 1.0 / KERNEL_TABLE_DENSITY, table


@lru_cache(maxsize=4)
def _splines(q_min: float, q_cut: float, n_q: int):
    """Cardinal cubic splines of the mesh in ln q: values and derivatives."""
    mesh = q_mesh(q_min, q_cut, n_q)
    spline = CubicSpline(np.log(mesh), np.eye(n_q), axis=0, bc_type="natural")
    return mesh, spline, spline.derivative()


def cardinal(q, q_min: float = Q_MIN, q_cut: float = Q_CUT, n_q: int = N_Q):
    r"""``(p, dp/dq)``, each ``(n_q, len(q))``: the cardinal splines
    :math:`p_a(q)` (in :math:`\ln q`) and their derivatives, constant below
    the mesh."""
    mesh, spline, derivative = _splines(q_min, q_cut, n_q)
    q = np.clip(np.asarray(q, dtype=float), mesh[0], mesh[-1])
    x = np.log(q)
    p = spline(x).T
    dp = derivative(x).T / q
    dp[:, np.asarray(q) <= mesh[0]] = 0.0
    return p, dp


@dataclass(frozen=True)
class NonlocalTerms:
    """rVV10 on one density: the energy (Hartree) and the GGA-type partials
    :math:`\\partial E/\\partial n` and :math:`\\partial E/\\partial\\sigma`
    on the flat grid (per unit volume)."""

    energy: float
    d_density: np.ndarray
    d_sigma: np.ndarray


def _local(n, sigma, b, C, q_min, q_cut, n_q):
    """theta_a and its partials in n and sigma, on the flat grid."""
    on = n > DENSITY_CUTOFF
    nn = np.where(on, n, 1.0)
    ss = np.where(on, sigma, 0.0)
    kappa = kappa_prefactor(b) * nn ** (1.0 / 6.0)
    omega_g2 = C * (ss / (nn * nn)) ** 2
    omega0 = np.sqrt(omega_g2 + 4.0 * np.pi * nn / 3.0)
    q = omega0 / kappa
    dw_dn = (-4.0 * omega_g2 / nn + 4.0 * np.pi / 3.0) / (2.0 * omega0)
    dw_ds = C * ss / (nn ** 4 * omega0)
    dq_dn = dw_dn / kappa - q / (6.0 * nn)
    dq_ds = dw_ds / kappa
    h, dh = saturate(q, q_cut)
    p, dp = cardinal(h, q_min, q_cut, n_q)
    weight = nn * kappa ** -1.5                 # n kappa^(-3/2)
    dweight_dn = kappa ** -1.5 * (1.0 - 0.25)   # d/dn [n^(1-1/4)] / n^(-1/4)
    theta = weight * p
    dtheta_dn = dweight_dn * p + weight * dp * dh * dq_dn
    dtheta_ds = weight * dp * dh * dq_ds
    mask = on.astype(float)
    return theta * mask, dtheta_dn * mask, dtheta_ds * mask


def _wavenumbers(grid, periodic):
    r"""``(|G|, padded shape)`` of the convolution box.

    A period's are averaged over the grid's lattice operations
    (:func:`~mandacaru.integrals.reciprocal.symmetric_aliases`), so the
    kernel, a function of :math:`|\mathbf G|`, is as symmetric as the
    crystal on any grid its operations map onto itself.
    """
    shape, step = tuple(grid.shape), grid.step
    if periodic:
        from .reciprocal import symmetric_aliases
        return symmetric_aliases(grid)[1], shape
    lengths = tuple(sfft.next_fast_len(ISOLATED_PADDING * n) for n in shape)
    cell = np.asarray(step, dtype=float) @ np.diag(np.asarray(lengths, float))
    B = 2.0 * np.pi * np.linalg.inv(cell).T
    m = [np.fft.fftfreq(L, d=1.0 / L) for L in lengths]
    shapes = [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]
    metric = B.T @ B
    g2 = np.zeros(lengths)
    for i in range(3):
        for j in range(3):
            g2 = g2 + metric[i, j] * m[i].reshape(shapes[i]) * m[j].reshape(shapes[j])
    return np.sqrt(np.maximum(g2, 0.0)), lengths


def nonlocal_correlation(grid, density: np.ndarray, sigma: np.ndarray, *,
                         b: float = B_R2SCAN, C: float = C_VV10,
                         q_min: float = Q_MIN, q_cut: float = Q_CUT,
                         n_q: int = N_Q) -> NonlocalTerms:
    r"""rVV10's :math:`E_c^{nl}` of ``density`` and its partials.

    ``density`` and ``sigma`` (:math:`|\nabla n|^2`) are flat arrays on
    ``grid``; the convolution is periodic on a periodic grid and zero-padded
    otherwise.
    """
    n = np.maximum(np.asarray(density, dtype=float), 0.0)
    sigma = np.asarray(sigma, dtype=float)
    shape, dV = tuple(grid.shape), float(grid.dV)
    theta, dtheta_dn, dtheta_ds = _local(n, sigma, b, C, q_min, q_cut, n_q)
    knorm, lengths = _wavenumbers(grid, bool(getattr(grid, "periodic", False)))
    slices = tuple(slice(0, s) for s in shape)
    transforms = []
    for a in range(n_q):
        pad = np.zeros(lengths)
        pad[slices] = theta[a].reshape(shape)
        transforms.append(sfft.rfftn(pad))
    half = knorm[..., : transforms[0].shape[-1]]
    dk, table = _kernel_table(q_min, q_cut, n_q,
                              round(float(half.max()) + 1.0, 0))
    position = half / dk
    index = position.astype(np.intp)
    frac = position - index
    accumulated = [np.zeros_like(transforms[0]) for _ in range(n_q)]
    for a in range(n_q):
        for b_ in range(a, n_q):
            row = table[a, b_]
            kernel = row[index] * (1.0 - frac) + row[index + 1] * frac
            accumulated[a] += kernel * transforms[b_]
            if b_ != a:
                accumulated[b_] += kernel * transforms[a]
    energy = 0.0
    u = np.zeros((n_q, n.size))
    for a in range(n_q):
        conv = sfft.irfftn(accumulated[a], s=lengths)[slices].reshape(-1)
        u[a] = conv
        energy += float(np.sum(theta[a] * conv) * dV)
    # E = -3/4 sum_ab int theta_a (phi_ab * theta_b) + beta N; dE/dtheta_a = -3/2 u_a.
    energy = -0.75 * energy + beta(b) * float(np.sum(n) * dV)
    d_density = beta(b) - 1.5 * np.sum(u * dtheta_dn, axis=0)
    d_sigma = -1.5 * np.sum(u * dtheta_ds, axis=0)
    return NonlocalTerms(energy=energy, d_density=d_density, d_sigma=d_sigma)
