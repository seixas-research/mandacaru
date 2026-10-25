# -*- coding: utf-8 -*-
"""Batched atom-centered sampling for molecular and periodic quadratures.

Use the same native spline and harmonic evaluator as Bloch sums, with a
single zero translation for isolated functions. All magnetic components of a
radial shell share interpolation. Quadrature nodes and weights are unchanged.
"""
from __future__ import annotations

import numpy as np


def radial_spline(function):
    """Return native cubic coefficients, cached by the actual spline object."""
    spline = getattr(function, "_spline", None)
    table = getattr(function, "table", None)
    values = getattr(table, "values", getattr(function, "_values", None))
    if spline is None or values is None or not hasattr(spline, "c") \
            or spline.c.shape[0] != 4:
        return None
    cached = getattr(function, "_native_radial_spline", None)
    if cached is None or cached[0] is not spline:
        rc = float(getattr(function, "r_c", np.inf))
        packed = (np.asarray(spline.x, dtype=float),
                  np.ascontiguousarray(spline.c.T, dtype=float), rc)
        function._native_radial_spline = (spline, packed)
    return function._native_radial_spline[1]


_TRANSFORM_CACHE: dict = {}


def _real_transform(l: int) -> np.ndarray:
    cached = _TRANSFORM_CACHE.get(l)
    if cached is None:
        from ..basis._angular import real_harmonic_transform
        cached = real_harmonic_transform(l)
        _TRANSFORM_CACHE[l] = cached
    return cached


def _center_node_indices(points, center, cache: dict) -> np.ndarray:
    hit = cache.get(center)
    if hit is None:
        cand = np.flatnonzero(np.abs(points[0] - center[0]) < 1e-12)
        if cand.size:
            dx = points[0][cand] - center[0]
            dy = points[1][cand] - center[1]
            dz = points[2][cand] - center[2]
            hit = cand[(dx * dx + dy * dy + dz * dz) < 1e-24]
        else:
            hit = cand
        cache[center] = hit
    return hit


def sample_functions(functions, points, *, gradients=False, delta=1e-3,
                     values=True, out_samples=None, out_slopes=None):
    """Return values ``(M, N)`` and optional gradients ``(M, 3, N)``.

    Real and complex harmonics, polarization and diffuse shells use the same
    native kernel. Arbitrary user-defined functions retain their evaluation
    contract. No spatial, functional, spin or element-specific approximation
    is made here.
    """
    from . import _backend

    points = tuple(np.ascontiguousarray(p, dtype=float).ravel()
                   for p in np.broadcast_arrays(*points))
    size = len(points[0])
    real = all(getattr(f, "real", False) for f in functions)
    dtype = float if real else complex

    if out_samples is not None:
        samples = out_samples
    elif values:
        samples = np.empty((len(functions), size), dtype=dtype)
    else:
        samples = None

    if out_slopes is not None:
        slopes_is_out = True
        slopes = out_slopes
    elif gradients:
        slopes_is_out = False
        slopes = np.empty((3, len(functions), size), dtype=dtype)
    else:
        slopes_is_out = False
        slopes = None

    groups, fallback = {}, []
    for row, function in enumerate(functions):
        spline = radial_spline(function)
        if spline is None or not _backend.HAS_C_BACKEND or not all(
                hasattr(function, name) for name in ("center", "l", "m", "real")):
            fallback.append(row)
            continue
        table = getattr(function, "table", None)
        table_vals = getattr(table, "values", getattr(function, "_values", None))
        key = (id(table_vals), tuple(np.asarray(function.center, dtype=float)),
               int(function.l), bool(function.real), float(spline[2]))
        groups.setdefault(key, []).append(row)

    zero = np.zeros((1, 3))
    phase = np.ones((1, 1), dtype=complex)
    axes = ([-1, 0, 1, 2] if (values and gradients)
            else ([0, 1, 2] if gradients else [-1]))
    sqrt2 = np.sqrt(2.0)
    center_nodes: dict = {}
    has_real_shell = _backend.HAS_C_BACKEND and hasattr(_backend._LIB, "mandacaru_real_shell")

    for (_, center, l, is_real, _rc), members in groups.items():
        first = functions[members[0]]
        spline = radial_spline(first)
        sup_r = getattr(first, "support_radius", None)
        max_break = float(spline[0][-1])
        if sup_r is not None:
            reach = min(max_break, float(sup_r) + (float(delta) if gradients else 0.0))
            support = np.nextafter(reach, np.inf)
        else:
            support = np.nextafter(max_break, np.inf)

        if is_real and has_real_shell:
            ms = [int(functions[i].m) for i in members]
            rows = members
            used = True
            for axis in axes:
                if axis == -1:
                    target = samples
                else:
                    target = slopes[axis] if (slopes.shape[0] == 3) else slopes[:, axis]
                ok = _backend.real_shell(points, center, support, spline, l, ms, rows, target,
                                         derivative=axis)
                if not ok:
                    used = False
                    break
            if not used:
                fallback.extend(members)
        else:
            if is_real:
                ms = sorted({abs(int(functions[i].m)) for i in members})
                m_pos = {m_val: idx for idx, m_val in enumerate(ms)}
            else:
                ms = [int(functions[i].m) for i in members]
            rows = list(range(len(ms)))
            native = np.empty((1, len(ms), size), dtype=complex)
            for axis in axes:
                native.fill(0.0)
                used = _backend.bloch_shell(points, center, zero, phase, support,
                                             spline, l, ms, rows, native,
                                             derivative=axis)
                if not used:
                    fallback.extend(members)
                    break
                if axis == -1:
                    target = samples
                else:
                    target = slopes[axis] if (slopes.shape[0] == 3) else slopes[:, axis]
                if is_real:
                    for row_idx in members:
                        m_val = int(functions[row_idx].m)
                        order = abs(m_val)
                        col = native[0, m_pos[order]]
                        if order == 0:
                            target[row_idx] = col.real
                        else:
                            scale = ((-1) ** order) * sqrt2
                            target[row_idx] = scale * (col.real if m_val > 0 else col.imag)
                else:
                    target[members] = native[0]

        if gradients:
            # Preserve the basis contract at a nucleus, including its tiny
            # offset convention for spline endpoint derivatives.
            on_center = _center_node_indices(points, center, center_nodes)
            if on_center.size:
                node_points = tuple(p[on_center] for p in points)
                for row in members:
                    g = functions[row].gradient(*node_points, delta)
                    if slopes.shape[0] == 3:
                        slopes[:, row, on_center] = g
                    else:
                        slopes[row, :, on_center] = g

    if fallback:
        from ..basis.base import values_and_gradients
        selected = [functions[i] for i in fallback]
        if gradients:
            val_arr, derivative = values_and_gradients(selected, *points, delta)
            if samples is not None:
                samples[fallback] = val_arr
            if slopes.shape[0] == 3:
                slopes[:, fallback] = derivative.transpose(1, 0, 2)
            else:
                slopes[fallback] = derivative
            if samples is not None:
                for row in fallback:
                    function = functions[row]
                    if hasattr(function, "center"):
                        c_key = tuple(np.asarray(function.center, dtype=float))
                        node = _center_node_indices(points, c_key, center_nodes)
                        if node.size:
                            samples[row, node] = function.evaluate(*(p[node] for p in points))
        else:
            # Share geometry across magnetic and zeta partners even without C.
            from ..basis._angular import spherical_coords, spherical_harmonic
            by_center = {}
            for row in fallback:
                function = functions[row]
                if not all(hasattr(function, name) for name in
                           ("radial", "l", "m", "real", "center")):
                    samples[row] = np.asarray(function.evaluate(*points)).ravel()
                    continue
                key = tuple(np.asarray(function.center, dtype=float))
                by_center.setdefault(key, []).append(row)
            for center, members in by_center.items():
                radius, theta, phi = spherical_coords(*points, center)
                harmonics, radials = {}, {}
                for row in members:
                    function = functions[row]
                    angular = (function.l, function.m, function.real)
                    if angular not in harmonics:
                        harmonics[angular] = spherical_harmonic(
                            *angular[:2], theta, phi, real=angular[2])
                    table = getattr(function, "table", None)
                    table_values = getattr(
                        table, "values", getattr(function, "_values", None))
                    radial = ((id(table_values), function.l)
                              if table_values is not None else (row,))
                    if radial not in radials:
                        radials[radial] = function.radial(radius)
                    samples[row] = radials[radial] * harmonics[angular]
    if slopes is not None and not slopes_is_out:
        slopes = slopes.transpose(1, 0, 2)
    return samples, slopes
