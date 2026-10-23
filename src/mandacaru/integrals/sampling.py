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


def sample_functions(functions, points, *, gradients=False, delta=1e-3):
    """Return values ``(M, N)`` and optional gradients ``(M, 3, N)``.

    Real and complex harmonics, polarization and diffuse shells use the same
    native kernel. Arbitrary user-defined functions retain their evaluation
    contract. No spatial, functional, spin or element-specific approximation
    is made here.
    """
    from . import _backend
    from ..basis._angular import real_harmonic_transform

    points = tuple(np.ascontiguousarray(p, dtype=float).ravel()
                   for p in np.broadcast_arrays(*points))
    size = len(points[0])
    real = all(getattr(f, "real", False) for f in functions)
    dtype = float if real else complex
    samples = np.empty((len(functions), size), dtype=dtype)
    slopes = np.empty((len(functions), 3, size), dtype=dtype) if gradients else None
    groups, fallback = {}, []
    for row, function in enumerate(functions):
        spline = radial_spline(function)
        if spline is None or not _backend.HAS_C_BACKEND or not all(
                hasattr(function, name) for name in ("center", "l", "m", "real")):
            fallback.append(row)
            continue
        table = getattr(function, "table", None)
        values = getattr(table, "values", getattr(function, "_values", None))
        key = (id(values), tuple(np.asarray(function.center, dtype=float)),
               int(function.l), bool(function.real), float(spline[2]))
        groups.setdefault(key, []).append(row)

    zero = np.zeros((1, 3))
    phase = np.ones((1, 1), dtype=complex)
    for (_, center, l, is_real, _rc), members in groups.items():
        first = functions[members[0]]
        spline = radial_spline(first)
        ms = list(range(-l, l + 1)) if is_real else [functions[i].m for i in members]
        rows = list(range(len(ms)))
        support = np.nextafter(float(spline[0][-1]), np.inf)
        transform = real_harmonic_transform(l) if is_real else None
        for axis in ([-1, 0, 1, 2] if gradients else [-1]):
            native = np.zeros((1, len(ms), size), dtype=complex)
            used = _backend.bloch_shell(points, center, zero, phase, support,
                                         spline, l, ms, rows, native,
                                         derivative=axis)
            if not used:
                fallback.extend(members)
                break
            values = ((transform @ native[0]).real if is_real else native[0])
            indices = [functions[i].m + l for i in members] if is_real else rows
            if axis == -1:
                samples[members] = values[indices]
            else:
                slopes[members, axis] = values[indices]
        if gradients:
            # Preserve the basis contract at a nucleus, including its tiny
            # offset convention for spline endpoint derivatives.
            on_center = sum((p - c)**2 for p, c in zip(points, center)) < 1e-24
            if np.any(on_center):
                node_points = tuple(p[on_center] for p in points)
                for row in members:
                    slopes[row][:, on_center] = functions[row].gradient(*node_points, delta)

    if fallback:
        from ..basis.base import values_and_gradients
        selected = [functions[i] for i in fallback]
        if gradients:
            values, derivative = values_and_gradients(selected, *points, delta)
            samples[fallback], slopes[fallback] = values, derivative
            for row in fallback:
                function = functions[row]
                if hasattr(function, "center"):
                    node = sum((p - c)**2 for p, c in zip(points, function.center)) < 1e-24
                    if np.any(node):
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
    return samples, slopes
