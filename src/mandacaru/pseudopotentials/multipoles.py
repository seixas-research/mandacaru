# -*- coding: utf-8 -*-
# file: pseudopotentials/multipoles.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Partial-wave pairs of a PAW-LCAO dataset, handed to Rust.

The radial multipole moments :math:`\Delta^{(L)}_{ij} = \int (R_iR_j -
\tilde R_i\tilde R_j)\,r^{L+2}\,dr` between any two partial waves of a dataset are
computed in Rust (``paw_smooth_waves``, ``paw_radial_moments``); this module adapts the
dataset's arrays to those routines and carries no numerics of its own.  The shape
function, the Gaunt coefficients and the angular rule it once wrapped are called through
``mandacaru._native`` (``paw_shape_function``, ``paw_gaunt``, ``paw_angular_grid``).
"""

from __future__ import annotations

import numpy as np

from .. import _native
from .native_context import context


def _channel_arrays(dataset, l: int):
    channel = dataset.channels[int(l)]
    ae = np.ascontiguousarray([np.asarray(w, dtype=float)
                               for w in channel.ae_waves])
    return channel, ae


def partial_waves(dataset, l: int):
    """``(all-electron R_i, smooth R~_i)`` of channel ``l`` on ``dataset.r``.

    The smooth waves are rebuilt from the Bessel expansion the dataset stores
    inside the channel's ``r_cut``; beyond it they *are* the all-electron ones
    (matched there in value and three derivatives), which is what the
    expansion itself, divergent outside, must never be read as.
    """
    channel, ae = _channel_arrays(dataset, l)
    smooth = _native.paw_smooth_waves(
        context(), int(l), np.ascontiguousarray(dataset.r, dtype=float),
        float(channel.r_cut), ae,
        np.ascontiguousarray(channel.wave_coefficients, dtype=float),
        np.ascontiguousarray(channel.wavevectors, dtype=float))
    return list(ae), list(smooth)


def _pair(dataset, l1: int, l2: int):
    """The arguments of the native pair routines after ``(context, L)``: the grid, both channels'
    cutoff radii (Rust takes the larger) and the all-electron and smooth waves of each."""
    ae1, ps1 = partial_waves(dataset, l1)
    ae2, ps2 = partial_waves(dataset, l2)
    return (np.ascontiguousarray(dataset.r, dtype=float),
            float(dataset.channels[int(l1)].r_cut),
            float(dataset.channels[int(l2)].r_cut),
            np.array(ae1), np.array(ps1), np.array(ae2), np.array(ps2))


def radial_moments(dataset, l1: int, l2: int, L: int) -> np.ndarray:
    r"""``Delta^{(L)}_{ij} = int (R_i R_j - R~_i R~_j) r^{L+2} dr`` over the
    augmentation sphere, for waves ``i`` of channel ``l1`` and ``j`` of ``l2``.
    At ``l1 == l2`` and ``L == 0`` this is the stored ``overlap_correction``."""
    return _native.paw_radial_moments(context(), int(L), *_pair(dataset, l1, l2))
