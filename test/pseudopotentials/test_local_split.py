# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_local_split.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The short-range sphere of the local-potential split holds the dataset tail."""

import numpy as np
import pytest

from mandacaru.pseudopotentials import local_split
from mandacaru.pseudopotentials.environment import LibraryPathError
from mandacaru.pseudopotentials.local_split import (
    TAIL_TOLERANCE, short_range_radius, tail_radius)
from mandacaru.pseudopotentials.paw import get_paw


@pytest.mark.parametrize("symbol", ["Cd", "Zn", "Cu"])
def test_sphere_holds_the_dataset_tail(symbol):
    try:
        dataset = get_paw(symbol, paw_dataset="pbe-sr")
    except LibraryPathError:
        pytest.skip("MANDACARU_PAW_PATH is not configured here")
    # A fine grid shrinks sigma so far that 6.5 sigma is inside the table's
    # transition region: the sphere must still reach the measured tail.
    radius = short_range_radius(dataset, sigma=0.5 * 0.10 * 1.4)
    assert radius >= tail_radius(dataset) > 0.0
    r = np.linspace(radius, radius + 4.0, 200)
    deviation = np.abs(dataset.local_potential(r)
                       + dataset.valence_charge / r)
    assert deviation.max() <= 2 * TAIL_TOLERANCE


def test_tail_radius_is_cached_on_the_dataset():
    class Table:
        valence_charge = 2.0

        def local_potential(self, r):
            return -2.0 / r + np.where(r < 1.5, 0.1, 0.0)

    table = Table()
    first = tail_radius(table)
    assert 1.5 <= first <= 1.6 and table._tail_radius == first
    assert local_split.tail_radius(table) == first
