# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_local_split.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The local-potential split: the short-range sphere holds the dataset tail, and
every number of the split reproduces the frozen output of the Python it replaced
(``test/migration/reference/paw/local_split.npz``)."""

from pathlib import Path
from types import SimpleNamespace

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


REFERENCE = (Path(__file__).resolve().parents[1] / "migration" / "reference"
             / "paw" / "local_split.npz")
SYMBOLS = ["H", "O", "Cu", "Cd", "Zn", "Li"]
#: The radii of the frozen potentials: the origin branch, two radii where the
#: Gaussian ion's derivative is a difference of 1e9-sized terms (round-off
#: noise of 1e-7 in either implementation), a small radius, and the sphere.
RADII = np.concatenate([[0.0, 1e-14, 1e-9, 1e-3], np.linspace(0.05, 12.0, 400)])
NOISY = [1, 2]


def close(value, frozen, rtol=1e-9):
    """The largest difference, relative to the largest frozen entry."""
    frozen = np.asarray(frozen)
    scale = max(float(np.abs(frozen).max()), 1e-300)
    assert float(np.abs(np.asarray(value) - frozen).max()) <= rtol * scale


@pytest.fixture(scope="module")
def frozen():
    return np.load(REFERENCE)


@pytest.fixture(scope="module")
def datasets():
    try:
        return {symbol: get_paw(symbol) for symbol in SYMBOLS}
    except (LibraryPathError, FileNotFoundError):
        pytest.skip("MANDACARU_PAW_PATH is not configured here")


class TestFrozenPythonOutputs:
    """Rust against the pre-port Python, on the shipped lda-sr datasets."""

    def test_the_gaussian_ion(self, frozen):
        # The Gaussian ion's derivative has no Python route (nothing consumes
        # it); its frozen values are covered through the short-range
        # derivative below and the finite-difference test in
        # crates/mandacaru-core/tests/paw_local_split.rs.
        potentials = [local_split.long_range_potential(charge, sigma, RADII)
                      for charge, sigma in ((1.0, 0.3), (6.0, 0.66),
                                            (3.0, 0.5))]
        close(np.stack(potentials), frozen["long_range"][0::2])

    def test_the_short_range_difference(self, frozen, datasets):
        values, slopes = [], []
        for symbol in SYMBOLS:
            for sigma in (0.3, 0.5, 0.8):
                values.append(local_split.short_range_potential(
                    datasets[symbol], sigma, RADII))
                slopes.append(local_split.short_range_potential_derivative(
                    datasets[symbol], sigma, RADII))
        keep = [i for i in range(RADII.size) if i not in NOISY]
        close(values, frozen["short_range"])
        close(np.stack(slopes)[:, keep],
              frozen["short_range_derivative"][:, keep])

    def test_the_radii(self, frozen, datasets):
        pp = [datasets[symbol] for symbol in SYMBOLS]
        close([local_split.tail_radius(d) for d in pp], frozen["tail_radius"])
        close([local_split.local_cutoff(d) for d in pp],
              frozen["local_cutoff"])
        close([[local_split.short_range_radius(d, sigma)
                for sigma in (0.1, 0.3, 0.5, 0.8)] for d in pp],
              frozen["short_range_radius"])

    def test_the_quadrature_rules(self, frozen):
        cases = [(3.0, 1.2, 64), (3.0, 0.0, 32), (2.2, 2.2, 32),
                 (0.7, 1.5, 16), (4.5, 1.0, 48)]
        for k, (radius, panel, points) in enumerate(cases):
            close(np.stack(local_split._radial_rule(radius, panel, points)),
                  frozen[f"radial_rule_{k}"])
        for k, (polar, azimuthal) in enumerate([(32, 64), (16, 36), (8, 12)]):
            directions, weights = local_split._angular_rule(polar, azimuthal)
            close(directions, frozen[f"angular_directions_{k}"])
            close(weights, frozen[f"angular_weights_{k}"])

    def test_the_gaussian_width_follows_the_coarsest_lattice_vector(self, frozen):
        steps = [np.diag([0.37, 0.37, 0.37]),
                 np.array([[0.4, 0.1, 0.0], [0.0, 0.3, 0.05],
                           [0.0, 0.0, 0.5]])]
        close([local_split.split_width(SimpleNamespace(step=s))
               for s in steps], frozen["split_width"])
