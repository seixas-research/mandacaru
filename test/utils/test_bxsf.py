# -*- coding: utf-8 -*-
# file: test/utils/test_bxsf.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""XCrySDen ``.bxsf`` band grids: the general grid's closing planes, the
header and a round trip."""

import numpy as np
import pytest

from mandacaru.utils.bxsf import read_bxsf, write_bxsf

CELL = np.array([[1.0, 0.2, 0.0], [0.0, 1.3, 0.1], [0.3, 0.0, 0.9]])


def _bands(shape=(2, 4, 3, 5), seed=7):
    return np.random.default_rng(seed).normal(size=shape) * 3.0


class TestTheFile:
    def test_a_round_trip_returns_the_periodic_mesh(self, tmp_path):
        energies = _bands()
        path = write_bxsf(tmp_path / "bands.bxsf", energies, CELL, 1.25)
        grid = read_bxsf(path)
        assert grid.energies.shape == energies.shape
        assert np.allclose(grid.energies, energies, atol=5e-7)
        assert np.allclose(grid.reciprocal_cell, CELL, atol=1e-10)
        assert grid.fermi_level == pytest.approx(1.25)

    def test_the_general_grid_closes_each_direction(self, tmp_path):
        """n + 1 points per direction, the last plane a copy of the first,
        k3 innermost."""
        energies = _bands(shape=(1, 2, 3, 4))
        text = write_bxsf(tmp_path / "b.bxsf", energies, CELL, 0.0)
        lines = open(text).read().splitlines()
        start = next(i for i, l in enumerate(lines)
                     if l.strip().startswith("BEGIN_BANDGRID_3D"))
        assert lines[start + 2].split() == ["3", "4", "5"]
        values = np.array([float(x) for l in lines[start + 8:-2]
                           for x in l.split()]).reshape(3, 4, 5)
        assert np.allclose(values[:-1, :-1, :-1], energies[0], atol=5e-7)
        assert np.allclose(values[-1], values[0])
        assert np.allclose(values[:, -1], values[:, 0])
        assert np.allclose(values[..., -1], values[..., 0])
        assert values[0, 0, 1] == pytest.approx(energies[0, 0, 0, 1], abs=5e-7)

    def test_bad_input_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="n_bands, n1, n2, n3"):
            write_bxsf(tmp_path / "x.bxsf", np.zeros((4, 3, 5)), CELL, 0.0)
        with pytest.raises(ValueError, match="independent"):
            write_bxsf(tmp_path / "x.bxsf", _bands(), np.zeros((3, 3)), 0.0)
        with pytest.raises(ValueError, match="finite"):
            write_bxsf(tmp_path / "x.bxsf", _bands(), CELL, float("nan"))
