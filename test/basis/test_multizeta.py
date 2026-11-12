# -*- coding: utf-8 -*-
# file: test/test_multizeta.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Multiple-zeta and polarized numerical atomic orbitals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.basis.multizeta import RadialTable, zeta_tables
from mandacaru.models.radial import DEFAULT_NAO_SIZE, ZETA_NAMES, resolve_zeta
from mandacaru.pseudopotentials.native_context import context
from mandacaru.integrals import Grid


FROZEN = Path(__file__).resolve().parents[1] / "migration" / "reference" / "basis_zetas"
HYDROGEN = json.loads((FROZEN / "zeta_tables.json").read_text())["hydrogen_1s"]


@pytest.fixture(scope="module")
def frozen():
    """What the scipy implementation (``split_radius``, ``split_valence_tail``,
    ``zeta_tables``) returned for the hydrogen 1s below, frozen before it was
    replaced by ``_native.radial_zetas``."""
    return np.load(FROZEN / "zeta_tables.npz")


@pytest.fixture(scope="module")
def hydrogen_1s():
    """A confined hydrogen 1s: the reference first zeta."""
    solved = _native.Context(threads=1).confined_radial(1, 0, 1.0, 4.0, intervals=800)
    return solved["r"], solved["radial"]


class TestZetaNames:
    """Parsing the basis-size notation."""

    @pytest.mark.parametrize("name, expected", [
        ("SZ", (1, 0)), ("DZ", (2, 0)), ("DZP", (2, 1)),
        ("TZP", (3, 1)), ("TZ2P", (3, 2)), ("QZP", (4, 1)),
    ])
    def test_known_names(self, name, expected):
        assert resolve_zeta(name) == expected

    def test_case_and_separators_are_ignored(self):
        assert resolve_zeta("dz-p") == resolve_zeta(" DZP ") == (2, 1)

    def test_explicit_pair_passes_through(self):
        assert resolve_zeta((3, 2)) == (3, 2)

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown basis size"):
            resolve_zeta("PZ")

    def test_every_name_is_self_consistent(self):
        """The name encodes its own (n_zeta, n_polarization)."""
        digits = {"S": 1, "D": 2, "T": 3, "Q": 4}
        for name, (n_zeta, n_pol) in ZETA_NAMES.items():
            assert digits[name[0]] == n_zeta
            assert name.count("P") == min(n_pol, 1)


class TestSplitValence:
    """The split-valence construction of the extra zetas."""

    @staticmethod
    def split_of(r, radial, **options):
        """The second zeta and the split radius it was cut at: the first node
        past its support, since the tail is exactly zero from there on."""
        zeta = zeta_tables(r, radial, 1, 0, 2, **options)[1].values
        return zeta, float(r[np.flatnonzero(zeta)[-1] + 1])

    def test_the_input_is_the_frozen_hydrogen(self, hydrogen_1s, frozen):
        r, radial = hydrogen_1s
        np.testing.assert_allclose(r, frozen["h1s_r"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(radial, frozen["h1s_values"], rtol=1e-12,
                                   atol=1e-14)

    def test_the_zetas_match_the_frozen_construction(self, hydrogen_1s, frozen):
        r, radial = hydrogen_1s
        tables = zeta_tables(r, radial, 1, 0, 4)
        assert len(tables) == HYDROGEN["count_out"]
        for k, table in enumerate(tables, start=1):
            expected = frozen[f"h1s_zeta{k}"]
            scale = np.abs(expected).max()
            assert np.abs(table.values - expected).max() <= 1e-12 * scale

    def test_split_radius_leaves_the_requested_tail_norm(self, hydrogen_1s,
                                                         frozen):
        r, radial = hydrogen_1s
        for norm, expected in zip(HYDROGEN["split_norms"],
                                  HYDROGEN["split_radii"]):
            _, r_s = self.split_of(r, radial, split_norm=norm)
            assert r_s == pytest.approx(expected, rel=1e-14)
        for target in (0.05, 0.15, 0.30):
            _, r_s = self.split_of(r, radial, split_norm=target)
            density = radial ** 2 * r ** 2
            tail = np.trapezoid(density[r >= r_s], r[r >= r_s])
            assert tail == pytest.approx(target, abs=2e-3)

    def test_larger_split_norm_pushes_the_radius_inward(self, hydrogen_1s):
        r, radial = hydrogen_1s
        assert (self.split_of(r, radial, split_norm=0.30)[1]
                < self.split_of(r, radial, split_norm=0.05)[1])

    def test_tail_vanishes_beyond_the_split_radius(self, hydrogen_1s):
        r, radial = hydrogen_1s
        zeta, r_s = self.split_of(r, radial)
        assert np.all(zeta[r >= r_s] == 0.0)
        assert np.any(np.abs(zeta[r < r_s]) > 1e-6)

    def test_tail_is_continuous_at_the_split_radius(self, hydrogen_1s, frozen):
        """Value and slope are matched there, so no discontinuity is injected:
        the (unnormalized) frozen tail is zero beyond ``r_s`` and the
        normalized zeta is that tail over its norm."""
        r, radial = hydrogen_1s
        zeta, r_s = self.split_of(r, radial)
        tail = frozen["h1s_tail_default"]
        inside = np.nonzero(r < r_s)[0]
        assert abs(tail[inside[-1]]) < 1e-3
        norm = np.sqrt(np.trapezoid(tail * tail * r * r, r))
        np.testing.assert_allclose(zeta * norm, tail, rtol=0,
                                   atol=1e-12 * np.abs(tail).max())

    def test_zeta_hierarchy_is_strictly_shorter_ranged(self, hydrogen_1s):
        """Each added zeta lives inside the previous one."""
        r, radial = hydrogen_1s
        tables = zeta_tables(r, radial, 1, 0, 4)
        assert [t.zeta for t in tables] == [1, 2, 3, 4]

        ranges, norms = [], []
        for table in tables:
            support = np.nonzero(np.abs(table.values) > 1e-12)[0]
            ranges.append(r[support.max()])
            norms.append(np.trapezoid(table.values ** 2 * r ** 2, r))

        assert ranges == sorted(ranges, reverse=True)
        # Every zeta is normalized (the extra ones used to carry a few percent
        # of the norm, which conditioned the overlap matrix needlessly).
        assert norms == pytest.approx([1.0] * len(norms), abs=1e-3)

    def test_zeta_count_is_honored(self, hydrogen_1s):
        r, radial = hydrogen_1s
        for n_zeta in (1, 2, 3, 4):
            assert len(zeta_tables(r, radial, 1, 0, n_zeta)) == n_zeta


def sample(table, m, points, *, center=(0.0, 0.0, 0.0), real=True, gradient=False):
    """``R(r) Y_lm`` (or its gradient) of a radial table at ``points`` (3, n), sampled by the engine."""
    x, y, z = (np.ascontiguousarray(a, dtype=float) for a in points)
    return _native.radial_orbital_sample(
        context(), np.ascontiguousarray(table.r, dtype=float),
        np.ascontiguousarray(table.values, dtype=float), table.l, m, x, y, z,
        list(center), real, gradient)


class TestSampledOrbital:
    """The basis function built on a radial table, sampled by ``_native.radial_orbital_sample``."""

    def test_confined_outside_the_cutoff(self, hydrogen_1s):
        r, radial = hydrogen_1s
        far = sample(RadialTable(r, radial, 1, 0), 0, ([50.0], [0.0], [0.0]))
        assert far[0] == 0.0

    def test_normalized_on_a_grid(self, hydrogen_1s):
        r, radial = hydrogen_1s
        grid = Grid(center=[0, 0, 0], box_size=6.0, grid_spacing=0.10)
        values = sample(RadialTable(r, radial, 1, 0), 0,
                        (grid.X.ravel(), grid.Y.ravel(), grid.Z.ravel()))
        norm = float(np.sum(np.abs(values) ** 2) * grid.dV)
        assert norm == pytest.approx(1.0, abs=2e-2)


class TestAnalyticGradient:
    """``gradient`` is the derivative of the samples -- the spline's and the
    harmonic's -- not a finite-difference estimate of them."""

    CENTER = np.array([0.1, -0.2, 0.3])

    @pytest.mark.parametrize("real", [True, False])
    @pytest.mark.parametrize("l, m", [(0, 0), (1, -1), (1, 0), (1, 1),
                                      (2, -2), (2, 1), (2, 2), (3, -2)])
    def test_it_is_the_derivative_of_the_samples(self, l, m, real):
        r = np.linspace(0.05, 6.0, 300)
        table = RadialTable(r, r ** l * np.exp(-r), l + 1, l)
        rng = np.random.default_rng(10 * l + m + 5)
        points = self.CENTER[:, None] + rng.uniform(-4.0, 4.0, size=(3, 120))
        # Inside the first table point (the power-law continuation), a few
        # hundredths of a Bohr from the center, and on the center itself.
        points = np.hstack([points,
                            self.CENTER[:, None]
                            + 0.02 * rng.normal(size=(3, 20)),
                            self.CENTER[:, None]])
        step = 1e-5

        def value(p):
            return sample(table, m, p, center=self.CENTER, real=real)

        numeric = np.stack([
            (value(points + step * np.eye(3)[k][:, None])
             - value(points - step * np.eye(3)[k][:, None])) / (2.0 * step)
            for k in range(3)])
        analytic = sample(table, m, points, center=self.CENTER, real=real, gradient=True)
        assert analytic.dtype == (np.float64 if real else np.complex128)
        assert np.abs(analytic - numeric).max() < 1e-7


class TestBasisConstruction:
    """What ``{"name": "NAO", "size": ...}`` builds, counted by the native engine."""

    @staticmethod
    def count(symbol, **options):
        from ase import Atoms

        from mandacaru.integrations.ase import geometry_from_atoms
        from mandacaru.specifications import basis_specification

        atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
        spec = json.dumps(basis_specification(dict(name="NAO", **options)))
        return _native.Context(threads=1).basis_size(geometry_from_atoms(atoms), basis_spec=spec)

    @pytest.mark.parametrize("symbol, size, n_functions", [
        # H: 1 s shell; polarization adds a p shell (3 functions).
        ("H", "SZ", 1), ("H", "DZ", 2), ("H", "TZ", 3), ("H", "QZ", 4),
        ("H", "DZP", 2 + 3), ("H", "TZP", 3 + 3),
        # O: s + p = 4 functions per zeta; polarization adds d (5 functions).
        ("O", "SZ", 4), ("O", "DZ", 8),
        ("O", "DZP", 8 + 5), ("O", "TZP", 12 + 5),
    ])
    def test_function_counts(self, symbol, size, n_functions):
        assert self.count(symbol, size=size) == n_functions

    def test_default_size_is_dzp(self):
        """DZP is the default: 8 (double-zeta s+p) + 5 (d polarization)."""
        assert DEFAULT_NAO_SIZE == "DZP"
        assert self.count("O") == 13 == self.count("O", size="DZP")
