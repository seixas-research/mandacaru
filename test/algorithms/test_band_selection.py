# -*- coding: utf-8 -*-
# file: test/algorithms/test_band_selection.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Choosing the states of a Wannierization by what they are: the
projectability onto target atomic orbitals, the windows it sets, and the
erfc-weighted selected columns of the density matrix."""

import numpy as np
import pytest
from ase.build import bulk

from mandacaru import Mandacaru
from mandacaru.algorithms.band_selection import (
    DEFAULT_FROZEN_PROJECTABILITY, DEFAULT_OUTER_PROJECTABILITY,
    erfc_fit, frozen_window, projectability_masks, resolve_selection,
    scdm_subspace)
from mandacaru.integrals import reciprocal as rc
from mandacaru.units import HARTREE_TO_EV


class TestTheWindows:
    def test_the_names_resolve(self):
        auto = resolve_selection("auto")
        assert auto.kind == "projectability"
        assert (auto.outer, auto.frozen) == (DEFAULT_OUTER_PROJECTABILITY,
                                             DEFAULT_FROZEN_PROJECTABILITY)
        given = resolve_selection({"projectability": (0.05, 0.8)})
        assert (given.outer, given.frozen) == (0.05, 0.8)
        assert resolve_selection("scdm").mu is None
        scdm = resolve_selection({"scdm": (3.0, 0.5)})
        assert (scdm.kind, scdm.mu, scdm.sigma) == ("scdm", 3.0, 0.5)
        assert resolve_selection("rpa").kind == "natural"
        assert resolve_selection(auto) is auto

    def test_energy_windows_are_not_a_selection(self):
        assert resolve_selection(None) is None
        assert resolve_selection({"outer": (-5.0, 5.0)}) is None
        assert resolve_selection({"outer": (-5.0, 5.0),
                                  "frozen": (-5.0, 0.0)}) is None

    @pytest.mark.parametrize("windows", [
        "everything", {"projectability": (0.9, 0.1)},
        {"projectability": (0.1, 1.5)}, {"scdm": (1.0, -0.2)},
        {"scdm": (1.0, 0.2), "outer": (-1.0, 1.0)},
        {"projectability": (0.1, 0.9), "inner": (0.0, 1.0)},
        {"rpa": "yes"}])
    def test_bad_windows_are_refused(self, windows):
        with pytest.raises(ValueError):
            resolve_selection(windows)

    def test_the_masks_hold_the_count(self):
        p = np.array([[1.0, 0.99, 0.95, 0.93, 0.5, 0.01],
                      [0.3, 0.2, 0.1, 0.05, 0.001, 0.0]])
        inside, fixed = projectability_masks(p, 3, 0.02, 0.9)
        # Four states clear 0.9 at the first point: the three most
        # projectable are frozen.
        assert fixed[0].tolist() == [True, True, True, False, False, False]
        assert inside[0].tolist() == [True] * 5 + [False]
        # Too few clear the outer threshold at the second: the most
        # projectable join until there are three.
        assert inside[1].sum() >= 3 and not fixed[1].any()

    def test_the_frozen_window_is_the_longest_projectable_run(self):
        """Target-like states far up in energy do not join the window: a
        state below the threshold between them ends it."""
        energies = np.array([[-5.0, -1.0, 0.5, 3.0, 9.0],
                             [-4.0, -0.5, 1.5, 2.0, 10.0]])
        p = np.array([[1.0, 0.99, 0.97, 0.3, 0.99],
                      [1.0, 0.98, 0.96, 0.5, 0.97]])
        assert frozen_window(energies, p, 0.95, 4) == (-5.0, 1.5)
        inside, fixed = projectability_masks(p, 4, 0.01, 0.95,
                                             energies=energies)
        assert fixed.tolist() == [[True, True, True, False, False],
                                  [True, True, True, False, False]]
        # Narrowed when a k-point would hold more states than functions.
        assert frozen_window(energies, p, 0.95, 2)[1] < 0.5
        assert frozen_window(energies, 0.5 * p, 0.95, 4) is None

    def test_the_erfc_fit_recovers_its_parameters(self):
        from scipy.special import erfc

        e = np.linspace(-10.0, 10.0, 400)
        p = 0.5 * erfc((e - 1.3) / 0.7)
        mu, sigma = erfc_fit(e, p)
        assert mu == pytest.approx(1.3, abs=1e-6)
        assert sigma == pytest.approx(0.7, abs=1e-6)

    def test_the_scdm_subspace_is_orthonormal(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(4, 9, 3)) + 1j * rng.normal(size=(4, 9, 3))
        energies = np.sort(rng.normal(size=(4, 9)), axis=1)
        S = scdm_subspace(energies, A, 0.0, 0.5)
        for s in S:
            assert np.allclose(s.conj().T @ s, np.eye(3), atol=1e-12)


@pytest.fixture(scope="module")
def silicon():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.35, trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZP"},
                           kpts={"size": (3, 3, 3), "gamma": True},
                           smearing={"method": "fermi-dirac",
                                     "width": 0.001})
    atoms.get_potential_energy()
    return atoms


class TestProjectability:
    def test_it_is_a_weight_that_counts_the_targets(self, silicon):
        """The targets are basis orbitals: over every band of the basis
        the projectabilities add up to their number, at each k-point."""
        p = silicon.calc.projectabilities("sp3")
        assert p.values.shape == (27, 18)
        assert p.values.min() >= 0.0 and p.values.max() <= 1.0
        assert np.allclose(p.values.sum(axis=1), 8.0, atol=1e-9)
        # The valence is s and p.
        assert p.values[:, :4].min() > 0.95
        assert "Si0:sp3-1" in p.labels

    def test_only_the_span_of_the_targets_matters(self, silicon):
        """Four sp3 hybrids on an atom span its s and p orbitals."""
        hybrids = silicon.calc.projectabilities("sp3", kpts=(2, 2, 2))
        shells = silicon.calc.projectabilities([("Si", ["s", "p"])],
                                               kpts=(2, 2, 2))
        assert np.allclose(hybrids.values, shells.values, atol=1e-10)

    def test_explicit_points_match_the_mesh(self, silicon):
        mesh = silicon.calc.projectabilities({"Si": "p"}, kpts=(2, 2, 2))
        points = silicon.calc.projectabilities({"Si": "p"},
                                               kpts=mesh.kpoints[:3])
        assert np.allclose(points.values, mesh.values[:3], atol=1e-10)
        assert np.allclose(points.energies, mesh.energies[:3], atol=1e-8)

    def test_targets_must_sit_on_atoms(self, silicon):
        with pytest.raises(ValueError, match="atom"):
            silicon.calc.projectabilities("bonds")


@pytest.fixture(scope="module")
def automatic(silicon):
    return silicon.calc.wannier(guess="sp3", windows="auto")


class TestAutomaticWindows:
    def test_the_targets_set_the_number_of_functions(self, automatic):
        assert automatic.n_functions == 8
        assert automatic.windows["selection"] == "projectability"

    def test_the_frozen_window_is_exact_on_the_mesh(self, silicon,
                                                    automatic):
        """The window holds the valence, and every state in it is an
        eigenvalue of the Wannier Hamiltonian at its mesh point."""
        lo, hi = automatic.windows["frozen"]
        solver = silicon.calc.solver._periodic_solver
        B = rc.reciprocal_vectors(solver.crystal.lattice)
        direct, _ = solver.bands(automatic.kpoints @ B.T)
        direct = np.asarray(direct) * HARTREE_TO_EV
        inside = (direct >= lo) & (direct <= hi)
        assert inside[:, :4].all()
        bands = automatic.interpolate(automatic.kpoints)
        worst = max(np.abs(bands[k] - e).min()
                    for k in range(len(direct)) for e in direct[k][inside[k]])
        assert worst < 1e-8
        # Every state in the window projects at least the threshold.
        p = silicon.calc.projectabilities("sp3")
        assert p.values[inside].min() >= DEFAULT_FROZEN_PROJECTABILITY

    def test_the_selection_is_cited_and_reported(self, silicon, automatic):
        keys = silicon.calc.citation_keys()
        assert {"Qiao2023", "Sayfutyarova2017", "Souza2001"} <= set(keys)
        assert "projectability" in automatic.summary()

    def test_scdm_needs_no_windows(self, silicon):
        w = silicon.calc.wannier(guess="sp3", windows="scdm")
        assert w.n_functions == 8 and w.windows["fitted"]
        assert w.windows["mu"] > silicon.calc.get_fermi_level()
        assert np.isfinite(w.omega_invariant)
        assert {"Damle2017", "Damle2018", "Vitale2020"} <= set(
            silicon.calc.citation_keys())
