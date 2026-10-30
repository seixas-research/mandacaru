# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_partial_waves.py

# This code is part of Mandacaru.
# MIT License

"""The radial machinery of PAW-LCAO generation
(:mod:`mandacaru.pseudopotentials.partial_waves`): Bessel matching, the
constrained minimization of the smooth partial waves, Numerov bound states,
and the ghost search on a stub generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn

from mandacaru import _native

from mandacaru.basis.atomic_solver import solve_atom
from mandacaru.pseudopotentials import partial_waves as pw


class TestConstruction:
    def test_bessel_wavevectors_interleave(self):
        qs = pw.bessel_wavevectors(0, 1.3, 8)
        x = qs * 1.3
        assert np.all(np.diff(x) > 0)
        assert x[0] == pytest.approx(np.pi, abs=1e-10)         # j_0 zero
        assert x[1] == pytest.approx(4.493409458, abs=1e-8)    # j_0' zero
        j, dj, _d2, _d3 = pw.bessel_derivatives(0, x)
        assert np.abs(j[::2]).max() < 1e-12 and np.abs(dj[1::2]).max() < 1e-12

    def test_bessel_derivatives_against_finite_differences(self):
        x = np.linspace(0.5, 12.0, 40)
        for l in (0, 1, 2):
            j, dj, d2j, d3j = pw.bessel_derivatives(l, x)
            h = 1e-3
            fd1 = (spherical_jn(l, x + h) - spherical_jn(l, x - h)) / (2 * h)
            fd2 = (spherical_jn(l, x + h) - 2 * j + spherical_jn(l, x - h)) / h ** 2
            assert np.abs(fd1 - dj).max() < 1e-6
            assert np.abs(fd2 - d2j).max() < 1e-5
            fd3 = (pw.bessel_derivatives(l, x + h)[2]
                   - pw.bessel_derivatives(l, x - h)[2]) / (2 * h)
            assert np.abs(fd3 - d3j).max() < 1e-5

    def test_constrained_minimum_satisfies_its_constraints(self):
        rng = np.random.default_rng(1)
        n = 8
        M = rng.normal(size=(n, n))
        K = M @ M.T
        G = np.eye(n) + 0.1 * (M + M.T)
        G = G @ G.T
        A = rng.normal(size=(3, n))
        b = rng.normal(size=3)
        k = rng.normal(size=n)
        c = pw.constrained_minimum(K, k, A, b, G, 5.0)
        assert np.abs(A @ c - b).max() < 1e-10
        assert c @ G @ c == pytest.approx(5.0, abs=1e-9)
        # KKT stationarity in the null space of A: Z^T (K c + k) = lam Z^T G c.
        _u, _s, vt = np.linalg.svd(A)
        Z = vt[3:].T
        g_obj, g_con = Z.T @ (K @ c + k), Z.T @ (G @ c)
        lam = (g_obj @ g_con) / (g_con @ g_con)
        assert np.abs(g_obj - lam * g_con).max() < 1e-8 * max(1.0, abs(lam))
        # Global: no feasible point (found by moving in the null space and
        # solving the norm condition for the step length) is lower.
        f = lambda v: v @ K @ v + 2 * k @ v
        for _ in range(50):
            y = Z @ rng.normal(size=n - 3)
            # (c + t y)^T G (c + t y) = 5  ->  quadratic in t
            qa, qb = y @ G @ y, 2 * c @ G @ y
            t = -qb / qa                        # the other intersection
            trial = c + t * y
            assert trial @ G @ trial == pytest.approx(5.0, abs=1e-8)
            assert f(trial) >= f(c) - 1e-9

    def test_constrained_minimum_in_the_hard_case(self):
        """The linear term has no component along the lowest eigenvector, so
        the norm stays finite at the pole: the minimizer takes the multiplier
        at the pole and meets the norm along that eigenvector."""
        K = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        k = np.array([0.0, 0.5, -0.3, 0.2, 0.0])
        A, b = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]), np.array([0.0])
        c = pw.constrained_minimum(K, k, A, b, np.eye(5), 4.0)
        others = -k[1:4] / (np.diag(K)[1:4] - 1.0)
        assert np.allclose(c[1:4], others, atol=1e-12)
        assert c[4] == pytest.approx(0.0, abs=1e-12)
        assert abs(c[0]) == pytest.approx(np.sqrt(4.0 - others @ others),
                                          rel=1e-12)

    def test_numerov_bound_state_matches_the_atom(self):
        atom = solve_atom(3, points=pw.generation_points(3), r_max=30.0,
                          tolerance=1e-7, mixing=0.25, xc="lda",
                          relativity="scalar")
        u, energy = pw.bound_state(atom.r, atom.v_effective, 0,
                                   atom.eigenvalues[(2, 0)], 3.0)
        assert abs(energy - atom.eigenvalues[(2, 0)]) < 5e-4      # FD vs Numerov
        assert np.trapezoid(u * u, atom.r) == pytest.approx(1.0, abs=1e-9)
        assert np.sum(np.diff(np.sign(u[(atom.r > 0.02) & (atom.r < 10)])) != 0) == 1

    def test_a_gga_local_potential_is_matched_over_many_nodes(self):
        """A PBE potential is a cubic spline through its derivative nodes,
        0.01 r apart.  Its fourth derivative read over the default +-25 grid
        points (two or three nodes) put oxygen's local potential at -13.6 Ha
        at the origin against LDA's -5.8; read over +-0.15 r_cl
        (GGA_LOCAL_FIT_FRACTION) it is LDA's to 0.1 Ha, and generate_paw
        uses that window for a gradient-corrected dataset (HISTORY.md,
        2026-10-07, "The PBE set")."""
        from mandacaru.pseudopotentials.paw import generate_paw

        r_cl = 1.305                           # oxygen's local radius
        depth = {}
        for xc in ("lda", "pbe"):
            atom = solve_atom(8, points=pw.generation_points(8), r_max=30.0,
                              tolerance=1e-7, mixing=0.25, xc=xc,
                              relativity="scalar")
            window = pw.GGA_LOCAL_FIT_FRACTION * r_cl
            depth[xc] = (
                pw.polynomial_local_potential(atom.r, atom.v_effective,
                                              r_cl)[0],
                pw.polynomial_local_potential(atom.r, atom.v_effective, r_cl,
                                              fit_window=window)[0])
        lda, lda_wide = depth["lda"]
        assert lda_wide == pytest.approx(lda, abs=0.05)
        assert depth["pbe"][0] < lda - 5.0               # the noise
        assert depth["pbe"][1] == pytest.approx(lda, abs=0.1)
        pp = generate_paw("O", xc="pbe", atom=atom, ghosts="keep")
        assert pp.r_cut_local == pytest.approx(r_cl, abs=0.01)
        assert pp.v_local_screened[0] - pp.local_shift == pytest.approx(
            depth["pbe"][1], abs=0.05)

    def test_generation_grid_scales_with_z(self):
        assert pw.generation_points(1) == 6000
        assert pw.generation_points(8) == 12000


class TestGhostSearch:
    """:func:`~mandacaru.pseudopotentials.pw.ghost_free` on a stub generator:
    the decisions, without paying for a generation."""

    class _Channel:
        def __init__(self, r_cut, reference):
            self.r_cut = r_cut
            self.reference_energies = [reference, reference + 1.0]

    class _Dataset:
        def __init__(self, levels, r_cuts=(3.0, 1.0)):
            self.channels = {l: TestGhostSearch._Channel(rc, -0.2)
                             for l, rc in zip((0, 2), r_cuts)}
            self.levels = levels            # {l: (lowest, second)}
            self.atom = "atom"

    CLEAN = {0: (-0.2, 0.3), 2: (-0.2, 0.3)}

    GHOST = {0: (-1.8, -0.2), 2: (-0.2, 0.3)}

    @staticmethod
    def _levels(pp, l):
        return pp.levels[l]

    @pytest.fixture(autouse=True)
    def _scattering(self, monkeypatch):
        """Phase errors by dataset, default clean -- of the constructed
        channels and of the ones without projectors."""
        from mandacaru.pseudopotentials import partial_waves as pw
        self.phases, self.unprojected = {}, {}
        monkeypatch.setattr(pw, "scattering_errors",
                            lambda pp, _ld, _cache=None:
                            self.phases.get(id(pp), {}))
        monkeypatch.setattr(pw, "unprojected_scattering_errors",
                            lambda pp, _ld, _cache=None:
                            self.unprojected.get(id(pp), {}))

    def _options(self, **overrides):
        options = {"r_cut": None, "r_cut_local": None, "local_shift": None,
                   "local_factor": 0.9, "atom": None}
        options.update(overrides)
        return options

    def _generator(self, outcomes):
        """Returns datasets from ``outcomes`` in order, recording each call."""
        calls = []

        def generate(symbol, *, ghosts, **options):
            assert ghosts == "keep"
            calls.append(options)
            return outcomes[len(calls) - 1]
        return generate, calls

    def _run(self, generate, mode="repair", overrides=None, acceptance=None,
             **options):
        from mandacaru.pseudopotentials.partial_waves import ghost_free
        return ghost_free(generate, self._levels, None, "X",
                          self._options(**options), mode, overrides,
                          acceptance=acceptance)

    @staticmethod
    def _s_miss(pp):
        """A stub acceptance: the dataset's own ``miss``, failing at 1."""
        miss = getattr(pp, "miss", 0.0)
        return {"s_miss": miss} if miss >= 1.0 else {}

    def _missing(self, miss):
        dataset = self._Dataset(self.CLEAN)
        dataset.miss = miss
        return dataset

    def test_a_failed_acceptance_is_repaired_with_more_bessel_functions(self):
        """Mg-LDA: clean spectrum and phases, but its s channel missed an
        intruding hydrogen 1s by 1.5; ten Bessel functions bring it to 0.07.
        The Bessel attempts come after the shorter local radii and the
        own-cutoff shifts."""
        from mandacaru.pseudopotentials.partial_waves import (
            OWN_CUTOFF_SHIFTS, SHORTER_LOCAL_FACTORS)

        first = self._missing(1.5)
        shifts = [self._missing(1.6) for _ in range(
            len(SHORTER_LOCAL_FACTORS) + len(OWN_CUTOFF_SHIFTS))]
        fixed = self._missing(0.07)
        generate, calls = self._generator([first] + shifts + [fixed])
        assert self._run(generate, acceptance=self._s_miss,
                         n_bessel=None) is fixed
        assert calls[-1]["n_bessel"] == 9
        assert calls[-1]["local_shift"] == 0.0

    def test_a_ghost_repair_must_also_pass_the_acceptance(self):
        """A shift that removes the ghost but leaves the s channel
        incomplete is not a repair."""
        first = self._Dataset(self.GHOST)
        incomplete = self._missing(2.4)
        complete = self._missing(0.3)
        generate, _calls = self._generator([first, incomplete, complete])
        assert self._run(generate, acceptance=self._s_miss,
                         n_bessel=None) is complete

    def test_flag_mode_keeps_the_smallest_miss(self):
        from mandacaru.pseudopotentials.partial_waves import (
            OWN_CUTOFF_SHIFTS, SHORTER_LOCAL_FACTORS)

        attempts = (len(SHORTER_LOCAL_FACTORS) + len(OWN_CUTOFF_SHIFTS)
                    + 2 * (len(OWN_CUTOFF_SHIFTS) + 1) + 3 * 3 + 5)
        outcomes = [self._missing(3.0)] + [self._missing(2.0 + 0.01 * i)
                                           for i in range(attempts)]
        generate, _calls = self._generator(outcomes)
        result = self._run(generate, mode="flag", acceptance=self._s_miss,
                           n_bessel=None)
        assert result is outcomes[1]
        assert result.defects["s_miss"] == pytest.approx(2.0)

    def test_a_huge_miss_ranks_below_a_small_phase_error(self):
        """Gd-LDA: flagged either way, the search kept a miss of 23.8 over a
        0.09 rad phase error with a miss of 1.4.  Each defect now counts by
        how far it exceeds its own tolerance."""
        from mandacaru.pseudopotentials.partial_waves import _defect_badness

        huge_miss = (None, {}, {}, {"s_miss": 23.8})
        small_phase = (None, {}, {3: (0.09, 0.09)}, {"s_miss": 1.4})
        ghost = (None, {0: -0.5}, {}, {})
        assert _defect_badness(small_phase) < _defect_badness(huge_miss)
        assert _defect_badness(huge_miss) < _defect_badness(ghost)

    def test_a_pinned_construction_is_kept_with_its_miss_recorded(self):
        """A caller who fixed the cutoffs chose what a repair would change;
        an incomplete s channel alone is recorded, not refused."""
        first = self._missing(1.4)
        generate, calls = self._generator([first])
        result = self._run(generate, acceptance=self._s_miss,
                           r_cut={0: 3.0, 2: 1.0})
        assert result is first and len(calls) == 1
        assert result.defects["s_miss"] == pytest.approx(1.4)

    def test_the_miss_survives_the_file_record(self):
        from mandacaru.pseudopotentials.partial_waves import (defect_message,
                                                     defects_record,
                                                     read_defects)

        defects = {"ghosts": {}, "phases": {}, "s_miss": 1.8}
        assert read_defects(defects_record(defects))["s_miss"] == 1.8
        assert "incomplete s channel" in defect_message("X", "paw-lcao",
                                                        defects)

    def test_a_clean_dataset_is_returned_unchanged(self):
        clean = self._Dataset(self.CLEAN)
        generate, calls = self._generator([clean])
        assert self._run(generate) is clean
        assert len(calls) == 1

    def test_a_clean_dataset_that_scatters_wrong_is_repaired(self):
        """Aluminum's first p channel: no ghost, 0.95 rad off.  Absence of a
        ghost alone no longer returns the first construction."""
        wrong = self._Dataset(self.CLEAN)
        right = self._Dataset(self.CLEAN)
        self.phases[id(wrong)] = {1: (0.949, 0.949)}
        generate, calls = self._generator([wrong, right])
        assert self._run(generate, overrides={"norm_deficit": 0.0}) is right
        assert calls[1]["norm_deficit"] == 0.0

    def test_flag_keeps_the_least_defective_attempt_and_records_it(self):
        """No remedy works: ``flag`` returns the attempt with the shallowest
        ghost (a scattering-only defect would beat any ghost) and records it."""
        from mandacaru.pseudopotentials.partial_waves import (
            GHOST_REMEDY_SHIFTS, OWN_CUTOFF_SHIFTS, SHORTER_LOCAL_FACTORS)

        deep = self._Dataset({0: (-5.0, -0.2), 2: (-0.2, 0.3)})
        shallow = self._Dataset({0: (-0.5, -0.2), 2: (-0.2, 0.3)})
        rest = [self._Dataset(self.GHOST) for _ in range(
            len(SHORTER_LOCAL_FACTORS) + len(OWN_CUTOFF_SHIFTS)
            + len(GHOST_REMEDY_SHIFTS))]
        generate, _calls = self._generator([deep, shallow] + rest)
        kept = self._run(generate, mode="flag",
                         overrides={"norm_deficit": 0.0})
        assert kept is shallow
        assert kept.defects == {"ghosts": {0: pytest.approx(-0.3)},
                                "phases": {}}

    def test_an_inaccurate_level_is_not_a_ghost(self):
        """One level 6e-4 Ha low with nothing displaced: accuracy, not a ghost."""
        from mandacaru.pseudopotentials.partial_waves import ghost_errors
        pp = self._Dataset({0: (-0.2006, 0.3), 2: (-0.2, 0.3)})
        assert ghost_errors(pp, self._levels) == {}
        assert ghost_errors(self._Dataset(self.GHOST), self._levels) == {
            0: pytest.approx(-1.6)}

    def test_overrides_are_tried_alone_first_then_kept_in_every_attempt(self):
        from mandacaru.pseudopotentials.partial_waves import (
            OWN_CUTOFF_SHIFTS, SHORTER_LOCAL_FACTORS)
        n_own, n_short = len(OWN_CUTOFF_SHIFTS), len(SHORTER_LOCAL_FACTORS)
        stills = [self._Dataset(self.GHOST) for _ in range(1 + n_short + n_own)]
        clean = self._Dataset(self.CLEAN)
        generate, calls = self._generator([self._Dataset(self.GHOST)]
                                          + stills + [clean])
        assert self._run(generate, overrides={"norm_deficit": 0.0}) is clean
        alone, shorter = calls[1], calls[2]
        raised, balanced = calls[2 + n_short], calls[2 + n_short + n_own]
        assert alone["norm_deficit"] == 0.0
        assert alone["r_cut"] is None           # the cutoffs are untouched
        assert alone["atom"] == "atom"          # the SCF atom is reused
        assert shorter["norm_deficit"] == 0.0   # a shorter radius, no raise
        assert shorter["local_shift"] == 0.0
        assert raised["r_cut"] is None          # raised at the own cutoffs
        assert raised["local_shift"] == OWN_CUTOFF_SHIFTS[0]
        assert balanced["norm_deficit"] == 0.0  # and kept with balanced ones
        assert balanced["r_cut"] == {0: 3.0, 2: 3.0}

    def test_a_ghost_is_repaired_with_balanced_cutoffs_and_a_raised_shift(self):
        from mandacaru.pseudopotentials.partial_waves import (
            GHOST_REMEDY_SHIFTS, OWN_CUTOFF_SHIFTS, SHORTER_LOCAL_FACTORS)
        n_own, n_short = len(OWN_CUTOFF_SHIFTS), len(SHORTER_LOCAL_FACTORS)
        own = [self._Dataset(self.GHOST) for _ in range(n_short + n_own)]
        still = self._Dataset({0: (-0.5, -0.2), 2: (-0.2, 0.3)})
        clean = self._Dataset(self.CLEAN)
        generate, calls = self._generator([self._Dataset(self.GHOST)] + own
                                          + [still, clean])
        assert self._run(generate) is clean
        assert [c["local_factor"] for c in calls[1:1 + n_short]] == \
            pytest.approx([0.9 * f for f in SHORTER_LOCAL_FACTORS])
        assert [c["local_shift"] for c in calls[1 + n_short:1 + n_short + n_own]] \
            == list(OWN_CUTOFF_SHIFTS)
        first, second = (calls[1 + n_short + n_own],
                         calls[2 + n_short + n_own])
        assert first["r_cut"] == {0: 3.0, 2: 3.0}
        assert first["r_cut_local"] == pytest.approx(2.7)
        assert first["local_shift"] == GHOST_REMEDY_SHIFTS[0]
        assert second["local_shift"] == GHOST_REMEDY_SHIFTS[1]

    def test_a_repair_that_breaks_the_scattering_is_rejected(self):
        """Iron at a 10 Ha raise: ghost-free and 0.74 rad wrong at +0.25 Ha."""
        wrong = self._Dataset(self.CLEAN)
        right = self._Dataset(self.CLEAN)
        self.phases[id(wrong)] = {0: (0.74, 0.74), 2: (0.005, 0.005)}
        self.phases[id(right)] = {0: (0.004, 0.025), 2: (0.007, 0.007)}
        generate, _calls = self._generator([self._Dataset(self.GHOST), wrong,
                                            right])
        assert self._run(generate) is right

    def test_a_resonance_just_outside_the_window_is_rejected(self):
        """Gallium at a 20 Ha raise: 0.027 rad near, 0.94 rad at +0.55 Ha."""
        resonant = self._Dataset(self.CLEAN)
        right = self._Dataset(self.CLEAN)
        self.phases[id(resonant)] = {0: (0.027, 0.936)}
        self.phases[id(right)] = {0: (0.001, 0.010), 2: (0.003, 0.062)}
        generate, _calls = self._generator([self._Dataset(self.GHOST),
                                            resonant, right])
        assert self._run(generate) is right

    def test_a_ghost_with_a_pinned_cutoff_is_refused_not_overridden(self):
        from mandacaru.pseudopotentials.partial_waves import GhostStateError
        generate, calls = self._generator([self._Dataset(self.GHOST)])
        with pytest.raises(GhostStateError, match="r_cut fixed by the caller"):
            self._run(generate, r_cut=2.0)
        assert len(calls) == 1

    def test_refuse_mode_refuses(self):
        from mandacaru.pseudopotentials.partial_waves import GhostStateError
        generate, _calls = self._generator([self._Dataset(self.GHOST)])
        with pytest.raises(GhostStateError, match="l=0"):
            self._run(generate, mode="refuse")

    def test_keep_mode_returns_the_ghost(self):
        ghosted = self._Dataset(self.GHOST)
        generate, _calls = self._generator([ghosted])
        assert self._run(generate, mode="keep") is ghosted

    def test_no_remedy_is_an_error_naming_every_attempt(self):
        from mandacaru.pseudopotentials.partial_waves import (
            GHOST_REMEDY_SHIFTS, OWN_CUTOFF_SHIFTS, SHORTER_LOCAL_FACTORS,
            GhostStateError)
        n = (1 + len(SHORTER_LOCAL_FACTORS) + len(OWN_CUTOFF_SHIFTS)
             + len(GHOST_REMEDY_SHIFTS))
        generate, calls = self._generator([self._Dataset(self.GHOST)] * n)
        with pytest.raises(GhostStateError, match="no remedy removed it") as error:
            self._run(generate)
        assert len(calls) == n
        assert "own cutoffs, shift 5" in str(error.value)
        assert "local radius x 0.6, no shift" in str(error.value)

    def test_a_wrongly_scattering_local_channel_is_repaired_without_a_raise(self):
        """K21: aluminum's first repair (5 Ha) left its d channel, which has
        no projectors, 0.52 rad off.  The check now sees it, and a shorter
        local radius at no raise is tried before any raise."""
        wrong = self._Dataset(self.CLEAN)
        right = self._Dataset(self.CLEAN)
        self.unprojected[id(wrong)] = {1: 0.52}
        self.unprojected[id(right)] = {1: 0.12}
        generate, calls = self._generator([wrong, right])
        assert self._run(generate) is right
        assert calls[1]["local_shift"] == 0.0
        assert calls[1]["local_factor"] == pytest.approx(0.9 * 0.75)

    def test_a_failed_miss_ranks_below_any_scattering_defect(self):
        """U, Ta, Re and Hf scatter wrongly in every construction; the
        search must keep one with a complete s channel over one that misses
        an intruding 1s, however large the scattering error."""
        from mandacaru.pseudopotentials.partial_waves import _defect_badness

        scattering = (None, {}, {1: (0.5, 0.5)},
                      {"unprojected": {2: 1.42}})
        missing = (None, {}, {}, {"s_miss": 1.05})
        ghost = (None, {0: -0.5}, {}, {})
        assert _defect_badness(scattering) < _defect_badness(missing) < \
            _defect_badness(ghost)

    def test_a_flagged_local_channel_survives_the_file_record(self):
        from mandacaru.pseudopotentials.partial_waves import (
            _defect_badness, defect_message, defects_record, read_defects)

        defects = {"ghosts": {}, "phases": {}, "unprojected": {1: 0.37}}
        assert read_defects(defects_record(defects))["unprojected"] == {1: 0.37}
        assert "local potential that scatters wrongly" in defect_message(
            "Na", "paw-lcao", defects)
        # 0.37 rad is 1.85x its tolerance: worse than a 0.08 rad phase
        # error of a constructed channel (1.6x), better than any ghost.
        loose = (None, {}, {}, {"unprojected": {1: 0.37}})
        phase = (None, {}, {0: (0.08, 0.08)}, {})
        ghost = (None, {0: -0.5}, {}, {})
        assert _defect_badness(phase) < _defect_badness(loose) < \
            _defect_badness(ghost)

    def test_a_level_between_two_bound_references_is_a_ghost(self):
        """A semicore channel (sodium's 2s and 3s) must have its 3s as its
        second level: an extra state between the two is a ghost."""
        from mandacaru.pseudopotentials.partial_waves import ghost_errors
        pp = self._Dataset({0: (-2.0, -0.5, -0.1), 2: (-0.2, 0.3)})
        pp.channels[0].reference_energies = [-2.0, -0.1]
        pp.channels[0].occupations = [2.0, 1.0]
        assert ghost_errors(pp, self._levels) == {0: pytest.approx(-0.4)}
        pp.levels[0] = (-2.0, -0.1, 0.4)
        assert ghost_errors(pp, self._levels) == {}

    def test_an_extra_bound_level_above_the_reference_is_a_ghost(self):
        """Tungsten's 4f channel: reference -1.236 Ha, its second reference
        -0.236 Ha still bound, and a level at -0.148 Ha that the atom does
        not have (one f state below -0.1 Ha, the 4f).  Outside every
        scattering window and above the reference, it was invisible."""
        from mandacaru.pseudopotentials.partial_waves import ghost_errors
        pp = self._Dataset({0: (-0.2, 0.3), 2: (-1.236, -0.148)})
        pp.channels[2].reference_energies = [-1.236, -0.236]
        pp.all_electron_count = lambda l, energy: {0: 1, 2: 1}[l]
        assert ghost_errors(pp, self._levels) == {2: pytest.approx(-0.148)}
        assert ghost_errors(pp, self._levels, with_atom=False) == {}
        pp.all_electron_count = lambda l, energy: {0: 1, 2: 2}[l]
        assert ghost_errors(pp, self._levels) == {}

    def test_a_scattering_channel_is_not_judged(self):
        from mandacaru.pseudopotentials.partial_waves import ghost_errors
        pp = self._Dataset({0: (-0.2, 0.3), 2: (-5.0, 0.3)})
        pp.channels[2].reference_energies = [0.3, 1.3]
        assert ghost_errors(pp, self._levels) == {}

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="ghosts must be one of"):
            self._run(None, mode="ignore")

    # A check that cannot be evaluated (a NaN level) fails a construction
    # like a defect, but it is never returned, not even flagged.
    NAN = {0: (float("nan"), float("nan")), 2: (-0.2, 0.3)}

    def test_an_unevaluated_first_construction_is_repaired(self):
        right = self._Dataset(self.CLEAN)
        generate, calls = self._generator([self._Dataset(self.NAN), right])
        assert self._run(generate, overrides={"norm_deficit": 0.0}) is right
        assert len(calls) == 2

    def test_an_unevaluated_construction_is_never_flagged(self):
        from mandacaru.pseudopotentials.partial_waves import GhostStateError
        outcomes = [self._Dataset(self.NAN) for _ in range(40)]
        generate, _calls = self._generator(outcomes)
        with pytest.raises(GhostStateError, match="could not be evaluated"):
            self._run(generate, mode="flag")

    def test_refuse_names_the_unevaluated_check(self):
        from mandacaru.pseudopotentials.partial_waves import GhostStateError
        generate, _calls = self._generator([self._Dataset(self.NAN)])
        with pytest.raises(GhostStateError, match="l=0 level is nan"):
            self._run(generate, mode="refuse")


class TestUnevaluatedChecks:
    """A check that cannot be evaluated is a failure, never a pass: NaN
    levels and phases used to leave ``max(0.0, nan) == 0.0`` and every
    comparison false, and a dataset without its all-electron atom returned
    ``{}`` -- both read as "clean" (audit of 2026-10-09; HISTORY.md,
    2026-10-11)."""

    class _Channel:
        r_cut = 1.0
        occupations = [1.0, 0.0]
        reference_energies = [-0.5, 0.5]

    class _Hydrogen:
        """A one-channel dataset with a bare Coulomb atom attached."""
        symbol, atomic_number, relativity = "H", 1, "none"
        r_cut_local = 1.0
        frozen_subshells = semicore_subshells = ()

        def __init__(self):
            self.r = np.arange(1, 3001) * 0.01
            self.channels = {0: TestUnevaluatedChecks._Channel()}
            self.v_local_screened = -1.0 / np.maximum(self.r, 0.5)
            self.atom = type("Atom", (), {"r": self.r,
                                          "v_effective": -1.0 / self.r,
                                          "occupations": {(1, 0): 1.0}})()

        def projector_radius(self, l):
            return 1.0

    @staticmethod
    def _nan(*_args):
        return float("nan")

    def test_a_nan_level_fails_the_ghost_check(self):
        pp = self._Hydrogen()
        with pytest.raises(pw.DiagnosticError, match=r"H: an l=0 level is nan"):
            pw.ghost_errors(pp, lambda _pp, _l: [float("nan")] * 2,
                            with_atom=False)

    def test_a_nan_phase_fails_the_scattering_checks(self):
        pp = self._Hydrogen()
        with pytest.raises(pw.DiagnosticError,
                           match=r"H: the l=0 phase error at -1\.500 Ha is nan"):
            pw.scattering_errors(pp, self._nan)
        with pytest.raises(pw.DiagnosticError,
                           match=r"H: the l=1 \(no projectors\) phase error "
                                 r"at -1\.000 Ha is nan"):
            pw.unprojected_scattering_errors(pp, self._nan)

    def test_a_channel_anchored_at_a_scattering_energy_is_checked(self):
        """Potassium's empty d is anchored at 0 Ha; without a frozen
        subshell such a channel used to be skipped.  Its phase is compared
        at the positive energies of the window."""
        pp = self._Hydrogen()
        channel = self._Channel()
        channel.reference_energies = [0.0, 1.0]
        channel.occupations = [0.0, 0.0]
        pp.channels[2] = channel
        with pytest.raises(pw.DiagnosticError,
                           match=r"l=2 phase error at \+0\.0\d0 Ha"):
            pw.scattering_errors(pp, lambda p, l, e, r: (
                float("nan") if l == 2 else pw.log_derivative_ae(
                    p.r, p.atom.v_effective, l, e, r, 1.0)))

    def test_a_nan_potential_fails_the_local_ghost_check(self):
        pp = self._Hydrogen()
        pp.v_local_screened = pp.v_local_screened.copy()
        pp.v_local_screened[10] = np.nan
        with pytest.raises(pw.DiagnosticError,
                           match="the screened local potential is nan"):
            pw.local_potential_ghosts(pp)

    def test_without_the_atom_a_check_is_not_evaluated(self):
        """A dataset read from a file has no atom: the checks against it
        raise rather than return ``{}``; the spectral check of the projected
        channels alone is asked for explicitly."""
        pp = self._Hydrogen()
        pp.atom = None
        for check in (lambda: pw.local_potential_ghosts(pp),
                      lambda: pw.unprojected_scattering_errors(pp, self._nan),
                      lambda: pw.scattering_errors(pp, self._nan)):
            with pytest.raises(pw.DiagnosticError, match="not evaluated"):
                check()
        pp.unconstructed_ghosts = lambda: pw.local_potential_ghosts(pp)
        levels = lambda _pp, _l: [-0.5, 0.3]                      # noqa: E731
        with pytest.raises(pw.DiagnosticError, match="not evaluated"):
            pw.ghost_errors(pp, levels)
        assert pw.ghost_errors(pp, levels, with_atom=False) == {}

    def test_the_atom_may_be_on_a_finer_grid(self):
        """A loaded dataset keeps every fourth point of the generation grid;
        the all-electron side of a check runs on the atom's own grid, at the
        dataset's node."""
        pp = self._Hydrogen()
        exact = pw.unprojected_scattering_errors(
            pp, lambda p, l, e, r: pw.log_derivative_ae(
                p.r, p.atom.v_effective, l, e, r, 1.0))
        fine = np.arange(1, 12001) * 0.0025
        pp.atom = type("Atom", (), {"r": fine, "v_effective": -1.0 / fine})()
        coarse = pw.unprojected_scattering_errors(
            pp, lambda p, l, e, r: pw.log_derivative_ae(
                p.r, -1.0 / p.r, l, e, r, 1.0))
        assert exact[1] < 1e-12 and coarse[1] < 1e-3


REFERENCE = (Path(__file__).resolve().parents[1] / "migration" / "reference"
             / "paw")


def _scaled(value, reference):
    return (np.abs(np.asarray(value) - reference).max()
            / max(np.abs(reference).max(), 1e-300))


@pytest.fixture(scope="module")
def frozen():
    """Outputs of the Python implementation, frozen before the port to Rust
    (the native non-relativistic LDA atom at 6000 points for H, C, O, Si,
    Cu; ``partial_waves.json`` records the inputs)."""
    return np.load(REFERENCE / "partial_waves.npz"), json.loads(
        (REFERENCE / "partial_waves.json").read_text())


class TestFrozenPythonOutputs:
    """The Rust routines reproduce the numbers the Python ones gave."""

    def test_the_inner_grid_has_the_documented_size(self):
        assert len(pw._inner_grid(1.7)) == pw.INNER_POINTS

    def test_the_special_functions(self, frozen):
        p, _ = frozen
        x = p["bessel_x"]
        for l in range(5):
            for k, row in enumerate(pw.bessel_derivatives(l, x)):
                assert _scaled(row, p[f"bessel_d{k}_{l}"]) < 1e-10
        for l in range(4):
            for rc, n in ((1.5, 8), (2.2, 10), (0.8, 6)):
                assert _scaled(pw.bessel_wavevectors(l, rc, n),
                               p[f"wavevectors_{l}_{rc}_{n}"]) < 1e-13
            table = pw._bessel_table(l, p["bessel_table_q"],
                                     p["bessel_table_r"])
            assert _scaled(table, p[f"bessel_table_{l}"]) < 1e-13

    @pytest.mark.parametrize("k", range(4))
    def test_the_constrained_minimum(self, frozen, k):
        p, _ = frozen
        c = pw.constrained_minimum(p[f"cm_K_{k}"], p[f"cm_k_{k}"],
                                   p[f"cm_A_{k}"], p[f"cm_b_{k}"],
                                   p[f"cm_G_{k}"], float(p[f"cm_norm_{k}"][0]))
        assert _scaled(c, p[f"cm_c_{k}"]) < 1e-12

    @pytest.mark.parametrize("treatment", ["none", "scalar"])
    @pytest.mark.parametrize("symbol", ["H", "C", "O", "Si", "Cu"])
    def test_the_waves_of_an_atom(self, frozen, symbol, treatment):
        p, meta = frozen
        z = float(meta["elements"][symbol])
        v = p[f"{symbol}_v"]
        r = p[f"{symbol}_h"][0] * np.arange(1, v.size + 1)
        t, stride = f"{symbol}_{treatment}", meta["stride"]
        r_c = float(r[int(np.argmin(np.abs(r - 1.5)))])
        for l in range(3):
            for energy in (-0.5, 0.2):
                u = pw.scattering_wave(r, v, l, energy, z, treatment)
                assert _scaled(u[::stride], p[f"{t}_scat_{l}_{energy}"]) < 1e-11
                assert pw.log_derivative_ae(r, v, l, energy, r_c, z,
                                            treatment) == pytest.approx(
                    p[f"{t}_logd_{l}_{energy}"][0], rel=1e-11)
                assert _scaled(pw.matching_targets(r, u, v, l, energy, r_c,
                                                   treatment, -1, z),
                               p[f"{t}_match_{l}_{energy}"]) < 1e-11
        for key in (k for k in p.files if k.startswith(f"{t}_bound_")
                    and k.endswith("_u")):
            n, l = (int(part) for part in key.split("_")[-3:-1])
            guess = [row[2] for row in p[f"{symbol}_eig"]
                     if int(row[0]) == n and int(row[1]) == l][0]
            u, energy = pw.bound_state(r, v, l, guess, z, treatment=treatment)
            assert energy == pytest.approx(p[key[:-1] + "e"][0], rel=1e-12)
            assert _scaled(u[::stride], p[key]) < 1e-11

    @pytest.mark.parametrize("treatment", ["none", "scalar"])
    @pytest.mark.parametrize("symbol", ["H", "O", "Si", "Cu"])
    def test_the_optimized_pseudo_waves(self, frozen, symbol, treatment):
        """Eight Bessel functions reproduce the coefficients to 1e-9; ten
        (norm factor 0.9, q_cut 6) are ill-conditioned -- a 1e-13 change of
        the potential moves their coefficients by 1e-8 -- so there the smooth
        *waves* are compared (1e-8)."""
        p, meta = frozen
        z = float(meta["elements"][symbol])
        v = p[f"{symbol}_v"]
        r = p[f"{symbol}_h"][0] * np.arange(1, v.size + 1)
        t = f"{symbol}_{treatment}"
        for l in range(4):
            if f"{t}_refpair_{l}_e" not in p.files:
                continue
            states = p[f"{t}_ref_{l}_e"]
            energies = list(p[f"{t}_refpair_{l}_e"])
            r_c = float(p[f"{t}_cut_{l}"][0])
            if states[0][2] == 0.0:
                waves = [pw._scattering_reference(r, v, l, e, z, r_c,
                                                  treatment) for e in energies]
            else:
                waves = []
                for n, _energy, _occupation in states:
                    guess = [row[2] for row in p[f"{symbol}_eig"]
                             if int(row[0]) == int(n) and int(row[1]) == l][0]
                    u, _ = pw.bound_state(r, v, l, guess, z,
                                          treatment=treatment)
                    waves.append(u / r)
                if len(states) == 1:
                    waves.append(pw._scattering_reference(
                        r, v, l, energies[1], z, r_c, treatment))
            assert _scaled(pw.inner_overlaps(r, waves, r_c),
                           p[f"{t}_inner_{l}"]) < 1e-12
            assert _scaled(pw.norm_targets(r, waves, energies, r_c, v,
                                           treatment, -1, z),
                           p[f"{t}_normtargets_{l}"]) < 1e-12
            for n_bessel, q_cut, factor in ((8, 5.0, 1.0), (10, 6.0, 0.9)):
                key = f"{t}_opt_{l}_{n_bessel}"
                if key + "_coef" not in p.files:
                    continue
                out = pw.optimize_pseudo_waves(
                    r, v, l, waves, energies, r_c, q_cut, n_bessel, factor,
                    treatment, -1, z)
                assert _scaled(out.wavevectors[0], p[key + "_q"][0]) < 1e-12
                assert _scaled(out.norms, p[key + "_norms"]) < 1e-12
                assert _scaled(out.targets, p[key + "_targets"]) < 1e-12
                assert _scaled(out.achieved, p[key + "_achieved"]) < 1e-9
                basis = pw._bessel_table(l, out.wavevectors[0],
                                         pw._inner_grid(out.r_cut))
                smooth = np.array(out.coefficients) @ basis
                reference = p[key + "_coef"] @ basis
                if n_bessel == 8:
                    assert _scaled(out.coefficients, p[key + "_coef"]) < 1e-9
                    assert _scaled(out.residual_kinetic,
                                   p[key + "_res"]) < 1e-9
                assert _scaled(smooth, reference) < 1e-8

    @pytest.mark.parametrize("symbol", ["H", "C", "O", "Si", "Cu"])
    def test_the_local_potential(self, frozen, symbol):
        p, meta = frozen
        v = p[f"{symbol}_v"]
        r = p[f"{symbol}_h"][0] * np.arange(1, v.size + 1)
        for r_local, shift, window in ((2.0, 0.0, None), (2.5, 0.3, None),
                                       (1.6, 0.0, 0.15 * 1.6)):
            r_local = pw._snap(r, r_local)
            reference = p[f"{symbol}_vloc_{r_local:.4f}_{shift}_{window}"]
            out = pw.polynomial_local_potential(r, v, r_local, shift, window)
            assert _scaled(out[:reference.size], reference) < 1e-8
            assert np.array_equal(out[reference.size:], v[reference.size:])
