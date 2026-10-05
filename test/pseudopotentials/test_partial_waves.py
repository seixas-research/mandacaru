# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_partial_waves.py

# This code is part of Mandacaru.
# MIT License

"""The radial machinery of PAW-LCAO generation
(:mod:`mandacaru.pseudopotentials.partial_waves`): Bessel matching, the
constrained minimization of the smooth partial waves, Numerov bound states,
and the ghost search on a stub generator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import spherical_jn

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
        """Phase errors by dataset, default clean."""
        from mandacaru.pseudopotentials import partial_waves as pw
        self.phases = {}
        monkeypatch.setattr(pw, "scattering_errors",
                            lambda pp, _ld, _cache=None:
                            self.phases.get(id(pp), {}))

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
        The Bessel attempts come after the own-cutoff shifts."""
        from mandacaru.pseudopotentials.partial_waves import OWN_CUTOFF_SHIFTS

        first = self._missing(1.5)
        shifts = [self._missing(1.6) for _ in OWN_CUTOFF_SHIFTS]
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
        from mandacaru.pseudopotentials.partial_waves import OWN_CUTOFF_SHIFTS

        attempts = (len(OWN_CUTOFF_SHIFTS) + 2 * (len(OWN_CUTOFF_SHIFTS) + 1)
                    + 3 * 3 + 5)
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
        deep = self._Dataset({0: (-5.0, -0.2), 2: (-0.2, 0.3)})
        shallow = self._Dataset({0: (-0.5, -0.2), 2: (-0.2, 0.3)})
        rest = [self._Dataset(self.GHOST) for _ in range(9)]
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
        from mandacaru.pseudopotentials.partial_waves import OWN_CUTOFF_SHIFTS
        n_own = len(OWN_CUTOFF_SHIFTS)
        stills = [self._Dataset(self.GHOST) for _ in range(1 + n_own)]
        clean = self._Dataset(self.CLEAN)
        generate, calls = self._generator([self._Dataset(self.GHOST)]
                                          + stills + [clean])
        assert self._run(generate, overrides={"norm_deficit": 0.0}) is clean
        alone, raised, balanced = calls[1], calls[2], calls[2 + n_own]
        assert alone["norm_deficit"] == 0.0
        assert alone["r_cut"] is None           # the cutoffs are untouched
        assert alone["atom"] == "atom"          # the SCF atom is reused
        assert raised["r_cut"] is None          # raised at the own cutoffs
        assert raised["local_shift"] == OWN_CUTOFF_SHIFTS[0]
        assert balanced["norm_deficit"] == 0.0  # and kept with balanced ones
        assert balanced["r_cut"] == {0: 3.0, 2: 3.0}

    def test_a_ghost_is_repaired_with_balanced_cutoffs_and_a_raised_shift(self):
        from mandacaru.pseudopotentials.partial_waves import (GHOST_REMEDY_SHIFTS,
                                                     OWN_CUTOFF_SHIFTS)
        n_own = len(OWN_CUTOFF_SHIFTS)
        own = [self._Dataset(self.GHOST) for _ in range(n_own)]
        still = self._Dataset({0: (-0.5, -0.2), 2: (-0.2, 0.3)})
        clean = self._Dataset(self.CLEAN)
        generate, calls = self._generator([self._Dataset(self.GHOST)] + own
                                          + [still, clean])
        assert self._run(generate) is clean
        assert [c["local_shift"] for c in calls[1:1 + n_own]] == list(
            OWN_CUTOFF_SHIFTS)
        first, second = calls[1 + n_own], calls[2 + n_own]
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
        from mandacaru.pseudopotentials.partial_waves import (GHOST_REMEDY_SHIFTS,
                                                     OWN_CUTOFF_SHIFTS,
                                                     GhostStateError)
        n = 1 + len(OWN_CUTOFF_SHIFTS) + len(GHOST_REMEDY_SHIFTS)
        generate, calls = self._generator([self._Dataset(self.GHOST)] * n)
        with pytest.raises(GhostStateError, match="no remedy removed it") as error:
            self._run(generate)
        assert len(calls) == n
        assert "own cutoffs, shift 5" in str(error.value)

    def test_a_scattering_channel_is_not_judged(self):
        from mandacaru.pseudopotentials.partial_waves import ghost_errors
        pp = self._Dataset({0: (-0.2, 0.3), 2: (-5.0, 0.3)})
        pp.channels[2].reference_energies = [0.3, 1.3]
        assert ghost_errors(pp, self._levels) == {}

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="ghosts must be one of"):
            self._run(None, mode="ignore")
