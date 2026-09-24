"""The self-consistent atom: mixing, convergence test, reference configuration
and the logarithmic-grid path of :func:`mandacaru.basis.atomic_solver.solve_atom`.
"""
import numpy as np
import pytest

from mandacaru.basis import atomic_solver
from mandacaru.basis._config import ground_state_config
from mandacaru.basis.atomic_solver import (_PulayMixer, density_change,
                                           relaxed_configuration, solve_atom)


def _radial_grid(points=400, r_max=20.0):
    step = r_max / (points + 1)
    r = np.arange(1, points + 1) * step
    return r, 4.0 * np.pi * r * r, step


class TestDensityChange:
    def test_it_sees_a_valence_change_the_plain_maximum_misses(self):
        """``max|dn|`` is a statement about the core; a change in the tail must
        still register, through the ``4 pi r^2`` weight."""
        r, shell, _step = _radial_grid()
        core = 1e4 * np.exp(-40.0 * r)
        tail = 1e-3 * np.exp(-r)
        moved = 1e-3 * np.exp(-0.9 * r)
        plain = float(np.max(np.abs(moved - tail)))
        assert plain < 1e-4
        assert density_change(core + tail, core + moved, shell) > 10.0 * plain

    def test_it_is_never_weaker_than_the_plain_maximum(self):
        r, shell, _step = _radial_grid()
        a, b = np.exp(-r), np.exp(-1.1 * r)
        assert density_change(a, b, shell) >= float(np.max(np.abs(a - b)))


class TestPulayMixer:
    def _linear_problem(self, r, shell):
        """A contracting linear map with a known positive fixed point."""
        fixed = np.exp(-r) + 0.1 * np.exp(-0.3 * r)
        rng = np.random.default_rng(7)
        basis = np.array([np.exp(-k * r) for k in (0.5, 1.0, 2.0, 3.0)])
        coupling = rng.uniform(-0.9, 0.9, (4, 4))

        def output(n):
            delta = n - fixed
            return fixed + 0.95 * delta + 0.02 * coupling.dot(basis.dot(delta)).dot(basis)
        return fixed, output

    def test_it_beats_linear_mixing_on_a_linear_fixed_point(self):
        r, shell, step = _radial_grid()
        fixed, output = self._linear_problem(r, shell)
        start = 2.0 * fixed

        def run(mixer, iterations=60):
            n = start.copy()
            for _ in range(iterations):
                n = mixer(n, output(n))
            return float(np.max(np.abs(n - fixed)))

        pulay = run(_PulayMixer(shell * step, 0.3, 1.0))
        linear = run(_PulayMixer(shell * step, 0.3, 1.0, history=1))
        assert pulay < 1e-8
        assert pulay < 1e-3 * linear

    def test_it_conserves_charge_without_renormalizing(self):
        r, shell, step = _radial_grid()
        weight = shell * step
        electrons = 3.0
        mixer = _PulayMixer(weight, 0.3, electrons)
        n = np.exp(-r)
        n *= electrons / np.sum(weight * n)
        for scale in (1.3, 0.8, 1.1, 0.95):
            out = np.exp(-scale * r)
            out *= electrons / np.sum(weight * out)
            n = mixer(n, out)
            assert np.sum(weight * n) == pytest.approx(electrons, rel=1e-12)

    def test_a_negative_extrapolation_falls_back_to_the_linear_step(self):
        r, shell, step = _radial_grid()
        mixer = _PulayMixer(shell * step, 0.3, 1.0)
        mixed = np.exp(-r) - 0.5 * np.exp(-0.5 * r)
        linear = np.exp(-r)
        assert mixer._accept(mixed, linear) is linear


class TestSolveAtom:
    def test_oxygen_converges_quickly(self):
        atom = solve_atom(8)
        assert atom.converged
        assert atom.iterations < 40

    def test_iron_on_the_log_grid(self):
        """The Thomas-Fermi start binds no 3d state on the log grid; the atom
        starts from the converged uniform one instead.  The eigenvalues are
        checked against the NIST LDA atomic reference data for Fe [Ar]3d6 4s2
        (Kotochigova et al., https://math.nist.gov/DFTdata/): 3d -0.295049,
        4s -0.197978, 3s -3.360621 Hartree.  The uniform grid misses the 1s
        by 0.24 Hartree at this size; the log grid is within 2e-3."""
        atom = solve_atom(26, relativity="none", grid="log", points=12000,
                          r_max=25.0, polish=0)
        assert atom.converged
        assert atom.eigenvalues[(3, 2)] == pytest.approx(-0.295049, abs=1e-3)
        assert atom.eigenvalues[(4, 0)] == pytest.approx(-0.197978, abs=1e-3)
        assert atom.eigenvalues[(3, 0)] == pytest.approx(-3.360621, abs=1e-3)
        assert atom.eigenvalues[(1, 0)] == pytest.approx(-254.225505, abs=2e-3)

    def test_the_relativistic_shift_of_iron_on_the_log_grid(self):
        """Scalar relativity pulls 4s in and pushes 3d out (indirect
        screening); the log grid has the textbook sign at any size."""
        kwargs = dict(grid="log", points=12000, r_max=25.0, polish=0)
        plain = solve_atom(26, relativity="none", **kwargs)
        scalar = solve_atom(26, relativity="scalar", **kwargs)
        assert scalar.converged
        assert scalar.eigenvalues[(4, 0)] < plain.eigenvalues[(4, 0)]
        assert scalar.eigenvalues[(3, 2)] > plain.eigenvalues[(3, 2)]


class TestRelaxedConfiguration:
    def test_no_f_shell_costs_no_solve(self, monkeypatch):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("solve_atom must not run for iron")
        monkeypatch.setattr(atomic_solver, "solve_atom", forbidden)
        assert relaxed_configuration(26) == ground_state_config(26)

    def test_the_grid_is_part_of_the_cache_key(self, monkeypatch):
        """Which configuration wins depends on the grid (cerium's aufbau 4f is
        unbound at 6000 points and bound at 12000), so a result at one grid
        must not answer for another."""
        calls = []

        class Atom:
            converged = True

        def fake(Z, *, points, **_kwargs):
            calls.append(points)
            return Atom()
        monkeypatch.setattr(atomic_solver, "solve_atom", fake)
        monkeypatch.setattr(atomic_solver, "bound_valence", lambda atom: True)
        monkeypatch.setattr(atomic_solver, "_CONFIGURATION_CACHE", {})
        relaxed_configuration(58, points=1000)
        relaxed_configuration(58, points=1000)
        relaxed_configuration(58, points=2000)
        assert calls == [1000, 2000]
