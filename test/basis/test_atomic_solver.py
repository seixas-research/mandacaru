"""The self-consistent atom: the Python interface of the native solver, the frozen
parity cases and the reference-configuration rule of
:func:`mandacaru.basis.atomic_solver.solve_atom`.

The Pulay mixer, the convergence measure and the configuration helpers live in Rust and are
tested in ``crates/mandacaru-core/tests/atom_solver.rs``, which also runs the same frozen
cases (``test/migration/reference/paw/atomic_solver.npz``) through the core directly.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru.basis._config import ground_state_config
from mandacaru.basis.atomic_solver import (hartree_potential, lda_correlation,
                                           lda_exchange, lda_xc,
                                           relaxed_configuration, solve_atom,
                                           solve_radial)

REFERENCE = Path(__file__).resolve().parents[1] / "migration" / "reference" / "paw"


def _frozen():
    with np.load(REFERENCE / "atomic_solver.npz") as data:
        arrays = {key: data[key] for key in data.files}
    with open(REFERENCE / "atomic_solver.json") as handle:
        return arrays, json.load(handle)


ARRAYS, INPUTS = _frozen()
CASES = sorted(INPUTS["cases"], key=lambda key: (int(key.split("/")[0]), key))
#: Heavy relativistic atoms: the Numerov polish of gold runs for about a minute.
SLOW = {key for key in CASES
        if key.startswith("79/") and not key.endswith("/none")}


class TestFrozenPythonAtoms:
    """Every number below was produced by the pre-port Python solver (git
    ``241d688^``, ``basis/atomic_solver.py``) on the grid recorded in the JSON and is
    reproduced by the native solver to the SCF tolerance (1e-8 there)."""

    @pytest.mark.parametrize("key", [pytest.param(k, marks=pytest.mark.slow)
                                     if k in SLOW else k for k in CASES])
    def test_the_native_atom_reproduces_the_python_one(self, key):
        z, xc, relativity = key.split("/")
        atom = solve_atom(int(z), points=INPUTS["points"], r_max=INPUTS["r_max"],
                          tolerance=INPUTS["tolerance"], mixing=INPUTS["mixing"],
                          max_iterations=INPUTS["max_iterations"],
                          polish=INPUTS["polish"], xc=xc, relativity=relativity)
        case = INPUTS["cases"][key]
        assert atom.converged == case["converged"]
        assert atom.xc == xc and atom.relativity == relativity
        assert atom.total_energy == pytest.approx(case["total_energy"], rel=2e-8)
        nl = ARRAYS[key + "/nl"]
        frozen = ARRAYS[key + "/eigenvalues"]
        assert sorted(atom.eigenvalues) == sorted(map(tuple, nl.tolist()))
        for (n, l), want in zip(nl.tolist(), frozen):
            assert atom.eigenvalues[(n, l)] == pytest.approx(want, rel=2e-8)
        assert [atom.occupations[tuple(k)] for k in nl.tolist()] == \
            ARRAYS[key + "/occupations"].tolist()
        if relativity == "dirac":
            for (n, l, k), want in zip(ARRAYS[key + "/nlk"].tolist(),
                                       ARRAYS[key + "/eigenvalues_j"]):
                assert atom.eigenvalues_j[(n, l, k)] == pytest.approx(want, rel=2e-8)
        else:
            assert not atom.eigenvalues_j
        # Fields on every 20th node: the grid is r_i = i h, h = r_max / (points + 1).
        # The far tail (rho < 1e-8) is not controlled by the SCF test, which weighs the
        # change by 4 pi r^2, so there only absolute agreement at 5e-6 Hartree is asked.
        # PBE between the kernel's two density floors (1e-12 < rho < 2e-12) applies the
        # uniform-gas exchange without the enhancement factor: that window (a handful of
        # nodes, up to 1e-4 Hartree in v_xc) is excluded from the potentials, widened to
        # 0.5-4e-12 because the unconverged tail moves it between runs.
        stride = slice(0, None, 20)
        rho = ARRAYS[key + "/density"]
        keep = np.ones_like(rho, dtype=bool)
        if xc == "pbe":
            keep = ~((rho > 0.5e-12) & (rho < 4e-12))
        for name in ("density", "v_effective", "v_hartree", "v_xc"):
            got = np.asarray(getattr(atom, name))[stride]
            want = ARRAYS[key + "/" + name]
            atol = 1e-8 * float(np.max(np.abs(want))) if name == "density" else 5e-6
            mask = keep if name in ("v_effective", "v_xc") else np.ones_like(keep)
            np.testing.assert_allclose(got[mask], want[mask], rtol=1e-5, atol=atol,
                                       err_msg=name)
        last = tuple(ARRAYS[key + "/last_nl"].tolist())
        got = atom.orbitals[last][stride]
        want = ARRAYS[key + "/u_last"]
        np.testing.assert_allclose(got, want, rtol=1e-5,
                                   atol=1e-7 * float(np.max(np.abs(want))))


class TestRelaxedConfiguration:
    def test_no_f_shell_returns_the_aufbau_filling(self):
        assert relaxed_configuration(26) == ground_state_config(26)
        assert relaxed_configuration(8) == ground_state_config(8)

    def test_an_unbound_f_shell_gives_up_one_electron_to_the_d_shell(self):
        """At 1500 points in a 60 Bohr box lanthanum's aufbau ``4f`` is a box state
        (-0.0006 Ha in the frozen Python run); 5d^1 6s^2 is chosen."""
        want = INPUTS["relaxed"]["relax3:57,1500,60.0"]
        chosen = relaxed_configuration(57, r_max=60.0, points=1500)
        assert {f"{n},{l}": q for (n, l), q in chosen.items()} == want
        assert chosen[(5, 2)] == 1 and (4, 3) not in chosen

    @pytest.mark.parametrize("name", ["relax2:57,3000,25.0", "relax2:90,6000,30.0",
                                      "relax3:89,800,60.0", "relax3:91,600,60.0"])
    def test_a_bound_f_shell_keeps_the_aufbau_filling(self, name):
        _, rest = name.split(":")
        z, points, r_max = rest.split(",")
        chosen = relaxed_configuration(int(z), r_max=float(r_max), points=int(points))
        assert {f"{n},{l}": q for (n, l), q in chosen.items()} == INPUTS["relaxed"][name]
        assert chosen == ground_state_config(int(z))


class TestSolveAtomInterface:
    def test_oxygen_converges_quickly(self):
        atom = solve_atom(8)
        assert atom.converged
        assert atom.iterations < 40

    def test_an_unsupported_request_raises(self):
        with pytest.raises(ValueError, match="r2scan|canonical"):
            solve_atom(2, points=300, xc="r2scan")
        with pytest.raises(ValueError, match="zora"):
            solve_atom(2, points=300, relativity="zora")
        with pytest.raises(TypeError):
            solve_atom(2, points=300, grid="log")

    def test_a_confinement_array_must_match_the_grid(self):
        with pytest.raises(ValueError, match="confinement"):
            solve_atom(2, points=300, confinement=np.zeros(10))

    def test_a_confining_wall_is_not_part_of_the_reported_potential(self):
        free = solve_atom(2, points=600, r_max=15.0)
        walled = solve_atom(2, points=600, r_max=15.0,
                            confinement=lambda r: 0.05 * r ** 2)
        assert walled.details["confined"] and not free.details["confined"]
        assert walled.eigenvalues[(1, 0)] > free.eigenvalues[(1, 0)]

    def test_the_dirac_atom_carries_its_j_levels(self):
        atom = solve_atom(10, points=3000, r_max=18.0, relativity="dirac")
        assert atom.relativity == "dirac"
        assert atom.details["relativistic_exchange"]
        assert set(atom.eigenvalues_j) >= {(2, 1, 1), (2, 1, -2), (2, 0, -1)}
        assert atom.spin_orbit_splitting(2, 1) > 0.0
        assert atom.spin_orbit_splitting(2, 0) == 0.0


#: NIST LSD reference eigenvalues (Hartree) for closed-shell atoms.
NIST_EIGENVALUES = {
    2: {(1, 0): -0.570425},
    4: {(1, 0): -3.856411, (2, 0): -0.205719},
}


@pytest.fixture(scope="module")
def oxygen_atom():
    return solve_atom(8, points=6000, r_max=30.0, tolerance=1e-7, mixing=0.25)


class TestAtomicSolver:
    @pytest.mark.parametrize("atomic_number", sorted(NIST_EIGENVALUES))
    def test_matches_published_lda_eigenvalues(self, atomic_number):
        """Closed-shell atoms, where spin-restricted LDA is the right model."""
        atom = solve_atom(atomic_number, points=6000, r_max=30.0,
                          tolerance=1e-7, mixing=0.25)
        assert atom.converged
        for state, reference in NIST_EIGENVALUES[atomic_number].items():
            assert atom.eigenvalues[state] == pytest.approx(reference, abs=5e-3)

    def test_iron_valence_matches_the_nist_lda_reference(self):
        """Fe [Ar]3d6 4s2 (Kotochigova et al., https://math.nist.gov/DFTdata/):
        3d -0.295049, 4s -0.197978, 3s -3.360621 Hartree.  The uniform grid reads the
        valence within 1e-3 and the 3s within 3e-3 (it misses the 1s by far more)."""
        atom = solve_atom(26, points=12000, r_max=25.0)
        assert atom.converged
        assert atom.eigenvalues[(3, 2)] == pytest.approx(-0.295049, abs=1e-3)
        assert atom.eigenvalues[(4, 0)] == pytest.approx(-0.197978, abs=1e-3)
        assert atom.eigenvalues[(3, 0)] == pytest.approx(-3.360621, abs=3e-3)

    def test_valence_eigenvalue_is_accurate(self, oxygen_atom):
        """The valence shell is what a pseudopotential is built from."""
        assert oxygen_atom.eigenvalues[(2, 1)] == pytest.approx(-0.338381, abs=2e-3)

    def test_density_integrates_to_the_electron_count(self, oxygen_atom):
        r = oxygen_atom.r
        electrons = np.trapezoid(4.0 * np.pi * oxygen_atom.density * r * r, r)
        assert electrons == pytest.approx(8.0, rel=2e-3)

    def test_orbitals_have_the_right_node_count(self, oxygen_atom):
        for (n, l), u in oxygen_atom.orbitals.items():
            interior = u[(oxygen_atom.r > 0.05) & (oxygen_atom.r < 12.0)]
            nodes = int(np.sum(np.diff(np.sign(interior)) != 0))
            assert nodes == n - l - 1, f"state {(n, l)} has {nodes} nodes"

    def test_hartree_potential_of_a_point_charge(self):
        """A tight spherical blob must look like ``1/r`` outside itself."""
        r = np.linspace(1e-3, 20.0, 4000)
        width = 0.2
        rho = np.exp(-(r / width) ** 2) / (np.pi ** 1.5 * width ** 3)
        potential = hartree_potential(r, rho)
        far = r > 2.0
        assert np.allclose(potential[far], 1.0 / r[far], rtol=1e-4)

    def test_lda_xc_is_negative(self):
        e_xc, v_xc = lda_xc(np.array([0.01, 0.1, 1.0]))
        assert np.all(e_xc < 0) and np.all(v_xc < 0)

    def test_lda_terms_follow_the_closed_forms(self):
        """Slater exchange ``-3/4 (3/pi)^(1/3) rho^(1/3)`` with ``v_x = 4/3 e_x``, and
        the Perdew-Zunger correlation in both of its branches (``r_s`` 0.5 and 5)."""
        rho = np.array([0.5, 1.0, 1e-3])
        e_x, v_x = lda_exchange(rho)
        np.testing.assert_allclose(e_x, -0.75 * (3.0 / np.pi) ** (1 / 3) * rho ** (1 / 3),
                                   rtol=1e-13)
        np.testing.assert_allclose(v_x, 4.0 / 3.0 * e_x, rtol=1e-13)
        density = 3.0 / (4.0 * np.pi * np.array([0.5, 5.0]) ** 3)
        e_c, v_c = lda_correlation(density)
        # rs = 5: the Pade form gamma / (1 + beta1 sqrt(rs) + beta2 rs)
        expected = -0.1423 / (1.0 + 1.0529 * np.sqrt(5.0) + 0.3334 * 5.0)
        assert e_c[1] == pytest.approx(expected, rel=1e-13)
        # v_c = e_c - (rs / 3) de_c/drs, checked by a central difference in rs
        for rs, e, v in zip((0.5, 5.0), e_c, v_c):
            h = 1e-5 * rs
            up, down = (lda_correlation(3.0 / (4.0 * np.pi * np.array([rs + h, rs - h]) ** 3))[0])
            assert v == pytest.approx(e - rs / 3.0 * (up - down) / (2 * h), rel=1e-7)

    def test_a_radial_state_matches_the_hydrogenic_level(self):
        r = np.arange(1, 6001) * (40.0 / 6001)
        u, eps = solve_radial(r, -2.0 / r, 1, 1)          # 3p of He+
        assert eps == pytest.approx(-4.0 / 18.0, rel=1e-3)
        assert np.trapezoid(u * u, r) == pytest.approx(1.0, rel=1e-12)
        assert u[np.argmax(np.abs(u[:200]))] > 0.0
