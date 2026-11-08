# -*- coding: utf-8 -*-
# file: test/basis/test_xc.py

"""Exchange-correlation functionals of the radial atom, through the native kernels.

The functionals are Rust (``models/xc.rs``, ``models/radial_xc.rs``); the Python
PW92/PBE implementations are deleted. The potential-is-the-derivative identity of
the radial GGA is pinned in Rust (``tests/xc_scheme.rs``). Here: the
Perdew-Zunger continuity of the native LDA, its spin form, the generator's
functional names, and a self-consistent PBE atom.
"""

import numpy as np
import pytest

from mandacaru.basis.atomic_solver import lda_correlation, lda_xc
from .xc_points import spin_partials


class TestPerdewZunger:
    """The two forms meet at :math:`r_s = 1` with no jump in the energy or
    the potential (the published C, D left 3.2e-5 Ha there), restricted
    and spin-polarized alike."""

    RS = np.array([1.0 - 1e-10, 1.0 + 1e-10])

    def test_the_restricted_form_is_continuous_at_rs_one(self):
        rho = 3.0 / (4.0 * np.pi * self.RS ** 3)
        e_c, v_c = lda_correlation(rho)
        assert abs(e_c[0] - e_c[1]) < 1e-10
        assert abs(v_c[0] - v_c[1]) < 1e-10

    @pytest.mark.parametrize("zeta", [0.0, 0.5, 1.0 - 1e-9])
    def test_the_polarized_form_is_continuous_at_rs_one(self, zeta):
        rho = 3.0 / (4.0 * np.pi * self.RS ** 3)
        up, dn = 0.5 * rho * (1 + zeta), 0.5 * rho * (1 - zeta)
        f, v_up, v_dn = spin_partials("lda", up, dn)[:3]
        assert abs(f[0] / rho[0] - f[1] / rho[1]) < 1e-9
        assert abs(v_up[0] - v_up[1]) < 1e-8
        assert abs(v_dn[0] - v_dn[1]) < 1e-8

    def test_the_spin_form_reduces_to_the_restricted_one(self):
        rho = np.array([1e-3, 0.05, 0.2387, 1.0, 10.0])
        f = spin_partials("lda", 0.5 * rho, 0.5 * rho)[0]
        assert np.allclose(f, rho * lda_xc(rho)[0], rtol=1e-12, atol=0)


class TestTheGeneratorsFunctionals:
    def test_an_unknown_functional_is_named_with_the_alternatives(self):
        from mandacaru.pseudopotentials.generation import generate_paw
        with pytest.raises(ValueError, match="canonical XC"):
            generate_paw("H", xc="b3lyp")


class TestAgainstTheSelfConsistentAtom:
    def test_a_pbe_atom_differs_from_an_lda_one_and_stays_bound(self):
        from mandacaru.basis.atomic_solver import solve_atom

        lda = solve_atom(8, points=2500, r_max=18.0, tolerance=1e-5)
        pbe = solve_atom(8, points=2500, r_max=18.0, tolerance=1e-5, xc="pbe")
        assert pbe.xc == "pbe" and lda.xc == "lda"
        assert pbe.total_energy != lda.total_energy
        # PBE binds oxygen more tightly than LDA does, but not by a lot.
        assert abs(pbe.total_energy - lda.total_energy) < 2.0
        assert pbe.eigenvalues[(2, 1)] < 0.0
