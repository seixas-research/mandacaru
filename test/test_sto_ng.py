# -*- coding: utf-8 -*-
# file: test_sto_ng.py

"""Native STO-nG generator: fits Gaussians to Slater orbitals from scratch."""

import numpy as np
import pytest

from carcara.basis.sto_ng import (
    _fit_reference,
    effective_principal_number,
    occupied_subshells,
    slater_exponent,
    sto_ng_contraction,
    sto_ng_shells,
)


class TestReferenceFit:
    def test_h1s_matches_published_sto3g(self):
        # The from-scratch fit must recover the standard STO-3G 1s (zeta=1)
        # exponents to a few figures.
        exps, coeffs, ov = _fit_reference(1, 0, 3)
        assert np.allclose(exps, [2.22766, 0.40577, 0.10982], rtol=2e-4)
        assert np.allclose(coeffs, [0.15433, 0.53533, 0.44463], atol=2e-3)
        assert ov > 0.999                         # squared overlap with the STO

    @pytest.mark.parametrize("n, l", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)])
    def test_fit_recovers_slater_orbital(self, n, l):
        # More Gaussians -> better overlap, always monotone and high.
        ov3 = _fit_reference(n, l, 3)[2]
        ov6 = _fit_reference(n, l, 6)[2]
        assert 0.99 < ov3 <= ov6 <= 1.0 + 1e-9

    def test_2s_has_negative_lead_coefficient(self):
        # The classic STO-nG signature: the tight s-Gaussian enters with a
        # negative coefficient to shape the r e^{-zr} node region.
        _exps, coeffs, _ov = _fit_reference(2, 0, 3)
        assert coeffs[0] < 0

    def test_is_cached(self):
        assert _fit_reference(1, 0, 3) is _fit_reference(1, 0, 3)


class TestSlaterExponent:
    def test_hydrogen_zeta_is_one(self):
        assert np.isclose(slater_exponent(1, 1, 0), 1.0)

    def test_effective_principal_numbers(self):
        assert effective_principal_number(1) == 1.0
        assert effective_principal_number(4) == 3.7

    def test_unsupported_shell_raises(self):
        with pytest.raises(ValueError):
            effective_principal_number(7)


class TestContraction:
    def test_zeta_squared_scaling(self):
        # Exponents scale as zeta^2 off the zeta=1 reference; coeffs unchanged.
        ref_exps, ref_coeffs, _ = _fit_reference(2, 1, 3)
        exps, coeffs = sto_ng_contraction(6, 2, 1, 3)     # carbon 2p
        zeta = slater_exponent(6, 2, 1)
        assert np.allclose(exps, np.array(ref_exps) * zeta ** 2)
        assert np.allclose(coeffs, ref_coeffs)

    def test_length_matches_n_gaussians(self):
        exps, coeffs = sto_ng_contraction(8, 2, 0, 4)
        assert exps.shape == coeffs.shape == (4,)


class TestOccupiedSubshells:
    @pytest.mark.parametrize("Z, expected", [
        (1, [(1, 0)]),
        (6, [(1, 0), (2, 0), (2, 1)]),
        (10, [(1, 0), (2, 0), (2, 1)]),
        (18, [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]),
    ])
    def test_minimal_basis_subshells(self, Z, expected):
        assert occupied_subshells(Z) == expected

    def test_shells_cover_every_subshell(self):
        shells = sto_ng_shells(6, 3)
        assert [(n, l) for (n, l, _e, _c) in shells] == occupied_subshells(6)
