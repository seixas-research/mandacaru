# -*- coding: utf-8 -*-
# file: test/basis/test_xc_spin.py

# This code is part of Mandacaru.
# MIT License

"""Spin-polarized LDA, PBE and r2SCAN (`basis/xc_spin.py`)."""

import numpy as np
import pytest

from mandacaru.basis import r2scan
from mandacaru.basis.xc import xc_partials
from mandacaru.basis.xc_spin import spin_partials


def _points(n=200, seed=1):
    rng = np.random.default_rng(seed)
    rho = 10 ** rng.uniform(-3, 0.5, n)
    gradient_up = rng.normal(size=(3, n)) * rho * 0.8
    gradient_dn = rng.normal(size=(3, n)) * rho * 0.8
    return rng, rho, gradient_up, gradient_dn


def _tau(rng, rho, gradient):
    """Above the von Weizsaecker bound, as an exact tau is."""
    return (np.sum(gradient * gradient, axis=0) / (8.0 * rho)
            + 0.3 * rho ** (5.0 / 3.0) * rng.uniform(0.1, 2.0, rho.size))


@pytest.mark.parametrize("functional, relativistic", [
    ("lda", False), ("lda", True), ("pbe", False), ("pbe", True),
    ("r2scan", False)])
def test_an_unpolarized_density_gives_the_unpolarized_functional(
        functional, relativistic):
    """At zeta = 0 the energy, the potential (each channel) and the gradient
    and tau derivatives equal :mod:`mandacaru.basis.xc` /
    :mod:`mandacaru.basis.r2scan` to round-off: every pinned closed-shell
    number in the suite rests on this."""
    rng, rho, gradient, _ = _points()
    half = 0.5 * rho
    s_half = np.sum((0.5 * gradient) ** 2, axis=0)
    tau_half = _tau(rng, half, 0.5 * gradient)
    f, v_up, v_dn, s_uu, s_ud, s_dd, t_up, _t_dn = spin_partials(
        functional, half, half, s_half, s_half, s_half, tau_half, tau_half,
        relativistic=relativistic)
    sigma = 4.0 * s_half
    # f(rho, sigma) = f_spin(rho/2, rho/2, sigma/4, sigma/4, sigma/4).
    d_sigma = (s_uu + s_ud + s_dd) / 4.0
    if functional == "r2scan":
        f0, v0, s0, t0 = r2scan.partials(rho, sigma, 2.0 * tau_half)
        assert np.allclose(t_up, t0, rtol=1e-12, atol=1e-14)
    else:
        f0, v0, dg = xc_partials(rho, np.sqrt(sigma), functional, relativistic)
        s0 = 0.5 * dg / np.sqrt(sigma)
    assert np.allclose(f, f0, rtol=1e-13, atol=1e-15)
    assert np.allclose(v_up, v0, rtol=1e-12, atol=1e-15)
    assert np.allclose(v_dn, v0, rtol=1e-12, atol=1e-15)
    if functional != "lda":
        assert np.allclose(d_sigma, s0, rtol=1e-10, atol=1e-14)


#: Against libxc: LDA exactly (PZ81 continuous at r_s = 1, libxc's
#: ``LDA_C_PZ_MOD``); PBE and r2SCAN to the parameter digits libxc
#: carries beyond the published fits (with its constants PBE agrees to 1e-13;
#: the unpolarized r2SCAN sits at the same 1e-6 level).
LIBXC = {"lda": ("LDA_X,LDA_C_PZ_MOD", 1, 1e-12),
         "pbe": ("PBE", 4, 3e-6),
         "r2scan": ("R2SCAN", 6, 3e-5)}


@pytest.mark.parametrize("functional", sorted(LIBXC))
@pytest.mark.parametrize("zeta", [0.3, 0.9])
def test_a_polarized_density_matches_libxc(functional, zeta):
    libxc = pytest.importorskip("pyscf.dft.libxc")
    code, rows, tolerance = LIBXC[functional]
    rng, rho, gradient_up, gradient_dn = _points()
    up, dn = rho * (1 + zeta) / 2, rho * (1 - zeta) / 2
    gu, gd = gradient_up * (1 + zeta) / 2, gradient_dn * (1 - zeta) / 2
    tu, td = _tau(rng, up, gu), _tau(rng, dn, gd)
    zero = np.zeros(rho.size)
    a = np.vstack([up, gu, zero, tu])[:rows]
    b = np.vstack([dn, gd, zero, td])[:rows]
    exc, vxc = libxc.eval_xc(code, (a[0], b[0]) if rows == 1 else (a, b),
                             spin=1, deriv=1)[:2]
    mine = spin_partials(functional, up, dn, np.sum(gu * gu, 0),
                         np.sum(gu * gd, 0), np.sum(gd * gd, 0), tu, td)

    def relative(ours, theirs):
        theirs = np.asarray(theirs)
        return np.abs(ours - theirs).max() / np.abs(theirs).max()

    assert relative(mine[0], exc * rho) < tolerance
    assert relative(np.stack(mine[1:3], 1), vxc[0]) < tolerance
    if functional != "lda":
        assert relative(np.stack(mine[3:6], 1), vxc[1]) < tolerance
    if functional == "r2scan":
        assert relative(np.stack(mine[6:8], 1), vxc[3]) < tolerance


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_full_polarization_and_empty_points_are_finite(functional):
    """A one-electron region (rho_down = 0) and vacuum (rho = 0) give finite
    values everywhere, no exchange for the empty channel, and the energy of
    the occupied channel alone."""
    rng, rho, gradient, _ = _points(50)
    rho[:5] = 0.0
    gradient[:, :5] = 0.0
    tau = _tau(rng, np.maximum(rho, 1e-30), gradient)
    tau[:5] = 0.0
    zero = np.zeros(rho.size)
    out = spin_partials(functional, rho, zero, np.sum(gradient ** 2, 0), zero,
                        zero, tau, zero)
    for array in out:
        assert np.all(np.isfinite(array))
    assert np.all(out[0][:5] == 0.0)
    assert np.all(out[0][5:] < 0.0)
