# -*- coding: utf-8 -*-
# file: test/integrals/test_noncollinear.py

# This code is part of Mandacaru.
# MIT License

"""The noncollinear (local spin axis) exchange-correlation of spinor states."""

import numpy as np
import pytest

from mandacaru.integrals import Grid
from mandacaru.integrals import exchange_correlation as xc_grid
from mandacaru.integrals.noncollinear import evaluate_noncollinear


def _grid():
    return Grid(center=np.zeros(3), box_size=6.0, grid_spacing=0.2,
                units="bohr")


def _gaussian(grid, alpha, center=(0.0, 0.0, 0.0), electrons=1.0):
    x, y, z = (np.asarray(a).reshape(-1) - c
               for a, c in zip((grid.X, grid.Y, grid.Z), center))
    return electrons * (alpha / np.pi) ** 1.5 * np.exp(-alpha * (x * x + y * y
                                                                 + z * z))


def _noncollinear(grid):
    """A density and a magnetization whose direction turns in space."""
    n = _gaussian(grid, 0.8, electrons=4.0)
    m = np.stack([0.6 * _gaussian(grid, 1.1, (0.3, 0.0, 0.0)),
                  0.4 * _gaussian(grid, 0.9, (0.0, -0.2, 0.1)),
                  0.5 * _gaussian(grid, 1.3, (0.0, 0.0, 0.25))])
    return n, m


#: Every functional with a noncollinear form (the hybrid's semilocal part).
FUNCTIONALS = ["lda", "pbe", "hse06", "r2scan", "pbe+rvv10",
               "r2scan+rvv10"]


def _tau(grid, density):
    """A kinetic-energy density above the von Weizsaecker bound of
    ``density`` (what an r2SCAN iso-orbital indicator needs)."""
    g = xc_grid.gradient(grid, density)
    safe = np.maximum(density, 1e-12)
    return 1.3 * np.sum(g * g, axis=0) / (8.0 * safe) + 0.4 * safe ** (5 / 3)


def _collinear(grid, functional, up, down, tau_up=None, tau_dn=None):
    """``evaluate_spin`` with what the functional needs."""
    meta = xc_grid.is_meta_gga(functional)
    screening = xc_grid.HYBRIDS.get(functional)
    return xc_grid.evaluate_spin(
        grid, up, down, functional,
        tau_up=(tau_up if tau_up is not None else _tau(grid, up))
        if meta else None,
        tau_dn=(tau_dn if tau_dn is not None else _tau(grid, down))
        if meta else None,
        screening=screening,
        exchange_densities=(up, down) if screening else None)


def _evaluate(grid, functional, n, m, tau=None, tau_m=None):
    """``evaluate_noncollinear`` with what the functional needs."""
    return evaluate_noncollinear(
        grid, n, m, functional,
        tau=tau if xc_grid.is_meta_gga(functional) else None,
        tau_magnetization=tau_m if xc_grid.is_meta_gga(functional) else None,
        screening=xc_grid.HYBRIDS.get(functional), exchange_density=n)


def _rotation(seed=3):
    q, _r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _channels_match(terms, collinear, rel):
    """The noncollinear potentials along z are the collinear channels'."""
    scale = np.max(np.abs(collinear.potential_up))
    assert np.max(np.abs(terms.potential + terms.field[2]
                         - collinear.potential_up)) < rel * scale
    assert np.max(np.abs(terms.potential - terms.field[2]
                         - collinear.potential_dn)) < rel * scale
    if collinear.tau_potential_up is not None:
        t_scale = np.max(np.abs(collinear.tau_potential_up))
        assert np.max(np.abs(terms.tau_potential + terms.tau_field[2]
                             - collinear.tau_potential_up)) < rel * t_scale
        assert np.max(np.abs(terms.tau_potential - terms.tau_field[2]
                             - collinear.tau_potential_dn)) < rel * t_scale


def _tolerance(functional) -> float:
    """Of the largest channel potential: round-off, amplified where the
    density is 1e-8 to 1e-10 -- r2SCAN's iso-orbital indicator and the
    short-range hole's reduced gradient are ratios of round-off there (up
    to 1e-9 of the peak; the energies agree to 1e-16)."""
    if xc_grid.is_meta_gga(functional):
        return 3e-9
    return 1e-9 if xc_grid.is_hybrid(functional) else 1e-11


def _along_z(grid, up, down):
    """``(n, m, tau, tau_m)`` of a collinear state along z."""
    zero = np.zeros_like(up)
    t_up, t_dn = _tau(grid, up), _tau(grid, down)
    return (up + down, np.stack([zero, zero, up - down]), t_up + t_dn,
            np.stack([zero, zero, t_up - t_dn]))


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_a_collinear_state_is_the_collinear_functional(functional):
    """m = m_z z with m_z >= 0: the energy and both channel potentials of
    the spin-polarized functional (and r2SCAN's tau potentials), to machine
    precision."""
    grid = _grid()
    up = _gaussian(grid, 0.8, electrons=2.5)
    down = _gaussian(grid, 0.8, electrons=1.5)
    collinear = _collinear(grid, functional, up, down)
    n, m, tau, tau_m = _along_z(grid, up, down)
    terms = _evaluate(grid, functional, n, m, tau, tau_m)
    assert terms.energy == pytest.approx(collinear.energy, rel=1e-13)
    _channels_match(terms, collinear, _tolerance(functional))
    assert np.max(np.abs(terms.field[:2])) == 0.0


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_a_moment_that_changes_sign_is_still_the_collinear_functional(
        functional):
    """Where m_z < 0 the local axis is -z and n_+ the down channel.  LDA is
    local; the gradient functionals take the gradient of m projected on the
    axis, not of |m|, so the node of the spin density puts no kink in its
    gradients; r2SCAN's tau channels follow the same axis."""
    grid = _grid()
    up = _gaussian(grid, 1.4, electrons=2.0)
    down = _gaussian(grid, 0.5, electrons=2.0)
    collinear = _collinear(grid, functional, up, down)
    n, m, tau, tau_m = _along_z(grid, up, down)
    assert np.min(m[2]) < 0.0 < np.max(m[2])
    terms = _evaluate(grid, functional, n, m, tau, tau_m)
    assert terms.energy == pytest.approx(collinear.energy, rel=1e-12)
    _channels_match(terms, collinear, _tolerance(functional))


def _noncollinear_tau(grid, n, m):
    """A kinetic-energy density matrix to go with :func:`_noncollinear`."""
    tau = _tau(grid, n)
    return tau, np.stack([0.3 * c * tau / np.maximum(n, 1e-12) for c in m])


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_a_global_spin_rotation_leaves_the_energy_and_turns_the_field(
        functional):
    grid = _grid()
    n, m = _noncollinear(grid)
    tau, tau_m = _noncollinear_tau(grid, n, m)
    R = _rotation()
    terms = _evaluate(grid, functional, n, m, tau, tau_m)
    turned = _evaluate(grid, functional, n, R @ m, tau, R @ tau_m)
    assert turned.energy == pytest.approx(terms.energy, rel=1e-13)
    scale = np.max(np.abs(terms.potential))
    tolerance = min(_tolerance(functional), 1e-9)
    assert np.max(np.abs(turned.potential - terms.potential)) \
        < tolerance * scale
    assert np.max(np.abs(turned.field - R @ terms.field)) < tolerance * scale
    if terms.tau_field is not None:
        t_scale = np.max(np.abs(terms.tau_potential))
        assert np.max(np.abs(turned.tau_field - R @ terms.tau_field)) \
            < 1e-12 * t_scale


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_the_potential_and_field_are_the_energy_s_derivative(functional):
    r"""dE = int (v dn + B . dm + v_tau dtau + B_tau . dtau_m): a central
    difference along a direction that changes the density and the axis of
    the magnetization (and the kinetic-energy density matrix)."""
    grid = _grid()
    n, m = _noncollinear(grid)
    tau, tau_m = _noncollinear_tau(grid, n, m)
    dn = 0.2 * _gaussian(grid, 1.0, (0.1, 0.2, 0.0))
    dm = np.stack([0.1 * _gaussian(grid, 1.2), -0.15 * _gaussian(grid, 0.9),
                   0.05 * _gaussian(grid, 1.0, (0.0, 0.3, 0.0))])
    dtau = 0.1 * _gaussian(grid, 0.9, (0.0, 0.1, -0.1))
    dtau_m = np.stack([0.02 * _gaussian(grid, 1.1), 0.03 * _gaussian(grid, 1.0),
                       -0.02 * _gaussian(grid, 1.2, (0.2, 0.0, 0.0))])
    terms = _evaluate(grid, functional, n, m, tau, tau_m)
    directional = float(np.sum(terms.potential * dn
                               + np.sum(terms.field * dm, axis=0)) * grid.dV)
    if terms.tau_potential is not None:
        directional += float(np.sum(terms.tau_potential * dtau + np.sum(
            terms.tau_field * dtau_m, axis=0)) * grid.dV)
    eps = 1e-4

    def energy(step):
        return _evaluate(grid, functional, n + step * dn, m + step * dm,
                         tau + step * dtau, tau_m + step * dtau_m).energy

    numerical = (energy(eps) - energy(-eps)) / (2 * eps)
    assert numerical == pytest.approx(directional, rel=1e-7)


@pytest.mark.parametrize("functional", ["r2scan", "r2scan+rvv10"])
def test_gradients_given_by_the_caller_give_the_divergence_form(functional):
    """``gradients=`` (the orbitals' product rule, for a molecule's
    meta-GGA): handed the grid's own gradients of ``n`` and ``m`` the
    evaluator returns the same energy, and its potential and field plus the
    divergences of the vector fields it hands back are the divergence-form
    ones."""
    grid = _grid()
    n, m = _noncollinear(grid)
    tau, tau_m = _noncollinear_tau(grid, n, m)
    plain = _evaluate(grid, functional, n, m, tau, tau_m)
    given = evaluate_noncollinear(
        grid, n, m, functional, tau=tau, tau_magnetization=tau_m,
        gradients=(xc_grid.gradient(grid, n), xc_grid.gradient(grid, m)))
    assert given.energy == pytest.approx(plain.energy, rel=1e-12)
    assert plain.gradient_potential is None
    potential = given.potential - xc_grid.divergence(
        grid, given.gradient_potential)
    field = given.field - np.stack([
        xc_grid.divergence(grid, given.gradient_field[:, a])
        for a in range(3)])
    scale = max(np.max(np.abs(plain.potential)), np.max(np.abs(plain.field)))
    assert np.max(np.abs(potential - plain.potential)) < 1e-8 * scale
    assert np.max(np.abs(field - plain.field)) < 1e-8 * scale
    assert np.allclose(given.tau_potential, plain.tau_potential)


@pytest.mark.parametrize("functional", ["r2scan", "r2scan+rvv10"])
def test_the_vector_fields_are_the_derivatives_in_the_given_gradients(
        functional):
    r"""dE = int (v dn + B . dm + W_n . d(grad n) + W_m : d(grad m)) for
    gradients that are not those of the fields they accompany -- the
    orbitals' product rule is only the density matrix's, not a function of
    ``n`` and ``m`` -- with the axis of the magnetization turning."""
    grid = _grid()
    n, m = _noncollinear(grid)
    tau, tau_m = _noncollinear_tau(grid, n, m)
    grad_n, grad_m = xc_grid.gradient(grid, n), xc_grid.gradient(grid, m)
    shape = np.exp(-0.2 * (grid.X ** 2 + grid.Y ** 2
                           + grid.Z ** 2).reshape(-1))
    d_grad_n = 0.05 * grad_n * shape * (1.0 + 0.3 * grid.Z.reshape(-1))
    d_grad_m = 0.05 * grad_m * shape * (1.0 - 0.2 * grid.X.reshape(-1))
    eps = 1e-5

    def energy(step):
        return evaluate_noncollinear(
            grid, n, m, functional, tau=tau, tau_magnetization=tau_m,
            gradients=(grad_n + step * d_grad_n,
                       grad_m + step * d_grad_m)).energy

    terms = evaluate_noncollinear(
        grid, n, m, functional, tau=tau, tau_magnetization=tau_m,
        gradients=(grad_n, grad_m))
    directional = float((np.sum(terms.gradient_potential * d_grad_n)
                         + np.sum(terms.gradient_field * d_grad_m))
                        * grid.dV)
    numerical = (energy(eps) - energy(-eps)) / (2 * eps)
    assert numerical == pytest.approx(directional, rel=1e-6)


def test_a_meta_gga_needs_the_kinetic_energy_density_matrix():
    grid = _grid()
    n, m = _noncollinear(grid)
    with pytest.raises(ValueError, match="kinetic-energy"):
        evaluate_noncollinear(grid, n, m, "r2scan")


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_a_nodal_plane_on_the_grid_is_the_collinear_functional(functional):
    """m_z odd about x = 0, with exact zeros on that grid plane and a
    nonzero gradient through it.  The zero vector has no axis of its own;
    the gradient functionals take it from the gradient, as the collinear
    channels on either side do (a zero axis dropped the spin gradient
    there: 2.65 mHa, and 5e-2 Ha in B_z)."""
    grid = _grid()
    plus = _gaussian(grid, 1.2, (0.8, 0.0, 0.0), electrons=2.0)
    minus = _gaussian(grid, 1.2, (-0.8, 0.0, 0.0), electrons=2.0)
    mz = plus - minus
    n = 1.3 * (plus + minus) + _gaussian(grid, 0.7)
    assert np.sum(mz == 0.0) > 100
    up, down = 0.5 * (n + mz), 0.5 * (n - mz)
    collinear = _collinear(grid, functional, up, down)
    n, m, tau, tau_m = _along_z(grid, up, down)
    terms = _evaluate(grid, functional, n, m, tau, tau_m)
    assert terms.energy == pytest.approx(collinear.energy, abs=1e-12)
    _channels_match(terms, collinear, _tolerance(functional))
    scale = np.max(np.abs(collinear.potential_up))
    assert np.max(np.abs(terms.field[:2])) < 1e-12 * scale


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_a_negative_channel_is_clipped_as_the_collinear_evaluator_does(
        functional):
    """A smooth PAW density can leave a spin channel negative (half a
    partial core does): the channel is clipped at zero, its gradient is the
    clipped channel's and its potential vanishes, as in the collinear
    evaluator -- without the clipped spin density matrix rVV10 along z
    missed the collinear potential by 0.9 on O2's crystal density."""
    grid = _grid()
    up = _gaussian(grid, 0.8, electrons=2.5)
    down = _gaussian(grid, 0.8, electrons=1.0) - _gaussian(
        grid, 2.5, (0.6, 0.0, 0.0), electrons=0.5)
    assert np.min(down) < 0.0
    # No kinetic-energy density where the channel is empty (its von
    # Weizsaecker bound of a clipped density is a ratio of round-off).
    t_up = _tau(grid, up)
    t_dn = np.where(down > 0.0, _tau(grid, np.maximum(down, 0.0)), 0.0)
    collinear = _collinear(grid, functional, up, down, t_up, t_dn)
    zero = np.zeros_like(up)
    terms = _evaluate(grid, functional, up + down,
                      np.stack([zero, zero, up - down]), t_up + t_dn,
                      np.stack([zero, zero, t_up - t_dn]))
    assert terms.energy == pytest.approx(collinear.energy, rel=1e-12)
    _channels_match(terms, collinear, _tolerance(functional))
