# -*- coding: utf-8 -*-
# file: test/algorithms/test_orbital_response.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The z-vector orbital response of an active-space energy.

The algebra is checked on a small random model against the derivative of the
energy of re-solved Hartree-Fock orbitals (the quantity a displaced calculation
reports), in a real and in a complex orthonormal frame.
"""

import numpy as np
import pytest

from mandacaru.algorithms import orbital_response as orr

M, NOCC = 6, 2
SUPPORT = [0, 1, 2, 3]


def symmetric(rng, shape):
    a = rng.normal(size=shape)
    return 0.5 * (a + a.T)


def two_body(L):
    """``<ab|cd> = sum_K L^K_ac L^K_bd``: a positive Coulomb-like tensor."""
    return np.einsum("kac,kbd->abcd", L, L)


def solve_scf(h, g):
    w, V = np.linalg.eigh(h)
    P = 2 * V[:, :NOCC] @ V[:, :NOCC].conj().T
    for _ in range(3000):
        w, V = np.linalg.eigh(h + orr._fock_response(g, P))
        new = 2 * V[:, :NOCC] @ V[:, :NOCC].conj().T
        if np.abs(new - P).max() < 1e-15:
            break
        P = 0.5 * (P + new)
    return w, V


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(3)
    h0 = np.diag(np.arange(M) * 1.0) + 0.1 * symmetric(rng, (M, M))
    dh = symmetric(rng, (M, M))
    L0 = 0.15 * np.array([symmetric(rng, (M, M)) for _ in range(4)])
    dL = 0.1 * np.array([symmetric(rng, (M, M)) for _ in range(4)])
    D = np.zeros((M, M))
    D[0, 0] = 2.0
    a = rng.normal(size=(3, 3))
    a = a @ a.T
    D[1:4, 1:4] = a / np.trace(a) * 2.0
    G = (np.einsum("pr,qs->pqrs", D, D)
         - 0.5 * np.einsum("ps,qr->pqrs", D, D))
    pert = rng.normal(size=(3, 3, 3, 3))
    pert = pert + pert.transpose(1, 0, 3, 2)
    pert = pert + pert.transpose(2, 3, 0, 1)
    G[1:4, 1:4, 1:4, 1:4] += 0.05 * pert
    return dict(h0=h0, dh=dh, L0=L0, dL=dL, D=D, G=G)


def relaxed_energy(model, lam, V_ref):
    h = model["h0"] + lam * model["dh"]
    g = two_body(model["L0"] + lam * model["dL"])
    _w, V = solve_scf(h, g)
    V = V * np.sign(np.einsum("ap,ap->p", V, V_ref))
    hm, gm = orr.transform(h, g, V)
    return (np.einsum("pq,pq->", model["D"], hm)
            + 0.5 * np.einsum("pqrs,pqrs->", model["G"], gm))


def directional(model, V, D, G):
    """``<D, dh> + 1/2 <G, dg>`` with the orbitals ``V`` held fixed."""
    L0, dL = model["L0"], model["dL"]
    dg = (np.einsum("kac,kbd->abcd", L0, dL)
          + np.einsum("kac,kbd->abcd", dL, L0))
    hm, gm = orr.transform(model["dh"], dg, V)
    return (np.einsum("pq,pq->", D, hm)
            + 0.5 * np.einsum("pqrs,pqrs->", G, gm))


def test_the_effective_densities_give_the_derivative_of_the_relaxed_energy(
        model):
    h = model["h0"]
    g = two_body(model["L0"])
    _w, V = solve_scf(h, g)
    step = 1e-4
    numeric = (relaxed_energy(model, step, V)
               - relaxed_energy(model, -step, V)) / (2 * step)
    response = orr.response_densities(h, g, V, model["D"], model["G"],
                                      n_occ=NOCC, support=SUPPORT,
                                      frame=False)
    unrelaxed = directional(model, V, model["D"], model["G"])
    relaxed = directional(model, V, response.D, response.G)
    assert abs(unrelaxed - numeric) > 0.1                # the response matters
    assert relaxed == pytest.approx(numeric, abs=1e-6)
    assert response.cphf_residual < 1e-9
    assert response.orbital_gradient > 0.1


def test_a_unitary_change_of_the_atomic_orbital_frame_changes_nothing(model):
    rng = np.random.default_rng(11)
    h = model["h0"]
    g = two_body(model["L0"])
    _w, V = solve_scf(h, g)
    real = orr.response_densities(h, g, V, model["D"], model["G"],
                                  n_occ=NOCC, support=SUPPORT, frame=False)
    W = np.linalg.qr(rng.normal(size=(M, M))
                     + 1j * rng.normal(size=(M, M)))[0]
    hc = W @ h @ W.conj().T
    gc = np.einsum("ap,bq,cr,ds,pqrs->abcd", W, W, W.conj(), W.conj(), g,
                   optimize=True)
    cplx = orr.response_densities(hc, gc, W @ V, model["D"], model["G"],
                                  n_occ=NOCC, support=SUPPORT, frame=False)
    assert cplx.D == pytest.approx(real.D, abs=1e-12)
    assert cplx.G == pytest.approx(real.G, abs=1e-12)


def test_the_orbital_gradient_vanishes_for_the_hartree_fock_determinant(model):
    h = model["h0"]
    g = two_body(model["L0"])
    _w, V = solve_scf(h, g)
    D = np.zeros((M, M))
    D[0, 0] = D[1, 1] = 2.0
    G = (np.einsum("pr,qs->pqrs", D, D)
         - 0.5 * np.einsum("ps,qr->pqrs", D, D))
    response = orr.response_densities(h, g, V, D, G, n_occ=NOCC,
                                      support=[0, 1], frame=False)
    assert response.orbital_gradient < 1e-9
    assert response.D == pytest.approx(D, abs=1e-9)


def test_orbitals_that_are_not_canonical_are_refused(model):
    h = model["h0"]
    g = two_body(model["L0"])
    _w, V = solve_scf(h, g)
    mixed = V.copy()
    mixed[:, [0, 1]] = V[:, [0, 1]] @ np.array([[0.8, 0.6], [-0.6, 0.8]])
    # Rotating the occupied pair leaves the Fock matrix canonical only up to
    # the rotation; rotating an occupied into a virtual does not.
    mixed[:, [1, 2]] = V[:, [1, 2]] @ np.array([[0.9, 0.436], [-0.436, 0.9]])
    with pytest.raises(orr.OrbitalResponseUnavailable, match="canonical"):
        orr.response_densities(h, g, mixed, model["D"], model["G"],
                               n_occ=NOCC, support=SUPPORT, frame=False)


def test_a_state_that_depends_on_a_degenerate_pair_is_refused():
    # Two virtual orbitals of equal energy that interact only with each
    # other: the Fock matrix (the occupied orbitals') does not split them,
    # yet a state with density in both depends on their gauge.
    h = np.diag([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    L = np.zeros((1, M, M))
    L[0, 2:4, 2:4] = 0.3 * np.array([[0.2, 0.5], [0.5, -0.1]])
    g = two_body(L)
    _w, V = solve_scf(h, g)
    D = np.zeros((M, M))
    D[0, 0] = 2.0
    D[2, 2], D[3, 3], D[2, 3], D[3, 2] = 0.5, 0.5, 0.3, 0.3
    G = np.einsum("pr,qs->pqrs", D, D)
    with pytest.raises(orr.DegenerateOrbitalError):
        orr.response_densities(h, g, V, D, G, n_occ=NOCC,
                               support=[0, 2, 3], frame=False)


def test_nearly_degenerate_orbitals_that_carry_density_are_reported(model):
    h = np.diag([0.0, 1.0, 2.0, 2.0 + 1e-4, 3.0, 4.0])
    g = 0.01 * two_body(model["L0"])
    _w, V = solve_scf(h, g)
    response = orr.response_densities(h, g, V, model["D"], model["G"],
                                      n_occ=NOCC, support=SUPPORT,
                                      frame=False)
    assert [(p, q) for p, q, _gap in response.near_degenerate] == [(2, 3)]
    assert response.near_degenerate[0][2] < orr.NEAR_DEGENERACY
