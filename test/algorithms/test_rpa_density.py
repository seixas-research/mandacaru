# -*- coding: utf-8 -*-
# file: test/algorithms/test_rpa_density.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The direct-RPA one-particle density matrix: the ring amplitudes against
an independent spin-orbital solution, the crystal's identities (trace,
occupations, three correlation-energy formulas, the static polarizability
of the screened interaction) and the natural orbitals that drive a
Wannierization."""

import types

import numpy as np
import pytest
from ase.build import bulk

from mandacaru import Mandacaru
from mandacaru.algorithms import rpa_density as rpa
from mandacaru.algorithms.wannier import polarizability
from mandacaru.integrals import reciprocal as rc


def model(seed=1, o=3, v=5, n_aux=7, scale=0.15):
    """Orbital energies and real symmetric Coulomb factors of a closed
    shell."""
    rng = np.random.default_rng(seed)
    e = np.sort(np.concatenate([rng.uniform(-1.5, -0.5, o),
                                rng.uniform(0.2, 2.0, v)]))
    L = rng.normal(size=(o + v, o + v, n_aux)) * scale
    return e, o, 0.5 * (L + L.transpose(1, 0, 2))


def spin_orbital_ring(e, o, L):
    """Direct ring CCD in spin orbitals, solved independently (a Sylvester
    iteration of the Riccati equation), and its density contracted pair by
    pair: ``(gamma spin summed, E_c)``."""
    from scipy.linalg import solve_sylvester

    M = len(e)
    pairs = [(i, a, s) for s in (0, 1) for i in range(o)
             for a in range(o, M)]
    Lp = np.array([L[i, a] for i, a, _s in pairs])
    K = Lp @ Lp.T                   # alpha and beta excitations couple too
    D = np.array([e[a] - e[i] for i, a, _s in pairs])
    A = np.diag(D) + K
    t = np.zeros_like(K)
    for _ in range(200):
        new = solve_sylvester(A, A, -(K + t @ K @ t))
        new = 0.5 * (new + new.T)
        if np.abs(new - t).max() < 1e-14:
            break
        t = new
    gamma = np.zeros((M, M))
    gamma[np.arange(o), np.arange(o)] = 2.0
    W = t @ t.T
    for x, (i, a, s) in enumerate(pairs):
        for y, (j, b, u) in enumerate(pairs):
            if (i, s) == (j, u):
                gamma[a, b] += W[x, y]
            if (a, s) == (b, u):
                gamma[j, i] -= W[x, y]
    return gamma, 0.5 * np.sum(K * t)


class TestTheRingAmplitudes:
    def test_the_density_is_the_spin_orbital_one(self):
        e, o, L = model()
        gamma, energy, plasmon = rpa.molecular_ring_density(e, o, L)
        reference, reference_energy = spin_orbital_ring(e, o, L)
        assert np.abs(gamma - reference).max() < 1e-12
        assert energy == pytest.approx(reference_energy, abs=1e-12)
        assert plasmon == pytest.approx(energy, abs=1e-12)

    def test_the_trace_and_the_occupations(self):
        e, o, L = model(seed=4, scale=0.3)
        gamma, _energy, _plasmon = rpa.molecular_ring_density(e, o, L)
        assert np.allclose(gamma, gamma.conj().T, atol=1e-14)
        assert np.trace(gamma).real == pytest.approx(2 * o, abs=1e-12)
        n = np.linalg.eigvalsh(gamma)
        assert n.min() > 0.0 and n.max() < 2.0

    def test_weak_coupling_is_direct_mp2(self):
        """At second order the energy is the direct (Coulomb) part of
        MP2."""
        e, o, L = model(seed=2)
        L = 1e-3 * L
        _gamma, energy, _plasmon = rpa.molecular_ring_density(e, o, L)
        Lov = L[:o, o:]
        K = np.einsum("iaQ,jbQ->iajb", Lov, Lov)
        gap = e[o:][None, :] - e[:o][:, None]
        direct = -2.0 * np.sum(K ** 2 / (gap[:, :, None, None]
                                         + gap[None, None, :, :]))
        assert energy == pytest.approx(direct, rel=1e-5)

    def test_a_block_and_its_partner_share_the_amplitudes(self):
        """Two momenta: the amplitudes of -q are the transpose of q's, and
        both energy formulas agree."""
        rng = np.random.default_rng(5)
        n, m, g = 6, 4, 9
        L = 0.2 * (rng.normal(size=(n, g)) + 1j * rng.normal(size=(n, g)))
        Lp = 0.2 * (rng.normal(size=(m, g)) + 1j * rng.normal(size=(m, g)))
        d, dp = rng.uniform(0.3, 1.5, n), rng.uniform(0.3, 1.5, m)
        forward = rpa.ring_amplitudes(d, L, dp, Lp)
        backward = rpa.ring_amplitudes(dp, Lp, d, L)
        assert np.allclose(backward.amplitudes, forward.amplitudes.T,
                           atol=1e-12)
        assert forward.energy == pytest.approx(backward.energy, abs=1e-12)
        assert forward.plasmon + forward.plasmon_partner == pytest.approx(
            2.0 * forward.energy, abs=1e-12)
        assert np.allclose(forward.partner_omega, backward.omega,
                           atol=1e-12)


@pytest.fixture(scope="module")
def silicon():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = Mandacaru(method="dft", xc="lda", h=0.35, trace=False,
                           basis={"name": "PAW-LCAO", "size": "SZP"},
                           kpts={"size": (2, 2, 2), "gamma": True},
                           smearing={"method": "fermi-dirac",
                                     "width": 0.001})
    atoms.get_potential_energy()
    return atoms, atoms.calc.natural_orbitals(method="rpa")


class TestTheCrystal:
    def test_the_trace_is_the_electron_count(self, silicon):
        _atoms, nos = silicon
        assert nos.electron_count == pytest.approx(8.0, abs=1e-10)
        assert nos.occupations.min() > 0.0
        assert nos.occupations.max() < 2.0
        assert 0.0 < nos.idempotency < 1.0

    def test_the_natural_orbitals_are_unitary(self, silicon):
        _atoms, nos = silicon
        for U in nos.vectors:
            assert np.allclose(U.conj().T @ U, np.eye(len(U)), atol=1e-12)

    def test_three_formulas_give_one_correlation_energy(self, silicon):
        """The iterated amplitudes are the eigenproblem's, whose
        excitation energies give the plasmon formula; the
        imaginary-frequency polarizability gives the same energy without
        amplitudes."""
        from mandacaru.units import HARTREE_TO_EV

        atoms, nos = silicon
        solver = atoms.calc.solver._periodic_solver
        fermi = float(atoms.calc.solver._scf.fermi_level)
        exact = rpa.crystal_rpa(solver, nos.size, fermi,
                                solver_method="eigen")
        assert nos.plasmon_energy is None
        assert np.abs(exact.occupations - nos.occupations).max() < 1e-9
        assert exact.correlation_energy == pytest.approx(
            nos.correlation_energy, abs=1e-9)
        assert exact.plasmon_energy == pytest.approx(
            exact.correlation_energy, abs=1e-8)
        fractional, states = rpa.gather_states(solver, nos.size, nos.bands,
                                               fermi)
        blocks = rpa.momentum_blocks(solver.crystal, fractional, states,
                                     nos.size, nos.cutoff)
        partners = rpa.partner_columns(blocks, nos.size)
        chi = rpa.chi_correlation_energy(blocks, partners) / len(
            fractional)
        assert nos.correlation_energy < 0.0
        assert chi * HARTREE_TO_EV == pytest.approx(nos.correlation_energy,
                                                    abs=1e-6)

    def test_the_pairs_are_those_of_the_screened_interaction(self, silicon):
        """At zero frequency the blocks' polarizability is the cRPA
        code's, which builds its transition densities independently."""
        atoms, nos = silicon
        solver = atoms.calc.solver._periodic_solver
        crystal = solver.crystal
        fermi = float(atoms.calc.solver._scf.fermi_level)
        fractional, states = rpa.gather_states(solver, nos.size, nos.bands,
                                               fermi)
        blocks = rpa.momentum_blocks(crystal, fractional, states, nos.size,
                                     nos.cutoff)
        partners = rpa.partner_columns(blocks, nos.size)
        size = np.asarray(nos.size)
        B = rc.reciprocal_vectors(crystal.lattice)
        block = blocks[3]
        index, columns = partners[3]
        Lp = blocks[index].L[:, columns]
        Pi = (2.0 * (block.L.T / -block.delta[None, :]) @ np.conj(block.L)
              - 2.0 * (Lp.conj().T / blocks[index].delta[None, :]) @ Lp)
        q = B @ (block.q_cell / size)
        qG = q[:, None] + B @ block.g
        root = np.sqrt(4.0 * np.pi / np.sum(qG * qG, axis=0))
        mine = Pi / root[:, None] / root[None, :]
        channel = types.SimpleNamespace(_solver=solver, _fermi_level=fermi)
        cells = np.indices(nos.size).reshape(3, -1).T
        partner = np.ravel_multi_index(tuple(((cells + block.q_cell)
                                              % size).T), nos.size)
        weighted = [s + (np.zeros(len(s[2])),) for s in states]
        reference, _qG = polarizability(
            crystal, [(channel, [(weighted[k], weighted[k2])
                                 for k, k2 in enumerate(partner)])],
            q, block.g, len(fractional))
        assert np.abs(mine - reference).max() < 1e-10 * np.abs(
            reference).max()

    def test_the_valence_is_depleted_into_the_antibonding_orbitals(
            self, silicon):
        """An insulator: the occupied natural orbitals are the valence
        bands' span, and the most occupied empty ones are sp3-like."""
        atoms, nos = silicon
        strong = nos.occupations[:, :4]
        weak = nos.occupations[:, 4:]
        assert strong.min() > 1.8 and weak.max() < 0.2
        U = nos.vectors
        assert np.allclose(np.sum(np.abs(U[:, :4, :4]) ** 2, axis=1), 1.0,
                           atol=1e-10)
        # The eight most correlated orbitals are the sp3 bonding and
        # antibonding ones: their projectability onto the sp3 targets beats
        # every other orbital's, at every k-point.
        from mandacaru.algorithms.band_selection import (projectability,
                                                         target_vectors)
        from mandacaru.algorithms.wannier import mesh_states, resolve_guess

        solver = atoms.calc.solver._periodic_solver
        targets = target_vectors(solver.crystal,
                                 resolve_guess("sp3", atoms, solver.crystal))
        _f, data, vectors, _e = mesh_states(solver, nos.size, nos.bands)
        chosen = nos.selected(8)
        for d, v, Uk, top in zip(data, vectors, U, chosen):
            p = projectability([d], [v @ Uk], targets)[0]
            rest = np.setdiff1d(np.arange(len(p)), top)
            assert p[top].min() > 0.8 > 0.2 > p[rest].max()

    def test_the_summary_and_the_citations(self, silicon):
        atoms, nos = silicon
        text = nos.summary()
        assert "strongly occupied" in text and "weakly occupied" in text
        assert {"Scuseria2008", "Furche2008"} <= set(
            atoms.calc.citation_keys())

    def test_a_spin_polarized_crystal_is_refused(self):
        solver = types.SimpleNamespace(n_spins=2)
        with pytest.raises(NotImplementedError):
            rpa.crystal_rpa(solver, (1, 1, 1), 0.0)

    def test_unknown_methods_are_refused(self, silicon):
        atoms, _nos = silicon
        with pytest.raises(ValueError, match="rpa"):
            atoms.calc.natural_orbitals(method="mp2")
