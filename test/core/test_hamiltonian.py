# -*- coding: utf-8 -*-
# file: test_hamiltonian.py

import numpy as np
import pytest

from mandacaru.core.hamiltonian import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.integrals.packed import dense_tensor


R = 0.74  # H2 bond length, Angstrom


@pytest.fixture(scope="module")
def h2_integrals():
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, grid_spacing=0.18)
    basis = minimal_hao_basis(nuclei)
    return MolecularIntegrals(nuclei, basis, grid, units="angstrom")


class TestMolecularIntegrals:
    def test_orbital_count(self, h2_integrals):
        assert h2_integrals.n_orbitals == 2

    def test_nuclear_repulsion(self, h2_integrals):
        # Z_A Z_B / |R_A - R_B| with R = 0.74 A = 1.39838 bohr.
        bohr = R / 0.52917721067
        assert np.isclose(h2_integrals.nuclear_repulsion, 1.0 / bohr, rtol=1e-3)

    def test_one_body_hermitian_and_symmetric(self, h2_integrals):
        h = h2_integrals.one_body()
        assert np.allclose(h, h.conj().T, atol=1e-8)
        # homonuclear: the two diagonal (on-site) energies are equal
        assert np.isclose(h[0, 0], h[1, 1], atol=1e-3)

    def test_orthonormalized_basis_is_identity_overlap(self, h2_integrals):
        # after Lowdin orthogonalization the effective overlap is I; check the
        # raw overlap is a sensible symmetric matrix with unit-ish diagonal
        S = h2_integrals.overlap()
        assert np.allclose(S, S.conj().T, atol=1e-8)
        assert np.allclose(np.diag(S).real, 1.0, atol=5e-3)

    def test_two_body_symmetries(self, h2_integrals):
        eri = h2_integrals.two_body()
        # Real, physicists'-notation <ab|cd>: the full 8-fold permutation symmetry.
        assert np.abs(eri.imag).max() < 1e-9
        assert np.allclose(eri, eri.transpose(2, 1, 0, 3), atol=1e-8)  # e-1 bra swap
        assert np.allclose(eri, eri.transpose(0, 3, 2, 1), atol=1e-8)  # e-2 bra swap
        assert np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-8)  # <ab|cd>=<cd|ab>

    def test_coulomb_exchange_not_swapped(self, h2_integrals):
        # Regression guard for the physicist/chemist mix-up: the opposite-spin
        # two-electron determinant |0a 1b> must feel the Coulomb J=<01|01>, not
        # the exchange K=<00|11>.
        eri = h2_integrals.two_body().real
        h = h2_integrals.one_body().real
        J01, K01 = eri[0, 1, 0, 1], eri[0, 0, 1, 1]
        H = h2_integrals.molecular_hamiltonian(include_nuclear_repulsion=False)
        n = H.n_modes()
        mat = H.to_matrix().real

        def index(occ):
            b = 0
            for j in occ:
                b |= 1 << (n - 1 - j)
            return b

        # spin-orbital P = p + sigma*M, M=2 -> alpha {0,1}, beta {2,3}
        two_body_ab = mat[index([0, 3]), index([0, 3])] - (h[0, 0] + h[1, 1])
        two_body_aa = mat[index([0, 1]), index([0, 1])] - (h[0, 0] + h[1, 1])
        assert np.isclose(two_body_ab, J01, atol=1e-8)
        assert np.isclose(two_body_aa, J01 - K01, atol=1e-8)


class TestMolecularHamiltonian:
    def test_hamiltonian_is_hermitian(self, h2_integrals):
        H = h2_integrals.molecular_hamiltonian()
        assert H.n_modes() == 4
        assert H.map_to_qubits("jordan_wigner").is_hermitian(atol=1e-7)

    def test_all_mappings_share_ground_state(self, h2_integrals):
        H = h2_integrals.molecular_hamiltonian()
        ref = None
        for method in ("jordan_wigner", "parity", "bravyi_kitaev"):
            mat = H.map_to_qubits(method).to_matrix()
            mat = 0.5 * (mat + mat.conj().T)
            e0 = np.linalg.eigvalsh(mat).min()
            if ref is None:
                ref = e0
            assert np.isclose(e0, ref, atol=1e-7), method

    def test_two_qubit_reduction_matches_sector(self, h2_integrals):
        H = h2_integrals.molecular_hamiltonian()
        n = H.n_modes()
        # Ground state of the 2-electron sector by occupation-block restriction.
        full = _herm(H.to_matrix())
        idx = [i for i in range(2 ** n) if bin(i).count("1") == 2]
        sector_min = float(np.linalg.eigvalsh(full[np.ix_(idx, idx)]).min())

        red = H.map_to_qubits("parity_reduced", num_particles=(1, 1))
        assert red.num_qubits == 2
        rmat = _herm(red.to_matrix())
        assert np.isclose(np.linalg.eigvalsh(rmat).min(), sector_min, atol=1e-7)

    def test_nuclear_repulsion_shifts_spectrum(self, h2_integrals):
        H_with = h2_integrals.molecular_hamiltonian(include_nuclear_repulsion=True)
        H_without = h2_integrals.molecular_hamiltonian(include_nuclear_repulsion=False)
        e_with = np.linalg.eigvalsh(
            _herm(H_with.map_to_qubits("jordan_wigner").to_matrix())).min()
        e_without = np.linalg.eigvalsh(
            _herm(H_without.map_to_qubits("jordan_wigner").to_matrix())).min()
        assert np.isclose(e_with - e_without, h2_integrals.nuclear_repulsion,
                          atol=1e-7)


    def test_it_is_the_spin_blocked_tensor_term_by_term(self, h2_integrals):
        # Built from the spatial integrals (Fermion.from_spatial_integrals),
        # never from the (2M)^4 spin-orbital tensor -- and the same operator,
        # term for term and in the same order, as the one that tensor gave.
        from mandacaru.core.hamiltonian import (molecular_orbital_integrals,
                                                spin_block_integrals)
        from mandacaru.core.mapping import Fermion

        H = h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        h_mo, eri_mo, _ = molecular_orbital_integrals(h2_integrals, 2)
        constant = (h2_integrals.nuclear_repulsion
                    + h2_integrals.constant_energy)
        reference = (Fermion.from_integrals(*spin_block_integrals(h_mo, eri_mo))
                     + Fermion({(): complex(constant)}, n_modes=4))
        assert list(H.terms.items()) == list(reference.terms.items())


def _herm(m):
    return 0.5 * (m + m.conj().T)


class TestShortRangeTwoBody:
    """``short_range_two_body``: a screened hybrid's exchange tensor."""

    def test_without_screening_it_is_the_full_tensor(self, h2_integrals):
        assert np.allclose(h2_integrals.short_range_two_body(0.0),
                           h2_integrals.two_body(), atol=1e-13)

    def test_it_keeps_the_symmetries_and_shrinks_with_omega(self,
                                                            h2_integrals):
        full = h2_integrals.two_body().real
        previous = full
        for omega in (0.11, 0.5, 2.0):
            eri = h2_integrals.short_range_two_body(omega)
            assert np.abs(eri.imag).max() < 1e-9
            eri = eri.real
            assert np.allclose(eri, eri.transpose(2, 1, 0, 3), atol=1e-8)
            assert np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-8)
            diagonal = np.einsum("pppp->p", eri)
            assert np.all(diagonal > 0.0)
            assert np.all(diagonal < np.einsum("pppp->p", previous))
            previous = eri

    def test_the_long_range_part_is_omega_over_root_pi_at_short_range(
            self, h2_integrals):
        r"""For :math:`\omega` much smaller than the orbitals' inverse size,
        :math:`\operatorname{erf}(\omega r)/r \approx 2\omega/\sqrt\pi`, so
        two short-range tensors differ by :math:`2(\omega_2 -
        \omega_1)/\sqrt\pi\,\delta_{pr}\delta_{qs}` in an orthonormal
        basis."""
        w1, w2 = 0.005, 0.01
        long_range = (h2_integrals.short_range_two_body(w1)
                      - h2_integrals.short_range_two_body(w2)).real
        M = h2_integrals.n_orbitals
        expected = (2.0 * (w2 - w1) / np.sqrt(np.pi)
                    * np.einsum("pr,qs->pqrs", np.eye(M), np.eye(M)))
        assert np.allclose(long_range, expected, atol=2e-5)

    def test_negative_omega_is_refused(self, h2_integrals):
        with pytest.raises(ValueError, match="omega"):
            h2_integrals.short_range_two_body(-0.1)


def _molecule_integrals(name, basis, grid_spacing=0.35, vacuum=2.5):
    """Fresh integrals (nothing built yet) and the electron count."""
    import warnings

    from ase.build import molecule

    from mandacaru.algorithms._hamiltonian_from_atoms import \
        build_basis_hamiltonian

    atoms = molecule(name)
    atoms.center(vacuum=vacuum)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        context = build_basis_hamiltonian(atoms, basis, None, grid_spacing, 0, None,
                                          hamiltonian=False)[4]
    return context["integrals"], context["n_electrons"]


def _both_routes(name, basis, active_space, **options):
    """``[(H, active space, integrals), ...]`` for the tensor and the
    integral-direct route."""
    built = []
    for direct in (False, True):
        integrals, n_electrons = _molecule_integrals(name, basis, **options)
        H = integrals.molecular_hamiltonian(
            mo_basis=True, n_electrons=n_electrons,
            active_space=active_space, direct=direct)
        built.append((H, integrals.active_space, integrals))
    return built


class TestEarlyActiveSpace:
    """An active space built integral-direct: the same Hamiltonian as the
    tensor route, without the ``M^4`` tensor."""

    @pytest.mark.parametrize("name, basis, spec", [
        ("H2O", {"name": "PAW-LCAO", "size": "SZ"},
         {"frozen": 1, "orbitals": 4}),
        ("LiH", {"name": "PAW-LCAO", "size": "DZ"}, {"orbitals": 3}),
        # Closed-shell MP2 from the <ij|ab> block alone.
        ("LiH", {"name": "PAW-LCAO", "size": "DZ"},
         {"orbitals": 3, "method": "mp2"}),
    ])
    def test_the_active_hamiltonian_is_the_tensor_route(self, name, basis,
                                                         spec, monkeypatch):
        # The two Fock builds agree to 4e-15 at a common density, but near
        # convergence DIIS solves an ill-conditioned system that amplifies
        # that round-off, so at the default SCF tolerance (1e-9) the two
        # SCFs can take different last steps and stop 3.8e-10 apart in the
        # density (water).  Both are converged to 1e-12 here, so what is
        # compared is the routes, not where each SCF stopped.
        from mandacaru.algorithms.hartree_fock import RHF

        run = RHF.run
        monkeypatch.setattr(RHF, "run", lambda self, **kw: run(
            self, **{**kw, "tol": 1e-12}))
        (_tensor, tensor_space, dense), (_direct, direct_space, grid) = \
            _both_routes(name, basis, spec)
        self._check(dense, tensor_space, grid, direct_space, 1e-10)

    @pytest.mark.parametrize("method", ["energy", "natural"])
    def test_an_open_shell_builds_direct_on_request(self, method,
                                                    monkeypatch):
        # The direct UHF's natural orbitals.  UHF's soft modes leave the
        # coefficients apart by the SCF tolerance, not round-off; the
        # active energies agree to round-off.  Both SCFs are converged to
        # 1e-12, as in the closed-shell test, and given the iterations to get
        # there: with the exact one-center spheres (2026-10-12) OH's pi-hole
        # mode is softer, the default 300 iterations stop short of
        # convergence, and the two unconverged stopping points were 2.4e-7
        # apart in the active one-body block.
        from mandacaru.algorithms.hartree_fock import UHF
        from mandacaru.core.sector import ParticleSector

        solve = UHF.solve
        monkeypatch.setattr(UHF, "solve", lambda self, **kw: solve(
            self, **{**kw, "tol": 1e-12, "max_iter": 3000}))
        built = _both_routes("OH", {"name": "PAW-LCAO", "size": "SZ"},
                             {"orbitals": 4, "method": method})
        (_t, tensor_space, dense), (_d, direct_space, grid) = built
        assert dense._last_uhf_result[1].converged
        assert grid._last_direct_uhf_result[1].converged
        self._check(dense, tensor_space, grid, direct_space, 1e-7)
        energies = []
        for H, _space, _integrals in built:
            P = H.map_to_qubits("jordan_wigner")
            sector = ParticleSector(P.num_qubits, (4, 3))
            energies.append(np.linalg.eigvalsh(
                sector.restrict(P).toarray()).min())
        assert abs(energies[0] - energies[1]) < 1e-10

    @staticmethod
    def _check(dense, tensor_space, grid, direct_space, tolerance):
        assert not dense.integral_direct and grid.integral_direct
        assert grid._eri is None and grid._eri_ao is None
        assert direct_space.active == tensor_space.active
        assert direct_space.frozen == tensor_space.frozen
        # The same active space, each orbital up to the phase its
        # eigensolver chose, and a degenerate set (OH's pi pair) up to a
        # rotation inside it: the two SCFs differ by round-off, which is
        # what decides those.  Bring the direct integrals to the tensor
        # route's orbitals, C_direct = C_tensor U.
        active = list(tensor_space.active)
        U = (np.asarray(dense.mo_coefficients)[:, active].conj().T
             @ np.asarray(grid.mo_coefficients)[:, active])
        assert np.abs(U.conj().T @ U - np.eye(len(active))).max() < 1e-8
        h = U @ grid.active_one_body @ U.conj().T
        g = np.einsum("pa,qb,rc,sd,abcd->pqrs", U, U, U.conj(), U.conj(),
                      grid.active_two_body, optimize=True)
        assert np.abs(h - dense.active_one_body).max() < tolerance
        # The tensor route keeps its reduced integrals packed.
        assert np.abs(g - dense_tensor(dense.active_two_body)).max() \
            < tolerance
        assert abs(grid.active_constant - dense.active_constant) < tolerance

    def test_no_four_index_tensor_at_a_fixed_active_space(self):
        import tracemalloc

        # Ethane in DZP on a coarse grid: M = 56, so one complex M^4 tensor
        # is 157 MB while the sampled basis is 2 MB.
        integrals, n_electrons = _molecule_integrals(
            "C2H6", {"name": "PAW-LCAO", "size": "DZP"}, grid_spacing=0.45, vacuum=2.0)
        integrals.one_body()                       # samples the basis
        M = integrals.n_orbitals
        tracemalloc.start()
        try:
            H = integrals.molecular_hamiltonian(
                mo_basis=True, n_electrons=n_electrons,
                active_space={"orbitals": {"occupied": 2, "virtual": 2}},
                direct=True)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert H.n_modes() == 8 and integrals.integral_direct
        # The SCF's own working set (DIIS, the exchange's pair blocks, the
        # projections) is a few tens of MB; no M^4 array is in it.
        assert peak < 16 * M ** 4 / 2

    def test_the_default_route_follows_the_memory_limit(self, monkeypatch):
        from mandacaru.algorithms.active_space import resolve_active_space_spec
        from mandacaru.core import hamiltonian, matrix_free

        integrals, n_electrons = _molecule_integrals("LiH", "HAO")
        monkeypatch.setattr(matrix_free, "memory_limit_bytes",
                            lambda: 2 ** 40)
        spec = {"orbitals": 3}
        integrals.molecular_hamiltonian(mo_basis=True,
                                        n_electrons=n_electrons,
                                        active_space=spec)
        assert not integrals.integral_direct
        # A machine too small for the tensor route's three tensors.
        monkeypatch.setattr(
            matrix_free, "memory_limit_bytes",
            lambda: hamiltonian.dense_route_bytes(integrals.n_orbitals,
                                                  integrals.packs_two_body))
        integrals.molecular_hamiltonian(mo_basis=True,
                                        n_electrons=n_electrons,
                                        active_space=spec)
        assert integrals.integral_direct
        # An open shell keeps the tensor route (its coefficients would agree
        # only to the UHF tolerance).
        assert not integrals._integral_direct_route(
            resolve_active_space_spec(spec), [], n_electrons, (3, 1), None)
        # Without an active space nothing is cut, so the tensor is built.
        integrals.molecular_hamiltonian(mo_basis=True,
                                        n_electrons=n_electrons)
        assert not integrals.integral_direct


# --------------------------------------------------------------------------- #
# The two-body operator against a determinant-by-determinant oracle.
# --------------------------------------------------------------------------- #

def _ladder(operation, mode, determinant):
    """``a`` or ``a+`` on spin-orbital ``mode`` of a determinant bit mask:
    ``(new mask, fermionic sign)``, ``None`` when it annihilates."""
    occupied = (determinant >> mode) & 1
    if occupied != (operation == "a"):
        return None
    below = bin(determinant & ((1 << mode) - 1)).count("1")
    return determinant ^ (1 << mode), -1.0 if below % 2 else 1.0


def determinant_hamiltonian(h, chemist, n_alpha, n_beta, constant=0.0):
    r"""``(matrix, determinants)`` of
    :math:`\sum h_{pq} a^\dagger_p a_q + \tfrac12 \sum (pr|qs)\,
    a^\dagger_p a^\dagger_q a_s a_r` in the ``(n_alpha, n_beta)`` determinant
    basis, spin-orbital ``sigma * M + p``.

    Written from the chemists' integrals ``(pr|qs) = chemist[p, r, q, s]``
    alone, applying the operators to bit masks: it shares no code with
    ``Fermion``, the mappings or the sector.  ``h`` is ``(M, M)`` or
    ``(2, M, M)``, ``chemist`` ``(M,)*4`` or ``(2, 2, M, M, M, M)`` (spin
    resolved: electron 1 of spin ``sigma``, electron 2 of spin ``tau``).
    """
    from itertools import combinations

    h, chemist = np.asarray(h), np.asarray(chemist)
    M = h.shape[-1]
    spin_h = lambda s: h[s] if h.ndim == 3 else h
    spin_g = lambda s, t: chemist[s, t] if chemist.ndim == 6 else chemist
    determinants = [sum(1 << i for i in a) | sum(1 << (M + i) for i in b)
                    for a in combinations(range(M), n_alpha)
                    for b in combinations(range(M), n_beta)]
    position = {d: i for i, d in enumerate(determinants)}
    matrix = np.zeros((len(determinants),) * 2, dtype=complex)
    for column, det in enumerate(determinants):
        matrix[column, column] += constant
        for s in range(2):
            for p, q in zip(*np.nonzero(spin_h(s))):
                first = _ladder("a", s * M + q, det)
                second = first and _ladder("c", s * M + p, first[0])
                if second:
                    matrix[position[second[0]], column] += \
                        spin_h(s)[p, q] * first[1] * second[1]
        for s in range(2):
            for t in range(2):
                for p, r, q, u in zip(*np.nonzero(spin_g(s, t))):
                    chain, sign = det, 1.0
                    for operation, mode in (("a", s * M + r), ("a", t * M + u),
                                            ("c", t * M + q), ("c", s * M + p)):
                        step = _ladder(operation, mode, chain)
                        if step is None:
                            break
                        chain, factor = step
                        sign *= factor
                    else:
                        matrix[position[chain], column] += \
                            0.5 * sign * spin_g(s, t)[p, r, q, u]
    return matrix, determinants


def _chemist_problem(M, complex_orbitals, seed, factors=4, spin_resolved=False):
    """Random Hermitian ``h`` and a sum of squares of Hermitian matrices
    ``(pr|qs) = sum_k L_k[p, r] L_k[q, s]``: real (8-fold symmetric) or
    complex (4-fold), as the integrals of real and complex orbitals are."""
    rng = np.random.default_rng(seed)

    def hermitian():
        a = rng.standard_normal((M, M))
        if complex_orbitals:
            a = a + 1j * rng.standard_normal((M, M))
        return (a + a.conj().T) / 2

    def one_set():
        g = sum(np.einsum("pr,qs->prqs", L, L)
                for L in (hermitian() for _ in range(factors)))
        return hermitian(), g

    if not spin_resolved:
        return one_set()
    blocks = [[one_set() for _ in range(2)] for _ in range(2)]
    h = np.stack([blocks[0][0][0], blocks[1][1][0]])
    g = np.stack([np.stack([blocks[s][t][1] for t in range(2)])
                  for s in range(2)])
    return h, g


class TestAgainstADeterminantOracle:
    """The qubit Hamiltonian's sector spectrum is the determinant FCI's.

    A wrong index order (chemists' for physicists'), a lost factor of 1/2, a
    swapped spin block or a missing symmetrization changes the spectrum of
    these random problems at once; they have no structure a lucky convention
    could hide behind.
    """

    @pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                         "bravyi_kitaev", "parity_reduced"])
    @pytest.mark.parametrize("complex_orbitals", [False, True])
    @pytest.mark.parametrize("M, n_alpha, n_beta", [(3, 1, 1), (4, 2, 1)])
    def test_the_sector_spectrum_is_the_determinant_spectrum(
            self, M, n_alpha, n_beta, complex_orbitals, mapping):
        from mandacaru.core.mapping import Fermion
        from mandacaru.core.sector import ParticleSector

        h, chemist = _chemist_problem(M, complex_orbitals, seed=7)
        operator = Fermion.from_spatial_integrals(
            h, chemist.transpose(0, 2, 1, 3))
        operator = operator + Fermion({(): 0.37 + 0j}, n_modes=2 * M)
        oracle, _ = determinant_hamiltonian(h, chemist, n_alpha, n_beta, 0.37)
        assert np.abs(oracle - oracle.conj().T).max() < 1e-13

        qubit = operator.map_to_qubits(mapping, n_modes=2 * M,
                                       num_particles=(n_alpha, n_beta))
        # Hermitian as an operator, conserving N and S_z: nothing leaves.
        assert max(abs(c.imag) for c in qubit.terms.values()) < 1e-12
        sector = ParticleSector(qubit.num_qubits, (n_alpha, n_beta), mapping)
        assert sector.conserves(qubit)
        matrix = sector.restrict(qubit).toarray()
        assert np.abs(matrix - matrix.conj().T).max() < 1e-12
        assert np.allclose(np.linalg.eigvalsh(matrix),
                           np.linalg.eigvalsh(oracle), atol=1e-11)

    def test_spin_resolved_integrals_follow_their_spin_blocks(self):
        from mandacaru.core.mapping import Fermion
        from mandacaru.core.sector import ParticleSector

        h, chemist = _chemist_problem(3, True, seed=9, spin_resolved=True)
        physicist = chemist.transpose(0, 1, 2, 4, 3, 5)
        oracle, _ = determinant_hamiltonian(h, chemist, 2, 1)
        qubit = Fermion.from_spatial_integrals(h, physicist).map_to_qubits(
            "jordan_wigner", n_modes=6)
        sector = ParticleSector(6, (2, 1), "jordan_wigner")
        assert sector.conserves(qubit)
        assert np.allclose(
            np.linalg.eigvalsh(sector.restrict(qubit).toarray()),
            np.linalg.eigvalsh(oracle), atol=1e-11)

    @pytest.fixture(scope="class")
    def molecules(self):
        return {}

    @pytest.mark.parametrize("name, frozen", [
        ("H2", None), ("LiH", None), ("LiH", 1), ("H2O", 1)])
    def test_molecular_hamiltonians_equal_the_independent_active_problem(
            self, name, frozen, molecules):
        """The pipeline's operator (AO tensor -> RHF orbitals -> frozen core
        -> spin blocks -> Jordan-Wigner) against the active problem written
        here from the atomic-orbital integrals: the transform by the
        orbitals, the frozen-core potential
        ``h_pq + sum_c [2 (pq|cc) - (pc|cq)]``, the core energy and the
        nuclear repulsion.  Also the Hartree-Fock energy of the SCF."""
        from mandacaru.core.sector import ParticleSector

        if name not in molecules:
            molecules[name] = _molecule_integrals(
                name, {"name": "PAW-LCAO", "size": "SZ"})
        integrals, n_electrons = molecules[name]
        spec = None if frozen is None else {"frozen": frozen}
        operator = integrals.molecular_hamiltonian(
            mo_basis=True, n_electrons=n_electrons, active_space=spec)

        C = np.asarray(integrals.mo_coefficients)
        h_mo = C.conj().T @ integrals.one_body() @ C
        g_mo = np.einsum("pi,qj,rk,sl,pqrs->ijkl", C.conj(), C.conj(), C, C,
                         integrals.two_body(), optimize=True)
        chemist = g_mo.transpose(0, 2, 1, 3)               # (pr|qs)
        core, active = range(frozen or 0), range(frozen or 0, h_mo.shape[0])
        h_eff = h_mo + sum(2 * chemist[:, :, c, c] - chemist[:, c, c, :]
                           for c in core)
        e_core = (sum(2 * h_mo[c, c] for c in core)
                  + sum(2 * chemist[c, c, d, d] - chemist[c, d, d, c]
                        for c in core for d in core))
        constant = e_core + integrals.nuclear_repulsion \
            + integrals.constant_energy
        ix = np.ix_(active, active)
        n_alpha = n_electrons // 2 - len(core)
        oracle, _ = determinant_hamiltonian(
            h_eff[ix], chemist[np.ix_(active, active, active, active)],
            n_alpha, n_alpha, constant)

        # The SCF's own energy is the reference determinant's.
        assert oracle[0, 0].real == pytest.approx(
            integrals.hartree_fock(n_electrons).electronic_energy
            + integrals.nuclear_repulsion + integrals.constant_energy,
            abs=1e-8)

        qubit = operator.map_to_qubits("jordan_wigner")
        sector = ParticleSector(qubit.num_qubits, (n_alpha, n_alpha),
                                "jordan_wigner")
        assert sector.conserves(qubit)
        assert np.allclose(
            np.linalg.eigvalsh(sector.restrict(qubit).toarray())[:4],
            np.linalg.eigvalsh(oracle)[:4], atol=1e-9)
