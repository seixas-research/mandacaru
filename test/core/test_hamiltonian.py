# -*- coding: utf-8 -*-
# file: test_hamiltonian.py

import numpy as np
import pytest

from mandacaru.core.hamiltonian import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid


R = 0.74  # H2 bond length, Angstrom


@pytest.fixture(scope="module")
def h2_integrals():
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.18)
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


def _molecule_integrals(name, basis, h=0.35, vacuum=2.5):
    """Fresh integrals (nothing built yet) and the electron count."""
    import warnings

    from ase.build import molecule

    from mandacaru.algorithms._hamiltonian_from_atoms import \
        build_basis_hamiltonian

    atoms = molecule(name)
    atoms.center(vacuum=vacuum)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        context = build_basis_hamiltonian(atoms, basis, None, h, 0, None,
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
    def test_an_open_shell_builds_direct_on_request(self, method):
        # The direct UHF's natural orbitals.  UHF's soft modes leave the
        # coefficients apart by the SCF tolerance, not round-off; the
        # active energies agree to round-off.
        from mandacaru.core.sector import ParticleSector

        built = _both_routes("OH", {"name": "PAW-LCAO", "size": "SZ"},
                             {"orbitals": 4, "method": method})
        (_t, tensor_space, dense), (_d, direct_space, grid) = built
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
        assert np.abs(g - dense.active_two_body).max() < tolerance
        assert abs(grid.active_constant - dense.active_constant) < tolerance

    def test_no_four_index_tensor_at_a_fixed_active_space(self):
        import tracemalloc

        # Ethane in DZP on a coarse grid: M = 56, so one complex M^4 tensor
        # is 157 MB while the sampled basis is 2 MB.
        integrals, n_electrons = _molecule_integrals(
            "C2H6", {"name": "PAW-LCAO", "size": "DZP"}, h=0.45, vacuum=2.0)
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
        # A machine too small for the tensor route's three M^4 arrays.
        monkeypatch.setattr(
            matrix_free, "memory_limit_bytes",
            lambda: hamiltonian.dense_route_bytes(integrals.n_orbitals))
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
