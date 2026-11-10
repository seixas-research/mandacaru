"""The quantum methods on a crystal through ``Mandacaru``: supercell, refusals, k-resolved views.

``vqe``, ``adapt-vqe``, ``vasqa`` and ``valsqa`` on a structure with a periodic axis solve the
Born-von Karman supercell of a Gamma-centered ``kpts`` mesh with the lattice-summed Hamiltonian
built in Rust
(``molecular::periodic``) and report the energy per primitive cell. The unfolded mean-field
bands, the Lehmann spectral function and the tetrahedron density of states are native too
(their numerical gates are in ``crates/mandacaru-core/tests/{molecular_periodic,spectral,
tetrahedron}.rs``); here they are checked through the calculator, on a two-cell hydrogen chain
(4 qubits, well under a second per calculation).
"""

import functools
import json
from functools import reduce

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

SLSQP = {"method": "SLSQP", "maxiter": 1000, "tol": 1e-12}
GAMMA2 = {"size": (2, 1, 1), "gamma": True}
#: Every crystal quantum run names its orbitals until crystal_orbitals="k-space" exists.
REAL = {"crystal_orbitals": "real-space"}
PAULIS = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
          "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}


def chain(a=2.0):
    """One H per primitive cell, 10 Angstrom of vacuum transversely."""
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.diag([a, 10.0, 10.0]),
                 pbc=[True, False, False])


@functools.lru_cache(maxsize=None)
def solved(method="vqe", size=(2, 1, 1), seed=None):
    atoms = chain()
    extra = {} if seed is None else {"seed": seed}
    atoms.calc = Mandacaru(method=method, basis="HAO", grid_spacing=0.35,
                           mapping="jordan_wigner", kpts={"size": size, "gamma": True},
                           optimizer=SLSQP, trace=False, **REAL, **extra)
    atoms.get_potential_energy()
    return atoms


def exact_per_cell(calc, electrons, cells):
    """Exact ground state of the run's qubit Hamiltonian in its electron-number sector."""
    n = calc.problem.num_qubits
    h = sum((re + 1j * im) * reduce(np.kron, [PAULIS[c] for c in label])
            for label, re, im in calc.problem.hamiltonian_terms)
    sector = [i for i in range(2 ** n) if bin(i).count("1") == electrons]
    return np.linalg.eigvalsh(h[np.ix_(sector, sector)])[0] * HARTREE_TO_EV / cells


class TestTheSupercellHamiltonian:
    def test_the_grid_is_periodic_and_commensurate(self):
        calc = solved().calc
        grid = calc.integrals.grid
        assert grid.periodic
        # round(2.0 / 0.35) = 6 nodes per primitive cell, two cells.
        assert grid.shape[0] == 12

    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
    def test_the_two_cell_energy_is_pinned(self, method):
        """Measured 2026-11-09: -12.422242881249 eV per cell for both solvers.

        The old periodic engine (removed in the Rust cutover) pinned -12.4222428989 on this
        case; the native supercell reproduces it to 2e-8 eV (`test/migration`).
        """
        energy = solved(method).get_potential_energy()
        assert energy == pytest.approx(-12.422242881249, abs=1e-7)

    def test_the_variational_energy_is_the_exact_ground_state_at_four_qubits(self):
        calc = solved().calc
        assert calc.results["energy"] == pytest.approx(exact_per_cell(calc, 2, 2), abs=1e-8)

    def test_it_is_not_the_isolated_cluster(self):
        """The lattice sums differ from the box: the cluster gave -12.0319 eV per cell."""
        assert abs(solved().get_potential_energy() - -12.031915584357828) > 0.3


class TestTheOptionChecks:
    @pytest.mark.parametrize("name", ["bloch-vqe", "bloch-adapt-vqe", "bloch-vasqa",
                                      "bloch-valsqa"])
    def test_a_bloch_name_is_not_a_method_and_names_the_plain_one(self, name):
        with pytest.raises(ValueError, match=f"use method='{name[6:]}'"):
            Mandacaru(method=name)

    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe", "vasqa", "valsqa"])
    def test_a_crystal_without_kpts_runs_at_gamma(self, method):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=np.diag([2.0, 6, 6]),
                      pbc=[True, False, False])
        extra = {"seed": 0} if method in ("vasqa", "valsqa") else {}
        atoms.calc = Mandacaru(method=method, basis="HAO", grid_spacing=0.4, trace=False,
                               **REAL, **extra)
        atoms.get_potential_energy()
        assert np.allclose(atoms.calc.kpoints, [[0, 0, 0]])
        assert atoms.calc.integrals.grid.periodic

    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe", "vasqa", "valsqa"])
    def test_k_space_orbitals_are_not_implemented_yet(self, method):
        atoms = chain()
        atoms.calc = Mandacaru(method=method, basis="HAO", grid_spacing=0.35, trace=False)
        assert atoms.calc._crystal_orbitals == "k-space"
        with pytest.raises(NotImplementedError, match="crystal_orbitals='real-space'"):
            atoms.get_potential_energy()

    def test_an_unknown_crystal_orbitals_value_raises(self):
        with pytest.raises(ValueError, match="crystal_orbitals"):
            Mandacaru(method="vqe", crystal_orbitals="momentum")

    def test_crystal_orbitals_is_refused_on_a_molecule(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6, 6, 6])
        atoms.calc = Mandacaru(method="vqe", basis="HAO", trace=False, **REAL)
        with pytest.raises(ValueError, match="crystal_orbitals applies to a crystal"):
            atoms.get_potential_energy()

    @pytest.mark.parametrize("method", ["rhf", "dft", "subspace-vqe"])
    def test_crystal_orbitals_is_refused_by_a_method_without_it(self, method):
        with pytest.raises(ValueError, match="does not use"):
            Mandacaru(method=method, **REAL)

    def test_a_shifted_mesh_is_refused(self):
        atoms = chain()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.35, kpts=(2, 1, 1),
                               trace=False, **REAL)
        with pytest.raises(ValueError, match="Gamma"):
            atoms.get_potential_energy()

    def test_an_odd_monkhorst_pack_mesh_contains_gamma(self):
        atoms = chain()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.35, kpts=(3, 1, 1),
                               trace=False, **REAL)
        atoms.get_potential_energy()
        assert len(atoms.calc.kpoints) == 3

    @pytest.mark.parametrize("options, error, match", [
        ({"basis": {"name": "PAW-LCAO", "size": "SZ"}}, NotImplementedError, "PAW-LCAO"),
        ({"kinetic": "fd"}, ValueError, "spectral"),
        ({"precision": "float32"}, ValueError, "float64"),
    ])
    def test_unsupported_supercell_options_raise(self, options, error, match):
        atoms = chain()
        atoms.calc = Mandacaru(method="vqe", kpts=GAMMA2, trace=False, **REAL,
                               **{"basis": "HAO", **options})
        with pytest.raises(error, match=match):
            atoms.get_potential_energy()

    def test_a_mesh_along_the_vacuum_is_refused(self):
        atoms = chain()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.35,
                               kpts={"size": (2, 2, 1), "gamma": True}, trace=False, **REAL)
        with pytest.raises(ValueError, match="not periodic"):
            atoms.get_potential_energy()

    def test_a_charged_cell_is_refused(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=np.diag([2.0, 6, 6]),
                      pbc=[True, False, False])
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, charge=1,
                               trace=False, **REAL)
        with pytest.raises(ValueError, match="neutral"):
            atoms.get_potential_energy()

    @pytest.mark.parametrize("method", ["subspace-vqe", "subspace-adapt-vqe"])
    def test_a_method_without_a_periodic_form_is_refused_on_a_crystal(self, method):
        atoms = chain()
        atoms.calc = Mandacaru(method=method, basis="HAO", grid_spacing=0.35, trace=False)
        with pytest.raises(ValueError, match="no periodic form"):
            atoms.get_potential_energy()


class TestTheArchitectureSearches:
    """VASQA and VALSQA on the same supercell: the Markov chain over ansatz architectures, and
    the chain with a learned proposal. Measured 2026-11-10 on seeds 0-4: every run of both
    methods reaches the exact ground state of the 2-cell (4 qubits) and the 3-cell (6 qubits)
    chain; at 4 cells (8 qubits, fermionic pool) all ten runs give -11.26294664 eV per cell, a
    spread of 1.8e-15 eV."""

    @pytest.mark.parametrize("method", ["vasqa", "valsqa"])
    @pytest.mark.parametrize("seed", [0, 3])
    def test_the_two_cell_chain_reaches_the_exact_ground_state(self, method, seed):
        calc = solved(method, seed=seed).calc
        assert calc.results["energy"] == pytest.approx(exact_per_cell(calc, 2, 2), abs=1e-8)
        assert calc.results["energy"] == pytest.approx(-12.422242881249, abs=1e-7)

    @pytest.mark.parametrize("method", ["vasqa", "valsqa"])
    def test_the_three_cell_chain_reaches_the_exact_ground_state(self, method):
        calc = solved(method, size=(3, 1, 1), seed=1).calc
        assert calc.results["energy"] == pytest.approx(exact_per_cell(calc, 3, 3), abs=1e-8)

    def test_the_spectral_function_reads_the_searched_state(self):
        spectral = solved("vasqa", seed=3).calc.get_spectral_function()
        assert np.allclose(spectral.sum_rule, 1.0, atol=1e-12)
        assert spectral.gap > 0

    def test_valsqa_without_a_trained_model_says_so(self, monkeypatch):
        monkeypatch.delenv("MANDACARU_PROPOSAL_DATA", raising=False)
        result = solved("valsqa", seed=0).calc.result
        assert not result.proposal_ready
        assert "no trained proposal model" in result.proposal_reason

    def test_valsqa_reads_the_trained_model_of_its_store(self, tmp_path, monkeypatch):
        from mandacaru.analysis.proposals import ProposalModel, fit

        store = tmp_path / "store"
        model = fit(store / "empty", fractions=(1.,), resources={"threads": 1})
        value = json.loads(model.native_json)
        value["ready"] = True
        value["mixing"] = .3
        ProposalModel(json.dumps(value)).save(store / "proposal_model.npz")
        monkeypatch.setenv("MANDACARU_PROPOSAL_DATA", str(store))
        atoms = chain()
        atoms.calc = Mandacaru(method="valsqa", basis="HAO", grid_spacing=0.35, seed=2,
                               mapping="jordan_wigner", kpts=GAMMA2, optimizer=SLSQP,
                               record=False, trace=False, **REAL)
        energy = atoms.get_potential_energy()
        assert atoms.calc.result.proposal_ready
        assert energy == pytest.approx(exact_per_cell(atoms.calc, 2, 2), abs=1e-8)


class TestTheUnfoldedMeanField:
    def test_the_mesh_points_are_gamma_and_the_zone_boundary(self):
        k = solved().calc.kpoints
        assert np.allclose(k, [[0, 0, 0], [0.5, 0, 0]])
        assert np.allclose(solved().calc.get_ibz_k_points(), k)

    def test_the_band_at_the_mesh_points_is_the_supercell_spectrum(self):
        """Born-von Karman identity through the calculator."""
        calc = solved().calc
        band = calc.band_structure(path="GX", npoints=5)
        assert band.energies.shape == (1, 5, 1)
        levels = np.sort(calc.native_reference.mo_energies) * HARTREE_TO_EV
        assert band.energies[0, 0, 0] == pytest.approx(levels[0], abs=1e-8)
        assert band.energies[0, -1, 0] == pytest.approx(levels[1], abs=1e-8)
        # One electron per cell: the Fermi level is the k = 0 level.
        assert calc.get_fermi_level() == pytest.approx(levels[0], abs=1e-8)

    def test_the_tetrahedron_density_holds_the_band_and_the_projection_adds_up(self):
        calc = solved().calc
        axis = np.linspace(-15.0, 15.0, 30001)
        energies, dos = calc.dos(energies=axis, kpts=(24, 1, 1), integration="tetrahedron")
        # One band per cell, two spins: the density integrates to 2. The linear interpolation
        # of a 1-D band has steps at the mesh energies, which the trapezoid rule on a 1 meV axis
        # resolves to 7e-4 (measured).
        assert np.trapezoid(dos, energies) == pytest.approx(2.0, abs=2e-3)
        _, pdos = calc.pdos(energies=axis, kpts=(24, 1, 1), integration="tetrahedron")
        assert list(pdos) == [(0, 0)]
        assert np.allclose(sum(pdos.values()), dos, atol=1e-10)

    def test_a_gaussian_pdos_of_unfolded_bands_is_refused(self):
        with pytest.raises(NotImplementedError, match="tetrahedron"):
            solved().calc.pdos()


class TestTheSpectralFunction:
    @pytest.fixture(scope="class")
    def spectral(self):
        return solved().calc.get_spectral_function(eta=0.1)

    def test_the_sum_rule_holds_for_every_channel(self, spectral):
        assert spectral.sum_rule.shape == (2, 1, 2)
        assert np.allclose(spectral.sum_rule, 1.0, atol=1e-12)

    def test_its_ground_energy_is_the_variational_energy(self, spectral):
        calc = solved().calc
        assert spectral.ground_energy == pytest.approx(2 * calc.results["energy"], abs=1e-9)

    def test_the_state_is_gapped_and_the_bands_follow_the_poles(self, spectral):
        assert spectral.gap > 0
        assert spectral.removal_edge < spectral.chemical_potential < spectral.addition_edge
        bands = solved().calc.bands()
        assert bands.shape == (2, 1)
        # k = 0 is occupied (its strongest pole is a removal), k = 1/2 empty.
        assert bands[0, 0] <= spectral.removal_edge + 1e-9
        assert bands[1, 0] >= spectral.addition_edge - 1e-9
        weights = solved().calc.band_weights()
        assert np.all((weights > 0.5) & (weights <= 1 + 1e-12))

    def test_the_broadened_trace_is_a_density(self, spectral):
        assert spectral.values.shape == (2, len(spectral.energies))
        assert np.all(spectral.values >= 0)

    def test_an_off_mesh_point_is_refused(self):
        with pytest.raises(ValueError, match="commensurate"):
            solved().calc.get_spectral_function(kpoints=[[0.25, 0, 0]])

    def test_an_active_space_is_refused(self):
        atoms = chain()
        atoms.calc = Mandacaru(method="vqe", basis={"name": "HAO", "virtual_orbitals": 1},
                               grid_spacing=0.35, kpts=GAMMA2, optimizer=SLSQP,
                               active_space={"orbitals": 2}, trace=False, **REAL)
        atoms.get_potential_energy()
        with pytest.raises(ValueError, match="full orbital space"):
            atoms.calc.get_spectral_function()
