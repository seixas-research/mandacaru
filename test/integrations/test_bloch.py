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

from mandacaru import Mandacaru, _native
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
        """Measured 2026-11-10: -13.382786723210 eV per cell for both solvers, the exact
        ground state of the band-limited supercell Hamiltonian. It was -12.422242881249 on
        the point-sampled Hamiltonian, which reproduced the removed periodic engine and depended
        on where the atom sat on the grid; see ``test/migration/test_native_bloch.py``.
        """
        energy = solved(method).get_potential_energy()
        assert energy == pytest.approx(-13.382786723210, abs=1e-7)

    def test_the_variational_energy_is_the_exact_ground_state_at_four_qubits(self):
        calc = solved().calc
        assert calc.results["energy"] == pytest.approx(exact_per_cell(calc, 2, 2), abs=1e-8)

    def test_it_is_not_the_isolated_cluster(self):
        """The lattice sums differ from the box: the isolated cluster gave -12.0319 eV per cell."""
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
        ({"basis": {"name": "UPAW-LCAO", "size": "SZ"}}, NotImplementedError, "UPAW-LCAO"),
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
    spread of 1.8e-15 eV. Re-measured on the band-limited Hamiltonian (2026-11-10): VASQA seeds
    0-4 give -12.382599787021714 eV per cell at 4 cells, spread 1e-15 eV."""

    @pytest.mark.parametrize("method", ["vasqa", "valsqa"])
    @pytest.mark.parametrize("seed", [0, 3])
    def test_the_two_cell_chain_reaches_the_exact_ground_state(self, method, seed):
        calc = solved(method, seed=seed).calc
        assert calc.results["energy"] == pytest.approx(exact_per_cell(calc, 2, 2), abs=1e-8)
        assert calc.results["energy"] == pytest.approx(-13.382786723210, abs=1e-7)

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
        model = fit(store / "empty", fractions=(1.,), parallelization={"threads": 1})
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

    def test_subspace_vasqa_solves_crystal_and_excited_spectral_function(self):
        atoms = chain()
        atoms.calc = Mandacaru(method="subspace-vasqa", basis="HAO", grid_spacing=0.35,
                               mapping="jordan_wigner", kpts=GAMMA2, num_states=2,
                               weights=[0.7, 0.3], max_steps=12, max_length=4, seed=5,
                               trace=False, **REAL)
        atoms.get_potential_energy()
        result = atoms.calc.result
        assert len(result.energies) == 2 and result.energies[0] <= result.energies[1] + 1e-12
        sf0 = atoms.calc.get_spectral_function(root=0)
        sf1 = atoms.calc.get_spectral_function(root=1)
        assert np.allclose(sf0.sum_rule, 1.0, atol=1e-12)
        assert np.allclose(sf1.sum_rule, 1.0, atol=1e-12)
        assert sf1.values.shape == sf0.values.shape

    def test_vasqa_checkpoint_resumes_identically(self, tmp_path):
        atoms = chain()
        ckpt = tmp_path / "vasqa_ckpt.json"
        common = dict(method="vasqa", basis="HAO", grid_spacing=0.35, mapping="jordan_wigner",
                      kpts=GAMMA2, optimizer=SLSQP, max_length=4, temperature=0.25, seed=7,
                      trace=False, **REAL)
        c_full = Mandacaru(max_steps=6, **common)
        r_full = c_full.run(atoms)
        c1 = Mandacaru(max_steps=3, checkpoint=str(ckpt), **common)
        c1.run(atoms)
        c2 = Mandacaru(max_steps=6, resume=str(ckpt), **common)
        r2 = c2.run(atoms)
        assert r2.energy_hartree == pytest.approx(r_full.energy_hartree, abs=1e-12)
        assert r2.final_energy == pytest.approx(r_full.final_energy, abs=1e-12)
        assert r2.operators == r_full.operators


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
        # the engine's own midpoint and difference, in eV
        assert spectral.chemical_potential == pytest.approx(
            0.5 * (spectral.removal_edge + spectral.addition_edge), abs=1e-12)
        assert spectral.gap == pytest.approx(spectral.addition_edge - spectral.removal_edge, abs=1e-12)
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


def lih_chain(active, **options):
    """LiH chain (a = 3.2 A, H off the axis) on three k-points: its +-k states are exactly
    degenerate pairs (a Li 1s pair and a sigma pair occupied, the lowest virtual pair at orbitals
    6 and 7), and a periodic grid has no deterministic frame for them."""
    atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [1.45, 0.05, 0.0]],
                  cell=np.diag([3.2, 6.0, 6.0]), pbc=[True, False, False])
    atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.3,
                           kpts={"size": (3, 1, 1), "gamma": True}, active_space=active,
                           optimizer=SLSQP, parallelization={"threads": 1}, **REAL, **options)
    return atoms


#: The three occupied sigma orbitals and one member of the degenerate virtual pair.
SPLIT = {"orbitals": 4, "frozen": [0, 1, 2]}
#: The same with the whole pair.
WHOLE = {"orbitals": 5, "frozen": [0, 1, 2]}


class TestADegenerateGroupSplitByTheActiveSpace:
    """The split-group guard (``MeanField::split_groups``; owner 2026-11-12, Q6): a force is
    refused by name, an energy run warns in the ``[ACTIVE SPACE]`` block. Before the guard the
    finite-difference force of ``SPLIT`` was not a derivative: net force (-0.0162, 0.0144,
    0.0018) eV/A, the H y component 0.01503, 0.01094, 0.00380 eV/A at 0.53, 2 and 4 mA, and a
    nonzero z force across the mirror plane z = 0."""

    def test_the_force_is_refused_by_name(self):
        atoms = lih_chain(SPLIT, trace=False)
        with pytest.raises(NotImplementedError,
                           match="splits 1 exactly degenerate orbital group"):
            atoms.get_forces()

    def test_the_energy_run_warns_in_the_active_space_block(self, tmp_path):
        from mandacaru.utils.logging import parse_output

        log = tmp_path / "split.log"
        atoms = lih_chain(SPLIT, trace=False, txt=str(log))
        atoms.get_potential_energy()
        warning = parse_output(str(log))["active_space"]["warning"]
        assert "splits 1 exactly degenerate orbital group" in warning
        # The fourth active orbital is one member of the virtual +-k pair (orbitals 6 and 7).
        assert "orbital energy" in warning and "active [6] deleted [7]" in warning

    def test_a_whole_group_runs_and_its_force_sums_to_zero(self, tmp_path):
        """Measured 2026-11-12 (finite differences, 1 mBohr): H (2.9484749, 0.0067956, 0)
        eV/A, net force 2e-10."""
        from mandacaru.utils.logging import parse_output

        log = tmp_path / "whole.log"
        atoms = lih_chain(WHOLE, trace=False, txt=str(log))
        forces = atoms.get_forces()
        assert np.abs(forces.sum(axis=0)).max() < 1e-7
        assert forces[1, 0] == pytest.approx(2.9484749, abs=5e-5)
        assert abs(forces[1, 2]) < 1e-7
        assert "warning" not in parse_output(str(log))["active_space"]


def paw_h2_chain(method="vqe", size=(2, 1, 1), **options):
    """An H2 chain (bond 0.74 A along the axis, H off it by 0.05 A) of period 3 A on PAW-LCAO:
    SZ, four supercell orbitals on two cells (8 qubits)."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.05, 0.0]],
                  cell=np.diag([3.0, 6.0, 6.0]), pbc=[True, False, False])
    atoms.calc = Mandacaru(method=method, basis="PAW-LCAO", grid_spacing=0.2,
                           mapping="jordan_wigner", kpts={"size": size, "gamma": True},
                           optimizer=SLSQP, trace=False, **REAL, **options)
    return atoms


@functools.lru_cache(maxsize=None)
def paw_solved(method="vqe"):
    atoms = paw_h2_chain(method)
    atoms.get_potential_energy()
    return atoms


class TestThePawLcaoSupercell:
    """Stage 4c of the Bloch methods (``BLOCH_VQE_PLAN.md`` 4.7): the PAW-LCAO supercell built from
    the crystal setup (``molecular::periodic::paw``) through the calculator. Its Rust gates (the
    Born-von Karman identity S4-1, the unfolding S4-2, the molecular limit S4-3, the placement
    S4-5) are in ``crates/mandacaru-core/tests/molecular_paw_supercell.rs``."""

    @pytest.mark.parametrize("method, tolerance", [("adapt-vqe", 1e-7), ("vqe", 1e-5)])
    def test_the_variational_energy_is_the_exact_ground_state_at_eight_qubits(self, method,
                                                                               tolerance):
        """S4-7: the solvers reach the exact ground state of the supercell's qubit Hamiltonian
        (eV per cell); ADAPT to the optimizer's tolerance, the fixed UCCSD ansatz of ``vqe`` to its
        own limit at four electrons in eight spin orbitals (2.0e-6 eV above, measured 2026-11-15)."""
        calc = paw_solved(method).calc
        assert calc.problem.num_qubits == 8
        assert calc.results["energy"] == pytest.approx(exact_per_cell(calc, 4, 2),
                                                       abs=tolerance)

    def test_the_spectral_function_keeps_its_sum_rule(self):
        """S4-7: the Lehmann spectral function of the correlated supercell state."""
        spectral = paw_solved().calc.get_spectral_function(eta=0.1)
        assert np.allclose(spectral.sum_rule, 1.0, atol=1e-12)
        assert spectral.ground_energy == pytest.approx(
            2 * paw_solved().calc.results["energy"], abs=1e-9)

    def test_the_analytic_force_sums_to_the_egg_box_and_is_the_energy_derivative(self):
        """S4-6 and Bloch-VQE stage 6b: the calculator's force on the PAW supercell is analytic
        (the Rust gradient of the energy per cell, the Z-vector included). It sums to the slope of
        the grid's egg box (the point-sampled basis moving against the grid: the central
        differences it replaced measured 7.3e-5 eV/A on this cell at grid_spacing 0.2 A, F_x(H 0)
        -5.897530 and F_x(H 1) 5.897603; the plan's 1e-6 eV/A is not reached) and equals the
        Richardson extrapolation of independent central differences of ``get_potential_energy``
        at 2 and 4 mA."""
        atoms = paw_h2_chain()
        forces = atoms.get_forces()
        assert atoms.calc.result.force_details["scheme"] == "analytic"
        for axis in range(3):
            assert forces[0][axis] == pytest.approx(-forces[1][axis], abs=1e-4)
        central = []
        for step in (0.002, 0.004):
            energies = []
            for sign in (1.0, -1.0):
                moved = atoms.copy()
                moved.positions = [[x + (sign * step if (i, a) == (1, 0) else 0.0)
                                    for a, x in enumerate(row)]
                                   for i, row in enumerate(atoms.positions.tolist())]
                moved.calc = paw_h2_chain().calc
                energies.append(moved.get_potential_energy())
            central.append((energies[1] - energies[0]) / (2 * step))
        assert forces[1][0] == pytest.approx((4 * central[0] - central[1]) / 3, abs=2e-5)

    def test_an_active_space_that_cuts_a_degenerate_pair_is_refused_for_forces(self):
        """S4-7: the split-group guard on a +-k cut of the PAW supercell (LiH, three cells)."""
        atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [1.6, 0.05, 0.0]],
                      cell=np.diag([3.2, 6.0, 6.0]), pbc=[True, False, False])
        atoms.calc = Mandacaru(method="vqe", basis="PAW-LCAO", grid_spacing=0.25,
                               kpts={"size": (3, 1, 1), "gamma": True},
                               active_space={"orbitals": 3, "frozen": [0, 1]},
                               optimizer=SLSQP, trace=False, **REAL)
        with pytest.raises(NotImplementedError, match="exactly degenerate orbital group"):
            atoms.get_forces()

    @pytest.mark.parametrize("case", ["LiH (3,1,1)", "Si 2x1x1 SZ"])
    def test_the_supercell_run_streams_its_pair_densities(self, case):
        """S4-8: every engine allocation is reserved through the calculation's context, and the
        pair densities are streamed per Cholesky column. So the reserved peak of the supercell
        run stays under the plan's 2 GB ceiling and under the crystal's own DFT run at Gamma plus
        what one stored copy of the pair densities (``m^2 P 16`` bytes) would add. Measured
        2026-11-15: 25.0 MB against 28.8 MB (LiH), 26.0 MB against 46.3 MB (Si)."""
        from ase.build import bulk

        if case.startswith("LiH"):
            def cell():
                return Atoms("LiH", positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]],
                             cell=np.diag([3.2, 6.0, 6.0]), pbc=[True, False, False])
            size, active = (3, 1, 1), {"orbitals": 4, "frozen": [0, 1]}
        else:
            def cell():
                return bulk("Si", "diamond", a=5.43)
            size, active = (2, 1, 1), {"orbitals": 4, "frozen": [0, 1, 2, 3, 4, 5]}
        common = dict(basis="PAW-LCAO", grid_spacing=0.25, trace=False)
        crystal = cell()
        crystal.calc = Mandacaru(method="dft", xc="lda", kpts={"size": (1, 1, 1), "gamma": True},
                                 **common)
        crystal.get_potential_energy()
        supercell = cell()
        supercell.calc = Mandacaru(method="vqe", kpts={"size": size, "gamma": True},
                                   active_space=active, **common, **REAL)
        supercell.get_potential_energy()
        calc = supercell.calc
        m = _native.integrals_report(calc.integrals)["n_orbitals"]
        points = reduce(lambda a, b: a * b, calc.integrals.grid.shape)
        peak = calc.context.memory_peak_bytes
        stored = m * m * points * 16
        print(f"{case}: m {m}, P {points}; reserved peak {peak / 1e6:.1f} MB, crystal DFT "
              f"{crystal.calc.context.memory_peak_bytes / 1e6:.1f} MB, stored pairs "
              f"{stored / 1e6:.1f} MB")
        assert peak < 2 << 30
        assert peak < crystal.calc.context.memory_peak_bytes + stored

    def test_the_hao_supercell_orders_the_bond_force_the_same_way(self):
        """S4-4, a report: PAW-LCAO (``lda-sr``) and HAO differ by dataset and basis, so only the
        sign of the bond force of the H2 chain is gated; the two forces are printed."""
        paw = paw_h2_chain().get_forces()
        atoms = paw_h2_chain()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.2,
                               mapping="jordan_wigner", kpts=GAMMA2, optimizer=SLSQP,
                               trace=False, **REAL)
        hao = atoms.get_forces()
        print(f"H2 chain bond force F_x(H 1): PAW-LCAO {paw[1][0]:+.6f}, HAO {hao[1][0]:+.6f} eV/A")
        assert (paw[1][0] > 0) == (hao[1][0] > 0)
