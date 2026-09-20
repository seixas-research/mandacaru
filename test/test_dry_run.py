# -*- coding: utf-8 -*-
# file: test/test_dry_run.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The dry run: qubit estimation without integrals, mapping or circuits.

Two things are pinned.  **Accuracy** -- the estimated qubit count equals the
qubit count of the Hamiltonian a real run builds (for a geometry, a frozen core,
a cached file and an explicit operator).  **Early stop** -- with ``dry_run=True``
no Hamiltonian is built and no solver loop runs: the ASE hook reports ``NaN``,
``run()`` returns the estimate, and even a reserved device is accepted.
"""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from mandacaru.algorithms import (count_basis_functions, estimate_qubits,
                                  Mandacaru, QubitEstimate)
from mandacaru.algorithms.base import VariationalDriver
from mandacaru.basis import BasisSet
from mandacaru.cli import main


def _h2(R=0.74, box=6.0):
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, R]], cell=[box] * 3,
                 pbc=False)


def _boxed(name, vacuum=3.0):
    atoms = molecule(name)
    atoms.center(vacuum=vacuum)
    return atoms


# --------------------------------------------------------------------------- #
# Counting.
# --------------------------------------------------------------------------- #

class TestEstimate:
    def test_h2_minimal_basis_is_four_qubits(self):
        est = estimate_qubits(_h2())
        assert isinstance(est, QubitEstimate)
        assert est.n_qubits == 4
        assert est.n_spatial_orbitals == 2 and est.n_spin_orbitals == 4
        assert est.num_particles == (1, 1) and est.n_electrons == 2
        assert est.per_atom == [("H", 1), ("H", 1)]
        assert est.n_qubits_reduced == 2
        assert est.source == "geometry"

    @pytest.mark.parametrize("name, basis", [
        ("H2O", "FAO"), ("LiH", "FAO"), ("H2O", "6-31G(d)"), ("NH3", "STO-3G"),
        ("H2O", {"name": "NAO", "size": "DZP"}),
    ])
    def test_matches_basis_function_count(self, name, basis):
        atoms = _boxed(name)
        spec = basis if isinstance(basis, str) else basis["name"]
        options = {} if isinstance(basis, str) else {
            k: v for k, v in basis.items() if k != "name"}
        bset = BasisSet.build(spec, **options)
        expected = sum(len(bset.atom(s)) for s in atoms.get_chemical_symbols())
        est = estimate_qubits(atoms, basis=basis)
        assert est.n_basis_functions == expected
        assert est.n_qubits == 2 * expected
        assert est.n_electrons == sum(atoms.get_atomic_numbers())

    def test_frozen_core_shrinks_the_register(self):
        water = _boxed("H2O")
        full = estimate_qubits(water)
        frozen = estimate_qubits(water, frozen_core=True)
        assert full.n_qubits == 14
        assert frozen.n_qubits == 12 and frozen.n_frozen_orbitals == 1
        assert frozen.n_electrons == 8 and frozen.num_particles == (4, 4)
        explicit = estimate_qubits(water, frozen_orbitals=[0])
        assert explicit.n_qubits == 12
        by_count = estimate_qubits(water, frozen_core=1)
        assert by_count.n_qubits == 12

    def test_charge_and_magmoms_set_the_occupation(self):
        water = _boxed("H2O")
        assert estimate_qubits(water, charge=2).num_particles == (4, 4)
        o2 = molecule("O2")
        o2.center(vacuum=3.0)
        o2.set_initial_magnetic_moments([1.0, 1.0])
        est = estimate_qubits(o2)
        assert est.num_particles == (9, 7) and est.n_qubits == 20

    def test_pseudopotentials_count_valence_only(self):
        est = estimate_qubits(_boxed("H2O"), basis="NCPP")
        assert est.n_electrons == 8 and est.n_frozen_orbitals == 0
        assert est.basis == "NCPP (SZ, pseudopotentials)"
        assert est.per_atom == [("O", 4), ("H", 1), ("H", 1)]
        assert est.n_qubits == 12
        assert any("pseudopotentials" in n for n in est.notes)

    @pytest.mark.parametrize("basis, expected, per_atom, label", [
        ({"name": "PAW", "size": "DZP"}, 46, [("O", 13), ("H", 5), ("H", 5)],
         "PAW (DZP, pseudopotentials)"),
        ({"name": "PAW", "size": "DZ"}, 24, [("O", 8), ("H", 2), ("H", 2)],
         "PAW (DZ, pseudopotentials)"),
        ({"name": "ONCVPSP", "size": "DZP"}, 46, [("O", 13), ("H", 5), ("H", 5)],
         "ONCVPSP (DZP, pseudopotentials)"),
        ({"O": {"name": "PAW", "size": "DZP"}, "H": "PAW"}, 30,
         [("O", 13), ("H", 1), ("H", 1)],
         'PAW (per-element sizes {"H": "SZ", "O": "DZP"}, pseudopotentials)'),
    ])
    def test_pseudopotential_size_hierarchy_is_counted(self, basis, expected,
                                                        per_atom, label):
        """The DZ/DZP size options of a pseudopotential family (PAW included)
        enlarge the valence basis exactly as a run would build it -- a 46-qubit
        water estimate is what makes the dry run indispensable here, since no
        state-vector driver could materialize that register."""
        from mandacaru.pseudopotentials.families import lookup_family
        from mandacaru.pseudopotentials.orbitals import pseudo_basis
        water = _boxed("H2O")
        est = estimate_qubits(water, basis=basis)
        assert est.n_qubits == expected and est.per_atom == per_atom
        assert est.basis == label
        # Cross-check against the family's own basis constructor.
        name = basis["name"] if "name" in basis else "PAW"
        size = (basis.get("size", "SZ") if "name" in basis
                else {"O": "DZP", "H": "SZ"})
        family = lookup_family(name)
        symbols = water.get_chemical_symbols()
        potentials = {s: family.get(s, None) for s in set(symbols)}
        fns, _ = pseudo_basis(symbols, water.get_positions(), potentials, size=size)
        assert est.n_qubits == 2 * len(fns)

    def test_plane_waves_match_the_engine(self):
        from mandacaru.core import PlaneWaveIntegrals
        cell = np.diag([3.0, 3.0, 3.0])
        atoms = Atoms("H2", positions=[[1.1, 1.5, 1.5], [1.9, 1.5, 1.5]],
                      cell=cell, pbc=True)
        pw = PlaneWaveIntegrals([(1.0, p) for p in atoms.positions], cell,
                                energy_cutoff=60.0, max_plane_waves=10_000)
        est = estimate_qubits(atoms, basis={"name": "PW", "energy_cutoff": 60.0})
        assert est.n_qubits == 2 * pw.n_orbitals
        assert est.per_atom == []           # not atom-centered
        with pytest.raises(NotImplementedError):
            estimate_qubits(atoms, basis={"name": "PW", "energy_cutoff": 60.0},
                            frozen_core=True)

    def test_count_basis_functions_labels(self):
        per_atom, label = count_basis_functions(_boxed("LiH"), "FAO")
        assert per_atom == [("Li", 2), ("H", 1)] and label == "FAO"
        _per_atom, label = count_basis_functions(
            _boxed("LiH"), {"name": "GTO", "n_gaussians": 3})
        assert "STO-3G" in label

    def test_too_small_basis_is_refused(self):
        with pytest.raises(ValueError, match="cannot fit"):
            estimate_qubits(_h2(), charge=-4)

    def test_needs_a_source(self):
        with pytest.raises(ValueError, match="needs atoms"):
            estimate_qubits()


# --------------------------------------------------------------------------- #
# Devices.
# --------------------------------------------------------------------------- #

class TestDevices:
    def test_qpu_capacity_is_compared(self):
        water = _boxed("H2O")
        small = estimate_qubits(water, frozen_core=True,
                                device="braket-ionq-aria")
        assert small.device_qubits == 25 and small.fits_device is True
        big = estimate_qubits(water, basis={"name": "NAO", "size": "DZP"},
                              device="braket-iqm-garnet")
        assert big.device_qubits == 20 and big.fits_device is False
        assert "DOES NOT FIT" in big.summary()

    def test_simulator_reports_memory_not_capacity(self):
        est = estimate_qubits(_h2())
        assert est.device == "AER_simulator"
        assert est.device_qubits is None and est.fits_device is None
        assert est.statevector_bytes == 16 * 2 ** 4
        assert "state vector" in est.summary()

    def test_reserved_device_is_estimated_not_refused(self):
        est = estimate_qubits(_h2(), device="ibm-quantum")
        assert est.device == "ibm-quantum" and est.n_qubits == 4
        assert any("shots > 0" in n for n in est.notes)

    def test_registry_exposes_capacities(self):
        from mandacaru.backends.hardware import device_qubits
        assert device_qubits("braket-ionq-forte") == 36
        assert device_qubits("braket-rigetti-ankaa") == 84
        assert device_qubits("AER_simulator") is None
        assert device_qubits("arn:aws:braket:us-east-1::device/qpu/x/y") is None

    def test_json_round_trip(self):
        est = estimate_qubits(_h2(), device="braket-ionq-aria")
        data = json.loads(est.to_json())
        assert data["n_qubits"] == 4 and data["fits_device"] is True
        assert data["num_particles"] == [1, 1]
        assert data["per_atom"] == [["H", 1], ["H", 1]]


# --------------------------------------------------------------------------- #
# Accuracy against real Hamiltonians.
# --------------------------------------------------------------------------- #

class TestAgainstRealRuns:
    def test_estimate_equals_the_built_hamiltonian(self, tmp_path):
        atoms = _h2()
        est = estimate_qubits(atoms, mapping="parity")
        path = str(tmp_path / "h2.json")
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="FAO", h=0.35, mapping="parity",
                               trace=False, profile=False, max_iterations=1,
                               save_hamiltonian=path,
                               hamiltonian_format="json")
        atoms.get_potential_energy()
        assert atoms.calc.n_qubits == est.n_qubits == 4
        assert atoms.calc.num_particles == est.num_particles

        # The cache file is a complete specification: header only, no mapping.
        cached = estimate_qubits(load_hamiltonian=path)
        assert cached.n_qubits == 4 and cached.source == "hamiltonian-file"
        assert cached.num_particles == (1, 1) and cached.mapping == "parity"

        # Direct-mode drivers estimate from the operator they hold.
        driver = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=path, trace=False, profile=False)
        assert driver.estimate_qubits().n_qubits == 4
        vqe = Mandacaru(method="vqe", load_hamiltonian=path, trace=False)
        assert vqe.estimate_qubits().num_particles == (1, 1)

    def test_frozen_core_estimate_equals_the_active_space(self):
        lih = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[7.0] * 3)
        est = estimate_qubits(lih, frozen_core=True)
        assert est.n_qubits == 4 and est.n_frozen_orbitals == 1
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            build_basis_hamiltonian
        h, particles, n_orb, _profile, _ctx = build_basis_hamiltonian(
            lih, "FAO", None, 0.4, 0, None, frozen_core=True)
        assert n_orb == est.n_spatial_orbitals == 2
        assert particles == est.num_particles == (1, 1)
        assert h.n_modes() == est.n_qubits


# --------------------------------------------------------------------------- #
# Early stop: nothing is built, nothing runs.
# --------------------------------------------------------------------------- #

def _forbid_execution(monkeypatch):
    """Make every expensive stage explode, so a dry run must avoid all of them."""
    def boom(*_a, **_k):
        raise AssertionError("a dry run must not reach this stage")
    monkeypatch.setattr(VariationalDriver, "_build_hamiltonian", boom)
    monkeypatch.setattr(VariationalDriver, "_make_timings", boom)
    monkeypatch.setattr(VariationalDriver, "_configure", boom)
    import mandacaru.algorithms.base as base
    monkeypatch.setattr(base, "build_provider", boom)


class TestEarlyStop:
    def test_ase_hook_stops_before_the_hamiltonian(self, monkeypatch):
        _forbid_execution(monkeypatch)
        atoms = _h2()
        atoms.calc = Mandacaru(dry_run=True)
        energy = atoms.get_potential_energy()
        assert np.isnan(energy)
        assert atoms.calc.result is None
        assert atoms.calc.dry_run_result.n_qubits == 4
        assert atoms.calc.solver.dry_run_result is atoms.calc.dry_run_result

    def test_forces_are_nan_in_a_dry_run(self, monkeypatch):
        _forbid_execution(monkeypatch)
        atoms = _h2()
        atoms.calc = Mandacaru(dry_run=True)
        forces = atoms.get_forces()
        assert forces.shape == (2, 3) and np.isnan(forces).all()

    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe", "subspace-vqe",
                                        "subspace-adapt-vqe"])
    def test_every_method_honors_dry_run(self, monkeypatch, method):
        _forbid_execution(monkeypatch)
        atoms = _h2()
        atoms.calc = Mandacaru(method=method, dry_run=True)
        assert np.isnan(atoms.get_potential_energy())
        assert atoms.calc.dry_run_result.method == method

    def test_direct_mode_run_returns_the_estimate(self, monkeypatch):
        from mandacaru.core import MolecularIntegrals, minimal_fao_basis
        from mandacaru.integrals import Grid
        nuclei = [(1.0, np.array([0.0, 0.0, -0.37])),
                  (1.0, np.array([0.0, 0.0, 0.37]))]
        grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.4)
        h = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei),
                               grid).molecular_hamiltonian(mo_basis=True,
                                                           n_electrons=2)
        driver = Mandacaru(method="adapt-vqe", hamiltonian=h, pool="fermionic",
                           num_particles=(1, 1), n_spatial_orbitals=2,
                           trace=False, profile=False, dry_run=True)
        monkeypatch.setattr(VariationalDriver, "_make_timings",
                            lambda *_a, **_k: (_ for _ in ()).throw(
                                AssertionError("ran")))
        est = driver.run()
        assert isinstance(est, QubitEstimate) and est.n_qubits == 4
        assert est.source == "hamiltonian" and driver.result is None
        from mandacaru.circuits import UCCSD
        vqe = Mandacaru(method="vqe", hamiltonian=h, ansatz=UCCSD(2, (1, 1)),
                        trace=False, dry_run=True)
        assert vqe.run().n_qubits == 4
        sub = Mandacaru(method="subspace-vqe", hamiltonian=h,
                        ansatz=UCCSD(2, (1, 1)), num_states=2, trace=False,
                        dry_run=True)
        assert sub.run().n_qubits == 4

    def test_nothing_is_materialized_in_a_dry_run(self, tmp_path, monkeypatch):
        """The 2^n matrix is what the estimate exists to warn about.

        Constructing a driver around a Hamiltonian used to build it eagerly --
        so asking whether a 40-qubit problem fits would first try to allocate
        it.
        """
        from mandacaru.core import PauliSum
        from mandacaru.core.serialization import save_hamiltonian

        for name in ("to_matrix", "to_sparse_matrix"):
            monkeypatch.setattr(
                PauliSum, name,
                lambda self, *a, _n=name, **k: pytest.fail(
                    f"PauliSum.{_n} was called in a dry run"))

        path = save_hamiltonian(tmp_path / "cached.json",
                                PauliSum({"IIII": -1.0, "ZIII": 0.2}),
                                num_particles=(1, 1), n_spatial_orbitals=2)
        driver = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=path, dry_run=True, trace=False,
                           profile=False)
        assert driver.run().n_qubits == 4
        assert Mandacaru(method="vqe", load_hamiltonian=path, dry_run=True,
                         trace=False).run().n_qubits == 4

    def test_a_cached_tapered_estimate_is_not_reduced_twice(self, tmp_path):
        """The stored width of a tapered file is already the reduced one."""
        from mandacaru.core import PauliSum
        from mandacaru.core.serialization import save_hamiltonian

        path = save_hamiltonian(tmp_path / "tapered.json",
                                PauliSum({"II": -1.0, "ZI": 0.2}),
                                mapping="parity_reduced",
                                num_particles=(1, 1), n_spatial_orbitals=2)
        driver = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=path, dry_run=True, trace=False,
                           profile=False)
        estimate = driver.run()
        assert estimate.n_qubits == 2 and estimate.mapping == "parity_reduced"
        assert any("already tapered" in note for note in estimate.notes)

    def test_reserved_device_and_qpu_do_not_raise(self, monkeypatch):
        _forbid_execution(monkeypatch)
        atoms = _h2()
        atoms.calc = Mandacaru(dry_run=True,
                                       device="ibm-quantum")
        assert np.isnan(atoms.get_potential_energy())
        assert atoms.calc.dry_run_result.device == "ibm-quantum"
        # A Braket QPU with shots: the shot path would build a provider.
        atoms.calc = Mandacaru(method="vqe", dry_run=True,
                                       device="braket-ionq-aria", shots=100)
        assert np.isnan(atoms.get_potential_energy())
        assert atoms.calc.dry_run_result.fits_device is True

    def test_verbose_dry_run_prints_the_summary(self, capsys):
        atoms = _h2()
        atoms.calc = Mandacaru(dry_run=True)
        atoms.get_potential_energy()
        out = capsys.readouterr().out
        assert "QUBITS REQUIRED   : 4" in out
        assert "no circuits executed" in out

    def test_calculator_dry_run_method_is_one_off(self, monkeypatch):
        _forbid_execution(monkeypatch)
        calc = Mandacaru(frozen_core=True)
        est = calc.dry_run(_boxed("H2O"))
        assert est.n_qubits == 12
        assert calc.dry_run_result is est and calc.result is None
        # One-off: the calculator itself was not switched to dry-run mode, so
        # the next energy / run() builds a real solver.
        assert "dry_run" not in calc.solver_kwargs


# --------------------------------------------------------------------------- #
# Command line.
# --------------------------------------------------------------------------- #

class TestCLI:
    def test_dry_run_flag(self, capsys, monkeypatch):
        _forbid_execution(monkeypatch)
        assert main(["H2O", "--cell", "8", "--frozen-core", "--dry-run",
                     "--quiet"]) == 0
        out = capsys.readouterr().out
        assert "QUBITS REQUIRED   : 12" in out
        assert "frozen core       : 1" in out

    def test_json_output(self, capsys, monkeypatch):
        _forbid_execution(monkeypatch)
        assert main(["LiH", "--cell", "8", "--dry-run", "--json",
                     "--mapping", "parity", "--device", "ibm-quantum",
                     "--device-qubits", "127"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["n_qubits"] == 6 and data["mapping"] == "parity"
        assert data["device"] == "ibm-quantum"
        assert data["device_qubits"] == 127 and data["fits_device"] is True

    def test_basis_options_and_files(self, capsys, monkeypatch, tmp_path):
        _forbid_execution(monkeypatch)
        from ase.io import write
        xyz = str(tmp_path / "water.xyz")
        write(xyz, _boxed("H2O"))                 # extxyz carries the Lattice
        assert main([xyz, "--basis", "NAO", "--basis-option", "size=DZ",
                     "--dry-run", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        bset = BasisSet.build("NAO", size="DZ")
        expected = sum(len(bset.atom(s)) for s in ("O", "H", "H"))
        assert data["n_basis_functions"] == expected
        assert data["n_qubits"] == 2 * expected
        assert "DZ" in data["basis"]

    def test_cached_hamiltonian_needs_no_geometry(self, capsys, tmp_path,
                                                  monkeypatch):
        path = str(tmp_path / "h2.json")
        atoms = _h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic", h=0.35,
                               trace=False, profile=False, max_iterations=1,
                               save_hamiltonian=path,
                               hamiltonian_format="json")
        atoms.get_potential_energy()
        _forbid_execution(monkeypatch)
        assert main(["--load-hamiltonian", path, "--dry-run", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["n_qubits"] == 4 and data["source"] == "hamiltonian-file"

    def test_geometry_is_required_without_a_cache(self):
        with pytest.raises(SystemExit):
            main(["--dry-run"])
        with pytest.raises(SystemExit):
            main(["not-a-molecule-or-file", "--dry-run"])

    def test_magmoms_and_spin_reach_the_estimate(self, capsys, monkeypatch):
        _forbid_execution(monkeypatch)
        assert main(["O2", "--cell", "8", "--magmoms", "1", "1", "--dry-run",
                     "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["num_particles"] == [9, 7]

    def test_a_bare_molecule_needs_a_cell(self, monkeypatch):
        """The box is the cell: a g2 name without --cell is refused."""
        _forbid_execution(monkeypatch)
        with pytest.raises(SystemExit, match="no unit cell.*--cell"):
            main(["H2", "--dry-run", "--json"])
        with pytest.raises(SystemExit, match="1, 3 or 9"):
            main(["H2", "--cell", "5", "5", "--dry-run", "--json"])

    def test_cell_flag_boxes_the_molecule(self, capsys, monkeypatch):
        from mandacaru.cli import load_geometry, parse_cell
        _forbid_execution(monkeypatch)
        assert main(["H2", "--cell", "5", "--dry-run", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["n_qubits"] == 4
        atoms = load_geometry("H2O", cell=[6.0, 7.0, 8.0])
        assert np.allclose(atoms.cell.lengths(), [6.0, 7.0, 8.0])
        assert np.allclose(atoms.get_center_of_mass(), [3.0, 3.5, 4.0],
                           atol=0.3)
        assert parse_cell([1, 0, 0, 0, 2, 0, 0, 0, 3]).shape == (3, 3)
        with pytest.raises(SystemExit, match="non-zero length"):
            parse_cell([0.0])

    def test_full_run_through_the_cli(self, capsys):
        # A real (tiny) run: the same entry point without --dry-run.
        assert main(["H2", "--method", "vqe", "--h", "0.4", "--cell", "5.0",
                     "--optimizer", "L-BFGS-B", "--quiet"]) == 0
        out = capsys.readouterr().out
        assert "Final energy:" in out
