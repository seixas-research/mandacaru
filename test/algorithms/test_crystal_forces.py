"""Crystal forces and stress through ``Mandacaru(method="dft")``: analytic (Rust), reduced meshes, saved crystals, refusals.

The numbers against the reconverged finite-difference oracles of the engines' freezes are in
``test/migration/test_native_periodic_response.py``; here is the behavior of the public route: the symmetry-reduced mesh
gives the whole mesh's force and stress, a saved crystal gives them after ``load_crystal`` (an old file without its setup block
raises naming it), everything the engines do not differentiate raises with its cause (no silent other calculation), and one
small crystal is differentiated by an independent ASE-style central difference computed in the test itself. Needs
``MANDACARU_PAW_PATH``.
"""
import json
import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import PropertyNotImplementedError

from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

pytestmark = pytest.mark.skipif(not os.environ.get("MANDACARU_PAW_PATH"), reason="MANDACARU_PAW_PATH is not set")

BASIS = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}
#: The oracle's SCF: free energy to 1e-11 Ha, density to 1e-9.
SCF = {"max_iterations": 400, "tolerance": 1e-11, "density_tolerance": 1e-9, "mixing": 0.25, "history": 7}
SMEARING = {"method": "fermi-dirac", "width": 0.1}


def lih():
    """LiH in a triclinic box at Gamma (the force freeze's cell; its lattice is not symmetric)."""
    return Atoms("LiH", cell=[[6.0, 0, 0], [0.4, 5.8, 0], [0.3, 0.2, 5.6]], pbc=True,
                 positions=[[2.0, 2.1, 2.2], [3.5, 2.2, 2.0]])


def calculator(**options):
    options.setdefault("grid_spacing", 0.45)
    options.setdefault("kpts", {"size": (1, 1, 1), "gamma": True})
    options.setdefault("symmetry", False)
    options.setdefault("periodic_scf_options", dict(SCF, width_ha=0.1 / HARTREE_TO_EV))
    return Mandacaru(method="dft", basis=BASIS, smearing=SMEARING, trace=False, **options)


# --------------------------------------------------------------------------- #
# The symmetry-reduced mesh (CF-6).
# --------------------------------------------------------------------------- #


def aluminum(symmetry):
    atoms = bulk("Al", "fcc", a=4.05)
    atoms.calc = Mandacaru(method="dft", basis=BASIS, grid_spacing=0.6, kpts={"size": (2, 2, 2), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.3}, symmetry=symmetry, trace=False,
                           periodic_scf_options=dict(SCF, width_ha=0.3 / HARTREE_TO_EV))
    return atoms


def displaced_silicon(symmetry):
    """Diamond Si, the second atom off its site along [111]: 6 operations, a force along the diagonal."""
    a = 5.43
    atoms = Atoms("Si2", cell=[[a / 2, a / 2, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2]], pbc=True,
                  positions=[[0, 0, 0], [a / 4 + 0.03, a / 4 + 0.03, a / 4 + 0.03]])
    atoms.calc = calculator(grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True}, symmetry=symmetry)
    return atoms


# The whole mesh and the reduced one are two module-scoped fixtures: each is charged to the first test that needs it, so
# no single test pays both SCFs and both engines (233 s together on a loaded machine, over the 180 s budget of one test).
@pytest.fixture(scope="module")
def aluminum_whole():
    atoms = aluminum(False)
    return atoms, atoms.get_stress()


@pytest.fixture(scope="module")
def aluminum_reduced():
    atoms = aluminum(True)
    return atoms, atoms.get_stress()


@pytest.fixture(scope="module")
def silicon_whole():
    atoms = displaced_silicon(False)
    return atoms, atoms.get_forces()


@pytest.fixture(scope="module")
def silicon_reduced():
    atoms = displaced_silicon(True)
    return atoms, atoms.get_forces()


@pytest.mark.slow
def test_the_stress_of_cubic_aluminum_on_the_reduced_mesh_is_an_isotropic_pressure(aluminum_reduced):
    """Cubic Al on the 2x2x2 mesh that 48 operations reduce: the stress is one pressure, not zero.

    Measured 57 s (the reduced SCF and its stress engine, charged to the fixture) at a machine load of 13-15."""
    atoms, stress = aluminum_reduced
    assert atoms.calc.periodic_problem.symmetry.n_operations == 48
    # The group average over 48 operations makes the tensor isotropic to rounding (7e-16 measured on the frozen pair in Rust):
    # the gate asserts the symmetrizer's exactness, not a physical tolerance.
    assert np.ptp(stress[:3]) < 1e-12 and np.abs(stress[3:]).max() < 1e-12
    assert abs(stress[0]) > 1e-3


@pytest.mark.slow
def test_the_stress_of_a_symmetry_reduced_mesh_is_the_whole_meshs(aluminum_whole, aluminum_reduced):
    """Cubic Al on the 2x2x2 mesh: the whole mesh (time reversal only) against the one 48 operations reduce.

    The whole mesh is charged to this test (116 s), the reduced one to the test above (57 s); together they took 233 s
    in one test before the split, over the 180 s budget. The stress of a displaced crystal (Si along [110] and [111]) is gated in
    Rust (`periodic_response_symmetry.rs`, grid 0.5). The whole-mesh model is cubic only to the quadrature error of its
    sphere rules (log P2 / 3c-i): at grid 0.6 its diagonal spread is of the order of the 3e-6 gate below, the reduced
    mesh's is exactly zero. Run alone with ``-k``, this test pays both module fixtures (116 + 57 = 173 s): pytest charges a
    module fixture to the first test that asks for it."""
    whole, stress_whole = aluminum_whole
    reduced, stress_reduced = aluminum_reduced
    assert len(whole.calc.periodic_problem.kpoints) > len(reduced.calc.periodic_problem.kpoints)
    # The plan's stress gates: 3e-6 eV/A^3 on a diagonal entry, 5e-8 on a shear.
    assert np.allclose(stress_reduced[:3], stress_whole[:3], atol=3e-6, rtol=0)
    assert np.allclose(stress_reduced[3:], stress_whole[3:], atol=5e-8, rtol=0)


@pytest.mark.slow
def test_the_reduced_mesh_force_follows_the_diagonal(silicon_reduced):
    """The reduced mesh keeps 6 operations: the force is along the diagonal and nowhere else, the symmetry holds in the
    answer."""
    atoms, force = silicon_reduced
    assert atoms.calc.periodic_problem.symmetry.n_operations == 6
    assert np.allclose(force[:, 0], force[:, 1], atol=1e-6)
    assert np.allclose(force[:, 0], force[:, 2], atol=1e-6)


@pytest.mark.slow
def test_the_force_of_a_symmetry_reduced_mesh_is_the_whole_meshs(silicon_whole, silicon_reduced):
    """Si displaced along [111] on 2x2x2: 8 points whole, 4 reduced. The whole mesh is charged to the fixture of this
    test, the reduced one to ``test_the_reduced_mesh_force_follows_the_diagonal`` above it. Measured 46 s and 26 s at a machine
    load of 13-15 (140 s together at load 15-20 before the split)."""
    whole, force_whole = silicon_whole
    reduced, force_reduced = silicon_reduced
    assert np.abs(force_whole).max() > 0.1
    assert np.allclose(force_reduced, force_whole, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# A saved crystal.
# --------------------------------------------------------------------------- #


def test_a_loaded_crystal_gives_forces_and_stress(tmp_path):
    path = tmp_path / "lih.json"
    atoms = lih()
    atoms.calc = calculator(save_crystal=path)
    forces, stress = atoms.get_forces(), atoms.get_stress()
    loaded = Mandacaru(method="dft", load_crystal=path, trace=False,
                       periodic_scf_options=dict(SCF, width_ha=0.1 / HARTREE_TO_EV), smearing=SMEARING)
    result = loaded.run()
    assert np.allclose(result.forces(), forces, atol=1e-6)
    assert np.allclose(result.stress(), stress, atol=1e-6)
    assert "forces" in loaded.implemented_properties and "stress" in loaded.implemented_properties
    # Through ASE, with the atoms attached to the calculator that loads the crystal.
    again = lih()
    again.calc = Mandacaru(method="dft", load_crystal=path, trace=False, smearing=SMEARING,
                           periodic_scf_options=dict(SCF, width_ha=0.1 / HARTREE_TO_EV))
    assert np.allclose(again.get_forces(), forces, atol=1e-6)
    assert np.allclose(again.get_stress(), stress, atol=1e-6)


def test_a_crystal_file_without_the_setup_block_has_an_energy_and_no_forces(tmp_path):
    """An old file: the energy never needed the block; a force or a stress raises naming it."""
    path = tmp_path / "lih.json"
    atoms = lih()
    atoms.calc = calculator(save_crystal=path)
    energy = atoms.get_potential_energy()
    payload = json.loads(path.read_text())
    assert payload["setup"]["version"] == 1 and payload["setup"]["operations"] is None
    del payload["setup"]
    old = tmp_path / "old.json"
    old.write_text(json.dumps(payload))
    result = Mandacaru(method="dft", load_crystal=old, trace=False, smearing=SMEARING,
                       periodic_scf_options=dict(SCF, width_ha=0.1 / HARTREE_TO_EV)).run()
    assert result.in_units("eV") == pytest.approx(energy, abs=1e-6)
    for property_ in (result.forces, result.stress):
        with pytest.raises(NotImplementedError, match="'setup' block"):
            property_()


# --------------------------------------------------------------------------- #
# The meta-GGA and rVV10 crystals reach the analytic route; float32 and hybrid +U raise.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("xc", ["r2scan", "pbe+rvv10"])
def test_r2scan_and_rvv10_crystals_have_analytic_forces(xc):
    """MG-F5/F6: the meta-GGA and rVV10 crystal forces are analytic in Rust (their
    finite-difference gates are in ``crates/mandacaru-core/tests/periodic_gradient_xc.rs``);
    here they reach the calculator."""
    atoms = lih()
    atoms.calc = calculator(xc=xc)
    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3) and np.all(np.isfinite(forces))
    assert np.abs(forces).max() > 1e-4


def test_an_r2scan_crystal_has_an_analytic_stress():
    """MG-F5b: the meta-GGA crystal stress is analytic in Rust (its reconverged strain gates are in
    ``crates/mandacaru-core/tests/periodic_stress_meta.rs``); here it reaches the calculator."""
    atoms = lih()
    atoms.calc = calculator(xc="r2scan")
    stress = atoms.get_stress(voigt=False)
    assert stress.shape == (3, 3) and np.all(np.isfinite(stress))
    assert np.allclose(stress, stress.T) and np.abs(stress).max() > 1e-6


def test_an_rvv10_crystal_has_an_analytic_stress():
    """AS-X: the rVV10 crystal stress is analytic in Rust (its reconverged strain gates are in
    ``crates/mandacaru-core/tests/periodic_stress_rvv10.rs``, the kernel term alone in
    ``nonlocal_strain.rs``); here it reaches the calculator."""
    atoms = lih()
    atoms.calc = calculator(xc="pbe+rvv10")
    stress = atoms.get_stress(voigt=False)
    assert stress.shape == (3, 3) and np.all(np.isfinite(stress))
    assert np.allclose(stress, stress.T) and np.abs(stress).max() > 1e-6


def test_float32_forces_and_stress_are_refused_before_the_scf(monkeypatch):
    atoms = lih()
    atoms.calc = calculator(precision="float32")
    ran = []
    monkeypatch.setattr(atoms.calc, "run", lambda *a, **k: ran.append("run"))
    for query in (atoms.get_forces, atoms.get_stress):
        with pytest.raises(ValueError, match="float32.*crystal forces or stress"):
            query()
    assert ran == [] and atoms.calc.result is None


def test_dft_plus_u_hybrid_raises():
    """Hybrid DFT+U raises NotImplementedError naming the functional."""
    with pytest.raises(NotImplementedError, match="hybrid functional"):
        calculator(xc="hse06", hubbard_u={"Cu": {"l": "d", "U": 4.0}})


def test_dft_plus_u_forces_and_stress_semilocal():
    """DFT+U on a semilocal crystal produces valid analytic forces and stress."""
    atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[7.0] * 3, pbc=True)
    atoms.center()
    atoms.calc = calculator(grid_spacing=0.45, hubbard_u={"Cu": {"l": "d", "U": 4.0}}, symmetry=False,
                            periodic_scf_options={"max_iterations": 50, "tolerance": 1e-8, "density_tolerance": 1e-7, "mixing": 0.25, "width_ha": 0.1 / HARTREE_TO_EV})
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert forces.shape == (2, 3)
    assert stress.shape == (6,)
    assert not np.isnan(forces).any()
    assert not np.isnan(stress).any()


def test_a_hybrid_with_symmetry_on_keeps_the_whole_mesh_and_differentiates_the_same_crystal():
    """A hybrid sums its exchange over the listed k-points, so the setup keeps the whole mesh for it
    whatever ``symmetry`` says (MG-F3 part 1, owner decision Q11): ``symmetry=True`` reduces
    nothing, and the forces and the stress are those of ``symmetry=False``. (The engines still
    refuse a hybrid crystal that does list a reduced mesh, from another source:
    ``periodic_response_refusals.rs``.) H2 in a cubic box at Gamma; measured about 14 s."""

    def hydrogen(symmetry):
        atoms = Atoms("H2", positions=[[3.0, 3.0, 2.63], [3.0, 3.0, 3.37]], cell=[6.0] * 3, pbc=True)
        atoms.calc = calculator(xc="hse06", symmetry=symmetry)
        return atoms

    reduced, whole = hydrogen(True), hydrogen(False)
    assert reduced.get_forces() == pytest.approx(whole.get_forces(), abs=1e-8)
    assert reduced.get_stress() == pytest.approx(whole.get_stress(), abs=1e-8)
    assert abs(reduced.get_forces()[1, 2]) > 0.1


def test_the_stress_of_a_molecule_is_refused_naming_the_cause():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = Mandacaru(method="dft", basis=BASIS, grid_spacing=0.4, trace=False)
    with pytest.raises(PropertyNotImplementedError, match="stress is only available for periodic structures"):
        atoms.get_stress()


def test_a_spin_polarized_crystal_has_forces_and_stress():
    """H2 triplet: the unrestricted engine differentiates the energy and computes forces and stress."""
    atoms = Atoms("HH", cell=[6.2] * 3, pbc=True, positions=[[3.1, 3.3, 2.0], [3.62, 3.0, 4.0]])
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    atoms.calc = calculator(grid_spacing=0.45, spin=True,
                            periodic_scf_options=dict(SCF, density_tolerance=1e-8, width_ha=0.1 / HARTREE_TO_EV))
    assert atoms.get_forces().shape == (2, 3)
    assert atoms.get_stress().shape == (6,)


# --------------------------------------------------------------------------- #
# An independent oracle: ASE-style central differences of the energy, computed here.
# --------------------------------------------------------------------------- #


def free_energy(atoms):
    return atoms.get_potential_energy(force_consistent=True)


@pytest.mark.parametrize("dispersion", [{}, {"vdw": "ts"}], ids=["plain", "ts"])
def test_lih_force_equals_the_reconverged_central_difference(dispersion):
    """Richardson over two steps of the free energy of the same discretized functional (no symmetry, tight SCF).

    With ``vdw="ts"`` the energy carries the Tkatchenko-Scheffler dispersion and so must the force (its stress is
    refused)."""
    atoms = lih()
    atoms.calc = calculator(**dispersion)
    analytic = atoms.get_forces()
    steps = (0.004, 0.008)
    for atom, axis in ((0, 0), (1, 2)):
        central = []
        for step in steps:
            energies = []
            for sign in (1, -1):
                moved = atoms.copy()
                moved.positions[atom, axis] += sign * step
                moved.calc = calculator(**dispersion)
                energies.append(free_energy(moved))
            central.append((energies[0] - energies[1]) / (2 * step))
        richardson = central[0] + (central[0] - central[1]) / 3
        assert abs(analytic[atom, axis] + richardson) < 1e-5, (atom, axis, analytic[atom, axis], -richardson)
        assert abs(analytic[atom, axis]) > 0.05


def test_the_dispersion_correction_has_no_crystal_stress():
    atoms = lih()
    atoms.calc = calculator(vdw="ts")
    with pytest.raises(NotImplementedError, match="Tkatchenko-Scheffler"):
        atoms.get_stress()


def strained_aluminum(strain, shape, sigma):
    """One Al atom in the fcc primitive cell under ``x' = (1 + strain) x`` with the node counts and the ion width pinned
    (the analytic stress is the derivative at the unstrained cell's pins; ``grid_spacing`` would re-mesh a strained cell)."""
    from mandacaru.integrals import Grid

    atoms = bulk("Al", "fcc", a=4.05)
    t = (np.eye(3) + strain).T
    cell = np.asarray(atoms.cell.array) @ t
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_positions(atoms.positions @ t)
    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0, units="angstrom", cell=cell, periodic=True,
                grid_spacing=list(np.linalg.norm(cell, axis=1) / np.asarray(shape)))
    assert tuple(grid.shape) == tuple(shape)
    atoms.calc = Mandacaru(method="dft", basis=BASIS, grid=grid, kpts={"size": (2, 1, 1), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 2.0}, symmetry=False, trace=False,
                           periodic_scf_options=dict(SCF, width_ha=2.0 / HARTREE_TO_EV))
    return atoms


@pytest.mark.slow
def test_aluminum_stress_equals_the_reconverged_strain_difference(monkeypatch):
    """ASE-style central differences of the free energy under strain, computed here (Richardson over two steps), against
    the analytic stress: the isotropic part and one shear. Measured 36 s at a machine load of 13-15."""
    from mandacaru.integrals import Grid
    from mandacaru.integrations import setup_block
    from mandacaru.units import ANGSTROM_TO_BOHR

    unstrained = bulk("Al", "fcc", a=4.05)
    reference = Grid(center=0.5 * unstrained.cell.array.sum(axis=0), box_size=0.0, grid_spacing=0.45, units="angstrom",
                     cell=unstrained.cell.array, periodic=True)
    shape = tuple(int(n) for n in reference.shape)
    # The Rust setup's ion width, 1.4 times the coarsest step of the unstrained grid, held for every strain.
    steps = np.linalg.norm(unstrained.cell.array, axis=1) / np.asarray(shape)
    sigma = 1.4 * float(steps.max()) * ANGSTROM_TO_BOHR
    monkeypatch.setattr(setup_block, "ion_width", lambda grid: sigma)
    analytic = strained_aluminum(np.zeros((3, 3)), shape, sigma)
    stress = analytic.get_stress()
    volume = abs(np.linalg.det(analytic.cell.array))

    def derivative(direction):
        """dF/de of the free energy (eV) along ``strain = e * direction``."""
        central = []
        for step in (0.0005, 0.001):
            energies = [free_energy(strained_aluminum(sign * step * direction, shape, sigma)) for sign in (1, -1)]
            central.append((energies[0] - energies[1]) / (2 * step))
        return central[0] + (central[0] - central[1]) / 3

    isotropic = derivative(np.eye(3)) / (3 * volume)
    shear = np.zeros((3, 3))
    shear[0, 1] = shear[1, 0] = 1.0
    xy = derivative(shear) / (2 * volume)                                  # g_xy + g_yx = 2 Omega sigma_xy
    # The gate is the floor of the oracle, not of the engine. Each free energy is converged to SCF["tolerance"] (1e-11 Ha =
    # 2.7e-10 eV); the Richardson combination (4 c(h) - c(2h)) / 3 of central differences at h = 5e-4 and 1e-3 weighs an
    # energy error by (4/3) / h + (1/3) / (2 h) = 3000 per unit strain, so at worst 8.2e-7 eV per unit strain: 1.6e-8 eV/A^3
    # on the isotropic part (divided by 3 V) and 2.5e-8 on a shear (by 2 V). The gate is 4-6 times that worst case; the
    # scratch run measured 1.7e-9 (so the gate has 60 times of room, not the 1800 times the old 3e-6 had).
    assert SCF["tolerance"] == 1e-11
    floor = 3000 * SCF["tolerance"] * HARTREE_TO_EV / (2 * volume)          # the shear's, the larger
    gate = 1e-7
    assert 3 * floor < gate < 10 * floor, (floor, gate)
    # The stress is a signal, not a number the gate hides: a flipped sign misses by 2 |stress|, >= 100 gates.
    assert abs(isotropic) > 1e3 * gate and abs(xy) > 1e3 * gate
    assert abs(stress[:3].mean() - isotropic) < gate                       # eV/A^3
    assert abs(stress[:3].mean() + isotropic) > 100 * gate                 # the control: the wrong sign bites
    assert abs(stress[5] - xy) < gate
    assert abs(stress[5] + xy) > 100 * gate
