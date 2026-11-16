"""``Mandacaru.get_excitations``: the request reaches the native linear-response solver, the
refusals name what is missing, and each call writes one ``[EXCITED STATES]`` block.

H2 on a PAW-LCAO DZP basis (ten orbitals, nine transitions; ``HAO`` on H2 has one). Needs
``MANDACARU_PAW_PATH``."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.utils import parse_output


def hydrogen(tmp_path=None, **options):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5])
    atoms.center()
    settings = dict(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                    grid_spacing=0.3)
    settings.update(options)
    if tmp_path is not None:
        settings["txt"] = str(tmp_path / "run.log")
    atoms.calc = Mandacaru(**settings)
    return atoms


def test_singlets_triplets_and_both_approximations():
    atoms = hydrogen()
    singlets = atoms.calc.get_excitations(3)
    assert len(singlets) >= 3 and singlets.requested == 3
    assert np.all(np.diff(singlets.energies) >= 0) and singlets.energies[0] > 0
    assert np.all(singlets.oscillator_strengths >= 0)
    assert singlets.transition_dipoles.shape == (len(singlets), 3)
    assert np.all(singlets.wavelengths > 0)
    triplets = atoms.calc.get_excitations(3, spin="triplet")
    assert np.all(triplets.oscillator_strengths == 0)
    assert triplets.energies[0] < singlets.energies[0]
    casida = atoms.calc.get_excitations(3, approximation="casida")
    assert casida.energies[0] <= singlets.energies[0] + 1e-9
    plus, minus = casida.amplitudes(0)
    assert abs(float(np.sum(plus * minus)) - 1) < 1e-10


def test_the_independent_kernel_gives_the_orbital_energy_differences():
    atoms = hydrogen()
    bare = atoms.calc.get_excitations(1, kernel="independent")
    levels = atoms.calc.get_eigenvalues()
    occupied = bare.occupied[0]
    assert abs(bare.energies[0] - (levels[occupied] - levels[occupied - 1])) < 1e-8


def test_unknown_names_raise_value_errors_listing_the_choices():
    atoms = hydrogen()
    with pytest.raises(ValueError, match="casida"):
        atoms.calc.get_excitations(1, approximation="rpa")
    with pytest.raises(ValueError, match="triplet"):
        atoms.calc.get_excitations(1, spin="quintet")
    with pytest.raises(ValueError, match="independent"):
        atoms.calc.get_excitations(1, kernel="alda")


def test_unsupported_states_raise_by_name():
    quantum = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5])
    quantum.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.35)
    with pytest.raises(ValueError, match="mean-field"):
        quantum.calc.get_excitations(1)
    crystal = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[3, 3, 3], pbc=True)
    crystal.calc = Mandacaru(method="dft", xc="lda", basis="HAO", grid_spacing=0.35)
    with pytest.raises(NotImplementedError, match="PAW-LCAO"):
        crystal.calc.get_excitations(1)
    with pytest.raises(ValueError, match="spin-conserving"):
        hydrogen(unrestricted=True).calc.get_excitations(1, spin="singlet")
    with pytest.raises(NotImplementedError, match="rVV10"):
        hydrogen(xc="pbe+rvv10").calc.get_excitations(1)


def test_each_request_writes_one_excited_states_block(tmp_path):
    atoms = hydrogen(tmp_path)
    found = atoms.calc.get_excitations(2, approximation="casida")
    text = (tmp_path / "run.log").read_text()
    assert text.count("[EXCITED STATES]") == 1
    block = parse_output(str(tmp_path / "run.log"))["excited_states"]
    assert block["approximation"] == "casida" and block["spin"] == "singlet"
    assert len(block["roots"]) == len(found)
    assert abs(block["roots"][0]["energy_eV"] - found.energies[0]) < 1e-7


def test_the_screened_hybrid_runs_and_both_gauges_are_reported():
    atoms = hydrogen(xc="hse06")
    singlets = atoms.calc.get_excitations(2, approximation="casida")
    assert singlets.energies[0] > 0 and np.all(singlets.velocity_oscillator_strengths >= 0)
    assert singlets.velocity_oscillator_strengths[0] > 0
    triplets = atoms.calc.get_excitations(2, spin="triplet")
    assert np.all(triplets.velocity_oscillator_strengths == 0)


def test_an_open_shell_gives_spin_conserving_roots_with_their_spin_contamination():
    from ase.build import molecule

    atoms = molecule("NH2")
    atoms.center(vacuum=3.5)
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                           grid_spacing=0.3)
    found = atoms.calc.get_excitations(3, approximation="casida")
    assert found.spin == "unrestricted" and len(found.occupied) == 2
    assert found.occupied[0] == found.occupied[1] + 1
    assert abs(found.reference_spin_squared - 0.75) < 0.05
    assert found.spin_squared.shape == found.energies.shape
    assert found.transitions(0)[0].spin in ("alpha", "beta")
    weights, occupied, empty = found.natural_transition_orbitals(0, channel=1)
    assert occupied.shape == empty.shape


def test_the_meta_gga_runs_through_the_same_request():
    found = hydrogen(xc="r2scan").calc.get_excitations(2, approximation="casida")
    assert found.energies[0] > 0 and found.functional == "r2scan"


def test_kohn_sham_and_hartree_fock_run_on_an_element_beyond_neon():
    """Sodium's frozen core is a closed 1s shell, but of an element past Z = 10: its Hartree-Fock
    exchange with the valence comes from the dataset's stored core states (owner Q9), so
    Hartree-Fock of NaH runs (the old closed-1s rule refused it by atomic number), and Kohn-Sham
    TD-DFT of the same molecule, which never reads a core-valence exchange, runs too."""
    atoms = Atoms("NaH", positions=[[0, 0, 0], [0, 0, 1.89]], cell=[7, 7, 7])
    atoms.center()
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                           grid_spacing=0.3)
    found = atoms.calc.get_excitations(2)
    assert np.all(np.isfinite(found.energies)) and np.all(found.energies > 0)
    hartree_fock = Atoms("NaH", positions=[[0, 0, 0], [0, 0, 1.89]], cell=[7, 7, 7])
    hartree_fock.center()
    hartree_fock.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                                  grid_spacing=0.3)
    assert np.isfinite(hartree_fock.get_potential_energy())


def test_a_crystal_has_bethe_salpeter_excitons(tmp_path):
    """The H2 chain (1.4 Bohr bond along x, 6 x 8 x 8 Bohr cell, Gamma only): the singlet,
    triplet and independent excitons of the native Bethe-Salpeter operator, its refusals and its
    ``[EXCITED STATES]`` block."""
    bohr = 0.529177210903
    atoms = Atoms("H2", positions=[[2.3 * bohr, 4.0 * bohr, 4.0 * bohr],
                                   [3.7 * bohr, 4.0 * bohr, 4.0 * bohr]],
                  cell=[6.0 * bohr, 8.0 * bohr, 8.0 * bohr], pbc=True)
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.4 * bohr, kpts={"size": (1, 1, 1), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.01}, trace=False,
                           txt=str(tmp_path / "run.log"))
    view = atoms.calc.get_excitations(1)  # SZ H2 has two bands: one transition
    assert view.kernel == "bse" and view.spin == "singlet" and view.approximation == "tda"
    assert view.energies.shape == (1,) and view.weights.shape == (1, 3) and view.transitions == 1
    assert view.kpoint_weights.shape == (1, 1) and view.mesh == (1, 1, 1)
    assert np.allclose(view.binding, view.onset - view.energies)
    assert view.hermiticity < 1e-10 and view.converged and view.solver == "dense"
    assert view.static_tensor[0, 0] > 1 and view.screened_head > 0
    triplet = atoms.calc.get_excitations(1, spin="triplet")
    assert triplet.energies[0] <= view.energies[0] + 1e-9 and triplet.weights.shape == (1, 3)
    bare = atoms.calc.get_excitations(1, kernel="independent")
    assert np.isclose(bare.energies[0], bare.onset) and np.isclose(bare.binding[0], 0.0)
    block = parse_output(str(tmp_path / "run.log"))["steps"][-1]["excited_states"]
    assert block["method"] == "bethe-salpeter" and block["kernel"] == "independent"
    epsilon = atoms.calc.get_dielectric_function([5.0, 10.0], kernel="bse", damping=1.0)
    assert epsilon.kernel == "bse" and epsilon.cutoff is not None and epsilon.tensor.shape == (2, 3, 3)
    with pytest.raises(NotImplementedError, match="casida"):
        atoms.calc.get_excitations(1, approximation="casida")
    with pytest.raises(NotImplementedError, match="adiabatic"):
        atoms.calc.get_excitations(1, kernel="adiabatic")
    with pytest.raises(TypeError):
        hydrogen().calc.get_excitations(1, bands=(1, 1))



def test_an_excited_state_calculation_returns_the_root_energy_and_its_forces(tmp_path):
    """``excited_state=``: the energy is the ground state's plus the root's excitation energy,
    the forces are the excited state's (H2's lowest singlet is antibonding: they stretch the
    bond far more than the ground state's at this length), ``get_excitations`` still answers for the ground state, and ``[SCF SUMMARY]`` names
    the root; ``track="overlap"`` follows it into the next geometry."""
    ground = hydrogen()
    omega = ground.calc.get_excitations(1, tolerance=1e-9).energies[0]  # eV, as the energies
    excited = hydrogen(tmp_path, excited_state={"root": 0, "track": "overlap"})
    energy = excited.get_potential_energy()
    assert abs(energy - (ground.get_potential_energy() + omega)) < 1e-6
    assert abs(excited.get_potential_energy(force_consistent=True) - energy) < 1e-10
    forces, pulled = excited.get_forces(), ground.get_forces()
    assert forces.shape == (2, 3) and forces[1, 2] > pulled[1, 2] + 5 and abs(forces[:, 2].sum()) < 1e-8
    assert abs(excited.calc.get_excitations(1, tolerance=1e-9).energies[0] - omega) < 1e-7
    assert "root 0 of the singlet roots (tda, adiabatic kernel), picked by index" in (tmp_path / "run.log").read_text()
    excited.positions[1, 2] += 0.02
    assert excited.get_potential_energy() < energy
    assert "followed by transition-density overlap" in (tmp_path / "run.log").read_text()


def test_the_dispersion_of_the_ground_state_is_added_to_the_excited_state():
    ground = hydrogen(vdw="ts")
    omega = ground.calc.get_excitations(1, tolerance=1e-9).energies[0]
    excited = hydrogen(vdw="ts", excited_state={"root": 0})
    assert abs(excited.get_potential_energy() - (ground.get_potential_energy() + omega)) < 1e-6


def test_an_excited_state_request_is_parsed_and_refused_by_name():
    with pytest.raises(ValueError, match="'root', 'spin', 'approximation' or 'track'"):
        hydrogen(excited_state={"state": 1})
    with pytest.raises(ValueError, match="'index' or 'overlap'"):
        hydrogen(excited_state={"track": "energy"})
    with pytest.raises(TypeError, match="must be a dict"):
        hydrogen(excited_state=1)
    with pytest.raises(ValueError, match="excited_state"):
        Mandacaru(method="ghf", excited_state={"root": 0})
    crystal = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[3, 3, 3], pbc=True)
    crystal.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                             grid_spacing=0.35, excited_state={"root": 0})
    with pytest.raises(NotImplementedError, match="crystal"):
        crystal.get_potential_energy()
    crystal.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                             grid_spacing=0.35, excited_state={"root": 0})
    with pytest.raises(NotImplementedError, match="native DFT"):
        crystal.get_potential_energy()


def test_a_spin_polarized_reference_has_excited_state_forces(tmp_path):
    """``spin=True``: the reference is unrestricted, its roots spin-conserving (on closed-shell H2
    root 0 is the Ms = 0 component of the lowest triplet), and ``spin`` stays unset."""
    ground = hydrogen(spin=True)
    omega = ground.calc.get_excitations(1, tolerance=1e-9).energies[0]
    excited = hydrogen(tmp_path, spin=True, excited_state={"root": 0, "track": "overlap"})
    assert abs(excited.get_potential_energy() - (ground.get_potential_energy() + omega)) < 1e-6
    assert excited.get_forces().shape == (2, 3)
    excited.positions[1, 2] += 0.02
    excited.get_potential_energy()
    assert "of the unrestricted roots" in (tmp_path / "run.log").read_text()
    assert "followed by transition-density overlap 0.99" in (tmp_path / "run.log").read_text()
    with pytest.raises(ValueError, match="leave spin unset"):
        hydrogen(spin=True, excited_state={"spin": "singlet"}).get_potential_energy()


def test_a_hartree_fock_reference_has_cis_and_tdhf_excited_states(tmp_path):
    """``method="rhf"`` (TDHF, ``"casida"``) and ``"uhf"`` (CIS, ``"tda"``): the energy is the
    reference's plus the root's, and the force follows the energy along the bond."""
    for method, state in (("rhf", {"approximation": "casida"}), ("uhf", {})):
        ground = hydrogen(method=method, xc=None)
        omega = ground.calc.get_excitations(1, tolerance=1e-9, **state).energies[0]
        excited = hydrogen(tmp_path, method=method, xc=None, excited_state=state)
        energy = excited.get_potential_energy()
        assert abs(energy - (ground.get_potential_energy() + omega)) < 1e-6
        force = excited.get_forces()[1, 2]
        moved = [hydrogen(method=method, xc=None, excited_state=state) for _ in range(2)]
        for atoms, step in zip(moved, (0.005, -0.005)):
            atoms.positions[1, 2] += step
        difference = (moved[1].get_potential_energy() - moved[0].get_potential_energy()) / 0.01
        assert abs(force - difference) < 2e-3, (method, force, difference)


def test_hubbard_u_reaches_the_excited_state():
    """hubbard_u on method="dft": the excited state is the +U reference's, and its
    forces run (gated in Rust on copper's 3d shell)."""
    hydride = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[7.0] * 3)
    hydride.center()
    settings = dict(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                    grid_spacing=0.3, hubbard_u={"Cu": {"l": "d", "U": 4.0, "J": 1.0}})
    ground = hydride.copy()
    ground.calc = Mandacaru(**settings)
    omega = ground.calc.get_excitations(1, tolerance=1e-9).energies[0]
    hydride.calc = Mandacaru(**settings, excited_state={"root": 0})
    assert abs(hydride.get_potential_energy() - (ground.get_potential_energy() + omega)) < 1e-6
    assert hydride.get_forces().shape == (2, 3)


def test_r2scan_reaches_the_excited_state():
    """xc="r2scan": the excited state is the meta-GGA's (E_0 + omega), and its force follows the
    energy along the bond (gated in Rust on the library's LiH, the current-density term
    included)."""
    ground = hydrogen(xc="r2scan")
    omega = ground.calc.get_excitations(1, tolerance=1e-9).energies[0]
    excited = hydrogen(xc="r2scan", excited_state={"root": 0})
    assert abs(excited.get_potential_energy() - (ground.get_potential_energy() + omega)) < 1e-6
    force = excited.get_forces()[1, 2]
    moved = [hydrogen(xc="r2scan", excited_state={"root": 0}) for _ in range(2)]
    for atoms, step in zip(moved, (0.005, -0.005)):
        atoms.positions[1, 2] += step
    difference = (moved[1].get_potential_energy() - moved[0].get_potential_energy()) / 0.01
    assert abs(force - difference) < 2e-3, (force, difference)


def test_hse06_reaches_the_excited_state():
    """xc="hse06": the excited state is the screened hybrid's (E_0 + omega), and its force
    follows the energy along the bond (gated in Rust on the library's LiH)."""
    ground = hydrogen(xc="hse06")
    omega = ground.calc.get_excitations(1, tolerance=1e-9).energies[0]
    excited = hydrogen(xc="hse06", excited_state={"root": 0})
    assert abs(excited.get_potential_energy() - (ground.get_potential_energy() + omega)) < 1e-6
    force = excited.get_forces()[1, 2]
    moved = [hydrogen(xc="hse06", excited_state={"root": 0}) for _ in range(2)]
    for atoms, step in zip(moved, (0.005, -0.005)):
        atoms.positions[1, 2] += step
    difference = (moved[1].get_potential_energy() - moved[0].get_potential_energy()) / 0.01
    assert abs(force - difference) < 2e-3, (force, difference)


def test_a_spin_orbit_dataset_gives_the_two_component_excitations(tmp_path):
    """A spin-orbit dataset (a spinor reference) routes to the two-component response. H2 has no
    spin-orbit splitting to speak of, so its lowest state is the triplet, three dark states."""
    atoms = hydrogen(tmp_path, paw_dataset="lda-fr")
    spinor = atoms.calc.get_excitations(4, approximation="casida")
    assert spinor.spin == "spinor" and len(spinor.occupied) == 1
    assert np.iscomplexobj(spinor.transition_dipoles)
    assert spinor.transition_dipoles.shape == (len(spinor), 3)
    assert np.ptp(spinor.energies[:3]) < 1e-6 and spinor.energies[3] > spinor.energies[2] + 1e-3
    assert np.all(spinor.oscillator_strengths[:3] < 1e-8)
    plus, minus = spinor.amplitudes(0)
    assert abs(np.vdot(plus, minus).real - 1) < 1e-8
    with pytest.raises(NotImplementedError, match="spinor"):
        spinor.transition_density_matrix(0)
    log = (tmp_path / "run.log").read_text()
    assert log.count("[EXCITED STATES]") == 1 and "spinor" in log


def test_a_spin_orbit_dataset_reaches_the_excited_state():
    """A spin-orbit dataset's spinor reference has an excited state (E_0 + omega) and forces
    (gated in Rust on Au-H): root 3, the state above the dark triplet's three (``spin`` unset),
    its force along the bond against a central difference."""
    ground = hydrogen(paw_dataset="lda-fr")
    omega = ground.calc.get_excitations(4, tolerance=1e-9).energies[3]
    excited = hydrogen(paw_dataset="lda-fr", excited_state={"root": 3})
    assert abs(excited.get_potential_energy() - (ground.get_potential_energy() + omega)) < 1e-6
    force = excited.get_forces()[1, 2]
    moved = [hydrogen(paw_dataset="lda-fr", excited_state={"root": 3}) for _ in range(2)]
    for atoms, step in zip(moved, (0.005, -0.005)):
        atoms.positions[1, 2] += step
    difference = (moved[1].get_potential_energy() - moved[0].get_potential_energy()) / 0.01
    assert abs(force - difference) < 2e-3, (force, difference)


def test_a_magnetized_spinor_reference_has_excited_state_forces():
    """A magnetized spinor reference without spin-orbit coupling (H2+ from noncollinear initial
    moments) has excited-state forces (gated in Rust on H2+ and H3): its lowest state, a spin
    flip (the turns of the moment, at omega = 0, are not states), against a central difference;
    with spin-orbit coupling the same cation refuses by name."""
    def ion(**options):
        atoms = hydrogen(charge=1, **options)
        atoms.set_initial_magnetic_moments([[0, 0, 0.5], [0, 0, 0.5]])
        return atoms
    ground = ion()
    omega = ground.calc.get_excitations(2, tolerance=1e-9).energies[0]
    assert omega > 1.0
    excited = ion(excited_state={"root": 0})
    assert abs(excited.get_potential_energy() - (ground.get_potential_energy() + omega)) < 1e-6
    force = excited.get_forces()[1, 2]
    moved = [ion(excited_state={"root": 0}) for _ in range(2)]
    for atoms, step in zip(moved, (0.005, -0.005)):
        atoms.positions[1, 2] += step
    difference = (moved[1].get_potential_energy() - moved[0].get_potential_energy()) / 0.01
    assert abs(force - difference) < 2e-3, (force, difference)
    with pytest.raises(NotImplementedError, match="magnetized"):
        hydrogen(charge=1, paw_dataset="lda-fr", excited_state={"root": 1}).get_forces()
