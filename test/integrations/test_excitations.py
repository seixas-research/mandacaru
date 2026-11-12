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
    with pytest.raises(NotImplementedError, match="crystal"):
        crystal.calc.get_excitations(1)
    with pytest.raises(ValueError, match="spin-conserving"):
        hydrogen(unrestricted=True).calc.get_excitations(1, spin="singlet")
    with pytest.raises(NotImplementedError, match="meta-GGA"):
        hydrogen(xc="r2scan").calc.get_excitations(1)


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
