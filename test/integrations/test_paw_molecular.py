"""The molecular PAW-LCAO adapter (``integrations/paw_molecular.py``)."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru



def _oxygen(cell):
    return Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]], cell=cell)


def test_a_molecule_without_a_cell_raises_as_on_every_molecular_route():
    """A zero cell used to become a 2 x 2 x 2 box on the PAW route, and the SCF
    converged there to an energy thousands of eV off (O2: -3902 eV against
    -860 eV in a real box). The all-electron route raised; both raise now."""
    for basis in ("HAO", {"name": "PAW-LCAO", "size": "SZ"}):
        atoms = _oxygen(None)
        atoms.calc = Mandacaru(method="rhf", basis=basis, grid_spacing=0.3)
        with pytest.raises(ValueError, match="nondegenerate cell or an explicit grid"):
            atoms.get_potential_energy()


def test_the_box_is_the_shared_molecular_box():
    """The PAW route samples the box every molecular route uses: centered on the
    mean atomic position, spanning the cell at the requested spacing."""
    atoms = _oxygen(np.diag([5.0, 5.0, 6.2]))
    atoms.center()
    atoms.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.3)
    atoms.get_potential_energy()
    assert tuple(atoms.calc.integrals.grid.shape) == (18, 18, 22)


def test_a_multi_atom_molecule_gives_the_same_bits_on_every_thread_count():
    """The one-center local and electrostatic matrices sum one partial per atom in atom order on
    every thread count: the SCF energy and the one-body matrix of H2O did differ by 4.4e-11 eV and
    1.3e-13 Ha between one and four threads when the serial route used a single accumulator."""
    runs = []
    for threads in (1, 2, 4):
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]])
        atoms.center(vacuum=3.0)
        atoms.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                               grid_spacing=0.4, parallelization={"threads": threads})
        runs.append((atoms.get_potential_energy(), atoms.calc.integrals.snapshot()))
    for energy, snapshot in runs[1:]:
        assert energy == runs[0][0]
        for name in ("overlap", "kinetic", "potential", "one_body"):
            assert np.array_equal(snapshot[name], runs[0][1][name]), name


def silane():
    """SiH4: silicon's frozen core is 1s 2s 2p, three closed subshells."""
    d = 1.48 / 3 ** 0.5
    atoms = Atoms("SiH4", positions=[[0, 0, 0], [d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]],
                  cell=[7, 7, 7])
    atoms.center()
    return atoms


@pytest.mark.parametrize("active_space", [None, {"orbitals": 6}])
def test_a_hamiltonian_over_a_multi_shell_core_carries_its_core_valence_exchange(active_space):
    """The second-quantized Hamiltonian over silicon's frozen 1s 2s 2p core (the dense model and
    an active space) carries the valence's exchange with that core, built from the dataset's
    stored core states (owner Q9); it was refused for any core beyond a closed 1s shell."""
    atoms = silane()
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.3, active_space=active_space)
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.calc.result.hamiltonian is not None


def test_hartree_fock_runs_over_a_multi_shell_core():
    atoms = silane()
    atoms.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"}, grid_spacing=0.3)
    assert np.isfinite(atoms.get_potential_energy())


def test_a_hybrid_on_a_multi_shell_core_is_refused_by_name():
    """HSE06 takes a fraction of exact exchange, which includes the valence's exchange with the
    frozen core, which the screened one-center terms carry for an empty core or a single closed 1s
    shell only: on SiH4 (a 1s 2s 2p core) it raises, naming the functional, while the semilocal
    functional of the same molecule runs."""
    def silane_with(xc):
        atoms = silane()
        atoms.calc = Mandacaru(method="dft", xc=xc, basis={"name": "PAW-LCAO", "size": "SZ"},
                               grid_spacing=0.3)
        return atoms

    assert np.isfinite(silane_with("pbe").get_potential_energy())
    with pytest.raises(ValueError, match="hse06 takes 0.25 of exact exchange.*screened core-valence exchange"):
        silane_with("hse06").get_potential_energy()
