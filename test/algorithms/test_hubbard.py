"""The ``hubbard_u`` option of ``Mandacaru``: what is accepted, what is refused and when.

The option is read by the native engine (``models::hubbard_request``, ``periodic::hubbard_table``);
Python hands the mapping over as JSON text. Every refusal here is a ``ValueError`` or a
``NotImplementedError`` with the reason, raised before any SCF. The refusals that need only the
elements and their datasets are raised by the constructor when ``atoms`` is given; the rest at
the first calculation. The numbers of a DFT+U run are pinned in
``test/migration/test_native_hubbard.py``. Needs ``MANDACARU_PAW_PATH``.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mandacaru import Mandacaru
from mandacaru.integrals import Grid
from mandacaru.units import HARTREE_TO_EV

SZ = {"name": "PAW-LCAO", "size": "SZ"}
NI_D = {"Ni": {"l": "d", "U": 3.0, "J": 0.0}}
CU_D = {"Cu": {"l": "d", "U": 4.0}}


def cuh():
    """CuH in a 7 A periodic box (Gamma only)."""
    atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[7.0] * 3, pbc=True)
    atoms.center()
    return atoms


def single(symbol, moment=0.0):
    """One atom of ``symbol`` in a 7 A periodic box."""
    atom = Atoms(symbol, positions=[[0, 0, 0]], cell=[7.0] * 3, pbc=True)
    atom.center()
    atom.set_initial_magnetic_moments([moment])
    return atom


# --------------------------------------------------------------------------- #
# The mapping: shape, values, scope.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec, message",
    [
        ({}, "must map an element symbol"),
        ("Ni", "must map an element symbol"),
        ({"Xx": {"l": "d", "U": 1.0}}, "not an element symbol"),
        ({"X": {"l": "d", "U": 1.0}}, "not an element symbol"),
        ({"Ni": 6.0}, "needs the keys"),
        ({"Ni": {"U": 6.0}}, "needs the keys"),
        ({"Ni": {"l": "d"}}, "needs the keys"),
        ({"Ni": {"l": "d", "U": 6.0, "weight": 1}}, "needs the keys"),
        ({"Ni": {"l": "d", "U": 6.0, "form": "x"}}, "unknown Hubbard form"),
        ({"Ni": {"l": "d", "U": 6.0, "form": 1}}, "form must be a string"),
        ({"Ni": {"l": "p", "U": 6.0}}, "'d' or 'f'"),
        ({"Ni": {"l": 2, "U": 6.0}}, "'d' or 'f'"),
        ({"Ni": {"l": "d", "U": -1.0}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": True}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": float("nan")}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": float("inf")}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": "6"}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": 1.0, "J": -0.1}}, "finite number >= 0"),
        ({"Ni": {"l": "d", "U": 1.0, "J": 2.0}}, "exceeds U"),
    ],
)
def test_a_malformed_option_is_refused_at_construction(spec, message):
    with pytest.raises(ValueError, match=message):
        Mandacaru(method="dft", hubbard_u=spec, basis=SZ, trace=False)


def test_numpy_scalars_are_numbers_and_numpy_booleans_are_not():
    Mandacaru(
        method="dft",
        hubbard_u={"Ni": {"l": "d", "U": np.float32(3.0), "J": np.int64(0)}},
        basis=SZ,
        trace=False,
    )
    with pytest.raises(ValueError, match="finite number >= 0"):
        Mandacaru(method="dft", hubbard_u={"Ni": {"l": "d", "U": np.bool_(True)}}, trace=False)
    with pytest.raises(ValueError, match="must map an element symbol"):
        Mandacaru(method="dft", hubbard_u={("Ni",): {"l": "d", "U": 1.0}}, trace=False)


def test_combinations_the_engine_does_not_implement_are_refused_at_construction():
    with pytest.raises(NotImplementedError, match="hybrid functional 'hse06'"):
        Mandacaru(method="dft", xc="hse06", hubbard_u=NI_D, basis=SZ, trace=False)
    with pytest.raises(ValueError, match="precision='float64'"):
        Mandacaru(method="dft", precision="float32", hubbard_u=NI_D, basis=SZ, trace=False)
    # a geometry given with the calculator: a molecule with non-DFT method
    molecule = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[8.0] * 3)
    with pytest.raises(ValueError, match="does not use: hubbard_u"):
        Mandacaru(method="rhf", atoms=molecule, hubbard_u=CU_D, basis=SZ, trace=False)


@pytest.mark.parametrize("basis", ["HAO", "STO-3G"])
def test_a_basis_without_a_pseudopotential_is_refused_naming_the_option(basis):
    """The shell is a projection on a PAW partial wave: the message says which basis is needed."""
    with pytest.raises(ValueError, match=r"hubbard_u needs a PAW-LCAO or UPAW-LCAO basis"):
        Mandacaru(method="dft", basis=basis, hubbard_u=NI_D, trace=False)
    Mandacaru(method="dft", basis=SZ, hubbard_u=NI_D, trace=False)


# --------------------------------------------------------------------------- #
# The geometry: refused by the constructor when atoms are given, otherwise at the first run.
# --------------------------------------------------------------------------- #


def test_a_molecule_is_refused_for_non_dft():
    molecule = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[8.0] * 3)
    molecule.center()
    with pytest.raises(ValueError, match="does not use: hubbard_u"):
        Mandacaru(method="rhf", hubbard_u=CU_D, trace=False, basis=SZ)


def test_a_molecule_calculates_dft_plus_u():
    molecule = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[8.0] * 3)
    molecule.center()
    calc = Mandacaru(method="dft", xc="lda", hubbard_u=CU_D, trace=False, basis=SZ, grid_spacing=0.3)
    molecule.calc = calc
    energy = molecule.get_potential_energy()
    assert np.isfinite(energy)
    assert calc.result.hubbard_energy is not None
    assert calc.result.hubbard_energy > 0.0
    assert len(calc.result.hubbard) == 1
    shell = calc.result.hubbard[0]
    assert shell["symbol"] == "Cu"
    assert shell["l"] == "d"
    assert np.allclose(shell["trace"], [4.886], atol=1e-2)


def frozen_cuh(spec, shift=0.0, *, spin=False):
    """CuH on a frozen 6 A grid, the H atom displaced by ``shift`` along z (A), for the
    force against its energy."""
    atoms = Atoms("CuH", positions=[[3.013, 2.979, 3.007], [3.013, 2.979, 4.467 + shift]], cell=[6.0] * 3)
    grid = Grid(center=[3.0] * 3, cell=np.eye(3) * 6.0, grid_spacing=0.3)
    options = {"spin": True} if spin else {}
    if spin:
        atoms.set_initial_magnetic_moments([2.0, 0.0])
    atoms.calc = Mandacaru(
        method="dft", xc="lda", basis=SZ, grid=grid, grid_spacing=0.3, hubbard_u=spec,
        project_translation=False, scf_options={"tolerance": 1e-10}, trace=False,
        resources={"threads": 2}, **options,
    )
    return atoms


@pytest.mark.parametrize(
    "spec, spin",
    [
        pytest.param({"Cu": {"l": "d", "U": 4.0, "form": "dudarev-fll"}}, False, id="dudarev"),
        pytest.param({"Cu": {"l": "d", "U": 4.0, "J": 1.0, "form": "liechtenstein-fll"}}, False, id="liechtenstein"),
        pytest.param({"Cu": {"l": "d", "U": 4.0, "form": "dudarev-fll"}}, True, id="dudarev-spin-polarized"),
        pytest.param(
            {"Cu": {"l": "d", "U": 4.0, "J": 1.0, "form": "liechtenstein-fll"}}, True, id="liechtenstein-spin-polarized"
        ),
    ],
)
def test_the_force_of_a_molecule_with_hubbard_u_is_the_derivative_of_its_energy(spec, spin):
    """The +U term of the molecular force (the energy's dependence on the shell projections of
    the moving atom) against a central difference of the +U energy on the frozen grid. Without
    the term the force is off by 0.07 eV/A (restricted) here, 150 times this tolerance; the
    residual with the term is 1.2e-4 eV/A, the step's and the same as with ``U = 0``."""
    h = 0.005
    atoms = frozen_cuh(spec, spin=spin)
    force = atoms.get_forces()[1, 2]
    if spin:
        up, down = atoms.calc.result.hubbard[0]["trace"]
        assert abs(up - down) > 0.3  # a polarized shell, not the restricted limit
    assert atoms.calc.result.force_details["scheme"] == "analytic"
    energies = [frozen_cuh(spec, shift, spin=spin).get_potential_energy() for shift in (h, -h)]
    numerical = -(energies[0] - energies[1]) / (2 * h)
    assert force == pytest.approx(numerical, abs=5e-4)


def test_a_molecule_naming_an_absent_element_raises_instead_of_running_without_u():
    molecule = Atoms("CuH", positions=[[3.0, 3.0, 3.0], [3.0, 3.0, 4.46]], cell=[6.0] * 3)
    molecule.calc = Mandacaru(method="dft", xc="lda", hubbard_u=NI_D, trace=False, basis=SZ, grid_spacing=0.3)
    with pytest.raises(ValueError, match="hubbard_u names Ni, which is not in the system"):
        molecule.get_potential_energy()


def test_a_molecule_whose_dataset_lacks_the_shell_raises_instead_of_running_without_u():
    molecule = Atoms("CuH", positions=[[3.0, 3.0, 3.0], [3.0, 3.0, 4.46]], cell=[6.0] * 3)
    molecule.calc = Mandacaru(
        method="dft", xc="lda", hubbard_u={"Cu": {"l": "f", "U": 4.0}}, trace=False, basis=SZ, grid_spacing=0.3
    )
    with pytest.raises(ValueError, match="does not have channel l = 3"):
        molecule.get_potential_energy()


def test_a_molecule_on_a_spin_orbit_dataset_refuses_hubbard_u():
    """The fully relativistic datasets run spinor Kohn-Sham, which has no +U."""
    molecule = Atoms("CuH", positions=[[3.0, 3.0, 3.0], [3.0, 3.0, 4.46]], cell=[6.0] * 3)
    molecule.calc = Mandacaru(
        method="dft", xc="lda", hubbard_u=CU_D, trace=False, grid_spacing=0.3,
        basis={**SZ, "paw_dataset": "lda-fr"},
    )
    with pytest.raises(NotImplementedError, match="spinor molecular Kohn-Sham is not available for DFT\\+U"):
        molecule.get_potential_energy()


GEOMETRY_REFUSALS = [
    pytest.param(
        lambda: bulk("Si", "diamond", a=5.43),
        NI_D,
        {},
        ValueError,
        "Ni, which is not in the system",
        id="element-absent",
    ),
    pytest.param(
        lambda: single("Ni", 2.0),
        {"Ni": {"l": "f", "U": 3.0}},
        {"spin": True},
        ValueError,
        "Ni dataset has no 'f' channel",
        id="no-shell",
    ),
    pytest.param(
        lambda: single("Ni", 2.0),
        NI_D,
        {"spin": True, "basis": {**SZ, "paw_dataset": "lda-fr"}},
        NotImplementedError,
        "spin-orbit",
        id="spin-orbit",
    ),
    pytest.param(
        lambda: single("Ba"),
        {"Ba": {"l": "d", "U": 3.0}},
        {},
        ValueError,
        "'d' shell of Ba has no occupied reference wave in the lda-sr dataset set",
        id="empty-reference-d",
    ),
    pytest.param(
        lambda: single("Au"),
        {"Au": {"l": "f", "U": 3.0}},
        {},
        ValueError,
        "'f' shell of Au has no occupied reference wave in the lda-sr dataset set",
        id="empty-reference-f",
    ),
]


@pytest.mark.parametrize("make, option, extra, error, message", GEOMETRY_REFUSALS)
def test_a_geometry_the_option_cannot_use_is_refused_at_construction_with_atoms(
    make, option, extra, error, message
):
    options = dict(method="dft", hubbard_u=option, trace=False, grid_spacing=0.4, basis=SZ)
    options.update(extra)
    with pytest.raises(error, match=message):
        Mandacaru(atoms=make(), **options)


@pytest.mark.parametrize("make, option, extra, error, message", GEOMETRY_REFUSALS)
def test_the_same_refusal_comes_at_the_first_calculation_when_atoms_come_later(
    make, option, extra, error, message
):
    options = dict(method="dft", hubbard_u=option, trace=False, grid_spacing=0.4, basis=SZ)
    options.update(extra)
    atoms = make()
    atoms.calc = Mandacaru(**options)
    with pytest.raises(error, match=message):
        atoms.get_potential_energy()


# --------------------------------------------------------------------------- #
# A k mesh reduced by symmetry (HU-4): the occupations are completed over the kept operations.
# --------------------------------------------------------------------------- #

SCF = {"max_iterations": 300, "tolerance": 1e-10, "density_tolerance": 5e-10, "mixing": 0.25, "history": 7,
       "width_ha": 0.05 / HARTREE_TO_EV}


def nickel_triangle(symmetry, mesh=(2, 2, 1)):
    """Three Ni atoms around a threefold axis of a hexagonal cell: 12 operations (D3h) that permute the three Hubbard
    atoms cyclically. The quadrature of the projector spheres is exactly invariant under them, so the reduced mesh
    meets the 1e-10 occupation gate (cubic operations leave a 1e-7 floor, plain DFT included)."""
    a, c, x = 5.0, 3.9, 0.28
    atoms = Atoms("Ni3", cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]], pbc=True,
                  scaled_positions=[[x, 0, 0], [0, x, 0], [-x, -x, 0]])
    atoms.calc = Mandacaru(method="dft", xc="lda", hubbard_u=NI_D, trace=False, basis=SZ, grid_spacing=0.42,
                           kpts={"size": mesh, "gamma": True}, symmetry=symmetry, periodic_scf_options=SCF,
                           smearing={"method": "fermi-dirac", "width": 0.05})
    return atoms


# Each quantity is its own module-scoped fixture, so that the test that first needs it is charged for it alone (the
# stress of the whole mesh is the slow one: 85 s against 45 s reduced, measured at a low machine load).
@pytest.fixture(scope="module")
def triangle_whole():
    atoms = nickel_triangle(False)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def triangle_reduced():
    atoms = nickel_triangle(True)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def force_whole(triangle_whole):
    return triangle_whole.get_forces()


@pytest.fixture(scope="module")
def force_reduced(triangle_reduced):
    return triangle_reduced.get_forces()


@pytest.fixture(scope="module")
def stress_whole(triangle_whole):
    return triangle_whole.get_stress()


@pytest.fixture(scope="module")
def stress_reduced(triangle_reduced):
    return triangle_reduced.get_stress()


@pytest.mark.slow
def test_a_reduced_mesh_gives_the_whole_meshs_energy_and_occupations(triangle_whole, triangle_reduced):
    """2x2x1 is the whole mesh with ``symmetry=False`` (every point is its own time-reversal partner): 4 points, 2 on the
    wedge. The reference shares nothing with the symmetrizer: no operation, no completion."""
    whole, reduced = triangle_whole, triangle_reduced
    assert reduced.calc.periodic_problem.symmetry.n_operations == 12
    assert len(reduced.calc.periodic_problem.kpoints) < len(whole.calc.periodic_problem.kpoints) == 4
    assert reduced.get_potential_energy() == pytest.approx(whole.get_potential_energy(), abs=1e-8)
    for shell_reduced, shell_whole in zip(reduced.calc.result.hubbard, whole.calc.result.hubbard):
        assert np.abs(shell_reduced["n"] - shell_whole["n"]).max() < 1e-10
    assert reduced.calc.result.hubbard_energy > 1e-3


@pytest.mark.slow
def test_the_force_of_a_reduced_mesh_is_the_whole_meshs(force_whole, force_reduced):
    """The +U operator is completed with the occupations, so the explicit +U force term of the wedge sum is covariant and
    the group average of the engine gives the whole mesh's. The three atoms feel a radial force (14 eV/A): 1.4e-10
    measured on 2x2x2, gate 1e-6."""
    assert np.abs(force_whole).max() > 1.0
    assert np.abs(force_reduced - force_whole).max() < 1e-6


@pytest.mark.slow
def test_the_stress_of_a_reduced_mesh_keeps_the_hexagonal_form(stress_reduced):
    """12 operations make the in-plane stress isotropic and the shear zero, to rounding."""
    assert abs(stress_reduced[0] - stress_reduced[1]) < 1e-12
    assert np.abs(stress_reduced[3:]).max() < 1e-12
    assert abs(stress_reduced[0] - stress_reduced[2]) > 1e-3


@pytest.mark.slow
def test_the_stress_of_a_reduced_mesh_is_the_whole_meshs(stress_whole, stress_reduced):
    """5e-11 eV/A^3 measured on 2x2x2, gate 1e-6."""
    assert np.abs(stress_reduced - stress_whole).max() < 1e-6


def test_the_response_of_a_reduced_mesh_refuses_atoms_the_operations_move():
    """The localized shift of one atom breaks the symmetry the wedge and the density completion assume."""
    atoms = nickel_triangle(True)
    atoms.get_potential_energy()
    with pytest.raises(ValueError, match="fixed by every kept operation") as error:
        atoms.calc.hubbard_response()
    assert "symmetry=False" in str(error.value)


@pytest.mark.slow
def test_the_response_of_a_reduced_mesh_is_the_whole_meshs_when_every_hubbard_atom_is_fixed():
    """fcc Ni has one Hubbard atom that all 48 operations fix: the shift keeps the symmetry, so the wedge gives U."""
    results = {}
    for symmetry in (True, False):
        nickel = bulk("Ni", "fcc", a=3.52)
        nickel.set_initial_magnetic_moments([0.0])  # ASE's default moment would keep the full mesh
        nickel.calc = Mandacaru(method="dft", xc="lda", hubbard_u=NI_D, trace=False, basis=SZ, grid_spacing=0.4,
                                kpts={"size": (2, 2, 2), "gamma": True}, symmetry=symmetry, periodic_scf_options=SCF,
                                smearing={"method": "fermi-dirac", "width": 0.05})
        nickel.get_potential_energy()
        results[symmetry] = (len(nickel.calc.periodic_problem.kpoints), nickel.calc.hubbard_response()["u_matrix"])
    assert results[True][0] < results[False][0] == 8
    assert results[True][1] == pytest.approx(results[False][1], abs=1e-6)


# --------------------------------------------------------------------------- #
# A saved crystal file owns the correction.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """CuH crystal files written before the SCF (one iteration is enough to write them)."""
    folder = tmp_path_factory.mktemp("saved")
    options = dict(
        method="dft",
        xc="lda",
        grid_spacing=0.3,
        trace=False,
        basis=SZ,
        symmetry=False,
        smearing={"method": "fermi-dirac", "width": 0.01},
        periodic_scf_options={"max_iterations": 1},
    )
    for name, extra in (("plain", {}), ("four", {"hubbard_u": CU_D})):
        atoms = cuh()
        atoms.calc = Mandacaru(save_crystal=folder / f"{name}.json", **options, **extra)
        atoms.get_potential_energy()
    return folder


def test_the_option_against_a_loaded_crystal_follows_the_mismatch_rule(saved):
    short = dict(method="dft", trace=False, periodic_scf_options={"max_iterations": 1})
    # the file's own definition is accepted (one iteration is enough to pass the check)
    Mandacaru(load_crystal=saved / "four.json", hubbard_u=CU_D, **short).run()
    for option in (
        {"Cu": {"l": "d", "U": 5.0}},
        {"Cu": {"l": "d", "U": 4.0, "J": 1.0}},
        {"Cu": {"l": "f", "U": 4.0}},
        {"Cu": {"l": "d", "U": 4.0}, "Ni": {"l": "d", "U": 4.0}},
    ):
        with pytest.raises(ValueError, match="disagrees with the crystal file"):
            Mandacaru(load_crystal=saved / "four.json", hubbard_u=option, **short).run()
    with pytest.raises(ValueError, match="disagrees with the crystal file"):
        Mandacaru(load_crystal=saved / "plain.json", hubbard_u=CU_D, **short).run()
