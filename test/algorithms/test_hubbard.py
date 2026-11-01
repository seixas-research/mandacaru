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
    # a geometry given with the calculator: a molecule has no periodic direction
    molecule = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[8.0] * 3)
    with pytest.raises(ValueError, match="needs a periodic cell"):
        Mandacaru(method="dft", atoms=molecule, hubbard_u=CU_D, basis=SZ, trace=False)


@pytest.mark.parametrize("basis", ["HAO", "STO-3G"])
def test_a_basis_without_a_pseudopotential_is_refused_naming_the_option(basis):
    """The shell is a projection on a PAW partial wave: the message says which basis is needed."""
    with pytest.raises(ValueError, match=r"hubbard_u needs a PAW-LCAO or UPAW-LCAO basis"):
        Mandacaru(method="dft", basis=basis, hubbard_u=NI_D, trace=False)
    Mandacaru(method="dft", basis=SZ, hubbard_u=NI_D, trace=False)


# --------------------------------------------------------------------------- #
# The geometry: refused by the constructor when atoms are given, otherwise at the first run.
# --------------------------------------------------------------------------- #


def test_a_molecule_is_refused_before_any_calculation():
    molecule = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[8.0] * 3)
    molecule.center()
    molecule.calc = Mandacaru(method="dft", hubbard_u=CU_D, trace=False, basis=SZ)
    with pytest.raises(ValueError, match="needs a periodic cell"):
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


def test_a_crystal_whose_mesh_is_reduced_by_symmetry_is_refused_naming_the_option():
    """Only the assembled crystal knows the mesh was reduced: this one is refused at the run."""
    nickel = bulk("Ni", "fcc", a=3.52)
    nickel.set_initial_magnetic_moments([0.0])  # ASE's default moment would keep the full mesh
    nickel.calc = Mandacaru(
        method="dft",
        hubbard_u=NI_D,
        trace=False,
        basis=SZ,
        grid_spacing=0.4,
        kpts={"size": (2, 2, 2), "gamma": True},
    )
    with pytest.raises(ValueError, match="symmetry=False"):
        nickel.get_potential_energy()


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
