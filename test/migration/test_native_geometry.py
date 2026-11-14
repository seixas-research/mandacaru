"""The lattice geometry the engine computes for the ASE integration (``geometry::*``,
``kpath::interpolate``), against ASE's own routines as the independent oracle.

Every cell here is triclinic or hexagonal: a symmetric cell hides a swap of rows and columns.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.cell import Cell
from ase.neighborlist import neighbor_list

from mandacaru import _native

TRICLINIC = np.array([[3.0, 0.2, -0.1], [0.5, 2.7, 0.3], [0.4, -0.6, 4.1]])


def triclinic_atoms(pbc=(True, True, True)):
    positions = [[0.2, 0.3, 0.4], [1.1, 1.3, 2.0], [-0.5, 4.2, 3.3], [4.0, -1.0, 7.5]]
    return Atoms("OHHC", positions=positions, cell=TRICLINIC, pbc=pbc)


@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False), (False, False, False)])
@pytest.mark.parametrize("wrap", [True, False])
def test_fractional_coordinates_are_ases_with_and_without_wrapping(pbc, wrap):
    atoms = triclinic_atoms(pbc)
    got = _native.fractional_coordinates(
        np.ascontiguousarray(atoms.cell.array), np.ascontiguousarray(atoms.positions), list(pbc), wrap)
    np.testing.assert_allclose(got, atoms.get_scaled_positions(wrap=wrap), atol=1e-13)


def test_a_cell_with_one_zero_vector_is_completed_like_ases_and_two_are_refused():
    atoms = Atoms("H2", positions=[[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]],
                  cell=[[3.0, 0.4, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 0.0]], pbc=(True, True, False))
    got = _native.fractional_coordinates(
        np.ascontiguousarray(atoms.cell.array), np.ascontiguousarray(atoms.positions),
        list(atoms.pbc), False)
    np.testing.assert_allclose(got, atoms.get_scaled_positions(wrap=False), atol=1e-13)
    with pytest.raises(ValueError, match="two zero vectors"):
        _native.fractional_coordinates(
            np.array([[3.0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.zeros((1, 3)), [True, False, False], False)


def test_the_supercell_is_ases_repeat_with_its_moments_tiled():
    atoms = triclinic_atoms()
    atoms.set_initial_magnetic_moments([1.0, 0.0, -2.0, 0.5])
    numbers, positions, cell, moments = _native.repeat_cell(
        [int(z) for z in atoms.numbers], np.ascontiguousarray(atoms.positions),
        np.ascontiguousarray(atoms.cell.array), [float(m) for m in atoms.get_initial_magnetic_moments()],
        [2, 1, 3])
    big = atoms.repeat((2, 1, 3))
    assert list(numbers) == list(big.numbers)
    np.testing.assert_allclose(positions, big.positions, atol=1e-13)
    np.testing.assert_allclose(cell, big.cell.array, atol=1e-13)
    np.testing.assert_allclose(moments, big.get_initial_magnetic_moments())
    with pytest.raises(ValueError):
        _native.repeat_cell([1], np.zeros((1, 3)), TRICLINIC, [0.0], [0, 1, 1])


@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False), (False, False, False)])
def test_the_neighbors_within_a_cutoff_are_ases_neighbor_list(pbc):
    atoms = triclinic_atoms(pbc)
    got = _native.neighbor_vectors(
        np.ascontiguousarray(atoms.cell.array), list(pbc), np.ascontiguousarray(atoms.positions), 6.0)
    first, vectors = neighbor_list("iD", atoms, 6.0)
    for i, found in enumerate(got):
        want = vectors[first == i]
        assert found.shape == want.shape, (i, found.shape, want.shape)
        order = lambda a: a[np.lexsort(np.round(a, 9).T[::-1])]  # noqa: E731
        np.testing.assert_allclose(order(found), order(want), atol=1e-10)


def test_the_wannier_neighbors_of_a_crystal_include_the_periodic_images():
    atoms = bulk("Si", "diamond", a=5.43)
    got = _native.neighbor_vectors(
        np.ascontiguousarray(atoms.cell.array), [True] * 3, np.ascontiguousarray(atoms.positions), 2.5)
    assert [len(v) for v in got] == [4, 4]
    assert all(abs(np.linalg.norm(v, axis=1) - 2.3511).max() < 1e-3 for v in got)


def test_the_reciprocal_columns_of_the_engine_are_ases_rows_times_two_pi():
    """The convention the DOS binding relies on: the engine's reciprocal vectors are the columns
    of ``2 pi inverse(cell)``, ASE's ``cell.reciprocal()`` is their transpose without ``2 pi``."""
    context = _native.Context(threads=1)
    columns = context.periodic_reciprocal_vectors(np.ascontiguousarray(TRICLINIC.T))
    np.testing.assert_allclose(columns, 2.0 * np.pi * Cell(TRICLINIC).reciprocal().array.T, atol=1e-13)
    np.testing.assert_allclose(Cell(TRICLINIC).reciprocal().array.T, np.linalg.inv(TRICLINIC), atol=1e-13)


def test_the_tetrahedron_dos_of_a_skewed_cell_counts_every_state_and_ignores_a_rotation():
    context = _native.Context(threads=1)
    size = (4, 4, 4)
    fractional = context.kmesh(list(size), gamma=True, time_reversal=False)["full"]
    k = np.asarray(fractional, dtype=float)
    levels = np.ascontiguousarray(
        (np.cos(2 * np.pi * k[:, 0]) + np.cos(2 * np.pi * k[:, 1]) + 0.5 * np.cos(2 * np.pi * k[:, 2]))
        .reshape(1, -1, 1))
    axis = np.linspace(-3.0, 3.0, 121)
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    def dos(cell):
        return _native.tetrahedron_dos(context, list(size), np.ascontiguousarray(cell), fractional,
                                       levels, axis)

    plain, turned = dos(TRICLINIC), dos(TRICLINIC @ rotation.T)
    assert plain["integrated"][-1] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(plain["dos"], turned["dos"], atol=1e-10)


# --------------------------------------------------------------------------- #
# The band path: the engine's k-points, labels and named points against ASE's.
# --------------------------------------------------------------------------- #

def _ase_path(cell, path, npoints, pbc=(True, True, True)):
    return Cell(cell).bandpath(path=path, npoints=npoints, pbc=pbc)


def _bismuth_telluride():
    from ase.spacegroup import crystal

    return crystal(symbols=["Te", "Bi", "Te"], basis=[(0, 0, 0.2117), (0, 0, 0.3990), (0, 0, 0.0)],
                   spacegroup=166, cellpar=[4.383, 4.383, 30.487, 90, 90, 120],
                   primitive_cell=True).cell.array


def _mos2():
    from ase.build import mx2

    return mx2("MoS2", kind="2H", a=3.16, thickness=3.17, vacuum=5.0).cell.array


LATTICES = {
    "cubic": (np.diag([2.0, 2.0, 2.0]), "GXMGRX,MR", (True,) * 3),
    "fcc": (bulk("Si", "diamond", a=5.43).cell.array, "GXWKGLUWLK,UX", (True,) * 3),
    "bcc": (bulk("Fe", "bcc", a=2.87).cell.array, "GHNGPH,PN", (True,) * 3),
    "hexagonal": (Cell.fromcellpar([3.0, 3.0, 5.0, 90, 90, 120]).array, "GMKGALHA,LM,KH", (True,) * 3),
    "tetragonal": (np.diag([3.0, 3.0, 4.5]), "GXMGZRAZ,XR,MA", (True,) * 3),
    "rhombohedral_acute": (Cell.fromcellpar([4, 4, 4, 70, 70, 70]).array, "GLB1,BZGX,QFP1Z,LP", (True,) * 3),
    "rhombohedral_obtuse": (Cell.fromcellpar([4, 4, 4, 100, 100, 100]).array, "GPZQGFP1Q1LZ", (True,) * 3),
    "bismuth_telluride": (_bismuth_telluride(), "ZGFZL", (True,) * 3),
    "copper": (bulk("Cu", "fcc", a=3.61).cell.array, "GXWLGK", (True,) * 3),
    "mos2": (_mos2(), "GMKG", (True, True, False)),
    "graphene": (Cell.fromcellpar([2.46, 2.46, 15.0, 90, 90, 120]).array, "GMKG", (True, True, False)),
}


@pytest.mark.parametrize("name", list(LATTICES))
@pytest.mark.parametrize("npoints", [20, 61])
def test_the_band_path_is_ases_cell_bandpath(name, npoints):
    cell, path, pbc = LATTICES[name]
    got = _native.ase_band_path(np.ascontiguousarray(cell), list(pbc), npoints, path)
    want = _ase_path(cell, path, npoints, pbc)
    assert got["path"] == want.path
    np.testing.assert_allclose(got["kpoints"], want.kpts, atol=1e-12)
    for label, point in want.special_points.items():
        np.testing.assert_allclose(got["special_points"][label], point, atol=1e-12)
    # the cumulative lengths along the path, and at its vertices, are `paths2kpts`'
    from ase.dft.kpoints import paths2kpts, resolve_kpt_path_string

    paths, coords = resolve_kpt_path_string(got["path"], want.special_points)
    _kpts, x, vertices = paths2kpts(coords, Cell(cell), npoints)
    np.testing.assert_allclose(got["distances"], x, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got["vertex_distances"], vertices, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", ["cubic", "fcc", "hexagonal", "rhombohedral_acute", "rhombohedral_obtuse"])
def test_the_default_path_of_a_lattice_is_the_one_ase_picks(name):
    cell, path, pbc = LATTICES[name]
    got = _native.ase_band_path(np.ascontiguousarray(cell), list(pbc), 40, None)
    assert got["path"] == path
    np.testing.assert_allclose(got["kpoints"], _ase_path(cell, None, 40, pbc).kpts, atol=1e-12)


def test_the_two_dimensional_defaults_are_ases_too():
    """The square (X on the second axis, ``MGXM``), the rectangle (``GXSYGS``), the chain and the
    60 degree hexagonal plane (M on the second axis) were aligned with ASE's conventions, so one
    definition serves ``band_path`` and ``band_structure``."""
    plane = [True, True, False]
    for cell, pbc in [(np.diag([3.0, 3.0, 15.0]), plane), (np.diag([3.0, 4.0, 15.0]), plane),
                      (Cell.fromcellpar([2.46, 2.46, 15, 90, 90, 60]).array, plane),
                      (Cell.fromcellpar([2.46, 2.46, 15, 90, 90, 120]).array, plane),
                      (np.diag([3.0, 15.0, 15.0]), [True, False, False])]:
        got = _native.ase_band_path(np.ascontiguousarray(cell), pbc, 25, None)
        want = _ase_path(cell, None, 25, tuple(pbc))
        assert got["path"] == want.path
        np.testing.assert_allclose(got["kpoints"], want.kpts, atol=1e-12)
        for label, point in want.special_points.items():
            np.testing.assert_allclose(got["special_points"][label], point, atol=1e-12)
    assert _native.ase_band_path(np.diag([3.0, 3.0, 15.0]), plane, 5)["special_points"]["X"] == [0.0, 0.5, 0.0]


def test_explicit_points_make_a_path_on_any_lattice_as_ases_custom_points_do():
    cell = Cell.fromcellpar([3.0, 4.0, 5.0, 70, 80, 85]).array         # triclinic: no convention
    points = [("A", (0.0, 0.0, 0.0)), ("B", (0.5, 0.0, 0.25)), ("C", (0.5, 0.5, 0.5))]
    got = _native.ase_band_path(np.ascontiguousarray(cell), [True] * 3, 30, None, points)
    ase = Cell(cell).bandpath("ABC", npoints=30, special_points={k: list(v) for k, v in points})
    assert got["path"] == "ABC" and got["labels"] == ["A", "B", "C"]
    np.testing.assert_allclose(got["kpoints"], ase.kpts, atol=1e-12)
    assert got["special_points"] == {k: list(v) for k, v in points}
    unlabeled = _native.ase_band_path(np.ascontiguousarray(cell), [True] * 3, 5, None,
                                      [("", (0, 0, 0)), ("", (0.5, 0, 0))])
    assert unlabeled["path"] == "" and unlabeled["special_points"] == {}
    with pytest.raises(ValueError, match="both|not both"):
        _native.ase_band_path(np.ascontiguousarray(cell), [True] * 3, 5, "GX", points)


def test_a_lattice_without_a_convention_raises_naming_its_lengths_and_angles():
    triclinic = Cell.fromcellpar([3.0, 4.0, 5.0, 70, 80, 85]).array
    with pytest.raises(ValueError, match="explicit points|oblique|triclinic|lattice"):
        _native.ase_band_path(np.ascontiguousarray(triclinic), [True] * 3, 20, None)
