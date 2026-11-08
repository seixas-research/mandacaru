"""The ``setup`` block of a crystal snapshot, and the request ``integrations/setup_block.py`` makes.

The block is written by the Rust setup next to the ``sites`` block and parsed and checked in Rust
(``periodic::description``); here is its content, read from a snapshot a one-iteration run saved (the SCF is not needed).
Needs ``MANDACARU_PAW_PATH``.
"""
import json
import os
import pathlib

import pytest
from ase import Atoms
from ase.build import bulk

from mandacaru import Mandacaru

pytestmark = pytest.mark.skipif(not os.environ.get("MANDACARU_PAW_PATH"), reason="MANDACARU_PAW_PATH is not set")

BASIS = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}


def snapshot(atoms, path, basis=BASIS, **options):
    atoms.calc = Mandacaru(method="dft", basis=basis, trace=False, save_crystal=path,
                           periodic_scf_options={"max_iterations": 1}, **options)
    try:
        atoms.get_potential_energy()
    except Exception:  # noqa: BLE001 - one iteration does not converge; the snapshot is saved before the SCF
        if not path.exists():
            raise
    return json.loads(path.read_text())


def test_the_block_describes_the_basis_the_projectors_and_the_ion_width(tmp_path):
    atoms = Atoms("LiH", cell=[6.0, 6.2, 6.4], pbc=True, positions=[[2.0, 2.1, 2.2], [3.5, 2.2, 2.0]])
    payload = snapshot(atoms, tmp_path / "lih.json", grid_spacing=0.45, symmetry=False)
    block = payload["setup"]
    assert block["version"] == 1 and block["operations"] is None
    functions = block["functions"]
    assert len(functions["atom"]) == len(functions["m"]) == len(functions["table"]) == payload["m"]
    assert set(functions["atom"]) == {0, 1}
    assert max(functions["table"]) < len(block["tables"])
    for table in block["tables"]:
        assert len(table["r"]) == len(table["values"]) >= 4
        assert all(b > a for a, b in zip(table["r"], table["r"][1:]))
    assert len(block["projectors"]) == payload["p"]
    assert all(len(row) == 4 for row in block["projectors"])
    assert block["sigma_bohr"] > 0
    # The datasets, their digests and the centers are the sites block's, not repeated here.
    assert {"centers", "datasets", "projector_basis"} <= set(payload["sites"])
    assert not {"datasets", "centers"} & set(block)


def test_a_reduced_mesh_lists_its_operations_with_their_atom_maps(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    payload = snapshot(atoms, tmp_path / "al.json", grid_spacing=0.45, kpts={"size": (3, 3, 3), "gamma": True},
                       smearing={"method": "fermi-dirac", "width": 0.3}, symmetry=True)
    operations = payload["setup"]["operations"]
    count = len(payload["permutations"])
    assert count == 48
    assert len(operations["rotations"]) == len(operations["translations"]) == len(operations["atom_maps"]) == count
    assert all(len(w) == 9 and all(isinstance(x, int) for x in w) for w in operations["rotations"])
    assert all(sorted(m) == [0] for m in operations["atom_maps"])


# --------------------------------------------------------------------------- #
# The Rust crystal basis honors the confinement options as the molecular builder does (MG-3).
# --------------------------------------------------------------------------- #


def _rust_tables(tmp_path, symbol, options):
    """The radial tables (in build order) of a one-atom Rust crystal, read from its saved snapshot."""
    atoms = Atoms(symbol, cell=[9.0] * 3, pbc=True, positions=[[4.5, 4.5, 4.5]])
    path = tmp_path / f"{symbol}.json"
    payload = snapshot(atoms, path, basis=options, grid_spacing=0.45, symmetry=False,
                       kpts={"size": (1, 1, 1), "gamma": True})
    return payload["setup"]["tables"]


#: The retired Python-orchestrated builder's tables (``orbitals.pseudo_basis`` at v26.11.13),
#: frozen 2026-11-08 at every 4th node: the oracle both routes are held to.
FROZEN = json.loads((pathlib.Path(__file__).resolve().parents[1] / "migration" / "reference"
                     / "basis_recipe.json").read_text())["cases"]
CRYSTAL_CASES = [i for i, c in enumerate(FROZEN) if "diffuse" not in c["options"]]


def _close(r, values, frozen, label):
    import numpy as np
    r, values = np.asarray(r)[::frozen["stride"]], np.asarray(values)[::frozen["stride"]]
    assert np.allclose(r, frozen["r"], rtol=0, atol=1e-12), label
    want = np.asarray(frozen["values"])
    assert np.max(np.abs(values - want)) < 1e-8 * np.max(np.abs(want)), label


@pytest.mark.parametrize("case", range(len(FROZEN)),
                         ids=[f"{c['symbol']}-{i}" for i, c in enumerate(FROZEN)])
def test_the_molecular_basis_reproduces_the_retired_python_builder(case):
    """``pseudo_basis`` (the Rust recipe) equals the frozen Python builder shell for shell."""
    from mandacaru.basis.filtering import resolve_filter_cutoff
    from mandacaru.pseudopotentials.orbitals import pseudo_basis
    from mandacaru.pseudopotentials.paw import get_paw
    record = FROZEN[case]
    symbol, options = record["symbol"], dict(record["options"])
    cutoff = (None if options["filter_cutoff"] is False
              else resolve_filter_cutoff(options["filter_cutoff"]))
    functions, _ = pseudo_basis([symbol], [[0.0, 0.0, 0.0]], {symbol: get_paw(symbol, "lda-sr")},
                                options=options, filter_wavevector=cutoff)
    shells, seen = [], set()
    for f in functions:
        key = (f.n, f.l, f.zeta, f.polarization, f.diffuse)
        if key not in seen:
            seen.add(key)
            shells.append(f)
    assert [(f.n, f.l, f.zeta, f.polarization, f.diffuse) for f in shells] == [
        (s["n"], s["l"], s["zeta"], s["polarization"], s["diffuse"]) for s in record["shells"]]
    for f, frozen in zip(shells, record["shells"]):
        _close(f.table.r, f.table.values, frozen, (symbol, f.n, f.l, f.zeta))


@pytest.mark.parametrize("case", CRYSTAL_CASES,
                         ids=[f"{FROZEN[i]['symbol']}-{i}" for i in CRYSTAL_CASES])
def test_the_crystal_basis_reproduces_the_retired_python_builder(tmp_path, case):
    """The Rust crystal setup's tables equal the frozen Python builder's for the same options."""
    record = FROZEN[case]
    tables = _rust_tables(tmp_path, record["symbol"], {"name": "PAW-LCAO", **record["options"]})
    assert [t["l"] for t in tables] == [s["l"] for s in record["shells"]]
    for table, frozen in zip(tables, record["shells"]):
        _close(table["r"], table["values"], frozen, (record["symbol"], frozen["n"], frozen["l"]))


def test_the_options_reach_the_rust_basis(tmp_path):
    """Non-vacuity of the parity above: a smaller shift and a softer wall change the tables."""
    import numpy as np
    default = _rust_tables(tmp_path, "Si", {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200})
    for change in ({"energy_shift": 0.02}, {"confinement_potential": (3.0, 0.3)}):
        changed = _rust_tables(tmp_path, "Si", {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200,
                                                 **change})
        values = [np.asarray(t["values"]) for t in default], [np.asarray(t["values"]) for t in changed]
        assert any(a.shape != b.shape or np.max(np.abs(a - b)) > 1e-3 for a, b in zip(*values)), change


def test_a_dual_projector_crystal_builds_and_its_force_is_refused_by_name(tmp_path):
    """``projector_basis="dual"`` builds the crystal in the dual basis (the snapshot says so); its
    energy equals the raw crystal's, the basis being a change of projector coordinates; the
    analytic force engine works in the raw basis and refuses the dual one, naming it."""
    energies = {}
    for basis in ("raw", "dual"):
        atoms = Atoms("LiH", cell=[6.0, 6.2, 6.4], pbc=True, positions=[[2.0, 2.1, 2.2], [3.5, 2.2, 2.0]])
        atoms.calc = Mandacaru(method="dft", basis={**BASIS, "projector_basis": basis},
                               grid_spacing=0.45, symmetry=False, trace=False,
                               save_crystal=tmp_path / f"{basis}.json",
                               smearing={"method": "fermi-dirac", "width": 0.1})
        energies[basis] = atoms.get_potential_energy()
        assert json.loads((tmp_path / f"{basis}.json").read_text())["sites"]["projector_basis"] == basis
        if basis == "dual":
            with pytest.raises(NotImplementedError, match="dual"):
                atoms.get_forces()
    assert energies["dual"] == pytest.approx(energies["raw"], abs=1e-6)


@pytest.mark.parametrize("size", ["SZ", "DZP"])
def test_the_paw_lcao_energy_shift_defaults_to_a_tenth_of_an_ev(tmp_path, size):
    """Leaving ``energy_shift`` out is ``energy_shift=0.1`` (eV), for a crystal and a molecule alike
    (an absent key once read as unconfined on the crystal route)."""
    import numpy as np
    from mandacaru.pseudopotentials.confinement import DEFAULT_ENERGY_SHIFT
    from mandacaru.pseudopotentials.orbitals import pseudo_basis
    from mandacaru.pseudopotentials.paw import get_paw
    assert DEFAULT_ENERGY_SHIFT == 0.1
    base = {"name": "PAW-LCAO", "size": size, "filter_cutoff": 200}
    for name in ("implicit", "explicit"):
        (tmp_path / name).mkdir()
    implicit = _rust_tables(tmp_path / "implicit", "Si", base)
    explicit = _rust_tables(tmp_path / "explicit", "Si", {**base, "energy_shift": 0.1})
    assert [t["values"] for t in implicit] == [t["values"] for t in explicit]
    pp = {"Si": get_paw("Si", "lda-sr")}
    molecule = [pseudo_basis(["Si"], [[0.0, 0.0, 0.0]], pp, options=options)[0]
                for options in ({"size": size}, {"size": size, "energy_shift": 0.1})]
    for a, b in zip(*molecule):
        np.testing.assert_array_equal(a.table.values, b.table.values)
    # and the crystal's default is the molecule's, table for table
    assert len(implicit) == len({(f.n, f.l, f.zeta, f.polarization) for f in molecule[0]})
