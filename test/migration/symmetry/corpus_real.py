# -*- coding: utf-8 -*-
"""Real systems, labelled magnetic cases, pinned test cases, and scale supercells. All deterministic."""

from ase import Atoms
from ase.build import bulk, fcc111, graphene, make_supercell, molecule, mx2
from ase.lattice.hexagonal import Graphite
from ase.spacegroup import crystal
import numpy as np


def _c(cid, cat, at, **kw):
    return dict(
        id=cid,
        cat=cat,
        cell=np.asarray(at.cell[:], float),
        frac=np.asarray(at.get_scaled_positions(wrap=False), float),
        labels=np.asarray(at.get_atomic_numbers(), int),
        pbc=tuple(at.pbc),
        **kw,
    )


def pinned():
    """Cases from test/core/test_symmetry.py."""
    sq = Atoms("H", positions=[[0, 0, 0]], cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 12.0]], pbc=[1, 1, 0])
    a = 2.5
    hexa = Atoms(
        "H",
        positions=[[0, 0, 0]],
        cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 12.0]],
        pbc=[1, 1, 0],
    )
    fcc = Atoms(
        "Cu",
        positions=[[0, 0, 0]],
        cell=0.5 * 4.0 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float),
        pbc=True,
    )
    chain = Atoms("H", positions=[[0, 0, 0]], cell=np.diag([1.0, 8.0, 8.0]), pbc=[1, 0, 0])
    out = [
        _c("pin/square P4/mmm", "pinned", sq, expect=(123, "P4/mmm", 16)),
        _c("pin/hexagonal P6/mmm", "pinned", hexa, expect=(191, "P6/mmm", 24)),
        _c("pin/fcc Fm-3m", "pinned", fcc, expect=(225, "Fm-3m", 48)),
        _c("pin/chain", "pinned", chain, expect=None),
    ]
    return out


def degenerate():
    return _c(
        "pin/degenerate-cell",
        "pinned-failure",
        Atoms("H", positions=[[0, 0, 0]], cell=np.zeros((3, 3)), pbc=True),
        expect="error",
    )


def real_systems():
    out = []

    def add(name, at, **kw):
        out.append(_c("real/" + name, "real", at, **kw))

    add("Al fcc", bulk("Al", "fcc", a=4.05))
    add("Al fcc cubic", bulk("Al", "fcc", a=4.05, cubic=True))
    add("Si diamond", bulk("Si", "diamond", a=5.43))
    add("Si diamond cubic", bulk("Si", "diamond", a=5.43, cubic=True))
    add("Cu fcc", bulk("Cu", "fcc", a=3.61))
    add("Au fcc", bulk("Au", "fcc", a=4.08))
    add("Fe bcc", bulk("Fe", "bcc", a=2.87))
    add("Mg hcp", bulk("Mg", "hcp", a=3.21, c=5.21))
    add("Ti hcp ideal c/a", bulk("Ti", "hcp", a=2.95, c=2.95 * 1.5871))
    add("NaCl rocksalt", bulk("NaCl", "rocksalt", a=5.64))
    add("NiO rocksalt prim", bulk("NiO", "rocksalt", a=4.17))
    add("ZnS zincblende", bulk("ZnS", "zincblende", a=5.41))
    add("CsCl", bulk("CsCl", "cesiumchloride", a=4.12))
    add("ZnO wurtzite", bulk("ZnO", "wurtzite", a=3.25, c=5.2))
    add(
        "SrTiO3 perovskite",
        crystal(
            ["Sr", "Ti", "O"],
            [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
            spacegroup=221,
            cellpar=[3.905] * 3 + [90] * 3,
        ),
    )
    add(
        "Bi2Te3 rhombohedral",
        crystal(
            ["Bi", "Te", "Te"],
            [(0, 0, 0.4), (0, 0, 0), (0, 0, 0.21)],
            spacegroup=166,
            cellpar=[4.38, 4.38, 30.5, 90, 90, 120],
            primitive_cell=True,
        ),
    )
    add(
        "Bi2Te3 hexagonal axes",
        crystal(
            ["Bi", "Te", "Te"],
            [(0, 0, 0.4), (0, 0, 0), (0, 0, 0.21)],
            spacegroup=166,
            cellpar=[4.38, 4.38, 30.5, 90, 90, 120],
        ),
    )
    add("graphene vac", graphene(a=2.46, vacuum=8.0))
    add("graphene vac pbc FFT", graphene(a=2.46, vacuum=8.0))
    m = mx2("MoS2", kind="2H", a=3.16, thickness=3.17, vacuum=5.0)
    add("MoS2 2H slab", m)
    add("MoS2 1T slab", mx2("MoS2", kind="1T", a=3.16, thickness=3.17, vacuum=5.0))
    add("MoS2 slab 2x2", m.repeat((2, 2, 1)))
    add("MoS2 slab sqrt3", make_supercell(m, [[2, 1, 0], [-1, 1, 0], [0, 0, 1]]))
    add("Al(111) slab 4L", fcc111("Al", size=(1, 1, 4), a=4.05, vacuum=8.0))
    add("Cu(111) 2x2x3", fcc111("Cu", size=(2, 2, 3), a=3.61, vacuum=8.0))
    add("graphite", Graphite("C", latticeconstant={"a": 2.46, "c": 6.7}))
    for mol in ("H2O", "CH4", "NH3", "C6H6", "H2", "CO2"):
        at = molecule(mol)
        at.center(vacuum=6.0)
        at.pbc = True
        add("molecule " + mol + " in cubic box", at)
    at = molecule("H2O")
    at.set_cell([9.0, 10.1, 11.3])
    at.center()
    at.pbc = True
    add("H2O in orthorhombic box off-centre", at)
    at = molecule("H2O")
    at.set_cell([9.0, 10.1, 11.3])
    at.center()
    at.pbc = (1, 1, 0)
    add("H2O 2D-pbc box", at)
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    at.pbc = (1, 1, 0)
    add("Si cubic pbc=(T,T,F)", at)
    return out


def labelled():
    """Distinct species/labels/collinear moments. labels are what the port receives: (Z, round(m,6)) folded."""
    out = []
    fe = bulk("Fe", "bcc", a=2.87, cubic=True)  # 2 atoms
    fe.set_initial_magnetic_moments([2.0, 2.0])
    out.append(("ferro", fe))
    fe2 = fe.copy()
    fe2.set_initial_magnetic_moments([2.0, -2.0])
    out.append(("antiferro CsCl-type", fe2))
    nio = bulk("NiO", "rocksalt", a=4.17, cubic=True)
    m = [2.0 if s == "Ni" else 0.0 for s in nio.get_chemical_symbols()]
    nio.set_initial_magnetic_moments(m)
    out.append(("NiO ferro", nio))
    m2 = [(2.0 if i % 2 == 0 else -2.0) if s == "Ni" else 0.0 for i, s in enumerate(nio.get_chemical_symbols())]
    nio2 = nio.copy()
    nio2.set_initial_magnetic_moments(m2)
    out.append(("NiO cubic cell AFM (arbitrary order)", nio2))
    # true type-II AFM along [111] needs the rhombohedral 2-atom-Ni cell: build from fcc111 stacking
    afm = bulk("NiO", "rocksalt", a=4.17)
    afm = afm.repeat((2, 1, 1))
    mm = [(2.0 if i < 2 else -2.0) if s == "Ni" else 0.0 for i, s in enumerate(afm.get_chemical_symbols())]
    afm.set_initial_magnetic_moments(mm)
    out.append(("NiO fcc 2x1x1 AFM along a1", afm))
    sz = bulk("Si", "diamond", a=5.43)
    sz.set_initial_magnetic_moments([0.0, 1.0])
    out.append(("Si two labels (m=0,1) on one Z: Fd-3m -> F-43m", sz))
    return out


def with_labels(at, scheme):
    Z = at.get_atomic_numbers()
    mag = at.get_initial_magnetic_moments()
    if scheme == "Z":
        return Z.astype(int)
    keys = sorted({(int(z), round(float(m), 6)) for z, m in zip(Z, mag)})
    return np.array([keys.index((int(z), round(float(m), 6))) + 1 for z, m in zip(Z, mag)])


def big_supercell(n_target=500):
    """Scale supercells: ~500-atom diamond-Si and an hcp Mg supercell (vacancy-free, exact)."""
    si = bulk("Si", "diamond", a=5.43, cubic=True).repeat((4, 4, 4))  # 8*64 = 512 atoms
    mg = bulk("Mg", "hcp", a=3.21, c=5.21).repeat((8, 8, 4))  # 2*256 = 512 atoms
    sh = si.copy()
    sh.rattle(1e-9, seed=1)
    return [
        _c("scale/Si 512 cubic 4x4x4", "scale", si, expect_ops=48 * 4 * 64 // 4 * 1),
        _c("scale/Mg 512 hcp 8x8x4", "scale", mg),
        _c("scale/Si 512 rattled 1e-9", "scale", sh),
    ]
