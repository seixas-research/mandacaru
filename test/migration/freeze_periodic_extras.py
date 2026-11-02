"""The 'wanted' extras of WP-F item 1: Mg hcp (skew grid, alias averaging, symmetry), r2SCAN Si (core tau) and an
f-channel crystal (compensation multipoles up to L = 6, so rotations for L >= 5).  Called by freeze_periodic_setup.py."""
from __future__ import annotations


def extra_definition(case: str, base: dict):
    import numpy as np
    from ase.build import bulk
    if case == "mg_hcp":
        atoms = bulk("Mg", "hcp", a=3.21, c=5.21)
        opts = dict(base, grid_spacing=.5, kpts={"gamma": True, "size": (2, 2, 2)}, smearing=.3)
        return atoms, opts, dict(skew=True, wedge=True, dataset="lda-sr", xc="lda")
    if case == "si_r2scan":
        atoms = bulk("Si", "diamond", a=5.43)
        opts = dict(base, xc="r2scan", grid_spacing=float(np.linalg.norm(atoms.cell[0])) / 10,
                    kpts={"gamma": False, "size": (2, 2, 2)}, smearing=.01)
        return atoms, opts, dict(shape=(10, 10, 10), generic_k=True, dataset="pbe-sr", xc="r2scan", core_tau=True)
    if case == "f_channel":
        atoms = bulk("W", "bcc", a=3.165, cubic=False)
        opts = dict(base, grid_spacing=float(np.linalg.norm(atoms.cell[0])) / 6, kpts={"gamma": True, "size": (2, 2, 2)}, smearing=.3)
        return atoms, opts, dict(wedge=True, min_L=6, dataset="lda-sr", xc="lda")
    raise SystemExit(f"unknown extra {case!r}")
