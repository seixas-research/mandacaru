# -*- coding: utf-8 -*-
"""Corpus generator, reproducible from one integer seed. Case = dict(id, cat, cell, frac, labels, n_ase, ...)."""

import itertools
import warnings

from ase import Atoms
from ase.spacegroup import Spacegroup, crystal
import numpy as np

SEED = 20261030
MAXN = 400  # corpus atoms cap (keeps one process well under 1 GB)
R_GROUPS = {146, 148, 155, 160, 161, 166, 167}


def system(n):
    return (
        "triclinic"
        if n <= 2
        else "monoclinic"
        if n <= 15
        else "orthorhombic"
        if n <= 74
        else "tetragonal"
        if n <= 142
        else "trigonal"
        if n <= 167
        else "hexagonal"
        if n <= 194
        else "cubic"
    )


def settings():
    out = []
    for n in range(1, 231):
        for s in (1, 2):
            try:
                Spacegroup(n, s)
                out.append((n, s))
            except Exception:
                pass
    return out


def cellpar(n, setting, rng, kind="generic"):
    """Cell parameters compatible with the lattice metric of (n,setting). kind='special' picks metric-
    degenerate values (c=a, c/a ideal ...) whose extra metric symmetry the atoms must not exhibit.
    """
    sy = system(n)
    a = rng.uniform(3.3, 6.0)
    b = a * rng.uniform(1.12, 1.55)
    c = a * rng.uniform(1.6, 2.1)
    sp = kind == "special"
    if sy == "triclinic":
        return [a, b, c, *(rng.uniform(70, 110, 3) if not sp else (90, 90, 90))]
    if sy == "monoclinic":
        be = rng.uniform(96, 118) if not sp else 90.0
        rots = Spacegroup(n, setting).get_op()[0]
        proper = [W for W in rots if round(np.linalg.det(W)) == 1 and np.trace(W) != 3]
        mirrors = [W for W in rots if round(np.linalg.det(W)) == -1 and np.trace(W) == 1]
        ax = (
            [i for i in range(3) if proper[0][i, i] == 1][0]
            if proper
            else [i for i in range(3) if mirrors[0][i, i] == -1][0]
        )  # the 2-fold axis, or the mirror normal
        cp = [a, b, c, 90, 90, 90]
        cp[3 + ax] = be  # angle opposite the unique axis
        return cp
    if sy == "orthorhombic":
        return [a, b, c, 90, 90, 90] if not sp else [a, a, c, 90, 90, 90]
    if sy == "tetragonal":
        return [a, a, c if not sp else a, 90, 90, 90]
    if sy == "trigonal":
        if n in R_GROUPS and setting == 2:
            al = rng.uniform(52, 78) if not sp else 60.0
            return [a, a, a, al, al, al]
        return [a, a, c, 90, 90, 120]
    if sy == "hexagonal":
        return [a, a, c if not sp else a * 1.6329931618554521, 90, 90, 120]
    return [a, a, a, 90, 90, 90]


SPECIAL_CANDIDATES = None


def _candidates():
    global SPECIAL_CANDIDATES
    if SPECIAL_CANDIDATES is None:
        S = (0.0, 0.25, 0.5, 0.75)
        pts = [p for p in itertools.product(S, repeat=3)]
        pts += [(1 / 3, 2 / 3, z) for z in (0.0, 0.25, 0.5)] + [
            (2 / 3, 1 / 3, z) for z in (0.0, 0.25, 0.5)
        ]
        u, v = 0.1234, 0.3187
        pts += [
            (u, 0, 0),
            (0, u, 0),
            (0, 0, u),
            (u, u, 0),
            (u, 0, u),
            (0, u, u),
            (u, u, u),
            (u, -u, 0),
            (u, 2 * u, 0),
            (u, 0, 0.25),
            (u, 0.25, 0),
            (0, u, 0.25),
            (u, u, 0.25),
            (u, 2 * u, 0.25),
            (u, v, 0),
            (u, v, 0.25),
            (u, v, 0.5),
            (u, 0, v),
            (0, u, v),
            (u, 0.5, 0),
            (0.5, u, 0),
            (u, u, 0.5),
            (u, 0.25, 0.25),
            (0.25, u, u),
            (u, -u, u),
            (u, 2 * u, u),
            (u, 0.5, v),
            (u, u, v),
            (u, -u, v),
        ]
        SPECIAL_CANDIDATES = pts
    return SPECIAL_CANDIDATES


def cellvol(cp):
    from ase.geometry import cellpar_to_cell

    return abs(np.linalg.det(cellpar_to_cell(cp)))


def multiplicity(sg, p):
    sites, _ = sg.equivalent_sites([p], symprec=1e-6)
    return len(sites)


def special_sites(sg, rng, max_kinds=3):
    """One site per distinct multiplicity (smallest first), at most max_kinds, none the general one."""
    gen = len(sg.get_op()[0])
    best = {}
    for p in _candidates():
        m = multiplicity(sg, p)
        if m < gen and m not in best:
            best[m] = p
    return [(m, best[m]) for m in sorted(best)[:max_kinds]], gen


def general_site(rng):
    return tuple(rng.uniform(0.07, 0.43, 3) + rng.integers(0, 2, 3) * 0.5 * rng.integers(0, 2, 3))


def build(n, setting, sites, species, cp, primitive=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return crystal(
            species,
            sites,
            spacegroup=n,
            setting=setting,
            cellpar=cp,
            primitive_cell=primitive,
            onduplicates="replace",
            symprec=1e-6,
        )


def too_close(at, dmin=0.6):
    if len(at) < 2:
        return False
    d = at.get_all_distances(mic=True)
    d[np.diag_indices(len(at))] = 9
    return d.min() < dmin


# ---------------------------------------------------------------- cell / atom transformations (pure functions)
def unimodular(rng, steps=4):
    M = np.eye(3, dtype=int)
    for _ in range(steps):
        i, j = rng.choice(3, 2, replace=False)
        E = np.eye(3, dtype=int)
        E[i, j] = rng.choice([-2, -1, 1, 2])
        M = E @ M
    return M


def transform(at, kind, rng):
    """at: ASE Atoms -> new Atoms describing the SAME crystal."""
    cell = at.cell[:].copy()
    frac = at.get_scaled_positions(wrap=False)
    Z = at.get_atomic_numbers()
    if kind == "conv" or kind == "prim":
        return at
    if kind == "s211":
        return at.repeat((2, 1, 1))
    if kind == "s222":
        return at.repeat((2, 2, 2))
    if kind == "perm":  # cyclic permutation (right-handed)
        P = [1, 2, 0]
        return Atoms(Z, scaled_positions=frac[:, P], cell=cell[P], pbc=True)
    if kind == "swap":  # (b, a, -c): right-handed, flips a row sign
        P = [1, 0, 2]
        sgn = np.array([1, 1, -1])
        return Atoms(Z, scaled_positions=frac[:, P] * sgn, cell=cell[P] * sgn[:, None], pbc=True)
    if kind == "skew":  # unimodular change of basis: cell' = M cell, frac' = frac M^-1
        M = unimodular(rng)
        return Atoms(Z, scaled_positions=frac @ np.linalg.inv(M), cell=M @ cell, pbc=True)
    if kind == "shift":  # origin shift putting one atom exactly on 0, lattice wrapping, 1e-13 jitter, shuffle
        f = frac - frac[rng.integers(len(frac))]
        f = f + rng.integers(-2, 3, f.shape) + rng.choice([-1e-13, 0.0, 1e-13], f.shape)
        m = np.abs(f - np.round(f)) < 1e-11  # atoms on a cell face: put them just inside or just outside
        f[m] = np.round(f[m]) + rng.choice([-1e-14, 1e-14, 0.0], m.sum())
        perm = rng.permutation(len(f))
        return Atoms(Z[perm], scaled_positions=f[perm], cell=cell, pbc=True)
    if kind == "rot":  # rigid Cartesian rotation: explicit Rodrigues formula (platform independent)
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        th = rng.uniform(0.3, 2.8)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Q = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        return Atoms(Z, scaled_positions=frac, cell=cell @ Q.T, pbc=True)
    raise KeyError(kind)


def case_dict(cid, cat, at, **meta):
    cell = np.asarray(at.cell[:], float)
    frac = np.asarray(at.get_scaled_positions(wrap=False), float)
    return dict(id=cid, cat=cat, cell=cell, frac=frac, labels=np.asarray(at.get_atomic_numbers(), int), **meta)


# ---------------------------------------------------------------- (i) space-group corpus
def space_group_cases(seed=SEED, only=None, max_atoms_super=260, extra_cells=True):
    for n, s in settings():
        if only is not None and (n, s) not in only:
            continue
        rng0 = np.random.default_rng([seed, n, s])
        sg = Spacegroup(n, s)
        cp = cellpar(n, s, rng0)
        V0 = cellvol(cp)
        base = None
        # general position, two species at two generic sites (reject close contacts: redraw)
        for attempt in range(30):
            g1, g2 = general_site(rng0), general_site(rng0)
            try:
                a = build(n, s, [g1, g2], ["Na", "Cl"], cp)
            except Exception:
                continue
            if not too_close(a):
                base = a
                break
        if base is not None:
            yield case_dict(
                f"sg{n}.{s}/gen/conv",
                "sg-general",
                base,
                n_ase=n,
                setting=s,
                vconv=V0,
                sites="general",
                variant="conv",
                cellpar=cp,
            )
            prim = build(n, s, [g1, g2], ["Na", "Cl"], cp, primitive=True)
            yield case_dict(
                f"sg{n}.{s}/gen/prim",
                "sg-general",
                prim,
                n_ase=n,
                setting=s,
                vconv=V0,
                sites="general",
                variant="prim",
            )
            for kind in ("perm", "swap", "skew", "shift", "rot") + (
                ("s211",) if len(base) * 2 <= MAXN else ()
            ) + (("s222",) if len(base) * 8 <= max_atoms_super else ()):
                rr = np.random.default_rng([seed, n, s, sum(map(ord, kind))])
                at = transform(
                    base if len(base) <= 100 or kind in ("perm", "swap", "rot") else prim,
                    kind,
                    rr,
                )
                yield case_dict(
                    f"sg{n}.{s}/gen/{kind}",
                    "sg-variant",
                    at,
                    n_ase=n,
                    setting=s,
                    vconv=V0,
                    sites="general",
                    variant=kind,
                )
            if extra_cells:
                cp2 = cellpar(n, s, np.random.default_rng([seed, n, s, 99]))
                try:
                    b2 = build(n, s, [g1, g2], ["Na", "Cl"], cp2)
                    if not too_close(b2):
                        yield case_dict(
                            f"sg{n}.{s}/gen/conv2",
                            "sg-general",
                            b2,
                            n_ase=n,
                            setting=s,
                            vconv=cellvol(cp2),
                            sites="general",
                            variant="conv",
                            cellpar=cp2,
                        )
                except Exception:
                    pass
                cp3 = cellpar(n, s, np.random.default_rng([seed, n, s, 7]), kind="special")
                try:
                    b3 = build(n, s, [g1, g2], ["Na", "Cl"], cp3)
                    if not too_close(b3):
                        yield case_dict(
                            f"sg{n}.{s}/gen/metric-special",
                            "sg-metric-special",
                            b3,
                            n_ase=n,
                            setting=s,
                            vconv=cellvol(cp3),
                            sites="general",
                            variant="conv",
                            cellpar=cp3,
                        )
                except Exception:
                    pass
        # special positions
        specials, gen = special_sites(sg, rng0)
        for m, p in specials:
            try:
                a = build(n, s, [p], ["Na"], cp)
            except Exception:
                continue
            if too_close(a):
                continue
            yield case_dict(
                f"sg{n}.{s}/sp{m}only/conv",
                "sg-special-only",
                a,
                n_ase=n,
                setting=s,
                vconv=V0,
                sites=f"special m={m}",
                variant="conv",
            )
            try:
                a = build(n, s, [p, general_site(rng0)], ["Na", "Cl"], cp)
            except Exception:
                continue
            if too_close(a):
                continue
            yield case_dict(
                f"sg{n}.{s}/sp{m}/conv",
                "sg-special",
                a,
                n_ase=n,
                setting=s,
                vconv=V0,
                sites=f"special m={m}+general",
                variant="conv",
            )
            if m == specials[0][0]:
                for kind in ("skew", "shift", "s222"):
                    rr = np.random.default_rng([seed, n, s, 1000 + sum(map(ord, kind))])
                    yield case_dict(
                        f"sg{n}.{s}/sp{m}/{kind}",
                        "sg-variant",
                        transform(a, kind, rr),
                        n_ase=n,
                        setting=s,
                        vconv=V0,
                        sites=f"special m={m}",
                        variant=kind,
                    )
