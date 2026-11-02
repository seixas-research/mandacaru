"""The periodic end-result cases (WP-F freeze item 3) and the worker that solves one of them.

``freeze_periodic_results.py`` is the driver; it spawns ``python freeze_periodic_cases.py CASE PRECISION THREADS`` once per
run, so every SCF has a process of its own and a clean peak.  The worker prints one ``RESULT {json}`` line.

Groups.  ``lda9``: the nine cases of ``reference/precision/periodic_energies.json`` (the shifted mesh ``kpts=(n, n, n)``, the
basis default filter).  ``lda_k3``: Si SZ grid_spacing=0.15 on the Gamma-centered 3x3x3 mesh.  ``companion``: the end results of the
boundary snapshots of ``freeze_periodic_setup.py`` (same atoms and options, so a Rust snapshot test can be cross-checked
against an energy).  ``extra``: Si SZ grid_spacing=0.3 on the shifted 2x2x2 mesh for PBE, r2SCAN, PBE+rVV10, the ``lda-fr`` set and HSE06
(Al for the second PBE), whose old numbers in a scratchpad were computed while crystals read the ``lda-sr`` set for PBE
(``prior_wrong_dataset``).  ``skew``: the two H cells of ``test_native_periodic.py``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCF = {"tolerance": 1e-12, "density_tolerance": 1e-10}
MB = 1024 ** 2
SKEW_CELL = [[4., .4, 0.], [0., 4., .3], [.1, 0., 4.5]]
LATTICE = {"Al": ("fcc", 4.05), "Si": ("diamond", 5.43)}
#: ``(symbol, size, grid_spacing, k)`` in the order of ``periodic_energies.json["cases"]``.
LDA9 = (("Al", "SZ", .30, 2), ("Al", "SZ", .30, 4), ("Al", "DZP", .30, 2), ("Al", "DZP", .25, 4), ("Al", "SZ", .25, 2),
        ("Si", "SZ", .30, 2), ("Si", "SZ", .30, 4), ("Si", "DZP", .30, 2), ("Si", "SZ", .25, 2))
#: Old scratchpad numbers (extension 39cf62c7, crystals then read lda-sr for PBE): context only, never a gate.
PRIOR_WRONG = {
    "extra_si_pbe": dict(E_eV=-214.47518194673447, energy=-7.881817556611394, free_energy=-7.88181756347414, n_iter=21),
    "extra_al_pbe": dict(E_eV=-56.13004821856553, energy=-2.060986865901114, free_energy=-2.064495585537114, n_iter=17),
    "extra_si_r2scan": dict(E_eV=-214.50269460297764, energy=-7.882828630653878, free_energy=-7.882828632368034, n_iter=19),
    "extra_si_pbe_rvv10": dict(E_eV=-214.333308572856, energy=-7.876603805912491, free_energy=-7.876603813523433, n_iter=22)}
#: Not stale (the set is named explicitly); the same extension and settings.
PRIOR_STILL_VALID = {"extra_si_lda_fr": dict(E_eV=-213.38118350217175, energy=-7.841613854667971,
                                              free_energy=-7.841613862819763, n_iter=19)}
EXTRAS = (("si_pbe", "Si", "pbe", None), ("al_pbe", "Al", "pbe", None), ("si_r2scan", "Si", "r2scan", None),
          ("si_pbe_rvv10", "Si", "pbe+rvv10", None), ("si_lda_fr", "Si", "lda", "lda-fr"),
          ("si_hse06", "Si", "hse06", None))
COMPANIONS = ("si_3x3x3", "si_shifted", "al_fcc", "al_fcc_gamma", "si_so", "si_so_scalar")
SKEWS = ("h2_skew_pbe", "h_skew_spin_lda")


def lda9_id(index: int) -> str:
    sym, size, grid_spacing, k = LDA9[index]
    return f"lda9_{sym.lower()}_{size.lower()}_spacing{grid_spacing:.2f}_k{k}"


def case_ids() -> list[str]:
    return ([lda9_id(i) for i in range(len(LDA9))] + ["lda_si_sz_k3_spacing0.15"] + [f"companion_{c}" for c in COMPANIONS]
            + [f"extra_{e[0]}" for e in EXTRAS] + ["extra_si_hse06_full_mesh"] + [f"skew_{s}" for s in SKEWS])


def definition(case: str):
    """``(atoms, Mandacaru options, meta)``; the options exclude ``precision`` and ``resources``."""
    import numpy as np
    from ase import Atoms
    from ase.build import bulk
    basis = lambda size="SZ", **extra: {"name": "PAW-LCAO", "size": size, **extra}
    meta = dict(metal=False, species={}, reference=None)
    if case.startswith("lda9_"):
        index = [lda9_id(i) for i in range(len(LDA9))].index(case)
        sym, size, grid_spacing, k = LDA9[index]
        atoms = bulk(sym, LATTICE[sym][0], a=LATTICE[sym][1])
        opts = dict(method="dft", xc="lda", basis=basis(size), grid_spacing=grid_spacing, kpts=(k, k, k))
        return atoms, opts, dict(meta, metal=sym == "Al", frozen_index=index, dataset="lda-sr", xc="lda")
    if case == "lda_si_sz_k3_spacing0.15":
        atoms = bulk("Si", "diamond", a=5.43)
        opts = dict(method="dft", xc="lda", basis=basis(), grid_spacing=.15, kpts={"size": (3, 3, 3), "gamma": True})
        return atoms, opts, dict(meta, dataset="lda-sr", xc="lda", memory_table_E_eV=-213.89170586640074)
    if case.startswith("extra_"):
        name = case[len("extra_"):]
        if name == "si_hse06_full_mesh":
            atoms = bulk("Si", "diamond", a=5.43)
            opts = dict(method="dft", xc="hse06", basis=basis(), grid_spacing=.3, kpts={"size": (1, 1, 1), "gamma": True},
                        symmetry=False)
            return atoms, opts, dict(meta, dataset="pbe-sr", xc="hse06", provisional_not_a_gate=True)
        _, sym, xc, dataset = next(e for e in EXTRAS if e[0] == name)
        atoms = bulk(sym, LATTICE[sym][0], a=LATTICE[sym][1])
        opts = dict(method="dft", xc=xc, basis=basis(), grid_spacing=.3, kpts=(2, 2, 2))
        if dataset:
            opts["paw_dataset"] = dataset
        folder = dataset or ("lda-sr" if xc == "lda" else "pbe-sr")
        return atoms, opts, dict(meta, metal=sym == "Al", dataset=folder, xc=xc, provisional_not_a_gate=name == "si_hse06")
    if case.startswith("companion_"):
        name = case[len("companion_"):]
        base = dict(method="dft", xc="lda", basis=basis(filter_cutoff=200))
        si = bulk("Si", "diamond", a=5.43)
        grid_spacing = float(np.linalg.norm(si.cell[0])) / 10
        if name in ("si_3x3x3", "si_shifted"):
            mesh = {"gamma": True, "size": (3, 3, 3)} if name == "si_3x3x3" else {"gamma": False, "size": (2, 2, 2)}
            return si, dict(base, grid_spacing=grid_spacing, kpts=mesh, smearing=.01), dict(meta, dataset="lda-sr", xc="lda")
        if name in ("si_so", "si_so_scalar"):
            fr = name == "si_so"
            opts = dict(base, grid_spacing=grid_spacing, kpts={"gamma": True, "size": (2, 1, 1)}, smearing=.01,
                        paw_dataset="lda-fr" if fr else "lda-sr")
            return si, opts, dict(meta, dataset="lda-fr" if fr else "lda-sr", xc="lda", spinor_expected=fr)
        al = bulk("Al", "fcc", a=4.05)
        size = (2, 2, 2) if name == "al_fcc" else (1, 1, 1)
        opts = dict(base, grid_spacing=float(np.linalg.norm(al.cell[0])) / 7, kpts={"gamma": True, "size": size})
        return al, opts, dict(meta, metal=True, dataset="lda-sr", xc="lda")
    if case.startswith("skew_"):
        name = case[len("skew_"):]
        cell = np.array(SKEW_CELL)
        kw = dict(basis=basis(filter_cutoff=200), grid_spacing=.7, kpts={"gamma": True, "size": (2, 1, 1)}, smearing=.3)
        if name == "h2_skew_pbe":
            atoms = Atoms("H2", positions=[[1.2, 1.6, 1.9], [2., 2.3, 2.1]], cell=cell, pbc=True)
            return atoms, dict(method="dft", xc="pbe", spin=False, **kw), dict(meta, dataset="pbe-sr", xc="pbe")
        atoms = Atoms("H", positions=[[1.2, 1.6, 1.9]], cell=cell, pbc=True)
        atoms.set_initial_magnetic_moments([1.])
        return atoms, dict(method="dft", xc="lda", spin=True, **kw), dict(meta, dataset="lda-sr", xc="lda", spins=2)
    raise SystemExit(f"unknown case {case!r}")


def dataset_files(atoms, folder: str) -> list[str]:
    import os
    root = os.environ["MANDACARU_PAW_PATH"]
    return sorted({str(Path(root) / folder / f"{s}.parquet") for s in atoms.get_chemical_symbols()})


def solve(case: str, precision: str, threads: int) -> dict:
    """Run one SCF in this process and return the frozen quantities (Hartree; nested lists)."""
    import resource
    import numpy as np
    from freeze_stamp import stamp
    from mandacaru import Mandacaru
    from mandacaru.pseudopotentials.paw import get_paw
    atoms, opts, meta = definition(case)
    files = dataset_files(atoms, meta["dataset"])
    out = dict(case=case, precision=precision, threads=threads, options={k: repr(v) for k, v in opts.items()},
               meta={k: v for k, v in meta.items() if k not in ("species",)}, symbols=atoms.get_chemical_symbols(),
               stamp=stamp(datasets=files, extra={"case": case, "precision": precision, "threads": threads}))
    start = time.time()
    try:
        calc = Mandacaru(**opts, precision=precision, periodic_scf_options=dict(SCF), trace=False,
                         resources={"threads": threads, "memory_limit_bytes": 900 * MB})
        atoms.calc = calc
        e_ev = float(atoms.get_potential_energy())
        sn = calc.result.scf.snapshot()
        w = np.asarray(sn["weights"], dtype=float)
        occ = np.asarray(sn["occupations"], dtype=float)
        eig = np.asarray(sn["eigenvalues"], dtype=float)
        electrons = float(sum(get_paw(s, meta["dataset"]).valence_charge for s in atoms.get_chemical_symbols()))
        out.update(
            ok=True, E_eV=e_ev, energy=float(sn["energy"]), free_energy=float(sn["free_energy"]),
            extrapolated_energy=float(sn["extrapolated_energy"]),
            spin_orbit_energy=None if sn["spin_orbit_energy"] is None else float(sn["spin_orbit_energy"]), eigenvalues=eig.tolist(), eigenvalues_shape=list(eig.shape),
            occupation_sum=float((w[None, :, None] * occ).sum()), expected_electrons=electrons,
            fermi_level_reported=float(sn["fermi_level"]), moment=float(sn["magnetic_moment"]), n_iter=int(sn["n_iterations"]),
            converged=bool(sn["converged"]), n_spins=int(sn["n_spins"]), spinor=bool(sn["spinor"]),
            kpoints_fractional=np.asarray(sn["fractional_kpoints"]).tolist(), weights=w.tolist(),
            terms={k: float(v) for k, v in sn["terms"].items()}, last_history=list(map(float, sn["history"][-1])),
            context_peak_bytes=int(calc.context.memory_peak_bytes))
    except Exception as exc:                      # a case that cannot run says why; the driver stores it
        out.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:400]}")
    scale = 1 if sys.platform == "darwin" else 1024
    out.update(wall_s=round(time.time() - start, 2), rss_peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale / MB)
    return out


if __name__ == "__main__":
    case, precision, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
    print("RESULT " + json.dumps(solve(case, precision, threads), allow_nan=False))
