"""Freeze the periodic PAW setup (WP-F items 1 and 2) from the live Python tree.

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set::

    python test/migration/freeze_periodic_setup.py all                 # every snapshot, floors, diagnostics
    python test/migration/freeze_periodic_setup.py snapshot si_so      # one boundary snapshot
    python test/migration/freeze_periodic_setup.py diagnostics         # item 2 (see freeze_periodic_diagnostics.py)
    python test/migration/freeze_periodic_setup.py list

Item 1 writes ``phase5/<case>.json`` in the ``save_crystal`` format (loadable by Rust), item 2
``reference/periodic/<topic>.json``.  The output directory (``--output``) defaults to ``test/migration``.
Every case runs in its own process (``all`` spawns them after ``scripts/memguard.py``); a case builds the
snapshot with ``Mandacaru(..., save_crystal=...)``, solves it once, reloads the file in Rust, asserts the
non-vacuity conditions and stamps it (``freeze_stamp.stamp``).  ``floor`` repeats the build in two more
processes (same threads, then 2 threads) and stores the run-to-run difference per array group.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SNAPSHOT_CASES = ("si_3x3x3", "si_shifted", "al_fcc", "al_fcc_gamma", "si_so", "h2_skew_pbe", "h_spin_lda")
EXTRA_CASES = ("mg_hcp", "si_r2scan", "f_channel")
SCF = {"tolerance": 1e-12, "density_tolerance": 1e-10}
SKEW_CELL = [[4., .4, 0.], [0., 4., .3], [.1, 0., 4.5]]
#: Arrays whose floor is reported together: the plan's tolerance classes.
GROUPS = {"T2_k_arrays": ("psi", "overlap", "fixed", "projections", "spin_orbit"),
          "T1_grid": ("core", "initial", "initial_m", "kernel", "xc_wave", "phase", "kerker", "ion", "hats", "u",
                      "uion", "linear", "blocks", "initial_q", "qsym")}
SCALARS = ("eion", "ion_constant", "constant", "reference", "electrons")


def repo_root() -> Path:
    from freeze_stamp import repo_root as root
    return root()


def definition(case: str):
    """``(atoms, calculator options, meta)`` of a snapshot case."""
    import numpy as np
    from ase import Atoms
    from ase.build import bulk
    basis = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}
    base = dict(method="dft", xc="lda", basis=basis, trace=False, periodic_scf_options=dict(SCF))
    if case.startswith("si_") and case not in ("si_so", "si_r2scan"):
        atoms = bulk("Si", "diamond", a=5.43)
        mesh = {"gamma": True, "size": (3, 3, 3)} if case == "si_3x3x3" else {"gamma": False, "size": (2, 2, 2)}
        opts = dict(base, grid_spacing=float(np.linalg.norm(atoms.cell[0])) / 10, kpts=mesh, smearing=.01)
        return atoms, opts, dict(shape=(10, 10, 10), generic_k=True, wedge=True, dataset="lda-sr", xc="lda")
    if case == "si_so":
        atoms = bulk("Si", "diamond", a=5.43)
        opts = dict(base, grid_spacing=float(np.linalg.norm(atoms.cell[0])) / 10, paw_dataset="lda-fr",
                    kpts={"gamma": True, "size": (2, 1, 1)}, smearing=.01)
        return atoms, opts, dict(shape=(10, 10, 10), spinor=True, dataset="lda-fr", xc="lda")
    if case in ("al_fcc", "al_fcc_gamma"):
        atoms = bulk("Al", "fcc", a=4.05)
        size = (2, 2, 2) if case == "al_fcc" else (1, 1, 1)
        opts = dict(base, grid_spacing=float(np.linalg.norm(atoms.cell[0])) / 7, kpts={"gamma": True, "size": size}, smearing=.3)
        return atoms, opts, dict(shape=(7, 7, 7), wedge=size != (1, 1, 1), dataset="lda-sr", xc="lda")
    cell = np.array(SKEW_CELL)
    if case == "h2_skew_pbe":
        atoms = Atoms("H2", positions=[[1.2, 1.6, 1.9], [2., 2.3, 2.1]], cell=cell, pbc=True)
        opts = dict(base, xc="pbe", spin=False, grid_spacing=.7, kpts={"gamma": True, "size": (2, 1, 1)}, smearing=.3)
        return atoms, opts, dict(skew=True, dataset="pbe-sr", xc="pbe")
    if case == "h_spin_lda":
        atoms = Atoms("H", positions=[[1.2, 1.6, 1.9]], cell=cell, pbc=True)
        atoms.set_initial_magnetic_moments([1.])
        opts = dict(base, spin=True, grid_spacing=.7, kpts={"gamma": True, "size": (2, 1, 1)}, smearing=.3)
        return atoms, opts, dict(skew=True, spins=2, dataset="lda-sr", xc="lda")
    if case in EXTRA_CASES:
        from freeze_periodic_extras import extra_definition
        return extra_definition(case, base)
    raise SystemExit(f"unknown case {case!r}; known: {SNAPSHOT_CASES + EXTRA_CASES}")


def decode(flat, complex_valued=None):
    """A numpy array from the snapshot's ``[re, im]`` pairs or flat floats."""
    import numpy as np
    a = np.asarray(flat, dtype=float)
    if complex_valued is None:
        complex_valued = a.ndim == 2 and a.shape[-1] == 2
    return a[..., 0] + 1j * a[..., 1] if complex_valued else a


def unit_fraction(value) -> bool:
    return abs(value - round(value)) < 1e-9


def non_vacuity(case, meta, payload, calc):
    """Assertions that a wrong k.R sign / conjugation / spin block changes the stored array."""
    import numpy as np
    out = {}
    M = payload["m"]
    ks = payload["kpoints"]
    if meta.get("generic_k"):
        crystal = calc._crystal_adapter
        rows = []
        for row in ks:
            frac = np.asarray(row["fractional"])
            if all(unit_fraction(2 * f) for f in frac):
                continue                   # k = -k modulo G: Gamma-like, complex harmonics alone give Im psi
            psi, S = decode(row["psi"]).reshape(M, -1), decode(row["overlap"]).reshape(M, M)
            minus = crystal.kpoint_matrices(-np.asarray(row["k"])[None])[0]
            scale = float(np.abs(psi).max())
            rows.append(dict(
                fractional=frac.tolist(), max_im_S=float(np.abs(S.imag).max()), max_im_psi_rel=float(np.abs(psi.imag).max() / scale),
                psi_minus_k_vs_k_rel=float(np.abs(minus.psi - psi).max() / scale),
                psi_minus_k_vs_conj_rel=float(np.abs(minus.psi - psi.conj()).max() / scale),
                S_minus_k_vs_k=float(np.abs(minus.overlap - S).max()), S_minus_k_vs_conj=float(np.abs(minus.overlap - S.conj()).max())))
        assert rows, f"{case}: no k with k != -k"
        for r in rows:
            for key in ("max_im_S", "max_im_psi_rel", "psi_minus_k_vs_k_rel", "psi_minus_k_vs_conj_rel",
                        "S_minus_k_vs_k", "S_minus_k_vs_conj"):
                assert r[key] > 1e-3, f"{case}: vacuous generic-k fixture, {key}={r[key]:.2e} at {r['fractional']}"
        out["generic_k"] = rows
    if meta.get("spinor"):
        assert payload.get("spinor") is True and payload["time_reversal"] is True
        blocks = []
        for row in ks:
            h = decode(row["spin_orbit"]).reshape(2, M, 2, M)
            fixed = decode(row["fixed"]).reshape(M, M)
            norm, flip = float(np.abs(h).max()), float(np.abs(h[0, :, 1, :]).max())
            swapped = float(np.abs(h[::-1, :, ::-1, :] - h).max())
            assert norm > 1e-4 and flip > 1e-4, f"{case}: spin-orbit block vanishes ({norm:.2e}, {flip:.2e})"
            assert swapped > 1e-4, f"{case}: alpha/beta swap leaves the block unchanged ({swapped:.2e})"
            assert np.abs(h.reshape(2 * M, 2 * M) - h.reshape(2 * M, 2 * M).conj().T).max() < 1e-13
            blocks.append(dict(fractional=row["fractional"], max_abs=norm, max_abs_spin_flip=flip, alpha_beta_swap_change=swapped,
                               drop_block_change=norm, max_abs_fixed=float(np.abs(fixed).max())))
        out["spin_orbit"] = blocks
    if meta.get("skew"):
        step = np.asarray(payload["grid"]["step"]).reshape(3, 3)
        assert np.abs(step - np.diag(np.diag(step))).max() > 1e-3, f"{case}: orthogonal grid in a skew case"
        out["grid_step_offdiagonal_max"] = float(np.abs(step - np.diag(np.diag(step))).max())
    if "spins" in meta:
        assert payload["spins"] == meta["spins"] and payload["permutations"] == []
        assert float(np.abs(decode(payload["initial_m"])).max()) > 0.0
        out["spins"] = payload["spins"]
    if meta.get("wedge"):
        assert payload["permutations"], f"{case}: no symmetry operation survives, the wedge is trivial"
        out["n_operations"] = len(payload["permutations"])
    if meta.get("min_L"):
        top = max(c[1] for c in payload["channel_labels"])
        assert top >= meta["min_L"], f"{case}: highest compensation L is {top}"
        out["max_L"] = top
    if meta.get("core_tau"):
        out["tau_keys"] = sorted(k for k in payload if "tau" in k)
        assert out["tau_keys"], f"{case}: the snapshot carries no core tau"
    if meta.get("shape"):
        assert tuple(payload["grid"]["shape"]) == meta["shape"], (payload["grid"]["shape"], meta["shape"])
    assert payload["functional"] == meta["xc"]
    return out


def snapshot(case: str, output: Path, threads: int):
    for name in ("MANDACARU_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    if not os.environ.get("MANDACARU_PAW_PATH"):
        raise SystemExit("set MANDACARU_PAW_PATH to the mandacaru-paw checkout")
    from mandacaru import Mandacaru, _native
    from mandacaru.pseudopotentials.paw import get_paw
    from freeze_stamp import stamp
    atoms, opts, meta = definition(case)
    target = output / "phase5" / f"{case}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    atoms.calc = Mandacaru(**opts, resources={"threads": threads}, save_crystal=str(target))
    error = None
    try:
        energy = atoms.get_potential_energy()
    except Exception as exc:            # the snapshot is written before the solve
        energy, error = None, f"{type(exc).__name__}: {exc}"
    payload = json.loads(target.read_text())
    checks = non_vacuity(case, meta, payload, atoms.calc)
    ctx = _native.Context(threads=threads)
    native = None
    if error is None:
        # The file alone must reproduce the energy of the run that wrote it (same smearing and tolerances).
        res = ctx.load_crystal(str(target)).solve(
            smearing="fermi-dirac", width_ha=opts["smearing"] / 27.211386245988, **SCF).snapshot()
        native = dict(converged=bool(res["converged"]), extrapolated_energy_ha=float(res["extrapolated_energy"]),
                      free_energy_ha=float(res["free_energy"]), magnetic_moment=float(res.get("magnetic_moment", 0.0)),
                      ase_minus_native_ev=energy - float(res["extrapolated_energy"]) * 27.211386245988)
        assert native["converged"] and abs(native["ase_minus_native_ev"]) < 2.7e-9, native
    loadable = ctx.load_crystal(str(target)) is not None
    datasets = [get_paw(s, paw_dataset=meta["dataset"]).source for s in sorted(set(atoms.get_chemical_symbols()))]
    payload["provenance"].update(
        reference_case=case, generator="test/migration/freeze_periodic_setup.py", energy_ev=energy, scf_error=error,
        stamp=stamp(datasets=datasets, extra=dict(case=case, threads=threads, options=json.loads(json.dumps(
            {k: v for k, v in opts.items() if k not in ("periodic_scf_options",)}, default=str)),
            loadable_by_rust=loadable, native_reload=native, non_vacuity=checks,
            n_kpoints=len(payload["kpoints"]), grid_shape=payload["grid"]["shape"])))
    target.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(case=case, path=str(target), bytes=target.stat().st_size, energy_ev=energy, error=error,
                          native=native, nk=len(payload["kpoints"])), default=str))


def group_floor(a: dict, b: dict) -> dict:
    """Run-to-run difference of two payloads, relative to each array's maximum, maximum per group."""
    import numpy as np

    def rel(x, y):
        x, y = decode(x), decode(y)
        return float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-300)) if x.size else 0.0
    out = {g: 0.0 for g in GROUPS}
    for g, names in GROUPS.items():
        for n in names:
            if n in a and a[n]:
                out[g] = max(out[g], rel(a[n], b[n]))
    for ka, kb in zip(a["kpoints"], b["kpoints"]):
        for n in GROUPS["T2_k_arrays"]:
            if n in ka:
                out["T2_k_arrays"] = max(out["T2_k_arrays"], rel(ka[n], kb[n]))
        out["T0_k_and_weight_abs"] = max(out.get("T0_k_and_weight_abs", 0.0),
                                         float(np.abs(np.array(ka["k"]) - kb["k"]).max()), abs(ka["weight"] - kb["weight"]))
    out["scalars_abs"] = max(abs(a[s] - b[s]) for s in SCALARS)
    out["permutations_same_rows_as_set"] = sorted(map(tuple, a["permutations"])) == sorted(map(tuple, b["permutations"]))
    out["permutations_same_order"] = a["permutations"] == b["permutations"]
    out["energy_ev_abs"] = abs(a["provenance"]["energy_ev"] - b["provenance"]["energy_ev"]) \
        if a["provenance"]["energy_ev"] is not None else None
    out["channel_labels_exact"] = a["channel_labels"] == b["channel_labels"] and a["grid"] == b["grid"]
    return out


def floor(case: str, output: Path, work: Path):
    """Two more builds in fresh processes (1 and 2 threads); stores the differences in the stamp."""
    target = output / "phase5" / f"{case}.json"
    main = json.loads(target.read_text())
    threads = main["provenance"]["stamp"]["extra"]["threads"]
    result = {}
    for tag, n in (("repeat_same_threads", threads), ("two_threads", 2 if threads != 2 else 1)):
        folder = work / f"{case}_{tag}"
        run_guarded([sys.executable, str(HERE / "freeze_periodic_setup.py"), "snapshot", case, "--output", str(folder),
                     "--threads", str(n)])
        other = folder / "phase5" / f"{case}.json"
        result[tag] = dict(threads=n, **group_floor(main, json.loads(other.read_text())))
        other.unlink()
    main["provenance"]["stamp"]["floor"] = result
    target.write_text(json.dumps(main, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(case=case, floor=result)))


def run_guarded(command):
    root = repo_root()
    guard = root / "scripts" / "memguard.py"
    if guard.exists():
        subprocess.run([sys.executable, str(guard), "--need-gb", "2", "--wait", "120"], cwd=root, check=True)
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("all", "snapshot", "floor", "diagnostics", "list"))
    parser.add_argument("case", nargs="?", help="a case or topic; for 'all', 'extras' adds the three wanted extras")
    parser.add_argument("--output", type=Path, default=None, help="default: test/migration of the repository")
    parser.add_argument("--work", type=Path, default=None, help="scratch folder of the floor repeats")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    output = args.output or repo_root() / "test" / "migration"
    work = args.work or Path(os.environ.get("TMPDIR", "/tmp")) / "freeze_periodic_work"
    me = [sys.executable, str(HERE / "freeze_periodic_setup.py")]
    if args.action == "list":
        print("\n".join(SNAPSHOT_CASES + EXTRA_CASES))
    elif args.action == "snapshot":
        snapshot(args.case, output, args.threads)
    elif args.action == "floor":
        floor(args.case, output, work)
    elif args.action == "diagnostics":
        from freeze_periodic_diagnostics import TOPICS
        for topic in ([args.case] if args.case else TOPICS):
            run_guarded([sys.executable, str(HERE / "freeze_periodic_diagnostics.py"), topic, "--output", str(output),
                         "--work", str(work)])
    else:
        cases = SNAPSHOT_CASES + tuple(c for c in EXTRA_CASES if args.case == "extras")
        for case in cases:
            run_guarded(me + ["snapshot", case, "--output", str(output)])
            run_guarded(me + ["floor", case, "--output", str(output), "--work", str(work)])
        run_guarded(me + ["diagnostics", "--output", str(output), "--work", str(work)])


if __name__ == "__main__":
    main()
