#!/usr/bin/env python
"""HU-0: freeze the OLD DFT+U numbers before the Rust port exists.

The old stack is the Python of commit a0fd3b5 (v26.10.38, the last tree with
``algorithms/hubbard.py``), restored from git into a scratch folder and run on
the CURRENT native extension.  Caveat, stamped into every record: it is the old
Python on the new extension, not the old extension.  The pure functions (shell
weights, E_U, V_U) and the closed-form models are the exact gates; the SCF
numbers are a parity target whose replay floor is in ``reproducibility``.

Always use the ``mandacaru`` interpreter and set ``MANDACARU_PAW_PATH``:

    python test/migration/freeze_hubbard.py restore /tmp/hubbard_old       # git archive a0fd3b5 + the current .so
    python test/migration/freeze_hubbard.py list
    python test/migration/freeze_hubbard.py verify --old /tmp/hubbard_old  # one command: replay the cheap cases
    python test/migration/freeze_hubbard.py freeze --old /tmp/hubbard_old [--cases a,b] [--out DIR] [--no-repro]

``verify`` replays every cheap case in a fresh interpreter (4 threads): the
weights and the exactly solvable models must match BITWISE; an SCF case must
match within the measured replay floor (energies ~1e-9 eV, n ~1e-9, see
``VERIFY_TOL``): the SCF is not bitwise at this level (4e-12 eV run to run, the
same at 1 and 4 threads).  A difference there is a finding, not a failure to
explain away.  The reference is ``reference/hubbard/frozen.json`` next to this
file; the occupation matrices and eigenvalues are in git-ignored ``.npz`` files.

Memory: every case runs as a monitored child (``top`` mem + compressed <= 1000 MB, the
owner's gate; RSS is reported too and the child is killed above 1.3 x the cap).  A case
over the cap is recorded as "not frozen: exceeds the cap at X MB" and is never retried
unchanged.  ``scripts/memguard.py`` runs before every child.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_COMMIT = "a0fd3b5"
PYTHON = sys.executable
THREADS = 4                       # the freeze thread count (repro records 1)
MEMGUARD = HERE.parents[1] / "scripts" / "memguard.py"    # present when this file sits in test/migration
CAP_MB = 1000.0                   # RSS + compressed, per process
NIO_SETTINGS = dict(a=4.17, h=0.25, kpts=(2, 2, 2), width=0.05, xc="lda", size="SZ")

# --------------------------------------------------------------------------- #
# Cases (name -> spec); spec "kind" picks the child implementation.
# --------------------------------------------------------------------------- #


def _cases() -> dict:
    c: dict = {}
    c["weights"] = dict(kind="weights", cheap=True)
    c["models"] = dict(kind="models", cheap=True)
    for u in (0.0, 3.0, 6.0):
        c[f"ni_atom_U{u:g}"] = dict(kind="scf", system="ni_atom", U=u, seed="A", cheap=True)
    # n at U -> 0; not in verify: two empty minority d levels sit at E_F of a degenerate atom, n varies by 7e-7 and
    # the iteration count by 2 between identical runs (frozen, floor recorded, never gated)
    c["ni_atom_U0_probe"] = dict(kind="scf", system="ni_atom", U=1e-9, seed="A", cheap=False)
    for u in (0.0, 3.0, 6.0):
        for seed in ("A", "B", "C", "D"):
            if u == 0.0 and seed in ("C", "D"):
                continue
            c[f"nio_U{u:g}_{seed}"] = dict(kind="scf", system="nio", U=u, seed=seed,
                                           cheap=(seed == "A"))
    c["nio_U0_probe"] = dict(kind="scf", system="nio", U=1e-9, seed="A", cheap=True)
    for u in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):         # extra grid for E(U), dF/dU
        if f"nio_U{u:g}_A" not in c:
            c[f"nio_U{u:g}_A"] = dict(kind="scf", system="nio", U=u, seed="A", cheap=False)
    # Hellmann-Feynman steps around U0 (Richardson in U)
    for u0 in (1.0, 3.0, 5.0):
        for d in (0.2, 0.1):
            for sgn in (-1, 1):
                u = round(u0 + sgn * d, 6)
                nm = f"nio_U{u:g}_A"
                c.setdefault(nm, dict(kind="scf", system="nio", U=u, seed="A", cheap=False))
    # CuH Gamma-box crystal named in the plan: 9 A, h 0.3.  Its RSS peaks at ~1.1 GB (any thread count) while
    # top's mem+cmprs (the owner's gate) is ~270 MB; the 7 A variant (RSS 0.8 GB) is the cheap verify case.
    for u in (0.0, 4.0):
        c[f"cuh_crystal_L9_U{u:g}"] = dict(kind="scf", system="cuh_crystal", U=u, seed="A", L=9.0, cheap=False)
    for u in (0.0, 4.0):
        c[f"cuh_crystal_L7_U{u:g}"] = dict(kind="scf", system="cuh_crystal", U=u, seed="A", L=7.0, cheap=True)
    c["cuh_crystal_L7_spin_U4"] = dict(kind="scf", system="cuh_crystal_spin", U=4.0, seed="A", L=7.0, cheap=True)
    for u in (0.0, 4.0):
        c[f"cuh_molecule_U{u:g}"] = dict(kind="scf", system="cuh_molecule", U=u, seed="A", cheap=True)
        c[f"cuh_molecule_L9_U{u:g}"] = dict(kind="scf", system="cuh_molecule", U=u, seed="A", L=9.0, cheap=False)
    return c


CASES = _cases()
REPRO = ["nio_U3_A", "nio_U0_A", "nio_U6_A", "ni_atom_U3", "ni_atom_U6", "ni_atom_U0_probe", "cuh_crystal_L7_U4", "cuh_molecule_U4"]   # repeated at 4 threads and run at 1 thread


# --------------------------------------------------------------------------- #
# Provenance.
# --------------------------------------------------------------------------- #


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(npz: Path) -> str:
    """sha256 of the arrays' content (key, dtype, shape, bytes), not of the zip file (its headers carry mtimes)."""
    import numpy as np
    h = hashlib.sha256()
    z = np.load(npz)
    for k in sorted(f for f in z.files if f != "stamp_json"):
        a = np.ascontiguousarray(z[k])
        h.update(f"{k}|{a.dtype}|{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def tree_digest(root: Path) -> tuple[str, int]:
    """sha256 over (relative path, file sha256) of every .py file under ``root``."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(sha256_file(p).encode())
        n += 1
    return h.hexdigest(), n


def stamp(old: Path, repo: str | None = None) -> dict:
    pkg = old / "src" / "mandacaru"
    digest, nfiles = tree_digest(pkg)
    so = sorted(pkg.glob("_native*.so"))
    paw = Path(os.environ.get("MANDACARU_PAW_PATH", ""))
    data = {}
    for el in ("H", "O", "Ti", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ce", "Gd"):
        f = paw / "lda-sr" / f"{el}.parquet"
        if f.exists():
            data[el] = sha256_file(f)
    ver = (old / "src" / "mandacaru" / "version.py")
    m = re.search(r'["\'](\d+\.\d+\.\d+)["\']', ver.read_text()) if ver.exists() else None
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo or HERE).stdout.strip() or None
    except Exception:
        head = None
    out = dict(
        old_version=m.group(1) if m else None, repo_head_when_frozen=head,
        old_commit=OLD_COMMIT, old_tree_sha256=digest, old_py_files=nfiles,
        extension=[dict(name=s.name, sha256=sha256_file(s), bytes=s.stat().st_size) for s in so],
        paw_path=str(paw), paw_set="lda-sr", paw_sha256=data,
        python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
    )
    try:
        import numpy, scipy, ase
        out.update(numpy=numpy.__version__, scipy=scipy.__version__, ase=ase.__version__)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Child side: everything below runs in a fresh interpreter (python -I).
# --------------------------------------------------------------------------- #


def peak_footprint_mb() -> float | None:
    """Lifetime peak physical footprint of this process (macOS proc_pid_rusage, rusage_info_v4), MB."""
    try:
        import ctypes, ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        buf = (ctypes.c_uint64 * 80)()
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
            return None
        # 16-byte uuid, then u64 fields; ri_lifetime_max_phys_footprint is index 28 (checked against a 300 MB allocation)
        raw = bytes(buf)
        import struct
        return struct.unpack_from("<Q", raw, 16 + 28 * 8)[0] / 2 ** 20
    except Exception:
        return None


def boot(old: str):
    if not os.environ.get("MANDACARU_PAW_PATH"):
        sys.exit("MANDACARU_PAW_PATH must name the mandacaru-paw checkout (mandacaru-paw --set DIR)")
    sys.path.insert(0, str(Path(old) / "src"))
    import warnings
    warnings.filterwarnings("ignore")
    import mandacaru
    here = str(Path(mandacaru.__file__).resolve())
    assert str(Path(old).resolve()) in here, f"imported {here}, not the restored old tree"
    return mandacaru


def cplx(a):
    import numpy as np
    a = np.asarray(a)
    return dict(re=a.real.tolist(), im=a.imag.tolist())


def child_weights(old: str):
    boot(old)
    import numpy as np
    from scipy.integrate import simpson
    from mandacaru.algorithms.hubbard import shell_weights
    from mandacaru.pseudopotentials import get_paw
    from mandacaru.pseudopotentials.multipoles import partial_waves
    rows = {}
    for sym, l in (("H", 0), ("O", 1), ("Ti", 2), ("Mn", 2), ("Fe", 2), ("Co", 2), ("Ni", 2),
                   ("Cu", 2), ("Zn", 2), ("Ce", 3), ("Gd", 3)):
        ds = get_paw(sym)
        ch = ds.channels[l]
        r = np.asarray(ds.r, float)
        ae, _ = partial_waves(ds, l)
        occ = [i for i, f in enumerate(getattr(ch, "occupations", ()) or ()) if float(f) > 0]
        chi = ae[occ[0] if occ else 0]
        ins = r <= float(ch.r_cut)
        full = float(simpson(chi ** 2 * r ** 2, x=r))
        inn = float(simpson(chi[ins] ** 2 * r[ins] ** 2, x=r[ins]))
        wr, wd = shell_weights(ds, l, "raw"), shell_weights(ds, l, "dual")
        rows[sym] = dict(l=l, r_cut=float(ch.r_cut), n_projectors=len(ae), occupations=[float(x) for x in ch.occupations],
                         reference_orbital_norm=full, fraction_in_sphere=inn / full,
                         w_raw=wr.tolist(), w_dual=wd.tolist(),
                         n_reference_dual=float(wd[0] ** 2),       # = fraction (dual coefficients (1,0))
                         r_grid_max=float(r[-1]), r_grid_points=int(len(r)))
    return dict(kind="weights", rows=rows), {}


def _atoms(system: str, seed: str):
    import numpy as np
    from ase import Atoms
    from ase.build import bulk, make_supercell
    s = NIO_SETTINGS
    if system == "nio":
        prim = bulk("NiO", "rocksalt", a=s["a"])
        a = make_supercell(prim, np.array([[1, 0, 0], [0, 1, 0], [-1, -1, 2]]))
        mom = 0.5 if seed == "B" else 2.0
        a.set_initial_magnetic_moments(_afm_moments(a.get_chemical_symbols(), mom))
        return a, dict(spin=True, kpts=s["kpts"], width=s["width"], periodic=True, element="Ni", h=s["h"], size=s["size"])
    if system == "ni_atom":
        a = Atoms("Ni", positions=[[0, 0, 0]], cell=[7.0] * 3, pbc=True)
        a.center()
        a.set_initial_magnetic_moments([2.0])
        return a, dict(spin=True, kpts=(1, 1, 1), width=0.02, periodic=True, element="Ni", h=0.25, size="SZ")
    if system in ("cuh_crystal", "cuh_crystal_spin", "cuh_molecule"):
        pbc = system != "cuh_molecule"
        a = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[9.0 if pbc else 8.0] * 3, pbc=pbc)
        a.center()
        if system == "cuh_crystal_spin":
            a.set_initial_magnetic_moments([1.0, 0.0])
        return a, dict(spin=(system == "cuh_crystal_spin"), kpts=(1, 1, 1), width=0.01, periodic=pbc,
                       element="Cu", h=0.3, size="SZ")
    raise KeyError(system)


def _afm_moments(symbols, mom):
    out, k = [], 0
    for x in symbols:
        if x == "Ni":
            out.append(mom if k % 2 == 0 else -mom)
            k += 1
        else:
            out.append(0.0)
    return out


def seed_operator_factory(U_h):
    """Seed C: for the first 6 evaluations the occupation matrices fed to the operator
    are the orbital-broken seed instead of the natural output (complex harmonics m=-2..2).

    Atom k (k even: spin-up majority) -- majority spin: 0.95 on all five; minority spin:
    t2g 0.95, d_z2 (m=0) 0.95 occupied, d_x2-y2 ((m=+2)+(m=-2))/sqrt2 0.0 empty
    (an eg-polarized d9-like minority hole on the other orbital).  Odd atoms swap spins."""
    import numpy as np
    dz2 = np.zeros(5, complex); dz2[2] = 1.0
    dxy = np.zeros(5, complex); dxy[0] = dxy[4] = 1 / np.sqrt(2)      # (m=-2)+(m=+2)
    eg_hole = np.outer(dxy, dxy.conj())
    n_full = 0.95 * np.eye(5, dtype=complex)
    n_min = n_full - 0.95 * eg_hole
    return n_full, n_min


def child_scf(old: str, spec: dict):
    boot(old)
    import resource
    import numpy as np
    from mandacaru import Mandacaru
    from mandacaru.algorithms import hubbard as H
    from mandacaru.algorithms import periodic_dft as P

    system, U, seed = spec["system"], float(spec["U"]), spec["seed"]
    atoms, cfg = _atoms(system, seed)
    if spec.get("L"):
        atoms.set_cell([spec["L"]] * 3); atoms.center()
    if spec.get("h"):
        cfg["h"] = spec["h"]

    rec: list = []                      # one entry per evaluation: [spin][shell] n
    counter = {"k": 0}
    orig = H.OnsiteHubbard.evaluate
    n_full, n_min = seed_operator_factory(U)

    def evaluate(self, projected, restricted):
        energy, operators = orig(self, projected, restricted)
        ns = []
        for R in projected:
            R = np.asarray(R)
            ns.append([n.copy() for n in self.occupations(0.5 * R if restricted else R)])
        rec.append(ns)
        counter["k"] += 1
        if seed in ("C", "D") and counter["k"] <= (6 if seed == "C" else 12) and not restricted:
            size = self.size
            ops = []
            for s in range(2):
                op = np.zeros((size, size), dtype=operators[s].dtype)
                for k, (u, positions, selection) in enumerate(self.shells):
                    majority = (s == 0) == (k % 2 == 0)
                    n = n_full if majority else n_min
                    op[np.ix_(positions, positions)] += selection @ H.onsite_potential(n, u) @ selection.T
                ops.append(op)
            return energy, ops
        return energy, operators

    H.OnsiteHubbard.evaluate = evaluate

    kw = {}
    if U > 0.0:
        kw["hubbard_u"] = {cfg["element"]: {"d": U}}
    common = dict(method="dft", xc="lda", grid_spacing=cfg["h"], trace=False,
                  basis={"name": "PAW-LCAO", "size": cfg["size"]})
    if cfg["periodic"]:
        common.update(kpts={"size": cfg["kpts"], "gamma": True}, spin=cfg["spin"],
                      smearing={"method": "fermi-dirac", "width": cfg["width"]})
    else:
        common.update(stability=False)
    atoms.calc = Mandacaru(**common, **kw)
    t0 = time.time()
    e = atoms.get_potential_energy()
    wall = time.time() - t0
    calc = atoms.calc
    res = calc.result
    out = dict(kind="scf", system=system, U_eV=U, seed=seed, symbols=atoms.get_chemical_symbols(),
               energy_eV=float(e), wall_s=wall)
    arrays = {}
    scf = getattr(res, "scf", None)
    if scf is not None and not hasattr(scf, "terms"):          # molecular KohnShamResult
        h_ha = float(getattr(scf, "hubbard_energy", 0.0))
        out.update(molecular=True, hubbard_Ha=h_ha, hubbard_eV=h_ha * 27.211386245988,
                   n_iterations=int(getattr(scf, "n_iterations", -1)), converged=bool(getattr(scf, "converged", False)),
                   xc_energy_Ha=float(getattr(scf, "xc_energy", float("nan"))))
        scf = None
    if scf is not None:
        terms = {k: float(v) for k, v in scf.terms.items()}
        out.update(free_energy_Ha=float(scf.free_energy), extrapolated_energy_Ha=float(scf.extrapolated_energy), total_energy_Ha=float(scf.energy),
                   n_iterations=int(scf.n_iterations), converged=bool(scf.converged), terms_Ha=terms,
                   hubbard_Ha=terms.get("hubbard", 0.0), hubbard_eV=terms.get("hubbard", 0.0) * 27.211386245988,
                   band_gap_eV=None if getattr(scf, "band_gap", None) is None else float(scf.band_gap) * 27.211386245988,
                   fermi_level_Ha=float(scf.fermi_level), magnetic_moment=float(scf.magnetic_moment),
                   history_last=dict(iter=int(scf.history[-1]["iter"]), energy_Ha=float(scf.history[-1]["energy"]),
                                     residual=float(scf.history[-1]["residual"])),
                   n_kpoints=int(len(scf.kpoints)))
        arrays["history_iter_energyHa_residual_moment"] = np.array(
            [[h["iter"], h["energy"], h["residual"], h.get("moment", 0.0)] for h in scf.history], dtype=float)
        ev = np.asarray(scf.eigenvalues)
        occ = np.asarray(scf.occupations)
        arrays["eigenvalues_Ha"] = ev
        arrays["occupations"] = occ
        arrays["kpoints"] = np.asarray(scf.kpoints)
        arrays["weights"] = np.asarray(scf.weights)
        # own gap: lowest level above E_F minus highest level at or below it, all spins/k
        ef = float(scf.fermi_level)
        above, below = ev[ev > ef], ev[ev <= ef]
        out["gap_from_levels_eV"] = float(above.min() - below.max()) * 27.211386245988 if above.size and below.size else None
        out["spin_channels"] = int(ev.shape[1]) if ev.ndim == 3 else 1
    elif not out.get("molecular"):
        out.update(note="no .scf on the result")
    try:
        out["magnetic_moment_total"] = float(atoms.get_magnetic_moment())
        out["magnetic_moments_local"] = [float(x) for x in atoms.get_magnetic_moments()]
    except Exception as ex:                     # restricted runs have no moment
        out["magnetic_moment_total"] = None
        out["moment_note"] = type(ex).__name__
    if rec:
        last = rec[-1]
        out["n_calls"] = len(rec)
        out["n_eigenvalues"] = [[np.linalg.eigvalsh(n).tolist() for n in per_spin] for per_spin in last]
        out["n_trace"] = [[float(np.trace(n).real) for n in per_spin] for per_spin in last]
        flat = np.array([w for per_spin in out["n_eigenvalues"] for sh in per_spin for w in sh])
        out["n_eigenvalue_min"], out["n_eigenvalue_max"] = float(flat.min()), float(flat.max())
        out["n_hermitian_err"] = float(max(np.abs(n - n.conj().T).max() for ps in last for n in ps))
        arrays["n_matrices"] = np.array([[n for n in per_spin] for per_spin in last])      # (spin, shell, 5, 5)
        # first and middle evaluations, to show the trajectory
        out["n_eigenvalue_trajectory_first_last"] = {
            "first": [[np.linalg.eigvalsh(n).tolist() for n in ps] for ps in rec[0]]}
    out["settings"] = dict(
        xc="lda", basis="PAW-LCAO", size=cfg["size"], grid_spacing_A=cfg["h"], kpts=list(cfg["kpts"]) if cfg["periodic"] else None,
        gamma_centered=True if cfg["periodic"] else None, smearing=("fermi-dirac", cfg["width"]) if cfg["periodic"] else None,
        spin=cfg["spin"], initial_moments=[float(x) for x in atoms.get_initial_magnetic_moments()],
        cell_A=[float(x) for x in atoms.cell.lengths()], pbc=bool(cfg["periodic"]),
        hubbard_u={cfg["element"]: {"d": U}} if U > 0 else None, J_eV=0.0,
        scf=dict(max_iter=300, tol_Ha=1e-7, density_tol=1e-5, hybrid_tol=P.HYBRID_TOL,
                 mixing_beta=P.MIXING_BETA, mixing_history=P.MIXING_HISTORY, kerker_q0=P.KERKER_Q0,
                 onsite_mix_weight=P.ONSITE_MIX_WEIGHT, spin_warmup_iterations=P.SPIN_WARMUP_ITERATIONS,
                 mixed_quantity="on-site operator C O C^dagger (the old loop mixes the operator, not n)",
                 initial_n="none: the operator starts at zero, the first iteration is plain DFT",
                 seed_description={"A": "initial moments +-2.0, zero operator start (the old default)",
                                   "B": "initial moments +-0.5, zero operator start",
                                   "C": "initial moments +-2.0 and, for the first 6 evaluations, operators from an "
                                        "orbital-broken n (majority 0.95*I; minority 0.95 on all but d_x2-y2, "
                                        "which is 0). A PARTIAL pin: the old loop linearly mixes the operator with "
                                        "beta = 0.25 for the first 12 iterations, so after 6 pinned evaluations the input "
                                        "holds 1-(0.75)^6 = 82 % of the pinned operator; then released",
                                   "D": "as C but pinned for 12 evaluations (the whole linear-mixing warmup): the input "
                                        "holds 1-(0.75)^12 = 97 % of the pinned operator at the release"}[seed] if system == "nio" else
                 "initial moments as listed, zero operator start"),
        symmetry="full k mesh: the old crystal is built on the whole mesh whenever the correction is on "
                 "(no symmetrization of n); molecules have no mesh",
        threads=int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
    )
    try:
        from mandacaru.integrals import _backend
        out["threads_seen_by_native"] = int(_backend.num_threads())
    except Exception as ex:
        out["threads_seen_by_native"] = f"unavailable: {type(ex).__name__}"
    out["maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    out["peak_footprint_MB"] = peak_footprint_mb()
    return out, arrays


# ----- exactly solvable models ----------------------------------------------- #


def child_models(old: str):
    boot(old)
    import numpy as np
    import mpmath as mp
    from mandacaru.algorithms.hubbard import onsite_energy, onsite_potential
    mp.mp.dps = 40

    def scf(h0, shells, nocc, U, mix=0.5, tol=1e-14, maxit=20000, n0=None):
        """Dudarev SCF with the OLD code's E_U and V_U.  ``shells`` is a list of index lists (one
        n per shell and spin); ``n0[s][k]`` seeds spin s, shell k (default 1/2 * identity)."""
        if shells and not isinstance(shells[0], (list, tuple)):
            shells = [list(shells)]
        S = len(h0)
        n = [[(0.5 * np.eye(len(sh)) if n0 is None else np.array(n0[s][k], float)) for k, sh in enumerate(shells)]
             for s in range(S)]
        for it in range(maxit):
            out, Cs, eps = [], [], []
            for s in range(S):
                Hm = h0[s].astype(complex).copy()
                for k, sh in enumerate(shells):
                    Hm[np.ix_(sh, sh)] += onsite_potential(n[s][k], U)      # the OLD code's V_U
                e, C = np.linalg.eigh(Hm)
                eps.append(e); Cs.append(C)
                Co = C[:, :nocc[s]]
                D = Co @ Co.conj().T
                out.append([D[np.ix_(sh, sh)] for sh in shells])
            err = max(np.abs(out[s][k] - n[s][k]).max() for s in range(S) for k in range(len(shells)))
            n = [[mix * out[s][k] + (1 - mix) * n[s][k] for k in range(len(shells))] for s in range(S)]
            if err < tol:
                break
        E = 0.0
        for s in range(S):
            Co = Cs[s][:, :nocc[s]]
            E += float(np.trace(h0[s] @ (Co @ Co.conj().T)).real)
            E += sum(onsite_energy(n[s][k], U) for k in range(len(shells)))   # the OLD code's E_U
        return E, n, eps, it

    out = {"kind": "models", "mp_dps": 40}
    # ---- A. 2x2 (d, L), one electron per spin ------------------------------------
    ed, eL, t = mp.mpf("-0.30"), mp.mpf("-0.20"), mp.mpf("-0.15")
    h2 = np.array([[float(ed), float(t)], [float(t), float(eL)]])

    def Ex(x, U):                        # both spins in the same orbital (cos, sin), n = x
        return 2 * (ed * x + eL * (1 - x) + 2 * t * mp.sqrt(x * (1 - x))) + U * x * (1 - x)

    def stationary(U):
        U = mp.mpf(U)
        f = lambda x: mp.diff(lambda y: Ex(y, U), x)
        roots, grid = [], [mp.mpf(i) / 2000 for i in range(1, 2000)]
        for a, b in zip(grid[:-1], grid[1:]):
            fa, fb = f(a), f(b)
            if fa == 0 or fa * fb < 0:
                r = mp.findroot(f, (a, b), solver="anderson", tol=1e-35, maxsteps=200)
                if not roots or abs(roots[-1] - r) > mp.mpf("1e-20"):
                    roots.append(r)
        res = []
        for r in roots:
            d2 = mp.diff(lambda y: Ex(y, U), r, 2)
            res.append(dict(n=mp.nstr(r, 20), E=mp.nstr(Ex(r, U), 20), minimum=bool(d2 > 0)))
        return res
    out["two_by_two"] = {"ed": -0.30, "eL": -0.20, "t": -0.15, "runs": []}
    for U in (0.0, 0.2, 0.6, 1.2):
        ana = stationary(U)
        for n0v in (0.5, 0.95, 0.05):
            E, n, eps, it = scf([h2, h2], [[0]], [1, 1], U, n0=[[np.array([[n0v]])]] * 2)
            nearest = min(ana, key=lambda a: abs(float(a["n"]) - n[0][0][0, 0].real))
            out["two_by_two"]["runs"].append(dict(U=U, seed_n=n0v, E_scf=E, n_scf=float(n[0][0][0, 0].real), iterations=it,
                                                  analytic_stationary_points=ana,
                                                  matched_E=nearest["E"], abs_diff=float(abs(E - float(mp.mpf(nearest["E"]))))))
    # t = 0 level shift: occupied d -> e_d - U/2, empty -> e_d + U/2
    h0 = np.array([[float(ed), 0.0], [0.0, float(eL)]])
    out["level_shift_t0"] = []
    for U in (0.0, 0.4, 0.8):
        _, _, eps, _ = scf([h0, h0], [[0]], [1, 0], U, n0=[[np.array([[1.0]])], [np.array([[0.0]])]])
        out["level_shift_t0"].append(dict(U=U, eps_d_occ=float(min(eps[0])), expect_occ=float(ed) - U / 2,
                                          eps_d_empty=float(eps[1][0]), expect_empty=float(ed) + U / 2))
    # ---- B. Hubbard dimer ------------------------------------------------------------
    t2 = -0.5
    hd = np.array([[0.0, t2], [t2, 0.0]])
    out["dimer"] = {"t": t2, "runs": []}
    for U in (0.5, 0.9, 1.0, 1.1, 1.5, 3.0, 6.0):
        Enm, nnm, eps_nm, it1 = scf([hd, hd], [[0], [1]], [1, 1], U, mix=0.3, n0=[[np.array([[0.5]]), np.array([[0.5]])]] * 2)
        n0 = [[np.array([[0.8]]), np.array([[0.2]])], [np.array([[0.2]]), np.array([[0.8]])]]
        Eaf, naf, eps_af, it2 = scf([hd, hd], [[0], [1]], [1, 1], U, mix=0.3, n0=n0)
        m = float(naf[0][0][0, 0] - naf[1][0][0, 0])
        ana_nm = -2 * abs(t2) + U / 2
        ana = dict(E_NM=ana_nm)
        if U > 2 * abs(t2):
            ana.update(E_AFM=-2 * t2 ** 2 / U, m=float(np.sqrt(1 - (2 * t2 / U) ** 2)), gap=U)
        gap = float(eps_af[0][1] - eps_af[0][0])
        out["dimer"]["runs"].append(dict(U=U, E_NM=Enm, E_AFM_scf=Eaf, m_scf=m, gap_scf=gap, iterations=[it1, it2],
                                         analytic=ana,
                                         err_NM=abs(Enm - ana_nm),
                                         err_AFM=abs(Eaf - ana["E_AFM"]) if "E_AFM" in ana else abs(Eaf - ana_nm)))
    out["dimer"]["note"] = "below U = 2|t| = 1.0 the AFM seed relaxes to the nonmagnetic state (m_scf ~ 0, E_AFM_scf = E_NM)"
    # rotation invariance of E_U under a unitary in the shell (the old functional)
    rng = np.random.default_rng(7)
    A = rng.normal(size=(4, 4)); Hm = 0.5 * (A + A.T)
    E1 = scf([Hm, Hm], [[0, 1, 2]], [2, 1], 0.7)[0]
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    W = np.eye(4, dtype=complex); W[:3, :3] = Q
    E2 = scf([W.conj().T @ Hm @ W] * 2, [[0, 1, 2]], [2, 1], 0.7)[0]
    out["shell_rotation_invariance"] = dict(E=E1, E_rotated=E2, abs_diff=abs(E1 - E2))
    # pure functions on fixed matrices (bitwise gate for the Rust model)
    nf = np.array([[0.9, 0.1j, 0], [-0.1j, 0.4, 0.05], [0, 0.05, 0.1]])
    out["pure"] = dict(n=cplx(nf), U=0.3, E=onsite_energy(nf, 0.3), V=cplx(onsite_potential(nf, 0.3)),
                       half_filling_identity=dict(l=2, expected=0.3 / 2 * 5 / 4,
                                                  got=onsite_energy(0.5 * np.eye(5), 0.3)))
    return out, {}


# --------------------------------------------------------------------------- #
# Parent side.
# --------------------------------------------------------------------------- #


def child_main(name: str, old: str, out_json: str, out_npz: str):
    spec = json.loads(os.environ["HU0_SPEC"]) if os.environ.get("HU0_SPEC") else CASES[name]
    if spec["kind"] == "weights":
        res, arr = child_weights(old)
    elif spec["kind"] == "models":
        res, arr = child_models(old)
    else:
        res, arr = child_scf(old, spec)
    res["case"] = name
    stamp_json = os.environ.get("HU0_STAMP", "")
    res["stamp_sha256"] = hashlib.sha256(stamp_json.encode()).hexdigest() if stamp_json else None
    if arr:
        import numpy as np
        arr["stamp_json"] = np.array(stamp_json)
    Path(out_json).write_text(json.dumps(res, indent=1))
    if arr:
        import numpy as np
        np.savez_compressed(out_npz, **arr)


def _to_mb(s: str) -> float:
    s = s.strip().rstrip("+-")
    m = re.match(r"([\d.]+)([KMGB]?)", s)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v * {"K": 1 / 1024, "M": 1, "G": 1024, "B": 1 / 2 ** 20, "": 1 / 2 ** 20}[m.group(2)]


def guard(path) -> None:
    """Refuse to start while the machine is short of memory (scripts/memguard.py, 1 GB need, 120 s wait)."""
    g = Path(path) if path else MEMGUARD
    if not g.exists():
        sys.exit(f"memguard missing at {g}: pass --memguard PATH (or --no-memguard to run unguarded)")
    if subprocess.run([PYTHON, str(g), "--need-gb", "1", "--wait", "120"]).returncode:
        sys.exit("memguard refused to start the job (memory short)")


def run_child(name: str, old: str, workdir: Path, threads: int, cap_mb: float = CAP_MB, tag: str = "", spec: dict | None = None, stamp_json: str = "") -> dict:
    """Run one case in a monitored child; return its record (or a 'not frozen' record)."""
    workdir.mkdir(parents=True, exist_ok=True)
    pj, pz = workdir / f"{name}{tag}.json", workdir / f"{name}{tag}.npz"
    for p in (pj, pz):
        if p.exists():
            p.unlink()
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               HU0_STAMP=stamp_json, MKL_NUM_THREADS=str(threads), MANDACARU_NUM_THREADS=str(threads), VECLIB_MAXIMUM_THREADS=str(threads), **({"HU0_SPEC": json.dumps(spec)} if spec else {}), MANDACARU_PAW_PATH=os.environ.get("MANDACARU_PAW_PATH", ""))
    cmd = [PYTHON, "-I", str(Path(__file__).resolve()), "child", name, "--old", old, "--json", str(pj), "--npz", str(pz)]
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak_rss, cmprs, top_mem, killed, nsamp = 0.0, 0.0, 0.0, False, 0
    last_top = 0.0
    while p.poll() is None:
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
        if r:
            rss = float(r) / 1024
            if time.time() - last_top > 0.8:
                last_top = time.time()
                top = subprocess.run(["top", "-l", "1", "-pid", str(p.pid), "-stats", "pid,mem,cmprs"],
                                     capture_output=True, text=True).stdout.splitlines()
                for line in top:
                    f = line.split()
                    if len(f) >= 3 and f[0] == str(p.pid):          # only the child's own line
                        cmprs = max(cmprs, _to_mb(f[2]))
                        top_mem = max(top_mem, _to_mb(f[1]))
            peak_rss = max(peak_rss, rss)
            nsamp += 1
            if top_mem + cmprs > cap_mb or rss > 1.3 * cap_mb:
                p.kill(); killed = True
                break
        time.sleep(0.2)
    outtxt = p.communicate()[0]
    mem = dict(peak_rss_MB=round(peak_rss, 1), peak_top_mem_MB=round(top_mem, 1), peak_cmprs_MB=round(cmprs, 1), samples=nsamp,
               wall_s=round(time.time() - t0, 2), threads=threads, cap_MB=cap_mb, gate="top mem + cmprs <= cap (owner rule); RSS reported, killed above 1.3 x cap")
    if killed:
        return dict(case=name, not_frozen=f"not frozen: exceeds the cap at {round(max(top_mem + cmprs, peak_rss))} MB", memory=mem)
    if p.returncode != 0 or not pj.exists():
        return dict(case=name, not_frozen=f"child failed rc={p.returncode}", tail=outtxt[-1500:], memory=mem)
    rec = json.loads(pj.read_text())
    mem["child_peak_footprint_MB"] = rec.get("peak_footprint_MB")
    mem["child_maxrss_MB"] = rec.get("maxrss_MB")
    rec["memory"] = mem
    if pz.exists():
        rec["npz"] = dict(file=pz.name, content_sha256=array_digest(pz))
    return rec


HA_EV = 27.211386245988


def derive(doc: dict) -> dict:
    """Quantities computed from frozen runs: Hellmann-Feynman check, E(U), gap(U), n summary."""
    C = doc["cases"]

    def F(u, seed="A"):
        r = C.get(f"nio_U{u:g}_{seed}")
        return None if r is None or "not_frozen" in r else r["free_energy_Ha"] * HA_EV

    def EU(u):
        r = C.get(f"nio_U{u:g}_A")
        return None if r is None or "not_frozen" in r else r["hubbard_eV"]

    out = {"hellmann_feynman": [], "E_of_U": [], "note": "dF/dU = E_U/U with F the Mermin free energy (eV) and U_eff in eV"}
    for u0 in (1.0, 3.0, 5.0):
        row = dict(U0=u0, E_U_over_U=None if EU(u0) is None else EU(u0) / u0)
        D = {}
        for d in (0.2, 0.1):
            lo, hi = F(round(u0 - d, 6)), F(round(u0 + d, 6))
            if lo is not None and hi is not None:
                D[d] = (hi - lo) / (2 * d)
        row["central_difference"] = {str(k): v for k, v in D.items()}
        if 0.2 in D and 0.1 in D:
            rich = (4 * D[0.1] - D[0.2]) / 3
            row["richardson_in_U"] = rich
            row["abs_err_central_0.2"] = abs(D[0.2] - row["E_U_over_U"])
            row["abs_err_central_0.1"] = abs(D[0.1] - row["E_U_over_U"])
            row["abs_err_richardson"] = abs(rich - row["E_U_over_U"])
            row["rel_err_richardson"] = row["abs_err_richardson"] / row["E_U_over_U"]
        out["hellmann_feynman"].append(row)
    for u in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        r = C.get(f"nio_U{u:g}_A")
        if r is None or "not_frozen" in r:
            continue
        out["E_of_U"].append(dict(U=u, F_eV=r["free_energy_Ha"] * HA_EV, E_eV=r["energy_eV"], E_U_eV=r.get("hubbard_eV", 0.0),
                                  gap_eV=r["band_gap_eV"], n_iterations=r["n_iterations"],
                                  local_moment=max(abs(x) for x in r["magnetic_moments_local"])))
    es = [x["F_eV"] for x in out["E_of_U"]]
    out["F_monotone_increasing"] = all(b > a for a, b in zip(es, es[1:]))
    sec = [es[i + 1] - 2 * es[i] + es[i - 1] for i in range(1, len(es) - 1)]
    out["F_second_differences_eV_(uniform U step 1)"] = sec
    out["F_concave"] = all(x < 0 for x in sec)
    gs = [x["gap_eV"] for x in out["E_of_U"]]
    out["gap_increasing"] = all(b > a for a, b in zip(gs, gs[1:]))
    # seeds: do B, C, D land on the A minimum?
    out["seeds"] = []
    for u in (3.0, 6.0):
        base = C.get(f"nio_U{u:g}_A")
        for sd in ("A", "B", "C", "D"):
            r = C.get(f"nio_U{u:g}_{sd}")
            if base is None or r is None or "not_frozen" in r:
                continue
            fa = [x for ps in base["n_eigenvalues"] for sh in ps for x in sh]
            fb = [x for ps in r["n_eigenvalues"] for sh in ps for x in sh]
            out["seeds"].append(dict(U=u, seed=sd, F_eV=r["free_energy_Ha"] * HA_EV,
                                     dF_vs_A_eV=(r["free_energy_Ha"] - base["free_energy_Ha"]) * HA_EV,
                                     n_iterations=r["n_iterations"], gap_eV=r["band_gap_eV"],
                                     max_abs_dn_eig_vs_A=max(abs(x - y) for x, y in zip(sorted(fa), sorted(fb))),
                                     local_moment=max(abs(x) for x in r["magnetic_moments_local"])))
    # occupation-matrix summary (sorted eigenvalues per spin and site)
    out["n_summary"] = {}
    for nm in ("ni_atom_U0_probe", "ni_atom_U3", "ni_atom_U6", "nio_U0_probe", "nio_U3_A", "nio_U6_A", "cuh_crystal_L9_U4",
               "cuh_crystal_L7_U4", "cuh_molecule_U4"):
        r = C.get(nm)
        if r and "n_eigenvalues" in r:
            out["n_summary"][nm] = dict(min=r["n_eigenvalue_min"], max=r["n_eigenvalue_max"], trace=r["n_trace"],
                                        eigenvalues=[[[round(x, 6) for x in sh] for sh in ps] for ps in r["n_eigenvalues"]])
    sh = {}
    for tag, a_, b_ in (("crystal_L9", "cuh_crystal_L9_U0", "cuh_crystal_L9_U4"), ("crystal_L7", "cuh_crystal_L7_U0", "cuh_crystal_L7_U4"),
                        ("molecule_L8", "cuh_molecule_U0", "cuh_molecule_U4"), ("molecule_L9", "cuh_molecule_L9_U0", "cuh_molecule_L9_U4")):
        if a_ in C and b_ in C and "energy_eV" in C[a_] and "energy_eV" in C[b_]:
            sh[tag] = C[b_]["energy_eV"] - C[a_]["energy_eV"]
    out["cuh_U4_energy_shift_eV"] = sh
    return out


def repro_metrics(a: dict, b: dict, work: Path, ra: str, rb: str) -> dict:
    """Differences between two records of the same case (a = frozen, b = repeat)."""
    import numpy as np
    out = {"energy_eV": abs(a["energy_eV"] - b["energy_eV"]),
           "free_energy_eV": abs(a["free_energy_Ha"] - b["free_energy_Ha"]) * HA_EV if "free_energy_Ha" in a else None,
           "hubbard_eV": abs(a.get("hubbard_eV", 0.0) - b.get("hubbard_eV", 0.0)),
           "gap_eV": None if a.get("band_gap_eV") is None else abs(a["band_gap_eV"] - b["band_gap_eV"]),
           "n_iterations": [a.get("n_iterations"), b.get("n_iterations")],
           "n_eigenvalues_max_abs": (max(abs(x - y) for pa, pb in zip(a["n_eigenvalues"], b["n_eigenvalues"])
                                         for xa, xb in zip(pa, pb) for x, y in zip(xa, xb)) if "n_eigenvalues" in a else None),
           "moments_max_abs": (max(abs(x - y) for x, y in zip(a["magnetic_moments_local"], b["magnetic_moments_local"]))
                               if a.get("magnetic_moments_local") else None)}
    out["npz_content_equal"] = (a.get("npz", {}).get("content_sha256") == b.get("npz", {}).get("content_sha256")) if "npz" in a else None
    fa, fb = work / ra, work / rb
    if fa.exists() and fb.exists():
        A, B = np.load(fa), np.load(fb)
        out["npz_max_abs"] = {k: float(np.abs(A[k] - B[k]).max()) for k in A.files if k != "stamp_json" and A[k].shape == B[k].shape}
    return out


def run_repro(doc: dict, old: str, out: Path, stamp_json: str = ""):
    """(f) reproducibility floor: each REPRO case twice more at THREADS and once at 1 thread."""
    doc["reproducibility"] = {}
    for nm in REPRO:
        ref = doc["cases"].get(nm)
        if not ref or "not_frozen" in ref:
            continue
        rows = {}
        for tag, th in (("_rep2", THREADS), ("_rep3", THREADS), ("_thr1", 1)):
            rec = run_child(nm, old, out / "repro", th, tag=tag, stamp_json=stamp_json)
            if "not_frozen" in rec:
                rows[tag] = rec["not_frozen"]
                continue
            m = repro_metrics(ref, rec, out, ref["npz"]["file"] if "npz" in ref else "", "repro/" + rec["npz"]["file"] if "npz" in rec else "")
            m["threads_requested"] = th
            m["threads_seen_by_native"] = rec.get("threads_seen_by_native")
            m["wall_s"] = rec["memory"]["wall_s"]
            rows[tag] = m
        doc["reproducibility"][nm] = rows
        print(f"[repro {nm}]", {k: (v if isinstance(v, str) else (v["energy_eV"], v["npz_content_equal"])) for k, v in rows.items()}, flush=True)


def cmd_restore(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    repo = a.repo or subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    arch = subprocess.run(["git", "archive", OLD_COMMIT, "src"], capture_output=True, cwd=repo)
    if arch.returncode:
        sys.exit(arch.stderr.decode())
    subprocess.run(["tar", "-x", "-C", str(d)], input=arch.stdout, check=True)
    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}-darwin"
    cands = sorted(Path(repo, "src", "mandacaru").glob(f"_native.{tag}.so")) or sorted(Path(repo, "src", "mandacaru").glob("_native*.so"))
    if not cands:
        sys.exit("no current extension found under src/mandacaru")
    import shutil
    shutil.copy2(cands[0], d / "src" / "mandacaru" / cands[0].name)
    print("restored", OLD_COMMIT, "->", d, "; extension", cands[0].name)
    print(json.dumps(stamp(d, repo), indent=1))


def _compare(new, ref, path="", out=None, tol=0.0):
    """Recursively compare two JSON records; return the largest abs difference and mismatching keys."""
    out = {} if out is None else out
    if isinstance(ref, dict):
        for k in ref:
            if k in ("memory", "wall_s", "maxrss_MB", "peak_footprint_MB", "npz", "reproducibility", "settings", "stamp_sha256"):
                continue
            if k not in new:
                out[path + "/" + k] = "missing"
            else:
                _compare(new[k], ref[k], path + "/" + k, out, tol)
    elif isinstance(ref, list):
        if len(ref) != len(new):
            out[path] = f"length {len(new)} != {len(ref)}"
        else:
            for i, (x, y) in enumerate(zip(new, ref)):
                _compare(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(ref, float):
        d = abs(float(new) - ref)
        if d > tol:
            out[path] = d
    elif new != ref:
        out[path] = f"{new!r} != {ref!r}"
    return out


def cmd_freeze(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    old = str(Path(a.old).resolve())
    names = [] if a.repro_only else (a.cases.split(",") if a.cases else list(CASES))
    result_path = out / "frozen.json"
    doc = json.loads(result_path.read_text()) if result_path.exists() else {}
    doc["stamp"] = stamp(Path(old), a.repo)
    stamp_json = json.dumps(doc["stamp"], sort_keys=True)
    doc.setdefault("cases", {})
    for nm in names:
        if not a.no_memguard:
            guard(a.memguard)
        if CASES[nm].get("skip"):
            doc["cases"][nm] = dict(case=nm, not_frozen=CASES[nm]["skip"])
            print(f"[{nm}] {CASES[nm]['skip']}", flush=True)
            continue
        rec = run_child(nm, old, out, THREADS, stamp_json=stamp_json)
        doc["cases"][nm] = rec
        status = rec.get("not_frozen") or f"E={rec.get('energy_eV')} t={rec['memory']['wall_s']}s rss={rec['memory']['peak_rss_MB']}MB"
        print(f"[{nm}] {status}", flush=True)
        result_path.write_text(json.dumps(doc, indent=1))
    doc["derived"] = derive(doc)
    if not a.no_repro:
        run_repro(doc, old, out, stamp_json)
    result_path.write_text(json.dumps(doc, indent=1))
    print("wrote", result_path)


#: replay tolerances for SCF records, from the measured floor (reproducibility section): the same case run
#: again gave 4e-12 eV in E, 3e-13 in n, 7e-14 Ha in eigenvalues (2.8e-8 Ha / 2.8e-7 eV gap for the Ni atom,
#: whose near-Fermi levels are degenerate); 1 and 4 threads were indistinguishable.  The gate is ~250x the floor.
VERIFY_TOL = dict(energy_eV=1e-9, free_energy_Ha=1e-10, extrapolated_energy_Ha=1e-10, total_energy_Ha=1e-10,
                  hubbard_Ha=1e-10, hubbard_eV=1e-9, band_gap_eV=1e-6, gap_from_levels_eV=1e-6, fermi_level_Ha=1e-6,
                  magnetic_moment=1e-9, magnetic_moment_total=1e-9, n_eigenvalues=1e-9, n_trace=1e-9,
                  n_eigenvalue_min=1e-9, n_eigenvalue_max=1e-9, n_hermitian_err=1e-9, xc_energy_Ha=1e-10)
NPZ_TOL = dict(n_matrices=1e-9, eigenvalues_Ha=1e-6, occupations=1e-8, kpoints=0.0, weights=0.0)


def _flat(x):
    out = []
    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flat(y))
    elif isinstance(x, (int, float)):
        out.append(float(x))
    return out


def verify_scf(new: dict, ref: dict, work: Path, ref_dir: Path) -> dict:
    bad = {}
    scale = 100.0 if ref.get("system") == "ni_atom" else 1.0     # the atom's near-Fermi levels are degenerate
    for k, tol in VERIFY_TOL.items():
        tol = tol * scale
        if k in ref:
            a, b = _flat(new.get(k)), _flat(ref[k])
            if len(a) != len(b):
                bad[k] = "length"
            elif a and max(abs(x - y) for x, y in zip(a, b)) > tol:
                bad[k] = max(abs(x - y) for x, y in zip(a, b))
    for k in ("n_iterations", "converged", "n_kpoints"):
        if k in ref and new.get(k) != ref[k]:
            if k == "n_iterations" and scale > 1 and abs(new.get(k, 0) - ref[k]) <= 3:
                continue                                  # the atom's iteration count varies by 1-2 run to run
            bad[k] = f"{new.get(k)!r} != {ref[k]!r}"
    if "magnetic_moments_local" in ref:
        a, b = _flat(new.get("magnetic_moments_local")), _flat(ref["magnetic_moments_local"])
        if max(abs(x - y) for x, y in zip(a, b)) > 1e-9 * scale:
            bad["magnetic_moments_local"] = max(abs(x - y) for x, y in zip(a, b))
    if "npz" in ref:
        import numpy as np
        fa, fb = work / new["npz"]["file"], ref_dir / ref["npz"]["file"]
        if not fb.exists():
            bad["npz"] = f"reference {fb.name} missing (git-ignored: restore it from the freeze archive)"
        else:
            A, B = np.load(fa), np.load(fb)
            for k in B.files:
                tol = NPZ_TOL.get(k)
                if tol is None or A[k].shape != B[k].shape:
                    continue
                tol = tol * scale
                d = float(np.abs(A[k] - B[k]).max())
                if d > tol:
                    bad["npz:" + k] = d
    return bad


def cmd_verify(a):
    ref_dir = Path(a.ref)
    ref = json.loads((ref_dir / "frozen.json").read_text())
    old = str(Path(a.old).resolve())
    work = Path(a.work)
    bad, ok = 0, 0
    st = stamp(Path(old), a.repo)
    if st["old_tree_sha256"] != ref["stamp"]["old_tree_sha256"]:
        print("FAIL: the restored old tree differs from the frozen one"); bad += 1
    if [e["sha256"] for e in st["extension"]] != [e["sha256"] for e in ref["stamp"]["extension"]]:
        print("NOTE: the native extension differs from the frozen one (a difference below is then a parity finding)")
    only = set(a.cases.split(",")) if a.cases else None
    for nm, spec in CASES.items():
        if not spec.get("cheap") or nm not in ref["cases"] or "not_frozen" in ref["cases"][nm]:
            continue
        if only and nm not in only:
            continue
        if not a.no_memguard:
            guard(a.memguard)
        rec = run_child(nm, old, work, THREADS, stamp_json=json.dumps(st, sort_keys=True))
        if "not_frozen" in rec:
            print(f"FAIL {nm}: {rec['not_frozen']}"); bad += 1; continue
        if spec["kind"] in ("weights", "models"):
            diff = _compare(rec, ref["cases"][nm])                       # tol 0: bitwise
            label = "BITWISE-OK"
        else:
            diff = verify_scf(rec, ref["cases"][nm], work, ref_dir)
            label = "WITHIN-FLOOR-OK"
        if diff:
            bad += 1
            print(f"DIFF {nm}: {dict(list(diff.items())[:6])}")
        else:
            ok += 1
            print(f"{label} {nm}  (rss {rec['memory']['peak_rss_MB']} MB, top {rec['memory']['peak_top_mem_MB']} MB, {rec['memory']['wall_s']} s)")
    print(f"verify: {ok} ok, {bad} not ok")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("restore"); s.add_argument("dir"); s.add_argument("--repo", help="repository root (default: git root of this file)")
    sub.add_parser("list")
    s = sub.add_parser("freeze"); s.add_argument("--old", required=True); s.add_argument("--cases"); s.add_argument("--out", default=str(HERE / "reference" / "hubbard")); s.add_argument("--memguard"); s.add_argument("--no-memguard", action="store_true"); s.add_argument("--no-repro", action="store_true"); s.add_argument("--repro-only", action="store_true", help="only redo the reproducibility stage on an existing frozen.json"); s.add_argument("--repo", help="repository root, for the HEAD stamp")
    s = sub.add_parser("verify"); s.add_argument("--old", required=True); s.add_argument("--repo"); s.add_argument("--cases", help="comma list (default: every cheap case)"); s.add_argument("--memguard"); s.add_argument("--no-memguard", action="store_true"); s.add_argument("--ref", default=str(HERE / "reference" / "hubbard")); s.add_argument("--work", default="/tmp/hubbard_verify")
    s = sub.add_parser("child"); s.add_argument("name"); s.add_argument("--old", required=True); s.add_argument("--json", required=True); s.add_argument("--npz", required=True)
    a = ap.parse_args()
    if a.cmd == "restore":
        cmd_restore(a)
    elif a.cmd == "list":
        for k, v in CASES.items():
            print(k, v)
    elif a.cmd == "freeze":
        cmd_freeze(a)
    elif a.cmd == "verify":
        cmd_verify(a)
    else:
        child_main(a.name, a.old, a.json, a.npz)


if __name__ == "__main__":
    main()
