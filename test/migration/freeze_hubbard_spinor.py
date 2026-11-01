#!/usr/bin/env python
"""Independent references for the noncollinear (spin-orbit) DFT+U functionals (plan SO-0, gates G-S1 / G-S2 model part).

Nothing here imports Mandacaru.  Written to ``test/migration/reference/hubbard/spinor.json`` (tracked, small).  Reuses the
exact Slater tensor of ``freeze_hubbard_slater.py`` (same directory) for the Liechtenstein part.

LAYOUT (every stored matrix is in it): 2p x 2p, SPIN-MAJOR, alpha block first, index = s*p + (m + l), s = 0 (up), 1 (down),
complex Y_lm, m = -l..l, Condon-Shortley.  ``n[(s,a),(s',b)] = <a s|rho|b s'>``.  ``dE = tr[V dn]``, i.e. ``V_ab = dE/dn_ba``.

* Dudarev (noncollinear, full spin matrix): ``E = U_eff/2 tr[n - n n]``, ``V = U_eff/2 (1 - 2n)``.
* Liechtenstein: ``E_Hub = 1/2 sum v_abcd [N_ca N_db - sum_{ss'} n^{s's}_{da} n^{ss'}_{cb}]``, ``N = n^{uu} + n^{dd}``.
* |j mj> = sum CG(l m, 1/2 s | j mj) |m s>; Condon-Shortley phases (``L.S = l/2`` for ``j = l + 1/2``, ``-(l+1)/2`` for ``j = l - 1/2``).

    python test/migration/freeze_hubbard_spinor.py run
    python test/migration/freeze_hubbard_spinor.py verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
import sympy
from sympy.physics.quantum.cg import CG

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_hubbard_slater as S  # noqa: E402  (same directory; python -I drops the script dir from sys.path)

REF = Path(os.environ.get("HU_REF", Path(__file__).resolve().parent / "reference" / "hubbard"))
OUT = REF / "spinor.json"
CASES = {"d": 2, "f": 3}
SEED = {"d": 20261111, "f": 20261112}
U_EFF = 3.7
F_SET = S.F_SET
sig = [np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], dtype=complex)]


def fs(x) -> str:
    return S.fs(Fraction(x))


def cplx(a):
    return S.cplx(a)


def dud(n, U):
    """E, V of the noncollinear Dudarev functional."""
    return 0.5 * U * np.trace(n - n @ n).real, 0.5 * U * (np.eye(n.shape[0]) - 2 * n)


def blocks(n):
    p = n.shape[0] // 2
    return n.reshape(2, p, 2, p)


def e_hub_nc(v, n):
    n4 = blocks(n)
    N = n4[0, :, 0, :] + n4[1, :, 1, :]
    return 0.5 * (np.einsum("abcd,ca,db->", v, N, N) - np.einsum("abcd,ydsa,scyb->", v, n4, n4)).real


def v_hub_nc(v, n):
    """The plan's V^{ss'}_{ac} = delta_ss' sum v_abcd N_db - sum v_abdc n^{ss'}_{db}, assembled spin-major."""
    n4 = blocks(n)
    p = n4.shape[1]
    N = n4[0, :, 0, :] + n4[1, :, 1, :]
    direct = np.einsum("abcd,db->ac", v, N)
    V4 = -np.einsum("abdc,sdtb->satc", v, n4)
    for s in (0, 1):
        V4[s, :, s, :] += direct
    return V4.reshape(2 * p, 2 * p)


def e_det(v, c):
    """Explicit antisymmetrized energy of orbitals c[:, i] (2p vectors, spin-major): 1/2 sum_{i != j} <ij|v|ij> - <ij|v|ji>."""
    p = c.shape[0] // 2
    c4 = c.reshape(2, p, -1)
    tot = 0.0
    nocc = c.shape[1]
    for i in range(nocc):
        for j in range(nocc):
            if i == j:
                continue
            ci, cj = c4[:, :, i], c4[:, :, j]                              # [s, a]
            direct = np.einsum("as,bt,cs,dt,abcd->", ci.T.conj(), cj.T.conj(), ci.T, cj.T, v)
            exch = np.einsum("as,bt,cs,dt,abcd->", ci.T.conj(), cj.T.conj(), cj.T, ci.T, v)
            tot += (direct - exch).real
    return 0.5 * tot


def rand_n2(rng, dim):
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    _, q = np.linalg.eigh(a + a.conj().T)
    return (q * rng.uniform(0.0, 1.0, size=dim)) @ q.conj().T


def spin_rot(rng, p):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = float(rng.uniform(0.3, 3.0))
    u = scipy.linalg.expm(-0.5j * th * sum(a * s for a, s in zip(ax, sig)))
    return np.kron(u, np.eye(p)), u


def stage(name, l):
    p, dim = 2 * l + 1, 2 * (2 * l + 1)
    rng = np.random.default_rng(SEED[name])
    doc = {"l": l, "p": p, "dim": dim, "U_eff": U_EFF}

    # ---- Dudarev on the full spin matrix
    n = rand_n2(rng, dim)
    E, V = dud(n, U_EFF)
    H = S.rand_herm(rng, dim)

    def f(t):
        return dud(n + t * H, U_EFF)[0]
    d = lambda h: (f(h) - f(-h)) / (2 * h)                                    # noqa: E731
    fd = (4 * d(0.04) - d(0.08)) / 3
    d_ = {"n": cplx(n), "E": float(E), "V": cplx(V), "direction": cplx(H),
          "eigenvalues_n_min_max": [float(np.linalg.eigvalsh(n)[0]), float(np.linalg.eigvalsh(n)[-1])],
          "fd_richardson": float(fd), "tr_V_H": float(np.trace(V @ H).real),
          "controls": {"tr_VT_H": float(np.trace(V.T @ H).real), "tr_conjV_H": float(np.trace(V.conj() @ H).real),
                       "tr_1.01V_H": float(1.01 * np.trace(V @ H).real)},
          "note": "E is quadratic in n: central FD is exact to roundoff; the direction has a non-symmetric imaginary part so V^T fails",
          "orientation": "dE = tr[V dn], V_ab = dE/dn_ba"}
    d_["fd_minus_tr_V_H"] = d_["fd_richardson"] - d_["tr_V_H"]
    # U(2p) invariance
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    E2, V2 = dud(Q @ n @ Q.conj().T, U_EFF)
    d_["U2p_invariance"] = {"E_diff": float(E2 - E), "V_covariance": float(np.abs(V2 - Q @ V @ Q.conj().T).max())}
    W, u = spin_rot(rng, p)
    E3, V3 = dud(W @ n @ W.conj().T, U_EFF)
    d_["SU2_spin_rotation"] = {"E_diff": float(E3 - E), "V_covariance": float(np.abs(V3 - W @ V @ W.conj().T).max()),
                               "u": cplx(u)}
    # block diagonal n == two collinear channels
    nb = np.zeros((dim, dim), dtype=complex)
    nb[:p, :p], nb[p:, p:] = rand_n2(rng, p), rand_n2(rng, p)
    Eb = dud(nb, U_EFF)[0]
    two = 0.5 * U_EFF * sum(np.trace(nb[s * p:(s + 1) * p, s * p:(s + 1) * p] - nb[s * p:(s + 1) * p, s * p:(s + 1) * p] @ nb[s * p:(s + 1) * p, s * p:(s + 1) * p]).real for s in (0, 1))
    d_["block_diagonal_equals_two_channel_collinear"] = float(Eb - two)
    # a collinear-only code drops the spin-flip blocks: how much does that change E for this n
    nd = n.copy()
    nd[:p, p:] = 0
    nd[p:, :p] = 0
    d_["control_spin_flip_blocks_dropped_dE"] = float(dud(nd, U_EFF)[0] - E)
    # Dudarev = noncollinear Liechtenstein with F0 = U_eff, F_k = 0 (J = 0), derived analytically and checked here
    ak = S.ak_tensors(l)
    af = {k: S.ffloat(ak[k]) for k in ak}
    ks = list(range(0, 2 * l + 1, 2))
    v0 = U_EFF * af[0].astype(complex)
    Ncount = np.trace(n).real
    e_u_from_lie = e_hub_nc(v0, n) - 0.5 * U_EFF * Ncount * (Ncount - 1)
    d_["liechtenstein_F0_only_minus_dudarev_E"] = float(e_u_from_lie - E)
    doc["dudarev"] = d_

    # ---- spin-orbit atom: L.S, CG projectors, analytic +U numbers
    Lx, Ly, Lz = S.ladder(l)
    LS = sum(np.kron(s / 2, L) for s, L in zip(sig, (Lx, Ly, Lz)))
    ev = np.linalg.eigvalsh(LS)
    lam = {"up": Fraction(l, 2), "dn": Fraction(-(l + 1), 2)}

    def cg_vec(j2, mj2):
        """|j mj> in spin-major layout; j2 = 2j, mj2 = 2 mj (integers)."""
        v = np.zeros(dim)
        for s, ms2 in ((0, 1), (1, -1)):
            m2 = mj2 - ms2
            if abs(m2) > 2 * l or (m2 % 2) != 0:
                continue
            m = m2 // 2
            v[s * p + m + l] = float(CG(sympy.Integer(l), sympy.Integer(m), sympy.Rational(1, 2), sympy.Rational(ms2, 2),
                                       sympy.Rational(j2, 2), sympy.Rational(mj2, 2)).doit())
        return v
    jj = {"j=l+1/2": 2 * l + 1, "j=l-1/2": 2 * l - 1}
    so = {"layout_spin_major_alpha_first": True, "LS_eigenvalues_expected": {k: fs(v) for k, v in lam.items()},
          "LS_eigenvalue_residual_distinct": float(max(abs(ev - np.array(sorted([float(lam['dn'])] * (2 * l) + [float(lam['up'])] * (2 * l + 2)))))),
          "projectors": {}}
    P = {}
    for key, j2 in jj.items():
        mjs = list(range(-j2, j2 + 1, 2))
        vecs = np.array([cg_vec(j2, m) for m in mjs])
        Pj = vecs.T @ vecs
        # explicit orthonormality + closed-form m,s-resolved weights
        gram = vecs @ vecs.T
        lamj = float(lam["up"] if j2 == 2 * l + 1 else lam["dn"])
        diag = np.diag(Pj).reshape(2, p)
        if j2 == 2 * l + 1:
            closed = np.array([[Fraction(l + m + 1, p) for m in range(-l, l + 1)], [Fraction(l - m + 1, p) for m in range(-l, l + 1)]])
        else:
            closed = np.array([[Fraction(l - m, p) for m in range(-l, l + 1)], [Fraction(l + m, p) for m in range(-l, l + 1)]])
        P[key] = Pj
        so["projectors"][key] = {
            "2j": j2, "rank": len(mjs), "trace": float(np.trace(Pj)), "gram_residual": float(np.abs(gram - np.eye(len(mjs))).max()),
            "P2_minus_P": float(np.abs(Pj @ Pj - Pj).max()), "LS_P_minus_lambda_P": float(np.abs(LS @ Pj - lamj * Pj).max()),
            "diag_up_dn_exact": [[fs(x) for x in row] for row in closed],
            "diag_residual_vs_exact": float(np.abs(diag - np.vectorize(float)(closed)).max()),
            "P_real_part": np.real(Pj).tolist(), "P_imag_max": float(np.abs(Pj.imag).max()),
            "spin_flip_block_max_abs": float(np.abs(Pj[:p, p:]).max()),
            "CG_first_state_vs_closed_formula": None}
    # spectral projectors of L.S (independent of CG) agree
    w, U_ = np.linalg.eigh(LS)
    Pspec = {k: U_[:, np.abs(w - float(lam[t])) < 1e-9] @ U_[:, np.abs(w - float(lam[t])) < 1e-9].conj().T for k, t in (("j=l+1/2", "up"), ("j=l-1/2", "dn"))}
    so["spectral_projector_vs_CG_projector"] = {k: float(np.abs(Pspec[k] - P[k]).max()) for k in P}
    so["sum_of_projectors_minus_identity"] = float(np.abs(P["j=l+1/2"] + P["j=l-1/2"] - np.eye(dim)).max())
    # closed-form CG mixing: <m s | j mj> for the highest mj, compared with sqrt((l +- mj + 1/2)/(2l+1))
    mjt = 2 * l + 1                                                          # 2*mj of the stretched state, fully polarized
    st = cg_vec(2 * l + 1, 2 * l + 1)
    so["stretched_state_is_m_l_up"] = float(abs(st[0 * p + 2 * l] - 1.0))
    mj1 = cg_vec(2 * l + 1, 1)                                               # mj = 1/2 of j = l + 1/2
    mj2 = cg_vec(2 * l - 1, 1)
    so["mixing_mj_1/2"] = {"j=l+1/2_up_weight_vs_sqrt((l+1)/(2l+1))": float(abs(mj1[0 * p + l] - math.sqrt((l + 1) / (2 * l + 1)))),
                           "j=l+1/2_dn_weight_vs_sqrt(l/(2l+1))": float(abs(mj1[1 * p + l + 1] - math.sqrt(l / (2 * l + 1)))),
                           "j=l-1/2_up_weight_vs_-sqrt(l/(2l+1))": float(abs(mj2[0 * p + l] + math.sqrt(l / (2 * l + 1)))),
                           "j=l-1/2_dn_weight_vs_sqrt((l+1)/(2l+1))": float(abs(mj2[1 * p + l + 1] - math.sqrt((l + 1) / (2 * l + 1))))}
    del mjt
    cases = []
    rk = {k: so["projectors"][k]["rank"] for k in P}

    def add(label, fr):
        """fr = {key: Fraction occupation per state of that j shell}; exact E_U/U_eff and the float cross-check."""
        N = sum(f_ * rk[k] for k, f_ in fr.items())
        tr2 = sum(f_ * f_ * rk[k] for k, f_ in fr.items())
        exact = Fraction(1, 2) * (N - tr2)
        nm = sum(float(f_) * P[k] for k, f_ in fr.items())
        Em, Vm = dud(nm, 1.0)
        evV = {k: fs(Fraction(1, 2) * (1 - 2 * f_)) for k, f_ in fr.items()}
        dropped = nm.copy()
        dropped[:p, p:] = 0
        dropped[p:, :p] = 0
        cases.append({"label": label, "occupation_per_state": {k: fs(f_) for k, f_ in fr.items()}, "N": fs(N),
                      "E_U_over_U_eff_exact": fs(exact), "E_U_over_U_eff_float": float(Em),
                      "float_minus_exact": float(Em - float(exact)),
                      "V_over_U_eff_eigenvalue_per_j_shell": evV,
                      "V_residual_vs_closed_form": float(np.abs(Vm - sum((0.5 - float(fr.get(k, 0))) * P[k] for k in P)).max()),
                      "control_spin_flip_blocks_dropped_E_U_over_U_eff": float(dud(dropped, 1.0)[0]),
                      "control_m_s_diagonal_only_E_U_over_U_eff": float(dud(np.diag(np.diag(nm)), 1.0)[0])})
    hi, lo = "j=l+1/2", "j=l-1/2"
    add("one electron, uniform on the j=l-1/2 shell", {lo: Fraction(1, rk[lo])})
    add("one electron, uniform on the j=l+1/2 shell", {hi: Fraction(1, rk[hi])})
    add("three electrons uniform on the j=l+1/2 shell (lambda = 1/2)", {hi: Fraction(3, rk[hi])})
    add("two electrons uniform on j=l-1/2 plus one on j=l+1/2", {lo: Fraction(2, rk[lo]), hi: Fraction(1, rk[hi])})
    add("filled j=l-1/2, empty j=l+1/2", {lo: Fraction(1)})
    add("all of both shells filled", {lo: Fraction(1), hi: Fraction(1)})
    add("one electron, weight 1/3 on j=l-1/2 and 2/3 on j=l+1/2 (uniform inside each)", {lo: Fraction(1, 3 * rk[lo]), hi: Fraction(2, 3 * rk[hi])})
    # pure single state (rank-1 idempotent): E_U = 0 exactly, V eigenvalues -1/2, +1/2
    pure = np.outer(cg_vec(jj[lo], 1), cg_vec(jj[lo], 1))
    Ep, Vp = dud(pure, 1.0)
    so["pure_single_mj_state"] = {"E_U_over_U_eff": float(Ep), "V_eigenvalues_min_max": [float(np.linalg.eigvalsh(Vp)[0]), float(np.linalg.eigvalsh(Vp)[-1])]}
    so["cases"] = cases
    doc["so_atom"] = so

    # ---- Liechtenstein noncollinear: formula vs explicit determinants, invariances, V orientation, d2 pure-j pairs
    F = F_SET[name]
    v = sum(F[i] * af[k] for i, k in enumerate(ks)).astype(complex)
    lie = {"F": F}
    c = np.linalg.qr(rng.normal(size=(dim, 3)) + 1j * rng.normal(size=(dim, 3)))[0]      # 3 random noncollinear orbitals
    nd3 = c @ c.conj().T
    lie["determinant_3_random_spinors"] = {"E_explicit": float(e_det(v, c)), "E_formula": float(e_hub_nc(v, nd3)),
                                           "diff": float(e_det(v, c) - e_hub_nc(v, nd3))}
    nl = rand_n2(rng, dim)
    Hl = S.rand_herm(rng, dim)
    El = e_hub_nc(v, nl)
    Vl = v_hub_nc(v, nl)

    def g(t):
        return e_hub_nc(v, nl + t * Hl)
    dd = lambda h: (g(h) - g(-h)) / (2 * h)                                   # noqa: E731
    fdl = (4 * dd(0.04) - dd(0.08)) / 3
    lie.update({"n": cplx(nl), "E_Hub": float(El), "V_plan_formula": cplx(Vl), "direction": cplx(Hl), "fd_richardson": float(fdl),
                "tr_V_H": float(np.trace(Vl @ Hl).real), "fd_minus_tr_V_H": float(fdl - np.trace(Vl @ Hl).real),
                "controls": {"tr_VT_H": float(np.trace(Vl.T @ Hl).real), "tr_conjV_H": float(np.trace(Vl.conj() @ Hl).real),
                             "tr_1.01V_H": float(1.01 * np.trace(Vl @ Hl).real)}})
    Hi = 1j * (lambda A: A - A.T)(rng.normal(size=(dim, dim)))                  # purely imaginary Hermitian direction
    gi = lambda t: e_hub_nc(v, nl + t * Hi)                                      # noqa: E731
    di = lambda h: (gi(h) - gi(-h)) / (2 * h)                                    # noqa: E731
    lie["imaginary_direction"] = {"direction": cplx(Hi), "fd_richardson": float((4 * di(0.04) - di(0.08)) / 3),
                                  "tr_V_H": float(np.trace(Vl @ Hi).real), "control_tr_VT_H": float(np.trace(Vl.T @ Hi).real),
                                  "control_tr_conjV_H": float(np.trace(Vl.conj() @ Hi).real)}
    Wl, ul = spin_rot(rng, p)
    lie["SU2_spin_rotation_dE"] = float(e_hub_nc(v, Wl @ nl @ Wl.conj().T) - El)
    Dax = S.D_axis(l, rng.normal(size=3), float(rng.uniform(0.3, 2.5)))
    So3 = np.kron(ul, Dax)                                                       # spin rotation x orbital rotation
    lie["SO3_orbital_x_SU2_spin_dE"] = float(e_hub_nc(v, So3 @ nl @ So3.conj().T) - El)
    lie["SO3_orbital_only_dE"] = float(e_hub_nc(v, np.kron(np.eye(2), Dax) @ nl @ np.kron(np.eye(2), Dax).conj().T) - El)
    Qp, _ = np.linalg.qr(rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p)))
    lie["control_random_Up_orbital_dE"] = float(e_hub_nc(v, np.kron(np.eye(2), Qp) @ nl @ np.kron(np.eye(2), Qp).conj().T) - El)
    lie["V_covariance_under_SO3xSU2"] = float(np.abs(v_hub_nc(v, So3 @ nl @ So3.conj().T) - So3 @ Vl @ So3.conj().T).max())
    # d2 pure-j pair determinants: E_int = F0 c0 + F2 c2 + ... exact fractions (linear in F)
    pairs = {}
    for key, j2 in jj.items():
        if l != 2:
            continue
        mjs = list(range(-j2, j2 + 1, 2))
        rows = []
        for a_ in range(len(mjs)):
            for b_ in range(a_ + 1, len(mjs)):
                cc = np.stack([cg_vec(j2, mjs[a_]), cg_vec(j2, mjs[b_])], axis=1)
                co = [e_det(af[k].astype(complex), cc) for k in ks]
                fr = [Fraction(x).limit_denominator(100000) for x in co]
                ok = max(abs(float(fr_) - x) for fr_, x in zip(fr, co)) < 1e-13
                rows.append({"2mj": [mjs[a_], mjs[b_]], "coeff_F0_F2_F4": [fs(x) for x in fr], "coeff_float": co, "rational_to_1e-13": bool(ok)})
        pairs[key] = rows
    if pairs:
        lie["d2_pure_j_pairs"] = pairs
        lie["d2_pairs_plan_values_to_check"] = "plan 14.3 quotes j=3/2 pair F0 - 0.12 F2 + 0 F4 and j=5/2 pair F0 + 0.052244898 F2 + 0.009070295 F4 (fractions -3/25, 64/1225, 4/441 recognized to 9 digits)"
    doc["liechtenstein_noncollinear"] = lie
    return doc


def build():
    doc = {"stamp": stamp()}
    for name, l in CASES.items():
        doc[name] = stage(name, l)
    doc["TODO_dirac_j_resolved_sphere_fractions"] = {
        "status": "NOT DONE in this wave: needs the datasets / a Dirac atom (j-resolved all-electron partial waves are not stored in the sets)",
        "needed": "for each fr dataset element: fraction of the j = l-1/2 and j = l+1/2 radial wave inside the projector sphere, from a Dirac all-electron "
                  "atom, vs the scalar-channel fraction used as the spin-independent weight W (plan 14.3: Ni 3d, Pt 5d, Fe 3d, Au 5d, Ce 4f)",
        "plan_quotes_unverified_here": "Ni 3d 0.64224 (j=3/2) / 0.63768 (5/2) vs 0.63981 scalar; Pt 5d 0.95526 / 0.94083 vs 0.94801"}
    return doc


def stamp():
    return {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "slater_script_sha256": hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__,
            "mandacaru_code_used": False, "layout": __doc__.split("LAYOUT")[1].split("    python test")[0].strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["run", "verify"])
    a = ap.parse_args()
    doc = build()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    if a.mode == "run":
        REF.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB), peak RSS {rss:.0f} MB")
        return 0
    ref = json.loads(OUT.read_text())
    diffs = S.compare(json.loads(json.dumps(doc)), ref)
    print(f"verify: {len(diffs)} differences, peak RSS {rss:.0f} MB")
    for dd in diffs[:20]:
        print("  ", dd)
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
