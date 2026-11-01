#!/usr/bin/env python
"""Independent references for the Liechtenstein (Slater-tensor) DFT+U form (plan HL-0, gate G-L0).

Nothing here imports Mandacaru.  numpy / scipy / sympy are used as a reference generator only.  Written to
``test/migration/reference/hubbard/slater.json`` (tracked, small).  Units: the tensor entries are pure numbers
(coefficients of the Slater integrals F^k); energies are in the units of the F^k you feed.

CONVENTIONS (also written into the JSON ``stamp.conventions``)

* Complex Y_lm with the Condon-Shortley phase, orbital order m = -l..l, flat tensor index ((a*p+b)*p+c)*p+d with
  a = m1 + l, etc.  ``<ab|v|cd> = int int phi_a*(1) phi_b*(2) v(1,2) phi_c(1) phi_d(2)``: electron 1 carries (a, c),
  electron 2 carries (b, d).  Multipole expansion 1/|r1-r2| = sum_k r<^k / r>^(k+1) P_k(cos gamma), so
  ``<ab|v|cd> = sum_k F^k a_k(a,b,c,d)`` with ``a_k = c^k(a,c) c^k(d,b)`` and
  ``c^k(m,m') = sqrt(4 pi/(2k+1)) int Y_lm* Y_{k,m-m'} Y_lm' dOmega`` (real, signed square root of a rational).
* Real harmonics (m_r = -l..l): ``phi_r = sum_m T[r,m] Y_m`` with T[0] = 1 for m_r = 0;
  m_r > 0: T[m_r] = 1/sqrt2, T[-m_r] = (-1)^m_r / sqrt2   (phi = sqrt2 Re Y_{l m_r});
  m_r < 0 (mu = |m_r|): T[mu] = -i/sqrt2, T[-mu] = +i (-1)^mu / sqrt2   (phi = sqrt2 Im Y_{l mu}).
  ``a_k(real)[r1 r2 r3 r4] = sum conj(T[r1,m1]) conj(T[r2,m2]) T[r3,m3] T[r4,m4] a_k(complex)[m1 m2 m3 m4]``.
* Occupation matrix ``n_ab = <a|rho|b>`` per spin; ``E_Hub = 1/2 sum v_abcd [N_ca N_db - sum_s n^s_da n^s_cb]``;
  ``E_dc = U/2 N(N-1) - J/2 sum_s N_s(N_s-1)``; ``U = F0``; J derived below.  ``dE = sum_s tr[V^s dn^s]``.

    python test/migration/freeze_hubbard_slater.py run
    python test/migration/freeze_hubbard_slater.py verify
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
import scipy.special as sp
import sympy
from sympy.physics.wigner import wigner_3j

REF = Path(os.environ.get("HU_REF", Path(__file__).resolve().parent / "reference" / "hubbard"))
OUT = REF / "slater.json"
TOL = 1e-12            # verify: |new - stored| <= TOL * max(1, |stored|) for every float
CASES = {"d": 2, "f": 3}
F_SET = {"d": [4.0, 6.5, 4.1], "f": [4.0, 7.0, 4.5, 3.0]}   # arbitrary fixed Slater integrals for the energy examples
SEED = {"d": 20261101, "f": 20261102}


# ----------------------------------------------------------------------------------------------- exact algebra
# A value is a "surd": {s: Fraction} meaning sum_s coef_s * sqrt(s), s a squarefree positive integer (key 1 = rational).
def sqfree(n: int):
    """n = r^2 * s with s squarefree -> (r, s)."""
    r, s = 1, 1
    for p_, e in sympy.factorint(n).items():
        r *= p_ ** (e // 2)
        s *= p_ ** (e % 2)
    return r, s


def sqrt_frac(q: Fraction):
    """sqrt(q) = coef * sqrt(s) -> (Fraction coef, int s)."""
    if q == 0:
        return Fraction(0), 1
    r, s = sqfree(q.numerator * q.denominator)
    return Fraction(r, q.denominator), s


def sadd(a, b):
    out = dict(a)
    for s, c in b.items():
        out[s] = out.get(s, 0) + c
        if out[s] == 0:
            del out[s]
    return out


def smul_sqrt(a, s2):
    """a * sqrt(s2)"""
    out = {}
    for s, c in a.items():
        r, s3 = sqfree(s * s2)
        out[s3] = out.get(s3, 0) + c * r
    return {s: c for s, c in out.items() if c != 0}


def rat(a) -> Fraction:
    assert set(a) <= {1}, a
    return a.get(1, Fraction(0))


def rsum(it) -> Fraction:
    out = {}
    for v in it:
        out = sadd(out, v)
    return rat(out)


def sfloat(a) -> float:
    return float(sum(float(c) * math.sqrt(s) for s, c in a.items()))


def sstr(a) -> str:
    return " + ".join(f"{fs(c)}" + ("" if s == 1 else f"*sqrt({s})") for s, c in sorted(a.items())) or "0"


def fs(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}"


def ck_table(l: int, k: int):
    """c^k(m,m') as (sign, Fraction square), indexed [m+l][m'+l]; zero entries are (0, 0)."""
    p = 2 * l + 1
    w000 = wigner_3j(l, k, l, 0, 0, 0)
    out = [[(0, Fraction(0))] * p for _ in range(p)]
    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            q = m - mp
            if abs(q) > k:
                continue
            w = wigner_3j(l, k, l, -m, q, mp)
            c = (-1) ** (m % 2) * (2 * l + 1) * w000 * w          # = c^k(m,m') exactly (sympy surd)
            if c == 0:
                continue
            sq = sympy.nsimplify(c * c)
            assert sq.is_Rational, (l, k, m, mp, sq)
            out[m + l][mp + l] = (1 if c > 0 else -1, Fraction(int(sq.p), int(sq.q)))
    return out


def ak_tensors(l: int):
    """{k: object ndarray(p,p,p,p) of surds}, k = 0..2l.  a_k(a,b,c,d) = c^k(a,c) c^k(d,b) (one q: a-c = d-b)."""
    p = 2 * l + 1
    res = {}
    for k in range(0, 2 * l + 1):
        tab = ck_table(l, k)
        t = np.empty((p, p, p, p), dtype=object)
        for i in np.ndindex(p, p, p, p):
            t[i] = {}
        for a in range(p):
            for c in range(p):
                s1, q1 = tab[a][c]
                if s1 == 0:
                    continue
                for d in range(p):
                    for b in range(p):
                        s2, q2 = tab[d][b]
                        if s2 == 0 or a - c != d - b:
                            continue
                        co, s = sqrt_frac(q1 * q2)
                        t[a, b, c, d] = {s: s1 * s2 * co}
        res[k] = t
    return res


def real_T(l: int):
    """T as exact rows: list of {m_complex: (re, im)} in units of 1/sqrt2 (m_r = 0 row has unit 1), + float matrix."""
    rows, p = [], 2 * l + 1
    Tf = np.zeros((p, p), dtype=complex)
    for r in range(-l, l + 1):
        mu = abs(r)
        if r == 0:
            row = {0: (1, 0)}
            Tf[r + l, l] = 1.0
        elif r > 0:
            row = {mu: (1, 0), -mu: ((-1) ** mu, 0)}
            Tf[r + l, mu + l] = 1 / math.sqrt(2)
            Tf[r + l, -mu + l] = (-1) ** mu / math.sqrt(2)
        else:
            row = {mu: (0, -1), -mu: (0, (-1) ** mu)}
            Tf[r + l, mu + l] = -1j / math.sqrt(2)
            Tf[r + l, -mu + l] = 1j * (-1) ** mu / math.sqrt(2)
        rows.append(row)
    return rows, Tf


def to_real_exact(l: int, ak):
    rows, _ = real_T(l)
    p = 2 * l + 1
    out = {}
    for k, t in ak.items():
        r = np.empty((p, p, p, p), dtype=object)
        for i in np.ndindex(p, p, p, p):
            nroot = sum(0 if x == l else 1 for x in i)
            acc = {}                                             # s -> [re, im]
            for m1, (a1, b1) in rows[i[0]].items():
                for m2, (a2, b2) in rows[i[1]].items():
                    for m3, (a3, b3) in rows[i[2]].items():
                        for m4, (a4, b4) in rows[i[3]].items():
                            v = t[m1 + l, m2 + l, m3 + l, m4 + l]
                            if not v:
                                continue
                            zr, zi = 1, 0                        # exact Gaussian-integer product, conj on the first two
                            for (xr, xi), cj in (((a1, b1), True), ((a2, b2), True), ((a3, b3), False), ((a4, b4), False)):
                                xi = -xi if cj else xi
                                zr, zi = zr * xr - zi * xi, zr * xi + zi * xr
                            for s, c in v.items():
                                e = acc.setdefault(s, [Fraction(0), Fraction(0)])
                                e[0] += c * zr
                                e[1] += c * zi
            val = {}
            for s, (re, im) in acc.items():
                assert im == 0, (k, i, im)
                if re != 0:
                    val = sadd(val, {s: re / 2 ** (nroot // 2)})
            if nroot % 2:                                        # one leftover 1/sqrt2 = sqrt2/2
                val = {s: c / 2 for s, c in smul_sqrt(val, 2).items()}
            r[i] = val
        out[k] = r
    return out


def pack(t):
    flat = [(int(i), v) for i, v in enumerate(t.reshape(-1)) if v]
    return {"idx": [i for i, _ in flat], "val": [sstr(v) for _, v in flat]}


def ffloat(t):
    return np.vectorize(sfloat, otypes=[float])(t)


# ----------------------------------------------------------------------------------------------- harmonics / quadrature
def ylm_matrix(l: int, x, phi):
    """Y_lm(theta, phi) for m = -l..l (rows), own implementation from the associated Legendre function (CS phase)."""
    out = np.empty((2 * l + 1, x.size), dtype=complex)
    for m in range(0, l + 1):
        nrm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
        y = nrm * sp.lpmv(m, l, x) * np.exp(1j * m * phi)
        out[m + l] = y
        out[-m + l] = (-1) ** m * np.conj(y)
    return out


def sphere_grid(nt, nph):
    x, w = np.polynomial.legendre.leggauss(nt)
    phi = 2 * np.pi * np.arange(nph) / nph
    X, PH = np.meshgrid(x, phi, indexing="ij")
    W = np.outer(w, np.full(nph, 2 * np.pi / nph))
    X, PH, W = X.ravel(), PH.ravel(), W.ravel()
    s = np.sqrt(1 - X * X)
    return X, PH, W, np.stack([s * np.cos(PH), s * np.sin(PH), X], axis=1)


def quad_tensor(l, kernel_fn, nt, nph):
    """A[a,b,c,d] = int int Ya*(1) Yc(1) K(1,2) Yb*(2) Yd(2)."""
    p = 2 * l + 1
    X, PH, W, u = sphere_grid(nt, nph)
    Y = ylm_matrix(l, X, PH)
    rho = (np.conj(Y)[:, None, :] * Y[None, :, :] * W).reshape(p * p, -1)    # [(a,c), point], weights included
    G = u @ u.T
    K = kernel_fn(G)
    left = rho @ K                                                          # [(a,c), j]
    A = (left @ rho.T)                                                      # [(a,c), (b,d)] with rho2 = conj(Y_b) Y_d W
    # rho.T used conj(Y_b)Y_d W for electron 2: same array.  index (a,c),(b,d) -> [a,b,c,d]
    return A.reshape(p, p, p, p).transpose(0, 2, 1, 3)


# ----------------------------------------------------------------------------------------------- Hubbard formulas
def e_hub(v, n_up, n_dn):
    """Collinear Liechtenstein E_Hub for tensor v (p^4, <ab|v|cd>), n^s_ab = <a|rho|b>."""
    N = n_up + n_dn
    hartree = np.einsum("abcd,ca,db->", v, N, N)
    exch = sum(np.einsum("abcd,da,cb->", v, n, n) for n in (n_up, n_dn))
    return 0.5 * (hartree - exch)


def e_dc(U, J, Nu, Nd):
    N = Nu + Nd
    return 0.5 * U * N * (N - 1) - 0.5 * J * (Nu * (Nu - 1) + Nd * (Nd - 1))


def v_hub(v, n_up, n_dn, U, J):
    N = n_up + n_dn
    Nt, Nus, Nds = np.trace(N).real, np.trace(n_up).real, np.trace(n_dn).real
    p = N.shape[0]
    out = []
    for n, Ns in ((n_up, Nus), (n_dn, Nds)):
        V = np.einsum("acbd,dc->ab", v, N) - np.einsum("acdb,dc->ab", v, n)
        V = V - np.eye(p) * (U * (Nt - 0.5) - J * (Ns - 0.5))
        out.append(V)
    return out


def rand_n(rng, p):
    a = rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p))
    _, q = np.linalg.eigh(a + a.conj().T)
    return (q * rng.uniform(0.05, 0.95, size=p)) @ q.conj().T


def rand_herm(rng, p):
    a = rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p))
    return (a + a.conj().T) / 2


def cplx(a):
    a = np.asarray(a)
    return {"re": a.real.tolist(), "im": a.imag.tolist()}


# ----------------------------------------------------------------------------------------------- rotations
def ladder(l):
    p = 2 * l + 1
    m = np.arange(-l, l + 1)
    Lp = np.zeros((p, p))
    for i in range(p - 1):
        Lp[i + 1, i] = math.sqrt(l * (l + 1) - m[i] * (m[i] + 1))
    Lz = np.diag(m.astype(float))
    return (Lp + Lp.T) / 2, (Lp - Lp.T) / (2j), Lz


def small_d(l, beta):
    p = 2 * l + 1
    d = np.zeros((p, p))
    c, s = math.cos(beta / 2), math.sin(beta / 2)
    f = math.factorial
    for mp in range(-l, l + 1):
        for m in range(-l, l + 1):
            tot = 0.0
            for t in range(0, 2 * l + 1):
                a, b, e, g = l + m - t, t, mp - m + t, l - mp - t
                if min(a, b, e, g) < 0:
                    continue
                tot += ((-1) ** (mp - m + t) * math.sqrt(f(l + mp) * f(l - mp) * f(l + m) * f(l - m))
                        / (f(a) * f(b) * f(e) * f(g)) * c ** (2 * l + m - mp - 2 * t) * s ** (mp - m + 2 * t))
            d[mp + l, m + l] = tot
    return d


def rot_axis(n, th):
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def axis_angle(R):
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    n = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return n, th


def D_axis(l, n, th):
    Lx, Ly, Lz = ladder(l)
    n = n / np.linalg.norm(n)
    return scipy.linalg.expm(-1j * th * (n[0] * Lx + n[1] * Ly + n[2] * Lz))


def spherical(r):
    return np.arccos(np.clip(r[:, 2], -1, 1)), np.arctan2(r[:, 1], r[:, 0])


# ----------------------------------------------------------------------------------------------- the stages
def stage(name, l):
    p, ks = 2 * l + 1, list(range(0, 2 * l + 1, 2))
    rng = np.random.default_rng(SEED[name])
    doc = {"l": l, "p": p, "ks": ks, "index_order": "flat = ((a*p+b)*p+c)*p+d; orbital index = m + l (complex) or m_r + l (real)"}
    ak = ak_tensors(l)
    for k in range(0, 2 * l + 1):
        if k % 2:
            assert not any(ak[k].reshape(-1)), "odd k must vanish by parity"
    rows, Tf = real_T(l)
    akr = to_real_exact(l, ak)
    doc["c_k_complex"] = {str(k): [[(("-" if s < 0 else "") + "sqrt(" + fs(q) + ")") if s else "0" for (s, q) in r]
                                   for r in ck_table(l, k)] for k in ks}
    doc["a_k_complex"] = {str(k): pack(ak[k]) for k in ks}
    doc["a_k_real"] = {str(k): pack(akr[k]) for k in ks}
    doc["T_real"] = cplx(Tf)
    T_unitary = float(np.abs(Tf @ Tf.conj().T - np.eye(p)).max())
    af = {k: ffloat(ak[k]) for k in ak}
    arf = {k: ffloat(akr[k]) for k in akr}
    v_of = lambda F, a: sum(F[i] * a[k] for i, k in enumerate(ks))          # noqa: E731

    # --- exact sum rules (all Fractions)
    sr = {}
    for k in ks:
        Sk = rsum(ak[k][a, b, b, a] for a in range(p) for b in range(p))
        diag = rsum(ak[k][a, a, a, a] for a in range(p))
        direct = rsum(ak[k][a, b, a, b] for a in range(p) for b in range(p))
        w000 = wigner_3j(l, k, l, 0, 0, 0)
        sq = sympy.nsimplify(w000 ** 2)
        sr[str(k)] = {"sum_ab_abab": fs(direct), "sum_ab_abba": fs(Sk), "sum_a_aaaa": fs(diag),
                      "3j_lkl_000_squared": f"{sq.p}/{sq.q}",
                      "S_k_equals_p2_3j2": Sk == p * p * Fraction(int(sq.p), int(sq.q)),
                      "real_basis_sum_ab_abba": fs(rsum(akr[k][a, b, b, a] for a in range(p) for b in range(p))),
                      "real_basis_sum_ab_abab": fs(rsum(akr[k][a, b, a, b] for a in range(p) for b in range(p)))}
    sr["hermiticity_ab_swap_and_exchange_symmetry"] = bool(all(
        np.array_equal(ak[k], ak[k].transpose(2, 3, 0, 1)) and np.array_equal(ak[k], ak[k].transpose(1, 0, 3, 2)) for k in ks))
    sr["real_basis_hermiticity_symmetry"] = bool(all(
        np.array_equal(akr[k], akr[k].transpose(2, 3, 0, 1)) and np.array_equal(akr[k], akr[k].transpose(1, 0, 3, 2)) for k in ks))
    sr["T_unitarity_residual"] = T_unitary
    doc["sum_rules"] = sr

    # --- J derivation
    X = {k: rsum(ak[k][a, b, b, a] for a in range(p) for b in range(p)) for k in ks}       # sum_ab v_abba coefficient
    dg = {k: rsum(ak[k][a, a, a, a] for a in range(p)) for k in ks}
    Ud = {k: rsum(ak[k][a, b, a, b] for a in range(p) for b in range(p)) for k in ks}
    assert Ud[0] == p * p and all(Ud[k] == 0 for k in ks if k)
    Jfill = {k: (X[k] - (p if k == 0 else 0)) / (p * (p - 1)) for k in ks}                 # (X - p F0)/(p(p-1))
    Jneq = {k: (X[k] - dg[k]) / (p * (p - 1)) for k in ks}                                  # a != b average of v_abba
    Jquad = {k: X[k] / (p * p) for k in ks}                                                  # a, b both, quadratic match
    assert Jfill[0] == 0
    tb = {"d": {2: Fraction(1, 14), 4: Fraction(1, 14)},
          "f": {2: Fraction(286, 6435), 4: Fraction(195, 6435), 6: Fraction(250, 6435)}}[name]
    doc["J"] = {
        "definition": "J is fixed by the filled-shell identity E_Hub(n=1) - E_dc = 0 with U = F0: "
                      "J = (sum_ab v_abba - p F0)/(p(p-1)) = sum_{k>=2} F_k S_k/(p(p-1)); equivalently U-J = mean over a!=b of (v_abab - v_abba)",
        "J_filled_shell_coeff": {str(k): fs(Jfill[k]) for k in ks},
        "J_textbook_coeff": {str(k): fs(tb[k]) for k in ks if k},
        "derived_equals_textbook": all(Jfill[k] == tb[k] for k in ks if k),
        "J_mean_a_neq_b_of_v_abba_coeff": {str(k): fs(Jneq[k]) for k in ks},
        "J_neq_equals_filled": all(Jfill[k] == Jneq[k] for k in ks),
        "pair_mean_of_(v_abab-v_abba)_over_a_neq_b_equals_F0_minus_J": all(
            Fraction(p * p * (1 if k == 0 else 0) - X[k], p * (p - 1)) == (1 if k == 0 else 0) - Jfill[k] for k in ks),
        "J_quadratic_match_X_over_p2_coeff": {str(k): fs(Jquad[k]) for k in ks},
        "note": "the k = 0 piece of sum_ab v_abba is p F0 (a = b only): it is the self-pair that FLL's N_s(N_s-1) removes; "
                "X/p^2 keeps it (F0/p), J_filled drops it and is the unique J making filled AND empty shells exact zero",
        "U_average": fs(Fraction(Ud[0], p * p)) + " * F0 (sum_ab v_abab / p^2)",
    }

    # --- Slater determinants: coefficient vectors over F_k
    def e_int(orbs, a):
        """orbs = list of (orbital index, spin); returns {k: Fraction} of 1/2 sum_{i!=j}[v_ijij - d_s v_ijji]."""
        res = {}
        for k in ks:
            t = a[k]
            s = Fraction(0)
            for i, (mi, si) in enumerate(orbs):
                for j, (mj, sj) in enumerate(orbs):
                    if i == j:
                        continue
                    s += rat(t[mi, mj, mi, mj]) - (rat(t[mi, mj, mj, mi]) if si == sj else 0)
            res[k] = s / 2
        return res

    def e_dc_vec(orbs):
        Nu = sum(1 for _, s in orbs if s == 0)
        Nd = len(orbs) - Nu
        N = Nu + Nd
        vec = {}
        for k in ks:
            vec[k] = (Fraction(N * (N - 1), 2) if k == 0 else 0) - Jfill[k] * Fraction(Nu * (Nu - 1) + Nd * (Nd - 1), 2)
        return vec

    cmpl = lambda ms, spins: [(m + l, s) for m, s in zip(ms, spins)]                      # noqa: E731
    cases = {
        "one_electron_m0_up": (cmpl([0], [0]), "complex"),
        f"all_up_{p}_electrons": (cmpl(range(-l, l + 1), [0] * p), "complex"),
        f"filled_{2 * p}_electrons": (cmpl(list(range(-l, l + 1)) * 2, [0] * p + [1] * p), "complex"),
        "three_up_m_top_three": (cmpl([l, l - 1, l - 2], [0, 0, 0]), "complex"),
        "three_mixed_m_l_up__m_-1_up__m_0_down": (cmpl([l, -1, 0], [0, 0, 1]), "complex"),
        "two_up_m_l__m_l-1": (cmpl([l, l - 1], [0, 0]), "complex"),
        "two_opposite_spin_same_orbital_m0": (cmpl([0, 0], [0, 1]), "complex"),
    }
    real_cases = {
        "real_three_mixed_(-2)_up__(-1)_up__(0)_down" if l == 2 else "real_three_mixed_(-3)_up__(-1)_up__(0)_down": (
            [(-l + l, 0), (-1 + l, 0), (0 + l, 1)], "real"),
        "real_all_up": ([(i, 0) for i in range(p)], "real"),
        "real_three_up_(0)(+1)(+2)": ([(l, 0), (l + 1, 0), (l + 2, 0)], "real"),
    }
    dets = {}
    for nm, (orbs, basis) in {**cases, **real_cases}.items():
        a = ak if basis == "complex" else akr
        ei = e_int(orbs, a)
        ed = e_dc_vec(orbs)
        eu = {k: ei[k] - ed[k] for k in ks}
        dets[nm] = {"basis": basis, "orbitals_index_spin": [[int(o), int(s)] for o, s in orbs],
                    "E_int_coeff": {str(k): fs(ei[k]) for k in ks}, "E_dc_coeff": {str(k): fs(ed[k]) for k in ks},
                    "E_U_coeff": {str(k): fs(eu[k]) for k in ks}}
    # identity checks with the textbook pair formula
    npair = p * (p - 1) // 2
    allup = dets[f"all_up_{p}_electrons"]["E_int_coeff"]
    dets["identity_all_up_equals_pairs_times_(F0-J)"] = all(
        Fraction(allup[str(k)]) == npair * ((1 if k == 0 else 0) - Jfill[k]) for k in ks)
    dets["identity_filled_and_one_electron_E_U_zero"] = all(
        Fraction(dets[nm]["E_U_coeff"][str(k)]) == 0 for nm in (f"filled_{2 * p}_electrons", "one_electron_m0_up", f"all_up_{p}_electrons") for k in ks)
    # basis independence: the same determinant is invariant under the real<->complex rotation of a full shell
    dets["basis_independence_all_up_complex_vs_real"] = dets["real_all_up"]["E_int_coeff"] == dets[f"all_up_{p}_electrons"]["E_int_coeff"]
    doc["determinants"] = dets

    # --- E_Hub, V, FD for a random collinear n (floats) and the determinant cross-check of the formula
    F = F_SET[name]
    v = v_of(F, af).astype(complex)
    nu, nd = rand_n(rng, p), rand_n(rng, p)
    Ju = sum(F[i] * float(Jfill[k]) for i, k in enumerate(ks))
    U = F[0]
    Eh = e_hub(v, nu, nd).real
    Nu_, Nd_ = np.trace(nu).real, np.trace(nd).real
    Edc = e_dc(U, Ju, Nu_, Nd_)
    Vu, Vd = v_hub(v, nu, nd, U, Ju)
    Hu, Hd = rand_herm(rng, p), rand_herm(rng, p)

    def Efun(t):
        a, b = nu + t * Hu, nd + t * Hd
        return e_hub(v, a, b).real - e_dc(U, Ju, np.trace(a).real, np.trace(b).real)
    dE = lambda h: (Efun(h) - Efun(-h)) / (2 * h)                                          # noqa: E731
    fd = (4 * dE(0.05) - dE(0.1)) / 3
    an = (np.trace(Vu @ Hu) + np.trace(Vd @ Hd)).real
    ant = (np.trace(Vu.T @ Hu) + np.trace(Vd.T @ Hd)).real
    Ec = [e_hub(af[k].astype(complex), nu, nd).real for k in ks]
    lie = {"F": F, "J_value": Ju, "E_Hub": Eh, "E_dc": Edc, "E_U": Eh - Edc,
           "E_Hub_coefficients_of_F": Ec,
           "n_up": cplx(nu), "n_dn": cplx(nd), "V_up": cplx(Vu), "V_dn": cplx(Vd), "direction_up": cplx(Hu), "direction_dn": cplx(Hd),
           "fd_richardson": fd, "tr_V_H": an, "tr_Vtranspose_H_control": ant, "fd_minus_analytic": fd - an,
           "note": "E is quadratic in n: central FD is exact to roundoff, Richardson kept for the convention",
           "dE_dn_orientation": "dE = sum_s tr[V^s dn^s]; V^s_ab = sum_cd[v_acbd N_dc - v_acdb n^s_dc] - delta_ab[U(N-1/2) - J(N_s-1/2)]"}
    # a determinant through the density-matrix formula (idempotent n) vs the explicit antisymmetrized sum
    orbs = cases["three_mixed_m_l_up__m_-1_up__m_0_down"][0]
    nup_d, ndn_d = np.zeros((p, p)), np.zeros((p, p))
    for m, s in orbs:
        (nup_d if s == 0 else ndn_d)[m, m] += 1.0
    e_formula = e_hub(v, nup_d.astype(complex), ndn_d.astype(complex)).real
    ei = dets["three_mixed_m_l_up__m_-1_up__m_0_down"]["E_int_coeff"]
    e_exact = sum(F[i] * float(Fraction(ei[str(k)])) for i, k in enumerate(ks))
    lie["determinant_formula_vs_exact_d3"] = {"E_Hub_formula": e_formula, "E_int_exact_coefficients": e_exact, "diff": e_formula - e_exact}
    # spherical (n_s = x_s * I): E_U closed form
    xs = (0.35, 0.8)
    Es = e_hub(v, xs[0] * np.eye(p, dtype=complex), xs[1] * np.eye(p, dtype=complex)).real - e_dc(U, Ju, xs[0] * p, xs[1] * p)
    Nu2, Nd2 = xs[0] * p, xs[1] * p
    Xk = sum(F[i] * float(X[k]) for i, k in enumerate(ks))
    closed = 0.5 * U * (Nu2 + Nd2) ** 2 - 0.5 * Xk / p**2 * (Nu2**2 + Nd2**2) - e_dc(U, Ju, Nu2, Nd2)
    lie["spherical_n_closed_form"] = {"x": list(xs), "E_U_tensor": Es, "E_U_closed": closed, "diff": Es - closed}
    doc["liechtenstein_collinear"] = lie

    # --- quadrature (floats): P_k kernel (exact) and the full 1/|r1-r2| kernel
    nt, nph = 16, 32
    err_k, odd_max = 0.0, 0.0
    for k in range(0, 2 * l + 3):
        A = quad_tensor(l, lambda G, k=k: sp.eval_legendre(k, G), nt, nph)
        assert np.abs(A.imag).max() < 1e-13
        if k in ks:
            err_k = max(err_k, float(np.abs(A.real - af[k]).max()))
        else:
            odd_max = max(odd_max, float(np.abs(A).max()))
    Ac = {k: quad_tensor(l, lambda G, k=k: sp.eval_legendre(k, G), nt, nph).real for k in ks}
    Xg, Pg, Wg, _ = sphere_grid(nt, nph)
    Yg = ylm_matrix(l, Xg, Pg)
    orth = float(np.abs((Yg * Wg) @ Yg.conj().T - np.eye(p)).max())
    # closed-form phases of the lowest harmonics (fixes the Condon-Shortley sign, which orthonormality does not)
    th, ph = 0.7, 1.9
    xx = np.array([math.cos(th)])
    Yp = ylm_matrix(2, xx, np.array([ph]))[:, 0]
    ref = {0: math.sqrt(5 / (16 * math.pi)) * (3 * math.cos(th) ** 2 - 1),
           1: -math.sqrt(15 / (8 * math.pi)) * math.sin(th) * math.cos(th) * np.exp(1j * ph),
           2: math.sqrt(15 / (32 * math.pi)) * math.sin(th) ** 2 * np.exp(2j * ph)}
    phase_err = max(abs(Yp[2 + m] - ref[m]) for m in (0, 1, 2))
    # swapped-pair negative control
    swapped = max(float(np.abs(Ac[k].transpose(0, 1, 3, 2) - af[k]).max()) for k in ks)
    # full kernel: spheres of radius 1 and 2
    r1, r2 = 1.0, 2.0
    Af = quad_tensor(l, lambda G: 1.0 / np.sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * G), 40, 80).real
    Fk = {k: r1**k / r2 ** (k + 1) for k in ks}
    full = float(np.abs(Af - sum(Fk[k] * af[k] for k in ks)).max())
    # real basis through the quadrature of real functions
    Yr = Tf @ Yg
    real_imag = float(np.abs(Yr.imag).max())
    rho = (Yr[:, None, :] * Yr[None, :, :] * Wg).reshape(p * p, -1)
    real_err = 0.0
    for k in ks:
        K = sp.eval_legendre(k, sphere_grid(nt, nph)[3] @ sphere_grid(nt, nph)[3].T)
        A = ((rho @ K) @ rho.T).reshape(p, p, p, p).transpose(0, 2, 1, 3)
        real_err = max(real_err, float(np.abs(A - arf[k]).max()))
    doc["quadrature"] = {
        "grid": {"n_cos_theta": nt, "n_phi": nph, "full_kernel_grid": [40, 80], "radii": [r1, r2]},
        "orthonormality_residual": orth, "cs_phase_residual_vs_closed_form_Y2m": float(phase_err),
        "P_k_kernel_vs_exact_a_k_max_abs_complex": err_k, "odd_and_higher_k_max_abs": odd_max,
        "full_1_over_r12_kernel_vs_sum_a_k_F_k_max_abs": full, "F_k_of_full_kernel": [Fk[k] for k in ks],
        "real_basis_P_k_kernel_vs_exact_max_abs": real_err, "real_harmonics_max_imag": real_imag,
        "negative_control_swapped_pair_max_abs": swapped,
        "tolerance_plan_expected": "1e-12 d, 1e-11 f"}

    # --- rotations
    rot = {}
    ang = rng.uniform(0.3, 2.8, size=3)
    n_ax = rng.normal(size=3)
    n_ax /= np.linalg.norm(n_ax)
    th_ax = float(rng.uniform(0.4, 2.6))
    Lx, Ly, Lz = ladder(l)
    comm = float(np.abs(Lx @ Ly - Ly @ Lx - 1j * Lz).max())
    L2 = float(np.abs(Lx @ Lx + Ly @ Ly + Lz @ Lz - l * (l + 1) * np.eye(p)).max())
    a_, b_, c_ = ang
    Dd = np.diag(np.exp(-1j * np.arange(-l, l + 1) * a_)) @ small_d(l, b_) @ np.diag(np.exp(-1j * np.arange(-l, l + 1) * c_))
    Dx = scipy.linalg.expm(-1j * a_ * Lz) @ scipy.linalg.expm(-1j * b_ * Ly) @ scipy.linalg.expm(-1j * c_ * Lz)
    Dax = D_axis(l, n_ax, th_ax)
    Rm = rot_axis(n_ax, th_ax)
    # R = Rz(a) Ry(b) Rz(c) active rotation matrices
    Rz = lambda t: rot_axis(np.array([0.0, 0, 1]), t)                                      # noqa: E731
    Ry = lambda t: rot_axis(np.array([0.0, 1, 0]), t)                                      # noqa: E731
    Reul = Rz(a_) @ Ry(b_) @ Rz(c_)
    pts = rng.normal(size=(24, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]

    def ytest(D, R):
        th0, ph0 = spherical(pts)
        lhs = ylm_matrix(l, np.cos(th0), ph0)                                              # Y_m'(r)
        r_in = pts @ R                                                                      # R^{-1} r = R^T r
        th1, ph1 = spherical(r_in)
        rhs = ylm_matrix(l, np.cos(th1), ph1)                                               # Y_m(R^{-1} r)
        return float(np.abs(rhs - D.T @ lhs).max())
    rot["generators_commutator_residual"] = comm
    rot["L2_residual"] = L2
    rot["euler_angles"] = [float(x) for x in ang]
    rot["small_d_formula_vs_expm_euler"] = float(np.abs(Dd - Dx).max())
    rot["unitarity"] = float(np.abs(Dd @ Dd.conj().T - np.eye(p)).max())
    rot["Y_lm(R^-1 r)=sum_m' Y_m'(r) D_m'm_euler"] = ytest(Dd, Reul)
    rot["axis"] = [float(x) for x in n_ax]
    rot["angle"] = th_ax
    rot["Y_lm(R^-1 r)=sum_m' Y_m'(r) D_m'm_axis_angle"] = ytest(Dax, Rm)
    n2 = rng.normal(size=3)
    th2 = float(rng.uniform(0.4, 2.6))
    R2 = rot_axis(n2, th2)
    n3, th3 = axis_angle(Rm @ R2)
    rot["homomorphism_D1D2_vs_D(R1R2)"] = float(np.abs(Dax @ D_axis(l, n2, th2) - D_axis(l, n3, th3)).max())
    # the exact d-matrix of the Y-test convention applied to the Slater tensor
    inv_c = 0.0
    for k in ks:
        r = np.einsum("ia,jb,kc,ld,ijkl->abcd", Dax.conj(), Dax.conj(), Dax, Dax, af[k])
        inv_c = max(inv_c, float(np.abs(r - af[k]).max()))
    Dr = Tf.conj() @ Dax @ Tf.T
    inv_r = 0.0
    for k in ks:
        r = np.einsum("ia,jb,kc,ld,ijkl->abcd", Dr.conj(), Dr.conj(), Dr, Dr, arf[k])
        inv_r = max(inv_r, float(np.abs(r - arf[k]).max()))
    # a random U(p) (not a rotation) must change the tensor (the negative control)
    Q, _ = np.linalg.qr(rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p)))
    ctrl = float(max(np.abs(np.einsum("ia,jb,kc,ld,ijkl->abcd", Q.conj(), Q.conj(), Q, Q, af[k]) - af[k]).max() for k in ks))
    Dn = Dax
    rot["tensor_invariance_complex_basis"] = inv_c
    rot["tensor_invariance_real_basis"] = inv_r
    rot["D_real_basis_max_imag"] = float(np.abs(Dr.imag).max())
    rot["tensor_change_under_random_U(p)_control"] = ctrl
    nr_u, nr_d = rand_n(rng, p), rand_n(rng, p)
    e0 = e_hub(v, nr_u, nr_d).real
    e1 = e_hub(v, Dn @ nr_u @ Dn.conj().T, Dn @ nr_d @ Dn.conj().T).real
    e2 = e_hub(v, Q @ nr_u @ Q.conj().T, Q @ nr_d @ Q.conj().T).real
    rot["E_Hub_rotated_n_minus_E"] = e1 - e0
    rot["E_Hub_random_unitary_n_minus_E_control"] = e2 - e0
    rot["V_covariance_residual"] = float(max(np.abs(a - Dn @ b @ Dn.conj().T).max() for a, b in zip(
        v_hub(v, Dn @ nr_u @ Dn.conj().T, Dn @ nr_d @ Dn.conj().T, U, Ju), v_hub(v, nr_u, nr_d, U, Ju))))
    doc["rotation"] = rot
    return doc


def build():
    doc = {"stamp": stamp()}
    for name, l in CASES.items():
        doc[name] = stage(name, l)
    return doc


def stamp():
    return {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__, "platform": platform.platform(),
            "mandacaru_code_used": False,
            "conventions": __doc__.split("CONVENTIONS (also written into the JSON ``stamp.conventions``)")[1].split("    python test")[0].strip()}


def compare(new, ref, path="", out=None):
    out = [] if out is None else out
    if path == ".stamp":
        return out
    if isinstance(ref, dict):
        for k in ref:
            if k not in new:
                out.append(f"{path}.{k}: missing")
            else:
                compare(new[k], ref[k], f"{path}.{k}", out)
    elif isinstance(ref, list):
        if len(new) != len(ref):
            out.append(f"{path}: length {len(new)} != {len(ref)}")
        else:
            for i, (a, b) in enumerate(zip(new, ref)):
                compare(a, b, f"{path}[{i}]", out)
    elif isinstance(ref, float):
        if abs(new - ref) > TOL * max(1.0, abs(ref)) and not (abs(ref) < 1e-9 and abs(new) < 1e-9):
            out.append(f"{path}: {new!r} != {ref!r}")
    elif new != ref:
        out.append(f"{path}: {new!r} != {ref!r}")
    return out


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
    diffs = compare(json.loads(json.dumps(doc)), ref)
    print(f"verify: {len(diffs)} differences, peak RSS {rss:.0f} MB")
    for d in diffs[:20]:
        print("  ", d)
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
