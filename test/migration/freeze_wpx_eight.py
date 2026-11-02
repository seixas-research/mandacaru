"""The eight WP-X functions of ``periodic_paw.py`` as frozen quantities (collectors for ``freeze_wpx_functions.py``).

Each collector evaluates the live Python function on the crystal, an input perturbation that must change the array
(non-vacuity), and the independent check of the quantity (a central difference of the same sum, a sum rule, a
hermiticity or limit identity).  ``Recorder`` holds the arrays, their perturbed twins, the checks and the replay inputs.
"""
from __future__ import annotations

import copy

import numpy as np

EXTRA = [0.02, -0.01, 0.015]    # Bohr: the displacement used as the non-vacuity perturbation


def enc(a):
    a = np.asarray(a)
    if a.dtype.kind == "c":
        return {"shape": list(a.shape), "dtype": "complex128", "re": a.real.ravel().tolist(),
                "im": a.imag.ravel().tolist()}
    kind = "int64" if a.dtype.kind in "iub" else "float64"
    return {"shape": list(a.shape), "dtype": kind, "re": a.astype(kind).ravel().tolist()}


def dec(e):
    a = np.array(e["re"], dtype=float)
    if e["dtype"] == "complex128":
        a = a + 1j * np.array(e["im"], dtype=float)
    elif e["dtype"] == "int64":
        a = a.astype(np.int64)
    return a.reshape(e["shape"])


class Recorder:
    """Frozen arrays, their perturbed twins, scalar checks, and the replay inputs."""

    def __init__(self):
        self.q, self.perturbed, self.notes, self.checks, self.inputs = {}, {}, {}, {}, {}

    def add(self, name, value, perturbed=None, note=""):
        self.q[name] = np.asarray(value)
        if perturbed is not None:
            self.perturbed[name] = np.asarray(perturbed)
        self.notes[name] = note

    def check(self, name, value, gate=None, note="", scale=None):
        value = float(value)
        self.checks[name] = {"value": value, "gate": gate, "pass": None if gate is None else bool(value <= gate),
                             "note": note, "array_max": scale, "relative": None if not scale else value / scale}


def moved(crystal, atom, delta, call):
    """``call()`` with atom ``atom`` displaced by ``delta`` (Bohr); the centers are restored."""
    saved = crystal.centers[atom].copy()
    crystal.centers[atom] = saved + np.asarray(delta, dtype=float)
    try:
        return call()
    finally:
        crystal.centers[atom] = saved


def central(function, crystal, atom, d, step):
    values = []
    for sign in (+1.0, -1.0):
        delta = np.zeros(3)
        delta[d] = sign * step
        values.append(moved(crystal, atom, delta, function))
    return values


def relative(a, b):
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-300)


def basis_gradient(rec, pp, crystal, rng):
    centers = np.array(crystal.centers)
    pts = np.vstack([centers[0] + rng.uniform(-1, 1, (44, 3)) * 5.0, centers[0], centers[-1],
                     centers[0] + [1e-13, 0, 0], centers[0] + [40.0, 0, 0]])
    x, y, z = pts.T
    value = np.array([[pp.BasisGradient(f, a).evaluate(x, y, z) for a in range(3)] for f in crystal.basis])
    shifted = np.array([[pp.BasisGradient(f, a).evaluate(x + 0.01, y, z) for a in range(3)] for f in crystal.basis])
    rec.inputs["basis_gradient_points_bohr"] = enc(pts)
    rec.add("BasisGradient.evaluate", value, shifted,
            "(function, axis, point) complex; perturbed: points moved by 0.01 Bohr in x. Rows 44 and 45 are the "
            "two nuclei, 46 is 1e-13 Bohr away (nucleus branch), 47 is beyond every cutoff (zero)")
    step, worst, scale, count = 1e-5, 0.0, 0.0, 0
    for i, f in enumerate(crystal.basis):
        away = np.linalg.norm(pts - f.center, axis=1) > 1e-3
        for a in range(3):
            plus, minus = (f.evaluate(*(pts + s * step * np.eye(3)[a]).T) for s in (1, -1))
            numeric = (plus - minus) / (2 * step)
            worst = max(worst, float(np.abs(numeric - value[i, a])[away].max()))
            scale = max(scale, float(np.abs(value[i, a]).max()))
            count += int(away.sum())
    rec.check("BasisGradient.fd_abs", worst, note=f"central difference of parent.evaluate, step {step} Bohr, "
              f"{count} points away from the nucleus", scale=scale)
    nucleus = [44 if a == 0 else 45 for a in crystal.atom_of_orbital]
    p_rows = [i for i, f in enumerate(crystal.basis) if f.l == 1 and f.m == 0]
    analytic = max((abs(value[i, 2, nucleus[i]] - pp.radial_slope(crystal.basis[i], np.zeros(1))[0]
                        * np.sqrt(3 / (4 * np.pi))) for i in p_rows), default=0.0)
    rec.check("BasisGradient.nucleus_p0_analytic", analytic, note="d_z(R'(0) r Y_10) = R'(0) sqrt(3/4pi) at the nucleus")
    rec.check("BasisGradient.s_nucleus_max", max((abs(value[i, :, nucleus[i]]).max() for i, f in enumerate(crystal.basis)
                                                  if f.l == 0), default=0.0),
              note="an s function has no gradient at its own nucleus")


def radial_slope(rec, pp, crystal):
    r = np.concatenate([[0.0, 1e-8, 1e-4, 1e-3], np.geomspace(1e-3, 40.0, 140)])
    rec.inputs["radial_slope_r_bohr"] = enc(r)
    value = np.array([pp.radial_slope(f, r) for f in crystal.basis])
    rec.add("radial_slope", value, np.array([pp.radial_slope(f, r * (1 + 1e-6)) for f in crystal.basis]),
            "(function, r); perturbed: r scaled by 1+1e-6 (rows are the basis functions in crystal order)")
    exact = all(np.array_equal(pp.radial_slope(pp.BasisGradient(f, 0), r), pp.radial_slope(f, r))
                for f in crystal.basis[:4])
    rec.check("radial_slope.gradient_wrapper_unwrapped", 0.0 if exact else 1.0, 0.0, "BasisGradient unwraps to its parent")
    step, worst, scale = 1e-6, 0.0, 0.0
    for i, f in enumerate(crystal.basis):
        r0, rm = f._r[0], f._r[-1]
        keep = (np.abs(r - r0) > 3 * step) & (np.abs(r - rm) > 3 * step) & (r > 3 * step)
        numeric = (f.radial(r[keep] + step) - f.radial(r[keep] - step)) / (2 * step)
        worst = max(worst, float(np.abs(numeric - value[i][keep]).max()))
        scale = max(scale, float(np.abs(value[i]).max()))
    rec.check("radial_slope.fd_abs", worst, note=f"central difference of function.radial, step {step} Bohr, "
              f"breakpoints excluded", scale=scale)


def short_range_cross(rec, pp, crystal, rng):
    ks = np.vstack([crystal.kpoints, crystal.cartesian_kpoints([[0.17, 0.31, -0.23]])])
    M = crystal.M
    rows, cols = list(range(0, M, 2)), list(range(1, M, 2))
    grads = [pp.BasisGradient(f, a) for a in range(3) for f in crystal.basis]
    rec.inputs["short_range_cross"] = {
        "kpoints_bohr_inv": enc(ks), "rows": rows, "cols": cols,
        "gradient_bra": "BasisGradient(basis[mu], axis) at index axis*M + mu; ket = the M basis functions"}
    for atom in range(len(crystal.centers)):
        eo = crystal.short_range_cross(atom, ks, [crystal.basis[i] for i in rows], [crystal.basis[i] for i in cols])
        eo_p = crystal.short_range_cross(atom, ks + 0.01, [crystal.basis[i] for i in rows],
                                         [crystal.basis[i] for i in cols])
        rec.add(f"short_range_cross.even_odd.atom{atom}", eo, eo_p,
                "(k, even rows, odd cols); perturbed: every k shifted by 0.01 Bohr^-1 along (1,1,1)")
        full = crystal.short_range_atom(atom, ks)
        rec.check(f"short_range_cross.block_of_full.atom{atom}", np.abs(eo - full[:, rows][:, :, cols]).max(), 1e-12,
                  "absolute; the gate of test_the_cross_block_is_the_block_of_the_full_matrix")
        gb = crystal.short_range_cross(atom, ks, grads, crystal.basis)
        gb_p = crystal.short_range_cross(atom, ks + 0.01, grads, crystal.basis)
        rec.add(f"short_range_cross.gradient_bra.atom{atom}", gb, gb_p,
                "(k, axis*M+mu, nu) complex <d_axis chi_mu|V_sr,atom|chi_nu>")
        rec.check(f"short_range_cross.im_over_max.atom{atom}", np.abs(gb.imag).max() / np.abs(gb).max(), None,
                  "non-vacuity: the generic k makes the array complex (a wrong k.R sign changes it)")
        step, worst = 1e-4, 0.0
        for a in range(3):
            fd = []
            for sign in (+1, -1):
                S = []
                for f in crystal.basis:
                    g = copy.copy(f)
                    g.center = np.array(f.center, dtype=float)
                    g.center[a] += sign * step
                    S.append(g)
                fd.append(crystal.short_range_cross(atom, ks, S, S))
            G = gb[:, a * M:(a + 1) * M, :]
            predicted = -(G + G.conj().transpose(0, 2, 1))
            worst = max(worst, float(np.abs((fd[0] - fd[1]) / (2 * step) - predicted).max()))
        rec.check(f"short_range_cross.translation_fd_abs.atom{atom}", worst,
                  note=f"-(G + G^H) against the central difference (step {step} Bohr) of the matrix with every basis "
                       f"center shifted and the sphere fixed", scale=np.abs(gb).max())


def remainder(rec, pp, crystal, rng):
    n_atoms = len(crystal.centers)
    base = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, -0.7], [0.4, 2.1, 1.1], [0.5, 0, 0], [0, 0.175, 0]])
    direction = rng.normal(size=(13, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    vectors = np.vstack([base, direction * rng.uniform(0.01, 13.5, 13)[:, None]]).T
    rec.inputs["remainder_vectors_bohr_inv"] = enc(vectors)
    rec.inputs["remainder_table"] = {"step": pp.REMAINDER_TABLE_STEP, "max": pp.REMAINDER_TABLE_MAX,
                                     "note": "linear interpolation; vectors 0 and 3,4 sit on table nodes, the largest "
                                             "random vectors are past the table end"}
    first = None
    for atom in range(n_atoms):
        value = crystal.remainder_transforms(atom, vectors)
        shifted = moved(crystal, atom, EXTRA, lambda atom=atom: crystal.remainder_transforms(atom, vectors))
        rec.add(f"remainder_transforms.atom{atom}", value, shifted,
                "(a, b, vector) complex; perturbed: the atom moved by (0.02,-0.01,0.015) Bohr (a phase exp(-i g.d))")
        minus = crystal.remainder_transforms(atom, -vectors)
        rec.check(f"remainder_transforms.hermitian.atom{atom}", np.abs(value - np.conj(minus).transpose(1, 0, 2)).max(),
                  1e-9, "E_ab(g) = conj(E_ba(-g)); the gate of test_it_is_hermitian_in_the_pair")
        rec.check(f"remainder_transforms.g0_over_max.atom{atom}", np.abs(value[:, :, 0]).max() / np.abs(value).max(),
                  None, "the remainder carries no net charge: E(0) is small against the rest")
        first = value if first is None else first
        if atom == 1:
            phase = np.exp(-1j * (vectors.T @ (crystal.centers[1] - crystal.centers[0])))
            rec.check("remainder_transforms.moves_with_atom", np.abs(value - first * phase[None, None, :]).max(), 1e-9,
                      "atom 1 = atom 0 times exp(-i g.(R1-R0)) (test_it_moves_with_the_atom)")
    old_step, cache = pp.REMAINDER_TABLE_STEP, crystal.__dict__.pop("_remainder_cache", None)
    try:
        pp.REMAINDER_TABLE_STEP = old_step / 4
        fine = crystal.remainder_transforms(0, vectors)
    finally:
        pp.REMAINDER_TABLE_STEP = old_step
        crystal.__dict__["_remainder_cache"] = cache
    rec.check("remainder_transforms.table_step_error", relative(crystal.remainder_transforms(0, vectors), fine), None,
              "relative to the maximum: step 0.025 against a table four times finer (interpolation error)")


def onsite(rec, pp, crystal):
    for atom in range(len(crystal.centers)):
        W = crystal.onsite_remainder(atom)
        rec.add(f"onsite_remainder.atom{atom}", W, W.transpose(0, 2, 1, 3),
                "(a, b, c, d) Hartree; perturbed: indices (a,c,b,d) (a wrong pair convention)")
        rec.check(f"onsite_remainder.hermitian.atom{atom}", np.abs(W - np.conj(W.transpose(1, 0, 3, 2))).max(), 1e-10,
                  "conj(W[a,b,c,d]) = W[b,a,d,c]; absolute Hartree")
        rec.check(f"onsite_remainder.pair_exchange.atom{atom}", np.abs(W - W.transpose(2, 3, 0, 1)).max()
                  / np.abs(W).max(), 5e-5, "pair-exchange asymmetry, relative; a property of the quadrature "
                  "(5e-5 measured), not roundoff: do not tighten")
    columns = {a: crystal.projector_columns(a) for a in range(len(crystal.centers))}
    for atom, cols in columns.items():
        rec.add(f"projector_columns.atom{atom}", np.array(cols, dtype=np.int64), None, "T0 exact")
    rec.add("projector_lm", np.array([[p.atom_index, p.l, p.m, p.index] for p in crystal.projectors], dtype=np.int64),
            None, "(P, 4): atom, l, m, radial index of every projector in crystal order (T0 exact)")


def gradients(rec, pp, crystal):
    n = len(crystal.centers)
    rec.inputs["lattice_sums"] = {"centers_bohr": enc(crystal.centers), "lattice_bohr": enc(crystal.lattice),
                                  "sigma_bohr": crystal.sigma, "charges": enc(crystal.charges),
                                  "ion_pair_reach_sigmas": pp.ION_PAIR_REACH,
                                  "channels": [list(map(int, c)) for c in crystal.channels]}
    grad = crystal.ion_constants_gradient()
    other = (moved(crystal, n - 1, EXTRA, crystal.ion_constants_gradient) if n > 1 else None)
    rec.add("ion_constants_gradient", grad, other, "(atom, axis) Ha/Bohr; perturbed: the last atom moved by "
            "(0.02,-0.01,0.015) Bohr" if n > 1 else "one atom: zero by translation invariance; no perturbation exists")
    shifted = [c + (EXTRA if i == n - 1 else 0.0) for i, c in enumerate(crystal.centers)]
    rec.add("ion_constants", np.float64(crystal.ion_constants()), np.float64(crystal.ion_constants(shifted)) if n > 1 else None,
            "Ha, the sum whose derivative is frozen; perturbed: the last atom moved by (0.02,-0.01,0.015) Bohr")
    worst = 0.0
    for atom in range(n):
        for d in range(3):
            plus, minus = central(crystal.ion_constants, crystal, atom, d, 1e-4)
            worst = max(worst, abs(grad[atom, d] - (plus - minus) / 2e-4))
    rec.check("ion_constants_gradient.fd_abs", worst, 1e-8, "Ha/Bohr, step 1e-4 Bohr", scale=np.abs(grad).max())
    rec.check("ion_constants_gradient.sum_rule", np.abs(grad.sum(axis=0)).max(), 1e-13, "translation of every atom")
    channels = sorted(crystal.channels)
    sr = crystal.short_range_ion_compensation_gradient()
    value = np.array([sr[c] for c in channels])
    other = (moved(crystal, n - 1, EXTRA, lambda: np.array([crystal.short_range_ion_compensation_gradient()[c]
                                                            for c in channels])) if n > 1 else None)
    rec.add("short_range_ion_compensation_gradient", value, other,
            "(channel, atom, axis) complex per Bohr, channels sorted as in inputs.sr_channels; perturbed as above")
    rec.inputs["sr_channels"] = [list(map(int, c)) for c in channels]
    rec.add("short_range_ion_compensation", np.array([crystal.short_range_ion_compensation()[c] for c in channels]),
            np.array([crystal.short_range_ion_compensation(shifted)[c] for c in channels]) if n > 1 else None,
            "(channel) complex, the sum whose derivative is frozen; perturbed as ion_constants")
    worst = 0.0
    for atom in range(n):
        for d in range(3):
            plus, minus = central(crystal.short_range_ion_compensation, crystal, atom, d, 1e-4)
            numeric = np.array([(plus[c] - minus[c]) / 2e-4 for c in channels])
            worst = max(worst, float(np.abs(value[:, atom, d] - numeric).max()))
    rec.check("short_range_ion_compensation_gradient.fd_abs", worst, 1e-7,
              "per Bohr, step 1e-4 Bohr, every atom and axis", scale=np.abs(value).max())
    rec.check("short_range_ion_compensation_gradient.sum_rule", np.abs(value.sum(axis=1)).max(), 1e-12,
              "translation of every atom")
    dataset = crystal.datasets[0]
    radius = np.concatenate([np.linspace(0.5 * dataset.r[0], 0.9 * dataset.r[0], 3),
                             np.linspace(1.05 * dataset.r[0], 0.97 * dataset.r[-1], 40),
                             np.linspace(1.05 * dataset.r[-1], 3.0 * dataset.r[-1], 5)])
    rec.inputs["local_potential_radius_bohr"] = enc(radius)
    slope = dataset.local_potential_derivative(radius)
    rec.add("local_potential_derivative", slope, dataset.local_potential_derivative(radius * (1 + 1e-6)),
            "Ha/Bohr, dataset 0 (the slope short_range_potential_derivative is built from); perturbed: r scaled by 1+1e-6")
    rec.add("local_potential", dataset.local_potential(radius), None, "Ha, context for the slope")
    step = 1e-6
    numeric = (dataset.local_potential(radius + step) - dataset.local_potential(radius - step)) / (2 * step)
    rec.check("local_potential_derivative.fd_abs", np.abs(slope - numeric).max(), 2e-6,
              "step 1e-6 Bohr; the gate of test_the_local_potential_derivative_is_the_slope", scale=np.abs(slope).max())


def radial_tables(rec, crystal):
    seen = {}
    for f in crystal.basis:
        seen.setdefault((f.l, id(f._values)), f)
    rec.inputs["basis_radial_tables"] = [
        {"l": f.l, "r": enc(f._r), "values": enc(f._values)} for f in seen.values()]
    rec.inputs["basis_functions"] = [{"l": f.l, "m": f.m, "atom": int(a), "center_bohr": list(map(float, f.center)),
                                      "table": [k for k, g in enumerate(seen.values()) if g._values is f._values][0]}
                                     for f, a in zip(crystal.basis, crystal.atom_of_orbital)]


def replay_inputs(rec, pp, crystal):
    """What a Rust test needs besides the arrays: the datasets, the quadrature rules and the sphere radii."""
    from mandacaru.pseudopotentials import local_split as ls, paw
    rec.inputs["replay"] = {
        "datasets": [f"lda-sr/{d.symbol}.parquet" for d in crystal.datasets],
        "compensation_radius_bohr": [float(d.compensation_radius) for d in crystal.datasets],
        "short_range_radius_bohr": [float(ls.short_range_radius(d, crystal.sigma)) for d in crystal.datasets],
        "short_range_panel_bohr": [float(ls.local_cutoff(d)) for d in crystal.datasets],
        "short_range_rule": [ls.CRYSTAL_RADIAL_POINTS, ls.CRYSTAL_POLAR_POINTS, ls.CRYSTAL_AZIMUTHAL_POINTS],
        "compensation_rule": [paw.COMPENSATION_RADIAL_POINTS, paw.PROJECTION_POLAR_POINTS,
                              paw.PROJECTION_AZIMUTHAL_POINTS],
        "projector_basis": str(getattr(crystal.projectors[0], "projector_basis", paw.DEFAULT_PROJECTOR_BASIS))}


def collect_eight(rec, pp, crystal, rng):
    replay_inputs(rec, pp, crystal)
    radial_tables(rec, crystal)
    basis_gradient(rec, pp, crystal, rng)
    radial_slope(rec, pp, crystal)
    short_range_cross(rec, pp, crystal, rng)
    remainder(rec, pp, crystal, rng)
    onsite(rec, pp, crystal)
    gradients(rec, pp, crystal)
