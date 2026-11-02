"""The dead-block functions of ``periodic_paw.py`` (WAVE3B D2) frozen with the recipe of ``freeze_wpx_functions.py``.

Imported by that script (``collect``); it is not run on its own.  Every function here has no caller outside the file
and its tests.  The D2 inventory (``D2_inventory.md``) classifies each: the ones whose behavior a Rust twin reproduces
are frozen so the twin can be compared, the ones with unported behavior are frozen so that whoever ports them has the
number.  Inputs that the functions take (density matrices, moment weights, shifts, kernels) are drawn from a seeded
generator and stored under ``inputs``; the Bloch states come from the crystal, which the setup freeze pins.
"""
from __future__ import annotations

import numpy as np

OMEGA = 0.11   # Bohr^-1, the HSE06 screening parameter, for the screened kernel of the dense compensation sums
SHIFTS = [[0.13, 0.07, -0.05], [0.0, 0.0, 0.3]]
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]


def screened_kernel(q2):
    """``4 pi (1 - exp(-q^2 / 4 omega^2)) / q^2``, finite at q = 0 (the Rust ``screened_kernel``)."""
    q2 = np.asarray(q2, dtype=float)
    safe = np.where(q2 > 1e-12, q2, 1.0)
    return np.where(q2 > 1e-12, 4 * np.pi * (1 - np.exp(-safe / (4 * OMEGA ** 2))) / safe, np.pi / OMEGA ** 2)


def psd(rng, n, rank):
    a = rng.normal(size=(n, rank)) + 1j * rng.normal(size=(n, rank))
    return a @ a.conj().T / rank


def cat(q, channels):
    return np.array([q[c] for c in channels])


def density_group(rec, enc, crystal, rng, channels):
    nk, M = len(crystal.kpoint_data), crystal.M
    P = np.array([psd(rng, M, 3) for _ in range(nk)])
    rec.inputs["density_matrices"] = enc(P)
    rec.inputs["density_channels"] = [list(map(int, c)) for c in channels]
    rho, q = crystal.density(list(P))
    rho2, q2 = crystal.density(list(1.001 * P))
    rec.add("density.rho", rho, rho2, "(grid) real, symmetrized like the SCF density; perturbed: P scaled by 1.001")
    rec.add("density.q", cat(q, channels), cat(q2, channels), "(channel) complex multipoles, rotated by the operations")
    raw = sum(d.weight * np.real(np.sum(d.psi * (p @ d.psi.conj()))) for d, p in zip(crystal.kpoint_data, P))
    rec.check("density.charge_kept_by_symmetrization", abs(rho.sum() - raw), None,
              f"|sum(rho) - unsymmetrized sum|; sum(rho) = {rho.sum():.6g}")
    tau = crystal.kinetic_energy_density(list(P))
    tau2 = crystal.kinetic_energy_density(list(1.001 * P))
    rec.add("kinetic_energy_density", tau, tau2, "(grid) real tau = 1/2 sum w P grad.grad*, symmetrized")
    kin = sum(d.weight * np.trace(p @ crystal._kinetic(d.psi, d.k)).real for d, p in zip(crystal.kpoint_data, P))
    rec.check("kinetic_energy_density.integral_vs_kinetic_matrix", abs(tau.sum() * crystal.grid.dV - kin) / abs(kin),
              None, f"relative: int tau = sum_k w tr(P T_k) with the kinetic matrix of _kinetic; int tau = {kin:.6g}")
    rec.check("kinetic_energy_density.min", -tau.min(), None, "minus the minimum: tau is nonnegative for a PSD P")
    rec.inputs["tau_minimum"] = float(tau.min())


def moments_group(rec, enc, crystal, rng, channels):
    n_p = len(crystal.projectors)
    w = {c: complex(rng.normal(), rng.normal()) for c in channels}
    R = psd(rng, n_p, 5)
    rec.inputs["moment_weights"] = enc(np.array([w[c] for c in channels]))
    rec.inputs["moment_R"] = enc(R)
    scaled = {c: v * 1.001 for c, v in w.items()}
    rec.add("moment_operator", crystal.moment_operator(w), crystal.moment_operator(scaled),
            "(P, P) complex = sum_c w_c blk_c on each atom's projector block; perturbed: w scaled by 1.001")
    rec.add("moment_traces", cat(crystal.moment_traces(R), channels), cat(crystal.moment_traces(R * 1.001), channels),
            "(channel) complex tr(R_AA blk_c); perturbed: R scaled by 1.001")


def gradients_group(rec, enc, crystal, pp, ks):
    d = crystal.kpoint_data[0]
    g = crystal.bloch_gradients(d.psi, d.k)
    rec.inputs["bloch_gradients_k_bohr_inv"] = enc(d.k)
    rec.add("bloch_gradients", g, crystal.bloch_gradients(d.psi, d.k + 0.01),
            "(axis, mu, grid) complex at the first k-point; perturbed: k shifted by 0.01 Bohr^-1 along (1,1,1)")
    T = crystal._kinetic(d.psi, d.k)
    half = 0.5 * np.einsum("amx,anx->mn", g.conj(), g) * crystal.grid.dV
    rec.check("bloch_gradients.kinetic_matrix", np.abs(half - T).max() / np.abs(T).max(), None,
              "relative: 1/2 <grad chi_m|grad chi_n> = _kinetic[m, n]; the gap is the alias mean of |G+k|^2 against the "
              "square of the mean G+k")
    sums = crystal.bloch_sums(ks)
    rec.inputs["bloch_sums_k_bohr_inv"] = enc(ks)
    rec.add("bloch_sums", sums, crystal.bloch_sums(ks + 0.01), "(1, mu, grid) complex at one generic k; perturbed: k shifted by 0.01")
    rec.check("bloch_sums.psi_difference", np.abs(crystal.bloch_sums(crystal.kpoints[:1])[0] - d.psi).max(), None,
              "the first k-point of the crystal is KPoint.psi: exact (not stored: the setup freeze holds psi)")
    from mandacaru.integrals import reciprocal as rc
    pick = np.random.default_rng(5).choice(crystal.grid.size, 40, replace=False)
    points = np.stack([c.ravel()[pick] for c in (crystal.grid.X, crystal.grid.Y, crystal.grid.Z)], axis=1)
    radius = crystal._cell_region()[1]
    reach = max(pp._support(f) for f in crystal.basis) + 2 * radius
    expected = np.zeros((len(crystal.basis), len(pick)), dtype=complex)
    for R in rc.lattice_translations(crystal.lattice, reach + radius):
        for mu, f in enumerate(crystal.basis):
            expected[mu] += np.exp(1j * (ks[0] @ R)) * f.evaluate(*(points - R).T)
    rec.check("bloch_sums.image_sum", np.abs(sums[0][:, pick] - expected).max() / np.abs(expected).max(), None,
              "relative: the plain sum over lattice images of exp(i k.R) chi(r - R) at 40 grid points (the test of "
              "bloch_values), independent of the C/Rust kernel")
    rec.check("bloch_sums.im_over_max", np.abs(sums.imag).max() / np.abs(sums).max(), None,
              "non-vacuity: a generic k makes the Bloch sums complex")
    proj = crystal.projections(ks)
    rec.add("projections", proj, crystal.projections(ks + 0.01), "(k, mu, p) complex <chi_k|p~>; perturbed as above")
    rec.check("projections.kpoint_difference", np.abs(crystal.projections(crystal.kpoints[:1])[0] - d.projections).max(),
              None,
              "the first k-point is KPoint.projections (the (M, P) block the crystal stores)")


def spinor_group(rec, enc, full, rng):
    M, nk = full.M, len(full.kpoint_data)
    P = np.array([psd(rng, 2 * M, 3) for _ in range(nk)])
    rec.inputs["spinor_density_matrices"] = enc(P)
    m = full.magnetization(list(P))
    rec.add("magnetization", m, full.magnetization(list(1.001 * P)),
            "(3, grid) real (mx, my, mz) of spinor P_k (2M x 2M, alpha first) on the full mesh")
    explicit = np.zeros_like(m)
    for data, p in zip(full.kpoint_data, P):
        blocks = [[p[s * M:(s + 1) * M, t * M:(t + 1) * M] for t in range(2)] for s in range(2)]
        rho = [[np.sum(data.psi * (blocks[s][t] @ data.psi.conj()), axis=0) for t in range(2)] for s in range(2)]
        for a in range(3):
            explicit[a] += data.weight * np.real(sum(PAULI[a][t, s] * rho[s][t] for s in range(2) for t in range(2)))
    rec.check("magnetization.pauli_trace", np.abs(m - explicit).max(), None,
              f"m_a = tr(rho sigma_a) with explicit Pauli matrices; array maximum {np.abs(m).max():.4g}")
    pa, pb = (np.array([psd(rng, M, 2) for _ in range(nk)]) for _ in range(2))
    block = np.array([np.block([[a, np.zeros((M, M))], [np.zeros((M, M)), b]]) for a, b in zip(pa, pb)])
    mc = full.magnetization(list(block))
    up, _ = full.density(list(pa))
    down, _ = full.density(list(pb))
    rec.check("magnetization.collinear_limit", max(np.abs(mc[2] - (up - down)).max(), np.abs(mc[:2]).max()), None,
              "P = diag(Pa, Pb): mz = rho(Pa) - rho(Pb), mx = my = 0")
    grads = [full.bloch_gradients(d.psi, d.k) for d in full.kpoint_data]
    tau, tau_m = full.spinor_kinetic_energy_density(list(P), grads)
    tau2, tau_m2 = full.spinor_kinetic_energy_density(list(1.001 * P), grads)
    rec.add("spinor_kinetic_energy_density.tau", tau, tau2, "(grid) real trace of the spinor tau matrix (UNPORTED)")
    rec.add("spinor_kinetic_energy_density.tau_m", tau_m, tau_m2, "(3, grid) real vector part (UNPORTED)")
    t_up, t_dn = (full.kinetic_energy_density(list(x)) for x in (pa, pb))
    t_c, t_cm = full.spinor_kinetic_energy_density(list(block), grads)
    rec.check("spinor_kinetic_energy_density.collinear_limit",
              max(np.abs(t_c - (t_up + t_dn)).max(), np.abs(t_cm[2] - (t_up - t_dn)).max(), np.abs(t_cm[:2]).max()),
              None, "P = diag(Pa, Pb): tau = tau(Pa) + tau(Pb), tau_z = tau(Pa) - tau(Pb), tau_x = tau_y = 0")


def dense_group(rec, enc, pp, crystal):
    rec.inputs["dense_compensation"] = {"shifts_bohr_inv": SHIFTS, "kernel": "screened, omega = 0.11 Bohr^-1: "
                                        "4 pi (1 - exp(-q2 / 4 omega^2)) / q2, pi / omega^2 at q = 0",
                                        "cutoff_bohr_inv": pp.COMPENSATION_CUTOFF, "channels": [list(c) for c in crystal.channels]}
    for i, shift in enumerate(SHIFTS):
        U = crystal.dense_compensation(shift, screened_kernel)
        rec.add(f"dense_compensation.shift{i}", U, crystal.dense_compensation(np.add(shift, 0.01), screened_kernel),
                "(channel, channel) complex; perturbed: the shift moved by 0.01 Bohr^-1 (UNPORTED)")
        grad = crystal.dense_compensation_gradient(shift, screened_kernel)
        rec.add(f"dense_compensation_gradient.shift{i}", grad,
                crystal.dense_compensation_gradient(np.add(shift, 0.01), screened_kernel),
                "(axis, channel, channel) complex (UNPORTED)")
        rec.check(f"dense_compensation.hermitian.shift{i}", np.abs(U - U.conj().T).max() / np.abs(U).max(), None, "relative")
        if i:
            continue
        atom, step, worst = len(crystal.centers) - 1, 1e-4, 0.0
        owner = np.array([c[0] == atom for c in crystal.channels], dtype=float)
        sign = owner[:, None] - owner[None, :]
        for d in range(3):
            values = []
            for s in (+1, -1):
                delta = np.zeros(3)
                delta[d] = s * step
                saved = crystal.centers[atom].copy()
                crystal.centers[atom] = saved + delta
                try:
                    values.append(crystal.dense_compensation(shift, screened_kernel))
                finally:
                    crystal.centers[atom] = saved
            worst = max(worst, float(np.abs((values[0] - values[1]) / (2 * step) - grad[d] * sign).max()))
        rec.check("dense_compensation_gradient.fd_abs", worst, None,
                  f"per Bohr; dU/dR_A = grad (delta_aA - delta_bA), A = atom {atom}, step {step}", scale=np.abs(grad).max())


def field_group(rec, pp, crystal, moved):
    from mandacaru.integrals.exchange_correlation import xc_core_density
    cores = [crystal.place_core(a, xc_core_density) for a in range(len(crystal.centers))]
    shifted = [crystal.place_core(a, xc_core_density, crystal.centers[a] + [0.02, -0.01, 0.015])
               for a in range(len(crystal.centers))]
    for a, (c, s) in enumerate(zip(cores, shifted)):
        rec.add(f"place_core.atom{a}", c, s, "(grid) real, the atom's xc core placed with its images; perturbed: center "
                "moved by (0.02,-0.01,0.015) Bohr")
    shares = crystal.valence_shares()
    other = (moved(crystal, len(crystal.centers) - 1, [0.02, -0.01, 0.015], crystal.valence_shares)
             if len(crystal.centers) > 1 else None)   # one atom: its share is 1 wherever it is
    rec.add("valence_shares", shares, other, "(atom, grid) real n_A / sum_B n_B, zero where the sum is (UNPORTED)")
    total = shares.sum(axis=0)
    rec.check("valence_shares.sum_to_one", np.abs(total[total > 0] - 1).max(), None, "exact: the shares of a point sum to 1")
    rec.add("n_electrons_neutral", np.float64(crystal.n_electrons_neutral), None, "sum of the valence charges")
    rec.add("local_split_width", np.float64(crystal.local_split_width()), None, "sigma in Bohr (the native split width)")


def mesh_group(rec, enc, pp, spec, atoms, crystal):
    from mandacaru.models.setup import monkhorst_pack_kpts
    operations = pp.grid_symmetry(atoms, crystal.grid, max_L=4)
    cases = [((4, 4, 4), False), ((4, 4, 4), True), ((3, 3, 3), False), ((2, 2, 2), False), ((2, 2, 1), True)]
    for index, (size, gamma) in enumerate(cases):
        _, _, mesh = monkhorst_pack_kpts({"size": size, "gamma": gamma})
        for label, ops in (("time_reversal", None), ("space_group", operations)):
            points, mapping, kept = pp.mesh_wedge(mesh, size, ops)
            tag = f"mesh_wedge.{index}.{label}"
            rotations = (np.zeros((0, 3, 3), dtype=np.int64) if kept is None
                         else np.asarray(kept.info.rotations, dtype=np.int64))
            rec.inputs[tag + ".case"] = {"size": list(size), "gamma": gamma}
            if label == "time_reversal":
                rec.add(f"mesh_wedge.{index}.mesh", mesh, None, "(nk, 3) fractional full mesh in Python's order (T0, 1e-12)")
            rec.add(tag + ".points", points, None, "(nw, 3) fractional wedge points")
            rec.add(tag + ".mapping", np.asarray(mapping, dtype=np.int64), None, "(nk) wedge index of each mesh point")
            rec.add(tag + ".rotations", rotations, None, "(nops, 3, 3) integer k-space matrices kept (replay input)")
            classes = len(points)
            rec.check(tag + ".classes", classes, None, f"{len(mesh)} mesh points fall in {classes} classes")
        rpoints, rweights, _ = pp.reduce_mesh(mesh, operations)
        rec.add(f"reduce_mesh.{index}.weights", rweights, None, "(nw) weights summing to one")
    lone = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    try:
        pp.mesh_wedge(lone, (4, 1, 1))
        rec.check("mesh_wedge.refusal_missing", 1.0, 0.0, "a mesh without its time reverse must raise")
    except ValueError as error:
        rec.inputs["mesh_wedge_refusal"] = {"mesh": lone.tolist(), "size": [4, 1, 1], "error": "ValueError",
                                            "message": str(error)}


def collect(rec, pp, spec, atoms, grid_spacing, crystal, rng, enc, moved, build):
    channels = sorted(crystal.channels)
    ks = crystal.cartesian_kpoints([[0.17, 0.31, -0.23]])
    density_group(rec, enc, crystal, rng, channels)
    moments_group(rec, enc, crystal, rng, channels)
    gradients_group(rec, enc, crystal, pp, ks)
    _, _, full = build(spec, full_mesh=True)
    spinor_group(rec, enc, full, rng)
    dense_group(rec, enc, pp, crystal)
    field_group(rec, pp, crystal, moved)
    mesh_group(rec, enc, pp, spec, atoms, crystal)
