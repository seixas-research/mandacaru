"""WP-F item 2: the periodic-setup diagnostics, ``reference/periodic/<topic>.json``.

    python test/migration/freeze_periodic_diagnostics.py TOPIC [--output DIR] [--work DIR]

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set; ``TOPIC`` is one of
``TOPICS`` (``freeze_periodic_setup.py diagnostics`` runs them all, one process each).  Every producer is
called as the live code calls it (``build_crystal`` and the ``PeriodicPAW`` methods), every topic carries
independent checks computed here (an image sum with a second harmonic implementation, a quadrature of the
shape transform, brute-force orbits of the k-mesh, characters of the harmonic rotations), the stamp of
``freeze_stamp.stamp`` and a measured floor (the topic is built twice, in two processes).  Arrays are
``{"shape", "re", "im"}`` (complex) or ``{"shape", "data"}`` (real or integer).
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

TOPICS = ("local_split", "orbitals_si_sz", "orbitals_si_dzp", "compensation_transforms", "shape_table", "symmetry",
          "mesh_wedge", "kpoint_matrices", "spin_orbit", "basis_table")
#: Fractional k with irrational components: a k, its -k and a third point (none a multiple of 1/n).
IRRATIONAL = ((0.2071067811865476, 0.3660254037844386, -0.1380711874576983),
              (-0.2071067811865476, -0.3660254037844386, 0.1380711874576983),
              (0.4142135623730951, -0.2679491924311227, 0.1213203435596424))


def enc(a):
    import numpy as np
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return dict(shape=list(a.shape), re=a.real.ravel().tolist(), im=a.imag.ravel().tolist())
    return dict(shape=list(a.shape), data=a.ravel().tolist())


def silicon():
    import numpy as np
    from ase.build import bulk
    atoms = bulk("Si", "diamond", a=5.43)
    return atoms, float(np.linalg.norm(atoms.cell[0])) / 10


def aluminum():
    import numpy as np
    from ase.build import bulk
    atoms = bulk("Al", "fcc", a=4.05)
    return atoms, float(np.linalg.norm(atoms.cell[0])) / 7


def build(atoms, grid_spacing, size="SZ", dataset=None, kpts=None, **kw):
    from mandacaru.pseudopotentials import periodic_paw as pp
    options = {"size": size, "filter_cutoff": 200}
    if dataset:
        options["paw_dataset"] = dataset
    return pp.build_crystal(atoms, grid_spacing, options, kpts=kpts or {"size": (1, 1, 1), "gamma": True}, **kw)


def shells(crystal):
    """Shell index of every basis function, by first appearance of its radial table."""
    from mandacaru.pseudopotentials import periodic_paw as pp
    seen: dict = {}
    return [seen.setdefault(pp._shell_key(f), len(seen)) for f in crystal.basis]


def local_split(_):
    """T, V_sr, C D_ion C^+ and S = psi^+ psi dV + C q C^+ at two off-Gamma k of Si SZ (shifted 2x2x2 mesh)."""
    import numpy as np
    atoms, grid_spacing = silicon()
    crystal, _ctx = build(atoms, grid_spacing, kpts={"size": (2, 2, 2), "gamma": False})
    out = dict(grid_dV=crystal.grid.dV, D_ion=enc(crystal.D_ion), q_overlap=enc(crystal.q_overlap), points=[])
    for d in crystal.kpoint_data:
        T = crystal._kinetic(d.psi, d.k)
        V = crystal._short_range_local(d.k[None])[0]
        N = d.projections @ crystal.D_ion @ d.projections.conj().T
        Sg = (d.psi.conj() @ d.psi.T) * crystal.grid.dV
        Sa = d.projections @ crystal.q_overlap @ d.projections.conj().T
        herm = lambda X: 0.5 * (X + X.conj().T)
        out["points"].append(dict(
            k_cartesian=d.k.tolist(), weight=d.weight, T=enc(T), V_sr=enc(V), C_D_ion_Cdag=enc(N), S_grid=enc(Sg),
            S_aug=enc(Sa), C=enc(d.projections), fixed=enc(d.fixed), overlap=enc(d.overlap),
            check=dict(fixed_residual=float(np.abs(herm(T + V + N) - d.fixed).max()),
                       overlap_residual=float(np.abs(herm(Sg + Sa) - d.overlap).max()),
                       T_hermiticity=float(np.abs(T - T.conj().T).max()), T_min_eigenvalue=float(np.linalg.eigvalsh(herm(T)).min()),
                       V_hermiticity=float(np.abs(V - V.conj().T).max()), S_grid_min_eigenvalue=float(np.linalg.eigvalsh(herm(Sg)).min()),
                       max_im_T=float(np.abs(T.imag).max()))))
        assert out["points"][-1]["check"]["fixed_residual"] < 1e-13 and out["points"][-1]["check"]["overlap_residual"] < 1e-13
        assert out["points"][-1]["check"]["max_im_T"] > 1e-3, "T has no phase at this k: the fixture would not see a k.R sign"
    return out


def orbitals(size):
    """Bloch sums at ~40 off-grid points for irrational k (a k and its -k), against two independent image sums."""
    import numpy as np
    from scipy.special import sph_harm_y
    from mandacaru.integrals import reciprocal as rc
    from mandacaru.pseudopotentials import periodic_paw as pp
    atoms, grid_spacing = silicon()
    crystal, _ctx = build(atoms, grid_spacing, size=size)
    A = crystal.lattice
    center, radius = crystal._cell_region()
    rng = np.random.default_rng(20261101)
    inner = (A @ rng.random((3, 32))).T + rc.grid_origin(crystal.grid)
    near = []
    for atom in (0, 1):
        for r in (0.05, 0.2, 0.45, 0.9):
            u = rng.normal(size=3)
            near.append(crystal.centers[atom] + r * u / np.linalg.norm(u))
    points = np.vstack([inner, near])
    nodes = (crystal.grid.X.ravel(), crystal.grid.Y.ravel(), crystal.grid.Z.ravel())
    nearest = min(np.min(np.linalg.norm(np.stack(nodes, 1) - p, axis=1)) for p in points)
    assert nearest > 1e-3, "a point sits on a grid node"
    kcart = crystal.cartesian_kpoints(np.array(IRRATIONAL))
    # The kernel culls images against the ball (center, radius): every point must lie inside it (a point near the
    # atom at the cell corner can lie outside the cell's bounding ball and silently lose images).
    radius = max(radius, float(np.linalg.norm(points - center, axis=1).max()) + 1e-6)
    values = pp.bloch_values(crystal.basis, A, kcart, tuple(points.T), center, radius)
    reach = max(pp._support(f) for f in crystal.basis) + 2 * radius
    images = rc.lattice_translations(A, reach + radius)
    expected = np.zeros_like(values)
    independent = np.zeros_like(values)
    for R in images:
        phases = np.exp(1j * (kcart @ R))
        for mu, f in enumerate(crystal.basis):
            expected[:, mu, :] += phases[:, None] * f.evaluate(*(points - R).T)[None, :]
            d = points - np.asarray(f.center) - R
            r = np.linalg.norm(d, axis=1)
            inside = (r < pp._support(f)) & (r > 0)
            chi = np.zeros(len(points), dtype=complex)
            theta = np.arccos(np.clip(d[inside, 2] / r[inside], -1, 1))
            chi[inside] = f.radial(r[inside]) * sph_harm_y(f.l, f.m, theta, np.arctan2(d[inside, 1], d[inside, 0]))
            independent[:, mu, :] += phases[:, None] * chi[None, :]
    scale = float(np.abs(expected).max())
    shift = []
    for j in range(3):
        edge = max(radius, float(np.linalg.norm(points + A[:, j] - center, axis=1).max()) + 1e-6)
        moved = pp.bloch_values(crystal.basis, A, kcart, tuple((points + A[:, j]).T), center, edge)
        law = np.exp(1j * (kcart @ A[:, j]))[:, None, None] * values
        shift.append(float(np.abs(moved - law).max() / scale))
    # Y_{l,-m} = (-1)^m conj(Y_{lm}), so chi_{-k,l,-m} = (-1)^m conj(chi_{k,l,m}) within a shell.
    tag = shells(crystal)
    where = {(t, f.l, f.m): i for i, (t, f) in enumerate(zip(tag, crystal.basis))}
    flip = 0.0
    for (t, l, m), mu in where.items():
        flip = max(flip, float(np.abs(values[1, where[t, l, -m]] - (-1) ** m * values[0, mu].conj()).max() / scale))
    wrong_sign = pp.bloch_values(crystal.basis, A, -kcart, tuple(points.T), center, radius)
    unphased = [float(np.abs(pp.bloch_values(crystal.basis, A, kcart, tuple((points + A[:, j]).T), center, max(radius, float(np.linalg.norm(points + A[:, j] - center, axis=1).max()) + 1e-6)) - values).max() / scale) for j in range(3)]
    unconjugated = max(float(np.abs(values[1, where[t, l, -m]] - (-1) ** m * values[0, mu]).max() / scale) for (t, l, m), mu in where.items())
    check = dict(kernel_vs_image_sum=float(np.abs(values - expected).max() / scale),
                 controls=dict(image_sum_with_wrong_k_sign=float(np.abs(wrong_sign - expected).max() / scale),
                               translation_without_phase=unphased, m_flip_without_conjugation=unconjugated),
                 kernel_vs_scipy_harmonics=float(np.abs(values - independent).max() / scale),
                 translation_law=shift, minus_k_m_flip=flip, min_distance_to_grid_node=float(nearest),
                 image_count=len(images), l_values=sorted({f.l for f in crystal.basis}))
    # image_sum keeps every image's full table tail; the kernel cuts each function where its table falls below
    # SUPPORT_TOLERANCE x peak (the documented support parity item), summed over ~1000 images: ~1e-10 for DZP.
    assert check["kernel_vs_image_sum"] < 1e-8 and max(shift) < 1e-12 and flip < 1e-12, check
    assert check["kernel_vs_scipy_harmonics"] < 1e-12, check
    assert check["controls"]["image_sum_with_wrong_k_sign"] > 1e-3 and min(unphased) > 1e-3 and unconjugated > 1e-3, check
    if size == "DZP":
        assert 2 in check["l_values"]
    return dict(size=size, k_fractional=np.array(IRRATIONAL).tolist(), k_cartesian=enc(kcart), points=enc(points),
                n_near_nucleus=len(near), values=enc(values), image_sum=enc(expected), check=check, shell_index=shells(crystal))


def compensation_transforms(_):
    """The compensation shapes at a fixed vector list, against the plane-wave formula with a quadrature of F_L."""
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import spherical_jn, sph_harm_y
    atoms, grid_spacing = silicon()
    crystal, _ctx = build(atoms, grid_spacing)
    B = crystal.cartesian_kpoints(np.eye(3)).T
    rng = np.random.default_rng(7)
    unit = rng.normal(size=(2, 3))
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    G = [np.zeros(3), *B.T, B @ [1, -1, 2], B @ [2, 2, -3], [0, 0, 1.3], [0, 0, -2.1], [1.7, 0, 0], [0, -0.9, 0],
         *rng.normal(size=(5, 3)) * 2.0, unit[0] * 14.9, unit[1] * 15.6, np.array([11., 7., -9.]) * 1.3]
    G = np.array(G).T
    out = crystal.compensation_transforms(G)
    r_g = float(crystal.datasets[0].compensation_radius)
    from mandacaru.pseudopotentials.multipoles import shape_function
    norm, ang, cache = {}, [], {}
    g = np.linalg.norm(G, axis=0)
    theta = np.arccos(np.clip(G[2] / np.where(g > 0, g, 1), -1, 1))
    phi = np.arctan2(G[1], G[0])
    for row, (atom, L, M) in enumerate(crystal.channels):
        if L not in norm:
            integrand = lambda r, L=L: float(shape_function(np.array([r]), r_g, L)[0]) * r ** (L + 2)
            norm[L] = quad(integrand, 0, r_g, epsabs=1e-14, epsrel=1e-13)[0]
            cache[L] = np.array([quad(lambda r, q=q, L=L: float(shape_function(np.array([r]), r_g, L)[0]) * spherical_jn(L, q * r) * r * r,
                                      0, r_g, epsabs=1e-14, epsrel=1e-12, limit=200)[0] for q in g])
        F = cache[L]
        ref = 4 * np.pi * (-1j) ** L * sph_harm_y(L, M, theta, phi) * F * np.exp(-1j * crystal.centers[atom] @ G)
        ang.append(np.abs(out[row] - ref))
    per_vector = np.max(np.array(ang), axis=0)
    check = dict(vs_quadrature_plane_wave=float(per_vector.max()), residual_by_vector=per_vector.tolist(), vector_norms=g.tolist(),
                 normalization_by_L={str(L): v for L, v in norm.items()},
                 L0_at_G0=complex(out[0, 0]).real, sqrt_4pi=float(np.sqrt(4 * np.pi)))
    assert abs(check["L0_at_G0"] - check["sqrt_4pi"]) < 1e-6, check
    # the table is a 1201-point trapezoid of F_L on [0, r_g] interpolated linearly in q (step 0.01): the error grows with |G|
    assert check["vs_quadrature_plane_wave"] < 1e-4, check
    return dict(channels=[list(c) for c in crystal.channels], vectors=enc(G), transforms=enc(out), compensation_radius=r_g,
                centers_bohr=enc(np.array(crystal.centers)), check=check)


def shape_table(_):
    """F_L(q) of the compensation shape: nodes of the 0.01 table (stride 100), off-node interpolations, quadrature."""
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import spherical_jn
    from mandacaru.pseudopotentials import periodic_paw as pp
    from mandacaru.pseudopotentials.multipoles import shape_function
    atoms, grid_spacing = silicon()
    crystal, _ctx = build(atoms, grid_spacing)
    r_g = float(crystal.datasets[0].compensation_radius)
    q_off = np.sort(np.random.default_rng(3).uniform(0.0, 125.0, 24))
    q_quad = np.array([0.3, 1.7, 4.9, 9.3, 15.0, 31.7])
    tables = {}
    for L in range(5):
        q, F = crystal._shape_table(0, L, 0.0)
        interp = crystal._shape_transform(0, L, q_off)
        exact = np.array([quad(lambda r, x=x: float(shape_function(np.array([r]), r_g, L)[0]) * spherical_jn(L, x * r) * r * r,
                               0, r_g, epsabs=1e-14, epsrel=1e-12, limit=200)[0] for x in q_quad])
        near = crystal._shape_transform(0, L, q_quad)
        tables[str(L)] = dict(nodes_q=enc(q[::100]), nodes_F=enc(F[::100]), q_off=enc(q_off), F_off=enc(interp), n_points=len(q),
                              sum_F=float(F.sum()), sum_abs_F=float(np.abs(F).sum()), q_quad=enc(q_quad), F_interpolated_at_q_quad=enc(near),
                              F_quadrature=enc(exact), max_vs_quadrature=float(np.abs(near - exact).max()))
        assert tables[str(L)]["max_vs_quadrature"] < 1e-5, (L, tables[str(L)]["max_vs_quadrature"])
    return dict(compensation_radius=r_g, step=pp.SHAPE_TABLE_STEP, q_max=pp.SHAPE_TABLE_MAX, shape_points=pp.SHAPE_POINTS, tables=tables)


def kpoint_matrices(_):
    """S, fixed, C at three off-mesh irrational k (psi on a stride), with S(k+G) and mesh-consistency identities."""
    import numpy as np
    stride = 37
    out = dict(psi_stride=stride)
    for name, (atoms, grid_spacing) in dict(si=silicon(), al=aluminum()).items():
        crystal, _ctx = build(atoms, grid_spacing, kpts={"size": (2, 2, 2), "gamma": False} if name == "si" else {"size": (2, 2, 2), "gamma": True})
        kcart = crystal.cartesian_kpoints(np.array(IRRATIONAL))
        data = crystal.kpoint_matrices(kcart)
        G = crystal.cartesian_kpoints([[1.0, -1.0, 2.0]])
        shifted = crystal.kpoint_matrices(kcart[:1] + G)[0]
        mesh = crystal.kpoint_matrices(crystal.kpoints)
        identities = dict(S_periodic_in_G=float(np.abs(shifted.overlap - data[0].overlap).max()),
                          fixed_periodic_in_G=float(np.abs(shifted.fixed - data[0].fixed).max()),
                          mesh_points_equal_kpoint_data=max(float(np.abs(a.overlap - b.overlap).max()) for a, b in zip(mesh, crystal.kpoint_data)),
                          max_im_S=[float(np.abs(d.overlap.imag).max()) for d in data])
        assert identities["S_periodic_in_G"] < 1e-11 and min(identities["max_im_S"]) > 1e-3, identities
        out[name] = dict(k_fractional=np.array(IRRATIONAL).tolist(), k_cartesian=enc(kcart), overlap=enc(np.array([d.overlap for d in data])),
                         fixed=enc(np.array([d.fixed for d in data])), projections=enc(np.array([d.projections for d in data])),
                         psi_stride=enc(np.array([d.psi[:, ::stride] for d in data])), psi_max=[float(np.abs(d.psi).max()) for d in data],
                         psi_fro=[float(np.linalg.norm(d.psi)) for d in data], grid_size=int(crystal.grid.size), identities=identities)
    return out


def spin_orbit(_):
    """h^SO(k) of Si lda-fr SZ: the mesh k(2,1,1), an irrational k and its -k, with Hermiticity, periodicity and Kramers."""
    import numpy as np
    from scipy.linalg import eigh
    atoms, grid_spacing = silicon()
    crystal, _ctx = build(atoms, grid_spacing, dataset="lda-fr", kpts={"size": (2, 1, 1), "gamma": True}, spin_orbit=True)
    M = crystal.M
    kcart = crystal.cartesian_kpoints(np.array(IRRATIONAL[:2]))
    data = crystal.kpoint_matrices(np.vstack([crystal.kpoints, kcart]))
    G = crystal.cartesian_kpoints([[1.0, -1.0, 2.0]])
    shifted = crystal.kpoint_matrices(kcart[:1] + G)[0]
    spectra = [eigh(d.spin_orbit, np.kron(np.eye(2), d.overlap), eigvals_only=True) for d in data]
    metric = [np.abs(np.trace(np.linalg.solve(np.kron(np.eye(2), d.overlap), d.spin_orbit))) for d in data]
    check = dict(hermiticity=max(float(np.abs(d.spin_orbit - d.spin_orbit.conj().T).max()) for d in data),
                 periodic_in_G=float(np.abs(shifted.spin_orbit - data[2].spin_orbit).max()),
                 mesh_equals_kpoint_data=max(float(np.abs(a.spin_orbit - b.spin_orbit).max()) for a, b in zip(data[:2], crystal.kpoint_data)),
                 kramers_k_vs_minus_k=float(np.abs(spectra[2] - spectra[3]).max()), gamma_pairs=float(np.abs(spectra[0][0::2] - spectra[0][1::2]).max()),
                 trace_in_overlap_metric=max(metric), spectrum_span=[float(np.ptp(x)) for x in spectra], norm=[float(np.abs(d.spin_orbit).max()) for d in data],
                 alpha_beta_swap_change=[float(np.abs(d.spin_orbit.reshape(2, M, 2, M)[::-1, :, ::-1] - d.spin_orbit.reshape(2, M, 2, M)).max()) for d in data])
    assert check["hermiticity"] < 1e-13 and check["periodic_in_G"] < 1e-11 and check["kramers_k_vs_minus_k"] < 1e-11, check
    assert min(check["norm"]) > 1e-4 and min(check["alpha_beta_swap_change"]) > 1e-5 and min(check["spectrum_span"]) > 1e-4, check
    return dict(layout="(2, M, 2, M), alpha first; the stored array is the flat (2M, 2M)", M=M,
                k_cartesian=enc(np.vstack([crystal.kpoints, kcart])), spin_orbit=enc(np.array([d.spin_orbit for d in data])),
                overlap=enc(np.array([d.overlap for d in data])), spectra_in_overlap_metric=enc(np.array(spectra)), check=check)


def basis_table(_):
    """(atom, l, m, shell, zeta, center) of every basis function, projectors, channels and wrapped positions."""
    import numpy as np
    from mandacaru.pseudopotentials import periodic_paw as pp
    cases = {"si_sz": (*silicon(), "SZ", None), "si_dzp": (*silicon(), "DZP", None), "al_sz": (*aluminum(), "SZ", None),
             "si_sz_lda_fr": (*silicon(), "SZ", "lda-fr")}
    out = {}
    for name, (atoms, grid_spacing, size, dataset) in cases.items():
        crystal, ctx = build(atoms, grid_spacing, size=size, dataset=dataset, spin_orbit=dataset is not None)
        wrapped = atoms.copy()
        wrapped.wrap()
        tag = shells(crystal)
        out[name] = dict(
            orbitals=[dict(atom=int(a), l=int(f.l), m=int(f.m), shell=t, zeta=getattr(f, 'zeta', None), kind=str(getattr(f, 'kind', type(f).__name__)),
                           r_c=getattr(f, 'r_c', None), support=float(pp._support(f)))
                      for a, f, t in zip(crystal.atom_of_orbital, crystal.basis, tag)],
            projectors=[dict(atom=int(p.atom_index), l=int(p.l), m=int(p.m), r_cut=float(p.r_cut), kb_energy=float(p.kb_energy)) for p in crystal.projectors],
            channels=[list(c) for c in crystal.channels], centers_bohr=enc(np.array(crystal.centers)),
            wrapped_positions_angstrom=enc(wrapped.get_positions()), lattice_bohr=enc(crystal.lattice), grid_shape=list(crystal.grid.shape),
            sigma=float(crystal.sigma), volume=float(crystal.volume), filter_wavevector=float(crystal.filter_wavevector),
            charges=crystal.charges.tolist(), n_electrons=float(ctx["n_electrons"]))
    return out


def topic_body(topic):
    if topic in ("symmetry", "mesh_wedge"):
        import freeze_periodic_symmetry
        return getattr(freeze_periodic_symmetry, topic)(None)
    if topic.startswith("orbitals_si_"):
        return orbitals(topic[len("orbitals_si_"):].upper())
    return globals()[topic](None)


DATASETS = dict(spin_orbit=[("Si", "lda-fr")], basis_table=[("Si", "lda-sr"), ("Si", "lda-fr"), ("Al", "lda-sr")],
                kpoint_matrices=[("Si", "lda-sr"), ("Al", "lda-sr")], symmetry=[("Si", "lda-sr"), ("Al", "lda-sr")],
                mesh_wedge=[("Si", "lda-sr"), ("Al", "lda-sr")])


def leaves(node, path=""):
    if isinstance(node, dict):
        if "shape" in node and ("re" in node or "data" in node):
            yield path, node
            return
        for key, value in node.items():
            yield from leaves(value, f"{path}/{key}")
    elif isinstance(node, list):
        if node and all(isinstance(x, (int, float)) for x in node):
            yield path, dict(shape=[len(node)], data=node)
        else:
            for i, value in enumerate(node):
                yield from leaves(value, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, dict(shape=[], data=[node])


def floor_between(a, b):
    """Run-to-run difference per array relative to its maximum.  The residual scalars of the ``check`` and
    ``identities`` blocks sit at roundoff (their own maximum is the noise), so they are compared in absolute terms."""
    import numpy as np
    worst, where, residual = 0.0, "", 0.0
    pa, pb = dict(leaves(a)), dict(leaves(b))
    assert pa.keys() == pb.keys()
    for path, x in pa.items():
        y = pb[path]
        parts = [(x["re"], y["re"]), (x["im"], y["im"])] if "re" in x else [(x["data"], y["data"])]
        scale = max([float(np.abs(u).max()) for u, _ in parts if len(u)] + [1e-300])
        diff = max([float(np.abs(np.asarray(u, float) - np.asarray(v, float)).max()) for u, v in parts if len(u)] + [0.0])
        if "/check" in path or "/identities" in path or "/controls" in path:
            residual = max(residual, diff)
        elif diff / scale > worst:
            worst, where = diff / scale, path
    return dict(max_relative_to_array_maximum=worst, at=where, n_arrays=len(pa), check_scalars_max_abs_difference=residual)


def run_topic(topic):
    from freeze_stamp import stamp
    from mandacaru.pseudopotentials.paw import get_paw
    body = topic_body(topic)
    datasets = [get_paw(sym, paw_dataset=ds).source for sym, ds in DATASETS.get(topic, [("Si", "lda-sr")])]
    return dict(topic=topic, body=body, stamp=stamp(datasets=datasets, extra=dict(topic=topic, generator="test/migration/freeze_periodic_diagnostics.py")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topic", choices=TOPICS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", action="store_true", help="internal: write the repeat build only")
    args = parser.parse_args()
    for name in ("MANDACARU_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    if not os.environ.get("MANDACARU_PAW_PATH"):
        raise SystemExit("set MANDACARU_PAW_PATH to the mandacaru-paw checkout")
    from freeze_stamp import repo_root
    root = repo_root()
    output = args.output or root / "test" / "migration"
    work = args.work or Path(os.environ.get("TMPDIR", "/tmp")) / "freeze_periodic_work"
    work.mkdir(parents=True, exist_ok=True)
    again = work / f"{args.topic}.json"
    if args.repeat:
        again.write_text(json.dumps(topic_body(args.topic), separators=(",", ":"), allow_nan=False))
        return
    payload = run_topic(args.topic)
    payload["stamp"]["extra"]["threads"] = args.threads
    guard = root / "scripts" / "memguard.py"
    floors = {}
    for tag, n in (("repeat_same_threads", args.threads), ("repeat_two_threads", 2 if args.threads != 2 else 1)):
        if guard.exists():
            subprocess.run([sys.executable, str(guard), "--need-gb", "2", "--wait", "120"], cwd=root, check=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), args.topic, "--repeat", "--work", str(work),
                        "--threads", str(n)], cwd=root, check=True)
        floors[tag] = dict(threads=n, **floor_between(json.loads(json.dumps(payload["body"])), json.loads(again.read_text())))
        again.unlink()
    payload["stamp"]["floor"] = floors
    target = output / "reference" / "periodic" / f"{args.topic}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(topic=args.topic, path=str(target), bytes=target.stat().st_size, floor=floors)))


if __name__ == "__main__":
    main()
