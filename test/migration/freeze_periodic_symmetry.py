"""WP-F item 2, symmetry part: the grid operations and harmonic rotations (L <= 4) and the k-mesh wedges.

Topics symmetry and mesh_wedge of freeze_periodic_diagnostics.py (which dispatches here).  The independent
checks are computed here: characters, fresh-direction fits and closure of the harmonic rotations, and brute-force
orbits of each mesh under the kept operations and time reversal (never reduce_mesh itself).
"""
from __future__ import annotations

from freeze_periodic_diagnostics import aluminum, build, enc, silicon


def operations_of(atoms, grid_spacing):
    crystal, _ctx = build(atoms, grid_spacing, kpts={"size": (2, 2, 2), "gamma": True})
    return crystal, crystal.symmetry


def symmetry(_):
    """Operations, atom maps and harmonic rotations (L <= 4) of the Si (10^3) and Al fcc (7^3) grids."""
    import hashlib
    import numpy as np
    from mandacaru.basis._angular import spherical_harmonic
    rng = np.random.default_rng(99)
    u = rng.normal(size=(40, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    out = {}
    for name, (atoms, grid_spacing) in dict(al=aluminum(), si=silicon()).items():
        crystal, sym = operations_of(atoms, grid_spacing)
        lattice = crystal.lattice
        W = np.asarray(sym.info.rotations)
        rot = {L: np.array([r[L] for r in sym.rotations]) for L in range(5)}
        resid = unitary = char = 0.0
        for s, w in enumerate(W):
            R = lattice @ w @ np.linalg.inv(lattice)
            det = np.sign(np.linalg.det(R))
            proper = det * R
            theta = np.arccos(np.clip((np.trace(proper) - 1) / 2, -1, 1))
            for L in range(5):
                T = rot[L][s]
                th = np.arccos(np.clip(u[:, 2], -1, 1)), np.arctan2(u[:, 1], u[:, 0])
                Y = np.stack([spherical_harmonic(L, M, *th, real=False) for M in range(-L, L + 1)], 1)
                v = u @ R.T
                YR = np.stack([spherical_harmonic(L, M, np.arccos(np.clip(v[:, 2], -1, 1)), np.arctan2(v[:, 1], v[:, 0]), real=False)
                               for M in range(-L, L + 1)], 1)
                resid = max(resid, float(np.abs(Y @ T - YR).max()))
                unitary = max(unitary, float(np.abs(T.conj().T @ T - np.eye(2 * L + 1)).max()))
                chi = (2 * L + 1) if theta < 1e-12 else np.sin((L + 0.5) * theta) / np.sin(theta / 2)
                char = max(char, abs(np.trace(T) - (det ** L) * chi))
        key = {tuple(w.ravel().astype(int)): s for s, w in enumerate(W)}
        order = 0.0
        for a in range(len(W)):
            for b in range(len(W)):
                c = key.get(tuple((W[a] @ W[b]).ravel().astype(int)))
                if c is not None:
                    order = max(order, min(float(np.abs(rot[3][c] - rot[3][a] @ rot[3][b]).max()),
                                           float(np.abs(rot[3][c] - rot[3][b] @ rot[3][a]).max())))
        check = dict(fresh_direction_residual=resid, unitarity=unitary, character_residual=char, closure_min_over_orders=order,
                     atom_maps_are_permutations=all(sorted(m.tolist()) == list(range(len(atoms))) for m in sym.atom_maps))
        assert max(resid, unitary, char, order) < 1e-9 and check["atom_maps_are_permutations"], check
        shared = {}
        if name == "si":
            # The harmonic rotation depends on the Cartesian R alone, and Si shares fcc's lattice shape: store the index
            # of the fcc operation with the same integer matrix instead of the same 165 matrices again.
            index = {tuple(w.ravel().astype(int)): i for i, w in enumerate(np.asarray(out["al_ops"]))}
            pick = [index[tuple(w.ravel().astype(int))] for w in W]
            diff = max(float(np.abs(rot[L] - np.asarray(out["al_rot"][L])[pick]).max()) for L in range(5))
            assert diff < 1e-12, diff
            shared = dict(harmonic_rotations_are_fcc_operations=pick, max_difference_to_fcc_rotations=diff)
        else:
            out["al_ops"], out["al_rot"] = W.copy(), {L: rot[L].copy() for L in range(5)}
        out[name] = dict(n_operations=sym.n_operations, n_space_group=sym.n_space_group, integer_rotations=enc(W.astype(int)),
                         translations=enc(np.asarray(sym.info.translations)), atom_maps=enc(np.array(sym.atom_maps)),
                         harmonic_rotations={} if shared else {str(L): enc(rot[L]) for L in range(5)}, **shared, grid_shape=list(crystal.grid.shape),
                         permutations_sha256=hashlib.sha256(np.asarray(sym.permutations, dtype=np.int64).tobytes()).hexdigest(),
                         permutations_shape=list(np.asarray(sym.permutations).shape), check=check)
    out.pop("al_ops")
    out.pop("al_rot")
    return out


def orbit_weights(mesh, rotations, tol=1e-8):
    """Brute-force orbit sizes of a fractional mesh under ``k -> k W`` and ``k -> -k`` (independent of reduce_mesh)."""
    import numpy as np
    n = len(mesh)
    label = -np.ones(n, dtype=int)
    for i in range(n):
        if label[i] >= 0:
            continue
        stack, label[i] = [i], i
        while stack:
            j = stack.pop()
            for w in rotations:
                for sign in (1.0, -1.0):
                    image = sign * (mesh[j] @ w)
                    delta = mesh - image
                    hit = np.nonzero(np.all(np.abs(delta - np.round(delta)) < tol, axis=1))[0]
                    for t in hit:
                        if label[t] < 0:
                            label[t] = i
                            stack.append(t)
    return label


def mesh_wedge(_):
    """Wedges of the fixture meshes against brute-force orbits, and the time-reversal-only reductions."""
    import numpy as np
    from mandacaru.models.setup import monkhorst_pack_kpts
    from mandacaru.pseudopotentials import periodic_paw as pp
    si, hs = silicon()
    al, ha = aluminum()
    cases = dict(si_3x3x3=(si, hs, (3, 3, 3), True), si_shifted_2x2x2=(si, hs, (2, 2, 2), False), si_2x1x1=(si, hs, (2, 1, 1), True),
                 al_2x2x2=(al, ha, (2, 2, 2), True), al_shifted_4x4x4=(al, ha, (4, 4, 4), False))
    out = {}
    for name, (atoms, grid_spacing, size, gamma) in cases.items():
        crystal, ctx = build(atoms, grid_spacing, kpts={"size": size, "gamma": gamma})
        _, _, mesh = monkhorst_pack_kpts({"size": size, "gamma": gamma})
        mesh = np.asarray(mesh, dtype=float)
        reduced, weights = ctx["kpoints_fractional"], crystal.weights
        kept = crystal.symmetry
        rotations = [] if kept is None else [np.asarray(w, dtype=float) for w in kept.info.rotations]
        label = orbit_weights(mesh, rotations or [np.eye(3)])
        sizes = {int(i): int(np.sum(label == i)) for i in set(label)}
        brute = sorted((round(s / len(mesh), 12)) for s in sizes.values())
        got = sorted(round(float(w), 12) for w in weights)
        tr_points, tr_weights = pp.time_reversal_reduce(mesh)
        points, mapping, _ops = pp.mesh_wedge(mesh, size, kept)
        check = dict(weights_sum=float(weights.sum()), n_orbits=len(sizes), weights_match_orbits=brute == got,
                     weight_times_nk_is_orbit_size=bool(np.allclose(np.sort(weights * len(mesh)), np.sort(list(sizes.values())))),
                     n_k_with_k_ne_minus_k=int(sum(1 for k in reduced if np.abs(2 * k - np.round(2 * k)).max() > 1e-9)))
        assert check["weights_match_orbits"] and check["weight_times_nk_is_orbit_size"] and abs(check["weights_sum"] - 1) < 1e-14, check
        out[name] = dict(size=list(size), gamma=gamma, mesh=enc(mesh), reduced=enc(reduced), weights=enc(weights), n_operations_kept=0 if kept is None else kept.n_operations,
                         time_reversal_only=dict(points=enc(tr_points), weights=enc(tr_weights)), mesh_wedge_points=enc(points), mesh_wedge_mapping=enc(mapping), check=check)
    return out
