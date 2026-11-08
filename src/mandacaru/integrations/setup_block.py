"""The ``setup`` block of a crystal snapshot: what the Rust derivative engines rebuild the geometry from.

Hand-over only. The arrays of the Python setup (the radial tables of the basis, the function and
projector lists, the ion width and the space-group operations a reduced k mesh was built with) are
copied into plain lists; the Rust side parses and checks them (``periodic::description``), and a
force or a stress is computed there. The dataset files, their digests, the centers and the
projector basis are the ``sites`` block's.
"""

from __future__ import annotations


def _reals(values):
    return [float(v) for v in values]


def setup_block(crystal):
    """The ``setup`` block of ``crystal`` (a ``PeriodicPAW``); ``operations`` is ``None`` on a whole mesh."""
    from ..pseudopotentials.periodic_paw import _radial_table

    tables, index = [], {}
    functions = {"atom": [], "m": [], "table": []}
    for function, atom in zip(crystal.basis, crystal.atom_of_orbital):
        table = _radial_table(function)
        if table is None:
            # Not tabulated (an analytic function): the energy needs no description of it, the
            # derivative engines do; the Rust side raises with this cause when asked for a force.
            return dict(version=1, unavailable=f"the basis function {type(function).__name__} has no radial table")
        r, values = table
        key = (int(function.l), id(values))
        if key not in index:
            index[key] = len(tables)
            radius = getattr(function, "r_c", None)
            finite = radius is not None and float(radius) < float("inf")
            tables.append(
                dict(
                    l=int(function.l),
                    r=_reals(r),
                    values=_reals(values),
                    confinement=float(radius) if finite else None,
                )
            )
        functions["atom"].append(int(atom))
        functions["m"].append(int(function.m))
        functions["table"].append(index[key])
    symmetry = crystal.symmetry
    operations = None
    if symmetry is not None:
        operations = dict(
            rotations=[[int(w) for row in matrix for w in row] for matrix in symmetry.info.rotations],
            translations=[_reals(t) for t in symmetry.info.translations],
            atom_maps=[[int(a) for a in atom_map] for atom_map in symmetry.atom_maps],
        )
    return dict(
        version=1,
        sigma_bohr=float(crystal.sigma),
        tables=tables,
        functions=functions,
        projectors=[[int(p.atom_index), int(p.l), int(p.m), int(p.index)] for p in crystal.projectors],
        operations=operations,
    )


def try_native_setup(driver, atoms, family, options, moments, spinor, time_reversal, start):
    """Attempt pure native Rust periodic setup, bypassing Python quadrature meshes."""
    import sys
    paw_mod = sys.modules.get("mandacaru.integrations.paw")
    if paw_mod is not None and getattr(paw_mod, "crystal_input", None) is not getattr(paw_mod, "_ORIGINAL_CRYSTAL_INPUT", None):
        return False
    if options.get("diffuse"):
        from ..pseudopotentials.diffuse import refuse_crystal
        refuse_crystal(options.get("diffuse"))
    can_use_native = (
        family.name in ("paw-lcao", "upaw-lcao")
        and options.get("filter_cutoff") is None
        and not getattr(driver, "_active_spec", None)
        and not getattr(driver, "_save_crystal", None)
    )
    if not can_use_native:
        return False
    try:
        import json
        from time import perf_counter
        from ..units import ANGSTROM_TO_BOHR
        from ..integrals import Grid
        from ..models.setup import monkhorst_pack_kpts
        from ..pseudopotentials.paw import get_paw, get_upaw
        from ..pseudopotentials.periodic_paw import grid_symmetry, reduce_mesh

        cell = atoms.cell.array
        if driver._explicit_grid is None:
            c0, c1, c2 = cell[0], cell[1], cell[2]
            center = [0.5 * (c0[i] + c1[i] + c2[i]) for i in range(3)]
            g = Grid(center=center, box_size=0.0,
                     grid_spacing=driver._grid_spacing, units="angstrom", cell=cell, periodic=True)
        else:
            g = driver._explicit_grid

        size, gamma, mesh = monkhorst_pack_kpts(driver._kpts_spec)
        symbols = list(atoms.symbols)
        load = get_upaw if family.name == "upaw-lcao" else get_paw
        unique_species = sorted(list(set(symbols)))
        datasets = [load(s, options.get("paw_dataset"), xc=driver.xc) for s in unique_species]
        from ..models.datasets import _warn_functional_mismatch
        _warn_functional_mismatch(datasets, driver.xc)

        has_moments = any(float(m) > 1e-12 or float(m) < -1e-12 for m in moments) if hasattr(moments, "__iter__") else False
        if has_moments or (spinor and not time_reversal):
            reduced = mesh
            weights = [1.0 / len(reduced)] * len(reduced)
        else:
            all_ch = [l for d in datasets for l in d.channels]
            max_L = max(4, 2 * max(all_ch)) if all_ch else 4
            operations = grid_symmetry(atoms, g, max_L=max_L) if driver._periodic_symmetry and not spinor else None
            reduced, weights, operations = reduce_mesh(mesh, operations)

        kpts_spec = [{'fractional': [float(x) for x in k], 'weight': float(w)} for k, w in zip(reduced, weights)]
        atoms_spec = []
        scaled = atoms.get_scaled_positions()
        for i, symbol in enumerate(symbols):
            species_idx = unique_species.index(symbol)
            m = float(moments[i]) if hasattr(moments, "__getitem__") else 0.0
            atoms_spec.append({
                'species': species_idx,
                'symbol': symbol,
                'fractional': [float(x) for x in scaled[i]],
                'moment': m,
            })

        dataset_paths = [ds.source for ds in datasets]
        lattice_bohr = [float(x) * ANGSTROM_TO_BOHR for row in cell for x in row]
        basis_size = options.get("size", "SZ").upper()
        request = {
            'lattice': lattice_bohr,
            'grid_shape': list(g.shape),
            'atoms': atoms_spec,
            'kpoints': kpts_spec,
            'functional': driver.xc,
            'precision': driver._precision,
            'basis_size': basis_size,
            'spinor': spinor,
            'spin': driver._unrestricted,
            'symmetry': driver._periodic_symmetry and not spinor,
            'charge': float(getattr(driver, "_charge", 0.0)),
        }
        if getattr(driver, "_hubbard_u", None) is not None:
            request["hubbard_u"] = (
                json.loads(driver._hubbard_u)
                if isinstance(driver._hubbard_u, str)
                else driver._hubbard_u
            )

        driver.periodic_problem = driver.context.periodic_problem_from_request(
            json.dumps(request), dataset_paths
        )
        driver._orbital_groups = driver.periodic_problem.orbital_groups
        driver._geometry_seconds = perf_counter() - start

        def _make_adapter():
            if not hasattr(driver, "_crystal_adapter"):
                driver._crystal_adapter = driver.periodic_problem
                driver._crystal_metadata = {
                    "n_electrons": getattr(driver.periodic_problem, "n_electrons", 0),
                    "kpts_size": list(size),
                    "kpts_gamma": gamma,
                }
            return driver._crystal_adapter

        driver._ensure_crystal_adapter = _make_adapter
        return True
    except Exception:
        return False

