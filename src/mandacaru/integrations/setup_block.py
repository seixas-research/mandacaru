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
