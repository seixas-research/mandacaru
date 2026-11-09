"""The request a crystal hands the Rust crystal setup (``periodic::setup``), and what it refuses.

Python only translates the ASE geometry and the calculator's options into the request: lattice
vectors (as columns, in Bohr), grid node counts, atoms, the Monkhorst-Pack mesh the setup reduces
itself, the functional, the basis size and the dataset files. Everything else (basis, projectors,
symmetry, k-point reduction, matrices, PAW spheres) is built in Rust.
"""

from __future__ import annotations


def ion_width(grid):
    """The Gaussian ion width (Bohr) of the local-potential split; ``None`` keeps the Rust default,
    1.4 times the coarsest grid step. A strain-difference oracle holds it at the unstrained cell's
    value, the pin of the analytic stress, by replacing this function."""
    return None


def _filter_wavevector(options):
    """The Fourier-filter cutoff wave-vector in Bohr^-1 for the Rust request (``None``: pi/h)."""
    spec = options.get("filter_cutoff")
    if spec is None or spec is True or spec == "auto":
        return None
    from ..basis.filtering import resolve_filter_cutoff
    return resolve_filter_cutoff(spec)


def spin_orbit_symbols(atoms, options: dict | None = None, family: str = "paw-lcao") -> list:
    """The chemical symbols of ``atoms`` whose datasets (of ``options``' ``paw_dataset``) carry a
    spin-orbit term, sorted; empty on a scalar set. The datasets are cached, one load per element."""
    from ..pseudopotentials.families import resolve_family
    from ..pseudopotentials.paw import get_paw, get_upaw

    options = resolve_family(family).resolved_options(dict(options or {}))
    load = get_upaw if family == "upaw-lcao" else get_paw
    return sorted(s for s in set(atoms.get_chemical_symbols())
                  if getattr(load(s, options.get("paw_dataset")), "has_spin_orbit", False))


def native_setup(driver, atoms, family, options, moments, spinor, time_reversal, start):
    """Set the crystal up with the Rust builder (``Context.periodic_problem_from_request``)."""
    if options.get("diffuse"):
        from ..pseudopotentials.diffuse import refuse_crystal
        refuse_crystal(options.get("diffuse"))
    import json
    from time import perf_counter
    from ..units import ANGSTROM_TO_BOHR
    from ..integrals import Grid
    from ..models.setup import monkhorst_pack_kpts
    from ..pseudopotentials.paw import get_paw, get_upaw

    cell = atoms.cell.array
    if driver._explicit_grid is None:
        c0, c1, c2 = cell[0], cell[1], cell[2]
        center = [0.5 * (c0[i] + c1[i] + c2[i]) for i in range(3)]
        g = Grid(center=center, box_size=0.0,
                 grid_spacing=driver._grid_spacing, units="angstrom", cell=cell, periodic=True)
    else:
        g = driver._explicit_grid

    size, gamma, _ = monkhorst_pack_kpts(driver._kpts_spec)
    symbols = list(atoms.symbols)
    load = get_upaw if family.name == "upaw-lcao" else get_paw
    unique_species = sorted(list(set(symbols)))
    datasets = [load(s, options.get("paw_dataset"), xc=driver.xc) for s in unique_species]
    from ..models.datasets import _warn_functional_mismatch
    _warn_functional_mismatch(datasets, driver.xc)

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
    # The Rust setup holds the lattice vectors as COLUMNS (``periodic::setup::parse``); ASE's cell
    # holds them as rows, so the request carries the transpose. A symmetric cell hides the swap.
    lattice_bohr = [float(x) * ANGSTROM_TO_BOHR for column in zip(*cell) for x in column]
    # Each element's size travels in its recipe; the global name only has to be a valid one.
    named = options.get("size", "SZ")
    basis_size = named.upper() if isinstance(named, str) else "SZ"
    request = {
        'lattice': lattice_bohr,
        'grid_shape': list(g.shape),
        'atoms': atoms_spec,
        # The Rust setup reduces the mesh itself: by the operations that keep it, then time
        # reversal; the whole mesh for a magnetic or a non-time-reversal spinor crystal and for
        # a hybrid.
        'mesh': {'size': list(map(int, size)), 'gamma': bool(gamma)},
        'functional': driver.xc,
        'precision': driver._precision,
        'basis_size': basis_size,
        'spinor': spinor,
        'spin': driver._unrestricted,
        'symmetry': driver._periodic_symmetry and not spinor,
        'charge': float(getattr(driver, "_charge", 0.0)),
        # raw (default) or dual projectors; the option is validated with the family's options
        'projector_basis': options.get("projector_basis", "raw"),
    }
    from .basis_recipe import element_recipe
    cutoff = _filter_wavevector(options)
    recipe_cutoff = None if options.get("filter_cutoff") is False else (
        "auto" if cutoff is None else float(cutoff))
    request["recipes"] = {symbol: element_recipe(family, options, symbol, recipe_cutoff)
                          for symbol in unique_species}
    if cutoff is not None:
        request["filter_cutoff"] = float(cutoff)
    width = ion_width(g)
    if width is not None:
        request["sigma"] = float(width)
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

