"""Dataset/basis-table adapter to the owned Rust molecular PAW engine.

The basis is the Rust recipe the crystal setup uses (``paw::basis_recipe``, bound as
``_native.paw_basis``). Sampling, quadratures, operator assembly, augmentation, factorization and
SCF execute in Rust.
"""

import json

import numpy as np

from .. import _native
from ..pseudopotentials.record import record_of
from .basis_recipe import basis_entries, basis_tables


def build(driver, geometry):
    from ase.data import chemical_symbols
    from ..basis.filtering import resolve_filter_cutoff
    from ..pseudopotentials.families import lookup_family

    spec = {"name": driver.basis} if isinstance(driver.basis, str) else driver.basis
    family = lookup_family(spec["name"])
    options = family.resolved_options({k: v for k, v in spec.items() if k != "name"})
    if driver.method == "dft":
        from ..pseudopotentials.environment import with_functional_set

        options = with_functional_set(family.name, options, driver.xc)
    unknown = options.keys() - set(family.options)
    if unknown:
        raise TypeError(f"unsupported {family.label} basis options: {sorted(unknown)}")
    projector_basis = options.get("projector_basis", "raw")
    if projector_basis not in ("raw", "dual"):
        raise ValueError("projector_basis must be raw or dual")
    symbols = [chemical_symbols[z] for z in geometry.numbers]
    unique = list(dict.fromkeys(symbols))
    species = {
        symbol: family.get(symbol, options.get("paw_dataset")) for symbol in unique
    }
    if driver.method == "dft":
        from ..models.datasets import _warn_functional_mismatch

        _warn_functional_mismatch(species.values(), driver.xc)
    grid = driver._grid(geometry)
    if grid is None:
        # The box every molecular route uses (the all-electron one too): a zero or
        # degenerate cell raises "molecular integration requires a nondegenerate cell or
        # an explicit grid" instead of becoming a 2 x 2 x 2 box.
        # grid_spacing is in Angstrom whatever the geometry's units.
        spacing = _native.length_to_bohr(float(driver._grid_spacing), "angstrom")
        grid = _native.SpatialGrid.from_geometry(geometry.native(), [spacing, spacing, spacing])
    cutoff = resolve_filter_cutoff(
        options.get("filter_cutoff"), _native.grid_step_max_norm(grid.step_bohr)
    )
    driver._basis_report = report = dict(confinement={}, polarization={}, diffuse={}, options=options, cutoff=cutoff, species=species)
    tables_of = basis_tables(driver.context, family, options, species, cutoff, report)
    arrays = {}

    def hold(value):
        key = f"a{len(arrays)}"
        arrays[key] = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
        return key

    metadata = dict(
        format="molecular-paw-input",
        version=1,
        length_unit="bohr", energy_unit="Ha",
        projector_basis=projector_basis,
        atoms=[unique.index(s) for s in symbols],
        species=[],
        basis=[],
        allow_heavy_dft=(driver.method == "dft"),
        spin_orbit=any(species[symbol].has_spin_orbit for symbol in unique),
        **({"hubbard": driver._hubbard_u} if getattr(driver, "_hubbard_u", None) is not None else {}),
    )
    # A spin-orbit dataset keeps its spin-orbit term for spinor Kohn-Sham only.
    if metadata["spin_orbit"] and driver.method != "dft":
        raise NotImplementedError("a molecule with a spin-orbit dataset needs method='dft' (spinor Kohn-Sham)")
    for index, symbol in enumerate(unique):
        payload, record = record_of(species[symbol])
        metadata["species"].append(
            dict(record=payload, tables=f"s{index}.", allow_heavy_dft=(driver.method == "dft")))
        for name, column in record.items():
            arrays[f"s{index}.{name}"] = np.ascontiguousarray(column, dtype=np.float64)
    metadata["basis"] = basis_entries(symbols, tables_of, hold)
    integrals = driver.context.paw_molecular_integrals(
        geometry.native(), grid, json.dumps(metadata, allow_nan=False), arrays,
        kinetic=driver._kinetic, cholesky_tolerance=driver._cholesky_tolerance,
        precision=driver._precision, consume=True,
    )
    cuts = [[s.r_cut for s in report["diffuse"].get(symbol, ()) if s.accepted] for symbol in symbols]
    box = _native.paw_diffuse_box(integrals, cuts)
    if box is not None:
        from ..pseudopotentials.diffuse import warn_if_cut

        report["diffuse_box"] = box
        warn_if_cut(box)
    return integrals
