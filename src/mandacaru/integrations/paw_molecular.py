"""Dataset/basis-table adapter to the owned Rust molecular PAW engine.

The basis is the Rust recipe the crystal setup uses (``paw::basis_recipe``, bound as
``_native.paw_basis``). Sampling, quadratures, operator assembly, augmentation, factorization and
SCF execute in Rust.
"""

import json

import numpy as np

from .. import _native
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
    driver._basis_report = report = dict(confinement={}, polarization={}, options=options, cutoff=cutoff, species=species)
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
    for symbol in unique:
        dataset = species[symbol]
        if getattr(dataset, "source", None) is not None:
            metadata["species"].append(
                dict(file=str(dataset.source), allow_heavy_dft=(driver.method == "dft"))
            )
            continue
        channels = []
        for l in range(8):
            if l not in dataset.channels:
                continue
            channel = dataset.channels[l]
            entry = dict(
                l=int(l),
                principal_n=int(channel.n),
                radius=float(channel.r_cut),
                occupations=list(map(float, channel.occupations)),
                ae=hold(channel.ae_waves),
                coupling=hold(channel.coupling),
                overlap=hold(channel.overlap_correction),
                vanderbilt=hold(channel.vanderbilt),
                wavevectors=[hold(q) for q in channel.wavevectors],
                wave_coefficients=[hold(c) for c in channel.wave_coefficients],
            )
            entry["raw_projectors" if projector_basis == "raw" else "projectors"] = (
                hold(
                    channel.raw_projectors
                    if projector_basis == "raw"
                    else channel.projectors
                )
            )
            channels.append(entry)
        nlcc = dataset.nlcc or {}
        if nlcc.get("applied") and nlcc.get("source") != "smooth_core" and nlcc.get("r_nlcc") is not None:
            raise NotImplementedError(
                "native PAW reference XC currently reads the dataset's stored smooth-core correction"
            )
        xc_core = (
            dataset.smooth_core_density if nlcc.get("applied") else np.zeros_like(dataset.r)
        )
        metadata["species"].append(
            dict(
                atomic_number=int(dataset.atomic_number),
                spin_orbit=bool(dataset.has_spin_orbit),
                frozen_subshells=[
                    list(map(int, shell)) for shell in dataset.frozen_subshells
                ],
                semicore_subshells=[
                    list(map(int, shell)) for shell in dataset.semicore_subshells
                ],
                r=hold(dataset.r),
                ionic=hold(dataset.v_local),
                core=hold(dataset.core_density),
                valence=hold(dataset.valence_density),
                xc_core=hold(xc_core),
                charge=float(dataset.valence_charge),
                local_radius=float(dataset.r_cut_local),
                compensation_radius=float(dataset.compensation_radius),
                functional=dataset.xc,
                one_center_energy=float(dataset.one_center_energy),
                relativistic_exchange=bool(dataset.relativistic_exchange),
                channels=channels,
            )
        )
    metadata["basis"] = basis_entries(symbols, tables_of, hold)
    return driver.context.paw_molecular_integrals(
        geometry.native(), grid, json.dumps(metadata, allow_nan=False), arrays,
        kinetic=driver._kinetic, cholesky_tolerance=driver._cholesky_tolerance,
        precision=driver._precision, consume=True,
    )
