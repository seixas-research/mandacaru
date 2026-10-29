"""Dataset/basis-table adapter to the owned Rust molecular PAW engine.

The existing PAW basis recipe supplies radial tables. Sampling, quadratures,
operator assembly, augmentation, factorization and SCF execute in Rust.
"""

import json
import numpy as np

from .. import _native


def build(driver, geometry):
    from ase.data import chemical_symbols
    from ..basis.filtering import resolve_filter_cutoff
    from ..integrals.grid import Grid
    from ..pseudopotentials.families import lookup_family, pseudo_basis_arguments
    from ..pseudopotentials.orbitals import pseudo_basis

    spec = {"name": driver.basis} if isinstance(driver.basis, str) else driver.basis
    family = lookup_family(spec["name"])
    options = family.resolved_options({k: v for k, v in spec.items() if k != "name"})
    if driver.method == "dft" and "paw_dataset" not in spec:
        from ..pseudopotentials.environment import default_library_folder

        options["paw_dataset"] = default_library_folder(driver.xc)
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
    grid = driver._grid(geometry)
    if grid is None:
        view = Grid(
            center=np.mean(geometry.positions, axis=0),
            cell=geometry.cell,
            grid_spacing=driver._grid_spacing
            if geometry.units == "angstrom"
            else driver._grid_spacing / _native.BOHR_TO_ANGSTROM,
            units=geometry.units,
        )
        grid = _native.SpatialGrid(
            view.shape,
            [float(view.X.flat[0]), float(view.Y.flat[0]), float(view.Z.flat[0])],
            np.ascontiguousarray(view.step),
        )
    cutoff = resolve_filter_cutoff(
        options.get("filter_cutoff"), max(np.linalg.norm(grid.step_bohr, axis=0))
    )
    confinement = {}
    functions, owners = pseudo_basis(
        symbols,
        geometry.positions,
        species,
        units=geometry.units,
        filter_wavevector=cutoff,
        **pseudo_basis_arguments(family, options, confinement=confinement),
    )
    arrays = {}

    def hold(value):
        key = f"a{len(arrays)}"
        arrays[key] = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
        return key

    metadata = dict(
        format="molecular-paw-input",
        version=1,
        length_unit="bohr",
        energy_unit="Ha",
        projector_basis=projector_basis,
        atoms=[unique.index(s) for s in symbols],
        species=[],
        basis=[],
    )
    for symbol in unique:
        dataset = species[symbol]
        if dataset.has_spin_orbit:
            raise NotImplementedError(
                "native molecular PAW spin-orbit projector terms are not yet implemented"
            )
        channels = []
        for l, channel in sorted(dataset.channels.items()):
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
        if (
            nlcc.get("applied")
            and nlcc.get("source") != "smooth_core"
            and nlcc.get("r_nlcc") is not None
        ):
            raise NotImplementedError(
                "native PAW reference XC currently reads the dataset's stored smooth-core correction"
            )
        xc_core = (
            dataset.smooth_core_density
            if nlcc.get("applied")
            else np.zeros_like(dataset.r)
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
    tables = {}
    for function, owner in zip(functions, owners):
        if hasattr(function, "table"):
            r, values = function.table.r, function.table.values
        else:
            r, values = function._r, function._values
        key = (id(r), id(values))
        if key not in tables:
            tables[key] = hold(r), hold(values)
        r_key, values_key = tables[key]
        metadata["basis"].append(
            dict(
                atom=int(owner),
                l=int(function.l),
                m=int(function.m),
                r=r_key,
                values=values_key,
            )
        )
    return driver.context.paw_molecular_integrals(
        geometry.native(),
        grid,
        json.dumps(metadata, allow_nan=False),
        arrays,
        kinetic=driver._kinetic,
        cholesky_tolerance=driver._cholesky_tolerance,
    )
