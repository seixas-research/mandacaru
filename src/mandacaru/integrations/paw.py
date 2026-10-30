"""Explicit PAW setup adapter and Python/ASE views of the native crystal engine.

PAW species/integral assembly remains Python in this migration stage. SCF,
occupations, mixing, eigensolves and spectral contractions belong to Rust.
"""

from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def band_input(crystal, fractional_kpoints):
    """Off-mesh setup matrices use the same typed boundary as a crystal."""
    points = crystal.kpoint_matrices(crystal.cartesian_kpoints(fractional_kpoints))
    arrays = {}
    metadata = []
    for index, (point, fractional) in enumerate(zip(points, fractional_kpoints)):
        row = dict(
            k=np.asarray(point.k).tolist(),
            fractional=np.asarray(fractional).tolist(),
            weight=float(point.weight),
        )
        for field in ("psi", "overlap", "fixed", "projections"):
            key = f"k{index}.{field}"
            arrays[key] = np.ascontiguousarray(
                getattr(point, field), dtype=point.psi.dtype if field == "psi" else complex
            ).reshape(-1)
            row[field] = key
        if point.spin_orbit is not None:
            key = f"k{index}.spin_orbit"
            arrays[key] = np.ascontiguousarray(point.spin_orbit, dtype=complex).reshape(-1)
            row["spin_orbit"] = key
        metadata.append(row)
    return json.dumps(metadata, allow_nan=False), arrays


def crystal_input(
    crystal,
    electrons,
    functional,
    *,
    constant=0.0,
    relativistic=False,
    magnetic_moments=None,
    provenance=None,
    precision="float64",
    time_reversal=False,
):
    """Snapshot the density-independent PAW setup; this performs no SCF steps."""
    if functional not in ("lda", "pbe", "r2scan", "pbe+rvv10", "r2scan+rvv10"):
        raise NotImplementedError("periodic screened exact exchange requires its reciprocal exchange operator")
    from ..integrals import reciprocal as rc, exchange_correlation as xc

    c = crystal
    arrays = {}

    def hold(name, value, dtype=float):
        arrays[name] = np.ascontiguousarray(value, dtype=dtype).reshape(-1)
        return name

    def point_input(d, frac, index):
        prefix = f"k{index}."
        row = dict(
            k=np.asarray(d.k).tolist(),
            fractional=np.asarray(frac).tolist(),
            weight=float(d.weight),
            psi=hold(prefix + "psi", d.psi, d.psi.dtype),
            overlap=hold(prefix + "overlap", d.overlap, complex),
            fixed=hold(prefix + "fixed", d.fixed, complex),
            projections=hold(prefix + "projections", d.projections, complex),
        )
        if d.spin_orbit is not None:
            row["spin_orbit"] = hold(prefix + "spin_orbit", d.spin_orbit, complex)
        return row
    n, p = c.grid.size, len(c.projectors)
    channels = c.channels
    blocks = np.zeros((len(channels), p, p), dtype=complex)
    for a, channel in enumerate(channels):
        own = c._positions[channel[0]]
        blocks[a][np.ix_(own, own)] = c.multipole_blocks[channel]
    sym = c.symmetry
    qsym = np.eye(len(channels), dtype=complex)
    if sym is not None:
        qsym[:] = 0.0
        index = {key: i for i, key in enumerate(channels)}
        for atom_map, rotation in zip(sym.atom_maps, sym.rotations):
            for a, (atom, degree, order) in enumerate(channels):
                for other in range(-degree, degree + 1):
                    source = index.get((int(atom_map[atom]), degree, other))
                    if source is not None:
                        qsym[a, source] += (
                            rotation[degree][order + degree, other + degree]
                            / sym.n_operations
                        )
    initial, moments = c.initial_density()
    core = c.core_density()
    u, uion, eion = c.dense_coulomb()
    hats = c.compensation_grid()
    short, onsite = c.short_range_ion_compensation(), c.onsite_ion_compensation()
    magnetization = c.initial_magnetization(magnetic_moments)
    if c.has_spin_orbit:
        # The public ASE scalar initial moments define the z direction.
        magnetization = np.stack([np.zeros(n), np.zeros(n), magnetization])
    g2 = np.sum(c.G * c.G, axis=0)
    kerker = g2 / (g2 + 1.0)
    kerker.flat[0] = 1.0
    phase = np.exp(-1j * np.einsum("c,cxyz->xyz", rc.grid_origin(c.grid), c.G))
    metadata = dict(
        schema_version=1,
        representation="periodic-paw-snapshot",
        energy_unit="Ha",
        length_unit="bohr",
        grid=dict(
            shape=list(c.grid.shape),
            origin=rc.grid_origin(c.grid).tolist(),
            step=c.grid.step.ravel().tolist(),
        ),
        m=c.M,
        p=p,
        channels=len(channels),
        channel_labels=[list(ch) for ch in channels],
        kpoints=[
            point_input(d, np.asarray(d.k) @ c.lattice / (2 * np.pi), i)
            for i, d in enumerate(c.kpoint_data)
        ],
        blocks=hold("blocks", blocks, complex),
        core=hold("core", np.zeros(n) if core is None else core),
        initial=hold("initial", initial),
        initial_m=hold("initial_m", magnetization),
        initial_q=hold("initial_q", [moments[ch] for ch in channels], complex),
        kernel=hold("kernel", c.kernel),
        xc_wave=hold("xc_wave", np.stack(xc._wavevectors(c.grid))),
        phase=hold("phase", phase, complex),
        kerker=hold("kerker", kerker),
        ion=hold("ion", c.ion_charge(), complex),
        hats=hold("hats", np.array([hats[ch] for ch in channels]), complex),
        u=hold("u", u, complex),
        uion=hold("uion", uion, complex),
        linear=hold(
            "linear", [short[ch] - onsite.get(ch, 0.0) for ch in channels], complex
        ),
        eion=float(eion),
        ion_constant=float(c.ion_constants()),
        constant=float(constant),
        reference=float(2 * np.pi * c.sigma**2 * np.sum(c.charges) / c.volume),
        electrons=float(electrons),
        spins=2
        if not c.has_spin_orbit and magnetic_moments is not None and np.any(np.asarray(magnetic_moments) != 0.0)
        else 1,
        spinor=c.has_spin_orbit,
        time_reversal=bool(time_reversal),
        functional=functional,
        precision=precision,
        relativistic=bool(relativistic),
        permutations=hold(
            "permutations", [] if sym is None else sym.permutations, np.uint64
        ),
        permutation_count=0 if sym is None else len(sym.permutations),
        qsym=hold("qsym", qsym, complex),
        provenance=dict(provenance or {}, setup_engine="python-paw-adapter"),
    )

    if functional.startswith("r2scan"):
        core_tau = c.core_tau()
        metadata["core_tau"] = hold("core_tau", np.zeros(n) if core_tau is None else core_tau)
    return json.dumps(metadata, allow_nan=False), arrays


def configure(driver, atoms):
    from ..models.setup import resolve_basis, pseudopotential_family
    from ..pseudopotentials.periodic_paw import build_crystal, spin_orbit_symbols
    from ..integrals.exchange_correlation import core_correction_offset
    from ..models.datasets import _datasets_relativistic, _warn_functional_mismatch

    if driver._load_crystal is not None:
        raise ValueError(
            "a cached crystal uses run(); it cannot be recomputed for ASE atoms"
        )
    if (
        driver._active_spec is not None
        or driver._save_hamiltonian
        or driver._periodic_two_body_requested
    ):
        raise ValueError(
            "periodic DFT has no molecular active space, two-body representation or Hamiltonian export; use save_crystal"
        )
    if driver._scf_options:
        raise ValueError(
            "periodic DFT uses periodic_scf_options; molecular scf_options do not apply"
        )
    if driver._kinetic != "spectral":
        raise ValueError("a crystal requires spectral kinetic energy")
    if driver._n_electrons is not None:
        raise ValueError("periodic PAW electron counts follow the datasets and charge")
    if driver._requested_mapping is not None:
        raise ValueError("a periodic Kohn-Sham problem has no qubit mapping")
    moments = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    if moments.ndim != 1:
        raise NotImplementedError(
            "native periodic SCF supports collinear initial moments"
        )
    name, options = resolve_basis(driver._periodic_basis)
    family = pseudopotential_family(name)
    if family is None or family.name not in ("paw-lcao", "upaw-lcao"):
        raise NotImplementedError(
            "native periodic DFT currently uses the explicit PAW-LCAO/UPAW-LCAO setup adapter"
        )
    options = dict(options)
    spinor = bool(spin_orbit_symbols(atoms, options, family.name))
    time_reversal = spinor and not driver._unrestricted and not np.any(moments)
    for axis, periodic in enumerate(atoms.pbc):
        size = (
            driver._kpts_spec.get("size", (1, 1, 1))
            if isinstance(driver._kpts_spec, dict)
            else driver._kpts_spec
        )
        if not periodic and size[axis] != 1:
            raise ValueError("nonperiodic directions require one k-point")
    start = perf_counter()
    crystal, context = build_crystal(
        atoms,
        driver._grid_spacing,
        options,
        kpts=driver._kpts_spec,
        family=family.name,
        grid=driver._explicit_grid,
        symmetry=driver._periodic_symmetry and not spinor,
        full_mesh=bool(np.any(moments)) or (spinor and not time_reversal),
        spin_orbit=spinor,
        native_context=driver.context,
        precision=driver._precision,
        filter_pin=getattr(driver, "filter_pin", None),
    )
    crystal._native_context = driver.context
    _warn_functional_mismatch(crystal.datasets, driver.xc)
    metadata, arrays = crystal_input(
        crystal,
        context["n_electrons"] - driver._charge,
        driver.xc,
        constant=sum(
            d.one_center_energy + core_correction_offset(d) for d in crystal.datasets
        ),
        relativistic=_datasets_relativistic(crystal.datasets),
        magnetic_moments=moments,
        precision=driver._precision,
        time_reversal=time_reversal,
        provenance=dict(
            mesh_size=list(context["kpts_size"]),
            mesh_gamma=context["kpts_gamma"],
            family=family.name,
            dataset_sources=[
                dict(
                    symbol=d.symbol,
                    sha256=getattr(d, "source_sha256", None),
                    file=Path(getattr(d, "source", "")).name,
                )
                for d in crystal.datasets
            ],
            geometry=dict(
                numbers=atoms.numbers.tolist(),
                positions_angstrom=atoms.positions.tolist(),
                cell_angstrom=atoms.cell.array.tolist(),
                pbc=atoms.pbc.tolist(),
            ),
        ),
    )
    # Explicitly retain the adapter for off-mesh PAW matrices only. The Rust
    # solver retains no callback or reference to this Python object.
    if driver._unrestricted and not spinor:
        options = json.loads(metadata)
        options["spins"] = 2
        metadata = json.dumps(options)
    driver._crystal_adapter, driver._crystal_metadata = crystal, context
    driver.periodic_problem = driver.context.periodic_problem(metadata, arrays)
    # This adapter owns recipes for new k-points. The SCF mesh and its samples
    # have been transferred to the native owner and need no Python duplicate.
    crystal.kpoint_data.clear()
    driver._geometry_seconds = perf_counter() - start
    driver._periodic_atoms = atoms.copy()
    driver._orbital_groups = [
        (int(a), int(f.l)) for a, f in zip(crystal.atom_of_orbital, crystal.basis)
    ]
    if spinor:
        driver._orbital_groups *= 2
    if driver._save_crystal:
        driver.periodic_problem.save(str(driver._save_crystal))
