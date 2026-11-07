"""Explicit PAW setup adapter and Python/ASE views of the native crystal engine.

PAW species/integral assembly remains Python in this migration stage. SCF,
occupations, mixing, eigensolves and spectral contractions belong to Rust.
"""

from __future__ import annotations
import json
from time import perf_counter
import numpy as np

from .setup_block import setup_block


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
    unrestricted=False,
    hubbard=None,
):
    """Snapshot the density-independent PAW setup; this performs no SCF steps."""
    if functional not in ("lda", "pbe", "r2scan", "pbe+rvv10", "r2scan+rvv10", "hse06"):
        raise NotImplementedError(f"periodic calculation does not support functional {functional!r}")
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
    native_ctx = getattr(c, "_native_context", None)
    if native_ctx is None:
        from .. import _native
        native_ctx = _native.Context(threads=1)
    positions_list = [c._positions[ch[0]] for ch in channels]
    blocks_list = [c.multipole_blocks[ch] for ch in channels]
    blocks = native_ctx.periodic_multipole_blocks(positions_list, blocks_list, p)
    sym = c.symmetry
    if sym is not None:
        qsym = native_ctx.periodic_channel_symmetry(
            channels, sym.atom_maps, sym.rotations, sym.n_operations
        )
    else:
        qsym = native_ctx.periodic_channel_symmetry(channels)
    initial, moments = c.initial_density()
    core = c.core_density()
    u, uion, eion = c.dense_coulomb()
    hats = c.compensation_grid_array()
    short, onsite = c.short_range_ion_compensation(), c.onsite_ion_compensation()
    magnetization = c.initial_magnetization(magnetic_moments)
    if c.has_spin_orbit:
        magnetization = native_ctx.periodic_spinor_initial_magnetization(magnetization, n)
    kerker = native_ctx.periodic_kerker(c.grid.shape, rc.grid_origin(c.grid), c.grid.step)
    phase = native_ctx.periodic_phase(c.grid.shape, rc.grid_origin(c.grid), c.grid.step)
    k_cart = np.asarray([d.k for d in c.kpoint_data], dtype=float) if c.kpoint_data else np.zeros((0, 3))
    k_frac = native_ctx.periodic_fractional_kpoints(k_cart, c.lattice)
    metadata = dict(
        schema_version=1,
        representation="periodic-paw-snapshot",
        energy_unit="Ha",
        length_unit="bohr",
        grid=dict(shape=list(c.grid.shape),
                  origin=rc.grid_origin(c.grid).tolist(),
                  step=c.grid.step.ravel().tolist()),
        m=c.M,
        p=p,
        channels=len(channels),
        channel_labels=[list(ch) for ch in channels],
        sites=dict(
            centers=[list(map(float, p)) for p in c.centers],
            projector_basis=c.projectors[0].projector_basis,
            datasets=[dict(symbol=d.symbol, file=getattr(d, "source", ""), sha256=getattr(d, "source_sha256", ""))
                      for d in c.datasets]),
        setup=setup_block(c),
        kpoints=[
            point_input(d, k_frac[i], i)
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
        hats=hold("hats", hats, complex),
        u=hold("u", u, complex),
        uion=hold("uion", uion, complex),
        linear=hold(
            "linear", [short[ch] - onsite.get(ch, 0.0) for ch in channels], complex
        ),
        eion=float(eion),
        ion_constant=float(c.ion_constants()),
        constant=float(constant),
        reference=float(native_ctx.periodic_reference(c.sigma, c.charges, c.volume)),
        electrons=float(electrons),
        spins=2 if not c.has_spin_orbit and unrestricted else 1,
        spinor=c.has_spin_orbit,
        hubbard=hubbard,
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
    from .. import hubbard
    from ..structure import initial_moments

    if driver._load_crystal is not None:
        raise ValueError("a cached crystal uses run(); it cannot be recomputed for ASE atoms")
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
    name, options = resolve_basis(driver._periodic_basis)
    family = pseudopotential_family(name)
    if family is None or family.name not in ("paw-lcao", "upaw-lcao"):
        raise NotImplementedError(
            "native periodic DFT currently uses the explicit PAW-LCAO/UPAW-LCAO setup adapter"
        )
    from ..pseudopotentials.environment import with_functional_set

    options = with_functional_set(family.name, options, driver.xc)
    spinor = bool(spin_orbit_symbols(atoms, options, family.name))
    if driver.method == "dft" and any(atoms.get_initial_magnetic_moments()):
        driver._unrestricted = True
    moments = initial_moments(atoms, spinor=spinor, unrestricted=driver._unrestricted)
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
    # The crystal of the previous geometry (a relaxation, a scan on one calculator) is released
    # before the next is built: two crystals and their setup arrays never coexist in memory.
    driver._drop_crystal()
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
        unrestricted=driver._unrestricted,
        hubbard=hubbard.table(driver._hubbard_u, driver.context, crystal, moments),
        provenance=dict(
            mesh_size=list(context["kpts_size"]),
            mesh_gamma=context["kpts_gamma"],
            family=family.name,
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
    driver._crystal_adapter, driver._crystal_metadata = crystal, context
    crystal.kpoint_data.clear()
    crystal._compensation_grid = None
    crystal._compensation_values = None
    driver.periodic_problem = driver.context.periodic_problem(metadata, arrays, consume=True)
    # This adapter owns recipes for new k-points. The SCF mesh and its samples
    # have been transferred to the native owner and need no Python duplicate.
    # The native problem owns its copy. Keep the adapter's lazy recipe for
    # later off-mesh work, but release the full reciprocal-grid cache now.
    del arrays
    driver._geometry_seconds = perf_counter() - start
    driver._orbital_groups = [
        (int(a), int(f.l)) for a, f in zip(crystal.atom_of_orbital, crystal.basis)
    ]
    if spinor:
        driver._orbital_groups *= 2
