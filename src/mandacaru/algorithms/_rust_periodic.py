"""Explicit PAW setup adapter and Python/ASE views of the native crystal engine.

PAW species/integral assembly remains Python in this migration stage. SCF,
occupations, mixing, eigensolves and spectral contractions belong to Rust.
"""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from ..units import HARTREE_TO_EV


def _complex(values):
    a = np.asarray(values, dtype=complex).ravel()
    return np.column_stack((a.real, a.imag)).tolist()


def kpoint_payload(crystal, data, fractional=None):
    if fractional is None:
        fractional = np.asarray(data.k) @ crystal.lattice / (2*np.pi)
    return dict(k=np.asarray(data.k).tolist(), fractional=np.asarray(fractional).tolist(),
                weight=float(data.weight), psi=_complex(data.psi), overlap=_complex(data.overlap),
                fixed=_complex(data.fixed), projections=_complex(data.projections))


def crystal_payload(crystal, electrons, functional, *, constant=0., relativistic=False,
                    magnetic_moments=None, provenance=None):
    """Snapshot the density-independent PAW setup; this performs no SCF steps."""
    if functional not in ("lda", "pbe"):
        raise NotImplementedError("native periodic SCF supports lda and pbe")
    from ..integrals import reciprocal as rc, exchange_correlation as xc
    c = crystal
    if c.has_spin_orbit:
        raise NotImplementedError("native periodic spinor/SOC SCF is not implemented")
    n, p = c.grid.size, len(c.projectors)
    channels = c.channels
    blocks = np.zeros((len(channels), p, p), dtype=complex)
    for a, channel in enumerate(channels):
        own = c._positions[channel[0]]
        blocks[a][np.ix_(own, own)] = c.multipole_blocks[channel]
    sym = c.symmetry
    qsym = np.eye(len(channels), dtype=complex)
    if sym is not None:
        qsym[:] = 0.
        index = {key: i for i, key in enumerate(channels)}
        for atom_map, rotation in zip(sym.atom_maps, sym.rotations):
            for a, (atom, degree, order) in enumerate(channels):
                for other in range(-degree, degree+1):
                    source = index.get((int(atom_map[atom]), degree, other))
                    if source is not None:
                        qsym[a, source] += rotation[degree][order+degree, other+degree] / sym.n_operations
    initial, moments = c.initial_density()
    core = c.core_density()
    u, uion, eion = c.dense_coulomb()
    hats = c.compensation_grid()
    short, onsite = c.short_range_ion_compensation(), c.onsite_ion_compensation()
    magnetization = c.initial_magnetization(magnetic_moments)
    g2 = np.sum(c.G*c.G, axis=0)
    kerker = g2/(g2+1.)
    kerker.flat[0] = 1.
    phase = np.exp(-1j*np.einsum('c,cxyz->xyz', rc.grid_origin(c.grid), c.G))
    return dict(schema_version=1, representation="periodic-paw-snapshot", energy_unit="Ha", length_unit="bohr",
                grid=dict(shape=list(c.grid.shape), origin=rc.grid_origin(c.grid).tolist(), step=c.grid.step.ravel().tolist()),
                m=c.M, p=p, channels=len(channels), channel_labels=[list(ch) for ch in channels], kpoints=[kpoint_payload(c, d) for d in c.kpoint_data],
                blocks=_complex(blocks), core=np.zeros(n).tolist() if core is None else np.asarray(core).ravel().tolist(),
                initial=np.asarray(initial).ravel().tolist(), initial_m=np.asarray(magnetization).ravel().tolist(),
                initial_q=_complex([moments[ch] for ch in channels]), kernel=np.asarray(c.kernel).ravel().tolist(),
                xc_wave=np.stack(xc._wavevectors(c.grid)).ravel().tolist(), phase=_complex(phase), kerker=kerker.ravel().tolist(),
                ion=_complex(c.ion_charge()), hats=_complex(np.array([hats[ch] for ch in channels])),
                u=_complex(u), uion=_complex(uion), linear=_complex([short[ch]-onsite.get(ch, 0.) for ch in channels]),
                eion=float(eion), ion_constant=float(c.ion_constants()), constant=float(constant),
                reference=float(2*np.pi*c.sigma**2*np.sum(c.charges)/c.volume), electrons=float(electrons),
                spins=2 if magnetic_moments is not None and np.any(np.asarray(magnetic_moments) != 0.) else 1,
                functional=functional, relativistic=bool(relativistic),
                permutations=[] if sym is None else [np.asarray(p).ravel().tolist() for p in sym.permutations],
                qsym=_complex(qsym), provenance=dict(provenance or {}, setup_engine="python-paw-adapter"))


def configure(driver, atoms):
    from .. import _native
    from ._hamiltonian_from_atoms import resolve_basis, pseudopotential_family
    from ..pseudopotentials.periodic_paw import build_crystal
    from ..integrals.exchange_correlation import core_correction_offset
    from .dft import _datasets_relativistic, _warn_functional_mismatch
    if driver._load_crystal is not None:
        raise ValueError("a cached crystal uses run(); it cannot be recomputed for ASE atoms")
    if driver._active_spec is not None or driver._save_hamiltonian or driver._periodic_two_body_requested:
        raise ValueError("periodic DFT has no molecular active space, two-body representation or Hamiltonian export; use save_crystal")
    if driver._scf_options:
        raise ValueError("periodic DFT uses periodic_scf_options; molecular scf_options do not apply")
    if driver._kinetic != "spectral":
        raise ValueError("a crystal requires spectral kinetic energy")
    if driver._n_electrons is not None:
        raise ValueError("periodic PAW electron counts follow the datasets and charge")
    if driver._requested_mapping is not None:
        raise ValueError("a periodic Kohn-Sham problem has no qubit mapping")
    moments = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    if moments.ndim != 1:
        raise NotImplementedError("native periodic SCF supports collinear initial moments")
    name, options = resolve_basis(driver._periodic_basis)
    family = pseudopotential_family(name)
    if family is None or family.name not in ("paw-lcao", "upaw-lcao"):
        raise NotImplementedError("native periodic DFT currently uses the explicit PAW-LCAO/UPAW-LCAO setup adapter")
    options = dict(options)
    for axis, periodic in enumerate(atoms.pbc):
        size = driver._kpts_spec.get("size", (1,1,1)) if isinstance(driver._kpts_spec, dict) else driver._kpts_spec
        if not periodic and size[axis] != 1:
            raise ValueError("nonperiodic directions require one k-point")
    start = perf_counter()
    driver.context = _native.Context(**driver.options)
    crystal, context = build_crystal(atoms, driver._grid_spacing, options, kpts=driver._kpts_spec,
                                     family=family.name, grid=driver._explicit_grid,
                                     symmetry=driver._periodic_symmetry, full_mesh=bool(np.any(moments)),
                                     filter_pin=getattr(driver, "filter_pin", None))
    _warn_functional_mismatch(crystal.datasets, driver.xc)
    payload = crystal_payload(crystal, context["n_electrons"]-driver._charge, driver.xc,
                             constant=sum(d.one_center_energy+core_correction_offset(d) for d in crystal.datasets),
                             relativistic=_datasets_relativistic(crystal.datasets), magnetic_moments=moments,
                             provenance=dict(mesh_size=list(context["kpts_size"]), mesh_gamma=context["kpts_gamma"],
                                             family=family.name, dataset_sources=[dict(symbol=d.symbol, sha256=getattr(d, "source_sha256", None),
                                                                                     file=Path(getattr(d, "source", "")).name) for d in crystal.datasets],
                                             geometry=dict(numbers=atoms.numbers.tolist(), positions_angstrom=atoms.positions.tolist(),
                                                           cell_angstrom=atoms.cell.array.tolist(), pbc=atoms.pbc.tolist())))
    # Explicitly retain the adapter for off-mesh PAW matrices only. The Rust
    # solver retains no callback or reference to this Python object.
    if driver._unrestricted:
        payload["spins"] = 2
    driver._crystal_adapter, driver._crystal_metadata = crystal, context
    driver.periodic_problem = driver.context.periodic_problem(json.dumps(payload, allow_nan=False))
    driver._periodic, driver._geometry_seconds = True, perf_counter()-start
    driver._periodic_atoms = atoms.copy()
    driver._orbital_groups = [(int(a), int(f.l)) for a, f in zip(crystal.atom_of_orbital, crystal.basis)]
    if driver._save_crystal:
        driver.periodic_problem.save(str(driver._save_crystal))


def run(driver, geometry=None):
    from .. import _native
    from .dft import PeriodicDFTResult
    from .periodic_dft import PeriodicKohnShamResult
    if driver.dry_run:
        raise NotImplementedError("native periodic dry-run estimates are not implemented")
    if geometry is not None:
        configure(driver, geometry)
    if not hasattr(driver, "periodic_problem"):
        if driver._load_crystal is None:
            raise ValueError("periodic DFT requires an ASE geometry or load_crystal")
        driver.context = _native.Context(**driver.options)
        start = perf_counter()
        driver.periodic_problem = driver.context.load_crystal(str(driver._load_crystal))
        if driver._explicit_xc and driver.xc != driver.periodic_problem.functional:
            raise ValueError("xc disagrees with the cached crystal functional")
        driver.xc = driver.periodic_problem.functional
        driver._periodic, driver._geometry_seconds = True, perf_counter()-start
    start = perf_counter()
    driver.native_periodic_result = driver.periodic_problem.solve(**driver._periodic_options)
    raw = driver.native_periodic_result.snapshot()
    driver.n_spatial_orbitals = driver.periodic_problem.n_orbitals
    driver.n_qubits = 0
    driver.fermion_hamiltonian = None
    n = int(round(sum(np.sum(f) * w for f, w in zip(raw["occupations"].sum(axis=0), raw["weights"])) ))
    driver.num_particles = (n // 2, n - n // 2)
    driver._periodic_snapshot = raw
    s = raw["n_spins"]
    eigenvalues = raw["eigenvalues"][0] if s == 1 else raw["eigenvalues"].transpose(1,0,2)
    occupations = raw["occupations"][0] if s == 1 else raw["occupations"].transpose(1,0,2)
    scf = PeriodicKohnShamResult(functional=driver.xc, free_energy=raw["free_energy"], energy=raw["energy"],
                               extrapolated_energy=raw["extrapolated_energy"], fermi_level=raw["fermi_level"],
                               kpoints=raw["kpoints"], weights=np.asarray(raw["weights"]), eigenvalues=eigenvalues,
                               occupations=occupations, converged=raw["converged"], n_iterations=raw["n_iterations"],
                               smearing=raw["smearing"], terms=raw["terms"], n_spins=s, magnetic_moment=raw["magnetic_moment"],
                               history=[dict(iter=i, energy=e, dE=de, residual=r, moment_residual=q, moment=m)
                                        for i,e,de,r,q,m in raw["history"]])
    driver._scf = scf
    driver.result = PeriodicDFTResult(method="dft", optimal_energy=raw["extrapolated_energy"]*driver._scale,
                                     free_energy=raw["free_energy"]*driver._scale, fermi_level=raw["fermi_level"]*driver._scale,
                                     scf=scf, energy_unit=driver.energy_unit,
                                     timings=dict(engine="rust", setup_engine="python-paw-adapter" if hasattr(driver,"_crystal_adapter") else "cached-crystal",
                                                  geometry_seconds=driver._geometry_seconds, scf_seconds=perf_counter()-start,
                                                  memory_peak_bytes=driver.context.memory_peak_bytes, threads=driver.context.threads))
    report = f"Rust periodic {driver.xc.upper()}: {driver.result.optimal_energy:.12f} {driver.energy_unit}; converged={driver.result.success}\n"
    if driver.verbose:
        print(report, end="")
    if driver.txt is not None:
        Path(driver.txt).write_text(report)
    return driver.result
