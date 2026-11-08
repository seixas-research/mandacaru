"""The periodic PAW-LCAO setup: option checks of a crystal and the request to the Rust setup.

Every crystal is built by the Rust crystal setup (``periodic::setup``); this module validates the
calculator's options, resolves the basis family and its datasets, and hands the request over
(:func:`.setup_block.native_setup`).
"""

from __future__ import annotations
from time import perf_counter
import numpy as np

from .setup_block import native_setup, spin_orbit_symbols


def configure(driver, atoms):
    from ..models.setup import resolve_basis, pseudopotential_family
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
            if isinstance(driver._kpts_spec, dict) else driver._kpts_spec)
        if not periodic and size[axis] != 1:
            raise ValueError("nonperiodic directions require one k-point")
    start = perf_counter()
    # The crystal of the previous geometry (a relaxation, a scan on one calculator) is released
    # before the next is built: two crystals and their setup arrays never coexist in memory.
    driver._drop_crystal()
    native_setup(driver, atoms, family, options, moments, spinor, time_reversal, start)
