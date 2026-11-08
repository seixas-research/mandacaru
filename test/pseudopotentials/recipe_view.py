"""Test helper: what the Rust basis recipe (``paw::basis_recipe``, ``_native.paw_basis``) resolved.

The Gaussian polarization shell is built and sized in Rust; this reads its report (``l``,
``base_l``, ``r_cut``, ``r_char``, ``reference_radius``) and its unfiltered radial table, so the
tests of ``test_confinement.py`` and ``test_paw_energy_shift.py`` pin the engine's numbers.
"""
import json
from types import SimpleNamespace

import numpy as np

from mandacaru import _native
from mandacaru.pseudopotentials.native_context import context
from mandacaru.pseudopotentials.record import record_of


def gaussian_shell(pp, shift, wall=(12.0, 0.6)):
    """The Gaussian polarization shell of ``pp`` at ``shift`` eV in ``wall``, unfiltered."""
    recipe = dict(zetas=1, polarizations=1, cutoff=None, energy_shift=float(shift),
                  wall=list(wall), split_norm=0.15, tail_norms=[0.15, 0.31, 0.54],
                  polarization="gaussian", diffuse=None)
    built = _native.paw_basis(context(), *record_of(pp), json.dumps(recipe))
    radial = next(values for _n, _l, _z, polarization, _d, _r, values in built["shells"]
                  if polarization)
    return SimpleNamespace(**built["polarization"], radial=np.asarray(radial))


def free_valence_rms(pp):
    """``{l: rms}`` (Bohr): the free-atom valence radius the Rust recipe sizes diffuse shells from,
    read from the reference of each channel's shell (a channel with no valence state takes the
    outermost one, so only the channels that are their own reference are kept)."""
    recipe = dict(zetas=1, polarizations=0, cutoff=None, energy_shift=None, wall=[12.0, 0.6],
                  split_norm=0.15, tail_norms=[0.15, 0.31, 0.54], polarization="orbital",
                  diffuse=dict(shells=[0, 1, 2, 3], count=1, ratio=3.5 ** 0.5, cutoff_norm=1e-4,
                               rms_ratio=2.5, rms_radius=None))
    built = _native.paw_basis(context(), *record_of(pp), json.dumps(recipe))
    return {row["l"]: row["reference_rms"] for row in built["diffuse"]
            if row["reference_l"] == row["l"] and row["index"] == 0}
