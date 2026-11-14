"""The radial multipole moments of a PAW-LCAO dataset in either projector basis (a test view).

The moments are the engine's (``paw_radial_moments``); this view only picks the stored monopole
over a reconstruction and applies the engine's basis transform for ``"raw"``.
"""

import numpy as np

from mandacaru import _native
from mandacaru.pseudopotentials.multipoles import radial_moments
from mandacaru.pseudopotentials.native_context import context


def multipole_moments(dataset, l1, l2, L, basis="dual"):
    """``Delta^(L)_ij`` of channels ``l1`` and ``l2`` in the ``"dual"`` or ``"raw"`` basis.

    The monopole of a channel with itself is the stored overlap correction: rebuilding it from the
    tabulated waves agrees only to about 1e-8, which the raw basis's large ``B^-1`` amplifies.
    """
    if int(l1) == int(l2) and int(L) == 0:
        delta = np.asarray(dataset.channels[int(l1)].overlap_correction, dtype=float)
    else:
        delta = radial_moments(dataset, l1, l2, L)
    if basis == "raw":
        return _native.paw_multipole_transform(
            context(),
            np.ascontiguousarray(dataset.channels[int(l1)].vanderbilt, dtype=np.float64),
            np.ascontiguousarray(dataset.channels[int(l2)].vanderbilt, dtype=np.float64),
            np.ascontiguousarray(delta, dtype=np.float64))
    if basis != "dual":
        raise ValueError(f"unknown projector basis {basis!r}")
    return delta
