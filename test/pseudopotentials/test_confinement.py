# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_confinement.py

# This code is part of Mandacaru.
# MIT License

"""The confined first zeta for every pseudopotential family.

PAW-LCAO's confinement is pinned in ``test_paw_energy_shift.py``; this file
checks that ONCVPSP (its own local potential and projectors, ``q = 0``)
solves the same confined problem:
the free level comes back when the wall recedes, the requested shift is the
achieved one, and the radii agree with PAW-LCAO's for the same atom.
"""

import numpy as np
import pytest

from mandacaru.pseudopotentials import get_oncv, get_paw
from mandacaru.pseudopotentials.confinement import (confined_energy,
                                                    confined_orbital,
                                                    free_energy)

LOADERS = {"PAW-LCAO": get_paw, "ONCVPSP": get_oncv}
NORM_CONSERVING = ("ONCVPSP",)


def _load(family, symbol):
    try:
        return LOADERS[family](symbol)
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"no {family} library: {error}")


@pytest.mark.parametrize("family", NORM_CONSERVING)
@pytest.mark.parametrize("symbol", ["H", "Li", "O"])
def test_a_far_wall_gives_back_the_free_level(family, symbol):
    """The confined problem is the dataset's own: with the wall at 25 Bohr the
    level is the free atom's."""
    pp = _load(family, symbol)
    for l in pp.channels:
        assert confined_energy(pp, l, 25.0) == pytest.approx(
            free_energy(pp, l), abs=1e-4)


@pytest.mark.parametrize("family", NORM_CONSERVING)
def test_the_requested_shift_is_the_achieved_one(family):
    pp = _load(family, "O")
    for l in pp.channels:
        orbital = confined_orbital(pp, l, 0.1)
        assert orbital.achieved_shift == pytest.approx(0.1, abs=1e-6)
        # Unit norm, and nothing beyond the wall.
        r = np.asarray(pp.r)
        assert np.trapezoid(orbital.radial ** 2 * r * r, r) == pytest.approx(
            1.0, abs=1e-3)
        assert not np.any(orbital.radial[r >= orbital.r_c])


@pytest.mark.parametrize("family", NORM_CONSERVING)
@pytest.mark.parametrize("symbol", ["H", "O"])
def test_the_radii_agree_with_paw(family, symbol):
    """Measured 2026-09-26 at 0.1 eV: O 2p 5.340 (PAW), 5.345 (ONCVPSP)
    Bohr -- the potentials differ slightly, the recipe does not."""
    pp, paw = _load(family, symbol), _load("PAW-LCAO", symbol)
    for l in pp.channels:
        assert confined_orbital(pp, l, 0.1).r_c == pytest.approx(
            confined_orbital(paw, l, 0.1).r_c, abs=0.05)
