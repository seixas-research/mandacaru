# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_native_context.py

# This code is part of Mandacaru.
# MIT License

"""The native context the PAW radial routines share
(:mod:`mandacaru.pseudopotentials.native_context`)."""

from __future__ import annotations

import numpy as np

from mandacaru import _native
from mandacaru.pseudopotentials import multipoles
from mandacaru.pseudopotentials.native_context import context


def test_one_context_serves_each_thread_request():
    """A request is served by one shared context; no request takes the
    engine's own thread policy (``MANDACARU_NUM_THREADS``, ``OMP_NUM_THREADS``,
    then the hardware), a request for ``n`` threads takes ``n``."""
    assert context() is context()
    assert context().threads == _native.num_threads()
    assert context(2) is context(2)
    assert context(2).threads == 2


def test_a_radial_call_releases_its_memory():
    """Every native call reserves its workspace through the context and gives
    it back."""
    before = context().memory_used_bytes
    multipoles.shape_function(np.linspace(0.0, 3.0, 50), 1.5, 2)
    assert context().memory_used_bytes == before
