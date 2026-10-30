# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_native_context.py

# This code is part of Mandacaru.
# MIT License

"""The native context the PAW radial routines share
(:mod:`mandacaru.pseudopotentials.native_context`)."""

from __future__ import annotations

import numpy as np

from mandacaru.pseudopotentials import multipoles
from mandacaru.pseudopotentials.native_context import context


def test_one_single_threaded_context_serves_the_process():
    assert context() is context()
    assert context().threads == 1


def test_a_radial_call_releases_its_memory():
    """Every native call reserves its workspace through the context and gives
    it back."""
    before = context().memory_used_bytes
    multipoles.shape_function(np.linspace(0.0, 3.0, 50), 1.5, 2)
    assert context().memory_used_bytes == before
