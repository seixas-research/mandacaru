# -*- coding: utf-8 -*-
# file: test/algorithms/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Tests for :mod:`mandacaru.algorithms`.

The test tree mirrors ``src/mandacaru/``: one package per source package, and
a module per source module wherever the tests have a single owner.  The
``__init__.py`` is what lets two packages hold a test module of the same name
(``basis/test_gaussian.py`` and ``core/test_hamiltonian.py`` live beside
``pseudopotentials/test_paw.py`` without pytest having to guess).
"""
