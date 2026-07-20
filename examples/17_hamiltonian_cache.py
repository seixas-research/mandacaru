# -*- coding: utf-8 -*-
# file: examples/17_hamiltonian_cache.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Saving and reloading the qubit Hamiltonian: Parquet and JSON.

Building a molecular Hamiltonian -- the real-space one- and two-body integrals
plus the fermion-to-qubit mapping -- is the most expensive stage of a Carcará
run, and it does not depend on the *algorithm* being run afterwards.  So it is
worth caching: build once, then sweep pools, optimizers, ansätze or temperature
schedules for free.

This example builds LiH once in each format, reloads it, and checks that

* the reloaded Hamiltonian is bit-for-bit the operator that was saved;
* a driver reconstructed from the file reproduces the original energy **without
  a geometry** -- no integrals, no mapping;
* the two formats are interchangeable, and :func:`~carcara.core.detect_format`
  identifies either one automatically (from the extension, or failing that from
  the file's own leading bytes).

Which format?
-------------
``"parquet"`` (default)
    Compressed, columnar and compact -- the right choice for the
    :math:`10^4`-:math:`10^6`-term Hamiltonians of a realistic active space, and
    queryable straight from pandas.
``"json"``
    Plain text: readable, diffable and dependency-free.  Use it when you want to
    inspect the operator by eye, or to avoid a Parquet engine entirely -- on some
    platforms ``pyarrow``'s writer is unstable in a process that has also run
    Qiskit's transpiler (see :mod:`carcara.core.serialization`).
"""

from __future__ import annotations

import os
import time

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.core import detect_format, load_hamiltonian

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

POOL = "qeb"
MAX_ITERATIONS = 10


def lih():
    return Atoms("LiH",
                 positions=[[7.5, 7.5, 7.5 - 0.7975], [7.5, 7.5, 7.5 + 0.7975]],
                 cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
                 pbc=True)


rule = "=" * 76
print(rule)
print("Hamiltonian cache: build once, reload in any format")
print(rule)

records = {}
for fmt in ("parquet", "json"):
    path = os.path.join(DATA, f"lih_cache.{fmt}")

    # -- build (integrals + mapping) and save ---------------------------- #
    atoms = lih()
    atoms.calc = ADAPTVQE(pool=POOL, basis={"name": "FAO"}, h=0.25,
                          verbose=False, profile=False,
                          max_iterations=MAX_ITERATIONS,
                          save_hamiltonian=path, hamiltonian_format=fmt)
    t0 = time.perf_counter()
    atoms.get_total_energy()
    build_seconds = time.perf_counter() - t0
    built = atoms.calc.adapt_result.optimal_energy
    live = atoms.calc.hamiltonian.simplify()

    # -- reload, with no geometry at all --------------------------------- #
    t0 = time.perf_counter()
    driver = ADAPTVQE(pool=POOL, load_hamiltonian=path, verbose=False,
                      profile=False, max_iterations=MAX_ITERATIONS)
    reloaded = driver.run()
    load_seconds = time.perf_counter() - t0

    record = load_hamiltonian(path)
    records[fmt] = record
    size_kb = os.path.getsize(path) / 1024.0

    # The file alone specifies the problem: operator + particles + orbitals.
    assert record.num_particles == atoms.calc.num_particles
    assert record.n_spatial_orbitals == atoms.calc.n_qubits // 2
    assert record.mapping == atoms.calc.mapping
    # ... and the operator survives the round trip exactly.
    assert set(record.hamiltonian.terms) == set(live.terms)
    for label, coeff in live.terms.items():
        assert abs(record.hamiltonian.terms[label] - coeff) < 1e-12
    # ... so the reloaded driver reproduces the original energy.  Saving applies
    # `simplify()`, which drops terms below 1e-12; those are physically
    # irrelevant but can nudge the classical optimizer onto a marginally
    # different path, so agreement is checked well inside chemical accuracy
    # rather than bit-for-bit.
    assert abs(reloaded.optimal_energy - built) < 1e-6

    print(f"\n[{fmt}]  {os.path.basename(path)}")
    print(f"    detected format        : {detect_format(path)}")
    print(f"    Pauli terms            : {len(record.hamiltonian.terms)}")
    print(f"    file size              : {size_kb:.1f} KiB")
    print(f"    build (integrals+map)  : {build_seconds:.2f} s  "
          f"-> E = {built:.8f} Ha")
    print(f"    reload + rerun         : {load_seconds:.2f} s  "
          f"-> E = {reloaded.optimal_energy:.8f} Ha")
    print(f"    metadata               : {record.metadata}")

# --------------------------------------------------------------------------- #
# The two formats hold the same operator.
# --------------------------------------------------------------------------- #

parquet, js = records["parquet"], records["json"]
assert set(parquet.hamiltonian.terms) == set(js.hamiltonian.terms)
difference = max(abs(parquet.hamiltonian.terms[label] - coeff)
                 for label, coeff in js.hamiltonian.terms.items())
print(f"\nParquet vs JSON: identical term set, max coefficient difference "
      f"{difference:.1e}")

# Detection also works when the extension says nothing at all.
opaque = os.path.join(DATA, "lih_cache.bin")
from carcara.core import save_hamiltonian                            # noqa: E402

save_hamiltonian(opaque, parquet.hamiltonian, mapping=parquet.mapping,
                 num_particles=parquet.num_particles,
                 n_spatial_orbitals=parquet.n_spatial_orbitals,
                 format="parquet")
print(f"{os.path.basename(opaque)!r} has no known extension -> detected as "
      f"{detect_format(opaque)!r} from its leading bytes")
assert load_hamiltonian(opaque).num_particles == parquet.num_particles

print(f"\n{rule}")
print("Both formats round-trip exactly and reload without any geometry,")
print("integrals or fermion-to-qubit mapping.")
print(rule)
