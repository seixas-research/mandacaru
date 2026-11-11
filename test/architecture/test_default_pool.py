"""The operator pool a quantum method uses when ``pool=`` is left out.

Owner decision 2026-11-09: the growing ADAPT methods default to ``ceo`` (on the 4-cell metallic
hydrogen chain the fermionic pool stops 0.0169 eV above the exact energy (0.0196 before the
band-limited supercell of 2026-11-10); ``ceo`` reaches it,
``test/migration/test_native_bloch.py``). Every other quantum method keeps ``fermionic``: a fixed
ansatz and the search methods are built on the fermionic excitations, and ``ansatz="hva"``
requires it. The one table is ``METHODS[method].default_pool``; an explicit ``pool=`` wins.
"""

import pytest
from ase import Atoms

from mandacaru import Mandacaru, _native
from mandacaru.specifications import METHODS
from mandacaru.utils import parse_output

GROWING = ("adapt-vqe", "subspace-adapt-vqe")
FIXED = ("vqe", "subspace-vqe", "vasqa", "valsqa")


def build(method, **options):
    return Mandacaru(method=method, trace=False, **options)


def test_the_table_covers_every_quantum_method_and_no_classical_one():
    quantum = {name for name, capability in METHODS.items() if capability.quantum}
    assert quantum == set(GROWING) | set(FIXED)
    assert {name: METHODS[name].default_pool for name in quantum} == {
        **dict.fromkeys(GROWING, "ceo"), **dict.fromkeys(FIXED, "fermionic")}
    assert all(c.default_pool is None for c in METHODS.values() if not c.quantum)


@pytest.mark.parametrize("method", GROWING)
def test_a_growing_method_defaults_to_ceo(method):
    assert build(method)._pool == "ceo"


@pytest.mark.parametrize("method", FIXED)
def test_every_other_quantum_method_stays_fermionic(method):
    assert build(method)._pool == "fermionic"


def test_the_unspecified_method_is_adapt_vqe_with_ceo():
    assert Mandacaru(trace=False)._pool == "ceo"


@pytest.mark.parametrize("method", GROWING + FIXED)
@pytest.mark.parametrize("pool", ["fermionic", "qeb", "ceo-ovp"])
def test_an_explicit_pool_always_wins(method, pool):
    assert build(method, pool=pool)._pool == pool


def test_a_classical_method_has_no_pool_and_refuses_one():
    for method in ("rhf", "uhf", "ghf", "dft"):
        assert Mandacaru(method=method, trace=False)._pool is None
        with pytest.raises(ValueError, match="does not use"):
            Mandacaru(method=method, pool="fermionic", trace=False)


def test_the_hamiltonian_variational_ansatz_works_at_the_default_and_refuses_ceo():
    assert build("vqe", ansatz="hva")._pool == "fermionic"
    with pytest.raises(ValueError, match="HVA"):
        build("vqe", ansatz="hva", pool="ceo")
    with pytest.raises(ValueError, match="HVA"):
        build("adapt-vqe", ansatz="hva")


def test_an_unknown_pool_still_raises():
    with pytest.raises(ValueError, match="unknown native operator pool"):
        build("adapt-vqe", pool="uccsd")


def test_the_resolved_pool_reaches_the_solver_and_the_log(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.center(vacuum=3.0)
    atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.4, trace=False,
                           profile=False, max_iterations=1, txt=str(tmp_path / "run.log"))
    atoms.get_total_energy()
    labels = atoms.calc.problem.pool_labels
    assert labels and all(label.startswith(("CEO[", "D(")) for label in labels)
    setup = parse_output(str(tmp_path / "run.log"))["setup"]
    assert setup["pool"] == "ceo"
    assert setup["pool_class"] == _native.pool_kind("ceo") == "coupled exchange operators"
