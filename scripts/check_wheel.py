"""Check a wheel's contents, then exercise it installed with compilers and integration imports blocked.

python scripts/check_wheel.py WHEEL --python VENV/bin/python
"""

import argparse
from pathlib import Path
import subprocess
import tempfile
import tomllib
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = (".c", ".h", ".cpp", ".f90")
BINARY_SUFFIXES = (".so", ".dylib", ".dll", ".pyd")
RETIRED_REQUIREMENTS = ("fastparquet", "pandas", "pyarrow")
RETIRED_FILES = ("mandacaru/basis/r2scan.py",)
SMOKE = r"""
import importlib.abc, importlib.metadata, json, pathlib, subprocess, sys
class NoOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','pandas','pyarrow','fastparquet','jax','qiskit','braket','cirq','matplotlib'}:
            raise AssertionError('optional import: '+fullname)
sys.meta_path.insert(0,NoOptionalImports())
def no_process(*args,**kwargs):
    raise AssertionError('runtime subprocess: '+repr(args))
subprocess.run=no_process
subprocess.Popen=no_process
import numpy as np
import mandacaru
from mandacaru import Geometry, Mandacaru, _native
assert pathlib.Path(sys.argv[1]).resolve() not in pathlib.Path(mandacaru.__file__).resolve().parents
requirements=importlib.metadata.requires('mandacaru')
required={r.split(';')[0].split('>')[0].split('=')[0].strip().lower()
          for r in requirements if ';' not in r}
assert {'numpy','ase','scipy','spglib','matplotlib',
        'qiskit','qiskit-ibm-runtime','amazon-braket-sdk','cirq'}.issubset(required), requirements
assert {'pandas','fastparquet','pyarrow'}.isdisjoint(required), requirements
g=Geometry((1,1),np.array([[2.5,2.5,2.13],[2.5,2.5,2.87]]),np.eye(3)*5)
with Mandacaru(method='dft',xc='pbe',grid_spacing=.5,resources={'threads':1}) as c:
    r=c.run(g)
    assert r.success and not r.scf.has_model and not c.integrals.has_eri
    molecular=r.optimal_energy
    assert np.isfinite(c.get_forces()).all()
with Mandacaru(method='adapt-vqe',load_hamiltonian=sys.argv[2],resources={'threads':1}) as c:
    r=c.run()
    assert r.success and abs(r.optimal_energy+25.871895455446616)<1e-8
from mandacaru.core.serialization import load_hamiltonian, save_hamiltonian
record=load_hamiltonian(sys.argv[2])
save_hamiltonian('cached.parquet',record.hamiltonian,mapping=record.mapping,
                 num_particles=record.num_particles,n_spatial_orbitals=record.n_spatial_orbitals)
assert pathlib.Path('cached.parquet').read_bytes()[:4]==b'PAR1'
assert load_hamiltonian('cached.parquet').hamiltonian.terms==record.hamiltonian.simplify().terms
with Mandacaru(method='adapt-vqe',load_hamiltonian='cached.parquet',resources={'threads':1}) as c:
    r=c.run()
    assert r.success and abs(r.optimal_energy+25.871895455446616)<1e-8
with Mandacaru(method='dft',xc='lda',load_crystal=sys.argv[3],resources={'threads':1}) as c:
    assert c.run().success
assert not pathlib.Path('.mandacaru').exists()
print(json.dumps({'version':mandacaru.__version__,'extension':_native.__file__,
                  'molecular_energy_ev':molecular,'runtime_compilation':False}))
"""


def project_scripts() -> dict[str, str]:
    """The console scripts ``pyproject.toml`` declares: the one list the wheel must match."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]


def content_problems(names, metadata, wheel_metadata, entry_points, scripts) -> list[str]:
    """Everything wrong with a wheel's file list and ``dist-info`` text (empty when sound)."""
    problems = []
    binaries = [n for n in names if n.endswith(BINARY_SUFFIXES)]
    if len(binaries) != 1 or not binaries[0].startswith("mandacaru/_native"):
        problems.append(f"want exactly one binary, mandacaru/_native*; found {binaries}")
    foreign = [n for n in names
               if n.endswith(SOURCE_SUFFIXES) or "libmandacaru_integrals" in n
               or "csrc" in n.split("/")]
    if foreign:
        problems.append(f"C/C++/Fortran sources, the retired integral library or csrc shipped: {foreign}")
    if "Root-Is-Purelib: false" not in wheel_metadata or "py3-none-any" in wheel_metadata:
        problems.append("the wheel must be a platform wheel (Root-Is-Purelib: false, no py3-none-any)")
    requires = {line.split(":", 1)[1].split(";")[0].strip().lower()
                for line in metadata.splitlines() if line.startswith("Requires-Dist:")}
    for retired in RETIRED_REQUIREMENTS:
        if any(r == retired or r.startswith(tuple(retired + mark for mark in " ><=~![("))
               for r in requires):
            problems.append(f"retired requirement {retired}: {sorted(requires)}")
    if "#gh-dark-mode-only" in metadata or "#gh-light-mode-only" in metadata:
        problems.append("the description still carries GitHub-only logo markup")
    for name, target in scripts.items():
        if f"{name} = {target}" not in entry_points:
            problems.append(f"entry point missing: {name} = {target}")
    stray = [n for n in names if {"target", "__pycache__"} & set(n.split("/"))]
    if stray:
        problems.append(f"build products in the wheel: {stray[:5]}")
    problems += [f"retired file shipped: {n}" for n in RETIRED_FILES if n in names]
    return problems


def wheel_problems(wheel: Path) -> list[str]:
    """Read ``wheel`` and return its content problems."""
    with ZipFile(wheel) as archive:
        names = archive.namelist()

        def text(suffix):
            return archive.read(next(n for n in names if n.endswith(suffix))).decode()

        return content_problems(names, text("/METADATA"), text("/WHEEL"),
                                text("/entry_points.txt"), project_scripts())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    options = parser.parse_args()
    problems = wheel_problems(options.wheel)
    if problems:
        raise SystemExit("wheel contents:\n  " + "\n  ".join(problems))
    with tempfile.TemporaryDirectory(prefix="mandacaru-wheel-") as temporary:
        subprocess.run(
            [
                str(options.python.absolute()),
                "-I",
                "-c",
                SMOKE,
                str(ROOT / "src"),
                str(ROOT / "test/migration/reference/h2_hamiltonian.json"),
                str(ROOT / "test/migration/phase5/si_crystal.json"),
            ],
            cwd=temporary,
            check=True,
            timeout=120,
        )


if __name__ == "__main__":
    main()
