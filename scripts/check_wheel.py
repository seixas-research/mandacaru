"""Exercise an installed wheel with compilers and optional imports blocked."""

import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SMOKE = r"""
import importlib.abc, importlib.metadata, json, pathlib, subprocess, sys
class NoOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ase','scipy','pandas','jax','qiskit','braket','cirq','matplotlib'}:
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
assert all('extra ==' in r or r.lower().startswith('numpy') for r in requirements), requirements
g=Geometry((1,1),np.array([[2.5,2.5,2.13],[2.5,2.5,2.87]]),np.eye(3)*5)
with Mandacaru(method='dft',xc='pbe',grid_spacing=.5,resources={'threads':1}) as c:
    r=c.run(g)
    assert r.success and not r.scf.has_model and not c.integrals.has_eri
    molecular=r.optimal_energy
    assert np.isfinite(c.get_forces()).all()
with Mandacaru(method='adapt-vqe',load_hamiltonian=sys.argv[2],resources={'threads':1}) as c:
    r=c.run()
    assert r.success and abs(r.optimal_energy+25.871895455446616)<1e-8
with Mandacaru(method='dft',xc='lda',load_crystal=sys.argv[3],resources={'threads':1}) as c:
    assert c.run().success
assert not pathlib.Path('.mandacaru').exists()
print(json.dumps({'version':mandacaru.__version__,'extension':_native.__file__,
                  'molecular_energy_ev':molecular,'runtime_compilation':False}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    options = parser.parse_args()
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
