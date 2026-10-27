"""Check a platform wheel and exercise it in an already provisioned clean venv.

python test/migration/check_wheel.py WHEEL --python VENV/bin/python --output PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent

SMOKE = r'''
import ctypes, json, os, pathlib, shutil, subprocess, sys
for name in ('MANDACARU_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['MANDACARU_BACKEND']='rust'
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(pathlib.Path.cwd()/'matplotlib')
import numpy as np
import mandacaru
from mandacaru import Mandacaru, _native
from mandacaru.integrals import ensure_backend
from ase import Atoms
prefix=pathlib.Path(sys.prefix).resolve()
assert pathlib.Path(mandacaru.__file__).resolve().is_relative_to(prefix)
assert pathlib.Path(_native.__file__).resolve().is_relative_to(prefix)
assert _native.ABI_VERSION==1
lib=ctypes.CDLL(_native.__file__)
assert lib.mandacaru_thread_policy_version()==1
status=ensure_backend(build=False)
assert status.available and status.is_rust
assert pathlib.Path(status.path).resolve()==pathlib.Path(_native.__file__).resolve()
compilers={'cargo','rustc','cc','clang','gcc','cmake'}
original_run,original_popen=subprocess.run,subprocess.Popen
def checked_call(original, command, *args, **kwargs):
    program=command[0] if isinstance(command,(list,tuple)) else command.split()[0]
    assert pathlib.Path(program).name not in compilers, 'Runtime compiler invocation is forbidden'
    return original(command,*args,**kwargs)
subprocess.run=lambda command,*a,**kw: checked_call(original_run,command,*a,**kw)
subprocess.Popen=lambda command,*a,**kw: checked_call(original_popen,command,*a,**kw)
original_which=shutil.which
shutil.which=lambda name, *a, **kw: None if pathlib.Path(name).name in compilers else original_which(name,*a,**kw)
spec=json.loads(sys.argv[1])
ctx=_native.Context(threads=1, memory_limit_bytes=1048576)
grid=_native.Grid.orthogonal([2]*3,[1.0]*3)
assert np.all(np.isfinite(ctx.laplacian(grid,np.ones((1,2,2,2)))))
case=spec['geometry']
atoms=Atoms(case['symbols'],positions=case['positions'],cell=case['cell'])
atoms.calc=Mandacaru(**case['options'],trace=False,references=False,dump_buffers=False)
energy=float(atoms.get_potential_energy())
assert atoms.calc.result.success and abs(energy-spec['rhf_energy_ev'])<1e-7
native_geometry={}
for method in ('rhf','vqe','adapt-vqe'):
    work=atoms.copy()
    options=dict(case['options'],method=method,engine='rust',trace=False)
    work.calc=Mandacaru(**options)
    value=float(work.get_potential_energy())
    expected=spec['rhf_energy_ev'] if method=='rhf' else spec['vqe_energy_ev']
    assert abs(value-expected)<1e-7
    assert (work.calc.result.converged if method=='adapt-vqe' else work.calc.result.success)
    assert work.calc.result.timings['engine']=='rust'
    native_geometry[method]=value
case=spec['dft_geometry']
work=Atoms(case['symbols'],positions=case['positions'],cell=case['cell'])
work.calc=Mandacaru(**case['options'],engine='rust',trace=False)
native_dft_energy=float(work.get_potential_energy())
assert abs(native_dft_energy-spec['dft_energy_ev'])<1e-7
native_forces=work.get_forces()
assert np.allclose(native_forces,spec['dft_forces'],atol=2e-6)
assert work.calc.force_result.details['engine']=='rust'
periodic=Mandacaru(method='dft',engine='rust',load_crystal=sys.argv[3],smearing=.01,trace=False).run()
assert periodic.success and abs(periodic.optimal_energy-spec['periodic_energy_ev'])<1e-6
assert periodic.timings['setup_engine']=='cached-crystal'
ewald=_native.Context().ewald(np.zeros((1,3)),np.ones(1),np.eye(3)*5)
assert abs(2*ewald['energy']*5+2.83729747948062)<1e-10
from mandacaru.basis.atomic_solver import solve_atom
assert solve_atom(2,engine='rust',points=300).converged
from mandacaru.algorithms._rust_molecular import basis_specification
from mandacaru._native_adapter import geometry_from_atoms
samples=_native.Context().sample_basis(geometry_from_atoms(work),np.zeros((2,3)),
    basis_spec=json.dumps(basis_specification({'name':'NAO','size':'DZP','n_grid':200})))
assert samples.shape==(10,2) and np.all(np.isfinite(samples))
ae_samples=_native.Context().sample_basis(geometry_from_atoms(work),np.zeros((2,3)),
    basis_spec=json.dumps(basis_specification({'name':'NAO-AE','tier':1,'points':400})))
assert ae_samples.shape==(10,2) and np.all(np.isfinite(ae_samples))
calc=Mandacaru(method='vqe',load_hamiltonian=sys.argv[2],trace=False,references=False,
              optimizer={'method':'L-BFGS','maxiter':500,'tol':1e-12},sector=False,sparse=False)
result=calc.run()
assert result.success and abs(result.optimal_energy-spec['vqe_energy_ev'])<1e-7
native_results={}
for method in ('vqe','adapt-vqe'):
    native_calc=Mandacaru(method=method,engine='rust',load_hamiltonian=sys.argv[2],trace=False,
                         engine_options={'threads':1},optimizer={'method':'L-BFGS','tol':1e-12})
    native_result=native_calc.run()
    assert native_result.timings['engine']=='rust'
    assert (native_result.success if method=='vqe' else native_result.converged)
    assert abs(native_result.optimal_energy-spec['vqe_energy_ev'])<1e-7
    native_results[method]=native_result.optimal_energy
from qiskit import QuantumCircuit, transpile
from mandacaru.algorithms.time_evolution import SuzukiTrotter
from mandacaru.algorithms.qpe import QuantumPhaseEstimation
from mandacaru.algorithms.mp2 import mp2_natural_orbitals
from mandacaru.core.mapping import PauliSum
simple=PauliSum({'X':.4,'Z':.3,'I':-.7},num_qubits=1)
psi=np.array([1.,0.],dtype=complex)
prop=SuzukiTrotter(simple,order=4,engine='rust')
evolved=prop.evolve(psi,.7,steps=3)
assert np.allclose(prop.evolve(evolved,.7,steps=3,inverse=True),psi,atol=1e-12)
qpe=QuantumPhaseEstimation(simple,4,engine='rust',verbose=False).run(psi)
assert abs(qpe.probabilities.sum()-1)<1e-12
hh=np.diag([-1.,1.]);gg=np.zeros((2,)*4);gg[0,0,1,1]=.2
mp2=mp2_natural_orbitals(hh,gg,1,engine='rust')
assert abs(mp2.correlation_energy+.01)<1e-14
for method in ('subspace-vqe','subspace-adapt-vqe'):
    levels=Mandacaru(method=method,engine='rust',load_hamiltonian=sys.argv[2],
                     trace=False,references=False).run()
    assert abs(levels.optimal_energy-spec['vqe_energy_ev'])<1e-7
    assert np.allclose(np.asarray(levels.states).conj()@np.asarray(levels.states).T,np.eye(2),atol=1e-12)
from mandacaru.core.serialization import load_hamiltonian, save_hamiltonian
circuit=QuantumCircuit(2)
circuit.h(0)
circuit.cx(0,1)
compiled=transpile(circuit,basis_gates=['u','cx'],optimization_level=1)
assert compiled.count_ops().get('cx')==1
record=load_hamiltonian(sys.argv[2])
save_hamiltonian('cached.parquet',record.hamiltonian,mapping=record.mapping,
                 num_particles=record.num_particles,n_spatial_orbitals=record.n_spatial_orbitals)
restored=load_hamiltonian('cached.parquet')
assert restored.hamiltonian.terms==record.hamiltonian.terms
print(json.dumps({'package':str(pathlib.Path(mandacaru.__file__).relative_to(prefix)),
                  'extension':str(pathlib.Path(_native.__file__).relative_to(prefix)),
                  'abi_version':_native.ABI_VERSION,'rhf_energy_ev':energy,
                  'cached_vqe_energy_ev':result.optimal_energy,'runtime_compilation':False,
                  'rust_engine_energies_ev':native_results,
                  'rust_geometry_energies_ev':native_geometry,
                  'rust_dft_energy_ev':native_dft_energy,
                   'rust_stationary_forces_and_radial_atoms':True,
                  'rust_periodic_energy_ev':periodic.optimal_energy,
                  'rust_periodic_scf_and_ewald':True,
                  'rust_dynamics_qpe_mp2_subspace':True,
                  'qiskit_transpile_and_parquet_roundtrip':True}))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.wheel) as wheel:
        names = wheel.namelist()
        binary = [name for name in names if name.endswith((".so", ".dylib", ".dll", ".pyd"))]
        assert len(binary) == 1 and binary[0].startswith("mandacaru/_native"), binary
        metadata = wheel.read(next(n for n in names if n.endswith("/METADATA"))).decode()
        wheel_metadata = wheel.read(next(n for n in names if n.endswith("/WHEEL"))).decode()
        entry_points = wheel.read(next(n for n in names if n.endswith("/entry_points.txt"))).decode()
        assert "Root-Is-Purelib: false" in wheel_metadata and "py3-none-any" not in wheel_metadata
        assert "#gh-dark-mode-only" not in metadata and "#gh-light-mode-only" not in metadata
        assert "mandacaru-build = mandacaru.pseudopotentials.build_cli:main" in entry_points
        assert "mandacaru = mandacaru.cli:main" in entry_points
        assert not any("/target/" in n or "/__pycache__/" in n for n in names)
    reference = json.loads((HERE / "reference/numpy.json").read_text())
    spec = {"geometry": reference["cases"]["h2_rhf"]["definition"],
            "dft_geometry": reference["cases"]["h2_dft_lda_forces"]["definition"],
            "dft_energy_ev": reference["cases"]["h2_dft_lda_forces"]["samples"][0]["values"]["energy_ev"],
            "dft_forces": reference["cases"]["h2_dft_lda_forces"]["samples"][0]["values"]["forces_ev_per_angstrom"],
            "periodic_energy_ev": reference["cases"]["si_periodic_lda"]["samples"][0]["values"]["energy_ev"],
            "rhf_energy_ev": reference["cases"]["h2_rhf"]["samples"][0]["values"]["energy_ev"],
            "vqe_energy_ev": reference["cases"]["h2_cached_vqe"]["samples"][0]["values"]["energy_ev"]}
    with tempfile.TemporaryDirectory(prefix="mandacaru-wheel-") as temporary:
        work = Path(temporary)
        fixture = work / "h2.json"
        fixture.write_bytes((HERE / "reference/h2_hamiltonian.json").read_bytes())
        crystal = work / "si_crystal.json"
        crystal.write_bytes((HERE / "phase5/si_crystal.json").read_bytes())
        result = subprocess.run([str(args.python.absolute()), "-I", "-c", SMOKE,
                                 json.dumps(spec), str(fixture), str(crystal)], cwd=work,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    report = {"schema_version": 1, "passed": True,
              "wheel": args.wheel.name, "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
              "wheel_metadata": wheel_metadata, "smoke": json.loads(result.stdout.splitlines()[-1])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
