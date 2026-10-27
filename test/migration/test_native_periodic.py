"""Native Phase 5 thermodynamics, crystal SCF, electrostatics and analysis."""
import copy
import gc
import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from mandacaru import Mandacaru, _native
from mandacaru.units import HARTREE_TO_EV

HERE=Path(__file__).parent
SI=HERE/'phase5/si_crystal.json'


def complex_values(a):
    a=np.asarray(a,complex).ravel()
    return np.column_stack([a.real,a.imag]).tolist()


def synthetic(functional='lda',spins=1):
    shape=(4,4,4)
    n=np.prod(shape)
    x=np.indices(shape)[0].ravel()
    psi=np.stack([np.ones(n),np.exp(2j*np.pi*x/shape[0])])/np.sqrt(n)
    # ngrid=volume, so the samples are exactly orthonormal.
    freq=np.stack(np.meshgrid(*[np.fft.fftfreq(i)*2*np.pi for i in shape],indexing='ij'))
    g2=np.sum(freq*freq,axis=0).ravel()
    kernel=np.zeros(n)
    kernel[g2>0]=4*np.pi/g2[g2>0]
    xc_wave=freq.reshape(3,-1).copy()
    # Derivatives of real fields have zero Nyquist multiplier.
    for a in range(3):
        nyq=np.indices(shape)[a].ravel()==shape[a]//2
        xc_wave[a,nyq]=0.
    return dict(schema_version=1,representation='periodic-paw-snapshot',energy_unit='Ha',length_unit='bohr',
                grid=dict(shape=shape,origin=[0.]*3,step=np.eye(3).ravel().tolist()),
                m=2,p=0,channels=0,channel_labels=[],
                kpoints=[dict(k=[0.]*3,fractional=[0.]*3,weight=1.,psi=complex_values(psi),
                              overlap=complex_values(np.eye(2)),fixed=complex_values(np.diag([-.5,.3])),projections=[])],
                blocks=[],core=np.zeros(n).tolist(),initial=np.full(n,2/n).tolist(),initial_m=np.zeros(n).tolist(),initial_q=[],
                kernel=kernel.tolist(),xc_wave=xc_wave.ravel().tolist(),phase=complex_values(np.ones(n)),kerker=np.ones(n).tolist(),
                ion=complex_values(np.zeros(n)),hats=[],u=[],uion=[],linear=[],eion=0.,ion_constant=0.,constant=.13,
                reference=.02,electrons=2.,spins=spins,functional=functional,relativistic=False,permutations=[],qsym=[])


@pytest.mark.parametrize('method',['fermi-dirac','gaussian','methfessel-paxton'])
@pytest.mark.parametrize('degeneracy',[1.,2.])
def test_metallic_fermi_occupations_entropy_and_variational_free_energy(method,degeneracy):
    from mandacaru.algorithms.periodic_dft import fermi_level,occupation,entropy
    levels=np.array([[-.3,-.01,.12,.4],[-.2,.03,.17,.5]])
    weights=np.array([.25,.75])
    width=.06
    electrons=degeneracy*1.7
    ctx=_native.Context()
    raw=ctx.fermi_distribution(levels.tolist(),weights.tolist(),electrons,method=method,width_ha=width,degeneracy=degeneracy)
    mu=fermi_level(levels,weights,electrons,method,width,degeneracy)
    assert raw['fermi_level']==pytest.approx(mu,abs=1e-12)
    f=degeneracy*occupation((levels-mu)/width,method)
    assert np.allclose(raw['occupations'],f,atol=2e-12)
    assert np.sum(f*weights[:,None])==pytest.approx(electrons,abs=2e-11)
    expected_entropy=degeneracy*np.sum(entropy((levels-mu)/width,method)*weights[:,None])
    assert raw['entropy']==pytest.approx(expected_entropy,abs=1e-11)
    def free(w):
        r=ctx.fermi_distribution(levels.tolist(),weights.tolist(),electrons,method=method,width_ha=w,degeneracy=degeneracy)
        return np.sum(np.array(r['occupations'])*levels*weights[:,None])-w*r['entropy']
    assert (free(width+1e-5)-free(width-1e-5))/2e-5==pytest.approx(-raw['entropy'],abs=1e-7)


@pytest.mark.parametrize('size',[(1,1,1),(2,3,4),(3,3,3)])
@pytest.mark.parametrize('gamma',[False,True])
def test_native_kmesh_matches_ase_and_time_reversal_orbits(size,gamma):
    from mandacaru.algorithms._hamiltonian_from_atoms import monkhorst_pack_kpts
    from mandacaru.pseudopotentials.periodic_paw import time_reversal_reduce
    _,_,full=monkhorst_pack_kpts(dict(size=size,gamma=gamma))
    points,weights=time_reversal_reduce(full)
    result=_native.Context().kmesh(size,gamma=gamma)
    assert np.allclose(result['full'],full,atol=1e-14)
    assert np.allclose(result['points'],points,atol=1e-14)
    assert np.allclose(result['weights'],weights,atol=1e-14)
    mapping=np.array(result['mapping'])
    assert np.allclose(np.bincount(mapping)/len(full),result['weights'])


def test_space_group_reduction_checks_mesh_preservation():
    ctx=_native.Context()
    rotation=[0,1,0,1,0,0,0,0,1]
    a=ctx.kmesh([3,3,1],gamma=True,time_reversal=False)
    b=ctx.kmesh([3,3,1],gamma=True,rotations=[rotation],time_reversal=False)
    assert len(b['points'])<len(a['points']) and sum(b['weights'])==pytest.approx(1.)
    with pytest.raises(ValueError,match='preserve'):
        ctx.kmesh([2,3,1],rotations=[rotation])
    with pytest.raises(ValueError,match='unimodular'):
        ctx.kmesh([2,2,2],rotations=[[0]*9])


@pytest.mark.parametrize('skew',[False,True])
@pytest.mark.parametrize('eta',[None,.18,.45])
def test_ewald_matches_python_for_energy_forces_stress_and_potential(skew,eta):
    from mandacaru.core.ewald import ewald_energy,ewald_forces,ewald_stress,ewald_potential
    cell=np.diag([6.,7.,8.])
    if skew:
        cell+=np.array([[0.,.3,.1],[.2,0.,.4],[.1,.2,0.]])
    positions=np.array([[.7,1.,1.2],[2.1,3.,3.1]])
    charges=np.array([1.,-2.])
    points=np.array([[.3,2.,1.1],[3.3,1.,.9]])
    ctx=_native.Context(threads=1)
    raw=ctx.ewald(positions,charges,cell,eta=eta,points_bohr=points,softening_bohr=.2)
    assert raw['energy']==pytest.approx(ewald_energy(positions,charges,cell,eta=eta),abs=3e-12)
    assert np.allclose(raw['forces'],ewald_forces(positions,charges,cell,eta=eta),atol=3e-12)
    assert np.allclose(raw['stress'],ewald_stress(positions,charges,cell,eta=eta),atol=3e-12)
    assert np.allclose(raw['potential'],ewald_potential(positions,charges,cell,points,eta=eta,softening=.2),atol=3e-12)
    assert np.linalg.norm(raw['forces'].sum(axis=0))<1e-13
    moved=positions.copy();moved[0]+=cell[0]-2*cell[1]
    assert ctx.ewald(moved,charges,cell,eta=eta)['energy']==pytest.approx(raw['energy'],abs=3e-12)
    assert ctx.memory_used_bytes==0


def test_ewald_strain_and_displacement_derivatives_and_madelung_limit():
    ctx=_native.Context(threads=1)
    r=np.array([[.7,1.,1.2],[2.1,3.,3.1]])
    q=np.array([1.,-2.]);cell=np.array([[6.,.3,.1],[.2,7.,.4],[.1,.2,8.]])
    raw=ctx.ewald(r,q,cell)
    eps=1e-5
    plus=r.copy();minus=r.copy();plus[0,1]+=eps;minus[0,1]-=eps
    fd=-(ctx.ewald(plus,q,cell)['energy']-ctx.ewald(minus,q,cell)['energy'])/(2*eps)
    assert raw['forces'][0,1]==pytest.approx(fd,abs=1e-9)
    for a,b in [(0,0),(0,2),(2,1)]:
        strain=np.zeros((3,3));strain[a,b]=eps
        eplus=ctx.ewald(np.ascontiguousarray(r@(np.eye(3)+strain).T),q,np.ascontiguousarray(cell@(np.eye(3)+strain).T))['energy']
        eminus=ctx.ewald(np.ascontiguousarray(r@(np.eye(3)-strain).T),q,np.ascontiguousarray(cell@(np.eye(3)-strain).T))['energy']
        assert raw['stress'][a,b]==pytest.approx((eplus-eminus)/(2*eps*np.linalg.det(cell)),abs=2e-10)
    unit=ctx.ewald(np.zeros((1,3)),np.ones(1),np.eye(3)*5)
    assert 2*unit['energy']*5==pytest.approx(-2.83729747948062,abs=1e-10)


@pytest.mark.parametrize('functional',['lda','pbe'])
@pytest.mark.parametrize('spins',[1,2])
def test_uniform_periodic_gas_matches_independent_grid_functional(functional,spins):
    from mandacaru.integrals import Grid
    from mandacaru.integrals.exchange_correlation import evaluate
    payload=synthetic(functional,spins)
    ctx=_native.Context(threads=1)
    problem=ctx.periodic_problem(json.dumps(payload))
    raw=problem.solve(width_ha=.06).snapshot()
    assert raw['converged']
    n=64
    grid=Grid(center=[2/1.8897261246257702]*3,cell=np.eye(3)*4/1.8897261246257702,
              grid_spacing=1/1.8897261246257702,periodic=True)
    rho=np.full(n,2/n)
    xc=evaluate(grid,rho,functional)
    band=sum(np.sum(raw['occupations'][s,0]*np.array([-.5,.3])) for s in range(spins))
    assert raw['energy']==pytest.approx(band+xc.energy+.13,abs=1e-10)
    assert np.sum(raw['density'])==pytest.approx(2.,abs=1e-10)
    assert np.allclose(raw['eigenvalues'][0,0],np.array([-.5,.3])+xc.potential[0]+.02,atol=1e-10)
    for d in raw['density_matrices'].reshape(-1,2,2):
        assert np.allclose(d,d.conj().T)
    del problem
    gc.collect()
    assert ctx.memory_used_bytes==0


@pytest.fixture(scope='module')
def si_native():
    ctx=_native.Context(threads=1)
    problem=ctx.load_crystal(str(SI))
    result=problem.solve(width_ha=.01/HARTREE_TO_EV)
    return ctx,problem,result


def test_cached_silicon_matches_frozen_phase0_with_explicit_fermi_plateau(si_native):
    _,_,result=si_native
    raw=result.snapshot()
    frozen=json.loads((HERE/'reference/numpy.json').read_text())['cases']['si_periodic_lda']
    expected=frozen['samples'][0]['values']
    assert raw['converged']
    assert raw['extrapolated_energy']*HARTREE_TO_EV==pytest.approx(expected['energy_ev'],abs=1e-6)
    assert np.allclose(raw['eigenvalues'][0]*HARTREE_TO_EV,expected['eigenvalues_ev'],atol=1e-6)
    assert np.allclose(raw['weights'],expected['kpoint_weights'],atol=1e-14)
    eps=raw['eigenvalues'][0]*HARTREE_TO_EV
    valence,conduction=eps[:,3].max(),eps[:,4].min()
    # At 0.01 eV the count saturates over an interval in this gap. The
    # chemical potential is nonunique at machine precision; both values must
    # be in the gap and the native occupations must give the exact count.
    assert valence<raw['fermi_level']*HARTREE_TO_EV<conduction
    assert valence<expected['fermi_level_ev']<conduction
    assert np.sum(raw['occupations'][0]*np.array(raw['weights'])[:,None])==pytest.approx(8.,abs=1e-10)
    assert raw['free_energy']==pytest.approx(raw['energy'],abs=1e-12)
    payload=json.loads(SI.read_text())
    assert payload['provenance']['dataset_sources'][0]['sha256']==frozen['dataset_inputs'][0]['sha256']


def test_cached_crystal_runs_without_python_scf_fft_linalg_or_xc_callbacks(monkeypatch):
    from mandacaru.algorithms import periodic_dft,periodic_device
    from mandacaru.integrals import exchange_correlation
    def forbidden(*args,**kwargs):
        raise AssertionError('Python numerical callback entered the native periodic solve')
    for obj,name in [(periodic_dft.PeriodicKohnSham,'run'),(periodic_dft.PulayMixer,'mix'),
                     (periodic_device.CPUDevice,'eigensolve'),(np.linalg,'eigh'),(np.linalg,'solve'),
                     (np.fft,'fftn'),(np.fft,'ifftn'),(exchange_correlation,'evaluate')]:
        monkeypatch.setattr(obj,name,forbidden)
    result=Mandacaru(method='dft',engine='rust',load_crystal=SI,
                     smearing={'width':.01},trace=False).run()
    assert result.success and result.timings['setup_engine']=='cached-crystal'
    assert result.optimal_energy==pytest.approx(-211.60193975895757,abs=1e-6)


def test_frozen_bands_dos_projection_and_trial_subspaces_are_consistent(si_native):
    _,_,result=si_native
    raw=result.snapshot();payload=json.loads(SI.read_text())
    bands=result.bands(json.dumps(payload['kpoints']))
    assert np.allclose(bands['eigenvalues'],raw['eigenvalues'],atol=1e-12)
    assert np.allclose(bands['projections'].sum(axis=2),1.,atol=1e-10)
    energies=np.linspace(-2.,2.,1601)
    total=result.dos(energies,width_ha=.03)[:,0]
    projected=result.dos(energies,width_ha=.03,projected=True)
    assert np.allclose(projected.sum(axis=1),total,atol=1e-12)
    assert np.trapezoid(total,energies)==pytest.approx(16.,abs=1e-9)
    full=result.projectability(np.eye(8,dtype=complex))
    assert np.allclose(full['weights'],1.,atol=1e-10)
    assert np.allclose(full['hamiltonians'],full['hamiltonians'].swapaxes(-1,-2).conj(),atol=1e-10)
    assert np.allclose(np.linalg.eigvalsh(full['hamiltonians']),raw['eigenvalues'],atol=1e-10)
    trials=np.eye(8,dtype=complex)[:,:2].copy()
    small=result.projectability(trials)
    assert np.all(small['weights']>=-1e-12) and np.all(small['weights']<=1+1e-10)
    assert np.allclose(small['weights'].sum(axis=-1),2.,atol=1e-10)
    with pytest.raises(ValueError,match='dependent'):
        result.projectability(np.ones((8,2),dtype=complex))


@pytest.mark.parametrize('hopping,expected',[(.5,np.pi),(1.5,0.)])
def test_wilson_loop_ssh_phase_is_invariant_under_local_gauges(hopping,expected):
    k=np.linspace(0.,2*np.pi,301,endpoint=False)
    phase=np.angle(hopping+np.exp(1j*k))
    occupied=np.stack([-np.exp(1j*phase),np.ones(len(k))],axis=1)/np.sqrt(2)
    links=np.array([np.vdot(occupied[i],occupied[(i+1)%len(k)]) for i in range(len(k))]).reshape(-1,1,1)
    ctx=_native.Context()
    berry,singular=ctx.wilson_loop(links)
    assert abs(np.angle(np.exp(1j*(berry-expected))))<1e-10
    gauge=np.exp(1j*np.random.default_rng(27).normal(size=len(k)))
    links[:,0,0]*=np.conj(gauge)*np.roll(gauge,-1)
    shifted,_=ctx.wilson_loop(links)
    assert abs(np.angle(np.exp(1j*(berry-shifted))))<1e-10
    assert singular>.99
    links[0]=0.
    with pytest.raises(ValueError,match='singular'):
        ctx.wilson_loop(links)


def test_cached_validation_failure_budget_and_result_ownership(tmp_path):
    payload=synthetic()
    for mutate in [lambda p:p.update(energy_unit='eV'),lambda p:p['kpoints'][0].update(weight=.5),
                   lambda p:p['kpoints'][0].update(overlap=complex_values(np.diag([1.,-1.]))),
                   lambda p:p.update(kernel=[1.]*64),lambda p:p.update(permutations=[[0]*64])]:
        bad=copy.deepcopy(payload);mutate(bad)
        ctx=_native.Context()
        with pytest.raises(ValueError):
            ctx.periodic_problem(json.dumps(bad))
        assert ctx.memory_used_bytes==0
    ctx=_native.Context(memory_limit_bytes=1024)
    with pytest.raises(MemoryError):
        ctx.periodic_problem(json.dumps(payload))
    assert ctx.memory_used_bytes==0
    ctx=_native.Context()
    problem=ctx.periodic_problem(json.dumps(payload))
    base=ctx.memory_used_bytes
    failed=problem.solve(max_iterations=1)
    assert not failed.converged
    del failed
    assert ctx.memory_used_bytes==base
    result=problem.solve()
    owned=ctx.memory_used_bytes
    result.dos(np.linspace(-1,1,20),width_ha=.1)
    assert ctx.memory_used_bytes==owned
    snap=result.snapshot();snap['eigenvalues'][:]=0
    assert np.linalg.norm(result.snapshot()['eigenvalues'])>.1
    path=tmp_path/'crystal.json';problem.save(str(path))
    saved=ctx.load_crystal(str(path)).solve().snapshot()
    assert np.allclose(saved['eigenvalues'],result.snapshot()['eigenvalues'])
    del result,problem
    gc.collect()
    assert ctx.memory_used_bytes==0


def test_ase_silicon_adapter_getters_and_native_off_mesh_spectra():
    from mandacaru.pseudopotentials.paw import get_paw
    try:
        get_paw('Si',paw_dataset='lda-sr')
    except FileNotFoundError:
        pytest.skip('external lda-sr/Si.parquet dataset unavailable; the portable native crystal tests still run')
    atoms=bulk('Si','diamond',a=5.43)
    atoms.calc=Mandacaru(method='dft',engine='rust',basis={'name':'PAW-LCAO','size':'SZ','filter_cutoff':200},
                        grid_spacing=.4,kpts={'size':(2,2,2),'gamma':True},smearing=.01,trace=False)
    assert atoms.get_potential_energy()==pytest.approx(-211.60193975895757,abs=1e-6)
    assert atoms.calc.result.timings['setup_engine']=='python-paw-adapter'
    assert atoms.calc.get_number_of_spins()==1
    assert atoms.calc.get_eigenvalues().shape==(8,)
    assert len(atoms.calc.get_ibz_k_points())==3
    assert sum(atoms.calc.get_k_point_weights())==pytest.approx(1.)
    assert atoms.calc.get_bz_k_points().shape==(8,3)
    band=atoms.calc.band_structure(path='GX',npoints=9)
    assert band.energies.shape==(1,9,8)
    axis,dos=atoms.calc.dos(npoints=801)
    _,pdos=atoms.calc.pdos(energies=axis)
    assert np.allclose(sum(pdos.values()),dos,atol=1e-10)
    with pytest.raises(NotImplementedError,match='derivatives'):
        atoms.get_forces()


@pytest.mark.parametrize('spin',[False,True])
def test_periodic_pbe_and_collinear_spin_match_python_on_skew_cells(spin):
    from ase import Atoms
    from mandacaru.pseudopotentials.paw import get_paw
    xc='lda' if spin else 'pbe'
    try:
        get_paw('H',paw_dataset=f'{xc}-sr')
    except FileNotFoundError:
        pytest.skip(f'external {xc}-sr/H.parquet dataset unavailable')
    cell=np.array([[4.,.4,0.],[0.,4.,.3],[.1,0.,4.5]])
    atoms=Atoms('H' if spin else 'H2',positions=[[1.2,1.6,1.9]] if spin else
                [[1.2,1.6,1.9],[2.,2.3,2.1]],cell=cell,pbc=True)
    if spin:
        atoms.set_initial_magnetic_moments([1.])
    opts=dict(method='dft',xc=xc,basis={'name':'PAW-LCAO','size':'SZ','filter_cutoff':200},
              grid_spacing=.7,kpts={'size':(2,1,1),'gamma':True},smearing=.3,trace=False,dump_buffers=False)
    py=atoms.copy();py.calc=Mandacaru(**opts)
    expected=py.get_potential_energy()
    atoms.calc=Mandacaru(**opts,engine='rust')
    actual=atoms.get_potential_energy()
    assert atoms.calc.result.success and py.calc.result.success
    assert actual==pytest.approx(expected,abs=3e-6)
    assert atoms.calc.result.scf.magnetic_moment==pytest.approx(py.calc.result.scf.magnetic_moment,abs=1e-5)
    assert atoms.calc.get_number_of_spins()==(2 if spin else 1)
    if spin:
        # Magnetic initial states must not be projected onto a nonmagnetic
        # spatial symmetry group during density assembly.
        assert atoms.calc.solver._crystal_adapter.symmetry is None
    assert atoms.calc.results['free_energy']==pytest.approx(py.calc.results['free_energy'],abs=3e-6)
    expected_levels=np.asarray(py.calc.result.scf.eigenvalues)
    actual_levels=np.asarray(atoms.calc.result.scf.eigenvalues)
    assert np.allclose(actual_levels,expected_levels,atol=2e-6)
