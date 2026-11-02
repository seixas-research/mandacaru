"""Freeze the reference of the crystal force on a real SCF crystal (CF-0 minimal, input of CF-4).

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set::

    python test/migration/freeze_periodic_gradients.py list
    python test/migration/freeze_periodic_gradients.py case si [--output DIR]

The collinear scalar LDA cases (``si``, ``al2``, ``lih``) use the ``lda-sr`` datasets; the hybrid cases (``lih_hse06``,
``si_hse06``, ``si_hse06_gb``) run HSE06 with the ``pbe-sr`` datasets and record the exact-exchange divergence treatment
(``scf.exxdiv_treatment``) the oracle energies were computed with (the Rust test must pass the same one). A case with a
``snapshot`` entry reuses the snapshot of that case (the snapshot does not depend on the divergence treatment).

The spinor cases (CS-3, ``lda-fr``/``pbe-fr`` datasets named by the case's ``paw_dataset``) are ``si_ldafr`` and ``si_pbefr``
(the displaced Si cell, time-reversal symmetric, 2x2x2 mesh), ``si_ldafr_tr3`` and ``si_ldafr_tr3_full`` (a 3x1x1 mesh, halved
by time reversal or kept whole), ``auh_ldafr`` (the heavy molecule whose spin-orbit force is resolved) and ``nne_ldafr`` (a
high-spin N atom next to Ne: the magnetic, non-time-reversal spinor, started along z; it needs ``density_tolerance`` 1e-6,
see ``nne_atoms``). The collinear spin-polarized cases (F1, ``beh_lda``, ``beh_pbe``: the BeH radical, ``h2_hse06``: the H2 triplet; the ``_b`` twins repeat a case with the second step family, ``spin=True``, a nonzero initial
moment) use the scalar-relativistic datasets. A case may override ``basis``, ``scf`` and ``width_ev``; an oracle energy must come from a converged SCF
(the freeze raises otherwise).

For each case (a small displaced crystal, full k mesh, no symmetry) the Python setup exports the crystal snapshot of
the undisplaced reference geometry (``<case>.snapshot.json``, the input of the Rust SCF) and the free energy of the
Rust SCF at displaced geometries (every one a fresh Python setup of the displaced crystal, solved with the oracle
tolerances). The manifest ``<case>.json`` stores the geometry, the discretization, the free energies and the
Richardson protocol of ``PAW_COMPLETION_PLAN.md`` 14.4 item 1:

    c_a = (F(+a) - F(-a)) / 2a,   R(a, b) = c_a + (c_a - c_b) / 3,   RR = R(2,4) + (R(2,4) - R(4,8)) / 15,

with the steps 0.002, 0.004 and 0.008 Angstrom, and the measured floors ``R(2,4) - R(4,8)`` and ``RR - R(2,4)``.
The displaced snapshots are NOT stored (they are as large as the reference one); the free energies are. The manifest
also carries what the Rust setup cannot yet derive from the snapshot and the dataset files alone: the radial tables of
the basis functions and the function list (the output of the Python basis generation; WP-M moves it to Rust), the
projector basis and the dataset file names. Units: Bohr, Hartree, Hartree/Bohr unless a name says Angstrom or eV.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from freeze_stamp import stamp  # noqa: E402
from freeze_wpx_eight import Recorder, enc, radial_tables  # noqa: E402

BASIS = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}
STEPS_A = (0.002, 0.004, 0.008)
#: The F1 cases (spin-polarized, 14.4: a floor measured with one family can be a 20x underestimate on a large force, so every
#: case has a twin with a second family): steps four times smaller than STEPS_A truncate at 1e-9 Ha/Bohr or less where the
#: 2/4/8 mA family truncates at 1e-7 (PBE on Li), and the 1e-11 Ha SCF noise over the step stays at 5e-9 Ha/Bohr (0.5 mA).
STEPS_FINE = (0.0005, 0.001, 0.002)
STEPS_FINE_B = (0.00075, 0.0015, 0.003)
#: Oracle tolerances: the free energy to 1e-11 Ha, the density to 1e-9 (14.4 item 1).
SCF = {"max_iterations": 400, "tolerance": 1e-11, "density_tolerance": 1e-9, "mixing": 0.25, "history": 7}
WIDTH_EV = 0.1
SI_A = 5.43


def si_atoms():
    """Diamond Si in a primitive cell whose lattice matrix is NOT symmetric (a circulant choice of the three
    face-centering vectors), the second atom off its ideal position by a generic vector."""
    from ase import Atoms
    a = SI_A
    cell = [[a / 2, a / 2, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2]]
    atoms = Atoms("Si2", cell=cell, pbc=True, positions=[[0, 0, 0], [a / 4, a / 4, a / 4]])
    atoms.positions[1] += [0.05, 0.02, -0.03]
    return atoms


def al_atoms():
    from ase.build import bulk
    atoms = bulk("Al", "fcc", a=4.05, orthorhombic=True)
    atoms.positions[1] += [0.05, 0.02, -0.03]
    return atoms


def lih_atoms():
    from ase import Atoms
    atoms = Atoms("LiH", cell=[[6.0, 0, 0], [0.4, 5.8, 0], [0.3, 0.2, 5.6]], pbc=True,
                  positions=[[2.0, 2.1, 2.2], [3.5, 2.2, 2.0]])
    return atoms


def auh_atoms():
    """AuH in a 6 Angstrom box, 1.9 Angstrom long (past the SZ-basis minimum near 1.78 Angstrom, scanned, so the bond
    carries a force small enough for the oracle to resolve) and tilted off every axis: a closed-shell, time-reversal-symmetric spinor whose spin-orbit force is large (Au 5d splits by about 1.5
    eV), the heavy case on which the spin-orbit term can be resolved (on Si it is 0.2 plan gates). The y and z coordinates of the
    Au atom are not frozen: its force along them is 0.2 Ha/Bohr on this coarse SZ grid (a mesh-bound force the hydrogen
    does not balance; x sits on a grid midpoint where it vanishes) and their oracle floors, 1e-5 and 4e-5, are above the
    plan gate."""
    from ase import Atoms
    return Atoms("AuH", cell=[6.0, 6.0, 6.0], pbc=True, positions=[[3.0, 3.1, 2.6], [3.172, 2.928, 4.481]])


def nne_atoms():
    """An N atom (high-spin, S = 3/2: 3 Bohr magnetons) with a Ne atom 2.6 Angstrom away, tilted off every axis, in an
    8 Angstrom box: a MAGNETIC spinor (not time-reversal symmetric) with the four field channels live. The N moment is
    robust (a half-filled p shell, a large exchange gap), unlike triplet Li2 (collapses to a non-magnetic state), triplet
    O2, PtH and hcp Co (their spinor SCFs do not reach the oracle tolerances). The spinor SCF of an atom in a box stalls
    above a density residual of about 5e-7 (the magnetization direction in the vacuum is ill-conditioned), so the
    density tolerance is 1e-6 while the free energy is converged to 1e-11 Ha. The converged moment is asserted by the
    Rust test, not assumed."""
    from ase import Atoms
    atoms = Atoms("NNe", cell=[8.0, 8.0, 8.0], pbc=True, positions=[[4.0, 4.2, 2.7], [4.26, 3.9, 5.3]])
    atoms.set_initial_magnetic_moments([3.0, 0.0])
    return atoms


def h2_atoms():
    """F1, the HSE06 cell: the H2 triplet (two H atoms 2.1 Angstrom apart, tilted off every axis, parallel initial moments of 1
    Bohr magneton each) in a 6.2 Angstrom box. Fully spin-polarized (the spin-down density is a tail), no partial core, so the
    spheres' spin-scaling remainder is the whole spin physics of the sphere. The converged moment is asserted by the Rust test."""
    from ase import Atoms
    atoms = Atoms("HH", cell=[6.2, 6.2, 6.2], pbc=True, positions=[[3.1, 3.3, 2.0], [3.62, 3.0, 4.0]])
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    return atoms


def beh_atoms():
    """F1: the BeH radical (a doublet: one unpaired electron, 1 Bohr magneton to start on Be) 1.33 Angstrom long, tilted off every
    axis, in a 6.2 Angstrom box, for the COLLINEAR spin-polarized crystal (``spin=True``, scalar-relativistic datasets). Be
    carries a large partial core, so the partial core meets a spin-split exchange-correlation potential. Cells tried and dropped,
    for the record: NH (the partial core of N makes the GGA and hybrid free energies rough at the 1e-9 Ha level: central
    differences are not monotone in the step and the step families disagree by 1e-6 Ha/Bohr; and the N core moves the force by
    1e-5 Ha/Bohr, too little for the spin-averaging control to bite 100 gates), NiH (the partial core of Ni moves the force by
    less than the gate), N next to Ne (a mesh-bound force of 0.1 Ha/Bohr on Ne with an oracle floor of 3e-7), the LiH triplet
    (its SCF collapses to zero moment). The converged moment is asserted by the Rust test."""
    from ase import Atoms
    atoms = Atoms("BeH", cell=[6.2, 6.2, 6.2], pbc=True, positions=[[3.1, 3.3, 2.0], [3.43, 3.0, 3.25]])
    atoms.set_initial_magnetic_moments([1.0, 0.0])
    return atoms


CASES = {
    "si": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True},
               moves=[(1, 0), (1, 1), (1, 2), (0, 1)]),
    "al2": dict(atoms=al_atoms, grid_spacing=0.45, kpts={"size": (2, 2, 2), "gamma": True},
                moves=[(1, 0), (1, 2), (0, 1)]),
    "lih": dict(atoms=lih_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True},
                moves=[(0, 0), (1, 1), (1, 2)]),
    # HSE06 (A2b / HX-W): an atom pair in a triclinic box at Gamma, then the displaced Si cell on the whole 2x2x2 mesh.
    "lih_hse06": dict(atoms=lih_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True},
                      moves=[(0, 0), (1, 1), (1, 2)], xc="hse06", exxdiv="none"),
    "si_hse06": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True},
                     moves=[(1, 0), (1, 1), (1, 2), (0, 1)], xc="hse06", exxdiv="none"),
    # The semilocal half of the hybrid on its own (PBE, the pbe-sr datasets): separates a gradient-correction
    # discrepancy from an exchange one.
    "si_pbe": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True},
                   moves=[(1, 0), (1, 1)], xc="pbe"),
    # CS-3: the spinor (spin-orbit) LDA crystal. Every point of the 2x2x2 Gamma-centered mesh is time-reversal invariant,
    # so nothing is halved: the crystal is a whole-mesh, time-reversal-symmetric spinor with identity moment symmetry.
    "si_ldafr": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True},
                     moves=[(1, 0), (1, 1), (1, 2), (0, 1)], paw_dataset="lda-fr"),
    # CS-3: the spinor GGA (PBE rotated to the local spin eigenframe), the pbe-fr datasets.
    "si_pbefr": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True},
                     moves=[(1, 0), (0, 1)], xc="pbe", paw_dataset="pbe-fr"),
    # CS-3 (half versus whole mesh): a 3x1x1 mesh whose points +k and -k differ. The natural build halves it by time
    # reversal (two points with weights 2/3 and 1/3, the moment symmetrization stays the identity); `full_mesh` keeps all three.
    "si_ldafr_tr3": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (3, 1, 1), "gamma": True},
                         moves=[(1, 0), (0, 1)], paw_dataset="lda-fr"),
    "si_ldafr_tr3_full": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (3, 1, 1), "gamma": True},
                              moves=[(1, 0), (0, 1)], paw_dataset="lda-fr", full_mesh=True),
    # CS-3: a magnetic (not time-reversal-symmetric) spinor with the four field channels live.
    "auh_ldafr": dict(atoms=auh_atoms, grid_spacing=0.35, kpts={"size": (1, 1, 1), "gamma": True},
                      moves=[(0, 0), (1, 0), (1, 1), (1, 2)], paw_dataset="lda-fr",
                      basis=dict(BASIS, filter_cutoff=250)),
    "nne_ldafr": dict(atoms=nne_atoms, grid_spacing=0.35, kpts={"size": (1, 1, 1), "gamma": True},
                      moves=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], paw_dataset="lda-fr",
                      basis=dict(BASIS, filter_cutoff=250), scf=dict(density_tolerance=1e-6)),
    # F1: collinear spin-polarized crystals (spins = 2), an open-shell diatomic in a box at Gamma: LDA, PBE (the
    # spin-resolved gradient correction) and HSE06 (K per spin and the spheres' spin-scaling remainder). Each case has a twin
    # ``_b`` with the second step family (the floors and the force must agree with the first).
    "beh_lda": dict(atoms=beh_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True},
                    moves=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], spin=True, steps=STEPS_FINE),
    "beh_lda_b": dict(atoms=beh_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True},
                      moves=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], spin=True, steps=STEPS_FINE_B, snapshot="beh_lda"),
    "beh_pbe": dict(atoms=beh_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True},
                    moves=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], xc="pbe", spin=True, steps=STEPS_FINE),
    "beh_pbe_b": dict(atoms=beh_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True},
                      moves=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], xc="pbe", spin=True, steps=STEPS_FINE_B, snapshot="beh_pbe"),
    # HSE06 on the H2 triplet: the hybrid free energy of the LiH triplet is not a single-valued function of the geometry (the
    # unrestricted SCF of four displaced geometries out of 36 converged, at the oracle tolerances, to a free energy 2e-5 to 5e-5 Ha
    # above the smooth trend of the others: central differences 6e-3 against 2.5e-3 Ha/Bohr), so no step family resolves it; the H2 triplet has none of that.
    "h2_hse06": dict(atoms=h2_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True},
                     moves=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], xc="hse06", exxdiv="none", spin=True, steps=STEPS_FINE),
    "h2_hse06_b": dict(atoms=h2_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True},
                       moves=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], xc="hse06", exxdiv="none", spin=True, steps=STEPS_FINE_B, snapshot="h2_hse06"),
    "si_hse06_gb": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True},
                        moves=[(1, 0), (0, 1)], xc="hse06", exxdiv="gygi-baldereschi", snapshot="si_hse06"),
}
DATASET_SETS = {"lda": "lda-sr", "pbe": "pbe-sr", "hse06": "pbe-sr"}


def calculator(spec, save=None):
    from mandacaru import Mandacaru
    options = dict(SCF, **spec.get("scf", {}), width_ha=spec.get("width_ev", WIDTH_EV) / 27.211386245988)
    hybrid = {"exxdiv_treatment": spec["exxdiv"]} if "exxdiv" in spec else {}
    library = {"paw_dataset": spec["paw_dataset"]} if "paw_dataset" in spec else {}
    if spec.get("spin"):
        library["spin"] = True
    return Mandacaru(method="dft", xc=spec.get("xc", "lda"), basis=spec.get("basis", BASIS), grid_spacing=spec["grid_spacing"],
                     kpts=spec["kpts"], smearing={"method": "fermi-dirac", "width": spec.get("width_ev", WIDTH_EV)},
                     periodic_scf_options=options, symmetry=False, save_crystal=save, **hybrid, **library)


def free_energy_ha(atoms, spec, save=None):
    from mandacaru._native import HARTREE_TO_EV
    atoms = atoms.copy()
    atoms.calc = calculator(spec, save)
    ev = atoms.get_potential_energy(force_consistent=True)
    if not atoms.calc.result.periodic.converged:
        raise RuntimeError("an oracle energy needs a converged SCF: this one stopped at the iteration limit")
    adapter = atoms.calc._crystal_adapter
    return ev / HARTREE_TO_EV, adapter, atoms.calc


def protocol(energies, steps_bohr):
    c = [(energies[(+1, i)] - energies[(-1, i)]) / (2 * steps_bohr[i]) for i in range(3)]
    r24 = c[0] + (c[0] - c[1]) / 3
    r48 = c[1] + (c[1] - c[2]) / 3
    rr = r24 + (r24 - r48) / 15
    return dict(central=c, r24=r24, r48=r48, rr=rr, floor_richardson=r24 - r48, floor_extrapolated=rr - r24)


def keep_whole_mesh():
    """The Python build halves the mesh of a time-reversal-symmetric spinor; this keeps every point (and no
    symmetry), as ``freeze_hybrid_spinor.py`` does for the spinor-hybrid fixtures."""
    from mandacaru.pseudopotentials import periodic_paw
    build = periodic_paw.build_crystal

    def whole(*args, **kwargs):
        kwargs.update(full_mesh=True, symmetry=False)
        return build(*args, **kwargs)

    periodic_paw.build_crystal = whole


def freeze(case, output):
    from mandacaru._native import BOHR_TO_ANGSTROM, HARTREE_TO_EV
    spec = CASES[case]
    if spec.get("full_mesh"):
        keep_whole_mesh()
    atoms = spec["atoms"]()
    to_bohr = 1.0 / BOHR_TO_ANGSTROM
    steps = spec.get("steps", STEPS_A)
    steps_bohr = [s * to_bohr for s in steps]
    output.mkdir(parents=True, exist_ok=True)
    shared = spec.get("snapshot")
    snapshot = output / f"{shared or case}.snapshot.json"
    start = time.time()
    base, adapter, calc = free_energy_ha(atoms, spec, save=None if shared else str(snapshot))
    dataset_set = spec.get("paw_dataset", DATASET_SETS[spec.get("xc", "lda")])
    rec = Recorder()
    radial_tables(rec, adapter)
    setup = dict(rec.inputs)
    setup["centers_bohr"] = [list(map(float, c)) for c in adapter.centers]
    setup["lattice_bohr"] = enc(adapter.lattice)
    setup["datasets"] = [{"symbol": d.symbol, "file": f"{dataset_set}/{d.symbol}.parquet"} for d in adapter.datasets]
    setup["projector_basis"] = str(adapter.projectors[0].projector_basis)
    setup["sigma_bohr"] = float(adapter.sigma)
    setup["projector_orbitals"] = [[int(p.atom_index), int(p.l), int(p.m), int(p.index)] for p in adapter.projectors]
    setup["orbital_atoms"] = [int(a) for a in adapter.atom_of_orbital]
    payload = json.loads(snapshot.read_text())
    if not shared:
        # A hybrid crystal reads the dataset files named in its snapshot: keep the library-relative names, not this
        # machine's absolute paths (the tests join them with MANDACARU_PAW_PATH).
        for entry, relative in zip(payload["sites"]["datasets"], setup["datasets"]):
            entry["file"] = relative["file"]
        snapshot.write_text(json.dumps(payload))
    mesh = payload["kpoints"]
    gc.collect()
    coordinates = []
    for atom, axis in spec["moves"]:
        energies = {}
        for index, step in enumerate(steps):
            for sign in (+1, -1):
                moved = atoms.copy()
                moved.positions[atom, axis] += sign * step
                energies[(sign, index)], _, _ = free_energy_ha(moved, spec)
                gc.collect()
        result = protocol(energies, steps_bohr)
        result.update(atom=atom, axis=axis, free_energies_ha={f"{'+' if s > 0 else '-'}{i}": v
                                                              for (s, i), v in energies.items()})
        coordinates.append(result)
        print(f"{case} atom {atom} axis {axis}: RR {result['rr']:.9e} Ha/Bohr, floors "
              f"{result['floor_richardson']:.2e} / {result['floor_extrapolated']:.2e} "
              f"({time.time() - start:.0f} s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB)",
              flush=True)
    ha_bohr_to_ev_a = HARTREE_TO_EV / BOHR_TO_ANGSTROM
    datasets = [str(Path(os.environ["MANDACARU_PAW_PATH"]) / d["file"]) for d in setup["datasets"]]
    manifest = dict(
        stamp=stamp(datasets=datasets, extra={"case": case, "generator": "test/migration/freeze_periodic_gradients.py"}),
        case=case, symbols=atoms.get_chemical_symbols(), cell_angstrom=atoms.cell.array.tolist(),
        positions_angstrom=atoms.positions.tolist(), grid_spacing_A=spec["grid_spacing"], kpts=spec["kpts"],
        k_points=len(mesh), basis=spec.get("basis", BASIS), functional=spec.get("xc", "lda"), symmetry=False,
        snapshot=shared or case,
        scf=dict(SCF, **spec.get("scf", {}), width_ev=spec.get("width_ev", WIDTH_EV),
                 width_ha=spec.get("width_ev", WIDTH_EV) / 27.211386245988, smearing="fermi-dirac",
                 **({"exxdiv_treatment": spec["exxdiv"]} if "exxdiv" in spec else {})),
        steps_angstrom=list(steps), steps_bohr=steps_bohr, free_energy_ha=base,
        coordinates=coordinates, ha_bohr_to_ev_per_angstrom=ha_bohr_to_ev_a,
        protocol="PAW_COMPLETION_PLAN.md 14.4 item 1: c_a, R(a,b) = c_a + (c_a - c_b)/3, RR = R(2,4) + (R(2,4) - R(4,8))/15; "
                 "the force is -RR; floors are R(2,4) - R(4,8) and RR - R(2,4)",
        setup=setup, seconds=time.time() - start,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    (output / f"{case}.json").write_text(json.dumps(manifest))
    print(f"{case}: manifest and snapshot written to {output}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("list", "case"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--output", type=Path, default=root / "reference" / "periodic_gradients")
    ns = parser.parse_args()
    for name in ("MANDACARU_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if ns.action == "list":
        print("\n".join(CASES))
    else:
        freeze(ns.name, ns.output)


if __name__ == "__main__":
    main()
