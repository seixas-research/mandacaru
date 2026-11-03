"""Freeze the reference of the analytic crystal stress on a real SCF crystal (AS-6, input of ``periodic_stress_crystal*``).

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set::

    python test/migration/freeze_periodic_strain.py list
    python test/migration/freeze_periodic_strain.py pins lih
    python test/migration/freeze_periodic_strain.py case lih [--output DIR] [--steps 0.0005 0.001 0.002]

AS-8 adds the spinor (spin-orbit) LDA cases, whose datasets are named by the case's ``paw_dataset`` (``lda-fr``): ``si_ldafr``
(the displaced Si cell, a 2x2x2 Gamma-centered mesh, every point time-reversal invariant, so nothing is halved),
``si_ldafr_tr3`` (a 3x1x1 mesh that the build halves by time reversal: two points, weights 2/3 and 1/3) and ``auh_fr`` (AuH in a
6 Angstrom box: a closed-shell, time-reversal spinor with a large spin-orbit term).

The cases are small scalar LDA (and one PBE) crystals on the whole k mesh (no symmetry): ``lih`` (a triclinic cell, one k-point),
``si`` (the displaced two-atom fcc cell of the force freeze, a 2x2x2 Gamma-centered mesh), ``si_ideal`` (the same cell
undisplaced: cubic), ``al`` (the one-atom fcc
primitive cell: a metal, Fermi-Dirac occupations on a 3x3x3 mesh, cubic, so its stress is proportional to the identity).

For every strain entry ``eps_ab`` (all nine, one at a time, UNSYMMETRIZED: ``x' = (1 + eps) x`` for the lattice vectors and the
atoms, fractional coordinates fixed) and for the isotropic series ``eps = e I``, the free energy of the Rust SCF is taken at
``+-e`` for each step ``e`` and the derivative is the Richardson extrapolation of ``PAW_COMPLETION_PLAN.md`` 14.4 item 1:

    c_e = (F(+e) - F(-e)) / 2e,   R(a, b) = c_a + (c_a - c_b) / 3,   RR = R(1, 2) + (R(1, 2) - R(2, 4)) / 15,

with the three steps ``e1 < e2 < e3 = 2 e2 = 4 e1``. The oracle is ``+dF/d eps_ab`` in Hartree (the analytic ``Stress::gradient``
has the same sign); no minus sign enters. The pins of ``ANALYTIC_STRESS_PLAN.md`` section 2 are held at the unstrained cell's value
through the options of the Python path, and asserted for every cell before its SCF runs:

* the node counts: an explicit periodic grid whose per-axis spacing is ``|a_m'| / n_m`` (the node count of the unstrained grid);
* the Gaussian ion width ``sigma``: ``local_split.split_width`` returns the unstrained value;
* the Fourier filter cutoff: the basis names an absolute energy (``filter_cutoff`` in eV), so it is a number, asserted
  through ``context["filter_wavevector"]``;
* the grid origin: the grid is centered on ``cell.sum(0) / 2``, so node 0 sits at the origin and the origin scales with the cell;
* the dense compensation sums: ``dense_coulomb`` sums over the integer triples ``n`` of the UNSTRAINED reciprocal ball
  (``|G| <= 15``), at the strained vectors ``n . B'``, never re-enumerating the list;
* the alias operation set: the number of operations of ``reciprocal.lattice_operations`` of every strained grid equals the
  unstrained one (a step that collapses the set raises);
* the k-points are fractional (a Monkhorst-Pack mesh), the occupations are Fermi-Dirac of a fixed width, the spheres are rigid
  (their quadrature is regenerated about the carried atoms).

The manifest ``<case>.json`` and the unstrained snapshot ``<case>.snapshot.json`` are written to ``reference/periodic_strain/``
in the shape the force freeze uses (the Rust loader of the stress tests reads both); the strained snapshots are NOT stored.
Units: Bohr, Hartree, Hartree per unit strain unless a name says otherwise.
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
from freeze_periodic_gradients import BASIS, SCF, auh_atoms, h2_atoms, keep_whole_mesh, lih_atoms, nne_atoms, si_atoms  # noqa: E402

STEPS = (0.0005, 0.001, 0.002)
WIDTH_EV = 0.1
ENTRIES = [(a, b) for a in range(3) for b in range(3)]
HARTREE_EV = 27.211386245988


def al_primitive():
    """One Al atom in the fcc primitive cell: cubic symmetry, so the stress is proportional to the identity."""
    from ase.build import bulk
    return bulk("Al", "fcc", a=4.05)


def au_primitive():
    """One Au atom in the fcc primitive cell (4.08 Angstrom): the heavy metal whose spin-orbit term the stress resolves (AS-8)."""
    from ase.build import bulk
    return bulk("Au", "fcc", a=4.08)


def si_ideal():
    """Diamond Si in the same circulant primitive cell with NO displacement: cubic symmetry, the isotropy case of Si."""
    from ase import Atoms
    a = 5.43
    cell = [[a / 2, a / 2, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2]]
    return Atoms("Si2", cell=cell, pbc=True, positions=[[0, 0, 0], [a / 4, a / 4, a / 4]])


CASES = {
    "lih": dict(atoms=lih_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True}),
    "si": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True}),
    "si_ideal": dict(atoms=si_ideal, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True}),
    "al": dict(atoms=al_primitive, grid_spacing=0.45, kpts={"size": (3, 3, 3), "gamma": True}, width_ev=0.3),
    # The same Al cell on a 2x1x1 mesh (two k-points) with a hot 2 eV smearing: on two k-points the SZ bands are gapped (about
    # 7 eV at the second k-point), and 1 eV leaves every occupation within 1e-3 of 0 or 1; 2 eV puts 15 percent on the
    # neighbors of the gap, so the Mermin occupations are really partial. The cheap metal, for the tests that evaluate the
    # stress twice.
    "al_small": dict(atoms=al_primitive, grid_spacing=0.45, kpts={"size": (2, 1, 1), "gamma": True}, width_ev=2.0),
    # The gradient correction on a real state (X-3 with the production gradients and flux): the pbe-sr datasets.
    "lih_pbe": dict(atoms=lih_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True}, xc="pbe"),
    # AS-9: HSE06 (the pbe-sr datasets), the scalar collinear hybrid on the whole mesh. ``exxdiv`` is the divergence treatment of the
    # exact exchange the oracle energies are computed with (the Rust test passes the same one); a ``_gb`` case shares the snapshot
    # of its ``none`` twin (the unstrained crystal does not depend on the treatment).
    "lih_hse06": dict(atoms=lih_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True}, xc="hse06", exxdiv="none"),
    "lih_hse06_gb": dict(atoms=lih_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True}, xc="hse06",
                         exxdiv="gygi-baldereschi", snapshot="lih_hse06"),
    "si_hse06": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True}, xc="hse06", exxdiv="none"),
    "si_hse06_gb": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True}, xc="hse06",
                        exxdiv="gygi-baldereschi", snapshot="si_hse06"),
    # AS-8: the spinor LDA crystals (spin-orbit coupling, time-reversal symmetric, no moment).
    "si_ldafr": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (2, 2, 2), "gamma": True}, paw_dataset="lda-fr"),
    "si_ldafr_tr3": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (3, 1, 1), "gamma": True}, paw_dataset="lda-fr"),
    # The heavy molecule whose spin-orbit term the stress resolves (on Si it is a small share): a coarser grid than the
    # force freeze (0.35), so a Python process stays near 1 GB over the 54 SCFs.
    "auh_fr": dict(atoms=auh_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True}, paw_dataset="lda-fr",
                   basis=dict(BASIS, filter_cutoff=250)),
    # The heavy metal: Au fcc, one atom, a 2x1x1 mesh (two time-reversal-invariant points) at 1 eV smearing. The cell is cubic but
    # the mesh is not, so the shear entries do not vanish (no isotropy gate).
    "au_fcc": dict(atoms=au_primitive, grid_spacing=0.4, kpts={"size": (2, 1, 1), "gamma": True}, paw_dataset="lda-fr", width_ev=1.0,
                   basis=dict(BASIS, filter_cutoff=250)),
    # AS-10: HSE06 WITH spin-orbit coupling, the time-reversal-symmetric spinor crystal on the WHOLE 3x1x1 mesh (a hybrid needs it:
    # ``full_mesh``; the points 0 and +-1/3, so the exchange runs at complex k), the ``pbe-fr`` datasets.
    "si_hsefr_tr3": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (3, 1, 1), "gamma": True}, xc="hse06", exxdiv="none",
                         paw_dataset="pbe-fr", full_mesh=True),
    # The scalar twin of ``si_hsefr_tr3`` (pbe-sr datasets, no spin-orbit): the same whole 3x1x1 mesh with complex pair wave vectors,
    # to tell a defect of the hybrid at complex k from one of the spinor frame.
    "si_hse06_tr3": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (3, 1, 1), "gamma": True}, xc="hse06", exxdiv="none",
                         full_mesh=True),
    # AS-10: the collinear spin-polarized HSE06 energy of the H2 triplet (``spin=True``, the pbe-sr datasets) is the oracle of the
    # MAGNETIC spinor hybrid stress with a zero spin-orbit matrix: the discretized energy is invariant under a global spin rotation,
    # so the unrestricted free energy is the spinor's whatever the direction of the moment (F2b).
    "h2_hse06": dict(atoms=h2_atoms, grid_spacing=0.45, kpts={"size": (1, 1, 1), "gamma": True}, xc="hse06", exxdiv="none", spin=True),
    # The spinor GGA (PBE rotated to the local spin eigenframe), time-reversal symmetric, two k-points.
    "si_pbefr_tr3": dict(atoms=si_atoms, grid_spacing=0.5, kpts={"size": (3, 1, 1), "gamma": True}, xc="pbe", paw_dataset="pbe-fr"),
    # The magnetic spinor (not time-reversal symmetric, the four Pauli channels live): a high-spin N atom next to Ne, started along
    # z with the spin-orbit coupling on; the spinor SCF stalls above a density residual of 5e-7, so the density tolerance is 1e-6.
    "nne_pbefr": dict(atoms=nne_atoms, grid_spacing=0.35, kpts={"size": (1, 1, 1), "gamma": True}, xc="pbe", paw_dataset="pbe-fr",
                      basis=dict(BASIS, filter_cutoff=250), scf=dict(density_tolerance=1e-6)),
    "nne_fr": dict(atoms=nne_atoms, grid_spacing=0.35, kpts={"size": (1, 1, 1), "gamma": True}, paw_dataset="lda-fr",
                   basis=dict(BASIS, filter_cutoff=250), scf=dict(density_tolerance=1e-6)),
}
DATASET_SETS = {"lda": "lda-sr", "pbe": "pbe-sr", "hse06": "pbe-sr"}


def strained(atoms, eps):
    """``x' = (1 + eps) x`` for the cell vectors (rows) and the atoms; fractional coordinates are unchanged."""
    import numpy as np
    t = (np.eye(3) + np.asarray(eps)).T
    moved = atoms.copy()
    moved.set_cell(np.asarray(atoms.cell.array) @ t, scale_atoms=False)
    moved.set_positions(np.asarray(atoms.positions) @ t)
    return moved


def entry_strain(a, b, e):
    import numpy as np
    eps = np.zeros((3, 3))
    eps[a, b] = e
    return eps


def pinned_grid(cell, shape):
    """The periodic grid of ``cell`` with exactly ``shape`` nodes: the per-axis spacing is ``|a_m| / n_m`` (Angstrom)."""
    import numpy as np
    from mandacaru.integrals import Grid
    lengths = np.linalg.norm(np.asarray(cell), axis=1)
    grid = Grid(center=0.5 * np.asarray(cell).sum(axis=0), box_size=0.0, grid_spacing=list(lengths / np.asarray(shape)),
                units="angstrom", cell=np.asarray(cell), periodic=True)
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(f"the pinned grid has {grid.shape} nodes, not {tuple(shape)}")
    return grid


def reference_grid(atoms, spec):
    import numpy as np
    from mandacaru.integrals import Grid
    cell = np.asarray(atoms.cell.array)
    return Grid(center=0.5 * cell.sum(axis=0), box_size=0.0, grid_spacing=spec["grid_spacing"], units="angstrom",
                cell=cell, periodic=True)


class Pins:
    """The held values of the unstrained cell, installed as patches of the Python setup for the freeze only."""

    def __init__(self, atoms, spec, unpin=()):
        import numpy as np
        from mandacaru.integrals import reciprocal as rc
        from mandacaru.pseudopotentials import local_split, periodic_paw
        self.atoms, self.spec = atoms, spec
        #: pins released on purpose, for the negative controls of the oracle (``--unpin``): the node counts, the ion width, the dense list
        self.unpin = tuple(unpin)
        grid = reference_grid(atoms, spec)
        self.shape = tuple(int(n) for n in grid.shape)
        self.operations = len(rc.lattice_operations(grid))
        self.sigma = float(local_split.split_width(grid))
        lattice = rc.lattice_vectors(grid)
        B = rc.reciprocal_vectors(lattice)
        radius = periodic_paw.COMPENSATION_CUTOFF
        vectors = rc.lattice_translations(B, radius)
        self.triples = np.rint(vectors @ np.linalg.inv(B.T)).astype(int)
        assert np.allclose(self.triples @ B.T, vectors, atol=1e-9), "the integer triples do not reproduce the ball"
        self.filter_wavevector = None
        self._split_width = local_split.split_width
        self._dense = periodic_paw.PeriodicPAW.dense_coulomb
        self._rc = rc
        self._radius = radius
        self.dense_sizes = []

    def install(self):
        import numpy as np
        from mandacaru.pseudopotentials import local_split, periodic_paw
        rc, triples, radius = self._rc, self.triples, self._radius
        sigma = self.sigma
        if "sigma" not in self.unpin:
            local_split.split_width = lambda grid: sigma
        original, sizes = self._dense, self.dense_sizes

        def dense(crystal):
            plain = rc.lattice_translations

            def pinned(B, r):
                if r == radius:
                    return triples @ np.asarray(B).T
                return plain(B, r)

            rc.lattice_translations = pinned
            try:
                sizes.append(len(triples))
                return original(crystal)
            finally:
                rc.lattice_translations = plain

        if "dense" not in self.unpin:
            periodic_paw.PeriodicPAW.dense_coulomb = dense

    def check(self, adapter, context, grid):
        from mandacaru.integrals import reciprocal as rc
        if "shape" not in self.unpin:
            assert tuple(grid.shape) == self.shape, "node counts"
        assert max(abs(float(grid.X.flat[0])), abs(float(grid.Y.flat[0])), abs(float(grid.Z.flat[0]))) < 1e-9, \
            "the grid origin must stay at the cell origin (it scales with the cell)"
        if "sigma" not in self.unpin:
            assert adapter.sigma == self.sigma, "ion width"
        assert len(rc.lattice_operations(grid)) == self.operations, "alias operation set collapsed under the strain"
        k_c = float(context["filter_wavevector"])
        if self.filter_wavevector is None:
            self.filter_wavevector = k_c
        assert k_c == self.filter_wavevector, "filter cutoff"


def calculator(spec, grid, save=None):
    from mandacaru import Mandacaru
    options = dict(SCF, **spec.get("scf", {}), width_ha=spec.get("width_ev", WIDTH_EV) / HARTREE_EV)
    library = {"paw_dataset": spec["paw_dataset"]} if "paw_dataset" in spec else {}
    library.update({"exxdiv_treatment": spec["exxdiv"]} if "exxdiv" in spec else {})
    library.update({"spin": True} if spec.get("spin") else {})
    return Mandacaru(method="dft", xc=spec.get("xc", "lda"), basis=spec.get("basis", BASIS), grid_spacing=spec["grid_spacing"], grid=grid,
                     kpts=spec["kpts"], smearing={"method": "fermi-dirac", "width": spec.get("width_ev", WIDTH_EV)},
                     periodic_scf_options=options, symmetry=False, save_crystal=save, **library)


def free_energy_ha(atoms, spec, pins, save=None):
    """The free energy (Hartree) of the Rust SCF at ``atoms`` with every pin asserted; also the adapter."""
    from mandacaru._native import HARTREE_TO_EV
    grid = (reference_grid(atoms, spec) if "shape" in pins.unpin else pinned_grid(atoms.cell.array, pins.shape))
    atoms = atoms.copy()
    atoms.calc = calculator(spec, grid, save)
    ev = atoms.get_potential_energy(force_consistent=True)
    if not atoms.calc.result.periodic.converged:
        raise RuntimeError("an oracle energy needs a converged SCF: this one stopped at the iteration limit")
    pins.check(atoms.calc._crystal_adapter, atoms.calc._crystal_metadata, grid)
    return ev / HARTREE_TO_EV, atoms.calc._crystal_adapter, atoms.calc


def protocol(energies, steps):
    """``energies[(sign, i)]`` at ``steps[i]``: the central differences and the Richardson ladder."""
    c = [(energies[(+1, i)] - energies[(-1, i)]) / (2 * steps[i]) for i in range(3)]
    r12 = c[0] + (c[0] - c[1]) / 3
    r24 = c[1] + (c[1] - c[2]) / 3
    rr = r12 + (r12 - r24) / 15
    return dict(central=c, r12=r12, r24=r24, rr=rr, floor_richardson=r12 - r24, floor_extrapolated=rr - r12)


def sweep(atoms, spec, pins, steps, eps_of, label, start):
    energies = {}
    for index, step in enumerate(steps):
        for sign in (+1, -1):
            energies[(sign, index)], _, _ = free_energy_ha(strained(atoms, eps_of(sign * step)), spec, pins)
            gc.collect()
    result = protocol(energies, steps)
    result["free_energies_ha"] = {f"{'+' if s > 0 else '-'}{i}": v for (s, i), v in energies.items()}
    print(f"{label}: RR {result['rr']:.9e} Ha, floors {result['floor_richardson']:.2e} / "
          f"{result['floor_extrapolated']:.2e} ({time.time() - start:.0f} s, "
          f"peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB)", flush=True)
    return result


def volume_a3(atoms):
    import numpy as np
    return abs(float(np.linalg.det(np.asarray(atoms.cell.array))))


def freeze(case, output, steps, only=None, tag=None, unpin=()):
    import numpy as np
    from mandacaru._native import BOHR_TO_ANGSTROM
    spec = CASES[case]
    if spec.get("full_mesh"):
        keep_whole_mesh()
    atoms = spec["atoms"]()
    output.mkdir(parents=True, exist_ok=True)
    pins = Pins(atoms, spec, unpin)
    pins.install()
    owner = spec.get("snapshot", case)
    snapshot = output / f"{owner}.snapshot.json"
    start = time.time()
    # A second step family (``--tag b``) shares the snapshot of the first and does not rewrite it.
    base, adapter, calc = free_energy_ha(atoms, spec, pins, save=None if tag or owner != case else str(snapshot))
    print(f"{case}: pins shape {pins.shape}, sigma {pins.sigma:.6f}, filter {pins.filter_wavevector:.6f}, "
          f"operations {pins.operations}, dense triples {len(pins.triples)}, free energy {base:.12f} Ha "
          f"({time.time() - start:.0f} s)", flush=True)
    if only == "pins":
        return
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
    if not tag and owner == case:
        for entry, relative in zip(payload["sites"]["datasets"], setup["datasets"]):
            entry["file"] = relative["file"]
        snapshot.write_text(json.dumps(payload))
    mesh = payload["kpoints"]
    gc.collect()
    entries = []
    for a, b in ENTRIES:
        result = sweep(atoms, spec, pins, steps, lambda e, a=a, b=b: entry_strain(a, b, e), f"{case} eps[{a}][{b}]", start)
        result.update(a=a, b=b)
        entries.append(result)
    iso = sweep(atoms, spec, pins, steps, lambda e: e * np.eye(3), f"{case} isotropic", start)
    datasets = [str(Path(os.environ["MANDACARU_PAW_PATH"]) / d["file"]) for d in setup["datasets"]]
    manifest = dict(
        stamp=stamp(datasets=datasets, extra={"case": case, "generator": "test/migration/freeze_periodic_strain.py"}),
        case=case, symbols=atoms.get_chemical_symbols(), cell_angstrom=atoms.cell.array.tolist(),
        positions_angstrom=atoms.positions.tolist(), grid_spacing_A=spec["grid_spacing"], kpts=spec["kpts"],
        k_points=len(mesh), basis=spec.get("basis", BASIS), functional=spec.get("xc", "lda"), symmetry=False, snapshot=owner,
        scf=dict(SCF, **spec.get("scf", {}), width_ev=spec.get("width_ev", WIDTH_EV),
                 width_ha=spec.get("width_ev", WIDTH_EV) / HARTREE_EV, smearing="fermi-dirac",
                 **({"exxdiv_treatment": spec["exxdiv"]} if "exxdiv" in spec else {})),
        pins=dict(shape=list(pins.shape), sigma_bohr=pins.sigma, filter_wavevector=pins.filter_wavevector,
                  operations=pins.operations, dense_triples=len(pins.triples), dense_cutoff=15.0, unpinned=list(unpin)),
        steps=list(steps), free_energy_ha=base, volume_bohr3=volume_a3(atoms) / BOHR_TO_ANGSTROM ** 3,
        entries=entries, isotropic=iso,
        protocol="c_e = (F(+e) - F(-e))/2e; R(a,b) = c_a + (c_a - c_b)/3; RR = R(1,2) + (R(1,2) - R(2,4))/15; "
                 "g_ab = +dF/d eps_ab (Hartree), one entry at a time, unsymmetrized; floors are R(1,2) - R(2,4) and RR - R(1,2)",
        setup=setup, seconds=time.time() - start,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    (output / f"{case}{'_' + tag if tag else ''}.json").write_text(json.dumps(manifest))
    print(f"{case}: manifest and snapshot written to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("list", "pins", "case"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--output", type=Path, default=HERE / "reference" / "periodic_strain")
    parser.add_argument("--steps", type=float, nargs=3, default=STEPS)
    parser.add_argument("--tolerance", type=float, nargs=2, default=None, metavar=("ENERGY", "DENSITY"),
                        help="override the oracle SCF tolerances (Ha, density); default 1e-11 / 1e-9")
    parser.add_argument("--unpin", nargs="*", default=(), choices=("dense", "sigma", "shape"),
                        help="release a pin on purpose (the negative controls of the oracle); use with --tag")
    parser.add_argument("--tag", default=None, help="a second step family: writes <case>_<tag>.json, keeps the snapshot")
    ns = parser.parse_args()
    for name in ("MANDACARU_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if ns.tolerance:
        SCF["tolerance"], SCF["density_tolerance"] = ns.tolerance
    if ns.action == "list":
        print("\n".join(CASES))
    else:
        freeze(ns.name, ns.output, tuple(ns.steps), only="pins" if ns.action == "pins" else None, tag=ns.tag,
               unpin=tuple(ns.unpin))


if __name__ == "__main__":
    main()
