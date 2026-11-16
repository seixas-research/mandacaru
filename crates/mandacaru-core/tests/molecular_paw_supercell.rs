//! Stages 4a and 4b of the Bloch methods, the PAW-LCAO supercell from the crystal setup
//! (`molecular::periodic::paw`, `BLOCH_VQE_PLAN.md` Stage 4).
//!
//! 4b: the pair-density engine on the supercell's Lowdin functions (orthonormal on the grid and
//! the projectors), gate S4-1 (the supercell Hartree-Fock energy of a 2 x 1 x 1 mesh equals the
//! doubled cell's at Gamma) and S4-5 (a rigid shift of the crystal on the grid).
//!
//! 4a: gate S4-2, the crystal setup's
//! `S(k)`, `fixed(k)` and projections `C(k)` recovered from the folded supercell blocks to 1e-12
//! at every point of the Gamma-centered mesh; the folded `S` and `fixed` real (time reversal; the
//! projections keep the complex harmonics of the projectors), the overlap symmetric positive
//! definite and a function of `R - R'` only, and a spin-orbit setup refused by name. Needs
//! `MANDACARU_PAW_PATH`.
mod paw_supercell_support;
mod setup_shells_support;

use paw_supercell_support::{
    cells_of, closed_shell_energy, hydrogen_chain_cell, lithium_hydride_cell, molecular_integrals, molecule_in_a_box,
    raw_atomic_orbitals,
};

use mandacaru_core::math::storage::MatrixSource;
use mandacaru_core::{
    molecular::{
        periodic::paw::{HamiltonianKind, PawSupercell},
        MolecularIntegrals, ScfOptions,
    },
    periodic::setup::PeriodicSetup,
    Context,
};
use std::sync::Arc;

/// LiH in a skew cell (`lda-sr`, SZ, raw projectors) on the Rust setup.
fn lithium_hydride(ctx: &Context, library: &str, spinor: bool) -> PeriodicSetup {
    cells_of(ctx, library, spinor, &lithium_hydride_cell(), 1, [0.; 3])
}

/// `cells_of` with a LiH primitive cell (the 4a gates).
fn cells_of_lithium_hydride(ctx: &Context, library: &str, spinor: bool, copies: usize) -> PeriodicSetup {
    cells_of(ctx, library, spinor, &lithium_hydride_cell(), copies, [0.; 3])
}

/// The RHF energy per mesh cell (Hartree) of the PAW supercell of `mesh` on `setup`, and its
/// Hamiltonian.
fn rhf_per_cell(ctx: &Context, setup: PeriodicSetup, mesh: [usize; 3]) -> (f64, MolecularIntegrals) {
    let integrals = MolecularIntegrals::paw_supercell(ctx, Arc::new(setup), mesh, None).unwrap();
    assert_eq!(integrals.kind(), HamiltonianKind::PawSupercell);
    let n = integrals.neutral_electrons();
    let options = ScfOptions {
        tolerance: 1e-11,
        ..ScfOptions::default()
    };
    let rhf = integrals.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
    let cells: usize = mesh.iter().product();
    ((rhf.electronic_energy + integrals.nuclear) / cells as f64, integrals)
}

/// The pair engine on the supercell's Lowdin functions: orthonormal through the grid overlap and
/// the projectors' overlap correction, real and eight-fold symmetric integrals, a positive
/// semidefinite pair matrix (the Cholesky accepts it).
#[test]
fn the_pair_engine_sees_the_lowdin_functions_orthonormal() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for mesh in [[1, 1, 1], [2, 1, 1]] {
        let setup = cells_of(&ctx, "lda-sr", false, &hydrogen_chain_cell(), 1, [0.; 3]);
        let integrals = MolecularIntegrals::paw_supercell(&ctx, Arc::new(setup), mesh, None).unwrap();
        let source = integrals.paw_supercell.as_ref().unwrap();
        let eri = integrals.eri().unwrap();
        let (imaginary, asymmetry) = source.columns.departures();
        println!(
            "H2 chain {mesh:?}: m {}, <phi|S|phi> - 1 {:.2e}, Im (pq|rs) {imaginary:.2e}, asymmetry {asymmetry:.2e}, Cholesky {:?}",
            integrals.m(),
            source.metric_error,
            eri.report().cholesky
        );
        assert!(source.metric_error < 1e-10, "{mesh:?}: {:e}", source.metric_error);
        assert!(imaginary < 1e-8 && asymmetry < 1e-8, "{mesh:?}");
    }
}

/// S4-1: the Born-von Karman identity. The supercell Hartree-Fock energy per cell at mesh
/// (2, 1, 1) equals the doubled cell's at (1, 1, 1), to 1e-9 Ha.
#[test]
fn a_two_cell_mesh_is_the_doubled_cell_at_gamma() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for (name, cell) in [("H2 chain", hydrogen_chain_cell()), ("LiH", lithium_hydride_cell())] {
        let (mesh, on_mesh) = rhf_per_cell(&ctx, cells_of(&ctx, "lda-sr", false, &cell, 1, [0.; 3]), [2, 1, 1]);
        let (doubled, on_doubled) = rhf_per_cell(&ctx, cells_of(&ctx, "lda-sr", false, &cell, 2, [0.; 3]), [1, 1, 1]);
        // The MP2 shift is v_M of the supercell's lattice (the doubled cell), not the primitive's:
        // the energy's constant 1/2 N_e v_M carries the same v_M.
        let v = on_mesh.exchange_madelung().unwrap();
        let from_constant = 2. * on_mesh.madelung / on_mesh.neutral_electrons() as f64;
        assert!((v - from_constant).abs() < 1e-10 && (v - on_doubled.exchange_madelung().unwrap()).abs() < 1e-10, "{name}: {v} {from_constant}");
        // The doubled cell holds two primitive cells.
        let doubled = doubled / 2.;
        let (gamma, _) = rhf_per_cell(&ctx, cells_of(&ctx, "lda-sr", false, &cell, 1, [0.; 3]), [1, 1, 1]);
        println!(
            "{name} RHF per cell: mesh (2,1,1) {mesh:.12}, doubled cell at Gamma {doubled:.12}, \
             difference {:.2e} Ha; Gamma only {gamma:.12}",
            mesh - doubled
        );
        assert!((mesh - doubled).abs() < 1e-9, "{name}: {:e}", mesh - doubled);
    }
}

/// S4-5: a rigid shift of the crystal on the grid changes the RHF energy per cell by the
/// placement residual of the point-sampled filtered basis (measured; the plan's 1e-6 Ha at a
/// grid spacing of 0.2 A, owner Q12: reported, as the crystal DFT does).
#[test]
fn a_rigid_shift_of_the_crystal_moves_the_energy_by_the_placement_residual() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let cell = hydrogen_chain_cell();
    let bohr = 1. / 0.529177210903;
    let (reference, _) = rhf_per_cell(&ctx, cells_of(&ctx, "lda-sr", false, &cell, 1, [0.; 3]), [2, 1, 1]);
    let mut worst = 0f64;
    for shift in [0.001, 0.013, 0.05, 0.137] {
        let moved = [shift * bohr, 0.6 * shift * bohr, -0.3 * shift * bohr];
        let (energy, _) = rhf_per_cell(&ctx, cells_of(&ctx, "lda-sr", false, &cell, 1, moved), [2, 1, 1]);
        println!("rigid shift {shift} A: E - E0 = {:.3e} Ha per cell", energy - reference);
        worst = worst.max((energy - reference).abs());
    }
    let spacing = (0..3).map(|a| cell.lattice.column(a).norm() / cell.shape[a] as f64).fold(0f64, f64::max);
    println!("largest placement residual {worst:.3e} Ha per cell at a grid spacing of {:.3} A", spacing / bohr);
    assert!(worst < 1e-6, "{worst:e}");
}

#[test]
fn the_supercell_blocks_unfold_to_the_crystal_matrices_at_every_mesh_point() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let setup = lithium_hydride(&ctx, "lda-sr", false);
    for mesh in [[2, 1, 1], [3, 1, 1], [2, 2, 1]] {
        let supercell = PawSupercell::new(&ctx, &setup, mesh).unwrap();
        assert_eq!(supercell.kind(), HamiltonianKind::PawSupercell);
        let count = supercell.n_cells();
        let (m, p) = (supercell.m, supercell.p);
        assert_eq!(supercell.overlap.shape(), (count * m, count * m));
        assert_eq!(supercell.projections.shape(), (count * m, count * p));
        let mut worst = [0f64; 3];
        let miss = |got: nalgebra::DMatrix<num_complex::Complex64>, want: &nalgebra::DMatrix<num_complex::Complex64>| {
            (&got - want).iter().fold(0f64, |a, v| a.max(v.norm()))
        };
        for (k, point) in supercell.kpoints.iter().zip(&supercell.points) {
            worst[0] = worst[0].max(miss(supercell.unfold(&supercell.overlap, *k).unwrap(), &point.overlap));
            worst[1] = worst[1].max(miss(supercell.unfold(&supercell.fixed, *k).unwrap(), &point.fixed));
            worst[2] = worst[2].max(miss(supercell.unfold(&supercell.projections, *k).unwrap(), &point.projections));
        }
        // The blocks depend on R - R' only: every cell's diagonal block is cell 0's.
        let shift = (1..count)
            .map(|r| {
                (supercell.overlap.view((r * m, r * m), (m, m)) - supercell.overlap.view((0, 0), (m, m))).abs().max()
            })
            .fold(0f64, f64::max);
        let symmetric = (&supercell.overlap - supercell.overlap.transpose()).abs().max();
        let lowest = supercell.overlap.clone().symmetric_eigen().eigenvalues.min();
        println!(
            "mesh {mesh:?} ({count} cells, m {m}, p {p}): unfold - matrices_at: S {:.2e}, fixed {:.2e}, C {:.2e}; translation {shift:.2e}, asymmetry {symmetric:.2e}, lowest S eigenvalue {lowest:.3e}",
            worst[0], worst[1], worst[2]
        );
        assert!(worst.iter().all(|w| *w < 1e-12), "{mesh:?}: {worst:?}");
        assert!(shift < 1e-12 && symmetric < 1e-12 && lowest > 0., "{mesh:?}");
    }
}

#[test]
fn a_spin_orbit_setup_is_refused_by_name() {
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let setup = lithium_hydride(&ctx, "lda-fr", true);
    let error = PawSupercell::new(&ctx, &setup, [2, 1, 1]).err().expect("a spinor setup is refused");
    assert!(error.to_string().contains("spin-orbit"), "{error}");
}


/// The fold's convention against an independent build: the 2 x 1 x 1 supercell's folded `S` and
/// `fixed` equal the doubled cell's own matrices at Gamma (atoms cell-major, the same spacing,
/// cutoff and ion width), block by block.
#[test]
fn the_folded_supercell_is_the_doubled_cell_at_gamma() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let supercell = PawSupercell::new(&ctx, &lithium_hydride(&ctx, "lda-sr", false), [2, 1, 1]).unwrap();
    let doubled = cells_of_lithium_hydride(&ctx, "lda-sr", false, 2);
    let gamma = &doubled.matrices_at(&ctx, &[[0.; 3]], &[1.]).unwrap()[0];
    assert_eq!(gamma.overlap.shape(), supercell.overlap.shape(), "the doubled cell's basis");
    let miss = |a: &nalgebra::DMatrix<f64>, b: &nalgebra::DMatrix<num_complex::Complex64>| {
        a.iter().zip(b.iter()).fold(0f64, |x, (u, v)| x.max((v - num_complex::Complex64::new(*u, 0.)).norm()))
    };
    let (s, f) = (miss(&supercell.overlap, &gamma.overlap), miss(&supercell.fixed, &gamma.fixed));
    let coupling = supercell.overlap.view((0, supercell.m), (supercell.m, supercell.m)).abs().max();
    println!("folded 2x1x1 - doubled cell at Gamma: S {s:.2e}, fixed {f:.2e} (largest cell-0/cell-1 overlap {coupling:.3e})");
    assert!(coupling > 1e-3, "the two cells do not overlap: the comparison would not see the phase");
    assert!(s < 1e-10 && f < 1e-10, "S {s:e}, fixed {f:e}");
}

/// The molecular PAW-LCAO RHF energy (Hartree) of the molecule of `setup` ([`molecular_integrals`]).
fn molecular_rhf(ctx: &Context, setup: &PeriodicSetup) -> f64 {
    let integrals = molecular_integrals(ctx, setup);
    let n = integrals.neutral_electrons();
    let options = ScfOptions {
        tolerance: 1e-11,
        ..ScfOptions::default()
    };
    let rhf = integrals.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
    rhf.electronic_energy + integrals.nuclear
}

/// The control of S4-3: the crystal's own LDA (the setup's SCF, Gamma) of a molecule in a box
/// against the molecular PAW-LCAO LDA on the same grid, datasets and tables. A gap here belongs to
/// the two formulations themselves (the crystal's and the molecule's electrostatics and one-center
/// constants), not to the supercell's Hartree-Fock assembly.
#[test]
#[ignore = "slow: a few minutes (six LDA crystals and molecules); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test molecular_paw_supercell -- --ignored the_crystal_and_the_molecule_agree_in_kohn_sham --nocapture"]
fn the_crystal_and_the_molecule_agree_in_kohn_sham_for_a_molecule_in_a_box() {
    use mandacaru_core::periodic::solver::Options;
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let bohr = 1. / 0.529177210903;
    let spacing = 0.2 * bohr;
    for (name, atoms) in [
        ("H2", vec![("H", [-0.7, 0., 0.]), ("H", [0.7, 0., 0.])]),
        ("LiH", vec![("Li", [-1.5, 0., 0.]), ("H", [1.5, 0., 0.])]),
        ("N2", vec![("N", [-1.04, 0., 0.]), ("N", [1.04, 0., 0.])]),
    ] {
        for side_angstrom in [10f64, 14.] {
            let nodes = (side_angstrom * bohr / spacing).round() as usize;
            let setup = molecule_in_a_box(&ctx, &atoms, nodes as f64 * spacing, nodes);
            let options = Options {
                tolerance: 1e-10,
                density_tolerance: 1e-8,
                max_iterations: 300,
                ..Default::default()
            };
            let crystal = setup.build(&ctx).unwrap().solve(&options).unwrap().energy;
            let molecule = molecular_integrals(&ctx, &setup);
            let n = molecule.neutral_electrons();
            let options = ScfOptions {
                tolerance: 1e-11,
                ..ScfOptions::default()
            };
            let lda = molecule.kohn_sham("lda", [n / 2, n / 2], false, &options).unwrap();
            let molecular = lda.electronic_energy + molecule.nuclear;
            println!(
                "{name} L {side_angstrom} A: crystal LDA {crystal:.10} molecule LDA {molecular:.10} difference {:.3e} Ha",
                crystal - molecular
            );
            // LiH keeps its dipole's L^-3 tail; the nonpolar molecules agree at 14 A.
            if name != "LiH" && side_angstrom > 12. {
                assert!((crystal - molecular).abs() < 1e-6, "{name}: {:e}", crystal - molecular);
            }
        }
    }
}

/// The RHF total energy (Hartree), the ADAPT-VQE total energy of the 4-qubit active space `active`
/// (the RHF energy plus the correlation the solver adds to its reference), the MP2 natural
/// occupations (occupied, then virtual) of the Hamiltonian `integrals`, and the levels of its MP2
/// denominators (the RHF levels, the occupied ones with `v_M` added on a supercell).
fn correlated(ctx: &Context, integrals: &MolecularIntegrals, active: &str) -> (f64, f64, Vec<f64>, Vec<f64>) {
    use mandacaru_core::quantum::SolveOptions;
    let n = integrals.neutral_electrons();
    let options = ScfOptions { tolerance: 1e-11, ..ScfOptions::default() };
    let rhf = integrals.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
    let total = rhf.electronic_energy + integrals.nuclear;
    let model = rhf.active_model(integrals, active).unwrap();
    assert_eq!(model.m, 2, "a 4-qubit active space");
    let problem = model.problem(ctx, "jordan_wigner", true, false, true).unwrap();
    let solved = problem.solve(&SolveOptions { adapt: true, ..Default::default() }, None).unwrap();
    let full = rhf.model(integrals).unwrap();
    let mp2 = mandacaru_core::molecular::correlation::natural_orbitals(ctx, full.m, full.one.as_slice(), full.two.as_slice(), n / 2, full.exchange_madelung).unwrap();
    let mut occupations = mp2.occupied_occupations.as_slice().to_vec();
    occupations.extend_from_slice(mp2.virtual_occupations.as_slice());
    let levels = mandacaru_core::molecular::correlation::denominator_levels(&rhf.energies[0], n / 2, integrals.exchange_madelung().unwrap());
    (total, total + solved.energy - solved.reference, occupations, levels)
}

/// Least squares of `d = a + b L^-3 + c L^-5` over `(L, d)` (the dipole and the quadrupole tails
/// of a molecule in a periodic box): `a`.
fn polar_tail_fit(rows: &[(f64, f64)]) -> f64 {
    let design = nalgebra::DMatrix::from_fn(rows.len(), 3, |i, j| rows[i].0.powi(-[0, 3, 5][j]));
    let values = nalgebra::DVector::from_iterator(rows.len(), rows.iter().map(|r| r.1));
    let normal = design.transpose() * &design;
    let solution = normal.lu().solve(&(design.transpose() * values)).unwrap();
    solution[0]
}

/// Least squares of `d = a + b L^-p` over `(L, d)`: `(a, b)`.
fn tail_fit(rows: &[(f64, f64)], power: i32) -> (f64, f64) {
    let (mut sx, mut sy, mut sxx, mut sxy) = (0., 0., 0., 0.);
    for (side, d) in rows {
        let x = side.powi(-power);
        sx += x;
        sy += d;
        sxx += x * x;
        sxy += x * d;
    }
    let n = rows.len() as f64;
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    ((sy - b * sx) / n, b)
}

/// S4-3, the molecular PAW limit: a molecule in a periodic cubic box of growing side, mesh
/// (1, 1, 1), against the molecular PAW-LCAO Hamiltonian on the same grid spacing, datasets and
/// radial tables. The supercell's difference to the molecule's converged value (its largest box;
/// the molecule's isolated grid converges from 12 A on) is fitted to `a + b L^-3` (measured: every
/// case falls as `L^-3`); the extrapolated `a` is the formulation gap between the crystal's
/// electrostatics (filtered shells, Bloch-style compensation, the periodic kernel and the Madelung
/// term) and the molecule's. Gated on every box and on the three largest (plan 4.7): the RHF
/// and the ADAPT-VQE energy of a 4-qubit active space below 1e-5 Ha. MP2: the supercell's
/// occupied levels carry `-v_M` (the exchange's dropped `G = 0`), which the MP2 denominators add
/// back (`MolecularIntegrals::exchange_madelung`); gated (vera, 2026-11-15): every level of the
/// denominators falls toward the molecule's as `L^-3` (log-log slope -3 +- 0.1 between the two
/// largest boxes; a wrong sign, a factor 1/2 or a shift on the virtuals leave a 1/L term), and
/// the natural occupations meet the plan's 1e-6 extrapolated by both windows. Every fit takes
/// only the boxes in the asymptotic regime (vera's ruling, 2026-11-15): a box is admitted when
/// the molecule in that box is at its own limit (its largest box), the RHF and VQE energies
/// within 1e-6 Ha and every MP2 natural occupation within 1e-7, and each case needs at least
/// five admitted boxes. LiH's 10 A box is shorter than the reach of the Li 2s function (table
/// support 15.16 Bohr against a half-box of 9.45 Bohr), so the molecule in it is not at its
/// limit either (4.9e-5 Ha) and it leaves the fits (measured by vera with it dropped: occupations
/// -7.4e-7 on every box, -4.2e-7 on the three largest).
#[test]
#[ignore = "slow: about 20 min (1153 s measured 2026-11-15; N2 on six boxes the longest); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test molecular_paw_supercell -- --ignored the_supercell_of_a_molecule_in_a_box_reaches_the_molecular_energy --nocapture"]
fn the_supercell_of_a_molecule_in_a_box_reaches_the_molecular_energy() {
    let ctx = Context::new(Some(4), 1800 << 20).unwrap();
    let bohr = 1. / 0.529177210903;
    let spacing = 0.2 * bohr;
    // The difference falls as L^-3 for every case (H2: (d(L_i) - d(L_j)) / (L_i^-3 - L_j^-3)
    // constant to 2 % over the four intervals of the first run, where L^-5 and 1/L do not fit).
    // The active spaces do not cut a degenerate pair (N2: the HOMO sigma and the sigma*, the pi*
    // pair left out, the four lower orbitals frozen).
    // The polar LiH carries two more boxes: its dipole's tail makes the fit window-dependent.
    // N2's occupations sit at the edge of 1e-6 on 10-16 A with a pure L^-3 fit (vera), so it carries
    // the 18 and 20 A boxes too.
    let cases: [(&str, Vec<(&'static str, [f64; 3])>, &str, Vec<f64>); 3] = [
        ("H2", vec![("H", [-0.7, 0., 0.]), ("H", [0.7, 0., 0.])], r#"{"orbitals": [0, 1]}"#, vec![10., 12., 14., 16., 18., 20.]),
        ("LiH", vec![("Li", [-1.5, 0., 0.]), ("H", [1.5, 0., 0.])], r#"{"orbitals": [0, 1]}"#, vec![10., 12., 14., 16., 18., 20.]),
        ("N2", vec![("N", [-1.04, 0., 0.]), ("N", [1.04, 0., 0.])], r#"{"orbitals": [4, 7], "frozen": [0, 1, 2, 3]}"#, vec![10., 12., 14., 16., 18., 20.]),
    ];
    let mut failures = Vec::new();
    for (name, atoms, active, boxes) in cases {
        let mut sides = Vec::new();
        let (mut crystals, mut molecules) = (Vec::new(), Vec::new());
        for &side_angstrom in &boxes {
            let nodes = (side_angstrom * bohr / spacing).round() as usize;
            let side = nodes as f64 * spacing;
            let setup = molecule_in_a_box(&ctx, &atoms, side, nodes);
            let molecule = correlated(&ctx, &molecular_integrals(&ctx, &setup), active);
            let integrals = MolecularIntegrals::paw_supercell(&ctx, Arc::new(setup), [1, 1, 1], None).unwrap();
            assert_eq!(integrals.kind(), HamiltonianKind::PawSupercell);
            let crystal = correlated(&ctx, &integrals, active);
            println!(
                "{name} L {side_angstrom} A ({nodes}^3): RHF supercell {:.10} molecule {:.10} ({:.3e}); VQE supercell {:.10} molecule {:.10} ({:.3e}) Ha",
                crystal.0, molecule.0, crystal.0 - molecule.0, crystal.1, molecule.1, crystal.1 - molecule.1
            );
            sides.push(side);
            crystals.push(crystal);
            molecules.push(molecule);
        }
        let limit = molecules.last().unwrap().clone();
        // The admission rule: the molecule in the box at its own limit.
        let departures: Vec<(f64, f64)> = molecules
            .iter()
            .map(|e| {
                let energy = (e.0 - limit.0).abs().max((e.1 - limit.1).abs());
                let occupation = e.2.iter().zip(&limit.2).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
                (energy, occupation)
            })
            .collect();
        let admitted: Vec<usize> = (0..boxes.len()).filter(|&i| departures[i].0 < 1e-6 && departures[i].1 < 1e-7).collect();
        println!(
            "{name}: molecule against its largest box (energy, occupation) {:?}; admitted boxes {:?} A",
            departures.iter().map(|(e, o)| format!("({e:.1e}, {o:.1e})")).collect::<Vec<_>>(),
            admitted.iter().map(|&i| boxes[i]).collect::<Vec<_>>()
        );
        let gap = |value: &dyn Fn(&(f64, f64, Vec<f64>, Vec<f64>)) -> f64, power: i32| {
            let rows: Vec<(f64, f64)> = admitted.iter().map(|&i| (sides[i], value(&crystals[i]) - value(&limit))).collect();
            // a + b L^-3 + c L^-5 on every admitted box and a + b L^-3 on the three largest, for
            // the energies and the occupations alike.
            let (large, slope) = tail_fit(&rows[rows.len() - 3..], power);
            let all = if power == 3 { polar_tail_fit(&rows) } else { tail_fit(&rows, power).0 };
            (all, large, slope)
        };
        let rhf = gap(&|e| e.0, 3);
        let vqe = gap(&|e| e.1, 3);
        // The levels of the MP2 denominators against the molecule's in the same box: L^-3.
        let (last, previous) = (crystals.len() - 1, crystals.len() - 2);
        let slopes: Vec<f64> = (0..limit.3.len())
            .map(|p| {
                let d = |i: usize| (crystals[i].3[p] - molecules[i].3[p]).abs();
                (d(last) / d(previous)).ln() / (sides[last] / sides[previous]).ln()
            })
            .collect();
        let level_gaps: Vec<String> = (0..limit.3.len())
            .map(|p| format!("{:+.3e}", crystals[0].3[p] - molecules[0].3[p]))
            .collect();
        println!(
            "{name}: denominator levels, supercell (occupied + v_M) - molecule at {} A {level_gaps:?}; log-log slopes between the two largest boxes {:?}",
            boxes[0],
            slopes.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
        );
        let mut occupations = (0f64, 0f64);
        for index in 0..limit.2.len() {
            let (all, large, _) = gap(&|e| e.2[index], 3);
            occupations = (occupations.0.max(all.abs()), occupations.1.max(large.abs()));
        }
        let raw: Vec<f64> = crystals
            .iter()
            .map(|c| c.2.iter().zip(&limit.2).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max))
            .collect();
        println!("{name}: largest MP2 occupation difference per box {:?}", raw.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>());
        println!(
            "{name}: {} boxes admitted; extrapolated gaps (a + b L^-3 + c L^-5 on every admitted box; a + b L^-3 on the three largest): RHF {:.3e} ({:.3e}, slope {:.3e} Ha Bohr^3), VQE {:.3e} ({:.3e}) Ha, MP2 natural occupations {:.3e} ({:.3e}); molecule's occupations {:?}",
            admitted.len(), rhf.0, rhf.1, rhf.2, vqe.0, vqe.1, occupations.0, occupations.1, limit.2
        );
        // Every case is measured before the gate is judged (each variant reported).
        if admitted.len() < 5 {
            failures.push(format!("{name}: {} boxes in the asymptotic regime, five needed", admitted.len()));
        }
        if rhf.0.abs() >= 1e-5 || rhf.1.abs() >= 1e-5 {
            failures.push(format!("{name}: RHF {rhf:?}"));
        }
        if vqe.0.abs() >= 1e-5 || vqe.1.abs() >= 1e-5 {
            failures.push(format!("{name}: VQE {vqe:?}"));
        }
        if !slopes.iter().all(|s| (s + 3.).abs() < 0.1) {
            failures.push(format!("{name}: level slopes {slopes:?}"));
        }
        if occupations.0 >= 1e-6 || occupations.1 >= 1e-6 {
            failures.push(format!("{name}: MP2 occupations (every box, three largest) {occupations:?}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The audit of the one-center Hartree-Fock operator (S4-3's N2 gap): the molecule's
/// `Delta W . reference` (from its operator, `-onsite - xc + core_exchange`) against the crystal's
/// remainder contracted with the same reference, per channel at `m = 0`.
#[test]
fn the_one_center_reference_hartree_is_the_molecules() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    for symbol in ["H", "Li", "N"] {
        let setup = molecule_in_a_box(&ctx, &[(symbol, [0., 0., 0.])], 12., 30);
        let crystal = setup.build(&ctx).unwrap();
        let ours = mandacaru_core::periodic::wannier::downfold::basis::reference_hartree(&crystal).unwrap();
        let channels = mandacaru_core::paw::molecular::hartree_fock_onsite_channels(&ctx, &setup.datasets[0]).unwrap();
        let mut worst = 0f64;
        for ch in &channels {
            let rows: Vec<usize> = (0..setup.projectors.len())
                .filter(|&j| setup.projectors[j].l == ch.l && setup.projectors[j].m == 0)
                .collect();
            for (i, &a) in rows.iter().enumerate() {
                for (j, &b) in rows.iter().enumerate() {
                    let theirs = -ch.onsite[(i, j)] - ch.xc[(i, j)] + ch.core_exchange[(i, j)];
                    worst = worst.max((ours[(a, b)].re - theirs).abs());
                    println!(
                        "{symbol} l {} ({i},{j}): dW.ref crystal {:+.8e} molecule {theirs:+.8e}; core exchange {:+.3e}",
                        ch.l, ours[(a, b)].re, ch.core_exchange[(i, j)]
                    );
                }
            }
        }
        println!("{symbol}: largest dW.ref difference {worst:.3e} Ha");
        assert!(worst < 1e-8, "{symbol}: {worst:e}");
    }
}

/// The pair engine on p functions (the crystal's complex harmonics turned real, `L > 0`
/// compensation charges): N2 in a small box, its Lowdin functions orthonormal and its integrals real
/// and symmetric, and its energy against the molecule's at the same box, printed.
#[test]
fn the_pair_engine_sees_p_functions_orthonormal_and_symmetric() {
    let ctx = Context::new(Some(4), 1 << 30).unwrap();
    let bohr = 1. / 0.529177210903;
    let spacing = 0.25 * bohr;
    let nodes = (8f64 * bohr / spacing).round() as usize;
    // Along x, then along z: the energy of a free molecule does not depend on its orientation;
    // the closed-shell atoms Be (s only) and Ne (s and p) separate the one-center terms from the
    // bond.
    for (axis, atoms) in [
        ("N2 along x", vec![("N", [-1.04, 0., 0.]), ("N", [1.04, 0., 0.])]),
        ("N2 along z", vec![("N", [0., 0., -1.04]), ("N", [0., 0., 1.04])]),
        ("Be", vec![("Be", [0., 0., 0.])]),
        ("Ne", vec![("Ne", [0., 0., 0.])]),
    ] {
        let setup = molecule_in_a_box(&ctx, &atoms, nodes as f64 * spacing, nodes);
        let molecule = molecular_rhf(&ctx, &setup);
        let (crystal, integrals) = rhf_per_cell(&ctx, setup, [1, 1, 1]);
        let source = integrals.paw_supercell.as_ref().unwrap();
        let (imaginary, asymmetry) = source.columns.departures();
        println!(
            "{axis}, 8 A box ({nodes}^3): m {}, <phi|S|phi> - 1 {:.2e}, Im (pq|rs) {imaginary:.2e}, asymmetry {asymmetry:.2e}; supercell {crystal:.10} molecule {molecule:.10}",
            integrals.m(),
            source.metric_error
        );
        assert!(source.metric_error < 1e-10 && imaginary < 1e-8 && asymmetry < 1e-8);
    }
}

/// The audit of the two-body integrals: at the crystal's own converged LDA density of a molecule
/// in a box (Gamma), the Hartree energy of the supercell's pair integrals,
/// `1/2 sum D_ab D_cd (ab|cd)` in its Lowdin frame, against the crystal's electrostatics
/// (`solver::hartree_energy`), which differ by the one-center remainder the crystal's frozen
/// one-center treatment leaves out: printed for H2 and N2.
#[test]
fn the_pair_integrals_carry_the_crystals_hartree_energy() {
    use mandacaru_core::periodic::solver::{hartree_energy, Options};
    let ctx = Context::new(Some(4), 1 << 30).unwrap();
    let bohr = 1. / 0.529177210903;
    let spacing = 0.25 * bohr;
    let nodes = (8f64 * bohr / spacing).round() as usize;
    for (name, atoms) in [
        ("H2", vec![("H", [-0.7, 0., 0.]), ("H", [0.7, 0., 0.])]),
        ("N2", vec![("N", [-1.04, 0., 0.]), ("N", [1.04, 0., 0.])]),
    ] {
        let setup = molecule_in_a_box(&ctx, &atoms, nodes as f64 * spacing, nodes);
        let crystal = setup.build(&ctx).unwrap();
        let options = Options { tolerance: 1e-10, density_tolerance: 1e-8, max_iterations: 300, ..Default::default() };
        let result = crystal.solve(&options).unwrap();
        let ours = hartree_energy(&crystal, &result.matrices).unwrap();
        let integrals = MolecularIntegrals::paw_supercell(&ctx, Arc::new(setup), [1, 1, 1], Some(1e-12)).unwrap();
        let cells = PawSupercell::new(&ctx, integrals.paw_supercell.as_ref().unwrap().setup.as_ref(), [1, 1, 1]).unwrap();
        let d = result.matrices[0].matrix(&ctx).unwrap();
        // The crystal rows to the supercell's real rows, then the Lowdin frame.
        let real = cells.harmonics.adjoint() * &*d * &cells.harmonics;
        let m = integrals.m();
        let lowdin = &integrals.lowdin;
        let inverse = lowdin.clone().try_inverse().unwrap();
        let frame = &inverse * real.map(|v| v.re) * &inverse;
        let eri = integrals.eri().unwrap();
        let mut pairs = 0.;
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    for e in 0..m {
                        pairs += 0.5 * frame[(a, b)] * frame[(c, e)] * eri.value(a, c, b, e);
                    }
                }
            }
        }
        // The one-center remainder the pair integrals add: 1/2 sum_A W_abcd P_ab P_cd with
        // P = C^T D C^* on the projectors of atom A (`pairs::onsite`).
        let c = &crystal.points[0].projections;
        let projected = c.transpose() * &*d * c.map(|v| v.conj());
        let mut onsite = 0.;
        let setup = integrals.paw_supercell.as_ref().unwrap().setup.clone();
        for atom in 0..setup.centers.len() {
            let own: Vec<usize> = (0..setup.projectors.len()).filter(|&j| setup.projectors[j].atom == atom).collect();
            let projectors: Vec<mandacaru_core::paw::onecenter::Projector> = own
                .iter()
                .map(|&j| mandacaru_core::paw::onecenter::Projector {
                    l: setup.projectors[j].l,
                    m: setup.projectors[j].m,
                    index: setup.projectors[j].index,
                    real: false,
                })
                .collect();
            let w = mandacaru_core::periodic::setup::remainder::onsite_remainder(
                &ctx,
                &setup.datasets[setup.atoms_species[atom]],
                mandacaru_core::paw::onecenter::Basis::Raw,
                &projectors,
            )
            .unwrap();
            let n = own.len();
            for a in 0..n {
                for b in 0..n {
                    for x in 0..n {
                        for y in 0..n {
                            onsite += 0.5
                                * (w.data[((a * n + b) * n + x) * n + y]
                                    * projected[(own[a], own[b])]
                                    * projected[(own[x], own[y])])
                                    .re;
                        }
                    }
                }
            }
        }
        println!(
            "{name}: crystal Hartree {ours:.10}, pair integrals {pairs:.10} (one-center {onsite:.10}), pairs - one-center - crystal {:.3e} Ha; Im D {:.2e}",
            pairs - onsite - ours,
            real.iter().fold(0f64, |x, v| x.max(v.im.abs()))
        );
        // Measured: 1.0e-7 (H2), 2.5e-5 Ha of 17.5 (N2), the grid sphere of the compensation.
        assert!((pairs - onsite - ours).abs() < 1e-4, "{name}: {:e}", pairs - onsite - ours);
    }
}

/// S4-3's audit at one density: N2 and H2 in an 8 A box, the supercell and the molecular PAW-LCAO
/// Hamiltonians in the raw atomic orbitals both share. The overlaps agree; the one-body operators
/// and the pair integrals agree once the shifts of the periodic kernel's dropped `G = 0` are taken
/// out (about `a S_ab` and `v S_ab S_cd`); each route's energy at the other's RHF density lies
/// above its own RHF energy (each SCF reached the minimum), and the two energies of one density
/// differ by the box's finite-size term only. The N2 gap of S4-3 (-0.667 Ha) was the molecule's
/// cold-start RHF on a saddle (Hessian root -0.325 Ha, the pi levels split), fixed by the SCF's
/// stability following: the cold start now equals the molecule's SCF seeded with the
/// supercell's orbitals.
#[test]
fn the_supercell_and_the_molecule_agree_at_one_density_and_each_reaches_its_minimum() {
    let ctx = Context::new(Some(4), 1 << 30).unwrap();
    let bohr = 1. / 0.529177210903;
    let spacing = 0.25 * bohr;
    let nodes = (8f64 * bohr / spacing).round() as usize;
    for (name, atoms) in [
        ("H2", vec![("H", [-0.7, 0., 0.]), ("H", [0.7, 0., 0.])]),
        ("N2", vec![("N", [-1.04, 0., 0.]), ("N", [1.04, 0., 0.])]),
    ] {
        let setup = molecule_in_a_box(&ctx, &atoms, nodes as f64 * spacing, nodes);
        let molecule = molecular_integrals(&ctx, &setup);
        let crystal = MolecularIntegrals::paw_supercell(&ctx, Arc::new(setup), [1, 1, 1], Some(1e-12)).unwrap();
        let m = molecule.m();
        assert_eq!(m, crystal.m());
        let ao = raw_atomic_orbitals;
        let (s_m, h_m, e_m) = ao(&molecule);
        let (s_c, h_c, e_c) = ao(&crystal);
        let overlap_gap = (&s_c - &s_m).amax();
        // The shifts of the dropped G = 0, fitted, and what is left of each operator.
        let (mut num, mut den) = (0., 0.);
        for ab in 0..m * m {
            for cd in 0..m * m {
                let w = s_m[(ab / m, ab % m)] * s_m[(cd / m, cd % m)];
                num += w * (e_c[ab * m * m + cd] - e_m[ab * m * m + cd]);
                den += w * w;
            }
        }
        let v = num / den;
        let alpha = (&h_c - &h_m).dot(&s_m) / s_m.dot(&s_m);
        let one_body_gap = (&h_c - &h_m - &s_m * alpha).amax();
        let pair_gap = (0..m * m * m * m)
            .map(|i| (e_c[i] - e_m[i] - v * s_m[(i / (m * m * m), (i / (m * m)) % m)] * s_m[((i / m) % m, i % m)]).abs())
            .fold(0f64, f64::max);
        println!(
            "{name}: S gap {overlap_gap:.2e}; shifts a {alpha:.6}, v {v:.6} Ha (Madelung of the box {:.6}); one-body left {one_body_gap:.2e}, pair integrals left {pair_gap:.2e} Ha",
            2. * crystal.madelung / crystal.neutral_electrons() as f64
        );
        assert!(overlap_gap < 1e-7, "{name}: {overlap_gap:e}");
        // The periodic images of the pair densities' dipoles and of the ions' field (L^-3).
        assert!(one_body_gap < 2e-2 && pair_gap < 5e-3, "{name}: {one_body_gap:e} {pair_gap:e}");
        let n = molecule.neutral_electrons();
        let options = ScfOptions { tolerance: 1e-11, ..ScfOptions::default() };
        let own = crystal.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
        let rhf = molecule.mean_field("rhf", [n / 2, n / 2], &options).unwrap();
        let single = ScfOptions { guesses: 1, ..options.clone() };
        let cold = molecule.mean_field("rhf", [n / 2, n / 2], &single).unwrap();
        // The oracle: the molecule's SCF seeded with the supercell's orbitals (the Lowdin frames
        // agree to the overlap gap).
        let seeded = molecule.continue_mean_field(&own, &options).unwrap();
        let roots = [molecule.stability(&rhf).unwrap().unwrap(), crystal.stability(&own).unwrap().unwrap()];
        println!(
            "{name}: molecule cold start {:.10} (one guess {:.10}), seeded from the supercell {:.10} Ha; lowest Hessian roots: molecule {:.4}, supercell {:.4}",
            rhf.electronic_energy, cold.electronic_energy, seeded.electronic_energy, roots[0], roots[1]
        );
        assert!(rhf.converged && cold.converged && seeded.converged);
        assert!((rhf.electronic_energy - seeded.electronic_energy).abs() < 1e-8, "{name}");
        assert!((cold.electronic_energy - seeded.electronic_energy).abs() < 1e-8, "{name}");
        assert!(roots.iter().all(|r| *r > 0.), "{name}: {roots:?}");
        // Each route's energy at both RHF densities (raw AOs, constants included).
        let total = closed_shell_energy;
        let ao_density = |state: &mandacaru_core::molecular::MeanField, frame: &nalgebra::DMatrix<f64>| {
            frame * state.densities[0].map(|v| v.re) * frame.transpose() * 2.
        };
        let (p_m, p_c) = (ao_density(&rhf, &molecule.lowdin), ao_density(&own, &crystal.lowdin));
        let molecular = [total(&h_m, &e_m, molecule.nuclear, &p_m), total(&h_m, &e_m, molecule.nuclear, &p_c)];
        let periodic = [total(&h_c, &e_c, crystal.nuclear, &p_m), total(&h_c, &e_c, crystal.nuclear, &p_c)];
        println!(
            "{name}: at the molecule's density: molecule {:.10}, supercell {:.10}; at the supercell's: molecule {:.10}, supercell {:.10}; supercell - molecule {:.4e} and {:.4e} Ha",
            molecular[0], periodic[0], molecular[1], periodic[1], periodic[0] - molecular[0], periodic[1] - molecular[1]
        );
        assert!((molecular[0] - rhf.electronic_energy - molecule.nuclear).abs() < 1e-8);
        assert!((periodic[1] - own.electronic_energy - crystal.nuclear).abs() < 1e-8);
        // The other route's density is idempotent in this route's metric only to the overlap
        // gap (H2: 1.6e-8; its energy then lies 2e-8 Ha below), hence 1e-6 (the saddle was 0.67 Ha).
        assert!(molecular[1] > molecular[0] - 1e-6 && periodic[0] > periodic[1] - 1e-6, "{name}: a route's own SCF is not its minimum");
        // The 8 A box's finite-size term (S4-3 extrapolates it away).
        assert!((periodic[0] - molecular[0]).abs() < 2e-2, "{name}");
    }
}

/// The analytic variational force dispatches on the Hamiltonian's kind (Bloch-VQE stage 6a): a
/// PAW-LCAO supercell's force is the derivative of the energy per cell, one row of three per
/// primitive atom (the image average folded in Rust), and rigid internal directions are refused by
/// name (rotations are not a symmetry of a crystal).
#[test]
fn the_analytic_force_of_a_paw_supercell_is_per_cell_and_cartesian() {
    use mandacaru_core::molecular::response;
    use mandacaru_core::quantum::SolveOptions;
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let setup = cells_of(&ctx, "lda-sr", false, &hydrogen_chain_cell(), 1, [0.; 3]);
    let atoms = setup.centers.len();
    let (_, integrals) = rhf_per_cell(&ctx, setup, [2, 1, 1]);
    let reference = integrals.mean_field("rhf", [2, 2], &Default::default()).unwrap();
    let model = reference.model(&integrals).unwrap();
    let problem = model.problem(&ctx, "parity_reduced", true, false, true).unwrap();
    let solved = problem.solve(&SolveOptions { adapt: true, ..Default::default() }, None).unwrap();
    let state = solved.state.complex128().unwrap();
    let gradient = response::gradient(&integrals, &reference, &model, &problem, state, &response::Options::default()).unwrap();
    println!("H2 chain (2,1,1) ADAPT: per-cell gradient {:?} Ha/Bohr ({} directions)", gradient.gradient.as_slice(), gradient.directions);
    assert_eq!(gradient.gradient.as_slice().len(), 3 * atoms);
    assert_eq!(gradient.directions, 3 * atoms);
    let error = response::gradient(&integrals, &reference, &model, &problem, state, &response::Options { internal: true, ..Default::default() })
        .err()
        .expect("internal directions are refused on a crystal");
    assert!(matches!(error, mandacaru_core::Error::Unsupported(_)), "{error}");
    assert!(error.to_string().contains("PAW-LCAO supercell"), "{error}");
}
