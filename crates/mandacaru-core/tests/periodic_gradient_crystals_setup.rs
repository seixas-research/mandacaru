//! CF-4, the consistency gate: the Rust setup built from the dataset files, the snapshot's sites
//! block and the basis sidecar describes the SAME crystal as the Python-built snapshot the Rust SCF
//! solves. Per k-point the Bloch sums, projections, overlap and fixed Hamiltonian; on the grid the
//! hats, ions, lattice sums, constants and the partial core. A force of this setup is then the
//! force of that energy. Also ties the setup's kinetic form to the kinetic matrix inside the
//! snapshot's `fixed` Hamiltonian (the SCF's own operator), reports its distance to the meta-GGA
//! tau operator, and ties `state_energy` to the SCF energy. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::{
    fields::kinetic_operator,
    gradient::scf,
    setup::{
        electrostatics::{couplings, hats, ion_charge},
        fields::{core, reference, Placed, Tables},
        ions::ion_constants,
        pulay::{kinetic_form, projection, Matrix, Mesh},
        shapes::{Sites, COMPENSATION_CUTOFF},
        shells::{DenseRows, Sampling},
    },
};
use num_complex::Complex64;
use periodic_gradient_crystals_support::*;

/// Relative gates: 100x the largest measured gap (8.1e-14 for the matrices, 3.4e-14 for the grid
/// quantities), rounded up.
const MATRIX_GATE: f64 = 1e-11;
const GRID_GATE: f64 = 1e-11;
/// The LDA crystals and the two HSE06 crystals (the `pbe-sr` datasets): the hybrid setup is the
/// semilocal one, and its snapshot is as consistent.
const CASES: [&str; 5] = ["si", "al2", "lih", "lih_hse06", "si_hse06"];

fn relative(got: &[Complex64], want: &[Complex64]) -> f64 {
    assert_eq!(got.len(), want.len(), "lengths differ");
    let scale = want.iter().map(|w| w.norm()).fold(0., f64::max).max(1e-300);
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).norm())
        .fold(0., f64::max)
        / scale
}

fn real(got: &[f64], want: &[f64]) -> f64 {
    let c = |v: &[f64]| v.iter().map(|x| Complex64::new(*x, 0.)).collect::<Vec<_>>();
    relative(&c(got), &c(want))
}

#[test]
#[ignore = "slow: 61 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_crystals_setup -- --ignored the_rust_setup_is_the_snapshot_crystal"]
fn the_rust_setup_is_the_snapshot_crystal() {
    for case in CASES {
        let data = Data::load(case);
        let c = &data.crystal;
        let ctx = &c.context;
        let (m, n, p) = (c.m, c.grid.index.points(), c.p);
        let mesh = Mesh::new(ctx, &c.grid).unwrap();
        let dv = c.grid.index.dv();
        let mut worst = [0f64; 4];
        for point in &c.points {
            let k = point.k;
            let mut psi = vec![Complex64::new(0., 0.); m * n];
            data.shells
                .values(
                    ctx,
                    &Sampling {
                        lattice: &data.lattice,
                        region: mesh.region(),
                        kpoints: &[k],
                    },
                    mesh.points(),
                    &mut DenseRows::new(&mut psi, m, n),
                )
                .unwrap();
            let snapshot = point.psi.read(ctx).unwrap();
            worst[0] = worst[0].max(relative(&psi, &snapshot));
            let cmat = projection(ctx, &data.shells, &data.lattice, k, &data.projectors).unwrap();
            worst[1] = worst[1].max(gap(&cmat, &point.projections));
            let basis = Matrix::from_row_slice(m, n, &psi);
            let grid_overlap = basis.conjugate() * basis.transpose() * Complex64::new(dv, 0.);
            let overlap = &grid_overlap + &cmat * &data.augmentation * cmat.adjoint();
            worst[2] = worst[2].max(gap(
                &((&overlap + overlap.adjoint()) * Complex64::new(0.5, 0.)),
                &point.overlap,
            ));
            let kinetic = kinetic_form(ctx, &c.grid, &mesh, k, (&psi, m), (&psi, m)).unwrap();
            let fixed = &kinetic
                + short_range_total(&data, k)
                + &cmat * &data.ion_operator * cmat.adjoint();
            worst[3] = worst[3].max(gap(
                &((&fixed + fixed.adjoint()) * Complex64::new(0.5, 0.)),
                &point.fixed,
            ));
        }
        eprintln!("{case}: psi, C, S, fixed relative gaps {worst:?} (p = {p})");
        for (name, w) in ["psi", "C", "S", "fixed"].iter().zip(worst) {
            assert!(w < MATRIX_GATE, "{case}: {name}: {w:e}");
        }
    }
}

/// Two Rust kinetic matrices exist: the setup's `kinetic_form` (the alias-averaged symbol
/// `|G + k|^2 / 2`, the kinetic matrix inside `fixed`) and `fields::kinetic_operator` (the
/// meta-GGA tau-potential operator, a product of two spectral gradients, here with a unit
/// potential, which does not average the squares of the aliases). The energy's kinetic matrix is
/// the first: it is tied to the snapshot by subtracting the short-range and projector parts from
/// the snapshot's `fixed` (the SCF's own operator), to the same gate as `fixed` itself. The
/// distance to the second is a report.
#[test]
#[ignore = "slow: 21 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_crystals_setup -- --ignored the_kinetic_form_is_the_kinetic_matrix_of_the_snapshot"]
fn the_kinetic_form_is_the_kinetic_matrix_of_the_snapshot() {
    for case in CASES {
        let data = Data::load(case);
        let c = &data.crystal;
        let ctx = &c.context;
        let (m, n) = (c.m, c.grid.index.points());
        let mesh = Mesh::new(ctx, &c.grid).unwrap();
        let (mut worst, mut tau_distance) = (0f64, 0f64);
        for point in &c.points {
            let psi = point.psi.read(ctx).unwrap();
            let kinetic = kinetic_form(ctx, &c.grid, &mesh, point.k, (&psi, m), (&psi, m)).unwrap();
            let rest = short_range_total(&data, point.k)
                + &point.projections * &data.ion_operator * point.projections.adjoint();
            let snapshot = &point.fixed - rest;
            worst = worst.max(gap(
                &((&kinetic + kinetic.adjoint()) * Complex64::new(0.5, 0.)),
                &((&snapshot + snapshot.adjoint()) * Complex64::new(0.5, 0.)),
            ));
            let tau = kinetic_operator(c, point, &vec![1.0; n]).unwrap();
            tau_distance = tau_distance.max(gap(
                &kinetic,
                &Matrix::from_column_slice(m, m, tau.as_slice()),
            ));
        }
        eprintln!(
            "{case}: kinetic_form vs snapshot fixed - rest {worst:e}; vs kinetic_operator(1) {tau_distance:e} (report)"
        );
        assert!(worst < MATRIX_GATE, "{case}: {worst:e}");
    }
}

#[test]
fn the_rust_setup_lattice_sums_are_the_snapshot_ones() {
    for case in CASES {
        let data = Data::load(case);
        let c = &data.crystal;
        let ctx = &c.context;
        let n = c.grid.index.points();
        let sites = data.sites();
        let channels = c.channels;
        let rust_hats = hats(ctx, &c.grid, &data.shapes, &sites).unwrap();
        let mut worst = [0f64; 7];
        for a in 0..channels {
            let table = c.hats.read_range(ctx, a * n, n).unwrap();
            worst[0] = worst[0].max(relative(&rust_hats.as_slice()[a * n..(a + 1) * n], &table));
        }
        worst[1] = relative(
            ion_charge(ctx, &c.grid, &sites).unwrap().as_slice(),
            c.ion.as_slice(),
        );
        let pair = couplings(ctx, &data.shapes, &sites, COMPENSATION_CUTOFF).unwrap();
        worst[2] = relative(pair.u.as_slice(), c.u.as_slice());
        worst[3] = relative(pair.u_ion.as_slice(), c.uion.as_slice());
        worst[4] = (pair.e_ion - c.eion).abs() / c.eion.abs()
            + (ion_constants(&sites).unwrap() - c.ion_constant).abs() / c.ion_constant.abs();
        worst[5] = (reference(&sites) - c.reference).abs() / c.reference.abs();
        let tables: Vec<Tables> = data
            .datasets
            .iter()
            .zip(data.cores())
            .map(|(ds, core)| Tables {
                radius: ds.r.as_slice(),
                valence_density: ds.valence_density.as_slice(),
                core: core.map(|c| c.values),
            })
            .collect();
        let placed: Vec<Placed> = data
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| Placed {
                species: i,
                center: a.center,
            })
            .collect();
        let rust_core = core(ctx, &c.grid, &tables, &placed, false).unwrap();
        worst[6] = real(rust_core.as_slice(), c.core.as_slice());
        eprintln!(
            "{case}: hats, ion, U, U_ion, E_ion+constant, reference, core relative gaps {worst:?}"
        );
        for (name, w) in [
            "hats",
            "ion",
            "U",
            "U_ion",
            "E_ion and ion constant",
            "reference",
            "core",
        ]
        .iter()
        .zip(worst)
        {
            assert!(w < GRID_GATE, "{case}: {name}: {w:e}");
        }
    }
}

#[test]
fn state_energy_is_the_scf_energy() {
    let data = Data::load("si");
    let result = data.solve();
    let energy = scf::state_energy(&data.crystal, &result.matrices).unwrap();
    eprintln!(
        "state_energy - energy = {:e} (energy {})",
        energy - result.energy,
        result.energy
    );
    assert!(
        (energy - result.energy).abs() < 1e-13,
        "{:e}",
        energy - result.energy
    );
    // The manifest's free energy is the SCF's: the oracle and the force differentiate one surface.
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    eprintln!(
        "SCF free energy - manifest = {:e}",
        result.free_energy - reference
    );
    assert!(
        (result.free_energy - reference).abs() < 1e-9,
        "{:e}",
        result.free_energy - reference
    );
}

/// The three frozen symmetric cells of the older fixtures cannot see a transposed lattice, and
/// neither can the Si cell here: a circulant fcc primitive matrix and its transpose generate the
/// same lattice (measured: the projections do not move, 2e-15). The triclinic LiH cell is another
/// lattice when transposed: the ion constant and the dense sums miss the snapshot by far more
/// than their gates.
#[test]
fn a_transposed_lattice_is_detected() {
    let data = Data::load("lih");
    let c = &data.crystal;
    let ctx = &c.context;
    let transposed = Sites {
        lattice: data.lattice.transpose(),
        ..data.sites()
    };
    let miss = (ion_constants(&transposed).unwrap() - c.ion_constant).abs() / c.ion_constant.abs();
    let right =
        (ion_constants(&data.sites()).unwrap() - c.ion_constant).abs() / c.ion_constant.abs();
    eprintln!("ion constant: lattice {right:e}, transposed lattice {miss:e}");
    assert!(right < GRID_GATE && miss > 100. * GRID_GATE);
    let pair = couplings(ctx, &data.shapes, &transposed, COMPENSATION_CUTOFF).unwrap();
    let gate = relative(pair.u.as_slice(), c.u.as_slice());
    eprintln!("compensation U, transposed lattice: {gate:e}");
    assert!(gate > 100. * GRID_GATE);
}
