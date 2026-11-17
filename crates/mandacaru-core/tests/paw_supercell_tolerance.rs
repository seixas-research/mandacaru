//! The Cholesky tolerance of the PAW-LCAO supercell against the accuracy it needs
//! (`molecular::periodic::paw::SUPERCELL_CHOLESKY_TOLERANCE`). The pair matrix of the Lowdin
//! functions has no low-rank structure (its eigenvalues fall off smoothly), so the rank, and the
//! memory of the factors, follow the tolerance; the largest unused pair integral bounds each
//! integral and the energy per cell follows it to within a size-dependent factor. The tolerance
//! the supercell takes by default (the calculator's 1e-8 eV) keeps the RHF and the active-space
//! energies per cell within 1e-6 Hartree of a tight factorization on every case measured. Needs `MANDACARU_PAW_PATH`.
mod paw_supercell_support;
mod setup_shells_support;

use paw_supercell_support::{cells_with_basis, silicon_cell, Primitive};

use mandacaru_core::{
    molecular::{periodic::paw::SUPERCELL_CHOLESKY_TOLERANCE, MolecularIntegrals, ScfOptions},
    quantum::SolveOptions,
    Context,
};
use std::sync::Arc;

const REFERENCE: f64 = 1e-12;

/// What one factorization gave: the rank, the RHF energy per cell and (with `active`) the
/// correlation energy per cell of the 4-qubit space of the HOMO and the LUMO.
struct Run {
    rank: usize,
    rhf: f64,
    correlation: f64,
}

fn run(
    setup: &Arc<mandacaru_core::periodic::setup::PeriodicSetup>,
    mesh: [usize; 3],
    tolerance: Option<f64>,
    active: bool,
) -> Run {
    let cells = mesh.iter().product::<usize>() as f64;
    let ctx = Context::new(Some(2), 3 << 30).unwrap();
    let integrals =
        MolecularIntegrals::paw_supercell(&ctx, setup.clone(), mesh, tolerance).unwrap();
    let n = integrals.neutral_electrons();
    let options = ScfOptions {
        tolerance: 1e-11,
        ..ScfOptions::default()
    };
    let rhf = integrals
        .mean_field("rhf", [n / 2, n / 2], &options)
        .unwrap();
    let rank = integrals.eri().unwrap().report().cholesky.unwrap().0;
    let mut correlation = 0.;
    if active {
        let frozen: Vec<String> = (0..n / 2 - 1).map(|i| i.to_string()).collect();
        let spec = format!(
            r#"{{"orbitals": [{}, {}], "frozen": [{}]}}"#,
            n / 2 - 1,
            n / 2,
            frozen.join(", ")
        );
        let model = rhf.active_model(&integrals, &spec).unwrap();
        let problem = model
            .problem(&ctx, "jordan_wigner", true, false, true)
            .unwrap();
        let solved = problem
            .solve(
                &SolveOptions {
                    adapt: true,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        correlation = (solved.energy - solved.reference) / cells;
    }
    Run {
        rank,
        rhf: (rhf.electronic_energy + integrals.nuclear) / cells,
        correlation,
    }
}

fn silicon(basis: &str, mesh: [usize; 3]) -> Arc<mandacaru_core::periodic::setup::PeriodicSetup> {
    let ctx = Context::new(Some(2), 3 << 30).unwrap();
    let cell: Primitive = silicon_cell();
    Arc::new(cells_with_basis(
        &ctx, "lda-sr", false, &cell, 1, [0.; 3], basis,
    ))
}

/// Si 2 x 1 x 1 at SZ (`m = 16`, 136 pairs): measured against the reference 1e-12 (rank 136),
/// 1e-6 drops three pairs and moves the RHF energy per cell by 1.3e-7 and the correlation by
/// 8e-10 Hartree; 1e-5 drops 29 and moves them by 2.5e-5 and 4.6e-7: the energy follows the
/// tolerance, and the default is exact on this cell.
#[test]
fn the_energy_per_cell_follows_the_tolerance_on_silicon_sz() {
    let setup = silicon("SZ", [2, 1, 1]);
    let reference = run(&setup, [2, 1, 1], Some(REFERENCE), true);
    assert_eq!(reference.rank, 136);
    for (tolerance, rank_below, rhf, correlation) in [
        (None, 137, 1e-9, 1e-9),
        (Some(1e-6), 134, 1e-6, 1e-8),
        (Some(1e-5), 108, 1e-4, 1e-6),
    ] {
        let got = run(&setup, [2, 1, 1], tolerance, true);
        println!(
            "Si SZ 2x1x1 tolerance {tolerance:?}: rank {}, RHF/cell {:+.2e}, correlation/cell {:+.2e}",
            got.rank,
            got.rhf - reference.rhf,
            got.correlation - reference.correlation
        );
        assert!(got.rank < rank_below, "rank {}", got.rank);
        assert!(
            (got.rhf - reference.rhf).abs() < rhf,
            "RHF {:e}",
            got.rhf - reference.rhf
        );
        assert!(
            (got.correlation - reference.correlation).abs() < correlation,
            "correlation {:e}",
            got.correlation - reference.correlation
        );
    }
}

/// Si 2 x 2 x 2 at SZ (`m = 64`), the largest mesh the RHF reference runs: at 1e-8 Hartree the
/// rank was 62 % of the reference's 92 % of the pairs and the energy per cell moved by 1.5e-7
/// Hartree (about 15 times the tolerance here, 1.7e-6 at 1e-7 and 1.6e-5 at 1e-6); the default
/// (1e-8 eV) is tighter.
#[test]
#[ignore = "slow: about 4 min (two RHF runs on m = 64); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test paw_supercell_tolerance -- --ignored --nocapture"]
fn the_default_tolerance_holds_the_energy_per_cell_on_silicon_222() {
    let setup = silicon("SZ", [2, 2, 2]);
    let reference = run(&setup, [2, 2, 2], Some(1e-10), false);
    let got = run(&setup, [2, 2, 2], None, false);
    let pairs = 64 * 65 / 2;
    println!(
        "Si SZ 2x2x2: default {SUPERCELL_CHOLESKY_TOLERANCE:e}: rank {} of {pairs} (reference {}), RHF/cell {:+.2e}",
        got.rank,
        reference.rank,
        got.rhf - reference.rhf
    );
    assert!(
        (got.rhf - reference.rhf).abs() < 1e-6,
        "{:e}",
        got.rhf - reference.rhf
    );
}

/// Si 2 x 1 x 1 at DZP (`m = 52`, 1378 pairs), the production basis: 1e-8 Hartree kept 79 % of the
/// pairs (the reference 1e-12 keeps all) and moved the RHF energy per cell by 1.0e-9 and the
/// correlation by 1.8e-10 Hartree (the same at 1e-10: 4.5e-12); the default (1e-8 eV) is tighter.
#[test]
#[ignore = "slow: about 1 min (two RHF runs on m = 52); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test paw_supercell_tolerance -- --ignored --nocapture"]
fn the_default_tolerance_holds_the_energy_per_cell_on_silicon_dzp() {
    let setup = silicon("DZP", [2, 1, 1]);
    let reference = run(&setup, [2, 1, 1], Some(REFERENCE), true);
    let got = run(&setup, [2, 1, 1], None, true);
    println!(
        "Si DZP 2x1x1: default {SUPERCELL_CHOLESKY_TOLERANCE:e}: rank {} of 1378 (reference {}), RHF/cell {:+.2e}, correlation/cell {:+.2e}",
        got.rank,
        reference.rank,
        got.rhf - reference.rhf,
        got.correlation - reference.correlation
    );
    assert!(
        (got.rhf - reference.rhf).abs() < 1e-7,
        "{:e}",
        got.rhf - reference.rhf
    );
    assert!((got.correlation - reference.correlation).abs() < 1e-8);
}

/// Si 2 x 2 x 1 at DZP (`m = 104`, 5460 pairs), the largest production-basis mesh whose RHF
/// reference runs: the default against the tolerance 1e-10 (which keeps 72 % of the pairs).
#[test]
#[ignore = "slow: about 20 min (two factorizations and RHF runs on m = 104); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test paw_supercell_tolerance -- --ignored silicon_dzp_four --nocapture"]
fn the_default_tolerance_holds_the_energy_per_cell_on_silicon_dzp_four_cells() {
    let setup = silicon("DZP", [2, 2, 1]);
    let reference = run(&setup, [2, 2, 1], Some(1e-10), false);
    let got = run(&setup, [2, 2, 1], None, false);
    println!(
        "Si DZP 2x2x1: default {SUPERCELL_CHOLESKY_TOLERANCE:e}: rank {} of 5460 (1e-10: {}), RHF/cell {:+.2e}",
        got.rank,
        reference.rank,
        got.rhf - reference.rhf
    );
    assert!(
        (got.rhf - reference.rhf).abs() < 1e-6,
        "{:e}",
        got.rhf - reference.rhf
    );
}
