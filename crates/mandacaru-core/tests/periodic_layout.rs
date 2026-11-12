//! The crystal SCF on every processor layout: its spin channel and k-point tasks run on `kpts`
//! lanes and its grid density is split by range, and the result (free energy, iteration count,
//! eigenvalues, occupations, density, moments) is the same, bit for bit, on every layout and every
//! processor count. Unpolarized, spin-polarized and spinor snapshots, crystals built by the Rust
//! setup on grids of several density tiles (LDA, spin-polarized, r2SCAN: the meta-GGA kinetic
//! operator of a task runs its own transforms), DFT+U, and k-point tasks reading stacks archived
//! on disk; and the refusal of more lanes than tasks. The built crystals need `MANDACARU_PAW_PATH`.
mod hubbard_support;
mod setup_shells_support;

use mandacaru_core::{
    math::storage::{ComplexSamples, Precision},
    models::hubbard_request::Request,
    periodic::{
        setup::parse::{create_setup, AtomSpec, SetupInput},
        solver::Options,
        wannier::sampling,
        Crystal, PeriodicResult,
    },
    Context, Error,
};
use nalgebra::{Matrix3, Vector3};
use std::sync::Arc;

const HARTREE_EV: f64 = 27.211386245988;

/// Every number of a result as its bit pattern.
#[derive(Debug, PartialEq)]
struct Bits {
    free: u64,
    iterations: usize,
    eigenvalues: Vec<u64>,
    occupations: Vec<u64>,
    density: Vec<u64>,
    moments: Vec<(u64, u64)>,
}
fn bits(state: &PeriodicResult) -> Bits {
    Bits {
        free: state.free_energy.to_bits(),
        iterations: state.history.len(),
        eigenvalues: state
            .eigenvalues
            .iter()
            .flatten()
            .map(|e| e.to_bits())
            .collect(),
        occupations: state
            .occupations
            .iter()
            .flatten()
            .map(|f| f.to_bits())
            .collect(),
        density: state
            .density
            .as_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect(),
        moments: state
            .moments
            .as_slice()
            .iter()
            .map(|q| (q.re.to_bits(), q.im.to_bits()))
            .collect(),
    }
}

/// `(kpts, threads)` layouts: equal products first, then other processor counts.
fn context(kpts: usize, threads: usize) -> Context {
    Context::new(Some(threads), 1 << 30)
        .unwrap()
        .with_layout(kpts, 1)
        .unwrap()
}

/// Solves `crystal(ctx)` on every layout and asserts every result equals the first one.
fn same_on(layouts: &[(usize, usize)], crystal: impl Fn(&Context) -> Crystal, options: &Options) {
    let mut reference: Option<Bits> = None;
    for &(kpts, threads) in layouts {
        let ctx = context(kpts, threads);
        let state = {
            let c = crystal(&ctx);
            let state = c.solve(options).unwrap();
            assert!(
                state.converged,
                "kpts={kpts} threads={threads}: not converged"
            );
            bits(&state)
        };
        match &reference {
            None => reference = Some(state),
            Some(first) => assert!(
                *first == state,
                "kpts={kpts} threads={threads} differs from layout {:?}",
                layouts[0]
            ),
        }
        assert_eq!(ctx.memory_used(), 0);
        // The k-point lanes ran (a layout the SCF ignored would compare equal too); a lane waiting
        // for the workers of its kernels is counted beside them, hence `threads + 1` per lane.
        if kpts > 1 {
            assert!(
                ctx.workers_total() >= kpts,
                "kpts={kpts} threads={threads}: the k-point lanes did not run"
            );
        }
        assert!(
            ctx.workers_peak() <= kpts * (threads + 1),
            "kpts={kpts} threads={threads}: {} workers at once",
            ctx.workers_peak()
        );
    }
}

fn snapshot(ctx: &Context, name: &str) -> Crystal {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../test/migration/phase5/{name}.json"));
    Crystal::load(ctx, path.to_str().unwrap()).unwrap()
}

fn options() -> Options {
    Options {
        width: 0.01 / HARTREE_EV,
        ..Default::default()
    }
}

#[test]
fn unpolarized_spin_polarized_and_spinor_snapshots_are_the_same_on_every_layout() {
    // si_crystal: 3 k-point tasks; h_spin_lda: 2 k-points x 2 spin channels; si_so: 2 spinor
    // k-points. Equal processor counts (3 and 6, 4, 2), then every other count against them.
    same_on(
        &[(1, 3), (3, 1), (1, 6), (3, 2), (1, 1), (2, 1), (2, 3)],
        |ctx| snapshot(ctx, "si_crystal"),
        &options(),
    );
    same_on(
        &[(1, 4), (4, 1), (2, 2), (1, 1), (3, 1), (2, 3)],
        |ctx| snapshot(ctx, "h_spin_lda"),
        &options(),
    );
    same_on(
        &[(1, 2), (2, 1), (1, 1), (2, 2)],
        |ctx| snapshot(ctx, "si_so"),
        &options(),
    );
}

#[test]
fn k_point_tasks_reading_archived_stacks_match_the_resident_serial_run() {
    let serial = Context::new(Some(1), 1 << 30).unwrap();
    let resident = bits(&snapshot(&serial, "si_crystal").solve(&options()).unwrap());
    for (kpts, threads) in [(3, 1), (3, 2)] {
        let ctx = context(kpts, threads);
        let state = {
            let mut c = snapshot(&ctx, "si_crystal");
            for point in &mut c.points {
                let samples = point.psi.read(&ctx).unwrap().to_vec();
                point.psi =
                    ComplexSamples::archived_copy(&ctx, &samples, Precision::Float64).unwrap();
            }
            bits(&c.solve(&options()).unwrap())
        };
        assert!(
            resident == state,
            "kpts={kpts} threads={threads} on archived stacks"
        );
        assert!(ctx.archived_bytes() > 0);
    }
}

#[test]
fn more_k_point_lanes_than_tasks_are_refused_with_the_largest_that_fits() {
    let ctx = context(4, 1);
    let error = snapshot(&ctx, "si_crystal")
        .solve(&options())
        .err()
        .expect("kpts=4 on 3 tasks");
    assert!(matches!(error, Error::Invalid(_)), "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains("kpts=4 exceeds the 3 k-point tasks")
            && message.contains("the largest kpts is 3"),
        "{message}"
    );
    // The spin channels are tasks too: 2 k-points x 2 spins take kpts = 4, not 5.
    assert!(snapshot(&context(4, 1), "h_spin_lda")
        .solve(&options())
        .is_ok());
    let message = snapshot(&context(5, 1), "h_spin_lda")
        .solve(&options())
        .err()
        .expect("kpts=5 on 4 tasks")
        .to_string();
    assert!(message.contains("the largest kpts is 4"), "{message}");
}

/// A cell built by the Rust setup (PAW-LCAO SZ) on the full `mesh`.
struct Cell {
    library: &'static str,
    symbols: Vec<&'static str>,
    lattice: Matrix3<f64>,
    grid: [usize; 3],
    fractional: Vec<[f64; 3]>,
    moments: Vec<f64>,
    spin: bool,
    functional: &'static str,
    mesh: [usize; 3],
    hubbard: Option<&'static str>,
}
fn build(ctx: &Context, cell: &Cell) -> Crystal {
    let points: usize = cell.grid.iter().product();
    assert!(
        points.div_ceil(mandacaru_core::math::projection::density_tile(points)) >= 2,
        "the cell must split its density over several tiles"
    );
    let mut species: Vec<&str> = vec![];
    for symbol in &cell.symbols {
        if !species.contains(symbol) {
            species.push(symbol);
        }
    }
    let datasets = species
        .iter()
        .map(|symbol| Arc::new(setup_shells_support::dataset(ctx, cell.library, symbol)))
        .collect();
    let count: usize = cell.mesh.iter().product();
    let input = SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: cell.lattice,
        grid_shape: cell.grid,
        atoms: cell
            .symbols
            .iter()
            .zip(&cell.fractional)
            .zip(&cell.moments)
            .map(|((symbol, fractional), moment)| AtomSpec {
                species: species.iter().position(|s| s == symbol).unwrap(),
                symbol: (*symbol).into(),
                fractional: *fractional,
                moment: *moment,
            })
            .collect(),
        kpoints_frac: (0..count)
            .map(|k| sampling::mesh_point(cell.mesh, k).unwrap())
            .collect(),
        weights: vec![1. / count as f64; count],
        functional: cell.functional.into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: false,
        spin: cell.spin,
        symmetry: false,
        filter_cutoff: None,
        sigma: None,
        dataset_files: None,
        hubbard_u: cell.hubbard.map(|text| Request::parse(text).unwrap()),
        charge: None,
    };
    create_setup(ctx, input, datasets)
        .unwrap()
        .build(ctx)
        .unwrap()
}

/// Silicon (two atoms, fcc cell of 10.26 Bohr) on 20^3 points: four density tiles.
fn silicon(spin: bool, functional: &'static str, library: &'static str) -> Cell {
    let a = 10.26 / 2.;
    Cell {
        library,
        symbols: vec!["Si", "Si"],
        lattice: Matrix3::from_columns(&[
            Vector3::new(0., a, a),
            Vector3::new(a, 0., a),
            Vector3::new(a, a, 0.),
        ]),
        grid: [20, 20, 20],
        fractional: vec![[0.; 3], [0.25; 3]],
        moments: if spin { vec![0.5, -0.5] } else { vec![0., 0.] },
        spin,
        functional,
        mesh: [2, 1, 1],
        hubbard: None,
    }
}

fn tight() -> Options {
    Options {
        width: 0.01 / HARTREE_EV,
        tolerance: 1e-9,
        ..Default::default()
    }
}

#[test]
fn a_crystal_built_on_several_density_tiles_is_the_same_on_every_layout() {
    // The setup itself runs on every processor count here (its sphere projections included).
    let lda = silicon(false, "lda", "lda-sr");
    same_on(&[(1, 2), (2, 1), (1, 1)], |ctx| build(ctx, &lda), &tight());
}

#[test]
fn a_spin_polarized_built_crystal_is_the_same_on_every_layout() {
    let spin = silicon(true, "lda", "lda-sr");
    same_on(&[(2, 2), (4, 1), (1, 1)], |ctx| build(ctx, &spin), &tight());
}

#[test]
fn a_meta_gga_built_crystal_is_the_same_on_every_layout() {
    let meta = silicon(false, "r2scan", "pbe-sr");
    same_on(&[(1, 2), (2, 1), (1, 1)], |ctx| build(ctx, &meta), &tight());
}

#[test]
fn a_dft_plus_u_crystal_is_the_same_on_every_layout() {
    same_on(
        &[(1, 2), (2, 1), (1, 1), (2, 2)],
        |ctx| {
            let mut c = hubbard_support::load(ctx, "cuh_7_spin");
            c.hubbard
                .as_mut()
                .unwrap()
                .set_interaction(4. / HARTREE_EV, 0.)
                .unwrap();
            c
        },
        &options(),
    );
}
