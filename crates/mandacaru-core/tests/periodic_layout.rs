//! The crystal SCF on every processor layout: its spin channel and k-point tasks run on `kpts`
//! lanes and its grid density is split by range, and the result (free energy, iteration count,
//! eigenvalues, occupations, density, moments) is the same, bit for bit, on every layout and every
//! processor count. Unpolarized, spin-polarized and spinor snapshots, crystals built by the Rust
//! setup on grids of several density tiles (LDA, spin-polarized, r2SCAN: the meta-GGA kinetic
//! operator of a task runs its own transforms), DFT+U, and k-point tasks reading stacks archived
//! on disk; the screened hybrid (HSE06) with its target points on the k-point lanes and its
//! occupied states on the band lanes, unpolarized, spin-polarized and spinor; and the refusal of
//! more lanes than tasks. The built and the hybrid crystals need `MANDACARU_PAW_PATH`.
mod hubbard_support;
mod setup_shells_support;
mod setup_support;

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

/// A context of `kpts x bands x threads` processors.
fn layout_context(kpts: usize, bands: usize, threads: usize) -> Context {
    Context::new(Some(threads), 1 << 30)
        .unwrap()
        .with_layout(kpts, bands)
        .unwrap()
}

/// A context of `kpts x 1 x threads` processors.
fn context(kpts: usize, threads: usize) -> Context {
    layout_context(kpts, 1, threads)
}

/// [`same_on_layouts`] on `(kpts, threads)` layouts with one band group.
fn same_on(layouts: &[(usize, usize)], crystal: impl Fn(&Context) -> Crystal, options: &Options) {
    let layouts: Vec<_> = layouts.iter().map(|&(k, t)| (k, 1, t)).collect();
    same_on_layouts(&layouts, crystal, options);
}

/// Solves `crystal(ctx)` on every `(kpts, bands, threads)` layout and asserts every result equals
/// the first one: equal processor counts first, then other counts.
fn same_on_layouts(
    layouts: &[(usize, usize, usize)],
    crystal: impl Fn(&Context) -> Crystal,
    options: &Options,
) {
    let mut reference: Option<Bits> = None;
    for &(kpts, bands, threads) in layouts {
        let ctx = layout_context(kpts, bands, threads);
        let name = format!("kpts={kpts} bands={bands} threads={threads}");
        let state = {
            let c = crystal(&ctx);
            let state = c.solve(options).unwrap();
            assert!(state.converged, "{name}: not converged");
            bits(&state)
        };
        match &reference {
            None => reference = Some(state),
            Some(first) => assert!(
                *first == state,
                "{name} differs from layout {:?}",
                layouts[0]
            ),
        }
        assert_eq!(ctx.memory_used(), 0);
        // The lanes ran (a layout the SCF ignored would compare equal too). A lane waiting for
        // the lanes or the workers it starts is counted beside them: a k-point lane, its band
        // lanes and, per band lane, `threads` kernel workers.
        if kpts > 1 || bands > 1 {
            assert!(
                ctx.workers_total() >= kpts.max(bands),
                "{name}: the lanes did not run"
            );
        }
        assert!(
            ctx.workers_peak() <= kpts * (1 + bands * (threads + 1)),
            "{name}: {} workers at once",
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
    // The files and their SHA-256: a hybrid's spheres read the datasets they name.
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap())
        .join(cell.library);
    let files = species
        .iter()
        .map(|symbol| {
            let path = root.join(format!("{symbol}.parquet"));
            (
                path.to_string_lossy().into_owned(),
                mandacaru_core::paw::digest::dataset_content_sha256(
                    &mandacaru_core::Context::new(Some(1), 1 << 30).unwrap(),
                    &path,
                )
                .unwrap(),
            )
        })
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
        dataset_files: Some(files),
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

/// A snapshot as a screened hybrid at its first point alone, a whole Gamma mesh (the
/// `periodic_hybrid.rs` fixture).
fn gamma_hybrid(ctx: &Context, fixture: &str) -> Crystal {
    use serde_json::{json, Value};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/phase5")
        .join(fixture);
    let mut value: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    value["functional"] = json!("hse06");
    value["sites"] = setup_support::sites(&value, setup_support::library_of(fixture));
    let mut gamma = value["kpoints"][0].clone();
    gamma["weight"] = json!(1.0);
    value["kpoints"] = json!([gamma]);
    value["permutations"] = json!([]);
    Crystal::from_json(ctx, &value.to_string()).unwrap()
}

fn hybrid_options() -> Options {
    Options {
        width: 0.01 / HARTREE_EV,
        tolerance: 1e-9,
        density_tolerance: 1e-7,
        ..Default::default()
    }
}

#[test]
fn a_gamma_hybrid_is_the_same_on_every_band_layout() {
    // Silicon at Gamma: four occupied states, on two and four band lanes and on one.
    same_on_layouts(
        &[(1, 2, 2), (1, 4, 1), (1, 1, 1)],
        |ctx| gamma_hybrid(ctx, "si_crystal.json"),
        &hybrid_options(),
    );
}

#[test]
#[ignore = "slow: 15 s (measured 2026-11-12, loaded machine); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_layout -- --ignored every_band_and_spin_layout"]
fn a_gamma_hybrid_is_the_same_on_every_band_and_spin_layout() {
    // Silicon at Gamma, unpolarized (a window of one state on three lanes) and spinor.
    same_on_layouts(
        &[(1, 1, 4), (1, 3, 1), (1, 2, 3)],
        |ctx| gamma_hybrid(ctx, "si_crystal.json"),
        &hybrid_options(),
    );
    same_on_layouts(
        &[(1, 1, 2), (1, 2, 1), (1, 1, 1), (1, 3, 2)],
        |ctx| gamma_hybrid(ctx, "si_so.json"),
        &hybrid_options(),
    );
    // A spin-polarized hydrogen atom at Gamma: two SCF tasks (the spin channels), one exchange
    // target per channel. It has one occupied state, which leaves band lanes nothing to share:
    // k-point lanes and threads only.
    same_on_layouts(
        &[(1, 1, 2), (2, 1, 1), (1, 1, 1)],
        |ctx| gamma_hybrid(ctx, "h_spin_lda.json"),
        &hybrid_options(),
    );
}

/// Silicon (fcc, 10.26 Bohr) as HSE06 on the whole 2 x 1 x 1 mesh, on 16^3 points.
fn silicon_hybrid(spin: bool) -> Cell {
    Cell {
        grid: [16, 16, 16],
        ..silicon(spin, "hse06", "pbe-sr")
    }
}

#[test]
#[ignore = "slow: 54 s (measured 2026-11-12, loaded machine); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_layout -- --ignored a_hybrid_crystal_is_the_same"]
fn a_hybrid_crystal_is_the_same_on_every_k_point_and_band_layout() {
    // Two target points on the k-point lanes, four occupied states per point on the band lanes;
    // the same processor counts (4, 2), then others.
    let layouts = [
        (1, 1, 4),
        (2, 2, 1),
        (2, 1, 2),
        (1, 4, 1),
        (1, 1, 1),
        (2, 3, 1),
    ];
    let unpolarized = silicon_hybrid(false);
    same_on_layouts(&layouts, |ctx| build(ctx, &unpolarized), &hybrid_options());
}

#[test]
#[ignore = "slow: 104 s (measured 2026-11-12, loaded machine); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_layout -- --ignored a_spin_polarized_hybrid_crystal"]
fn a_spin_polarized_hybrid_crystal_is_the_same_on_every_k_point_and_band_layout() {
    // Two channels x two points: four SCF tasks, two exchange targets per channel.
    let spin = silicon_hybrid(true);
    same_on_layouts(
        &[(4, 1, 1), (2, 2, 1), (1, 2, 2), (1, 1, 1)],
        |ctx| build(ctx, &spin),
        &hybrid_options(),
    );
}

#[test]
fn band_lanes_that_do_not_fit_in_the_budget_are_refused_with_the_largest_that_does() {
    // The budget of a run with one band lane: 512 band lanes need at least 512 times the weighted
    // pairs of the target basis (8 functions x 1000 points x 16 bytes each), more than all of it.
    let probe = Context::new(Some(1), 1 << 30).unwrap();
    let options = Options {
        max_iterations: 2,
        ..hybrid_options()
    };
    let needed = {
        let c = gamma_hybrid(&probe, "si_crystal.json");
        drop(c.solve(&options));
        probe.memory_peak()
    };
    assert!(
        512 * 8 * 1000 * 16 > needed,
        "the probe peak {needed} is too large for the test"
    );
    let ctx = Context::new(Some(1), needed)
        .unwrap()
        .with_layout(1, 512)
        .unwrap();
    let c = gamma_hybrid(&ctx, "si_crystal.json");
    let error = c
        .solve(&options)
        .err()
        .expect("512 band lanes in a budget sized for one");
    assert!(matches!(error, Error::LayoutBudget(_)), "{error:?}");
    assert!(
        error.to_string().contains("bands=512")
            && error.to_string().contains("the largest bands that fits is"),
        "{error}"
    );
}

/// H2 (1.4 Bohr bond along x) in a 6 x 8 x 8 Bohr cell on the whole 2 x 1 x 1 mesh: two small
/// k-point tasks whose forces and stress are cheap.
fn h2_chain(functional: &'static str, library: &'static str, spin: bool) -> Cell {
    Cell {
        library,
        symbols: vec!["H", "H"],
        lattice: Matrix3::from_diagonal(&Vector3::new(6., 8., 8.)),
        grid: [15, 20, 20],
        fractional: vec![[2.3 / 6., 0.5, 0.5], [3.7 / 6., 0.5, 0.5]],
        moments: if spin { vec![0.5, 0.5] } else { vec![0., 0.] },
        spin,
        functional,
        mesh: [2, 1, 1],
        hubbard: None,
    }
}

/// The forces and the stress of `crystal(ctx)` on every `(kpts, bands, threads)` layout: the
/// per-k-point derivatives (and a hybrid's pair passes) run on the lanes and are added in the
/// order of the serial sweep, so every bit equals the first layout's.
fn same_derivatives_on_layouts(
    layouts: &[(usize, usize, usize)],
    crystal: impl Fn(&Context) -> Crystal,
    options: &Options,
) {
    use mandacaru_core::periodic::response;
    let mut reference: Option<(Vec<u64>, Vec<u64>)> = None;
    for &(kpts, bands, threads) in layouts {
        let ctx = layout_context(kpts, bands, threads);
        let name = format!("kpts={kpts} bands={bands} threads={threads}");
        let got = {
            let c = crystal(&ctx);
            let state = c.solve(options).unwrap();
            assert!(state.converged, "{name}: not converged");
            let forces = response::forces(&c, &state).unwrap();
            let stress = response::stress(&c, &state).unwrap();
            (
                forces
                    .iter()
                    .flatten()
                    .map(|f| f.to_bits())
                    .collect::<Vec<_>>(),
                stress
                    .gradient
                    .iter()
                    .flatten()
                    .map(|g| g.to_bits())
                    .collect::<Vec<_>>(),
            )
        };
        match &reference {
            None => reference = Some(got),
            Some(first) => {
                assert!(
                    first.0 == got.0,
                    "{name}: the forces differ from {:?}",
                    layouts[0]
                );
                assert!(
                    first.1 == got.1,
                    "{name}: the stress differs from {:?}",
                    layouts[0]
                );
            }
        }
        assert_eq!(ctx.memory_used(), 0);
    }
}

#[test]
#[ignore = "slow: 92 s (measured 2026-11-12, load 9-12); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_layout -- --ignored semilocal_forces_and_stress"]
fn semilocal_forces_and_stress_are_the_same_on_every_layout() {
    let lda = h2_chain("lda", "lda-sr", false);
    same_derivatives_on_layouts(
        &[(1, 1, 2), (2, 1, 1), (1, 1, 1)],
        |ctx| build(ctx, &lda),
        &tight(),
    );
    // Two spin channels x two points: four tasks for the bands and the derivatives.
    let spin = h2_chain("pbe", "pbe-sr", true);
    same_derivatives_on_layouts(
        &[(2, 1, 2), (4, 1, 1), (1, 1, 1)],
        |ctx| build(ctx, &spin),
        &tight(),
    );
}

#[test]
#[ignore = "slow: 362 s (measured 2026-11-12, load 25); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test periodic_layout -- --ignored hybrid_forces_and_stress"]
fn hybrid_forces_and_stress_are_the_same_on_every_k_point_and_band_layout() {
    // HSE06: the exchange operator of the bands (P4) and the pair passes of its derivatives on the
    // k-point and band lanes. LiH in a skew polar cell has two occupied states per point, so the
    // band lanes have states to share.
    let hybrid = Cell {
        library: "pbe-sr",
        symbols: vec!["Li", "H"],
        lattice: Matrix3::new(7., 0.8, 0.4, 0., 6.5, 0.6, 0., 0., 7.5),
        grid: [20, 19, 22],
        fractional: vec![[0.1, 0.2, 0.15], [0.1 + 3. / 7., 0.2, 0.15]],
        moments: vec![0., 0.],
        spin: false,
        functional: "hse06",
        mesh: [2, 1, 1],
        hubbard: None,
    };
    same_derivatives_on_layouts(
        &[(1, 2, 1), (2, 1, 1), (1, 1, 2), (1, 1, 1), (2, 2, 1)],
        |ctx| build(ctx, &hybrid),
        &tight(),
    );
}
