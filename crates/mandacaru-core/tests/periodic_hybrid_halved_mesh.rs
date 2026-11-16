//! Q11/SH-5, the time-reversal-halved k mesh of a screened hybrid on crystals built by the Rust
//! setup (exact under time reversal). The half lists each point without its `-k` at twice the
//! weight; the exact exchange builds the states at `-k` from those at `k` (the conjugates, the
//! Kramers partners `i sigma_y psi^*` for a time-reversal-symmetric spinor) with their projections
//! reversed through the complex-harmonic projectors.
//!
//! * the SCF energy of the half is that of the whole mesh to 1e-10 Ha, the levels at the listed
//!   points to 1e-8 Ha: Si HSE06 and HSE06 + spin-orbit on a 3 x 1 x 1 mesh;
//! * the analytic forces and stress of the half are those of the whole mesh to 1e-8 (Ha/Bohr, Ha)
//!   and match the central differences of the reconverged free energy of the half to 1e-6: LiH
//!   HSE06 in a skew cell on a 3 x 1 x 1 mesh (the strained builds keep the node counts, the
//!   filter cutoff and the ion width of the unstrained cell, the pins the stress holds);
//! * the setup's own reduction of a 3 x 1 x 1 mesh (the route `Mandacaru` takes) halves a hybrid
//!   by time reversal and keeps the whole mesh's energy, unpolarized, spinor and with two spin
//!   channels.
//!
//! Needs `MANDACARU_PAW_PATH`. Run: `MANDACARU_PAW_PATH=... cargo test --profile release-test -p
//! mandacaru-core --test periodic_hybrid_halved_mesh -- --ignored --test-threads=1`.
mod setup_shells_support;

use mandacaru_core::{
    periodic::{
        response,
        setup::parse::{create_setup, AtomSpec, SetupInput},
        solver::Options,
        Crystal, PeriodicResult,
    },
    Context,
};
use nalgebra::{Matrix3, Vector3};
use std::sync::Arc;

const HARTREE_EV: f64 = 27.211386245988;

/// A cell for the Rust setup (PAW-LCAO SZ, raw projectors).
#[derive(Clone)]
struct Cell {
    library: &'static str,
    symbols: Vec<&'static str>,
    /// Lattice vectors as columns (Bohr).
    lattice: Matrix3<f64>,
    grid: [usize; 3],
    fractional: Vec<[f64; 3]>,
    spinor: bool,
    /// Two collinear spin channels (zero moments).
    spin: bool,
    /// The unstrained lattice: its spacing pins the filter cutoff and the ion width (the setup's
    /// defaults, `pi / spacing` and `1.4 spacing`), so a strained build keeps them.
    reference: Matrix3<f64>,
}

/// Where the k-points come from: listed (points and weights) or a mesh the setup reduces.
enum Points {
    Listed(Vec<[f64; 3]>, Vec<f64>),
    Mesh([usize; 3]),
}

/// The whole 3 x 1 x 1 mesh and its time-reversal half (Gamma, and 1/3 standing for 2/3).
fn whole() -> Points {
    Points::Listed(
        vec![[0.; 3], [1. / 3., 0., 0.], [2. / 3., 0., 0.]],
        vec![1. / 3.; 3],
    )
}
fn half() -> Points {
    Points::Listed(vec![[0.; 3], [1. / 3., 0., 0.]], vec![1. / 3., 2. / 3.])
}

fn context() -> Context {
    Context::new(Some(2), 1 << 30).unwrap()
}

fn build(ctx: &Context, cell: &Cell, points: Points) -> Crystal {
    let (mesh, kpoints, weights) = match points {
        Points::Listed(kpoints, weights) => (None, kpoints, weights),
        Points::Mesh(size) => (Some((size, true)), vec![], vec![]),
    };
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
            let digest = mandacaru_core::paw::digest::dataset_content_sha256(
                &Context::new(Some(1), 1 << 30).unwrap(),
                &path,
            )
            .unwrap();
            (path.to_string_lossy().into_owned(), digest)
        })
        .collect();
    let spacing = (0..3)
        .map(|a| cell.reference.column(a).norm() / cell.grid[a] as f64)
        .fold(0f64, f64::max);
    let input = SetupInput {
        mesh,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: cell.lattice,
        grid_shape: cell.grid,
        atoms: cell
            .symbols
            .iter()
            .zip(&cell.fractional)
            .map(|(symbol, fractional)| AtomSpec {
                species: species.iter().position(|s| s == symbol).unwrap(),
                symbol: (*symbol).into(),
                fractional: *fractional,
                moment: 0.,
            })
            .collect(),
        kpoints_frac: kpoints,
        weights,
        functional: "hse06".into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: cell.spinor,
        spin: cell.spin,
        symmetry: false,
        filter_cutoff: Some(std::f64::consts::PI / spacing),
        sigma: Some(1.4 * spacing),
        dataset_files: Some(files),
        hubbard_u: None,
        charge: None,
    };
    create_setup(ctx, input, datasets)
        .unwrap()
        .build(ctx)
        .unwrap()
}

fn solve(c: &Crystal, options: &Options) -> PeriodicResult {
    let r = c.solve(options).unwrap();
    assert!(r.converged, "not converged");
    r
}

/// Silicon (fcc, 10.26 Bohr) on 16^3 points.
fn silicon(library: &'static str, spinor: bool) -> Cell {
    let a = 10.26 / 2.;
    let lattice = Matrix3::from_columns(&[
        Vector3::new(0., a, a),
        Vector3::new(a, 0., a),
        Vector3::new(a, a, 0.),
    ]);
    Cell {
        library,
        symbols: vec!["Si", "Si"],
        lattice,
        reference: lattice,
        grid: [16, 16, 16],
        fractional: vec![[0.; 3], [0.25; 3]],
        spinor,
        spin: false,
    }
}

/// LiH in a skew polar cell: no symmetry fixes a force or a shear.
fn lithium_hydride() -> Cell {
    let lattice = Matrix3::new(7., 0.8, 0.4, 0., 6.5, 0.6, 0., 0., 7.5);
    Cell {
        library: "pbe-sr",
        symbols: vec!["Li", "H"],
        lattice,
        reference: lattice,
        grid: [20, 19, 22],
        fractional: vec![[0.1, 0.2, 0.15], [0.1 + 3. / 7., 0.2, 0.15]],
        spinor: false,
        spin: false,
    }
}

#[test]
#[ignore = "slow: four hybrid SCFs, 50 s (measured 2026-11-15)"]
fn a_time_reversal_halved_hybrid_mesh_gives_the_whole_mesh_energy() {
    let options = Options {
        width: 0.01 / HARTREE_EV,
        tolerance: 1e-9,
        density_tolerance: 1e-7,
        ..Default::default()
    };
    let ctx = context();
    let mut failures = Vec::new();
    for (label, cell) in [
        ("HSE06", silicon("pbe-sr", false)),
        ("HSE06 + spin-orbit", silicon("pbe-fr", true)),
    ] {
        let w = solve(&build(&ctx, &cell, whole()), &options);
        let h = solve(&build(&ctx, &cell, half()), &options);
        let gap = (w.free_energy - h.free_energy).abs();
        let mut levels = 0f64;
        for (k, listed) in h.eigenvalues.iter().enumerate() {
            for (x, y) in listed.iter().zip(&w.eigenvalues[k]) {
                levels = levels.max((x - y).abs());
            }
        }
        println!(
            "{label}: whole {:.12} Ha, half {:.12} Ha, difference {gap:.3e}, levels {levels:.3e}",
            w.free_energy, h.free_energy
        );
        failures.extend((gap >= 1e-10).then(|| format!("{label}: energy {gap:e}")));
        failures.extend((levels >= 1e-8).then(|| format!("{label}: levels {levels:e}")));
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// The route `Mandacaru` takes: the setup reduces the 3 x 1 x 1 Gamma-centered mesh itself (by
/// time reversal alone for a hybrid), for HSE06, HSE06 + spin-orbit and two collinear spin
/// channels with zero moments; each equals the whole mesh listed point by point, the energy to
/// 1e-10 Ha and the levels of every kept point (matched by its fractional coordinates, modulo the
/// reciprocal lattice and -k) to 1e-8 Ha.
#[test]
#[ignore = "slow: six hybrid SCFs, about 80 s"]
fn the_setup_halves_a_hybrid_mesh_by_time_reversal_and_keeps_its_energy() {
    let options = Options {
        width: 0.01 / HARTREE_EV,
        tolerance: 1e-9,
        density_tolerance: 1e-7,
        ..Default::default()
    };
    let ctx = context();
    let mut failures = Vec::new();
    let polarized = Cell {
        spin: true,
        ..silicon("pbe-sr", false)
    };
    for (label, cell) in [
        ("HSE06", silicon("pbe-sr", false)),
        ("HSE06 + spin-orbit", silicon("pbe-fr", true)),
        ("HSE06, two spin channels", polarized),
    ] {
        let listed = build(&ctx, &cell, whole());
        let reduced = build(&ctx, &cell, Points::Mesh([3, 1, 1]));
        let weights: Vec<f64> = reduced.points.iter().map(|p| p.weight).collect();
        println!("{label}: the setup lists {} points, weights {weights:?}", weights.len());
        if reduced.points.len() != 2 {
            failures.push(format!("{label}: {} points, not the half", reduced.points.len()));
        }
        let (w, h) = (solve(&listed, &options), solve(&reduced, &options));
        let gap = (w.free_energy - h.free_energy).abs();
        let channels = h.eigenvalues.len() / reduced.points.len();
        let mut levels = 0f64;
        for (k, point) in reduced.points.iter().enumerate() {
            let same = |a: [f64; 3], b: [f64; 3], sign: f64| {
                (0..3).all(|i| {
                    let d = a[i] - sign * b[i];
                    (d - d.round()).abs() < 1e-8
                })
            };
            let at = (0..listed.points.len())
                .find(|&j| {
                    same(listed.points[j].fractional, point.fractional, 1.)
                        || same(listed.points[j].fractional, point.fractional, -1.)
                })
                .expect("a kept point of the whole mesh");
            for s in 0..channels {
                let (kept, all) = (s * reduced.points.len() + k, s * listed.points.len() + at);
                for (x, y) in h.eigenvalues[kept].iter().zip(&w.eigenvalues[all]) {
                    levels = levels.max((x - y).abs());
                }
            }
        }
        println!(
            "{label}: listed whole {:.12} Ha, reduced by the setup {:.12} Ha, difference {gap:.3e}, levels {levels:.3e}",
            w.free_energy, h.free_energy
        );
        failures.extend((gap >= 1e-10).then(|| format!("{label}: energy {gap:e}")));
        failures.extend((levels >= 1e-8).then(|| format!("{label}: levels {levels:e}")));
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// Tight enough that the central differences below are limited by their step, not the SCF.
fn tight() -> Options {
    Options {
        width: 0.01 / HARTREE_EV,
        tolerance: 1e-12,
        density_tolerance: 1e-10,
        max_iterations: 300,
        ..Default::default()
    }
}

/// The free energy of the half mesh of `cell`, reconverged.
fn free_energy(ctx: &Context, cell: &Cell) -> f64 {
    solve(&build(ctx, cell, half()), &tight()).free_energy
}

#[test]
#[ignore = "slow: two hybrid force and stress evaluations and twelve reconverged SCFs, minutes"]
fn the_forces_and_stress_of_a_halved_hybrid_mesh_are_those_of_the_whole_mesh_and_of_the_energy() {
    let ctx = context();
    let cell = lithium_hydride();
    let derivatives = |points| {
        let c = build(&ctx, &cell, points);
        let r = solve(&c, &tight());
        let engine = response::Response::new(&c).unwrap();
        let forces = engine.forces(&c, &r).unwrap();
        let stress = engine.stress(&c, &r).unwrap();
        (r.free_energy, forces, stress.gradient)
    };
    let (fw, forces_whole, g_whole) = derivatives(whole());
    let (fh, forces_half, g_half) = derivatives(half());
    let mut failures = Vec::new();
    println!("free energy: whole {fw:.12} Ha, half {fh:.12} Ha, difference {:.3e}", (fw - fh).abs());
    failures.extend(((fw - fh).abs() >= 1e-10).then(|| format!("energy {:e}", (fw - fh).abs())));
    let force_gap = forces_whole
        .iter()
        .flatten()
        .zip(forces_half.iter().flatten())
        .fold(0f64, |a, (x, y)| a.max((x - y).abs()));
    let stress_gap = g_whole
        .iter()
        .flatten()
        .zip(g_half.iter().flatten())
        .fold(0f64, |a, (x, y)| a.max((x - y).abs()));
    let force_peak = forces_half.iter().flatten().fold(0f64, |a, x| a.max(x.abs()));
    let stress_peak = g_half.iter().flatten().fold(0f64, |a, x| a.max(x.abs()));
    println!(
        "half vs whole: forces {force_gap:.3e} Ha/Bohr (peak {force_peak:.3e}), stress gradient \
         {stress_gap:.3e} Ha (peak {stress_peak:.3e})"
    );
    failures.extend((force_gap >= 1e-8).then(|| format!("forces vs whole {force_gap:e}")));
    failures.extend((stress_gap >= 1e-8).then(|| format!("stress vs whole {stress_gap:e}")));

    // The reconverged central differences of the half's free energy.
    let step = 1e-3;
    let inverse = cell.lattice.try_inverse().unwrap();
    for (atom, axis) in [(0, 0), (1, 1), (0, 2)] {
        let moved = |sign: f64| {
            let mut c = cell.clone();
            let shift = inverse * (Vector3::ith(axis, 1.) * (sign * step));
            for a in 0..3 {
                c.fractional[atom][a] += shift[a];
            }
            free_energy(&ctx, &c)
        };
        let slope = -(moved(1.) - moved(-1.)) / (2. * step);
        let miss = (forces_half[atom][axis] - slope).abs();
        println!(
            "force atom {atom} axis {axis}: analytic {:+.9e}, reconverged {slope:+.9e}, miss {miss:.2e}",
            forces_half[atom][axis]
        );
        failures.extend((miss >= 1e-6).then(|| format!("force {atom},{axis} vs FD {miss:e}")));
    }
    for (a, b) in [(0, 0), (0, 1), (1, 2)] {
        let strained = |sign: f64| {
            let mut e = Matrix3::<f64>::zeros();
            e[(a, b)] += 0.5 * sign * step;
            e[(b, a)] += 0.5 * sign * step;
            let mut c = cell.clone();
            c.lattice = (Matrix3::identity() + e) * cell.lattice;
            free_energy(&ctx, &c)
        };
        let slope = (strained(1.) - strained(-1.)) / (2. * step);
        let analytic = 0.5 * (g_half[a][b] + g_half[b][a]);
        let miss = (analytic - slope).abs();
        println!(
            "stress gradient ({a},{b}): analytic {analytic:+.9e} Ha, reconverged {slope:+.9e} Ha, miss {miss:.2e}"
        );
        failures.extend((miss >= 1e-6).then(|| format!("stress {a},{b} vs FD {miss:e}")));
    }
    assert!(failures.is_empty(), "{failures:?}");
}
