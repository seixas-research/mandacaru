//! The center derivative of the projections (`G_C = dC/dc`) and of the short-range local
//! matrices of the periodic setup, and the analytic slope of a smooth core
//! (`periodic::setup::pulay`). The gates are central differences of the quantities themselves
//! with the atom's functions, projectors and quadrature spheres moved together (the identities
//! the force engine uses), and the end-to-end derivative of a trace against a random density.
//! Needs `MANDACARU_PAW_PATH`.
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;

use mandacaru_core::{
    math::spline::Cubic,
    paw::fields::place,
    periodic::setup::{
        basis_gradient::ShortRange,
        fields::core_table,
        pulay::{
            core_gradient, mesh_dot, projection, projection_gradient, short_range_gradients, Matrix,
        },
    },
};
use num_complex::Complex64;
use setup_gradients_support::context;
use setup_pulay_support::*;
use setup_shells_support::dataset;

const GATE: f64 = 1e-8;
const H: f64 = 2e-4;
/// The step of the core's central differences. A real table has a knot every 0.006 Bohr and its
/// spline has third-derivative jumps (the ten digits of the data), so the difference of the placed
/// core has an error that falls as the step squared: a smaller step than the others'.
const CORE_STEP: f64 = 5e-5;
const CASES: [&str; 2] = ["si", "al2"];
/// The pairs `(atom, axis)` that are differenced for each fixture (every projection rebuild is a
/// full quadrature of the projector spheres).
const MOVES: [(&str, [(usize, usize); 3]); 2] = [
    ("si", [(0, 0), (1, 1), (1, 2)]),
    ("al2", [(0, 2), (1, 0), (1, 2)]),
];

/// `P G - G Q`: the rows of `own_rows` kept, the columns of `own_columns` taken away.
fn projection_identity(g: &Matrix, own_rows: &[bool], own_columns: &[bool]) -> Matrix {
    Matrix::from_fn(g.nrows(), g.ncols(), |mu, p| {
        let mut x = Complex64::new(0., 0.);
        if own_rows[mu] {
            x += g[(mu, p)];
        }
        if own_columns[p] {
            x -= g[(mu, p)];
        }
        x
    })
}

#[test]
fn the_projection_gradient_obeys_the_ownership_identity_si() {
    projection_identity_of(MOVES[0].0, &MOVES[0].1);
}

#[test]
fn the_projection_gradient_obeys_the_ownership_identity_al2() {
    projection_identity_of(MOVES[1].0, &MOVES[1].1);
}

fn projection_identity_of(name: &'static str, moves: &[(usize, usize)]) {
    let ctx = context();
    let fx = Fixture::new(&ctx, name);
    let g = projection_gradient(&ctx, &fx.shells, &fx.lattice, KPOINT, &fx.projectors).unwrap();
    let c = projection(&ctx, &fx.shells, &fx.lattice, KPOINT, &fx.projectors).unwrap();
    assert!(peak(&c) > 1e-3 && peak(&g[0]) > 1e-3, "non-vacuous");
    for (index, &(atom, axis)) in moves.iter().enumerate() {
        let rows = fx.rows_of(atom);
        let columns: Vec<bool> = fx.projectors.iter().map(|p| p.atom == atom).collect();
        let moved = |basis: bool, projectors: bool, d: f64| {
            let shells = if basis {
                fx.moved_basis(atom, axis, d)
            } else {
                fx.shells.clone()
            };
            let list = if projectors {
                fx.moved_projectors(atom, axis, d)
            } else {
                fx.projectors.clone()
            };
            projection(&ctx, &shells, &fx.lattice, KPOINT, &list).unwrap()
        };
        let none = vec![false; fx.projectors.len()];
        let no_rows = vec![false; fx.functions()];
        // The functions alone, the projectors alone (once), both together.
        let mut cases = vec![(true, true, projection_identity(&g[axis], &rows, &columns))];
        if index == 0 {
            cases.push((true, false, projection_identity(&g[axis], &rows, &none)));
            cases.push((
                false,
                true,
                projection_identity(&g[axis], &no_rows, &columns),
            ));
        }
        for (basis, projectors, want) in cases {
            let fd = central(H, &mut |d| moved(basis, projectors, d));
            let error = gap(&fd, &want);
            assert!(
                error < GATE,
                "{name} atom {atom} axis {axis} basis {basis} projectors {projectors}: {error:.2e}"
            );
            if peak(&want) > 1e-3 {
                // Controls: the opposite sign, the value in place of the derivative.
                assert!(gap(&fd, &(-&want)) > 100. * GATE);
                assert!(gap(&fd, &c) > 100. * GATE);
            }
        }
        // The projector-moving term is not zero and cannot be dropped.
        let full = projection_identity(&g[axis], &rows, &columns);
        let rows_only = projection_identity(&g[axis], &rows, &none);
        assert!(gap(&full, &rows_only) > 100. * GATE);
    }
}

/// `2 Re sum_{mu in rows} (G D)_{mu mu}`.
fn row_trace(g: &Matrix, d: &Matrix, rows: &[bool]) -> f64 {
    let gd = g * d;
    2. * (0..g.nrows())
        .filter(|&mu| rows[mu])
        .map(|mu| gd[(mu, mu)].re)
        .sum::<f64>()
}

#[test]
fn the_short_range_force_needs_the_sphere_that_moves_with_its_atom() {
    for name in CASES {
        let ctx = context();
        let fx = Fixture::new(&ctx, name);
        let m = fx.functions();
        let d = density(m);
        let spheres: Vec<ShortRange> = fx
            .centers
            .iter()
            .enumerate()
            .map(|(atom, center)| ShortRange {
                center: *center,
                potential: &fx.potentials[atom],
                sigma: fx.sigma,
            })
            .collect();
        let g = short_range_gradients(&ctx, &fx.shells, &fx.lattice, KPOINT, &spheres).unwrap();
        assert!(peak(&g[0][0]) > 1e-4, "non-vacuous");
        let everyone = vec![true; m];
        for atom in 0..fx.atoms() {
            let own = fx.rows_of(atom);
            for axis in 0..3 {
                // Atom `atom` moves: its functions and its sphere.
                let energy = |d_shift: f64| {
                    let mut centers = fx.centers.clone();
                    centers[atom][axis] += d_shift;
                    let shells = fx.moved_basis(atom, axis, d_shift);
                    Matrix::from_element(
                        1,
                        1,
                        Complex64::new(
                            trace(&d, &short_range_total(&fx, &ctx, &shells, &centers, KPOINT)),
                            0.,
                        ),
                    )
                };
                let fd = central(H, &mut |s| energy(s))[(0, 0)].re;
                // The functions' share: the rows of the moved atom of every sphere's `G`.
                let rows: f64 = g.iter().map(|ga| row_trace(&ga[axis], &d, &own)).sum();
                // The sphere's share: the translation law, minus every function moving.
                let sphere = -row_trace(&g[atom][axis], &d, &everyone);
                let scale = fd.abs().max(rows.abs()).max(1e-3);
                let error = (fd - rows - sphere).abs() / scale;
                assert!(
                    error < GATE,
                    "{name} atom {atom} axis {axis}: {error:.2e} (fd {fd:.6e}, rows {rows:.6e}, sphere {sphere:.6e})"
                );
                // Controls: the functions alone miss by far more than the gate (the third
                // center is real), as does a sphere term of the opposite sign.
                assert!(
                    (fd - rows).abs() / scale > 100. * GATE,
                    "{name}: sphere share is zero"
                );
                assert!((fd - rows + sphere).abs() / scale > 100. * GATE);
            }
        }
    }
}

#[test]
fn the_core_gradient_equals_the_center_derivative_of_the_placed_core() {
    let ctx = context();
    let fx = Fixture::new(&ctx, "si");
    // A smooth, non-spherically-trivial radial core on a table that starts at the first node.
    let r: Vec<f64> = (0..400).map(|i| 0.01 + 0.015 * i as f64).collect();
    let core: Vec<f64> = r
        .iter()
        .map(|x| (1. + 0.3 * x) * (-x * x / 0.8).exp())
        .collect();
    // The table's slope: the spline against the analytic derivative of the function it holds.
    let profile = Cubic::new(
        &ctx,
        ctx.copy_slice(&r).unwrap(),
        ctx.copy_slice(&core).unwrap(),
    )
    .unwrap();
    for &x in &[0.3f64, 1.1, 2.7, 4.2] {
        let analytic = (0.3 - 2. * x * (1. + 0.3 * x) / 0.8) * (-x * x / 0.8).exp();
        assert!((profile.value_slope(x).1 - analytic).abs() < 1e-6);
    }
    check_core(&ctx, &fx, &r, &core);
}

/// E5 on a real table: the smooth core a shipped dataset was unscreened with, for the first of
/// these species that carries one (a species without a core contributes nothing to E5).
#[test]
fn the_core_gradient_of_a_shipped_dataset_core_equals_the_center_derivative() {
    let ctx = context();
    // A core is compact: a finer grid than the recorded one puts nodes inside it.
    let fx = Fixture::with_shape(&ctx, "si", Some([48, 48, 48]));
    let mut checked = Vec::new();
    for symbol in ["Si", "Al", "Cu", "Ni", "Fe", "Zn", "Ge", "Ga"] {
        let ds = dataset(&ctx, "lda-sr", symbol);
        let source = ds.nlcc.source.clone().unwrap_or_default();
        let Some(smooth) =
            core_table(ds.nlcc.applied, &source, ds.smooth_core_density.as_slice()).unwrap()
        else {
            continue;
        };
        let top = smooth.iter().fold(0f64, |a, v| a.max(v.abs()));
        let reach = ds.r.as_slice()[smooth.iter().rposition(|v| v.abs() > 1e-8 * top).unwrap()];
        println!(
            "{symbol}: smooth core max {top:.3e}, reach {reach:.2} Bohr, r_nlcc {:?}",
            ds.nlcc.r_nlcc
        );
        check_core(&ctx, &fx, ds.r.as_slice(), smooth);
        checked.push(symbol);
    }
    println!("shipped smooth cores checked: {checked:?}");
    assert!(
        !checked.is_empty(),
        "no shipped species of the list carries a smooth core"
    );
}

/// `core_gradient` of the radial table `(r, core)` against the central difference of the placed
/// core with the center moved, on the Si grid, for every axis.
fn check_core(ctx: &mandacaru_core::Context, fx: &Fixture, r: &[f64], core: &[f64]) {
    let mesh = fx.mesh(ctx);
    let profile = Cubic::new(
        ctx,
        ctx.copy_slice(r).unwrap(),
        ctx.copy_slice(core).unwrap(),
    )
    .unwrap();
    let center = [1.7, 2.3, 0.9];
    for axis in 0..3 {
        let field = |d: f64| {
            let mut c = center;
            c[axis] += d;
            Matrix::from_column_slice(
                mesh.len(),
                1,
                &place(ctx, &fx.grid, &profile, &[c])
                    .unwrap()
                    .as_slice()
                    .iter()
                    .map(|v| Complex64::new(*v, 0.))
                    .collect::<Vec<_>>(),
            )
        };
        let fd = central(CORE_STEP, &mut |d| field(d));
        let got = core_gradient(ctx, &fx.grid, &r, &core, center, axis).unwrap();
        let want = Matrix::from_column_slice(
            mesh.len(),
            1,
            &got.as_slice()
                .iter()
                .map(|v| Complex64::new(*v, 0.))
                .collect::<Vec<_>>(),
        );
        let error = gap(&fd, &want);
        println!(
            "core gradient axis {axis}: {error:.2e} (peak {:.2e})",
            peak(&want)
        );
        assert!(error < GATE, "axis {axis}: {error:.2e}");
        // Non-vacuous: the sampled gradient is a fraction of the table's own scale.
        let scale = core.iter().fold(0f64, |a, v| a.max(v.abs()));
        assert!(
            peak(&want) > 1e-3 * scale,
            "axis {axis}: {:.2e}",
            peak(&want)
        );
        // Controls: the opposite sign, and another axis' field.
        assert!(gap(&fd, &(-&want)) > 100. * GATE);
        let other = core_gradient(ctx, &fx.grid, &r, &core, center, (axis + 1) % 3).unwrap();
        let other = Matrix::from_column_slice(
            mesh.len(),
            1,
            &other
                .as_slice()
                .iter()
                .map(|v| Complex64::new(*v, 0.))
                .collect::<Vec<_>>(),
        );
        assert!(gap(&fd, &other) > 100. * GATE);
        // A center one lattice vector away is the same atom.
        let shifted = [
            center[0] + fx.lattice[(0, 0)],
            center[1] + fx.lattice[(1, 0)],
            center[2] + fx.lattice[(2, 0)],
        ];
        let again = core_gradient(ctx, &fx.grid, &r, &core, shifted, axis).unwrap();
        let worst = got
            .as_slice()
            .iter()
            .zip(again.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        assert!(worst < 1e-12 * peak(&want), "periodic image: {worst:.2e}");
        // `<v|d rho>` is the mesh dot product, and the derivative of `<v|rho>` for a fixed v.
        let v = rough_potential(mesh.len());
        let dot = mesh_dot(&mesh, &v, got.as_slice()).unwrap();
        let energy = |d: f64| {
            let mut c = center;
            c[axis] += d;
            let rho = place(ctx, &fx.grid, &profile, &[c]).unwrap();
            Matrix::from_element(
                1,
                1,
                Complex64::new(mesh_dot(&mesh, &v, rho.as_slice()).unwrap(), 0.),
            )
        };
        let fd_dot = central(H, &mut |d| energy(d))[(0, 0)].re;
        assert!(
            (dot - fd_dot).abs() < GATE * dot.abs().max(1e-3),
            "axis {axis}"
        );
    }
}
