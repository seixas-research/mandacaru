//! The strain derivatives of the grid arrays (`strain_grid`: the Coulomb kernel, the Gaussian
//! ions, the compensation charge `sum_c q_c hat_c` and the smooth core) against central
//! differences of the plain builders over a strained `Sites` and `SpatialGrid` (cell, atoms,
//! grid step and grid origin scale together; node counts, width, shape tables and lattice
//! operations pinned; no SCF), on Si, Al2 and the skew LiH cell. The hats are a piecewise-linear
//! table: the finite difference carries a kink error at every frequency that straddles a node,
//! which is measured at the steps 1e-5, 5e-5, 2e-4 and excluded, element by element and
//! counted, from the gated comparison. Needs `MANDACARU_PAW_PATH`.
mod strain_support;

use mandacaru_core::{
    math::{reciprocal, spline::Cubic},
    paw::fields::place,
    periodic::setup::{
        electrostatics::{coulomb, hats, ion_charge, kernel, max_wavenumber},
        shapes::Shapes,
        strain_grid::{core_strain, hats_strain, ions_strain, kernel_strain, VOIGT},
    },
};
use num_complex::Complex64;
use strain_support::*;

/// Gate for the smooth arrays, relative to the array's maximum (the plan's E-2/E-4 number).
const GATE: f64 = 1e-9;
/// Gate for the arrays that carry the shape tables and the core spline.
const TABLE_GATE: f64 = 1e-8;
/// The step of the gated core difference.
const CORE_STEP: f64 = 1e-5;

#[test]
fn the_kernel_strain_is_the_difference_of_the_kernel_on_the_strained_grid() {
    let ctx = context();
    for name in ["si", "al2", "lih"] {
        let cell = Cell::load(&ctx, name);
        let n = cell.grid.index.points();
        let got = kernel_strain(&ctx, &cell.grid).unwrap();
        let table = reciprocal::alias_table(&ctx, &cell.grid).unwrap();
        assert!(
            !table.is_empty(),
            "{name}: the alias average must act for the control"
        );
        let b = reciprocal::reciprocal_cell(&cell.grid).unwrap();
        let shape = cell.grid.index.shape();
        let (mut miss, mut scale_all) = (0f64, 0f64);
        let (mut unaliased, mut flipped) = (0f64, 0f64);
        for (row, &(a, c)) in VOIGT.iter().enumerate() {
            let side = |h: f64| -> Vec<f64> {
                let strained = cell.strained(entry(a, c, h));
                assert_pinned(&ctx, &cell, &strained);
                kernel(&ctx, &strained.grid).unwrap().as_slice().to_vec()
            };
            let difference = |h: f64| -> Vec<f64> {
                side(h)
                    .iter()
                    .zip(side(-h))
                    .map(|(p, m)| (p - m) / (2. * h))
                    .collect()
            };
            let (one, two) = (difference(1e-4), difference(2e-4));
            let want: Vec<f64> = one
                .iter()
                .zip(&two)
                .map(|(x, y)| (4. * x - y) / 3.)
                .collect();
            let have = &got.as_slice()[row * n..(row + 1) * n];
            let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
            miss = miss.max(worst(have, &want, &|_| false) / scale);
            scale_all = scale_all.max(scale);
            flipped = flipped.max(
                have.iter()
                    .zip(&want)
                    .map(|(x, y)| (-x - y).abs())
                    .fold(0., f64::max)
                    / scale,
            );
            // The box vector alone, with no alias average.
            let plain: Vec<f64> = (0..n)
                .map(|at| {
                    let g = reciprocal::box_vector(&b, shape, at);
                    let g2 = g.iter().map(|v| v * v).sum::<f64>();
                    if g2 > 0. {
                        2. * coulomb(g) * g[a] * g[c] / g2
                    } else {
                        0.
                    }
                })
                .collect();
            unaliased = unaliased.max(worst(&plain, &want, &|_| false) / scale);
        }
        println!("{name}: dK {miss:.2e} of max {scale_all:.2e}; controls: sign {flipped:.2e}, no alias mean {unaliased:.2e}");
        assert!(miss < GATE, "{name}");
        assert!(flipped > 100. * GATE && unaliased > 100. * GATE, "{name}");
    }
}

#[test]
fn the_ion_strain_is_the_difference_of_the_ions_on_the_strained_grid() {
    let ctx = context();
    for name in ["si", "al2", "lih"] {
        let cell = Cell::load(&ctx, name);
        let n = cell.grid.index.points();
        let got = ions_strain(&ctx, &cell.grid, &cell.sites()).unwrap();
        let (mut miss, mut scale_all, mut flipped) = (0f64, 0f64, 0f64);
        for (row, &(a, c)) in VOIGT.iter().enumerate() {
            let side = |h: f64| -> Vec<Complex64> {
                let strained = cell.strained(entry(a, c, h));
                assert_pinned(&ctx, &cell, &strained);
                ion_charge(&ctx, &strained.grid, &strained.sites())
                    .unwrap()
                    .as_slice()
                    .to_vec()
            };
            let difference = |h: f64| -> Vec<Complex64> {
                side(h)
                    .iter()
                    .zip(side(-h))
                    .map(|(p, m)| (p - m) / (2. * h))
                    .collect()
            };
            let (one, two) = (difference(1e-4), difference(2e-4));
            let want: Vec<Complex64> = one
                .iter()
                .zip(&two)
                .map(|(x, y)| (x * 4. - y) / 3.)
                .collect();
            let have = &got.as_slice()[row * n..(row + 1) * n];
            let scale = maximum(&want);
            miss = miss.max(worst(have, &want, &|_| false) / scale);
            scale_all = scale_all.max(scale);
            let negated: Vec<Complex64> = have.iter().map(|x| -x).collect();
            flipped = flipped.max(worst(&negated, &want, &|_| false) / scale);
        }
        println!("{name}: dI {miss:.2e} of max {scale_all:.2e}; sign control {flipped:.2e}");
        assert!(scale_all > 1e-3 && miss < GATE, "{name}");
        assert!(flipped > 100. * GATE, "{name}");
    }
}

/// `h(G) = sum_c q_c hat_c(G)` of the cell on its own grid.
fn compensation_charge(
    ctx: &mandacaru_core::Context,
    cell: &Cell,
    shapes: &Shapes,
    q: &[Complex64],
) -> Vec<Complex64> {
    let n = cell.grid.index.points();
    let all = hats(ctx, &cell.grid, shapes, &cell.sites()).unwrap();
    (0..n)
        .map(|at| {
            (0..cell.channels.len())
                .map(|c| q[c] * all.as_slice()[c * n + at])
                .sum()
        })
        .collect()
}

#[test]
fn the_hat_strain_is_the_difference_of_the_hats_away_from_the_table_kinks() {
    let ctx = context();
    for name in ["si", "al2", "lih"] {
        let cell = Cell::load(&ctx, name);
        let n = cell.grid.index.points();
        let q_max = max_wavenumber(&ctx, &cell.grid).unwrap();
        let shapes = Shapes::new(&ctx, &cell.sites(), q_max).unwrap();
        let moments = cell.moments(5);
        let got = hats_strain(&ctx, &cell.grid, &shapes, &cell.sites(), &moments).unwrap();
        assert!(
            cell.channels.iter().any(|c| c.1 > 0),
            "{name}: the shapes must have a direction (L > 0) for the index order to matter"
        );
        let scale = maximum(got.as_slice());
        assert!(scale > 1e-3, "{name}");
        // Per step: the share of frequencies that straddle a node, the worst miss of the others
        // and of the straddling ones (the kink floor of the finite difference).
        for h in [1e-5, 5e-5, 2e-4] {
            let (mut kept_miss, mut kink_miss, mut kinked, mut total) =
                (0f64, 0f64, 0usize, 0usize);
            let (mut flipped, mut transposed) = (0f64, 0f64);
            for a in 0..3 {
                for b in 0..3 {
                    let (plus, minus) = (
                        cell.strained(entry(a, b, h)),
                        cell.strained(entry(a, b, -h)),
                    );
                    assert_pinned(&ctx, &cell, &plus);
                    assert_pinned(&ctx, &cell, &minus);
                    let (cp, cm) = (
                        compensation_charge(&ctx, &plus, &shapes, &moments),
                        compensation_charge(&ctx, &minus, &shapes, &moments),
                    );
                    let flags = tainted(&ctx, &plus.grid, &minus.grid);
                    kinked += flags.iter().filter(|f| **f).count();
                    total += n;
                    let want: Vec<Complex64> = cp
                        .iter()
                        .zip(&cm)
                        .map(|(x, y)| (x - y) / (2. * h))
                        .collect();
                    let have = &got.as_slice()[(a * 3 + b) * n..(a * 3 + b + 1) * n];
                    kept_miss = kept_miss.max(worst(have, &want, &|i| flags[i]) / scale);
                    kink_miss = kink_miss.max(worst(have, &want, &|i| !flags[i]) / scale);
                    let negated: Vec<Complex64> = have.iter().map(|x| -x).collect();
                    flipped = flipped.max(worst(&negated, &want, &|i| flags[i]) / scale);
                    let swapped = &got.as_slice()[(b * 3 + a) * n..(b * 3 + a + 1) * n];
                    transposed = transposed.max(worst(swapped, &want, &|i| flags[i]) / scale);
                }
            }
            println!(
                "{name} h = {h:.0e}: {kinked} of {total} elements straddle a node ({:.2}%); miss {kept_miss:.2e} elsewhere, {kink_miss:.2e} at the straddlers; controls: sign {flipped:.2e}, transposed {transposed:.2e}",
                100. * kinked as f64 / total as f64
            );
            if h == 1e-5 {
                assert!((kinked as f64) < 0.05 * total as f64, "{name}");
                assert!(kept_miss < TABLE_GATE, "{name}");
                assert!(
                    flipped > 100. * TABLE_GATE && transposed > 100. * TABLE_GATE,
                    "{name}"
                );
            }
        }
    }
}

#[test]
fn the_core_strain_is_the_difference_of_the_placed_core_on_the_strained_grid() {
    let ctx = context();
    let mut checked = 0;
    for name in ["si", "al2", "lih"] {
        let cell = Cell::load(&ctx, name);
        for (atom, table) in cell.tables.iter().enumerate() {
            let Some(core) = &table.core else { continue };
            checked += 1;
            let profile = |c: &Cell| {
                Cubic::new(
                    &ctx,
                    ctx.copy_slice(&table.radius).unwrap(),
                    ctx.copy_slice(core).unwrap(),
                )
                .map(|p| place(&ctx, &c.grid, &p, &[c.atoms[atom].center]).unwrap())
                .unwrap()
            };
            // The spline of a shipped table has third-derivative jumps (ten digits of data), so
            // the difference converges as the step squared until round-off: the ladder shows it.
            for h in [5e-5, 2e-5, 1e-5, 5e-6] {
                let (mut miss, mut scale_all, mut wrong_axes, mut flipped) =
                    (0f64, 0f64, 0f64, 0f64);
                for (a, b) in VOIGT {
                    let want: Vec<f64> = profile(&cell.strained(entry(a, b, h)))
                        .as_slice()
                        .iter()
                        .zip(profile(&cell.strained(entry(a, b, -h))).as_slice())
                        .map(|(p, m)| (p - m) / (2. * h))
                        .collect();
                    let field = |axes: (usize, usize)| {
                        core_strain(
                            &ctx,
                            &cell.grid,
                            &table.radius,
                            core,
                            &[cell.atoms[atom].center],
                            axes,
                        )
                        .unwrap()
                    };
                    let have = field((a, b));
                    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                    miss = miss.max(worst(have.as_slice(), &want, &|_| false) / scale);
                    scale_all = scale_all.max(scale);
                    let negated: Vec<f64> = have.as_slice().iter().map(|x| -x).collect();
                    flipped = flipped.max(worst(&negated, &want, &|_| false) / scale);
                    if a != b {
                        wrong_axes = wrong_axes
                            .max(worst(field((a, a)).as_slice(), &want, &|_| false) / scale);
                    }
                }
                println!("{name} atom {atom} h = {h:.0e}: dcore {miss:.2e} of max {scale_all:.2e}; controls: sign {flipped:.2e}, (a, a) for (a, b) {wrong_axes:.2e}");
                if h == CORE_STEP {
                    assert!(scale_all > 1e-4 && miss < TABLE_GATE, "{name}");
                    assert!(
                        flipped > 100. * TABLE_GATE && wrong_axes > 100. * TABLE_GATE,
                        "{name}"
                    );
                }
            }
        }
    }
    assert!(checked >= 2, "the Si and Al2 datasets carry a smooth core");
}
