//! The +U strain contraction against the independent oracle `hubbard_gradient::Gradient::strain`
//! (AS-2, plan section 7): the explicit +U term `dE_U/d eps_ab = 2 w Re sum dC_ab[mu, p]
//! (O C^+ D)[p, mu]` built on the library's `projection_strain` equals the layer's image-resolved
//! sum `sum_R e^{-ik.R} G^a_R d_R^b`, whose real image blocks the TEST supplies.
//!
//! The analytic-`C(X)` chain of `periodic_hubbard_gradient.rs` has no `Shells` to differentiate, so
//! the toy here is a real cell (Si, Al2, the skew LiH cell) with its basis and projectors
//! restricted to the `m = 0` components (their image blocks are real: asserted) and synthetic
//! Hubbard data: shell weights, Hermitian shell potentials, Hermitian densities (two k-points
//! on Si and LiH, one on the Al2 cells, whose sphere rebuilds are the slowest). The gate is the plan's 1e-10 of the toy. The controls: the image vector dropped from
//! `d_R` (through the layer's seam), the sign, the grid form `A_ab` used on the sphere, and the
//! lattice read as rows (asserted on the skew LiH cell and the skewed Al2). Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_spheres_support;

use mandacaru_core::periodic::{
    hubbard_gradient::displacement,
    setup::{
        pulay::{projection, Matrix},
        shells::Shells,
        spheres::Projector,
        strain_spheres::projection_strain,
    },
};
use mandacaru_core::Context;
use nalgebra::Matrix3;
use periodic_gradient_crystals_support::Data;
use setup_gradients_support::context;
use setup_pulay_support::{density, Fixture, KPOINT};
use strain_spheres_support::{
    hubbard::{build, without_image, Toy},
    local::{skew_lattice, Control},
    variant, Form,
};

/// The plan's number for the +U toy.
const GATE: f64 = 1e-10;

/// Two k-points (weights 1/3 and 2/3) for the cells that are cheap; one for the Al2 cells, whose
/// sphere rebuilds are the slowest.
fn kpoints(count: usize) -> Vec<[f64; 3]> {
    [KPOINT, [-0.0612, 0.1127, 0.0431]][..count].to_vec()
}

/// A deterministic Hermitian density per k-point, the second the conjugate of the first.
fn density_at(m: usize) -> impl Fn(usize) -> Matrix {
    move |k| {
        let d = density(m);
        if k == 0 {
            d
        } else {
            d.conjugate()
        }
    }
}

fn si_like(ctx: &Context, name: &'static str, count: usize) -> Toy {
    let fx = Fixture::new(ctx, name);
    let m = count_zero(&fx.shells);
    build(
        ctx,
        fx.lattice,
        &fx.shells,
        &fx.owner,
        &fx.projectors,
        fx.atoms(),
        kpoints(count),
        density_at(m),
    )
}

fn lih(ctx: &Context) -> Toy {
    let d = Data::load("lih");
    let m = count_zero(&d.shells);
    build(
        ctx,
        d.lattice,
        &d.shells,
        &d.owner,
        &d.projectors,
        d.atoms.len(),
        kpoints(2),
        density_at(m),
    )
}

/// Functions with `m = 0`.
fn count_zero(shells: &Shells) -> usize {
    shells
        .shells()
        .iter()
        .map(|s| s.ms.iter().filter(|&&m| m == 0).count())
        .sum()
}

fn projections(ctx: &Context, toy: &Toy, lattice: &Matrix3<f64>) -> Vec<Matrix> {
    toy.kpoints
        .iter()
        .map(|&k| projection(ctx, &toy.shells, lattice, k, &toy.projectors).unwrap())
        .collect()
}

fn strain_of(
    ctx: &Context,
    toy: &Toy,
    lattice: &Matrix3<f64>,
    projectors: &[Projector],
) -> Vec<[[Matrix; 3]; 3]> {
    toy.kpoints
        .iter()
        .map(|&k| projection_strain(ctx, &toy.shells, lattice, k, projectors).unwrap())
        .collect()
}

fn gap(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut worst = 0f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[i][j] - b[i][j]).abs());
        }
    }
    worst
}

fn check(name: &str, toy: &Toy, transposed: Control) {
    let ctx = context();
    let start = std::time::Instant::now();
    assert!(
        toy.imaginary < 1e-12,
        "{name}: the m = 0 image blocks are not real ({:.2e})",
        toy.imaginary
    );
    assert!(
        toy.images.len() > 1,
        "{name}: the toy must reach neighbor cells"
    );
    let c = projections(&ctx, toy, &toy.lattice);
    let layer = toy.layer(&c, displacement);
    let scale = layer.iter().flatten().fold(0f64, |m, x| m.max(x.abs()));
    assert!(
        scale > 1e-3,
        "{name}: the +U strain is not vacuous ({scale:.2e})"
    );
    let library = toy.contract(&c, &strain_of(&ctx, toy, &toy.lattice, &toy.projectors));
    let error = gap(&library, &layer);
    assert!(
        error < GATE,
        "{name}: the contraction differs by {error:.2e}"
    );
    // The controls, each against the layer.
    let bite = |label: &str, wrong: &[[f64; 3]; 3]| {
        let miss = gap(wrong, &layer);
        assert!(
            miss > 100. * GATE,
            "{name}: {label} misses by only {miss:.2e}"
        );
        miss
    };
    bite("d_R without R", &toy.layer(&c, without_image));
    bite("sign", &library.map(|row| row.map(|x| -x)));
    // The grid form `A_ab` on the sphere (the atom's own displacement dropped).
    // Three components keep the cost down: the contraction of the other six is not formed, so the
    // comparison is over the entries formed.
    let pairs = [(0, 1), (2, 0), (1, 1)];
    let grid_form: Vec<[[Matrix; 3]; 3]> = toy
        .kpoints
        .iter()
        .map(|&k| {
            variant(
                &ctx,
                &toy.shells,
                &toy.lattice,
                k,
                &toy.projectors,
                Form::NoSphereTerm,
                &pairs,
            )
        })
        .collect();
    let formed = toy.contract(&c, &grid_form);
    let miss = pairs
        .iter()
        .map(|&(a, b)| (formed[a][b] - layer[a][b]).abs())
        .fold(0f64, f64::max);
    assert!(
        miss > 100. * GATE,
        "{name}: grid form on the sphere misses by only {miss:.2e}"
    );
    if transposed != Control::Skip {
        let l = toy.lattice;
        assert!(
            (l - l.transpose()).norm() > 1e-3 * l.norm(),
            "{name}: the skew guard needs a lattice that is not symmetric"
        );
        let t = l.transpose();
        let wrong = toy.contract(&c, &strain_of(&ctx, toy, &t, &toy.projectors));
        let miss = gap(&wrong, &layer);
        if transposed == Control::Asserted {
            bite("transposed lattice", &wrong);
        }
        println!(
            "{name}: transposed lattice misses by {miss:.2e} ({:.0} gates)",
            miss / GATE
        );
    }
    println!(
        "{name}: +U strain from projection_strain against the layer {error:.2e} (scale {scale:.2e}, {} images, {:.0} s)",
        toy.images.len(),
        start.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "slow: 67 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres_hubbard -- --ignored the_plus_u_strain_of_si_equals_the_layer"]
fn the_plus_u_strain_of_si_equals_the_layer() {
    check("si", &si_like(&context(), "si", 2), Control::Skip);
}

#[test]
#[ignore = "slow: 62 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres_hubbard -- --ignored the_plus_u_strain_of_al2_equals_the_layer"]
fn the_plus_u_strain_of_al2_equals_the_layer() {
    check("al2", &si_like(&context(), "al2", 1), Control::Skip);
}

#[test]
fn the_plus_u_strain_of_the_skew_lih_cell_equals_the_layer() {
    check("lih", &lih(&context()), Control::Asserted);
}

#[test]
#[ignore = "slow: 92 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres_hubbard -- --ignored the_plus_u_strain_of_a_skewed_al2_equals_the_layer"]
fn the_plus_u_strain_of_a_skewed_al2_equals_the_layer() {
    let ctx = context();
    let fx = Fixture::new(&ctx, "al2");
    let toy = build(
        &ctx,
        skew_lattice(),
        &fx.shells,
        &fx.owner,
        &fx.projectors,
        fx.atoms(),
        kpoints(1),
        density_at(count_zero(&fx.shells)),
    );
    check("al2_skew", &toy, Control::Asserted);
}
