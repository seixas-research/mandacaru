//! The strain derivative of the short-range ion-compensation sum
//! (`periodic::setup::strain_spheres::linear_strain`, AS-2, plan E-6): the compensation sphere of
//! atom `A` is rigid, the ion `B` in the image `T` sits at `Delta = R_B + T - R_A` and the explicit
//! image-potential term `d v^sr(|y|)/d eps_ab = -(v^sr)'(|y|) y_a Delta_b / |y|` carries the OWN
//! images `T != 0`, which an atom displacement cancels and a strain does not.
//!
//! The gate is a Richardson central difference of `short_range_ion_compensation` over the
//! carried `Sites` (lattice and atoms by `x -> (1 + eps) x`, the Gaussian width, the compensation
//! radii and the sphere rule fixed), with the image list asserted unchanged at every strained
//! point (a pinned list). The on-site long-range integral that `linear` subtracts is a radial
//! integral at a pinned radius and width: it has no strain derivative. The controls compare a
//! WRONG variant with the difference: the own images skipped (right for forces), the sign, the
//! long-range slope not subtracted, and the lattice read as rows (asserted on the skewed Al2; on
//! LiH only two neighbors reach a sphere, so it is reported there). Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_spheres_support;

use mandacaru_core::periodic::setup::strain_spheres::linear_strain;
use num_complex::Complex64;
use setup_gradients_support::context;
use strain_spheres_support::{
    linear::{linear_derivative, linear_variant, members, Switch},
    local::{Control, Local},
};

/// The plan's number for the sphere sum of the ions.
const GATE: f64 = 1e-7;
/// Equality of two exact routes.
const IDENTITY: f64 = 1e-12;
/// The step of the Richardson pair (`FD_STEP`, `2 FD_STEP`). The error of the difference is
/// roundoff over the step and falls with it: worst absolute error over the nine components, Si
/// 6.5e-12 / 3.9e-12 / 1.5e-12 / 1.5e-12 and Al2 3.8e-12 / 3.5e-12 / 6.1e-13 / 6.3e-13 at the steps
/// 5e-6 / 1e-5 / 2e-5 / 4e-5, a plateau from 2e-5 on.
const FD_STEP: f64 = 2e-5;

type Tensor = Vec<[[Complex64; 3]; 3]>;

fn peak(t: &Tensor) -> f64 {
    t.iter()
        .flatten()
        .flatten()
        .map(|x| x.norm())
        .fold(0., f64::max)
}

/// Largest `|a - b|` over the channels and the nine components (absolute, Hartree per unit
/// strain times the moment of the channel).
fn gap(a: &Tensor, b: &Tensor) -> f64 {
    let mut worst = 0f64;
    for (x, y) in a.iter().zip(b) {
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((x[i][j] - y[i][j]).norm());
            }
        }
    }
    worst
}

/// The gate: the plan's 1e-7 of the term's maximum.
fn gate(fd: &Tensor) -> f64 {
    GATE * peak(fd)
}

fn check(local: &Local, own_images: Control, transposed: Control) {
    let ctx = context();
    let start = std::time::Instant::now();
    let sites = local.cell.sites();
    let got = linear_strain(&ctx, &sites, &local.potentials).unwrap();
    // The image list is a pin: no member changes at any strained point of the difference.
    let base = members(&sites, &local.potentials);
    let own = base.iter().filter(|m| m.0 == m.1).count();
    // The control "own images skipped" is vacuous unless some own image reaches a sphere.
    assert!(
        own > 0,
        "{}: no own image reaches a compensation sphere",
        local.name
    );
    let mut fd: Tensor = vec![[[Complex64::new(0., 0.); 3]; 3]; got.len()];
    for a in 0..3 {
        for b in 0..3 {
            for sign in [-2., -1., 1., 2.] {
                let mut eps = [[0.; 3]; 3];
                eps[a][b] = sign * FD_STEP;
                assert_eq!(
                    members(&local.cell.strained(eps).sites(), &local.potentials),
                    base,
                    "{}: the image list changed at eps_{a}{b} = {:.0e}",
                    local.name,
                    sign * FD_STEP
                );
            }
            for (row, value) in linear_derivative(&ctx, local, (a, b), FD_STEP)
                .iter()
                .enumerate()
            {
                fd[row][a][b] = *value;
            }
        }
    }
    assert!(
        peak(&fd) > 1e-6,
        "{}: the derivative is not vacuous",
        local.name
    );
    let gate = gate(&fd);
    // The library is the variant with everything switched on.
    let all = Switch {
        own_images: true,
        long_range: true,
    };
    let written = gap(&got, &linear_variant(&ctx, &sites, &local.potentials, all));
    assert!(
        written < IDENTITY,
        "{} written form: {written:.2e}",
        local.name
    );
    let error = gap(&got, &fd);
    assert!(
        error < gate,
        "{}: {error:.2e} (gate {gate:.2e})",
        local.name
    );
    let bite = |label: &str, miss: f64, how: Control| {
        if how == Control::Asserted {
            assert!(
                miss > 100. * gate,
                "{} {label} misses by only {miss:.2e} (gate {gate:.2e})",
                local.name
            );
        }
        println!(
            "{}: {label} misses by {miss:.2e} ({:.0} gates)",
            local.name,
            miss / gate
        );
    };
    let negated: Tensor = got.iter().map(|t| t.map(|r| r.map(|x| -x))).collect();
    bite("sign", gap(&negated, &fd), Control::Asserted);
    let variant = |own_images, long_range| {
        let switch = Switch {
            own_images,
            long_range,
        };
        gap(
            &linear_variant(&ctx, &sites, &local.potentials, switch),
            &fd,
        )
    };
    bite(
        "long-range slope kept",
        variant(true, false),
        Control::Asserted,
    );
    bite("own images skipped", variant(false, true), own_images);
    if transposed != Control::Skip {
        let l = local.cell.lattice;
        assert!(
            (l - l.transpose()).norm() > 1e-3 * l.norm(),
            "{}: the skew guard needs a lattice that is not symmetric",
            local.name
        );
        let mut cell = local.cell.clone();
        cell.lattice = l.transpose();
        let t = linear_strain(&ctx, &cell.sites(), &local.potentials).unwrap();
        bite("transposed lattice", gap(&t, &fd), transposed);
    }
    println!(
        "{}: linear strain against the central difference {error:.2e} (gate {gate:.2e}, peak {:.2e}, own images {own}, {:.0} s)",
        local.name,
        peak(&fd),
        start.elapsed().as_secs_f64()
    );
}

#[test]
fn the_linear_strain_is_the_central_difference_of_the_ion_compensation_of_si() {
    check(
        &Local::wpx(&context(), "si"),
        Control::Asserted,
        Control::Skip,
    );
}

#[test]
fn the_linear_strain_is_the_central_difference_of_the_ion_compensation_of_al2() {
    check(
        &Local::wpx(&context(), "al2"),
        Control::Asserted,
        Control::Skip,
    );
}

#[test]
fn the_linear_strain_is_the_central_difference_of_the_ion_compensation_of_the_skew_lih_cell() {
    // Two own images sit at the edge of the reach and only two neighbors reach a sphere across
    // the 11 Bohr box: neither control has much to bite on, both are reported.
    check(&Local::lih(), Control::Reported, Control::Reported);
}

#[test]
fn the_linear_strain_is_the_central_difference_of_the_ion_compensation_of_a_skewed_al2() {
    // The own images and the neighbors of the skewed tetragonal cell reach every sphere: the
    // row/column convention of the lattice shows.
    check(
        &Local::al2_skew(&context()),
        Control::Asserted,
        Control::Asserted,
    );
}
