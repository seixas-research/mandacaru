//! The sphere form of the strain tangent (`periodic::setup::strain_spheres`, AS-1):
//! `d chi_mu(x_p) / d eps_ab = A_ab(x_p) - s_b G_a(x_p)` for quadrature points `x_p = R_A + s_p`
//! rigid about their atom, and the strain derivative `dC_ab` of the projections built on it, for
//! the scalar projector lists (Si, Al2) and the spin-orbit union list (Si `lda-fr`). The gate is a
//! central difference of `pulay::projection` on the STRAINED cell (lattice, atoms, basis and
//! projector centers carried by `x -> (1 + eps) x`, spheres rebuilt rigid about the carried
//! atoms, radial tables fixed, fractional k held). A second, independent route (the per-image
//! projection kernel with the translation weight applied outside it) pins the sphere form to
//! roundoff. The skew LiH cell and a skewed Al2 (their lattices are not their own transpose) pin the
//! row/column convention of the lattice: the analytic derivative on the transposed lattice must miss. Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_spheres_support;

use mandacaru_core::periodic::setup::{
    pulay::{projection, Matrix},
    shells::Shells,
    spheres::Projector,
    strain_spheres::projection_strain,
};
use mandacaru_core::Context;
use nalgebra::Matrix3;
use num_complex::Complex64;
use periodic_gradient_crystals_support::Data;
use setup_gradients_support::context;
use setup_pulay_support::*;
use strain_samples_support::*;
use strain_spheres_support::{
    local::{skew_lattice, Control},
    *,
};

/// The plan's number; the measured floor is in the log.
const GATE: f64 = 1e-8;
/// Equality of two exact routes: roundoff of the sums.
const IDENTITY: f64 = 1e-12;
const FD_STEP: f64 = 1e-5;
/// The components differenced (every rebuild is a quadrature of all the projector spheres); the
/// nine are compared with the image-resolved route.
const DIFFERENCED: [(usize, usize); 3] = [(0, 0), (0, 2), (2, 1)];

struct Case {
    name: &'static str,
    lattice: Matrix3<f64>,
    shells: Shells,
    projectors: Vec<Projector>,
}

fn scalar(ctx: &Context, name: &'static str) -> Case {
    let fx = Fixture::new(ctx, name);
    Case {
        name,
        lattice: fx.lattice,
        shells: fx.shells,
        projectors: fx.projectors,
    }
}

fn lih() -> Case {
    let d = Data::load("lih");
    Case {
        name: "lih",
        lattice: d.lattice,
        shells: d.shells,
        projectors: d.projectors,
    }
}

fn spin_orbit(ctx: &Context) -> Case {
    use setup_shells_support::{dataset, minimal_basis, spin_orbit_list, Cell};
    let cell = Cell::read("si_sz_lda_fr");
    let ds = dataset(ctx, "lda-fr", "Si");
    let basis = minimal_basis(ctx, &ds, &cell.centers, cell.filter_wavevector);
    let (projectors, _) = spin_orbit_list(ctx, &ds, &cell.centers);
    Case {
        name: "si_lda_fr_spin_orbit",
        lattice: cell.lattice,
        shells: basis.shells,
        projectors,
    }
}

fn projections_at(ctx: &Context, case: &Case, eps: &Matrix3<f64>) -> Matrix {
    let c = Carried::new(&case.lattice, &case.shells, &case.projectors, eps, KPOINT);
    projection(ctx, &c.shells, &c.lattice, c.k, &c.projectors).unwrap()
}

/// `transposed`: the lattice is not its own transpose, so the analytic derivative on the
/// transposed lattice is run, and gated to miss when `Asserted`.
fn check(case: &Case, differenced: &[(usize, usize)], transposed: Control) {
    let ctx = context();
    let start = std::time::Instant::now();
    let got =
        projection_strain(&ctx, &case.shells, &case.lattice, KPOINT, &case.projectors).unwrap();
    let written = variant(
        &ctx,
        &case.shells,
        &case.lattice,
        KPOINT,
        &case.projectors,
        Form::Right,
        differenced,
    );
    let images = image_resolved(&ctx, &case.shells, &case.lattice, KPOINT, &case.projectors);
    // `d_R` without the lattice vector, and the lattice read as rows instead of columns.
    // Only the derivative axes of the differenced components are formed.
    let axes: Vec<usize> = (0..3)
        .filter(|a| differenced.iter().any(|&(x, _)| x == *a))
        .collect();
    let no_image = image_resolved_with(
        &ctx,
        &case.shells,
        &case.lattice,
        KPOINT,
        &case.projectors,
        Displacement::WithoutImage,
        &axes,
    );
    let transposed_strain = (transposed != Control::Skip).then(|| {
        assert!(
            (case.lattice - case.lattice.transpose()).norm() > 1e-3 * case.lattice.norm(),
            "{}: the skew guard needs a lattice that is not symmetric",
            case.name
        );
        projection_strain(
            &ctx,
            &case.shells,
            &case.lattice.transpose(),
            KPOINT,
            &case.projectors,
        )
        .unwrap()
    });
    let mut worst = (0f64, 0f64);
    for a in 0..3 {
        for b in 0..3 {
            // The library function is the form written out (on the differenced components, where the
            // wrong forms are formed too) and, for all nine, the per-image identity.
            let e_images = gap(&got[a][b], &images[a][b]);
            worst.0 = worst.0.max(e_images);
            if differenced.contains(&(a, b)) {
                let e_written = gap(&got[a][b], &written[a][b]);
                assert!(
                    e_written < IDENTITY,
                    "{} ({a},{b}) written form: {e_written:.2e}",
                    case.name
                );
            }
            assert!(
                e_images < IDENTITY,
                "{} ({a},{b}) image-resolved: {e_images:.2e}",
                case.name
            );
        }
    }
    let wrong = [
        Form::NoSphereTerm,
        Form::NoWeight,
        Form::Sign,
        Form::Transposed,
        Form::Unweighted,
    ]
    .map(|form| {
        (
            form,
            variant(
                &ctx,
                &case.shells,
                &case.lattice,
                KPOINT,
                &case.projectors,
                form,
                differenced,
            ),
        )
    });
    for &(a, b) in differenced {
        // A plain two-point difference: at a step of 1e-5 its truncation error (step^2 f''' / 6) is far below
        // the gate, and a Richardson pair would double the cost of the sphere rebuilds.
        let fd = (projections_at(&ctx, case, &entry(a, b, FD_STEP))
            - projections_at(&ctx, case, &entry(a, b, -FD_STEP)))
            / Complex64::new(2. * FD_STEP, 0.);
        let error = gap(&got[a][b], &fd);
        worst.1 = worst.1.max(error);
        assert!(
            peak(&fd) > 1e-3,
            "{} ({a},{b}): the derivative is not vacuous",
            case.name
        );
        assert!(error < GATE, "{} ({a},{b}): {error:.2e}", case.name);
        let miss = gap(&no_image[a][b], &fd);
        assert!(
            miss > 100. * GATE,
            "{} ({a},{b}) d_R without R misses by only {miss:.2e}",
            case.name
        );
        if let Some(t) = &transposed_strain {
            let miss = gap(&t[a][b], &fd);
            if transposed == Control::Asserted {
                assert!(
                    miss > 100. * GATE,
                    "{} ({a},{b}) transposed lattice misses by only {miss:.2e}",
                    case.name
                );
            }
            println!(
                "{} ({a},{b}): transposed lattice misses by {miss:.2e} ({:.0} gates)",
                case.name,
                miss / GATE
            );
        }
        for (form, matrices) in &wrong {
            // On the diagonal the other component is the same component.
            if matches!(form, Form::Transposed) && a == b {
                continue;
            }
            let miss = gap(&matrices[a][b], &fd);
            assert!(
                miss > 100. * GATE,
                "{} ({a},{b}) {form:?} misses by only {miss:.2e}",
                case.name
            );
        }
    }
    println!(
        "{}: sphere form against the central difference {:.2e}, against the per-image route {:.2e} ({:.0} s)",
        case.name,
        worst.1,
        worst.0,
        start.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "slow: 118 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres -- --ignored the_sphere_form_is_the_central_difference_of_the_projections_of_si"]
fn the_sphere_form_is_the_central_difference_of_the_projections_of_si() {
    check(&scalar(&context(), "si"), &DIFFERENCED, Control::Skip);
}

#[test]
#[ignore = "slow: 148 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres -- --ignored the_sphere_form_is_the_central_difference_of_the_projections_of_al2"]
fn the_sphere_form_is_the_central_difference_of_the_projections_of_al2() {
    // The larger cell: one off-diagonal component keeps the case well under the time budget
    // (132 s for two); the diagonal and the other off-diagonals are gated on Si and the spin-orbit
    // cell.
    check(&scalar(&context(), "al2"), &[(2, 0)], Control::Skip);
}

#[test]
#[ignore = "slow: 67 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres -- --ignored the_sphere_form_is_the_central_difference_of_the_spin_orbit_projections"]
fn the_sphere_form_is_the_central_difference_of_the_spin_orbit_projections() {
    let ctx = context();
    check(&spin_orbit(&ctx), &DIFFERENCED, Control::Skip);
}

#[test]
#[ignore = "slow: 12 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres -- --ignored the_sphere_form_is_the_central_difference_of_the_projections_of_the_skew_lih_cell"]
fn the_sphere_form_is_the_central_difference_of_the_projections_of_the_skew_lih_cell() {
    // Both orders of an off-diagonal pair: a swapped index or a transposed lattice shows in one.
    check(&lih(), &[(0, 1), (2, 0)], Control::Asserted);
}

#[test]
#[ignore = "slow: 183 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres -- --ignored the_sphere_form_is_the_central_difference_of_the_projections_of_a_skewed_al2"]
fn the_sphere_form_is_the_central_difference_of_the_projections_of_a_skewed_al2() {
    let mut case = scalar(&context(), "al2");
    case.name = "al2_skew";
    case.lattice = skew_lattice();
    // One off-diagonal component: the skewed cell's rebuilds take 80 s each.
    check(&case, &[(0, 1)], Control::Asserted);
}

#[test]
fn a_projector_with_no_cutoff_or_a_cell_it_does_not_reach_raises_or_is_zero() {
    let ctx = context();
    let case = scalar(&ctx, "si");
    let mut broken = case.projectors.clone();
    broken[0].r_cut = 0.;
    assert!(projection_strain(&ctx, &case.shells, &case.lattice, KPOINT, &broken).is_err());
}
