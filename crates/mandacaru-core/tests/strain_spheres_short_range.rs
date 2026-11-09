//! The strain derivative of the short-range local matrix of an atom's sphere
//! (`periodic::setup::strain_spheres::short_range_strain`, AS-2): `V_sr^A` is the atom's OWN radial
//! potential on its rigid quadrature sphere, so only the two functions move,
//! `dV_ab = sum_x w v [conj(d chi_mu) chi_nu + conj(chi_mu) d chi_nu]` with
//! `d chi = A_ab - s_b G_a`, and there is no potential term (`du_A = 0`).
//!
//! The gate is a central difference of `short_range_cross` on the STRAINED cell (lattice, atoms
//! and basis centers carried by `x -> (1 + eps) x`, the sphere rebuilt rigid about the carried
//! atom, the radial tables and the potential fixed, the fractional k held). The controls compare
//! a WRONG variant with the same difference: the plan's pre-correction `du_A` (the image
//! potentials of the other atoms) added to `V_sr`, the grid form `A_ab` used on the sphere, the
//! image weight or the sign wrong, the other component (reported on LiH, whose derivative is nearly
//! symmetric), and, on the skew LiH cell and a skewed Al2, the lattice read as rows. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod setup_gradients_support;
mod setup_pulay_support;
mod setup_shells_support;
mod strain_samples_support;
mod strain_spheres_support;

use mandacaru_core::periodic::setup::{pulay::Matrix, strain_spheres::short_range_strain};
use num_complex::Complex64;
use setup_gradients_support::context;
use setup_pulay_support::{gap, peak, KPOINT};
use strain_spheres_support::{
    local::{image_potential_term, v_sr_derivative, v_variant, Control, Local},
    Form,
};

/// The plan's number for the matrix terms.
const GATE: f64 = 1e-8;
/// Equality of two exact routes: roundoff of the sums.
const IDENTITY: f64 = 1e-12;
/// The step of the Richardson pair (`FD_STEP`, `2 FD_STEP`). The one-step difference at 1e-5 has a
/// noise of 1e-8 on the H sphere of LiH (the radial tables of the basis are cubic splines with
/// jumps of the third derivative that the moving points cross), 4e-9 at 2e-5.
const FD_STEP: f64 = 2e-5;

/// `swapped`: the control that reads component `(b, a)` for `(a, b)`; it can only bite when the
/// strain derivative of the matrix is not symmetric in `a, b` (the antisymmetric part is a rotation
/// of the neighbors about the sphere, and is tiny where no neighbor reaches the sphere).
fn check(local: &Local, pairs: &[(usize, usize)], transposed: Control, swapped: Control) {
    let ctx = context();
    let start = std::time::Instant::now();
    let spheres = local.spheres();
    let got =
        short_range_strain(&ctx, &local.shells, &local.cell.lattice, KPOINT, &spheres).unwrap();
    let transposed_strain = (transposed != Control::Skip).then(|| {
        let l = local.cell.lattice;
        assert!(
            (l - l.transpose()).norm() > 1e-3 * l.norm(),
            "{}: the skew guard needs a lattice that is not symmetric",
            local.name
        );
        short_range_strain(&ctx, &local.shells, &l.transpose(), KPOINT, &spheres).unwrap()
    });
    let mut worst = (0f64, 0f64);
    // The smallest miss of the controls that must bite, in units of the gate.
    let (mut least_du, mut least_form) = (f64::INFINITY, f64::INFINITY);
    for (atom, sphere) in spheres.iter().enumerate() {
        // The library function is the form written out, on the pairs where the wrong forms are formed.
        let right = v_variant(
            &ctx,
            &local.shells,
            &local.cell.lattice,
            KPOINT,
            sphere,
            Form::Right,
            pairs,
        );
        let wrong: Vec<(Form, [[Matrix; 3]; 3])> = [
            Form::NoSphereTerm,
            Form::NoWeight,
            Form::Sign,
            Form::Transposed,
            Form::Unweighted,
        ]
        .map(|form| {
            let m = v_variant(
                &ctx,
                &local.shells,
                &local.cell.lattice,
                KPOINT,
                sphere,
                form,
                pairs,
            );
            (form, m)
        })
        .into();
        let du = image_potential_term(&ctx, local, atom, KPOINT, pairs);
        for &(a, b) in pairs {
            let hermitian = peak(&(&got[atom][a][b] - got[atom][a][b].adjoint()));
            assert!(
                hermitian < 1e-12 * peak(&got[atom][a][b]).max(1.),
                "{} atom {atom} ({a},{b}) is not Hermitian: {hermitian:.2e}",
                local.name
            );
            let written = gap(&got[atom][a][b], &right[a][b]);
            assert!(
                written < IDENTITY,
                "{} atom {atom} ({a},{b}) written form: {written:.2e}",
                local.name
            );
            worst.0 = worst.0.max(written);
        }
        for &(a, b) in pairs {
            let fd = &v_sr_derivative(&ctx, local, KPOINT, (a, b), FD_STEP)[atom];
            let error = gap(&got[atom][a][b], fd);
            worst.1 = worst.1.max(error);
            assert!(
                peak(fd) > 1e-3,
                "{} atom {atom} ({a},{b}): the derivative is not vacuous ({:.2e})",
                local.name,
                peak(fd)
            );
            assert!(
                error < GATE,
                "{} atom {atom} ({a},{b}): {error:.2e}",
                local.name
            );
            for (form, matrices) in &wrong {
                if matches!(form, Form::Transposed) && a == b {
                    continue;
                }
                let miss = gap(&matrices[a][b], fd);
                if !matches!(form, Form::Transposed) || swapped == Control::Asserted {
                    least_form = least_form.min(miss / GATE);
                    assert!(
                        miss > 100. * GATE,
                        "{} atom {atom} ({a},{b}) {form:?} misses by only {miss:.2e}",
                        local.name
                    );
                } else {
                    println!(
                        "{} atom {atom} ({a},{b}): components swapped misses by {miss:.2e} ({:.0} gates)",
                        local.name,
                        miss / GATE
                    );
                }
            }
            // The potential term that does not belong here.
            let with_du = &got[atom][a][b] + &du[a][b];
            let miss = gap(&with_du, fd);
            least_du = least_du.min(miss / GATE);
            assert!(
                miss > 100. * GATE,
                "{} atom {atom} ({a},{b}) du_A added misses by only {miss:.2e}",
                local.name
            );
            let miss = gap(&(&got[atom][a][b] * Complex64::new(-1., 0.)), fd);
            assert!(
                miss > 100. * GATE,
                "{} atom {atom} ({a},{b}) sign: {miss:.2e}",
                local.name
            );
            if let Some(t) = &transposed_strain {
                let miss = gap(&t[atom][a][b], fd);
                if transposed == Control::Asserted {
                    assert!(
                        miss > 100. * GATE,
                        "{} atom {atom} ({a},{b}) transposed lattice misses by only {miss:.2e}",
                        local.name
                    );
                }
                println!(
                    "{} atom {atom} ({a},{b}): transposed lattice misses by {miss:.2e} ({:.0} gates)",
                    local.name,
                    miss / GATE
                );
            }
        }
    }
    println!(
        "{}: short-range strain against the central difference {:.2e}, against the written form {:.2e}; du_A added misses by >= {least_du:.0} gates, wrong forms by >= {least_form:.0} gates ({:.0} s)",
        local.name,
        worst.1,
        worst.0,
        start.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "slow: 47 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres_short_range -- --ignored the_short_range_strain_is_the_central_difference_of_the_sphere_matrix_of_si"]
fn the_short_range_strain_is_the_central_difference_of_the_sphere_matrix_of_si() {
    check(
        &Local::wpx(&context(), "si"),
        &[(0, 0), (0, 2), (2, 1)],
        Control::Skip,
        Control::Asserted,
    );
}

#[test]
#[ignore = "slow: 61 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres_short_range -- --ignored the_short_range_strain_is_the_central_difference_of_the_sphere_matrix_of_al2"]
fn the_short_range_strain_is_the_central_difference_of_the_sphere_matrix_of_al2() {
    check(
        &Local::wpx(&context(), "al2"),
        &[(2, 0), (1, 1)],
        Control::Skip,
        Control::Asserted,
    );
}

#[test]
fn the_short_range_strain_is_the_central_difference_of_the_sphere_matrix_of_the_skew_lih_cell() {
    // The lattice read as rows moves the images that reach the spheres: asserted. The swapped
    // components differ only by the antisymmetric part of the derivative, a rotation of the
    // neighbors about the sphere, which is tiny across the 11 Bohr box: reported.
    check(
        &Local::lih(),
        &[(0, 1), (2, 0)],
        Control::Asserted,
        Control::Reported,
    );
}

#[test]
#[ignore = "slow: 71 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test strain_spheres_short_range -- --ignored the_short_range_strain_is_the_central_difference_of_the_sphere_matrix_of_a_skewed_al2"]
fn the_short_range_strain_is_the_central_difference_of_the_sphere_matrix_of_a_skewed_al2() {
    check(
        &Local::al2_skew(&context()),
        &[(0, 1), (2, 0)],
        Control::Asserted,
        Control::Asserted,
    );
}
