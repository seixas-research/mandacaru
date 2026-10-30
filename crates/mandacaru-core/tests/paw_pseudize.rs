use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::pseudize::{
    inner_overlaps, matching_targets, norm_targets, optimize, snap, Equation, NormCondition,
    PseudoInput,
};
use mandacaru_core::paw::pseudize_tables::{
    bessel_table, gram, inner_grid, linspace, q_grid, resample, tail_transform, INNER_POINTS,
};
use mandacaru_core::paw::smooth::{bessel, wavevectors};
use mandacaru_core::Context;
use std::f64::consts::PI;

const NONRELATIVISTIC: Equation = Equation {
    treatment: Treatment::None,
    kappa: None,
    z: 1.,
};

fn grid() -> Vec<f64> {
    (1..=6000).map(|i| i as f64 * 0.005).collect()
}
fn hydrogen(r: &[f64]) -> (Vec<f64>, Vec<f64>) {
    (
        r.iter().map(|x| 2. * (-x).exp()).collect(),
        r.iter().map(|x| -1. / x).collect(),
    )
}

#[test]
fn bessel_derivatives_obey_the_bessel_equation_and_the_zeros_interlace() {
    for l in 0..4 {
        for x in [0.7, 2.3, 9.1, 31.4] {
            let [j, d1, d2, d3] = bessel(l, x);
            let ll = (l * (l + 1)) as f64;
            assert!((x * x * d2 + 2. * x * d1 + (x * x - ll) * j).abs() < 1e-12 * (1. + x * x));
            let h = 1e-5;
            let fd = (bessel(l, x + h)[2] - bessel(l, x - h)[2]) / (2. * h);
            assert!((fd - d3).abs() < 1e-6);
        }
        let qs = wavevectors(l, 1.3, 8).unwrap();
        assert!(qs.windows(2).all(|w| w[1] > w[0]));
        let x = qs.iter().map(|q| q * 1.3).collect::<Vec<_>>();
        // Every wave vector is a zero of the value or of the slope, and the two kinds alternate.
        let kinds = x
            .iter()
            .map(|x| {
                let [value, slope, ..] = bessel(l, *x);
                assert!(value.abs().min(slope.abs()) < 1e-10, "l = {l}, x = {x}");
                value.abs() < slope.abs()
            })
            .collect::<Vec<_>>();
        assert!(kinds.windows(2).all(|w| w[0] != w[1]), "l = {l}");
    }
}

#[test]
fn the_tables_agree_with_closed_forms() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    assert_eq!(inner_grid(1.7).len(), INNER_POINTS);
    assert_eq!(q_grid().len(), 601);
    assert_eq!(linspace(0., 1., 5), vec![0., 0.25, 0.5, 0.75, 1.]);
    // The s-wave Gram integral is analytic: int_0^R j0(a r) j0(b r) r^2 dr.
    let (a, b, rc) = (PI / 1.7, 2. * PI / 1.7, 1.7);
    let r_in = inner_grid(rc);
    let g = gram(0, &[a, b], &r_in).unwrap();
    let expected = rc
        * rc
        * (a * bessel(1, a * rc)[0] * bessel(0, b * rc)[0]
            - b * bessel(0, a * rc)[0] * bessel(1, b * rc)[0])
        / (a * a - b * b);
    assert!((g[1] - expected).abs() < 1e-9);
    assert!((g[1] - g[2]).abs() < 1e-15);
    // Cubic resampling reproduces a cubic exactly (not-a-knot), including extrapolation.
    let r = (1..=200).map(|i| i as f64 * 0.01).collect::<Vec<_>>();
    let cubic = r
        .iter()
        .map(|x| 1. + x - 2. * x * x + 0.5 * x.powi(3))
        .collect::<Vec<_>>();
    let new = [0.0, 0.123, 1.5];
    let out = resample(&ctx, &r, &cubic, &new).unwrap();
    for (x, v) in new.iter().zip(&out) {
        assert!((v - (1. + x - 2. * x * x + 0.5 * x.powi(3))).abs() < 1e-9);
    }
    assert_eq!(bessel_table(0, &[1., 2.], &[0.5, 1.]).len(), 4);
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn matching_targets_are_the_derivatives_of_the_wave() {
    let r = grid();
    let (radial, v) = hydrogen(&r);
    let u = r
        .iter()
        .zip(&radial)
        .map(|(r, w)| r * w)
        .collect::<Vec<_>>();
    let rc = snap(&r, 1.2);
    let t = matching_targets(&r, &u, &v, 0, -0.5, rc, &NONRELATIVISTIC).unwrap();
    let e = 2. * (-rc).exp();
    for (k, value) in t.iter().enumerate() {
        let expected = if k % 2 == 0 { e } else { -e };
        assert!(
            (value - expected).abs() < 2e-5 * e,
            "derivative {k}: {value} vs {expected}"
        );
    }
    assert!(matching_targets(&r, &u, &v, 0, -0.5, r[0], &NONRELATIVISTIC).is_err());
}

#[test]
fn inner_overlaps_and_targets_match_the_analytic_hydrogen_norm() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let r = grid();
    let (radial, v) = hydrogen(&r);
    let rc = snap(&r, 1.3);
    let s = inner_overlaps(&ctx, &r, &radial, 1, rc).unwrap()[(0, 0)];
    assert!((s - (1. - (-2. * rc).exp() * (1. + 2. * rc + 2. * rc * rc))).abs() < 1e-8);
    let t = norm_targets(&ctx, &r, &radial, &[-0.5], rc, &v, &NONRELATIVISTIC).unwrap();
    assert_eq!(t[(0, 0)], s);
}

fn tilde(
    waves: &[f64],
    energies: &[f64],
    norm: NormCondition,
    ctx: &Context,
) -> mandacaru_core::paw::pseudize::PseudoWaves {
    let r = grid();
    let (_, v) = hydrogen(&r);
    optimize(
        ctx,
        &PseudoInput {
            r: &r,
            v_ae: &v,
            l: 0,
            waves,
            energies,
            r_cut: 1.3,
            q_cut: 5.,
            n_bessel: 8,
            norm,
            equation: NONRELATIVISTIC,
        },
    )
    .unwrap()
}

#[test]
fn the_smooth_wave_matches_value_slopes_and_norm() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let r = grid();
    let (radial, v) = hydrogen(&r);
    let rc = snap(&r, 1.3);
    let u = r
        .iter()
        .zip(&radial)
        .map(|(r, w)| r * w)
        .collect::<Vec<_>>();
    let target = matching_targets(&r, &u, &v, 0, -0.5, rc, &NONRELATIVISTIC).unwrap();
    for (norm, factor) in [
        (NormCondition::Factor(1.), 1.),
        (NormCondition::Factor(0.9), 0.9),
        (NormCondition::Free, f64::NAN),
    ] {
        let pw = tilde(&radial, &[-0.5], norm, &ctx);
        let derivs = pw
            .wavevectors
            .iter()
            .map(|q| bessel(0, q * rc))
            .collect::<Vec<_>>();
        for row in 0..4 {
            let value = (0..8)
                .map(|a| {
                    pw.coefficients[(0, a)] * pw.wavevectors[a].powi(row as i32) * derivs[a][row]
                })
                .sum::<f64>();
            assert!(
                (value - target[row]).abs() < 1e-7 * (1. + target[row].abs()),
                "row {row}"
            );
        }
        if factor.is_finite() {
            assert!(
                (pw.achieved[(0, 0)] - factor * pw.norms[(0, 0)]).abs() < 1e-9,
                "norm {factor}"
            );
        }
        assert!(pw.residuals[0] > 0. && pw.residuals[0].is_finite());
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn a_tail_transform_of_a_decaying_wave_is_finite_and_the_scattering_tail_tapers() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let r = grid();
    let (radial, _) = hydrogen(&r);
    let q = q_grid();
    let bound = tail_transform(&ctx, 0, &r, &radial, 1.3, true, &q).unwrap();
    assert!(bound.iter().all(|v| v.is_finite()) && bound[0].abs() > 0.);
    let flat = vec![1.; r.len()];
    let scattering = tail_transform(&ctx, 0, &r, &flat, 1.3, false, &q).unwrap();
    assert!(scattering.iter().all(|v| v.is_finite()));
    assert!(tail_transform(&ctx, 0, &r, &vec![0.; r.len()], 1.3, true, &q).is_err());
}
