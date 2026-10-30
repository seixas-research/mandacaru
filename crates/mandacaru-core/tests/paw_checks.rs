use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::checks::{
    bound_levels, interpolate_linear, log_derivative_of_u, log_derivatives_ae, node_count,
    peak_radius, phase_summary, spectrum_extent, spurious_nodes, window_offsets,
};
use mandacaru_core::Context;
use std::f64::consts::PI;

fn grid() -> Vec<f64> {
    (1..=6000).map(|i| i as f64 * 0.005).collect()
}

#[test]
fn hydrogen_levels_and_nodes() {
    let r = grid();
    let v = r.iter().map(|x| -1. / x).collect::<Vec<_>>();
    let levels = bound_levels(&r, &v, 0, None, 0.).unwrap();
    assert!(levels.len() > 3);
    assert!((levels[0] + 0.5).abs() < 1e-4 && (levels[1] + 0.125).abs() < 1e-4);
    // l = 1 starts at 2p.
    assert!((bound_levels(&r, &v, 1, None, 0.).unwrap()[0] + 0.125).abs() < 1e-4);
    assert_eq!(
        bound_levels(&r, &v, 0, Some(-0.3), 0.).unwrap().len(),
        levels.len() - 1
    );
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    // Between 1s and 2s the regular solution has one node.
    let (nodes, sum) = node_count(&ctx, &r, &v, 0, -0.2, 1., Treatment::None).unwrap();
    assert_eq!(nodes, 1);
    assert!(sum.is_finite());
    assert!(bound_levels(&r[..3], &v[..3], 0, None, 0.).is_err());
}

#[test]
fn logarithmic_derivatives() {
    let ctx = Context::new(Some(1), 64 << 20).unwrap();
    let r = grid();
    let u = r.iter().map(|x| x * (-x).exp()).collect::<Vec<_>>();
    assert!((log_derivative_of_u(&r, &u, 1.5).unwrap() + 1.).abs() < 1e-8);
    assert!(log_derivative_of_u(&r, &u, r[0]).is_err());
    // At the hydrogen 1s energy the all-electron log derivative of R is -1.
    let v = r.iter().map(|x| -1. / x).collect::<Vec<_>>();
    let l = log_derivatives_ae(&ctx, &r, &v, 0, &[-0.5], 1.5, 1., Treatment::None, None).unwrap();
    assert!((l[0] + 1.).abs() < 1e-5);
}

#[test]
fn phases_wrap_at_the_poles_and_flag_nonfinite_values() {
    let offsets = window_offsets(1., 0.05);
    assert_eq!(offsets.len(), 41);
    assert_eq!(offsets[0], -1.);
    assert!((offsets[40] - 1.).abs() < 1e-12);
    let same = vec![0.3; 41];
    assert_eq!(
        phase_summary(&same, &same, &offsets, 0.5).unwrap(),
        (0., 0., None)
    );
    // A phase shift of 0.1 rad everywhere, and 0.4 only outside the near window.
    let ae = vec![0.3; 41];
    let ps = offsets
        .iter()
        .map(|o| (0.3f64.atan() + if o.abs() <= 0.5 { 0.1 } else { 0.4 }).tan())
        .collect::<Vec<_>>();
    let (near, far, bad) = phase_summary(&ps, &ae, &offsets, 0.5).unwrap();
    assert!((near - 0.1).abs() < 1e-12 && (far - 0.4).abs() < 1e-12 && bad.is_none());
    // A pole of L is no error: atan jumps by pi between +inf and -inf.
    assert!(phase_summary(&[1e16], &[-1e16], &[0.], 0.5).unwrap().1 < 1e-8);
    let mut broken = same.clone();
    broken[7] = f64::NAN;
    assert_eq!(
        phase_summary(&broken, &same, &offsets, 0.5).unwrap().2,
        Some(7)
    );
    assert!(phase_summary(&same[..3], &same, &offsets, 0.5).is_err());
    let _ = PI;
}

#[test]
fn interpolation_peaks_extent_and_spurious_nodes() {
    assert_eq!(
        interpolate_linear(&[-1., 0.5, 2., 9.], &[0., 1., 2.], &[0., 10., 0.]).unwrap(),
        vec![0., 5., 0., 0.]
    );
    assert!(interpolate_linear(&[0.], &[1., 0.], &[0., 0.]).is_err());
    let r = grid();
    let radial = r.iter().map(|x| 2. * (-x).exp()).collect::<Vec<_>>();
    assert!((peak_radius(&r, &radial).unwrap() - 1.).abs() < 1e-12);
    let extent = spectrum_extent(&r, &radial, 1e-5, 4., 24.).unwrap();
    // 2 r exp(-r) falls to 1e-5 of its peak 2/e near r = 15.2.
    assert!((extent - 15.2).abs() < 0.3, "{extent}");
    assert_eq!(spectrum_extent(&r, &radial, 1e-5, 30., 31.).unwrap(), 30.);
    // j_0 expansions: one smooth lobe has no node inside, the second zero of j_0 does.
    let ae = r.iter().map(|x| (-x).exp()).collect::<Vec<_>>();
    let mut waves = ae.clone();
    waves.extend(&ae);
    let flat = vec![PI / 2., 3. * PI / 2.];
    let coefficients = vec![1., 0., 0., 1.];
    let found = spurious_nodes(
        0,
        2.,
        &r,
        &waves,
        &[flat.clone(), flat].concat(),
        &coefficients,
        2,
    )
    .unwrap();
    assert_eq!(found, vec![1]);
}
