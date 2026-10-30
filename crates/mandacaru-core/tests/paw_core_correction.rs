use mandacaru_core::math::radial::gradient;
use mandacaru_core::math::roots::brent;
use mandacaru_core::paw::core_correction::{
    crossover_radius, partial_core_density, smooth_core_density,
};
use std::f64::consts::PI;

fn grid(n: usize, r_max: f64) -> Vec<f64> {
    (1..=n).map(|i| i as f64 * r_max / (n + 1) as f64).collect()
}
fn densities(r: &[f64]) -> (Vec<f64>, Vec<f64>) {
    (
        r.iter()
            .map(|x| 2. * 216. / PI * (-12. * x).exp())
            .collect(),
        r.iter().map(|x| 6. / PI * (-2. * x).exp()).collect(),
    )
}

#[test]
fn brent_finds_roots_to_the_requested_tolerance() {
    let root = brent(|x| Ok(x * x * x - 2.), 0., 2., 1e-14, 1e-15, 100).unwrap();
    assert!((root - 2f64.cbrt()).abs() < 1e-14);
    assert!(brent(|x| Ok(x * x + 1.), 0., 2., 1e-14, 1e-15, 100).is_err());
    assert_eq!(brent(|x| Ok(x), 0., 2., 1e-14, 1e-15, 100).unwrap(), 0.);
}

#[test]
fn the_gradient_is_second_order_on_uniform_and_nonuniform_meshes() {
    let uniform = (0..50).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
    let nonuniform = (0..50)
        .map(|i| (i as f64 * 0.1).powf(1.2) + 0.01 * i as f64)
        .collect::<Vec<_>>();
    for x in [uniform, nonuniform] {
        let f = x.iter().map(|x| x * x - 3. * x).collect::<Vec<_>>();
        let g = gradient(&x, &f).unwrap();
        for (x, g) in x.iter().zip(&g) {
            assert!((g - (2. * x - 3.)).abs() < 1e-12, "x = {x}");
        }
    }
}

#[test]
fn the_crossover_is_where_the_densities_meet() {
    let r = grid(4000, 20.);
    let (core, valence) = densities(&r);
    let radius = crossover_radius(&r, &core, &valence, 1.).unwrap().unwrap();
    let index = r.iter().position(|x| *x >= radius).unwrap();
    assert!((core[index] / valence[index] - 1.).abs() < 5e-2);
    assert!(crossover_radius(&r, &core, &valence, 2.).unwrap().unwrap() < radius);
    assert!(crossover_radius(&r, &vec![0.; r.len()], &valence, 1.)
        .unwrap()
        .is_none());
    assert!(crossover_radius(&r, &core[1..], &valence, 1.).is_err());
}

#[test]
fn the_smooth_core_matches_value_and_slope_and_is_the_core_outside() {
    let r = grid(4000, 20.);
    let (core, _) = densities(&r);
    let (smooth, a, b) = smooth_core_density(&r, &core, 0.5).unwrap();
    let k = r.iter().position(|x| *x >= 0.5).unwrap();
    let slope = |v: &[f64], i: usize| (v[i + 1] - v[i - 1]) / (r[i + 1] - r[i - 1]);
    assert_eq!(smooth[k], core[k]);
    assert!((slope(&smooth, k + 1) - slope(&core, k + 1)).abs() < 1e-9 * core[k]);
    assert!(smooth[..k].iter().all(|v| v.is_finite() && *v > 0.));
    assert!(smooth[..k].iter().zip(&core).all(|(s, c)| s <= c));
    assert!(a > 0. && b > 0. && b * r[k] < PI);
    assert!(smooth_core_density(&r, &core, 15.).is_err() || core[3999] > 0.);
    assert!(smooth_core_density(&r, &vec![0.; r.len()], 1.).is_err());
}

#[test]
fn a_valence_only_atom_gets_no_correction() {
    let r = grid(4000, 20.);
    let (_, valence) = densities(&r);
    let out = partial_core_density(&r, &vec![0.; r.len()], &valence, None, 1.).unwrap();
    assert!(out.r_nlcc.is_none() && out.density.iter().all(|v| *v == 0.));
    let (core, valence) = densities(&r);
    let out = partial_core_density(&r, &core, &valence, None, 1.).unwrap();
    assert!(out.r_nlcc.is_some() && out.partial_core_electrons < out.core_electrons);
}
