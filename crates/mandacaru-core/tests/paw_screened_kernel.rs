//! The radial pieces of the screened one-center terms for every multipole level
//! `L`, not only the monopole: the erfc-screened Coulomb energy of two
//! concentric `r^L exp(-a r^2) Y_LM` charges has a closed form, and the
//! compensation charge `g_L Y_LM`, its bare potential and its erf part must
//! agree with direct quadrature of the same radial kernel. A wrong factor in
//! the `L >= 1` channels (p and d shells make `L` up to 2 and 4) would show
//! here and nowhere in the unit-charge sum rule, which only sees `L = 0`.
use mandacaru_core::{
    math::{radial::gauss_legendre, spline::simpson},
    paw::{
        multipole::{shape_function, shape_potential},
        onecenter_radial::{long_range_radial_kernel, long_range_shape_potential, radial_coulomb},
        pair_tables::{potential, shape},
    },
    Context,
};
use std::f64::consts::PI;

/// `Gamma(L + 1/2) = sqrt(pi) (2L-1)!! / 2^L`.
fn gamma_half(l: usize) -> f64 {
    (1..=l).fold(PI.sqrt(), |g, k| g * (k as f64 - 0.5))
}

/// `int_0^inf f_a f_b r^2 s^2 r_<^L / r_>^(L+1)` for `f = r^L exp(-a r^2)` under
/// the bare kernel, and under its erf part, from the plane-wave representation
/// `r_<^L / r_>^(L+1) = (2L+1) (2/pi) int dq j_L(qr) j_L(qs)`, where
/// `int r^(L+2) exp(-a r^2) j_L(qr) dr = c_a q^L exp(-q^2 / 4a)`.
fn closed_form(l: usize, a: f64, b: f64, omega: f64) -> (f64, f64) {
    let c = |e: f64| PI.sqrt() / (2f64.powi(l as i32 + 2) * e.powf(l as f64 + 1.5));
    let beta = 0.25 / a + 0.25 / b;
    let long = beta + 0.25 / (omega * omega);
    let scale = (2 * l + 1) as f64 * (2. / PI) * c(a) * c(b) * 0.5 * gamma_half(l);
    let power = -(l as f64 + 0.5);
    (scale * beta.powf(power), scale * long.powf(power))
}

#[test]
#[ignore = "slow: 17 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_screened_kernel -- --ignored the_erfc_screened_energy_of_two_multipole_gaussians_is_analytic_at_every_level"]
fn the_erfc_screened_energy_of_two_multipole_gaussians_is_analytic_at_every_level() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let r: Vec<f64> = (0..1600).map(|i| 9. * i as f64 / 1599.).collect();
    let weights: Vec<f64> = (0..r.len())
        .map(|i| {
            let mut unit = vec![0.; r.len()];
            unit[i] = 1.;
            simpson(&r, &unit).unwrap() * r[i] * r[i]
        })
        .collect();
    let mut worst = 0f64;
    for l in 0..=4usize {
        let kernel = &long_range_radial_kernel(&ctx, &r, &r, &[l], 0.11, None).unwrap()[0];
        for (a, b, omega) in [(0.7, 1.9, 0.11), (1.3, 1.3, 0.11), (0.9, 2.5, 0.4)] {
            let kernel_here;
            let kernel = if omega == 0.11 {
                kernel
            } else {
                kernel_here = long_range_radial_kernel(&ctx, &r, &r, &[l], omega, None).unwrap();
                &kernel_here[0]
            };
            let density = |e: f64| -> Vec<f64> {
                r.iter()
                    .map(|r| r.powi(l as i32) * (-e * r * r).exp())
                    .collect()
            };
            let (fa, fb) = (density(a), density(b));
            let bare = radial_coulomb(&r, &fa, &fb, l).unwrap();
            let long: f64 = (0..r.len())
                .map(|i| {
                    fa[i]
                        * weights[i]
                        * (0..r.len())
                            .map(|j| kernel[i * r.len() + j] * fb[j] * weights[j])
                            .sum::<f64>()
                })
                .sum();
            let (bare_exact, long_exact) = closed_form(l, a, b, omega);
            let (e_bare, e_long) = (
                (bare - bare_exact).abs() / bare_exact,
                (long - long_exact).abs() / bare_exact,
            );
            println!("L {l} (a {a}, b {b}, omega {omega}): bare {e_bare:.2e}, erf {e_long:.2e}");
            worst = worst.max(e_bare).max(e_long);
            // The erf part is a smooth kernel and agrees to round-off (3e-11 at
            // worst here, relative to the bare energy); the bare radial integral carries the quadrature error
            // of the integrator (2.4e-5 at L 3 on 1600 points, growing with L).
            assert!(e_bare < 1e-4, "L {l}: bare {bare} vs {bare_exact}");
            assert!(e_long < 1e-9, "L {l}: erf {long} vs {long_exact}");
        }
    }
    println!("worst relative deviation {worst:.2e}");
}

#[test]
fn the_compensation_shape_and_its_potentials_agree_with_the_radial_kernels() {
    let ctx = Context::new(Some(1), 256 << 20).unwrap();
    let (rc, omega) = (1.1, 0.11);
    let nodes = gauss_legendre(120).unwrap();
    let s: Vec<f64> = nodes.iter().map(|(x, _)| 0.5 * rc * (x + 1.)).collect();
    let r: Vec<f64> = [0.05, 0.3, 0.7, 1.0, 1.1, 1.4, 2.5, 4.0].to_vec();
    for l in 0..=4usize {
        let g = shape_function(&s, rc, l).unwrap();
        // The two implementations of the shape and of its bare potential agree.
        for (k, &s_k) in s.iter().enumerate() {
            assert!(
                (shape(s_k, rc, l) - g[k]).abs() < 1e-12 * (1. + g[k].abs()),
                "shape L {l}"
            );
        }
        let bare_other = shape_potential(&r, rc, l).unwrap();
        let long = long_range_shape_potential(&ctx, &r, rc, l, omega).unwrap();
        let kernel = &long_range_radial_kernel(&ctx, &r, &s, &[l], omega, None).unwrap()[0];
        let scale = 4. * PI / (2 * l + 1) as f64;
        let weight: Vec<f64> = nodes
            .iter()
            .zip(&s)
            .zip(&g)
            .map(|(((_, w), s), g)| 0.5 * rc * w * s * s * g)
            .collect();
        for (i, &r_i) in r.iter().enumerate() {
            // The bare potential by quadrature of r_<^L / r_>^(L+1), split at the
            // kink of the kernel (s = r) so that the Gauss rule converges.
            let direct: f64 = {
                let edge = r_i.min(rc);
                let mut total = 0.;
                for (lo, hi) in [(0., edge), (edge, rc)] {
                    if hi <= lo {
                        continue;
                    }
                    for (x, w) in &nodes {
                        let s = lo + 0.5 * (hi - lo) * (x + 1.);
                        let weight = 0.5 * (hi - lo) * w * s * s * shape(s, rc, l);
                        let (small, large) = (r_i.min(s), r_i.max(s));
                        total += small.powi(l as i32) / large.powi(l as i32 + 1) * weight;
                    }
                }
                scale * total
            };
            let analytic = potential(r_i, rc, l);
            let erf_part: f64 = scale
                * (0..s.len())
                    .map(|j| kernel[i * s.len() + j] * weight[j])
                    .sum::<f64>();
            let size = 1. + analytic.abs();
            assert!(
                (analytic - direct).abs() < 1e-9 * size,
                "bare L {l} r {r_i}: {analytic} vs {direct}"
            );
            assert!(
                (bare_other[i] - analytic).abs() < 1e-12 * size,
                "multipole.rs L {l} r {r_i}"
            );
            assert!(
                (long[i] - erf_part).abs() < 2e-6 * size,
                "erf L {l} r {r_i}: {} vs {erf_part}",
                long[i]
            );
            assert!(
                long[i] <= analytic + 1e-12,
                "the erf part is part of the bare potential"
            );
        }
    }
}
