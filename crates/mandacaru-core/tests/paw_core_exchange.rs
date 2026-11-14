//! The core-valence exchange of PAW Hartree-Fock (`paw::core_exchange`, stage 4-0c):
//!
//! - C2: the angular factor `(q_c / 2) (l L l_c; 0 0 0)^2` of the closed form equals an
//!   independent product quadrature over the sphere (Gauss-Legendre in `cos theta`, uniform in
//!   `phi`) of the explicit `m_c` sum of real Gaunt integrals, for `l, l_c <= 3`, at every `m`;
//!   and the term reduces, for a closed `s` core, to the one-multipole form the 1s-only path used.
//! - C0: for every Li-Ne dataset of `lda-sr/` and `pbe-sr/` (a single closed 1s core), the term
//!   built from the core orbital solved again on the dataset's own atom equals the term built
//!   from the orbital the stored density gives (`sqrt(2 pi n_c)`), to twice the density's
//!   reproduction error.
//!
//! Needs `MANDACARU_PAW_PATH` (read only).
use mandacaru_core::math::angular::real;
use mandacaru_core::paw::{
    core_exchange::{core_valence_exchange, CoreShell},
    generate::core_states::add_core_states,
    onecenter_radial::three_j_squared,
    pair_tables::coulomb,
    record::PawDataset,
};
use mandacaru_core::Context;
use std::f64::consts::PI;
use std::path::PathBuf;

/// Gauss-Legendre nodes and weights on [-1, 1] (Newton on the Legendre recurrence).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.; n];
    let mut w = vec![0.; n];
    for i in 0..n {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p0, mut p1) = (1., z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (z * p1 - p0) / (z * z - 1.);
            let step = p1 / dp;
            z -= step;
            if step.abs() < 1e-16 {
                let (mut p0, mut p1) = (1., z);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                let dp = n as f64 * (z * p1 - p0) / (z * z - 1.);
                x[i] = z;
                w[i] = 2. / ((1. - z * z) * dp * dp);
                break;
            }
        }
    }
    (x, w)
}

#[test]
fn the_angular_factor_is_the_m_sum_of_gaunt_integrals_by_quadrature() {
    let (nodes, weights) = gauss_legendre(24);
    let phis = 48;
    // Integral over the sphere of Y_a Y_b Y_c (real harmonics), by the product rule.
    let integral = |a: (usize, i32), b: (usize, i32), c: (usize, i32)| -> f64 {
        let mut sum = 0.;
        for (x, w) in nodes.iter().zip(&weights) {
            let theta = x.acos();
            for k in 0..phis {
                let phi = 2. * PI * k as f64 / phis as f64;
                sum += w
                    * (2. * PI / phis as f64)
                    * real(a.0, a.1, theta, phi)
                    * real(b.0, b.1, theta, phi)
                    * real(c.0, c.1, theta, phi);
            }
        }
        sum
    };
    for l in 0..=3usize {
        for lc in 0..=3usize {
            for big_l in l.abs_diff(lc)..=l + lc {
                let closed = (2 * lc + 1) as f64 * three_j_squared(l, big_l, lc).unwrap();
                for m in -(l as i32)..=(l as i32) {
                    // A full core shell, summed over m_c and over the multipole's M.
                    let mut quadrature = 0.;
                    for mc in -(lc as i32)..=(lc as i32) {
                        for big_m in -(big_l as i32)..=(big_l as i32) {
                            let g = integral((l, m), (lc, mc), (big_l, big_m));
                            quadrature += 4. * PI / (2 * big_l + 1) as f64 * g * g;
                        }
                    }
                    assert!(
                        (quadrature - closed).abs() < 1e-12,
                        "l={l} lc={lc} L={big_l} m={m}: quadrature {quadrature} closed {closed}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_closed_s_core_reduces_to_the_single_multipole_term() {
    let n = 400;
    let r: Vec<f64> = (1..=n).map(|i| 0.01 * i as f64).collect();
    let core: Vec<f64> = r.iter().map(|x| 2. * (-2. * x).exp()).collect();
    let waves: Vec<Vec<f64>> = (0..2)
        .map(|k| {
            r.iter()
                .map(|x| x.powi(2) * (-(1. + k as f64) * x).exp())
                .collect()
        })
        .collect();
    for l in 0..=3usize {
        let got = core_valence_exchange(
            &r,
            l,
            &waves,
            &[CoreShell {
                l: 0,
                occupation: 2.,
                radial: &core,
            }],
            n,
        )
        .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let left: Vec<f64> = (0..n).map(|k| waves[i][k] * core[k]).collect();
                let right: Vec<f64> = (0..n).map(|k| waves[j][k] * core[k]).collect();
                let want = -0.5 * (coulomb(&r, &left, &right, l) + coulomb(&r, &right, &left, l))
                    / (2 * l + 1) as f64;
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-14 * want.abs().max(1e-300),
                    "l={l} ({i}, {j}): {} {want}",
                    got[(i, j)]
                );
            }
        }
    }
}

/// The first `extent` points of `r` that hold the core: where the trapezoid charge of `n_c`
/// reaches `1 - 1e-14` of its total, plus two (the 1s-only path's rule).
fn extent(r: &[f64], core: &[f64]) -> usize {
    let cumulative = |upto: usize| -> f64 {
        (1..upto)
            .map(|i| {
                0.5 * (r[i] - r[i - 1])
                    * 4.
                    * PI
                    * (r[i - 1].powi(2) * core[i - 1] + r[i].powi(2) * core[i])
            })
            .sum()
    };
    let total = cumulative(r.len());
    let mut integrated = 0.;
    for i in 1..r.len() {
        integrated += 0.5
            * (r[i] - r[i - 1])
            * 4.
            * PI
            * (r[i - 1].powi(2) * core[i - 1] + r[i].powi(2) * core[i]);
        if integrated >= (1. - 1e-14) * total {
            return (i + 2).min(r.len());
        }
    }
    r.len()
}

#[test]
#[ignore = "slow: 17 s (16 reference atoms solved again); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test paw_core_exchange -- --ignored"]
fn the_solved_1s_orbital_gives_the_term_of_the_stored_density_for_li_to_ne() {
    let root = PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"));
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    for set in ["lda-sr", "pbe-sr"] {
        for symbol in ["Li", "Be", "B", "C", "N", "O", "F", "Ne"] {
            let path = root.join(set).join(format!("{symbol}.parquet"));
            let dataset = PawDataset::read(&ctx, &path).unwrap();
            let r = dataset.r.as_slice().to_vec();
            let density = dataset.core_density.as_slice().to_vec();
            let (upgraded, found) = add_core_states(&ctx, dataset).unwrap();
            let states = upgraded.core_states.as_ref().unwrap();
            assert_eq!(states.len(), 1, "{set}/{symbol}: one closed 1s core");
            let solved: Vec<f64> = states[0]
                .orbital
                .as_slice()
                .iter()
                .zip(&r)
                .map(|(u, x)| (u / x).abs())
                .collect();
            let from_density: Vec<f64> = density.iter().map(|n| (2. * PI * n).sqrt()).collect();
            let size = extent(&r, &density);
            let mut worst = 0f64;
            for (&l, channel) in &upgraded.channels {
                let waves: Vec<Vec<f64>> = channel
                    .ae_waves
                    .iter()
                    .map(|w| w.as_slice().to_vec())
                    .collect();
                let shell = |radial| CoreShell {
                    l: 0,
                    occupation: 2.,
                    radial,
                };
                let old =
                    core_valence_exchange(&r, l as usize, &waves, &[shell(&from_density)], size)
                        .unwrap();
                let new =
                    core_valence_exchange(&r, l as usize, &waves, &[shell(&solved)], size).unwrap();
                let scale = old.iter().fold(0f64, |m, v| m.max(v.abs()));
                let deviation = (&new - &old).iter().fold(0f64, |m, v| m.max(v.abs())) / scale;
                worst = worst.max(deviation);
            }
            println!(
                "{set}/{symbol}: density deviation {:.2e}, exchange deviation {worst:.2e}",
                found.density_deviation
            );
            assert!(
                worst <= 2. * found.density_deviation + 1e-12,
                "{set}/{symbol}: {worst:e} against 2 x {:e}",
                found.density_deviation
            );
        }
    }
}
