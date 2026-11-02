//! The compensation charges in the periodic screened exact exchange (W-E4P / P1), on the Si, Al,
//! H2 and spinor Si fixtures of `test/migration/phase5` made hybrids on a whole mesh: the
//! unit-charge sum rule, the independent direct Fourier sum, Hermiticity and positivity, and the
//! old operator reproduced bit for bit when no compensation is attached.
//!
//! The states are the occupied generalized eigenvectors of each point's fixed Hamiltonian
//! (a mechanics state, not an SCF one). Needs `MANDACARU_PAW_PATH` for the compensation radii.
mod periodic_screened_augmentation_support;
mod setup_support;

use mandacaru_core::{
    periodic::{
        screened_augmentation::Augmentation,
        screened_exchange::{
            exchange_operator_at_k_with, exchange_operators_with, ExxDivTreatment, Screening,
        },
    },
    Context,
};
use num_complex::Complex64;
use periodic_screened_augmentation_support::{reference::Compensation, *};

#[test]
fn the_augmented_pair_of_a_normalized_state_integrates_to_the_overlap() {
    let mut worst = 0f64;
    let mut plain = 0f64;
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let (n, dv) = (c.grid.index.points(), c.grid.index.dv());
        for point in &c.points {
            let shapes = aug.pair_shapes(c, [0.; 3]).unwrap();
            let samples = point.psi.read(&c.context).unwrap();
            let u = |mu: usize, r: usize| {
                let x = c.grid.point(r);
                let angle = -(0..3).map(|a| point.k[a] * x[a]).sum::<f64>();
                samples[mu * n + r] * Complex64::from_polar(1., angle)
            };
            let states = coefficients(point);
            for state in 0..c.m {
                let column: Vec<Complex64> = states.column(state).iter().copied().collect();
                let projected = aug.project(c, point, &column);
                let moments = aug.moments(c, point, &projected);
                let overlap = &point.overlap * states.column(state);
                for mu in 0..c.m {
                    let smooth: Complex64 = (0..n)
                        .map(|r| {
                            u(mu, r).conj()
                                * (0..c.m).map(|nu| column[nu] * u(nu, r)).sum::<Complex64>()
                        })
                        .sum();
                    let comp: Complex64 = (0..c.channels)
                        .map(|a| moments[(mu, a)] * shapes.as_slice()[a * n])
                        .sum();
                    worst = worst.max((dv * (smooth + comp) - overlap[mu]).norm());
                    plain = plain.max((dv * smooth - overlap[mu]).norm());
                }
            }
        }
    }
    println!("unit-charge sum rule: augmented {worst:.2e}, grid pair alone {plain:.2e}");
    assert!(worst < 1e-10, "sum rule {worst:e}");
    assert!(
        plain > 100. * worst.max(1e-10),
        "the grid pair alone must miss it"
    );
}

#[test]
fn the_operator_equals_an_independent_direct_fourier_sum() {
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let comp = Compensation {
            sites: &sites,
            shapes: &rig.shapes,
            alias_hats: None,
        };
        for exxdiv in [ExxDivTreatment::None, GB] {
            let all =
                exchange_operators_with(c, &rig.d, screening(0.11), exxdiv, Some(&aug)).unwrap();
            let plain = exchange_operators_with(c, &rig.d, screening(0.11), exxdiv, None).unwrap();
            for spin in 0..c.spins {
                for k in 0..c.points.len() {
                    let want = reference::exchange(c, &rig.d, spin, k, 0.11, exxdiv, Some(&comp));
                    let got = &all.operators[spin * c.points.len() + k];
                    let at_k = exchange_operator_at_k_with(
                        c,
                        &rig.d,
                        &c.points[k],
                        spin,
                        screening(0.11),
                        exxdiv,
                        Some(&aug),
                    )
                    .unwrap();
                    let (e1, e2) = (relative(got, &want), relative(&at_k, &want));
                    let pseudo = reference::exchange(c, &rig.d, spin, k, 0.11, exxdiv, None);
                    let e3 = relative(&plain.operators[spin * c.points.len() + k], &pseudo);
                    let gap = relative(got, &plain.operators[spin * c.points.len() + k]);
                    println!("{fixture} {exxdiv:?} spin {spin} k {k}: mesh {e1:.1e}, at_k {e2:.1e}, pseudo-only {e3:.1e}; compensation moves K by {gap:.2e}");
                    assert!(e1 < 1e-10 && e2 < 1e-10 && e3 < 1e-10);
                    assert!(
                        got.norm() == 0.
                            || gap > 1e-6
                            || rig.d[spin * c.points.len() + k].norm() == 0.
                    );
                }
            }
        }
    }
}

#[test]
fn a_spinor_crystal_equals_the_direct_fourier_sum() {
    let rig = Rig::new("si_so.json", true, &[8]);
    let c = &rig.crystal;
    assert!(c.spinor);
    let sites = rig.geometry.sites();
    let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
    let comp = Compensation {
        sites: &sites,
        shapes: &rig.shapes,
        alias_hats: None,
    };
    let got = exchange_operators_with(c, &rig.d, screening(0.11), GB, Some(&aug)).unwrap();
    let plain = exchange_operators_with(c, &rig.d, screening(0.11), GB, None).unwrap();
    let want = reference::exchange(c, &rig.d, 0, 0, 0.11, GB, Some(&comp));
    let at_k =
        exchange_operator_at_k_with(c, &rig.d, &c.points[0], 0, screening(0.11), GB, Some(&aug))
            .unwrap();
    let (e1, e2, gap) = (
        relative(&got.operators[0], &want),
        relative(&at_k, &want),
        relative(&got.operators[0], &plain.operators[0]),
    );
    println!("spinor Si: mesh {e1:.1e}, at_k {e2:.1e}; the compensation moves K by {gap:.2e}");
    assert_eq!(got.operators[0].nrows(), 2 * c.m);
    assert!(e1 < 1e-10 && e2 < 1e-10 && gap > 1e-6);
}

#[test]
fn k_is_hermitian_and_positive_semidefinite() {
    for (fixture, gamma, occupied) in CASES.iter() {
        let rig = Rig::new(fixture, *gamma, occupied);
        let c = &rig.crystal;
        let sites = rig.geometry.sites();
        let aug = Augmentation::new(c, sites, &rig.shapes).unwrap();
        let out = exchange_operators_with(c, &rig.d, screening(0.11), GB, Some(&aug)).unwrap();
        for k in &out.operators {
            let skew = (k - k.adjoint())
                .iter()
                .map(|v| v.norm())
                .fold(0., f64::max);
            let scale = k.iter().map(|v| v.norm()).fold(0., f64::max);
            let (eig, _) = mandacaru_core::math::linalg::eigen(k.clone()).unwrap();
            println!(
                "{fixture}: |K - K^+| {skew:.1e}, lowest eigenvalue {:.2e} of {scale:.2e}",
                eig[0]
            );
            assert!(skew <= 1e-14 * scale.max(1.));
            assert!(eig[0] >= -1e-12 * scale.max(1.));
        }
    }
}

#[test]
fn without_compensation_the_old_operator_is_reproduced_bit_for_bit() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/periodic/exchange_pseudo_pair.json");
    let frozen: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["fixture"].as_str().unwrap();
        let gamma = case["gamma"].as_bool().unwrap();
        let ctx = Context::new(Some(1), 512 << 20).unwrap();
        let c = hybrid(&ctx, name, gamma);
        let m = case["m"].as_u64().unwrap() as usize;
        let d: Vec<Matrix> = case["densities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| from_bits(m, m, v))
            .collect();
        for (label, exxdiv) in [("none", ExxDivTreatment::None), ("gb", GB)] {
            let want = &case[label];
            let with =
                exchange_operators_with(&c, &d, Screening::of(&c).unwrap(), exxdiv, None).unwrap();
            let (ops, energy) = (&with.operators, with.energy);
            assert_eq!(
                energy.to_bits(),
                want["energy"].as_u64().unwrap(),
                "{name} {label} energy"
            );
                        for (k, (op, frozen)) in ops
                .iter()
                .zip(want["operators"].as_array().unwrap())
                .enumerate()
            {
                let frozen = from_bits(m, m, frozen);
                assert!(
                    op.iter()
                        .zip(frozen.iter())
                        .all(|(a, b)| a.re.to_bits() == b.re.to_bits()
                            && a.im.to_bits() == b.im.to_bits()),
                    "{name} {label} K[{k}]"
                );
                            }
            let mut at = 0;
            for spin in 0..c.spins {
                for point in &c.points {
                    let got = exchange_operator_at_k_with(
                        &c,
                        &d,
                        point,
                        spin,
                        Screening::of(&c).unwrap(),
                        exxdiv,
                        None,
                    )
                    .unwrap();
                    let frozen = from_bits(m, m, &want["at_k"][at]);
                    assert!(
                        got.iter()
                            .zip(frozen.iter())
                            .all(|(a, b)| a.re.to_bits() == b.re.to_bits()
                                && a.im.to_bits() == b.im.to_bits()),
                        "{name} {label} at_k {at}"
                    );
                    at += 1;
                }
            }
        }
        println!("{name}: grid-pair operator and energy bit for bit with the frozen pre-augmentation numbers");
    }
}
