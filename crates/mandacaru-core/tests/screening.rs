use mandacaru_core::{
    periodic::{
        screening::{self, Options, Polarizability, State},
        spectrum::Smearing,
    },
    quantum::C64,
    Context,
};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

fn options(constrained: bool) -> Options {
    Options {
        cell_volume: 7.,
        kpoints: 5,
        fermi: 0.,
        width: 0.03,
        degeneracy: 2.,
        smearing: Smearing::FermiDirac,
        constrained,
    }
}

#[test]
fn complex_two_level_response_and_dyson_match_the_rank_one_analytic_solution() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    {
        let density = [C64::new(0.3, -0.2), C64::new(-0.4, 0.6), C64::new(0.2, 0.7)];
        let low = State {
            energy: -0.4,
            target: 0.3,
        };
        let high = State {
            energy: 0.4,
            target: 0.4,
        };
        let mut accumulator = Polarizability::new(&ctx, 3, options(false)).unwrap();
        accumulator.add(low, high, &density).unwrap();
        accumulator.add(high, low, &density).unwrap();
        let (chi, _memory) = accumulator.finish().unwrap();
        let coefficient = 4.
            * (Smearing::FermiDirac.occupation(-0.4 / 0.03)
                - Smearing::FermiDirac.occupation(0.4 / 0.03))
            / (-0.8 * 35.);
        let expected = DMatrix::from_fn(3, 3, |i, j| coefficient * density[i] * density[j].conj());
        assert!((&chi - &expected).norm() < 1e-14);
        assert!(chi[(0, 1)].im.abs() > 1e-3);
        let vectors = [[0.; 3], [1., 0., 0.], [0., 2., 0.]];
        let interaction = screening::dyson(&ctx, &chi, &vectors).unwrap();
        let root = [0., (4. * PI).sqrt(), PI.sqrt()];
        let weighted = DVector::from_iterator(3, density.iter().zip(root).map(|(v, s)| v * s));
        let inverse = DMatrix::identity(3, 3)
            + (&weighted * weighted.adjoint())
                * C64::new(
                    coefficient / (1. - coefficient * weighted.norm_squared()),
                    0.,
                );
        assert!((&interaction.inverse_dielectric - &inverse).norm() < 1e-13);
        let correction = DMatrix::from_fn(3, 3, |i, j| {
            root[i] * (inverse[(i, j)] - C64::new(f64::from(i == j), 0.)) * root[j]
        });
        assert!((&interaction.correction - correction).norm() < 1e-12);
        assert!((interaction.inverse_dielectric[(0, 0)] - C64::new(1., 0.)).norm() < 1e-14);
        assert!(interaction.correction[(0, 0)].norm() < 1e-14);
        let (e, _) = mandacaru_core::math::linalg::eigen(interaction.correction.clone()).unwrap();
        assert!(e[e.len() - 1] < 1e-12 && e[0] < -0.1);
        let mut partial = Polarizability::new(&ctx, 3, options(true)).unwrap();
        partial.add(low, high, &density).unwrap();
        partial.add(high, low, &density).unwrap();
        let (partial, _partial_memory) = partial.finish().unwrap();
        assert!((partial - expected * C64::new(1. - 0.3 * 0.4, 0.)).norm() < 1e-14);
        let mut excluded = Polarizability::new(&ctx, 3, options(true)).unwrap();
        excluded
            .add(
                State { target: 1., ..low },
                State { target: 1., ..high },
                &density,
            )
            .unwrap();
        let (zero, _zero_memory) = excluded.finish().unwrap();
        assert_eq!(zero.norm(), 0.);
        assert_eq!(
            screening::dyson(&ctx, &zero, &vectors)
                .unwrap()
                .correction
                .norm(),
            0.
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn degenerate_response_and_batched_complex_transition_tails_match_direct_sums() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    {
        let mut response = Polarizability::new(&ctx, 3, options(false)).unwrap();
        let mut expected = DMatrix::<C64>::zeros(3, 3);
        let step = 1e-6;
        for t in 0..137 {
            let energy = (t as f64 - 68.) * 0.002;
            let density = std::array::from_fn::<_, 3, _>(|g| {
                C64::new(
                    (0.3 * t as f64 + g as f64).sin(),
                    (0.2 * t as f64 - 2. * g as f64).cos(),
                )
            });
            response
                .add(
                    State { energy, target: 0. },
                    State {
                        energy: energy + 1e-11,
                        target: 0.,
                    },
                    &density,
                )
                .unwrap();
            let occupation = |e: f64| 2. * Smearing::FermiDirac.occupation(e / 0.03);
            let slope = (occupation(energy + 0.5e-11 + step) - occupation(energy + 0.5e-11 - step))
                / (2. * step * 35.);
            for i in 0..3 {
                for j in 0..3 {
                    expected[(i, j)] += slope * density[i] * density[j].conj();
                }
            }
        }
        let (actual, _memory) = response.finish().unwrap();
        assert!((&actual - expected).norm() < 2e-9);
        assert!((&actual - actual.adjoint()).norm() < 1e-13);
        for smearing in [
            Smearing::FermiDirac,
            Smearing::Gaussian,
            Smearing::MethfesselPaxton,
        ] {
            for x in [-4., -1.2, 0., 0.4, 3.] {
                let numerical =
                    (smearing.occupation(x + step) - smearing.occupation(x - step)) / (2. * step);
                assert!((smearing.slope(x) - numerical).abs() < 1e-10);
            }
            assert_eq!(smearing.slope(1e300), 0.);
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn invalid_or_unstable_screening_is_rejected_without_retained_allocations() {
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    assert!(screening::dyson(
        &ctx,
        &DMatrix::from_element(1, 1, C64::new(1., 0.)),
        &[[1., 0., 0.]]
    )
    .is_err());
    assert_eq!(ctx.memory_used(), 0);
    {
        let mut response = Polarizability::new(&ctx, 1, options(true)).unwrap();
        assert!(response
            .add(
                State {
                    energy: 0.,
                    target: 1.1
                },
                State {
                    energy: 1.,
                    target: 0.
                },
                &[C64::new(1., 0.)]
            )
            .is_err());
        assert!(response
            .add(
                State {
                    energy: 0.,
                    target: 0.
                },
                State {
                    energy: 1.,
                    target: 0.
                },
                &[C64::new(f64::NAN, 0.)]
            )
            .is_err());
    }
    assert_eq!(ctx.memory_used(), 0);
}
