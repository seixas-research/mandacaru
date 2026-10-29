use mandacaru_core::{
    quantum::{properties, Basis, Encoding, C64},
    Context,
};

#[test]
fn occupied_transition_rdms_match_independent_spin_orbital_expectations_for_every_mapping() {
    let context = Context::new(Some(1), 32 << 20).unwrap();
    {
        for mapping in ["jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"] {
            for particles in [[2, 1], [1, 1]] {
                let m = 3;
                let n = 2 * m;
                let encoding = Encoding::new(mapping, n, particles).unwrap();
                let basis = Basis::new(&context, &encoding, true).unwrap();
                let mut state: Vec<_> = (0..basis.len())
                    .map(|i| C64::new((i as f64 + 0.2).sin(), (0.7 * i as f64).cos()))
                    .collect();
                let norm = state.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                for value in &mut state {
                    *value /= norm;
                }
                let full = properties::rdms(&context, &encoding, &basis, &state, true).unwrap();
                let spatial =
                    properties::spatial_rdms(&context, &encoding, &basis, &state).unwrap();
                for p in 0..m {
                    for q in 0..m {
                        let expected = full.one.as_slice()[p * n + q]
                            + full.one.as_slice()[(p + m) * n + q + m];
                        assert!((spatial.one.as_slice()[p * m + q] - expected).norm() < 1e-13);
                        for r in 0..m {
                            for s in 0..m {
                                let mut expected = C64::new(0., 0.);
                                for alpha in 0..2 {
                                    for beta in 0..2 {
                                        expected += full.two.as_ref().unwrap().as_slice()[(((p
                                            + alpha * m)
                                            * n
                                            + q
                                            + beta * m)
                                            * n
                                            + r
                                            + alpha * m)
                                            * n
                                            + s
                                            + beta * m];
                                    }
                                }
                                assert!(
                                    (spatial.two.as_slice()[((p * m + q) * m + r) * m + s]
                                        - expected)
                                        .norm()
                                        < 1e-13,
                                    "{mapping} {p}{q}{r}{s}"
                                );
                            }
                        }
                    }
                }
                let electrons = (particles[0] + particles[1]) as f64;
                let trace = (0..m)
                    .map(|p| spatial.one.as_slice()[p * m + p].re)
                    .sum::<f64>();
                assert!((trace - electrons).abs() < 1e-12);
                let pair_trace = (0..m)
                    .flat_map(|p| (0..m).map(move |q| (p, q)))
                    .map(|(p, q)| spatial.two.as_slice()[((p * m + q) * m + p) * m + q].re)
                    .sum::<f64>();
                assert!((pair_trace - electrons * (electrons - 1.)).abs() < 1e-12);
            }
        }
    }
    assert_eq!(context.memory_used(), 0);
}
