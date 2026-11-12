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

#[test]
fn transition_rdms_are_the_bilinear_form_of_the_rdms_on_every_thread_count() {
    for threads in [1, 3] {
        let context = Context::new(Some(threads), 32 << 20).unwrap();
        {
            for mapping in ["jordan_wigner", "parity_reduced"] {
                let m = 3;
                let encoding = Encoding::new(mapping, 2 * m, [2, 1]).unwrap();
                let basis = Basis::new(&context, &encoding, true).unwrap();
                let state = |shift: f64| -> Vec<C64> {
                    (0..basis.len())
                        .map(|i| C64::new((i as f64 + shift).sin(), (0.7 * i as f64 - shift).cos()))
                        .collect()
                };
                let (a, b) = (state(0.2), state(1.3));
                // With bra == ket: the RDMs themselves.
                let same = properties::spatial_transition_rdms(&context, &encoding, &basis, &a, &a)
                    .unwrap();
                let rdm = properties::spatial_rdms(&context, &encoding, &basis, &a).unwrap();
                for (x, y) in same.two.as_slice().iter().zip(rdm.two.as_slice()) {
                    assert!((x - y).norm() < 1e-12);
                }
                // The polarization identity: <a|O|b> + <b|O|a> = RDM(a + b) - RDM(a) - RDM(b) (real part).
                let sum: Vec<C64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
                let ab = properties::spatial_transition_rdms(&context, &encoding, &basis, &a, &b)
                    .unwrap();
                let ba = properties::spatial_transition_rdms(&context, &encoding, &basis, &b, &a)
                    .unwrap();
                let (rs, ra, rb) = (
                    properties::spatial_rdms(&context, &encoding, &basis, &sum).unwrap(),
                    rdm,
                    properties::spatial_rdms(&context, &encoding, &basis, &b).unwrap(),
                );
                for k in 0..ab.two.as_slice().len() {
                    let lhs = ab.two.as_slice()[k] + ba.two.as_slice()[k];
                    let rhs = rs.two.as_slice()[k] - ra.two.as_slice()[k] - rb.two.as_slice()[k];
                    assert!((lhs - rhs).norm() < 1e-11, "{mapping} {k}");
                }
            }
        }
        assert_eq!(context.memory_used(), 0);
    }
}

#[test]
fn transition_rdms_of_a_sector_above_one_chunk_are_bitwise_the_same_on_every_thread_count() {
    // 8 orbitals, 4 + 4 electrons: 4900 determinants, two fixed chunks of
    // `properties::RDM_CHUNK` columns whatever the number of threads.
    let m = 8;
    let mut reference: Option<(Vec<C64>, Vec<C64>)> = None;
    for threads in [1, 2, 3, 4] {
        let context = Context::new(Some(threads), 64 << 20).unwrap();
        {
            let encoding = Encoding::new("jordan_wigner", 2 * m, [4, 4]).unwrap();
            let basis = Basis::new(&context, &encoding, true).unwrap();
            assert!(basis.len() > properties::RDM_CHUNK);
            let state = |shift: f64| -> Vec<C64> {
                (0..basis.len())
                    .map(|i| {
                        C64::new(
                            (i as f64 * 0.37 + shift).sin(),
                            (0.11 * i as f64 - shift).cos(),
                        )
                    })
                    .collect()
            };
            let rdm = properties::spatial_transition_rdms(
                &context,
                &encoding,
                &basis,
                &state(0.2),
                &state(1.3),
            )
            .unwrap();
            let got = (rdm.one.as_slice().to_vec(), rdm.two.as_slice().to_vec());
            match &reference {
                None => reference = Some(got),
                Some(first) => assert!(*first == got, "{threads} threads differ from one"),
            }
        }
        assert_eq!(context.memory_used(), 0);
    }
}
