use mandacaru_core::{
    models::dispersion::{self, ChargeParameters, Damping},
    Context, Geometry, LengthUnit,
};

#[test]
fn damped_pair_has_coefficient_response_and_correct_asymptote() {
    let damping = Damping {
        s6: 1.,
        s8: 0.8,
        a1: 0.4,
        a2: 4.5,
    };
    let x = [8., 12., 96.];
    let (e, d) = dispersion::pair(x[0], x[1], x[2], damping).unwrap();
    assert!(e < 0. && d[0] > 0.);
    for i in 0..3 {
        let mut p = x;
        let mut m = x;
        let h = 1e-5 * x[i];
        p[i] += h;
        m[i] -= h;
        let fd = (dispersion::pair(p[0], p[1], p[2], damping).unwrap().0
            - dispersion::pair(m[0], m[1], m[2], damping).unwrap().0)
            / (2. * h);
        assert!((fd - d[i]).abs() < 1e-12);
    }
    let far = dispersion::pair(10000., 12., 96., damping).unwrap().0;
    assert!((far / (-12. / 10000f64.powi(6)) - 1.).abs() < 1e-7);
    assert!(dispersion::pair(0., 12., 96., damping).is_err());
}
#[test]
fn atm_sign_and_all_nine_partials_follow_the_energy() {
    let x = [8., 9., 10., 12., 13., 14., 4., 4.5, 5.];
    let eval = |x: [f64; 9]| {
        dispersion::three_body(
            x[..3].try_into().unwrap(),
            x[3..6].try_into().unwrap(),
            x[6..].try_into().unwrap(),
            1.,
        )
        .unwrap()
    };
    let (e, d) = eval(x);
    assert!(e > 0.);
    for i in 0..9 {
        let mut p = x;
        let mut m = x;
        let h = 1e-5 * x[i];
        p[i] += h;
        m[i] -= h;
        assert!(((eval(p).0 - eval(m).0) / (2. * h) - d[i]).abs() < 1e-13);
    }
    assert!(
        dispersion::three_body([8., 8., 16.], [12.; 3], [4.; 3], 1.)
            .unwrap()
            .0
            < 0.
    );
    assert!(dispersion::three_body([1., 1., 3.], [12.; 3], [4.; 3], 1.).is_err());
}

fn molecule(positions: [[f64; 3]; 3]) -> Geometry {
    Geometry::new(
        vec![1, 8, 1],
        positions.to_vec(),
        [[0.; 3]; 3],
        [false; 3],
        LengthUnit::Bohr,
    )
    .unwrap()
}
fn environment(geometry: &Geometry) -> (Vec<f64>, Vec<f64>) {
    let mut chi = vec![3., 4., 3.];
    let mut gradient = vec![0.; 27];
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                let delta = std::array::from_fn::<_, 3, _>(|a| {
                    geometry.positions()[i][a] - geometry.positions()[j][a]
                });
                let weight = (-delta.iter().map(|v| v * v).sum::<f64>() / 4.).exp();
                chi[i] += 0.2 * weight;
                for a in 0..3 {
                    let d = -0.1 * delta[a] * weight;
                    gradient[(i * 3 + i) * 3 + a] += d;
                    gradient[(i * 3 + j) * 3 + a] -= d;
                }
            }
        }
    }
    (chi, gradient)
}
#[test]
fn constrained_charge_response_includes_environment_and_conserves_charge() {
    let ctx = Context::new(Some(1), 4 << 20).unwrap();
    let positions = [[0., 0., 0.], [1.5, 0.3, -0.2], [2.5, 1., 0.1]];
    let solve = |g: &Geometry| {
        let (chi, gradient) = environment(g);
        dispersion::equilibrate(
            &ctx,
            g,
            ChargeParameters {
                electronegativities: &chi,
                hardnesses: &[0.3, 0.4, 0.3],
                widths: &[0.7, 0.9, 0.7],
                electronegativity_gradient: Some(&gradient),
            },
            0.5,
        )
        .unwrap()
    };
    let base = solve(&molecule(positions));
    assert!((base.values.as_slice().iter().sum::<f64>() - 0.5).abs() < 1e-14);
    for atom in 0..3 {
        for axis in 0..3 {
            let mut p = positions;
            let mut m = positions;
            let h = 1e-5;
            p[atom][axis] += h;
            m[atom][axis] -= h;
            let plus = solve(&molecule(p));
            let minus = solve(&molecule(m));
            let mut charge_response = 0.;
            for i in 0..3 {
                let analytic = base.response.as_slice()[(i * 3 + atom) * 3 + axis];
                let fd = (plus.values.as_slice()[i] - minus.values.as_slice()[i]) / (2. * h);
                assert!(
                    (fd - analytic).abs() < 3e-10,
                    "site {i}, moved {atom}, axis {axis}: {fd} vs {analytic}"
                );
                charge_response += analytic;
            }
            assert!(charge_response.abs() < 1e-13);
        }
    }
    for i in 0..3 {
        for axis in 0..3 {
            assert!(
                (0..3)
                    .map(|atom| base.response.as_slice()[(i * 3 + atom) * 3 + axis])
                    .sum::<f64>()
                    .abs()
                    < 1e-13
            );
        }
    }
    drop(base);
    assert_eq!(ctx.memory_used(), 0);
}
