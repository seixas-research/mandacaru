use mandacaru_core::{
    math::fft::Fft3,
    periodic::{spinor, Crystal, Options},
    quantum::C64,
    Context,
};
use nalgebra::DMatrix;
use serde_json::json;

fn complex(values: impl IntoIterator<Item = C64>) -> serde_json::Value {
    json!(values.into_iter().map(|v| [v.re, v.im]).collect::<Vec<_>>())
}
fn source(spinor: bool, coupling: f64) -> String {
    let n = 64;
    let psi = complex((0..2 * n).map(|i| {
        if i < n {
            C64::new(0.125, 0.)
        } else {
            C64::from_polar(
                0.125,
                2. * std::f64::consts::PI * ((i - n) / 16) as f64 / 4.,
            )
        }
    }));
    let mut point = json!({"k":[0.,0.,0.], "fractional":[0.,0.,0.], "weight":1.,
        "psi":psi, "overlap":[[1.,0.],[0.,0.],[0.,0.],[1.,0.]],
        "fixed":[[-0.5,0.],[0.,0.],[0.,0.],[0.3,0.]], "projections":[]});
    if spinor {
        let mut soc = DMatrix::<C64>::zeros(4, 4);
        // A transverse field on the first spatial orbital. Its eigenstates
        // carry y magnetization, so a z-only implementation cannot pass.
        soc[(0, 2)] = C64::new(0., -coupling);
        soc[(2, 0)] = C64::new(0., coupling);
        point["spin_orbit"] = complex((0..4).flat_map(|i| {
            let s = &soc;
            (0..4).map(move |j| s[(i, j)])
        }));
    }
    json!({"schema_version":1,"representation":"periodic-paw-snapshot","energy_unit":"Ha","length_unit":"bohr",
        "grid":{"shape":[4,4,4],"origin":[0.,0.,0.],"step":[1.,0.,0.,0.,1.,0.,0.,0.,1.]},
        "m":2,"p":0,"channels":0,"channel_labels":[],"kpoints":[point], "blocks":[],
        "core":vec![0.;n], "initial":vec![2./n as f64;n],"initial_m":vec![0.;if spinor {3*n} else {n}], "initial_q":[],
        "kernel":vec![0.;n],"xc_wave":vec![0.;3*n],"phase":complex(vec![C64::new(1.,0.);n]), "kerker":vec![1.;n],
        "ion":complex(vec![C64::default();n]),"hats":[],"u":[],"uion":[],"linear":[],
        "eion":0.,"ion_constant":0.,"constant":0.13,"reference":0.02,"electrons":2.,"spins":1,
        "spinor":spinor,"functional":"lda","relativistic":false,"permutations":[],"qsym":[]}).to_string()
}
#[test]
fn spinor_zero_soc_recovers_scalar_energy_without_double_counting_states() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let scalar = Crystal::from_json(&ctx, &source(false, 0.)).unwrap();
        let spinor = Crystal::from_json(&ctx, &source(true, 0.)).unwrap();
        let options = Options {
            width: 0.06,
            ..Default::default()
        };
        let a = scalar.solve(&options).unwrap();
        let b = spinor.solve(&options).unwrap();
        let symmetric = Crystal::from_json(
            &ctx,
            &source(true, 0.).replacen('{', "{\"time_reversal\":true,", 1),
        )
        .unwrap();
        let symmetric = symmetric.solve(&options).unwrap();
        assert!(a.converged && b.converged);
        assert!((a.free_energy - b.free_energy).abs() < 1e-11);
        assert!((symmetric.free_energy - b.free_energy).abs() < 1e-11);
        assert_eq!(b.eigenvalues[0].len(), 4);
        assert!((b.occupations[0].iter().sum::<f64>() - 2.).abs() < 1e-11);
        assert!(b.occupations[0].iter().all(|f| *f <= 1.));
        assert!(b.magnetization < 1e-12);
        for (a, b) in a.density.as_slice().iter().zip(b.density.as_slice()) {
            assert!((a - b).abs() < 1e-12);
        }
        let trials = DMatrix::<C64>::identity(4, 4);
        let values: Vec<_> = (0..4)
            .flat_map(|i| {
                let t = &trials;
                (0..4).map(move |j| t[(i, j)])
            })
            .collect();
        let projection =
            mandacaru_core::periodic::analysis::projectability(&spinor, &b, &values, 4).unwrap();
        assert!(projection
            .weights
            .as_slice()
            .iter()
            .all(|w| (w - 1.).abs() < 1e-12));
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn spinor_xc_is_the_derivative_in_all_four_density_components() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let mut c = Crystal::from_json(&ctx, &source(true, 0.)).unwrap();
        let n = c.grid.index.points();
        // Derivatives of real fields: zero Nyquist, all three directions.
        for a in 0..3 {
            for k in 0..n {
                let q = (k / 4usize.pow(2 - a as u32)) % 4;
                c.xc_wave.as_mut_slice()[a * n + k] = if q == 1 {
                    std::f64::consts::PI / 2.
                } else if q == 3 {
                    -std::f64::consts::PI / 2.
                } else {
                    0.
                };
            }
        }
        let rho: Vec<_> = (0..4 * n)
            .map(|i| {
                if i < n {
                    0.04 + 0.003 * (i as f64 * 0.2).sin()
                } else {
                    0.002 * (i as f64 * 0.4).cos()
                }
            })
            .collect();
        let direction: Vec<_> = (0..n).map(|i| 0.01 * (i as f64 * 0.3).cos()).collect();
        for functional in ["lda", "pbe"] {
            c.functional = functional.into();
            let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
            let terms = spinor::exchange_correlation(&c, &mut fft, &rho).unwrap();
            for axis in 0..4 {
                let energy = |sign: f64| {
                    let mut shifted = rho.clone();
                    for k in 0..n {
                        shifted[axis * n + k] += sign * 1e-5 * direction[k];
                    }
                    let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
                    spinor::exchange_correlation(&c, &mut fft, &shifted)
                        .unwrap()
                        .energy
                };
                let fd = (energy(1.) - energy(-1.)) / (2e-5);
                let analytic = (0..n)
                    .map(|k| terms.potential.as_slice()[axis * n + k] * direction[k])
                    .sum::<f64>();
                assert!(
                    (fd - analytic).abs() < 2e-9,
                    "{functional}, component {axis}: {fd} vs {analytic}"
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn spinor_density_and_hamiltonian_are_dual_including_complex_transverse_blocks() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let c = Crystal::from_json(&ctx, &source(true, 0.07)).unwrap();
        let n = c.grid.index.points();
        let d = DMatrix::from_fn(4, 4, |i, j| {
            C64::new(
                if i == j { 0.5 } else { 0.04 },
                0.03 * (i as f64 - j as f64),
            )
        });
        let (rho, _) = spinor::density(&c, std::slice::from_ref(&d)).unwrap();
        let potential: Vec<_> = (0..4 * n).map(|i| 0.1 * (i as f64 * 0.17).cos()).collect();
        let (eps, _, vectors) = spinor::diagonalize(&c, &c.points[0], &potential, &[]).unwrap();
        let h = &vectors
            * DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                4,
                eps.iter().map(|e| C64::new(*e, 0.)),
            ))
            * vectors.adjoint();
        let projected = (&d * h).trace().re;
        let direct = spinor::fixed_energy(&c.points[0], &d, c.m)
            + rho
                .as_slice()
                .iter()
                .zip(&potential)
                .map(|(r, v)| r * v)
                .sum::<f64>();
        assert!(
            (projected - direct).abs() < 1e-11,
            "{projected} vs {direct}"
        );
        assert!(rho.as_slice()[2 * n..3 * n].iter().any(|v| v.abs() > 1e-5));
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn spinor_scf_soc_energy_matches_hellmann_feynman_free_energy_derivative() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let options = Options {
            width: 0.12,
            tolerance: 1e-12,
            density_tolerance: 1e-10,
            ..Default::default()
        };
        let coupling = 0.04;
        let c = Crystal::from_json(&ctx, &source(true, coupling)).unwrap();
        let state = c.solve(&options).unwrap();
        assert!(state.converged);
        assert!(state.magnetization > 1e-4);
        let shifted = |delta: f64| {
            let c = Crystal::from_json(&ctx, &source(true, coupling + delta)).unwrap();
            let state = c.solve(&options).unwrap();
            assert!(state.converged);
            state.free_energy
        };
        let fd = (shifted(1e-5) - shifted(-1e-5)) / (2e-5);
        assert!((fd - state.spin_orbit_energy.unwrap() / coupling).abs() < 2e-8);
        let path =
            std::env::temp_dir().join(format!("mandacaru-spinor-{}.json", std::process::id()));
        c.save(path.to_str().unwrap()).unwrap();
        let loaded = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(loaded.spinor);
        assert_eq!(loaded.points[0].spin_orbit, c.points[0].spin_orbit);
        assert_eq!(loaded.initial_m.as_slice().len(), 3 * c.grid.index.points());
    }
    assert_eq!(ctx.memory_used(), 0);
}
