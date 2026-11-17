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
            let terms = spinor::exchange_correlation(&c, &mut fft, &rho, None).unwrap();
            for axis in 0..4 {
                let energy = |sign: f64| {
                    let mut shifted = rho.clone();
                    for k in 0..n {
                        shifted[axis * n + k] += sign * 1e-5 * direction[k];
                    }
                    let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
                    spinor::exchange_correlation(&c, &mut fft, &shifted, None)
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

/// A crystal of the meta-GGA tests: a 4 x 4 x 4 grid whose derivative is the spectral one up to
/// `pi / 2`, without a core, and a state whose spin frame turns (`rho` is charge and three
/// magnetization channels, `tau` its kinetic channels).
fn meta_state(ctx: &Context) -> (Crystal, Vec<f64>, Vec<f64>) {
    let mut c = Crystal::from_json(ctx, &source(true, 0.)).unwrap();
    let n = c.grid.index.points();
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
    c.core_tau = Some(ctx.allocate::<f64>(n).unwrap());
    c.functional = "r2scan".into();
    let rho = (0..4 * n)
        .map(|i| {
            if i < n {
                0.04 + 0.003 * (i as f64 * 0.2).sin()
            } else {
                0.002 * (i as f64 * 0.4).cos()
            }
        })
        .collect();
    let tau = (0..4 * n)
        .map(|i| {
            if i < n {
                0.05 + 0.004 * (i as f64 * 0.7).cos()
            } else {
                0.003 * (i as f64 * 0.9).sin()
            }
        })
        .collect();
    (c, rho, tau)
}

/// F5c: the spinor meta-GGA in the local spin frame is the derivative of its energy in all four
/// density components (which includes the frame term: the spin projections depend on the frame)
/// and in all four kinetic-density components.
#[test]
fn spinor_meta_gga_xc_is_the_derivative_in_every_density_and_kinetic_component() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (c, rho, tau) = meta_state(&ctx);
        let n = c.grid.index.points();
        let direction: Vec<_> = (0..n).map(|i| 0.01 * (i as f64 * 0.3).cos()).collect();
        let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
        let terms = spinor::exchange_correlation(&c, &mut fft, &rho, Some(&tau)).unwrap();
        let kinetic = terms
            .kinetic_potential
            .as_ref()
            .expect("a meta-GGA returns v_tau");
        for (which, field) in [("density", 0), ("kinetic density", 1)] {
            for axis in 0..4 {
                let energy = |sign: f64| {
                    let (mut r, mut t) = (rho.clone(), tau.clone());
                    let target = if field == 0 { &mut r } else { &mut t };
                    for k in 0..n {
                        target[axis * n + k] += sign * 1e-5 * direction[k];
                    }
                    let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
                    spinor::exchange_correlation(&c, &mut fft, &r, Some(&t))
                        .unwrap()
                        .energy
                };
                let fd = (energy(1.) - energy(-1.)) / (2e-5);
                let potential = if field == 0 {
                    terms.potential.as_slice()
                } else {
                    kinetic.as_slice()
                };
                let analytic = (0..n)
                    .map(|k| potential[axis * n + k] * direction[k])
                    .sum::<f64>();
                eprintln!(
                    "r2SCAN {which}, component {axis}: fd {fd:+.10e}, analytic {analytic:+.10e}"
                );
                assert!(
                    (fd - analytic).abs() < 2e-9,
                    "{which}, component {axis}: {fd} vs {analytic}"
                );
            }
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The partial core of a spinor meta-GGA enters the frame's spin densities by halves,
/// `n_up/dn = (n + core +- u . m) / 2` and `tau_up/dn = (tau_0 + core_tau +- u . tau_vec) / 2`, and
/// never the frame, which the valence fields alone build: `Terms::core` is the exact derivative of
/// the energy in the core and in its kinetic density. The Pauli charge channels of the potentials
/// carry the turn of the frame besides, so on a state whose frame turns they miss it (the second
/// assertion of each pair, the floor that makes the first one count).
#[test]
fn spinor_meta_gga_core_potentials_are_the_derivatives_in_the_core_and_its_kinetic_density() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mut c, rho, tau) = meta_state(&ctx);
        let n = c.grid.index.points();
        let (core, core_tau): (Vec<f64>, Vec<f64>) = (0..n)
            .map(|k| {
                (
                    0.01 + 0.002 * (k as f64 * 0.5).sin(),
                    0.012 + 0.002 * (k as f64 * 0.8).cos(),
                )
            })
            .unzip();
        c.core.as_mut_slice().copy_from_slice(&core);
        c.core_tau
            .as_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&core_tau);
        let direction: Vec<_> = (0..n).map(|i| 0.01 * (i as f64 * 0.3).cos()).collect();
        let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
        let terms = spinor::exchange_correlation(&c, &mut fft, &rho, Some(&tau)).unwrap();
        let sources = terms
            .core
            .as_ref()
            .expect("a spinor meta-GGA returns its core sources");
        let pauli_kinetic = terms
            .kinetic_potential
            .as_ref()
            .expect("a meta-GGA returns v_tau");
        for which in ["core", "core kinetic density"] {
            let mut energy = |sign: f64| {
                let (target, base) = if which == "core" {
                    (c.core.as_mut_slice(), &core)
                } else {
                    (c.core_tau.as_mut().unwrap().as_mut_slice(), &core_tau)
                };
                for k in 0..n {
                    target[k] = base[k] + sign * 1e-5 * direction[k];
                }
                let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
                spinor::exchange_correlation(&c, &mut fft, &rho, Some(&tau))
                    .unwrap()
                    .energy
            };
            let fd = (energy(1.) - energy(-1.)) / 2e-5;
            energy(0.);
            let (exact, pauli) = if which == "core" {
                (sources.potential.as_slice(), terms.potential.as_slice())
            } else {
                (sources.kinetic.as_slice(), pauli_kinetic.as_slice())
            };
            let contract = |v: &[f64]| (0..n).map(|k| v[k] * direction[k]).sum::<f64>();
            let (analytic, charge) = (contract(exact), contract(&pauli[..n]));
            eprintln!("r2SCAN {which}: fd {fd:+.10e}, exact {analytic:+.10e} ({:.2e}), Pauli charge channel {charge:+.10e} ({:.2e})", (fd - analytic).abs(), (fd - charge).abs());
            assert!((fd - analytic).abs() < 2e-9, "{which}: {fd} vs {analytic}");
            assert!(
                (fd - charge).abs() > 1e-8,
                "{which}: the frame does not turn here, the Pauli channel is as good: {charge}"
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The spinor meta-GGA takes the molecule's gradient form: the density gradients of its frame are
/// the projections `(grad n +- u . grad m) / 2` of the Pauli channels' spectral gradients, not the
/// gradients of the rotated densities `u . m`, which differ by `m . grad u` where the frame turns.
/// The state turns its frame (the second assertion is the floor that makes the first one count).
#[test]
fn spinor_meta_gga_gradients_are_the_projections_of_the_pauli_gradients() {
    use mandacaru_core::models::noncollinear::{rotate, rotate_gradients};
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (c, rho, tau) = meta_state(&ctx);
        let n = c.grid.index.points();
        let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
        let mut derivative = |field: &[f64], axis: usize| -> Vec<f64> {
            let mut work: Vec<C64> = field.iter().map(|v| C64::new(*v, 0.)).collect();
            fft.transform(&mut work, false).unwrap();
            for (k, w) in work.iter_mut().enumerate() {
                *w *= C64::new(0., c.xc_wave.as_slice()[axis * n + k]);
            }
            fft.transform(&mut work, true).unwrap();
            work.iter().map(|v| v.re).collect()
        };
        let pauli: Vec<f64> = (0..12)
            .flat_map(|row| derivative(&rho[(row / 3) * n..(row / 3 + 1) * n], row % 3))
            .collect();
        let rotated = rotate(&ctx, n, &rho, Some(&tau)).unwrap();
        let projected = rotate_gradients(&ctx, n, &rho, &rotated, &pauli).unwrap();
        let of_rotated: Vec<f64> = (0..6)
            .flat_map(|row| {
                derivative(
                    &rotated.densities.as_slice()[(row / 3) * n..(row / 3 + 1) * n],
                    row % 3,
                )
            })
            .collect();
        let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
        let terms = spinor::exchange_correlation(&c, &mut fft, &rho, Some(&tau)).unwrap();
        let seen = terms
            .gradients
            .as_ref()
            .expect("a meta-GGA returns its gradients");
        let gap = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0., f64::max)
        };
        let (to_projection, to_rotated) = (
            gap(seen.as_slice(), projected.as_slice()),
            gap(seen.as_slice(), &of_rotated),
        );
        eprintln!("frame gradients: to the projections {to_projection:.2e}, to the gradients of the rotated densities {to_rotated:.2e}");
        assert!(
            to_projection < 1e-14,
            "the gradients the functional saw are not the projections: {to_projection:e}"
        );
        assert!(
            to_rotated > 1e-6,
            "this state does not turn its frame: {to_rotated:e}"
        );
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The strain of the spectral gradient (the stress's X-3): scaling the wave vectors of axis `a` by
/// `1 - eps` at fixed nodes moves the energy by `-dv sum_s sum_i flux^s_a (partial_a n_s)`, with the
/// arrays the functional saw, so the frame's gradients and fluxes are the contraction's operands.
#[test]
fn spinor_meta_gga_spectral_gradient_strain_is_the_flux_contraction() {
    let ctx = Context::new(Some(1), 32 << 20).unwrap();
    {
        let (mut c, rho, tau) = meta_state(&ctx);
        let n = c.grid.index.points();
        let dv = c.grid.index.dv();
        let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
        let terms = spinor::exchange_correlation(&c, &mut fft, &rho, Some(&tau)).unwrap();
        let (g, f) = (
            terms.gradients.as_ref().unwrap(),
            terms.flux.as_ref().unwrap(),
        );
        let reference = c.xc_wave.as_slice().to_vec();
        for axis in 0..3 {
            let mut energy = |eps: f64| {
                c.xc_wave.as_mut_slice().copy_from_slice(&reference);
                for k in 0..n {
                    c.xc_wave.as_mut_slice()[axis * n + k] *= 1. - eps;
                }
                let mut fft = Fft3::new(&ctx, c.grid.index.shape()).unwrap();
                spinor::exchange_correlation(&c, &mut fft, &rho, Some(&tau))
                    .unwrap()
                    .energy
            };
            let h = 1e-4;
            let fd = (energy(h) - energy(-h)) / (2. * h);
            let analytic = -dv
                * (0..2)
                    .map(|s| {
                        let row = 3 * s + axis;
                        (0..n)
                            .map(|k| f.as_slice()[row * n + k] * g.as_slice()[row * n + k])
                            .sum::<f64>()
                    })
                    .sum::<f64>();
            eprintln!("strain of axis {axis}: fd {fd:+.10e}, X-3 {analytic:+.10e}");
            assert!(
                (fd - analytic).abs() < 1e-8 * fd.abs().max(1e-3),
                "axis {axis}: {fd} vs {analytic}"
            );
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
