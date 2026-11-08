//! The echo correlation and its spectrum (`quantum::dynamics`).
use mandacaru_core::{
    quantum::{dynamics::{fourier_spectrum, Propagator}, C64},
    Context,
};

fn context() -> Context {
    Context::new(Some(1), 64 * 1024 * 1024).unwrap()
}

#[test]
fn the_echo_correlation_of_a_flipped_spin_is_its_two_level_phase() {
    // H = Z, V = X on |0>: U V psi = e^{it}|1>, V U psi = e^{-it}|1>, so C(t) = e^{-2it}.
    let ctx = context();
    let h = Propagator::new(&ctx, 1, &[("Z".to_string(), C64::new(1., 0.))], 1).unwrap();
    let v = Propagator::new(&ctx, 1, &[("X".to_string(), C64::new(1., 0.))], 1).unwrap();
    let psi = [C64::new(1., 0.), C64::new(0., 0.)];
    let dt = 0.3;
    let c = h.echo_correlation(&ctx, &v, &psi, dt, 6, 4).unwrap();
    for (i, value) in c.iter().enumerate() {
        let expected = C64::from_polar(1., -2. * dt * i as f64);
        assert!((value - expected).norm() < 1e-12, "{i}: {value} vs {expected}");
    }
    assert!(h.echo_correlation(&ctx, &v, &psi, 0., 6, 4).is_err());
}

#[test]
fn the_spectrum_follows_the_shifted_fft_convention() {
    let ctx = context();
    for n in [8usize, 9] {
        let dt = 0.5;
        let k = 2usize;
        let omega = 2. * std::f64::consts::PI * k as f64 / (n as f64 * dt);
        let signal: Vec<C64> = (0..n).map(|t| C64::from_polar(1., omega * dt * t as f64)).collect();
        let (energies, amplitudes) = fourier_spectrum(&ctx, &signal, dt, 0., false).unwrap();
        // energies = 2 pi fftshift(fftfreq(n, dt)), ascending, zero at index n/2.
        assert!((energies[n / 2]).abs() < 1e-15);
        assert!(energies.windows(2).all(|w| w[1] > w[0]));
        let peak = (0..n).max_by(|&a, &b| amplitudes[a].norm().total_cmp(&amplitudes[b].norm())).unwrap();
        assert!((energies[peak] - omega).abs() < 1e-12, "n={n}");
        assert!((amplitudes[peak] - C64::new(dt * n as f64, 0.)).norm() < 1e-10);
    }
    let short = vec![C64::new(1., 0.); 3];
    assert!(fourier_spectrum(&ctx, &short, 0.5, 0., true).is_err());
}

#[test]
fn the_nested_otoc_at_zero_time_is_the_bare_operator_product() {
    // t = 0, B = X, M = Z: U_1 psi = X psi. On |0> (an M eigenstate, s = +1) the one-branch form
    // gives C = <1|Z|1> = -1; on |+> both branches give C = <Z X + | X Z +> = -1 as well.
    let ctx = context();
    let one = C64::new(1., 0.);
    let h = Propagator::new(&ctx, 1, &[("Z".to_string(), one)], 1).unwrap();
    let b = Propagator::new(&ctx, 1, &[("X".to_string(), one)], 1).unwrap();
    let m = Propagator::new(&ctx, 1, &[("Z".to_string(), one)], 1).unwrap();
    let zero = [one, C64::new(0., 0.)];
    let (state, c, measured, eigenvalue) = h.nested_otoc(&ctx, &b, &m, &zero, 0., 1, 1, 1e-10).unwrap();
    assert_eq!(eigenvalue, Some(1));
    assert!((measured + 1.).abs() < 1e-14 && (c - C64::new(-1., 0.)).norm() < 1e-14);
    assert!((state.as_slice()[1] - one).norm() < 1e-14);
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let plus = [C64::new(r, 0.), C64::new(r, 0.)];
    let (_, c, _, eigenvalue) = h.nested_otoc(&ctx, &b, &m, &plus, 0., 1, 1, 1e-10).unwrap();
    assert_eq!(eigenvalue, None);
    assert!((c - C64::new(-1., 0.)).norm() < 1e-14);
    // The series' first sample is the zero-time correlator.
    let series = h.nested_otoc_series(&ctx, &b, &m, &plus, 0.4, 3, 2, 2, 1e-10).unwrap();
    let (_, c2, _, _) = h.nested_otoc(&ctx, &b, &m, &plus, 0., 1, 2, 1e-10).unwrap();
    assert!((series[0] - c2).norm() < 1e-14 && series.len() == 3);
    assert!(h.nested_otoc(&ctx, &b, &m, &plus, 0., 1, 0, 1e-10).is_err());
}

#[test]
fn the_qpe_window_is_the_widened_gershgorin_bound() {
    use mandacaru_core::quantum::{dynamics::qpe_window, Operator};
    let ctx = context();
    let h = Operator::new(&ctx, 2, &[("II".into(), C64::new(-0.75, 0.)), ("ZI".into(), C64::new(0.25, 0.)),
                                     ("XY".into(), C64::new(0., 0.1))]).unwrap();
    let [lo, hi] = qpe_window(&h, 4, None).unwrap();
    assert!((lo + 1.1).abs() < 1e-15 && (hi - (lo + 0.7 * (1. + 2. / 16.))).abs() < 1e-15);
    // Only the identity: a unit window centred on it.
    let flat = Operator::new(&ctx, 1, &[("I".into(), C64::new(2., 0.))]).unwrap();
    let [lo, hi] = qpe_window(&flat, 3, None).unwrap();
    assert!((lo - 1.5).abs() < 1e-15 && (hi - (1.5 + 1.25)).abs() < 1e-15);
    let [lo, hi] = qpe_window(&h, 2, Some([-2., 2.])).unwrap();
    assert!(lo == -2. && (hi - (-2. + 4. * 1.5)).abs() < 1e-15);
    assert!(qpe_window(&h, 0, None).is_err());
}

#[test]
fn the_qpe_memory_counts_the_full_register() {
    use mandacaru_core::quantum::dynamics::{qpe_memory, QPE_WORKING_SET_FACTOR};
    let m = qpe_memory(4, 10, None, 16).unwrap();
    assert_eq!((m.statevector, m.unitary, m.dense), (16 << 14, 16 * 256, true));
    assert_eq!(m.working, QPE_WORKING_SET_FACTOR * m.statevector);
    let big = qpe_memory(30, 20, None, 8).unwrap();
    assert!(!big.dense && big.unitary == 0 && big.working > 1 << 50);
    assert!(qpe_memory(0, 4, None, 16).is_err());
    assert!(qpe_memory(100, 100, None, 16).is_err());
}
