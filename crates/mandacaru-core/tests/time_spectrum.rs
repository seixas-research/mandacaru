//! The time-to-frequency transforms of `models::time_spectrum` against closed forms: a damped
//! sinusoid, a sum of lines (the kicked dipole of a discrete spectrum) and the free electron gas
//! (TDDFT_PLAN.md section 13.5.3: the `1 / (omega + i gamma)` prefactor of `epsilon`).
use mandacaru_core::{
    models::time_spectrum::{
        damped_transform, dielectric_function, polarizability, strength_function, Window,
    },
    Context,
};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

fn context() -> Context {
    Context::new(Some(1), 64 * 1024 * 1024).unwrap()
}

fn samples(dt: f64, duration: f64, signal: impl Fn(f64) -> f64) -> Vec<f64> {
    let n = (duration / dt).round() as usize;
    (0..=n).map(|i| signal(i as f64 * dt)).collect()
}

const LINES: [(f64, f64); 3] = [(0.35, 0.8), (0.62, 0.15), (1.1, 0.45)];

fn lines_alpha(z: C64) -> C64 {
    LINES.iter().map(|&(w, f)| f / (w * w - z * z)).sum()
}

#[test]
fn a_damped_sinusoid_transforms_to_its_complex_frequency_pole_at_second_order() {
    let ctx = context();
    let (omega_n, gamma, duration) = (0.7, 0.05, 700.);
    let frequencies: Vec<f64> = (0..60).map(|i| 0.02 + 0.025 * i as f64).collect();
    let exact: Vec<C64> = frequencies
        .iter()
        .map(|&w| {
            let z = C64::new(w, gamma);
            omega_n / (omega_n * omega_n - z * z)
        })
        .collect();
    let error = |dt: f64| {
        let s = samples(dt, duration, |t| (omega_n * t).sin());
        let t = damped_transform(&ctx, &s, dt, &frequencies, gamma, Window::Exponential).unwrap();
        assert!(t.tail < 1e-15);
        t.values
            .as_slice()
            .iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).norm() / b.norm())
            .fold(0., f64::max)
    };
    let (coarse, fine) = (error(0.2), error(0.1));
    assert!(fine < 3e-3, "{fine:.3e}");
    let order = (coarse / fine).log2();
    assert!(
        (order - 2.).abs() < 0.1,
        "{coarse:.3e} {fine:.3e} order {order:.3}"
    );
}

#[test]
fn a_kicked_discrete_spectrum_gives_its_polarizability_and_line_areas() {
    let ctx = context();
    let (kappa, gamma, dt, duration) = (1e-3, 0.01, 0.1, 3000.);
    let dipole = samples(dt, duration, |t| {
        kappa
            * LINES
                .iter()
                .map(|&(w, f)| f / w * (w * t).sin())
                .sum::<f64>()
    });
    let frequencies: Vec<f64> = (0..400).map(|i| 0.005 * i as f64).collect();
    let alpha = polarizability(
        &ctx,
        &dipole,
        dt,
        kappa,
        &frequencies,
        gamma,
        Window::Exponential,
    )
    .unwrap();
    for (a, &w) in alpha.values.as_slice().iter().zip(&frequencies) {
        let exact = lines_alpha(C64::new(w, gamma));
        assert!(
            (a - exact).norm() < 5e-3 * exact.norm().max(1.),
            "omega {w}: {a} vs {exact}"
        );
    }
    // Static limit: alpha(0 + i gamma) is the sum over states, sum f / omega^2, up to gamma^2.
    let static_alpha: f64 = LINES.iter().map(|&(w, f)| f / (w * w)).sum();
    assert!((alpha.values.as_slice()[0].re - static_alpha).abs() < 2e-3 * static_alpha);
    // The strength function: its integral is the oscillator-strength sum (the lines' Lorentzian
    // tails outside [0, 3] carry about 1 % of it at this damping).
    let axis: Vec<f64> = (1..3000).map(|i| 1e-3 * i as f64).collect();
    let (s, tail) =
        strength_function(&ctx, &dipole, dt, kappa, &axis, gamma, Window::Exponential).unwrap();
    assert!(tail < 1e-12);
    let area: f64 = s.as_slice().iter().sum::<f64>() * 1e-3;
    let total: f64 = LINES.iter().map(|l| l.1).sum();
    assert!(
        (area / total - 1.).abs() < 0.03,
        "area {area:.4} for {total}"
    );
    // and it is positive: the sign convention of the kick.
    assert!(s.as_slice().iter().all(|v| *v > -1e-6));
    // The Gaussian window gives a line shape too: its peak sits at the line.
    let narrow: Vec<f64> = (0..201).map(|i| 0.30 + 0.0005 * i as f64).collect();
    let (g, _) =
        strength_function(&ctx, &dipole, dt, kappa, &narrow, 0.01, Window::Gaussian).unwrap();
    let peak = g
        .as_slice()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert!((narrow[peak] - 0.35).abs() < 1e-3);
}

#[test]
fn the_free_electron_gas_gives_the_drude_dielectric_function_with_the_complex_prefactor() {
    let ctx = context();
    let (density, a0, gamma, dt, duration) = (0.03, 1e-3, 0.02, 0.1, 1500.);
    // j = -n A_0 theta(t): the 0+ sample carries the jump.
    let current = samples(dt, duration, |_| -density * a0);
    let frequencies: Vec<f64> = (1..80).map(|i| 0.01 * i as f64).collect();
    let eps = dielectric_function(
        &ctx,
        &current,
        dt,
        a0,
        &frequencies,
        gamma,
        Window::Exponential,
    )
    .unwrap();
    let raw =
        damped_transform(&ctx, &current, dt, &frequencies, gamma, Window::Exponential).unwrap();
    for ((e, j), &w) in eps
        .values
        .as_slice()
        .iter()
        .zip(raw.values.as_slice())
        .zip(&frequencies)
    {
        let z = C64::new(w, gamma);
        let exact = C64::new(1., 0.) - 4. * PI * density / (z * z);
        // The trapezoid error, (z dt)^2 / 12 relative, is on the Drude term, not on epsilon,
        // which crosses zero at the plasma frequency.
        let drude = 4. * PI * density / (z * z).norm();
        assert!(
            (e - exact).norm() < 1e-3 * drude.max(exact.norm()),
            "omega {w}: {e} vs {exact}"
        );
        // The 1/omega prefactor is wrong: at omega = 0.05 it misses Im epsilon badly.
        if (w - 0.05).abs() < 1e-12 {
            let wrong = C64::new(1., 0.) - C64::new(0., 4. * PI) * j / (w * a0);
            assert!(
                (wrong - exact).norm() > 0.2 * exact.norm(),
                "{wrong} vs {exact}"
            );
        }
    }
}

#[test]
fn transforms_refuse_what_they_cannot_form() {
    let ctx = context();
    let s = vec![0., 1., 0.5];
    assert!(Window::parse("hann").is_err());
    assert!(dielectric_function(&ctx, &s, 0.1, 1e-3, &[0.1], 0.1, Window::Gaussian).is_err());
    assert!(dielectric_function(&ctx, &s, 0.1, 1e-3, &[0.], 0., Window::Exponential).is_err());
    assert!(damped_transform(&ctx, &s[..1], 0.1, &[0.1], 0.1, Window::Exponential).is_err());
    assert!(damped_transform(&ctx, &s, -0.1, &[0.1], 0.1, Window::Exponential).is_err());
    assert!(damped_transform(&ctx, &s, 0.1, &[f64::NAN], 0.1, Window::Exponential).is_err());
    assert!(polarizability(&ctx, &s, 0.1, 0., &[0.1], 0.1, Window::Exponential).is_err());
}

#[test]
fn the_user_unit_spectrum_is_the_atomic_one_in_converted_units() {
    use mandacaru_core::{
        models::time_spectrum::{spectrum_in_user_units, Kind},
        units::{Quantity, HARTREE_TO_EV},
    };
    let ctx = context();
    let (kappa, dt) = (1e-3, 0.1);
    let dipole = samples(dt, 500., |t| kappa * 0.8 / 0.5 * (0.5 * t).sin());
    let hartree = [0.4, 0.5, 0.6];
    let ev: Vec<f64> = hartree.iter().map(|w| w * HARTREE_TO_EV).collect();
    let (atomic, _) = strength_function(
        &ctx,
        &dipole,
        dt,
        kappa,
        &hartree,
        0.02,
        Window::Exponential,
    )
    .unwrap();
    let (user, _) = spectrum_in_user_units(
        &ctx,
        Kind::Strength,
        &dipole,
        dt,
        kappa,
        &ev,
        0.02 * HARTREE_TO_EV,
        Window::Exponential,
        false,
    )
    .unwrap();
    for (a, u) in atomic.as_slice().iter().zip(&user) {
        let expected = a * Quantity::InverseEnergy.factor(false);
        assert!((u.re - expected).abs() < 1e-12 * expected.abs().max(1e-12) && u.im == 0.);
    }
    let alpha = polarizability(
        &ctx,
        &dipole,
        dt,
        kappa,
        &hartree,
        0.02,
        Window::Exponential,
    )
    .unwrap();
    let (user, _) = spectrum_in_user_units(
        &ctx,
        Kind::Polarizability,
        &dipole,
        dt,
        kappa,
        &hartree,
        0.02,
        Window::Exponential,
        true,
    )
    .unwrap();
    assert!(alpha
        .values
        .as_slice()
        .iter()
        .zip(&user)
        .all(|(a, u)| (a - u).norm() == 0.));
    assert!(Kind::parse("loss").is_err());
}
