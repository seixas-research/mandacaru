//! `models::hubbard_response::measure`: the odd-part fit, the diagnostics, the checked
//! inversion and the full-matrix inversion on exactly solvable Kohn-Sham models. Every
//! reference is a closed form or an independent solver in `hubbard_response_support`,
//! never the formula that builds the model under test.
mod hubbard_response_support;

use hubbard_response_support::*;
use mandacaru_core::models::hubbard_response::{measure, Fit, Options};
use mandacaru_core::{Context, Error};
use nalgebra::DMatrix;

// ---- the fit -----------------------------------------------------------------------

#[test]
fn the_odd_part_drops_the_constant_and_the_even_term_and_inversion_is_exact() {
    let m0 = mat(2, &[-0.8, 0.1, 0.1, -0.5]);
    let m = mat(2, &[-0.3, 0.05, 0.05, -0.2]);
    let mut probe = Linear::new(m0.clone(), m.clone(), 1);
    probe.even = 3.0;
    let r = run(&mut probe, &[0.05], Fit::Central);
    assert!(worst(&r.chi0, &m0) < 1e-13 && worst(&r.chi, &m) < 1e-13);
    assert!(worst(&r.u, &(invert(&m0) - invert(&m))) < 1e-12);
    assert_eq!(r.nonlinearity, None, "one alpha gives no linearity control");
    assert_eq!(r.evaluations, 2 * 2 * 2);
}

#[test]
fn richardson_removes_the_cubic_term_and_central_does_not() {
    let m0 = mat(1, &[-0.8]);
    let m = mat(1, &[-0.3]);
    let mut probe = Linear::new(m0.clone(), m, 1);
    probe.cubic = 2.0;
    let mut options = Options::new(&[0.1, 0.2], Fit::Central).unwrap();
    options.nonlinearity_limit = f64::INFINITY;
    let central = measure(&context(), &mut probe, &options).unwrap();
    // D(a) = chi + cubic a^2, so the central estimate at a = 0.1 is off by 0.02.
    assert!((central.chi0[(0, 0)] - (-0.8 + 2.0 * 0.01)).abs() < 1e-13);
    options.fit = Fit::Richardson;
    let richardson = measure(&context(), &mut probe, &options).unwrap();
    assert!((richardson.chi0[(0, 0)] + 0.8).abs() < 1e-13);
    // The reported nonlinearity is the drift of the quotient over the alphas.
    // Both passes drift by 2 (0.04 - 0.01) = 0.06; chi is the smaller (-0.28 at a = 0.1).
    let drift = 0.06 / 0.28;
    assert!((richardson.nonlinearity.unwrap() - drift).abs() < 1e-12);
}

#[test]
fn central_error_scales_as_alpha_squared_and_richardson_as_alpha_to_the_fourth() {
    let mut toy = TwoLevel {
        w: 0.,
        kappa: 0.,
        u_in: 0.,
        vu: Vu::Absent,
        ..two_level(Vu::Absent)
    };
    let exact = toy.slope(toy.d0);
    let mut options = Options::new(&[0.1], Fit::Central).unwrap();
    options.nonlinearity_limit = f64::INFINITY;
    let mut error = |alphas: &[f64], fit: Fit| {
        options.alphas = alphas.to_vec();
        options.fit = fit;
        (measure(&context(), &mut toy, &options).unwrap().chi0[(0, 0)] - exact).abs()
    };
    let c: Vec<f64> = [0.2, 0.1, 0.05]
        .iter()
        .map(|a| error(&[*a], Fit::Central))
        .collect();
    let r: Vec<f64> = [0.025, 0.0125, 0.00625]
        .iter()
        .map(|a| error(&[*a, 2. * a], Fit::Richardson))
        .collect();
    println!(
        "central errors {c:?} ratios {:.3} {:.3}",
        c[0] / c[1],
        c[1] / c[2]
    );
    println!(
        "richardson errors {r:?} ratios {:.3} {:.3}",
        r[0] / r[1],
        r[1] / r[2]
    );
    for ratio in [c[0] / c[1], c[1] / c[2]] {
        assert!((3.7..4.3).contains(&ratio), "central ratio {ratio}");
    }
    for ratio in [r[0] / r[1], r[1] / r[2]] {
        assert!((14.0..18.0).contains(&ratio), "richardson ratio {ratio}");
    }
    // The pairs (0.1, 0.2) and (0.05, 0.1) are not yet asymptotic: the a^6 term flips the sign
    // of the error (scratch values 1.0e-6, -4.9e-8, -4.9e-9, -3.4e-10 at a = 0.1 ... 0.0125).
    assert!(r[2] < 1e-10 && r[2] < c[2] / 1e5);
}

#[test]
fn alphas_are_validated() {
    for bad in [
        &[][..],
        &[0.][..],
        &[-0.1][..],
        &[f64::NAN][..],
        &[0.1, 0.1][..],
    ] {
        assert!(
            matches!(Options::new(bad, Fit::Central), Err(Error::Invalid(_))),
            "{bad:?}"
        );
    }
    assert!(matches!(
        Options::new(&[0.1], Fit::Richardson),
        Err(Error::Invalid(_))
    ));
    // Unordered alphas are sorted: the smallest one drives the central estimate.
    assert_eq!(
        Options::new(&[0.2, 0.1], Fit::Richardson).unwrap().alphas,
        [0.1, 0.2]
    );
}

#[test]
fn nonlinearity_above_its_limit_is_refused_with_the_number() {
    let mut probe = Linear::new(mat(1, &[-1.]), mat(1, &[-0.5]), 1);
    probe.cubic = 40.;
    let options = Options::new(&[0.05, 0.1], Fit::Richardson).unwrap();
    let message = refusal(&mut probe, &options);
    assert!(message.contains("nonlinearity"), "{message}");
    probe.cubic = 0.;
    assert!(measure(&context(), &mut probe, &options).is_ok());
}

#[test]
fn a_misbehaving_probe_raises_and_never_returns_numbers() {
    let options = Options::new(&[0.05], Fit::Central).unwrap();
    assert!(refusal(&mut Faulty(Fault::ShortVector), &options).contains("occupations"));
    assert!(refusal(&mut Faulty(Fault::NotFinite), &options).contains("finite"));
    let failure = measure(&context(), &mut Faulty(Fault::Fails), &options);
    assert!(matches!(failure, Err(Error::Kernel(m)) if m.contains("diverged")));
}

#[test]
fn the_matrices_are_reserved_from_the_context() {
    let mut probe = Linear::new(
        mat(2, &[-1., 0., 0., -1.]),
        mat(2, &[-0.5, 0., 0., -0.5]),
        1,
    );
    let options = Options::new(&[0.05], Fit::Central).unwrap();
    let tiny = Context::new(Some(1), 64).unwrap();
    assert!(matches!(
        measure(&tiny, &mut probe, &options),
        Err(Error::Budget { .. })
    ));
    let ctx = context();
    measure(&ctx, &mut probe, &options).unwrap();
    assert_eq!(ctx.memory_used(), 0);
    assert!(ctx.memory_peak() > 0);
}

// ---- diagnostics and checked inversion ---------------------------------------------

#[test]
fn asymmetry_is_reported_before_symmetrizing_and_refused_above_its_limit() {
    let m0 = mat(2, &[-0.8, 0.10, 0.12, -0.5]);
    let m = mat(2, &[-0.3, 0.05, 0.05, -0.2]);
    let mut probe = Linear::new(m0.clone(), m, 1);
    let mut options = Options::new(&[0.05], Fit::Central).unwrap();
    assert!(refusal(&mut probe, &options).contains("asymmetric"));
    options.asymmetry_limit = 1.0;
    let r = measure(&context(), &mut probe, &options).unwrap();
    assert!((r.asymmetry - 0.02 / 0.8).abs() < 1e-12, "{}", r.asymmetry);
    assert!((r.chi0[(0, 1)] - 0.11).abs() < 1e-13 && r.chi0[(0, 1)] == r.chi0[(1, 0)]);
}

#[test]
fn an_ill_conditioned_response_raises_with_its_condition_number() {
    let mut probe = Linear::new(
        mat(2, &[-1., 0., 0., -1e-9]),
        mat(2, &[-0.5, 0., 0., -0.5e-9]),
        1,
    );
    let mut options = Options::new(&[0.01], Fit::Central).unwrap();
    let message = refusal(&mut probe, &options);
    assert!(
        message.contains("condition number 1.0") && message.contains("1e8"),
        "{message}"
    );
    options.condition_limit = 1e12;
    let r = measure(&context(), &mut probe, &options).unwrap();
    assert!((r.condition_chi0 / 1e9 - 1.).abs() < 1e-3);
}

#[test]
fn a_singular_interacting_response_raises_even_when_the_bare_one_is_regular() {
    let mut probe = Linear::new(mat(2, &[-1., 0., 0., -1.]), mat(2, &[-0.5, 0., 0., 0.]), 1);
    let message = refusal(&mut probe, &Options::new(&[0.01], Fit::Central).unwrap());
    assert!(message.starts_with("chi is singular"), "{message}");
    assert!(message.contains("condition number inf"), "{message}");
}

#[test]
fn a_bare_response_that_is_not_negative_definite_is_refused() {
    let mut probe = Linear::new(mat(2, &[-1., 0., 0., 0.5]), mat(2, &[-0.5, 0., 0., 0.2]), 1);
    let message = refusal(&mut probe, &Options::new(&[0.01], Fit::Central).unwrap());
    assert!(message.contains("negative definite"), "{message}");
}

#[test]
fn a_leaky_probe_violates_the_fixed_n_sum_rule_and_is_refused_on_request() {
    let m = mat(2, &[-0.5, 0.1, 0.1, -0.4]);
    let mut probe = Linear::new(m.clone(), mat(2, &[-0.3, 0.05, 0.05, -0.2]), 1);
    let mut options = Options::new(&[0.01], Fit::Central).unwrap();
    let r = measure(&context(), &mut probe, &options).unwrap();
    // Column sums: chi0 -0.4, -0.3 (largest entry 0.5); chi -0.25, -0.15 (largest 0.3).
    assert!((r.sum_rule - 0.25 / 0.3).abs() < 1e-9, "{}", r.sum_rule);
    options.sum_rule_limit = Some(1e-3);
    assert!(refusal(&mut probe, &options).contains("sum rule"));
}

// ---- exactly solvable Kohn-Sham models ---------------------------------------------

#[test]
fn two_level_model_recovers_every_kernel_the_potential_realizes() {
    let alphas = [0.005, 0.01];
    for (vu, name) in [
        (Vu::Absent, "absent"),
        (Vu::Frozen, "frozen"),
        (Vu::Responds, "responds"),
    ] {
        let mut toy = two_level(vu);
        let r = run(&mut toy, &alphas, Fit::Richardson);
        let f = r.u[(0, 0)];
        println!(
            "{name}: U {f:.12} exact {:.12} shifted-potential {:.12}",
            toy.exact_kernel(),
            toy.measured_kernel()
        );
        assert!((f - toy.exact_kernel()).abs() < 1e-7, "{name}");
        assert!(
            (toy.measured_kernel() - toy.exact_kernel()).abs() < 1e-7,
            "{name}"
        );
        // Dyson: chi^-1 = chi0^-1 - F with F the kernel measured on the potential.
        let dyson = 1. / r.chi0[(0, 0)] - 1. / r.chi[(0, 0)];
        assert!((dyson - toy.measured_kernel()).abs() < 1e-7, "{name}");
    }
}

#[test]
fn a_frozen_dudarev_potential_makes_u_independent_of_u_in_and_a_responding_one_does_not() {
    let alphas = [0.005, 0.01];
    let kernel = |vu, u_in| {
        let mut toy = TwoLevel {
            u_in,
            w: 0.9,
            kappa: 0.,
            ..two_level(vu)
        };
        run(&mut toy, &alphas, Fit::Richardson).u[(0, 0)]
    };
    for u_in in [0., 0.6, 1.5] {
        assert!(
            (kernel(Vu::Frozen, u_in) - 0.9).abs() < 1e-7,
            "frozen, U_in = {u_in}"
        );
        assert!(
            (kernel(Vu::Responds, u_in) - (0.9 - 0.5 * u_in)).abs() < 1e-7,
            "responds, U_in = {u_in}"
        );
    }
}

#[test]
fn the_hubbard_dimer_recovers_the_perturbation_theory_value() {
    let alphas = [0.005, 0.01];
    for (t, u) in [(1.0, 4.0), (0.7, 2.5), (1.0, 0.0)] {
        let mut dimer = Dimer { t, u };
        let r = run(&mut dimer, &alphas, Fit::Richardson);
        println!(
            "dimer t {t} U {u}: chi0 {:.10} chi {:.10} U_resp {:.10} analytic {:.10}",
            r.chi0[(0, 0)],
            r.chi[(0, 0)],
            r.u[(0, 0)],
            dimer.analytic_u()
        );
        assert!((r.chi0[(0, 0)] + 1. / (2. * t)).abs() < 1e-7);
        assert!(
            (r.u[(0, 0)] - dimer.analytic_u()).abs() < 1e-6,
            "t {t} U {u}"
        );
    }
}

fn kernel_block() -> DMatrix<f64> {
    // Symmetric, with sizeable off-diagonal terms so no shortcut can find them.
    DMatrix::from_fn(4, 4, |i, j| {
        if i == j {
            0.8
        } else {
            0.6 / (1. + (i as f64 - j as f64).abs())
        }
    })
}

fn ring(vu: Vu, u_in: f64) -> Ring {
    let mut ring = Ring::new(12, 10, vec![0, 1, 4, 7], kernel_block());
    ring.vu = vu;
    ring.u_in = u_in;
    ring
}

#[test]
fn on_a_twelve_site_ring_only_the_full_matrix_inversion_finds_the_kernel() {
    let alphas = [0.005, 0.01];
    for (vu, u_in) in [(Vu::Frozen, 1.0), (Vu::Responds, 1.0), (Vu::Absent, 0.)] {
        let mut toy = ring(vu, u_in);
        let exact = toy.exact_kernel();
        let r = run(&mut toy, &alphas, Fit::Richardson);
        let full = worst(&r.u, &exact);
        let scalar = (1. / r.chi0[(0, 0)] - 1. / r.chi[(0, 0)]) - exact[(0, 0)];
        let block = |k: usize| {
            let (a, b) = (
                r.chi0.view((0, 0), (k, k)).into_owned(),
                r.chi.view((0, 0), (k, k)).into_owned(),
            );
            worst(
                &(invert(&a) - invert(&b)),
                &exact.view((0, 0), (k, k)).into_owned(),
            )
        };
        println!(
            "ring {}: full {full:.2e} scalar {scalar:.3} 2-site block {:.3} 3-site block {:.3}",
            u_in,
            block(2),
            block(3)
        );
        assert!(full < 1e-6, "full-matrix inversion error {full}");
        assert!(
            scalar.abs() > 0.05 && block(2) > 0.05 && block(3) > 0.02,
            "the shortcuts are the failing controls"
        );
    }
}

#[test]
fn a_frozen_dudarev_term_gives_the_same_ring_kernel_at_any_u_in() {
    let alphas = [0.005, 0.01];
    let f = |u_in| run(&mut ring(Vu::Frozen, u_in), &alphas, Fit::Richardson).u;
    assert!(worst(&f(0.), &f(1.5)) < 1e-6);
}

#[test]
fn chi0_taken_at_the_bare_hamiltonian_is_wrong_by_much_more_than_the_floor() {
    let alphas = [0.005, 0.01];
    let mut toy = ring(Vu::Frozen, 1.0);
    toy.bare_chi0 = true;
    let error = worst(
        &run(&mut toy, &alphas, Fit::Richardson).u,
        &toy.exact_kernel(),
    );
    println!("bare-H0 chi0 error {error:.3}");
    assert!(error > 0.05);
}

#[test]
fn correlating_every_electron_holding_site_makes_the_response_singular() {
    let mut toy = Ring::new(12, 10, (0..12).collect(), DMatrix::zeros(12, 12));
    let options = Options::new(&[0.01], Fit::Central).unwrap();
    let message = refusal(&mut toy, &options);
    println!("{message}");
    assert!(
        message.contains("condition number") && message.contains("sum rule"),
        "{message}"
    );
    assert!(message.contains("every electron"), "{message}");
}
