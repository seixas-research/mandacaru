//! `models::hubbard_response::kernel`: spin-resolved kernels, their reductions, `j_response`
//! and the grouping of shells by key.
mod hubbard_response_support;

use hubbard_response_support::*;
use mandacaru_core::models::hubbard_response::{Fit, Kernel, Response};
use mandacaru_core::Error;
use nalgebra::DMatrix;

// ---- spin channels -----------------------------------------------------------------

fn hund(delta: f64) -> Hund {
    Hund {
        t: 0.5,
        u: 2.0,
        j: 0.3,
        delta,
    }
}

#[test]
fn both_spin_channels_recover_the_full_two_by_two_kernel_and_j() {
    let alphas = [0.005, 0.01];
    for delta in [0., 0.3] {
        let mut toy = hund(delta);
        let r = run(&mut toy, &alphas, Fit::Richardson);
        let exact = mat(2, &[1.7, 2.0, 2.0, 1.7]);
        assert!(worst(&r.u, &exact) < 1e-6, "delta {delta}: {}", r.u);
        assert!((r.kernel(Kernel::SameSpin).unwrap()[(0, 0)] - 1.7).abs() < 1e-6);
        assert!((r.kernel(Kernel::Charge).unwrap()[(0, 0)] - 1.85).abs() < 1e-6);
        assert!(
            (r.j_response().unwrap()[0] - 0.3).abs() < 1e-6,
            "j_response"
        );
    }
}

#[test]
fn the_spin_summed_scalar_equals_the_charge_kernel_only_for_equivalent_channels() {
    let alphas = [0.005, 0.01];
    let summed = |delta| {
        let mut toy = hund(delta);
        let r = run(&mut toy, &alphas, Fit::Richardson);
        let chi0 = toy.chi0();
        let a = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
            1. / chi0[0],
            1. / chi0[1],
        ]));
        let chi = invert(&(a - mat(2, &[1.7, 2.0, 2.0, 1.7])));
        let independent = 1. / (chi0[0] + chi0[1]) - 1. / chi.sum();
        (
            r.kernel(Kernel::SpinSummed).unwrap()[(0, 0)],
            independent,
            r.kernel(Kernel::Charge).unwrap()[(0, 0)],
        )
    };
    let (symmetric, independent, charge) = summed(0.);
    assert!((symmetric - charge).abs() < 1e-6 && (independent - 1.85).abs() < 1e-12);
    let (inequivalent, independent, charge) = summed(0.3);
    println!("inequivalent channels: spin-summed {inequivalent:.8} charge {charge:.8}");
    assert!((inequivalent - independent).abs() < 1e-6);
    assert!(
        (inequivalent - charge).abs() > 1e-5,
        "the two numbers differ when the channels do"
    );
}

#[test]
fn a_restricted_probe_has_only_the_spin_summed_kernel() {
    let r = run(&mut two_level(Vu::Frozen), &[0.005, 0.01], Fit::Richardson);
    assert!((r.kernel(Kernel::SpinSummed).unwrap()[(0, 0)] - r.u[(0, 0)]).abs() < 1e-15);
    for kernel in [Kernel::SameSpin, Kernel::Charge] {
        assert!(
            matches!(r.kernel(kernel), Err(Error::Invalid(m)) if m.contains("two spin channels"))
        );
    }
    assert!(matches!(r.j_response(), Err(Error::Invalid(_))));
}

// ---- grouping ----------------------------------------------------------------------

fn diagonal_kernel(values: &[f64]) -> Response {
    let n = values.len();
    let chi0 = DMatrix::from_diagonal_element(n, n, -1.);
    let chi = invert(
        &(invert(&chi0) - DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(values))),
    );
    run(&mut Linear::new(chi0, chi, 1), &[0.01], Fit::Central)
}

#[test]
fn shells_group_by_key_and_a_spread_above_the_tolerance_refuses() {
    let r = diagonal_kernel(&[4.0, 4.2, 1.0]);
    let groups = r
        .hubbard_u(Kernel::SpinSummed, &["Ni", "Ni", "O"], 0.25)
        .unwrap();
    assert_eq!(
        groups.iter().map(|g| g.key.as_str()).collect::<Vec<_>>(),
        ["Ni", "O"]
    );
    assert!((groups[0].u - 4.1).abs() < 1e-9 && (groups[0].spread - 0.1).abs() < 1e-9);
    assert_eq!(groups[0].shells, [0, 1]);
    let message = match r.hubbard_u(Kernel::SpinSummed, &["Ni", "Ni", "O"], 0.05) {
        Err(Error::Invalid(m)) => m,
        other => panic!("{other:?}"),
    };
    assert!(
        message.contains("Ni") && message.contains("spread"),
        "{message}"
    );
    assert!(r.hubbard_u(Kernel::SpinSummed, &["Ni", "O"], 1.0).is_err());
    assert!(r
        .hubbard_u(Kernel::SpinSummed, &["Ni", "Ni", "O"], -1.0)
        .is_err());
}
