//! Tests for linear-response Hubbard U measurement on crystals (periodic_hubbard_response).
mod hubbard_support;

use hubbard_support::{load, HARTREE_EV};
use mandacaru_core::{
    periodic::{
        hubbard_response::{hubbard_response, sum_over_states_chi0, PeriodicProbe},
        Crystal, Options, PeriodicResult,
    },
    models::hubbard_response::Probe,
    Context,
};

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

fn solve(c: &Crystal, width_ev: f64) -> PeriodicResult {
    let state = c
        .solve(&Options {
            width: width_ev / HARTREE_EV,
            ..Default::default()
        })
        .unwrap();
    assert!(state.converged, "the SCF did not converge");
    state
}

fn with_u(c: &mut Crystal, u_ev: f64) {
    c.hubbard
        .as_mut()
        .unwrap()
        .set_interaction(u_ev / HARTREE_EV, 0.)
        .unwrap();
}

#[test]
fn periodic_probe_shells_and_channels_match_crystal() {
    let ctx = context();
    let mut c = load(&ctx, "cuh_7");
    with_u(&mut c, 0.0);
    let state = solve(&c, 0.01);
    let probe = PeriodicProbe::new(&c, &state).unwrap();
    assert_eq!(probe.shells(), 1);
    assert_eq!(probe.channels(), 1);
}

#[test]
fn cuh_linear_response_returns_valid_matrices_and_positive_u() {
    let ctx = context();
    let mut c = load(&ctx, "cuh_7");
    with_u(&mut c, 0.0);
    let state = solve(&c, 0.01);

    // Shifts in Hartree: ~0.05 and 0.1 eV
    let alphas = [0.05 / HARTREE_EV, 0.1 / HARTREE_EV];
    let res = hubbard_response(&c, &state, &alphas).unwrap();

    assert_eq!(res.chi0.nrows(), 1);
    assert_eq!(res.chi0.ncols(), 1);
    assert_eq!(res.chi.nrows(), 1);
    assert_eq!(res.chi.ncols(), 1);
    assert_eq!(res.u_matrix.nrows(), 1);
    assert_eq!(res.u_matrix.ncols(), 1);

    let chi0 = res.chi0[(0, 0)];
    let chi = res.chi[(0, 0)];
    let u = res.u_matrix[(0, 0)];

    println!("CuH chi0: {chi0:.6} Ha^-1, chi: {chi:.6} Ha^-1, U: {u:.6} Ha ({} eV)", u * HARTREE_EV);

    // Occupations must fall as shift rises -> chi0 and chi negative
    assert!(chi0 < 0.0, "chi0 must be negative");
    assert!(chi < 0.0, "chi must be negative");
    // Screening reduces response: |chi| < |chi0|, meaning chi > chi0
    assert!(chi > chi0, "dielectric screening must reduce response");
    // Therefore U = (chi0)^-1 - (chi)^-1 > 0
    assert!(u > 0.0, "screened Hubbard U must be positive");
    assert!((res.u_charge[(0, 0)] - u).abs() < 1e-12, "u_charge equals u for 1 channel");

    // Also verify that sum_over_states_chi0 agrees closely with finite-difference chi0
    let analytic_chi0 = sum_over_states_chi0(&c, &state).unwrap();
    let diff = (analytic_chi0[(0, 0)] - chi0).abs();
    println!("CuH sum-over-states chi0: {:.6}, diff from FD: {:.3e}", analytic_chi0[(0, 0)], diff);
    assert!(diff < 1e-3, "sum-over-states chi0 must match finite difference");
}

#[test]
fn nio_linear_response_spin_resolved_kernel_and_j() {
    let ctx = context();
    let mut c = load(&ctx, "nio_afm2");
    with_u(&mut c, 0.0);
    let state = solve(&c, 0.05);

    // 1 alpha (central difference) for fast test
    let alphas = [0.05 / HARTREE_EV];
    let res = hubbard_response(&c, &state, &alphas).unwrap();

    // 2 Ni shells, 2 spin channels -> 4x4 matrix
    assert_eq!(res.chi0.nrows(), 4);
    assert_eq!(res.chi0.ncols(), 4);
    assert_eq!(res.chi.nrows(), 4);
    assert_eq!(res.u_matrix.nrows(), 4);
    assert_eq!(res.u_charge.nrows(), 2);
    assert_eq!(res.u_charge.ncols(), 2);

    // Check diagonal elements of u_charge (effective U per Ni atom in eV)
    let u_ni0 = res.u_charge[(0, 0)] * HARTREE_EV;
    let u_ni1 = res.u_charge[(1, 1)] * HARTREE_EV;
    println!("NiO Ni(0) U_charge: {u_ni0:.3} eV, Ni(1) U_charge: {u_ni1:.3} eV");
    assert!(u_ni0 > 0.0, "effective U on Ni(0) must be positive");
    assert!(u_ni1 > 0.0, "effective U on Ni(1) must be positive");
    // Due to symmetry between the two Ni sites, their U should be very close
    assert!((u_ni0 - u_ni1).abs() < 0.2, "Ni sites must have similar U values");

    // j_response should be Some with 2 entries
    let j = res.j_response.as_ref().unwrap();
    assert_eq!(j.len(), 2);
    println!("NiO j_response: {:.3} eV, {:.3} eV", j[0] * HARTREE_EV, j[1] * HARTREE_EV);

    // Compare with sum-over-states chi0
    let sos_chi0 = sum_over_states_chi0(&c, &state).unwrap();
    assert_eq!(sos_chi0.nrows(), 4);
    for i in 0..4 {
        println!("NiO diag chi0[{i}] FD: {:.4}, SOS: {:.4}", res.chi0[(i, i)], sos_chi0[(i, i)]);
        assert!(res.chi0[(i, i)] < 0.0);
        assert!(sos_chi0[(i, i)] < 0.0);
    }
}
