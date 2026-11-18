//! Tests for linear-response Hubbard U measurement on crystals (periodic_hubbard_response).
mod hubbard_support;
mod setup_shells_support;

use hubbard_support::{load, HARTREE_EV};
use mandacaru_core::{
    models::{
        hubbard::Form,
        hubbard_request::Request,
        hubbard_response::{ColumnSource, Fit, Kernel, Options as ResponseOptions, Probe},
    },
    paw::digest::dataset_content_sha256,
    periodic::{
        hubbard_response::{
            hubbard_response, hubbard_response_with_symmetry, sum_over_states_chi0, PeriodicProbe,
        },
        response_scf::{converge_hubbard_u, Options as ScfLoopOptions},
        setup::parse::{create_setup, AtomSpec, SetupInput},
        Crystal, Options, PeriodicResult,
    },
    Context,
};
use nalgebra::Matrix3;
use setup_shells_support::dataset;
use std::sync::Arc;

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

    println!(
        "CuH chi0: {chi0:.6} Ha^-1, chi: {chi:.6} Ha^-1, U: {u:.6} Ha ({} eV)",
        u * HARTREE_EV
    );

    // Occupations must fall as shift rises -> chi0 and chi negative
    assert!(chi0 < 0.0, "chi0 must be negative");
    assert!(chi < 0.0, "chi must be negative");
    // Screening reduces response: |chi| < |chi0|, meaning chi > chi0
    assert!(chi > chi0, "dielectric screening must reduce response");
    // Therefore U = (chi0)^-1 - (chi)^-1 > 0
    assert!(u > 0.0, "screened Hubbard U must be positive");
    assert!(
        (res.u_charge[(0, 0)] - u).abs() < 1e-12,
        "u_charge equals u for 1 channel"
    );

    // Also verify that sum_over_states_chi0 agrees closely with finite-difference chi0
    let analytic_chi0 = sum_over_states_chi0(&c, &state).unwrap();
    let diff = (analytic_chi0[(0, 0)] - chi0).abs();
    println!(
        "CuH sum-over-states chi0: {:.6}, diff from FD: {:.3e}",
        analytic_chi0[(0, 0)],
        diff
    );
    assert!(
        diff < 1e-3,
        "sum-over-states chi0 must match finite difference"
    );
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
    assert!(
        (u_ni0 - u_ni1).abs() < 0.2,
        "Ni sites must have similar U values"
    );

    // j_response should be Some with 2 entries
    let j = res.j_response.as_ref().unwrap();
    assert_eq!(j.len(), 2);
    println!(
        "NiO j_response: {:.3} eV, {:.3} eV",
        j[0] * HARTREE_EV,
        j[1] * HARTREE_EV
    );

    // Compare with sum-over-states chi0
    let sos_chi0 = sum_over_states_chi0(&c, &state).unwrap();
    assert_eq!(sos_chi0.nrows(), 4);
    for i in 0..4 {
        println!(
            "NiO diag chi0[{i}] FD: {:.4}, SOS: {:.4}",
            res.chi0[(i, i)],
            sos_chi0[(i, i)]
        );
        assert!(res.chi0[(i, i)] < 0.0);
        assert!(sos_chi0[(i, i)] < 0.0);
    }
}

#[test]
fn symmetry_column_reduction_and_lr_scf_match_direct_response_on_crystals() {
    let ctx = context();
    // LR-6: NiO AFM-2 with mapped spin-flipped second Ni site matches unreduced full-column probe.
    let mut nio = load(&ctx, "nio_afm2");
    with_u(&mut nio, 0.0);
    let nio_state = solve(&nio, 0.05);
    let alphas = [0.05 / HARTREE_EV];
    let opts = ResponseOptions::new(&alphas, Fit::Central).unwrap();
    let full = hubbard_response_with_symmetry(
        &nio,
        &nio_state,
        &opts,
        &[ColumnSource::Direct, ColumnSource::Direct],
    )
    .unwrap();
    let reduced = hubbard_response_with_symmetry(
        &nio,
        &nio_state,
        &opts,
        &[
            ColumnSource::Direct,
            ColumnSource::Mapped {
                rep: 0,
                rows: vec![(1, true), (0, true)],
            },
        ],
    )
    .unwrap();
    let diff_u = (&full.u_matrix - &reduced.u_matrix).amax();
    let diff_chi = (&full.chi - &reduced.chi).amax();
    assert!(
        diff_u < 1e-4 && diff_chi < 1e-4,
        "LR-6 mismatch: dU={diff_u:e}, dchi={diff_chi:e}"
    );
    let spread_ok = full
        .hubbard_u(Form::Liechtenstein, Kernel::Charge, None, 0.5 / HARTREE_EV)
        .unwrap();
    assert_eq!(spread_ok.len(), 1);
    assert_eq!(spread_ok[0].0.key, "Ni");

    // LR-12: outer self-consistent U convergence on CuH.
    let mut cuh = load(&ctx, "cuh_7");
    with_u(&mut cuh, 0.0);
    let cuh_state = solve(&cuh, 0.01);
    let mut lr_opts = ScfLoopOptions::default();
    lr_opts.alphas = vec![0.05 / HARTREE_EV];
    lr_opts.max_iterations = 6;
    lr_opts.tolerance = 0.05 / HARTREE_EV;
    lr_opts.damping = 0.7;
    lr_opts.scf_options = Options {
        width: 0.01 / HARTREE_EV,
        ..Default::default()
    };
    let sc = converge_hubbard_u(&cuh, &cuh_state, &lr_opts).unwrap();
    assert!(sc.converged, "LR-12 did not converge: {:?}", sc.history);
    assert!(sc.response.self_consistent);
    assert_eq!(
        sc.final_hubbard.shells[0].source.as_deref(),
        Some("lr-scf")
    );
    assert!(sc.elements[0].1 > 0.0);
}

#[test]
fn spinor_crystal_linear_response_respects_trs_and_measures_magnetic_j() {
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap())
        .join("lda-fr");
    let build = |sym: &str, moment: f64, hub_spec: &str| {
        let species = [sym, "H"];
        let files: Vec<_> = species
            .iter()
            .map(|s| {
                let path = root.join(format!("{s}.parquet"));
                (
                    path.to_string_lossy().into_owned(),
                    dataset_content_sha256(&ctx, &path).unwrap(),
                )
            })
            .collect();
        let datasets: Vec<_> = species
            .iter()
            .map(|s| Arc::new(dataset(&ctx, "lda-fr", s)))
            .collect();
        let input = SetupInput {
            mesh: None,
            recipes: None,
            projector_basis: "raw".into(),
            lattice: Matrix3::from_diagonal(&nalgebra::Vector3::new(7.5, 7.5, 7.5)),
            grid_shape: [20, 20, 20],
            atoms: vec![
                AtomSpec {
                    species: 0,
                    symbol: sym.into(),
                    fractional: [0.15, 0.2, 0.2],
                    moment,
                },
                AtomSpec {
                    species: 1,
                    symbol: "H".into(),
                    fractional: [0.50, 0.2, 0.2],
                    moment: 0.0,
                },
            ],
            kpoints_frac: vec![[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]],
            weights: vec![0.5, 0.5],
            functional: "lda".into(),
            precision: "float64".into(),
            basis_size: "SZ".into(),
            spinor: true,
            spin: false,
            symmetry: false,
            filter_cutoff: None,
            sigma: None,
            dataset_files: Some(files),
            hubbard_u: Some(Request::parse(hub_spec).unwrap()),
            charge: None,
        };
        create_setup(&ctx, input, datasets)
            .unwrap()
            .build(&ctx)
            .unwrap()
    };

    // TRS spinor CuH: channels == 1, j_response == None, SOS chi0 agrees with FD.
    let cuh = build("Cu", 0.0, r#"{"Cu": {"l": "d", "U": 0.0}}"#);
    assert!(cuh.spinor && cuh.time_reversal);
    let cuh_state = solve(&cuh, 0.01);
    let res_trs = hubbard_response(&cuh, &cuh_state, &[0.05 / HARTREE_EV]).unwrap();
    assert_eq!(res_trs.response.channels, 1);
    assert!(res_trs.j_response.is_none());
    assert!(res_trs.u_matrix[(0, 0)] > 0.0);
    let sos_trs = sum_over_states_chi0(&cuh, &cuh_state).unwrap();
    assert!(
        (sos_trs[(0, 0)] - res_trs.chi0[(0, 0)]).abs() < 2e-3,
        "TRS spinor chi0 SOS {} vs FD {}",
        sos_trs[(0, 0)],
        res_trs.chi0[(0, 0)]
    );

    // Magnetic spinor NiH (!time_reversal): channels == 2, j_response is Some, u_cross is Some.
    let nih = build("Ni", 1.5, r#"{"Ni": {"l": "d", "U": 0.0}}"#);
    assert!(nih.spinor && !nih.time_reversal);
    let nih_state = solve(&nih, 0.05);
    let res_mag = hubbard_response(&nih, &nih_state, &[0.05 / HARTREE_EV]).unwrap();
    assert_eq!(res_mag.response.channels, 2);
    assert_eq!(res_mag.u_matrix.nrows(), 2);
    assert!(res_mag.j_response.is_some());
    assert!(res_mag.u_cross().is_some());
    assert!(res_mag.u_charge[(0, 0)] > 0.0);
    let sos_mag = sum_over_states_chi0(&nih, &nih_state).unwrap();
    assert_eq!(sos_mag.nrows(), 2);
    for i in 0..2 {
        assert!(
            (sos_mag[(i, i)] - res_mag.chi0[(i, i)]).abs() < 5e-3,
            "Magnetic spinor chi0[{i}] SOS {} vs FD {}",
            sos_mag[(i, i)],
            res_mag.chi0[(i, i)]
        );
    }
}
