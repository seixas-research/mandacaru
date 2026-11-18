//! Molecular DFT+U linear-response U/J tests (`MP-6`, `MP-7`, and spinor smearing `U3`).
mod paw_molecular_support;

use mandacaru_core::{
    models::{
        hubbard::Form,
        hubbard_response::{Kernel, Probe},
    },
    molecular::{
        hubbard_response::{hubbard_response, sum_over_states_chi0, MolecularProbe},
        occupation::DEFAULT_WIDTH,
        MolecularIntegrals, ScfOptions,
    },
    paw::molecular::Arrays,
    Context,
};
use paw_molecular_support::{analytic_shells, dataset, molecule_with};
use serde_json::json;

const HARTREE_EV: f64 = 27.211_386_245_988;

fn cuh_with_hubbard(library: &str) -> (Context, MolecularIntegrals) {
    let ctx = Context::new(Some(2), 2 << 30).unwrap();
    let cu_ds = dataset::read(&ctx, library, "Cu");
    let h_ds = dataset::read(&ctx, library, "H");
    let mut arrays = Arrays::new();
    let cu_sp = dataset::species(&ctx, &mut arrays, &cu_ds);
    let h_sp = dataset::species(&ctx, &mut arrays, &h_ds);
    let shells = analytic_shells(&[0, 1, 2]);
    let spec_str = json!({"Cu": {"l": "d", "U": 0.0}}).to_string();
    let extra = if library.ends_with("-fr") {
        vec![("hubbard", json!(spec_str)), ("spin_orbit", json!(true))]
    } else {
        vec![("hubbard", json!(spec_str))]
    };
    let integrals = molecule_with(
        &ctx,
        &[29, 1],
        &[[0.0, 0.0, -1.4], [0.0, 0.0, 1.4]],
        14.0,
        0.5,
        (vec![cu_sp, h_sp], vec![0, 1], arrays),
        "raw",
        &shells,
        &extra,
    );
    (ctx, integrals)
}

#[test]
fn closed_shell_and_unrestricted_molecular_hubbard_response_match_sos() {
    let (_ctx, integrals) = cuh_with_hubbard("lda-sr");
    let opts = ScfOptions::default();

    let ne = integrals.neutral_electrons();
    let n_pair = ne / 2;
    let rhf = integrals
        .kohn_sham("lda", [n_pair, n_pair], false, &opts)
        .unwrap();
    assert!(rhf.converged);
    let res = hubbard_response(&integrals, &rhf, "lda", &[0.05 / HARTREE_EV]).unwrap();
    assert_eq!(res.response.channels, 1);
    assert!(res.u_matrix[(0, 0)] > 0.0);
    let sos = sum_over_states_chi0(&integrals, &rhf, DEFAULT_WIDTH).unwrap();
    assert!(
        (sos[(0, 0)] - res.chi0[(0, 0)]).abs() < 1e-3,
        "molecular restricted chi0 SOS {} vs FD {}",
        sos[(0, 0)],
        res.chi0[(0, 0)]
    );
    let grouped = res
        .hubbard_u(Form::Dudarev, Kernel::SpinSummed, None, 0.1 / HARTREE_EV)
        .unwrap();
    assert_eq!(grouped[0].0.key, "Cu");

    // Open-shell CuH+ (unrestricted: 2 channels, j_response and u_cross present).
    let uhf = integrals
        .kohn_sham("lda", [n_pair, n_pair - 1], true, &opts)
        .unwrap();
    assert!(uhf.converged);
    let probe = MolecularProbe::new(&integrals, &uhf, "lda").unwrap();
    assert_eq!(probe.channels(), 2);
    let res_u = hubbard_response(&integrals, &uhf, "lda", &[0.05 / HARTREE_EV]).unwrap();
    assert_eq!(res_u.response.channels, 2);
    assert!(res_u.j_response.is_some());
    assert!(res_u.u_cross().is_some());
    assert!(res_u.u_charge[(0, 0)] > 0.0);
}

#[test]
fn spinor_molecular_smearing_and_hubbard_response_work_end_to_end() {
    let (_ctx, integrals) = cuh_with_hubbard("lda-fr");
    let opts = ScfOptions::default();
    let ne = integrals.neutral_electrons();
    let mf = integrals
        .spinor_kohn_sham_smeared("lda", ne, 0.01, &opts)
        .unwrap();
    assert!(mf.converged);
    assert_eq!(mf.smearing_width, Some(0.01));
    assert_eq!(mf.occupations.len(), 1);
    let res = hubbard_response(&integrals, &mf, "lda", &[0.05 / HARTREE_EV]).unwrap();
    assert_eq!(res.response.channels, 1);
    assert!(res.u_matrix[(0, 0)] > 0.0);
    let sos = sum_over_states_chi0(&integrals, &mf, 0.01).unwrap();
    assert!(
        (sos[(0, 0)] - res.chi0[(0, 0)]).abs() < 2e-3,
        "spinor molecular chi0 SOS {} vs FD {}",
        sos[(0, 0)],
        res.chi0[(0, 0)]
    );
}
