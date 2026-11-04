//! Integration test verifying DFT+U crystal forces and stresses wiring.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use mandacaru_core::{
    models::hubbard::Form,
    periodic::{
        gradient::{scf, Derivatives},
        hubbard::{Seed, Shell, Spec},
        stress_terms::crystal::{crystal_strain, crystal_stress, refuse as stress_refuse},
    },
    Error,
};
use periodic_gradient_crystals_support::{inputs, provider, Data};
use periodic_stress_crystals_support::load;

fn dummy_shell(u: f64) -> Shell {
    Shell {
        atom: 0,
        symbol: "Li".into(),
        l: 0,
        u,
        j: 0.,
        projectors: vec![0],
        weights: vec![1.],
        basis: None,
        seed: Seed::Moment {
            n_electrons: 1.,
            moment: 0.,
        },
    }
}

#[test]
fn hubbard_crystal_forces_and_stress_refusals_and_acceptance() {
    let mut data = Data::load("lih");

    // 1. Inactive/None hubbard hook returns None
    assert!(data.crystal.hubbard_hook().is_none());
    scf::refuse(&data.crystal).expect("plain crystal is differentiated");
    stress_refuse(&data.crystal).expect("plain crystal stress is differentiated");

    // Inactive hubbard (U - J = 0)
    data.crystal.hubbard = Some(Spec {
        form: Form::DudarevFll,
        shells: vec![dummy_shell(0.)],
    });
    assert!(!data.crystal.hubbard.as_ref().unwrap().is_active());
    assert!(data.crystal.hubbard_hook().is_none());
    scf::refuse(&data.crystal).expect("inactive hubbard crystal is differentiated");
    stress_refuse(&data.crystal).expect("inactive hubbard crystal stress is differentiated");

    // 2. Active hubbard (U = 0.1) on semilocal scalar crystal is ACCEPTED
    data.crystal.hubbard = Some(Spec {
        form: Form::DudarevFll,
        shells: vec![dummy_shell(0.1)],
    });
    assert!(data.crystal.hubbard.as_ref().unwrap().is_active());
    assert!(data.crystal.hubbard_hook().is_some());
    scf::refuse(&data.crystal).expect("active DFT+U semilocal crystal must be accepted");
    stress_refuse(&data.crystal).expect("active DFT+U semilocal crystal stress must be accepted");

    // 3. Active hubbard on spinor crystal must be REFUSED
    data.crystal.spinor = true;
    let err = scf::refuse(&data.crystal).expect_err("spinor DFT+U forces must be refused");
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("DFT+U")),
        "unexpected error: {err:?}"
    );
    let err_stress =
        stress_refuse(&data.crystal).expect_err("spinor DFT+U stress must be refused");
    assert!(
        matches!(&err_stress, Error::Unsupported(m) if m.contains("DFT+U")),
        "unexpected error: {err_stress:?}"
    );
    data.crystal.spinor = false;

    // 4. Active hubbard on hybrid crystal must be REFUSED
    let original_functional = data.crystal.functional.clone();
    data.crystal.functional = "hse06".to_string();
    let err_hyb = scf::refuse(&data.crystal).expect_err("hybrid DFT+U forces must be refused");
    assert!(
        matches!(&err_hyb, Error::Unsupported(m) if m.contains("DFT+U")),
        "unexpected error: {err_hyb:?}"
    );
    let err_hyb_stress =
        stress_refuse(&data.crystal).expect_err("hybrid DFT+U stress must be refused");
    assert!(
        matches!(&err_hyb_stress, Error::Unsupported(m) if m.contains("DFT+U")),
        "unexpected error: {err_hyb_stress:?}"
    );
    data.crystal.functional = original_functional;
}

#[test]
fn hubbard_provider_and_stress_wiring() {
    let mut data = load("lih");
    data.crystal.hubbard = Some(Spec {
        form: Form::DudarevFll,
        shells: vec![dummy_shell(0.2)],
    });

    let cores = data.cores();
    let prov = provider(&data, &cores);

    // Provider automatically carries the hubbard hook when crystal has active hubbard
    assert!(prov.hubbard().is_some());

    // Solve and obtain state
    let result = data.solve();
    let state = data.state(&result);

    // crystal_strain should successfully construct and carry the hubbard operator
    let strain_result = crystal_strain(
        &data.crystal.context,
        inputs(&data, &cores),
        data.layout(),
        data.operators(),
        &data.crystal,
        &data.options,
        &state,
        None,
    );
    assert!(strain_result.is_ok(), "crystal_strain failed: {:?}", strain_result.err());

    let c_strain = strain_result.unwrap();
    // Verify frozen.hubbard is populated!
    assert!(
        c_strain.frozen.hubbard.is_some(),
        "frozen.hubbard must be populated in CrystalStrain"
    );
    let hub_ops = c_strain.frozen.hubbard.as_ref().unwrap();
    assert_eq!(hub_ops.len(), data.crystal.spins);
    assert_eq!(hub_ops[0].shape(), (data.crystal.p, data.crystal.p));

    // Calculate crystal stress
    let stress_res = crystal_stress(
        &data.crystal.context,
        inputs(&data, &cores),
        data.layout(),
        data.operators(),
        &data.crystal,
        &data.options,
        &state,
        None,
    );
    assert!(stress_res.is_ok(), "crystal_stress failed: {:?}", stress_res.err());
    let stress = stress_res.unwrap();
    assert!(stress.volume > 0.0);
    assert!(stress.pressure().is_finite());
}
