//! What `periodic::response::{forces, stress}` refuse, each with its cause: a dataset file whose
//! SHA-256 is not the one the snapshot was built from, a dataset that was not read from a file, a
//! dual projector basis, a state that did not converge, a hybrid or a spin-orbit crystal on a mesh
//! reduced by symmetry operations. Every refusal asserts the exception kind AND its cause text, and
//! the energy of the same snapshot stands. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_response_support;

use mandacaru_core::{
    periodic::{
        gradient::scf,
        response::{forces, stress},
        stress_terms::crystal,
        Crystal,
    },
    Context, Error, Result,
};
use periodic_gradient_crystals_support::Data;
use periodic_gradient_spinor_support::Spinor;
use periodic_response_support::{options_of, solve_snapshot, ForceCase};
use serde_json::Value;

fn refusal<T>(r: Result<T>) -> Error {
    match r {
        Err(e) => e,
        Ok(_) => panic!("expected a refusal"),
    }
}

/// The forces' and the stress' refusal of a crystal solved from `snapshot`: the same error kind and
/// message for both, the kind returned for the test to assert.
fn refused(snapshot: &Value, manifest: &Value) -> Error {
    let (crystal, result) = solve_snapshot(snapshot, manifest);
    assert!(result.converged);
    let (f, s) = (
        refusal(forces(&crystal, &result)),
        refusal(stress(&crystal, &result)),
    );
    assert_eq!(
        f.to_string(),
        s.to_string(),
        "forces and stress word it alike"
    );
    f
}

/// The control of the file/digest/basis refusals below: the snapshot they edit has forces and a
/// stress, so each edit alone is the cause.
#[test]
fn the_unedited_snapshot_has_forces_and_stress() {
    let c = ForceCase::load("lih");
    let (crystal, result) = solve_snapshot(&c.snapshot, &c.manifest);
    assert_eq!(forces(&crystal, &result).unwrap().len(), 2);
    assert!(stress(&crystal, &result).unwrap().volume > 0.);
}

#[test]
fn a_dataset_file_that_is_not_the_one_the_snapshot_was_built_from_is_refused() {
    let c = ForceCase::load("lih");
    let mut edited = c.snapshot.clone();
    edited["sites"]["datasets"][0]["sha256"] = "0".repeat(64).into();
    let error = refused(&edited, &c.manifest);
    assert!(matches!(error, Error::Invalid(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("SHA-256 differs"), "{message}");
    assert!(
        message.contains("is not the dataset file the snapshot was built from"),
        "{message}"
    );
}

#[test]
fn a_dataset_that_was_not_read_from_a_file_is_refused() {
    let c = ForceCase::load("lih");
    let mut edited = c.snapshot.clone();
    edited["sites"]["datasets"][0]["file"] = "".into();
    edited["sites"]["datasets"][0]["sha256"] = "".into();
    let error = refused(&edited, &c.manifest);
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("was not read from a file"), "{message}");
}

#[test]
fn a_dual_projector_basis_is_refused() {
    let c = ForceCase::load("lih");
    let mut edited = c.snapshot.clone();
    edited["sites"]["projector_basis"] = "dual".into();
    let error = refused(&edited, &c.manifest);
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains("differentiate the raw projector basis") && message.contains("\"dual\""),
        "{message}"
    );
}

#[test]
fn a_state_that_did_not_converge_is_refused() {
    let c = ForceCase::load("lih");
    let ctx = Context::new(Some(1), 1 << 30).unwrap();
    let crystal = Crystal::from_json(&ctx, &c.snapshot.to_string()).unwrap();
    let mut options = options_of(&c.manifest["scf"]);
    options.max_iterations = 2;
    let result = crystal.solve(&options).unwrap();
    assert!(!result.converged, "two iterations must not converge");
    for error in [
        refusal(forces(&crystal, &result)),
        refusal(stress(&crystal, &result)),
    ] {
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let message = error.to_string();
        assert!(
            message.contains("need a converged SCF") && message.contains("first order"),
            "{message}"
        );
    }
}

/// A hybrid with the symmetry table of a reduced mesh: Gamma is one point, so the load accepts the
/// table (the energy path refuses it only beside several points), and the derivative engines name
/// the reduction instead of averaging a k-sum that was never reduced. The Mandacaru setup never
/// builds such a crystal (a hybrid keeps the whole mesh, MG-F3 part 1); a snapshot from another
/// source can.
#[test]
fn a_hybrid_with_symmetry_operations_on_a_gamma_mesh_is_refused() {
    let mut data = Data::load("lih_hse06");
    assert_eq!(data.crystal.points.len(), 1, "a Gamma-only mesh");
    scf::refuse(&data.crystal).expect("the plain hybrid is differentiated");
    data.crystal.permutations = vec![vec![0]];
    let forces_cause = refusal(scf::refuse(&data.crystal)).to_string();
    let stress_cause = refusal(crystal::refuse(&data.crystal)).to_string();
    for message in [&forces_cause, &stress_cause] {
        assert!(
            message.contains("a hybrid crystal on a k mesh reduced by space-group symmetry")
                && message.contains("needs the whole mesh"),
            "{message}"
        );
    }
}

/// The loader forbids a spinor snapshot to list node permutations, so the only spinor crystal the
/// derivative engines can meet on a reduced mesh carries a moment symmetrization that is not the
/// identity (a hand-edited snapshot); it is refused with the cause.
#[test]
fn a_spin_orbit_crystal_on_a_reduced_mesh_is_refused() {
    let mut spinor = Spinor::load("si_ldafr");
    let crystal = &mut spinor.data.crystal;
    scf::refuse(crystal).expect("the spinor crystal on a whole mesh is differentiated");
    crystal::refuse(crystal).expect("and so is its stress");
    crystal.qsym.as_mut_slice()[1] = num_complex::Complex64::new(0.5, 0.);
    let forces_cause = refusal(scf::refuse(crystal)).to_string();
    let stress_cause = refusal(crystal::refuse(crystal)).to_string();
    for message in [&forces_cause, &stress_cause] {
        assert!(
            message.contains("a spin-orbit crystal on a k mesh reduced by space-group symmetry")
                && message.contains("spin rotation")
                && message.contains("symmetry=False"),
            "{message}"
        );
    }
}
