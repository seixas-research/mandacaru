//! The hybrid crystal as a whole (W-E4P / P2): its refusals (no spheres, a changed dataset file,
//! a magnetic spinor), the spheres kept by a saved crystal, and the spinor without spin-orbit
//! coupling against the scalar hybrid. Needs `MANDACARU_PAW_PATH`.
mod periodic_screened_augmentation_support;
mod setup_support;

use mandacaru_core::{periodic::Crystal, Context};
use periodic_screened_augmentation_support::GB;
use setup_support::library_of;

/// The fixture as a hybrid on its first point, with `edit` applied to the JSON first.
fn gamma(
    ctx: &Context,
    name: &str,
    edit: &dyn Fn(&mut serde_json::Value),
) -> mandacaru_core::Result<Crystal> {
    use serde_json::json;
    let mut value = periodic_screened_augmentation_support::fixture(name);
    value["functional"] = json!("hse06");
    value["sites"] = setup_support::sites(&value, library_of(name));
    let mut first = value["kpoints"][0].clone();
    first["weight"] = json!(1.0);
    value["kpoints"] = json!([first]);
    value["permutations"] = json!([]);
    edit(&mut value);
    Crystal::from_json(ctx, &value.to_string())
}

#[test]
fn a_hybrid_crystal_without_its_spheres_is_refused_and_never_runs_unaugmented() {
    use serde_json::json;
    let ctx = Context::new(Some(1), 768 << 20).unwrap();
    let refused = |edit: &dyn Fn(&mut serde_json::Value)| -> String {
        gamma(&ctx, "si_crystal.json", edit)
            .err()
            .expect("the snapshot must be refused")
            .to_string()
    };
    assert!(refused(&|v| v["sites"] = serde_json::Value::Null).contains("sites"));
    assert!(
        refused(&|v| v["sites"]["datasets"][0]["sha256"] = json!("0".repeat(64)))
            .contains("SHA-256")
    );
    assert!(refused(&|v| v["sites"]["datasets"][1]["file"] = json!("")).contains("dataset file"));
    assert!(refused(&|v| v["sites"]["projector_basis"] = json!("both")).contains("projector basis"));
    assert!(
        refused(&|v| v["sites"]["datasets"] = json!([v["sites"]["datasets"][0].clone()]))
            .contains("one dataset per atom")
    );
    // A magnetic spinor hybrid is built: the spheres evaluate the spin-density functional.
    let magnetic = gamma(&ctx, "si_so.json", &|v| v["time_reversal"] = json!(false))
        .expect("a magnetic spinor hybrid is built");
    assert!(magnetic.spinor && !magnetic.time_reversal);
    // The same hybrid with its spheres loads, and the plain functional needs none.
    assert!(gamma(&ctx, "si_crystal.json", &|_| {}).is_ok());
    assert!(gamma(&ctx, "si_crystal.json", &|v| {
        v["functional"] = json!("pbe");
        v["sites"] = serde_json::Value::Null;
    })
    .is_ok());
}

#[test]
fn a_saved_hybrid_crystal_keeps_its_spheres() {
    let ctx = Context::new(Some(1), 768 << 20).unwrap();
    let crystal = gamma(&ctx, "al_fcc_gamma.json", &|_| {}).unwrap();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("p2_saved_hybrid.json");
    crystal.save(path.to_str().unwrap()).unwrap();
    let loaded = Crystal::load(&ctx, path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();
    let d = periodic_screened_augmentation_support::densities(&crystal, &[2]);
    let (a, b) = (
        crystal.hybrid_exchange(&d, GB).unwrap(),
        loaded.hybrid_exchange(&d, GB).unwrap(),
    );
    assert_eq!(a.energy.to_bits(), b.energy.to_bits());
    assert!(a.operators.iter().zip(&b.operators).all(|(x, y)| x == y));
}

#[test]
fn a_spinor_without_spin_orbit_coupling_is_the_scalar_hybrid() {
    use mandacaru_core::periodic::Options;
    use serde_json::json;
    let ctx = Context::new(Some(1), 768 << 20).unwrap();
    let options = Options {
        tolerance: 1e-11,
        density_tolerance: 1e-9,
        max_iterations: 200,
        ..Options::default()
    };
    let spinor = gamma(&ctx, "si_so.json", &|v| {
        let points = v["kpoints"].as_array_mut().unwrap();
        for point in points.iter_mut() {
            let size = point["spin_orbit"].as_array().unwrap().len();
            point["spin_orbit"] = json!(vec![[0.0, 0.0]; size]);
        }
    })
    .unwrap();
    let scalar = gamma(&ctx, "si_so.json", &|v| {
        let n = spinor_free_points(v);
        v["spinor"] = json!(false);
        v["time_reversal"] = json!(false);
        v["spins"] = json!(1);
        v["initial_m"] = json!(vec![0.0; n]);
        for point in v["kpoints"].as_array_mut().unwrap() {
            point.as_object_mut().unwrap().remove("spin_orbit");
        }
    })
    .unwrap();
    let (a, b) = (
        spinor.solve(&options).unwrap(),
        scalar.solve(&options).unwrap(),
    );
    println!(
        "Si lda-fr Gamma, no spin-orbit: spinor {:.12} scalar {:.12} ({:.2e}); {} and {} iterations",
        a.energy,
        b.energy,
        (a.energy - b.energy).abs(),
        a.history.len(),
        b.history.len()
    );
    assert!(a.converged && b.converged);
    assert!((a.energy - b.energy).abs() < 1e-9);
}

/// The number of grid points of a fixture (for its scalar magnetization field).
fn spinor_free_points(v: &serde_json::Value) -> usize {
    v["grid"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .product()
}
