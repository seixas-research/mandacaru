//! AS-6 negative controls of the ORACLE: the same differencing with one pin of
//! `ANALYTIC_STRESS_PLAN.md` section 2 released does not describe the analytic stress. Each
//! frozen file (`lih_unpin_<pin>.json`, `freeze_periodic_strain.py --unpin <pin>`, the steps
//! 0.001/0.002/0.004) strains the same cell with the pin released: the Gaussian ion width
//! re-derived from the strained grid spacing, the node counts re-derived from the strained cell
//! (the third axis of the LiH cell, 12.47 nodes, becomes 13 at +0.4 percent), the dense
//! compensation list re-enumerated at every strained cell. The analytic stress is that of the
//! pinned functional, so the first two must miss it by more than 100 of the pinned oracle's
//! gates; the pinned oracle (`lih.json`, `lih_b.json`) meets them. The dense list is the finding
//! that does NOT bite: the Python build of the compensation sums changes by 2e-11 in `U` under a
//! 4e-3 strain (the energy by less than 1e-12 Ha), far below the estimate of AS-0 U3, so on
//! this route the pin is kept for correctness and measured to be below the gates. Needs
//! `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use periodic_stress_crystals_support::{
    gates::entries,
    load, manifest, oracle,
    run::{solve, SCF_DENSITY_TOLERANCE, SCF_TOLERANCE},
};

#[test]
fn an_oracle_that_releases_a_pin_misses_the_analytic_stress() {
    let mut data = load("lih");
    let volume = data.manifest["volume_bohr3"].as_f64().unwrap();
    let want = entries(&data.manifest, volume);
    let (_, _, stress) = solve(&mut data, SCF_TOLERANCE, SCF_DENSITY_TOLERANCE);
    for (pin, binding) in [("sigma", true), ("shape", true), ("dense", false)] {
        let released = manifest(&format!("lih_unpin_{pin}"));
        assert_eq!(
            released["pins"]["unpinned"],
            serde_json::json!([pin]),
            "the frozen file must release exactly the pin it is named for"
        );
        let wrong = oracle(&released);
        let mut worst = 0f64;
        for a in 0..3 {
            for b in 0..3 {
                let miss = (stress.gradient[a][b] - wrong[a][b]).abs() / want[a][b].gate;
                assert!(!miss.is_nan());
                worst = worst.max(miss);
            }
        }
        println!("  oracle without the {pin} pin: worst entry misses the analytic stress by {worst:.2e} gates");
        if binding {
            assert!(
                worst > 100.,
                "the {pin} pin does not matter: {worst:.2e} gates"
            );
        } else {
            assert!(
                worst < 1.,
                "the {pin} pin matters after all: {worst:.2e} gates"
            );
        }
    }
}
