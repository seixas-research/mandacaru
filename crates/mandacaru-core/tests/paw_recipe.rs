//! The recipe of a PAW-LCAO / UPAW-LCAO dataset (`paw::generate::recipe` and the policy tables
//! behind it, A3): every entry of every table against the frozen resolution of the Python
//! generator (`test/migration/reference/paw_recipe/policy.json`, written by
//! `test/migration/freeze_paw_recipe.py` from the live tables), the Bessel counts and norm
//! deficits against the parameters stored in the 184 shipped scalar-relativistic files
//! (`census_<set>.json` of the A0 freeze), and the refusals.
use mandacaru_core::models::configuration::{ground_state, valence_subshells};
use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::generate::recipe::{Cutoffs, Nlcc, Recipe};
use mandacaru_core::paw::generate::{
    tables_cutoffs, tables_elements as elements, tables_repairs, tables_upaw,
};
use mandacaru_core::paw::record::Family;
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

fn reference(relative: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference")
        .join(relative);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "frozen reference {} is missing ({e}); restore it with `python test/migration/freeze_paw_recipe.py`",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap()
}

fn radii(v: &Value) -> BTreeMap<usize, f64> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(l, r)| (l.parse().unwrap(), r.as_f64().unwrap()))
        .collect()
}

fn subshells(v: &Value) -> Vec<(usize, usize)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s[0].as_u64().unwrap() as usize,
                s[1].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

fn occupancies(v: &Value) -> Vec<((usize, usize), f64)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                (
                    s[0].as_u64().unwrap() as usize,
                    s[1].as_u64().unwrap() as usize,
                ),
                s[2].as_f64().unwrap(),
            )
        })
        .collect()
}

/// The elements whose reference configuration is relaxed away from the aufbau filling.
const RELAXED: [&str; 4] = ["La", "Ac", "Th", "Pa"];

#[test]
fn the_generic_constants_equal_the_frozen_ones() {
    let policy = reference("paw_recipe/policy.json");
    let c = &policy["constants"];
    let number = |key: &str| c[key].as_f64().unwrap();
    assert_eq!(elements::N_BESSEL as f64, number("n_bessel"));
    assert_eq!(elements::Q_CUT, number("q_cut"));
    assert_eq!(elements::ENERGY_OFFSET, number("energy_offset"));
    assert_eq!(elements::NORM_DEFICIT, number("norm_deficit"));
    assert_eq!(elements::RC_FACTOR, number("rc_factor"));
    assert_eq!(elements::LOCAL_FACTOR, number("local_factor"));
    assert_eq!(
        elements::SEMICORE_CUTOFF_FRACTION,
        number("semicore_cutoff_fraction")
    );
    assert_eq!(
        elements::FROZEN_SCATTERING_ENERGY,
        number("frozen_scattering_energy")
    );
    assert_eq!(
        elements::GGA_LOCAL_FIT_FRACTION,
        number("gga_local_fit_fraction")
    );
    let anchors = c["extra_anchor_energies"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(elements::EXTRA_ANCHOR_ENERGIES.to_vec(), anchors);
    for (symbol, points) in policy["generation_points"].as_object().unwrap() {
        let z = mandacaru_core::models::elements::atomic_number(symbol).unwrap();
        assert_eq!(
            elements::generation_points(z) as u64,
            points.as_u64().unwrap(),
            "{symbol}"
        );
    }
}

#[test]
fn every_dataset_resolves_to_the_frozen_recipe() {
    let policy = reference("paw_recipe/policy.json");
    let datasets = policy["datasets"].as_object().unwrap();
    assert_eq!(datasets.len(), 2 * 92);
    for (key, frozen) in datasets {
        let (symbol, functional) = key.split_once('/').unwrap();
        let recipe = Recipe::for_dataset(symbol, functional, Treatment::Scalar).unwrap();
        let number = |name: &str| frozen[name].as_f64().unwrap();
        assert_eq!(
            recipe.norm_deficit,
            number("norm_deficit"),
            "{key} norm deficit"
        );
        assert_eq!(
            recipe.n_bessel as u64,
            frozen["n_bessel"].as_u64().unwrap(),
            "{key} n_bessel"
        );
        assert_eq!(
            recipe.energy_offset,
            number("energy_offset"),
            "{key} energy offset"
        );
        assert_eq!(
            recipe.local_shift,
            number("local_shift"),
            "{key} local shift"
        );
        assert_eq!(
            recipe.second_reference,
            radii(&frozen["second_reference"]),
            "{key} second reference"
        );
        assert_eq!(
            recipe.cutoff_table,
            radii(&frozen["cutoff_table"]),
            "{key} cutoff table"
        );
        assert_eq!(
            recipe.frozen_subshells,
            subshells(&frozen["frozen_subshells"]),
            "{key} frozen"
        );
        assert_eq!(
            recipe.semicore_subshells,
            subshells(&frozen["semicore_subshells"]),
            "{key} semicore"
        );
        assert_eq!(
            recipe.highest_channel.map(|l| l as u64),
            frozen["highest_channel"].as_u64(),
            "{key} highest channel"
        );
        assert_eq!(
            recipe.reference_configuration,
            occupancies(&frozen["reference_configuration"]),
            "{key} reference configuration"
        );
        assert_eq!(
            recipe.core_radius,
            frozen["core_radius"].as_f64(),
            "{key} core radius"
        );
        assert_eq!(recipe.family, Family::PawLcao);
        assert_eq!(recipe.cutoffs, Cutoffs::Rule);
        assert_eq!(recipe.nlcc, Nlcc::SmoothCore);
        assert_eq!(recipe.extra_l, 0);
        assert!(recipe.local_radius.is_none());
        recipe.validate().unwrap();
    }
}

#[test]
fn every_upaw_dataset_resolves_to_the_frozen_recipe() {
    let policy = reference("paw_recipe/policy.json");
    let upaw = policy["upaw"].as_object().unwrap();
    let datasets = policy["datasets"].as_object().unwrap();
    for (key, frozen) in datasets {
        let (symbol, functional) = key.split_once('/').unwrap();
        let base = Recipe::for_dataset(symbol, functional, Treatment::Scalar).unwrap();
        let recipe = Recipe::upaw(symbol, functional, Treatment::Scalar).unwrap();
        let own = upaw.get(key);
        let option = |name: &str| own.and_then(|o| o.get(name));
        assert_eq!(recipe.family, Family::UpawLcao);
        assert_eq!(recipe.norm_deficit, 0., "{key}");
        assert_eq!(
            recipe.extra_l as u64,
            option("extra_l").map_or(0, |v| v.as_u64().unwrap()),
            "{key}"
        );
        assert_eq!(
            recipe.local_shift,
            option("local_shift").map_or(base.local_shift, |v| v.as_f64().unwrap()),
            "{key} local shift"
        );
        assert_eq!(
            recipe.energy_offset,
            option("energy_offset").map_or(base.energy_offset, |v| v.as_f64().unwrap()),
            "{key} energy offset"
        );
        assert_eq!(
            recipe.n_bessel as u64,
            option("n_bessel").map_or(base.n_bessel as u64, |v| v.as_u64().unwrap()),
            "{key} n_bessel"
        );
        assert_eq!(
            recipe.semicore_subshells,
            option("semicore_subshells").map_or(base.semicore_subshells.clone(), subshells),
            "{key} semicore"
        );
        assert_eq!(
            recipe.reference_configuration,
            option("reference_configuration")
                .map_or(base.reference_configuration.clone(), occupancies),
            "{key} reference configuration"
        );
        match option("r_cut") {
            Some(table) => assert_eq!(recipe.cutoffs, Cutoffs::PerChannel(radii(table)), "{key}"),
            None => assert_eq!(recipe.cutoffs, Cutoffs::Rule, "{key}"),
        }
        recipe.validate().unwrap();
        // The PAW-LCAO side of the same dataset is untouched by the family's options.
        assert_eq!(base.norm_deficit, frozen["norm_deficit"].as_f64().unwrap());
    }
    // The upaw entries of the table are exactly the frozen ones.
    let ours = tables_upaw::entries()
        .map(|(s, xc, _)| format!("{s}/{xc}"))
        .collect::<std::collections::BTreeSet<_>>();
    let theirs = upaw
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ours, theirs);
}

/// The tables hold exactly the entries of the frozen ones and no more: a lookup test above
/// cannot see an extra row.
#[test]
fn the_tables_hold_no_entry_the_freeze_does_not_have() {
    let policy = reference("paw_recipe/policy.json");
    let datasets = policy["datasets"].as_object().unwrap();
    let frozen = |key: String| datasets.contains_key(&key);
    for (symbol, functional, _) in tables_cutoffs::dataset_entries() {
        assert!(
            frozen(format!("{symbol}/{functional}")),
            "{symbol} {functional}"
        );
        let table = &datasets[&format!("{symbol}/{functional}")]["cutoff_table"];
        assert!(!table.as_object().unwrap().is_empty());
    }
    for (symbol, functional, _, _) in tables_repairs::s_entries() {
        assert!(frozen(format!("{symbol}/{functional}")));
    }
    for (symbol, functional, _, _) in tables_repairs::d_entries() {
        assert!(frozen(format!("{symbol}/{functional}")));
    }
    for (symbol, functional, count) in elements::n_bessel_entries() {
        assert_eq!(
            datasets[&format!("{symbol}/{functional}")]["n_bessel"]
                .as_u64()
                .unwrap(),
            count as u64
        );
    }
    for (symbol, functional, references) in elements::second_reference_entries() {
        assert_eq!(
            radii(&datasets[&format!("{symbol}/{functional}")]["second_reference"]),
            references.iter().copied().collect::<BTreeMap<_, _>>()
        );
    }
    for (name, entries) in elements::scalar_entries() {
        for (symbol, value) in entries {
            for functional in ["lda", "pbe"] {
                let frozen = &datasets[&format!("{symbol}/{functional}")];
                if name == "core_radius" {
                    assert_eq!(frozen[name].as_f64(), Some(*value));
                }
            }
        }
    }
}

/// The census of the shipped scalar-relativistic files stores, per file, the norm deficit and Bessel counts it
/// was built with and which parameters deviate from what the tables would give a first
/// construction. Where nothing deviates the recipe must say what the file says.
#[test]
fn the_recipe_reproduces_the_parameters_of_the_shipped_files() {
    let mut compared = 0;
    // The `fr` sets are not an oracle for the policy tables: 35 of their elements were built
    // before the tables of 10-20/21 (A0 finding, `PAW_COMPLETION_PLAN.md` 3.1).
    for (set, functional) in [("lda-sr", "lda"), ("pbe-sr", "pbe")] {
        let census = reference(&format!("paw_generate/census_{set}.json"));
        for (symbol, entry) in census["files"].as_object().unwrap() {
            let deviates = entry["deviates"].as_array().unwrap();
            let recipe = Recipe::for_dataset(symbol, functional, Treatment::Scalar).unwrap();
            if !deviates.iter().any(|d| d == "norm_deficit") {
                // The census omits a deficit of zero.
                let stored = entry.get("nd").map_or(0., |v| v.as_f64().unwrap());
                assert_eq!(recipe.norm_deficit, stored, "{set} {symbol}");
                compared += 1;
            }
            if !deviates.iter().any(|d| d == "n_bessel") {
                let counts = entry["nb"].as_array().unwrap();
                assert!(
                    counts
                        .iter()
                        .all(|c| c.as_u64().unwrap() as usize == recipe.n_bessel),
                    "{set} {symbol}: file {counts:?}, recipe {}",
                    recipe.n_bessel
                );
                compared += 1;
            }
            let frozen = entry.get("frozen").map(subshells).unwrap_or_default();
            assert_eq!(recipe.frozen_subshells, frozen, "{set} {symbol} frozen");
            let semicore = entry.get("semicore").map(subshells).unwrap_or_default();
            assert_eq!(
                recipe.semicore_subshells, semicore,
                "{set} {symbol} semicore"
            );
            // The empty channels above the valence: from the aufbau valence of the element with
            // the recipe's frozen and semicore moves, as the generator derives them.
            // La, Ac, Th and Pa are solved in a relaxed configuration (their aufbau 4f or 5f is a
            // box state), which only the atom knows: the generator tests cover them.
            if RELAXED.contains(&symbol.as_str()) {
                continue;
            }
            let z = recipe.atomic_number;
            let ground = ground_state(z).unwrap();
            let mut valence = valence_subshells(&ground);
            valence.retain(|s| !recipe.frozen_subshells.contains(s));
            valence.extend(recipe.semicore_subshells.iter().copied());
            let top = valence.iter().map(|s| s.1).max().unwrap();
            let frozen_top = recipe
                .frozen_subshells
                .iter()
                .map(|s| s.1)
                .max()
                .unwrap_or(top);
            let extra = frozen_top
                .saturating_sub(top)
                .max(recipe.highest_channel.unwrap_or(top).saturating_sub(top));
            let stored = entry
                .get("extra_l")
                .map_or(0, |v| v.as_u64().unwrap() as usize);
            assert_eq!(extra, stored, "{set} {symbol} extra_l");
            compared += 3;
        }
    }
    assert!(compared > 700, "only {compared} comparisons");
}

#[test]
fn what_the_generator_cannot_honor_is_refused() {
    let refused = |result: mandacaru_core::Result<Recipe>, needle: &str| {
        let message = result.err().expect("must be refused").to_string();
        assert!(message.contains(needle), "{message}");
    };
    refused(Recipe::for_dataset("O", "lda", Treatment::Dirac), "Dirac");
    refused(Recipe::upaw("O", "lda", Treatment::Dirac), "Dirac");
    for functional in ["hse06", "r2scan", "LDA", "pbe+rvv10", ""] {
        refused(
            Recipe::for_dataset("O", functional, Treatment::Scalar),
            "supports the functionals",
        );
    }
    refused(
        Recipe::for_dataset("Xx", "lda", Treatment::Scalar),
        "chemical symbol",
    );
    let base = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    let broken = |edit: &dyn Fn(&mut Recipe)| {
        let mut recipe = base.clone();
        edit(&mut recipe);
        recipe
            .validate()
            .err()
            .expect("must be refused")
            .to_string()
    };
    assert!(broken(&|r| r.q_cut = 0.).contains("q_cut"));
    assert!(broken(&|r| r.q_cut = f64::NAN).contains("q_cut"));
    assert!(broken(&|r| r.norm_deficit = 1.).contains("norm_deficit"));
    assert!(broken(&|r| r.norm_deficit = -0.1).contains("norm_deficit"));
    assert!(broken(&|r| r.n_bessel = 0).contains("Bessel"));
    assert!(broken(&|r| r.local_radius = Some(-1.)).contains("r_cut_local"));
    assert!(broken(&|r| r.cutoffs = Cutoffs::Uniform(0.)).contains("r_cut"));
    assert!(broken(&|r| r.nlcc = Nlcc::Radius(f64::INFINITY)).contains("nlcc"));
    assert!(broken(&|r| r.relativity = Treatment::Dirac).contains("Dirac"));
    assert!(broken(&|r| r.functional = "hse06".into()).contains("lda and pbe"));
    let mut upaw = Recipe::upaw("O", "lda", Treatment::Scalar).unwrap();
    upaw.norm_deficit = 0.05;
    assert!(upaw
        .validate()
        .unwrap_err()
        .to_string()
        .contains("UPAW-LCAO"));
}
