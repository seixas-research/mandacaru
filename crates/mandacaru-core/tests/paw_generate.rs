//! The PAW generator (`paw::generate`, A3) against the A0 freeze of the Python generator
//! (`test/migration/reference/paw_generate/gen_*`): scalar-relativistic LDA and PBE datasets and
//! the unitary family, first construction (`ghosts="keep"`), the ghost-search remedy of the
//! selected attempt applied as recipe fields where the search had to repair it.
//!
//! Layers, so that a miss localizes itself: the reference atom (iteration count, energy, every
//! level), then the references and smooth waves against the inputs the Python assemble was
//! called with (the hand-off to A2), the local potential, every table and scalar of the dataset.
//! The A0 freeze is Python and uses the atom-solver radial exchange-correlation scheme, so the
//! parity gate runs through that scheme (`AtomSolver`, the legacy seam); the production path
//! (`generate`, the variational scheme) is gated where the two schemes agree and its documented
//! difference is measured, not pinned to the freeze. Negative controls are in
//! `paw_generate_controls.rs`.
mod paw_assemble_support;
mod paw_generate_support;

use mandacaru_core::models::xc_scheme::{AtomSolver, Variational};
use mandacaru_core::paw::generate::reference::{solve_reference, solve_reference_with};
use mandacaru_core::paw::generate::{generate, generate_with_atom};
use mandacaru_core::paw::record::PawDataset;
use mandacaru_core::{Context, Error};
use paw_assemble_support::Frozen;
use paw_generate_support::compare::{compare_atom, Scheme};
use paw_generate_support::dataset::compare_dataset;
use paw_generate_support::{arrays, recipe_of, Decimated, HEAVY, LIGHT};

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

/// One frozen dataset through the legacy seam, atom first, checked layer by layer.
fn parity(name: &str) {
    let f = Frozen::open(name);
    let data = Decimated::open(name);
    let recipe = recipe_of(&f);
    let ctx = context();
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    compare_atom(&f, &atom).assert_within(&format!("{name} atom"));
    let started = std::time::Instant::now();
    let generated = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    println!(
        "{name}: generated in {:.1} s, context peak {} B",
        started.elapsed().as_secs_f64(),
        ctx.memory_peak()
    );
    compare_dataset(&f, &data, &generated, Scheme::Legacy).assert_within(name);
    drop(generated);
    drop(atom);
    assert_eq!(
        ctx.memory_used(),
        0,
        "{name}: the generator gave its memory back"
    );
}

#[test]
fn the_light_sample_equals_the_frozen_datasets() {
    for name in LIGHT {
        parity(name);
    }
}

macro_rules! heavy {
    ($($test:ident => $name:expr),*) => {$(
        #[test]
        #[ignore = "minutes: cargo test --release -p mandacaru-core --test paw_generate -- --ignored"]
        fn $test() {
            parity($name);
        }
    )*};
}

heavy!(
    heavy_lda_sr_ce => "lda-sr_Ce",
    heavy_pbe_sr_ce => "pbe-sr_Ce",
    heavy_lda_sr_au => "lda-sr_Au",
    heavy_lda_sr_pb => "lda-sr_Pb",
    heavy_lda_sr_u => "lda-sr_U"
);

#[test]
fn the_sample_covers_every_frozen_scalar_relativistic_dataset() {
    let mut frozen = paw_assemble_support::datasets()
        .into_iter()
        .filter(|n| !n.starts_with("lda-fr") && !n.starts_with("pbe-fr"))
        .collect::<Vec<_>>();
    let mut ours = LIGHT
        .iter()
        .chain(HEAVY.iter())
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    frozen.sort();
    ours.sort();
    assert_eq!(frozen, ours, "the Dirac sets are the spin-orbit package's");
}

/// Items whose value reads the all-electron potential beyond the density floor of the exchange
/// kernel (about 14 Bohr): the local potential, and the waves of a reference above zero, whose
/// outer part follows the box (the reference atom's tail potential differs from the atom-solver
/// one by the uniform-gas continuation below a density of 2e-12).
fn reads_the_tail(name: &str) -> bool {
    name.contains("v_loc")
        || (name.contains("wave") && name.rsplit('_').next().is_some_and(|k| k == "1"))
}

/// The round-off diagnostics of the generator (duality error, asymmetry, minimum overlap...) and
/// the norm correction (a round-off array here) are compared by order of magnitude in the parity
/// test, not value by value: a production run does not reproduce them to a gate and they are not
/// part of what the scheme changes.
fn is_round_off_diagnostic(name: &str) -> bool {
    // `norm_correction` is the round-off of the smooth waves' norm condition (entries of 1e-14 to
    // 1e-11 for these zero-deficit constructions): its maximum is noise, so it has no scale.
    name.starts_with("diagnostic.") || name.contains("norm_correction")
}

/// The production path (variational scheme, atom included: D-2b) against the freeze of the
/// atom-solver scheme, LDA. The two schemes differ for LDA only in the far tail (the atom-solver
/// scheme continues the uniform gas below a density of 2e-12, a step of 1.86e-4 Ha at about 14
/// Bohr; the kernel drops it), and the tail reaches the dataset through the reference atom: the
/// eigenvalues move by 2.4e-10 Ha (oxygen), the bound-state quantities of the sphere by about
/// that, and whatever follows a reference above zero (the second scattering reference, the
/// couplings built on it) by more, since its wave is normalized over the box. The gates are the
/// measured worst of each dataset rounded up (`sphere`: the items that do not read the tail,
/// `tail`: those that do), in units of the maximum of the array. Measured worst over the three:
/// oxygen 3.0e-10 and 9.5e-5, iron 2.3e-4 (a projector of the s channel, whose reference sits
/// next to a box-sensitive second one) and 7.3e-4, unitary aluminum 1.8e-4 (the empty d channel's
/// reference above zero) and 5.3e-5; "LDA unchanged at 1e-12" holds
/// through the seam (`paw_generate` parity, `paw_one_center`, the exact terms), not for the
/// production atom.
const PRODUCTION_LDA: [(&str, f64, f64); 3] = [
    ("lda-sr_O", 1e-9, 3e-4),
    ("lda-sr_Fe", 1e-3, 3e-3),
    ("upaw_Al", 1e-3, 3e-4),
];

#[test]
fn the_production_path_equals_the_freeze_for_lda_to_the_size_of_the_far_tail() {
    for (name, sphere_gate, tail_gate) in PRODUCTION_LDA {
        let f = Frozen::open(name);
        let data = Decimated::open(name);
        let recipe = recipe_of(&f);
        let ctx = context();
        let generated = generate(&ctx, &recipe).unwrap();
        let comparison = compare_dataset(&f, &data, &generated, Scheme::Production);
        let (mut worst_sphere, mut worst_tail) = ((0f64, String::new()), (0f64, String::new()));
        for item in comparison
            .items
            .iter()
            .filter(|i| !is_round_off_diagnostic(&i.name))
        {
            let (gate, slot) = if reads_the_tail(&item.name) {
                (tail_gate, &mut worst_tail)
            } else {
                (sphere_gate, &mut worst_sphere)
            };
            assert!(
                item.deviation <= gate,
                "{name} {}: {:.3e} > {gate:.0e}",
                item.name,
                item.deviation
            );
            if item.deviation > slot.0 {
                *slot = (item.deviation, item.name.clone());
            }
        }
        println!("production LDA {name}: sphere {worst_sphere:?}, tail {worst_tail:?}");
        let mut all = comparison
            .items
            .iter()
            .filter(|i| !is_round_off_diagnostic(&i.name))
            .map(|i| (i.deviation, i.name.as_str()))
            .collect::<Vec<_>>();
        all.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!(
            "production LDA {name}: largest {:?}",
            &all[..8.min(all.len())]
        );
        // The gates are not slack: each is within a factor of thirty of what was measured.
        assert!(
            worst_sphere.0 > sphere_gate / 30. && worst_tail.0 > tail_gate / 30.,
            "{name}: tighten the gates"
        );
    }
}

#[test]
fn the_production_path_moves_the_pbe_dataset_by_the_measured_scheme_difference() {
    for (name, e1c_bounds) in [("pbe-sr_O", (1e-5, 1e-3)), ("pbe-sr_Cu", (1e-5, 1e-3))] {
        let f = Frozen::open(name);
        let data = Decimated::open(name);
        let recipe = recipe_of(&f);
        let ctx = context();
        let production = generate(&ctx, &recipe).unwrap();
        let seam = {
            let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
            generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap()
        };
        let comparison = compare_dataset(&f, &data, &production, Scheme::Legacy);
        let worst = comparison
            .items
            .iter()
            .filter(|i| !reads_the_tail(&i.name) && !is_round_off_diagnostic(&i.name))
            .fold((0f64, String::new()), |w, i| {
                if i.deviation > w.0 {
                    (i.deviation, i.name.clone())
                } else {
                    w
                }
            });
        let mut movers = comparison
            .items
            .iter()
            .filter(|i| !is_round_off_diagnostic(&i.name))
            .map(|i| (i.deviation, i.name.as_str()))
            .collect::<Vec<_>>();
        movers.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!(
            "{name}: largest movers {:?}",
            &movers[..6.min(movers.len())]
        );
        let e1c = (production.dataset.one_center_energy - seam.dataset.one_center_energy).abs();
        println!(
            "{name}: worst moved item {worst:?} ({:.0}x the LDA gate); E1c moved by {e1c:.3e} Ha",
            worst.0 / PRODUCTION_LDA[0].1
        );
        assert!(worst.0 / PRODUCTION_LDA[0].1 > 100.0, "{name}: {worst:?}");
        assert!(worst.0 < 1.0, "{name}: {worst:?}");
        assert!(
            e1c > e1c_bounds.0 && e1c < e1c_bounds.1,
            "{name}: E1c moved by {e1c:e}"
        );
        // The legacy seam is still the freeze: the port is proved by `parity`, and the tail of
        // v_local is the freeze's to round-off in the sphere.
        let still = compare_dataset(&f, &data, &seam, Scheme::Legacy);
        assert!(
            still.worst().ratio() <= 1.0,
            "{name}: {}",
            still.worst().name
        );
    }
}

#[test]
fn a_generated_dataset_survives_the_record_round_trip() {
    let f = Frozen::open("lda-sr_Na");
    let recipe = recipe_of(&f);
    let ctx = context();
    let generated = generate(&ctx, &recipe).unwrap();
    let dir = std::env::temp_dir().join(format!("mandacaru_paw_generate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("Na.parquet");
    let spec = mandacaru_core::paw::record::WriteSpec {
        stride: 4,
        codec: mandacaru_core::parquet::Codec::Zstd,
    };
    generated.dataset.write(&ctx, &path, &spec).unwrap();
    let back = PawDataset::read(&ctx, &path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(back.symbol, "Na");
    assert_eq!(back.family, generated.dataset.family);
    assert_eq!(back.points() * 4, generated.dataset.points());
    // The tables are the generated ones at every fourth point, to ten digits.
    for (name, values) in generated.dataset.tables() {
        let (_, stored) = back
            .tables()
            .into_iter()
            .find(|(n, _)| *n == name)
            .expect(&name);
        let scale = values.iter().fold(0., |m: f64, x| m.max(x.abs()));
        for (k, s) in stored.iter().enumerate() {
            assert!(
                (s - values[4 * k + 3]).abs() <= 1e-9 * scale + 1e-300,
                "{name}[{k}]"
            );
        }
    }
    assert_eq!(back.one_center_energy, generated.dataset.one_center_energy);
    assert_eq!(back.channels.len(), generated.dataset.channels.len());
}

#[test]
fn the_generator_draws_its_memory_from_the_context() {
    let f = Frozen::open("lda-sr_O");
    let recipe = recipe_of(&f);
    let measuring = context();
    let generated = generate(&measuring, &recipe).unwrap();
    let peak = measuring.memory_peak();
    drop(generated);
    assert_eq!(measuring.memory_used(), 0);
    println!("oxygen: context peak {peak} B for {} points", 12000);
    // A context that holds the peak generates; one a fifth of it is refused by name before
    // anything large is allocated.
    let exact = Context::new(Some(1), peak + (peak >> 4)).unwrap();
    generate(&exact, &recipe).unwrap();
    let starved = Context::new(Some(1), peak / 5).unwrap();
    match generate(&starved, &recipe) {
        Err(Error::Budget { .. }) => {}
        other => panic!("expected a budget refusal, found {:?}", other.map(|_| ())),
    }
    assert_eq!(starved.memory_used(), 0);
}

/// What the generator cannot honor raises, naming the reason, and gives its memory back: no
/// other construction is substituted.
#[test]
fn an_impossible_recipe_is_refused_by_name() {
    use mandacaru_core::models::relativity::Treatment;
    use mandacaru_core::paw::generate::recipe::{Cutoffs, Recipe};
    let ctx = context();
    let run = |recipe: &Recipe| -> String {
        match generate(&ctx, recipe) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("the recipe was not refused"),
        }
    };
    let oxygen = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    let mut recipe = Recipe::for_dataset("B", "lda", Treatment::Scalar).unwrap();
    recipe.rc_factor = 1e4;
    let message = run(&recipe);
    assert!(
        message.contains("exceeds") && message.contains("radial grid"),
        "{message}"
    );
    let mut recipe = oxygen.clone();
    recipe.cutoffs = Cutoffs::PerChannel([(0, 1.45)].into());
    assert!(run(&recipe).contains("names no radius for the channel l=1"));
    let mut recipe = oxygen.clone();
    recipe.second_reference.insert(1, -100.);
    assert!(run(&recipe).contains("is not above the bound state"));
    let mut recipe = oxygen.clone();
    recipe.frozen_subshells = vec![(3, 1)];
    assert!(run(&recipe).contains("not an occupied valence subshell"));
    let mut recipe = oxygen.clone();
    recipe.frozen_subshells = vec![(2, 0), (2, 1)];
    assert!(run(&recipe).contains("at least one occupied valence subshell must remain"));
    let mut recipe = oxygen.clone();
    recipe.semicore_subshells = vec![(2, 1)];
    assert!(run(&recipe).contains("not an occupied core subshell"));
    let mut recipe = Recipe::for_dataset("Na", "lda", Treatment::Scalar).unwrap();
    recipe.semicore_subshells = vec![(1, 0)];
    assert!(run(&recipe).contains("not directly below"));
    // An atom solved with another functional is not the reference of this recipe.
    let pbe = Recipe::for_dataset("O", "pbe", Treatment::Scalar).unwrap();
    let atom = solve_reference(&ctx, &pbe).unwrap();
    let message = generate_with_atom(&ctx, &oxygen, &atom, &Variational)
        .err()
        .expect("refused")
        .to_string();
    assert!(message.contains("was solved with xc"), "{message}");
    drop(atom);
    let nonrelativistic = Recipe::for_dataset("O", "lda", Treatment::None).unwrap();
    let atom = solve_reference(&ctx, &nonrelativistic).unwrap();
    let message = generate_with_atom(&ctx, &oxygen, &atom, &Variational)
        .err()
        .expect("refused")
        .to_string();
    assert!(message.contains("relativity"), "{message}");
    drop(atom);
    assert_eq!(ctx.memory_used(), 0, "a refusal leaves nothing reserved");
}

/// A fallback cutoff beyond the usual range is generated and noted, not hidden.
#[test]
fn a_far_fallback_cutoff_is_noted() {
    use mandacaru_core::models::relativity::Treatment;
    use mandacaru_core::paw::generate::recipe::Recipe;
    let ctx = context();
    let mut recipe = Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap();
    let ordinary = generate(&ctx, &recipe).unwrap();
    assert!(ordinary.notes.is_empty(), "{:?}", ordinary.notes);
    recipe.rc_factor = 1.6;
    let generated = generate(&ctx, &recipe).unwrap();
    assert!(
        generated
            .notes
            .iter()
            .any(|n| n.contains("fallback cutoff")),
        "{:?}",
        generated.notes
    );
}

/// The explicit-radius core correction has no frozen end-to-end case (the shipped library never
/// used it), so this checks what can be checked without one: the correction is applied at the
/// requested radius, its record is consistent with the density it made, only the quantities that
/// read the core density move, and every table that does not (waves, projectors, couplings) is
/// the default construction's, to round-off.
#[test]
fn an_explicit_core_radius_changes_only_what_reads_the_core() {
    use mandacaru_core::models::relativity::Treatment;
    use mandacaru_core::paw::generate::recipe::{Nlcc, Recipe};
    let ctx = context();
    let base = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    let atom = solve_reference_with(&ctx, &base, &AtomSolver).unwrap();
    let default = generate_with_atom(&ctx, &base, &atom, &AtomSolver)
        .unwrap()
        .dataset;
    let mut recipe = base.clone();
    recipe.nlcc = Nlcc::Radius(0.9);
    let explicit = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver)
        .unwrap()
        .dataset;
    assert!(explicit.nlcc.applied);
    // As the Python generator records it, the explicit radius is used and stored as given.
    assert_eq!(explicit.nlcc.r_nlcc, Some(0.9), "{:?}", explicit.nlcc);
    // The record's electron counts are those of the frozen core and of the density used.
    let core_electrons = explicit.nlcc.core_electrons.unwrap();
    assert!(
        (core_electrons - 2.).abs() < 1e-9,
        "the oxygen 1s core holds two electrons: {core_electrons}"
    );
    let partial = explicit.nlcc.partial_core_electrons.unwrap();
    assert!(partial > 0. && partial < core_electrons, "{partial}");
    let (a, b) = (arrays(&default), arrays(&explicit));
    for (name, values) in &a {
        let deviation = values
            .iter()
            .zip(&b[name])
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max);
        let scale = values
            .iter()
            .fold(0., |m: f64, x| m.max(x.abs()))
            .max(1e-300);
        let reads_the_core =
            name == "v_local" || name == "smooth_core_density" || name.starts_with("coupling_l");
        if reads_the_core {
            continue;
        }
        assert!(
            deviation / scale < 1e-12,
            "{name} moved by {:.2e}",
            deviation / scale
        );
    }
    let moved = a["v_local"]
        .iter()
        .zip(&b["v_local"])
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max);
    assert!(
        moved > 1e-3,
        "the core correction did not reach the local potential: {moved:e}"
    );
}
