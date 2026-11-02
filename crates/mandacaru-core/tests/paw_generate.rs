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

use mandacaru_core::paw::generate::energies::{AtomSolver, Variational};
use mandacaru_core::paw::generate::reference::solve_reference;
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
    let atom = solve_reference(&ctx, &recipe).unwrap();
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

/// The production path is the variational scheme. For LDA it equals the freeze wherever the two
/// radial schemes agree (density at least 2e-12) and differs in the far tail by the documented
/// 1.9e-4 Ha of `v_local`.
#[test]
fn the_production_path_equals_the_freeze_for_lda_outside_the_far_tail() {
    for name in ["lda-sr_O", "lda-sr_Fe", "upaw_Al"] {
        let f = Frozen::open(name);
        let data = Decimated::open(name);
        let recipe = recipe_of(&f);
        let ctx = context();
        let generated = generate(&ctx, &recipe).unwrap();
        compare_dataset(&f, &data, &generated, Scheme::Production)
            .assert_within(&format!("{name} (production)"));
        // The variational potential and the freeze part ways below 2e-12 (printed, bounded).
        let legacy = generate_with_atom(
            &ctx,
            &recipe,
            &solve_reference(&ctx, &recipe).unwrap(),
            &AtomSolver,
        )
        .unwrap();
        let (a, b) = (arrays(&generated.dataset), arrays(&legacy.dataset));
        let tail = a["v_local"]
            .iter()
            .zip(&b["v_local"])
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max);
        println!("{name}: production against legacy v_local {tail:.3e}");
        // The upper bound only: whether this tail survives is the owner decision of XC-SW.
        assert!(tail < 2.5e-4, "{name}: tail {tail:e}");
    }
}

/// For PBE every table that does not depend on the radial scheme is the freeze's; `v_local` and
/// the one-center constant carry the scheme (D-2: the variational one is the source of truth)
/// and move by the amount measured in A2 (3.5e-5 Ha for oxygen), well inside the 4.9e-4 Ha scale
/// of the plan.
#[test]
fn the_production_path_moves_only_the_scheme_dependent_pbe_numbers() {
    for (name, e1c_bounds) in [("pbe-sr_O", (1e-5, 1e-3)), ("pbe-sr_Cu", (1e-5, 1e-3))] {
        let f = Frozen::open(name);
        let data = Decimated::open(name);
        let recipe = recipe_of(&f);
        let ctx = context();
        let atom = solve_reference(&ctx, &recipe).unwrap();
        let production = generate_with_atom(&ctx, &recipe, &atom, &Variational).unwrap();
        let comparison = compare_dataset(&f, &data, &production, Scheme::Legacy);
        let scheme_dependent = |n: &str| {
            n.contains("v_local") && !n.contains("screened")
                || n == "scalar.one_center_energy"
                || n.starts_with("energy.")
        };
        let moved = comparison
            .items
            .iter()
            .filter(|i| scheme_dependent(&i.name))
            .map(|i| (i.name.clone(), i.deviation))
            .collect::<Vec<_>>();
        println!("{name}: scheme-dependent deviations {moved:?}");
        let e1c = comparison.deviation("scalar.one_center_energy")
            * production.dataset.one_center_energy.abs().max(1.);
        assert!(
            e1c > e1c_bounds.0 && e1c < e1c_bounds.1,
            "{name}: E1c moved by {e1c:e}"
        );
        assert!(comparison.deviation("array.v_local") * 1e0 > 1e-6);
        comparison
            .without(scheme_dependent)
            .assert_within(&format!("{name} (production, scheme-independent)"));
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
    let atom = solve_reference(&ctx, &base).unwrap();
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
