//! The PAW generator (`paw::generate`): the dataset record survives a write and a read, the
//! generator draws its memory from the context, and what it cannot honor is refused by name.
//! The explicit-radius core correction moves only what reads the core.

mod paw_generate_support;

use mandacaru_core::models::xc_scheme::{AtomSolver, Variational};
use mandacaru_core::paw::generate::reference::{solve_reference, solve_reference_with};
use mandacaru_core::paw::generate::{generate, generate_with_atom};
use mandacaru_core::paw::record::PawDataset;
use mandacaru_core::{Context, Error};
use paw_generate_support::arrays;

fn context() -> Context {
    Context::new(Some(1), 1 << 30).unwrap()
}

#[test]
fn a_generated_dataset_survives_the_record_round_trip() {
    use mandacaru_core::models::relativity::Treatment;
    use mandacaru_core::paw::generate::recipe::Recipe;
    let recipe = Recipe::for_dataset("Na", "lda", Treatment::Scalar).unwrap();
    let ctx = context();
    let generated = generate(&ctx, &recipe).unwrap();
    let dir = std::env::temp_dir().join(format!("mandacaru_paw_generate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("Na.parquet");
    let spec = mandacaru_core::paw::record::WriteSpec {
        stride: 4,
        codec: mandacaru_core::parquet::Codec::Zstd,
    };
    // The record carries the core it summed (format version 3): the same subshells, whose
    // `sum occupation u^2 / (4 pi r^2)` is the core density to roundoff.
    let states = generated
        .dataset
        .core_states
        .as_ref()
        .expect("a generated record has its core");
    assert!(!states.is_empty(), "Na has a core");
    let r = generated.dataset.r.as_slice();
    let core = generated.dataset.core_density.as_slice();
    let peak = core.iter().fold(0f64, |m, x| m.max(x.abs()));
    for (i, (&ri, &want)) in r.iter().zip(core).enumerate() {
        let sum: f64 = states
            .iter()
            .map(|s| {
                s.occupation * s.orbital.as_slice()[i].powi(2)
                    / (4. * std::f64::consts::PI * ri * ri)
            })
            .sum();
        assert!((sum - want).abs() <= 1e-13 * peak, "core density at r[{i}]");
    }
    generated.dataset.write(&ctx, &path, &spec).unwrap();
    let back = PawDataset::read(&ctx, &path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        back.format_version(),
        mandacaru_core::paw::record::FORMAT_VERSION
    );
    assert_eq!(back.core_states.as_ref().map(Vec::len), Some(states.len()));
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
    use mandacaru_core::models::relativity::Treatment;
    use mandacaru_core::paw::generate::recipe::Recipe;
    let recipe = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
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
    // Silicon tabulates its radii, so the table is emptied: every channel then takes the fallback,
    // `rc_factor` times the outermost peak of its first reference.
    recipe.cutoff_table.clear();
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

/// The PBE reference atom of every lanthanide, scalar-relativistic and Dirac, on the production
/// path (`solve_reference`: the relaxed configuration on the generation grid, then the reference
/// SCF at the generator's tolerance and mixing). Before the gradient smoothing of
/// `spherical::exchange_correlation` the scalar Ce, Pr, Nd, Yb and the Dirac Gd atoms raised as
/// unconverged after 11-28 minutes each and the scalar Gd was refused after 600 iterations.
#[test]
#[ignore = "slow: 2980 s (measured 2026-11-09); run: cargo test --profile release-test -p mandacaru-core --test paw_generate -- --ignored every_pbe_lanthanide_reference_atom_converges --nocapture"]
fn every_pbe_lanthanide_reference_atom_converges() {
    use mandacaru_core::models::{configuration::valence_subshells, relativity::Treatment};
    use mandacaru_core::paw::generate::recipe::Recipe;
    let ctx = context();
    let mut failed = Vec::new();
    for z in 57u8..=71 {
        let symbol = mandacaru_core::models::elements::SYMBOLS[z as usize - 1];
        for treatment in [Treatment::Scalar, Treatment::Dirac] {
            let recipe = Recipe::for_dataset(symbol, "pbe", treatment).unwrap();
            let started = std::time::Instant::now();
            let result = solve_reference(&ctx, &recipe);
            let seconds = started.elapsed().as_secs_f64();
            match result {
                Ok(atom) => {
                    let valence = valence_subshells(&atom.occupations);
                    let levels = atom
                        .occupations
                        .iter()
                        .zip(&atom.eigenvalues)
                        .filter(|(s, _)| valence.contains(&(s.0, s.1)))
                        .map(|(s, e)| format!("{}{}:{e:.5}", s.0, "spdf".as_bytes()[s.1] as char))
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!(
                        "{symbol} {}: converged {} in {} it, {seconds:.0} s, E = {:.6} Ha; {levels}",
                        treatment.name(),
                        atom.converged,
                        atom.iterations,
                        atom.energy
                    );
                    if !atom.converged {
                        failed.push(format!("{symbol} {}", treatment.name()));
                    }
                }
                Err(e) => {
                    println!("{symbol} {}: {e} ({seconds:.0} s)", treatment.name());
                    failed.push(format!("{symbol} {}", treatment.name()));
                }
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}
