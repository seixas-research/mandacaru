//! The proposed state of a search step is evaluated lazily, and the chain is the one it was.
//!
//! A proposal is relaxed first. Its reverse probability reads one half of its distribution (an
//! insertion is undone by a deletion, which reads the removal weights; a deletion by an insertion,
//! which reads the operator probabilities; a replacement by a replacement, which reads both), so
//! only that half is computed, and the other half only if the proposal is accepted and becomes the
//! current state. The rows keep every field, so the frozen chain of
//! `test/migration/test_native_search_chain.py` is the pin of the whole chain; this file proves the
//! two pieces that pin cannot see: the matched-only `log_q` is bit-equal to the full one, and the
//! lazily completed states and the work counts are what the staging says.
use mandacaru_core::{
    math::random::Rng,
    quantum::{
        search::{
            driver::Options,
            kernel::{self, Kernel},
        },
        evolution::Work,
        CachedProblem, Encoding, Operator, C64,
    },
    Context,
};
use serde_json::Value;
use std::collections::BTreeMap;

fn architectures(pool: usize, maximum: usize) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    let mut last = vec![vec![]];
    for _ in 0..maximum {
        let next: Vec<_> = last
            .iter()
            .flat_map(|source: &Vec<usize>| {
                (0..pool).map(move |op| {
                    let mut next = source.clone();
                    next.push(op);
                    next
                })
            })
            .collect();
        all.extend(next.clone());
        last = next;
    }
    all
}

/// Distinct, positive, normalized weights that depend on the state (as the driver's do).
fn probabilities(state: &[usize], pool: usize) -> Vec<f64> {
    let gradients: Vec<f64> = (0..pool)
        .map(|i| 0.3 + 0.7 * ((i + 2 * state.len() + state.iter().sum::<usize>()) % 5) as f64)
        .collect();
    kernel::softmax(&gradients, 0.4).unwrap()
}

fn removal(state: &[usize]) -> Vec<f64> {
    let importances: Vec<f64> = state
        .iter()
        .enumerate()
        .map(|(i, &op)| 1. / (1. + i as f64) + 0.1 * op as f64)
        .collect();
    kernel::removal(&importances, 0.4).unwrap()
}

#[test]
fn the_matched_only_reverse_probability_is_bit_equal_to_the_full_one() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    // Every move kind, the empty sequence and the length bound, adjacent duplicates (an insertion
    // next to an equal operator matches at two positions), a one-operator pool (no replacement),
    // and a zero replacement weight.
    let cases: [(usize, usize, [f64; 3]); 5] = [
        (3, 3, [1.; 3]),
        (3, 3, [1., 2., 3.]),
        (1, 3, [1., 1., 1.]),
        (2, 4, [3., 1., 0.]),
        (4, 2, [0.5, 2., 1.]),
    ];
    for (pool, maximum, weights) in cases {
        let kernel = Kernel::new(pool, maximum, weights).unwrap();
        let states = architectures(pool, maximum);
        let (mut matched, mut duplicates) = (0, 0);
        for source in &states {
            let (p, w) = (probabilities(source, pool), removal(source));
            for target in &states {
                let full = kernel.log_q(&context, target, source, &p, &w).unwrap();
                // Whatever the pair, the one kind connecting it reads no more than its own half.
                let by_halves = kernel
                    .log_q_matched(target, source, Some(&p), Some(&w))
                    .unwrap();
                assert_eq!(by_halves.to_bits(), full.to_bits(), "{source:?} -> {target:?}");
                let own = match (target.len() as isize - source.len() as isize).abs() {
                    1 if target.len() > source.len() => Some((Some(&p[..]), None)),
                    1 => Some((None, Some(&w[..]))),
                    0 => Some((Some(&p[..]), Some(&w[..]))),
                    _ => None,
                };
                if let Some((p_only, w_only)) = own {
                    let own = kernel.log_q_matched(target, source, p_only, w_only).unwrap();
                    assert_eq!(own.to_bits(), full.to_bits(), "{source:?} -> {target:?}");
                }
                if full.is_finite() {
                    matched += 1;
                    if target.len() == source.len() + 1
                        && (0..target.len() - 1).any(|i| target[i] == target[i + 1])
                    {
                        duplicates += 1;
                    }
                }
            }
        }
        assert!(matched > 0);
        if pool > 1 {
            assert!(duplicates > 0, "no adjacent duplicate was covered");
        }
    }
    // A half that the kind reads must be given.
    let kernel = Kernel::new(3, 3, [1.; 3]).unwrap();
    assert!(kernel.log_q_matched(&[0], &[], None, Some(&[])).is_err());
    assert!(kernel.log_q_matched(&[], &[0], Some(&probabilities(&[0], 3)), None).is_err());
    assert_eq!(context.memory_used(), 0);
}

fn problem(context: &Context) -> CachedProblem {
    let operator = Operator::new(
        context,
        4,
        &[
            ("ZIII".into(), C64::new(0.3, 0.)),
            ("XXII".into(), C64::new(0.2, 0.)),
            ("YYII".into(), C64::new(0.2, 0.)),
            ("IZZI".into(), C64::new(0.1, 0.)),
            ("IIZI".into(), C64::new(-0.15, 0.)),
        ],
    )
    .unwrap();
    CachedProblem::new(
        context,
        Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
        operator,
        true,
        false,
        true,
    )
    .unwrap()
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn indices(value: &Value) -> Vec<usize> {
    value.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
}

/// The work a chain with `warm_start = true` does, from its rows: the reference is screened once,
/// a deletion or replacement screens its proposal, an insertion only when accepted, and the
/// importances are spent on the proposals that are inserted, replaced or accepted.
#[test]
fn a_warm_chain_screens_and_weighs_only_what_a_decision_reads() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let problem = problem(&context);
        let options = Options {
            max_steps: 120,
            max_length: 4,
            seed: 7,
            record: true,
            screen_insertions: 1,
            ..Default::default()
        };
        let result = problem.search(&options, &[], &[]).unwrap();
        let (mut screens, mut weighed, mut naive_screens, mut naive_weighed) = (1usize, 0usize, 1usize, 0usize);
        // The importances of a state of `l` operators: one prefix and sweep per position.
        let (mut work, mut separate) = (Work::default(), Work::default());
        let mut weigh = |length: usize| {
            separate += Work::separate_evaluations(length);
            work += Work {
                rotations: if length == 0 { 0 } else { (length - 1) + length * (length - 1) / 2 },
                adjoint_steps: length * (length + 1) / 2,
                expectations: length,
            };
            length
        };
        let mut skipped_screens = 0;
        let mut skipped_weights = 0;
        for row in &result.steps {
            let accepted = row["accepted"].as_bool().unwrap();
            let length = row["proposed"].as_array().unwrap().len();
            naive_screens += 1;
            naive_weighed += length;
            match row["move"].as_str().unwrap() {
                "insert" => {
                    weighed += weigh(length);
                    if accepted {
                        screens += 1;
                    } else {
                        skipped_screens += 1;
                    }
                }
                "delete" => {
                    screens += 1;
                    if accepted {
                        weighed += weigh(length);
                    } else {
                        skipped_weights += length;
                    }
                }
                _ => {
                    screens += 1;
                    weighed += weigh(length);
                }
            }
        }
        // A screened insertion is only recorded: it reads the relaxation alone.
        let screened_rows: Vec<_> = result.records.iter().filter(|r| r["screened"] == true).collect();
        assert!(!screened_rows.is_empty());
        for row in &screened_rows {
            naive_screens += 1;
            naive_weighed += row["proposed"].as_array().unwrap().len();
            skipped_screens += 1;
            skipped_weights += row["proposed"].as_array().unwrap().len();
        }
        // The final gradient screens `best` once when it is a rejected insertion (or a screened one).
        assert!(
            (screens..=screens + 1).contains(&result.num_screenings),
            "screens {} not in {screens}..={}",
            result.num_screenings,
            screens + 1
        );
        assert_eq!(result.num_importance_evaluations, weighed);
        // The work is what the shared prefix and the truncated sweeps say, against one full
        // evaluation per position; the optimizer read a gradient for fewer points than it
        // evaluated (its rejected trials computed the energy alone).
        assert_eq!(result.importance_work, work);
        assert!(work.rotations < separate.rotations && work.adjoint_steps < separate.adjoint_steps);
        assert_eq!(work.expectations, separate.expectations);
        assert!(result.num_gradient_evaluations > 0);
        assert!(result.num_gradient_evaluations <= result.solution.history.len());
        assert!(result.num_screenings < naive_screens);
        assert!(result.num_importance_evaluations < naive_weighed);
        assert!(skipped_screens > 0 && skipped_weights > 0);
        println!(
            "steps {}: screens {} of {}, importance evaluations {} of {} (skipped screens {skipped_screens}, importances {skipped_weights})",
            result.steps.len(),
            result.num_screenings,
            naive_screens,
            result.num_importance_evaluations,
            naive_weighed
        );
    }
    assert_eq!(context.memory_used(), 0);
}

/// An SLSQP relaxation accepts one trial per iteration on its energy alone, so it asks for a
/// gradient at its start and at each accepted point and for no other: a rejected trial of its
/// line search computes the energy only. Every objective call is still one `history` entry.
#[test]
fn a_chain_computes_a_gradient_only_where_its_optimizer_reads_one() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let problem = problem(&context);
        let options = Options {
            max_steps: 120,
            max_length: 4,
            seed: 7,
            optimizer: mandacaru_core::quantum::optimizer::Options {
                method: mandacaru_core::quantum::optimizer::Method::Slsqp,
                tolerance: Some(1e-10),
                ..Default::default()
            },
            ..Default::default()
        };
        let result = problem.search(&options, &[], &[]).unwrap();
        let relaxed = result.steps.iter().filter(|row| !row["proposed"].as_array().unwrap().is_empty()).count();
        assert_eq!(result.num_gradient_evaluations, relaxed + result.solution.optimizer_steps);
        // This model is quadratic enough that SLSQP's first trial is always accepted: the count
        // can only meet the evaluations here (the strict inequality is `optimizers.rs`' Rosenbrock).
        let evaluations = result.solution.history.len();
        assert!(result.num_gradient_evaluations <= evaluations);
        println!("gradients {} of {evaluations} optimizer evaluations", result.num_gradient_evaluations);
    }
    assert_eq!(context.memory_used(), 0);
}

/// With `warm_start = false` a state's distribution is a function of the sequence. Every row that
/// sources it, whether its distribution was completed on an acceptance or at a revisit of a memoized
/// relaxation, carries the same bits as an independent evaluation of that sequence, and every
/// recorded reverse probability is the full `log_q` of the proposed state's recorded distribution.
#[test]
fn a_lazily_completed_state_equals_a_fully_evaluated_one() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let problem = problem(&context);
        for uniform in [false, true] {
            let options = Options {
                max_steps: 200,
                max_length: 3,
                seed: 5,
                warm_start: false,
                uniform,
                record: true,
                screen_insertions: 1,
                ..Default::default()
            };
            let result = problem.search(&options, &[], &[]).unwrap();
            let kernel = Kernel::new(problem.pool.len(), options.max_length, options.move_weights).unwrap();
            let mut first: BTreeMap<Vec<usize>, &Value> = BTreeMap::new();
            for row in &result.records {
                let source = indices(&row["source"]);
                let seen = first.entry(source.clone()).or_insert(row);
                for key in [
                    "source_parameters",
                    "source_gradients",
                    "source_probabilities",
                    "source_importances",
                    "source_removal",
                ] {
                    assert_eq!(row[key], seen[key], "{source:?}: {key}");
                }
                // The weights are those of the state's own relaxation, computed fully and apart.
                let parameters = floats(&row["source_parameters"]);
                for (i, importance) in floats(&row["source_importances"]).into_iter().enumerate() {
                    let mut zeroed = parameters.clone();
                    zeroed[i] = 0.;
                    let full = problem.evaluate(&source, &zeroed, true).unwrap().gradient[i].abs();
                    assert_eq!(importance.to_bits(), full.to_bits(), "{source:?}[{i}]");
                }
                assert_eq!(
                    floats(&row["source_removal"]),
                    kernel::removal(&floats(&row["source_importances"]), options.proposal_temperature).unwrap()
                );
                if !uniform {
                    assert_eq!(
                        floats(&row["source_probabilities"]),
                        kernel::softmax(&floats(&row["source_gradients"]), options.proposal_temperature).unwrap()
                    );
                }
            }
            assert!(first.len() > 3, "the chain sourced only {} states", first.len());
            let mut checked = 0;
            for row in result.records.iter().filter(|r| r["screened"] == false) {
                let (source, target) = (indices(&row["source"]), indices(&row["proposed"]));
                let Some(to) = first.get(&target) else { continue };
                let reverse = kernel
                    .log_q(
                        &context,
                        &source,
                        &target,
                        &floats(&to["source_probabilities"]),
                        &floats(&to["source_removal"]),
                    )
                    .unwrap();
                assert_eq!(
                    row["log_q_reverse"].as_f64().unwrap().to_bits(),
                    reverse.to_bits(),
                    "{source:?} -> {target:?}"
                );
                checked += 1;
            }
            assert!(checked > 20, "only {checked} rows had a sourced proposal");
        }
    }
    assert_eq!(context.memory_used(), 0);
}

/// The bound behind plan row A5's early rejection, checked on a replay of the chain's own stream:
/// with `ln q_reverse <= 0` the acceptance is at most `B = -dF/T - ln q_forward`, so `ln u >= B`
/// with `B < 0` is a rejection that needs no reverse probability. The driver does not take that
/// shortcut (the step rows carry the reverse probability and the pinned chain reads it); this test
/// proves the bound algebra on every row and reports how many rows it would have covered.
#[test]
fn the_early_rejection_bound_agrees_with_every_decision_it_covers() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let problem = problem(&context);
        for (seed, uniform) in [(7u64, false), (3, false), (11, true)] {
            let options = Options { max_steps: 300, max_length: 4, seed, uniform, ..Default::default() };
            let result = problem.search(&options, &[], &[]).unwrap();
            let mut random = Rng(seed);
            let (mut covered, mut rejected, mut tight) = (0, 0, 0);
            for row in &result.steps {
                let _draw = random.uniform();
                let (temperature, forward) = (row["temperature"].as_f64().unwrap(), row["log_q_forward"].as_f64().unwrap());
                let (current, proposed) = (row["source_cost"].as_f64().unwrap(), row["cost"].as_f64().unwrap());
                let ratio = row["log_acceptance"].as_f64().unwrap();
                let accepted = row["accepted"].as_bool().unwrap();
                let bound = kernel::log_acceptance(temperature, current, proposed, forward, 0.);
                assert!(ratio <= bound + 1e-12, "ratio {ratio} above its bound {bound}");
                let log_u = (ratio < 0.).then(|| random.uniform().ln());
                // The replay is aligned with the chain's stream: it reproduces every recorded decision.
                assert_eq!(accepted, ratio >= 0. || log_u.is_some_and(|u| u < ratio));
                if !accepted {
                    rejected += 1;
                }
                if bound < 0. {
                    // The decision draws exactly when the ratio is negative, and the bound makes it so.
                    let log_u = log_u.expect("a negative bound forces a negative ratio, so a draw");
                    if log_u >= bound {
                        covered += 1;
                        assert!(!accepted, "the bound rejects a proposal the chain accepted");
                    }
                    if log_u >= ratio {
                        tight += 1;
                    }
                }
            }
            println!("seed {seed} uniform {uniform}: {covered} of {} steps ({rejected} rejected) rejected by the bound alone", result.steps.len());
            assert!(covered <= tight);
        }
    }
    assert_eq!(context.memory_used(), 0);
}
