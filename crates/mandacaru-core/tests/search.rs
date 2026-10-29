use mandacaru_core::{
    math::random::Rng,
    quantum::{
        search::{
            driver::Options,
            kernel::{self, Action, Kernel, Move},
        },
        CachedProblem, Encoding, Operator, C64,
    },
    Context,
};
use std::collections::BTreeMap;

fn architectures(pool: usize, maximum: usize) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    let mut last = vec![vec![]];
    for _ in 0..maximum {
        let next: Vec<_> = last
            .iter()
            .flat_map(|source| {
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

#[test]
fn proposal_mass_and_metropolis_balance_include_every_duplicate_edit() {
    let context = Context::new(Some(1), 16 << 20).unwrap();
    let kernel = Kernel::new(3, 3, [1.; 4]).unwrap();
    let states = architectures(3, 3);
    let probabilities =
        |state: &[usize]| kernel::softmax(&[1. + state.len() as f64, 0.2, 0.5], 0.4).unwrap();
    let cost =
        |state: &[usize]| 0.1 * state.len() as f64 + 0.03 * state.iter().sum::<usize>() as f64;
    for source in &states {
        let p = probabilities(source);
        let actions = kernel.actions(&context, source, &p).unwrap();
        assert!((actions.items.iter().map(|(_, p)| p).sum::<f64>() - 1.).abs() < 1e-12);
        let mut exhaustive = BTreeMap::<Vec<usize>, f64>::new();
        for &(action, probability) in &actions.items {
            *exhaustive.entry(action.apply(source)).or_default() += probability;
        }
        for target in &states {
            let forward = kernel.log_q(&context, target, source, &p).unwrap();
            assert!((forward.exp() - exhaustive.get(target).copied().unwrap_or(0.)).abs() < 1e-13);
            let reverse = kernel
                .log_q(&context, source, target, &probabilities(target))
                .unwrap();
            if forward.is_finite() && reverse.is_finite() {
                let temperature = 0.2;
                let lhs = (-cost(source) / temperature).exp()
                    * forward.exp()
                    * kernel::log_acceptance(
                        temperature,
                        cost(source),
                        cost(target),
                        forward,
                        reverse,
                    )
                    .min(0.)
                    .exp();
                let rhs = (-cost(target) / temperature).exp()
                    * reverse.exp()
                    * kernel::log_acceptance(
                        temperature,
                        cost(target),
                        cost(source),
                        reverse,
                        forward,
                    )
                    .min(0.)
                    .exp();
                assert!((lhs - rhs).abs() < 1e-13);
            }
        }
    }
    let q = kernel
        .log_q(&context, &[0, 0], &[0], &[0.2, 0.3, 0.5])
        .unwrap()
        .exp();
    assert!((q - 0.2 / 3.).abs() < 1e-14);
    let action = Action {
        kind: Move::Insert,
        position: 1,
        operator: 2,
        other: 0,
    };
    assert_eq!(action.transfer(&[0.1, 0.2], false), vec![0.1, 0., 0.2]);
    assert_eq!(kernel::log_acceptance(0., 1., 0., -1., -2.), f64::INFINITY);
    assert_eq!(
        kernel::log_acceptance(0., 0., 1., -1., -2.),
        f64::NEG_INFINITY
    );
    assert!(kernel::softmax(&[1., 0.], 1e-10).is_err());
    let mut random = Rng(17);
    for _ in 0..100 {
        assert!(random.index(7) < 7);
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn native_search_reaches_the_independent_ground_state_and_screening_preserves_chain_draws() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let operator = Operator::new(
            &context,
            4,
            &[
                ("ZIII".into(), C64::new(0.3, 0.)),
                ("XXII".into(), C64::new(0.2, 0.)),
                ("YYII".into(), C64::new(0.2, 0.)),
            ],
        )
        .unwrap();
        let problem = CachedProblem::new(
            &context,
            Encoding::new("jordan_wigner", 4, [1, 0]).unwrap(),
            operator,
            true,
            false,
            true,
        )
        .unwrap();
        let mut options = Options {
            max_steps: 40,
            max_length: 4,
            warm_start: false,
            seed: 7,
            ..Default::default()
        };
        let first = problem.search(&options, &[], &[]).unwrap();
        assert!((first.solution.energy + 0.5).abs() < 1e-9);
        assert!(!first.solution.success);
        assert!(first
            .solution
            .message
            .contains("no eigenstate convergence certificate"));
        let action = problem
            .hamiltonian
            .apply(&context, &problem.basis, first.solution.state.as_slice())
            .unwrap();
        let expected = mandacaru_core::quantum::evolution::inner(
            first.solution.state.as_slice(),
            action.as_slice(),
        )
        .re;
        assert!((expected - first.solution.energy).abs() < 1e-12);
        options.record = true;
        options.screen_insertions = 1;
        let recorded = problem.search(&options, &[], &[]).unwrap();
        assert_eq!(first.steps, recorded.steps);
        assert!(recorded.records.iter().any(|r| r["screened"] == true));
        assert_eq!(
            recorded
                .acceptance
                .iter()
                .map(|(_, tried)| tried)
                .sum::<usize>(),
            40
        );
        options.length_penalty = 1.;
        let penalized = problem.search(&options, &[], &[]).unwrap();
        assert!(penalized.solution.indices.is_empty());
        assert!((penalized.optimal_cost - penalized.solution.energy).abs() < 1e-12);
        options.transfer_steps = Some(3);
        options.length_penalty = 0.;
        let carried = problem
            .search(
                &options,
                &first.solution.indices,
                &first.solution.parameters,
            )
            .unwrap();
        assert!(carried.solution.seed.as_ref().unwrap().kept);
        assert_eq!(carried.steps.len(), 3);
        options.length_penalty = 1.;
        let refused = problem
            .search(
                &options,
                &first.solution.indices,
                &first.solution.parameters,
            )
            .unwrap();
        assert!(!refused.solution.seed.as_ref().unwrap().kept);
        assert_eq!(refused.steps.len(), 40);
    }
    assert_eq!(context.memory_used(), 0);
}
