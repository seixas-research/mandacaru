use mandacaru_core::{
    math::autodiff::Tape,
    quantum::{
        search::learning::{
            graph::{BipartiteGraph, Candidates, Matrix},
            network::{self, Parameters},
        },
        C64,
    },
    Context,
};

fn graph(context: &Context) -> BipartiteGraph {
    BipartiteGraph::new(
        context,
        4,
        0b1001,
        &[
            ("ZIII".into(), C64::new(-0.3, 0.)),
            ("XXII".into(), C64::new(0.2, 0.)),
            ("IYYI".into(), C64::new(0.1, 0.)),
            ("IIIZ".into(), C64::new(0.4, 0.)),
        ],
        &[
            vec![
                ("XIXI".into(), C64::new(0., 1.)),
                ("YIYI".into(), C64::new(0., 0.5)),
            ],
            vec![
                ("IYIZ".into(), C64::new(0., 1.)),
                ("ZIXI".into(), C64::new(0., -0.2)),
            ],
        ],
    )
    .unwrap()
}

#[test]
fn graph_descriptors_match_reference_occupations_and_weighted_pauli_supports() {
    let context = Context::new(Some(1), 16 << 20).unwrap();
    {
        let graph = graph(&context);
        assert!((graph.reference_energy + 0.1).abs() < 1e-14);
        assert!((graph.descriptors[(0, 0)] - 0.5).abs() < 1e-14);
        assert!((graph.descriptors[(0, 1)] - 2. / 3.).abs() < 1e-14);
        assert!((graph.descriptors[(0, 2)] - 1. / 3.).abs() < 1e-14);
        for operator in 0..2 {
            let weight = graph
                .pool
                .iter()
                .map(|edges| edges.row(operator).iter().sum::<f64>())
                .sum::<f64>();
            assert!((weight - 1.).abs() < 1e-14);
        }
        let rows =
            Candidates::new(&context, &graph, &[1, 0, 0], &[0, 1], &[0.3, 0.7], 0.1).unwrap();
        assert_eq!(rows.occupancy.shape(), (1, 2));
        assert!((rows.occupancy[(0, 0)] - 2. / 3.).abs() < 1e-14);
        assert!((rows.positional[(0, 0)] - 5. / 6.).abs() < 1e-14);
        assert!((rows.descriptors[(0, 4)] - 3. / 7.).abs() < 1e-14);
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn message_passing_parameter_gradients_and_serialization_match_independent_displacements() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let graph = graph(&context);
        let rows =
            Candidates::new(&context, &graph, &[1, 0, 0], &[0, 1], &[0.3, 0.7], 0.1).unwrap();
        let mut parameters = Parameters::new(&context, 17).unwrap();
        let mut tape = Tape::new(&context);
        let variables = parameters.variables(&mut tape).unwrap();
        let scores = network::scores(
            &mut tape,
            &variables,
            &graph,
            &rows,
            &[0.; 9],
            &[1.; 9],
            None,
            None,
            mandacaru_core::quantum::search::learning::data::EditKind::Insert,
        )
        .unwrap();
        let seed = Matrix::from_column_slice(2, 1, &[0.2, -0.3]);
        tape.backward(scores, &seed).unwrap();
        let loss = |parameters: &Parameters| {
            let mut tape = Tape::new(&context);
            let variables = parameters.variables(&mut tape).unwrap();
            let scores = network::scores(
                &mut tape,
                &variables,
                &graph,
                &rows,
                &[0.; 9],
                &[1.; 9],
                None,
                None,
                mandacaru_core::quantum::search::learning::data::EditKind::Insert,
            )
            .unwrap();
            tape.value(scores)
                .iter()
                .zip(seed.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        for name in [
            "qubit_in",
            "term_in",
            "to_term.0.1",
            "to_qubit.1.2",
            "term_update.1",
            "qubit_update_bias.0",
            "readout.0",
            "w1",
            "b1",
            "w2",
            "w_desc",
        ] {
            let length = parameters.values[name].len();
            for index in [0, length / 2, length - 1] {
                let original = parameters.values[name][index];
                parameters.values.get_mut(name).unwrap()[index] = original + 1e-6;
                let plus = loss(&parameters);
                parameters.values.get_mut(name).unwrap()[index] = original - 1e-6;
                let minus = loss(&parameters);
                parameters.values.get_mut(name).unwrap()[index] = original;
                let derivative = tape.gradient(variables[name])[index];
                assert!(
                    (derivative - (plus - minus) / 2e-6).abs() < 3e-7,
                    "{name}[{index}]"
                );
            }
        }
        let restored = Parameters::from_json(&context, &parameters.as_json()).unwrap();
        assert!((loss(&restored) - loss(&parameters)).abs() < 1e-14);
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn bipartite_graph_is_invariant_under_mo_sign_gauge() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    let base = graph(&context);
    let flipped = BipartiteGraph::new(
        &context,
        4,
        0b1001,
        &[
            ("ZIII".into(), C64::new(-0.3, 0.)),
            ("XXII".into(), C64::new(-0.2, 0.)),
            ("IYYI".into(), C64::new(-0.1, 0.)),
            ("IIIZ".into(), C64::new(0.4, 0.)),
        ],
        &[
            vec![
                ("XIXI".into(), C64::new(0., -1.)),
                ("YIYI".into(), C64::new(0., -0.5)),
            ],
            vec![
                ("IYIZ".into(), C64::new(0., -1.)),
                ("ZIXI".into(), C64::new(0., 0.2)),
            ],
        ],
    )
    .unwrap();
    assert!((&base.term - &flipped.term).amax() < 1e-14);
    assert!((&base.descriptors - &flipped.descriptors).amax() < 1e-14);
    assert!((&base.overlap - &flipped.overlap).amax() < 1e-14);
    for s in 0..3 {
        assert!((&base.weighted[s] - &flipped.weighted[s]).amax() < 1e-14);
        assert!((&base.pool[s] - &flipped.pool[s]).amax() < 1e-14);
    }
    let rows_a = Candidates::new(&context, &base, &[0, 1], &[0, 1], &[0.3, 0.2], 0.05).unwrap();
    let rows_b = Candidates::new(&context, &flipped, &[0, 1], &[0, 1], &[0.3, 0.2], 0.05).unwrap();
    assert!((&rows_a.descriptors - &rows_b.descriptors).amax() < 1e-14);
    assert_eq!(rows_a.sequence_overlap, rows_b.sequence_overlap);
}

#[test]
fn bipartite_graph_overlap_and_sequence_overlap_capture_anticommuting_support() {
    let context = Context::new(Some(1), 64 << 20).unwrap();
    let bg = BipartiteGraph::new(
        &context,
        4,
        0b1001,
        &[("ZIII".into(), C64::new(-0.3, 0.))],
        &[
            vec![("XIII".into(), C64::new(0., 1.))],
            vec![("IYII".into(), C64::new(0., 1.))],
            vec![("YIII".into(), C64::new(0., 1.))],
        ],
    )
    .unwrap();
    assert!((bg.overlap[(0, 0)] - 0.).abs() < 1e-14);
    assert!((bg.overlap[(0, 2)] - 1.).abs() < 1e-14);
    assert!((bg.overlap[(2, 0)] - 1.).abs() < 1e-14);
    assert!((bg.overlap[(1, 2)] - 0.).abs() < 1e-14);

    let early = Candidates::new(&context, &bg, &[0, 1], &[2], &[0., 0., 0.1], 0.01).unwrap();
    let late = Candidates::new(&context, &bg, &[1, 0], &[2], &[0., 0., 0.1], 0.01).unwrap();
    assert!((early.sequence_overlap[0] - 1. / 3.).abs() < 1e-14);
    assert!((late.sequence_overlap[0] - 2. / 3.).abs() < 1e-14);
}

#[test]
fn extended_gnn_bloch_curvature_and_delete_heads_match_finite_differences_and_roundtrip() {
    use mandacaru_core::quantum::{search::learning::data::EditKind, CrystalMomentum};

    let context = Context::new(Some(1), 64 << 20).unwrap();
    {
        let momentum = CrystalMomentum::new([2, 1, 1], vec![[0, 0, 0], [1, 0, 0]]).unwrap();
        let bg = graph(&context).with_momentum(&momentum);
        assert!((bg.qubit[(0, 4)] - 0.).abs() < 1e-14);
        assert!((bg.qubit[(1, 4)] - 0.5).abs() < 1e-14);
        assert!((bg.qubit[(2, 4)] - 0.).abs() < 1e-14);
        assert!((bg.qubit[(3, 4)] - 0.5).abs() < 1e-14);
        assert!((bg.qubit[(0, 7)] - 1.).abs() < 1e-14);

        let rows = Candidates::new(&context, &bg, &[1, 0], &[0, 1], &[0.3, 0.7], 0.1).unwrap();
        let curvatures = vec![-0.5, 0.2];
        let positions = vec![0usize, 1];
        let mut parameters = Parameters::new(&context, 23).unwrap();
        // Seed non-zero weights on all move/feature readout heads so autodiff gradients exercise every column.
        for name in [
            "w_desc",
            "w_delete_head",
            "w_delete_desc",
            "w_replace_head",
            "w_replace_desc",
        ] {
            for (idx, val) in parameters
                .values
                .get_mut(name)
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *val = 0.07 * ((idx % 5) as f64 - 2.0);
            }
        }
        let seed = Matrix::from_column_slice(2, 1, &[0.4, -0.5]);
        for (move_kind, names) in [
            (EditKind::Insert, vec!["qubit_in", "w1", "w2", "w_desc"]),
            (
                EditKind::Delete,
                vec!["qubit_in", "w1", "w_delete_head", "w_delete_desc"],
            ),
            (
                EditKind::Replace,
                vec!["qubit_in", "w1", "w_replace_head", "w_replace_desc"],
            ),
        ] {
            let mut tape = Tape::new(&context);
            let variables = parameters.variables(&mut tape).unwrap();
            let scores = network::scores(
                &mut tape,
                &variables,
                &bg,
                &rows,
                &[0.; 9],
                &[1.; 9],
                Some(&curvatures),
                Some(&positions),
                move_kind,
            )
            .unwrap();
            tape.backward(scores, &seed).unwrap();
            let loss = |params: &Parameters| {
                let mut t = Tape::new(&context);
                let vars = params.variables(&mut t).unwrap();
                let s = network::scores(
                    &mut t,
                    &vars,
                    &bg,
                    &rows,
                    &[0.; 9],
                    &[1.; 9],
                    Some(&curvatures),
                    Some(&positions),
                    move_kind,
                )
                .unwrap();
                t.value(s)
                    .iter()
                    .zip(seed.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            };
            for name in names {
                let len = parameters.values[name].len();
                for idx in [0, len / 2, len - 1] {
                    let orig = parameters.values[name][idx];
                    parameters.values.get_mut(name).unwrap()[idx] = orig + 1e-6;
                    let plus = loss(&parameters);
                    parameters.values.get_mut(name).unwrap()[idx] = orig - 1e-6;
                    let minus = loss(&parameters);
                    parameters.values.get_mut(name).unwrap()[idx] = orig;
                    let grad = tape.gradient(variables[name])[idx];
                    assert!(
                        (grad - (plus - minus) / 2e-6).abs() < 3e-7,
                        "{name}[{idx}]: analytic={grad} fd={}",
                        (plus - minus) / 2e-6
                    );
                }
            }
            let restored = Parameters::from_json(&context, &parameters.as_json()).unwrap();
            assert!((loss(&restored) - loss(&parameters)).abs() < 1e-14);
        }
    }
    assert_eq!(context.memory_used(), 0);
}
