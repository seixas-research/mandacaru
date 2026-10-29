use mandacaru_core::{
    math::autodiff::Tape,
    quantum::{
        search::learning::{
            graph::{Candidates, Graph, Matrix},
            network::{self, Parameters},
        },
        C64,
    },
    Context,
};

fn graph(context: &Context) -> Graph {
    Graph::new(
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
        let scores =
            network::scores(&mut tape, &variables, &graph, &rows, &[0.; 9], &[1.; 9]).unwrap();
        let seed = Matrix::from_column_slice(2, 1, &[0.2, -0.3]);
        tape.backward(scores, &seed).unwrap();
        let loss = |parameters: &Parameters| {
            let mut tape = Tape::new(&context);
            let variables = parameters.variables(&mut tape).unwrap();
            let scores =
                network::scores(&mut tape, &variables, &graph, &rows, &[0.; 9], &[1.; 9]).unwrap();
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
