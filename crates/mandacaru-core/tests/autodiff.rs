use mandacaru_core::{math::autodiff::Tape, Context};
use nalgebra::DMatrix;

fn model(
    tape: &mut Tape,
    x: &DMatrix<f64>,
    w: &DMatrix<f64>,
    b: &DMatrix<f64>,
) -> (usize, [usize; 3]) {
    let xi = tape.leaf(x).unwrap();
    let wi = tape.leaf(w).unwrap();
    let bi = tape.leaf(b).unwrap();
    let y = tape.matmul(xi, wi).unwrap();
    let y = tape.add(y, bi).unwrap();
    let y = tape.tanh(y).unwrap();
    let normalized = tape.normalize_rows(y, 1e-6).unwrap();
    let both = tape.concat(&[y, normalized]).unwrap();
    let gathered = tape.gather(both, &[1, 0, 1]).unwrap();
    let weighted = tape.scale_rows(gathered, &[0.3, -0.2, 0.5]).unwrap();
    let ones = tape.leaf(&DMatrix::from_element(1, 3, 1.)).unwrap();
    let pooled = tape.matmul(ones, weighted).unwrap();
    let repeated = tape.repeat(pooled, 3).unwrap();
    let head = tape
        .leaf(&DMatrix::from_column_slice(4, 1, &[0.2, -0.3, 0.7, 0.1]))
        .unwrap();
    (tape.matmul(repeated, head).unwrap(), [xi, wi, bi])
}

#[test]
fn matrix_reverse_derivatives_match_independent_parameter_displacements() {
    let context = Context::new(Some(1), 16 << 20).unwrap();
    {
        let x = DMatrix::from_row_slice(2, 3, &[0.1, -0.3, 0.7, 0.4, 0.2, -0.5]);
        let w = DMatrix::from_row_slice(3, 2, &[0.2, -0.1, 0.4, 0.3, -0.5, 0.6]);
        let b = DMatrix::from_row_slice(1, 2, &[0.1, -0.2]);
        let seed = DMatrix::from_column_slice(3, 1, &[0.3, -0.1, 0.7]);
        let mut tape = Tape::new(&context);
        let (output, variables) = model(&mut tape, &x, &w, &b);
        tape.backward(output, &seed).unwrap();
        let loss = |x: &DMatrix<f64>, w: &DMatrix<f64>, b: &DMatrix<f64>| {
            let mut tape = Tape::new(&context);
            let (out, _) = model(&mut tape, x, w, b);
            tape.value(out)
                .iter()
                .zip(seed.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        for (which, base) in [&x, &w, &b].into_iter().enumerate() {
            for i in 0..base.len() {
                let mut plus = base.clone();
                let mut minus = base.clone();
                plus[i] += 1e-6;
                minus[i] -= 1e-6;
                let ep = match which {
                    0 => loss(&plus, &w, &b),
                    1 => loss(&x, &plus, &b),
                    _ => loss(&x, &w, &plus),
                };
                let em = match which {
                    0 => loss(&minus, &w, &b),
                    1 => loss(&x, &minus, &b),
                    _ => loss(&x, &w, &minus),
                };
                let derivative = tape.gradient(variables[which])[i];
                assert!(
                    (derivative - (ep - em) / 2e-6).abs() < 2e-7,
                    "variable {which}, element {i}"
                );
            }
        }
    }
    assert_eq!(context.memory_used(), 0);
    let small = Context::new(Some(1), 64).unwrap();
    assert!(Tape::new(&small).leaf(&DMatrix::zeros(10, 10)).is_err());
    assert_eq!(small.memory_used(), 0);
}

#[test]
fn shared_multioutput_derivatives_accumulate_all_heads_and_treat_graph_inputs_as_constants() {
    let context = Context::new(Some(1), 1 << 20).unwrap();
    {
        let x = DMatrix::from_row_slice(2, 2, &[0.3, -0.2, 0.7, 0.4]);
        let w = DMatrix::from_column_slice(2, 1, &[0.1, -0.4]);
        let mut tape = Tape::new(&context);
        let input = tape.constant(&x).unwrap();
        let weights = tape.leaf(&w).unwrap();
        let linear = tape.matmul(input, weights).unwrap();
        let nonlinear = tape.tanh(linear).unwrap();
        let a = DMatrix::from_column_slice(2, 1, &[0.2, -0.3]);
        let b = DMatrix::from_column_slice(2, 1, &[0.5, 0.7]);
        tape.backward_many(&[
            (linear, a.clone()),
            (nonlinear, b.clone()),
            (linear, a.clone()),
        ])
        .unwrap();
        let loss = |weights: &DMatrix<f64>| {
            let linear = &x * weights;
            linear
                .iter()
                .zip(a.iter().zip(b.iter()))
                .map(|(v, (a, b))| 2. * a * v + b * v.tanh())
                .sum::<f64>()
        };
        for i in 0..2 {
            let mut plus = w.clone();
            let mut minus = w.clone();
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            assert!((tape.gradient(weights)[i] - (loss(&plus) - loss(&minus)) / 2e-6).abs() < 1e-9);
        }
        assert!(tape.gradient(input).iter().all(|v| *v == 0.));
        tape.backward(nonlinear, &b).unwrap();
        let expected = x.transpose()
            * DMatrix::from_fn(2, 1, |r, _| b[r] * (1. - tape.value(nonlinear)[r].powi(2)));
        assert!((tape.gradient(weights) - expected).norm() < 1e-14);
    }
    assert_eq!(context.memory_used(), 0);
}
