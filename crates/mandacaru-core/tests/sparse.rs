use mandacaru_core::{
    quantum::{sparse::SparseMatrix, Basis, Operator, C64},
    Context,
};

#[test]
fn sparse_storage_counts_connected_rows_instead_of_pauli_terms() {
    // 256 diagonal Pauli strings still have one connected row per column.
    // Reserving dimension * term_count would exceed this valid budget.
    let context = Context::new(Some(1), 128 << 10).unwrap();
    {
        let basis = Basis::register(&context, 8).unwrap();
        let terms: Vec<_> = (0..256)
            .map(|bits| {
                let label = (0..8)
                    .map(|q| if bits & (1 << q) == 0 { 'I' } else { 'Z' })
                    .collect();
                (label, C64::new(1. / ((bits + 1) * (bits + 1)) as f64, 0.))
            })
            .collect();
        let operator = Operator::new(&context, 8, &terms).unwrap();
        let sparse = SparseMatrix::new(&context, &basis, &operator).unwrap();
        let mut state = context.allocate::<C64>(256).unwrap();
        for (i, value) in state.as_mut_slice().iter_mut().enumerate() {
            *value = C64::new((i as f64 * 0.13).sin(), (i as f64 * 0.07).cos());
        }
        let expected = operator.apply(&context, &basis, state.as_slice()).unwrap();
        let mut actual = context.allocate(256).unwrap();
        sparse.apply_into(state.as_slice(), actual.as_mut_slice());
        assert!(expected
            .as_slice()
            .iter()
            .zip(actual.as_slice())
            .all(|(a, b)| (a - b).norm() < 2e-12));
    }
    assert_eq!(context.memory_used(), 0);
}

#[test]
fn sparse_apply_of_a_complex64_state_accumulates_in_float64() {
    // The stored values and the output row are float64 whatever the state's
    // precision: the result equals the float64 apply of the same (rounded) state
    // to float64 round-off, not to float32 round-off.
    let context = Context::new(Some(1), 8 << 20).unwrap();
    {
        let n = 10;
        let basis = Basis::register(&context, n).unwrap();
        let mut terms = vec![("I".repeat(n), C64::new(0.4, 0.))];
        for q in 0..n - 1 {
            for (a, b) in [('X', 'X'), ('Y', 'Y'), ('Z', 'Z')] {
                let label: String = (0..n)
                    .map(|j| {
                        if j == q {
                            a
                        } else if j == q + 1 {
                            b
                        } else {
                            'I'
                        }
                    })
                    .collect();
                terms.push((label, C64::new(0.5 + 0.01 * q as f64, 0.)));
            }
        }
        let operator = Operator::new(&context, n, &terms).unwrap();
        let sparse = SparseMatrix::new(&context, &basis, &operator).unwrap();
        let narrow: Vec<num_complex::Complex32> = (0..basis.len())
            .map(|i| num_complex::Complex32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()))
            .collect();
        let widened: Vec<C64> = narrow
            .iter()
            .map(|c| C64::new(f64::from(c.re), f64::from(c.im)))
            .collect();
        let mut from_narrow = vec![C64::default(); basis.len()];
        let mut from_wide = vec![C64::default(); basis.len()];
        sparse.apply_into(&narrow, &mut from_narrow);
        sparse.apply_into(&widened, &mut from_wide);
        assert_eq!(from_narrow, from_wide);
        let mut matrix_free = vec![C64::default(); basis.len()];
        operator.apply_into(&basis, &narrow, &mut matrix_free);
        assert!(matrix_free
            .iter()
            .zip(&from_narrow)
            .all(|(a, b)| (a - b).norm() < 1e-12));
    }
    assert_eq!(context.memory_used(), 0);
}
