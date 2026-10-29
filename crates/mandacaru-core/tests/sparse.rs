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
