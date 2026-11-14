//! Shared by the `search_*` tests: a small number-conserving model whose architecture space is
//! enumerable. Six qubits, one alpha and one beta electron in three orbitals; its fermionic pool
//! with singles has eight operators, so `max_length = 2` gives exactly 73 sequences.
#![allow(dead_code)]
use mandacaru_core::{
    quantum::{CachedProblem, Encoding, Operator, C64},
    Context,
};
use std::collections::BTreeMap;

pub const MODES: usize = 6;

/// A number-conserving Hermitian Hamiltonian on three spatial orbitals (modes 0-2 alpha, 3-5 beta)
/// with orbital energies, hopping, an on-site and an inter-orbital density interaction.
pub fn hamiltonian(context: &Context) -> Operator {
    let mut terms = BTreeMap::<String, f64>::new();
    let string = |paulis: &[(usize, char)]| -> String {
        let mut chars = vec!['I'; MODES];
        for &(mode, pauli) in paulis {
            chars[mode] = pauli;
        }
        chars.into_iter().collect()
    };
    let mut add = |key: String, value: f64| *terms.entry(key).or_default() += value;
    // n_p = (1 - Z_p) / 2.
    for offset in [0, 3] {
        for (orbital, energy) in [0.0, 0.5, 1.1].into_iter().enumerate() {
            add(string(&[]), energy / 2.);
            add(string(&[(offset + orbital, 'Z')]), -energy / 2.);
        }
        // Hopping between orbitals p < q of one spin: (X Z..Z X + Y Z..Z Y) / 2.
        for (p, q, t) in [(0, 1, -0.35), (1, 2, -0.2), (0, 2, 0.15)] {
            for pauli in ['X', 'Y'] {
                let mut paulis = vec![(offset + p, pauli), (offset + q, pauli)];
                paulis.extend((offset + p + 1..offset + q).map(|mode| (mode, 'Z')));
                add(string(&paulis), t / 2.);
            }
        }
    }
    // Density-density terms n_a n_b = (1 - Z_a - Z_b + Z_a Z_b) / 4.
    let mut density = |a: usize, b: usize, u: f64| {
        add(string(&[]), u / 4.);
        add(string(&[(a, 'Z')]), -u / 4.);
        add(string(&[(b, 'Z')]), -u / 4.);
        add(string(&[(a, 'Z'), (b, 'Z')]), u / 4.);
    };
    for p in 0..3 {
        density(p, 3 + p, 0.6);
    }
    density(0, 4, 0.25);
    density(1, 5, 0.15);
    density(2, 3, 0.3);
    let terms: Vec<_> = terms
        .into_iter()
        .map(|(key, value)| (key, C64::new(value, 0.)))
        .collect();
    Operator::new(context, MODES, &terms).unwrap()
}

pub fn architectures(pool: usize, maximum: usize) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    let mut last = vec![vec![]];
    for _ in 0..maximum {
        let next: Vec<Vec<usize>> = last
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

/// The model as a cached problem (Jordan-Wigner, the fermionic pool with singles).
pub fn problem(context: &Context) -> CachedProblem {
    let problem = CachedProblem::new(
        context,
        Encoding::new("jordan_wigner", MODES, [1, 1]).unwrap(),
        hamiltonian(context),
        true,
        false,
        true,
    )
    .unwrap();
    assert_eq!(
        problem.pool.len(),
        8,
        "the model's pool is the eight singles and doubles"
    );
    problem
}
