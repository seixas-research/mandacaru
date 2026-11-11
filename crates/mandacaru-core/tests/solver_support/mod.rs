//! The model problem the solver tests share.
use mandacaru_core::{
    quantum::{CachedProblem, Encoding, Operator, C64},
    Context,
};

/// A correlated two-electron problem on three spatial orbitals (six spin orbitals), in the
/// particle-number sector: one-body levels and hoppings, on-site repulsion and a pair hopping,
/// so ADAPT grows several operators. With the alpha orbitals on qubits 0-2 and the beta ones on
/// 3-5, an alpha and a beta excitation of the fermionic pool act on disjoint qubits.
pub fn problem(ctx: &Context, pool: &str) -> CachedProblem {
    let encoding = Encoding::new("jordan_wigner", 6, [1, 1]).unwrap();
    let mut h = Operator::new(ctx, 6, &[]).unwrap();
    let add = |h: Operator, term: &[(usize, bool)], c: f64| {
        h.add(ctx, &encoding.monomial(ctx, term).unwrap(), C64::new(c, 0.))
            .unwrap()
    };
    for (p, e) in [-1.1, -0.35, 0.4].into_iter().enumerate() {
        for s in [0, 3] {
            h = add(h, &[(p + s, true), (p + s, false)], e);
        }
    }
    for (p, q, t) in [(0, 1, -0.21), (1, 2, -0.17), (0, 2, 0.06)] {
        for s in [0, 3] {
            h = add(h, &[(p + s, true), (q + s, false)], t);
            h = add(h, &[(q + s, true), (p + s, false)], t);
        }
    }
    for (p, u) in [(0, 0.55), (1, 0.48), (2, 0.41)] {
        h = add(
            h,
            &[(p, true), (p, false), (p + 3, true), (p + 3, false)],
            u,
        );
    }
    // Pair hopping between orbitals 0 and 2, both directions.
    h = add(h, &[(2, true), (5, true), (3, false), (0, false)], 0.12);
    h = add(h, &[(0, true), (3, true), (5, false), (2, false)], 0.12);
    let mut problem = CachedProblem::new(ctx, encoding, h, true, false, true).unwrap();
    problem.use_pool(pool, true).unwrap();
    problem
}
