//! A Hubbard chain with a product ansatz of coupled-exchange operators, shared by the tests of the
//! shared-prefix importances and of the deferred gradient.
use mandacaru_core::{
    math::storage::Precision,
    quantum::{CachedProblem, Encoding, Operator, C64},
    Context,
};

/// A Hubbard chain of `sites` sites (spin-blocked modes: `spin * sites + site`).
pub fn hubbard(ctx: &Context, encoding: &Encoding, sites: usize) -> Operator {
    let modes = 2 * sites;
    let mut h = Operator::new(ctx, modes, &[]).unwrap();
    for spin in 0..2 {
        for site in 0..sites - 1 {
            let (i, j) = (spin * sites + site, spin * sites + site + 1);
            for (p, q) in [(i, j), (j, i)] {
                h = h
                    .add(
                        ctx,
                        &encoding.monomial(ctx, &[(p, true), (q, false)]).unwrap(),
                        C64::new(-1., 0.),
                    )
                    .unwrap();
            }
        }
    }
    for site in 0..sites {
        let (up, down) = (site, sites + site);
        h = h
            .add(
                ctx,
                &encoding
                    .monomial(ctx, &[(up, true), (up, false), (down, true), (down, false)])
                    .unwrap(),
                C64::new(4., 0.),
            )
            .unwrap();
    }
    h
}

/// A problem whose pool is `operators` coupled-exchange generators, combined four-string ones and
/// their eight-string members alternating (the eight-string ones cross the parallel threshold on a
/// 44100-row sector; the four-string ones stay in place), with angles of both signs and an exact
/// zero among them.
pub fn problem(
    threads: usize,
    modes: usize,
    electrons: [usize; 2],
    operators: usize,
    precision: Precision,
) -> (CachedProblem, Vec<usize>, Vec<f64>) {
    let ctx = Context::new(Some(threads), 1 << 30).unwrap();
    let encoding = Encoding::new("jordan_wigner", modes, electrons).unwrap();
    let h = hubbard(&ctx, &encoding, modes / 2);
    let mut problem = CachedProblem::new(&ctx, encoding, h, true, false, true)
        .unwrap()
        .with_precision(precision);
    let pool = mandacaru_core::quantum::pools::build(&ctx, &problem.encoding, "ceo", true).unwrap();
    let (mut combined, mut members) = (Vec::new(), Vec::new());
    for (g, visible) in pool.operators.into_iter().zip(pool.screenable) {
        if visible && g.label.contains(")+D(") && g.operator.terms.len() == 4 {
            combined.push(g);
        } else if !visible && g.operator.terms.len() == 8 {
            members.push(g);
        }
    }
    let half = operators / 2;
    let (a, b) = (
        (combined.len() / half).max(1),
        (members.len() / half).max(1),
    );
    let chosen: Vec<_> = combined
        .into_iter()
        .step_by(a)
        .take(half)
        .zip(members.into_iter().step_by(b).take(half))
        .flat_map(|(x, y)| [x, y])
        .collect();
    assert_eq!(chosen.len(), operators);
    problem.pool = chosen;
    let theta: Vec<f64> = (0..operators)
        .map(|k| {
            if k == 3 {
                0.
            } else {
                0.3 * ((k as f64) * 1.7 + 0.4).sin()
            }
        })
        .collect();
    ((problem), (0..operators).collect(), theta)
}
