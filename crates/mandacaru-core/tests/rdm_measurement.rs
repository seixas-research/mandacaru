//! Measured reduced density matrices (`quantum::rdm_measurement`): assembled from the
//! expectation values of their Pauli strings, they equal the state-vector route's
//! (`quantum::properties::spatial_rdms`) in every mapping, and the plan refuses a budget it
//! would exceed.
use mandacaru_core::{
    quantum::{
        properties::spatial_rdms, rdm_measurement::RdmMeasurement, Basis, Encoding, Operator, C64,
    },
    Context,
};

fn state(n: usize, seed: u64, complex: bool) -> Vec<C64> {
    let mut x = seed;
    let mut next = || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut v: Vec<C64> = (0..n)
        .map(|_| C64::new(next(), if complex { next() } else { 0. }))
        .collect();
    let norm = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    v.iter_mut().for_each(|c| *c /= norm);
    v
}

/// The value of every measured string, products included, read as a device reads it: the
/// rotated `Z` string after the basis's rotations, `<psi|U^dag Z U|psi>`.
fn measured_values(
    ctx: &Context,
    basis: &Basis,
    n: usize,
    plan: &RdmMeasurement,
    psi: &[C64],
) -> Vec<f64> {
    let mut values = vec![f64::NAN; plan.measured()];
    for (tail, observables) in plan.requests() {
        let mut unitary = Operator::new(ctx, n, &[("I".repeat(n), C64::new(1., 0.))]).unwrap();
        for (label, angle) in &tail {
            unitary = rotation(ctx, n, label, *angle)
                .compose(ctx, &unitary)
                .unwrap();
        }
        for (z, index) in observables {
            let measured = Operator::new(ctx, n, &[(z, C64::new(1., 0.))]).unwrap();
            let back = unitary
                .dagger(ctx)
                .unwrap()
                .compose(ctx, &measured.compose(ctx, &unitary).unwrap())
                .unwrap();
            let out = back.apply(ctx, basis, psi).unwrap();
            values[index as usize] = psi
                .iter()
                .zip(out.as_slice())
                .map(|(a, b)| (a.conj() * b).re)
                .sum();
        }
    }
    assert!(values.iter().all(|v| v.is_finite()));
    values
}

/// `<psi|P|psi>` of every observable, from the state: what a noiseless device reports.
fn expectations(
    ctx: &Context,
    basis: &Basis,
    n: usize,
    labels: &[String],
    psi: &[C64],
) -> Vec<f64> {
    labels
        .iter()
        .map(|label| {
            let p = Operator::new(ctx, n, &[(label.clone(), C64::new(1., 0.))]).unwrap();
            let out = p.apply(ctx, basis, psi).unwrap();
            psi.iter()
                .zip(out.as_slice())
                .map(|(a, b)| (a.conj() * b).re)
                .sum()
        })
        .collect()
}

#[test]
fn measured_rdms_are_the_state_rdms_in_every_mapping() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    for mapping in ["jordan_wigner", "parity", "bravyi_kitaev"] {
        for complex in [false, true] {
            let encoding = Encoding::new(mapping, 8, [2, 1]).unwrap();
            let basis = Basis::new(&ctx, &encoding, true).unwrap();
            let psi = state(basis.len(), 17, complex);
            let plan = RdmMeasurement::new(&ctx, &encoding, None).unwrap();
            let labels = plan.observables();
            let values = measured_values(&ctx, &basis, encoding.n_qubits(), &plan, &psi);
            // The rotated reading of each string is its unrotated expectation.
            let direct = expectations(&ctx, &basis, encoding.n_qubits(), &labels, &psi);
            let rotated = direct
                .iter()
                .zip(&values)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            assert!(rotated < 1e-12, "{rotated:.2e}");
            let errors = vec![1e-3; values.len()];
            let (measured, spread) = plan
                .assemble(&ctx, &values, Some(&errors), Some(4000))
                .unwrap();
            let exact = spatial_rdms(&ctx, &encoding, &basis, &psi).unwrap();
            let worst = |a: &[C64], b: &[C64]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x.re - y.re).abs())
                    .fold(0., f64::max)
            };
            let (one, two) = (
                worst(measured.one.as_slice(), exact.one.as_slice()),
                worst(measured.two.as_slice(), exact.two.as_slice()),
            );
            eprintln!(
                "{mapping} complex={complex}: {} strings, 1-RDM {one:.2e}, 2-RDM {two:.2e}",
                labels.len()
            );
            assert!(one < 1e-12 && two < 1e-12, "{mapping}: {one:.2e} {two:.2e}");
            // The trace is the electron count, the 2-RDM trace N(N-1).
            let m = 4;
            let trace: f64 = (0..m).map(|p| measured.one.as_slice()[p * m + p].re).sum();
            assert!((trace - 3.).abs() < 1e-12);
            let pairs: f64 = (0..m)
                .flat_map(|p| (0..m).map(move |q| (p, q)))
                .map(|(p, q)| measured.two.as_slice()[((p * m + q) * m + p) * m + q].re)
                .sum();
            assert!((pairs - 6.).abs() < 1e-12);
            let (one_error, two_error) = spread.unwrap();
            assert!(one_error
                .iter()
                .chain(&two_error)
                .all(|e| e.is_finite() && *e >= 0.));
            assert!(one_error.iter().any(|e| *e > 0.));
        }
    }
}

#[test]
fn the_plan_refuses_a_budget_it_would_exceed_and_wrong_values() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let encoding = Encoding::new("jordan_wigner", 8, [2, 1]).unwrap();
    let plan = RdmMeasurement::new(&ctx, &encoding, None).unwrap();
    let count = plan.observables().len();
    assert!(RdmMeasurement::new(&ctx, &encoding, Some(count)).is_ok());
    let refused = RdmMeasurement::new(&ctx, &encoding, Some(count / 2))
        .err()
        .unwrap();
    assert!(
        refused.to_string().contains("observables budget"),
        "{refused}"
    );
    let measured = plan.measured();
    assert!(measured > count);
    assert!(plan
        .assemble(&ctx, &vec![0.; measured - 1], None, None)
        .is_err());
    assert!(plan
        .assemble(&ctx, &vec![f64::NAN; measured], None, None)
        .is_err());
    assert!(plan
        .assemble(&ctx, &vec![0.; measured], None, Some(0))
        .is_err());
}

/// Fifty qubits: the plan is polynomial (no register, basis or state is formed).
#[test]
fn a_fifty_qubit_plan_is_polynomial() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let encoding = Encoding::new("jordan_wigner", 50, [5, 5]).unwrap();
    let plan = RdmMeasurement::new(&ctx, &encoding, None).unwrap();
    let count = plan.observables().len();
    eprintln!(
        "50 qubits: {count} strings, native peak {} MB",
        ctx.memory_peak() >> 20
    );
    assert!(count < 50usize.pow(4));
    assert!(ctx.memory_peak() < 400 << 20);
}

/// `exp(i a P)` as an operator.
fn rotation(ctx: &Context, n: usize, label: &str, angle: f64) -> Operator {
    Operator::new(
        ctx,
        n,
        &[
            ("I".repeat(n), C64::new(angle.cos(), 0.)),
            (label.to_string(), C64::new(0., angle.sin())),
        ],
    )
    .unwrap()
}

/// Every basis is qubit-wise commuting, every string is in exactly one, and the rotated `Z`
/// string measured after a basis's rotations is, conjugated back, the string it stands for.
#[test]
fn each_basis_rotates_its_strings_to_z_exactly() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    for mapping in ["jordan_wigner", "bravyi_kitaev"] {
        let encoding = Encoding::new(mapping, 8, [2, 1]).unwrap();
        let n = encoding.n_qubits();
        let plan = RdmMeasurement::new(&ctx, &encoding, None).unwrap();
        let labels = plan.observables();
        let requests = plan.requests();
        assert_eq!(requests.len(), plan.bases());
        assert!(plan.bases() < labels.len());
        let mut seen = vec![0usize; labels.len()];
        for (tail, observables) in &requests {
            let mut unitary = Operator::new(&ctx, n, &[("I".repeat(n), C64::new(1., 0.))]).unwrap();
            for (label, angle) in tail {
                unitary = rotation(&ctx, n, label, *angle)
                    .compose(&ctx, &unitary)
                    .unwrap();
            }
            for (z, index) in observables {
                assert!(z.chars().all(|c| c == 'I' || c == 'Z'), "{z}");
                assert!((*index as usize) < plan.measured());
                if *index as usize >= labels.len() {
                    // A covariance product: read in this basis, checked by its value elsewhere.
                    continue;
                }
                seen[*index as usize] += 1;
                let measured = Operator::new(&ctx, n, &[(z.clone(), C64::new(1., 0.))]).unwrap();
                let back = unitary
                    .dagger(&ctx)
                    .unwrap()
                    .compose(&ctx, &measured.compose(&ctx, &unitary).unwrap())
                    .unwrap();
                let expected = &labels[*index as usize];
                let terms: Vec<_> = back
                    .labels()
                    .into_iter()
                    .filter(|(_, c)| c.norm() > 1e-12)
                    .collect();
                assert_eq!(terms.len(), 1, "{mapping} {expected}");
                assert_eq!(&terms[0].0, expected);
                assert!(
                    (terms[0].1 - C64::new(1., 0.)).norm() < 1e-12,
                    "{expected} {:?}",
                    terms[0].1
                );
            }
        }
        assert!(seen.iter().all(|&k| k == 1));
    }
}

/// Fifty qubits: the million strings are grouped in polynomial time and memory.
#[test]
fn a_fifty_qubit_grouping_finishes() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let encoding = Encoding::new("jordan_wigner", 50, [5, 5]).unwrap();
    let start = std::time::Instant::now();
    let plan = RdmMeasurement::new(&ctx, &encoding, None).unwrap();
    eprintln!(
        "50 qubits: {} strings in {} bases, {:.1} s",
        plan.observables().len(),
        plan.bases(),
        start.elapsed().as_secs_f64()
    );
    assert!(plan.bases() < plan.observables().len());
}

/// The structured grouping against the general first-fit grouper (`execution::qwc`), where the
/// latter still finishes: no more bases (measured 11,661 against 12,743 on 24 qubits).
#[test]
fn the_structured_grouping_is_close_to_first_fit() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let encoding = Encoding::new("jordan_wigner", 24, [3, 3]).unwrap();
    let plan = RdmMeasurement::new(&ctx, &encoding, None).unwrap();
    let labels = plan.observables();
    let start = std::time::Instant::now();
    let first_fit = mandacaru_core::execution::qwc(&ctx, &labels, None)
        .unwrap()
        .groups
        .len();
    eprintln!(
        "24 qubits: {} strings, structured {} bases, first-fit {first_fit} bases ({:.1} s)",
        labels.len(),
        plan.bases(),
        start.elapsed().as_secs_f64()
    );
    assert!(plan.bases() <= first_fit);
}

/// The covariance term, against a direct computation: a state with perfectly paired alpha and
/// beta occupations (`a|1100> + b|0011>` on 2 spatial orbitals) makes `gamma_11` the sum of two
/// strings of one basis that move together, so its variance is twice the independent estimate.
#[test]
fn paired_occupations_double_the_variance_of_their_sum() {
    let ctx = Context::new(Some(1), 512 << 20).unwrap();
    let encoding = Encoding::new("jordan_wigner", 4, [1, 1]).unwrap();
    let basis = Basis::new(&ctx, &encoding, true).unwrap();
    let plan = RdmMeasurement::new(&ctx, &encoding, None).unwrap();
    // The sector holds |a0 b0>, |a0 b1>, |a1 b0>, |a1 b1>; keep only the paired two.
    let mut psi = vec![C64::default(); basis.len()];
    let (paired, excited) = (0.95f64, (1. - 0.95f64 * 0.95).sqrt());
    for k in 0..basis.len() {
        let ket = basis.indices.as_slice()[k];
        let alpha = ket >> 2;
        let beta = ket & 0b11;
        if alpha == beta {
            psi[k] = C64::new(if alpha == 0b10 { paired } else { excited }, 0.);
        }
    }
    let values = measured_values(&ctx, &basis, 4, &plan, &psi);
    let shots = 1000u64;
    // Per-string errors as a counting device reports them.
    let errors: Vec<f64> = values
        .iter()
        .map(|v| ((1. - v * v).max(0.) / shots as f64).sqrt())
        .collect();
    let (rdms, spread) = plan
        .assemble(&ctx, &values, Some(&errors), Some(shots))
        .unwrap();
    let (_, independent) = plan.assemble(&ctx, &values, Some(&errors), None).unwrap();
    let occupation = rdms.one.as_slice()[3].re; // gamma_11, the excited pair's orbital
                                                // gamma_11 = 2 n with n = excited^2 (both spins at once): sd = 2 sqrt(n (1 - n) / shots).
    let n = excited * excited;
    let exact = 2. * (n * (1. - n) / shots as f64).sqrt();
    let (with, without) = (spread.unwrap().0[3], independent.unwrap().0[3]);
    eprintln!(
        "gamma_11 {occupation:.4}: error {with:.5} (exact {exact:.5}), independent {without:.5}"
    );
    assert!((occupation - 2. * n).abs() < 1e-12);
    assert!(
        (with - exact).abs() < 1e-12 * exact.max(1.),
        "{with} {exact}"
    );
    assert!((with / without - 2f64.sqrt()).abs() < 1e-9);
}
