//! The deterministic frame of degenerate orbital clusters (`molecular::frame`): the pure helper
//! on constructed moments, then linear hydrogen chains on the synthetic PAW-LCAO set (no
//! dataset files), whose virtual pi orbitals are exactly degenerate pairs.
mod paw_forces_support;

use mandacaru_core::{
    molecular::{
        frame::{clusters, fix_cluster_gauge, lab_moments, Moments},
        MeanField, MolecularIntegrals,
    },
    quantum::{
        transfer::{self, Frame},
        C64,
    },
    Context, Result,
};
use nalgebra::{DMatrix, DVector};
use paw_forces_support::synthetic::{displaced, integrals, solve, Method};

/// Moments of a cluster from fixed symmetric operators `a` and vectors `g` over the whole space:
/// the quadratic forms and linear functionals of the cluster's columns.
fn provider(
    operators: Vec<DMatrix<f64>>,
    vectors: Vec<DVector<f64>>,
) -> impl FnMut(&DMatrix<f64>, bool) -> Result<Moments> {
    move |block, with_operators| {
        Ok(Moments {
            operators: operators
                .iter()
                .filter(|_| with_operators)
                .map(|a| block.transpose() * a * block)
                .collect(),
            functionals: vectors
                .iter()
                .map(|g| (block.transpose() * g, g.norm()))
                .collect(),
            resolution: 1e-6,
        })
    }
}

fn symmetric(n: usize, seed: f64) -> DMatrix<f64> {
    let raw = DMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 3 + 1) as f64 * seed).sin());
    (&raw + raw.transpose()) * 0.5
}

fn complex_identity(n: usize) -> DMatrix<C64> {
    DMatrix::identity(n, n)
}

/// `orbitals` with columns `0..3` mixed by a rotation about the first two axes and the second
/// column negated: any orthogonal change of the cluster's basis.
fn rotated(orbitals: &DMatrix<C64>, a: f64, b: f64, flip: bool) -> DMatrix<C64> {
    let (sa, ca, sb, cb) = (a.sin(), a.cos(), b.sin(), b.cos());
    let u = DMatrix::from_row_slice(
        3,
        3,
        &[ca, -sa, 0., sa * cb, ca * cb, -sb, sa * sb, ca * sb, cb],
    ) * if flip { -1. } else { 1. };
    let mut out = orbitals.clone();
    for r in 0..orbitals.nrows() {
        for c in 0..3 {
            out[(r, c)] = (0..3).map(|k| orbitals[(r, k)] * u[(k, c)]).sum();
        }
    }
    out
}

fn five_orbitals(operators: Vec<DMatrix<f64>>) -> (DMatrix<C64>, DMatrix<C64>) {
    let labels = [0., 0., 0., 1., 2.];
    let vectors = vec![
        DVector::from_fn(5, |i, _| 0.5 + ((i + 1) as f64 * 1.3).sin()),
        DVector::from_fn(5, |i, _| ((i + 2) as f64 * 0.7).cos()),
    ];
    let domains = [(0..5).collect::<Vec<_>>()];
    let mut straight = complex_identity(5);
    let mut mixed = rotated(&complex_identity(5), 0.83, -1.17, false);
    let mut provide = provider(operators, vectors);
    assert_eq!(
        fix_cluster_gauge(&mut straight, &labels, &domains, &mut provide).unwrap(),
        1
    );
    assert_eq!(
        fix_cluster_gauge(&mut mixed, &labels, &domains, &mut provide).unwrap(),
        1
    );
    (straight, mixed)
}

#[test]
fn a_rotation_inside_a_cluster_leaves_the_frame_unchanged() {
    let (straight, mixed) = five_orbitals(vec![symmetric(5, 0.9), symmetric(5, 1.7)]);
    assert!(
        (&straight - &mixed).norm() < 1e-12,
        "{}",
        (&straight - &mixed).norm()
    );
    // The frame is orthonormal and leaves the orbitals outside the cluster alone.
    assert!((straight.adjoint() * &straight - complex_identity(5)).norm() < 1e-12);
    // The orbitals outside the cluster keep their place and take only a sign.
    for c in 3..5 {
        assert!((straight[(c, c)].norm() - 1.).abs() < 1e-14);
    }
    // A negated input column changes nothing either: the sign belongs to the orbital function.
    let labels = [0., 0., 0., 1., 2.];
    let domains = [(0..5).collect::<Vec<_>>()];
    let mut flipped = rotated(&complex_identity(5), 0.83, -1.17, true);
    let mut provide = provider(
        vec![symmetric(5, 0.9), symmetric(5, 1.7)],
        vec![
            DVector::from_fn(5, |i, _| 0.5 + ((i + 1) as f64 * 1.3).sin()),
            DVector::from_fn(5, |i, _| ((i + 2) as f64 * 0.7).cos()),
        ],
    );
    fix_cluster_gauge(&mut flipped, &labels, &domains, &mut provide).unwrap();
    assert!((&straight - &flipped).norm() < 1e-12);
}

#[test]
fn a_second_operator_resolves_what_the_first_leaves_degenerate() {
    // The first operator is a multiple of the identity on the cluster, so only the second
    // orders its orbitals; the frame is still independent of the incoming rotation.
    let first = DMatrix::from_diagonal(&DVector::from_vec(vec![2., 2., 2., 5., 7.]));
    let (straight, mixed) = five_orbitals(vec![first, symmetric(5, 2.3)]);
    assert!((&straight - &mixed).norm() < 1e-12);
}

#[test]
fn a_cluster_no_operator_resolves_raises() {
    let flat = DMatrix::from_diagonal(&DVector::from_vec(vec![2., 2., 2., 5., 7.]));
    let mut orbitals = complex_identity(5);
    let mut provide = provider(vec![flat.clone(), flat], vec![DVector::from_element(5, 1.)]);
    let error = fix_cluster_gauge(
        &mut orbitals,
        &[0., 0., 0., 1., 2.],
        &[(0..5).collect()],
        &mut provide,
    )
    .unwrap_err();
    assert!(error.to_string().contains("deterministic frame"), "{error}");
}

#[test]
fn a_frame_without_a_cluster_asks_for_no_operators_and_takes_only_signs() {
    let vectors = vec![
        DVector::from_fn(5, |i, _| 0.5 + ((i + 1) as f64 * 1.3).sin()),
        DVector::from_fn(5, |i, _| ((i + 2) as f64 * 0.7).cos()),
    ];
    let labels = [0., 0.1, 0.2, 1., 2.];
    let domains = [(0..5).collect::<Vec<_>>()];
    let mut operators_asked = false;
    let mut inner = provider(vec![symmetric(5, 0.9)], vectors.clone());
    let mut watched = |block: &DMatrix<f64>, with_operators: bool| {
        operators_asked |= with_operators;
        inner(block, with_operators)
    };
    let mut orbitals = rotated(&complex_identity(5), 0.4, 0.2, false);
    let before = orbitals.clone();
    let changed = fix_cluster_gauge(&mut orbitals, &labels, &domains, &mut watched).unwrap();
    assert_eq!(changed, 0);
    assert!(!operators_asked);
    for c in 0..5 {
        let same = (&orbitals.column(c) - before.column(c)).norm();
        let flipped = (&orbitals.column(c) + before.column(c)).norm();
        assert!(
            same < 1e-15 || flipped < 1e-15,
            "column {c} changed beyond its sign"
        );
    }
}

#[test]
fn a_sign_flip_of_an_orbital_outside_every_cluster_is_undone() {
    let vectors = vec![
        DVector::from_fn(5, |i, _| 0.5 + ((i + 1) as f64 * 1.3).sin()),
        DVector::from_fn(5, |i, _| ((i + 2) as f64 * 0.7).cos()),
    ];
    let labels = [0., 0.1, 0.2, 1., 2.];
    let domains = [(0..5).collect::<Vec<_>>()];
    let mut provide = provider(vec![symmetric(5, 0.9)], vectors);
    let mut straight = complex_identity(5);
    fix_cluster_gauge(&mut straight, &labels, &domains, &mut provide).unwrap();
    for column in 0..5 {
        let mut flipped = complex_identity(5);
        flipped.column_mut(column).scale_mut(-1.);
        fix_cluster_gauge(&mut flipped, &labels, &domains, &mut provide).unwrap();
        assert_eq!(flipped, straight, "flip of column {column}");
    }
}

#[test]
fn clusters_stay_inside_their_domain_and_within_the_tolerance_of_their_lowest_label() {
    let labels = [0., 0., 0., 0.];
    assert_eq!(
        clusters(&labels, &[vec![0, 1], vec![2, 3]]).unwrap(),
        vec![vec![0, 1], vec![2, 3]]
    );
    assert_eq!(
        clusters(&labels, &[vec![0], vec![1, 2, 3]]).unwrap(),
        vec![vec![1, 2, 3]]
    );
    // 5e-9 is inside the 1e-8 tolerance of the lowest label, 2e-8 is outside it.
    assert_eq!(
        clusters(&[2e-8, 0., 5e-9], &[vec![0, 1, 2]]).unwrap(),
        vec![vec![1, 2]]
    );
    assert!(clusters(&[f64::NAN, 0.], &[vec![0, 1]]).is_err());
}

/// Linear H4: its virtual pi orbitals are exactly degenerate pairs.
const H4: [[f64; 3]; 4] = [[0., 0., -2.5], [0., 0., -0.8], [0., 0., 0.9], [0., 0., 2.6]];
/// The doubly occupied and the empty orbitals of the closed-shell H4 reference.
fn domains(m: usize) -> [Vec<usize>; 2] {
    [(0..2).collect(), (2..m).collect()]
}

/// The first virtual cluster of the state and the active space that cuts it: the occupied
/// orbitals and the cluster's first member.
fn cutting_spec(state: &MeanField) -> (Vec<usize>, String) {
    let m = state.model_orbitals.ncols();
    let found = clusters(&state.energies[0], &domains(m)).unwrap();
    let cluster = found
        .first()
        .expect("linear H4 has degenerate pi pairs")
        .clone();
    assert_eq!(cluster.len(), 2, "{found:?}");
    (
        cluster.clone(),
        format!(r#"{{"orbitals": [0, 1, {}]}}"#, cluster[0]),
    )
}

/// The energy of a fixed operator sequence at fixed angles on the active space `spec`.
fn sequence_energy(base: &MolecularIntegrals, state: &MeanField, spec: &str) -> f64 {
    let model = state.active_model(base, spec).unwrap();
    let problem = model
        .problem(&base.context, "jordan_wigner", true, false, true)
        .unwrap();
    let count = problem.pool.len().min(6);
    let indices: Vec<usize> = (0..count).collect();
    let angles: Vec<f64> = (0..count).map(|k| 0.3 * ((k + 1) as f64).cos()).collect();
    problem.evaluate(&indices, &angles, true).unwrap().energy
}

#[test]
fn the_scf_frame_of_a_linear_molecule_ignores_a_rotation_of_its_degenerate_orbitals() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H4, "lda");
        let mut state = solve(&base, Method::Hf("rhf"), [2, 2]);
        let m = state.model_orbitals.ncols();
        let (cluster, spec) = cutting_spec(&state);
        let reference = state.model_orbitals.clone();
        let before = sequence_energy(&base, &state, &spec);
        // Rotate the pair by an arbitrary angle, as another eigensolver run could return it.
        let (s, c) = 0.7f64.sin_cos();
        for r in 0..m {
            let (a, b) = (
                state.model_orbitals[(r, cluster[0])],
                state.model_orbitals[(r, cluster[1])],
            );
            state.model_orbitals[(r, cluster[0])] = a * c + b * s;
            state.model_orbitals[(r, cluster[1])] = b * c - a * s;
        }
        // And an occupied orbital that no cluster touches comes back with the opposite sign.
        state.model_orbitals.column_mut(0).scale_mut(-1.);
        // The orbitals as rotated give another truncated-ansatz energy (printed, not asserted:
        // a symmetric sequence would not feel it).
        let unfixed = sequence_energy(&base, &state, &spec);
        println!("fixed-sequence energy: frame {before:.12}, rotated {unfixed:.12}");
        let labels = state.energies[0].clone();
        let changed = fix_cluster_gauge(
            &mut state.model_orbitals,
            &labels,
            &domains(m),
            &mut lab_moments(&base),
        )
        .unwrap();
        assert!(changed >= 1);
        assert!(
            (&state.model_orbitals - &reference).norm() < 1e-10,
            "frame moved by {:.3e}",
            (&state.model_orbitals - &reference).norm()
        );
        let after = sequence_energy(&base, &state, &spec);
        assert!((after - before).abs() < 1e-10, "{before} {after}");
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn the_mp2_natural_orbitals_of_a_degenerate_cluster_ignore_the_canonical_frame() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H4, "lda");
        let mut state = solve(&base, Method::Hf("rhf"), [2, 2]);
        let m = state.model_orbitals.ncols();
        let (cluster, _) = cutting_spec(&state);
        let spec = r#"{"orbitals": 6, "method": "mp2"}"#;
        let columns = |state: &MeanField| {
            let model = state.active_model(&base, spec).unwrap();
            let rotation = model.orbital_rotation.as_ref().unwrap();
            (
                &state.model_orbitals * &rotation.matrix,
                rotation.occupations.clone(),
            )
        };
        let (straight, occupations) = columns(&state);
        let (s, c) = 0.7f64.sin_cos();
        for r in 0..m {
            let (a, b) = (
                state.model_orbitals[(r, cluster[0])],
                state.model_orbitals[(r, cluster[1])],
            );
            state.model_orbitals[(r, cluster[0])] = a * c + b * s;
            state.model_orbitals[(r, cluster[1])] = b * c - a * s;
        }
        // A flipped occupied orbital too: its natural orbital comes back with its own sign.
        state.model_orbitals.column_mut(0).scale_mut(-1.);
        let (mixed, _) = columns(&state);
        // Every natural orbital agrees with its sign: the frame is the same.
        let degenerate = clusters(&occupations, &[(2..m).collect()]).unwrap();
        assert!(!degenerate.is_empty(), "no degenerate natural occupations");
        for column in 0..m {
            let overlap = straight.column(column).adjoint() * mixed.column(column);
            assert!(
                (overlap[(0, 0)].re - 1.).abs() < 1e-8,
                "natural orbital {column} moved: overlap {}",
                overlap[(0, 0)]
            );
        }
    }
    assert_eq!(ctx.memory_used(), 0);
}

#[test]
fn an_axial_step_keeps_the_frame_the_energy_and_the_handover_continuous() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H4, "lda");
        let state = solve(&base, Method::Hf("rhf"), [2, 2]);
        let (_, spec) = cutting_spec(&state);
        // Moving the last atom along the axis keeps every pi pair degenerate.
        let step = 1e-5;
        let moved = displaced(&base, 3, 2, step);
        let after = solve(&moved, Method::Hf("rhf"), [2, 2]);
        let jump =
            (sequence_energy(&moved, &after, &spec) - sequence_energy(&base, &state, &spec)).abs();
        println!("fixed-sequence energy change over a {step:e} Bohr axial step: {jump:.3e} Ha");
        assert!(jump < 2e-5, "{jump:.3e} Ha over a {step:e} Bohr axial step");

        let old_model = state.active_model(&base, &spec).unwrap();
        let new_model = after.active_model(&moved, &spec).unwrap();
        let old_problem = old_model
            .problem(&ctx, "jordan_wigner", true, false, true)
            .unwrap();
        let new_problem = new_model
            .problem(&ctx, "jordan_wigner", true, false, true)
            .unwrap();
        let count = old_problem.pool.len().min(3);
        let indices: Vec<usize> = (0..count).collect();
        let angles = vec![0.2; count];
        let captured = transfer::capture(
            Frame {
                integrals: &base,
                reference: &state,
                model: &old_model,
                problem: &old_problem,
            },
            &indices,
            &angles,
        )
        .unwrap();
        let seed = transfer::carry(
            &captured,
            Frame {
                integrals: &moved,
                reference: &after,
                model: &new_model,
                problem: &new_problem,
            },
            0.99,
        )
        .unwrap();
        println!(
            "handover confidence {:.8}: {}",
            seed.confidence, seed.reason
        );
        assert!(seed.accepted, "{}", seed.reason);
        assert!(seed.confidence > 1. - 1e-4, "{}", seed.confidence);
    }
    assert_eq!(ctx.memory_used(), 0);
}

/// The unrestricted frame: the natural orbitals of `P_a + P_b` have one exact zero-occupation
/// cluster (the empty block, which no symmetry protects) beside the symmetric pi pairs. An
/// arbitrary orthogonal rotation of that whole cluster leaves the frame and a fixed-sequence
/// energy unchanged.
#[test]
fn the_unrestricted_empty_block_has_a_deterministic_frame() {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    {
        let base = integrals(&ctx, &H4, "lda");
        let mut state = solve(&base, Method::Hf("uhf"), [2, 1]);
        let m = state.model_orbitals.ncols();
        let labels = state.natural_occupations.clone();
        let domains = [
            (0..1).collect::<Vec<_>>(),
            (1..2).collect(),
            (2..m).collect(),
        ];
        let found = clusters(&labels, &domains).unwrap();
        let empty = found
            .iter()
            .max_by_key(|c| c.len())
            .expect("the empty block is a cluster")
            .clone();
        assert!(empty.len() > 2, "{found:?}");
        let spec = format!(r#"{{"orbitals": [0, 1, {}]}}"#, empty[0]);
        let reference = state.model_orbitals.clone();
        let before = sequence_energy(&base, &state, &spec);
        // A fixed orthogonal matrix (the Q factor of a deterministic matrix) over the cluster.
        let q = DMatrix::from_fn(empty.len(), empty.len(), |i, j| {
            ((i * 5 + j * 11 + 1) as f64 * 0.37).sin() + if i == j { 2. } else { 0. }
        })
        .qr()
        .q();
        let columns = state.model_orbitals.clone();
        for r in 0..m {
            for (c, &target) in empty.iter().enumerate() {
                state.model_orbitals[(r, target)] = (0..empty.len())
                    .map(|k| columns[(r, empty[k])] * q[(k, c)])
                    .sum();
            }
        }
        let changed = fix_cluster_gauge(
            &mut state.model_orbitals,
            &labels,
            &domains,
            &mut lab_moments(&base),
        )
        .unwrap();
        assert!(changed >= 1);
        assert!(
            (&state.model_orbitals - &reference).norm() < 1e-9,
            "frame moved by {:.3e}",
            (&state.model_orbitals - &reference).norm()
        );
        let after = sequence_energy(&base, &state, &spec);
        assert!((after - before).abs() < 1e-9, "{before} {after}");
    }
    assert_eq!(ctx.memory_used(), 0);
}
