//! CF-6, the symmetrizer on its own (no SCF): the group average of a per-atom vector and of a
//! rank-2 tensor over the operations a reduced k mesh was built with, and the checks that refuse
//! operations which do not reproduce the snapshot's node permutations, atoms or lattice.
use mandacaru_core::{periodic::point_group::PointGroup, Context};
use nalgebra::{Matrix3, Vector3};
use serde_json::{json, Value};

const N: usize = 4;

fn cubic() -> Matrix3<f64> {
    Matrix3::identity() * 5.0
}

/// A body-centred pair in a cubic cell.
fn centers() -> Vec<[f64; 3]> {
    vec![[0., 0., 0.], [2.5, 2.5, 2.5]]
}

/// The node permutation of `x -> W x` on an `N^3` grid, flat C order (as the SCF carries it).
fn permutation(w: [[i64; 3]; 3]) -> Vec<usize> {
    let mut out = Vec::new();
    for a in 0..N as i64 {
        for b in 0..N as i64 {
            for c in 0..N as i64 {
                let n = [a, b, c];
                let image: Vec<i64> = (0..3)
                    .map(|i| {
                        (0..3)
                            .map(|j| w[i][j] * n[j])
                            .sum::<i64>()
                            .rem_euclid(N as i64)
                    })
                    .collect();
                out.push(((image[0] * N as i64 + image[1]) * N as i64 + image[2]) as usize);
            }
        }
    }
    out
}

const IDENTITY: [[i64; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
/// A half turn about z (with the identity a closed group, so the average is a projector).
const C2: [[i64; 3]; 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]];

fn flat(w: [[i64; 3]; 3]) -> Vec<i64> {
    w.iter().flatten().copied().collect()
}

fn operations() -> Value {
    json!({
        "rotations": [flat(IDENTITY), flat(C2)],
        "translations": [[0., 0., 0.], [0., 0., 0.]],
        "atom_maps": [[0, 1], [0, 1]],
    })
}

fn parse(operations: &Value) -> mandacaru_core::Result<PointGroup> {
    PointGroup::parse(
        &Context::new(Some(1), 1 << 20).unwrap(),
        operations,
        &cubic(),
        &centers(),
        [N; 3],
        &[permutation(IDENTITY), permutation(C2)],
    )
}

#[test]
fn the_average_of_a_vector_is_invariant_and_idempotent() {
    let group = parse(&operations()).unwrap();
    assert_eq!(group.len(), 2);
    let mut f = [[1.0, 0.4, 0.3], [-0.4, 0.7, -0.3]];
    group.vectors(&mut f);
    // The half turn flips x and y: only the component along the axis survives on an atom it maps to itself.
    assert_eq!(f, [[0.0, 0.0, 0.3], [0.0, 0.0, -0.3]]);
    let once = f;
    group.vectors(&mut f);
    assert_eq!(f, once, "the average is a projector");
    // The result is covariant: F[s(A)] = R_s F[A].
    for (r, map) in group.rotations().iter().zip(group.atom_maps()) {
        for a in 0..2 {
            let rotated: [f64; 3] = (r * Vector3::from(once[a])).into();
            for i in 0..3 {
                assert!((once[map[a]][i] - rotated[i]).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn the_average_of_a_tensor_is_the_group_average() {
    let group = parse(&operations()).unwrap();
    let mut g = [[1., 0.2, 0.7], [0.5, 3., 0.1], [0.4, 0.3, 7.]];
    group.tensor(&mut g);
    // The half turn about z keeps the xx, yy, zz and the xy block and flips every entry that has one z index.
    assert_eq!(g, [[1., 0.2, 0.], [0.5, 3., 0.], [0., 0., 7.]]);
    let before = g;
    group.tensor(&mut g);
    assert_eq!(g, before);
}

#[test]
fn a_wrong_atom_map_changes_the_average() {
    let group = parse(&operations()).unwrap();
    let swapped = PointGroup::from_parts(group.rotations().to_vec(), vec![vec![1, 0]; 2]);
    let mut right = [[0., 0., 1.], [0., 0., 0.]];
    let mut wrong = right;
    group.vectors(&mut right);
    swapped.vectors(&mut wrong);
    assert_eq!(right, [[0., 0., 1.], [0., 0., 0.]]);
    assert_eq!(wrong, [[0., 0., 0.], [0., 0., 1.]]);
}

#[test]
fn operations_that_do_not_belong_to_the_snapshot_are_refused() {
    let refused = |edit: &dyn Fn(&mut Value)| {
        let mut ops = operations();
        edit(&mut ops);
        parse(&ops).is_err()
    };
    // A fractional rotation, a rotation that is not orthogonal in the cell, an atom map that is
    // not a permutation, one that sends an atom where it does not go, a translation the grid
    // cannot carry, and the wrong count.
    assert!(refused(&|o| o["rotations"][1][0] = json!(0.5)));
    assert!(refused(
        &|o| o["rotations"][1] = json!([1, 1, 0, 0, 1, 0, 0, 0, 1])
    ));
    assert!(refused(&|o| o["atom_maps"][1] = json!([0, 0])));
    assert!(refused(&|o| o["atom_maps"][1] = json!([1, 0])));
    assert!(refused(&|o| o["translations"][1] = json!([0.3, 0., 0.])));
    assert!(refused(&|o| {
        o["rotations"].as_array_mut().unwrap().pop();
    }));
    // A node permutation that is not the one the operation generates.
    let mut wrong = permutation(C2);
    wrong.swap(1, 2);
    assert!(PointGroup::parse(
        &Context::new(Some(1), 1 << 20).unwrap(),
        &operations(),
        &cubic(),
        &centers(),
        [N; 3],
        &[permutation(IDENTITY), wrong]
    )
    .is_err());
    // The unedited block parses.
    assert!(parse(&operations()).is_ok());
}
