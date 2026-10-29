use mandacaru_core::math::symmetry::operations;
use nalgebra::{Matrix3, Vector3};

#[test]
fn generated_point_groups_have_expected_orders_and_are_closed_under_composition() {
    let cases = vec![
        (vec![[0., 0., 0.]], vec![2], 120),
        (vec![[0., 0., -1.], [0., 0., 1.]], vec![1, 1], 32),
        (vec![[0., 0., -1.], [0., 0., 1.]], vec![3, 1], 16),
        (
            vec![[0., 0., 0.], [0., 0.8, 0.6], [0., -0.8, 0.6]],
            vec![8, 1, 1],
            4,
        ),
        (
            vec![
                [0., 0., 0.],
                [1., 1., 1.],
                [1., -1., -1.],
                [-1., 1., -1.],
                [-1., -1., 1.],
            ],
            vec![6, 1, 1, 1, 1],
            24,
        ),
    ];
    for (points, labels, expected) in cases {
        let (center, group) = operations(&points, &labels, 1e-6).unwrap();
        assert_eq!(group.len(), expected);
        for a in &group {
            assert!((a.transpose() * a - Matrix3::identity()).norm() < 1e-10);
            for b in &group {
                assert!(group.iter().any(|c| (a * b - c).norm() < 1e-8));
            }
        }
        let q = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1., 2., 3.)),
            0.7,
        )
        .into_inner();
        let shift = Vector3::new(3., -2., 1.);
        let moved: Vec<[f64; 3]> = points
            .iter()
            .map(|p| (q * Vector3::from(*p) + shift).into())
            .collect();
        let (new_center, rotated) = operations(&moved, &labels, 1e-6).unwrap();
        assert!((new_center - q * center - shift).norm() < 1e-10);
        // Atomic spherical representatives may choose a different orientation.
        if points.len() > 2 {
            assert_eq!(rotated.len(), group.len());
            for r in &rotated {
                assert!(group
                    .iter()
                    .any(|g| (r - q * g * q.transpose()).norm() < 1e-8));
            }
        }
    }
}
