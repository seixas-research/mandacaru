use mandacaru_core::periodic::space_group::classify::{
    classify_point_group, CrystalSystem,
};
use nalgebra::Matrix3;

#[test]
fn test_classify_all_trivial_and_cubic() {
    // 1 (trivial)
    let rots_1 = vec![Matrix3::identity()];
    let pg_1 = classify_point_group(&rots_1).unwrap();
    assert_eq!(pg_1.symbol, "1");
    assert_eq!(pg_1.crystal_system, CrystalSystem::Triclinic);
    assert_eq!(pg_1.order, 1);

    // -1 (inversion)
    let rots_inv = vec![Matrix3::identity(), -Matrix3::identity()];
    let pg_inv = classify_point_group(&rots_inv).unwrap();
    assert_eq!(pg_inv.symbol, "-1");
    assert_eq!(pg_inv.crystal_system, CrystalSystem::Triclinic);
    assert_eq!(pg_inv.order, 2);

    // m-3m (full cubic, 48 operations)
    let a = 3.5;
    let sc = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    let rots_48 = mandacaru_core::math::lattice::input_lattice_rotations(&sc, 1e-5).unwrap();
    assert_eq!(rots_48.len(), 48);
    let pg_48 = classify_point_group(&rots_48).unwrap();
    assert_eq!(pg_48.symbol, "m-3m");
    assert_eq!(pg_48.crystal_system, CrystalSystem::Cubic);
    assert_eq!(pg_48.laue_class, "m-3m");
    assert_eq!(pg_48.order, 48);
}

#[test]
fn test_classify_invalid() {
    // Arbitrary non-point-group set
    let bad = vec![
        Matrix3::identity(),
        Matrix3::new(1, 1, 0, 0, 1, 0, 0, 0, 1),
    ];
    assert!(classify_point_group(&bad).is_err());
}
