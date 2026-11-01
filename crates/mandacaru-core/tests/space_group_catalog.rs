use mandacaru_core::periodic::space_group::catalog::{centering_vectors, iter_groups, lookup};
use mandacaru_core::periodic::space_group::classify::{classify_point_group, CrystalSystem};
use nalgebra::Matrix3;
use std::collections::HashSet;

#[test]
fn test_catalog_counts_per_crystal_system() {
    let mut counts = [0usize; 7];
    let mut total = 0;

    for g in iter_groups() {
        total += 1;
        match g.crystal_system {
            CrystalSystem::Triclinic => counts[0] += 1,
            CrystalSystem::Monoclinic => counts[1] += 1,
            CrystalSystem::Orthorhombic => counts[2] += 1,
            CrystalSystem::Tetragonal => counts[3] += 1,
            CrystalSystem::Trigonal => counts[4] += 1,
            CrystalSystem::Hexagonal => counts[5] += 1,
            CrystalSystem::Cubic => counts[6] += 1,
        }
    }

    assert_eq!(total, 230);
    // Counts: 2, 13, 59, 68, 25, 27, 36
    assert_eq!(counts[0], 2);
    assert_eq!(counts[1], 13);
    assert_eq!(counts[2], 59);
    assert_eq!(counts[3], 68);
    assert_eq!(counts[4], 25);
    assert_eq!(counts[5], 27);
    assert_eq!(counts[6], 36);
}

#[test]
fn test_catalog_numbers_and_lookup() {
    for n in 1..=230 {
        let g = lookup(n).unwrap_or_else(|| panic!("lookup({}) returned None", n));
        assert_eq!(g.number, n);
        assert!(!g.symbol.is_empty());
        assert!(!g.point_group.is_empty());
    }
    assert!(lookup(0).is_none());
    assert!(lookup(231).is_none());
}

#[test]
fn test_catalog_point_group_consistency() {
    for g in iter_groups() {
        let rots: Vec<Matrix3<i32>> = g
            .operations
            .iter()
            .map(|(r, _)| Matrix3::from_row_slice(r))
            .collect();
        let pg_info = classify_point_group(&rots)
            .unwrap_or_else(|e| panic!("SG {} point group failed: {}", g.number, e));
        assert_eq!(
            pg_info.symbol, g.point_group,
            "SG {} ({}) point group mismatch: expected {}, got {}",
            g.number, g.symbol, g.point_group, pg_info.symbol
        );
        assert_eq!(
            pg_info.crystal_system, g.crystal_system,
            "SG {} ({}) crystal system mismatch",
            g.number, g.symbol
        );
    }
}

#[test]
fn test_catalog_closure_and_centering() {
    for g in iter_groups() {
        let c_trans = centering_vectors(g.centering);

        // Expand full group operations: (W, (t + c) mod 24)
        let mut full_group = Vec::new();
        let mut full_set = HashSet::new();

        for (r, t) in g.operations {
            let w = Matrix3::from_row_slice(r);
            for c in c_trans {
                let t_full = [
                    (t[0] + c[0]).rem_euclid(24),
                    (t[1] + c[1]).rem_euclid(24),
                    (t[2] + c[2]).rem_euclid(24),
                ];
                let key = (
                    [w[(0, 0)], w[(0, 1)], w[(0, 2)],
                     w[(1, 0)], w[(1, 1)], w[(1, 2)],
                     w[(2, 0)], w[(2, 1)], w[(2, 2)]],
                    t_full,
                );
                if full_set.insert(key) {
                    full_group.push((w, t_full));
                }
            }
        }

        // 1. Group size must equal |point_group| * |centering|
        let expected_size = g.operations.len() * c_trans.len();
        assert_eq!(
            full_group.len(),
            expected_size,
            "SG {} ({}) size mismatch",
            g.number,
            g.symbol
        );

        // 2. Identity operation must be present
        let id = Matrix3::identity();
        assert!(
            full_group.iter().any(|(w, t)| *w == id && *t == [0, 0, 0]),
            "SG {} ({}) lacks identity",
            g.number,
            g.symbol
        );

        // 3. Multiplication closure with exact 1/24 integer arithmetic
        for (w1, t1) in &full_group {
            for (w2, t2) in &full_group {
                let w_prod = w1 * w2;
                let t_prod = [
                    (w1[(0, 0)] * t2[0] + w1[(0, 1)] * t2[1] + w1[(0, 2)] * t2[2] + t1[0]).rem_euclid(24),
                    (w1[(1, 0)] * t2[0] + w1[(1, 1)] * t2[1] + w1[(1, 2)] * t2[2] + t1[1]).rem_euclid(24),
                    (w1[(2, 0)] * t2[0] + w1[(2, 1)] * t2[1] + w1[(2, 2)] * t2[2] + t1[2]).rem_euclid(24),
                ];
                let key = (
                    [w_prod[(0, 0)], w_prod[(0, 1)], w_prod[(0, 2)],
                     w_prod[(1, 0)], w_prod[(1, 1)], w_prod[(1, 2)],
                     w_prod[(2, 0)], w_prod[(2, 1)], w_prod[(2, 2)]],
                    t_prod,
                );
                assert!(
                    full_set.contains(&key),
                    "SG {} ({}) not closed under multiplication",
                    g.number,
                    g.symbol
                );
            }
        }
    }
}
