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

fn to_i8_ops(ops: &'static [([i32; 9], [i32; 3])]) -> Vec<([[i8; 3]; 3], [i8; 3])> {
    ops.iter()
        .map(|(r, t)| {
            (
                [
                    [r[0] as i8, r[1] as i8, r[2] as i8],
                    [r[3] as i8, r[4] as i8, r[5] as i8],
                    [r[6] as i8, r[7] as i8, r[8] as i8],
                ],
                [t[0] as i8, t[1] as i8, t[2] as i8],
            )
        })
        .collect()
}

#[test]
fn test_catalog_snf_invariant_factors_divide_24() {
    for g in iter_groups() {
        let rots: Vec<[[i8; 3]; 3]> = g
            .operations
            .iter()
            .map(|(r, _)| [
                [r[0] as i8, r[1] as i8, r[2] as i8],
                [r[3] as i8, r[4] as i8, r[5] as i8],
                [r[6] as i8, r[7] as i8, r[8] as i8],
            ])
            .collect();

        let m_l: [[i32; 3]; 3] = match g.centering {
            'P' => [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'A' => [[1, 0, 0], [0, 1, 1], [0, -1, 1]],
            'B' => [[1, 0, 1], [0, 1, 0], [-1, 0, 1]],
            'C' => [[1, 1, 0], [-1, 1, 0], [0, 0, 1]],
            'I' => [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            'F' => [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
            'R' => [[2, 1, 1], [-1, 1, 1], [-1, -2, 1]],
            _ => panic!("unknown centering"),
        };

        let factors = mandacaru_core::periodic::space_group::shift::invariant_factors_for_group(
            &rots,
            &m_l,
        );

        for &factor in &factors {
            assert!(
                factor > 0 && 24 % factor == 0,
                "SG {} ({}) invariant factor {} does not divide 24",
                g.number,
                g.symbol,
                factor
            );
        }
    }
}

#[test]
fn test_catalog_affine_inequivalence_and_enantiomorphic_pairs() {
    let groups: Vec<_> = iter_groups().collect();
    let expected_enantiomorphic = vec![
        (76, 78),
        (91, 95),
        (92, 96),
        (144, 145),
        (151, 153),
        (152, 154),
        (169, 170),
        (171, 172),
        (178, 179),
        (180, 181),
        (212, 213),
    ];

    let mut proper_equiv = Vec::new();
    let mut improper_equiv = Vec::new();

    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            let g1 = groups[i];
            let g2 = groups[j];

            if g1.crystal_system != g2.crystal_system
                || g1.point_group != g2.point_group
                || g1.centering != g2.centering
            {
                continue;
            }

            let ops1 = to_i8_ops(g1.operations);
            let ops2 = to_i8_ops(g2.operations);

            let is_proper = mandacaru_core::periodic::space_group::matching::are_groups_affine_equivalent(
                &ops1,
                &ops2,
                g1.crystal_system,
                g2.crystal_system,
                g1.centering,
                g2.centering,
                false,
            );
            if is_proper {
                proper_equiv.push((g1.number, g2.number));
            }

            let is_improper = mandacaru_core::periodic::space_group::matching::are_groups_affine_equivalent(
                &ops1,
                &ops2,
                g1.crystal_system,
                g2.crystal_system,
                g1.centering,
                g2.centering,
                true,
            );
            if is_improper {
                improper_equiv.push((g1.number, g2.number));
            }
        }
    }

    assert_eq!(
        proper_equiv,
        Vec::<(usize, usize)>::new(),
        "Expected 0 proper affine equivalent pairs, found {:?}",
        proper_equiv
    );
    assert_eq!(
        improper_equiv,
        expected_enantiomorphic,
        "Improper affine equivalent pairs mismatch"
    );
    assert_eq!(
        230 - improper_equiv.len(),
        219,
        "Expected 219 affine classes"
    );
}

