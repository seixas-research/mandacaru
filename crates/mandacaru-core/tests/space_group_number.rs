use mandacaru_core::periodic::space_group::{find, identify, Cell};
use nalgebra::Matrix3;

#[test]
fn test_identify_square_slab() {
    let a = 2.5;
    let vac = 12.0;
    let lat = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, vac);
    let frac = [[0.0, 0.0, 0.0]];
    let labels = [1];
    let cell = Cell {
        lattice: lat,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    let id = identify(&ops, &cell).unwrap();
    assert_eq!(id.number, 123);
    assert_eq!(id.symbol, "P4/mmm");
}

#[test]
fn test_identify_hexagonal_slab() {
    let a = 2.5;
    let vac = 12.0;
    let lat = Matrix3::new(
        a, 0.0, 0.0,
        -a / 2.0, a * 3.0f64.sqrt() / 2.0, 0.0,
        0.0, 0.0, vac,
    );
    let frac = [[0.0, 0.0, 0.0]];
    let labels = [1];
    let cell = Cell {
        lattice: lat,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    let id = identify(&ops, &cell).unwrap();
    assert_eq!(id.number, 191);
    assert_eq!(id.symbol, "P6/mmm");
}

#[test]
fn test_identify_cu_fcc_primitive() {
    let a = 3.61;
    let fcc: Matrix3<f64> = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    let frac = [[0.0, 0.0, 0.0]];
    let labels = [29];
    let cell = Cell {
        lattice: fcc,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    let id = identify(&ops, &cell).unwrap();
    assert_eq!(id.number, 225);
    assert_eq!(id.symbol, "Fm-3m");
}

#[test]
fn test_identify_si_diamond_primitive() {
    let a = 5.43;
    let fcc: Matrix3<f64> = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    let frac = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]];
    let labels = [14, 14];
    let cell = Cell {
        lattice: fcc,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    let id = identify(&ops, &cell).unwrap();
    assert_eq!(id.number, 227);
    assert_eq!(id.symbol, "Fd-3m");
}

#[test]
fn test_identify_mg_hcp() {
    let a = 3.21;
    let c = 5.21;
    let hex = Matrix3::new(
        a, 0.0, 0.0,
        -a / 2.0, a * 3.0f64.sqrt() / 2.0, 0.0,
        0.0, 0.0, c,
    );
    let frac = [[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]];
    let labels = [12, 12];
    let cell = Cell {
        lattice: hex,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    let id = identify(&ops, &cell).unwrap();
    assert_eq!(id.number, 194);
    assert_eq!(id.symbol, "P6_3/mmc");
}

#[test]
fn test_identify_aln_wurtzite() {
    let a = 3.11;
    let c = 4.98;
    let u = 0.382;
    let hex = Matrix3::new(
        a, 0.0, 0.0,
        -a / 2.0, a * 3.0f64.sqrt() / 2.0, 0.0,
        0.0, 0.0, c,
    );
    let frac = [
        [1.0 / 3.0, 2.0 / 3.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.5],
        [1.0 / 3.0, 2.0 / 3.0, u],
        [2.0 / 3.0, 1.0 / 3.0, 0.5 + u],
    ];
    let labels = [13, 13, 7, 7];
    let cell = Cell {
        lattice: hex,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    let id = identify(&ops, &cell).unwrap();
    assert_eq!(id.number, 186);
    assert_eq!(id.symbol, "P6_3mc");
}

fn standard_cell_for_group(number: usize) -> Matrix3<f64> {
    let g = mandacaru_core::periodic::space_group::catalog::lookup(number).unwrap();
    match g.crystal_system {
        mandacaru_core::periodic::space_group::CrystalSystem::Triclinic => {
            Matrix3::new(4.1, 0.2, 0.3, 0.4, 4.8, 0.5, 0.6, 0.7, 5.5)
        }
        mandacaru_core::periodic::space_group::CrystalSystem::Monoclinic => {
            Matrix3::new(4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 6.0)
        }
        mandacaru_core::periodic::space_group::CrystalSystem::Orthorhombic => {
            Matrix3::new(4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0)
        }
        mandacaru_core::periodic::space_group::CrystalSystem::Tetragonal => {
            Matrix3::new(4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0)
        }
        mandacaru_core::periodic::space_group::CrystalSystem::Trigonal
        | mandacaru_core::periodic::space_group::CrystalSystem::Hexagonal => {
            let a = 4.0;
            let c = 6.0;
            Matrix3::new(
                a, 0.0, 0.0,
                -a / 2.0, a * 3.0f64.sqrt() / 2.0, 0.0,
                0.0, 0.0, c,
            )
        }
        mandacaru_core::periodic::space_group::CrystalSystem::Cubic => {
            Matrix3::new(5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0)
        }
    }
}

fn generate_atoms_from_catalog(
    number: usize,
    cell: Matrix3<f64>,
    shift: [f64; 3],
    transform_c: Option<Matrix3<i32>>,
) -> (Matrix3<f64>, Vec<[f64; 3]>, Vec<u32>) {
    let g = mandacaru_core::periodic::space_group::catalog::lookup(number)
        .unwrap_or_else(|| panic!("lookup({}) failed", number));
    let c_trans = mandacaru_core::periodic::space_group::catalog::centering_vectors(g.centering);

    let base_positions = [
        [0.1234, 0.2789, 0.3912],
        [0.4111, 0.1923, 0.6345],
    ];
    let base_labels = [1u32, 2u32];

    let mut frac_vec: Vec<[f64; 3]> = Vec::new();
    let mut labels_vec = Vec::new();

    let c_inv_t = transform_c.map(|c| {
        let c_f64 = Matrix3::new(
            c[(0, 0)] as f64, c[(0, 1)] as f64, c[(0, 2)] as f64,
            c[(1, 0)] as f64, c[(1, 1)] as f64, c[(1, 2)] as f64,
            c[(2, 0)] as f64, c[(2, 1)] as f64, c[(2, 2)] as f64,
        );
        c_f64.try_inverse().expect("transform_c must be invertible").transpose()
    });

    for (k, &pos0) in base_positions.iter().enumerate() {
        let label = base_labels[k];
        for (r, t) in g.operations {
            let w = Matrix3::from_row_slice(r);
            for c in c_trans {
                let tx = (t[0] + c[0]) as f64 / 24.0;
                let ty = (t[1] + c[1]) as f64 / 24.0;
                let tz = (t[2] + c[2]) as f64 / 24.0;

                let x = w[(0, 0)] as f64 * pos0[0] + w[(0, 1)] as f64 * pos0[1] + w[(0, 2)] as f64 * pos0[2] + tx + shift[0];
                let y = w[(1, 0)] as f64 * pos0[0] + w[(1, 1)] as f64 * pos0[1] + w[(1, 2)] as f64 * pos0[2] + ty + shift[1];
                let z = w[(2, 0)] as f64 * pos0[0] + w[(2, 1)] as f64 * pos0[1] + w[(2, 2)] as f64 * pos0[2] + tz + shift[2];

                let p_coord = if let Some(inv_t) = &c_inv_t {
                    let v = inv_t * nalgebra::Vector3::new(x, y, z);
                    [v[0].rem_euclid(1.0), v[1].rem_euclid(1.0), v[2].rem_euclid(1.0)]
                } else {
                    [x.rem_euclid(1.0), y.rem_euclid(1.0), z.rem_euclid(1.0)]
                };

                // Deduplicate within tolerance 1e-4
                let already_present = frac_vec.iter().zip(labels_vec.iter()).any(|(p, &l)| {
                    if l != label {
                        return false;
                    }
                    let dx = (p[0] - p_coord[0]).rem_euclid(1.0);
                    let dy = (p[1] - p_coord[1]).rem_euclid(1.0);
                    let dz = (p[2] - p_coord[2]).rem_euclid(1.0);
                    let dist_x = if dx > 0.5 { 1.0 - dx } else { dx };
                    let dist_y = if dy > 0.5 { 1.0 - dy } else { dy };
                    let dist_z = if dz > 0.5 { 1.0 - dz } else { dz };
                    dist_x < 1e-4 && dist_y < 1e-4 && dist_z < 1e-4
                });

                if !already_present {
                    frac_vec.push(p_coord);
                    labels_vec.push(label);
                }
            }
        }
    }

    let lat = if let Some(c) = transform_c {
        let c_f64 = Matrix3::new(
            c[(0, 0)] as f64, c[(0, 1)] as f64, c[(0, 2)] as f64,
            c[(1, 0)] as f64, c[(1, 1)] as f64, c[(1, 2)] as f64,
            c[(2, 0)] as f64, c[(2, 1)] as f64, c[(2, 2)] as f64,
        );
        c_f64 * cell
    } else {
        cell
    };

    (lat, frac_vec, labels_vec)
}

fn build_from_catalog(
    number: usize,
    cell: Matrix3<f64>,
    shift: [f64; 3],
    transform_c: Option<Matrix3<i32>>,
) -> mandacaru_core::periodic::space_group::Identity {
    let (lat, frac_vec, labels_vec) = generate_atoms_from_catalog(number, cell, shift, transform_c);
    let cell_obj = Cell {
        lattice: lat,
        fractional: &frac_vec,
        labels: &labels_vec,
    };

    let ops = find(&cell_obj, 1e-5).unwrap_or_else(|e| panic!("find failed for SG {}: {}", number, e));
    identify(&ops, &cell_obj).unwrap_or_else(|e| panic!("identify failed for SG {}: {}", number, e))
}

#[test]
fn test_hard_case_a_enantiomorphic_pairs_and_left_handed_cells() {
    let enantiomorphic_pairs = [
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

    for &(num_a, num_b) in &enantiomorphic_pairs {
        let lat = standard_cell_for_group(num_a);

        // 1. Exact numbers
        let id_a = build_from_catalog(num_a, lat, [0.0; 3], None);
        assert_eq!(id_a.number, num_a, "Expected SG {}, got {}", num_a, id_a.number);

        let id_b = build_from_catalog(num_b, lat, [0.0; 3], None);
        assert_eq!(id_b.number, num_b, "Expected SG {}, got {}", num_b, id_b.number);

        // 2. Reflecting atomic coordinates (x -> -x) swaps chirality to partner
        let (lat_a, frac_a, labels_a) = generate_atoms_from_catalog(num_a, lat, [0.0; 3], None);
        let frac_refl_a: Vec<[f64; 3]> = frac_a
            .iter()
            .map(|p| [(-p[0]).rem_euclid(1.0), (-p[1]).rem_euclid(1.0), (-p[2]).rem_euclid(1.0)])
            .collect();
        let cell_refl_a = Cell {
            lattice: lat_a,
            fractional: &frac_refl_a,
            labels: &labels_a,
        };
        let ops_refl_a = find(&cell_refl_a, 1e-5).unwrap();
        let id_refl_a = identify(&ops_refl_a, &cell_refl_a).unwrap();
        assert_eq!(
            id_refl_a.number, num_b,
            "Reflecting SG {} coordinates did not swap to partner {}",
            num_a, num_b
        );

        let (lat_b, frac_b, labels_b) = generate_atoms_from_catalog(num_b, lat, [0.0; 3], None);
        let frac_refl_b: Vec<[f64; 3]> = frac_b
            .iter()
            .map(|p| [(-p[0]).rem_euclid(1.0), (-p[1]).rem_euclid(1.0), (-p[2]).rem_euclid(1.0)])
            .collect();
        let cell_refl_b = Cell {
            lattice: lat_b,
            fractional: &frac_refl_b,
            labels: &labels_b,
        };
        let ops_refl_b = find(&cell_refl_b, 1e-5).unwrap();
        let id_refl_b = identify(&ops_refl_b, &cell_refl_b).unwrap();
        assert_eq!(
            id_refl_b.number, num_a,
            "Reflecting SG {} coordinates did not swap to partner {}",
            num_b, num_a
        );

        // 3. Negating lattice vectors (left-handed cell describing same Cartesian crystal) preserves chiral SG number
        let cell_left_a = Cell {
            lattice: -lat_a,
            fractional: &frac_refl_a,
            labels: &labels_a,
        };
        let ops_left_a = find(&cell_left_a, 1e-5).unwrap();
        let id_left_a = identify(&ops_left_a, &cell_left_a).unwrap();
        assert_eq!(
            id_left_a.number, num_a,
            "Left-handed cell of SG {} did not preserve chiral number",
            num_a
        );

        let cell_left_b = Cell {
            lattice: -lat_b,
            fractional: &frac_refl_b,
            labels: &labels_b,
        };
        let ops_left_b = find(&cell_left_b, 1e-5).unwrap();
        let id_left_b = identify(&ops_left_b, &cell_left_b).unwrap();
        assert_eq!(
            id_left_b.number, num_b,
            "Left-handed cell of SG {} did not preserve chiral number",
            num_b
        );
    }
}

#[test]
fn test_hard_case_b_two_origin_choices() {
    // Fd-3m (#227) shifted by [0.125, 0.125, 0.125]
    let lat227 = standard_cell_for_group(227);
    let id227_shift = build_from_catalog(227, lat227, [0.125, 0.125, 0.125], None);
    assert_eq!(id227_shift.number, 227);
    assert_eq!(id227_shift.symbol, "Fd-3m");

    let id227_zero = build_from_catalog(227, lat227, [0.0, 0.0, 0.0], None);
    assert_eq!(id227_zero.number, 227);
    assert_eq!(id227_zero.symbol, "Fd-3m");

    // Pn-3n (#222) shifted by [0.25, 0.25, 0.25]
    let lat222 = standard_cell_for_group(222);
    let id222 = build_from_catalog(222, lat222, [0.25, 0.25, 0.25], None);
    assert_eq!(id222.number, 222);
    assert_eq!(id222.symbol, "Pn-3n");

    // P4/nmm (#129) shifted by [0.25, -0.25, 0.0]
    let lat129 = standard_cell_for_group(129);
    let id129 = build_from_catalog(129, lat129, [0.25, -0.25, 0.0], None);
    assert_eq!(id129.number, 129);
    assert_eq!(id129.symbol, "P4/nmm");

    // Arbitrary continuous origin shift on Pnma (#62)
    let lat62 = standard_cell_for_group(62);
    let id62 = build_from_catalog(62, lat62, [0.173, 0.412, 0.889], None);
    assert_eq!(id62.number, 62);
    assert_eq!(id62.symbol, "Pnma");
}

#[test]
fn test_hard_case_c_orthorhombic_six_axis_labelings() {
    let proper_perms = [
        Matrix3::new(1, 0, 0, 0, 1, 0, 0, 0, 1),   // abc
        Matrix3::new(0, 0, 1, 1, 0, 0, 0, 1, 0),   // cab
        Matrix3::new(0, 1, 0, 0, 0, 1, 1, 0, 0),   // bca
        Matrix3::new(0, 1, 0, 1, 0, 0, 0, 0, -1),  // ba-c
        Matrix3::new(1, 0, 0, 0, 0, -1, 0, 1, 0),  // a-cb
        Matrix3::new(0, 0, -1, 0, 1, 0, 1, 0, 0),  // -cba
    ];

    let cases = [
        (62, "Pnma"),
        (64, "Cmce"),
        (74, "Imma"),
        (70, "Fddd"),
    ];

    for &(num, expected_symbol) in &cases {
        let lat = standard_cell_for_group(num);
        for &perm in &proper_perms {
            let id = build_from_catalog(num, lat, [0.0; 3], Some(perm));
            assert_eq!(
                id.number, num,
                "SG {} failed under axis permutation {:?}",
                num, perm
            );
            assert_eq!(
                id.symbol, expected_symbol,
                "SG {} symbol mismatch under axis permutation {:?}",
                num, perm
            );
        }
    }
}

#[test]
fn test_hard_case_d_monoclinic_cell_choices_and_unique_axes() {
    let mono_transforms = [
        Matrix3::identity(),
        Matrix3::new(1, 0, 1, 0, 1, 0, 0, 0, 1),
        Matrix3::new(1, 0, 0, 0, 1, 0, 1, 0, 1),
        Matrix3::new(0, 1, 0, 1, 0, 0, 0, 0, -1), // unique axis along a
        Matrix3::new(-1, 0, 0, 0, 0, 1, 0, 1, 0), // unique axis along c
    ];

    let cases = [(14, "P2_1/c"), (15, "C2/c")];

    for &(num, expected_symbol) in &cases {
        let lat = standard_cell_for_group(num);
        for &t in &mono_transforms {
            let id = build_from_catalog(num, lat, [0.0; 3], Some(t));
            assert_eq!(
                id.number, num,
                "Monoclinic SG {} failed under transform {:?}",
                num, t
            );
            assert_eq!(id.symbol, expected_symbol);
        }
    }
}

#[test]
fn test_hard_case_e_rhombohedral_obverse_reverse_and_primitive() {
    let r_groups = [146, 148, 155, 160, 161, 166, 167];

    for &num in &r_groups {
        let hex_lat = standard_cell_for_group(num);

        // 1. Hexagonal obverse setting
        let id_obv = build_from_catalog(num, hex_lat, [0.0; 3], None);
        assert_eq!(id_obv.number, num, "SG {} failed in hexagonal obverse", num);

        // 2. Hexagonal reverse setting (a -> -a, b -> -b)
        let rev_c = Matrix3::new(-1, 0, 0, 0, -1, 0, 0, 0, 1);
        let id_rev = build_from_catalog(num, hex_lat, [0.0; 3], Some(rev_c));
        assert_eq!(id_rev.number, num, "SG {} failed in hexagonal reverse", num);

        // 3. Primitive rhombohedral cell
        let (lat_hex, frac_hex, labels_hex) = generate_atoms_from_catalog(num, hex_lat, [0.0; 3], None);
        // Transformation matrix from primitive to hexagonal coordinates: M_prim
        let m_prim = Matrix3::new(1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0);
        let m_prim_inv_t = m_prim.try_inverse().unwrap().transpose();
        let lat_prim = m_prim_inv_t * lat_hex;

        let mut frac_prim: Vec<[f64; 3]> = Vec::new();
        let mut labels_prim = Vec::new();

        for (&p, &l) in frac_hex.iter().zip(labels_hex.iter()) {
            let v = m_prim * nalgebra::Vector3::new(p[0], p[1], p[2]);
            let p_wrapped = [v[0].rem_euclid(1.0), v[1].rem_euclid(1.0), v[2].rem_euclid(1.0)];

            let already = frac_prim.iter().zip(labels_prim.iter()).any(|(q, &ql)| {
                if ql != l { return false; }
                let dx = (q[0] - p_wrapped[0]).rem_euclid(1.0);
                let dy = (q[1] - p_wrapped[1]).rem_euclid(1.0);
                let dz = (q[2] - p_wrapped[2]).rem_euclid(1.0);
                let dist_x = if dx > 0.5 { 1.0 - dx } else { dx };
                let dist_y = if dy > 0.5 { 1.0 - dy } else { dy };
                let dist_z = if dz > 0.5 { 1.0 - dz } else { dz };
                dist_x < 1e-4 && dist_y < 1e-4 && dist_z < 1e-4
            });

            if !already {
                frac_prim.push(p_wrapped);
                labels_prim.push(l);
            }
        }

        let cell_prim = Cell {
            lattice: lat_prim,
            fractional: &frac_prim,
            labels: &labels_prim,
        };
        let ops_prim = find(&cell_prim, 1e-5).unwrap_or_else(|e| panic!("find failed for primitive SG {}: {}", num, e));
        let id_prim = identify(&ops_prim, &cell_prim).unwrap_or_else(|e| panic!("identify failed for primitive SG {}: {}", num, e));
        assert_eq!(id_prim.number, num, "Primitive cell of SG {} returned {}", num, id_prim.number);

        if num == 148 {
            assert_eq!(id_prim.symbol, "R-3");
        }
        if num == 166 {
            assert_eq!(id_prim.symbol, "R-3m");
        }
    }
}

#[test]
fn test_hard_case_g_h_i_metric_special_supercells_and_spellings() {
    // 1. Metric-cubic lattice 5.0 * I_3 for triclinic (#2), monoclinic (#14), tetragonal (#123)
    let cubic_lat = Matrix3::new(5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0);

    let id2 = build_from_catalog(2, cubic_lat, [0.0; 3], None);
    assert_eq!(id2.number, 2, "Metric-cubic lattice: expected SG 2, got {}", id2.number);

    let id14 = build_from_catalog(14, cubic_lat, [0.0; 3], None);
    assert_eq!(id14.number, 14, "Metric-cubic lattice: expected SG 14, got {}", id14.number);

    let id123 = build_from_catalog(123, cubic_lat, [0.0; 3], None);
    assert_eq!(id123.number, 123, "Metric-cubic lattice: expected SG 123, got {}", id123.number);

    // 2. Supercells 2x1x1, 2x2x2, and skewed cells for SG 123
    let tet_lat = standard_cell_for_group(123);
    let (lat123, frac123, labels123) = generate_atoms_from_catalog(123, tet_lat, [0.0; 3], None);

    // 2x1x1 supercell
    let mut lat_211 = lat123;
    lat_211[(0, 0)] *= 2.0;
    lat_211[(0, 1)] *= 2.0;
    lat_211[(0, 2)] *= 2.0;

    let mut frac_211 = Vec::new();
    let mut labels_211 = Vec::new();
    for (&p, &l) in frac123.iter().zip(labels123.iter()) {
        frac_211.push([0.5 * p[0], p[1], p[2]]);
        labels_211.push(l);
        frac_211.push([0.5 * p[0] + 0.5, p[1], p[2]]);
        labels_211.push(l);
    }
    let cell_211 = Cell {
        lattice: lat_211,
        fractional: &frac_211,
        labels: &labels_211,
    };
    let ops_211 = find(&cell_211, 1e-5).unwrap();
    let id_211 = identify(&ops_211, &cell_211).unwrap();
    assert_eq!(id_211.number, 123);
    assert_eq!(id_211.symbol, "P4/mmm");

    // 2x2x2 supercell
    let lat_222 = 2.0 * lat123;
    let mut frac_222 = Vec::new();
    let mut labels_222 = Vec::new();
    for (&p, &l) in frac123.iter().zip(labels123.iter()) {
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    frac_222.push([
                        0.5 * (p[0] + di as f64),
                        0.5 * (p[1] + dj as f64),
                        0.5 * (p[2] + dk as f64),
                    ]);
                    labels_222.push(l);
                }
            }
        }
    }
    let cell_222 = Cell {
        lattice: lat_222,
        fractional: &frac_222,
        labels: &labels_222,
    };
    let ops_222 = find(&cell_222, 1e-5).unwrap();
    let id_222 = identify(&ops_222, &cell_222).unwrap();
    assert_eq!(id_222.number, 123);
    assert_eq!(id_222.symbol, "P4/mmm");

    // Skewed cell
    let skew_mat = Matrix3::new(1, 1, 0, 0, 1, 0, 0, 0, 1);
    let id_skew = build_from_catalog(123, tet_lat, [0.0; 3], Some(skew_mat));
    assert_eq!(id_skew.number, 123);
    assert_eq!(id_skew.symbol, "P4/mmm");

    // 3. Modern spglib 2.x spellings
    let check_spelling = |num: usize, expected: &str| {
        let g = mandacaru_core::periodic::space_group::catalog::lookup(num).unwrap();
        assert_eq!(g.symbol, expected, "SG {} spelling mismatch", num);
    };
    check_spelling(39, "Aem2");
    check_spelling(41, "Aea2");
    check_spelling(64, "Cmce");
    check_spelling(67, "Cmme");
    check_spelling(68, "Ccce");
}

