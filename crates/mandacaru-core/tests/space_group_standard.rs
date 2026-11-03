use mandacaru_core::periodic::space_group::catalog::centering_vectors;
use mandacaru_core::periodic::space_group::standard::standardize_operations;
use mandacaru_core::periodic::space_group::{find, identify, Cell};
use nalgebra::Matrix3;
use std::collections::HashSet;

#[test]
fn test_standardize_triclinic_p1_and_p_bar_1() {
    // SG 1: P1
    let lat = Matrix3::new(4.2, 0.2, 0.3, 0.4, 4.8, 0.5, 0.6, 0.7, 5.5);
    let frac1 = [[0.0, 0.0, 0.0], [0.13, 0.27, 0.41], [0.52, 0.19, 0.73]];
    let labels1 = [1, 2, 3];
    let cell1 = Cell {
        lattice: lat,
        fractional: &frac1,
        labels: &labels1,
    };
    let ops1 = find(&cell1, 1e-5).unwrap();
    let std1 = standardize_operations(&ops1, &cell1).unwrap();
    assert_eq!(std1.centering, 'P');
    assert_eq!(std1.centering_vectors, vec![[0, 0, 0]]);
    assert_eq!(std1.conv_operations.len(), 1);
    let id1 = identify(&ops1, &cell1).unwrap();
    assert_eq!(id1.number, 1);
    assert_eq!(id1.symbol, "P1");

    // SG 2: P-1
    let frac2 = [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]];
    let labels2 = [1, 1];
    let cell2 = Cell {
        lattice: lat,
        fractional: &frac2,
        labels: &labels2,
    };
    let ops2 = find(&cell2, 1e-5).unwrap();
    let std2 = standardize_operations(&ops2, &cell2).unwrap();
    assert_eq!(std2.centering, 'P');
    assert_eq!(std2.conv_operations.len(), 2);
    let id2 = identify(&ops2, &cell2).unwrap();
    assert_eq!(id2.number, 2);
    assert_eq!(id2.symbol, "P-1");
}

#[test]
fn test_standardize_monoclinic_p2_m_and_c2_m() {
    let lat_p = Matrix3::new(4.5, 0.0, 0.0, 0.0, 5.2, 0.0, 1.2, 0.0, 6.1);
    // P2/m (#10): 2-fold about b and mirror ac
    let frac_p = [
        [0.15, 0.18, 0.35],
        [-0.15, 0.18, -0.35],
        [0.15, -0.18, 0.35],
        [-0.15, -0.18, -0.35],
    ];
    let labels_p = [1, 1, 1, 1];
    let cell_p = Cell {
        lattice: lat_p,
        fractional: &frac_p,
        labels: &labels_p,
    };
    let ops_p = find(&cell_p, 1e-5).unwrap();
    let std_p = standardize_operations(&ops_p, &cell_p).unwrap();
    assert_eq!(std_p.centering, 'P');
    let id_p = identify(&ops_p, &cell_p).unwrap();
    assert_eq!(id_p.number, 10);
    assert_eq!(id_p.symbol, "P2/m");

    // C2/m (#12): Base-centered monoclinic
    let mut frac_c = Vec::new();
    let mut labels_c = Vec::new();
    for p in &frac_p {
        frac_c.push([p[0].rem_euclid(1.0), p[1].rem_euclid(1.0), p[2].rem_euclid(1.0)]);
        labels_c.push(1);
        frac_c.push([(p[0] + 0.5).rem_euclid(1.0), (p[1] + 0.5).rem_euclid(1.0), p[2].rem_euclid(1.0)]);
        labels_c.push(1);
    }
    let cell_c = Cell {
        lattice: lat_p,
        fractional: &frac_c,
        labels: &labels_c,
    };
    let ops_c = find(&cell_c, 1e-5).unwrap();
    let std_c = standardize_operations(&ops_c, &cell_c).unwrap();
    assert_eq!(std_c.centering, 'C');
    let exp_cents: HashSet<[i32; 3]> = centering_vectors('C').iter().copied().collect();
    let act_cents: HashSet<[i32; 3]> = std_c.centering_vectors.into_iter().collect();
    assert_eq!(act_cents, exp_cents);
    let id_c = identify(&ops_c, &cell_c).unwrap();
    assert_eq!(id_c.number, 12);
    assert_eq!(id_c.symbol, "C2/m");
}

#[test]
fn test_standardize_orthorhombic_pmmm_cmmm_immm_fmmm() {
    let lat = Matrix3::new(4.1, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 6.3);

    // Pmmm (#47)
    let pos_base: [f64; 3] = [0.15, 0.22, 0.31];
    let signs: [f64; 2] = [-1.0, 1.0];
    let mut frac_p: Vec<[f64; 3]> = Vec::new();
    for &sx in &signs {
        for &sy in &signs {
            for &sz in &signs {
                frac_p.push([
                    (sx * pos_base[0]).rem_euclid(1.0),
                    (sy * pos_base[1]).rem_euclid(1.0),
                    (sz * pos_base[2]).rem_euclid(1.0),
                ]);
            }
        }
    }
    let labels_p = vec![1; frac_p.len()];
    let cell_p = Cell {
        lattice: lat,
        fractional: &frac_p,
        labels: &labels_p,
    };
    let ops_p = find(&cell_p, 1e-5).unwrap();
    let std_p = standardize_operations(&ops_p, &cell_p).unwrap();
    assert_eq!(std_p.centering, 'P');
    let id_p = identify(&ops_p, &cell_p).unwrap();
    assert_eq!(id_p.number, 47);
    assert_eq!(id_p.symbol, "Pmmm");

    // Immm (#71): add [0.5, 0.5, 0.5]
    let mut frac_i = frac_p.clone();
    for p in &frac_p {
        frac_i.push([
            (p[0] + 0.5).rem_euclid(1.0),
            (p[1] + 0.5).rem_euclid(1.0),
            (p[2] + 0.5).rem_euclid(1.0),
        ]);
    }
    let labels_i = vec![1; frac_i.len()];
    let cell_i = Cell {
        lattice: lat,
        fractional: &frac_i,
        labels: &labels_i,
    };
    let ops_i = find(&cell_i, 1e-5).unwrap();
    let std_i = standardize_operations(&ops_i, &cell_i).unwrap();
    assert_eq!(std_i.centering, 'I');
    let id_i = identify(&ops_i, &cell_i).unwrap();
    assert_eq!(id_i.number, 71);
    assert_eq!(id_i.symbol, "Immm");

    // Fmmm (#69): add FCC centering translations
    let mut frac_f = frac_p.clone();
    for p in &frac_p {
        frac_f.push([(p[0]).rem_euclid(1.0), (p[1] + 0.5).rem_euclid(1.0), (p[2] + 0.5).rem_euclid(1.0)]);
        frac_f.push([(p[0] + 0.5).rem_euclid(1.0), (p[1]).rem_euclid(1.0), (p[2] + 0.5).rem_euclid(1.0)]);
        frac_f.push([(p[0] + 0.5).rem_euclid(1.0), (p[1] + 0.5).rem_euclid(1.0), (p[2]).rem_euclid(1.0)]);
    }
    let labels_f = vec![1; frac_f.len()];
    let cell_f = Cell {
        lattice: lat,
        fractional: &frac_f,
        labels: &labels_f,
    };
    let ops_f = find(&cell_f, 1e-5).unwrap();
    let std_f = standardize_operations(&ops_f, &cell_f).unwrap();
    assert_eq!(std_f.centering, 'F');
    let id_f = identify(&ops_f, &cell_f).unwrap();
    assert_eq!(id_f.number, 69);
    assert_eq!(id_f.symbol, "Fmmm");
}

#[test]
fn test_standardize_tetragonal_p4_mmm_and_i4_mmm() {
    let lat = Matrix3::new(4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0);
    // P4/mmm (#123)
    let frac_p = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
    let labels_p = [1, 2];
    let cell_p = Cell {
        lattice: lat,
        fractional: &frac_p,
        labels: &labels_p,
    };
    let ops_p = find(&cell_p, 1e-5).unwrap();
    let std_p = standardize_operations(&ops_p, &cell_p).unwrap();
    assert_eq!(std_p.centering, 'P');
    let id_p = identify(&ops_p, &cell_p).unwrap();
    assert_eq!(id_p.number, 123);
    assert_eq!(id_p.symbol, "P4/mmm");

    // I4/mmm (#139)
    let frac_i = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ];
    let labels_i = [1, 1];
    let cell_i = Cell {
        lattice: lat,
        fractional: &frac_i,
        labels: &labels_i,
    };
    let ops_i = find(&cell_i, 1e-5).unwrap();
    let std_i = standardize_operations(&ops_i, &cell_i).unwrap();
    assert_eq!(std_i.centering, 'I');
    let id_i = identify(&ops_i, &cell_i).unwrap();
    assert_eq!(id_i.number, 139);
    assert_eq!(id_i.symbol, "I4/mmm");
}

#[test]
fn test_standardize_trigonal_and_hexagonal() {
    let a = 4.0;
    let c = 6.0;
    let hex_lat = Matrix3::new(
        a, 0.0, 0.0,
        -a / 2.0, a * 3.0f64.sqrt() / 2.0, 0.0,
        0.0, 0.0, c,
    );

    // P6/mmm (#191)
    let frac_hex = [[0.0, 0.0, 0.0]];
    let labels_hex = [1];
    let cell_hex = Cell {
        lattice: hex_lat,
        fractional: &frac_hex,
        labels: &labels_hex,
    };
    let ops_hex = find(&cell_hex, 1e-5).unwrap();
    let std_hex = standardize_operations(&ops_hex, &cell_hex).unwrap();
    assert_eq!(std_hex.centering, 'P');
    let id_hex = identify(&ops_hex, &cell_hex).unwrap();
    assert_eq!(id_hex.number, 191);
    assert_eq!(id_hex.symbol, "P6/mmm");

    // R-3m (#166): Obverse triple hexagonal cell
    let frac_r = [
        [0.0, 0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
    ];
    let labels_r = [1, 1, 1];
    let cell_r = Cell {
        lattice: hex_lat,
        fractional: &frac_r,
        labels: &labels_r,
    };
    let ops_r = find(&cell_r, 1e-5).unwrap();
    let std_r = standardize_operations(&ops_r, &cell_r).unwrap();
    assert_eq!(std_r.centering, 'R');
    let id_r = identify(&ops_r, &cell_r).unwrap();
    assert_eq!(id_r.number, 166);
    assert_eq!(id_r.symbol, "R-3m");
}

#[test]
fn test_standardize_cubic_pm3m_im3m_fm3m() {
    let a = 5.0;
    let sc_lat = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);

    // Pm-3m (#221)
    let frac_p = [[0.0, 0.0, 0.0]];
    let labels_p = [1];
    let cell_p = Cell {
        lattice: sc_lat,
        fractional: &frac_p,
        labels: &labels_p,
    };
    let ops_p = find(&cell_p, 1e-5).unwrap();
    let std_p = standardize_operations(&ops_p, &cell_p).unwrap();
    assert_eq!(std_p.centering, 'P');
    let id_p = identify(&ops_p, &cell_p).unwrap();
    assert_eq!(id_p.number, 221);
    assert_eq!(id_p.symbol, "Pm-3m");

    // Im-3m (#229)
    let frac_i = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
    let labels_i = [1, 1];
    let cell_i = Cell {
        lattice: sc_lat,
        fractional: &frac_i,
        labels: &labels_i,
    };
    let ops_i = find(&cell_i, 1e-5).unwrap();
    let std_i = standardize_operations(&ops_i, &cell_i).unwrap();
    assert_eq!(std_i.centering, 'I');
    let id_i = identify(&ops_i, &cell_i).unwrap();
    assert_eq!(id_i.number, 229);
    assert_eq!(id_i.symbol, "Im-3m");

    // Fm-3m (#225) from FCC primitive cell
    let fcc_lat = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    let frac_f = [[0.0, 0.0, 0.0]];
    let labels_f = [1];
    let cell_f = Cell {
        lattice: fcc_lat,
        fractional: &frac_f,
        labels: &labels_f,
    };
    let ops_f = find(&cell_f, 1e-5).unwrap();
    let std_f = standardize_operations(&ops_f, &cell_f).unwrap();
    assert_eq!(std_f.centering, 'F');
    let id_f = identify(&ops_f, &cell_f).unwrap();
    assert_eq!(id_f.number, 225);
    assert_eq!(id_f.symbol, "Fm-3m");
}

#[test]
fn test_standardize_supercells_and_skewed_cells() {
    let a = 3.61;
    let fcc: Matrix3<f64> = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    let frac = [[0.0, 0.0, 0.0]];
    let labels = [29];

    // 2x2x2 supercell
    let lat_222 = 2.0 * fcc;
    let mut frac_222 = Vec::new();
    let mut labels_222 = Vec::new();
    for di in 0..2 {
        for dj in 0..2 {
            for dk in 0..2 {
                frac_222.push([
                    0.5 * di as f64,
                    0.5 * dj as f64,
                    0.5 * dk as f64,
                ]);
                labels_222.push(29);
            }
        }
    }
    let cell_222 = Cell {
        lattice: lat_222,
        fractional: &frac_222,
        labels: &labels_222,
    };

    let ops_222 = find(&cell_222, 1e-5).unwrap();
    assert_eq!(ops_222.n_pure_translations, 8);
    let std_222 = standardize_operations(&ops_222, &cell_222).unwrap();
    assert_eq!(std_222.centering, 'F');
    let id_222 = identify(&ops_222, &cell_222).unwrap();
    assert_eq!(id_222.number, 225);
    assert_eq!(id_222.symbol, "Fm-3m");

    // Skewed cell: S = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    let skew = Matrix3::new(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let lat_skew = skew * fcc;
    let cell_skew = Cell {
        lattice: lat_skew,
        fractional: &frac,
        labels: &labels,
    };
    let ops_skew = find(&cell_skew, 1e-5).unwrap();
    let std_skew = standardize_operations(&ops_skew, &cell_skew).unwrap();
    assert_eq!(std_skew.centering, 'F');
    let id_skew = identify(&ops_skew, &cell_skew).unwrap();
    assert_eq!(id_skew.number, 225);
    assert_eq!(id_skew.symbol, "Fm-3m");
}

#[test]
fn test_standardize_left_handed_cell() {
    let a = 3.61;
    let mut fcc: Matrix3<f64> = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    // Invert first axis to make cell left-handed: det(lattice) < 0
    fcc[(0, 0)] = -fcc[(0, 0)];
    fcc[(0, 1)] = -fcc[(0, 1)];
    fcc[(0, 2)] = -fcc[(0, 2)];
    assert!(fcc.determinant() < 0.0);

    let frac = [[0.0, 0.0, 0.0]];
    let labels = [29];
    let cell_lh = Cell {
        lattice: fcc,
        fractional: &frac,
        labels: &labels,
    };

    let ops_lh = find(&cell_lh, 1e-5).unwrap();
    let std_lh = standardize_operations(&ops_lh, &cell_lh).unwrap();
    assert_eq!(std_lh.centering, 'F');
    let id_lh = identify(&ops_lh, &cell_lh).unwrap();
    assert_eq!(id_lh.number, 225);
    assert_eq!(id_lh.symbol, "Fm-3m");
}
