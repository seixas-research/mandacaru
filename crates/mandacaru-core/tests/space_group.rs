use mandacaru_core::periodic::space_group::{find, Cell};
use nalgebra::Matrix3;

#[test]
fn test_si_diamond_primitive() {
    let a = 5.43;
    let fcc = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    let frac = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]];
    let labels = [14, 14];
    let cell = Cell {
        lattice: fcc,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    assert_eq!(ops.rotations.len(), 48);
    assert_eq!(ops.n_distinct_rotations, 48);
    assert_eq!(ops.n_pure_translations, 1);
    assert_eq!(ops.point_group.symbol, "m-3m");

    // Atom map check
    let mut map = vec![0; 2];
    for i in 0..ops.len() {
        ops.atom_map(i, &mut map);
        assert_eq!(map.len(), 2);
        assert!(map[0] < 2 && map[1] < 2 && map[0] != map[1]);
    }
}

#[test]
fn test_cu_fcc_primitive() {
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
    assert_eq!(ops.rotations.len(), 48);
    assert_eq!(ops.n_distinct_rotations, 48);
    assert_eq!(ops.n_pure_translations, 1);
    assert_eq!(ops.point_group.symbol, "m-3m");
}

#[test]
fn test_mg_hcp() {
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
    assert_eq!(ops.rotations.len(), 24);
    assert_eq!(ops.point_group.symbol, "6/mmm");
}

#[test]
fn test_aln_wurtzite() {
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
    let labels = [13, 13, 7, 7]; // Al, Al, N, N
    let cell = Cell {
        lattice: hex,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    assert_eq!(ops.rotations.len(), 12);
    assert_eq!(ops.point_group.symbol, "6mm");
}

#[test]
fn test_mgo_rocksalt() {
    let a = 4.2;
    let fcc = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    let frac = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
    let labels = [12, 8]; // Mg, O
    let cell = Cell {
        lattice: fcc,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    assert_eq!(ops.rotations.len(), 48);
    assert_eq!(ops.point_group.symbol, "m-3m");
}

#[test]
fn test_slab_3d_group() {
    // 2-D square hydrogen layer with vacuum in z
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
    assert_eq!(ops.rotations.len(), 16);
    assert_eq!(ops.point_group.symbol, "4/mmm");
}

#[test]
fn test_boundary_atoms() {
    // Atoms at boundary near 0 and 1
    let a = 3.61;
    let sc = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    let frac = [
        [1.0 - 1e-12, 1e-13, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5 - 1e-12, 1.0 - 1e-13],
    ];
    let labels = [29, 29, 29, 29];
    let cell = Cell {
        lattice: sc,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    assert_eq!(ops.rotations.len(), 192);
}

#[test]
fn test_cu_supercells() {
    let a = 3.61;
    let fcc: Matrix3<f64> = 0.5 * a * Matrix3::new(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    // 2x1x1 supercell of primitive Cu: cell has double volume, 2 atoms
    let lat = Matrix3::new(
        2.0 * fcc[(0, 0)], 2.0 * fcc[(0, 1)], 2.0 * fcc[(0, 2)],
        fcc[(1, 0)], fcc[(1, 1)], fcc[(1, 2)],
        fcc[(2, 0)], fcc[(2, 1)], fcc[(2, 2)],
    );
    let frac = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
    let labels = [29, 29];
    let cell = Cell {
        lattice: lat,
        fractional: &frac,
        labels: &labels,
    };

    let ops = find(&cell, 1e-5).unwrap();
    // In this non-cubic 2x1x1 representation of FCC, metric symmetry gives 12 rotations x 2 pure = 24 ops
    assert_eq!(ops.rotations.len(), 24);
    assert_eq!(ops.n_distinct_rotations, 12);
    assert_eq!(ops.n_pure_translations, 2);

    // 2x2x2 supercell of primitive FCC: 8 atoms, 48 point group ops * 8 pure translations = 384 ops
    let lat_222 = 2.0 * fcc;
    let mut frac_222 = Vec::new();
    let mut labels_222 = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                frac_222.push([0.5 * i as f64, 0.5 * j as f64, 0.5 * k as f64]);
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
    assert_eq!(ops_222.rotations.len(), 384);
    assert_eq!(ops_222.n_distinct_rotations, 48);
    assert_eq!(ops_222.n_pure_translations, 8);
}

#[test]
fn test_error_handling() {
    let a = 3.0;
    let lat = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);

    // Zero atoms
    let empty_cell = Cell {
        lattice: lat,
        fractional: &[],
        labels: &[],
    };
    assert!(find(&empty_cell, 1e-5).is_err());

    // Atoms too close
    let close_frac = [[0.0, 0.0, 0.0], [1e-7, 0.0, 0.0]];
    let close_labels = [1, 1];
    let close_cell = Cell {
        lattice: lat,
        fractional: &close_frac,
        labels: &close_labels,
    };
    let err = find(&close_cell, 1e-5).unwrap_err();
    assert!(err.to_string().contains("too close distance between atoms"));
}

#[test]
fn test_million_length_axis_sparse_grid() {
    use mandacaru_core::periodic::space_group::search::SparseGrid;
    let lat = Matrix3::new(2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1e6);
    let frac = [[0.0, 0.0, 0.0]];
    let labels = [1];
    let grid = SparseGrid::new(&frac, &lat, 1e-5).unwrap();
    let matched = grid.find_match(&[0.0, 0.0, 0.0], 1, &frac, &labels, &lat, 1e-5);
    assert_eq!(matched, Some(0));
}

#[test]
fn test_find_with_context_respects_budget() {
    use mandacaru_core::periodic::space_group::find_with_context;
    use mandacaru_core::{Context, Error};

    let lat = Matrix3::new(2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 12.0);
    let frac = [[0.0, 0.0, 0.0]];
    let labels = [1];
    let cell = Cell {
        lattice: lat,
        fractional: &frac,
        labels: &labels,
    };

    // Tiny 64-byte budget must return Error::Budget
    let ctx_tiny = Context::new(Some(1), 64);
    let err = find_with_context(&ctx_tiny, &cell, 1e-5).unwrap_err();
    match err {
        Error::Budget { .. } => {}
        other => panic!("expected Error::Budget, got {:?}", other),
    }

    // Sufficient budget succeeds
    let ctx_ok = Context::new(Some(1), 10_000_000);
    let ops = find_with_context(&ctx_ok, &cell, 1e-5).unwrap();
    assert_eq!(ops.rotations.len(), 16);
}

