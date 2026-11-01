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
