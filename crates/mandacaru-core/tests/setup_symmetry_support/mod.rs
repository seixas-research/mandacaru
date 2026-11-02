//! Shared by the `setup_symmetry*` tests: the cells (Si, Al, Mg hcp), the frozen
//! reader and the group-closure check.
#![allow(dead_code)]

use mandacaru_core::{
    periodic::setup::symmetry::{Geometry, Symmetry},
    units::BOHR_TO_ANGSTROM,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

pub const SYMPREC: f64 = 1e-5 / BOHR_TO_ANGSTROM;
pub const T2: f64 = 1e-12;

pub fn reference(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/periodic")
        .join(name);
    let all: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    all["body"].clone()
}
pub fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
pub fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// A cell with its atoms; `rows` are ASE-style lattice rows in Angstrom.
pub struct Cell {
    pub lattice: Matrix3<f64>,
    pub fractional: Vec<[f64; 3]>,
    pub species: Vec<u32>,
    pub labels: Vec<u32>,
    pub grid: [usize; 3],
    pub symprec: f64,
    pub max_l: usize,
}
impl Cell {
    pub fn new(
        rows: [[f64; 3]; 3],
        fractional: &[[f64; 3]],
        species: u32,
        grid: [usize; 3],
    ) -> Self {
        let rows = Matrix3::from_fn(|i, j| rows[i][j] / BOHR_TO_ANGSTROM);
        Self {
            lattice: rows.transpose(),
            fractional: fractional.to_vec(),
            species: vec![species; fractional.len()],
            labels: vec![species; fractional.len()],
            grid,
            symprec: SYMPREC,
            max_l: 4,
        }
    }
    pub fn geometry(&self) -> Geometry<'_> {
        Geometry {
            lattice: self.lattice,
            fractional: &self.fractional,
            species: &self.species,
            labels: &self.labels,
            grid: self.grid,
            symprec: self.symprec,
            max_l: self.max_l,
        }
    }
    pub fn symmetry(&self) -> Symmetry {
        Symmetry::new(&self.geometry()).unwrap()
    }
}
pub fn fcc(a: f64) -> [[f64; 3]; 3] {
    let h = a / 2.;
    [[0., h, h], [h, 0., h], [h, h, 0.]]
}
pub fn silicon(n: usize) -> Cell {
    Cell::new(fcc(5.43), &[[0.; 3], [0.25; 3]], 14, [n; 3])
}
pub fn aluminum() -> Cell {
    Cell::new(fcc(4.05), &[[0.; 3]], 13, [7; 3])
}
pub fn magnesium(n: usize) -> Cell {
    let (a, c) = (3.21, 5.21);
    let rows = [
        [a, 0., 0.],
        [-a / 2., a * 3f64.sqrt() / 2., 0.],
        [0., 0., c],
    ];
    Cell::new(
        rows,
        &[[1. / 3., 2. / 3., 0.25], [2. / 3., 1. / 3., 0.75]],
        12,
        [n, n, 10],
    )
}

pub fn key(w: &Matrix3<i32>, t: [f64; 3]) -> ([i32; 9], [i64; 3]) {
    let t = t.map(|x| ((x * 1e6).round() as i64).rem_euclid(1_000_000));
    (std::array::from_fn(|i| w[(i / 3, i % 3)]), t)
}
/// Index of each kept operation in the frozen list (matched by `(W, t)`).
pub fn frozen_index(sym: &Symmetry, frozen: &Value) -> Vec<usize> {
    let w = ints(&frozen["integer_rotations"]["data"]);
    let t = reals(&frozen["translations"]["data"]);
    let table: HashMap<_, _> = (0..w.len() / 9)
        .map(|j| {
            let m = Matrix3::from_fn(|r, c| w[9 * j + 3 * r + c] as i32);
            (key(&m, [t[3 * j], t[3 * j + 1], t[3 * j + 2]]), j)
        })
        .collect();
    (0..sym.len())
        .map(|k| {
            *table
                .get(&key(sym.rotation(k), sym.translation(k)))
                .expect("an operation the reference lacks")
        })
        .collect()
}
pub fn matrix(v: &Value, row: usize, size: usize) -> Vec<Complex64> {
    let (re, im) = (reals(&v["re"]), reals(&v["im"]));
    (0..size * size)
        .map(|i| Complex64::new(re[row * size * size + i], im[row * size * size + i]))
        .collect()
}

/// Cartesian operation of kept operation `k`.
pub fn cartesian(cell: &Cell, sym: &Symmetry, k: usize) -> Matrix3<f64> {
    cell.lattice * sym.rotation(k).map(f64::from) * cell.lattice.try_inverse().unwrap()
}

/// The kept node permutations are a group: every product is a kept permutation.
pub fn closed_under_composition(sym: &Symmetry, nodes: usize) {
    let perms: Vec<Vec<usize>> = (0..sym.len())
        .map(|k| {
            let mut p = vec![0; nodes];
            sym.permutation(k, &mut p);
            p
        })
        .collect();
    let set: std::collections::HashSet<_> = perms.iter().collect();
    for a in &perms {
        for b in &perms {
            let product: Vec<usize> = b.iter().map(|&i| a[i]).collect();
            assert!(
                set.contains(&product),
                "the kept permutations are not closed"
            );
        }
    }
}
