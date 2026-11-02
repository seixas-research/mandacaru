//! Shared by the `strain_*` tests: the Si, Al2 and (non-symmetric, skew) LiH cells of the
//! `periodic_gradients` freeze as the plain inputs of `periodic::setup` (grid, atoms, channels,
//! lattice, width, radial tables), read from the manifest and the snapshot WITHOUT loading a
//! crystal; the strain map every oracle applies; and the central differences.
//!
//! One strain map for every input: `x -> (1 + eps) x` on the lattice, the atoms, the grid step
//! and the grid origin (the node counts, the width, the shape tables and the lattice operations
//! stay). Fractional k is fixed, so a Cartesian k moves with `(1 - eps^T)`.
#![allow(dead_code)]

pub mod pinned;

use mandacaru_core::{
    math::{reciprocal, SpatialGrid},
    paw::{digest::sha256_file, record::PawDataset},
    periodic::setup::{
        fields::core_table,
        shapes::{Atom, Channel, Sites},
    },
    Context,
};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
use serde_json::Value;
use std::path::PathBuf;

pub type Eps = [[f64; 3]; 3];

pub fn context() -> Context {
    Context::new(Some(1), 768 << 20).unwrap()
}

/// `eps` with the single entry `(a, b)` set to `delta`.
pub fn entry(a: usize, b: usize, delta: f64) -> Eps {
    let mut eps = [[0.; 3]; 3];
    eps[a][b] = delta;
    eps
}

pub fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// One atom species' radial table and its smooth core, if the dataset applied one.
#[derive(Clone)]
pub struct Table {
    pub radius: Vec<f64>,
    pub core: Option<Vec<f64>>,
}

/// A cell: everything the lattice sums and the grid arrays take.
#[derive(Clone)]
pub struct Cell {
    pub name: &'static str,
    pub grid: SpatialGrid,
    pub atoms: Vec<Atom>,
    pub channels: Vec<Channel>,
    pub lattice: Matrix3<f64>,
    pub sigma: f64,
    pub tables: Vec<Table>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/reference/periodic_gradients")
}

fn rebuild(grid: &SpatialGrid, map: &Matrix3<f64>) -> SpatialGrid {
    let step = map * grid.step;
    let mut values = [0.; 9];
    for (i, value) in values.iter_mut().enumerate() {
        *value = step[(i / 3, i % 3)];
    }
    let origin = map * Vector3::from(grid.origin);
    SpatialGrid::new(grid.index.shape(), origin.into(), values, true).unwrap()
}

impl Cell {
    pub fn load(ctx: &Context, name: &'static str) -> Self {
        let read = |file: String| -> Value {
            let text = std::fs::read_to_string(root().join(&file))
                .unwrap_or_else(|e| panic!("{file}: {e}"));
            serde_json::from_str(&text).unwrap()
        };
        let manifest = read(format!("{name}.json"));
        let snapshot = read(format!("{name}.snapshot.json"));
        let setup = &manifest["setup"];
        let shape: Vec<usize> = snapshot["grid"]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let origin = reals(&snapshot["grid"]["origin"]);
        let step: [f64; 9] = reals(&snapshot["grid"]["step"]).try_into().unwrap();
        let grid = SpatialGrid::new(
            [shape[0], shape[1], shape[2]],
            [origin[0], origin[1], origin[2]],
            step,
            true,
        )
        .unwrap();
        let lattice = Matrix3::from_row_slice(&reals(&setup["lattice_bohr"]["re"]));
        let cell = grid.step
            * Matrix3::from_diagonal(&Vector3::new(
                shape[0] as f64,
                shape[1] as f64,
                shape[2] as f64,
            ));
        assert!(
            (lattice - cell).norm() < 1e-9 * lattice.norm(),
            "{name}: the lattice is the grid's cell"
        );
        let paw = std::env::var_os("MANDACARU_PAW_PATH")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .expect("MANDACARU_PAW_PATH is unset: the strain tests read the shipped datasets");
        let mut atoms = Vec::new();
        let mut tables = Vec::new();
        for (atom, center) in snapshot["sites"]["centers"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let center = reals(center);
            let file = paw.join(setup["datasets"][atom]["file"].as_str().unwrap());
            assert_eq!(
                sha256_file(&file).unwrap(),
                snapshot["sites"]["datasets"][atom]["sha256"]
                    .as_str()
                    .unwrap(),
                "{file:?} is not the dataset the snapshot was built from"
            );
            let ds = PawDataset::read(ctx, &file).unwrap();
            atoms.push(Atom {
                center: [center[0], center[1], center[2]],
                charge: ds.valence_charge,
                compensation_radius: ds.compensation_radius,
            });
            let core = core_table(
                ds.nlcc.applied,
                ds.nlcc.source.as_deref().unwrap_or(""),
                ds.smooth_core_density.as_slice(),
            )
            .unwrap()
            .map(|c| c.to_vec());
            tables.push(Table {
                radius: ds.r.as_slice().to_vec(),
                core,
            });
        }
        let channels = snapshot["channel_labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c[0].as_u64().unwrap() as usize,
                    c[1].as_u64().unwrap() as usize,
                    c[2].as_i64().unwrap() as i32,
                )
            })
            .collect();
        Self {
            name,
            grid,
            atoms,
            channels,
            lattice,
            sigma: setup["sigma_bohr"].as_f64().unwrap(),
            tables,
        }
    }

    pub fn sites(&self) -> Sites<'_> {
        Sites {
            atoms: &self.atoms,
            channels: &self.channels,
            lattice: self.lattice,
            sigma: self.sigma,
        }
    }

    pub fn map(eps: Eps) -> Matrix3<f64> {
        Matrix3::identity() + Matrix3::from_fn(|a, b| eps[a][b])
    }

    /// The whole cell under `x -> (1 + eps) x`: lattice, atoms, grid step and grid origin.
    pub fn strained(&self, eps: Eps) -> Self {
        let map = Self::map(eps);
        let mut out = self.clone();
        out.lattice = map * self.lattice;
        for atom in &mut out.atoms {
            atom.center = (map * Vector3::from(atom.center)).into();
        }
        out.grid = rebuild(&self.grid, &map);
        out
    }

    /// A deterministic complex moment per channel (nonzero, of mixed phase).
    pub fn moments(&self, seed: u64) -> Vec<Complex64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        (0..self.channels.len())
            .map(|_| Complex64::new(next(), next()))
            .collect()
    }
}

/// The pin of the alias average: the strained grid keeps the reference lattice operations (the
/// count and the alias table, entry by entry).
pub fn assert_pinned(ctx: &Context, reference: &Cell, strained: &Cell) {
    assert_eq!(
        reciprocal::operation_count(ctx, &reference.grid).unwrap(),
        reciprocal::operation_count(ctx, &strained.grid).unwrap(),
        "{}: the strain changed the lattice operations of the alias average",
        reference.name
    );
    let (a, b) = (
        reciprocal::alias_table(ctx, &reference.grid).unwrap(),
        reciprocal::alias_table(ctx, &strained.grid).unwrap(),
    );
    assert_eq!(a.index, b.index);
    assert_eq!(a.weights, b.weights);
}

/// Largest `|got - want|` over the entries `skip` does not name.
pub fn worst<T: Copy + Into<Complex64>>(
    got: &[T],
    want: &[T],
    skip: &dyn Fn(usize) -> bool,
) -> f64 {
    got.iter()
        .zip(want)
        .enumerate()
        .filter(|(i, _)| !skip(*i))
        .map(|(_, (g, w))| ((*g).into() - (*w).into()).norm())
        .fold(0., f64::max)
}

pub fn maximum<T: Copy + Into<Complex64>>(values: &[T]) -> f64 {
    values.iter().map(|v| (*v).into().norm()).fold(0., f64::max)
}

/// The shape tables are linear between nodes `0.01` apart (`SHAPE_TABLE_STEP`); a central
/// difference of a table value straddles a node when the two strained wave numbers lie in
/// different cells (or one sits on a node).
pub fn straddles(plus: f64, minus: f64) -> bool {
    let step = mandacaru_core::periodic::setup::shapes::SHAPE_TABLE_STEP;
    let (a, b) = (plus / step, minus / step);
    a.floor() != b.floor() || (a - a.round()).abs() < 1e-6 || (b - b.round()).abs() < 1e-6
}

/// Per frequency of the grid's reciprocal set: does any wave vector that enters its alias mean
/// (the box vector and every alias) straddle a table node between the grids strained by `+h` and
/// `-h`? The array elements of such a frequency carry the finite difference's kink error.
pub fn tainted(ctx: &Context, plus: &SpatialGrid, minus: &SpatialGrid) -> Vec<bool> {
    let n = plus.index.points();
    let shape = plus.index.shape();
    let norm = |g: [f64; 3]| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let (bp, bm) = (
        reciprocal::reciprocal_cell(plus).unwrap(),
        reciprocal::reciprocal_cell(minus).unwrap(),
    );
    let mut out: Vec<bool> = (0..n)
        .map(|at| {
            straddles(
                norm(reciprocal::box_vector(&bp, shape, at)),
                norm(reciprocal::box_vector(&bm, shape, at)),
            )
        })
        .collect();
    let (tp, tm) = (
        reciprocal::alias_table(ctx, plus).unwrap(),
        reciprocal::alias_table(ctx, minus).unwrap(),
    );
    for e in 0..tp.len() {
        if straddles(norm(tp.vectors[e]), norm(tm.vectors[e])) {
            out[tp.index[e]] = true;
        }
    }
    out
}
