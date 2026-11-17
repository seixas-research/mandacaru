//! Shared by the `periodic_*_reduced_mesh` tests (SH-5): crystals built by the Rust setup from a
//! cell, so one cell is solved on its whole mesh (listed point by point), on the mesh the setup
//! reduces by the space group, and, displaced, on the whole mesh again for the reconverged finite
//! differences. The filter cutoff and the ion width come from the UNdisplaced cell's grid spacing,
//! so a displaced build keeps the pins the derivatives hold fixed.
#![allow(dead_code)]

use mandacaru_core::{
    periodic::{
        setup::parse::{create_setup, AtomSpec, SetupInput},
        solver::Options,
        spectrum, Crystal, PeriodicResult,
    },
    Context,
};
use nalgebra::{Matrix3, Vector3};
use std::sync::Arc;

#[path = "../setup_shells_support/mod.rs"]
mod shells;

pub const HARTREE_EV: f64 = 27.211386245988;

/// A cell for the Rust setup (PAW-LCAO SZ, raw projectors).
#[derive(Clone)]
pub struct Cell {
    pub library: &'static str,
    pub functional: &'static str,
    pub symbols: Vec<&'static str>,
    /// Lattice vectors as columns (Bohr).
    pub lattice: Matrix3<f64>,
    pub grid: [usize; 3],
    pub fractional: Vec<[f64; 3]>,
    pub spinor: bool,
    /// Two collinear spin channels, with the initial moment of each atom (`None`: zero moments).
    pub spin: bool,
    pub moments: Option<Vec<f64>>,
    pub charge: Option<f64>,
    /// The k mesh of the whole listing: `(size, gamma)`.
    pub mesh: ([usize; 3], bool),
}

/// Where the k-points come from: the whole mesh listed point by point, or the mesh the setup
/// reduces (with the space group when `symmetry`).
#[derive(Clone, Copy, PartialEq)]
pub enum Points {
    Whole,
    Reduced { symmetry: bool },
}

pub fn context() -> Context {
    Context::new(Some(3), 1 << 30).unwrap()
}

pub fn build(ctx: &Context, reference: &Matrix3<f64>, cell: &Cell, points: Points) -> Crystal {
    let (mesh, kpoints, weights, symmetry) = match points {
        Points::Whole => {
            // The mesh point by point, as the setup lists it before any reduction.
            let whole = spectrum::mesh(ctx, cell.mesh.0, cell.mesh.1, &[], false).unwrap();
            (None, whole.points, whole.weights, false)
        }
        Points::Reduced { symmetry } => (Some(cell.mesh), vec![], vec![], symmetry),
    };
    let mut species: Vec<&str> = vec![];
    for symbol in &cell.symbols {
        if !species.contains(symbol) {
            species.push(symbol);
        }
    }
    let datasets = species
        .iter()
        .map(|symbol| Arc::new(shells::dataset(ctx, cell.library, symbol)))
        .collect();
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap())
        .join(cell.library);
    let files = species
        .iter()
        .map(|symbol| {
            let path = root.join(format!("{symbol}.parquet"));
            let digest = mandacaru_core::paw::digest::dataset_content_sha256(
                &Context::new(Some(1), 1 << 30).unwrap(),
                &path,
            )
            .unwrap();
            (path.to_string_lossy().into_owned(), digest)
        })
        .collect();
    let spacing = (0..3)
        .map(|a| reference.column(a).norm() / cell.grid[a] as f64)
        .fold(0f64, f64::max);
    let input = SetupInput {
        mesh,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: cell.lattice,
        grid_shape: cell.grid,
        atoms: cell
            .symbols
            .iter()
            .zip(&cell.fractional)
            .enumerate()
            .map(|(i, (symbol, fractional))| AtomSpec {
                species: species.iter().position(|s| s == symbol).unwrap(),
                symbol: (*symbol).into(),
                fractional: *fractional,
                moment: cell.moments.as_ref().map_or(0., |m| m[i]),
            })
            .collect(),
        kpoints_frac: kpoints,
        weights,
        functional: cell.functional.into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: cell.spinor,
        spin: cell.spin,
        symmetry,
        filter_cutoff: Some(std::f64::consts::PI / spacing),
        sigma: Some(1.4 * spacing),
        dataset_files: Some(files),
        hubbard_u: None,
        charge: cell.charge,
    };
    create_setup(ctx, input, datasets)
        .unwrap()
        .build(ctx)
        .unwrap()
}

pub fn solve(c: &Crystal, options: &Options) -> PeriodicResult {
    let r = c.solve(options).unwrap();
    assert!(r.converged, "not converged");
    r
}

/// Tight enough that a force, or a central difference at a 1e-3 step, is limited by the model.
pub fn tight() -> Options {
    Options {
        width: 0.01 / HARTREE_EV,
        tolerance: 1e-12,
        density_tolerance: 1e-10,
        max_iterations: 300,
        ..Default::default()
    }
}

/// Silicon (fcc, 10.26 Bohr) on `grid^3` points with the second atom moved `shift` Bohr along the
/// cubic z axis (the point group keeps the operations that fix z: four on an isotropic grid).
pub fn silicon(
    library: &'static str,
    functional: &'static str,
    spinor: bool,
    grid: usize,
    shift: f64,
) -> Cell {
    let a = 10.26 / 2.;
    let lattice = Matrix3::from_columns(&[
        Vector3::new(0., a, a),
        Vector3::new(a, 0., a),
        Vector3::new(a, a, 0.),
    ]);
    let mut cell = Cell {
        library,
        functional,
        symbols: vec!["Si", "Si"],
        lattice,
        grid: [grid; 3],
        fractional: vec![[0.; 3], [0.25; 3]],
        spinor,
        spin: false,
        moments: None,
        charge: None,
        mesh: ([2, 2, 2], true),
    };
    cell.move_atom(1, Vector3::new(0., 0., shift));
    cell
}

impl Cell {
    /// Moves `atom` by the Cartesian vector `by` (Bohr).
    pub fn move_atom(&mut self, atom: usize, by: Vector3<f64>) {
        let shift = self.lattice.try_inverse().unwrap() * by;
        for a in 0..3 {
            self.fractional[atom][a] += shift[a];
        }
    }
}
