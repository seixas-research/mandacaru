//! Shared by `berry_crystal` and `berry_functionals`: small PAW-LCAO SZ cells built by the Rust
//! setup and solved, with the setup kept (Berry phases need states off the SCF mesh). Needs
//! `MANDACARU_PAW_PATH`.
#![allow(dead_code)]
use crate::setup_shells_support::dataset;
use mandacaru_core::{
    models::hubbard_request::Request,
    paw::digest::sha256_file,
    periodic::{
        berry::Sampler,
        setup::{
            parse::{create_setup, AtomSpec, SetupInput},
            PeriodicSetup,
        },
        solver::Options,
        wannier::sampling,
        Crystal, PeriodicResult,
    },
    Context,
};
use nalgebra::{Matrix3, Vector3};
use std::sync::Arc;

#[derive(Clone)]
pub struct Cell {
    pub library: &'static str,
    pub symbols: Vec<&'static str>,
    /// Lattice vectors as columns, Bohr.
    pub lattice: Matrix3<f64>,
    pub grid: [usize; 3],
    pub fractional: Vec<[f64; 3]>,
    pub moments: Vec<f64>,
    pub spin: bool,
    pub spinor: bool,
    pub functional: &'static str,
    /// The full Gamma-anchored SCF mesh.
    pub mesh: [usize; 3],
    /// A `hubbard_u` request, JSON text.
    pub hubbard: Option<&'static str>,
    /// Fermi-Dirac width, Hartree.
    pub width: f64,
}

pub struct Solved {
    pub ctx: Context,
    pub setup: Arc<PeriodicSetup>,
    pub crystal: Arc<Crystal>,
    pub state: Arc<PeriodicResult>,
}

impl Solved {
    pub fn sampler(&self) -> Sampler<'_> {
        Sampler::new(&self.crystal, &self.state, &self.setup).unwrap()
    }
}

pub fn solve(cell: &Cell) -> Solved {
    let ctx = Context::new(Some(2), 1 << 30).unwrap();
    let mut species: Vec<&str> = vec![];
    for symbol in &cell.symbols {
        if !species.contains(symbol) {
            species.push(symbol);
        }
    }
    let root = std::path::PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").unwrap())
        .join(cell.library);
    let files = species
        .iter()
        .map(|symbol| {
            let path = root.join(format!("{symbol}.parquet"));
            (
                path.to_string_lossy().into_owned(),
                sha256_file(&path).unwrap(),
            )
        })
        .collect();
    let datasets = species
        .iter()
        .map(|symbol| Arc::new(dataset(&ctx, cell.library, symbol)))
        .collect();
    let count: usize = cell.mesh.iter().product();
    let input = SetupInput {
        mesh: None,
        recipes: None,
        projector_basis: "raw".into(),
        lattice: cell.lattice,
        grid_shape: cell.grid,
        atoms: cell
            .symbols
            .iter()
            .zip(&cell.fractional)
            .zip(&cell.moments)
            .map(|((symbol, fractional), moment)| AtomSpec {
                species: species.iter().position(|s| s == symbol).unwrap(),
                symbol: (*symbol).into(),
                fractional: *fractional,
                moment: *moment,
            })
            .collect(),
        kpoints_frac: (0..count)
            .map(|k| sampling::mesh_point(cell.mesh, k).unwrap())
            .collect(),
        weights: vec![1. / count as f64; count],
        functional: cell.functional.into(),
        precision: "float64".into(),
        basis_size: "SZ".into(),
        spinor: cell.spinor,
        spin: cell.spin,
        symmetry: false,
        filter_cutoff: None,
        sigma: None,
        dataset_files: Some(files),
        hubbard_u: cell.hubbard.map(|text| Request::parse(text).unwrap()),
        charge: None,
    };
    let setup = Arc::new(create_setup(&ctx, input, datasets).unwrap());
    let crystal = Arc::new(setup.build(&ctx).unwrap());
    let state = Arc::new(
        crystal
            .solve(&Options {
                tolerance: 1e-10,
                density_tolerance: 1e-8,
                width: cell.width,
                ..Options::default()
            })
            .unwrap(),
    );
    assert!(state.converged, "the SCF of the test cell did not converge");
    Solved {
        ctx,
        setup,
        crystal,
        state,
    }
}

/// H2 (1.4 Bohr bond along x) centered in a 6 x 8 x 8 Bohr cell: a centrosymmetric insulator.
pub fn h2_chain(library: &'static str, functional: &'static str) -> Cell {
    Cell {
        library,
        symbols: vec!["H", "H"],
        lattice: Matrix3::from_diagonal(&Vector3::new(6., 8., 8.)),
        grid: [15, 20, 20],
        fractional: vec![[2.3 / 6., 0.5, 0.5], [3.7 / 6., 0.5, 0.5]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional,
        mesh: [1, 1, 1],
        hubbard: None,
        width: 1e-3,
    }
}

/// LiH (3 Bohr bond along a_1) in a skew cell with 4 Bohr to the next Li along a_1: polar, no
/// inversion, and dispersive along a_1, so its curvature is not zero.
pub fn lih(library: &'static str) -> Cell {
    Cell {
        library,
        symbols: vec!["Li", "H"],
        // Columns a_1 = (7, 0, 0), a_2 = (0.8, 6.5, 0), a_3 = (0.4, 0.6, 7.5).
        lattice: Matrix3::new(7., 0.8, 0.4, 0., 6.5, 0.6, 0., 0., 7.5),
        grid: [20, 19, 22],
        fractional: vec![[0.1, 0.2, 0.15], [0.1 + 3. / 7., 0.2, 0.15]],
        moments: vec![0., 0.],
        spin: false,
        spinor: false,
        functional: "lda",
        mesh: [2, 1, 1],
        hubbard: None,
        width: 1e-3,
    }
}

/// `cell` with every atom moved by `steps` grid steps along the three lattice vectors.
pub fn shifted(cell: &Cell, steps: [i64; 3]) -> Cell {
    let mut out = cell.clone();
    for f in &mut out.fractional {
        for a in 0..3 {
            f[a] += steps[a] as f64 / cell.grid[a] as f64;
        }
    }
    out
}

const ANGSTROM_TO_BOHR: f64 = 1. / 0.529177210903;

/// Lattice vectors given as rows in Angstrom, returned as columns in Bohr.
pub fn angstrom_cell(rows: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| rows[j][i] * ANGSTROM_TO_BOHR)
}

/// The grid of at most `spacing` Angstrom along every lattice vector.
pub fn grid_of(lattice: &Matrix3<f64>, spacing: f64) -> [usize; 3] {
    std::array::from_fn(|j| {
        (lattice.column(j).norm() / (spacing * ANGSTROM_TO_BOHR)).ceil() as usize
    })
}

/// An eV smearing width in Hartree.
pub fn ev(width: f64) -> f64 {
    width / 27.211386245988
}

/// `cell` with its first two lattice vectors swapped (a left-handed relabeling of one crystal).
pub fn swapped(cell: &Cell) -> Cell {
    let mut out = cell.clone();
    out.lattice.swap_columns(0, 1);
    out.grid.swap(0, 1);
    for f in &mut out.fractional {
        f.swap(0, 1);
    }
    out.mesh.swap(0, 1);
    out
}
