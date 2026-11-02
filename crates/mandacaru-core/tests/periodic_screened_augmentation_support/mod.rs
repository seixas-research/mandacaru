//! Shared by the `periodic_screened_augmentation*` tests: the phase-5 fixtures as hybrid
//! crystals on a whole mesh, the plain geometry inputs of the compensation transforms read the
//! way `setup_support` reads them, and occupied density matrices built from the fixtures' own
//! fixed Hamiltonians (a mechanics state, not a self-consistent one).
#![allow(dead_code)]

use crate::setup_support::{library_of, reals, sites, symbol, Shipped};
use mandacaru_core::{
    math::linalg::eigen,
    periodic::{
        screened_exchange::{ExxDivTreatment, Screening},
        setup::{
            electrostatics::max_wavenumber,
            shapes::{Atom, Channel, Shapes, Sites},
        },
        Crystal,
    },
    units::BOHR_TO_ANGSTROM,
    Context,
};
use nalgebra::{DMatrix, Matrix3};
use num_complex::Complex64;
use serde_json::{json, Value};
use std::path::PathBuf;

pub mod reference;

pub type Matrix = DMatrix<Complex64>;

/// (fixture, first point alone, occupied states per spin channel) of the collinear cases.
pub const CASES: [(&str, bool, &[usize]); 4] = [
    ("si_crystal.json", true, &[4]),
    ("al_fcc_gamma.json", true, &[2]),
    ("h2_skew_pbe.json", false, &[1]),
    ("h_spin_lda.json", false, &[1, 0]),
];
pub const GB: ExxDivTreatment = ExxDivTreatment::GygiBaldereschi;

pub fn screening(omega: f64) -> Screening {
    Screening {
        omega,
        fraction: 0.25,
    }
}

pub fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/phase5")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The fixture as a hybrid on a whole mesh: `gamma` keeps the first point alone (Gamma is a
/// whole mesh), otherwise the mesh must already be whole (the two-point H2 and H meshes).
pub fn hybrid(ctx: &Context, name: &str, gamma: bool) -> Crystal {
    hybrid_at(ctx, name, gamma, [0.; 3])
}
/// [`hybrid`] with the grid's first node moved by `shift` (Bohr): the sampled numbers are the
/// same, so the crystal is the original translated by `shift`.
pub fn hybrid_at(ctx: &Context, name: &str, gamma: bool, shift: [f64; 3]) -> Crystal {
    let mut value = fixture(name);
    let origin = reals(&value["grid"]["origin"]);
    value["grid"]["origin"] = json!([
        origin[0] + shift[0],
        origin[1] + shift[1],
        origin[2] + shift[2]
    ]);
    value["functional"] = json!("hse06");
    value["sites"] = sites(&value, library_of(name));
    if gamma {
        let mut first = value["kpoints"][0].clone();
        first["weight"] = json!(1.0);
        value["kpoints"] = json!([first]);
        value["permutations"] = json!([]);
    }
    Crystal::from_json(ctx, &value.to_string()).unwrap()
}

/// Atoms, channels and lattice of a fixture, with the compensation radius of the shipped
/// `lda-sr` table of each species.
pub struct Geometry {
    pub atoms: Vec<Atom>,
    pub channels: Vec<Channel>,
    pub lattice: Matrix3<f64>,
}
pub fn geometry(name: &str, crystal: &Crystal) -> Geometry {
    let json = fixture(name);
    let to_bohr = 1.0 / BOHR_TO_ANGSTROM;
    let geometry = &json["provenance"]["geometry"];
    let numbers = geometry["numbers"].as_array().unwrap();
    let atoms = geometry["positions_angstrom"]
        .as_array()
        .unwrap()
        .iter()
        .zip(numbers)
        .map(|(p, z)| {
            let p = reals(p);
            let shipped = Shipped::read(library_of(name), symbol(z.as_u64().unwrap()));
            Atom {
                center: [p[0] * to_bohr, p[1] * to_bohr, p[2] * to_bohr],
                charge: shipped.charge,
                compensation_radius: shipped.compensation_radius,
            }
        })
        .collect();
    let channels = json["channel_labels"]
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
    let shape = crystal.grid.index.shape();
    let lattice = crystal.grid.step
        * Matrix3::from_diagonal(&nalgebra::Vector3::new(
            shape[0] as f64,
            shape[1] as f64,
            shape[2] as f64,
        ));
    Geometry {
        atoms,
        channels,
        lattice,
    }
}

impl Geometry {
    /// The plain inputs of the compensation transforms (the ion width is not read by them).
    pub fn sites(&self) -> Sites<'_> {
        Sites {
            atoms: &self.atoms,
            channels: &self.channels,
            lattice: self.lattice,
            sigma: 1.0,
        }
    }
    /// The same geometry with every atom moved by `shift`.
    pub fn translated(&self, shift: [f64; 3]) -> Self {
        Self {
            atoms: self
                .atoms
                .iter()
                .map(|a| Atom {
                    center: std::array::from_fn(|i| a.center[i] + shift[i]),
                    ..*a
                })
                .collect(),
            channels: self.channels.clone(),
            lattice: self.lattice,
        }
    }
}

/// A hybrid fixture with its geometry, shape tables and occupied density matrices.
pub struct Rig {
    pub ctx: Context,
    pub shape_ctx: Context,
    pub crystal: Crystal,
    pub geometry: Geometry,
    pub shapes: Shapes,
    pub d: Vec<Matrix>,
}
impl Rig {
    pub fn new(fixture: &str, gamma: bool, occupied: &[usize]) -> Self {
        let ctx = Context::new(Some(1), 512 << 20).unwrap();
        let crystal = hybrid(&ctx, fixture, gamma);
        let geometry = geometry(fixture, &crystal);
        let q_max = max_wavenumber(&ctx, &crystal.grid).unwrap();
        // The tables are charged to a context of their own: the crystal's peak is the exchange's.
        let shape_ctx = Context::new(Some(1), 512 << 20).unwrap();
        let shapes = Shapes::new(&shape_ctx, &geometry.sites(), q_max).unwrap();
        let d = densities(&crystal, occupied);
        Self {
            ctx,
            shape_ctx,
            crystal,
            geometry,
            shapes,
            d,
        }
    }
}

/// Largest entry of `a - b` over the largest entry of `b`.
pub fn relative(a: &Matrix, b: &Matrix) -> f64 {
    let scale = b.iter().map(|v| v.norm()).fold(0., f64::max);
    (a - b).iter().map(|v| v.norm()).fold(0., f64::max) / scale.max(1e-300)
}

/// Occupied states from the fixed Hamiltonian of each point: `D = degeneracy * sum_n c_n c_n^+`
/// for the lowest `n_occupied` generalized eigenvectors (`c^+ S c = 1`), one matrix per
/// (spin, k) as the solver stores them. A spin channel beyond `occupied` holds no state.
pub fn densities(c: &Crystal, occupied: &[usize]) -> Vec<Matrix> {
    if c.spinor {
        return spinor_densities(c, occupied[0]);
    }
    let weight = c.degeneracy();
    let mut out = Vec::new();
    for spin in 0..c.spins {
        for point in &c.points {
            let h = point.lowdin.adjoint() * &point.fixed * &point.lowdin;
            let (_, y) = eigen(h).unwrap();
            let coefficients = &point.lowdin * y;
            let mut d = Matrix::zeros(c.m, c.m);
            for n in 0..occupied[spin] {
                let column = coefficients.column(n);
                d += column * column.adjoint() * Complex64::new(weight, 0.);
            }
            out.push(d);
        }
    }
    out
}

/// The generalized-eigenvector coefficients `c_n` (columns, `c^+ S c = 1`) at a point.
pub fn coefficients(point: &mandacaru_core::periodic::KPoint) -> Matrix {
    let h = point.lowdin.adjoint() * &point.fixed * &point.lowdin;
    let (_, y) = eigen(h).unwrap();
    &point.lowdin * y
}

pub fn bits(matrix: &Matrix) -> Value {
    json!(matrix
        .row_iter()
        .flat_map(|row| row
            .iter()
            .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect::<Vec<_>>())
        .collect::<Vec<_>>())
}
pub fn from_bits(rows: usize, columns: usize, value: &Value) -> Matrix {
    let raw: Vec<u64> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect();
    Matrix::from_fn(rows, columns, |i, j| {
        let at = 2 * (i * columns + j);
        Complex64::new(f64::from_bits(raw[at]), f64::from_bits(raw[at + 1]))
    })
}

/// A spinor crystal: the lowest `occupied` eigenvectors of each point's spin-orbit Hamiltonian
/// (the `2m x 2m` matrix of the snapshot), `D = sum c c^+`. Mechanics only: the vectors are
/// orthonormal in the Euclidean metric, not in the overlap.
fn spinor_densities(c: &Crystal, occupied: usize) -> Vec<Matrix> {
    c.points
        .iter()
        .map(|point| {
            let h = point.spin_orbit.clone().unwrap();
            let (_, vectors) = eigen(h).unwrap();
            let mut d = Matrix::zeros(2 * c.m, 2 * c.m);
            for n in 0..occupied {
                let column = vectors.column(n);
                d += column * column.adjoint();
            }
            d
        })
        .collect()
}

/// `S D_spin S`, the analytic unit-charge amount that multiplies `-2 d_omega / sqrt(pi)`.
pub fn overlap_square(rig: &Rig, spin: usize, k: usize) -> Matrix {
    let c = &rig.crystal;
    let scale = if c.spins == 1 { 0.5 } else { 1.0 };
    let s = &c.points[k].overlap;
    s * &rig.d[spin * c.points.len() + k] * s * Complex64::new(scale, 0.)
}
