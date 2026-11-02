//! Shared by the `periodic_hybrid_spinor*` tests (SH-1a): the spinor snapshots of
//! `test/migration/reference/hybrid_spinor` (and the phase-5 `si_so.json`) as hybrid crystals, the
//! signed permutations of the time-reversal operator in the complex-harmonic basis, the SU(2)
//! rotation in the spin-major layout and the occupied density matrices the K-level gates use.
//!
//! The `sites` block is rebuilt here from the shipped dataset files with the library passed
//! explicitly (a fixture of another library would otherwise fall back to the wrong sphere model).
#![allow(dead_code)]

use mandacaru_core::{
    math::linalg::eigen,
    paw::{
        onecenter::Basis,
        pair_tables::{layout, Projector},
        record::PawDataset,
        species_tables::SpeciesTables,
    },
    periodic::{screened_exchange::ExxDivTreatment, Crystal},
    units::BOHR_TO_ANGSTROM,
    Context,
};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;
use serde_json::{json, Value};
use std::path::PathBuf;

pub mod rig;

pub type Matrix = DMatrix<C64>;
pub const GB: ExxDivTreatment = ExxDivTreatment::GygiBaldereschi;
pub const NONE: ExxDivTreatment = ExxDivTreatment::None;

pub fn context() -> Context {
    Context::new(Some(1), 768 << 20).unwrap()
}

/// A fixture by file name: the hybrid-spinor reference folder first, then the phase-5 folder.
pub fn read(name: &str) -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/migration");
    let path = ["reference/hybrid_spinor", "phase5"]
        .iter()
        .map(|folder| root.join(folder).join(name))
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("no fixture {name}: run test/migration/freeze_hybrid_spinor.py"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The dataset library a fixture was built from: the freeze records it, the older `si_so.json`
/// is `lda-fr` by its name.
pub fn library(fixture: &Value, name: &str) -> String {
    match fixture["provenance"]["library"].as_str() {
        Some(library) => library.to_owned(),
        None if name == "si_so.json" => "lda-fr".to_owned(),
        None => panic!("{name} records no dataset library"),
    }
}

pub fn symbol(z: u64) -> &'static str {
    match z {
        1 => "H",
        14 => "Si",
        78 => "Pt",
        79 => "Au",
        other => panic!("no dataset mapped for Z = {other}"),
    }
}

fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn dataset_file(library: &str, symbol: &str) -> PathBuf {
    let root = std::env::var_os("MANDACARU_PAW_PATH")
        .filter(|v| !v.is_empty())
        .expect("MANDACARU_PAW_PATH is unset: a hybrid crystal reads the shipped dataset files");
    PathBuf::from(root).join(format!("{library}/{symbol}.parquet"))
}

/// The `sites` block of the shipped `library` for the geometry the fixture recorded.
pub fn sites(fixture: &Value, library: &str) -> Value {
    let geometry = &fixture["provenance"]["geometry"];
    let to_bohr = 1.0 / BOHR_TO_ANGSTROM;
    // The centers the projections were built with when the snapshot records them, else the geometry.
    let centers: Vec<Vec<f64>> = match fixture["sites"]["centers"].as_array() {
        Some(recorded) => recorded.iter().map(reals).collect(),
        None => geometry["positions_angstrom"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| reals(p).iter().map(|x| x * to_bohr).collect())
            .collect(),
    };
    let datasets: Vec<Value> = geometry["numbers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|z| {
            let symbol = symbol(z.as_u64().unwrap());
            let path = dataset_file(library, symbol);
            json!({"symbol": symbol, "file": path.to_str().unwrap(),
                   "sha256": mandacaru_core::paw::digest::sha256_file(&path).unwrap()})
        })
        .collect();
    json!({"centers": centers, "projector_basis": "raw", "datasets": datasets})
}

/// The fixture as an HSE06 JSON value with its spheres, `edit` applied last.
pub fn value(name: &str, edit: &dyn Fn(&mut Value)) -> Value {
    let mut v = read(name);
    let library = library(&v, name);
    v["functional"] = json!("hse06");
    v["sites"] = sites(&v, &library);
    edit(&mut v);
    v
}

pub fn hybrid(ctx: &Context, name: &str, edit: &dyn Fn(&mut Value)) -> Crystal {
    Crystal::from_json(ctx, &value(name, edit).to_string()).unwrap()
}

/// The first point alone as a whole (Gamma) mesh.
pub fn gamma_only(v: &mut Value) {
    let mut first = v["kpoints"][0].clone();
    first["weight"] = json!(1.0);
    v["kpoints"] = json!([first]);
    v["permutations"] = json!([]);
}

// ---------------------------------------------------------------- the two-component structure

/// `iσ_y (x) 1_m` in the spin-major layout, with `pi` (`m x m`) in the spatial slot.
pub fn time_reversal(pi: &Matrix) -> Matrix {
    let m = pi.nrows();
    Matrix::from_fn(2 * m, 2 * m, |i, j| match (i / m, j / m) {
        (0, 1) => pi[(i % m, j % m)],
        (1, 0) => -pi[(i % m, j % m)],
        _ => C64::default(),
    })
}

/// `u (x) 1_m`: an SU(2) rotation of the spin index of a spin-major matrix.
pub fn spin_rotation(u: &Matrix, m: usize) -> Matrix {
    Matrix::from_fn(2 * m, 2 * m, |i, j| {
        if i % m == j % m {
            u[(i / m, j / m)]
        } else {
            C64::default()
        }
    })
}

/// The wrong layout `1_m (x) u` (spin index fastest): the plan's loose notation, as a control.
pub fn spin_rotation_interleaved(u: &Matrix, m: usize) -> Matrix {
    Matrix::from_fn(2 * m, 2 * m, |i, j| {
        if i / 2 == j / 2 {
            u[(i % 2, j % 2)]
        } else {
            C64::default()
        }
    })
}

/// A reproducible SU(2) matrix with all four entries populated (`|alpha|^2 + |beta|^2 = 1`).
pub fn su2(seed: f64) -> Matrix {
    let a = 0.7 + 0.31 * seed;
    let (alpha, beta) = (
        C64::from_polar(a.cos(), 1.1 + seed),
        C64::from_polar(a.sin(), 0.3 * seed - 0.8),
    );
    let u = Matrix::from_row_slice(2, 2, &[alpha, -beta.conj(), beta, alpha.conj()]);
    assert!(
        (u.adjoint() * &u - Matrix::identity(2, 2)).norm() < 1e-14
            && (u.determinant() - C64::new(1., 0.)).norm() < 1e-14
    );
    u
}

pub fn sigma_x() -> Matrix {
    Matrix::from_row_slice(
        2,
        2,
        &[
            C64::default(),
            C64::new(1., 0.),
            C64::new(1., 0.),
            C64::default(),
        ],
    )
}

/// Largest entry of `a - b` over the largest entry of `b`.
pub fn relative(a: &Matrix, b: &Matrix) -> f64 {
    let scale = b.iter().map(|v| v.norm()).fold(0., f64::max);
    (a - b).iter().map(|v| v.norm()).fold(0., f64::max) / scale.max(1e-300)
}

/// Spin blocks of a `2m x 2m` matrix.
pub fn block(a: &Matrix, m: usize, s: usize, t: usize) -> Matrix {
    a.view((s * m, t * m), (m, m)).into_owned()
}

/// `a` with its cross-spin blocks zeroed.
pub fn spin_diagonal(a: &Matrix, m: usize) -> Matrix {
    Matrix::from_fn(2 * m, 2 * m, |i, j| {
        if i / m == j / m {
            a[(i, j)]
        } else {
            C64::default()
        }
    })
}

// ---------------------------------------------------------------- snapshot edits

/// The spin-orbit block of point `k` of a snapshot value (`2m x 2m`, row-major `[re, im]` pairs).
pub fn spin_orbit_of(v: &Value, k: usize) -> Matrix {
    let flat = v["kpoints"][k]["spin_orbit"]
        .as_array()
        .expect("no spin_orbit block");
    let n = (flat.len() as f64).sqrt() as usize;
    Matrix::from_fn(n, n, |i, j| {
        let pair = &flat[i * n + j];
        C64::new(pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap())
    })
}

pub fn set_spin_orbit(v: &mut Value, k: usize, h: &Matrix) {
    let pairs: Vec<[f64; 2]> = (0..h.nrows())
        .flat_map(|i| (0..h.ncols()).map(move |j| [h[(i, j)].re, h[(i, j)].im]))
        .collect();
    v["kpoints"][k]["spin_orbit"] = json!(pairs);
}

/// Every point's spin-orbit block replaced by `f(block)`.
pub fn map_spin_orbit(v: &mut Value, f: &dyn Fn(&Matrix) -> Matrix) {
    for k in 0..v["kpoints"].as_array().unwrap().len() {
        let h = f(&spin_orbit_of(v, k));
        set_spin_orbit(v, k, &h);
    }
}

/// The spinor snapshot as the scalar restricted one (no spin-orbit block, no moment): what a
/// spinor crystal without spin-orbit coupling must reproduce.
pub fn scalar_twin(v: &mut Value) {
    let n: usize = v["grid"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .product();
    v["spinor"] = json!(false);
    v["time_reversal"] = json!(false);
    v["spins"] = json!(1);
    v["initial_m"] = json!(vec![0.0; n]);
    for point in v["kpoints"].as_array_mut().unwrap() {
        point.as_object_mut().unwrap().remove("spin_orbit");
    }
}

// ---------------------------------------------------------------- labels and signed permutations

/// `(atom, l, m)` of every basis function, as the freeze recorded them.
pub fn orbital_labels(fixture: &Value) -> Vec<(usize, usize, i32)> {
    fixture["provenance"]["orbital_labels"]
        .as_array()
        .expect("the fixture records no orbital labels")
        .iter()
        .map(|t| {
            (
                t[0].as_u64().unwrap() as usize,
                t[1].as_u64().unwrap() as usize,
                t[2].as_i64().unwrap() as i32,
            )
        })
        .collect()
}

/// The `(atom, projector)` of every projector of the crystal, from the shipped datasets.
pub fn projector_labels(fixture: &Value, library: &str) -> Vec<(usize, Projector)> {
    let ctx = context();
    let mut out = Vec::new();
    for (atom, z) in fixture["provenance"]["geometry"]["numbers"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let path = dataset_file(library, symbol(z.as_u64().unwrap()));
        let record = PawDataset::read(&ctx, &path).unwrap();
        let tables =
            SpeciesTables::from_record(&ctx, &record, Basis::parse("raw").unwrap()).unwrap();
        out.extend(layout(&tables.view()).into_iter().map(|p| (atom, p)));
    }
    out
}

/// The signed permutation `e_(l,m) -> (-1)^m e_(l,-m)` over keys `(group, m)`: the entry
/// `[i, j]` is `(-1)^m_j` when `key_i = (group_j, -m_j)`; `occurrence` separates entries that
/// share `(group, m)` (several zetas) by their order of appearance.
pub fn signed_permutation<G: Copy + PartialEq>(keys: &[(G, i32)]) -> Matrix {
    let n = keys.len();
    let occurrence = |i: usize| keys[..i].iter().filter(|k| **k == keys[i]).count();
    let mut pi = Matrix::zeros(n, n);
    for j in 0..n {
        let (group, m) = keys[j];
        let target = (0..n)
            .find(|&i| keys[i] == (group, -m) && occurrence(i) == occurrence(j))
            .expect("no partner (l, -m)");
        pi[(target, j)] = C64::new(if m % 2 == 0 { 1. } else { -1. }, 0.);
    }
    pi
}

pub fn pi_ao(fixture: &Value) -> Matrix {
    let keys: Vec<_> = orbital_labels(fixture)
        .into_iter()
        .map(|(a, l, m)| ((a, l), m))
        .collect();
    signed_permutation(&keys)
}

pub fn pi_projector(fixture: &Value, library: &str) -> Matrix {
    let keys: Vec<_> = projector_labels(fixture, library)
        .into_iter()
        .map(|(a, p)| ((a, p.channel, p.radial, p.l), p.m))
        .collect();
    signed_permutation(&keys)
}

/// The index of the point at `-k` (modulo a reciprocal vector).
pub fn partner(c: &Crystal, i: usize) -> usize {
    (0..c.points.len())
        .find(|&j| {
            (0..3).all(|a| {
                let s = c.points[i].fractional[a] + c.points[j].fractional[a];
                (s - s.round()).abs() < 1e-8
            })
        })
        .expect("the mesh is not closed under k -> -k")
}

// ---------------------------------------------------------------- density matrices

/// The lowest `count` eigenvectors of `diag(fixed, fixed) + H_SO` at a point as `sum c c^+`.
pub fn occupied(c: &Crystal, k: usize, count: usize) -> Matrix {
    let point = &c.points[k];
    let m = c.m;
    let mut h = point.spin_orbit.clone().expect("a spinor fixture");
    for s in 0..2 {
        let mut part = h.view_mut((s * m, s * m), (m, m));
        part += &point.fixed;
    }
    let (_, v) = eigen((&h + h.adjoint()) * C64::new(0.5, 0.)).unwrap();
    let mut d = Matrix::zeros(2 * m, 2 * m);
    for n in 0..count {
        d += v.column(n) * v.column(n).adjoint();
    }
    d
}

/// A time-reversal-symmetric ensemble on a mesh closed under `k -> -k`: `D_-k = U D_k^* U^T`
/// with `u` the operator of the dataset level, Gamma-like points symmetrized in place.
pub fn kramers_densities(c: &Crystal, u: &Matrix, count: usize) -> Vec<Matrix> {
    let mut out: Vec<Option<Matrix>> = vec![None; c.points.len()];
    for k in 0..c.points.len() {
        if out[k].is_some() {
            continue;
        }
        let d = occupied(c, k, count);
        let j = partner(c, k);
        let image = u * d.conjugate() * u.transpose();
        if j == k {
            out[k] = Some((&d + &image) * C64::new(0.5, 0.));
        } else {
            out[k] = Some(d);
            out[j] = Some(image);
        }
    }
    out.into_iter().map(Option::unwrap).collect()
}

/// A spin-polarized ensemble rotated by a generic SU(2): `up` and `down` spatial occupations
/// of the lowest generalized states of each point, `D = R diag(D_up, D_dn) R^+`. Every spin block is
/// populated (the cross blocks carry the transverse moment).
pub fn rotated_polarized(c: &Crystal, up: usize, down: usize, u: &Matrix) -> Vec<Matrix> {
    let m = c.m;
    let r = spin_rotation(u, m);
    c.points
        .iter()
        .map(|point| {
            let h = (&point.fixed + point.fixed.adjoint()) * C64::new(0.5, 0.);
            let (_, v) = eigen(h).unwrap();
            let mut d = Matrix::zeros(2 * m, 2 * m);
            for (spin, count) in [(0, up), (1, down)] {
                for n in 0..count {
                    let mut vector = nalgebra::DVector::<C64>::zeros(2 * m);
                    vector.rows_mut(spin * m, m).copy_from(&v.column(n));
                    d += &vector * vector.adjoint();
                }
            }
            &r * d * r.adjoint()
        })
        .collect()
}
