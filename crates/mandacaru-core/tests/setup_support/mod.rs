//! Shared by the `setup_*` tests: the frozen Si crystal (`si_crystal.json`, 2 atoms,
//! SZ lda-sr, 10^3 grid) read key by key, and the shipped `lda-sr` Si table read
//! straight from its Parquet file. The typed dataset record does not exist yet; the
//! tests build the plain inputs of `periodic::setup` from the file themselves.
#![allow(dead_code)]

use mandacaru_core::{
    math::SpatialGrid,
    parquet::read_table,
    paw::local_split::grid_spacing,
    periodic::setup::shapes::{Atom, Channel},
    units::BOHR_TO_ANGSTROM,
    Context,
};
use nalgebra::Matrix3;
use num_complex::Complex64;
use serde_json::Value;
use std::path::PathBuf;

/// The Gaussian width factor of the local-potential split.
pub const SIGMA_FACTOR: f64 = 1.4;

pub fn frozen() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test/migration/phase5/si_crystal.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}
pub fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
/// A list of `[re, im]` pairs.
pub fn complexes(v: &Value) -> Vec<Complex64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| Complex64::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
        .collect()
}
/// Largest difference over the largest magnitude of the reference (T1).
pub fn relative<T: Copy + Into<Complex64>>(got: &[T], want: &[T]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().map(|w| (*w).into().norm()).fold(0., f64::max);
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| ((*g).into() - (*w).into()).norm())
        .fold(0., f64::max);
    worst / scale.max(1e-300)
}

pub struct Shipped {
    pub radius: Vec<f64>,
    pub v_local: Vec<f64>,
    pub valence_density: Vec<f64>,
    pub smooth_core: Vec<f64>,
    pub charge: f64,
    pub compensation_radius: f64,
    pub compensation_charge: f64,
    pub local_cutoff: f64,
    pub nlcc_applied: bool,
    pub nlcc_source: String,
}
impl Shipped {
    /// `$MANDACARU_PAW_PATH/<library>/<symbol>.parquet`; fails loudly when the variable is unset.
    pub fn read(library: &str, symbol: &str) -> Self {
        let root = std::env::var_os("MANDACARU_PAW_PATH")
            .filter(|v| !v.is_empty())
            .expect("MANDACARU_PAW_PATH is unset: the setup tests read the shipped lda-sr tables");
        let path = PathBuf::from(root).join(format!("{library}/{symbol}.parquet"));
        let ctx = Context::new(Some(1), 256 << 20).unwrap();
        let table = read_table(&ctx, &path).unwrap_or_else(|e| panic!("{symbol}: {e}"));
        let meta: Value =
            serde_json::from_str(&table.metadata["mandacaru.pseudopotential"]).unwrap();
        let column = |name: &str| -> Vec<f64> {
            table
                .column(name)
                .and_then(|c| c.as_f64())
                .unwrap()
                .to_vec()
        };
        Self {
            radius: column("r"),
            v_local: column("v_local"),
            valence_density: column("valence_density"),
            smooth_core: column("smooth_core_density"),
            charge: meta["valence_charge"].as_f64().unwrap(),
            compensation_radius: meta["compensation_radius"].as_f64().unwrap(),
            compensation_charge: meta["compensation_charge"].as_f64().unwrap(),
            local_cutoff: meta["r_cut_local"].as_f64().unwrap(),
            nlcc_applied: meta["nlcc"]["applied"].as_bool().unwrap_or(false),
            nlcc_source: meta["nlcc"]["source"].as_str().unwrap_or("").to_owned(),
        }
    }
}

/// The frozen Si crystal as the plain inputs of `periodic::setup`.
pub struct Silicon {
    pub grid: SpatialGrid,
    pub atoms: Vec<Atom>,
    pub channels: Vec<Channel>,
    pub lattice: Matrix3<f64>,
    pub sigma: f64,
    pub centers: Vec<[f64; 3]>,
    pub json: Value,
}
impl Silicon {
    pub fn load() -> Self {
        let json = frozen();
        let shipped = Shipped::read("lda-sr", "Si");
        let shape: Vec<usize> = json["grid"]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let step = reals(&json["grid"]["step"]);
        let origin = reals(&json["grid"]["origin"]);
        let grid = SpatialGrid::new(
            [shape[0], shape[1], shape[2]],
            [origin[0], origin[1], origin[2]],
            step.clone().try_into().unwrap(),
            true,
        )
        .unwrap();
        let lattice = grid.step
            * Matrix3::from_diagonal(&nalgebra::Vector3::new(
                shape[0] as f64,
                shape[1] as f64,
                shape[2] as f64,
            ));
        let to_bohr = 1.0 / BOHR_TO_ANGSTROM;
        let centers: Vec<[f64; 3]> = json["provenance"]["geometry"]["positions_angstrom"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let p = reals(p);
                [p[0] * to_bohr, p[1] * to_bohr, p[2] * to_bohr]
            })
            .collect();
        let atoms = centers
            .iter()
            .map(|&center| Atom {
                center,
                charge: shipped.charge,
                compensation_radius: shipped.compensation_radius,
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
        let sigma = SIGMA_FACTOR * grid_spacing(&step.try_into().unwrap());
        Self {
            grid,
            atoms,
            channels,
            lattice,
            sigma,
            centers,
            json,
        }
    }
}

/// Element symbols of the fixtures' atomic numbers.
pub fn symbol(z: u64) -> &'static str {
    match z {
        1 => "H",
        13 => "Al",
        14 => "Si",
        other => panic!("no dataset mapped for Z = {other}"),
    }
}

/// The shipped library each phase-5 fixture was built from.
pub fn library_of(fixture: &str) -> &'static str {
    match fixture {
        "h2_skew_pbe.json" => "pbe-sr",
        "si_so.json" => "lda-fr",
        _ => "lda-sr",
    }
}

/// The `sites` block a hybrid snapshot of `fixture` carries when its datasets are the shipped
/// files of `library`: the atom centers in Bohr, the raw projector basis, and per atom the file
/// with its SHA-256 as it is today (the digest the fixture recorded is that of the library it was
/// made from, which has been regenerated since).
pub fn sites(fixture: &Value, library: &str) -> Value {
    let root = std::env::var_os("MANDACARU_PAW_PATH")
        .filter(|v| !v.is_empty())
        .expect("MANDACARU_PAW_PATH is unset: a hybrid crystal reads the shipped dataset files");
    let geometry = &fixture["provenance"]["geometry"];
    let to_bohr = 1.0 / BOHR_TO_ANGSTROM;
    let centers: Vec<Vec<f64>> = geometry["positions_angstrom"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| reals(p).iter().map(|x| x * to_bohr).collect())
        .collect();
    let datasets: Vec<Value> = geometry["numbers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|z| {
            let symbol = symbol(z.as_u64().unwrap());
            let path = PathBuf::from(&root).join(format!("{library}/{symbol}.parquet"));
            serde_json::json!({
                "symbol": symbol,
                "file": path.to_str().unwrap(),
                "sha256": mandacaru_core::paw::digest::sha256_file(&path).unwrap(),
            })
        })
        .collect();
    serde_json::json!({"centers": centers, "projector_basis": "raw", "datasets": datasets})
}
